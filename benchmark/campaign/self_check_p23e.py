"""Offline P23E archive integrity, pinned to the approved P23D base."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'benchmark/campaign'
BASE = 'a34087c8d14cc02f0ff685cfa5b9d892cdd5ac90'
FROZEN = 'benchmark/campaign/frozen_manifest_campaign_p23e.json'
UNHASHED = {
    FROZEN: 'self-reference',
    'benchmark/campaign/checks/p23e/self_check_p23e.stdout': 'produced by the check',
    'benchmark/campaign/checks/p23e/self_check_p23e.stderr': 'produced by the check',
}
REPORTS = {
    'review_campaign_p23d_r2_result.txt': (12040, 'fffcf1aeaba54f88620254cc960eb678e366595332aca3eb2fae71fb8793fb91'),
    'p23d_commit_receipt.json': (434, 'b97d9cf0fdcd7d983b4e98994fb532aaf43e6d4e4386409b84e6d5fa465b09d2'),
    'CAMPAIGN_P23D_CODER_REPORT.md': (72606, 'fb2e3d83d13b45b22778184b01578ca0055663b4a92a80e60644fcefa4560949'),
}
PATTERNS = {
    'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    'GitHub token': r'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})',
    'API token': r'\bsk-[A-Za-z0-9_-]{16,}',
    'AWS key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'private key': r'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----',
    'password': r'''(?i)\b(?:password|passwd|pwd)\s*[:=]\s*["']?[^\s"'`,;]{3,}''',
}
LINK = re.compile(r'\[[^\]\n]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def identity(path):
    content = path.read_bytes()
    return len(content), hashlib.sha256(content).hexdigest()


def inventory(manifest, text):
    rows = {}
    for line in text.splitlines():
        cols = line.split('|')
        if len(cols) == 6 and re.fullmatch(r'`[a-f0-9]{64}`', cols[-2].strip()):
            target = Path(cols[1].strip().strip('`'))
            if target.is_absolute():
                continue
            name = str((manifest.parent / target).resolve().relative_to(ROOT))
            assert name not in rows, 'Duplicate inventory row: ' + name
            rows[name] = (int(cols[-3]), cols[-2].strip().strip('`'))
    return rows


def nearest_manifest(name, manifests):
    matches = [m for m in manifests if (ROOT / m).parent in (ROOT / name).parents]
    return max(matches, key=lambda m: len(Path(m).parts)) if matches else None


def dead_links(path):
    for number, line in enumerate(path.read_text().splitlines(), 1):
        for match in LINK.finditer(line):
            target = match[1]
            if urlsplit(target).scheme or target.startswith(('#', '/')):
                continue
            if not (path.parent / unquote(target.split('#')[0])).exists():
                yield number, target


def main():
    frozen = json.loads((ROOT / FROZEN).read_bytes())
    assert frozen['task'] == 'CAMPAIGN_P23E' and frozen['base_commit'] == BASE
    entries = frozen['files']
    hashed = {r['path'] for r in entries}
    assert len(hashed) == len(entries), 'Duplicate frozen path'
    unhashed = {r['path']: r['reason'] for r in frozen['unhashed_paths']}
    assert len(unhashed) == len(frozen['unhashed_paths']) and unhashed == UNHASHED
    assert not hashed.intersection(unhashed), 'Hashed/unhashed overlap'
    changed = hashed | unhashed.keys()
    actual = set(git('diff', '--name-only', BASE).decode().splitlines())
    actual.update(git('ls-files', '--others', '--exclude-standard').decode().splitlines())
    assert actual == changed, 'Frozen/changed set differs: ' + str(sorted(actual ^ changed))
    assert all((ROOT / name).is_file() for name in changed), 'Missing frozen file'
    for row in entries:
        assert identity(ROOT / row['path']) == (row['bytes'], row['sha256']), row['path']
    archives = {'benchmark/campaign/reports/' + name for name in REPORTS}
    for name, expected in REPORTS.items():
        assert identity(HERE / 'reports' / name) == expected, name

    names = json.loads(git('show', BASE + ':benchmark/campaign/checks/p23d/artifact_manifests.json'))
    assert len(names) == 24 and len(set(names)) == 24
    payloads = {name for name in changed if name not in names and name not in unhashed
                and nearest_manifest(name, names) is not None}
    p23e_rows = set()
    preserved = 0
    for name in names:
        manifest = ROOT / name
        base = git('show', BASE + ':' + name)
        current = manifest.read_bytes()
        assert current.startswith(base), 'Historical manifest rewritten: ' + name
        assert b'## P23D current inventory' in base, name
        old_text = base.decode().split('## P23D current inventory', 1)[1]
        for target, expected in inventory(manifest, old_text).items():
            if target not in changed:
                assert identity(ROOT / target) == expected, target
                preserved += 1
        added = current[len(base):].decode()
        expected_payloads = {p for p in payloads if nearest_manifest(p, names) == name}
        if expected_payloads:
            assert added.count('## P23E current inventory') == 1, name
        else:
            assert not added, 'Unexpected manifest append: ' + name
        rows = inventory(manifest, added)
        assert set(rows) == expected_payloads, 'P23E inventory differs: ' + name
        for target, expected in rows.items():
            assert identity(ROOT / target) == expected, target
            assert target not in p23e_rows, 'Duplicate P23E row: ' + target
            p23e_rows.add(target)
    assert p23e_rows == payloads

    for name in sorted(changed):
        path = ROOT / name
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                assert not re.search(pattern, line), f'Safety match: {name}:{number}: {kind} [MASKED]'
        # Owner exemption: the three pinned archives retain historical dead links.
        if name not in archives:
            for number, target in dead_links(path):
                raise AssertionError(f'Dead link: {name}:{number}: {target}')
    print(f'PASS P23E: 3 pinned reports; 24 preserved manifest prefixes; {preserved} unchanged P23D rows; '
          f'{len(changed)} frozen paths ({len(hashed)} hashed, 3 externally hashed); '
          f'{len(payloads)} nearest-manifest rows; edited-file links and all-file public-safety scan.')


if __name__ == '__main__':
    main()
