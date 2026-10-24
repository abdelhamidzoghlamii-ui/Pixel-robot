"""VIS-3C review harness (Coder tool, outside the repo): build the isolated copy, take integrity snapshots.
Usage: python3 -B review_tool.py copy ROUND | snap ROUND NAME"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROBOT = Path('/termux-home/robot')
RV = Path('/termux-home/review/vis3c')
SETTINGS = Path('/root/.gemini/antigravity-cli/settings.json')
NEW = 'benchmark/llm_objective_setting/conversation/chat_mtp_repro'
SCRATCH = Path('/tmp/claude-0/-data-data-com-termux-files-home-robot/92f772f9-54c5-4779-bac1-626911bd8a9b/scratchpad')
SKIP = {'.jpg', '.jpeg', '.png', '.onnx', '.gguf', '.bin', '.npz', '.pth', '.mp4'}


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tree(root, skip_git):
    out = {}
    for d, dirs, files in os.walk(root):
        dirs.sort()
        if skip_git and '.git' in dirs:
            dirs.remove('.git')
        for n in sorted(files):
            p = Path(d) / n
            if p.is_symlink() or not p.is_file():
                out[str(p.relative_to(root))] = 'symlink' if p.is_symlink() else 'special'
            else:
                out[str(p.relative_to(root))] = sha(p)
    return out


def git(*a):
    return subprocess.run(['git', '-C', str(ROBOT), *a], capture_output=True, text=True).stdout


def copy(rnd):
    c = RV / f'round{rnd}_copy'
    if c.exists():
        raise SystemExit(f'{c} exists')
    for rel in git('ls-files').splitlines():
        if Path(rel).suffix.lower() in SKIP or Path(rel).parts[0] in ('test_photos', 'bench_photos'):
            continue
        if (ROBOT / rel).is_file():
            (c / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROBOT / rel, c / rel)
    for n in ('README.md', 'oneshot.sh', 'repro.py', 'test_repro.py'):
        (c / NEW).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROBOT / NEW / n, c / NEW / n)
    ctx = c / '_review_context'
    ctx.mkdir()
    for f in ('candidate.diff', 'checks.txt', 'frozen_candidate.sha256', 'current_hashes.txt', 'TASK.md'):
        shutil.copy2(RV / f, ctx / f)
    for rel, name in (('chat.py', 'chat_2091f42.py'), ('server_manager.py', 'server_manager_2091f42.py'),
                      ('benchmark/llm_objective_setting/conversation/chat_mtp/chat_mtp.py', 'chat_mtp_2091f42.py')):
        (ctx / name).write_text(git('show', f'2091f42:{rel}'))
    for f in ('VIS3B_MTP_INVESTIGATION.md', 'review_r2_stdout.txt'):
        shutil.copy2(SCRATCH / f, ctx / ('VIS3B_MTP_REVIEW_ROUND2.json' if f.startswith('review') else f))
    print(c, sum(1 for _ in c.rglob('*') if _.is_file()), 'files')


def snap(rnd, name):
    s = {'head': git('rev-parse', 'HEAD', 'origin/main'), 'ls_remote': git('ls-remote', 'origin', 'refs/heads/main'),
         'stash': git('stash', 'list'), 'status': git('status', '--porcelain=v1', '-uall', '--ignored'),
         'settings': sha(SETTINGS), 'copy': tree(RV / f'round{rnd}_copy', False), 'live': tree(ROBOT, True)}
    (RV / f'round{rnd}_{name}.json').write_text(json.dumps(s, indent=0, sort_keys=True))
    print(name, 'copy', len(s['copy']), 'live', len(s['live']), 'settings', s['settings'][:12], s['head'].split())


def compare(rnd):
    a, b = (json.loads((RV / f'round{rnd}_{n}.json').read_text()) for n in ('pre', 'post'))
    for k in a:
        if a[k] != b[k]:
            if isinstance(a[k], dict):
                diff = sorted(set(a[k].items()) ^ set(b[k].items()))
                print(f'DIFFERENT {k}: {len(diff)} entries, e.g. {diff[:6]}')
            else:
                print(f'DIFFERENT {k}')
        else:
            print(f'identical {k}' + (f' ({len(a[k])} files)' if isinstance(a[k], dict) else ''))


if __name__ == '__main__':
    {'copy': lambda: copy(sys.argv[2]), 'snap': lambda: snap(sys.argv[2], sys.argv[3]),
     'compare': lambda: compare(sys.argv[2])}[sys.argv[1]]()
