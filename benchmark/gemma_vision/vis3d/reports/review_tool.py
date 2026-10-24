"""VIS-3D review harness (Coder tool, outside the repo; adapted from review/vis3c/review_tool.py): build the isolated
copy, take integrity snapshots, compare them. Usage: python3 -B review_tool.py copy N | snap N NAME | compare N"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROBOT = Path('/termux-home/robot')
RV = Path('/termux-home/review/vis3d')
SETTINGS = Path('/root/.gemini/antigravity-cli/settings.json')
NEW = 'benchmark/gemma_vision/vis3d'
DEP = 'benchmark/llm_objective_setting/conversation/chat_mtp_repro'  # untracked VIS-3C runner, unchanged
FRAMES = 'benchmark/relate_anything/quality1/runs/20261017T213604Z/frames'
SKIP = {'.jpg', '.jpeg', '.png', '.onnx', '.gguf', '.bin', '.npz', '.pth', '.mp4'}
CONTEXT = ('TASK.md', 'candidate.diff', 'checks.txt', 'frozen_candidate.sha256', 'identity.txt',
           'b11716_source_excerpts.txt')


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
    frames = {f'{FRAMES}/{f}.jpg' for f in json.loads((ROBOT / NEW / 'references.json').read_text())['frames']}
    for rel in git('ls-files').splitlines():
        if rel not in frames and (Path(rel).suffix.lower() in SKIP or Path(rel).parts[0] in ('test_photos', 'bench_photos')):
            continue
        if (ROBOT / rel).is_file():
            (c / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROBOT / rel, c / rel)
    for d, names in ((NEW, ('README.md', 'oneshot.sh', 'references.json', 'test_vis3d.py', 'vis3d.py')),
                     (DEP, ('README.md', 'oneshot.sh', 'repro.py', 'test_repro.py'))):
        (c / d).mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy2(ROBOT / d / n, c / d / n)
    ctx = c / '_review_context'
    ctx.mkdir()
    for f in CONTEXT:
        shutil.copy2(RV / f, ctx / f)
    print(c, sum(1 for _ in c.rglob('*') if _.is_file()), 'files')


def snap(rnd, name):
    s = {'head': git('rev-parse', 'HEAD', 'origin/main'), 'ls_remote': git('ls-remote', 'origin', 'refs/heads/main'),
         'stash': git('stash', 'list'), 'status': git('status', '--porcelain=v1', '-uall', '--ignored'),
         'settings': sha(SETTINGS), 'copy': tree(RV / f'round{rnd}_copy', False), 'live': tree(ROBOT, True)}
    (RV / f'round{rnd}_{name}.json').write_text(json.dumps(s, indent=0, sort_keys=True))
    print(name, 'copy', len(s['copy']), 'live', len(s['live']), 'settings', s['settings'][:12], s['head'].split())


def compare(rnd, a='pre', b='post', skip=()):
    a, b = (json.loads((RV / f'round{rnd}_{n}.json').read_text()) for n in (a, b))
    for k in a:
        if k in skip:
            continue
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
     'compare': lambda: compare(sys.argv[2], *sys.argv[3:5])}[sys.argv[1]]()
