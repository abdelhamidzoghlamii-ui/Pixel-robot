"""Check every 'file (N lines), line L:' + code-block citation of a review against the copy (Coder tool)."""
import re
import sys
from pathlib import Path

review, copy = Path(sys.argv[1]).read_text(), Path(sys.argv[2])
bad = ok = 0
cur = None  # (path, stated line count)
pat = re.compile(r'\[`([^`]+)`\]\(file://[^)]*?/round\d+_copy/([^)#]+)[^)]*\)\s*\((\d+) lines\)|(?:line|lines) (\d+)(?:[–-](\d+))?:\s*\n\s*```[a-z]*\n(.*?)\n\s*```', re.S)
for m in pat.finditer(review):
    if m.group(1):
        p = copy / m.group(2)
        cur = (p, int(m.group(3)))
        n = len(p.read_text().splitlines()) if p.is_file() else None
        if n != cur[1]:
            print(f'BAD line count: {m.group(2)} stated {cur[1]}, actual {n}'); bad += 1
        continue
    if cur is None:
        print(f'BAD: line {m.group(4)} cited without a file'); bad += 1
        continue
    lines = cur[0].read_text().splitlines() if cur[0].is_file() else []
    lo = int(m.group(4)); hi = int(m.group(5) or lo)
    quoted = [q.strip() for q in m.group(6).splitlines() if q.strip()]
    actual = [l.strip() for l in lines[lo - 1:hi]] if hi <= len(lines) else None
    if actual is None or [q.rstrip('`') for q in quoted] != [a for a in actual if a]:
        print(f'BAD: {cur[0].name}:{lo}{"-" + str(hi) if hi != lo else ""} quoted {quoted[:2]!r} actual {actual and actual[:2]!r}'); bad += 1
    else:
        ok += 1
print(f'citations ok {ok}, bad {bad}')
