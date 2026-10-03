#!/usr/bin/env python3
"""Read-only view of recorded POWERMAP1 CONFIRM data; no phone imports."""
import argparse
import json
import math
import statistics
from pathlib import Path

TOKEN = '20261002T090821Z_CONFIRM'
MISSING = 'not recorded'


def find_run():
    repo = Path(__file__).resolve().parents[2]
    roots = [repo / 'benchmark/power_map/runs', repo.parent / 'power_map']
    matches = [p for root in roots if root.exists() for p in root.iterdir()
               if TOKEN in p.name and p.is_dir() and (p / 'run.json').exists()]
    if len(matches) != 1:
        raise SystemExit(f'Expected one {TOKEN} run in {roots}; found {matches}')
    return matches[0]


def statistic(rows, field, op):
    if not rows or any(field not in r or r[field] is None for r in rows):
        return MISSING
    return round(op(r[field] for r in rows), 3)


def cap_seconds(block, policy, lo, hi):
    if policy not in block.get('cpuinfo_max_khz', {}) or 'fast' not in block:
        return MISSING
    seconds, previous, seen = 0., 0., False
    for row in block['fast']:
        t = row.get('t')
        if t is None or t < 0 or t > block.get('duration_s', 0):
            continue
        dt = max(0, min(t, hi) - max(previous, lo))
        previous = t
        if dt:
            seen = True
            value = row.get('max', {}).get(policy)
            if value is None:
                return MISSING
            if value < block['cpuinfo_max_khz'][policy]:
                seconds += dt
    return round(seconds, 3) if seen else MISSING


def render(run):
    lines = [f'CONFIRM source: {run}']
    paths = list(run.glob('block_*CONFIRM*.json'))
    if len(paths) != 1:
        return '\n'.join(lines + ['CONFIRM block: not recorded']) + '\n'
    b = json.loads(paths[0].read_text())
    duration = b.get('duration_s')
    if duration is None:
        return '\n'.join(lines + ['duration: not recorded']) + '\n'
    lines += ['Capped s uses the POWERMAP later-reading convention; last-sample tail is uncredited.',
              'minute | skin mean/max C | battery mean W | capped s policy0/4/6 | Android status max | selector calls/correct']
    for minute in range(math.ceil(duration / 60)):
        lo, hi = minute * 60, min(duration, (minute + 1) * 60)
        ds = [r for r in b.get('dumps', []) if lo <= r.get('t', -1) < hi]
        ss = [r for r in b.get('samples', []) if lo <= r.get('t', -1) < hi and r.get('t_end', hi + 1) <= duration]
        cs = [r for r in b.get('selector_calls', []) if lo <= r.get('t', -1) < hi]
        calls = len(cs) if 'selector_calls' in b else MISSING
        correct = sum(c['correct'] for c in cs) if 'selector_calls' in b and all('correct' in c for c in cs) else MISSING
        caps = '/'.join(str(cap_seconds(b, p, lo, hi)) for p in ('policy0', 'policy4', 'policy6'))
        lines.append(f'{minute+1} | {statistic(ds, "skin", statistics.mean)}/{statistic(ds, "skin", max)} | '
                     f'{statistic(ss, "battery_w", statistics.mean)} | {caps} | {statistic(ds, "status", max)} | {calls}/{correct}')
    for p in ('policy4', 'policy6'):
        top = b.get('cpuinfo_max_khz', {}).get(p)
        first = next((r['t'] for r in b.get('fast', []) if 0 <= r.get('t', -1) <= duration
                      and r.get('max', {}).get(p) is not None and top is not None and r['max'][p] < top), None)
        lines.append(f'{p} first recorded cap s: {first if first is not None else MISSING}')
    lines.append('Selector calls not recorded correct: time s | case id | expected | returned | ms')
    # Expected answers are absent in POWERMAP calls. Never fill them from today's case file.
    if 'selector_calls' not in b:
        lines.append(MISSING)
    for c in b.get('selector_calls', []):
        if c.get('correct') is not True:
            lines.append(' | '.join(str(c.get(k, MISSING)) for k in ('t', 'case_id', 'expected', 'choice', 'ms')))
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path)
    a = ap.parse_args()
    print(render(a.run or find_run()), end='')


if __name__ == '__main__':
    main()
