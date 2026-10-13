"""DRYRUN-1 analysis of the HEAT-A heat trace (read-only; run: python3 -I analyze.py TRACE)."""
import json, statistics as st, sys, collections
from datetime import datetime, timezone

raw = open(sys.argv[1], 'rb').read()
print('bytes', len(raw), 'ends_with_newline', raw.endswith(b'\n'))
rows, bad = [], []
for i, l in enumerate(raw.decode().splitlines(), 1):
    try: rows.append(json.loads(l))
    except Exception: bad.append(i)
print('lines', len(raw.decode().splitlines()), 'parsed', len(rows), 'bad', bad)
t0 = rows[0]['t_mono']
rel = lambda r: r['t_mono'] - t0
print('first', rows[0]['t_utc'], rows[0]['t_mono'], 'last', rows[-1]['t_utc'], rows[-1]['t_mono'])
dur = rows[-1]['t_mono'] - t0
u = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
print(f'duration mono {dur:.1f} s = {dur/60:.2f} min = {dur/3600:.3f} h; utc span {(u(rows[-1]["t_utc"])-u(rows[0]["t_utc"])).total_seconds():.0f} s')
nonmono = [i for i in range(1, len(rows)) if rows[i]['t_mono'] <= rows[i-1]['t_mono']]
print('t_mono non-increasing at rows', nonmono[:10], 'count', len(nonmono))
# utc vs mono drift
print('utc-mono offset first/last', (u(rows[0]['t_utc']).timestamp()-rows[0]['t_mono']), (u(rows[-1]['t_utc']).timestamp()-rows[-1]['t_mono']))
print('events', dict(collections.Counter(r['event'] for r in rows)))

# 4 dumpsys cadence
dz = [r for r in rows if r['event'] == 'dumpsys']
gaps = [(b['t_mono'] - a['t_mono'], a, b) for a, b in zip(dz, dz[1:])]
g = [x[0] for x in gaps]
print(f'dumpsys n {len(dz)} gap median {st.median(g):.3f} min {min(g):.3f} max {max(g):.3f} mean {st.mean(g):.3f}')
print('gaps >10s', sum(x > 10 for x in g), '>60s', sum(x > 60 for x in g))
for x, a, b in gaps:
    if x > 10: print(f'  gap {x:.1f} s  {a["t_utc"]} -> {b["t_utc"]}  (+{rel(a):.0f} s)')
for r in dz:
    if r['detail'] or r['skin_c'] is None or r['status'] is None:
        print('  dumpsys err/null', r)

# 5 phases
pcs = [r for r in rows if r['event'] in ('phase_change', 'hold_start', 'hold_end')]
start = next(r for r in rows if r['event'] == 'start')
print('phase changes', len(pcs))
viol = []
since = start['t_mono']  # ACTIVE since heat.start
act, pau = [], []
prev = start
for k, r in enumerate(rows):
    if r['event'] != 'phase_change': continue
    # decide() reading = latest dumpsys row with values before this row's camera op
    j = k - 1
    while rows[j]['event'] != 'dumpsys' or rows[j]['skin_c'] is None: j -= 1
    dec = rows[j]
    # preceding camera row (stop_ok/start_ok) right before enter()
    cam = rows[k - 1]
    el = r['t_mono'] - prev['t_mono']
    if r['phase'] == 'PAUSE':
        act.append(el)
        ok = dec['skin_c'] >= 37.0 and el >= 60
        rule = 'ACTIVE->PAUSE'
    else:
        pau.append(el)
        ok = dec['skin_c'] <= 35.0 and el >= 30
        rule = 'PAUSE->ACTIVE'
    print(f'  {r["t_utc"]} +{rel(r):7.1f}s {rule} decide-skin {dec["skin_c"]:.3f} (@{dec["t_utc"]}) row-skin {r["skin_c"]:.3f} status {r["status"]} prev-phase {el:6.1f}s cam={cam["event"]} {"OK" if ok else "VIOLATION"}')
    if not ok: viol.append(r)
    prev = r
print('violations', len(viol))
f = lambda xs: f'n {len(xs)} median {st.median(xs):.1f} min {min(xs):.1f} max {max(xs):.1f} sum {sum(xs):.1f}'
print('active (excl. trailing):', f(act)); print('  first active', f'{act[0]:.1f}', 'rest', f(act[1:]))
print('pause:', f(pau))
last_pc = pcs[-1]
tail_el = rows[-1]['t_mono'] - last_pc['t_mono']
print('trailing phase', last_pc['phase'], f'{tail_el:.1f} s')
tot_act = sum(act) + (tail_el if last_pc['phase'] == 'ACTIVE' else 0)
print(f'inference duty (whole) {tot_act/dur:.3f}')
fp = next(r for r in pcs if r['phase'] == 'PAUSE')
after = sum(act[1:]) + (tail_el if last_pc['phase'] == 'ACTIVE' else 0)
print(f'duty from first pause {after/(rows[-1]["t_mono"]-fp["t_mono"]):.3f}')

# 6 extremes
def mx(key, rs=rows):
    rs = [r for r in rs if r[key] is not None]
    m = max(r[key] for r in rs); hits = [r for r in rs if r[key] == m]
    return m, len(hits), hits[0]['t_utc'], hits[-1]['t_utc']
for key in ('skin_c', 'status', 'battery_c', 'cpu_max_c'):
    print('max', key, mx(key, dz))
print('min skin', min(r['skin_c'] for r in dz if r['skin_c'] is not None))
print('status counts', dict(collections.Counter(r['status'] for r in dz)))
print('HOLD rows', sum(r['phase'] == 'HOLD' for r in rows))
afterfp = [r for r in dz if r['t_mono'] >= fp['t_mono'] and r['skin_c'] is not None]
sk = [r['skin_c'] for r in afterfp]
print(f'after first pause ({fp["t_utc"]}): skin min {min(sk):.3f} max {max(sk):.3f} n {len(sk)}; <34.7: {sum(s<34.7 for s in sk)}, >37.4: {sum(s>37.4 for s in sk)}')
for r in afterfp:
    if r['skin_c'] < 34.7 or r['skin_c'] > 37.4: print('   out', r['t_utc'], r['skin_c'], r['phase'])
# battery trend
b = [r['battery_c'] for r in dz if r['battery_c'] is not None]
print('battery_c first/last', b[0], b[-1], 'cpu last', dz[-1]['cpu_max_c'], 'skin last', dz[-1]['skin_c'])
dist = collections.Counter(r['distance_cm'] for r in rows); print('distance_cm values', dist.most_common(5))
