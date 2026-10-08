"""Receipt-time power integration and de-phased, pinned root sampling."""
import math
import os
import threading
import time

import runtime as rt

PERIOD = .37


def safe_mask(allowed, measured):
    mask = set(allowed) - set(measured)
    if not mask or mask & set(measured):
        raise RuntimeError('empty/unsafe monitor CPU mask')
    return mask


def sample(shell, battery, t0, average=False):
    fields = ['current_now', 'voltage_now', 'status']
    script = 'for f in ' + ' '.join(battery+'/'+f for f in fields)
    script += '; do v=; read -r v <"$f"; printf "%s\\n" "$v"; done'
    if average:
        script += f'; if [ -e {battery}/current_avg ]; then v=; read -r v <{battery}/current_avg; echo "$v"; else echo ABSENT; fi'
    def read():
        start = time.monotonic() - t0
        values = shell.run(script)
        receipt = time.monotonic() - t0
        if rt.shell_nothing(shell, values):  # no stdout and no su stderr: the only re-read (H2, FIX4C/FIX4D)
            raise rt.Blip('empty power answer: '+repr(values))
        # Base parser and checks (5629699), unchanged.
        if len(values) != 3 + int(average) or values[2] != 'Discharging':
            raise RuntimeError('power read missing/charger connected: '+repr(values))
        current, voltage = map(int, values[:2])
        if voltage <= 0:
            raise RuntimeError('invalid battery voltage')
        row = dict(t_start=start, t=receipt, current_now_uA=current, voltage_now_uV=voltage,
                   battery_w=-current*voltage/1e12, battery_status=values[2])
        if average:
            row['current_avg_uA'] = None if values[3] == 'ABSENT' else int(values[3])
        return row
    return rt.reread('power', read)


def weighted(rows, lo, hi, max_gap=1.5):
    """Trapezoids clipped/interpolated to boundaries; never extrapolate a missing tail."""
    if hi <= lo:
        return dict(mean_battery_w=None, covered_s=0., samples=0, gaps=0)
    points = sorted((r['t'], r['battery_w']) for r in rows if 'battery_w' in r)
    energy = covered = 0.
    gaps = 0
    for (a, wa), (b, wb) in zip(points, points[1:]):
        l, h = max(lo, a), min(hi, b)
        if h <= l:
            continue
        if b-a > max_gap:
            gaps += 1
            continue
        wl, wh = wa+(wb-wa)*(l-a)/(b-a), wa+(wb-wa)*(h-a)/(b-a)
        energy += (wl+wh)*.5*(h-l)
        covered += h-l
    complete = abs(covered-(hi-lo)) < 1e-6 and not gaps
    return dict(mean_battery_w=energy/(hi-lo) if complete else None,
                covered_s=covered, samples=sum(lo <= t <= hi for t, _ in points), gaps=gaps)


def summary(rows, calls, duration):
    inside = lambda r: any(c['started_s'] <= r['t'] < c['ended_s'] for c in calls)
    block = [r for r in rows if 0 <= r['t'] <= duration]
    return dict(**weighted(rows, 0, duration), inside_samples=sum(inside(r) for r in block),
                outside_samples=sum(not inside(r) for r in block),
                read_overlaps_call_boundary=sum(any(r['t_start'] < c[k] <= r['t']
                    for c in calls for k in ('started_s', 'ended_s')) for r in block),
                period_s=PERIOD, method='trapezoid over receipt times; lag remains a method limitation')


def sampler(shell, battery, t0, mask, measured, stop, rows, errors, verify, period=PERIOD, average=False):
    try:
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, mask)
        safe_mask(os.sched_getaffinity(tid), measured)
        if os.sched_getaffinity(tid) != set(mask):
            raise RuntimeError('sampler pinning failed')
        k = 0
        next_verify = -math.inf
        while not stop.is_set():
            now = time.monotonic()
            mark = len(rt.READ_RETRIES)
            if period or now >= next_verify:
                verify()  # cadence mode: each read; fastest probe: once per second
                next_verify = now + 1.
            row = sample(shell, battery, t0, average)
            row['retries'] = rt.retries_since(mark)
            rows.append(row)
            if period:
                k = max(k+1, math.ceil((time.monotonic()-t0)/period))
                if stop.wait(max(0., t0+k*period-time.monotonic())):
                    break
        mark = len(rt.READ_RETRIES)
        verify()
        row = sample(shell, battery, t0, average)  # end bracket, same reader owns shell
        row['retries'] = rt.retries_since(mark)
        rows.append(row)
    except BaseException as e:
        errors.append(f'{type(e).__name__}: {e}')
        stop.set()


def lag_response(rows, load_start, load_end, key='current_now_uA'):
    """Baseline/plateau from final 5 s of quiet/load; crossing interpolation, absent => null."""
    # Discharging current is negative; convert to positive load magnitude.
    points = sorted((r['t'], -r[key]) for r in rows if r.get(key) is not None)
    def mean(lo, hi):
        xs = [v for t, v in points if lo <= t < hi]
        return sum(xs)/len(xs) if xs else None
    baseline = mean(load_start-5, load_start)
    plateau = mean(max(load_start, load_end-5), load_end)
    result = dict(baseline_uA=baseline, plateau_uA=plateau, step_uA=None,
                  rise_50_s=None, rise_90_s=None, fall_50_s=None, fall_10_s=None)
    if baseline is None or plateau is None or plateau <= baseline:
        result['unresolved'] = 'no positive observed step; probe cannot identify lag'
        return result
    result['step_uA'] = step = plateau-baseline
    def crossing(at, end, fraction, rising):
        target = baseline+fraction*step
        candidates = [(t, v) for t, v in points if at <= t <= end]
        if candidates and (candidates[0][1] >= target if rising else candidates[0][1] <= target):
            return candidates[0][0]-at
        for (a, va), (b, vb) in zip(candidates, candidates[1:]):
            if (va < target <= vb) if rising else (va > target >= vb):
                return a+(target-va)*(b-a)/(vb-va)-at
        return None
    for pct in (50, 90):
        result[f'rise_{pct}_s'] = crossing(load_start, load_end, pct/100, True)
    for pct in (50, 10):
        result[f'fall_{pct}_s'] = crossing(load_end, points[-1][0], pct/100, False)
    result['estimator'] = 'observed last-5-s plateau, not a fitted physical delay; unresolved crossings are null'
    return result
