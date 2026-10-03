#!/data/data/com.termux/files/usr/bin/python
"""DUTY1: native Termux only, motors off. Human launches oneshot.sh."""
import argparse
import gc
import json
import math
import os
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'power_map'))
import power_map as pm
cr, dp = pm.cr, pm.dp
OUT_ROOT = cr.HOME / 'duty_cycle'
NOTE = 'CPU-core reading, not a heat state'
BATTERY_S = .5


def blocks_for(run_set, smoke=False):
    if run_set == 'PARTS':
        return [dict(name=n, duration=60 if smoke and n == 'LOAD' else 20 if smoke else 120)
                for n in ('P0U', 'LOAD', 'P0', 'CAM', 'YOLO', 'YOLO_NOSPIN', 'SEL')]
    if run_set == 'CYCLE':
        return [dict(name=n, duration=50 if smoke and n == 'HEATCOOL' else 20 if smoke else
                     540 if n == 'HEATCOOL' else 240)
                for n in ('CONT', 'CYC50', 'CYC25', 'HEATCOOL')]
    raise ValueError(run_set)


def phase_plan(spec, smoke):
    name, duration = spec['name'], spec['duration']
    if name == 'HEATCOOL':
        heat = 30 if smoke else 360
        return [('active', heat, 0, True), ('pause', duration-heat, 0, True)]
    if name in ('CYC50', 'CYC25'):
        on, off = (10, 5) if smoke else (50, 10) if name == 'CYC50' else (25, 5)
        rows, left, cycle = [], duration, 0
        while left:
            whole = left >= on + off
            for phase, length in (('active', on), ('pause', off)):
                if left:
                    use = min(left, length)
                    rows.append((phase, use, cycle, whole))
                    left -= use
            cycle += 1
        return rows
    return [('active' if name in ('CONT', 'YOLO', 'YOLO_NOSPIN') else
             'selector' if name == 'SEL' else 'camera' if name == 'CAM' else 'pause', duration, 0, True)]


def selector_slots(active_before, active_s):
    """A slot at a phase boundary belongs to the next active phase, never to a pause."""
    k = math.ceil(active_before / cr.SELECTOR_S)
    return [k * cr.SELECTOR_S - active_before + j * cr.SELECTOR_S
            for j in range(max(0, math.ceil((active_before + active_s) / cr.SELECTOR_S) - k))]


def energy_above(rows, baseline, lo, hi):
    """Trapezoidal, interpolated boundaries; no extrapolation or gaps > 1.5 s."""
    if baseline is None or hi <= lo:
        return None
    rows = sorted((r['t'], r['battery_w']) for r in rows if r.get('battery_w') is not None)
    total, covered = 0., 0.
    for (a, wa), (b, wb) in zip(rows, rows[1:]):
        l, h = max(a, lo), min(b, hi)
        if h <= l:
            continue
        if b-a > 3 * BATTERY_S:
            return None
        wl, wh = wa+(wb-wa)*(l-a)/(b-a), wa+(wb-wa)*(h-a)/(b-a)
        total += ((wl+wh)/2-baseline)*(h-l)
        covered += h-l
    return total if abs(covered-(hi-lo)) < 1e-6 else None


def mean_power(rows, windows):
    energy = [energy_above(rows, 0., a, b) for a, b in windows if b > a]
    seconds = sum(b-a for a, b in windows if b > a)
    return sum(energy)/seconds if seconds and energy and None not in energy else None


def battery_sample(shell, t0):
    start = time.monotonic()-t0
    try:
        out = shell.run(f'cat {cr.BATTERY}/current_now {cr.BATTERY}/voltage_now {cr.BATTERY}/status')
        if len(out) != 3:
            raise ValueError(f'expected 3 battery fields: {out!r}')
        return dict(t=(start+time.monotonic()-t0)/2, t_start=start, t_end=time.monotonic()-t0,
                    battery_w=-int(out[0])*int(out[1])/1e12, battery_status=out[2])
    except (RuntimeError, ValueError) as e:
        return dict(t=time.monotonic()-t0, error=str(e))


def parse_stat(text):
    pid = int(text.split(' ', 1)[0])
    fields = text.rsplit(')', 1)[1].split()
    return dict(pid=pid, ticks=int(fields[11])+int(fields[12]), born_ticks=int(fields[19]))


def cpu_snapshot(server=None):
    """One root query, records identities so PID reuse never becomes negative CPU time."""
    script = (f'echo runner; cat /proc/{os.getpid()}/stat; echo llama_server; ' +
              (f'cat /proc/{server.proc.pid}/stat; ' if server else '') +
              'echo robotcam_app; for p in $(pidof com.pixelrobot.robotcam); do cat /proc/$p/stat; done; '
              'echo camera_provider; ps -A -o PID,NAME | grep android.hardware.camera.provider | '
              'while read p n; do cat /proc/$p/stat; done')
    rc, out = cr.root(script, 'duty_cpu')
    result = dict(t=time.monotonic(), groups={n: [] for n in
                  ('runner', 'llama_server', 'robotcam_app', 'camera_provider')}, errors=[])
    group = None
    for line in out.splitlines():
        if line in result['groups']:
            group = line
        elif group:
            try:
                result['groups'][group].append(parse_stat(line))
            except (ValueError, IndexError):
                result['errors'].append(line)
    if rc or not result['groups']['runner'] or server and not result['groups']['llama_server']:
        result['errors'].append(f'CPU query rc={rc} or required process missing')
    return result


def cpu_seconds(snapshots, boot_start):
    hz = os.sysconf('SC_CLK_TCK')
    totals = {}
    for group in ('runner', 'llama_server', 'robotcam_app', 'camera_provider'):
        pids = {}
        for snapshot in snapshots:
            for row in snapshot['groups'][group]:
                key = row['pid'], row['born_ticks']
                pids.setdefault(key, []).append(row['ticks'])
        totals[group] = sum(max(v)-(0 if born/hz >= boot_start else min(v))
                            for (pid, born), v in pids.items())/hz
    return totals


def build_detector(nospin=False):
    if not nospin:
        return pm.build_detector('mid')
    before = pm.thread_ids()
    opts = dp.ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 2, 1
    accepted = {}
    for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
        opts.add_session_config_entry(key, '0')  # rejection propagates: INCOMPLETE, no substitution
        accepted[key] = opts.get_session_config_entry(key) == '0'
        if not accepted[key]:
            raise RuntimeError(f'ORT did not retain {key}=0')
    detector = dp.Detector.__new__(dp.Detector)
    detector.sessions = {size: dp.ort.InferenceSession(path, sess_options=opts) for size, path in dp.MODELS.items()}
    for session in detector.sessions.values():
        for key in accepted:
            if session.get_session_options().get_session_config_entry(key) != '0':
                raise RuntimeError(f'ORT session did not retain {key}=0')
    workers = sorted(pm.thread_ids()-before)
    if len(workers) != 2:
        raise RuntimeError(f'cannot identify ORT MID workers: {workers}')
    for tid in workers:
        os.sched_setaffinity(tid, {4, 5})
        if os.sched_getaffinity(tid) != {4, 5}:
            raise RuntimeError(f'ORT worker {tid} pinning failed')
    wrapped = pm.PinnedDetector(detector, {4, 5})
    return wrapped, dict(version=dp.ort.__version__, setting='mid', caller_tid=wrapped.tid,
                         worker_tids=workers, spinning_accepted=accepted,
                         intra_op_num_threads={str(s): 2 for s in detector.sessions})


def stop_camera():
    began = time.monotonic()
    try:
        cr.camera_stop()
    finally:
        result = cr.camera_end_check()
    return dict(stop_to_stopped_s=time.monotonic()-began, **result)


def load_once(ctx, t0, index):
    pages = cr.resident_pages([cr.MODEL])
    before = cr.meminfo_mib()['mem_available_mib']
    began = time.monotonic()-t0
    server = None
    try:
        server = cr.Server(ctx['out'] / f'load_{time.time_ns()}.log')
        health_at = time.monotonic()-t0
        health_cpu = cpu_snapshot(server)
        ctx['cpu'].append(health_cpu)
        # Selector connection setup is separate from first-call CPU and latency.
        selector = cr.make_selector()
        pre_call = cpu_snapshot(server)
        case = ctx['cases'][index % len(ctx['cases'])]
        call_start = time.monotonic()-t0
        row = cr.select(selector, case)
        call_end = time.monotonic()-t0
        post_call = cpu_snapshot(server)
        ctx['cpu'].append(post_call)
        hz = os.sysconf('SC_CLK_TCK')
        ticks = lambda s: sum(r['ticks'] for r in s['groups']['llama_server'])
        return dict(index=index, resident_pages=pages, cache_state='cold' if pages == 0 else 'warm',
                    cache_note='GGUF pages only; partial residency is warm; no cache drop',
                    started_s=began, health_s=health_at, spawn_to_health_s=server.load_s,
                    first_call_ms=row['ms'], load_cpu_s=ticks(health_cpu)/hz,
                    call_cpu_s=(ticks(post_call)-ticks(pre_call))/hz,
                    cpu_errors=health_cpu['errors']+pre_call['errors']+post_call['errors'],
                    call={**row, 'expected': case['answer'], 'started_s': call_start, 'ended_s': call_end},
                    mem_before_mib=before)
    finally:
        if server is not None:
            server.stop()



def supervised_load(ctx, t0, index, limit):
    """Keep block-limit checks live during Server health waiting and the first request."""
    answer, error = [], []
    def work():
        try:
            answer.append(load_once(ctx, t0, index))
        except BaseException as e:
            error.append(e)
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    hit = None
    try:
        while thread.is_alive():
            if (hit := limit()):
                break
            thread.join(timeout=.05)
    finally:
        if hit or sys.exc_info()[0] is not None:
            # Repeat while spawn registration completes: LIVE owns even an interrupted load.
            deadline = time.monotonic()+120
            while thread.is_alive() and time.monotonic() < deadline:
                for server in list(cr.LIVE):
                    server.stop()
                thread.join(timeout=.05)
            if thread.is_alive():
                raise RuntimeError('LOAD worker still running; run aborted')
    if hit:
        return None, hit
    if error:
        raise error[0]
    return answer[0], None


def selector_phase(ctx, t0, began, length, active_before, stop, calls):
    for offset in selector_slots(active_before, length):
        at = began+offset
        if stop.wait(max(0, t0+at-time.monotonic())) or time.monotonic() >= t0+began+length:
            return
        case = ctx['cases'][len(calls) % len(ctx['cases'])]
        started = time.monotonic()-t0
        try:
            row = cr.select(ctx['selector'], case)
        except Exception as e:
            row = dict(case_id=case['id'], error=f'{type(e).__name__}: {e}')
        calls.append(dict(t=at, active_slot_s=active_before+offset, started_s=started,
                          ended_s=time.monotonic()-t0, expected=case['answer'], **row))



def camera_frame(detector, policy, session, last, t0, reads, rate=1):
    """One camera iteration; success pacing and repeat retry match power_map.frame_loop."""
    began = time.perf_counter()
    r = cr.read_frame(cr.FRAME_DIR, session=session)
    row = dict(t=time.monotonic()-t0, status=r['status'])
    if r['status'] == 'ok':
        row.update(frame=r['frame'], age_s=r['age_s'])
        if last is not None and r['frame'] <= last:
            row['status'] = 'repeat'
        else:
            last = r['frame']
            if detector:
                size, a = pm.next_size(policy), time.perf_counter()
                d = detector.detect(r['image'], size)
                row.update(size=size, detect_ms=(time.perf_counter()-a)*1000, n_detections=len(d))
    reads.append(row)
    if row['status'] == 'ok':
        # power_map.frame_loop: time spent reading/decoding/detecting consumes the period.
        time.sleep(max(0.0, began + 1 / rate - time.perf_counter()))
    else:
        time.sleep(.02 if row['status'] == 'repeat' else .1)
    return last


def drain_selector(thread, stop, ctx, record, name, limit, hit):
    """No block retry or exit until its selector has finished, including failed limit checks."""
    stop.set()
    began = time.monotonic()
    original_error = sys.exc_info()[1]
    error = original_error
    try:
        deadline = began+120
        while thread.is_alive() and time.monotonic() < deadline:
            try:
                if error is None and not hit and (hit := limit()):
                    for server in list(cr.LIVE):
                        server.stop()
                thread.join(timeout=.1)
            except BaseException as e:
                error = e if isinstance(e, (SystemExit, KeyboardInterrupt)) else error or e
    finally:
        # A drain timeout cancels the HTTP request via server shutdown.
        # Join even if shutdown or a signal raises: nothing may overlap a retry or exit.
        if thread.is_alive():
            error = error or RuntimeError('selector drain exceeded 120 s; request cancelled')
            for server in list(cr.LIVE):
                try:
                    server.stop()
                except BaseException as e:
                    error = e if isinstance(e, (SystemExit, KeyboardInterrupt)) else error or e
        while thread.is_alive():
            try:
                thread.join(timeout=.1)
            except BaseException as e:
                error = e if isinstance(e, (SystemExit, KeyboardInterrupt)) else error or e
        record['selector_drain_s'] = time.monotonic()-began
        row = dict(block=name, phase_start_s=record['start'], drain_s=record['selector_drain_s'],
                   error=f'{type(error).__name__}: {error}' if error is not None else None)
        # Discarded/interrupted blocks have no block JSON, so retain their drain evidence separately.
        with (ctx['out'] / 'selector_drains.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(f'[{name}] selector drain {row["drain_s"]:.3f}s; {row["error"] or "finished"}', flush=True)
    if error is not None and error is not original_error:
        raise error
    return hit


def run_block(spec, ctx):
    name = spec['name']
    cr.check_cores('before '+name)
    unloaded = name in ('P0U', 'LOAD')
    if unloaded and ctx.get('server'):
        ctx['server'].stop()
        ctx['server'] = None
    if not unloaded and not ctx.get('server'):
        ctx['server'] = cr.Server(ctx['out'] / f'resident_{time.time_ns()}.log')
        ctx['selector'] = cr.make_selector()
    gate = pm.thermal_gate(ctx['thermal_log'], ctx['idle'], name, ctx['smoke'])
    detector, ort = None, {}
    try:
        if any(p == 'active' for p, _, _, _ in phase_plan(spec, ctx['smoke'])):
            try:
                detector, ort = build_detector(name == 'YOLO_NOSPIN')
            except Exception as e:
                if name != 'YOLO_NOSPIN':
                    raise
                return dict(block=name, spec=spec, incomplete=[f'ORT spinning rejected: {e}'], ort={'spinning_accepted': False})
        ctx['cpu'] = [cpu_snapshot(ctx.get('server'))]
        boot_start = time.clock_gettime(time.CLOCK_BOOTTIME)
        fast, dumps, power, thread_rows, memory = [], [], [], [], []
        stop = threading.Event()
        began = t0 = time.monotonic()
        power.append(battery_sample(ctx['battery_shell'], t0))
        workers = ort.get('worker_tids', [])
        monitors = [threading.Thread(target=cr.monitor_loop, args=(period, read, rows, stop), daemon=True)
                    for period, read, rows in (
                        (1, lambda: cr.fast_sample(ctx['shell'], cr.fast_keys(ctx['layout'])), fast),
                        (5, cr.read_dump, dumps),
                        (.5, lambda: battery_sample(ctx['battery_shell'], t0), power),
                        (5, lambda: dict(t=time.monotonic()-t0, **cr.meminfo_mib()), memory),
                        (1, lambda: pm.thread_sample(t0, workers), thread_rows))]
        for thread in monitors:
            thread.start()
        phases, calls, reads, restarts, loads, issues = [], [], [], [], [], []
        hit, active_before = None, 0.
        lmk_since = time.time()
        def limit():
            cr.check_cores('during '+name)
            h = cr.block_limit(time.monotonic(), began, fast, dumps)
            return dict(limit=h[0], reason=h[1], time_to_limit_s=time.monotonic()-t0,
                        reading=cr.latest(fast, dumps)) if h else None
        try:
            # First/last samples bracket energy windows; first sensors precede workload.
            deadline = time.monotonic()+10
            while not (fast and dumps and power) and time.monotonic() < deadline:
                time.sleep(.05)
            if name == 'LOAD':
                for i in range(2):
                    due = t0 + i*(30 if ctx['smoke'] else 60)
                    while time.monotonic() < due:
                        if (hit := limit()):
                            break
                        time.sleep(.05)
                    if hit or (hit := limit()):
                        break
                    row, hit = supervised_load(ctx, t0, i, limit)
                    if hit:
                        break
                    row.update(ended_s=time.monotonic()-t0, mem_after_stop_mib=cr.meminfo_mib()['mem_available_mib'])
                    loads.append(row)
                while time.monotonic() < t0+spec['duration'] and not (hit := limit()):
                    time.sleep(.05)
            else:
                for kind, length, cycle, whole in phase_plan(spec, ctx['smoke']):
                    if (hit := limit()):
                        break
                    nominal_length = length
                    start = time.monotonic()-t0
                    if kind == 'pause' and phases and 'stop_end' in phases[-1]:
                        length = max(0., length-(start-phases[-1]['end']))
                    selector_stop, selector_thread = threading.Event(), None
                    camera_on = kind in ('active', 'camera')
                    cam, last = None, None
                    record = dict(kind=kind, start=start, cycle=cycle, whole=whole, planned_s=length, nominal_s=nominal_length, completed=False)
                    phases.append(record)
                    try:
                        if camera_on:
                            restart_at = time.monotonic()
                            try:
                                cam = pm.camera_start(1)  # reader enforces capture_boot > restart request
                                restarts.append(dict(t=start, start_to_first_fresh_s=time.monotonic()-restart_at,
                                                     retries=cam['attempts']-1, failures=0, **cam))
                            except Exception as e:
                                restarts.append(dict(t=start, failures=1, error=str(e)))
                                issues.append(f'camera restart failed: {e}')
                                break
                        if kind == 'selector' or kind == 'active' and name not in ('YOLO', 'YOLO_NOSPIN'):
                            selector_thread = threading.Thread(target=selector_phase,
                                args=(ctx, t0, start, length, active_before, selector_stop, calls), daemon=True)
                            selector_thread.start()
                        policy = pm.size_policy('5')
                        while time.monotonic() < t0+start+length:
                            if (hit := limit()):
                                break
                            if camera_on:
                                last = camera_frame(detector, policy, cam['session'], last, t0, reads, rate=1)
                            else:
                                time.sleep(.05)
                        record['completed'] = not hit and (not camera_on or cam is not None)
                    finally:
                        if selector_thread:
                            hit = drain_selector(selector_thread, selector_stop, ctx, record, name, limit, hit)
                        # Pause starts at STOP request; transition overhead is part of its power.
                        if camera_on:
                            if cam and last is not None:
                                deadline = time.monotonic()+2
                                end = cr.read_frame(cr.FRAME_DIR, session=cam['session'])
                                while not (end['status'] == 'ok' and end['frame'] > last) and time.monotonic() < deadline:
                                    time.sleep(.1)
                                    end = cr.read_frame(cr.FRAME_DIR, session=cam['session'])
                                record['survival'] = bool(cr.robotcam_pids()) and end['status'] == 'ok' and end['frame'] > last
                            record['end'] = time.monotonic()-t0
                            snapshot = cpu_snapshot(ctx.get('server'))
                            if cam and any(not snapshot['groups'][n] for n in ('robotcam_app', 'camera_provider')):
                                snapshot['errors'].append('camera CPU process missing while camera on')
                            ctx['cpu'].append(snapshot)
                            stopped = stop_camera()
                            if restarts:
                                restarts[-1]['stop'] = stopped
                            if cr.camera_end_failed(stopped):
                                issues.append('camera end: '+cr.camera_end_failed(stopped))
                            record['stop_end'] = time.monotonic()-t0
                        else:
                            record['end'] = time.monotonic()-t0
                    active_before += length if kind in ('active', 'selector') else 0
                    if hit:
                        break
            duration = time.monotonic()-t0
            ctx['cpu'].append(cpu_snapshot(ctx.get('server')))
            # A sample after the end brackets the final energy interval, but is excluded as a status measurement.
        finally:
            stop.set()
            for thread in monitors:
                thread.join(timeout=60)
            if any(t.is_alive() for t in monitors):
                raise RuntimeError('monitor still running; aborting run')
        power.append(battery_sample(ctx['battery_shell'], t0))
        for rows in (fast, dumps):
            for r in rows:
                r['t'], r['t_start'] = r['t']-t0, r['t_start']-t0
        # Attribute camera teardown to pause; retain actual phase spans and planned timing separately.
        for i, p in enumerate(phases):
            if p['kind'] == 'pause' and i and 'stop_end' in phases[i-1]:
                p['start'] = phases[i-1]['end']
        cr.check_cores('after '+name)
        return dict(block=name, spec={**spec, 'threads': 'mid'}, duration_s=duration, planned_s=spec['duration'],
                    phases=phases, loads=loads, reads=reads, selector_calls=calls, restarts=restarts,
                    fast=fast, dumps=dumps, power=power, memory=memory, thread_samples=thread_rows, ort=ort,
                    cpu_snapshots=ctx['cpu'], cpu_seconds=cpu_seconds(ctx['cpu'], boot_start),
                    thermal_start=gate, cpuinfo_max_khz=ctx['layout']['policies'], cpu_zone_note=NOTE,
                    heat_stop=dict(reached_limit=bool(hit) and hit['limit'] != 'fail_closed',
                                   **(hit or dict(limit=None, reason=None, time_to_limit_s=None))),
                    server_alive=ctx['server'].alive() if ctx.get('server') else None,
                    lmk=cr.lmk_lines(lmk_since), incomplete=issues)
    finally:
        if isinstance(detector, pm.PinnedDetector):
            detector.close()
        del detector
        gc.collect()


def block_problems(b):
    bad = list(b.get('incomplete', []))
    if 'duration_s' not in b:
        return bad or ['block did not run']
    duration = b['duration_s']
    heat = b['heat_stop']
    if heat['limit'] == 'fail_closed':
        bad.append('stopped fail-closed: '+heat['reason'])
    if heat['reached_limit'] and heat['time_to_limit_s'] < 1:
        bad.append('not run: limit at start')
    if not b['lmk']['ok']:
        bad.append('LMK query failed')
    power = [r for r in b['power'] if 0 <= r['t'] <= duration]
    if not power or any(r.get('battery_status') != 'Discharging' or 'error' in r for r in power):
        bad.append('battery missing/error or not Discharging')
    if mean_power(b['power'], [(max(0, b['power'][0]['t']), duration)]) is None:
        bad.append('battery energy coverage missing or gap >1.5 s')
    if not b.get('memory'):
        bad.append('MemAvailable sample missing')
    if any(r['errors'] for r in b['cpu_snapshots']):
        bad.append('per-process CPU query error')
    if b['block'] not in ('P0U', 'LOAD') and b['server_alive'] is not True:
        bad.append('llama-server did not survive')
    if not any(0 <= r['t'] <= duration for r in b['fast']):
        bad.append('no cpufreq samples')
    for p, cap in pm.capped_by_policy(b).items():
        if p not in b['cpuinfo_max_khz'] or cap['unknown_s']:
            bad.append(p+' scaling_max missing')
    if pm.skin_slope(b['dumps'], duration) is None:
        bad.append('skin slope missing')
    if b['ort']:
        _, errors = pm.observed_workers(b)
        bad.extend('ORT verification: '+e for e in errors)
    failed = [r for r in b['reads'] if r['status'] not in ('ok', 'repeat')]
    if failed:
        bad.append(f'{len(failed)} failed camera reads')
    for phase in b['phases']:
        if phase['kind'] == 'pause' and phase.get('planned_s') == 0:
            bad.append('camera stop/check consumed the entire pause')
        if phase['kind'] in ('active', 'camera'):
            processed = [r for r in b['reads'] if phase['start'] <= r['t'] <= phase['end'] and r['status'] == 'ok']
            if not processed or phase.get('survival') is not True:
                bad.append('camera phase has no frames or failed end survival')
            if phase['kind'] == 'active' and not heat['reached_limit']:
                for size in (320, 640):
                    if not any(r.get('size') == size for r in processed):
                        bad.append(f'camera phase missing {size} detection')
    calls = b['selector_calls']
    for c in calls:
        phase = next((p for p in b['phases'] if p['start'] <= c['started_s'] < p['end']), None)
        if 'error' in c or c['started_s']-c['t'] > 5 or phase is None or phase['kind'] not in ('active', 'selector') or c['ended_s'] > phase['end']:
            bad.append('selector error, late slot or call outside active phase')
    if b['block'] in ('SEL', 'CONT', 'CYC50', 'CYC25', 'HEATCOOL'):
        expected = sum(len(selector_slots(sum(p['planned_s'] for p in b['phases'][:i] if p['kind'] in ('active', 'selector')),
                                          p['planned_s'])) for i, p in enumerate(b['phases']) if p['kind'] in ('active', 'selector'))
        if not heat['reached_limit'] and len(calls) != expected:
            bad.append(f'selector cadence missed: {len(calls)}/{expected}')
    if b['block'] == 'LOAD':
        if len(b['loads']) != 2:
            bad.append('LOAD missing reloads (need 2)')
        for row in b['loads']:
            if row['cpu_errors']:
                bad.append('LOAD CPU measurements missing')
            if row['call'].get('correct') is None:
                bad.append('LOAD first-call result missing')
            if row['ended_s'] > b['planned_s']:
                bad.append('LOAD overran planned window')
    return bad


def window_block(b, lo, hi):
    return {**b, 'duration_s': hi-lo,
            'fast': [{**r, 't': r['t']-lo} for r in b['fast'] if lo <= r['t'] <= hi],
            'dumps': [{**r, 't': r['t']-lo} for r in b['dumps'] if lo <= r['t'] <= hi]}


def report(out):
    run = json.loads((out / 'run.json').read_text())
    blocks, bad = {}, []
    for spec in run['blocks']:
        path = out / f'block_{spec["name"]}.json'
        if not path.exists():
            bad.append(spec['name']+': block missing')
            continue
        b = json.loads(path.read_text())
        blocks[b['block']] = b
        bad.extend(b['block']+': '+p for p in block_problems(b))
    means = {n: mean_power(b['power'], [(b['power'][0]['t'], b['duration_s'])])
             for n, b in blocks.items() if 'duration_s' in b}
    baselines = {n: means.get(n) if not block_problems(blocks[n]) else None for n in ('P0', 'P0U') if n in blocks}
    lines = [f'DUTY1 {out.name} set {run["set"]}'+(' [SMOKE: not a measurement]' if run['smoke'] else ''),
             *['INCOMPLETE: '+p for p in bad],
             'Battery: current_now x voltage_now, sampled every 0.5 s; fuel-gauge averaging is unverified.',
             'Energy: derived trapezoidal integral with interpolated edges; missing coverage/gap >1.5 s = n/a.',
             'Block mean excludes the initial sampler startup sliver before the first battery reading.',
             'Capped s: POWERMAP later-reading convention, skipped gaps belong to later sample, no tail credit.',
             'zone9/10/11: '+NOTE, 'Skin slope: least-squares in-block VIRTUAL-SKIN C/min.',
             'Selector cadence: 20 s accumulated planned active time; first slot zero; pauses add no time.',
             'Actual active spans include camera startup and drained calls; stop/check overhead belongs to pause.',
             'CPU seconds: utime+stime snapshots at block boundaries and before process stops; newly born PIDs count from zero.',
             'Server: '+' '.join(run.get('server_cmd', []))]
    fmt = lambda v: cr.f(v, '.3f')
    for name, b in blocks.items():
        lines += ['', name]
        if 'duration_s' not in b:
            lines.append('ORT spinning accepted: '+str(b.get('ort')))
            continue
        duration = b['duration_s']
        calls = b['selector_calls']
        lines += [f'actual/planned s: {duration:.3f}/{b["planned_s"]}',
                  'mean battery W: '+fmt(means[name]),
                  'CPU seconds: '+json.dumps(b['cpu_seconds']),
                  'gate: '+json.dumps(b['thermal_start'])+'; '+NOTE,
                  'heat stop: '+json.dumps(b['heat_stop'])+'; '+NOTE,
                  'ORT: '+json.dumps(b['ort']),
                  'MemAvailable samples: '+json.dumps(b.get('memory', [])),
                  'selector ok/err/correct: '+str(sum('ms' in c for c in calls))+'/'+str(sum('error' in c for c in calls))+'/'+str(sum(c.get('correct', False) for c in calls)),
                  'Android status changes: '+pm.status_changes(b),
                  'LMK: '+json.dumps(b['lmk'])]
        ds = [d['skin'] for d in b['dumps'] if 0 <= d['t'] <= duration and d.get('skin') is not None]
        lines += ['skin start/end/max C: '+('/'.join(fmt(v) for v in (ds[0], ds[-1], max(ds))) if ds else 'n/a'),
                  'skin slope C/min: '+fmt(pm.skin_slope(b['dumps'], duration))]
        for size in (320, 640):
            detected = [r for r in b['reads'] if r.get('size') == size and 'detect_ms' in r]
            if detected:
                ms = [r['detect_ms'] for r in detected]
                drift = [cr.med([r['detect_ms'] for r in detected if r['t'] < 30]),
                         cr.med([r['detect_ms'] for r in detected if r['t'] >= duration-30])]
                lines.append(f'detect{size} ms median/P95, first/last 30s medians: '+json.dumps([cr.med(ms), cr.p95(ms), drift]))
        for p, cap in pm.capped_by_policy(b).items():
            lines.append(f'{p} capped s/% / lowest scaling_max MHz / unknown s: '+json.dumps(cap))
        base = 'P0U' if name in ('P0U', 'LOAD') else 'P0'
        baseline = baselines.get(base)
        lines.append(f'derived cost above base {base} W: '+fmt(means[name]-baseline if means[name] is not None and baseline is not None else None))
        for c in calls:
            lines.append(f'derived selector energy J above P0, {c["case_id"]} at {c["started_s"]:.3f}s: '+
                         fmt(energy_above(b['power'], baselines.get('P0'), c['started_s'], c['ended_s'])))
        for row in b['loads']:
            lines.append('LOAD: '+json.dumps(row))
            lines.append('derived LOAD energy J above P0U: '+fmt(energy_above(b['power'], baselines.get('P0U'), row['started_s'], row['ended_s'])))
        phases = b['phases']
        if phases:
            active = [(p['start'], p['end']) for p in phases if p['kind'] == 'active']
            pause = [(p['start'], p['end']) for p in phases if p['kind'] == 'pause']
            full = [p for p in phases if p['whole'] and p.get('completed') and 'end' in p]
            windows = [(p['start'], p['end']) for p in full]
            if name in ('CYC50', 'CYC25'):
                cycles = {p['cycle'] for p in full}
                windows = [(min(p['start'] for p in full if p['cycle'] == c), max(p['end'] for p in full if p['cycle'] == c))
                           for c in sorted(cycles) if {p['kind'] for p in full if p['cycle'] == c} == {'active', 'pause'}
]
            lines += ['whole-cycle mean W (complete cycles only): '+fmt(mean_power(b['power'], windows)),
                      'active mean W: '+fmt(mean_power(b['power'], active)),
                      'pause mean W: '+fmt(mean_power(b['power'], pause)),
                      'phases: '+json.dumps(phases)]
            expected = sum(p['planned_s'] for p in phases if p['kind'] in ('active', 'camera'))
            lines.append(f'frames processed/expected at 1/s: {sum(r["status"] == "ok" for r in b["reads"])}/{expected}')
        for r in b['restarts']:
            lines.append('camera restart: '+json.dumps(r))
        if name == 'HEATCOOL' and len(phases) == 2:
            cool = phases[1]
            lo, hi = cool['start'], cool['end']
            skin = next((d['skin'] for d in reversed(b['dumps']) if d['t'] <= lo and d.get('skin') is not None), None)
            lines.append('COOL starting skin C (latest preceding dump): '+fmt(skin))
            for t in range(0, math.ceil(hi-lo), 5):
                a, z = lo+t, min(hi, lo+t+5)
                d = next((r for r in b['dumps'] if a <= r['t'] < z), {})
                f = next((r for r in b['fast'] if a <= r['t'] < z), {})
                lines.append(f'COOL {t}s: skin={d.get("skin", "not recorded")} power_W={fmt(mean_power(b["power"], [(a,z)]))} scaling_max_khz={f.get("max", "not recorded")}')
            for t in range(0, math.ceil(hi-lo), 30):
                w = window_block(b, lo+t, min(hi, lo+t+30))
                lines.append(f'COOL {t}-{min(hi-lo,t+30):.1f}s slope C/min: '+fmt(pm.skin_slope(w['dumps'], w['duration_s'])))
    return '\n'.join(lines)+'\n', bad


def completed(out, spec):
    # HEATCOOL has one atomic JSON: an interruption during COOL redoes HEAT too.
    return (out / f'block_{spec["name"]}.json').exists()


def save_block(out, spec, b):
    path = out / f'block_{spec["name"]}.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(b)+'\n')
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--set', choices=('PARTS', 'CYCLE'), required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--thermal-log', default=str(OUT_ROOT / 'thermal.log'))
    a = ap.parse_args(argv)
    cr.require_native()  # before root, Android or server, including --resume
    specs = blocks_for(a.set, a.smoke)
    lower = 360+sum(s['duration'] for s in specs)
    upper = lower+(0 if a.smoke else 480*(len(specs)+1))
    print('Blocks: '+', '.join(f'{s["name"]} {s["duration"]}s' for s in specs), flush=True)
    print(f'Estimated wall time {lower/60:.1f}-{upper/60:.1f} min + loads/warm-up, camera checks and call drain; includes launcher 5 min idle and 60 s post-warm-up wait.', flush=True)
    if a.resume:
        out = a.resume
        run = json.loads((out / 'run.json').read_text())
        if (run['set'], run['blocks'], run['smoke']) != (a.set, specs, a.smoke):
            raise SystemExit('--resume: set/smoke mismatch')
    else:
        out = OUT_ROOT / f'run_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{a.set}{"_smoke" if a.smoke else ""}'
        out.mkdir(parents=True)
    ctx = dict(out=out, smoke=a.smoke, thermal_log=a.thermal_log, server=None, shell=None, battery_shell=None)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cr.exit_on_signal)
    try:
        rc, who = cr.root('id', 'id')
        if rc:
            raise SystemExit('su failed: '+who)
        cr.wait_cores('start')
        ctx['shell'], ctx['battery_shell'] = cr.RootShell(), cr.RootShell()
        ctx['layout'] = cr.discover(ctx['shell'])
        if not set(pm.POLICIES) <= set(ctx['layout']['policies']):
            raise SystemExit('layout lacks policy0/4/6')
        ctx['idle'] = run['idle'] if a.resume else {**cr.read_thermal(a.thermal_log), 'skin': cr.read_dump()['skin']}
        if ctx['idle']['skin'] is None:
            raise SystemExit('no idle VIRTUAL-SKIN')
        meta = dict(set=a.set, smoke=a.smoke, blocks=specs, idle=ctx['idle'], layout=ctx['layout'],
                    started=cr.utc(), server_cmd=cr.server_cmd(), sample_interval_s=.5,
                    sha256={str(p): cr.sha256(p) for p in (Path(__file__), Path(pm.__file__), Path(cr.__file__), Path(dp.__file__), *map(Path, dp.MODELS.values()))})
        (out / (f'resume_{time.time_ns()}.json' if a.resume else 'run.json')).write_text(json.dumps(meta, indent=2))
        if any(not completed(out, s) for s in specs):
            pm.thermal_gate(a.thermal_log, ctx['idle'], 'run-start warm-up', a.smoke)
            ctx['server'] = cr.Server(out / f'warmup_{time.time_ns()}.log')
            ctx['selector'], ctx['cases'] = cr.make_selector(), cr.load_cases()
            ctx['selector'].decide(*cr.WARMUP)
            print('Post-warm-up: waiting 60 s unrecorded', flush=True)
            time.sleep(60)
            cr.check_cores('after post-warm-up wait')
            if why := cr.camera_end_failed(stop_camera()):
                raise RuntimeError('initial camera stop: '+why)
        for spec in specs:
            if completed(out, spec):
                print(spec['name']+': already completed, kept', flush=True)
                continue
            while True:
                cr.wait_cores(spec['name'])
                try:
                    b = run_block(spec, ctx)
                    break
                except cr.CoresLost as e:
                    print(f'{e}; discard and redo {spec["name"]}', flush=True)
                    stop_camera()
                    if ctx.get('server') and not ctx['server'].alive():
                        ctx['server'].stop()
                        ctx['server'] = None
            save_block(out, spec, b)
        text, bad = report(out)
        (out / 'report.txt').write_text(text)
        print(text, flush=True)
        if cr.DOWNLOADS.is_dir():
            (cr.DOWNLOADS / f'duty_cycle_{out.name}_report.txt').write_text(text)
        if bad:
            raise SystemExit('RUN INCOMPLETE: '+'; '.join(bad))
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            for server in list(cr.LIVE):
                server.stop()
        finally:
            try:
                why = cr.camera_end_failed(stop_camera())
                if why:
                    print('final camera cleanup INCOMPLETE: '+why, file=sys.stderr)
                    raise RuntimeError(why)
            finally:
                for key in ('shell', 'battery_shell'):
                    if ctx.get(key):
                        ctx[key].close()


if __name__ == '__main__':
    main()
