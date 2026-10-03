#!/data/data/com.termux/files/usr/bin/python
"""YOLO power map, DECISIONS #127. Native Termux, motors off; launch with oneshot.sh."""
import argparse
import gc
import json
import math
import os
import queue
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'coresidency'))
import coresidency as cr
import detect_person as dp

OUT_ROOT = cr.HOME / 'power_map'
POLICIES = ('policy0', 'policy4', 'policy6')
GATE_MAX_S = 8 * 60


def blocks_for(run_set, rate=None, cadence=None, threads=None):
    if run_set in ('R1', 'R2'):
        if any(v is not None for v in (rate, cadence, threads)):
            raise ValueError('R1/R2 take no --rate, --cadence or --threads')
        rate = 2 if run_set == 'R1' else 1
        return [{'name': f'{run_set}_r{rate}_c{c}' if c != 'off' else f'{run_set}_r{rate}_off',
                 'rate': rate, 'cadence': c, 'threads': 'default'} for c in ('5', '10', 'off')]
    if rate not in (1, 2) or cadence not in ('5', '10', 'off'):
        raise ValueError('R3/CONFIRM require --rate {1,2} --cadence {5,10,off}')
    if run_set == 'R3' and threads is None:
        return [{'name': f'R3_{t}', 'rate': rate, 'cadence': cadence, 'threads': t}
                for t in ('default', 'mid', 'little')]
    if run_set == 'CONFIRM' and threads in ('default', 'mid', 'little'):
        return [{'name': 'CONFIRM', 'rate': rate, 'cadence': cadence, 'threads': threads}]
    raise ValueError('R3 takes no --threads; CONFIRM requires --threads {default,mid,little}')


def size_policy(cadence):
    return None if cadence == 'off' else cr.SizePolicy(interval_s=int(cadence))


def next_size(policy, now=None):
    return 320 if policy is None else policy.next_size(now)


def thermal_gate(path, idle, label, smoke):
    """Same skin gate, without the unrelated CPU-core term; bounded to eight minutes."""
    began = time.monotonic()
    while True:
        thermal, dump = cr.read_thermal(path), cr.read_dump()
        cool = dump['skin'] is not None and dump['skin'] <= idle['skin'] + cr.SKIN_GATE_C
        waited = time.monotonic() - began
        if smoke or cool or waited >= GATE_MAX_S:
            return {**thermal, 'skin': dump['skin'], 'status': dump['status'],
                    'waited_s': round(waited), 'warm_start': not (smoke or cool)}
        print(f'[{label}] waiting: skin {dump["skin"]}, need <= {idle["skin"] + cr.SKIN_GATE_C:.1f}; '
              f'{waited:.0f} s', flush=True)
        time.sleep(cr.GATE_POLL_S)


def thread_ids():
    return {int(p.name) for p in Path(f'/proc/{os.getpid()}/task').iterdir()}


class PinnedDetector:
    """One detector-only calling thread: ORT includes its caller in the intra-op count."""
    def __init__(self, detector, cpus):
        self.detector = detector
        self.requests, self.answers = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(target=self.loop, args=(cpus,), daemon=True)
        self.thread.start()
        ok, result = self.answers.get()
        if not ok:
            self.thread.join()
            raise result
        self.tid = result

    def loop(self, cpus):
        try:
            tid = threading.get_native_id()
            os.sched_setaffinity(tid, cpus)
            if os.sched_getaffinity(tid) != cpus:
                raise RuntimeError('detector caller affinity did not take effect')
            self.answers.put((True, tid))
        except BaseException as e:
            self.answers.put((False, e))
            return
        while (request := self.requests.get()) is not None:
            try:
                self.answers.put((True, self.detector.detect(*request)))
            except BaseException as e:
                self.answers.put((False, e))

    def detect(self, image, size):
        self.requests.put((image, size))
        ok, result = self.answers.get()
        if not ok:
            raise result
        return result

    def close(self):
        self.requests.put(None)
        self.thread.join(timeout=120)
        if self.thread.is_alive():
            raise RuntimeError('detector calling thread still running')


def build_detector(setting):
    """Only newly created session workers are pinned; never set the process affinity.
    A detector-only calling thread is pinned too, since it participates in intra-op work.
    """
    before = thread_ids()
    if setting == 'default':
        detector = dp.Detector()  # exactly the deployed constructor, no session options
    else:
        opts = dp.ort.SessionOptions()
        opts.intra_op_num_threads = 2 if setting == 'mid' else 4
        opts.inter_op_num_threads = 1
        detector = dp.Detector.__new__(dp.Detector)
        detector.sessions = {size: dp.ort.InferenceSession(path, sess_options=opts)
                             for size, path in dp.MODELS.items()}
    workers = sorted(thread_ids() - before)
    if setting != 'default':
        cpus = {4, 5} if setting == 'mid' else {0, 1, 2, 3}
        expected = 2 * ((2 if setting == 'mid' else 4) - 1)
        if len(workers) != expected:
            raise RuntimeError(f'cannot identify ORT workers: expected {expected}, got {workers}')
        for tid in workers:
            os.sched_setaffinity(tid, cpus)
            if os.sched_getaffinity(tid) != cpus:
                raise RuntimeError(f'ORT worker {tid} affinity did not take effect')
    caller_tid = None
    if setting != 'default':
        detector = PinnedDetector(detector, cpus)
        caller_tid = detector.tid
    sessions = detector.sessions if setting == 'default' else detector.detector.sessions
    info = {'caller_tid': caller_tid, 'version': dp.ort.__version__, 'setting': setting, 'worker_tids': workers,
            'intra_op_num_threads': {str(size): s.get_session_options().intra_op_num_threads
                                     for size, s in sessions.items()},
            'observed_default_intra_threads_per_session': len(workers) // 2 + 1 if setting == 'default' else None,
            'default_count_note': 'observed created workers / two sessions + one calling thread; option 0 = auto'}
    return detector, info


def thread_sample(t0, workers):
    rows = []
    for tid in sorted(thread_ids()):
        try:
            base = Path(f'/proc/{os.getpid()}/task/{tid}')
            # comm may contain spaces or parentheses; processor is stat field 39.
            stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
            status = (base / 'status').read_text().splitlines()
            allowed = next(l.split(':', 1)[1].strip() for l in status if l.startswith('Cpus_allowed_list:'))
            rows.append({'tid': tid, 'ort_worker': tid in workers, 'processor': int(stat[36]),
                         'allowed': allowed, 'cpu_ticks': int(stat[11]) + int(stat[12])})
        except (OSError, ValueError, IndexError, StopIteration) as e:
            rows.append({'tid': tid, 'ort_worker': tid in workers, 'error': str(e)})
    return {'t': round(time.monotonic() - t0, 3), 'threads': rows}


def camera_start(rate):
    """main.Robot.start_camera, retried: overlapping opens on restart can need a retry (DECISIONS #122)."""
    for attempt in range(1, 4):
        started_boot_s = time.clock_gettime(time.CLOCK_BOOTTIME)
        cr.am('start', '-n', 'com.pixelrobot.robotcam/.StartActivity', '--es', 'mode', 'B', '--ei', 'rate', str(rate))
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            r = cr.read_frame(cr.FRAME_DIR, min_capture_boot_s=started_boot_s)
            if r['status'] == 'ok':
                return {'session': r['session'], 'frame': r['frame'], 'attempts': attempt}
            time.sleep(0.1)
        cr.camera_stop()
        time.sleep(2)
    raise RuntimeError('RobotCam did not publish a usable frame in 3 attempts')


def frame_loop(detector, spec, session, t0, duration, reads, limit):
    """Reads every new frame and detects on it until the block ends or limit() returns a stop record.
    Returns (last frame number processed, stop record or None)."""
    policy = size_policy(spec['cadence'])
    last = None
    while time.monotonic() < t0 + duration:
        # ponytail: checked between frames, so a stop comes up to one detection (~1.5 s) plus the 1 s / 5 s
        # reading interval after the sensor crossed; a watcher thread could cut the frame part
        if (hit := limit()) is not None:
            return last, hit
        cr.check_cores('during the block')
        cr._Stamp.at = None
        a = time.perf_counter()
        r = cr.read_frame(cr.FRAME_DIR, session=session)
        b = time.perf_counter()
        row = {'t': round(time.monotonic() - t0, 3), 'status': r['status']}
        if cr._Stamp.at is not None:
            row['read_ms'] = round((cr._Stamp.at - a) * 1000, 2)
            row['decode_ms'] = round((b - cr._Stamp.at) * 1000, 2)
        if r['status'] == 'ok':
            row.update(frame=r['frame'], age_s=round(r['age_s'], 3))
            if last is not None and r['frame'] <= last:
                row['status'] = 'repeat'
        reads.append(row)
        if row['status'] != 'ok':
            time.sleep(0.02 if row['status'] == 'repeat' else 0.1)
            continue
        last = r['frame']
        size = next_size(policy)
        c = time.perf_counter()
        detections = detector.detect(r['image'], size)
        row.update(size=size, detect_ms=round((time.perf_counter() - c) * 1000, 2), n_detections=len(detections))
        # Pace at the selected camera rate; repeat frames retry in 20 ms.
        time.sleep(max(0.0, a + 1 / spec['rate'] - time.perf_counter()))
    return last, None

def run_block(spec, ctx):
    """One block; CoresLost makes the caller discard and redo it."""
    name, mode, gemma = spec['name'], 'mix', True
    cr.check_cores('before ' + name)
    therm_start = thermal_gate(ctx['thermal_log'], ctx['idle'], name, ctx['smoke'])
    print(f'[{name}] start: skin {therm_start["skin"]} degC, z9 {therm_start["z9"] / 1000:.1f} degC (CPU-core reading, not a heat state; waited '
          f'{therm_start["waited_s"]} s{", WARM START" if therm_start["warm_start"] else ""})', flush=True)
    server = ctx['server'] if gemma else None
    if gemma and not server.alive():
        raise RuntimeError('llama-server is not healthy at block start')
    stop_mon, fast, dumps, keys = threading.Event(), [], [], cr.fast_keys(ctx['layout'])
    monitors = [threading.Thread(target=cr.monitor_loop, args=(cr.FAST_S, lambda: cr.fast_sample(ctx['shell'], keys), fast,
                                                            stop_mon), daemon=True),
                threading.Thread(target=cr.monitor_loop, args=(cr.DUMP_S, cr.read_dump, dumps, stop_mon), daemon=True)]
    for t in monitors:
        t.start()
    began = time.monotonic()
    threads = []
    try:
        while not (fast and dumps) and time.monotonic() - began < 10:  # first readings in before the first frame
            time.sleep(0.05)
        lmk_since = time.time()
        cam = camera_start(spec['rate'])
        t0 = time.monotonic()
        duration = ctx['duration']

        def limit():
            now = time.monotonic()
            hit = cr.block_limit(now, began, fast, dumps)
            return hit and {'limit': hit[0], 'reason': hit[1], 'time_to_limit_s': round(now - t0, 2),
                            'reading': cr.latest(fast, dumps)}
        stop, samples, calls, reads = threading.Event(), [], [], []
        thread_rows = []
        threads = [threading.Thread(target=cr.sampler_loop, args=(t0, ctx['thermal_log'], server, stop, samples),
                                    daemon=True)]
        if gemma:
            threads.append(threading.Thread(target=cr.selector_loop, args=(ctx['selector'], ctx['cases'], t0, duration,
                                                                        stop, calls), daemon=True))
        if ctx['observe_threads']:
            threads.append(threading.Thread(target=cr.monitor_loop,
                           args=(1, lambda: thread_sample(t0, ctx['ort']['worker_tids']), thread_rows, stop),
                           daemon=True))
        for t in threads:
            t.start()
        try:
            last, hit = frame_loop(ctx['detector'], spec, cam['session'], t0, duration, reads, limit)
            elapsed = time.monotonic() - t0
            stop.set()  # a limit stop ends the selector's cadence here too (no slot starts after the block)
            deadline = time.monotonic() + 2  # survival: a newer frame within 2 s (rate 2)
            while True:
                end = cr.read_frame(cr.FRAME_DIR, session=cam['session'])
                if (end['status'] == 'ok' and last is not None and end['frame'] > last) or time.monotonic() > deadline:
                    break
                time.sleep(0.1)
            robotcam_pid = cr.robotcam_pids()
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=120)
            try:
                cr.camera_stop()
            finally:  # also on CoresLost/abort or a failed broadcast: the next block or run starts from a stopped app
                cam_end = cr.camera_end_check()
    finally:
        stop_mon.set()
        for t in monitors:
            t.join(timeout=60)
    if any(t.is_alive() for t in threads + monitors):  # e.g. a stalled selector request: it would overlap the next block
        raise RuntimeError(f'{name}: a sampler/selector/monitor thread still running after the block; run stopped')
    cr.check_cores('after ' + name)
    therm_end = cr.read_thermal(ctx['thermal_log'])
    for r in fast + dumps:  # monotonic -> s from the block start (readings before the camera start are negative)
        r['t'], r['t_start'] = round(r['t'] - t0, 3), round(r['t_start'] - t0, 3)
    return {'cpu_zone_note': 'CPU-core reading, not a heat state', 'block': name, 'spec': spec, 'ort': ctx['ort'], 'thread_samples': thread_rows, 'mode': mode, 'gemma': gemma, 'duration_s': round(elapsed, 2), 'planned_s': duration,
            'heat_stop': {'reached_limit': bool(hit) and hit['limit'] != 'fail_closed',
                          **(hit or {'limit': None, 'reason': None, 'time_to_limit_s': None, 'reading': None})},
            'camera': cam, 'camera_end': cam_end, 'thermal_start': therm_start, 'thermal_end': therm_end,
            'cpuinfo_max_khz': ctx['layout']['policies'], 'fast': fast, 'dumps': dumps,
            'survived': {'robotcam_process': bool(robotcam_pid), 'robotcam_pids': robotcam_pid,
                         'robotcam_new_frame_at_end': end['status'] == 'ok' and last is not None and end['frame'] > last,
                         'robotcam_end_status': end['status'],
                         'llama_server': server.alive() if server else None},
            'lmk': cr.lmk_lines(lmk_since), 'samples': samples, 'selector_calls': calls, 'reads': reads}


def problems(blocks, sums):
    """Why the run cannot count as a complete measurement (blocks are still kept: a kill is evidence)."""
    out = []
    for b, s in zip(blocks, sums):
        heat = b['heat_stop']
        if why := cr.camera_end_failed(b['camera_end']):
            out.append(f'{b["block"]}: RobotCam end check: {why}')
        if heat['reached_limit'] and s['frames'] == 0 and not s['failed']:  # met before the first frame
            r = heat['reading']
            out.append(f'{b["block"]}: not run: limit at start ({heat["reason"]}; skin {r["skin"]}, status '
                       f'{r["status"]}, battery {r["bat_c"]}, CPU max {r["cpu_c"]} degC, scaling_max {r["scaling_max"]})')
            continue
        if heat['limit'] == 'fail_closed':
            out.append(f'{b["block"]}: stopped fail-closed at {heat["time_to_limit_s"]} s: {heat["reason"]}')
        # a limit stop is a result: rules that need time only apply as far as the block ran
        # (no_sample: stopped before a second sample was due, so the first may finish after it)
        no_sample = heat['reached_limit'] and b['duration_s'] < cr.SAMPLE_S
        if b['gemma']:  # the Gemma workload is one call per 20 s slot, each inside the block
            calls = b['selector_calls']
            # slots that can start inside the block: k * cr.SELECTOR_S < planned length (selector_loop's bound; the
            # block itself runs a little past it), or before the limit stop, which also stops the selector; a slot
            # due in the last second before a limit stop is not required (the stop may beat the thread to it)
            end = heat['time_to_limit_s'] - 1 if heat['reached_limit'] else b['planned_s']
            slots = max(0, math.ceil(end / cr.SELECTOR_S))
            late = [c['t'] for c in calls if c['started_s'] - c['t'] > 5]
            # a call in flight at a limit stop ends after it by construction: not an overrun
            over = [] if heat['reached_limit'] else [c['t'] for c in calls if c['ended_s'] > b['duration_s']]
            if slots and s['sel'][0] == 0:
                out.append(f'{b["block"]}: no successful selector call ({s["sel"][1]} failed)')
            elif s['sel'][0] < slots or late or over:
                out.append(f'{b["block"]}: selector cadence missed: {s["sel"][0]}/{slots} successful calls, '
                           f'started >5 s late at slots {late}, ended after the block at slots {over}')
        if s['max_unread'][0]:
            out.append(f'{b["block"]}: {s["max_unread"][0]} 1 s sample(s) without every scaling_max_freq '
                       f'({s["max_unread"][1]:.1f} s of the block not known capped or not)')
        if s['sample_errors'] or (s['min_avail'] is None and not no_sample):
            out.append(f'{b["block"]}: {s["sample_errors"]} sample(s) with root/PSS/thermal errors, '
                       f'{0 if s["min_avail"] is None else "some"} usable samples')
        sv = b['survived']
        if s['failed'] or s['frames'] == 0 or \
                not (sv['robotcam_new_frame_at_end'] and sv['robotcam_process']):
            out.append(f'{b["block"]}: RobotCam: {s["frames"]} frames, failed reads {s["failed"] or "none"}, '
                       f'new frame at end {sv["robotcam_new_frame_at_end"]}, process at end {sv["robotcam_process"]}')
        for size in (320,) if b['spec']['cadence'] == 'off' else (320, 640):
            if size == 640 and b['mode'] == 'mix' and heat['reached_limit'] and heat['time_to_limit_s'] < int(b['spec']['cadence']):
                continue  # stopped before this size's first frame was due
            if s[f'n{size}'] == 0 or (None in s[f'drift{size}'] and not s['drift_na']):
                out.append(f'{b["block"]}: {s[f"n{size}"]} detections at {size}, drift first/last {cr.DRIFT_S} s '
                           f'{s[f"drift{size}"]} (a required measurement is missing)')
        if b['gemma'] and b['survived']['llama_server'] is not True:
            out.append(f'{b["block"]}: llama-server did not survive the block (see LMK lines)')
        if s['bat_status'] != ['Discharging'] and not (no_sample and s['bat_status'] == []):
            out.append(f'{b["block"]}: battery status {s["bat_status"]} (power needs Discharging throughout)')
        if not b['lmk']['ok']:
            out.append(f'{b["block"]}: LMK logcat query failed (rc {b["lmk"]["logcat_rc"]}): {b["lmk"]["raw_head"][:120]!r}')
    return out


def capped_by_policy(b):
    """Later in-block sample owns the gap; unreadable is unknown; no last-sample tail credit."""
    rows = [r for r in b['fast'] if 0 <= r['t'] <= b['duration_s']]
    result = {}
    for policy in POLICIES:
        capped = unknown = 0.0
        previous = 0.0
        values = []
        for row in rows:
            dt = row['t'] - previous
            previous = row['t']
            value = row['max'].get(policy)
            top = b['cpuinfo_max_khz'].get(policy)
            if value is None or top is None:
                unknown += dt
            else:
                values.append(value)
                if value < top:
                    capped += dt
        result[policy] = {'seconds': capped, 'percent': 100 * capped / b['duration_s'] if b['duration_s'] else None,
                          'unknown_s': unknown, 'lowest_mhz': min(values) / 1000 if values else None}
    return result


def skin_slope(dumps, duration):
    points = [(r['t'], r['skin']) for r in dumps if 0 <= r['t'] <= duration and r['skin'] is not None]
    if len(points) < 2:
        return None
    mt = sum(t for t, v in points) / len(points)
    mv = sum(v for t, v in points) / len(points)
    denominator = sum((t - mt)**2 for t, v in points)
    return 60 * sum((t - mt) * (v - mv) for t, v in points) / denominator if denominator else None


def observed_workers(b):
    cpus, allowed, errors = {}, {}, []
    previous_ticks, observed = {}, set()
    workers = set(b['ort']['worker_tids'])
    if b['ort'].get('caller_tid'):
        workers.add(b['ort']['caller_tid'])
    for sample in b['thread_samples']:
        if not 0 <= sample['t'] <= b['duration_s']:
            continue
        present = set()
        for row in sample['threads']:
            tid = row['tid']
            if tid not in workers:
                continue
            present.add(tid)
            if 'error' in row:
                errors.append(f'{tid}: {row["error"]}')
                continue
            observed.add(tid)
            # Idle workers retain their construction CPU: only advancing ticks are execution evidence.
            ticks = row['cpu_ticks']
            if tid in previous_ticks and ticks > previous_ticks[tid]:
                cpus.setdefault(tid, set()).add(row['processor'])
            previous_ticks[tid] = ticks
            allowed.setdefault(tid, set()).add(row['allowed'])
        if missing := workers - present:
            errors.append(f'workers missing: {sorted(missing)}')
    setting = b['spec']['threads']
    expected = {'mid': '4-5', 'little': '0-3'}.get(setting)
    if expected:
        for tid in workers:
            if allowed.get(tid) != {expected}:
                errors.append(f'{tid}: allowed {sorted(allowed.get(tid, set()))}, expected {expected}')
            target = {4, 5} if setting == 'mid' else {0, 1, 2, 3}
            if not cpus.get(tid, set()) <= target:
                errors.append(f'{tid}: observed CPUs {sorted(cpus.get(tid, set()))}, expected {sorted(target)}')
    if not workers or not workers <= observed:
        errors.append('no complete ORT worker observations')
    text = '; '.join(f'{tid}: CPUs {sorted(cpus.get(tid, set())) or "no sampled CPU tick advance"}, '
                     f'allowed {sorted(allowed.get(tid, set()))}' for tid in sorted(observed))
    return text or 'none', errors


def report(out):
    run = json.loads((out / 'run.json').read_text())
    blocks = [json.loads((out / f'block_{s["name"]}.json').read_text()) for s in run['blocks']
              if (out / f'block_{s["name"]}.json').exists()]
    sums = [cr.block_summary(b) for b in blocks]
    bad = problems(blocks, sums)
    missing = [s['name'] for s in run['blocks'] if not (out / f'block_{s["name"]}.json').exists()]
    bad += [f'{name}: block missing' for name in missing]
    observations = {}
    for b in blocks:
        if run['set'] in ('R3', 'CONFIRM'):
            observations[b['block']], errors = observed_workers(b)
            bad += [f'{b["block"]}: ORT verification: {error}' for error in errors]
        if skin_slope(b['dumps'], b['duration_s']) is None and not b['heat_stop']['reached_limit']:
            bad.append(f'{b["block"]}: skin slope missing')
        for p in POLICIES:
            if p not in b['cpuinfo_max_khz']:
                bad.append(f'{b["block"]}: {p} cpuinfo_max_freq missing')
    fmt = cr.f
    rows = [
        ('rate / cadence / threads', lambda b, s: f'{b["spec"]["rate"]} / {b["spec"]["cadence"]} / {b["spec"]["threads"]}'),
        ('frames processed', lambda b, s: str(s['frames'])),
        ('failed reads', lambda b, s: str(s['failed']) if s['failed'] else 'none'),
        ('detect320 ms median/P95 (n)', lambda b, s: f'{fmt(s["det320"][0])}/{fmt(s["det320"][1])} ({s["n320"]})'),
        ('detect640 ms median/P95 (n)', lambda b, s: f'{fmt(s["det640"][0])}/{fmt(s["det640"][1])} ({s["n640"]})'),
        ('640 drift first/last 30 s', lambda b, s: 'n/a' if s['drift_na'] or b['spec']['cadence'] == 'off' else
         f'{fmt(s["drift640"][0])} -> {fmt(s["drift640"][1])}'),
        ('frame age s median', lambda b, s: fmt(s['age'], '.2f')),
        ('selector ms median/P95', lambda b, s: f'{fmt(s["sel"][3])}/{fmt(s["sel"][4])}'),
        ('selector ok/err/correct', lambda b, s: '/'.join(str(v) for v in s['sel'][:3])),
        ('mean battery W', lambda b, s: fmt(s['w'], '.2f')),
        ('VIRTUAL-SKIN start/end/max C', lambda b, s: '/'.join(fmt(v, '.1f') for v in s['skin'])),
        ('skin rise rate C/min', lambda b, s: fmt(skin_slope(b['dumps'], b['duration_s']), '.3f')),
        ('Android status max', lambda b, s: fmt(s['status_max'])),
        ('Android status changes', lambda b, s: status_changes(b)),
        ('min MemAvailable MiB', lambda b, s: fmt(s['min_avail'])),
        ('max swap MiB', lambda b, s: fmt(s['max_swap'])),
        ('LMK lines (kill lines)', lambda b, s: f'{b["lmk"]["n_lines"]} ({b["lmk"]["n_kills"]})' if b['lmk']['ok'] else 'QUERY FAILED'),
        ('RobotCam / llama-server survival', lambda b, s: f'{"yes" if b["survived"]["robotcam_process"] and b["survived"]["robotcam_new_frame_at_end"] else "NO"} / {"yes" if b["survived"]["llama_server"] else "NO"}'),
        ('gate wait s', lambda b, s: str(b['thermal_start']['waited_s'])),
        ('warm start', lambda b, s: 'yes' if b['thermal_start']['warm_start'] else 'no'),
        ('stop limit / time s', lambda b, s: f'{b["heat_stop"]["limit"] or "none"} / {fmt(b["heat_stop"]["time_to_limit_s"], ".1f")}'),
        ('ORT version', lambda b, s: b['ort']['version']),
        ('ORT intra-op option (0=auto)', lambda b, s: str(b['ort']['intra_op_num_threads'])),
        ('default intra-op count observed', lambda b, s: str(b['ort']['observed_default_intra_threads_per_session'])),
        ('ORT workers observed CPUs / allowed', lambda b, s: observations.get(b['block'], 'not requested')),
    ]
    for p in POLICIES:
        rows += [(f'{p} capped s (% of block)', lambda b, s, p=p:
                  f'{capped_by_policy(b)[p]["seconds"]:.1f} ({fmt(capped_by_policy(b)[p]["percent"], ".1f")}%)'),
                 (f'{p} lowest scaling_max MHz', lambda b, s, p=p: fmt(capped_by_policy(b)[p]['lowest_mhz']))]
    lines = [f'Power map {out.name}, set {run["set"]}' + (' [SMOKE: functional check only]' if run['smoke'] else ''),
             *[f'INCOMPLETE: {p}' for p in bad], '',
             '| metric | ' + ' | '.join(b['block'] for b in blocks) + ' |',
             '| --- | ' + ' | '.join('---' for b in blocks) + ' |']
    for label, fn in rows:
        lines.append('| ' + label + ' | ' + ' | '.join(fn(b, s).replace('|', '/') for b, s in zip(blocks, sums)) + ' |')
    lines += ['', 'Skin rise rate: least-squares slope of in-block 5 s VIRTUAL-SKIN readings versus monotonic time, C/min.',
              'Capped time is separate per policy: scaling_max_freq < cpuinfo_max_freq. Each in-block 1 s sample owns '
              'the gap since the previous in-block sample (first owns time from zero); skipped-slot gaps belong to '
              'the later reading; unreadable is unknown and INCOMPLETE; tail after last sample is not counted. '
              'Percent denominator is actual block duration. policy4/policy6 are the target; policy0 is reported only.',
              'Skin start includes the first monitor dump during camera startup; status changes use the latest pre-block state at t=0.',
              'Gate: VIRTUAL-SKIN <= idle + 1.5 C; after 8 min start marked warm. Smoke skips waiting.',
              'zone9/10/11 in raw JSON and thermal.log: CPU-core reading, not a heat state; CPU readings are fault checks only.',
              'ORT observations: /proc task stat processor is the last scheduled CPU at each 1 s sample, not a full scheduler trace. '
              'CPU lists count samples only after per-thread utime+stime ticks advance since the preceding sample; '
              'the first sample is a baseline. Idle workers retain stale processor values, which are not execution evidence. '
              'Sub-tick work and the initial sampling interval may be missed. '
              'Cpus_allowed_list is sampled for every runner thread, including idle workers. Only newly created ORT session workers are pinned; '
              'mid/little also pin a dedicated detector calling thread, which participates in intra-op work. Default uses the deployed caller. '
              'Default count = observed created workers / two sessions + one caller; intra-op option 0 means automatic.',
              'Detect ms includes resize, inference and NMS. 640 drift = first/last 30 s medians. Battery W = '
              '-current_now*voltage_now/1e12; sampler/PSS every 5 s. Selector uses coresidency cases and letter scoring every 20 s. '
              'Block limits and INCOMPLETE rules inherited from coresidency, with required sizes adapted to cadence.',
              'Gemma resident for this invocation; no cold load, cache drop or MTP snapshot.',
              'Server command: ' + ' '.join(run['server_cmd'])]
    return '\n'.join(lines) + '\n', bad


def status_changes(b):
    last = None
    changes = []
    initial = [d for d in b['dumps'] if d['t'] < 0 and d['status'] is not None]
    rows = initial[-1:] + [d for d in b['dumps'] if d['t'] >= 0]
    for d in rows:
        if d['t'] > b['duration_s'] or d['status'] is None:
            continue
        if d['status'] != last:
            last = d['status']
            changes.append(f'{max(0, d["t"]):.1f}s {cr.STATUS_NAMES[last]}({last})')
    return ', '.join(changes) or 'none'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--set', required=True, choices=('R1', 'R2', 'R3', 'CONFIRM'))
    ap.add_argument('--rate', type=int, choices=(1, 2))
    ap.add_argument('--cadence', choices=('5', '10', 'off'))
    ap.add_argument('--threads', choices=('default', 'mid', 'little'))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--thermal-log', default=str(OUT_ROOT / 'thermal.log'))
    a = ap.parse_args(argv)
    try:
        specs = blocks_for(a.set, a.rate, a.cadence, a.threads)
    except ValueError as e:
        ap.error(str(e))
    cr.require_native()  # before any root command, camera or server
    duration = (60 if a.smoke else 1200) if a.set == 'CONFIRM' else (20 if a.smoke else 180)
    print('Blocks: ' + ', '.join(s['name'] for s in specs), flush=True)
    lower = 300 + duration * len(specs)
    upper = lower + (0 if a.smoke else GATE_MAX_S * (len(specs) + 1))
    print(f'Estimated wall time {lower/60:.1f}-{upper/60:.1f} min + load/warm-up and camera checks '
          '(includes 5 min launcher idle and up to 8 min per block/start gate).', flush=True)
    if a.resume:
        out = a.resume
        run = json.loads((out / 'run.json').read_text())
        if (run['set'], run['blocks'], run['smoke']) != (a.set, specs, a.smoke):
            raise SystemExit('--resume: set/settings/smoke do not match run.json')
    else:
        out = OUT_ROOT / f'run_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}_{a.set}{"_smoke" if a.smoke else ""}'
        out.mkdir(parents=True)
    ctx = {'thermal_log': a.thermal_log, 'smoke': a.smoke, 'duration': duration,
           'shell': None, 'observe_threads': a.set in ('R3', 'CONFIRM')}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cr.exit_on_signal)
    try:
        rc, who = cr.root('id', 'id')
        if rc:
            raise SystemExit(f'su failed: {who.strip()}')
        cr.wait_cores('start')
        ctx['shell'] = cr.RootShell()
        ctx['layout'] = cr.discover(ctx['shell'])
        if not set(POLICIES) <= set(ctx['layout']['policies']):
            raise SystemExit('layout lacks policy0/4/6')
        ctx['idle'] = {**cr.read_thermal(a.thermal_log), 'skin': cr.read_dump()['skin']}
        if ctx['idle']['skin'] is None:
            raise SystemExit('no VIRTUAL-SKIN at idle')
        if a.resume:
            ctx['idle'] = run['idle']  # retain original gate baseline as coresidency intends
        meta = {'set': a.set, 'blocks': specs, 'smoke': a.smoke, 'block_s': duration,
                'idle': ctx['idle'], 'started': cr.utc(), 'server_cmd': cr.server_cmd(),
                'sha256': {p.name: cr.sha256(p) for p in (Path(__file__), Path(cr.__file__),
                           Path(dp.__file__), Path(cr.ROBOT / 'detector_size_policy.py'),
                           *map(Path, dp.MODELS.values()))}, 'layout': ctx['layout']}
        if a.resume:
            (out / f'resume_{time.time_ns()}.json').write_text(json.dumps(meta, indent=2))
        else:
            (out / 'run.json').write_text(json.dumps(meta, indent=2))
        if any(not (out / f'block_{s["name"]}.json').exists() for s in specs):
            gate = thermal_gate(a.thermal_log, ctx['idle'], 'load/warm-up', a.smoke)
            ctx['server'] = cr.Server(out / f'llama_server_{time.time_ns()}.log')
            ctx['selector'], ctx['cases'] = cr.make_selector(), cr.load_cases()
            ctx['selector'].decide(*cr.WARMUP)  # one untimed warm-up
            cr.check_cores('after warm-up')
            (out / f'load_{time.time_ns()}.json').write_text(json.dumps(
                {'load_s': ctx['server'].load_s, 'gate': gate, 'cmd': ctx['server'].cmd}, indent=2))
        cr.camera_stop()
        default_detector, default_info = build_detector('default')
        del default_detector
        gc.collect()
        for spec in specs:
            path = out / f'block_{spec["name"]}.json'
            if path.exists():
                print(f'[{spec["name"]}] already completed, skipped (--resume)', flush=True)
                continue
            ctx['detector'], ctx['ort'] = build_detector(spec['threads'])
            ctx['ort']['observed_default_intra_threads_per_session'] = default_info['observed_default_intra_threads_per_session']
            while True:
                cr.wait_cores(spec['name'])
                try:
                    b = run_block(spec, ctx)
                    break
                except cr.CoresLost as e:
                    print(f'[{spec["name"]}] {e}; discarding block and redoing', flush=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(b) + '\n')
            tmp.replace(path)
            if isinstance(ctx['detector'], PinnedDetector):
                ctx['detector'].close()
            del ctx['detector']
            gc.collect()  # ensure previous session pools are gone before identifying next workers
        text, bad = report(out)
        (out / 'report.txt').write_text(text)
        print(text, flush=True)
        if cr.DOWNLOADS.is_dir():
            shutil.copy(out / 'report.txt', cr.DOWNLOADS / f'power_map_{out.name}_report.txt')
        else:
            print(f'WARNING: {cr.DOWNLOADS} missing; report only in {out}', flush=True)
        if bad:
            raise SystemExit('RUN INCOMPLETE: ' + '; '.join(bad))
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        for server in list(cr.LIVE):
            server.stop()
        try:
            cr.camera_stop()
        finally:
            try:
                why = cr.camera_end_failed(cr.camera_end_check())
                if why:
                    print(f'[CAM] end check: {why}', file=sys.stderr)
            finally:
                try:
                    if ctx['shell']:
                        ctx['shell'].close()
                finally:
                    if isinstance(ctx.get('detector'), PinnedDetector):
                        ctx['detector'].close()


if __name__ == '__main__':
    main()
