"""One model per process, MID cores 4-5. Full runs are HUMAN ONLY, native Termux.
Run M1 then M2 in separate invocations; each idles 5 min before its skin gate.
--dry-run: exactly 2 calls, NOT VALID; skips agent/charger/thermal gates and sensors.
"""
import argparse
import hashlib
import fcntl
import json
import os
import signal
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
from desk_check import HERE, ROBOT, HOME, Head, NAMES, require_parity, open_image, peak_rss_kib, write_json
import numpy as np

sys.path.insert(0, str(ROBOT / 'benchmark/power_map'))
import power_map as pm
cr = pm.cr
CPUS = {4, 5}


def processes_clear():
    # Existing oneshot pgrep checks, extended to robot and all benchmark Python runners.
    pattern = (r'claude|agy|node|codex|llama-server|chat\.py|main\.py|run_mission\.py|'
               r'power_map\.py|coresidency\.py|thermal_char\.py|duty_cycle\.py|'
               r'camera_power\.py|speed_block\.py|benchmark/.*\.py|run_.*\.sh|oneshot\.sh')
    result = subprocess.run(['pgrep', '-fa', pattern], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError('process check failed: ' + result.stderr)
    others = [line for line in result.stdout.splitlines() if int(line.split()[0]) != os.getpid()]
    if others:
        raise RuntimeError('agent/robot/benchmark resident: ' + '\n'.join(others))


class Screen:
    """coresidency/oneshot.sh wake lock + validated timeout, using cr.root (#123)."""
    def __init__(self):
        self.old = None
        self.locked = False

    def setting(self, command):
        rc, output = cr.root(command, 'relate_screen')
        if rc:
            raise RuntimeError('screen setting failed: ' + output)
        return output.strip()

    def start(self):
        subprocess.run(['termux-wake-lock'], check=True, timeout=30)
        self.locked = True
        old = self.setting('settings get system screen_off_timeout')
        if not re.fullmatch(r'[0-9]+', old):
            raise RuntimeError('invalid screen timeout: ' + repr(old))
        self.old = old  # cleanup restores from here, even if setting the new value fails
        if old == '2147483647':
            print('WARNING: saved screen timeout is already 2147483647; restore preserves it', flush=True)
        self.setting('settings put system screen_off_timeout 2147483647')
        if self.setting('settings get system screen_off_timeout') != '2147483647':
            raise RuntimeError('screen timeout set/readback failed')
        print('screen timeout saved ' + old + ' ms; set/read back 2147483647', flush=True)

    def restore(self):
        try:
            if self.old is not None:
                self.setting('settings put system screen_off_timeout ' + self.old)
                if self.setting('settings get system screen_off_timeout') != self.old:
                    raise RuntimeError('screen timeout restore/readback failed; restore manually to ' + self.old)
                print('screen timeout restored/read back ' + self.old + ' ms', flush=True)
        finally:
            if self.locked:
                subprocess.run(['termux-wake-unlock'], check=True, timeout=30)


def battery_sample():
    start = time.monotonic()
    rc, output = cr.root(f'cat {cr.BATTERY}/current_now {cr.BATTERY}/voltage_now {cr.BATTERY}/status', 'relate_battery')
    fields = output.split()
    if rc or len(fields) != 3:
        raise RuntimeError(f'root battery read failed: rc={rc}, {output!r}')
    if fields[2] != 'Discharging':
        raise RuntimeError('charger/battery status: ' + fields[2])
    return {'t_start': start, 't': time.monotonic(), 'battery_w': -(int(fields[0]) * int(fields[1])) / 1e12,
            'battery_status': fields[2], **cr.meminfo_mib(), 'VmHWM_KiB': peak_rss_kib()}


def status_memory_sample():
    start = time.monotonic()
    rc, status = cr.root(f'cat {cr.BATTERY}/status', 'relate_battery')
    if rc or status.strip() != 'Discharging':
        raise RuntimeError('charger/battery status: ' + status)
    return {'t_start': start, 't': time.monotonic(), 'battery_status': status.strip(),
            **cr.meminfo_mib(), 'VmHWM_KiB': peak_rss_kib()}


def require_policies(layout):
    if not set(pm.POLICIES) <= set(layout['policies']):
        raise RuntimeError('missing required CPU frequency policies; not started')


def read_fast(shell, keys, power_rows, root_pid, cpus=CPUS):
    row = cr.fast_sample(shell, keys)
    tid = threading.get_native_id()
    allowed = os.sched_getaffinity(tid)
    root_allowed = root_mask(shell, root_pid)
    require_monitor_mask(root_mask(shell, shell.p.pid), cpus)
    if not allowed or not root_allowed or allowed & cpus or root_allowed & cpus:
        raise RuntimeError('fast sampler/root shell may run on inference cores')
    pump_masks = {str(t): sorted(os.sched_getaffinity(t)) for t in getattr(shell, 'monitor_tids', ())}
    for mask in pump_masks.values():
        require_monitor_mask(set(mask), cpus)
    start = time.monotonic()
    values = shell.run(f'for f in {cr.BATTERY}/current_now {cr.BATTERY}/voltage_now; '
                       'do v=; read -r v <"$f"; echo "$v"; done')
    ended = time.monotonic()
    if len(values) != 2:
        raise RuntimeError('fast battery read missing values')
    current, voltage = map(int, values)
    power_rows.append({'t_start': start, 't': ended, 'battery_w': -current * voltage / 1e12,
                       'current_now_uA': current, 'voltage_now_uV': voltage,
                       'sampler_tid': tid, 'sampler_cpus': sorted(allowed),
                       'root_shell_pid': root_pid, 'root_shell_cpus': sorted(root_allowed),
                       'pump_cpus': pump_masks})
    return row


def power_summary(rows, calls, duration):
    block = [r for r in rows if 0 <= r['t'] <= duration]
    inside = [r for r in block if any(c['started_s'] <= r['t'] < c['ended_s'] for c in calls)]
    outside = [r for r in block if not any(c['started_s'] <= r['t'] < c['ended_s'] for c in calls)]
    def mean(group):
        return sum(r['battery_w'] for r in group) / len(group) if group else None
    return {'mean_battery_w': mean(block), 'mean_inside_calls_w': mean(inside),
            'mean_outside_calls_w': mean(outside), 'samples': len(block),
            'inside_samples': len(inside), 'outside_samples': len(outside),
            'classification': 'sample receipt t in [started_s, ended_s); block 0 <= t <= duration',
            'read_overlaps_call_boundary': sum(any(r['t_start'] < c[k] <= r['t']
                for c in calls for k in ('started_s', 'ended_s')) for r in block)}


class Calls:
    def __init__(self, head):
        self.head = head

    def detect(self, image, detections):
        return self.head.infer(image, detections)


def build(name, cpus=CPUS, threads=2, head_factory=Head):
    before = pm.thread_ids()
    head = head_factory(name, threads=threads)
    workers = sorted(pm.thread_ids() - before)
    expected = threads - 1
    if len(workers) != expected:
        raise RuntimeError(f'expected {expected} ORT workers, got {workers}')
    for tid in workers:
        os.sched_setaffinity(tid, cpus)
        if os.sched_getaffinity(tid) != cpus:
            raise RuntimeError('ORT worker affinity failed')
    caller = pm.PinnedDetector(Calls(head), cpus)
    return head, caller, workers + [caller.tid]


def check_pinning(tids, cpus=CPUS):
    for tid in tids:
        if os.sched_getaffinity(tid) != cpus:
            raise RuntimeError(f'worker/caller {tid} lost cluster affinity')


def require_monitor_mask(mask, cpus=CPUS, *, process="unspecified monitor", how="caller-supplied mask", allowed=None):
    if not mask or mask & cpus:
        raise RuntimeError('root shell/monitor mask empty or includes inference cores: '
                           f'actual={sorted(mask)} measured={sorted(cpus)} '
                           f'allowed={sorted(allowed) if allowed is not None else None} '
                           f'process={process} how={how}')


def root_mask(shell, pid):
    """Read a root-owned PID through root, never through Termux's affinity syscall."""
    if type(pid) is not int or pid <= 0:
        raise RuntimeError('invalid root mask PID')
    lines = shell.run('while IFS= read -r line; do case "$line" in '
                      'Cpus_allowed_list:*) printf \'%s\\n\' "$line";; esac; done '
                      f'</proc/{pid}/status')
    fields = [line.partition(':')[2].strip() for line in lines
              if line.startswith('Cpus_allowed_list:')]
    if len(fields) != 1 or not re.fullmatch(r'[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*', fields[0]):
        raise RuntimeError('missing/malformed root Cpus_allowed_list')
    mask = set()
    for part in fields[0].split(','):
        ends = list(map(int, part.split('-')))
        first, last = ends[0], ends[-1]
        if first > last or last >= os.cpu_count():
            raise RuntimeError('invalid root CPU range')
        mask.update(range(first, last + 1))
    return mask


def prepare_monitor(cpus=CPUS):
    previous = os.sched_getaffinity(0)
    monitor_cpus = previous - cpus
    require_monitor_mask(monitor_cpus, cpus, process=f"calling thread tid={threading.get_native_id()} (no root shell created)",
                         how="os.sched_getaffinity(0) minus measured", allowed=previous)
    shell = None
    try:
        try:
            os.sched_setaffinity(0, monitor_cpus)
            before = pm.thread_ids()
            shell = cr.RootShell()
            shell.monitor_tids = pm.thread_ids() - before
        finally:
            os.sched_setaffinity(0, previous)
        mask = format(sum(1 << c for c in monitor_cpus), 'x')
        shell.run(f'taskset -p {mask} {shell.p.pid}')
        shell.run(f'taskset -p {mask} $$')
        root_pid = int(shell.run('echo $$')[0])
        shell.root_pid = root_pid
        require_monitor_mask(root_mask(shell, root_pid), cpus, process=f"persistent shell pid={root_pid}",
                             how="root /proc/PID/status Cpus_allowed_list", allowed=monitor_cpus)
        require_monitor_mask(root_mask(shell, shell.p.pid), cpus, process=f"su pid={shell.p.pid}",
                             how="root /proc/PID/status Cpus_allowed_list", allowed=monitor_cpus)
        layout = cr.discover(shell)
        require_policies(layout)
        return shell, monitor_cpus, root_pid, layout
    except BaseException:
        if shell:
            shell.close()
        raise


def load_speed_input():
    datum = json.loads((HERE / 'speed_input.json').read_text())
    image_path = HERE / datum['image']
    image = open_image(image_path)
    if list(image.size) != datum['size'] or max(image.size) != 640 or not 2 <= len(datum['detections']) <= 32:
        raise RuntimeError('saved 640 image/boxes contract failed')
    return datum, image_path, image


def run_block(args, *, cpus=CPUS, threads=2, head_factory=Head, session=None):
    """Shared block implementation; a session owns screen, lock, idle and root shell."""
    preflight = getattr(args, 'preflight', False)
    if preflight and args.dry_run:
        raise RuntimeError('--preflight and --dry-run are mutually exclusive')
    if preflight:
        print('PREFLIGHT ONLY — NO TIMING', flush=True)
    require_parity(args.model)
    if sys.platform != 'android':
        raise RuntimeError('native Termux Python required')
    datum, image_path, image = load_speed_input()
    out = args.output or HERE / 'results' / args.model / f'{"dry_run_desk3" if args.dry_run else "speed"}_{time.monotonic_ns()}.json'
    if out.exists():
        raise RuntimeError('output exists; choose a new evidence filename')
    lock = None
    if session is None:
        lock_path = HOME / '.cache/relate_anything/speed.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('another RelateAnything speed runner holds the lock')
    if args.dry_run:
        print('INFORMAL — agents resident, NOT VALID TIMING — dry-run; no charger/agent/idle/skin gates or sensor sampling', flush=True)
    shell, caller, monitors = None, None, []
    screen = Screen()
    evidence_origin = time.monotonic()
    t0 = None
    stop = threading.Event()
    fast, dumps, samples, powers, affinities, calls, failures = [], [], [], [], [], [], []
    result = {'model': args.model, 'label': 'INFORMAL — agents resident, NOT VALID TIMING — DRY RUN' if args.dry_run else 'CANDIDATE TIMING; validity checked below',
              'photo': str(image_path), 'photo_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
              'input': datum, 'cadence_s': 5, 'planned_s': 180, 'intra_op_threads': threads, 'cpus': sorted(cpus),
              'sampling': {'caps_cpu_battery_temp_s': 1, 'battery_power_s': 1,
                           'skin_android_status_s': 5, 'battery_status_memory_s': 5},
              'power_source': {'current': cr.BATTERY + '/current_now',
                               'voltage': cr.BATTERY + '/voltage_now',
                               'kind': 'current_now sysfs reading; no runner averaging; driver filtering unspecified'},
              'fast': fast, 'dumps': dumps, 'status_memory': samples, 'power': powers,
              'affinity': affinities, 'calls': calls}
    result.update(variant=getattr(args, 'variant', 'fp32'), cluster=getattr(args, 'cluster', 'MID'))
    try:
        if not args.dry_run and session is None:
            cr.require_native()
            processes_clear()
            cr.check_cores('before idle')
            battery_sample()
            shell, monitor_cpus, root_pid, layout = prepare_monitor(cpus)
            head, caller, tids = build(args.model, cpus, threads, head_factory)
            check_pinning(tids, cpus)
            reading = cr.read_dump()
            if reading['skin'] is None or reading['status'] is None or reading['status'] >= cr.STATUS_STOP:
                raise RuntimeError('pre-idle skin/status missing or thermal stop')
            screen.start()
            if preflight:
                result.update(label='PREFLIGHT ONLY — NO TIMING',
                              root_shell_cpus=sorted(root_mask(shell, root_pid)),
                              cpuinfo_max_khz=layout['policies'], ort_tids=tids)
                return result
            caller.close()
            head = caller = None
            print('Idling 300 s; unplugged, screen on, Termux foreground', flush=True)
            for _ in range(60):
                time.sleep(5)
                processes_clear()
                battery_sample()
                cr.check_cores('during idle')
            idle = cr.read_dump()
            if idle['skin'] is None or idle['status'] is None:
                raise RuntimeError('no idle skin/status; not started')
            result['idle'] = idle
        if not args.dry_run and session is not None:
            idle = dict(session['idle'])
            result['idle'] = idle
            shell, monitor_cpus, root_pid, layout = session['monitor']
            require_monitor_mask(monitor_cpus, cpus)
            require_monitor_mask(root_mask(shell, root_pid), cpus)
        head, caller, tids = build(args.model, cpus, threads, head_factory)
        result.update(load_ms=head.load_ms, graph=head.metadata, ort_tids=tids)
        result.update(providers=head.session.get_providers(),
                      optimization_level=str(head.session.get_session_options().graph_optimization_level),
                      artifact_identity=getattr(head, 'artifact_identity', None))
        check_pinning(tids, cpus)
        if not args.dry_run:
            # power_map skin-only gate, bounded at 8 min; a timeout is recorded WARM START.
            began = time.monotonic()
            while True:
                processes_clear()
                battery_sample()
                cr.check_cores('during gate')
                reading = cr.read_dump()
                if reading['skin'] is None or reading['status'] is None:
                    raise RuntimeError('missing gate skin/status')
                cool = reading['skin'] <= idle['skin'] + cr.SKIN_GATE_C
                if cool or time.monotonic() - began >= pm.GATE_MAX_S:
                    result['gate'] = {**reading, 'warm_start': not cool, 'waited_s': time.monotonic() - began}
                    break
                time.sleep(cr.GATE_POLL_S)
            result['power_monitor_cpus'] = sorted(monitor_cpus)
            result['root_shell_cpus'] = sorted(root_mask(shell, root_pid))
            result['cpuinfo_max_khz'] = layout['policies']
            keys = cr.fast_keys(layout)
            def monitor(period, read, rows):
                try:
                    if rows is fast:
                        os.sched_setaffinity(0, monitor_cpus)
                        if os.sched_getaffinity(0) != monitor_cpus:
                            raise RuntimeError('fast monitor affinity readback failed')
                    cr.monitor_loop(period, read, rows, stop)
                except BaseException as error:
                    failures.append(str(error))
                    stop.set()
            def affinity_sample():
                check_pinning(tids, cpus)
                return pm.thread_sample(0, tids)
            for period, read, rows in [(1, affinity_sample, affinities),
                                       (1, lambda: read_fast(shell, keys, powers, root_pid, cpus), fast),
                                       (5, cr.read_dump, dumps), (5, status_memory_sample, samples)]:
                thread = threading.Thread(target=monitor, args=(period, read, rows), daemon=True)
                thread.start()
                monitors.append(thread)
            waiting = time.monotonic()
            while not (fast and dumps and samples and powers):
                if failures or time.monotonic() - waiting > 30:
                    raise RuntimeError('initial sensors failed: ' + repr(failures))
                time.sleep(0.05)
        if not args.dry_run:
            processes_clear()  # once before the measured block
        t0 = time.monotonic()
        result['block_start_s'] = 0.0
        result['time_base'] = 'time.monotonic; all sample times relative to block start'
        duration = 10 if args.dry_run else 180
        slot = 0
        while slot < (2 if args.dry_run else 36):
            due = t0 + slot * 5
            while time.monotonic() < due:
                if stop.wait(min(0.2, due - time.monotonic())):
                    raise RuntimeError('sensor monitor failed: ' + repr(failures))
                check_pinning(tids, cpus)
                if not args.dry_run:
                    cr.check_cores('between calls')
                    if hit := cr.block_limit(time.monotonic(), t0, fast, dumps):
                        raise RuntimeError('block limit: ' + repr(hit))
            if not args.dry_run:
                cr.check_cores('before call')
                if failures or (hit := cr.block_limit(time.monotonic(), t0, fast, dumps)):
                    raise RuntimeError('monitor/limit: ' + repr(failures or hit))
                if time.monotonic() >= t0 + duration:
                    raise RuntimeError('call cadence overran block')
            check_pinning(tids, cpus)
            start = time.monotonic()
            triplets, ort_ms = caller.detect(image, datum['detections'])
            ended = time.monotonic()
            check_pinning(tids, cpus)
            affinities.append(pm.thread_sample(0, tids))
            calls.append({'slot_s': slot * 5, 'started_s': start - t0, 'ended_s': ended - t0,
                          'ort_ms': ort_ms, 'pipeline_ms': (time.monotonic() - start) * 1000,
                          'triplets_count': len(triplets)})
            slot += 1
        if not args.dry_run:
            while time.monotonic() < t0 + duration:
                time.sleep(0.2)
                cr.check_cores('block tail')
                check_pinning(tids, cpus)
                if failures or (hit := cr.block_limit(time.monotonic(), t0, fast, dumps)):
                    raise RuntimeError('monitor/limit: ' + repr(failures or hit))
        result['duration_s'] = time.monotonic() - t0
        if not args.dry_run:
            processes_clear()  # once after the measured block
            battery_sample()
        result.update(median_ort_ms=float(np.median([c['ort_ms'] for c in calls])),
                      p95_ort_ms=float(np.percentile([c['ort_ms'] for c in calls], 95)),
                      VmHWM_KiB=peak_rss_kib(),
                      min_MemAvailable_MiB=min((s['mem_available_mib'] for s in samples
                                                if 0 <= s['t'] - t0 <= duration), default=None))
        if not args.dry_run:
            if any(c['started_s'] - c['slot_s'] >= 5 or c['ended_s'] > 180 for c in calls):
                raise RuntimeError('cadence missed/overrun')
            result['label'] = 'TIMING COMPLETE; warm starts flagged; no cold-cache load claim'
    except BaseException as error:
        result.update(label='NOT VALID — INCOMPLETE', error=f'{type(error).__name__}: {error}')
        if session is None or not isinstance(error, Exception):
            raise
    finally:
        stop.set()
        for thread in monitors:
            thread.join(timeout=60)
        if any(t.is_alive() for t in monitors):
            result.update(label='NOT VALID — INCOMPLETE', error='monitor still alive')
        for resource in (caller, shell if session is None else None):
            if resource:
                try:
                    resource.close()
                except BaseException as error:
                    result.update(label='NOT VALID — INCOMPLETE', cleanup_error=str(error))
        if failures:
            result.update(label='NOT VALID — INCOMPLETE', monitor_errors=failures)
        try:
            if session is None:
                screen.restore()
        except BaseException as error:
            result.update(label='NOT VALID — INCOMPLETE', cleanup_error=str(error))
        if lock is not None:
            lock.close()
        # Shared sensor/limit helpers use absolute monotonic readings during execution.
        # Convert every saved sample (including idle/gate and pre-block rows) once, here.
        origin = t0 if t0 is not None else evidence_origin
        if t0 is None:
            result['time_base'] = 'time.monotonic; relative to setup start, block never started'
        for row in fast + dumps + samples + powers + affinities + [result.get('idle', {}), result.get('gate', {})]:
            for key in ('t', 't_start'):
                if key in row:
                    row[key] -= origin
        result['power_summary'] = power_summary(powers, calls, min(result.get('duration_s', 0), 180))
        if 'duration_s' in result and fast:
            block = {'fast': fast, 'duration_s': min(result['duration_s'], 180),
                     'cpuinfo_max_khz': result['cpuinfo_max_khz']}
            result['caps_by_policy'] = pm.capped_by_policy(block)
            if any(r.get('error') or None in r['max'].values() for r in fast):
                result.update(label='NOT VALID — INCOMPLETE', error='unreadable CPU cap sample')
        result['validity'] = ('NOT VALID — INCOMPLETE' if result['label'] == 'NOT VALID — INCOMPLETE' else
                              'PREFLIGHT ONLY — NO TIMING' if preflight else
                              'NOT VALID — DRY RUN' if args.dry_run else
                              'NOT VALID — WARM START' if result.get('gate', {}).get('warm_start') else 'VALID')
        write_json(out, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('fast', 'dumps', 'status_memory', 'power', 'affinity', 'calls', 'input', 'graph')}, indent=2), flush=True)
        if preflight and result.get('cleanup_error'):
            raise RuntimeError('preflight cleanup failed: ' + result['cleanup_error'])
        if preflight:
            print('Evidence:', out, flush=True)
    print('Evidence:', out, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=NAMES)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    result = run_block(args)
    if result['label'] == 'NOT VALID — INCOMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, cr.exit_on_signal)
    main()
