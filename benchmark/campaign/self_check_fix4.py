"""CAMPAIGN_P1_FIX4 focused offline mocks: audit 1 fixes (H1-H5, M1, M3-M5, M8/M9, L1, L3-L7) and owner
decisions D1 (L1 M2 every 10 s) and D2 (bounded extra cooling). Never runs hardware, root, models or a timed phase."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import diagnostics as d
import phase1 as p
import power
import runtime as rt
from self_check import FastOps, InlineThread, phase_stubs

HERE = Path(__file__).resolve().parent
BASE = '56296995ea780141d757ba74ee768e07e27c2979'  # FIX3D, the audited commit
v, sb, pm, cr, _ = rt.imports()


def base_phase1():
    source = subprocess.run(['git', '-C', str(HERE), 'show', f'{BASE}:benchmark/campaign/phase1.py'],
                            capture_output=True, text=True, check=True).stdout
    path = Path(tempfile.mkdtemp())/'phase1_fix3d.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('phase1_fix3d', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def blip_once(value, error=rt.Blip('empty')):
    """A read failing once with `error`, then returning value; calls counted in .n."""
    def read():
        read.n += 1
        if read.n == 1:raise error
        return value
    read.n = 0
    return read


# ---------------------------------------------------------------- rest() on a fake clock

def rest_run(module=p, duration=600, camera_on=False, prepare_s=0., skin=lambda e: 30., t_ref=None, cool_max=0,
             events=None, check=None, cores=None, battery=None, cleanup=None, gap=None, mark=None):
    """rest() with mocks on a fake clock. Each loop check (block_limit) appends a thermal, fast and memory row
    (skin(elapsed)) and runs check(elapsed, record); the power sampler mock starts its 0.37 s rows when its thread
    starts. The second rt.prepare (the fast reader) takes prepare_s. cores/battery: side effects of rt.check_cores and
    diag.battery; cleanup=(n, error): the n-th rt.stop_camera call raises; gap=(lo, hi): no power rows in that interval.
    Returns (exception or None, record, events)."""
    events = [] if events is None else events
    clock, record, stops = [100.], {}, []
    def stop_camera(cr_):
        stops.append(1)
        if cleanup and len(stops) == cleanup[0]:raise cleanup[1]
    monitor = (NS(close=MagicMock()), {0}, 1, {'policies': {'policy0': 1803000}})
    fast = (NS(close=MagicMock()), {0}, 2, monitor[3])
    class Stop:
        done = False
        def set(self):self.done = True
        def is_set(self):return self.done
        def wait(self, dt):clock[0] += dt;return self.done
    def sampler(shell, battery, began, mask, measured, stop, rows, errors, verify):
        first = clock[0]-began
        rows.extend(dict(t=t, t_start=t-.01, battery_w=1.) for t in (first+k*.37 for k in range(int((duration+cool_max+20)/.37)))
                    if not gap or not gap[0] < t < gap[1])
    class Deferred:
        def __init__(self, target, args=(), **kw):self.target, self.args = target, args
        def start(self):
            events.append(('start', round(clock[0], 3)))
            if self.target is sampler:self.target(*self.args)
        def join(self, timeout=None):pass
        def is_alive(self):return False
    def prepare(sb_, measured):
        events.append(('prepare', round(clock[0], 3)))
        if sum(e[0] == 'prepare' for e in events) == 2:
            clock[0] += prepare_s
            return fast
        return monitor
    def bracket(shell, battery, t0, average=False):
        events.append(('bracket', round(clock[0], 3)))
        return dict(t_start=clock[0]-t0, t=clock[0]-t0, battery_w=1.)
    def block_limit(now, began, fast_rows, dumps):
        elapsed = now-began
        dumps.append(dict(t=now, t_start=now, rc=0, skin=skin(elapsed), status=0, attempts=1))
        fast_rows.append(dict(t=now, t_start=now, bat_c=30., cpu_c=50., cpu={}, max={'policy0': 1803000}))
        record['memory'].append(dict(t=now, mem_available_mib=2500, root_rc=0, battery_status='Discharging', pss_kb={}))
        if check:check(elapsed, record)
    server = NS(proc=NS(pid=7), alive=lambda: True)
    with contextlib.ExitStack() as stack:
        for m in (patch.object(module.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(module.threading, 'Event', Stop),
                  patch.object(module.threading, 'Thread', Deferred), patch.object(rt, 'prepare', side_effect=prepare),
                  patch.object(rt, 'root_mask_scope', return_value=contextlib.nullcontext()), patch.object(d, 'battery', side_effect=battery, return_value=85),
                  patch.object(d, 'snapshot', side_effect=lambda cr, on: dict(camera_on=on)),
                  patch.object(rt, 'stop_camera', side_effect=stop_camera), patch.object(rt, 'camera_start', return_value={'session': 'm'}),
                  patch.object(pm, 'camera_start', return_value={'session': 'm'}), patch.object(rt, 'clear_processes'),
                  patch.object(cr, 'check_cores'), patch.object(module.power, 'sample', side_effect=bracket),
                  patch.object(module.power, 'sampler', sampler), patch.object(cr, 'block_limit', side_effect=block_limit),
                  patch.object(module.os, 'sched_setaffinity'), patch.object(module.os, 'sched_getaffinity', return_value={0}),
                  patch.object(pm, 'capped_by_policy', return_value={}), patch.object(rt.time, 'sleep'),
                  patch.object(rt, 'battery_sample', side_effect=AssertionError('M3: rest loop battery_sample')),
                  *phase_stubs()):
            stack.enter_context(m)
        if module is not p:  # the audited loop still calls these (M3 removed them)
            stack.enter_context(patch.object(sb, 'battery_sample'))
            stack.enter_context(patch.object(rt, 'dump_check', return_value={'skin': 30, 'status': 0}))
        else:
            stack.enter_context(patch.object(rt, 'dump_check', side_effect=AssertionError('M3: rest loop dump_check')))
        pin = stack.enter_context(patch.object(rt, 'pin_main', side_effect=lambda mask: events.append(('pin_main', round(clock[0], 3)))))
        if cores:stack.enter_context(patch.object(rt, 'check_cores', side_effect=cores))
        kw = dict(t_ref=t_ref, cool_max=cool_max, mark=mark) if module is p else {}
        try:module.rest('FIXED PAUSE', duration, server, record, camera_on, **kw)
        except BaseException as e:return e, record, events
        return None, record, events


def check_h1():
    base = base_phase1()
    for prepare_s in (.4, 1.6):
        error, record, events = rest_run(base, prepare_s=prepare_s)
        if prepare_s > 1.5:
            assert str(error) == 'pause power coverage missing', error  # the audited order: bracket, then the shell
        else:
            assert error is None
        assert [e[0] for e in events][:3] == ['prepare', 'bracket', 'prepare'], events
    for prepare_s in (.4, 1.6, 5.):
        error, record, events = rest_run(prepare_s=prepare_s)
        assert error is None and record['power_summary']['mean_battery_w'] == 1., (error, record.get('power_summary'))
        kinds = [e[0] for e in events]
        assert kinds[:5] == ['prepare', 'prepare', 'pin_main', 'bracket', 'start'] and kinds.count('start') == 4, kinds
        bracket = next(t for k, t in events if k == 'bracket')
        assert all(t == bracket for k, t in events if k == 'start'), events  # threads start at the bracket: no gap
    print('PASS H1: audited rest() (bracket before the fast shell) loses the pause with a 1.6 s shell setup '
          '("pause power coverage missing"); fixed: both shells, then pin, bracket and all 4 threads at the same instant, '
          'power covered with 0.4/1.6/5 s setups')


# ---------------------------------------------------------------- H2

def check_h2_helper():
    rt.READ_RETRIES.clear()
    with patch.object(rt.time, 'sleep') as sleep:
        read = blip_once(42)
        assert rt.reread('x', read) == 42 and read.n == 2 and sleep.call_args_list == [((rt.READ_PAUSE_S,),)]
        (row,) = rt.READ_RETRIES
        assert row['what'] == 'x' and row['first'] == 'Blip: empty' and row['recovered'] is True and row['tid'] == threading.get_native_id()
        from self_check_fix4c import launch_error
        read = blip_once(1, launch_error())  # a real launch failure (FIX4D: only these are re-read)
        assert rt.reread('x', read) == 1 and read.n == 2
        def always():always.n += 1;raise rt.Blip(f'empty {always.n}')
        always.n = 0
        try:rt.reread('pgrep', always)
        except rt.Blip:raise AssertionError('second failure must be ReadFailed, not a Blip (no caller re-reads it again)')
        except rt.ReadFailed as e:assert str(e) == 'pgrep: Blip: empty 2 (re-read once; first: Blip: empty 1)' and always.n == 2, e
        assert rt.READ_RETRIES[-1]['recovered'] is False and len(rt.READ_RETRIES) == 3
        # Parsed but bad (any non-Blip error): no re-read.
        def bad():bad.n += 1;raise RuntimeError('charger/battery status: Charging')
        bad.n = 0
        try:rt.reread('x', bad)
        except RuntimeError as e:assert 'Charging' in str(e) and bad.n == 1
        assert len(rt.READ_RETRIES) == 3
        # Value-returning reads: blip(value) names the reason; the second value stands.
        values = iter([False, True])
        assert rt.reread('/health', lambda: next(values), lambda ok: None if ok else 'no') is True
        assert rt.server_ok(NS(alive=MagicMock(side_effect=[False, True]))) is True
        server = NS(alive=MagicMock(return_value=False))
        assert rt.server_ok(server) is False and server.alive.call_count == 2  # 2 consecutive failures
        # Per row and per phase.
        mark = len(rt.READ_RETRIES)
        row = rt.counted(lambda: rt.reread('x', blip_once({'v': 1})))
        assert row == {'v': 1, 'retries': 1}
        record = {}
        assert rt.retry_check(record, mark) is False and record['read_retry_check'] == 'OK' and len(record['read_retries']) == 1
        for _ in range(3):rt.reread('x', blip_once(0))
        assert rt.retry_check(record, mark) is True and record['read_retry_check'] == 'NOT VALID — READ RETRIES (4 > 3)'
    print('PASS H2 helper: one re-read after 0.3 s on Blip/OSError/blip(value), recorded (what, first error, tid, recovered); '
          'persistent failure -> RuntimeError carrying both errors, never re-read again; other errors not re-read; '
          '/health needs 2 consecutive failures; per-row retries; > 3 per phase -> NOT VALID — READ RETRIES')


def block_run(cycle, patches=(), mark=None):
    """live_block('L4') with mocked hardware; cycle(name, duration, ops, record, workers) replaces run_cycle."""
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        monitor = (NS(close=MagicMock()), {0}, 10, {'policies': {'policy0': 1}, 'cpu_zones': {}})
        workers = []
        class Deferred:
            def __init__(self, target, args=(), **kw):self.target, self.args = target, args;workers.append(self)
            def start(self):pass
            def join(self, timeout=None):pass
            def is_alive(self):return False
        for m in (patch.object(rt, 'prepare', return_value=monitor), patch.object(pm, 'build_detector', return_value=(NS(close=MagicMock()), {'worker_tids': [], 'caller_tid': 1})),
                  patch.object(v, 'build', return_value=(None, NS(close=MagicMock()), [])), patch.object(rt, 'clear_processes'),
                  patch.object(rt, 'battery_sample'), patch.object(rt, 'fast_check', return_value={}), patch.object(pm, 'camera_start', return_value={'session': 's'}),
                  patch.object(rt, 'dump_check', return_value={'skin': 30, 'status': 0}), patch.object(p, 'bounded_selector'), patch.object(cr, 'load_cases', return_value=[{}]),
                  patch.object(p.power, 'sample', return_value={'t': 0, 't_start': 0, 'battery_w': 1}), patch.object(p.threading, 'Thread', Deferred),
                  patch.object(rt, 'stop_camera'), patch.object(cr, 'lmk_lines', return_value={'ok': True, 'n_kills': 0, 'raw_head': 'logcat_rc=0\n'}), patch.object(cr, 'block_limit', return_value=None),
                  patch.object(d, 'snapshot', return_value={}), patch.object(rt, 'fallback_input', return_value=('img', [{}, {}])),
                  patch.object(pm, 'capped_by_policy', return_value={}), *phase_stubs(), *patches):
            stack.enter_context(m)
        def run(name, length, ops, record):
            origin = record['cycle_origin']
            record.update(duration_s=length, m2=[], selector=[], yolo=[], reads=[])
            record['power'] = [dict(t=i, t_start=i-.01, battery_w=1) for i in range(-1, length+2)]
            record['memory'] = [dict(t=origin, mem_available_mib=1, root_rc=0, pss_kb={'llama_server': 1}, battery_status='Discharging')]
            cycle(name, length, ops, record, workers)
        stack.enter_context(patch.object(p, 'run_cycle', side_effect=run))
        return p.live_block('L4', Path(tmp)/'b.json', NS(proc=NS(pid=1), alive=lambda: True), None, False, 180, mark=mark)


def check_h2_cap():
    with patch.object(rt.time, 'sleep'):
        for n, validity in ((3, 'VALID'), (4, 'NOT VALID — READ RETRIES')):
            def cycle(name, length, ops, record, workers, n=n):
                for _ in range(n):rt.reread('x', blip_once(0))
            record = block_run(cycle)
            assert record['validity'] == validity and record.get('failure_kind') is None and len(record['read_retries']) == n, record.get('error')
            assert p.can_continue(record) or not record.get('failure_kind')  # the session goes on
        for n in (3, 4):
            def check(elapsed, record, done=[0], n=n):
                if done[0] < n:done[0] += 1;rt.reread('x', blip_once(0))
            error, record, _ = rest_run(check=check)
            assert error is None and len(record['read_retries']) == n, error
            assert record.get('validity') == (None if n == 3 else 'NOT VALID — READ RETRIES (4 > 3)'), record.get('validity')
    print('PASS H2 cap: 3 re-reads in a block/pause keep it VALID; 4 make it NOT VALID — READ RETRIES without failure_kind '
          '(no session stop); the re-reads are listed in the phase record')


# ---------------------------------------------------------------- H3, H4, D1 (run_cycle)

original_wait = threading.Event.wait
def fast_wait(event, timeout=None):return original_wait(event, None if timeout is None else max(0, timeout/1000))


def check_h3():
    for n in (32, 40):
        boxes = [dict(class_name=f'c{i}', box_xyxy=[i, i, i+1, i+1], conf=(i*37 % 41)/41) for i in range(n)]
        ops, seen = FastOps('L4'), []
        ops.detect = lambda image, size: boxes
        ops.relate = lambda image, given: (seen.append(given), ([], 1.))[1]
        record = {}
        with patch.object(threading.Event, 'wait', fast_wait):p.run_cycle('L4', 21, ops, record)
        keep = sorted(range(n), key=lambda i: -boxes[i]['conf'])[:32]
        assert seen and all(given == [boxes[i] for i in sorted(keep)] for given in seen)  # top 32, detector order
        assert all(r['live_boxes'] == n and r['input_boxes'] == 32 and r['truncated_boxes'] == n-32 for r in record['m2'])
        assert record['summary']['max_live_boxes'] == n and record['summary']['m2']['truncated_slots'] == (len(seen) if n > 32 else 0)
        assert not record.get('worker_errors')
    ops = p.LiveOps(cr, NS(check_pinning=lambda *a: None), None, None, NS(detect=lambda i, b: ([], 1.)), [], set(), {}, None, None, [], None)
    try:ops.relate(None, [{}]*33)  # Head's limit stays guarded below run_cycle
    except RuntimeError:pass
    else:raise AssertionError('33 boxes reached M2')
    print('PASS H3: 40 live boxes -> the 32 highest-confidence ones in detector order, live_boxes 40, truncated_boxes 8, '
          'max_live_boxes 40, no error (32 -> unchanged); LiveOps.relate still refuses > 32')


def check_h4():
    def run(statuses, name='L0'):
        ops, plan = FastOps(name), list(statuses)
        n = [0]
        def frame(last):
            status = plan.pop(0) if plan else 'ok'
            if status in ('ok', 'repeat'):n[0] += status == 'ok';return dict(status='ok', frame=n[0], age_s=.1, image=None)
            return dict(status=status)
        ops.frame = frame
        record = {}
        with patch.object(threading.Event, 'wait', fast_wait):
            try:p.run_cycle(name, 30, ops, record)
            except RuntimeError as e:return str(e), record
        return None, record
    error, record = run(['ok', 'missing'])
    assert error is None and [m['status'] for m in record['camera_misses']] == ['missing'] and record['cadence_missed']
    assert record['camera_late'] >= 1 and record['frames_skipped']['320'] >= 1 and record['summary']['camera_slots_skipped'] == 1
    error, record = run(['ok', 'repeat'])
    assert error is None and [m['status'] for m in record['camera_misses']] == ['repeat']
    error, record = run(['ok', 'missing', 'missing', 'ok', 'missing', 'missing'])
    assert error is None and len(record['camera_misses']) == 4  # never 3 in a row
    error, record = run(['ok', 'missing', 'missing', 'missing'])
    assert error.startswith('camera failed: no new frame in 3 consecutive slots'), error
    for status in ('bad', 'other_session'):
        error, record = run(['ok', status])
        assert error == 'camera failed/repeated frame: '+status and not record['camera_misses'], error
    # LiveOps.frame polls a missing frame like a repeat until FRAME_WAIT_S; bad/other_session return at once.
    ops = p.LiveOps(cr, None, None, None, None, [], set(), {'session': 's'}, None, None, [], None)
    clock = [0.]
    def tick(dt):clock[0] += dt
    for answers, want, reads in (([dict(status='missing')]*3+[dict(status='ok', frame=2)], 'ok', 4),
                                 ([dict(status='missing')]*200, 'missing', 69), ([dict(status='bad')], 'bad', 1),
                                 ([dict(status='other_session')], 'other_session', 1)):
        clock[0] = 0.
        read = MagicMock(side_effect=answers)
        with patch.object(cr, 'read_frame', read), patch.object(p.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(p.time, 'sleep', side_effect=tick):
            assert ops.frame(1)['status'] == want and read.call_count == reads, (want, read.call_count)
    print('PASS H4: one missing or repeated frame -> skipped slot (camera_misses, camera_late, cadence miss), block goes on; '
          '2+2 with a frame between goes on; 3 in a row stops; bad/other_session stop at once; LiveOps.frame polls '
          '"missing" until 1.35 s (69 reads, 20 ms apart) and returns bad/other_session after 1 read')


class TenMsOps(FastOps):
    def now(self):return (time.monotonic()-self.base)*100  # 1 virtual s = 10 ms


def check_d1():
    def slow_wait(event, timeout=None):return original_wait(event, None if timeout is None else max(0, timeout/100))
    counts = {}
    for name in ('L1', 'L2', 'L4'):
        for attempt in range(5):
            ops = TenMsOps(name);record = {}
            with patch.object(threading.Event, 'wait', slow_wait):p.run_cycle(name, 180, ops, record)
            m2 = [r['slot_s'] for r in record['m2']]
            if name != 'L2':  # M2 only after a 640 frame on its M2_EVERY grid, on every run
                assert m2 == [r['slot_s'] for r in record['yolo'] if r['size'] == 640 and r['slot_s'] % p.M2_EVERY[name] == 0], (name, m2)
            if not sum(record['frames_skipped'].values()):
                break  # a proot scheduling stall skipped a slot: correct catch-up, but the count proves nothing
        else:raise AssertionError(f'{name}: 5 runs with skipped slots')
        counts[name] = m2
        assert not record.get('cadence_missed'), (name, record['summary'])
    assert counts['L1'] == list(range(0, 180, 10)), counts['L1']  # 18 calls, every second 640 frame
    assert len(counts['L2']) == 9 and len(counts['L4']) == 36, counts
    assert p.M2_EVERY == dict(L1=10, L3=5, L4=5)
    print('PASS D1: L1 runs M2 after every second 640 frame (slots 0,10..170: 18 calls per 180 s, no cadence miss); '
          'L2 9, L4 36 unchanged; expected-call count per layout from M2_EVERY')


# ---------------------------------------------------------------- D2

def check_d2():
    error, record, _ = rest_run(t_ref=None, cool_max=300, skin=lambda e: 40.)  # first pause: no T_ref yet
    assert error is None and 'cooling' not in record and 600 <= record['duration_s'] < 606
    error, record, _ = rest_run(t_ref=30., cool_max=300, skin=lambda e: 33. if e < 650 else 31.4)
    c = record['cooling']
    assert error is None and c['reached'] and 50 <= c['extra_s'] <= 55 and c['target_skin_c'] == 31.5 and c['end_skin_c'] == 31.4, c
    error, record, _ = rest_run(t_ref=30., cool_max=300, skin=lambda e: 33.)
    c = record['cooling']
    assert error is None and not c['reached'] and c['extra_s'] == 300 and not c['below_band'], c
    assert record['power_summary']['mean_battery_w'] == 1.  # still sampled through the extension
    error, record, _ = rest_run(t_ref=30., cool_max=300, skin=lambda e: 28.)
    c = record['cooling']
    assert error is None and c['extra_s'] == 0 and c['below_band'] and c['reached'], c
    error, record, _ = rest_run(t_ref=30., cool_max=300, skin=lambda e: 31.5)  # at the band edge: comparable, no wait
    assert record['cooling']['extra_s'] == 0
    # Same checks during the extension: a sensor stop there still stops the pause.
    def check(elapsed, record):
        if elapsed > 700:raise RuntimeError('pause thermal/sensor stop: mock')
    error, record, _ = rest_run(t_ref=30., cool_max=300, skin=lambda e: 33., check=check)
    assert str(error) == 'pause thermal/sensor stop: mock', error
    # Block labels: start skin within, above (after the extension ran out) and below T_ref.
    for skin, label in ((31.4, 'VALID'), (33., 'NOT COMPARABLE — START TEMP'), (28., 'NOT COMPARABLE — START TEMP')):
        row = dict(skin_start={'skin': skin}, validity='VALID')
        p.compare_start(row, 30.);assert row['validity'] == label
    print('PASS D2: no T_ref -> fixed pause; above T_ref+1.5 -> extended (same checks, still sampled) until within '
          '(+50..55 s here) or 300 s (not reached, block later NOT COMPARABLE); below T_ref-1.5 -> no wait (below_band, '
          'NOT COMPARABLE); edge 31.5 -> no wait')


# ---------------------------------------------------------------- main(): M1, M4, L3, L4, L5, H5, D2 wiring

def run_main(argv, rest_fail=None, battery_stop=False, server_stop_error=None, pause_files=True, screen_restore_error=None):
    calls, seen = [], {}
    server = NS(stop=MagicMock(side_effect=lambda: (seen.update(sigterm=signal.getsignal(signal.SIGTERM)),
                                                    server_stop_error and (_ for _ in ()).throw(server_stop_error))))
    screen = NS(restore=MagicMock(side_effect=screen_restore_error))
    def preflight(names, out, result, resources):
        seen['first_label'] = json.loads(out.read_text())['label']
        resources.update(server=server, screen=screen)
    def rest(label, duration, srv, record, camera_on=False, dry=False, **d2):
        calls.append(('rest', duration, d2))
        record.update(label=label, duration_s=duration)
        if rest_fail and sum(c[0] == 'rest' for c in calls) == rest_fail:
            record.update(error='RuntimeError: pause sampler stopped: mock cause')
            raise RuntimeError('pause sampler stopped: mock cause')
    starts = iter([20., 30., 30.5, 33., 31., 29., 30.])  # warm-up, then six measured blocks (T_ref 30)
    def block(name, path, srv, idle, dry=False, duration=180, **mark):
        calls.append(('block', name, dict(mark,idle=idle)))
        record = dict(block=name, validity=p.REHEARSAL if dry else 'VALID', skin_start={'skin': next(starts)},
                      summary=dict(camera_frames_late=2, camera_slots_skipped=1, max_live_boxes=9,
                                   m2=dict(live_calls=3, fallback_calls=1)),
                      selector=[{'context': '(fallback scene)\n'}], read_retries=[])
        p.write(path, record);return record
    def battery(cr, minimum):
        if battery_stop and minimum == 25 and sum(c[0] == 'rest' for c in calls) == 3:raise d.BatteryStop('battery 24% below 25%')
        return 85
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        out = io.StringIO()
        for m in (patch.object(p, 'preflight', side_effect=preflight), patch.object(p, 'rest', side_effect=rest),
                  patch.object(p, 'live_block', side_effect=block), patch.object(d, 'battery', side_effect=battery),
                  patch.object(rt, 'HOME', Path(tmp)), contextlib.redirect_stdout(out), *phase_stubs()):
            stack.enter_context(m)
        output = Path(tmp)/'s.json'
        try:p.main(argv+['--output', str(output)])
        except (RuntimeError, SystemExit) as e:seen['raised'] = e
        seen['handlers_after'] = signal.getsignal(signal.SIGTERM)
        files = sorted(f.name for f in Path(tmp).iterdir() if f.suffix == '.json')
        return calls, json.loads(output.read_text()), out.getvalue(), files, seen


def check_main():
    calls, result, out, files, seen = run_main([])
    assert seen['first_label'] == 'IN PROGRESS — NOT VALID' and result['label'] == 'SESSION COMPLETE; inspect individual validity'
    pauses = [c[2] for c in calls if c[0] == 'rest'][3:]
    # P23: cooling follows setup in live_block; pauses are fixed and cannot decide comparability early.
    assert all('t_ref' not in d and 'cool_max' not in d for d in pauses)
    gates=[c[2]['idle'] for c in calls if c[0]=='block'][1:]
    assert gates==[dict(reference=None,cool_max=300)]+[dict(reference=30.,cool_max=300)]*5,gates
    assert all(isinstance(d['mark'], int) for d in pauses)  # H2: the capacity check before each pause counts in it
    assert [b['validity'] for b in result['blocks']] == ['VALID', 'VALID', 'NOT COMPARABLE — START TEMP', 'VALID', 'VALID', 'VALID']
    assert files == sorted(['s.json', 's_warmup_L0.json']+[f's_idle_{i}.json' for i in (1, 2, 3)]
                           +[f's_pause_{i:02d}_{n}.json' for i, n in enumerate(p.DEFAULT, 1)]+[f's_block_{i:02d}_{n}.json' for i, n in enumerate(p.DEFAULT, 1)]), files
    assert '] START IDLE |' in out and '] END block 02 L1 VALID; camera late 2, camera slots skipped 1, M2 live 3 / fallback 1, max_live_boxes 9' in out, out
    assert 'about 95–100 min including setup/loads/cleanup, plus up to 25 min of D2 cooling' in out
    assert seen['sigterm'] == signal.SIG_IGN and seen['handlers_after'] != signal.SIG_IGN  # L4, restored afterwards
    assert result['read_retries'] == rt.READ_RETRIES
    # A failing pause: its own file exists with the error; the session stops with the cause (M4) and screen state (H5).
    calls, result, out, files, seen = run_main([], rest_fail=5)
    assert 's_pause_02_L1.json' in files and 's_block_02_L1.json' not in files
    assert result['error'] == 'RuntimeError: pause sampler stopped: mock cause' and result['label'] == 'NOT VALID — SESSION INCOMPLETE'
    assert 'END FIXED PAUSE FAILED: RuntimeError: pause sampler stopped: mock cause' in out and 'screen_state_at_failure' in result
    # L5: a cleanup error never replaces BatteryStop's label; it is recorded and exits 1.
    calls, result, out, files, seen = run_main([], battery_stop=True, server_stop_error=RuntimeError('stop failed'))
    assert result['label'].startswith('SESSION STOPPED — BATTERY') and result['cleanup_error'] == 'server: RuntimeError: stop failed', result['label']
    assert isinstance(seen['raised'], SystemExit) and seen['raised'].code == 1
    calls, result, out, files, seen = run_main([], server_stop_error=RuntimeError('stop failed'))
    assert result['label'] == 'NOT VALID — SESSION INCOMPLETE' and result['cleanup_error'] == 'server: RuntimeError: stop failed'
    assert result['error'] == 'cleanup failed: server: RuntimeError: stop failed'  # M4: a cleanup-only failure has a cause
    # Rehearsal: D2 at most 6 s; rehearsal_pass needs max_live_boxes for every block and no phase over the re-read cap.
    calls, result, out, files, seen = run_main(['--dry-run'])
    gates=[c[2]['idle'] for c in calls if c[0]=='block'][1:]
    assert [g['cool_max'] for g in gates]==[6]*6
    assert result['rehearsal_coverage']['rehearsal_pass'] is True, result['rehearsal_coverage']
    cov = dict(result, blocks=[dict(b, summary=dict(b['summary'], max_live_boxes=None)) if i == 2 else b for i, b in enumerate(result['blocks'])])
    assert p.coverage(cov)['rehearsal_pass'] is False
    cov = dict(result, pauses=[dict(r, read_retry_check='NOT VALID — READ RETRIES (4 > 3)') if i == 1 else r for i, r in enumerate(result['pauses'])])
    assert p.coverage(cov)['rehearsal_pass'] is False and p.coverage(cov)['read_retry_over_cap'] == ['NOT VALID — REHEARSAL: FIXED PAUSE']
    # M1: existing per-phase evidence refuses the stem.
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp)/'s_pause_03_L2.json').write_text('{}')
        try:p.main(['--output', str(Path(tmp)/'s.json')])
        except SystemExit as e:assert e.code == 2
        else:raise AssertionError('existing pause evidence overwritten')
    print('PASS main: IN PROGRESS — NOT VALID from the first write; idle/pause phases in their own files (also a failed '
          'pause); START/END lines with retries, block END with camera lateness, live/fallback M2, max_live_boxes (L3/H3); '
          'D2 T_ref from the first measured block, 300 s session / 6 s rehearsal; signals ignored during cleanup and restored '
          '(L4); a cleanup error keeps the battery label (L5); session failure keeps cause + screen state; rehearsal_pass '
          'needs max_live_boxes and no phase over the re-read cap; existing pause evidence refused')


def check_atomic_write():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'e.json'
        p.write(path, {'label': 'first'})
        with patch.object(p.os, 'fsync', side_effect=OSError('killed mid-write')):
            try:p.write(path, {'label': 'second', 'big': 'x'*100000})
            except OSError:pass
        assert json.loads(path.read_text()) == {'label': 'first'}  # the previous complete file survives
        with patch.object(p.os, 'replace', wraps=os.replace) as replace:p.write(path, {'label': 'third'})
        (src, dst), _ = replace.call_args
        assert Path(dst) == path and Path(src).parent == path.parent and json.loads(path.read_text()) == {'label': 'third'}
    print('PASS M1: write() goes through a same-directory temp file, flush, fsync and os.replace; a failure mid-write '
          'leaves the previous complete JSON')


# ---------------------------------------------------------------- M5, H5, M3, M4, L5, L7

def check_m5():
    for limit, kills in ((15, 0), (-1, 1)):
        kill, outcome = MagicMock(), {}
        def cycle(name, length, ops, record, workers):
            shared = next(w for w in workers if w.args and w.args[0] == 5.23)
            ops.guard()
            try:outcome['row'] = shared.args[1]()
            except RuntimeError as e:outcome['error'] = str(e)
        with patch.object(rt, 'HEARTBEAT_S', limit), patch.object(p.os, 'kill', kill), contextlib.redirect_stdout(io.StringIO()):
            record = block_run(cycle)
        assert kill.call_count == kills, kill.call_args_list  # setup and post-cycle shared checks are disarmed
        if kills:
            assert kill.call_args.args == (os.getpid(), signal.SIGTERM) and outcome['error'].startswith('main loop heartbeat')
        else:
            assert outcome['row']['ok'] and outcome['row']['retries'] == 0
    # Bounded transport from make_selector's first tokenize call; termux-wake-* have timeouts.
    posts = []
    def post(path, body):
        posts.append(path)
        return {'tokens': [2] if body.get('add_special') else [ord(body['content'][0])]}
    s = cr.make_selector(post)
    assert s._post is post and posts and set(posts) == {'/tokenize'}
    run = MagicMock(return_value=NS(returncode=0))
    with patch.object(sb.subprocess, 'run', run), patch.object(sb.cr, 'root', return_value=(0, '60000')), contextlib.redirect_stdout(io.StringIO()):
        screen = sb.Screen()
        try:screen.start()
        except RuntimeError:pass  # readback mock; only the wake lock call matters here
        screen.restore()
    assert [c.kwargs.get('timeout') for c in run.call_args_list] == [30, 30], run.call_args_list
    print('PASS M5: a stale main-loop heartbeat in the shared-check worker -> SIGTERM to self + error; fresh heartbeat -> '
          'ok; setup/post-cycle checks disarmed; make_selector uses the bounded transport from its first tokenize call; '
          'pgrep (10 s) and termux-wake-lock/unlock (30 s) have timeouts')


def check_h5():
    text = ('  mWakefulness=Awake\n  Display Power: state=ON\n  mHoldingDisplaySuspendBlocker=true\n===\n'
            '  topResumedActivity=ActivityRecord{1 u0 com.termux/.app.TermuxActivity t2}\n')
    state = rt.screen_state(NS(root=lambda cmd, tag, timeout=60: (0, text)))
    assert state == dict(rc=0, power=['mWakefulness=Awake', 'Display Power: state=ON', 'mHoldingDisplaySuspendBlocker=true'],
                         resumed_activity='topResumedActivity=ActivityRecord{1 u0 com.termux/.app.TermuxActivity t2}'), state
    root = MagicMock(side_effect=OSError('su gone'))  # not a launch failure (FIX4D): recorded at once, never re-read
    with patch.object(rt.time, 'sleep'):state = rt.screen_state(NS(root=root))
    assert state['error'] == 'OSError: su gone' and root.call_count == 1, state
    with patch.object(rt.time, 'sleep'):assert 'error' in rt.screen_state(NS(root=lambda *a, **k: (1, '')))
    nodes = '/b/capacity\t85\n/b/temp\t287\n/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq\t1401000\n'
    root = lambda cmd, tag, timeout=60: (0, text if tag == 'campaign_screen' else nodes)
    with patch.object(rt, 'read_dump', return_value=dict(status=0, skin=30., attempts=1)):
        assert d.snapshot(NS(BATTERY='/b', root=root), True)['screen']['power'][0] == 'mWakefulness=Awake'
    # Failure paths record it: a pause, a block and the session (check_main).
    error, record, _ = rest_run(cores=MagicMock(side_effect=cr.CoresLost('cores 4-7 not all allowed pause')))
    assert isinstance(error, cr.CoresLost) and record['screen_state_at_failure'] == {'mock': True}  # rest_run's stub
    def cycle(name, length, ops, record, workers):raise RuntimeError('camera failed: no new frame in 3 consecutive slots')
    record = block_run(cycle, (patch.object(rt, 'screen_state', return_value={'power': ['mWakefulness=Asleep']}),))
    assert record['failure_kind'] == 'shared' and record['screen_state_at_failure'] == {'power': ['mWakefulness=Asleep']}
    print('PASS H5: screen state (wakefulness/display lines, resumed activity) parsed, in every diagnostics snapshot, one '
          're-read then recorded, never raises; recorded in pause, block and session failure paths')


def check_m3_m4_l5_l7():
    # M3: rest_run fails on any rest-loop battery_sample/dump_check call; a full pause passes.
    error, record, events = rest_run()
    assert error is None
    # M4: causes in the messages.
    def worker_errors(elapsed, record):
        record['monitor_errors'][:] = ['RuntimeError: missing/malformed root Cpus_allowed_list', 'b', 'c']
    error, _, _ = rest_run(check=worker_errors)
    assert str(error) == 'pause monitor failed: RuntimeError: missing/malformed root Cpus_allowed_list; b', error
    assert rt.causes(['x'*400], ['y']) == 'x'*300+'; y'
    # L5: CoresLost or BatteryStop propagating + a cleanup error -> the original error propagates, cleanup recorded.
    cleanup = RuntimeError('camera cleanup: pid left')
    for failure, kw in ((cr.CoresLost('cores 4-7 not all allowed pause'), dict(cleanup=(2, cleanup))),
                        (d.BatteryStop('battery 24% below 25%'), dict(cleanup=(1, cleanup)))):
        if isinstance(failure, cr.CoresLost):kw['cores'] = MagicMock(side_effect=failure)
        else:kw['battery'] = failure
        error, record, _ = rest_run(**kw)
        assert error is failure and record['cleanup_errors'] == ['camera cleanup: pid left'], (error, record.get('cleanup_errors'))
    error, record, _ = rest_run(cleanup=(2, cleanup))  # nothing propagating: the cleanup error fails the pause
    assert str(error) == 'pause cleanup failed: camera cleanup: pid left' and record['screen_state_at_failure']
    # L7: the main thread is pinned to the monitor mask; cores 4-7 are checked on the unpinned sentinel.
    tid = rt.sentinel()
    with patch.object(rt.os, 'sched_getaffinity', side_effect=lambda t: {0, 1, 2, 3} if t == tid else set(range(8))):
        try:rt.check_cores(cr, 'pause')
        except cr.CoresLost as e:assert 'allowed [0, 1, 2, 3], sentinel thread' in str(e)
        else:raise AssertionError('lost cores accepted')
    with patch.object(rt.os, 'sched_getaffinity', side_effect=lambda t: set(range(8)) if t == tid else {0, 1, 2, 3}):
        rt.check_cores(cr, 'pause')  # a pinned main thread is not a cores loss
    calls = []
    with patch.object(rt.os, 'sched_setaffinity', side_effect=lambda t, m: calls.append((t, set(m)))), \
         patch.object(rt.os, 'sched_getaffinity', return_value={0, 1, 2, 3}):
        rt.pin_main({0, 1, 2, 3})
        assert calls == [(0, {0, 1, 2, 3})]
        try:rt.pin_main({6, 7})
        except RuntimeError as e:assert 'main thread pin failed' in str(e)
        else:raise AssertionError('pin readback mismatch accepted')
    calls.clear()
    with patch.object(rt.os, 'sched_setaffinity', side_effect=lambda t, m: calls.append((t, set(m)))), \
         patch.object(rt.os, 'sched_getaffinity', return_value=set(range(8))), patch.object(rt.os, 'cpu_count', return_value=8), \
         patch.object(cr, 'check_cores'):
        rt.setup_affinity(cr, {4, 5})
    assert calls == [(0, set(range(8))), (tid, set(range(8)))], calls
    print('PASS M3 (no rest-loop battery_sample/dump_check), M4 (two causes, 300 chars), L5 (CoresLost/BatteryStop keep '
          'propagating over a cleanup error; alone, the cleanup error fails the pause), L7 (main pinned with readback; '
          'cores checked on the sentinel, refreshed with the main thread at setup)')


def check_run_md():
    text = (HERE/'RUN.md').read_text()
    for needle in ('MANDATORY', 'rehearsal_pass', 'max_live_boxes', 'Airplane mode', 'Do Not Disturb', 'Magisk',
                   'no shelves', 'Battery ≥ 80 %', 'unplugged', 'hands off', '95–100 min', '25 min',
                   'no wrapper', 'oneshot.sh', 'settings put system screen_off_timeout', 'am force-stop com.pixelrobot.robotcam',
                   'pkill -f llama-server', 'termux-wake-unlock'):
        assert needle in text, needle
    print('PASS RUN.md: owner checklist, mandatory rehearsal gate, no-wrapper rule (M9), 95–100 min + D2 (L1), recovery block (M8)')


# ---------------------------------------------------------------- review round 1 regressions

def check_round1():
    # F1: catch-up skips (slow frames) must not feed the consecutive camera-failure counter.
    ops, calls = FastOps('L0'), [0]
    def frame(last):
        calls[0] += 1
        if calls[0] == 2:time.sleep(.004)  # 4 virtual s: the next loop skips past slots (catch-up)
        if calls[0] == 3:return dict(status='missing')  # right after the catch-up
        return dict(status='ok', frame=calls[0], age_s=.1, image=None)
    ops.frame = frame
    record = {}
    with patch.object(threading.Event, 'wait', fast_wait):p.run_cycle('L0', 40, ops, record)  # no stop
    assert [m['status'] for m in record['camera_misses']] == ['missing'] and record['frames_skipped']['320'] >= 3, record['frames_skipped']
    # F2, F3 (fast/charger/pgrep re-reads): FIX4C narrowed the rule; self_check_fix4c re-runs them on the actual parsers.
    # F4: a power gap inside the D2 extension fails the pause; the fixed-600 s summary alone would not see it.
    error, record, _ = rest_run(t_ref=30., cool_max=300, skin=lambda e: 33., gap=(700., 702.))
    assert str(error).startswith('pause power coverage missing (power_summary_with_cooling)'), error
    assert record['power_summary']['mean_battery_w'] == 1. and record['cooling']['end_skin_age_s'] is not None
    error, record, _ = rest_run(t_ref=30., cool_max=300, skin=lambda e: 33.)
    assert error is None and record['power_summary_with_cooling']['mean_battery_w'] == 1. and record['power_summary_with_cooling']['covered_s'] == 900
    # F5: screen state and screen-timeout settings get the re-read; setup has its own re-read record and cap.
    with patch.object(rt.time, 'sleep'):
        root = MagicMock(side_effect=[(1, ''), (0, 'mWakefulness=Awake\n===\ntopResumedActivity=x\n')])
        assert rt.screen_state(NS(root=root))['power'] == ['mWakefulness=Awake'] and root.call_count == 2
        assert 'error' in rt.screen_state(NS(root=MagicMock(return_value=(1, '')))), 'persistent: recorded, never raised'
        answers = iter([RuntimeError('screen setting failed: '), '60000', '', '2147483647', '', '60000'])
        def setting(self, command):
            a = next(answers)
            if isinstance(a, BaseException):raise a
            return a
        run = MagicMock(return_value=NS(returncode=0))
        with patch.object(sb.Screen, 'setting', setting), patch.object(sb.subprocess, 'run', run), contextlib.redirect_stdout(io.StringIO()):
            screen = rt.screen(sb);screen.start();screen.restore()
        assert screen.old == '60000'
    calls, result, out, files, seen = run_main([])
    assert result['setup'] == dict(read_retries=[], read_retry_check='OK') and '] END setup (preflight) OK |' in out, out
    cov = p.coverage(dict(result, setup=dict(read_retry_check='NOT VALID — READ RETRIES (4 > 3)'), plan=result['plan']))
    assert cov['read_retry_over_cap'] == ['setup'] and cov['rehearsal_pass'] is False
    # F6: a block downgraded for failed memory rows carries the cause into the session error.
    def cycle(name, length, ops, record, workers):
        record['memory'][0].update(pss_error="su rc 0; PSS missing for ['camera_provider']: x")
    record = block_run(cycle)
    assert record['failure_kind'] == 'shared' and record['error'].startswith("memory/battery read failed: su rc 0; PSS missing for ['camera_provider']"), record.get('error')
    # F7: screen state on the BatteryStop path and on a session cleanup failure; setup END line on failure.
    calls, result, out, files, seen = run_main([], battery_stop=True)
    assert result['label'].startswith('SESSION STOPPED') and 'screen_state_at_failure' in result
    calls, result, out, files, seen = run_main([], server_stop_error=RuntimeError('stop failed'))
    assert 'screen_state_at_failure' in result
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        out = io.StringIO()
        for m in (patch.object(p, 'preflight', side_effect=RuntimeError('agents/robot/other runners resident: 1 codex')),
                  patch.object(rt, 'HOME', Path(tmp)), contextlib.redirect_stdout(out), *phase_stubs()):
            stack.enter_context(m)
        try:p.main(['--output', str(Path(tmp)/'s.json')])
        except RuntimeError:pass
        assert '] END setup (preflight) FAILED: RuntimeError: agents/robot/other runners resident: 1 codex |' in out.getvalue(), out.getvalue()
    print('PASS review round 1: catch-up skips never count as camera failures (one miss after catch-up goes on); a power gap inside the D2 extension '
          'fails the pause; empty screen state and screen-timeout settings re-read once; setup re-reads recorded and capped; '
          'memory-row failures carry their cause; screen state on BatteryStop and cleanup failure; setup END line on failure')


# ---------------------------------------------------------------- review round 2 regressions

def check_round2():
    with patch.object(rt.time, 'sleep'):
        # R2-1 and the FIX4B sweep: FIX4C narrowed the rule; self_check_fix4c re-runs them on the actual parsers.
        # R2-2: an empty settings get and an empty transient mask readback are re-read once (FIX4C: a separator-only
        # screen state and the missing-policy check are non-empty answers, judged once; self_check_fix4c).
        answers = iter(['', '60000', '', '2147483647', '', '60000'])
        with patch.object(sb.Screen, 'setting', lambda self, command: next(answers)), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(sb.subprocess, 'run', return_value=NS(returncode=0)):
            mark = len(rt.READ_RETRIES);screen = rt.screen(sb);screen.start();screen.restore()
        assert screen.old == '60000' and [r['what'] for r in rt.READ_RETRIES[mark:]] == ['settings']
        unreadable = (98, 'CAMPAIGN_ROOT_MASK pid=77 actual=\n')
        mismatch = (98, "CAMPAIGN_ROOT_MASK pid=77 actual=pid 77's current affinity mask: 30\n")
        for first, retried in ((unreadable, True), (mismatch, False), ((97, 'CAMPAIGN_ROOT_MASK pid=77 actual=\n'), False)):
            root = MagicMock(side_effect=[first, (0, 'CAMPAIGN_ROOT_MASK pid=78 actual=x\nversionCode=7 versionName=1.0\n')])
            fake = NS(root=root)
            with rt.root_mask_scope(fake, sb, {0, 1}, {4, 5}):
                try:assert rt.camera_version(fake) == dict(versionCode='7', versionName='1.0') and retried
                except RuntimeError as e:assert not retried and 'pin/readback failed' in str(e) and not isinstance(e, rt.Blip), e
            assert root.call_count == (2 if retried else 1), (first, root.call_count)
        # FIX4B (R3-3): the end check runs once (a re-run could erase 'capture not stopped'); an unreadable pinned
        # readback in its root reads becomes the OSError it records as force_stop_rc None, and only the root reads are
        # redone (self_check_fix4c.check_stop_camera).
        end = dict(capture_stopped=True, force_stop_rc=0, pidof_root_rc=0, pidof_rc=1, pids_after_force_stop=[])
        check = MagicMock(return_value=end)
        with patch.object(cr, 'camera_stop'), patch.object(cr, 'camera_end_check', check):
            assert rt.stop_camera(cr) == end and check.call_count == 1
        rootf = check.call_args.kwargs['rootf']
        with patch.object(cr, 'root', side_effect=rt.Blip('transient root shell pin/readback failed')):
            try:rootf('pidof com.pixelrobot.robotcam; echo "pidof_rc=$?"', 'pidof')
            except OSError as e:assert 'pin/readback failed' in str(e)
            else:raise AssertionError('readback Blip not turned into OSError')
        # A listed RobotCam pid (non-empty) is read once; a pin/readback mismatch is not absorbed into the row.
        sample = dict(root_rc=0, battery_status='Discharging', pss_kb={}, pss_pids={}, pss_error="su rc 0; PSS missing for ['robotcam_app']: x")
        root = MagicMock(return_value=(1, '4321\npidof_rc=0\n'))
        with patch.object(cr, 'root_sample', return_value=dict(sample)), patch.object(cr, 'meminfo_mib', return_value={}), patch.object(cr, 'root', root):
            assert 'app absence unconfirmed' in rt.memory_sample(cr, 1, False)['pss_error'] and root.call_count == 1
        mismatch_error = RuntimeError('transient root shell pin/readback failed: mismatch')
        root = MagicMock(side_effect=mismatch_error)
        with patch.object(cr, 'root_sample', return_value=dict(sample)), patch.object(cr, 'meminfo_mib', return_value={}), patch.object(cr, 'root', root):
            try:rt.memory_sample(cr, 1, False)
            except RuntimeError as e:assert e is mismatch_error and root.call_count == 1
            else:raise AssertionError('non-read pidof failure absorbed into the row')
    # Between-phase capacity checks count in the phase they guard (main passes the mark taken before them).
    with patch.object(rt.time, 'sleep'):
        mark = len(rt.READ_RETRIES);rt.reread('battery capacity', blip_once(85))
        error, record, _ = rest_run(mark=mark)
        assert error is None and [r['what'] for r in record['read_retries']] == ['battery capacity']
        mark = len(rt.READ_RETRIES);rt.reread('battery capacity', blip_once(85))
        record = block_run(lambda *a: None, mark=mark)
        assert [r['what'] for r in record['read_retries']] == ['battery capacity'] and record['validity'] == 'VALID'
    calls, result, out, files, seen = run_main([])
    assert all(isinstance(c[2].get('mark'), int) for c in calls if c[0] == 'rest' and 'cool_max' in c[2])
    assert [isinstance(c[2].get('mark'), int) for c in calls if c[0] == 'block'] == [True]*7  # warm-up + 6 blocks
    # R2-3: the lag probe's fast rows carry their re-reads; more than 3 make the probe incomplete.
    for blips, complete in ((2, True), (4, False)):
        result = lag_run(blips)
        assert [r['retries'] for r in result['fast']][:blips] == [1]*blips and result['complete'] is complete, (blips, result.get('errors'))
        assert result['read_retry_check'] == ('OK' if complete else 'NOT VALID — READ RETRIES (4 > 3)')
    # R2-4: cleanup failures count in rehearsal coverage; both causes kept.
    calls, result, out, files, seen = run_main(['--dry-run'], server_stop_error=RuntimeError('stop failed'), screen_restore_error=RuntimeError('timeout restore'))
    assert result['cleanup_error'] == 'server: RuntimeError: stop failed; screen: RuntimeError: timeout restore', result['cleanup_error']
    cov = result['rehearsal_coverage']
    assert cov['rehearsal_pass'] is False and 'cleanup: server: RuntimeError: stop failed; screen: RuntimeError: timeout restore' in cov['errors'], cov
    # R2-5: preflight unwinding CoresLost keeps it when the screen restore also fails; alone, the cleanup error is raised.
    for failure in (cr.CoresLost('cores 4-7 not all allowed preflight'), None):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            screen = NS(restore=MagicMock(side_effect=RuntimeError('timeout restore')))
            def preflight(names, out, result, resources):
                resources['screen'] = screen
                if failure:raise failure
            for m in (patch.object(p, 'preflight', side_effect=preflight), patch.object(rt, 'HOME', Path(tmp)),
                      contextlib.redirect_stdout(io.StringIO()), *phase_stubs()):
                stack.enter_context(m)
            try:p.main(['--preflight', '--output', str(Path(tmp)/'s.json')])
            except BaseException as e:raised = e
            else:raised = None
            result = json.loads((Path(tmp)/'s.json').read_text())
        assert result['cleanup_error'] == 'screen: RuntimeError: timeout restore'
        if failure:assert raised is failure and result['error'].startswith('CoresLost'), raised
        else:assert str(raised) == 'preflight cleanup failed: screen: RuntimeError: timeout restore', raised
    print('PASS review round 2: an empty settings get and an unreadable (empty) transient mask readback are re-read once (a '
          'read-back mismatch or a failed pin is not); the end check runs once (only its root reads are redone); a listed '
          'RobotCam pid and a pin/readback mismatch in the camera-OFF pidof are never re-read; lag probe rows carry their '
          're-reads and > 3 make it incomplete; cleanup failures fail rehearsal_pass with both causes; preflight keeps an '
          'unwinding CoresLost over a cleanup error and raises the cleanup error alone; between-phase capacity re-reads '
          'count in the pause or block they guard')


def lag_run(blips):
    """lag_probe.main with mocks (self_check.check_lag_lifecycle shape); the fast worker reads 6 rows through the real
    cr.fast_sample, the first `blips` of them answering once with no output (the only re-read, FIX4C) before a complete one."""
    import lag_probe as lp
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        clock = [100.]
        stack.enter_context(patch.object(lp.time, 'monotonic', side_effect=lambda: clock[0]))
        stack.enter_context(patch.object(lp.time, 'sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0]+dt)))
        good = ['50000', '50000', '50000', '300']  # BIG, MID, LITTLE, battery
        answers = ([[], good] * blips) + [good] * (6 - blips)
        monitor = (NS(close=MagicMock(), run=MagicMock(side_effect=answers)), {0, 1, 2, 3}, 10,
                   {'policies': {}, 'cpu_zones': {'BIG': '9', 'MID': '10', 'LITTLE': '11'}})
        caller = NS(close=MagicMock())
        def detect(*a):
            clock[0] += .6
            return [], 600.
        caller.detect = detect
        def sampler(shell, battery, t0, mask, measured, stop, rows, errors, verify, *args):
            rows.extend(dict(t=t, t_start=t-.01, current_now_uA=-100 if t < 20 or t > 32 else -200, current_avg_uA=None,
                             battery_w=1, voltage_now_uV=4000000) for t in range(52))
        def monitor_loop(period, read, rows, stop):
            for _ in range(6 if period == 1 else 1):rows.append(read())
        for m in (patch.object(cr, 'require_native'), patch.object(rt, 'clear_processes'), patch.object(cr, 'check_cores'),
                  patch.object(rt, 'battery_sample'), patch.object(rt, 'require_hashes', return_value={}),
                  patch.object(rt, 'dump_check'), patch.object(sb, 'load_speed_input', return_value=({'detections': [{}, {}]}, None, None)),
                  patch.object(sb, 'Screen'), patch.object(rt, 'stop_camera'), patch.object(rt, 'prepare', return_value=monitor),
                  patch.object(v, 'build', return_value=(None, caller, [])), patch.object(sb, 'check_pinning'),
                  patch.object(power, 'sampler', side_effect=sampler), patch.object(lp.threading, 'Thread', InlineThread),
                  patch.object(cr, 'monitor_loop', side_effect=monitor_loop),
                  patch.object(rt, 'dump_once', side_effect=lambda cr, timeout=30: {'t': 100., 'rc': 0, 'skin': 30, 'status': 0}),
                  patch.object(cr, 'block_limit', return_value=None), contextlib.redirect_stdout(io.StringIO())):
            stack.enter_context(m)
        output = Path(tmp)/'lag.json'
        try:lp.main(['--output', str(output)])
        except (RuntimeError, SystemExit):pass
        return json.loads(output.read_text())


if __name__ == '__main__':
    check_h1();check_h2_helper();check_h2_cap();check_h3();check_h4();check_d1();check_d2()
    check_atomic_write();check_main();check_m5();check_h5();check_m3_m4_l5_l7();check_run_md();check_round1();check_round2()
    print('PASS CAMPAIGN_P1_FIX4 self-check (offline; agents resident; proot; NOT VALID for timing)')
