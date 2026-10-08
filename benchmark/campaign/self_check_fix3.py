"""CAMPAIGN_P1_FIX3 focused offline mocks. Never run hardware."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import diagnostics as d
import phase1 as p
import runtime as rt
from self_check import FastOps, phase_stubs

HERE = Path(__file__).resolve().parent
OWNER = HERE/'runs/owner_session_p1_fix2.json'
BASE = '914ce55e792a76f7ba24726bb739e7c4b531dbec'
PROVIDER = '=== camera_provider 1048\n TOTAL PSS: 244984\n'
APP = '=== robotcam_app 4321\n TOTAL PSS: 38000\n'


def root_text(app=False, provider=True, battery='-232812\n4122812\nDischarging\n'):
    """coresidency.root_sample's su output shape; the owner idle had no robotcam_app section."""
    return (battery+'=== runner 29452\n TOTAL PSS: 71809\n=== llama_server 31733\n TOTAL PSS: 3426951\n'
            + (APP if app else '')+(PROVIDER if provider else ''))


NO_APP = (0, 'pidof_rc=1\n')  # status-checked pidof: nothing matched


def rooted(sample, pidof=NO_APP):
    """Mock cr.root: root_sample's su call by tag, the campaign's absence confirmation otherwise."""
    return lambda command, tag, timeout=60: sample if tag == 'sample' else pidof


def bad(row):
    """Unchanged phase1 rest/live_block memory predicate."""
    return row.get('root_rc') != 0 or row.get('pss_error') or row.get('battery_status') != 'Discharging'


def check_memory_sample():
    _, _, _, cr, _ = rt.imports()
    owner = json.loads(OWNER.read_text())['idle_phases'][0]['memory']
    assert len(owner) == 37 and all(r['pss_error'].startswith("su rc 0; PSS missing for ['robotcam_app']: ") for r in owner)
    meminfo = {'mem_available_mib': 2640, 'swap_used_mib': 0, 'cached_mib': 1}
    # The 37 owner samples, as root_sample returned them: pre-fix all fail, fixed camera OFF passes, camera ON fails.
    for row in owner:
        sample = {k: v for k, v in row.items() if k not in ('t', *meminfo)}
        with patch.object(cr, 'meminfo_mib', return_value=meminfo), patch.object(cr, 'root_sample', side_effect=lambda pid: json.loads(json.dumps(sample))), \
             patch.object(cr, 'root', side_effect=rooted(None)):
            assert bad(dict(**cr.meminfo_mib(), **cr.root_sample(1)))  # pre-fix expression
            off = rt.memory_sample(cr, 1, False)
            assert not bad(off) and off['robotcam_app'] == 'not running (camera OFF)' and off['pss_kb']['robotcam_app'] is None
            assert off['pss_kb']['llama_server'] == row['pss_kb']['llama_server']
            assert bad(rt.memory_sample(cr, 1, True))
    # The real root_sample parser on the owner's output shape reproduces the owner's message exactly.
    cases = [  # (root rc, su output, camera_on, must fail[, pidof confirmation reply])
        (0, root_text(), False, False),
        (0, root_text(), False, True, (0, 'pidof_rc=2\n')),           # pidof itself failed: enumeration unknown
        (0, root_text(), False, True, (0, '')),                        # confirmation output lost
        (0, root_text(), False, True, (1, 'pidof_rc=1\n')),           # confirmation root call failed
        (0, root_text(), False, True, (0, '4321\npidof_rc=0\n')),     # app exists but was not sampled
        (0, root_text(), True, True),                       # camera ON, app missing
        (0, root_text(app=True), True, False),
        (0, root_text(app=True), False, False),             # app still cached while OFF: measured, fine
        (1, root_text(), False, True),                      # root failure
        (0, root_text(provider=False), False, True),        # camera provider missing: never optional
        (0, root_text(battery='\n'), False, True),          # battery fields missing
        (0, root_text(battery='-1\n4000000\nCharging\n'), False, True),
        (0, root_text().replace('=== llama_server', '=== llama_srv'), False, True),
        (0, root_text(app=True).replace(' TOTAL PSS: 38000', 'garbage'), False, True),  # app present, unparsable
    ]
    for rc, text, camera_on, fails, *pidof in cases:
        with patch.object(cr, 'root', side_effect=rooted((rc, text), *pidof)), patch.object(cr, 'meminfo_mib', return_value=meminfo):
            row = rt.memory_sample(cr, 31733, camera_on)
        assert bool(bad(row)) == fails, (rc, camera_on, row)
    with patch.object(cr, 'root', return_value=(0, root_text())), patch.object(cr, 'meminfo_mib', return_value=meminfo):
        n = owner[0]['pss_error'].index('runner 29452')+len('runner 29452')  # owner bytes then diverge (dumpsys body)
        assert cr.root_sample(31733)['pss_error'][:n] == owner[0]['pss_error'][:n]
    with patch.object(cr, 'root', side_effect=rooted((0, root_text()))), patch.object(cr, 'meminfo_mib', side_effect=KeyError('MemAvailable')):
        try:rt.memory_sample(cr, 1, False)
        except KeyError:pass
        else:raise AssertionError('missing MemAvailable accepted')
    print('PASS 37 owner camera-OFF samples: pre-fix fail, fixed pass, camera ON fail; root/battery/provider/server/parse and unconfirmed-absence (pidof) failures still fail')


def rest_harness(module, camera_on, text, pidof=NO_APP, late_failure=False, late_dump=None, late_row=None, reads=None):
    """self_check_fix2.check_rest wiring, with the real root_sample parser behind a mocked root.
    late_dump: the thermal reader runs once more during its join and returns this row (FIX3B);
    late_row: this row is appended to the thermal rows during the join, bypassing the reader;
    reads: rt.dump_once replacement, with the real rt.dump_check (FIX3D)."""
    v, sb, pm, cr, _ = rt.imports()
    with contextlib.ExitStack() as stack:
        clock = [100.]; pending = []; joining = []
        monitor = (NS(close=MagicMock()), {0}, 1, {'policies': {'policy0': 1803000}})
        class Stop:
            done = False
            def set(self):self.done = True
            def is_set(self):return self.done
            def wait(self, dt):clock[0] += dt;return self.done
        class Deferred:
            def __init__(self, target, args=(), **kwargs):self.target, self.args = target, args
            def start(self):pending.append(self)
            def join(self, timeout=None):
                # late_failure: the memory reader runs once more while being joined, after the deadline loop.
                if late_failure and self.args and self.args[0] == 5.13:joining.append(1);self.target(*self.args)
                if self.args and self.args[0] == 4.87:
                    if late_dump is not None:dump[0] = late_dump;self.target(*self.args)
                    if late_row is not None:self.args[2].append(late_row)
            def is_alive(self):return False
        dump = [{'t': 100., 'rc': 0, 'skin': 30, 'status': 0}]
        def meminfo():
            if joining:raise KeyError('MemAvailable')
            return {'mem_available_mib': 2640}
        def block_limit(*args):
            workers = pending[:];pending.clear()
            for worker in workers:worker.target(*worker.args)
        def monitor_loop(period, read, rows, stop):rows.append(read())
        def sampler(shell, battery, began, mask, measured, stop, rows, errors, verify):
            rows.extend(dict(t=t, t_start=t-.01, battery_w=1) for t in range(-1, 602))
        for m in (patch.object(module.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(module.threading, 'Event', Stop),
                  patch.object(module.threading, 'Thread', Deferred), patch.object(rt, 'prepare', side_effect=[monitor, (NS(close=MagicMock()), {0}, 2, monitor[3])]),
                  patch.object(rt, 'root_mask_scope', return_value=contextlib.nullcontext()), patch.object(d, 'battery', return_value=85),
                  patch.object(d, 'snapshot', side_effect=lambda cr, on: dict(camera_on=on)), patch.object(rt, 'stop_camera'),
                  patch.object(pm, 'camera_start', return_value={'session': 'mock'}), patch.object(rt, 'clear_processes'),
                  patch.object(rt, 'battery_sample'), patch.object(sb, 'battery_sample'), patch.object(cr, 'check_cores'),  # sb: the base loop
                  patch.object(rt, 'dump_check', **({'wraps': rt.dump_check} if reads else {})),
                  patch.object(rt, 'fast_check', return_value={'t': 100., 'max': {'policy0': 1803000}}),
                  patch.object(rt, 'dump_once', side_effect=reads or (lambda cr, timeout=30: dict(dump[0]))),
                  patch.object(cr, 'meminfo_mib', side_effect=meminfo),
                  patch.object(cr, 'root', side_effect=rooted((0, text), pidof)),
                  patch.object(cr, 'monitor_loop', side_effect=monitor_loop), patch.object(cr, 'block_limit', side_effect=block_limit),
                  patch.object(module.power, 'sample', return_value={'t': -1, 't_start': -1, 'battery_w': 1}),
                  patch.object(module.power, 'sampler', side_effect=sampler), patch.object(module.os, 'sched_setaffinity'),
                  patch.object(module.os, 'sched_getaffinity', return_value={0}), patch.object(pm, 'capped_by_policy', return_value={}),
                  *phase_stubs()):
            stack.enter_context(m)
        record = {}
        try:module.rest('IDLE', 180, NS(proc=NS(pid=31733), alive=lambda: True), record, camera_on)
        except RuntimeError as e:return str(e), record
        return None, record


def base_phase1():
    try:source = subprocess.run(['git', '-C', str(HERE), 'show', f'{BASE}:benchmark/campaign/phase1.py'],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):return None
    path = Path(tempfile.mkdtemp())/'phase1_base.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('phase1_base', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def check_rest_camera_states():
    base = base_phase1()
    if base is None:print('SKIP base-module rest reproduction: git history unavailable here (see checks/p1_fix3)')
    else:
        error, _ = rest_harness(base, False, root_text())
        assert error == 'pause memory/battery read failed', error
        print('PASS pre-fix reproduction: base 914ce55 rest() camera OFF, app absent -> "pause memory/battery read failed"')
        error, record = rest_harness(base, True, root_text(app=True), late_failure=True)
        assert error is None and record['monitor_errors'], (error, record.get('monitor_errors'))
        print('PASS pre-fix reproduction (review r2 F1): base rest() returns normally despite a late MemAvailable failure')
    error, record = rest_harness(p, False, root_text())
    assert error is None and record['memory'] and all(r['robotcam_app'] == 'not running (camera OFF)' for r in record['memory']), (error, record.get('memory'))
    assert rest_harness(p, True, root_text())[0].startswith('pause memory/battery read failed: ')
    assert rest_harness(p, True, root_text(app=True))[0] is None
    assert rest_harness(p, False, root_text(provider=False))[0].startswith('pause memory/battery read failed: ')
    assert rest_harness(p, False, root_text(), (0, 'pidof_rc=2\n'))[0].startswith('pause memory/battery read failed: ')
    for camera_on in (False, True):
        error, record = rest_harness(p, camera_on, root_text(app=camera_on), late_failure=True)
        assert error.startswith('pause monitor failed') and 'after deadline' in record['error'], (error, record.get('error'))
        assert p.coverage(dict(idle_phases=[record], pauses=[], blocks=[], plan=[]))['idle_phases_done'] == 0
    print('PASS fixed rest(): camera OFF app absent passes, camera ON app absent fails, camera ON app present passes, provider missing/pidof failure/late reader failure fail')


SCALE = 10  # virtual s per real s: 1 virtual s = 100 ms


class SlowOps(FastOps):
    """FastOps with 1 virtual s = 100 ms and a live M2 call of 3 virtual s (the phone's take 4-5 s). Flake at base
    (3 of 4 runs in proot, where every syscall is traced): 1 virtual s = 1 ms, and the instant live M2 launched
    at slot 20 could store its context before the slot-20 selector read the forced fallback one."""
    def now(self):return (time.monotonic()-self.base)*SCALE
    def relate(self, image, boxes):
        if image != 'fixed':time.sleep(3/SCALE)
        return [], 1


def check_forced_fallback():
    original = threading.Event.wait
    def fast_wait(event, timeout=None):return original(event, None if timeout is None else max(0, timeout/SCALE))
    def conclusive(name, target, record):
        """The run reached the forced 640 frame and the slot-20 selector read came before a later live M2 ended;
        a scheduling stall in proot can skip a slot (correct catch-up behaviour), which proves nothing here."""
        reached = name == 'L2' or any(r['size'] == 640 and r['slot_s'] == target for r in record['yolo'])
        read = next((r['started_s'] for r in record['selector'] if r['slot_s'] == 20), None)
        later = [r['ended_s'] for r in record['m2'] if r['slot_s'] > target and r['scene'] == 'live']
        return reached and (name == 'L2' or read is not None and all(read < end for end in later))
    for name in ('L1', 'L2', 'L3', 'L4'):
        target = 0 if name == 'L2' else 10 if name == 'L1' else 15  # D1: L1 runs M2 at 0/10/20 s
        for forced in (False, True):
            for attempt in range(5):
                ops = SlowOps(name);ops.detect = lambda image, size: [{}]*3
                ops.fallback = MagicMock(return_value=('fixed', [{}, {}]))
                if forced:ops.force_fallback_slot = target
                record = {}
                with patch.object(threading.Event, 'wait', fast_wait):p.run_cycle(name, 25, ops, record)
                if conclusive(name, target, record):
                    break
            else:raise AssertionError(f'{name} forced={forced}: 5 inconclusive runs (scheduling stalls)')
            fallback = [r['slot_s'] for r in record['m2'] if r['scene'] == 'fallback']
            assert fallback == ([target] if forced else []), (name, forced, fallback)
            assert all(r.get('fallback_forced', False) == (r['scene'] == 'fallback') for r in record['m2'])
            assert record['summary']['m2']['live_calls'] >= 1
            fallback_contexts = [r['slot_s'] for r in record['selector'] if r['context'].startswith('(fallback scene)')]
            assert fallback_contexts == ([target if name == 'L2' else 20] if forced else []), (name, forced, fallback_contexts)
    print('PASS rehearsal forces only the selector-feeding M2 slot (L2 0 s, L1 10 s, L3/L4 15 s) onto the real fallback path; '
          'live M2 and a fallback-context selector both run per layout; session unchanged')


def run_main(argv, block_failure=None, no_live=None):
    """Real main() control flow; preflight/rest/live_block mocked, arguments recorded."""
    calls = []; server = NS(stop=MagicMock()); screen = NS(restore=MagicMock())
    def preflight(names, out, result, resources):
        calls.append(('preflight',));resources.update(server=server, screen=screen)
    def rest(label, duration, srv, record, camera_on=False, dry=False, **d2):
        calls.append(('rest', label, duration, camera_on, dry))
        record.update(label=label, duration_s=duration)
    def block(name, path, srv, idle, dry=False, duration=180, **mark):
        calls.append(('block', name, dry, duration))
        m2 = name != 'L0' and name != no_live
        record = dict(block=name, validity=p.REHEARSAL if dry else 'VALID', skin_start={'skin': 30},
                      summary={'m2': {'live_calls': 4 if m2 else 0, 'fallback_calls': 1} if name != 'L0' else {}, 'max_live_boxes': 7},
                      selector=[{'context': None if name == 'L0' else '(fallback scene)\nM2 relations:'}, {'context': 'x'}])
        if name == block_failure:record.update(error='RuntimeError: mock', failure_kind='shared')
        p.write(path, record);return record
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        for m in (patch.object(p, 'preflight', side_effect=preflight), patch.object(p, 'rest', side_effect=rest),
                  patch.object(p, 'live_block', side_effect=block), patch.object(d, 'battery', return_value=85),
                  patch.object(rt, 'HOME', Path(tmp)), contextlib.redirect_stdout(io.StringIO()), *phase_stubs()):
            stack.enter_context(m)
        output = Path(tmp)/'out.json'
        try:p.main(argv+['--output', str(output)])
        except RuntimeError:assert block_failure
        result = json.loads(output.read_text())
        server.stop.assert_called_once();screen.restore.assert_called_once()
        return calls, result


def check_modes():
    calls, result = run_main(['--dry-run'])
    rests = [c for c in calls if c[0] == 'rest'];blocks = [c for c in calls if c[0] == 'block']
    assert calls[0] == ('preflight',)
    assert [(c[2], c[3], c[4]) for c in rests[:3]] == [(6, False, False), (6, False, False), (6, True, False)]
    assert all(c[1].startswith('NOT VALID — REHEARSAL: ') for c in rests)
    assert [(c[2], c[3]) for c in rests[3:]] == [(6, False)]*6
    assert blocks == [('block', 'L0', True, 20)]+[('block', n, True, 25) for n in p.DEFAULT]
    assert result['label'].startswith(p.REHEARSAL+' COMPLETE') and result['setup_complete']
    assert result['warmup']['validity'] == 'WARM-UP — NOT A RESULT' and all(b['validity'] == p.REHEARSAL for b in result['blocks'])
    cov = result['rehearsal_coverage']
    assert cov['rehearsal_pass'] is True and cov['pauses_done'] == 6 and cov['idle_phases_done'] == 3 and cov['errors'] == [], cov
    assert [(b['block'], b['m2_live_calls'], b['m2_fallback_calls'], b['selector_fallback_contexts']) for b in cov['blocks']] == \
        [('L0', 0, 0, 0), ('L1', 4, 1, 1), ('L2', 4, 1, 1), ('L3', 4, 1, 1), ('L4', 4, 1, 1), ('L0', 0, 0, 0)], cov
    # One layout without a live M2 call fails the rehearsal even though other layouts had live calls.
    assert run_main(['--dry-run'], no_live='L2')[1]['rehearsal_coverage']['rehearsal_pass'] is False
    # Full session: identical schedule to fix2, never rehearsal.
    calls, result = run_main([])
    assert [(c[2], c[3], c[4]) for c in calls if c[0] == 'rest'] == [(180, False, False), (60, False, False), (60, True, False)]+[(600, False, False)]*6
    assert all(not c[1].startswith('NOT VALID') for c in calls if c[0] == 'rest')
    assert [c[1:] for c in calls if c[0] == 'block'] == [('L0', False, 180)]+[(n, False, 180) for n in p.DEFAULT]
    assert 'rehearsal_coverage' not in result and result['label'] == 'SESSION COMPLETE; inspect individual validity'
    # Rehearsal stops on a shared block failure exactly as the session does, and still reports coverage.
    calls, result = run_main(['--dry-run'], block_failure='L2')
    assert [b['block'] for b in result['rehearsal_coverage']['blocks']] == ['L0', 'L1', 'L2'] and result['unrun_blocks'] == ['L3', 'L4', 'L0']
    assert result['rehearsal_coverage']['rehearsal_pass'] is False
    assert result['rehearsal_coverage']['errors'] == ['L2: RuntimeError: mock', 'session: RuntimeError: block invalidates later blocks: RuntimeError: mock']
    try:p.main(['--mock', '--output', '/nonexistent/x.json'])
    except SystemExit as e:assert e.code == 2
    else:raise AssertionError('--mock without --dry-run accepted')
    print('PASS --dry-run = live rehearsal (setup, 3 idle incl. camera ON, warm-up, 6 pauses, 6 blocks, cleanup, coverage); session schedule unchanged; --mock needs --dry-run')


def check_live_block_rehearsal():
    v, sb, pm, cr, _ = rt.imports()
    for dry, duration in ((False, 180), (True, 25)):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            monitor = (NS(close=MagicMock()), {0}, 10, {'policies': {'policy0': 1}, 'cpu_zones': {}})
            seen = {}
            for m in (patch.object(rt, 'prepare', return_value=monitor), patch.object(pm, 'build_detector', return_value=(NS(close=MagicMock()), {'worker_tids': [], 'caller_tid': 1})),
                      patch.object(v, 'build', return_value=(None, NS(close=MagicMock()), [])), patch.object(rt, 'clear_processes'), patch.object(cr, 'check_cores'),
                      patch.object(rt, 'battery_sample'), patch.object(rt, 'fast_check', return_value={}), patch.object(pm, 'camera_start', return_value={'session': 's'}),
                      patch.object(rt, 'dump_check', return_value={'skin': 30, 'status': 0}), patch.object(p, 'bounded_selector'), patch.object(cr, 'load_cases', return_value=[{}]),
                      patch.object(p.power, 'sample', return_value={'t': 0, 't_start': 0, 'battery_w': 1}),
                      patch.object(p.threading, 'Thread', return_value=NS(start=lambda: None, join=lambda timeout=None: None, is_alive=lambda: False)),
                      patch.object(rt, 'stop_camera'), patch.object(cr, 'lmk_lines', return_value={'ok': True, 'n_kills': 0}), patch.object(cr, 'block_limit', return_value=None),
                      patch.object(d, 'snapshot', return_value={}), patch.object(rt, 'fallback_input', return_value=('img', [{}, {}])),
                      patch.object(pm, 'capped_by_policy', side_effect=lambda b: seen.setdefault('caps', b['duration_s'])), *phase_stubs()):
                stack.enter_context(m)
            def cycle(name, length, ops, record):
                seen.update(length=length, forced=ops.force_fallback_slot)
                origin = record['cycle_origin']
                record.update(duration_s=length, m2=[], selector=[], yolo=[], reads=[])
                record['power'] = [dict(t=i, t_start=i-.01, battery_w=1) for i in range(-1, length+2)]
                record['memory'] = [dict(t=origin, mem_available_mib=1, root_rc=0, pss_kb={'llama_server': 1}, battery_status='Discharging')]
            stack.enter_context(patch.object(p, 'run_cycle', side_effect=cycle))
            record = p.live_block('L4', Path(tmp)/'b.json', NS(proc=NS(pid=1), alive=lambda: True), None, dry, duration)
            assert seen == dict(length=duration, forced=15 if dry else None, caps=duration), seen
            lo, hi = record['lmk_window_epoch_s'];assert abs(hi-lo-duration) < 1e-6
            assert record['power_summary']['mean_battery_w'] == 1
            assert record['validity'] == (p.REHEARSAL if dry else 'VALID') and record.get('rehearsal_block_validity') == ('VALID' if dry else None)
            assert json.loads((Path(tmp)/'b.json').read_text())['validity'] == record['validity']
    print('PASS live_block rehearsal: shortened duration reaches cycle/LMK/power/caps, first-fallback flag, NOT VALID label on file')


if __name__ == '__main__':
    check_memory_sample();check_rest_camera_states();check_forced_fallback();check_modes();check_live_block_rehearsal()
