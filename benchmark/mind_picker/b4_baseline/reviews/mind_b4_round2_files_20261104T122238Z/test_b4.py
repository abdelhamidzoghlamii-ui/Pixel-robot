"""Offline tests for b4.py with fakes: the skin gate, block order and sessions, the PT sampler and the watcher, the
work-phase deadlines (fake clock), camera ownership and validity, readings, block labels, the frozen definition, the
V70/V140 commands and the P/PT comparability rule. No server, camera, root or phone."""
import io
import json
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4  # noqa: E402

cr, pm = b4.cr, b4.pm
SCHED = {'tid': 1, 'requested_cpus': [0, 1, 2, 3], 'requested_nice': 10, 'cpus': [0, 1, 2, 3], 'nice': 10}


def good(skin=30.0, battery=90, mono=0.0, mem=('runner',)):
    return {'utc': 'T', 'mono': mono, 'caps': {p: 2000000 for p in b4.POLICIES}, 'root_rc': 0, 'battery_rc': 0,
            'dump_rc': 0, 'skin': skin, 'status': 0, 'battery_pct': battery, 'battery_c': 30.0, 'charge_uah': 3000000,
            'voltage_uv': 4000000, **{f'mem_{n}': {'Pss': 1000, 'VmHWM': 2000, 'pids': [1], 'rc': 0} for n in mem}}


def watch(hits=()):
    return {'period_s': 30, 'sched': SCHED, 'error': None, 'alive': False,
            'samples': [{'mono': 0, 'hits': []}, {'mono': 30, 'hits': list(hits)}, {'mono': 60, 'hits': []}]}


def record(block='P', rows=(), **kw):
    mem = ('runner', 'llama_server') if block in b4.SERVER_BLOCKS else ('runner',)
    rec = {'block': block, 'errors': [], 'cores_before': True, 'cores_after': True, 'agents_before': '',
           'agents_after': '', 'start': good(mono=0, mem=mem), 'end': good(mono=180, mem=mem), 'watch': watch(),
           'phases': [{'phase': block, 'overrun_s': 2.0}]}
    return {**rec, **kw}, list(rows)


DECISION = {'kind': 'decision', 'wall_s': 2.0, 'cores_ok': True}
CYCLE = {'kind': 'cycle', 'cores_ok': True}


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Clocked(unittest.TestCase):
    """b4.clock / b4.pause replaced by a fake clock that only operations advance."""

    def setUp(self):
        self.saved = b4.clock, b4.pause, b4.cores_ok
        self.c = FakeClock()
        b4.clock, b4.pause, b4.cores_ok = self.c, self.c.sleep, lambda: True

    def tearDown(self):
        b4.clock, b4.pause, b4.cores_ok = self.saved


class Gate(unittest.TestCase):
    def setUp(self):
        self.saved = cr.read_thermal, cr.read_dump, cr.GATE_POLL_S, pm.GATE_MAX_S, cr.SKIN_GATE_C
        cr.read_thermal = lambda path: {'at': 'T', 'z9': 30000, 'z10': 30000, 'z11': 30000}
        cr.GATE_POLL_S, cr.SKIN_GATE_C = 0, b4.SKIN_MATCH_C

    def tearDown(self):
        cr.read_thermal, cr.read_dump, cr.GATE_POLL_S, pm.GATE_MAX_S, cr.SKIN_GATE_C = self.saved

    def skins(self, values):
        it = iter(values)
        cr.read_dump = lambda: {'skin': next(it), 'status': 0}

    def test_waits_until_skin_within_one_degree(self):
        pm.GATE_MAX_S = 60
        self.skins([32.0, 31.5, 31.0])
        with unittest.mock.patch('sys.stdout', io.StringIO()):
            g = b4.gate(30.0, 'b')
        self.assertTrue(g['passed'])
        self.assertEqual(g['skin'], 31.0)
        self.assertIn('waited_s', g)

    def test_timed_run_waits_30_min_smoke_keeps_the_default(self):
        stop = unittest.mock.Mock(side_effect=SystemExit('stop'))
        pm.GATE_MAX_S = 480
        with unittest.mock.patch.object(b4.b2, 'load_set', stop), unittest.mock.patch.object(cr, 'require_native', stop), \
                unittest.mock.patch.object(b4.signal, 'signal'):
            self.assertRaises(SystemExit, b4.main, ['--smoke'])
            self.assertEqual(pm.GATE_MAX_S, 480)
            self.assertRaises(SystemExit, b4.main, [])
            self.assertEqual(pm.GATE_MAX_S, 30 * 60)

    def test_not_passed_after_the_limit(self):
        pm.GATE_MAX_S = 0.05
        self.skins([35.0] * 10 ** 6)
        with unittest.mock.patch('sys.stdout', io.StringIO()):
            g = b4.gate(30.0, 'b')
        self.assertFalse(g['passed'])


class Labels(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(b4.block_problems(*record('P', [DECISION])), [])
        self.assertEqual(b4.block_problems(*record('I')), [])

    def test_agent_cores_errors(self):
        self.assertTrue(b4.block_problems(*record('P', [DECISION], agents_after='123 claude')))
        self.assertTrue(b4.block_problems(*record('P', [DECISION], cores_before=False)))
        self.assertTrue(b4.block_problems(*record('P', [{**DECISION, 'cores_ok': False}])))
        self.assertTrue(b4.block_problems(*record('P', [DECISION, {**DECISION, 'error': 'x'}])))
        self.assertTrue(b4.block_problems(*record('P', [DECISION], errors=['RuntimeError: x'])))
        self.assertTrue(b4.block_problems(*record('P')))  # no decisions

    def test_violation_between_clean_endpoints(self):
        rec = record('I', watch=watch(['agent pid 42: node /usr/bin/claude']))
        self.assertEqual(rec[0]['agents_before'] + rec[0]['agents_after'], '')
        self.assertTrue(any('agent pid 42' in w for w in b4.block_problems(*rec)))
        rec = record('P', [DECISION], watch=watch(['llama-server Cpus_allowed_list:\t4-5']))
        self.assertTrue(any('llama-server' in w for w in b4.block_problems(*rec)))
        rec = record('Y', [CYCLE], watch=watch(["detector threads {7: [0, 1, 2, 3, 4, 5]}"]))
        self.assertTrue(any('detector threads' in w for w in b4.block_problems(*rec)))

    def test_overrun_limit(self):
        self.assertEqual(b4.block_problems(*record('P', [DECISION], phases=[{'phase': 'P', 'overrun_s': 15.0}])), [])
        self.assertTrue(b4.block_problems(*record('P', [DECISION], phases=[{'phase': 'P', 'overrun_s': 15.1}])))

    def test_readings(self):
        rec, rows = record('P', [DECISION])
        rec['end'] = {**rec['end'], 'skin': None}
        self.assertTrue(b4.block_problems(rec, rows))
        rec, rows = record('P', [DECISION])
        del rec['end']['mem_llama_server']
        self.assertTrue(b4.block_problems(rec, rows))
        rec, rows = record('Y', [CYCLE])
        self.assertTrue(any('robotcam' in w for w in b4.block_problems(rec, rows)))  # RobotCam memory needed
        rec['start']['mem_robotcam'] = rec['end']['mem_robotcam'] = {'Pss': 5, 'VmHWM': 6, 'pids': [9], 'rc': 0}
        self.assertEqual(b4.block_problems(rec, rows), [])
        rec['camera_stop'] = {'error': 'capture not shown stopped'}
        self.assertTrue(b4.block_problems(rec, rows))

    def test_partial_reads(self):
        base = good(mem=('runner', 'llama_server'))
        need = ['runner', 'llama_server']
        self.assertEqual(b4.reading_problems(base, 's', need), [])
        cases = {'battery_rc': 1, 'dump_rc': None, 'voltage_uv': None, 'charge_uah': None, 'status': None}
        for k, v in cases.items():
            self.assertTrue(b4.reading_problems({**base, k: v}, 's', need), k)
        self.assertTrue(b4.reading_problems({**base, 'caps': {'policy0': 'error: x'}}, 's', need))
        no_hwm = {**base, 'mem_llama_server': {'Pss': 1, 'pids': [1], 'rc': 1}}  # Pss read, status read failed
        why = b4.reading_problems(no_hwm, 's', need)
        self.assertTrue(any('VmHWM' in w for w in why) and any('rc 1' in w for w in why))

    def test_py_handoff_and_p0(self):
        rows = [DECISION, CYCLE]
        rec, _ = record('PY')
        rec['end']['mem_robotcam'] = {'Pss': 5, 'VmHWM': 6, 'rc': 0}
        self.assertIn('handoff reading missing', b4.block_problems(rec, rows))
        rec['handoff'] = good(mono=90, mem=('runner', 'llama_server', 'robotcam'))
        self.assertEqual(b4.block_problems(rec, rows), [])
        rec, _ = record('P0')
        self.assertTrue(b4.block_problems(rec, [{'kind': 'cold_erase'}]))
        self.assertEqual(b4.block_problems(rec, [{'kind': 'cold_erase'}, {'kind': 'cold_restore'}]), [])

    def test_exit_code_is_a_finding_not_invalid(self):
        self.assertEqual(b4.block_problems(*record('P', [DECISION], exit_code=1, killed=False)), [])


class Comparability(unittest.TestCase):
    def pair(self, skin=30.5, battery=85, label='VALID'):
        p = {'block': 'P', 'label': 'VALID', 'start': good(30.0, 90)}
        pt = {'block': 'PT', 'label': label, 'start': good(skin, battery)}
        return b4.comparable(p, pt)

    def test_rule(self):
        self.assertEqual(self.pair()[0], 'COMPARABLE')
        self.assertEqual(self.pair(skin=31.0, battery=80)[0], 'COMPARABLE')  # limits inclusive
        self.assertEqual(self.pair(skin=31.1)[0], 'NOT COMPARABLE')
        self.assertEqual(self.pair(battery=79)[0], 'NOT COMPARABLE')
        self.assertEqual(self.pair(label='INVALID')[0], 'NOT COMPARABLE')


class Threads(unittest.TestCase):
    def test_sampler_samples_and_problems(self):
        t = time.monotonic()
        s = b4.Sampler(read=lambda: good(mono=time.monotonic()), period=0.05).start()
        time.sleep(0.3)
        res = s.stop()
        self.assertFalse(res['alive'])
        self.assertGreaterEqual(len(res['samples']), 4)
        self.assertEqual((res['sched']['cpus'], res['sched']['nice']), ([0, 1, 2, 3], 10))
        self.assertEqual(b4.sampler_problems(res, t, time.monotonic()), [])
        gap = {**res, 'samples': [good(mono=t + 1), good(mono=t + 20)]}
        self.assertTrue(any('gap' in w for w in b4.sampler_problems(gap, t, t + 21)))
        bad = {**res, 'samples': res['samples'] + [{**good(mono=t), 'root_rc': 1}]}
        self.assertTrue(any('failed' in w for w in b4.sampler_problems(bad, t, time.monotonic())))
        self.assertTrue(b4.sampler_problems({**res, 'error': 'OSError: x'}, t, time.monotonic()))

    def test_pt_sampler_needs_cores_0_3_nice_10(self):
        res = {'period_s': 5, 'error': None, 'alive': False, 'samples': [good(mono=1), good(mono=6)]}
        self.assertEqual(b4.sampler_problems({**res, 'sched': SCHED}, 0, 7), [])
        for sched in ({**SCHED, 'cpus': [0, 1, 2, 3, 4, 5]}, {**SCHED, 'nice': 0}, None):
            self.assertTrue(any('need' in w for w in b4.sampler_problems({**res, 'sched': sched}, 0, 7)))

    def test_slow_read_skips_slots(self):
        s = b4.Sampler(read=lambda: (time.sleep(0.12), good(mono=time.monotonic()))[1], period=0.05).start()
        time.sleep(0.4)
        res = s.stop()
        monos = [x['mono'] for x in res['samples']]
        self.assertTrue(all(b - a >= 0.1 for a, b in zip(monos, monos[1:])))  # no catch-up burst

    def test_watch_scans_at_start_between_and_at_stop(self):
        hits = iter([[], ['agent pid 7: codex'], []] + [[]] * 100)
        w = b4.Periodic(lambda: {'mono': time.monotonic(), 'hits': next(hits)}, 0.05).start()
        time.sleep(0.12)
        res = w.stop()
        self.assertGreaterEqual(len(res['samples']), 3)  # at once, every period, once more at stop
        self.assertTrue(any('codex' in p for p in b4.watch_problems(res)))
        self.assertEqual(b4.watch_problems({**watch(), 'sched': SCHED}), [])

    def test_watch_check_affinities(self):
        class Det:
            def __init__(self, cpus):
                self.cpus = cpus

            def affinity(self):
                return {11: self.cpus, 12: [4, 5]}
        saved = b4.cores_ok, b4.AGENT_RE
        b4.cores_ok, b4.AGENT_RE = (lambda: True), b4.re.compile('no-such-agent-name-xyz')
        try:
            self.assertEqual(b4.watch_check({'detector': Det([4, 5])})['hits'], [])
            self.assertTrue(b4.watch_check({'detector': Det([4, 5, 6])})['hits'])
            self.assertTrue(b4.watch_check({'server_pid': 10 ** 9})['hits'])  # unreadable -> a hit
        finally:
            b4.cores_ok, b4.AGENT_RE = saved
        self.assertEqual(b4.parse_cpus('4-7'), {4, 5, 6, 7})
        self.assertEqual(b4.parse_cpus('0-1,4,6-7'), {0, 1, 4, 6, 7})


class Deadlines(Clocked):
    def picker(self, durations, dur=180.0, restore_s=0.0):
        it = iter(durations)
        acts, rows = [], []

        def slot(action):
            self.c.sleep(restore_s)
            acts.append(action)
            return {'action': action, 'wall_s': restore_s}

        def decide(s, item, p):
            d = next(it)
            self.c.sleep(d)
            return {'id': item['id'], 'wall_s': d}
        saved = b4.slot, b4.decide
        b4.slot, b4.decide = slot, decide
        try:
            s = {'items': [{'id': i} for i in b4.ITEMS]}
            rec = {'work_s': dur}
            b4.phase(rec, 'P', lambda d: b4.run_picker((s, {('V3', 'FWD', i): None for i in b4.ITEMS}, []), d,
                                                       rows.append))
        finally:
            b4.slot, b4.decide = saved
        return rows, rec['phases'][0]['overrun_s']

    def test_variable_durations(self):
        rows, over = self.picker([7, 3, 11, 5] * 30)
        self.assertEqual(sum(r['wall_s'] for r in rows[:-1]) < 180, True)  # the last one started before the deadline
        self.assertEqual(round(sum(r['wall_s'] for r in rows) - 180, 3), over)
        self.assertLessEqual(over, 11)
        self.assertEqual([r['n'] for r in rows if 'restore' in r], list(range(5, len(rows) + 1, 5)))
        self.assertEqual([r['id'] for r in rows[:21]], list(b4.ITEMS) + [1])

    def test_oversized_first_call(self):
        rows, over = self.picker([200])
        self.assertEqual((len(rows), over), (1, 20.0))
        self.assertTrue(b4.block_problems(*record('P', rows, phases=[{'phase': 'P', 'overrun_s': over}])))

    def test_restore_overhead_counts(self):
        rows, over = self.picker([9.0] * 50, restore_s=4.0)
        spent = sum(r['wall_s'] + r.get('restore', {}).get('wall_s', 0) for r in rows)
        self.assertEqual(round(spent - 180, 3), over)
        self.assertLess(spent - (rows[-1]['wall_s'] + rows[-1].get('restore', {}).get('wall_s', 0)), 180)

    def test_idle_and_vision(self):
        rec = {'work_s': 180.0}
        b4.phase(rec, 'I', lambda d: b4.pause(max(0.0, d - b4.clock())))
        self.assertEqual(rec['phases'][0]['overrun_s'], 0.0)
        saved = b4.post
        b4.post = lambda path, body: (self.c.sleep(25), {'choices': [{'message': {'content': 'x'}}], 'timings': {}})[1]
        try:
            rows, rec = [], {'work_s': 180.0}
            fr = {i: ({'frame_file': 'f.jpg'}, 'AAAA') for i in b4.FRAME_IDS}
            b4.phase(rec, 'V70', lambda d: b4.run_vision(fr, 'V70', d, rows.append))
        finally:
            b4.post = saved
        self.assertEqual((len(rows), rec['phases'][0]['overrun_s']), (8, 20.0))  # 8th starts at 175 s


class FakeRobot:
    def __init__(self, frames, detect_s=0.2, fail_start=False):
        self.frames, self.detect_s, self.fail_start = iter(frames), detect_s, fail_start
        self.camera_session, self.capturing, self.stops = None, False, 0
        self.detector = self
        self.size_policy = self

    def start_camera(self):
        self.capturing = True
        if self.fail_start:
            raise RuntimeError('RobotCam did not publish a usable frame')

    def stop_camera(self):
        self.stops += 1
        self.capturing = False

    def next_size(self):
        return 320

    def observe(self, dets, h):
        pass

    def detect(self, image, size):
        b4.clock.sleep(self.detect_s)
        return []

    def affinity(self):
        return {1: [4, 5]}

    def config(self):
        return {}


class Image:
    height = 480


class Camera(Clocked):
    def setUp(self):
        super().setUp()
        self.saved2 = b4.read_frame, b4.move_s
        b4.move_s = lambda: 1.5

    def tearDown(self):
        b4.read_frame, b4.move_s = self.saved2
        super().tearDown()

    def run_frames(self, frames, detect_s=0.2, dur=180.0):
        robot, rows, rec = FakeRobot([]), [], {'work_s': dur}
        it = iter(frames)
        b4.read_frame = lambda **kw: next(it)
        robot.detect_s = detect_s
        b4.phase(rec, 'Y', lambda d: b4.run_camera(robot, d, rows.append))
        return rows, rec['phases'][0]['overrun_s']

    def ok(self, n):
        return {'status': 'ok', 'session': 's', 'frame': n, 'age_s': 0.5, 'image': Image()}

    def test_good_cycle_then_persistent_failure_is_invalid(self):
        rows, _ = self.run_frames([self.ok(1)] + [{'status': 'missing'}] * 1000)
        self.assertEqual([r['kind'] for r in rows], ['cycle', 'camera_stop'])  # the phase ends at the first one
        rec, _ = record('Y', rows)
        rec['start']['mem_robotcam'] = rec['end']['mem_robotcam'] = {'Pss': 5, 'VmHWM': 6, 'rc': 0}
        self.assertTrue(any('camera stop: missing' in w for w in b4.block_problems(rec, rows)))

    def test_no_new_allowed_three_times(self):
        frames = [self.ok(1), self.ok(1), self.ok(1), self.ok(1)] + [self.ok(n) for n in range(2, 500)]
        rows, _ = self.run_frames(frames)
        self.assertEqual(sum(r['kind'] == 'camera_stop' for r in rows), 3)
        rec, _ = record('Y', rows)
        rec['start']['mem_robotcam'] = rec['end']['mem_robotcam'] = {'Pss': 5, 'VmHWM': 6, 'rc': 0}
        self.assertEqual(b4.block_problems(rec, rows), [])
        rows, _ = self.run_frames([self.ok(1)] * 10 + [self.ok(n) for n in range(2, 500)])
        self.assertEqual(sum(r['kind'] == 'camera_stop' for r in rows), 4)  # the 4th ends the phase
        self.assertTrue(any('"no new"' in w for w in b4.block_problems(record('Y', rows)[0], rows)))

    def test_camera_overrun_and_last_move_cut(self):
        rows, over = self.run_frames([self.ok(n) for n in range(1, 500)], detect_s=0.5)
        self.assertEqual(over, 0.0)  # the move after the last cycle is cut at the deadline
        self.assertTrue(all(r['t'] < 180 for r in rows))
        self.assertEqual(round(rows[1]['t'] - rows[0]['t'], 3), 2.0)
        rows, over = self.run_frames([self.ok(n) for n in range(1, 500)], detect_s=40.0)
        self.assertGreater(over, 0)
        self.assertLessEqual(over, 40.0)


class Ownership(unittest.TestCase):
    """A camera start that begins capture and then raises: stop_camera is still called; every cleanup runs."""

    def setUp(self):
        self.saved = (b4.make_robot, b4.reading, b4.cores_ok, b4.repro.agents, b4.read_frame, b4.watch_check,
                      b4.start_server, b4.stop_server, b4.picker_setup, b4.warm, b4.stop_camera)
        b4.reading = lambda pids=None, **kw: good()
        b4.cores_ok = lambda: True
        b4.repro.agents = lambda: ''
        b4.read_frame = lambda **kw: {'status': 'missing'}
        b4.watch_check = lambda targets: {'mono': 0, 'hits': []}

    def tearDown(self):
        (b4.make_robot, b4.reading, b4.cores_ok, b4.repro.agents, b4.read_frame, b4.watch_check,
         b4.start_server, b4.stop_server, b4.picker_setup, b4.warm, b4.stop_camera) = self.saved

    def ctx(self, smoke=False):
        return {'session': 1, 'smoke': smoke, 'dur': 0.05, 'test': {}, 'frames': {}}

    def test_failed_start_is_stopped(self):
        robot = FakeRobot([], fail_start=True)
        b4.make_robot = lambda: robot
        rec, rows = b4.run_block('Y', Path('.'), self.ctx(), lambda r: None)
        self.assertEqual(robot.stops, 1)
        self.assertFalse(robot.capturing)
        self.assertEqual(rec['label'], 'INVALID')

    def test_each_cleanup_runs_when_another_fails(self):
        robot = FakeRobot([], fail_start=True)
        b4.make_robot = lambda: robot
        stopped = []

        class Proc:
            pid = 4242
        b4.start_server = lambda out, name, cmd: (Proc(), None, 0.1)
        b4.stop_server = lambda proc, log: stopped.append(proc) or (0, False)
        b4.picker_setup = lambda test: ({'items': [{'id': 1}]}, {('V3', 'FWD', 1): None}, [])
        b4.warm = lambda prefix: {}
        b4.stop_camera = lambda r: (_ for _ in ()).throw(OSError('broadcast failed'))
        with unittest.mock.patch.object(b4.Periodic, 'stop', side_effect=RuntimeError('join')), \
                unittest.mock.patch.object(b4, 'decide', return_value={'id': 1, 'wall_s': 0.0}), \
                unittest.mock.patch.object(b4, 'slot', return_value={'action': 'restore', 'wall_s': 0.0}):
            rec, rows = b4.run_block('PY', Path('.'), {**self.ctx(), 'dur': 0.01}, lambda r: None)
        self.assertEqual(len(stopped), 1)  # the server is stopped although watch and camera cleanup failed
        self.assertTrue(any('cleanup camera' in e for e in rec['errors']))
        self.assertTrue(any('cleanup watch' in e for e in rec['errors']))
        self.assertEqual(rec['label'], 'INVALID')

    def test_initial_checks_abort_before_any_setup(self):
        b4.repro.agents = lambda: '42 claude'
        b4.start_server = lambda *a: self.fail('server started')
        b4.make_robot = lambda: self.fail('camera started')
        rec, rows = b4.run_block('PY', Path('.'), self.ctx(), lambda r: None)
        self.assertEqual(rec['label'], 'INVALID')
        self.assertIn('no workload started', ' '.join(rec['errors']))


class Definition(unittest.TestCase):
    def test_changes_are_refused(self):
        d = {'files': {'main.py': 'a', 'benchmark/x.py': 'b'}, 'commands': {'P': 'x'}, 'frames': {'1': 'f'}}
        self.assertEqual(b4.definition_diff(d, json.loads(json.dumps(d))), [])
        self.assertEqual(b4.definition_diff(d, {**d, 'files': {**d['files'], 'main.py': 'z'}}),
                         [('files', 'main.py', 'a', 'z')])
        self.assertTrue(b4.definition_diff(d, {**d, 'frames': {'1': 'g'}}))
        self.assertTrue(b4.definition_diff(d, {**d, 'files': {**d['files'], 'server_manager.py': 'c'}}))  # new dep
        self.assertTrue(b4.definition_diff(d, {**d, 'commands': {'P': 'x --threads 8'}}))

    def test_real_definition(self):
        fr = b4.frames()
        with tempfile.TemporaryDirectory() as t:
            d = b4.definition(Path(t), fr)
        files = d['files']
        for f in ('benchmark/mind_picker/b4_baseline/b4.py', 'benchmark/mind_picker/b4_baseline/oneshot.sh',
                  'main.py', 'server_manager.py', 'detect_person.py', 'benchmark/mind_picker/bc_build/bc.py'):
            self.assertIn(f, files)
        self.assertEqual(sorted(d['frames']), sorted(str(i) for i in b4.FRAME_IDS))
        self.assertEqual(set(d['commands']), set(b4.SERVER_BLOCKS))

    def test_resume_refuses_a_changed_frame(self):
        fr = b4.frames()
        with tempfile.TemporaryDirectory() as t:
            before = b4.definition(Path(t), fr)
            frame = b4.bv.FRAMES_DIR / Path(fr[1][0]['frame_file']).name
            with unittest.mock.patch.object(b4, 'sha', side_effect=lambda b: 'changed' if b == frame.read_bytes()
                                            else b4.bc.sha(b)):
                after = b4.definition(Path(t), fr)
        self.assertEqual([x[:2] for x in b4.definition_diff(before, after)], [('frames', '1')])


class Commands(unittest.TestCase):
    def test_v70_v140(self):
        out = Path('/run/x')
        base = b4.bc.command(out / 'slots')
        for k in ('P', 'P0', 'PT', 'PY', 'V70'):
            self.assertEqual(b4.server_command(out, k), base)
        v140 = b4.server_command(out, 'V140')
        diff = [(a, b) for a, b in zip(base, v140) if a != b]
        self.assertEqual(diff, [('70', '140'), ('70', '140')])
        self.assertEqual(v140[v140.index('--image-min-tokens') + 1], '140')
        self.assertEqual(b4.ORDER, ('I', 'P', 'P0', 'PT', 'V70', 'V140', 'Y', 'PY'))


class Sessions(unittest.TestCase):
    def setUp(self):
        self.saved = b4.gate, b4.run_block, b4.reading, b4.cores_ok, b4.repro.agents, b4.cr.wait_cores
        self.ran = []
        b4.reading = lambda **kw: good()
        b4.cores_ok = lambda: True
        b4.repro.agents = lambda: ''
        b4.cr.wait_cores = lambda label: None
        b4.gate = lambda ref, label: {'passed': True}

    def tearDown(self):
        b4.gate, b4.run_block, b4.reading, b4.cores_ok, b4.repro.agents, b4.cr.wait_cores = self.saved

    def fake_blocks(self, labels):
        def run_block(k, out, ctx, emit):
            self.ran.append(k)
            return {'block': k, 'label': labels.get(k, 'VALID'), 'problems': []}, []
        b4.run_block = run_block

    def session(self, out, smoke=False):
        ctx = {'smoke': smoke, 'session': 1, 'dur': 1}
        with open(out / 'blocks.jsonl', 'a') as blog, unittest.mock.patch('sys.stdout', io.StringIO()):
            return b4.session_loop(out, {'start': {'skin': 30.0}}, ctx, lambda r: None, blog)

    def test_order_invalid_stops_and_resume_redoes_it(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.fake_blocks({'PT': 'INVALID'})
            self.assertIn('block PT INVALID', self.session(out))
            self.assertEqual(self.ran, ['I', 'P', 'P0', 'PT'])
            self.ran.clear()
            self.fake_blocks({})
            self.assertEqual(self.session(out), 'all blocks done')
            self.assertEqual(self.ran, ['PT', 'V70', 'V140', 'Y', 'PY'])
            self.assertEqual(b4.next_blocks(b4.read_jsonl(out / 'blocks.jsonl')), [])

    def test_gate_agent_and_safety_defer_before_the_block(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.fake_blocks({})
            b4.gate = lambda ref, label: {'passed': False, 'skin': 33.0, 'waited_s': 480}
            self.assertIn('deferred before block I', self.session(out))
            b4.gate = lambda ref, label: {'passed': True}
            b4.repro.agents = lambda: '42 claude'
            self.assertIn('agent processes running', self.session(out))
            b4.repro.agents = lambda: ''
            b4.reading = lambda **kw: {**good(), 'status': b4.STATUS_STOP}
            self.assertIn('stopped before block I', self.session(out))
            self.assertEqual(self.ran, [])
            self.assertEqual(b4.read_jsonl(out / 'blocks.jsonl'), [])

    def test_smoke_runs_every_block_once_gating_only_the_first(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake_blocks({k: b4.SMOKE_LABEL for k in b4.ORDER})
            gated = []
            b4.gate = lambda ref, label: gated.append(label) or {'passed': True}
            b4.repro.agents = lambda: '42 claude'
            self.assertEqual(self.session(Path(d), smoke=True), 'all blocks done')
            self.assertEqual(self.ran, list(b4.ORDER))
            self.assertEqual(gated, [f'block {b4.ORDER[0]}'])

    def test_timed_session_gates_every_block(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake_blocks({})
            gated = []
            b4.gate = lambda ref, label: gated.append(label) or {'passed': True}
            self.assertEqual(self.session(Path(d)), 'all blocks done')
            self.assertEqual(gated, [f'block {k}' for k in b4.ORDER])

    def test_signal_in_a_block_keeps_the_record_and_stops(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            b4.run_block = lambda k, o, c, e: ({'block': k, 'label': 'INVALID', 'problems': ['x'],
                                                'interrupted': True}, [])
            with self.assertRaises(SystemExit):
                self.session(out)
            self.assertEqual(len(b4.read_jsonl(out / 'blocks.jsonl')), 1)


class Readings(unittest.TestCase):
    OUT = ('=== battery\n88\nDischarging\n301\n2900000\n4100000\nrc=0\n'
           '=== thermalservice\nThermal Status: 1\nCurrent temperatures from HAL:\n'
           '\tTemperature{mValue=33.5, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\nrc=0\n'
           '=== mem_llama_server\npid: 7\nRss: 300 kB\nPss: 200 kB\nVmHWM: 400 kB\nVmRSS: 300 kB\nrc=0\n'
           '=== mem_robotcam\npid: 8\nPss: 10 kB\npid: 9\nPss: 5 kB\nVmHWM: 20 kB\nrc=0\n')

    def read(self, out):
        saved = cr.root
        cr.root = lambda script, tag, timeout: (0, out)
        try:
            return b4.reading({'llama_server': 7})
        finally:
            cr.root = saved

    def test_parse_root_output(self):
        c = self.read(self.OUT)
        self.assertEqual((c['battery_pct'], c['charge_uah'], c['status'], c['skin']), (88, 2900000, 1, 33.5))
        self.assertEqual((c['battery_rc'], c['dump_rc']), (0, 0))
        self.assertEqual(c['mem_llama_server']['Pss'], 200)
        self.assertEqual(c['mem_robotcam'], {'pids': [8, 9], 'Pss': 15, 'VmHWM': 20, 'rc': 0})
        self.assertEqual(b4.reading_problems(c, 's', ['llama_server', 'robotcam']), [])

    def test_unfinished_and_failed_sections(self):
        cut = self.OUT.split('=== mem_robotcam')[0].replace('VmHWM: 400 kB\nVmRSS: 300 kB\nrc=0', 'rc=1')
        c = self.read(cut)
        why = b4.reading_problems(c, 's', ['llama_server', 'robotcam'])
        self.assertTrue(any('llama_server section rc 1' in w for w in why))
        self.assertTrue(any('robotcam section rc None' in w for w in why))
        c = self.read(self.OUT.replace('Thermal Status: 1\n', '').replace('4100000\nrc=0', 'rc=1'))
        why = b4.reading_problems(c, 's')
        self.assertTrue(any('battery_rc 1' in w for w in why) and any('status missing' in w for w in why))

    def test_root_scripts_parse_in_cr_root_wrapper(self):
        for light in (True, False):  # the native smoke failed on '; ;' (light script, trailing ';')
            script = b4.root_script({'runner': 1, 'llama_server': 2}, light)
            wrapped = f'{{ {script}; }} </dev/null >/dev/null 2>&1; rc=$?; exit $rc'  # coresidency.root's form
            r = b4.subprocess.run(['sh', '-n', '-c', wrapped], capture_output=True, text=True)
            self.assertEqual((r.returncode, r.stderr), (0, ''), light)

    def test_incomplete_launch_start_refuses_cleanly(self):
        saved = b4.reading
        try:
            for bad in ({'skin': None}, {'status': None}, {'voltage_uv': None}, {'root_rc': 1, 'root_stderr': 'x'}):
                b4.reading = lambda **kw: {**good(), **bad}
                with self.assertRaises(SystemExit) as e:
                    b4.launch_start()
                self.assertIn('incomplete launch-start reading', str(e.exception))
            b4.reading = lambda **kw: good(31.0)
            self.assertEqual(b4.launch_start()['skin'], 31.0)
        finally:
            b4.reading = saved

    def test_heat_and_energy(self):
        a, b = good(30.0, mono=0), {**good(33.0, mono=180), 'charge_uah': 2990000}
        self.assertEqual(b4.heat(a, b), (3.0, 3.0, 1.0))
        self.assertEqual(b4.energy(a, b), (10.0, 144.0))

    def test_report(self):
        p = {**record('P')[0], 'label': 'VALID', 'problems': [], 'session': 1,
             'start': {**good(30.0), 'avail_kb': 3000000}}
        pt = {**record('PT')[0], 'label': 'VALID', 'problems': [], 'session': 2,
              'warm': {'prefix_tokens': 731, 'prompt_n': 731, 'save': {'wall_s': 0.01}}}
        py = {**record('PY')[0], 'label': 'INVALID', 'problems': ['x'], 'session': 2, 'handoff': good(mono=90),
              'camera': {'start_s': 2.0, 'affinity_start': [4, 5]}, 'camera_stop': {'stopped_s': 1.0}}
        dec = {'kind': 'decision', 'wall_s': 6.0, 'reason_tokens': 13, 'gen_tok_s': 6.2, 'step1_prompt_n': 85,
               'step1_cache_n': 731, 'step2_prompt_n': 3, 'step2_cache_n': 828, 'endpoint_s': {'/tokenize': 0.06}}
        rows = [{**dec, 'block': 'P', 'session': 1}, {**dec, 'block': 'PT', 'session': 2, 'wall_s': 6.6},
                {**dec, 'block': 'PY', 'session': 2},
                {'kind': 'cycle', 'block': 'PY', 'session': 2, 't': 0.0, 'age_s': 0.4, 'size': 320, 'detect_ms': 150},
                {'kind': 'cycle', 'block': 'PY', 'session': 2, 't': 1.8, 'age_s': 0.6, 'size': 640, 'detect_ms': 520}]
        run = {'utc': 'T', 'git_head': 'h', 'start': good(), 'smoke': False, 'work_s': 180,
               'definition': {'files': {'benchmark/mind_picker/b4_baseline/b4.py': 'c' * 64},
                              'commands': {'P': 'x', 'V140': 'y'}, 'frames': {}}}
        text = b4.report(run, [p, pt, py], rows)
        self.assertIn('P vs PT (monitoring cost): COMPARABLE', text)
        self.assertIn('PT/P 1.100', text)
        self.assertIn('== PY: INVALID', text)
        self.assertIn(' Y phase start', text)
        self.assertIn('== V70: not run', text)
        self.assertIn('== V140: not run', text)
        self.assertIn('code cccccccccccc', text)


if __name__ == '__main__':
    unittest.main()
