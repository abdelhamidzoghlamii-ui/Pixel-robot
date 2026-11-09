"""Offline tests for b4.py with fakes: the skin gate, block order and sessions, the PT sampler, the restore cadence,
block labels and the P/PT comparability rule. No server, camera, root or phone."""
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


def good(skin=30.0, battery=90, mono=0.0, mem=('runner',)):
    return {'utc': 'T', 'mono': mono, 'caps': {p: 2000000 for p in b4.POLICIES}, 'root_rc': 0, 'skin': skin,
            'status': 0, 'battery_pct': battery, 'battery_c': 30.0, 'charge_uah': 3000000, 'voltage_uv': 4000000,
            **{f'mem_{n}': {'Pss': 1000, 'VmHWM': 2000, 'pids': [1]} for n in mem}}


def record(block='P', rows=(), **kw):
    mem = ('runner', 'llama_server') if block in b4.SERVER_BLOCKS else ('runner',)
    rec = {'block': block, 'errors': [], 'cores_before': True, 'cores_after': True, 'agents_before': '',
           'agents_after': '', 'start': good(mono=0, mem=mem), 'end': good(mono=180, mem=mem)}
    return {**rec, **kw}, list(rows)


DECISION = {'kind': 'decision', 'wall_s': 2.0, 'cores_ok': True}


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
            g = b4.gate(30.0, 'b', False)
        self.assertTrue(g['passed'])
        self.assertEqual(g['skin'], 31.0)
        self.assertIn('waited_s', g)

    def test_not_passed_after_the_limit(self):
        pm.GATE_MAX_S = 0.05
        self.skins([35.0] * 10 ** 6)
        with unittest.mock.patch('sys.stdout', io.StringIO()):
            g = b4.gate(30.0, 'b', False)
        self.assertFalse(g['passed'])

    def test_smoke_skips(self):
        self.assertEqual(b4.gate(30.0, 'b', True)['skipped'], b4.SMOKE_LABEL)


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

    def test_readings(self):
        rec, rows = record('P', [DECISION])
        rec['end'] = {**rec['end'], 'skin': None}
        self.assertTrue(b4.block_problems(rec, rows))
        rec, rows = record('P', [DECISION])
        del rec['end']['mem_llama_server']
        self.assertTrue(b4.block_problems(rec, rows))
        rec, rows = record('Y', [{'kind': 'cycle', 'cores_ok': True}])
        self.assertTrue(any('robotcam' in w for w in b4.block_problems(rec, rows)))  # RobotCam memory needed
        rec['start']['mem_robotcam'] = rec['end']['mem_robotcam'] = {'Pss': 5, 'pids': [9]}
        self.assertEqual(b4.block_problems(rec, rows), [])
        rec['camera_stop'] = {'error': 'capture not shown stopped'}
        self.assertTrue(b4.block_problems(rec, rows))

    def test_py_handoff_and_p0(self):
        rows = [DECISION, {'kind': 'cycle', 'cores_ok': True}]
        rec, _ = record('PY')
        rec['end']['mem_robotcam'] = {'Pss': 5}
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


class SamplerTest(unittest.TestCase):
    def test_samples_and_problems(self):
        t = time.monotonic()
        s = b4.Sampler(read=lambda: good(mono=time.monotonic()), period=0.05).start()
        time.sleep(0.3)
        res = s.stop()
        self.assertFalse(res['alive'])
        self.assertGreaterEqual(len(res['samples']), 4)
        self.assertEqual(res['sched']['requested_cpus'], [0, 1, 2, 3])
        self.assertEqual(b4.sampler_problems(res, t, time.monotonic()), [])
        gap = {**res, 'samples': [good(mono=t + 1), good(mono=t + 20)]}
        self.assertTrue(any('gap' in w for w in b4.sampler_problems(gap, t, t + 21)))
        bad = {**res, 'samples': res['samples'] + [{**good(mono=t), 'root_rc': 1}]}
        self.assertTrue(any('failed' in w for w in b4.sampler_problems(bad, t, time.monotonic())))
        self.assertTrue(b4.sampler_problems({**res, 'error': 'OSError: x'}, t, time.monotonic()))

    def test_slow_read_skips_slots(self):
        s = b4.Sampler(read=lambda: (time.sleep(0.12), good(mono=time.monotonic()))[1], period=0.05).start()
        time.sleep(0.4)
        res = s.stop()
        monos = [x['mono'] for x in res['samples']]
        self.assertTrue(all(b - a >= 0.1 for a, b in zip(monos, monos[1:])))  # no catch-up burst


class Picker(unittest.TestCase):
    def test_restore_before_every_fifth_and_time_limit(self):
        saved = b4.slot, b4.decide, b4.cores_ok
        acts, rows = [], []
        b4.slot = lambda action: acts.append(action) or {'action': action, 'wall_s': 0.0}
        b4.decide = lambda s, it, p: {'id': it['id'], 'wall_s': 0.01}
        b4.cores_ok = lambda: True
        try:
            s = {'items': [{'id': i} for i in b4.ITEMS]}
            prompts = {('V3', 'FWD', i): None for i in b4.ITEMS}
            b4.run_picker((s, prompts, []), 0.3, rows.append)
        finally:
            b4.slot, b4.decide, b4.cores_ok = saved
        n = len(rows)
        self.assertGreater(n, 20)  # the item list repeats
        self.assertEqual([r['n'] for r in rows if 'restore' in r], list(range(5, n + 1, 5)))
        self.assertEqual([r['id'] for r in rows[:21]], list(b4.ITEMS) + [1])
        self.assertLessEqual(rows[-1]['t'] + rows[-1]['wall_s'], 0.3 + 0.05)


class Sessions(unittest.TestCase):
    def setUp(self):
        self.saved = b4.gate, b4.run_block, b4.reading, b4.cores_ok, b4.repro.agents, b4.cr.wait_cores
        self.ran = []
        b4.reading = lambda **kw: good()
        b4.cores_ok = lambda: True
        b4.repro.agents = lambda: ''
        b4.cr.wait_cores = lambda label: None
        b4.gate = lambda ref, label, smoke: {'passed': True}

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
            self.assertEqual(self.ran, ['PT', 'V', 'Y', 'PY'])
            self.assertEqual(b4.next_blocks(b4.read_jsonl(out / 'blocks.jsonl')), [])

    def test_gate_agent_and_safety_defer_before_the_block(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.fake_blocks({})
            b4.gate = lambda ref, label, smoke: {'passed': False, 'skin': 33.0, 'waited_s': 480}
            self.assertIn('deferred before block I', self.session(out))
            b4.gate = lambda ref, label, smoke: {'passed': True}
            b4.repro.agents = lambda: '42 claude'
            self.assertIn('agent processes running', self.session(out))
            b4.repro.agents = lambda: ''
            b4.reading = lambda **kw: {**good(), 'status': b4.STATUS_STOP}
            self.assertIn('stopped before block I', self.session(out))
            self.assertEqual(self.ran, [])
            self.assertEqual(b4.read_jsonl(out / 'blocks.jsonl'), [])

    def test_smoke_runs_every_block_once_unenforced(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake_blocks({k: b4.SMOKE_LABEL for k in b4.ORDER})
            b4.repro.agents = lambda: '42 claude'
            self.assertEqual(self.session(Path(d), smoke=True), 'all blocks done')
            self.assertEqual(self.ran, list(b4.ORDER))

    def test_signal_in_a_block_keeps_the_record_and_stops(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            b4.run_block = lambda k, o, c, e: ({'block': k, 'label': 'INVALID', 'problems': ['x'],
                                                'interrupted': True}, [])
            with self.assertRaises(SystemExit):
                self.session(out)
            self.assertEqual(len(b4.read_jsonl(out / 'blocks.jsonl')), 1)


class Readings(unittest.TestCase):
    def test_parse_root_output(self):
        out = ('=== battery\n88\nDischarging\n301\n2900000\n4100000\nrc=0\n'
               '=== thermalservice\nThermal Status: 1\nCurrent temperatures from HAL:\n'
               '\tTemperature{mValue=33.5, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\nrc=0\n'
               '=== mem_llama_server\npid: 7\nRss: 300 kB\nPss: 200 kB\nVmHWM: 400 kB\nVmRSS: 300 kB\nrc=0\n'
               '=== mem_robotcam\npid: 8\nPss: 10 kB\npid: 9\nPss: 5 kB\nVmHWM: 20 kB\nrc=0\n')
        saved = cr.root
        cr.root = lambda script, tag, timeout: (0, out)
        try:
            c = b4.reading({'llama_server': 7})
        finally:
            cr.root = saved
        self.assertEqual((c['battery_pct'], c['charge_uah'], c['status']), (88, 2900000, 1))
        self.assertEqual(c['mem_llama_server']['Pss'], 200)
        self.assertEqual(c['mem_robotcam'], {'pids': [8, 9], 'Pss': 15, 'VmHWM': 20, 'rc': 0})
        self.assertEqual(c['skin'], 33.5)

    def test_report(self):
        p = {**record('P')[0], 'label': 'VALID', 'session': 1, 'start': {**good(30.0), 'avail_kb': 3000000}}
        pt = {**record('PT')[0], 'label': 'VALID', 'session': 2, 'warm': {'prefix_tokens': 731, 'prompt_n': 731,
                                                                         'save': {'wall_s': 0.01}}}
        py = {**record('PY')[0], 'label': 'INVALID', 'problems': ['x'], 'session': 2, 'handoff': good(mono=90),
              'camera': {'start_s': 2.0, 'affinity_start': [4, 5]}, 'camera_stop': {'stopped_s': 1.0}}
        dec = {'kind': 'decision', 'wall_s': 6.0, 'reason_tokens': 13, 'gen_tok_s': 6.2, 'step1_prompt_n': 85,
               'step1_cache_n': 731, 'step2_prompt_n': 3, 'step2_cache_n': 828, 'endpoint_s': {'/tokenize': 0.06}}
        rows = [{**dec, 'block': 'P', 'session': 1}, {**dec, 'block': 'PT', 'session': 2, 'wall_s': 6.6},
                {**dec, 'block': 'PY', 'session': 2},
                {'kind': 'cycle', 'block': 'PY', 'session': 2, 't': 0.0, 'age_s': 0.4, 'size': 320, 'detect_ms': 150},
                {'kind': 'cycle', 'block': 'PY', 'session': 2, 't': 1.8, 'age_s': 0.6, 'size': 640, 'detect_ms': 520}]
        recs = [{**p, 'problems': []}, {**pt, 'problems': []}, py]
        run = {'utc': 'T', 'code_sha256': {'b4.py': 'c' * 64}, 'git_head': 'h', 'command': ['x'], 'start': good(),
               'smoke': False, 'work_s': 180}
        text = b4.report(run, recs, rows)
        self.assertIn('P vs PT (monitoring cost): COMPARABLE', text)
        self.assertIn('PT/P 1.100', text)
        self.assertIn('== PY: INVALID', text)
        self.assertIn(' Y phase start', text)
        self.assertIn('== V: not run', text)

    def test_heat_and_energy(self):
        a, b = good(30.0, mono=0), {**good(33.0, mono=180), 'charge_uah': 2990000}
        self.assertEqual(b4.heat(a, b), (3.0, 3.0, 1.0))
        self.assertEqual(b4.energy(a, b), (10.0, 144.0))


if __name__ == '__main__':
    unittest.main()
