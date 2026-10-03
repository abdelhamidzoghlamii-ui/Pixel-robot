#!/usr/bin/env python3
"""Offline checks only: no su, Android services, real camera, model or server."""
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_map as pm
cr = pm.cr


def synthetic(spec, duration=180):
    reads = []
    policy = pm.size_policy(spec['cadence'])
    if policy:
        policy.last_large = 0
    for i in range(duration * spec['rate']):
        t = i / spec['rate']
        reads.append({'t': t, 'status': 'ok', 'size': pm.next_size(policy, t), 'detect_ms': 100,
                      'read_ms': 3, 'decode_ms': 10, 'age_s': .3})
    return {'block': spec['name'], 'spec': spec, 'mode': 'mix', 'gemma': True,
            'duration_s': duration, 'planned_s': duration, 'reads': reads,
            'heat_stop': {'reached_limit': False, 'limit': None, 'reason': None, 'reading': None,
                          'time_to_limit_s': None},
            'camera_end': {'capture_stopped': True, 'force_stop_rc': 0},
            'thermal_start': {'z9': 30000, 'waited_s': 2, 'warm_start': False}, 'thermal_end': {'z9': 50000},
            'cpuinfo_max_khz': dict.fromkeys(pm.POLICIES, 2000000),
            'fast': [{'t': i, 'max': {'policy0': 1000000, 'policy4': 2000000, 'policy6': 1500000}}
                     for i in range(1, duration)],
            'dumps': [{'t': i, 'skin': 32 + i / 60, 'status': 0 if i < 60 else 1}
                      for i in range(0, duration, 5)],
            'samples': [{'t': i, 't_end': i + 1, 'mem_available_mib': 2500, 'swap_used_mib': 1000,
                         'pss_kb': {}, 'z9': 50000, 'battery_w': 4, 'battery_status': 'Discharging'}
                        for i in range(0, duration, 5)],
            'selector_calls': [{'t': i, 'started_s': i, 'ended_s': i + 2, 'ms': 2000, 'correct': True}
                               for i in range(0, duration, 20)],
            'survived': {'robotcam_process': True, 'robotcam_new_frame_at_end': True, 'llama_server': True},
            'lmk': {'ok': True, 'n_lines': 2, 'n_kills': 0},
            'ort': {'version': 'fake', 'worker_tids': [44], 'caller_tid': None,
                    'intra_op_num_threads': {'320': 0, '640': 0},
                    'observed_default_intra_threads_per_session': 2},
            'thread_samples': [{'t': 1, 'threads': [{'tid': 44, 'processor': 4, 'allowed': '0-7', 'cpu_ticks': 100}]}]}


def check_blocks():
    assert [s['name'] for s in pm.blocks_for('R1')] == ['R1_r2_c5', 'R1_r2_c10', 'R1_r2_off']
    assert [s['name'] for s in pm.blocks_for('R2')] == ['R2_r1_c5', 'R2_r1_c10', 'R2_r1_off']
    assert [s['name'] for s in pm.blocks_for('R3', 1, '10')] == ['R3_default', 'R3_mid', 'R3_little']
    assert pm.blocks_for('CONFIRM', 2, 'off', 'little')[0]['threads'] == 'little'
    for args in [('R1', 1), ('R3',), ('R3', 2, '5', 'mid'), ('CONFIRM', 1, '5')]:
        try:
            pm.blocks_for(*args)
            raise AssertionError(args)
        except ValueError:
            pass
    print('block lists and CLI setting validation: PASS')


def check_cadence_and_loop():
    for rate in (1, 2):
        for cadence in ('5', '10', 'off'):
            policy = pm.size_policy(cadence)
            if policy:
                policy.last_large = 0
            sizes = [pm.next_size(policy, i/rate) for i in range(21 * rate)]
            wanted = [640 if cadence != 'off' and i > 0 and i % (int(cadence)*rate) == 0 else 320
                      for i in range(21*rate)]
            assert sizes == wanted, (rate, cadence, sizes)
            # Exercise the real pacing loop with a virtual clock and synthetic frames.
            clock = [0.0]
            reads, detected = [], []
            class Detector:
                def detect(self, image, size):
                    detected.append(size)
                    return []
            class Policy(cr.SizePolicy):
                def __init__(self, **kw):
                    super().__init__(**kw)
                    self.last_large = 0
                def next_size(self, now=None):
                    return super().next_size(clock[0])
            def frame(*args, **kw):
                return {'status': 'ok', 'session': 'x', 'frame': len(detected)+1,
                        'age_s': .3, 'image': None}
            with patch.object(pm.time, 'monotonic', lambda: clock[0]), \
                 patch.object(pm.time, 'perf_counter', lambda: clock[0]), \
                 patch.object(pm.time, 'sleep', lambda s: clock.__setitem__(0, clock[0]+s)), \
                 patch.object(cr, 'SizePolicy', Policy), patch.object(cr, 'read_frame', frame), \
                 patch.object(cr, 'check_cores', lambda when: None):
                last, hit = pm.frame_loop(Detector(), {'rate': rate, 'cadence': cadence}, 'x', 0, 21, reads, lambda: None)
            assert detected == wanted and last == 21*rate and hit is None
    hit = {'limit': 'battery'}
    reads = []
    assert pm.frame_loop(None, {'cadence': 'off'}, 'x', pm.time.monotonic(), 20, reads, lambda: hit) == (None, hit)
    assert not reads
    print('5/10/off size sequences and frame pacing at rates 1/2; limit before first read: PASS')


def check_numbers():
    b = synthetic(pm.blocks_for('R1')[0], 10)
    b['fast'] = [{'t': t, 'max': {'policy0': p0, 'policy4': p4, 'policy6': p6}}
                 for t, p0, p4, p6 in [(-1, 1, 1, 1), (1, 1e6, 2e6, 2e6), (2, 1e6, 1e6, 2e6),
                                        (5, 2e6, 1e6, 1e6), (6, 2e6, 2e6, None), (11, 1, 1, 1)]]
    caps = pm.capped_by_policy(b)
    assert caps['policy0']['seconds'] == 2 and caps['policy0']['percent'] == 20
    assert caps['policy4']['seconds'] == 4 and caps['policy4']['percent'] == 40
    assert caps['policy6']['seconds'] == 3 and caps['policy6']['unknown_s'] == 1
    assert caps['policy6']['lowest_mhz'] == 1000
    dumps = [{'t': t, 'skin': 31 + t*.025} for t in (0, 5, 15, 20)]
    assert abs(pm.skin_slope(dumps, 20) - 1.5) < 1e-10
    assert pm.skin_slope([{'t': 0, 'skin': None}], 20) is None
    assert abs(pm.skin_slope([{'t': -5, 'skin': 80}] + dumps + [{'t': 25, 'skin': 80}], 20)-1.5) < 1e-10
    print('separate capped policies, skipped gaps, unknown readings, excluded edges/tail; least-squares skin slope: PASS')


def check_report():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        specs = pm.blocks_for('R1')
        (out / 'run.json').write_text(json.dumps({'set': 'R1', 'blocks': specs, 'smoke': False, 'server_cmd': ['fake']}))
        for spec in specs:
            (out / f'block_{spec["name"]}.json').write_text(json.dumps(synthetic(spec)))
        text, bad = pm.report(out)
        assert bad == [], bad
        for name in ('R1_r2_c5', 'R1_r2_c10', 'R1_r2_off', 'policy0 capped', 'policy4 capped', 'policy6 capped',
                     'least-squares', 'CPU-core reading, not a heat state', '60.0s LIGHT(1)', '1.000'):
            assert name in text, name
        path = out / f'block_{specs[0]["name"]}.json'
        b = json.loads(path.read_text())
        b['fast'][0]['max']['policy4'] = None
        path.write_text(json.dumps(b))
        assert any('scaling_max' in p for p in pm.report(out)[1])
        path.unlink()
        assert any('block missing' in p for p in pm.report(out)[1])
    b = synthetic(pm.blocks_for('R3', 1, 'off')[1])
    b['thread_samples'][0]['threads'][0]['allowed'] = '4-5'
    assert pm.observed_workers(b)[1] == []
    b['thread_samples'].append({'t': 2, 'threads': [{'tid': 44, 'processor': 6, 'allowed': '4-5', 'cpu_ticks': 101}]})
    assert pm.observed_workers(b)[1]
    b['ort']['caller_tid'] = 55
    assert any('no complete' in e for e in pm.observed_workers(b)[1])
    # Round-1 regression: idle 640-session worker retains its pre-pin CPU 6.
    b = synthetic(pm.blocks_for('R3', 1, 'off')[2])
    b['thread_samples'] = [
        {'t': t, 'threads': [{'tid': 44, 'processor': cpu, 'allowed': '0-3', 'cpu_ticks': ticks}]}
        for t, cpu, ticks in [(0, 6, 100), (1, 6, 100), (2, 6, 100)]]
    text, errors = pm.observed_workers(b)
    assert errors == [] and 'no sampled CPU tick advance' in text, (text, errors)
    b['thread_samples'].append({'t': 3, 'threads': [{'tid': 44, 'processor': 2, 'allowed': '0-3', 'cpu_ticks': 101}]})
    text, errors = pm.observed_workers(b)
    assert errors == [] and 'CPUs [2]' in text and 'CPUs [6]' not in text, (text, errors)
    b['thread_samples'].append({'t': 4, 'threads': [{'tid': 44, 'processor': 6, 'allowed': '0-3', 'cpu_ticks': 102}]})
    assert any('observed CPUs' in e for e in pm.observed_workers(b)[1])
    print('stale pre-pin processor excluded for idle workers; active in/out-of-target CPUs verified: PASS')
    print('synthetic multi-column report, off cadence, INCOMPLETE cpufreq/missing block, ORT observations: PASS')


def check_native_and_threads():
    for prefix in ('', '/data/data/com.termux/files/usr'):
        with patch.dict(os.environ, {'PREFIX': prefix}):
            try:
                cr.require_native()
                raise AssertionError('proot allowed')
            except SystemExit as e:
                assert 'native Termux' in str(e)
    # Verify the /proc parser against this test's own main thread, without changing affinity.
    rows = pm.thread_sample(pm.time.monotonic(), {os.getpid()})['threads']
    row = next(r for r in rows if r['tid'] == os.getpid())
    assert 'error' not in row and row['processor'] >= 0 and row['allowed']
    class Detector:
        def detect(self, image, size):
            return (image, size)
    # Fake only affinity syscalls: no real pinning tested here.
    with patch.object(os, 'sched_setaffinity') as setter, patch.object(os, 'sched_getaffinity', lambda tid: {4, 5}):
        detector = pm.PinnedDetector(Detector(), {4, 5})
        assert detector.detect('frame', 320) == ('frame', 320)
        assert setter.call_args.args == (detector.tid, {4, 5})
        detector.close()
    print('native/proot refusal, /proc stat/status parser, detector-only caller lifecycle (mock affinity): PASS')


def fake_child(root_dir, resume=None):
    sys.path.insert(0, str(pm.HERE.parent / 'coresidency'))
    import test_coresidency as harness
    harness.PORT = 18180  # independent of the archived suite running concurrently
    real_run = pm.run_block

    def entry(args):
        pm.OUT_ROOT = cr.OUT_ROOT
        pm.dp.MODELS = {320: cr.MODEL, 640: cr.MODEL}
        def detector(setting):
            return cr.Detector(), {'version': 'fake', 'setting': setting, 'worker_tids': [],
                                   'caller_tid': None, 'intra_op_num_threads': {'320': 0, '640': 0},
                                   'observed_default_intra_threads_per_session': 1}
        pm.build_detector = detector
        def block(spec, ctx):
            if not resume and spec['name'] == 'R1_r2_c10':
                raise SystemExit(143)  # completed first block must survive interruption
            return real_run(spec, ctx)
        pm.run_block = block
        pm.main(['--set', 'R1', *args])
    cr.main = entry
    harness.child(root_dir, ['--resume', resume] if resume else [])


def check_integration():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        result = subprocess.run([sys.executable, __file__, '--fake-child', tmp], capture_output=True, text=True, timeout=120)
        assert result.returncode == 143, result.stdout[-3000:] + result.stderr[-3000:]
        run = next(root.glob('run_*_R1_smoke'))
        first = run / 'block_R1_r2_c5.json'
        assert first.exists() and not (run / 'block_R1_r2_c10.json').exists()
        saved = first.read_bytes()
        result = subprocess.run([sys.executable, __file__, '--fake-child', tmp, str(run)],
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        assert first.read_bytes() == saved
        assert 'already completed, skipped' in result.stdout
        assert not pm.report(run)[1]
        assert (root / 'downloads' / f'power_map_{run.name}_report.txt').exists()
        assert not list(root.glob('.drop_request'))
        for b in run.glob('block_*.json'):
            data = json.loads(b.read_text())
            assert data['camera_end']['capture_stopped'] and data['camera_end']['force_stop_rc'] == 0
            assert data['survived']['llama_server']
            assert data['cpu_zone_note'] == 'CPU-core reading, not a heat state'
    print('fake integration: interruption/resume preserves completed block, resident Gemma, report copy, camera cleanup: PASS')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--fake-child']:
        fake_child(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        raise SystemExit
    check_blocks()
    check_cadence_and_loop()
    check_numbers()
    check_report()
    check_native_and_threads()
    check_integration()
    print('POWERMAP offline checks: PASS')
