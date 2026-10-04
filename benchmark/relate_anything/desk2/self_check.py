"""Small native check for preprocessing, score ranking, masks and retained failed thresholds."""
import sys
sys.dont_write_bytecode = True
from desk_check import Head, preprocess, np, Image

image = Image.new('RGB', (100, 200), (255, 0, 128))
feed = preprocess(image, [[0, 0, 100, 200], [20, 40, 60, 120]])
assert feed['image'].shape == (1, 3, 448, 448) and feed['image'].dtype == np.float32
np.testing.assert_allclose(feed['image'][0, :, 0, 0], [1, 0, 128 / 255])
np.testing.assert_allclose(feed['boxes'][0, :2], [[.5, .5, 1, 1], [.4, .4, .4, .4]])
assert not feed['boxes'][0, 2:].any() and feed['box_counts'].dtype == np.int64
for invalid in ([[0, 0, 1, 1]], [[0, 0, float('nan'), 1], [0, 0, 1, 1]]):
    try:
        preprocess(image, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid boxes accepted')

pred = np.full((1, 128, 35), -10, np.float32)
pred[0, 0, 1] = 1
pred[0, 1, 2] = 2
subject = np.zeros((1, 128), np.int64)
obj = np.ones((1, 128), np.int64)
subject[0, 1], obj[0, 1] = 1, 0
subject[0, 2], obj[0, 3], obj[0, 4] = -1, 10, 0
valid = np.zeros((1, 128), bool)
valid[0, :5] = True
class Session:
    def run(self, names, inputs):
        return [pred, np.zeros((1, 128), np.float32), subject, obj, valid]
head = Head.__new__(Head)
head.W, head.alpha = np.zeros((35, 512), np.float32), np.zeros(35, np.float32)
head.a, head.b = 1, 0
head.thresholds = np.ones(35, np.float32) * .99
head.predicates = [str(i) for i in range(35)]
head.session = Session()
detections = [{'class_name': 'a', 'box_xyxy': [0, 0, 100, 200], 'conf': 1},
              {'class_name': 'b', 'box_xyxy': [20, 40, 60, 120], 'conf': .001}]
rows, _ = head.infer(image, detections)
assert len(rows) == 2 and [r['predicate'] for r in rows] == ['2', '1']
assert not any(r['pass'] for r in rows) and rows[0]['subject_idx'] == 1
assert all(r['subject_idx'] != r['object_idx'] for r in rows)
print('PASS: RGB /255 CHW, square resize, cxcywh padding/count, invalid inputs, masks, score order, one predicate/pair, below-threshold retention')


# Our mapping deliberately promotes to float64 before clipping. Upstream uses
# float32 bank scalars/clip endpoints and stores mapped thresholds as float32;
# this checks our stated formula, not bitwise equivalence with that runtime.
from desk_check import mapped_thresholds, require_parity
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
import desk_check as dc
import speed_block as sb
raw = np.array([0, .1234567, .5, .9999999, 1, np.nan, np.inf], np.float32)
a, b = 1.7345, -.2468
expected = []
for v in raw:
    t = float(np.clip(float(v), 1e-6, 1 - 1e-6)) if np.isfinite(v) else .5
    expected.append(float(1 / (1 + np.exp(-(a * np.log(t / (1 - t)) + b)))) if np.isfinite(v) else .4)
actual = mapped_thresholds(raw, a, b)
assert actual.dtype == np.float64
np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-16)
with tempfile.TemporaryDirectory() as tmp, patch.object(dc, 'HERE', Path(tmp)), patch.object(dc, 'MODELS', Path(tmp)):
    graph = Path(tmp) / 'relsgg-vits16/relateanything.onnx'
    graph.parent.mkdir()
    graph.write_bytes(b'tested graph')
    p = Path(tmp) / 'parity_valid_relsgg-vits16.json'
    for value in (None, {'passed': False}, {'passed': 'true'}):
        if value is not None: p.write_text(json.dumps(value))
        try:
            dc.Head('relsgg-vits16')
        except (FileNotFoundError, RuntimeError): pass
        else: raise AssertionError('M2 loaded without true new parity evidence')
        with patch.object(sys, 'argv', ['speed_block.py', '--model', 'relsgg-vits16', '--dry-run']):
            try: sb.main()
            except (FileNotFoundError, RuntimeError): pass
            else: raise AssertionError('speed M2 guard did not refuse before loading/locking')
    import hashlib
    graph_hash = hashlib.sha256(graph.read_bytes()).hexdigest()
    p.write_text(json.dumps({'passed': True, 'graph_sha256': graph_hash}))
    require_parity('relsgg-vits16')
    graph.write_bytes(b'replaced graph')
    for entry in (lambda: dc.Head('relsgg-vits16'), sb.main):
        with patch.object(sys, 'argv', ['speed_block.py', '--model', 'relsgg-vits16', '--dry-run']):
            try: entry()
            except RuntimeError as error: assert 'SHA-256' in str(error)
            else: raise AssertionError('stale parity unlocked replaced M2 graph')
    require_parity('relsgg-vits16plus')
# Process gate excludes only its own PID and refuses another speed runner or robot.
import subprocess
import os
for name in ('llama-server', 'chat.py', 'main.py', 'run_mission.py', 'power_map.py',
             'coresidency.py', 'thermal_char.py', 'duty_cycle.py', 'camera_power.py', 'speed_block.py'):
    with patch.object(sb.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
              f'{os.getpid()} speed_block.py\n{os.getpid()+1} {name}\n', '')):
        try: sb.processes_clear()
        except RuntimeError: pass
        else: raise AssertionError('resident process not rejected: ' + name)
with patch.object(sb.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
                      f'{os.getpid()} speed_block.py\n', '')):
    sb.processes_clear()
# Screen restoration runs even when set/readback fails after saving the original.
commands=[]
def setting(command):
    commands.append(command)
    if command == 'settings get system screen_off_timeout':
        return 0, '60000'  # setting the new timeout fails readback
    return 0, ''
with patch.object(sb.cr, 'root', side_effect=lambda c,t: setting(c)), patch.object(sb.subprocess, 'run'):
    screen = sb.Screen()
    try: screen.start()
    except RuntimeError: pass
    else: raise AssertionError('screen readback failure accepted')
    screen.restore()
assert 'settings put system screen_off_timeout 60000' in commands
print('PASS: explicit float64 formula (not runtime bitwise parity), missing/false/stale-graph M2 guards in both entry points, process refusal, screen restore after set failure')

# Extras must not repair a first-five shortfall (even if a later detector call would qualify).
from types import SimpleNamespace
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / 'test_photos').mkdir(); (root / 'bench_photos').mkdir()
    for n in range(4): (root / 'test_photos' / f'{n}.jpg').touch()
    for n in ('blocked_1.jpg', 'roomsig_couch_1.jpg'): (root / 'bench_photos' / n).touch()
    seen=[]
    def detect(p, size):
        seen.append(p)
        return detections if 'test_photos' in str(p) or len(seen)>6 else detections[:1]
    fake = SimpleNamespace(get_detector=lambda: SimpleNamespace(detect=detect))
    with patch.object(dc, 'ROBOT', root), patch.object(dc, 'open_image', lambda p:p), patch.dict(sys.modules, {'detect_person':fake}):
        try: dc.select_photos()
        except RuntimeError as error: assert 'before extras' in str(error)
        else: raise AssertionError('extras repaired first-five shortfall')
    assert len(seen)==6
print('PASS: first-five shortfall checked before adding extras')

sb.require_policies({'policies': dict.fromkeys(sb.pm.POLICIES, 1)})
try: sb.require_policies({'policies': {'policy0': 1, 'policy4': 1}})
except RuntimeError: pass
else: raise AssertionError('missing policy6 accepted')
windows = [{'started_s': 1, 'ended_s': 2}, {'started_s': 5, 'ended_s': 6}]
power = [{'t': t, 't_start': t-.01, 'battery_w': w} for t,w in
         [(-.1,99), (0,1), (1,3), (2,1), (5.5,5), (10,2), (10.1,99)]]
summary = sb.power_summary(power, windows, 10)
assert summary['samples'] == 5 and summary['inside_samples'] == 2 and summary['outside_samples'] == 3
assert summary['mean_battery_w'] == 2.4 and summary['mean_inside_calls_w'] == 4
assert summary['mean_outside_calls_w'] == 4/3
assert summary['read_overlaps_call_boundary'] == 2
assert sb.power_summary([], windows, 10)['mean_battery_w'] is None
class Shell:
    def run(self, command):
        assert '/current_now' in command and '/voltage_now' in command
        return ['-1000000', '4000000']
power=[]
with patch.object(sb.cr, 'fast_sample', return_value={'t': 1}), \
     patch.object(sb.os, 'sched_getaffinity', return_value={6,7}), \
     patch.object(sb.time, 'monotonic', side_effect=[1.1, 1.2]):
    assert sb.read_fast(Shell(), [], power, 123) == {'t': 1}
assert power[0]['battery_w'] == 4 and power[0]['t_start'] == 1.1 and power[0]['t'] == 1.2
assert not set(power[0]['sampler_cpus']) & sb.CPUS
with patch.object(sb.cr, 'fast_sample', return_value={}), \
     patch.object(sb.os, 'sched_getaffinity', side_effect=[{6,7},{4,5}]):
    try: sb.read_fast(Shell(), [], [], 123)
    except RuntimeError: pass
    else: raise AssertionError('root sampler permitted on MID cores')
print('PASS: required policies; one-second current_now fields, sign/units/monotonic window means, boundary counts, null dry means and non-MID sampler/root refusal')
