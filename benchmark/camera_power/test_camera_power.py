#!/usr/bin/env python3
"""Offline assertions. Android/root/model calls are mocked; never runs a camera."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import camera_power as cp
import launcher
from PIL import Image


def plans_mapping():
    specs = cp.blocks_for()
    assert specs[0]['extras'] == specs[-1]['extras'] == {}
    assert cp.extras_args(specs[0]) == cp.extras_args(specs[-1]) == []
    assert all(s['duration'] == 120 and s['rate'] == 1 for s in specs)
    assert [s['name'] for s in specs] == ['baseline_first', 'manual_200', 'manual_500',
            'manual_1000', 'mode_A_manual_1000', 'mode_A_manual_500', 'mode_A', 'baseline_last']
    assert sum(s['duration'] for s in specs) == 16*60
    assert all(s['duration'] == 20 for s in cp.blocks_for(True))
    assert cp.extras_args(next(s for s in specs if s['name'] == 'mode_A')) == ['--es', 'mode', 'A']
    assert cp.extras_args(next(s for s in specs if s['name'] == 'manual_500')) == ['--ei', 'frame_ms', '500']
    assert cp.extras_args(specs[4]) == ['--es', 'mode', 'A', '--ei', 'frame_ms', '1000']
    optional = cp.blocks_for(camera_id='0', blocks='preview,record,fast,off,focus_1m,dump,other_camera')
    assert cp.extras_args(optional[-1]) == ['--es', 'camera_id', '0']
    assert cp.extras_args(optional[-2]) == ['--ez', 'dump_characteristics', 'true']
    assert cp.extras_args(optional[-3]) == ['--ef', 'focus_diopters', '1.0']
    assert set().union(*(set(s['extras']) for s in specs+optional)) == set(cp.TYPES)
    for names in ('', 'other_camera', 'wrong', 'manual_500,manual_500'):
        try: cp.blocks_for(blocks=names)
        except ValueError: pass
        else: raise AssertionError('invalid block selection accepted')
    print('block order/duration/baseline no extras/all six extras: PASS')


def metrics():
    flat = Image.new('RGB', (640, 480), (100, 100, 100))
    assert cp.image_metrics(flat) == dict(luma=100., sharpness=0.)
    edge = Image.new('RGB', (320, 320))
    edge.paste('white', (160, 0, 320, 320))
    m = cp.image_metrics(edge)
    assert m['luma'] == 127.5 and m['sharpness'] > 0
    # DUTY1 math: constant 2 W -> mean 2 W, long sampling gaps are n/a.
    rows = [dict(t=i*.5, battery_w=2.) for i in range(5)]
    assert cp.dc.mean_power(rows, [(0, 2)]) == 2.
    assert cp.dc.mean_power([rows[0], rows[-1]], [(0, 2)]) is None
    assert cp.pm.skin_slope([dict(t=0, skin=30), dict(t=120, skin=32)], 120) == 1.
    groups = dict(runner=[], llama_server=[], robotcam_app=[], camera_provider=[])
    hz = os.sysconf('SC_CLK_TCK')
    a = dict(groups={**groups, 'camera_provider': [dict(pid=1, born_ticks=0, ticks=10*hz)]})
    b = dict(groups={**groups, 'camera_provider': [dict(pid=1, born_ticks=0, ticks=13*hz)]})
    assert cp.dc.cpu_seconds([a,b], 100)['camera_provider'] == 3.
    print('luma/Laplacian/power gaps/skin slope/CPU math: PASS')


def refusals_pairing():
    spec = cp.blocks_for()[0]
    meta = dict(variant='default', mode='B', rate=1, session_id='abc', frame=4,
                power=dict(schema_version=2, options=cp.DEFAULTS))
    cp.check_build(meta, spec)
    for bad in ({}, {**meta, 'power': {}}, {**meta, 'power': dict(options=cp.DEFAULTS)}, {**meta, 'mode': 'A'}):
        try:
            cp.check_build(bad, spec)
        except RuntimeError:
            pass
        else:
            raise AssertionError('bad build/options accepted')
    with tempfile.TemporaryDirectory() as tmp, patch.object(cp.reader, 'FRAME_DIR', tmp):
        p = Path(tmp)/'frame.json'
        p.write_text(json.dumps(meta))
        assert cp.diagnostic(dict(session='abc', frame=5), spec) is None
        assert cp.diagnostic(dict(session='abc', frame=4), spec) == meta
        p.write_text('{bad')
        assert cp.diagnostic(dict(session='abc', frame=4), spec) is None
    with patch.object(cp.cr, 'RootShell', side_effect=AssertionError('hardware touched')):
        try:
            cp.main(['--smoke'])
        except SystemExit as e:
            assert 'native Termux' in str(e)
        else:
            raise AssertionError('proot allowed')
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        cp.dc.save_block(out, spec, dict(evidence=1))
        original = (out/'block_baseline_first.json').read_bytes()
        assert cp.dc.completed(out, spec)
        assert not cp.dc.completed(out, cp.blocks_for()[-1])
        assert (out/'block_baseline_first.json').read_bytes() == original
    print('old APK/wrong options/proot refusal; sidecar pair/malformed; atomic resume: PASS')


def block_integration(core_loss=False):
    spec = {**cp.blocks_for(True)[0], 'duration': .35}
    groups = {n:[dict(pid=i+1, born_ticks=0, ticks=10)] for i,n in enumerate(
        ('runner', 'llama_server', 'robotcam_app', 'camera_provider'))}
    stop_result = dict(capture_stopped=True, force_stop_rc=0, pids_after_force_stop=[], pidof_root_rc=0, pidof_rc=1)
    frame = [0]
    stopped = []
    image = Image.new('RGB', (640,480), (100,100,100))
    def sample(*args):
        frame[0] += 1
        return (dict(status='ok', session='abc', frame=frame[0], age_s=.1,
                     luma=100., sharpness=0., diagnostics=None), image, 'abc', frame[0])
    def fast(*args):
        t = time.monotonic()
        return dict(t=t, t_start=t, bat_c=30., cpu_c=30., max={}, cpu={})
    def dump():
        t = time.monotonic()
        return dict(t=t, t_start=t, status=0, skin=30.)
    def cores(when):
        if core_loss and 'during' in when:
            raise cp.cr.CoresLost('mock core loss')
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        replacements = [(cp.pm, 'thermal_gate', lambda *a:dict(warm_start=False)),
                        (cp.cr, 'check_cores', cores), (cp.cr, 'am', lambda *a:None),
                        (cp.cr, 'fast_keys', lambda *a:[]), (cp.cr, 'fast_sample', fast),
                        (cp.cr, 'read_dump', dump), (cp, 'read_sample', sample),
                        (cp.dc, 'cpu_snapshot', lambda:dict(groups=groups, errors=[], t=time.monotonic())),
                        (cp.dc, 'battery_sample', lambda sh,t0:dict(t=time.monotonic()-t0, battery_w=2., battery_status='Discharging')),
                        (cp.dc, 'stop_camera', lambda:stopped.append(True) or stop_result)]
        for obj,key,value in replacements:
            stack.enter_context(patch.object(obj,key,value))
        ctx = dict(out=Path(tmp), smoke=True, thermal_log='unused', idle={}, shell=None,
                   battery_shell=None, layout={})
        try:
            b = cp.run_block(spec, ctx)
        except cp.cr.CoresLost:
            assert core_loss
        else:
            assert not core_loss
            assert b['first_frame_s'] is not None and len(b['saved_frames']) == 3
            assert all((Path(tmp)/v['path']).exists() for v in b['saved_frames'].values())
            s = cp.summary(b)
            assert s['mean_battery_w'] == 2. and s['luma']['median'] == 100.
            assert any('no paired diagnostics' in x for x in cp.problems(b,s))
        assert stopped
    print('mocked block monitors/frames/images/metrics/cleanup'+(' on core loss' if core_loss else '')+': PASS')


def poll_classification():
    statuses = ['missing', 'missing', 'ok', 'repeat', 'missing', 'bad', 'other_session', 'ok']
    rows = [dict(status=status) for status in statuses]
    c = cp.poll_counts(rows)
    assert c['startup_wait_polls'] == 2 and c['frames_failed'] == 3
    assert c['raw_status_counts'] == dict(missing=3, ok=2, repeat=1, bad=1, other_session=1)
    assert cp.poll_counts(rows[:2])['frames_failed'] == 0  # no frame still fails problems()
    spec = cp.blocks_for()[0]
    r = dict(status='ok', session='abc', frame=1, age_s=.1, image=Image.new('RGB',(10,10)))
    with patch.object(cp.reader, 'read_frame', return_value=r), patch.object(cp, 'diagnostic', return_value=None):
        row, image, pinned, last = cp.read_sample(spec, None, None, 0)
        assert row['status'] == 'ok' and row['diagnostics'] is None and last == 1
        row, image, pinned, last = cp.read_sample(spec, pinned, last, 0)
        assert row['status'] == 'repeat' and image is None
    with patch.object(cp.reader, 'read_frame', return_value=dict(status='missing')):
        assert cp.read_sample(spec, 'abc', 1, 0)[0] == dict(status='missing')
    print('startup waits / post-start rejection / repeats / unpaired sidecars: PASS')


def smoke_log_regression():
    root = Path('/termux-home/camera_power')
    if not root.is_dir():
        print('existing phone smoke-log regression: SKIP (logs unavailable)')
        return
    for name, failures in [('run_20261003T004418Z_smoke',76), ('run_20261003T062831Z_smoke',71)]:
        folder = root/name
        if not folder.is_dir(): continue
        for p in folder.glob('block_*.json'):
            b = json.loads(p.read_text())
            stats = cp.summary(b)
            if p.name == 'block_manual_500.json':
                assert stats['frames_ok'] == 6 and stats['frames_failed'] == failures
                assert stats['new_frame_gap_max_s'] > 2.8
                assert 'publication gap exceeds rate/frame-duration tolerance' in cp.problems(b,stats)
            else:
                assert stats['frames_failed'] == 0, p.name
                assert stats['startup_wait_polls'] == stats['raw_status_counts']['missing']
    print('existing dark/lit smoke regression: ordinary failures -> startup waits; manual500 remains INCOMPLETE: PASS')


def android_contracts_and_cadence():
    # Static Android contracts plus a deterministic timing model; no HAL emulation.
    root = Path('/termux-home/robotcam-exp/android/robotcam/app/src/main/java/com/pixelrobot/robotcam')
    power = (root/'CameraPower.kt').read_text()
    service = (root/'CameraService.kt').read_text()
    assert 'p.frameMs in 100..1000' in power
    assert 'manualExposure = minOf(exposure, frameMs * 1_000_000L)' in power
    assert 'b.set(CaptureRequest.SENSOR_FRAME_DURATION, frameMs * 1_000_000L)' in power
    assert 'c.availableCaptureRequestKeys.contains(CaptureRequest.CONTROL_POST_RAW_SENSITIVITY_BOOST)' in power
    assert 'result.get(CaptureResult.CONTROL_POST_RAW_SENSITIVITY_BOOST) else null' in power
    assert 'manualBoost?.let { b.set(CaptureRequest.CONTROL_POST_RAW_SENSITIVITY_BOOST, it) }' in power
    assert 'field("post_raw_boost", CaptureRequest.CONTROL_POST_RAW_SENSITIVITY_BOOST, CaptureResult.CONTROL_POST_RAW_SENSITIVITY_BOOST)' in power
    assert 'now - phaseAt >= 30_000' in power and 'now - phaseAt >= 1_000' in power
    assert 'if (mode == MODE_B && power.frameMs == 0) {' in service
    start = service[service.index('nextStreamDue = 0L', service.index('private fun startSession')):service.index('status("streaming")')]
    manual = start.split('} else if (mode == MODE_A)')[0]
    assert 'if (power.frameMs != 0)' in manual and 's.setRepeatingRequest' in manual
    assert 'power.stillTemplate' in manual and 'frameSurface' in manual
    assert 'stillTick' not in manual and 's.capture' not in manual
    assert 'mode == MODE_A || power.frameMs != 0' in service
    assert 'if (mode == MODE_B && power.frameMs == 0) stillRequest' in service
    pacing = service[service.index('if (power.frameMs != 0) {', service.index('private fun onFrame')):service.index('} else if (mode == MODE_A)',service.index('private fun onFrame'))]
    assert 'if (now < nextStreamDue - tolerance) return' in pacing
    assert 'maxOf(nextStreamDue + periodMs, now + periodMs - tolerance)' in pacing
    # 100..1000 ms streams, +/- 4 ms jitter, a 30 s AE re-convergence burst.
    # Non-divisors of 1000 select nearest frames; instantaneous gaps are quantized.
    for frame_ms in range(100, 1001):
        due, selected = 0, []
        timestamps = list(range(0, 30000, frame_ms)) + list(range(30000, 32000, 67)) + list(range(32000, 62000, frame_ms))
        for i, t in enumerate(timestamps):
            now = t + (4 if i % 2 else -4) + 10000
            tolerance = min(frame_ms, 1000)//2
            if now < due - tolerance: continue
            due = now+1000 if due == 0 else max(due+1000, now+1000-tolerance)
            selected.append(now)
        assert 60 <= len(selected) <= 64, (frame_ms,len(selected))
        assert max(b-a for a,b in zip(selected,selected[1:])) <= 2008, frame_ms
    # Exposure clamp does not increase a converged value, even above the period.
    for frame_ms in (100,200,500,1000):
        for exposure in (1, 66666316, 150000000, 1500000000):
            manual = min(exposure, frame_ms*1000000)
            assert manual <= exposure and manual <= frame_ms*1000000
    print('Android source contracts: manual streaming A/B, boost copy, exposure clamp, 30 s recovery: PASS')
    print('manual pacing model 100..1000 ms with jitter/re-convergence: PASS (HAL unverified)')


def launcher_checks():
    text = launcher.launcher_text()
    assert 'camera_power\\.py|duty_cycle\\.py|power_map' in text
    assert 'pause 300' in text and 'screen_off_timeout' in text and 'LOGGER=$!' in text
    assert 'claude|agy|node|codex' in text
    # Run inherited offline suite only against its fake su/settings/processes/temp sysfs.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        (path/'oneshot.sh').write_text(text)
        test = (HERE.parent/'duty_cycle/test_oneshot.sh').read_text()
        for before,after in [('robot/benchmark/duty_cycle','robot/benchmark/camera_power'),
                             ('run_duty_cycle.sh','run_camera_power.sh'), ('H/duty_cycle','H/camera_power')]:
            test = test.replace(before,after)
        (path/'test.sh').write_text(test)
        result = subprocess.run(['bash',str(path/'test.sh')], capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout+result.stderr
        print(result.stdout, end='')
    print('inherited launcher offline fake-only suite: PASS')


def stalled_accounting():
    """A safety stop must precede accounting drain, even when that accounting fails."""
    stop_called = threading.Event()
    timeline = []
    groups = {n: [] for n in ('runner', 'llama_server', 'robotcam_app', 'camera_provider')}
    cpu_calls = [0]
    def snapshot():
        cpu_calls[0] += 1
        if cpu_calls[0] >= 3:  # initial boundary, periodic, final boundary
            assert stop_called.wait(2), 'STOP waited behind stalled accounting'
            timeline.append('accounting failed')
            raise RuntimeError('mock failed boundary CPU query')
        return dict(t=time.monotonic(), groups=groups, errors=[])
    def monitor(period, read, rows, stop):
        rows.append(read())
        stop.wait(2)
    def stop_camera():
        timeline.append('STOP/end-check/force-stop')
        stop_called.set()
        return dict(capture_stopped=True, force_stop_rc=0, pids_after_force_stop=[], pidof_root_rc=0, pidof_rc=1)
    def fast(*a):
        t = time.monotonic()
        return dict(t=t, t_start=t, bat_c=45., cpu_c=30., max={}, cpu={})
    def dump():
        t = time.monotonic()
        return dict(t=t, t_start=t, status=0, skin=30.)
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        replacements = [(cp.pm,'thermal_gate',lambda *a:{}), (cp.cr,'check_cores',lambda *a:None),
                        (cp.cr,'am',lambda *a:None), (cp.cr,'fast_keys',lambda *a:[]),
                        (cp.cr,'fast_sample',fast), (cp.cr,'read_dump',dump),
                        (cp.cr,'monitor_loop',monitor), (cp.cr,'block_limit',lambda *a:('battery','mock limit')),
                        (cp.dc,'cpu_snapshot',snapshot), (cp.dc,'stop_camera',stop_camera),
                        (cp.dc,'battery_sample',lambda sh,t0:dict(t=time.monotonic()-t0,battery_w=2.,battery_status='Discharging'))]
        for obj,key,value in replacements: stack.enter_context(patch.object(obj,key,value))
        b = cp.run_block({**cp.blocks_for(True)[0], 'duration':.35},
                         dict(out=Path(tmp),smoke=True,thermal_log='unused',idle={},shell=None,
                              battery_shell=None,layout={}))
        assert timeline == ['STOP/end-check/force-stop','accounting failed'], timeline
        assert any(x['errors'] for x in b['cpu_snapshots'])
        assert b['heat_stop'] == ('battery','mock limit')
    print('limit + stalled/failed accounting: STOP/end-check/force-stop precedes drain; CPU error retained: PASS')


def fail_closed_orchestration():
    starts, cleanup, reports = [], [], []
    class Shell:
        def close(self): pass
    def run(spec, ctx):
        starts.append(spec['name'])
        return dict(block=spec['name'], spec=spec, first_frame_s=1.,
                    heat_stop=('fail_closed','mock battery monitor failed after startup'))
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        replacements = [(cp,'OUT_ROOT',root),(cp.cr,'require_native',lambda:None),
                        (cp.cr,'RootShell',Shell),(cp.cr,'discover',lambda sh:{}),
                        (cp.cr,'read_thermal',lambda path:dict(z9=30000)),
                        (cp.cr,'read_dump',lambda:dict(skin=30.)), (cp.cr,'wait_cores',lambda *a:None),
                        (cp.signal,'signal',lambda *a:None), (cp,'run_block',run),
                        (cp,'write_report',lambda *a:reports.append(True) or []),
                        (cp.dc,'stop_camera',lambda:cleanup.append(True) or
                         dict(capture_stopped=True,force_stop_rc=0,pids_after_force_stop=[],pidof_root_rc=0,pidof_rc=1))]
        for obj,key,value in replacements: stack.enter_context(patch.object(obj,key,value))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        try: cp.main(['--smoke'])
        except RuntimeError as e: assert 'fail-closed sensor stop' in str(e)
        else: raise AssertionError('continued after sensor monitoring failed')
        assert starts == ['baseline_first'] and len(cleanup) == 2 and reports
        out = next(root.glob('run_*'))
        path = out/'block_baseline_first.json'
        original = path.read_bytes()
        assert json.loads(original)['first_frame_s'] == 1., 'failure was after valid startup'
        try: cp.main(['--smoke','--resume',str(out)])
        except RuntimeError as e: assert 'saved fail-closed' in str(e)
        else: raise AssertionError('resume reset the failed sensor grace period')
        assert starts == ['baseline_first'] and len(cleanup) == 4
        assert path.read_bytes() == original
    print('fail-closed after startup: evidence retained, no next variant, resume refuses, final cleanup: PASS')


if __name__ == '__main__':
    plans_mapping(); poll_classification(); smoke_log_regression(); android_contracts_and_cadence(); metrics(); refusals_pairing(); block_integration(); block_integration(True); stalled_accounting(); fail_closed_orchestration(); launcher_checks()
    assert 'motors' not in sys.modules and 'main' not in sys.modules
    print('CAMPOWER2 offline checks: PASS (no hardware)')
