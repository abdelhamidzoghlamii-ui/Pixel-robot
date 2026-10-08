"""CAMPAIGN_P1_FIX3B focused offline mocks: checked thermal reader, camera cleanup pidof checks. Never run hardware."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import diagnostics as d
import phase1 as p
import runtime as rt
from self_check import phase_stubs
from self_check_fix3 import rest_harness, root_text

GOOD = {'t': 1., 'rc': 0, 'skin': 30.5, 'status': 0}
BAD = {'rc=1': dict(GOOD, rc=1), 'rc None': dict(GOOD, rc=None), 'error': dict(GOOD, rc=None, error='TimeoutExpired: su'),
       'skin None': dict(GOOD, skin=None), 'status None': dict(GOOD, status=None), 'skin nan': dict(GOOD, skin=math.nan),
       'status str': dict(GOOD, status='0'), 'rc missing': {k: v for k, v in GOOD.items() if k != 'rc'}}


def check_reader():
    cr = NS(STATUS_STOP=4)
    def reads(row):
        return patch.object(rt, 'dump_once', side_effect=lambda cr, timeout=30: dict(row)), patch.object(rt, 'DUMP_PAUSE_S', 0)
    with contextlib.ExitStack() as stack:
        for m in reads(GOOD):stack.enter_context(m)
        assert rt.read_dump(cr) == dict(GOOD, attempts=1) and rt.dump_check(cr) == dict(GOOD, attempts=1)
        assert rt.thermal_row_failures([GOOD]) == []
    for name, row in BAD.items():
        with contextlib.ExitStack() as stack:
            for m in reads(row):stack.enter_context(m)
            for reader in (rt.read_dump, rt.dump_check):
                try:reader(cr)
                except RuntimeError as e:assert 'thermal read failed' in str(e), (name, e)
                else:raise AssertionError(f'{name} accepted by {reader.__name__}')
        assert len(rt.thermal_row_failures([GOOD, row])) == 1, name
    with contextlib.ExitStack() as stack:
        for m in reads(dict(GOOD, status=4)):stack.enter_context(m)
        try:rt.dump_check(cr)
        except RuntimeError as e:assert 'thermal stop' in str(e)
        else:raise AssertionError('thermal stop accepted')
    # Diagnostics read the same checked reader.
    text = '/battery/capacity\t85\n/battery/temp\t287\n/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq\t1401000\n'
    cr = NS(BATTERY='/battery', root=lambda *a, **k: (0, text))
    with contextlib.ExitStack() as stack:
        for m in reads(GOOD):stack.enter_context(m)
        assert d.snapshot(cr, True)['skin'] == 30.5
    with contextlib.ExitStack() as stack:
        for m in reads(BAD['rc=1']):stack.enter_context(m)
        try:d.snapshot(cr, False)
        except RuntimeError as e:assert 'thermal read failed' in str(e)
        else:raise AssertionError('diagnostics accepted a failed thermal read')
    print('PASS checked thermal reader: rc=1/None/missing, error, skin None/nan, status None/str rejected by read_dump, dump_check and diagnostics; thermal stop kept')


def check_rest_late_thermal():
    for camera_on in (False, True):
        text = root_text(app=camera_on)
        assert rest_harness(p, camera_on, text)[0] is None
        for name in ('rc=1', 'error', 'skin None', 'status None'):
            # Reader completes during the join and returns a failure row: the checked reader raises.
            error, record = rest_harness(p, camera_on, text, late_dump=BAD[name])
            assert error.startswith('pause monitor failed') and 'thermal read failed' in str(record['monitor_errors']), (name, error, record.get('monitor_errors'))
            assert p.coverage(dict(idle_phases=[record], pauses=[], blocks=[], plan=[]))['idle_phases_done'] == 0
            # A failure row already collected (bypassing the reader) is caught by the post-join validation.
            error, record = rest_harness(p, camera_on, text, late_row=BAD[name])
            assert error.startswith('pause monitor failed') and 'thermal rows failed' in str(record['monitor_errors']), (name, error, record.get('monitor_errors'))
    print('PASS rest() camera OFF and ON: late failed thermal reads (rc=1, error, skin None, status None) during join stop the phase; coverage excludes it')


def live_record(dry, rows, reads=None):
    """live_block with the fix3 harness shape; run_cycle mock leaves `rows` as the collected thermal rows.
    reads: rt.dump_once replacement, with the real rt.dump_check (FIX3D)."""
    v, sb, pm, cr, _ = rt.imports()
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        monitor = (NS(close=MagicMock()), {0}, 10, {'policies': {'policy0': 1}, 'cpu_zones': {}})
        for m in (patch.object(rt, 'prepare', return_value=monitor), patch.object(pm, 'build_detector', return_value=(NS(close=MagicMock()), {'worker_tids': [], 'caller_tid': 1})),
                  patch.object(v, 'build', return_value=(None, NS(close=MagicMock()), [])), patch.object(rt, 'clear_processes'), patch.object(cr, 'check_cores'),
                  patch.object(rt, 'battery_sample'), patch.object(rt, 'fast_check', return_value={}), patch.object(pm, 'camera_start', return_value={'session': 's'}),
                  patch.object(rt, 'dump_check', **({'wraps': rt.dump_check} if reads else {'return_value': {'skin': 30, 'status': 0}})),
                  patch.object(rt, 'dump_once', side_effect=reads), patch.object(p, 'bounded_selector'), patch.object(cr, 'load_cases', return_value=[{}]),
                  patch.object(p.power, 'sample', return_value={'t': 0, 't_start': 0, 'battery_w': 1}),
                  patch.object(p.threading, 'Thread', return_value=NS(start=lambda: None, join=lambda timeout=None: None, is_alive=lambda: False)),
                  patch.object(rt, 'stop_camera'), patch.object(cr, 'lmk_lines', return_value={'ok': True, 'n_kills': 0}), patch.object(cr, 'block_limit', return_value=None),
                  patch.object(d, 'snapshot', return_value={}), patch.object(rt, 'fallback_input', return_value=('img', [{}, {}])),
                  patch.object(pm, 'capped_by_policy', return_value={}), *phase_stubs()):
            stack.enter_context(m)
        def cycle(name, length, ops, record):
            origin = record['cycle_origin']
            record.update(duration_s=length, m2=[], selector=[], yolo=[], reads=[])
            record['power'] = [dict(t=i, t_start=i-.01, battery_w=1) for i in range(-1, length+2)]
            record['memory'] = [dict(t=origin, mem_available_mib=1, root_rc=0, pss_kb={'llama_server': 1}, battery_status='Discharging')]
            record['dumps'].extend(dict(r, t=origin) for r in rows)
        stack.enter_context(patch.object(p, 'run_cycle', side_effect=cycle))
        return p.live_block('L4', Path(tmp)/'b.json', NS(proc=NS(pid=1), alive=lambda: True), None, dry, 25 if dry else 180)


def check_live_block_thermal():
    assert live_record(False, [GOOD])['validity'] == 'VALID'
    for name in ('rc=1', 'error', 'skin None', 'status None'):
        for dry in (False, True):
            record = live_record(dry, [GOOD, BAD[name]])
            assert record['failure_kind'] == 'shared' and 'thermal rows failed' in str(record['monitor_errors']), (name, record.get('monitor_errors'))
            assert record['validity'] == (p.REHEARSAL if dry else 'NOT VALID — INCOMPLETE')
            if dry:assert record['rehearsal_block_validity'] == 'NOT VALID — INCOMPLETE'
    print('PASS live_block session and rehearsal: a failed thermal row collected in the block makes it NOT VALID / shared (stops later blocks)')


def check_rehearsal_rejects_thermal():
    """Real main() rehearsal flow; a thermal failure in an idle phase or in the LAST block must fail rehearsal_pass."""
    def run(fail_at):
        server = NS(stop=MagicMock()); screen = NS(restore=MagicMock())
        def preflight(names, out, result, resources):resources.update(server=server, screen=screen)
        def rest(label, duration, srv, record, camera_on=False, dry=False, **d2):
            record.update(label=label, duration_s=duration, monitor_errors=[])
            if fail_at == 'idle' and 'CAMERA ON' in label:
                record.update(monitor_errors=['thermal read failed: rc 1'], error='RuntimeError: pause monitor failed')
                raise RuntimeError('pause monitor failed')
        blocks = []
        def block(name, path, srv, idle, dry=False, duration=180, **mark):
            blocks.append(name)
            record = dict(block=name, validity=p.REHEARSAL, skin_start={'skin': 30}, monitor_errors=[],
                          summary={'m2': {'live_calls': 2, 'fallback_calls': 1}, 'max_live_boxes': 5}, selector=[{'context': '(fallback scene)\n'}])
            if fail_at == 'last' and len(blocks) == 7:  # warm-up + six blocks: the final L0
                record.update(monitor_errors=['thermal rows failed: t=1: rc 1'], failure_kind='shared',
                              rehearsal_block_validity='NOT VALID — INCOMPLETE')
            p.write(path, record);return record
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            for m in (patch.object(p, 'preflight', side_effect=preflight), patch.object(p, 'rest', side_effect=rest),
                      patch.object(p, 'live_block', side_effect=block), patch.object(d, 'battery', return_value=85),
                      patch.object(rt, 'HOME', Path(tmp)), contextlib.redirect_stdout(io.StringIO()), *phase_stubs()):
                stack.enter_context(m)
            output = Path(tmp)/'out.json'
            try:p.main(['--dry-run', '--output', str(output)])
            except RuntimeError:assert fail_at
            return json.loads(output.read_text())['rehearsal_coverage']
    assert run(None)['rehearsal_pass'] is True
    for fail_at in ('idle', 'last'):
        cov = run(fail_at)
        assert cov['rehearsal_pass'] is False and cov['errors'], (fail_at, cov)
    print('PASS rehearsal with a thermal failure in the camera-ON idle or in the final block reports rehearsal_pass=false')


def check_camera_end():
    _, _, _, cr, _ = rt.imports()
    def run(pidof, force=(0, ''), pidof_raises=False):
        def rootf(command, tag, timeout=60):
            if tag == 'forcestop':return force
            if pidof_raises:raise OSError('su gone')
            assert command == 'pidof com.pixelrobot.robotcam; echo "pidof_rc=$?"', command
            return pidof
        with patch.object(cr, 'read_frame', return_value={'status': 'missing'}), patch.object(cr.time, 'sleep'):
            r = cr.camera_end_check(rootf, quiet_s=0, limit_s=1)
        return r, cr.camera_end_failed(r)
    r, why = run((0, 'pidof_rc=1\n'))
    assert why is None and r['pids_after_force_stop'] == [] and r['pidof_rc'] == 1 and r['pidof_root_rc'] == 0, (r, why)
    r, why = run((0, '4321 4400\npidof_rc=0\n'))
    assert r['pids_after_force_stop'] == ['4321', '4400'] and 'still running' in why, why
    for reply in ((0, 'pidof_rc=2\n'), (0, ''), (1, 'pidof_rc=1\n'), (0, 'garbage\n')):
        r, why = run(reply)
        assert why and ('unconfirmed' in why or 'still running' in why), (reply, r, why)
    r, why = run(None, pidof_raises=True)
    assert r['force_stop_rc'] is None and 'force-stop failed' in why
    r, why = run((0, 'pidof_rc=1\n'), force=(1, 'Error'))
    assert why == "am force-stop failed (rc 1: 'Error')", why
    # Fail closed: a result without a pidof status is not accepted.
    assert 'unconfirmed' in cr.camera_end_failed(dict(capture_stopped=True, force_stop_rc=0, pids_after_force_stop=[]))
    # Campaign cleanup uses the same check; a remaining pid is re-checked once after 0.5 s (FIX4 H2).
    for again, fails in (('4321\npidof_rc=0\n', True), ('pidof_rc=1\n', False)):
        sleeps = []
        with patch.object(cr, 'camera_stop'), patch.object(cr, 'camera_end_check', return_value=dict(capture_stopped=True, force_stop_rc=0,
                          pidof_root_rc=0, pidof_rc=0, pids_after_force_stop=['4321'])), \
             patch.object(cr, 'root', return_value=(0, again)), patch.object(rt.time, 'sleep', side_effect=sleeps.append):
            try:rt.stop_camera(cr)
            except RuntimeError as e:assert fails and 'still running' in str(e), e
            else:assert not fails, 'campaign accepted a RobotCam pid after force-stop and its re-check'
        assert sleeps == [rt.PID_RECHECK_S], sleeps
    print('PASS camera_end_check keeps pidof status; camera_end_failed fails on remaining pids, unconfirmed absence and missing status; campaign stop_camera fails closed')


if __name__ == '__main__':
    check_reader();check_rest_late_thermal();check_live_block_thermal();check_rehearsal_rejects_thermal();check_camera_end()
