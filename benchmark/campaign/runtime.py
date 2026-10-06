"""Lazy imports of unchanged benchmark runners; native operations owner only."""
import gc
import hashlib
import contextlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parents[1]
HOME = Path('/data/data/com.termux/files/home')

FALLBACK = ROBOT/'benchmark/relate_anything/desk2'
FALLBACK_HASHES = {
    'speed_photo.jpg': '4da2d789dd9a60405658184de81e9f46b3c55e09024a6f2ce26a934d899d0fd8',
    'speed_input.json': '35694e263f3894597e711bd2c2045bc55b3790f0bcd32d75baae11e60c20e32f',
}


def fallback_hashes():
    got = {n:hashlib.sha256((FALLBACK/n).read_bytes()).hexdigest() for n in FALLBACK_HASHES}
    if got != FALLBACK_HASHES:
        raise RuntimeError('fallback photo/committed YOLO boxes SHA-256 mismatch')
    return got


def fallback_input():
    fallback_hashes()
    _, sb, _, _, _ = imports()
    datum, _, image = sb.load_speed_input()
    return image, datum['detections']


def imports():
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROBOT), str(ROBOT/'benchmark/power_map'),
                   str(ROBOT/'benchmark/relate_anything/speed1')]
    import variants as v
    import session as speed_session
    return v, v.sb, v.sb.pm, v.sb.cr, speed_session


def clear_processes(server_pid=None):
    pattern = (r'claude|agy|node|codex|llama-server|chat\.py|main\.py|run_mission\.py|'
               r'benchmark/.*\.py|phase1\.py|lag_probe\.py|power_map\.py|coresidency\.py|thermal_char\.py|'
               r'duty_cycle\.py|camera_power\.py|speed_block\.py|session\.py|run_.*\.sh|oneshot\.sh')
    result = subprocess.run(['pgrep', '-fa', pattern], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError('process enumeration failed: '+result.stderr)
    allowed = {os.getpid(), server_pid}
    others = [l for l in result.stdout.splitlines() if int(l.split()[0]) not in allowed]
    if others:
        raise RuntimeError('agents/robot/other runners resident: '+'\n'.join(others))


def require_mask(mask, measured, allowed, process, how):
    if not mask or set(mask) & set(measured) or not set(mask) <= set(allowed):
        raise RuntimeError('root shell/monitor mask empty or includes inference cores: '
                           f'actual={sorted(mask)} measured={sorted(measured)} allowed={sorted(allowed)} '
                           f'process={process} how={how}')


def verify_monitor(sb, monitor, measured):
    shell, mask, pid, _ = monitor
    require_mask(mask, measured, mask, 'configured monitor', 'prepare result')
    for root_pid, process in ((pid, 'persistent shell'), (shell.p.pid, 'su')):
        observed = sb.root_mask(shell, root_pid)
        require_mask(observed, measured, mask, f'{process} pid={root_pid}', 'root /proc/PID/status Cpus_allowed_list')
        if observed != set(mask):
            raise RuntimeError(f'root/su affinity changed: actual={sorted(observed)} measured={sorted(measured)} '
                               f'allowed={sorted(mask)} process={process} pid={root_pid} how=root /proc/PID/status')
    for tid in shell.monitor_tids:
        observed = os.sched_getaffinity(tid)
        require_mask(observed, measured, mask, f'pump tid={tid}', 'os.sched_getaffinity(tid)')
        if observed != set(mask):
            raise RuntimeError(f'root pump affinity changed: actual={sorted(observed)} measured={sorted(measured)} '
                               f'allowed={sorted(mask)} process=pump tid={tid} how=os.sched_getaffinity(tid)')


def setup_affinity(cr, measured):
    """Refresh a stale caller mask; the kernel still enforces Android's cpuset.

    Never retry inside a measured interval. A continuing restriction refuses setup.
    """
    before = set(os.sched_getaffinity(0))
    requested = set(range(os.cpu_count()))
    os.sched_setaffinity(0, requested)
    actual = set(os.sched_getaffinity(0))
    try:
        cr.check_cores('campaign monitor setup after affinity refresh')
    except Exception as e:
        raise RuntimeError(f'campaign setup affinity restricted: before={sorted(before)} '
                           f'requested={sorted(requested)} actual={sorted(actual)} '
                           f'measured={sorted(measured)} allowed={sorted(actual)} '
                           f'process=calling thread how=os.sched_setaffinity/getaffinity: {e}') from e
    return dict(before=sorted(before), requested=sorted(requested), actual=sorted(actual))


def prepare(sb, measured, restrict=None):
    _, _, _, cr, _ = imports()
    affinity = setup_affinity(cr, measured)
    monitor = sb.prepare_monitor(measured)
    try:
        monitor[0].campaign_setup_affinity = affinity
        if restrict is not None:
            shell, allowed, pid, layout = monitor
            mask = set(allowed) & set(restrict)
            require_mask(mask, measured, allowed, "restricted monitor", "allowed intersect restrict")
            for tid in shell.monitor_tids:
                os.sched_setaffinity(tid, mask)
            bits = format(sum(1 << c for c in mask), 'x')
            shell.run(f'taskset -p {bits} {shell.p.pid}')
            shell.run(f'taskset -p {bits} {pid}')
            monitor = shell, mask, pid, layout
        verify_monitor(sb, monitor, measured)
        return monitor
    except BaseException:
        monitor[0].close()
        raise


def require_hashes(v, cr, pm, gemma=True):
    expected = json.loads((HERE/'expected_hashes.json').read_text())
    paths = {str(p): p for p in (v.directory('fp32')/n for n in
             ('relateanything.onnx', 'relateanything.json', 'predicate_bank.npz'))}
    if gemma:
        paths.update({str(p): Path(p) for p in (cr.MODEL, *pm.dp.MODELS.values(), cr.server_manager.LLAMA_SERVER)})
    got = {name: cr.sha256(p) for name, p in paths.items()}
    if any(expected.get(name) != digest for name, digest in got.items()):
        raise RuntimeError('model/graph/runtime hash changed or missing')
    v.dc.require_parity(v.NAME)
    return got


def thermal_failure(d):
    """Why a cr.read_dump() row is not a usable reading (it reports failures as data), else None."""
    skin, status = d.get('skin'), d.get('status')
    if d.get('rc') != 0 or d.get('error'):
        return f"rc {d.get('rc')!r}, error {d.get('error')!r}"
    if type(status) is not int or type(skin) not in (int, float) or not math.isfinite(skin):
        return f'skin {skin!r}, status {status!r}'
    return None


def read_dump(cr):
    """Checked thermal reader: a failed or incomplete thermalservice read raises."""
    d = cr.read_dump()
    if why := thermal_failure(d):
        raise RuntimeError('thermal read failed: '+why)
    return d


def thermal_row_failures(rows):
    return [f"t={r.get('t')}: {why}" for r in rows if (why := thermal_failure(r))]


def dump_check(cr):
    d = read_dump(cr)
    if d['status'] >= cr.STATUS_STOP:
        raise RuntimeError('skin/status missing or thermal stop')
    return d


def camera_version(cr):
    rc, text = cr.root('dumpsys package com.pixelrobot.robotcam', 'campaign_version')
    code = re.search(r'\bversionCode=(\d+)', text)
    name = re.search(r'\bversionName=([^\s]+)', text)
    if rc or not code or not name:
        raise RuntimeError('installed RobotCam version unreadable')
    return dict(versionCode=code.group(1), versionName=name.group(1))


def stop_camera(cr):
    try:
        cr.camera_stop()
    finally:
        check = cr.camera_end_check()
    if why := cr.camera_end_failed(check):
        raise RuntimeError('camera cleanup: '+why)
    return check


def memory_sample(cr, server_pid, camera_on):
    """MemAvailable plus one root battery/PSS read.

    coresidency.root_sample expects the RobotCam app in every state. With the camera OFF the
    app is force-stopped, so its absence alone is "not running", not a read failure. Root rc,
    battery fields, runner/server/provider PSS and a missing app with the camera ON still fail.
    """
    s = dict(t=time.monotonic(), **cr.meminfo_mib(), **cr.root_sample(server_pid))
    # Exact root_sample message when rc is 0 and the app is the only missing process.
    if not camera_on and s.get('pss_error', '').startswith("su rc 0; PSS missing for ['robotcam_app']: "):
        # root_sample discards pidof's status, so confirm absence: pidof exits 1 only when nothing matches.
        rc, out = cr.root('pidof com.pixelrobot.robotcam; echo "pidof_rc=$?"', 'campaign_pidof')
        if rc == 0 and out.split() == ['pidof_rc=1']:
            del s['pss_error']
            s['pss_kb']['robotcam_app'] = None
            s['robotcam_app'] = 'not running (camera OFF)'
        else:
            s['pss_error'] += f'; app absence unconfirmed: rc {rc}, {out[:100]!r}'
    return s


def release(caller):
    if caller:
        caller.close()
    gc.collect()


def fast_check(cr, monitor, sb=None, measured=None):
    if sb is not None:
        verify_monitor(sb, monitor, measured)
    row = cr.fast_sample(monitor[0], cr.fast_keys(monitor[3]))
    if row.get('error') or row.get('bat_c') is None or row.get('cpu_c') is None or any(v is None for v in row['max'].values()):
        raise RuntimeError('missing root CPU/battery/policy reading')
    if row['bat_c'] >= cr.BATTERY_STOP_C:
        raise RuntimeError('battery thermal stop')
    return row


@contextlib.contextmanager
def root_mask_scope(cr, sb, mask, measured):
    """Adapt legacy cr.root calls: pin/read back the root command shell before services.

    Magisk may spawn the shell from its daemon, so caller inheritance alone is insufficient.
    Binder service work in Android's existing processes remains outside our affinity control.
    """
    require_mask(mask, measured, mask, "transient shell configured", "root_mask_scope argument")
    bits = format(sum(1 << c for c in mask), 'x')
    original = cr.root
    def bound(command, tag, timeout=60):
        prefix = (f'taskset -p {bits} $$ >/dev/null; pin_rc=$?; '
                  'actual=$(taskset -p $$); read_rc=$?; '
                  "printf 'CAMPAIGN_ROOT_MASK pid=%s actual=%s\\n' \"$$\" \"$actual\"; "
                  '[ "$pin_rc" = 0 ] || exit 97; [ "$read_rc" = 0 ] || exit 98; '
                  f'case "$actual" in *": {bits}") ;; *) exit 98;; esac; ')
        rc, text = original(prefix+command, tag, timeout=timeout)
        if rc in (97, 98):
            raise RuntimeError(f'transient root shell pin/readback failed: actual={text!r} '
                               f'measured={sorted(measured)} allowed={sorted(mask)} '
                               f'process=transient shell (pid in output) how=taskset -p $$ rc={rc}')
        text = '\n'.join(line for line in text.splitlines() if not line.startswith('CAMPAIGN_ROOT_MASK '))
        return rc, text
    cr.root = bound
    try:
        yield
    finally:
        cr.root = original
