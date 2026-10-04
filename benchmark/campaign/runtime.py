"""Lazy imports of unchanged benchmark runners; native operations owner only."""
import gc
import contextlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parents[1]
HOME = Path('/data/data/com.termux/files/home')


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


def verify_monitor(sb, monitor, measured):
    shell, mask, pid, _ = monitor
    sb.require_monitor_mask(mask, measured)
    for root_pid in (pid, shell.p.pid):
        observed = sb.root_mask(shell, root_pid)
        if observed != set(mask):
            raise RuntimeError('root/su affinity changed')
        sb.require_monitor_mask(observed, measured)
    for tid in shell.monitor_tids:
        if os.sched_getaffinity(tid) != set(mask):
            raise RuntimeError('root pump affinity changed')


def prepare(sb, measured, restrict=None):
    monitor = sb.prepare_monitor(measured)
    try:
        if restrict is not None:
            shell, allowed, pid, layout = monitor
            mask = set(allowed) & set(restrict)
            sb.require_monitor_mask(mask, measured)
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


def dump_check(cr):
    d = cr.read_dump()
    if d['skin'] is None or d['status'] is None or d['status'] >= cr.STATUS_STOP:
        raise RuntimeError('skin/status missing or thermal stop')
    return d


def skin_gate(cr, baseline, label, guard):
    # Adapt power_map.thermal_gate's skin condition/bound to avoid its external thermal.log dependency.
    # CPU fault readings are collected directly by this campaign, not via a second root logger.
    began = time.monotonic()
    while True:
        guard()
        d = dump_check(cr)
        cool = d['skin'] <= baseline['skin'] + cr.SKIN_GATE_C
        waited = time.monotonic()-began
        if cool or waited >= 480:
            return dict(**d, waited_s=waited, warm_start=not cool)
        print(f'[{label}] skin {d["skin"]}; gate <= {baseline["skin"]+cr.SKIN_GATE_C:.1f}; waited {waited:.0f}s', flush=True)
        time.sleep(10)


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
    sb.require_monitor_mask(mask, measured)
    bits = format(sum(1 << c for c in mask), 'x')
    original = cr.root
    def bound(command, tag, timeout=60):
        prefix = (f'taskset -p {bits} $$ >/dev/null || exit 97; '
                  'actual=$(taskset -p $$) || exit 98; '
                  f'case "$actual" in *": {bits}") ;; *) exit 98;; esac; ')
        rc, text = original(prefix+command, tag, timeout=timeout)
        if rc in (97, 98):
            raise RuntimeError('transient root shell pin/readback failed')
        return rc, text
    cr.root = bound
    try:
        yield
    finally:
        cr.root = original
