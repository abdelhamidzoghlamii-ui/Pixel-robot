#!/usr/bin/env python3
"""Four camera heat blocks; --helper runs in native Termux, --run under oneshot's Debian."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

NATIVE = Path('/data/data/com.termux/files/home')
SHARED = (Path('/termux-home') if Path('/termux-home').is_dir() else NATIVE) / 'ladder'
BATTERY = '/sys/class/power_supply/battery'
FIELDS = [f'/sys/class/thermal/thermal_zone{i}/temp' for i in (9, 10, 11)] + [
    f'{BATTERY}/{name}' for name in ('current_now', 'voltage_now', 'status')]
BLOCKS = ('idle', 'robotcam_1', 'robotcam_2', 'old_path')


def put(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def shell(command, timeout=20):
    return subprocess.run(command, stdin=subprocess.DEVNULL, text=True,
                          capture_output=True, timeout=timeout, check=True).stdout


def root(command, timeout=20):
    return shell(['su', '-c', command], timeout)


def sample():
    vals = root('cat ' + ' '.join(FIELDS)).splitlines()
    if len(vals) != len(FIELDS):
        raise RuntimeError(f'root sensor fields: expected {len(FIELDS)}, got {len(vals)}')
    return {'t': time.monotonic(), 'utc': datetime.now(timezone.utc).isoformat(),
            **dict(zip(('z9', 'z10', 'z11', 'current_ua', 'voltage_uv', 'status'),
                       (int(x) if i < 5 else x for i, x in enumerate(vals))))}


def pss(label, target, run_id):
    # Android binder commands must get an EOF on stdin and write to /data/local/tmp.
    path = f'/data/local/tmp/camera_heat_{run_id}_{label}.txt'
    root(f'dumpsys meminfo {target} </dev/null > {path}', 30)
    raw = root(f'cat {path}')
    match = re.search(r'TOTAL PSS:\s*([\d,]+)', raw)
    if not match:
        match = re.search(r'^\s*TOTAL\s+([\d,]+)\b', raw, re.M)
    return {'pss_kb': int(match.group(1).replace(',', '')) if match else None,
            'raw': path, 'error': None if match else 'TOTAL PSS not found'}


def memory(run_id):
    out = {}
    try:
        out['app'] = pss('app', 'com.pixelrobot.robotcam', run_id)
    except (OSError, subprocess.SubprocessError) as e:
        out['app'] = {'pss_kb': None, 'error': str(e)}
    try:
        processes = root('ps -A -o PID,NAME').splitlines()
        pids = [line.split()[0] for line in processes if 'android.hardware.camera.provider' in line]
        out['provider'] = [pss(f'provider_{pid}', pid, run_id) for pid in pids]
        if not pids:
            out['provider_error'] = 'camera provider process not found'
    except (OSError, subprocess.SubprocessError) as e:
        out['provider'] = []
        out['provider_error'] = str(e)
    return out


def old_capture(state):
    while time.monotonic() < state['deadline']:
        path = Path(state['dir']) / f"{state['ok'] + state['failed']:05d}.jpg"
        try:
            subprocess.run(['termux-camera-photo', str(path)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                           timeout=15)
            good = path.exists() and path.stat().st_size > 1000
        except (OSError, subprocess.SubprocessError):
            good = False
        state['ok' if good else 'failed'] += 1
        time.sleep(max(0, min(.5, state['deadline'] - time.monotonic())))


def helper():
    root('id')  # fail before the five-minute oneshot idle if Magisk is unavailable
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_{os.getpid()}'
    directory = SHARED / f'camera_heat_{run_id}'
    directory.mkdir()
    put(SHARED / 'camera_heat_active.json', {'name': directory.name, 'id': run_id})
    readings = directory / 'sensors.jsonl'
    reader = None
    reader_log = None
    old = None
    stop_samples = threading.Event()
    sample_error = []

    def log_sensors():
        while not stop_samples.is_set():
            try:
                with readings.open('a') as f:
                    f.write(json.dumps(sample()) + '\n')
            except (OSError, subprocess.SubprocessError, RuntimeError) as e:
                sample_error.append(str(e))
                return
            stop_samples.wait(5)

    sensor_thread = threading.Thread(target=log_sensors, daemon=True)
    sensor_thread.start()
    number = 0
    camera_active_since = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        while True:
            if sample_error:
                raise RuntimeError(f'root sensor logger failed: {sample_error[-1]}')
            if camera_active_since and time.monotonic() - camera_active_since > 240:
                raise TimeoutError('camera active without runner response for 240s')
            request = directory / f'request_{number}.json'
            if not request.exists():
                time.sleep(.1)
                continue
            req = json.loads(request.read_text())
            action = req['action']
            try:
                if action == 'start':
                    rate = req['rate']
                    shell(['am', 'start', '-n', 'com.pixelrobot.robotcam/.StartActivity',
                           '--es', 'mode', 'B', '--ei', 'rate', str(rate)])
                    camera_active_since = time.monotonic()
                    result = {'ok': True}
                elif action == 'stop':
                    shell(['am', 'broadcast', '-n', 'com.pixelrobot.robotcam/.ControlReceiver',
                           '-a', 'com.pixelrobot.robotcam.STOP'])
                    camera_active_since = None
                    result = {'ok': True}
                elif action == 'reader_start':
                    reader_log = (directory / f"{req['block']}.reader.txt").open('w')
                    reader = subprocess.Popen([sys.executable, str(NATIVE / 'robot/benchmark/robotcam/robotcam_test.py'),
                                               '-n', '10000', '--interval', str(req['interval'])],
                                              stdin=subprocess.DEVNULL, stdout=reader_log,
                                              stderr=subprocess.STDOUT,
                                              env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                    camera_active_since = time.monotonic()
                    result = {'ok': True}
                elif action == 'reader_stop':
                    reader.terminate()
                    try:
                        reader.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        reader.kill()
                        reader.wait()
                    reader_log.close()
                    lines = (directory / f"{req['block']}.reader.txt").read_text().splitlines()
                    reads = [s for s in lines if re.match(r'^\s*\d+\s+(frame|MISSING|BAD|OTHER_SESSION)', s)]
                    failed = sum(bool(re.search(r'\b(MISSING|BAD|OTHER_SESSION)\b', s)) for s in reads)
                    result = {'ok': len(reads) - failed, 'failed': failed, 'attempts': len(reads),
                              'exit_code': reader.returncode}
                    reader = None
                elif action == 'old_start':
                    old = {'dir': tempfile.mkdtemp(prefix='camera_heat_frames_', dir=directory),
                           'ok': 0, 'failed': 0, 'deadline': time.monotonic() + req['duration']}
                    old['thread'] = threading.Thread(target=old_capture, args=(old,), daemon=True)
                    old['thread'].start()
                    camera_active_since = time.monotonic()
                    result = {'ok': True}
                elif action == 'old_stop':
                    old['deadline'] = min(old['deadline'], time.monotonic())
                    old['thread'].join(timeout=17)
                    if old['thread'].is_alive():
                        raise RuntimeError('old camera capture did not stop')
                    result = {'ok': old['ok'], 'failed': old['failed']}
                    shutil.rmtree(old['dir'])  # only old_path temporary frames
                    old = None
                    camera_active_since = None
                elif action == 'meminfo':
                    result = memory(run_id)
                elif action == 'finish':
                    result = {'ok': True}
                else:
                    raise ValueError(f'unknown action {action}')
            except Exception as e:
                result = {'error': f'{type(e).__name__}: {e}'}
            put(directory / f'response_{number}.json', result)
            number += 1
            if action == 'finish':
                break
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        stop_samples.set()
        sensor_thread.join(timeout=10)
        if reader is not None:
            reader.terminate()
            reader.wait()
            reader_log.close()
        if old is not None:
            old['deadline'] = time.monotonic()
            old['thread'].join(timeout=17)
            if not old['thread'].is_alive():
                shutil.rmtree(old['dir'])
        try:
            shell(['am', 'broadcast', '-n', 'com.pixelrobot.robotcam/.ControlReceiver',
                   '-a', 'com.pixelrobot.robotcam.STOP'])
        except (OSError, subprocess.SubprocessError) as e:
            print(f'RobotCam stop failed: {e}', file=sys.stderr)


def run():
    duration = 20 if '--toy' in sys.argv else 180
    active = json.loads((SHARED / 'camera_heat_active.json').read_text())
    directory = SHARED / active['name']
    readings = directory / 'sensors.jsonl'
    seq = 0

    def ask(action, **kwargs):
        nonlocal seq
        current = seq
        seq += 1
        put(directory / f'request_{current}.json', {'action': action, **kwargs})
        answer = directory / f'response_{current}.json'
        limit = time.monotonic() + (40 if action == 'meminfo' else 25)
        while not answer.exists():
            if time.monotonic() > limit:
                raise TimeoutError(f'helper did not answer {action}')
            time.sleep(.1)
        value = json.loads(answer.read_text())
        if 'error' in value:
            raise RuntimeError(f'{action}: {value["error"]}')
        return value

    def samples():
        lines = readings.read_text().splitlines()
        rows = [json.loads(line) for line in lines[:-1]]
        if lines:
            try:
                rows.append(json.loads(lines[-1]))
            except json.JSONDecodeError:
                pass  # the sampler is still appending its last line
        if not rows or time.monotonic() - rows[-1]['t'] > 15:
            raise RuntimeError('native root sensor logger is stale')
        return rows

    idle = samples()[-1]['z9']
    results = []
    print(f'camera heat {duration}s toy={duration == 20}, idle z9={idle / 1000:.1f}C', flush=True)
    try:
        ask('stop')
        for name in BLOCKS:
            while samples()[-1]['z9'] > idle + 4000:
                print(f'{name}: waiting for z9 <= {(idle + 4000) / 1000:.1f}C', flush=True)
                time.sleep(10)
            if name.startswith('robotcam'):
                rate = 1 if name == 'robotcam_1' else 2
                ask('start', rate=rate)
                ask('reader_start', block=name, interval=1 / rate)
            elif name == 'old_path':
                ask('old_start', duration=duration)
            start = time.monotonic()
            middle = start + duration / 2
            end = start + duration
            mem = None
            if name != 'idle':
                time.sleep(max(0, middle - time.monotonic()))
                mem = ask('meminfo')
            time.sleep(max(0, end - time.monotonic()))
            end = time.monotonic()
            if name.startswith('robotcam'):
                counts = ask('reader_stop', block=name)
                if counts['attempts'] == 0:
                    raise RuntimeError(f'{name}: reader produced no reads; see reader log')
                ask('stop')
            elif name == 'old_path':
                counts = ask('old_stop')
            else:
                counts = {'ok': 0, 'failed': 0}
            rows = [s for s in samples() if start <= s['t'] <= end]
            if len(rows) < 2:
                raise RuntimeError(f'{name}: too few sensor samples')
            watts = [-(s['current_ua'] * s['voltage_uv']) / 1e12 for s in rows]
            result = {'block': name, 'seconds': end - start, 'temp_start_c': rows[0]['z9'] / 1000,
                      'temp_end_c': rows[-1]['z9'] / 1000,
                      'rise_c_per_min': (rows[-1]['z9'] - rows[0]['z9']) / 1000 /
                                        ((rows[-1]['t'] - rows[0]['t']) / 60),
                      'mean_battery_draw_w': sum(watts) / len(watts),
                      'charging_status': sorted({s['status'] for s in rows}),
                      'frames': counts, 'meminfo': mem, 'samples': len(rows)}
            results.append(result)
            put(directory / 'results.json', {'duration_s': duration, 'idle_z9_mc': idle, 'blocks': results})
            print(json.dumps(result), flush=True)
    finally:
        try:
            ask('stop')
        finally:
            ask('finish')
    print(f'results: {directory / "results.json"}', flush=True)


if __name__ == '__main__':
    helper() if '--helper' in sys.argv else run()
