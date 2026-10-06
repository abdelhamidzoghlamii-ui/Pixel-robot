#!/data/data/com.termux/files/usr/bin/python
"""Co-residency benchmark (DECISIONS #124). Research only, motors off: never imports motors.py or
opens USB/serial. Native Termux only (the robot runs ONNX Runtime and llama-server natively);
started by oneshot.sh in this folder, which runs the thermal logger and answers cache-drop requests.

Blocks, in order (180 s each; --smoke: 20 s, no cooldown gate). Before each block and the loads, the cooldown gate
waits for VIRTUAL-SKIN <= idle + 1.5 degC and zone9 <= idle + 4 degC; after 15 min the block starts as a warm start:
  B1_mix_nollm      RobotCam mode B rate 2 -> read_frame (fixed session, frame must advance) ->
                    Detector at SizePolicy drive mode sizes (320 each frame, 640 every 5 s)
  B2_only640_nollm  same, always 640
  (loads)           Gemma cold load after a page-cache drop (handshake), then warm load by restart
  B3_mix_gemma      B1 + resident llama-server (server_manager setup_q4 flags) + a selector
                    letter-scoring call every 20 s in its own thread (s1o, variant s1o_b1609dp_q40 prompts)
  B4_only640_gemma  B2 + the same
  B5_mtp_ram_snapshot  (not timed) Gemma with the conv_mtp drafter flags, one short request, RAM snapshot
B1-B4 end early at the first block limit: Android thermal status >= CRITICAL, battery >= 45.0 degC, or a CPU zone
(BIG/MID/LITTLE) >= 110 degC in 3 consecutive 1 s samples (fault stop): a result, not a failure. Fail closed (block
stopped, INCOMPLETE): no VIRTUAL-SKIN + status reading for 60 s, no CPU zone or battery reading for 5 s. A limit met
before the first frame: "not run: limit at start", INCOMPLETE.

Every 5 s: MemAvailable/swap, PSS (root dumpsys meminfo) of the runner, llama-server, RobotCam app and
camera provider, zone9/10/11 from the thermal log, battery power (camera_heat.py's method). Per block:
LMK log lines, RobotCam/llama-server survival. Per read: status, read/decode/detect ms, size, frame age.
Per block, every 1 s (one persistent root shell): CPU zones, battery temp, scaling_max_freq per cpufreq policy;
every 5 s: Android thermal status and VIRTUAL-SKIN from `dumpsys thermalservice` (DECISIONS #123 form). After each
block's STOP: capture must stop (no new frame for 3 s), then `am force-stop` the app (it stays cached after STOP).

Usage: coresidency.py [--smoke] [--resume RUN_DIR] [--thermal-log PATH]
"""
import argparse
import hashlib
import io
import json
import math
import os
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True  # no .pyc into the ladder/v3 archive folders this imports from
HOME = Path('/data/data/com.termux/files/home')
ROBOT = HOME / 'robot'
LADDER = ROBOT / 'benchmark/strategic_selector/ladder'
sys.path[:0] = [str(ROBOT), str(LADDER)]
import robotcam_reader  # noqa: E402
from robotcam_reader import read_frame  # noqa: E402
from detector_size_policy import SizePolicy  # noqa: E402
from detect_person import Detector  # noqa: E402
import server_manager  # noqa: E402  (binary path and thread flags only; nothing started through it)
from measure import allowed_cpus  # noqa: E402  (the ladder's cpuset guard)

OUT_ROOT = HOME / 'coresidency'  # oneshot.sh's folder: thermal.log, handshake files, run folders
DOWNLOADS = HOME / 'storage/downloads'
FRAME_DIR = robotcam_reader.FRAME_DIR
CASES = LADDER / 'cases/ladder_cases_v1.jsonl'
V3 = ROBOT / 'benchmark/strategic_selector/v3'  # adapters.S1O
S1O_SRC = HOME / 'sel-candidates/s1o/src'       # s1.schema (stdlib only)
MODEL = str(HOME / 'models/gemma-4-E2B-it-Q4_0.gguf')  # server_manager.start_setup('setup_q4')
DRAFT = str(HOME / 'models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf')
MTP_ARGS = ['--model-draft', DRAFT, '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']  # conv_speed_mtp.py
PORT = 8080
BATTERY = '/sys/class/power_supply/battery'
BLOCK_S, SMOKE_S = 180, 20
SAMPLE_S, SELECTOR_S, DRIFT_S = 5, 20, 30
BLOCKS = [('B1_mix_nollm', 'mix', False), ('B2_only640_nollm', '640', False),
          ('B3_mix_gemma', 'mix', True), ('B4_only640_gemma', '640', True)]
LARGE_S = 5  # main.LARGE_FRAME_INTERVAL_S: drive mode runs one 640 frame every 5 s
# Block limits. zone9 is no heat signal (capped near 100 degC within seconds, then held at 62-68 degC while the
# phone heats); Android throttles on VIRTUAL-SKIN: status LIGHT 39, MODERATE 43, SEVERE 45, CRITICAL 46.5 (hard caps).
THERMAL, CPUFREQ = '/sys/class/thermal', '/sys/devices/system/cpu/cpufreq'
SKIN, CPU_TYPES = 'VIRTUAL-SKIN', ('BIG', 'MID', 'LITTLE')
STATUS_NAMES = ['NONE', 'LIGHT', 'MODERATE', 'SEVERE', 'CRITICAL', 'EMERGENCY', 'SHUTDOWN']
STATUS_STOP, BATTERY_STOP_C, CPU_FAULT_C, CPU_N = 4, 45.0, 110.0, 3
FAST_S, DUMP_S, SKIN_WAIT_S, SENSOR_WAIT_S = 1, 5, 60, 5
SKIN_GATE_C, GATE_MAX_S, GATE_POLL_S = 1.5, 15 * 60, 10
DEFAULT_INSTRUCTION = 'Choose the single best next action for the robot.'  # ladder.py
WARMUP = ('The robot is idle in the hallway.', {'wait': '', 'explore': ''}, 'Pick one.')  # ladder_worker.py
MTP_PROMPT = 'In one short sentence, what does a home robot do?'
LMK_RE = re.compile(r'lowmemorykiller|lmkd|killinfo', re.I)
LMK_KILL_RE = re.compile(r'(lowmemorykiller|lmkd)\b.*\bkill|killinfo', re.I)  # lmkd kill lines, killinfo events
DEVNULL = subprocess.DEVNULL


def require_native():
    if Path('/termux-home').is_dir() or os.environ.get('PREFIX') != '/data/data/com.termux/files/usr':
        raise SystemExit('Run from native Termux, not Debian/proot.')


_deferred = {'on': False, 'signum': None}  # set while a detached llama-server is spawned but not yet owned


def exit_on_signal(signum, frame):
    if _deferred['on']:
        _deferred['signum'] = signum  # raised by Server.__init__ once it holds the process
        return
    raise SystemExit(128 + signum)


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- root and Android

def root(cmd, tag, timeout=60):
    """DECISIONS #123 form: su -c "<cmd> </dev/null >FILE 2>&1", then the file is read. Returns (rc, text)."""
    path = f'/data/local/tmp/coresidency_{tag}.txt'
    r = subprocess.run(['su', '-c', f'{{ {cmd}; }} </dev/null >{path} 2>&1; rc=$?; cat {path}; exit $rc'],
                       stdin=DEVNULL, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout


def am(*args):
    """Native Termux `am`, exactly as main.Robot.start_camera/stop_camera call it (no root)."""
    subprocess.run(['am', *args], check=True, timeout=10, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)


def camera_stop():
    am('broadcast', '-n', 'com.pixelrobot.robotcam/.ControlReceiver', '-a', 'com.pixelrobot.robotcam.STOP')


def camera_start():
    """main.Robot.start_camera, retried: overlapping opens on restart can need a retry (DECISIONS #122)."""
    for attempt in range(1, 4):
        started_boot_s = time.clock_gettime(time.CLOCK_BOOTTIME)
        am('start', '-n', 'com.pixelrobot.robotcam/.StartActivity', '--es', 'mode', 'B', '--ei', 'rate', '2')
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            r = read_frame(FRAME_DIR, min_capture_boot_s=started_boot_s)
            if r['status'] == 'ok':
                return {'session': r['session'], 'frame': r['frame'], 'attempts': attempt}
            time.sleep(0.1)
        camera_stop()
        time.sleep(2)
    raise RuntimeError('RobotCam did not publish a usable frame in 3 attempts')


def camera_end_check(rootf=None, quiet_s=3, limit_s=15):
    """After the STOP broadcast: capture has stopped once every read_frame for quiet_s s (up to limit_s) shows
    evidence of no new frame: 'missing' (STOP deletes the files; an old frame reads as missing) or the same
    (session, frame) again. A new frame or an unreadable one ('bad': no evidence) restarts the window. Then
    `am force-stop` and pidof, also when capture did not stop, in the DECISIONS #123 form (rootf: coresidency.root
    or thermal_char.root_file). The app process stays cached after STOP, so a pid before the force-stop is no failure."""
    rootf = rootf or root
    began = quiet_from = time.monotonic()
    last, seen, unreadable = None, [], 0
    while time.monotonic() - quiet_from < quiet_s and time.monotonic() - began < limit_s:
        r = read_frame(FRAME_DIR)
        if r['status'] == 'ok' and (r['session'], r['frame']) != last:
            last, quiet_from = (r['session'], r['frame']), time.monotonic()
            seen.append({'session': r['session'], 'frame': r['frame'], 't': round(quiet_from - began, 2)})
        elif r['status'] not in ('ok', 'missing'):
            unreadable, quiet_from = unreadable + 1, time.monotonic()
        time.sleep(0.1)
    out = {'capture_stopped': time.monotonic() - quiet_from >= quiet_s, 'frames_seen': seen,
           'unreadable_reads': unreadable, 'check_s': round(time.monotonic() - began, 2)}
    try:
        rc, text = rootf('am force-stop com.pixelrobot.robotcam', 'forcestop')
        out.update(force_stop_rc=rc, force_stop_output=text.strip()[:200])
        # Keep pidof's own status: it exits 1 only when no process matches (0 = pids remain, other = failed).
        rc, text = rootf('pidof com.pixelrobot.robotcam; echo "pidof_rc=$?"', 'pidof')
        status = re.search(r'^pidof_rc=(\d+)$', text, re.M)
        out.update(pidof_root_rc=rc, pidof_rc=int(status.group(1)) if status else None,
                   pids_after_force_stop=[w for w in text.split() if w.isdigit()])
    except (OSError, subprocess.SubprocessError) as e:
        out.update(force_stop_rc=None, force_stop_output=f'{type(e).__name__}: {e}')
    return out


def camera_end_failed(c):
    """Why a camera_end_check result fails the run (capture kept advancing, force-stop failed, RobotCam still
    running or its absence unconfirmed after the force-stop), else None. Fails closed on a missing pidof status."""
    why = []
    if not c.get('capture_stopped'):
        why.append(f'capture not shown stopped {c.get("check_s")} s after STOP ({len(c.get("frames_seen", []))} '
                   f'new frames, {c.get("unreadable_reads")} unreadable reads)')
    if c.get('force_stop_rc') != 0:
        why.append(f'am force-stop failed (rc {c.get("force_stop_rc")}: {c.get("force_stop_output")!r})')
    if c.get('pids_after_force_stop'):
        why.append(f'RobotCam still running after force-stop (pids {c["pids_after_force_stop"]})')
    elif c.get('force_stop_rc') == 0 and (c.get('pidof_root_rc') != 0 or c.get('pidof_rc') != 1):
        why.append(f'RobotCam absence after force-stop unconfirmed (root rc {c.get("pidof_root_rc")}, '
                   f'pidof rc {c.get("pidof_rc")})')
    return '; '.join(why) or None


def meminfo_mib():
    m = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return {'mem_available_mib': m['MemAvailable'] // 1024, 'swap_used_mib': (m['SwapTotal'] - m['SwapFree']) // 1024,
            'cached_mib': m['Cached'] // 1024}


def pss_kb(text):
    """TOTAL PSS from one `dumpsys meminfo` section (camera_heat.py's parsing)."""
    m = re.search(r'TOTAL PSS:\s*([\d,]+)', text) or re.search(r'^\s*TOTAL\s+([\d,]+)\b', text, re.M)
    return int(m.group(1).replace(',', '')) if m else None


def root_sample(server_pid):
    """One su call: battery (camera_heat.py's sysfs fields) and PSS of the runner, llama-server, RobotCam app
    and camera provider. Watts = -(current_now uA * voltage_now uV) / 1e12, as camera_heat.py."""
    pids = [('runner', os.getpid())] + ([('llama_server', server_pid)] if server_pid else [])
    script = (f'cat {BATTERY}/current_now {BATTERY}/voltage_now {BATTERY}/status; ' +
              ''.join(f'echo "=== {n} {p}"; dumpsys meminfo {p} </dev/null; ' for n, p in pids) +
              'for p in $(pidof com.pixelrobot.robotcam); do echo "=== robotcam_app $p"; '
              'dumpsys meminfo $p </dev/null; done; '
              'ps -A -o PID,NAME | grep android.hardware.camera.provider | while read p n; do '
              'echo "=== camera_provider $p"; dumpsys meminfo $p </dev/null; done')
    began = time.monotonic()
    rc, out = root(script, 'sample')
    parts = re.split(r'^=== (\S+) (\d+)$', out, flags=re.M)
    head = parts[0].split()
    s = {'root_rc': rc, 'root_s': round(time.monotonic() - began, 2), 'pss_kb': {}, 'pss_pids': {}}
    try:
        s['battery_w'] = -(int(head[0]) * int(head[1])) / 1e12
        s['battery_status'] = head[2]
    except (IndexError, ValueError):
        s['error'] = f'battery fields: {parts[0][:200]!r}'
    missing = []
    for name, pid, text in zip(parts[1::3], parts[2::3], parts[3::3]):
        kb = pss_kb(text)
        s['pss_pids'].setdefault(name, []).append(int(pid))
        if kb is None:
            missing.append(f'{name} {pid}')  # any failed pid, even when another pid of that name parsed
        else:
            s['pss_kb'][name] = s['pss_kb'].get(name, 0) + kb  # several camera provider processes: summed
    expected = ['runner', 'robotcam_app', 'camera_provider'] + (['llama_server'] if server_pid else [])
    missing += [n for n in expected if n not in s['pss_pids']]
    if rc != 0 or missing:
        s['pss_error'] = f'su rc {rc}; PSS missing for {missing}: {out[:200]!r}'
    return s


def robotcam_pids():
    rc, out = root('pidof com.pixelrobot.robotcam', 'pidof')
    return [int(p) for p in out.split() if p.isdigit()]


def lmk_lines(since_epoch):
    """LMK-related log lines since since_epoch (logcat -T, all buffers); a failed logcat query is reported, not 0."""
    raw = '/data/local/tmp/coresidency_lmk_raw.txt'
    rc, out = root(f'logcat -d -b all -v epoch -T {since_epoch:.3f} >{raw} 2>&1; echo "logcat_rc=$?"; '
                   f'head -c 300 {raw}; echo; echo "=== lmk lines"; grep -iE "lowmemorykiller|lmkd|killinfo" {raw}', 'lmk')
    head, _, found = out.partition('=== lmk lines\n')
    m = re.search(r'^logcat_rc=(\d+)$', head, re.M)
    lines = [l for l in found.splitlines() if LMK_RE.search(l)]
    return {'lines': lines, 'n_lines': len(lines), 'n_kills': sum(bool(LMK_KILL_RE.search(l)) for l in lines),
            'logcat_rc': int(m.group(1)) if m else None, 'ok': m is not None and m.group(1) == '0',
            'raw_head': head[:400]}


# ---------------------------------------------------------------- thermal and cold load (from ladder.py)

THERMAL_MAX_AGE_S = 30
GATE_MC = 4000


def read_thermal(path):
    """ladder.read_thermal: last line of the root thermal log, e.g. '2026-09-25T21:51:27Z z9=36000 z10=... z11=...'."""
    last = Path(path).read_text().strip().splitlines()[-1].split()
    stamp = datetime.strptime(last[0], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if age > THERMAL_MAX_AGE_S:
        raise SystemExit(f'thermal log {path} last line is {age:.0f} s old: the logger is not running')
    z = dict(kv.split('=') for kv in last[1:])
    return {'at': last[0], **{k: int(z[k]) for k in ('z9', 'z10', 'z11')}}


def thermal_gate(path, idle, label, smoke):
    """Cooldown gate: wait until VIRTUAL-SKIN <= idle skin + 1.5 degC and z9 <= idle + 4 degC (ladder.thermal_gate);
    not reached in 15 min: start anyway, marked warm_start. --smoke skips the wait."""
    began = time.monotonic()
    while True:
        t, d = read_thermal(path), read_dump()
        cool = t['z9'] <= idle['z9'] + GATE_MC and d['skin'] is not None and d['skin'] <= idle['skin'] + SKIN_GATE_C
        waited = time.monotonic() - began
        if smoke or cool or waited >= GATE_MAX_S:
            break
        print(f'  [{label}] waiting: skin {d["skin"]} degC, need <= {idle["skin"] + SKIN_GATE_C:.1f}; z9 '
              f'{t["z9"] / 1000:.1f} degC, need <= {(idle["z9"] + GATE_MC) / 1000:.1f}; {waited:.0f} s', flush=True)
        time.sleep(GATE_POLL_S)
    return {**t, 'skin': d['skin'], 'status': d['status'], 'waited_s': round(waited), 'warm_start': not (smoke or cool)}


class CoresLost(Exception):
    pass


def check_cores(when):
    if not set(range(4, 8)) <= allowed_cpus():
        raise CoresLost(f'cores 4-7 not all allowed {when} (allowed {sorted(allowed_cpus())})')


def wait_cores(label):
    began = time.monotonic()
    while not set(range(4, 8)) <= allowed_cpus():
        print(f'  [{label}] paused: cores 4-7 not allowed (allowed {sorted(allowed_cpus())}); bring Termux to the '
              f'foreground. {time.monotonic() - began:.0f} s', flush=True)
        time.sleep(10)


HANDSHAKE_S = 60


def make_cold():
    """ladder.make_cold, handshake form: oneshot.sh's watcher drops the page cache as root on .drop_request and
    writes ok/failed to .drop_done; checked in /proc/meminfo. Else the GGUF alone is evicted (posix_fadvise)."""
    request, done = OUT_ROOT / '.drop_request', OUT_ROOT / '.drop_done'
    before = meminfo_mib()['cached_mib']
    done.unlink(missing_ok=True)
    request.touch()
    print(f'[loads] cold load next: handshake, waiting up to {HANDSHAKE_S} s for {done}', flush=True)
    deadline, answer = time.monotonic() + HANDSHAKE_S, None
    try:
        while time.monotonic() < deadline:
            if done.exists() and (text := done.read_text().strip()):
                answer = text
                break
            time.sleep(0.5)
    finally:
        request.unlink(missing_ok=True)
        done.unlink(missing_ok=True)
    after = meminfo_mib()['cached_mib']
    # Cached keeps what drop_caches cannot free (shmem, mapped pages), so a Cached ratio is no test of the drop;
    # the test is that no page of what the cold load reads (GGUF, llama-server, its build libs) is still cached.
    try:
        resident = resident_pages(cold_files())
    except (OSError, AttributeError, ValueError) as e:
        resident = f'{type(e).__name__}: {e}'
    state = {'cached_mib_before': before, 'cached_mib_after': after, 'resident_pages_after': resident}
    if answer == 'ok' and resident == 0:
        return {'mode': 'full-cold (page cache dropped by the handshake watcher)', **state}
    fd = os.open(MODEL, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return {'mode': 'weights-cold (posix_fadvise DONTNEED on the GGUF)', **state,
            'fallback_reason': f'handshake answer {answer!r}, {resident} load-file pages still cached, '
                               f'Cached {before} -> {after} MiB'}


def cold_files():
    """What a llama-server start reads from disk: the GGUF, the binary and the libraries beside it (RUNPATH)."""
    binary = Path(server_manager.LLAMA_SERVER)
    return sorted({Path(MODEL), binary, *(p.resolve() for p in binary.parent.glob('lib*.so*'))})


def resident_pages(paths):
    """Page-cache pages of these files, by mincore(2) on a read-only shared mapping (mapping reads nothing).
    The kernel reports every page as resident for a file the caller neither owns nor may write: fails closed."""
    import ctypes
    import mmap
    import numpy as np
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    total = 0
    for p in paths:
        size = os.path.getsize(p)
        with open(p, 'rb') as fh, mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_READ) as m:
            view = np.frombuffer(m, dtype=np.uint8)
            vec = ctypes.create_string_buffer((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE)
            rc = libc.mincore(view.ctypes.data, size, vec)
            del view  # the mapping cannot close while exported
            if rc != 0:
                raise OSError(ctypes.get_errno(), f'mincore {p}')
            total += sum(b & 1 for b in vec.raw)
    return total


# ---------------------------------------------------------------- llama-server and selector

def server_cmd(extra=()):
    """server_manager.start_server for setup_q4: same binary, model and flags in the same order (its trailing
    `2>/dev/null &` replaced by a log file). No taskset: the robot does not pin the server."""
    return [server_manager.LLAMA_SERVER, '-m', MODEL, '--port', str(PORT), '--ctx-size', '2048',
            '--threads', server_manager.THREADS, '--threads-batch', server_manager.BATCH_THREADS,
            '--parallel', '1', '--swa-full', '--cache-ram', '0', '--host', '127.0.0.1', *extra]


def post(path, body, timeout=600):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def healthy():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
            return r.status == 200 and json.loads(r.read()).get('status') == 'ok'
    except (OSError, ValueError):
        return False


LIVE = set()  # every started llama-server until stopped; main's finally stops them all, whoever owns them


class Server:
    def __init__(self, log_path, extra=()):
        if healthy():
            raise RuntimeError(f'something already answers on port {PORT}')
        self.cmd = server_cmd(extra)
        self.log = open(log_path, 'ab')
        began = time.perf_counter()
        _deferred.update(on=True, signum=None)
        try:  # signals are deferred until the process is registered in LIVE
            self.proc = subprocess.Popen(self.cmd, stdin=DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            LIVE.add(self)
        finally:
            _deferred['on'] = False
        try:  # errors and signals from here on stop it at once (main's finally would too, through LIVE)
            if _deferred['signum'] is not None:
                raise SystemExit(128 + _deferred['signum'])
            while not healthy():
                if self.proc.poll() is not None or time.perf_counter() - began > 180:
                    raise RuntimeError(f'llama-server not healthy (exit={self.proc.poll()}); see {log_path}')
                time.sleep(0.05)
        except BaseException:
            self.stop()
            raise
        self.load_s = round(time.perf_counter() - began, 3)

    def alive(self):
        return self.proc.poll() is None and healthy()

    def status_kb(self):
        """VmRSS/VmHWM of the server (our own child, readable without root)."""
        try:
            return {l.split(':')[0]: int(l.split()[1]) for l in open(f'/proc/{self.proc.pid}/status')
                    if l.startswith(('VmRSS', 'VmHWM', 'VmSwap'))}
        except OSError as e:
            return {'error': str(e)}

    def stop(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
            except ProcessLookupError:
                pass
        LIVE.discard(self)
        self.log.close()


def make_selector():
    """adapters.S1O (the ladder's s1o letter scoring) pointed at our server: S1O.__init__ would start its own
    llama-server on port 8091, so its connection setup is repeated here; decide/completion_probs are S1O's."""
    sys.path[:0] = [str(V3), str(S1O_SRC)]
    from adapters import S1O
    from s1.schema import Example, Q, LETTERS, build
    s = S1O.__new__(S1O)
    s.Example, s.Q, s.LETTERS, s.build = Example, Q, LETTERS, build
    s.url = f'http://127.0.0.1:{PORT}'
    with_bos = s._post('/tokenize', {'content': '', 'add_special': True})['tokens']
    s.bos_token_id = with_bos[0] if with_bos else None
    s.letter_ids = [s.encode(L)[0] for L in LETTERS]
    assert all(len(s.encode(L)) == 1 for L in LETTERS[:12])
    s.extra = {}
    return s


def load_cases():
    """ladder.load_cases subset: list options become {option: ''}, as in the s1o_b1609dp_q40 run."""
    out = []
    for line in CASES.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            opts = c['options'] if isinstance(c['options'], dict) else {o: '' for o in c['options']}
            out.append({'id': c['id'], 'situation': c['situation'], 'options': opts, 'answer': c['answer'],
                        'instruction': c.get('instruction', DEFAULT_INSTRUCTION)})
    return out


def select(sel, case):
    began = time.perf_counter()
    dist = sel.decide(case['situation'], case['options'], case['instruction'])
    ms = (time.perf_counter() - began) * 1000
    keys = list(case['options'])
    choice = max(keys, key=lambda k: dist[k])  # ties: first offered, as in ladder.py
    return {'case_id': case['id'], 'ms': round(ms, 1), 'letter': sel.LETTERS[keys.index(choice)], 'choice': choice,
            'correct': choice == case['answer'], 'prompt_tokens': sel.extra.get('prompt_tokens')}


def selector_loop(sel, cases, t0, duration, stop, calls):
    k = 0
    while (at := t0 + k * SELECTOR_S) < t0 + duration:
        if stop.wait(max(0.0, at - time.monotonic())):
            return
        case = cases[k % len(cases)]
        started = time.monotonic() - t0
        try:
            row = select(sel, case)
        except Exception as e:
            row = {'case_id': case['id'], 'error': f'{type(e).__name__}: {e}'}
        calls.append({'t': round(at - t0, 1), 'started_s': round(started, 2),
                      'ended_s': round(time.monotonic() - t0, 2), **row})
        k += 1


# ---------------------------------------------------------------- sampling and frames

def sample(t0, thermal_log, server):
    s = {'t': round(time.monotonic() - t0, 2), 'utc': utc(), **meminfo_mib()}
    try:
        t = read_thermal(thermal_log)
        s.update(z9=t['z9'], z10=t['z10'], z11=t['z11'], thermal_at=t['at'])
    except (SystemExit, OSError, ValueError, IndexError, KeyError) as e:
        s['thermal_error'] = str(e)
    try:
        s.update(root_sample(server.proc.pid if server else None))
    except (OSError, subprocess.SubprocessError) as e:
        s['error'] = f'{type(e).__name__}: {e}'
    s['t_end'] = round(time.monotonic() - t0, 2)  # the su call can take seconds: its values are from t..t_end
    return s


def sampler_loop(t0, thermal_log, server, stop, samples):
    k = 0
    while not stop.wait(max(0.0, t0 + k * SAMPLE_S - time.monotonic())):
        samples.append(sample(t0, thermal_log, server))
        k = max(k + 1, math.ceil((time.monotonic() - t0) / SAMPLE_S))  # an overrun skips slots, never bunches


class _Stamp:
    """Stand-in for robotcam_reader's `io` module: read_frame reads frame.jpg, then wraps the bytes in
    io.BytesIO before decoding, so stamping that call splits read_frame into read and decode time."""
    at = None

    @staticmethod
    def BytesIO(data):
        _Stamp.at = time.perf_counter()
        return io.BytesIO(data)


robotcam_reader.io = _Stamp


# ---------------------------------------------------------------- block limit readings
# Copied from ../thermal_char/thermal_char.py (it imports this file, so no import back): RootShell and TEMP_RE/
# parse_dump verbatim (END marker renamed), to_int; discover and the 1 s sample script cut down to the limit inputs.

def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class RootShell:
    """One persistent root shell for the sysfs reads (no new su per sample). Android service calls do not go
    through it: they use root (DECISIONS #123)."""
    END = '__coresidency_end_'

    def __init__(self):
        self.p = subprocess.Popen(['su'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=DEVNULL, text=True,
                                  bufsize=1, start_new_session=True)
        self.lines, self.n = queue.Queue(), 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.lines.put(line.rstrip('\n'))
        self.lines.put(None)

    def run(self, script, timeout=5):
        """Output lines of script; raises RuntimeError on timeout or a dead shell."""
        self.n += 1
        end = f'{self.END}{self.n}__'
        try:
            self.p.stdin.write(f'{script}\necho {end}\n')
            self.p.stdin.flush()
        except OSError as e:
            raise RuntimeError(f'root shell: {e}') from e
        out, deadline = [], time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                raise RuntimeError(f'root shell: no answer in {timeout} s') from None
            if line is None:
                raise RuntimeError(f'root shell exited (rc {self.p.poll()})')
            if line == end:
                return out
            if line.startswith(self.END):  # the end of an earlier command that timed out: its output is not ours
                out = []
                continue
            out.append(line)

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()


def discover(shell):
    """The BIG/MID/LITTLE zone numbers (lowest zone of each type) and every cpufreq policy's cpuinfo_max_freq."""
    script = (f'for f in {THERMAL}/thermal_zone*/type {CPUFREQ}/policy*/cpuinfo_max_freq; do [ -e "$f" ] || continue; '
              "v=; read -r v 2>/dev/null <\"$f\"; printf '%s\\t%s\\n' \"$f\" \"$v\"; done")
    zones, policies = {}, {}
    for line in shell.run(script, timeout=30):
        path, _, v = line.partition('\t')
        if m := re.fullmatch(re.escape(THERMAL) + r'/thermal_zone(\d+)/type', path):
            zones[int(m.group(1))] = v
        elif m := re.fullmatch(re.escape(CPUFREQ) + r'/policy(\d+)/cpuinfo_max_freq', path):
            policies[int(m.group(1))] = to_int(v)
    cpu = {t: next((str(i) for i in sorted(zones) if zones[i] == t), None) for t in CPU_TYPES}
    bad = [f'policy{p}' for p, v in policies.items() if not v]
    if None in cpu.values() or not policies or bad:
        raise SystemExit(f'layout: CPU zones {cpu}, policies {sorted(policies)}, without cpuinfo_max_freq {bad}; '
                         'not started')
    return {'cpu_zones': cpu, 'policies': {f'policy{p}': policies[p] for p in sorted(policies)}}


def fast_keys(layout):
    return ([('cpu', t, f'{THERMAL}/thermal_zone{i}/temp') for t, i in layout['cpu_zones'].items()] +
            [('bat', 'temp', f'{BATTERY}/temp')] +
            [('max', p, f'{CPUFREQ}/{p}/scaling_max_freq') for p in layout['policies']])


def fast_sample(shell, keys):
    """One 1 s reading (shell builtins only; an unreadable file gives an empty line). t = monotonic s when in hand."""
    start = time.monotonic()
    try:
        lines = shell.run(f'for f in {" ".join(k[2] for k in keys)}; do v=; read -r v 2>/dev/null <"$f"; echo "$v"; done')
        if len(lines) != len(keys):
            raise RuntimeError(f'expected {len(keys)} values, got {len(lines)}: {lines[:5]!r}')
        err = None
    except RuntimeError as e:
        lines, err = [''] * len(keys), str(e)
    s = {'t': time.monotonic(), 't_start': start, 'cpu': {}, 'max': {}, **({'error': err} if err else {})}
    for (kind, name, _), v in zip(keys, lines):
        if kind == 'bat':
            s['bat_c'] = None if to_int(v) is None else to_int(v) / 10  # power_supply temp is in tenths of degC
        else:
            s[kind][name] = to_int(v)
    s['cpu_c'] = None if None in s['cpu'].values() else max(s['cpu'].values()) / 1000  # the fault stop's input
    return s


TEMP_RE = re.compile(r'Temperature\{mValue=([^,]+), mType=(-?\d+), mName=([^,]+), mStatus=(\d+)\}')


def parse_dump(text):
    """Android thermal status and the 'Current temperatures from HAL' section (name -> degC)."""
    status = re.search(r'^Thermal Status: (\d+)\s*$', text, re.M)
    sections = {}
    for m in re.finditer(r'^(\S[^\n]*):\n((?:[ \t]+[^\n]*\n?)*)', text, re.M):
        sections[m.group(1).strip()] = m.group(2)
    hal = {}
    for m in TEMP_RE.finditer(sections.get('Current temperatures from HAL', '')):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if not math.isnan(v):
            hal[m.group(3)] = v
    cached = sorted({m.group(3) for m in TEMP_RE.finditer(sections.get('Cached temperatures', ''))})
    return {'status': int(status.group(1)) if status else None, 'skin': hal.get(SKIN), 'hal': hal,
            'cached_names': cached}


def read_dump():
    """`dumpsys thermalservice` (DECISIONS #123 form): status and VIRTUAL-SKIN, None when absent; t = monotonic s
    when the reading was in hand (the dump ran t_start..t)."""
    start = time.monotonic()
    try:
        rc, text = root('dumpsys thermalservice', 'thermalservice', timeout=30)
        d = {'rc': rc, **{k: v for k, v in parse_dump(text).items() if k in ('status', 'skin')}}
    except (OSError, subprocess.SubprocessError) as e:
        d = {'rc': None, 'error': f'{type(e).__name__}: {e}', 'status': None, 'skin': None}
    return {'t': time.monotonic(), 't_start': start, **d}


def monitor_loop(period, read, rows, stop):
    k, began = 0, time.monotonic()
    while not stop.wait(max(0.0, began + k * period - time.monotonic())):
        rows.append(read())
        k = max(k + 1, math.ceil((time.monotonic() - began) / period))  # an overrun skips slots, never bunches


def block_limit(now, began, fast, dumps):
    """The block limit met at `now` as (limit, text), else None; times are monotonic s, began = monitor start.
    Each limit from its own readings (thermal_char.stop_reasons). Pure: tested offline."""
    status = next((d for d in reversed(dumps) if d['status'] is not None), None)
    skin = next((d for d in reversed(dumps) if d['skin'] is not None), None)
    if status and status['status'] >= STATUS_STOP:
        return 'android_status', f'Android thermal status {status["status"]} >= {STATUS_STOP} ({STATUS_NAMES[STATUS_STOP]})'
    bat = [s for s in fast[-SENSOR_WAIT_S * 4:] if s['bat_c'] is not None]
    if bat and bat[-1]['bat_c'] >= BATTERY_STOP_C:
        return 'battery', f'battery {bat[-1]["bat_c"]:.1f} >= {BATTERY_STOP_C} degC'
    cpu = [s for s in fast[-SENSOR_WAIT_S * 4:] if s['cpu_c'] is not None]
    if len(cpu) >= CPU_N and all(s['cpu_c'] >= CPU_FAULT_C for s in cpu[-CPU_N:]):
        return 'cpu_fault', (f'CPU zone >= {CPU_FAULT_C:.0f} degC in {CPU_N} consecutive 1 s samples '
                             f'({[s["cpu_c"] for s in cpu[-CPU_N:]]})')
    fresh = min(skin['t'] if skin else -math.inf, status['t'] if status else -math.inf)
    if now - max(fresh, began) > SKIN_WAIT_S:
        return 'fail_closed', f'no {SKIN} and Android status reading for {SKIN_WAIT_S} s'
    for name, rows in (('CPU zone', cpu), ('battery temperature', bat)):
        if now - max(rows[-1]['t'] if rows else -math.inf, began) > SENSOR_WAIT_S:
            return 'fail_closed', f'no {name} reading for {SENSOR_WAIT_S} s'
    return None


def frame_loop(detector, mode, session, t0, duration, reads, limit):
    """Reads every new frame and detects on it until the block ends or limit() returns a stop record.
    Returns (last frame number processed, stop record or None)."""
    policy = SizePolicy(LARGE_S, 1/3)  # main.LARGE_FRAME_INTERVAL_S, PERSON_HEIGHT_FRACTION; drive mode
    last = None
    while time.monotonic() < t0 + duration:
        # ponytail: checked between frames, so a stop comes up to one detection (~1.5 s) plus the 1 s / 5 s
        # reading interval after the sensor crossed; a watcher thread could cut the frame part
        if (hit := limit()) is not None:
            return last, hit
        check_cores('during the block')
        _Stamp.at = None
        a = time.perf_counter()
        r = read_frame(FRAME_DIR, session=session)
        b = time.perf_counter()
        row = {'t': round(time.monotonic() - t0, 3), 'status': r['status']}
        if _Stamp.at is not None:
            row['read_ms'] = round((_Stamp.at - a) * 1000, 2)
            row['decode_ms'] = round((b - _Stamp.at) * 1000, 2)
        if r['status'] == 'ok':
            row.update(frame=r['frame'], age_s=round(r['age_s'], 3))
            if last is not None and r['frame'] <= last:
                row['status'] = 'repeat'
        reads.append(row)
        if row['status'] != 'ok':
            time.sleep(0.02 if row['status'] == 'repeat' else 0.1)
            continue
        last = r['frame']
        size = policy.next_size() if mode == 'mix' else 640
        c = time.perf_counter()
        detections = detector.detect(r['image'], size)
        row.update(size=size, detect_ms=round((time.perf_counter() - c) * 1000, 2), n_detections=len(detections))
        # rate 2: the next frame appears about 0.5 s after this one was first seen (a repeat retries in 20 ms)
        time.sleep(max(0.0, a + 0.5 - time.perf_counter()))
    return last, None


# ---------------------------------------------------------------- blocks

def latest(fast, dumps):
    """The newest value of each limit input (the readings recorded with a stop)."""
    pick = lambda rows, k: next((r[k] for r in reversed(rows) if r.get(k) is not None), None)
    return {'skin': pick(dumps, 'skin'), 'status': pick(dumps, 'status'), 'bat_c': pick(fast, 'bat_c'),
            'cpu_c': pick(fast, 'cpu_c'), 'scaling_max': next((r['max'] for r in reversed(fast)), None)}


def run_block(name, mode, gemma, ctx):
    """One block; raises CoresLost if cores 4-7 go away (the caller discards it and redoes it)."""
    check_cores('before ' + name)
    therm_start = thermal_gate(ctx['thermal_log'], ctx['idle'], name, ctx['smoke'])
    print(f'[{name}] start: skin {therm_start["skin"]} degC, z9 {therm_start["z9"] / 1000:.1f} degC (waited '
          f'{therm_start["waited_s"]} s{", WARM START" if therm_start["warm_start"] else ""})', flush=True)
    server = ctx['server'] if gemma else None
    if gemma and not server.alive():
        raise RuntimeError('llama-server is not healthy at block start')
    stop_mon, fast, dumps, keys = threading.Event(), [], [], fast_keys(ctx['layout'])
    monitors = [threading.Thread(target=monitor_loop, args=(FAST_S, lambda: fast_sample(ctx['shell'], keys), fast,
                                                            stop_mon), daemon=True),
                threading.Thread(target=monitor_loop, args=(DUMP_S, read_dump, dumps, stop_mon), daemon=True)]
    for t in monitors:
        t.start()
    began = time.monotonic()
    threads = []
    try:
        while not (fast and dumps) and time.monotonic() - began < 10:  # first readings in before the first frame
            time.sleep(0.05)
        lmk_since = time.time()
        cam = camera_start()
        t0 = time.monotonic()
        duration = ctx['duration']

        def limit():
            now = time.monotonic()
            hit = block_limit(now, began, fast, dumps)
            return hit and {'limit': hit[0], 'reason': hit[1], 'time_to_limit_s': round(now - t0, 2),
                            'reading': latest(fast, dumps)}
        stop, samples, calls, reads = threading.Event(), [], [], []
        threads = [threading.Thread(target=sampler_loop, args=(t0, ctx['thermal_log'], server, stop, samples),
                                    daemon=True)]
        if gemma:
            threads.append(threading.Thread(target=selector_loop, args=(ctx['selector'], ctx['cases'], t0, duration,
                                                                        stop, calls), daemon=True))
        for t in threads:
            t.start()
        try:
            last, hit = frame_loop(ctx['detector'], mode, cam['session'], t0, duration, reads, limit)
            elapsed = time.monotonic() - t0
            stop.set()  # a limit stop ends the selector's cadence here too (no slot starts after the block)
            deadline = time.monotonic() + 2  # survival: a newer frame within 2 s (rate 2)
            while True:
                end = read_frame(FRAME_DIR, session=cam['session'])
                if (end['status'] == 'ok' and last is not None and end['frame'] > last) or time.monotonic() > deadline:
                    break
                time.sleep(0.1)
            robotcam_pid = robotcam_pids()
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=120)
            try:
                camera_stop()
            finally:  # also on CoresLost/abort or a failed broadcast: the next block or run starts from a stopped app
                cam_end = camera_end_check()
    finally:
        stop_mon.set()
        for t in monitors:
            t.join(timeout=60)
    if any(t.is_alive() for t in threads + monitors):  # e.g. a stalled selector request: it would overlap the next block
        raise RuntimeError(f'{name}: a sampler/selector/monitor thread still running after the block; run stopped')
    check_cores('after ' + name)
    therm_end = read_thermal(ctx['thermal_log'])
    for r in fast + dumps:  # monotonic -> s from the block start (readings before the camera start are negative)
        r['t'], r['t_start'] = round(r['t'] - t0, 3), round(r['t_start'] - t0, 3)
    return {'block': name, 'mode': mode, 'gemma': gemma, 'duration_s': round(elapsed, 2), 'planned_s': duration,
            'heat_stop': {'reached_limit': bool(hit) and hit['limit'] != 'fail_closed',
                          **(hit or {'limit': None, 'reason': None, 'time_to_limit_s': None, 'reading': None})},
            'camera': cam, 'camera_end': cam_end, 'thermal_start': therm_start, 'thermal_end': therm_end,
            'cpuinfo_max_khz': ctx['layout']['policies'], 'fast': fast, 'dumps': dumps,
            'survived': {'robotcam_process': bool(robotcam_pid), 'robotcam_pids': robotcam_pid,
                         'robotcam_new_frame_at_end': end['status'] == 'ok' and last is not None and end['frame'] > last,
                         'robotcam_end_status': end['status'],
                         'llama_server': server.alive() if server else None},
            'lmk': lmk_lines(lmk_since), 'samples': samples, 'selector_calls': calls, 'reads': reads}


def ensure_server(ctx, out):
    """Resident Gemma for B3/B4: reuses the running server, else starts one; one untimed warm-up call."""
    if ctx['server'] is not None and ctx['server'].alive():
        return None
    if ctx['server'] is not None:
        ctx['server'].stop()
    ctx['server'] = Server(out / 'llama-server.log')
    ctx['selector'] = make_selector()
    ctx['selector'].decide(*WARMUP)
    return ctx['server'].load_s


def loads(ctx, out):
    """Cold then warm load; raises CoresLost if cores 4-7 go away around either (the caller redoes both)."""
    therm = thermal_gate(ctx['thermal_log'], ctx['idle'], 'loads', ctx['smoke'])
    check_cores('before the cold load')
    cold_state = make_cold()
    s = Server(out / 'llama-server.log')
    cold_s = s.load_s
    s.stop()
    check_cores('after the cold load')
    s = Server(out / 'llama-server.log')  # page cache now warm from the cold load
    ctx['server'] = s
    check_cores('after the warm load')
    ctx['selector'] = make_selector()
    ctx['selector'].decide(*WARMUP)
    return {'cold_load_s': cold_s, 'cold_state': cold_state, 'warm_load_s': s.load_s,
            'load_definition': 'spawn to first /health status ok (0.05 s poll)', 'thermal_start': therm,
            'cmd': s.cmd}


def mtp_snapshot(ctx, out):
    """Not timed: Gemma with the MTP drafter beside RobotCam and both detector sessions; one short request."""
    cam = camera_start()
    server = None
    try:
        server = Server(out / 'llama-server-mtp.log', MTP_ARGS)
        res = post('/completion', {'prompt': MTP_PROMPT, 'n_predict': 32, 'temperature': 0, 'cache_prompt': False})
        deadline = time.monotonic() + 3
        while (r := read_frame(FRAME_DIR, session=cam['session']))['status'] != 'ok' and time.monotonic() < deadline:
            time.sleep(0.1)
        if r['status'] != 'ok':
            raise RuntimeError(f'B5: no usable RobotCam frame for the 640 detection ({r["status"]})')
        ctx['detector'].detect(r['image'], 640)
        snap = sample(time.monotonic(), ctx['thermal_log'], server)
        rec = {'block': 'B5_mtp_ram_snapshot', 'cmd': server.cmd, 'load_s': server.load_s,
               'reply': res.get('content'), 'timings': res.get('timings'), 'detected_640_on_frame': r['frame'],
               'server_status_kb': server.status_kb(), 'sample': snap}
    finally:
        if server:
            server.stop()
        try:
            camera_stop()
        finally:  # also when the broadcast failed: the force-stop is the fallback
            cam_end = camera_end_check()
    return {**rec, 'camera_end': cam_end}


# ---------------------------------------------------------------- report

def med(xs):
    return statistics.median(xs) if xs else None


def p95(xs):
    s = sorted(xs)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)] if s else None


def f(v, spec='.0f'):
    return 'n/a' if v is None else format(v, spec)


def block_summary(b):
    done = [r for r in b['reads'] if 'detect_ms' in r]
    failed = {}
    for r in b['reads']:
        if r['status'] not in ('ok', 'repeat'):
            failed[r['status']] = failed.get(r['status'], 0) + 1
    last_t = b['duration_s']
    s = {'frames': len(done), 'failed': failed, 'repeats': sum(r['status'] == 'repeat' for r in b['reads'])}
    # a heat stop before both drift windows fit: drift is n/a, not missing
    s['drift_na'] = b['heat_stop']['reached_limit'] and last_t < 2 * DRIFT_S
    for size in (320, 640):
        ms = [r['detect_ms'] for r in done if r['size'] == size]
        s[f'n{size}'] = len(ms)
        s[f'det{size}'] = (med(ms), p95(ms))
        s[f'drift{size}'] = (None, None) if s['drift_na'] else (
            med([r['detect_ms'] for r in done if r['size'] == size and r['t'] < DRIFT_S]),
            med([r['detect_ms'] for r in done if r['size'] == size and r['t'] >= last_t - DRIFT_S]))
    s['read'] = med([r['read_ms'] for r in done])
    s['decode'] = med([r['decode_ms'] for r in done])
    s['age'] = med([r['age_s'] for r in done])
    ok_calls = [c for c in b['selector_calls'] if 'ms' in c]
    s['sel'] = (len(ok_calls), len(b['selector_calls']) - len(ok_calls), sum(c['correct'] for c in ok_calls),
                med([c['ms'] for c in ok_calls]), p95([c['ms'] for c in ok_calls]))
    sm = [x for x in b['samples'] if x['t_end'] <= b['duration_s']]  # finished after the block: not counted
    s['late_samples'] = len(b['samples']) - len(sm)
    s['min_avail'] = min((x['mem_available_mib'] for x in sm), default=None)
    s['max_swap'] = max((x['swap_used_mib'] for x in sm), default=None)
    for name in ('runner', 'llama_server', 'robotcam_app', 'camera_provider'):
        s['pss_' + name] = max((x['pss_kb'][name] for x in sm if name in x.get('pss_kb', {})), default=None)
    z9 = [b['thermal_start']['z9'], b['thermal_end']['z9']] + [x['z9'] for x in sm if 'z9' in x]
    s['z9'] = (b['thermal_start']['z9'] / 1000, b['thermal_end']['z9'] / 1000, max(z9) / 1000)
    # limit readings up to the block end (those taken before the camera start included: the start values)
    skin = [d['skin'] for d in b['dumps'] if d['skin'] is not None and d['t'] <= b['duration_s']]
    s['skin'] = (skin[0], skin[-1], max(skin)) if skin else (None, None, None)
    s['status_max'] = max((d['status'] for d in b['dumps'] if d['status'] is not None and d['t'] <= b['duration_s']),
                          default=None)
    # capped time: each in-block 1 s sample stands for the time since the previous one (a skipped slot is counted);
    # a sample with an unreadable scaling_max_freq is no evidence either way: counted apart, and INCOMPLETE
    fast = [x for x in b['fast'] if 0 <= x['t'] <= b['duration_s']]
    top = b['cpuinfo_max_khz']
    capped = unread = 0.0
    for prev, x in zip([0.0] + [x['t'] for x in fast], fast):
        if any(x['max'].get(p) is None for p in top):
            unread += x['t'] - prev
        elif any(x['max'][p] < top[p] for p in top):
            capped += x['t'] - prev
    s['capped'] = (capped, 100 * capped / b['duration_s'] if b['duration_s'] else None)
    s['max_unread'] = (sum(any(x['max'].get(p) is None for p in top) for x in fast), unread)
    s['low_max'] = [min((x['max'][p] for x in fast if x['max'].get(p) is not None), default=None) for p in top]
    w = [x['battery_w'] for x in sm if 'battery_w' in x]
    s['w'] = statistics.mean(w) if w else None
    s['bat_status'] = sorted({x['battery_status'] for x in sm if 'battery_status' in x})
    s['sample_errors'] = sum(any(k in x for k in ('error', 'thermal_error', 'pss_error')) for x in sm)
    # intervals between sample completions (a slow su call skips slots); a sample holds values from t..t_end
    edges = [0.0] + [x['t_end'] for x in sm] + [b['duration_s']]
    gaps = [q - p for p, q in zip(edges, edges[1:])]
    s['gaps'] = (sum(g > 1.5 * SAMPLE_S for g in gaps), max(gaps))
    return s


def problems(blocks, sums):
    """Why the run cannot count as a complete measurement (blocks are still kept: a kill is evidence)."""
    out = []
    for b, s in zip(blocks, sums):
        heat = b['heat_stop']
        if why := camera_end_failed(b['camera_end']):
            out.append(f'{b["block"]}: RobotCam end check: {why}')
        if heat['reached_limit'] and s['frames'] == 0 and not s['failed']:  # met before the first frame
            r = heat['reading']
            out.append(f'{b["block"]}: not run: limit at start ({heat["reason"]}; skin {r["skin"]}, status '
                       f'{r["status"]}, battery {r["bat_c"]}, CPU max {r["cpu_c"]} degC, scaling_max {r["scaling_max"]})')
            continue
        if heat['limit'] == 'fail_closed':
            out.append(f'{b["block"]}: stopped fail-closed at {heat["time_to_limit_s"]} s: {heat["reason"]}')
        # a limit stop is a result: rules that need time only apply as far as the block ran
        # (no_sample: stopped before a second sample was due, so the first may finish after it)
        no_sample = heat['reached_limit'] and b['duration_s'] < SAMPLE_S
        if b['gemma']:  # the Gemma workload is one call per 20 s slot, each inside the block
            calls = b['selector_calls']
            # slots that can start inside the block: k * SELECTOR_S < planned length (selector_loop's bound; the
            # block itself runs a little past it), or before the limit stop, which also stops the selector; a slot
            # due in the last second before a limit stop is not required (the stop may beat the thread to it)
            end = heat['time_to_limit_s'] - 1 if heat['reached_limit'] else b['planned_s']
            slots = max(0, math.ceil(end / SELECTOR_S))
            late = [c['t'] for c in calls if c['started_s'] - c['t'] > 5]
            # a call in flight at a limit stop ends after it by construction: not an overrun
            over = [] if heat['reached_limit'] else [c['t'] for c in calls if c['ended_s'] > b['duration_s']]
            if slots and s['sel'][0] == 0:
                out.append(f'{b["block"]}: no successful selector call ({s["sel"][1]} failed)')
            elif s['sel'][0] < slots or late or over:
                out.append(f'{b["block"]}: selector cadence missed: {s["sel"][0]}/{slots} successful calls, '
                           f'started >5 s late at slots {late}, ended after the block at slots {over}')
        if s['max_unread'][0]:
            out.append(f'{b["block"]}: {s["max_unread"][0]} 1 s sample(s) without every scaling_max_freq '
                       f'({s["max_unread"][1]:.1f} s of the block not known capped or not)')
        if s['sample_errors'] or (s['min_avail'] is None and not no_sample):
            out.append(f'{b["block"]}: {s["sample_errors"]} sample(s) with root/PSS/thermal errors, '
                       f'{0 if s["min_avail"] is None else "some"} usable samples')
        sv = b['survived']
        if s['failed'] or s['frames'] == 0 or \
                not (sv['robotcam_new_frame_at_end'] and sv['robotcam_process']):
            out.append(f'{b["block"]}: RobotCam: {s["frames"]} frames, failed reads {s["failed"] or "none"}, '
                       f'new frame at end {sv["robotcam_new_frame_at_end"]}, process at end {sv["robotcam_process"]}')
        for size in (320, 640) if b['mode'] == 'mix' else (640,):
            if size == 640 and b['mode'] == 'mix' and heat['reached_limit'] and heat['time_to_limit_s'] < LARGE_S:
                continue  # stopped before this size's first frame was due
            if s[f'n{size}'] == 0 or (None in s[f'drift{size}'] and not s['drift_na']):
                out.append(f'{b["block"]}: {s[f"n{size}"]} detections at {size}, drift first/last {DRIFT_S} s '
                           f'{s[f"drift{size}"]} (a required measurement is missing)')
        if b['gemma'] and b['survived']['llama_server'] is not True:
            out.append(f'{b["block"]}: llama-server did not survive the block (see LMK lines)')
        if s['bat_status'] != ['Discharging'] and not (no_sample and s['bat_status'] == []):
            out.append(f'{b["block"]}: battery status {s["bat_status"]} (power needs Discharging throughout)')
        if not b['lmk']['ok']:
            out.append(f'{b["block"]}: LMK logcat query failed (rc {b["lmk"]["logcat_rc"]}): {b["lmk"]["raw_head"][:120]!r}')
    return out


def report(out):
    """Returns (report text, problems)."""
    blocks = [json.loads((out / f'block_{n}.json').read_text()) for n, _, _ in BLOCKS if (out / f'block_{n}.json').exists()]
    sums = [block_summary(b) for b in blocks]
    bad = problems(blocks, sums)
    mib = lambda kb: None if kb is None else kb / 1024
    pols = '/'.join(p[6:] for p in blocks[0]['cpuinfo_max_khz']) if blocks else ''
    rows = [
        ('frames processed', lambda s: str(s['frames'])),
        ('failed reads by status', lambda s: ', '.join(f'{k} {v}' for k, v in s['failed'].items()) or 'none'),
        ('repeat reads (not failures)', lambda s: str(s['repeats'])),
        ('detect320 ms median/P95', lambda s: f'{f(s["det320"][0])}/{f(s["det320"][1])} (n {s["n320"]})'),
        ('detect640 ms median/P95', lambda s: f'{f(s["det640"][0])}/{f(s["det640"][1])} (n {s["n640"]})'),
        (f'drift320 first/last {DRIFT_S}s', lambda s: f'{f(s["drift320"][0])} -> {f(s["drift320"][1])}'),
        (f'drift640 first/last {DRIFT_S}s', lambda s: f'{f(s["drift640"][0])} -> {f(s["drift640"][1])}'),
        ('read/decode ms median', lambda s: f'{f(s["read"], ".1f")}/{f(s["decode"], ".1f")}'),
        ('frame age s median', lambda s: f(s['age'], '.2f')),
        ('selector ms median/P95', lambda s: f'{f(s["sel"][3])}/{f(s["sel"][4])}' if s['sel'][0] else '-'),
        ('selector calls ok/err/correct', lambda s: f'{s["sel"][0]}/{s["sel"][1]}/{s["sel"][2]}'),
        ('min MemAvailable MiB', lambda s: f(s['min_avail'])),
        ('max swap used MiB', lambda s: f(s['max_swap'])),
        ('peak PSS runner MiB', lambda s: f(mib(s['pss_runner']))),
        ('peak PSS llama-server MiB', lambda s: f(mib(s['pss_llama_server']))),
        ('peak PSS RobotCam app MiB', lambda s: f(mib(s['pss_robotcam_app']))),
        ('peak PSS camera provider MiB', lambda s: f(mib(s['pss_camera_provider']))),
        ('LMK log lines (kill lines)', None),
        ('survived RobotCam / llama', None),
        ('zone9 start/end/max degC', lambda s: '/'.join(f'{v:.1f}' for v in s['z9'])),
        ('VIRTUAL-SKIN start/end/max degC', lambda s: '/'.join(f(v, '.1f') for v in s['skin'])),
        ('Android status max', lambda s: f(s['status_max'], 'd')),
        ('policy capped s (% of block)', lambda s: f'{s["capped"][0]:.0f} ({f(s["capped"][1])}%)'),
        (f'lowest scaling_max MHz {pols}', lambda s: '/'.join(f(v and v / 1000) for v in s['low_max'])),
        ('gate wait s', None),
        ('stop limit', None),
        ('time to limit s', None),
        ('mean battery W', lambda s: f(s['w'], '.2f')),
        ('battery status', lambda s: ','.join(s['bat_status']) or 'n/a'),
        ('sample errors', lambda s: str(s['sample_errors'])),
        (f'sample gaps >{1.5 * SAMPLE_S:g}s (max s)', lambda s: f'{s["gaps"][0]} ({s["gaps"][1]:.1f})'),
        ('samples past block end (dropped)', lambda s: str(s['late_samples'])),
    ]
    extra = {'LMK log lines (kill lines)': lambda b: (f'{b["lmk"]["n_lines"]} ({b["lmk"]["n_kills"]})' if b['lmk']['ok']
                                                      else f'QUERY FAILED rc {b["lmk"]["logcat_rc"]}'),
             'survived RobotCam / llama': lambda b: (f'{"yes" if b["survived"]["robotcam_new_frame_at_end"] and b["survived"]["robotcam_process"] else "NO"} / '
                                                     f'{ {True: "yes", False: "NO", None: "-"}[b["survived"]["llama_server"]]}'),
             'gate wait s': lambda b: f'{b["thermal_start"]["waited_s"]}' + (' warm start' if b['thermal_start']['warm_start']
                                                                              else ''),
             'stop limit': lambda b: b['heat_stop']['limit'] or '-',
             'time to limit s': lambda b: (f'{b["heat_stop"]["time_to_limit_s"]:.1f}' if b['heat_stop']['reached_limit']
                                           else '-')}
    run = json.loads((out / 'run.json').read_text())
    lines = [f'Co-residency benchmark (DECISIONS #124), run {out.name}{"  [SMOKE: not a measurement]" if run["smoke"] else ""}',
             f'block length {run["block_s"]} s; idle z9 {run["idle"]["z9"] / 1000:.1f} degC, idle skin '
             f'{run["idle"]["skin"]:.1f} degC; runner sha256 {run["sha256"]["coresidency.py"][:12]}']
    lines += [f'INCOMPLETE: {p}' for p in bad] + ['']
    width = 32
    lines.append(f'{"":{width}}' + ''.join(f'{b["block"]:>20}' for b in blocks))
    for label, fn in rows:
        vals = [extra[label](b) if label in extra else fn(s) for b, s in zip(blocks, sums)]
        lines.append(f'{label:{width}}' + ''.join(f'{v:>20}' for v in vals))
    if (out / 'loads.json').exists():
        ld = json.loads((out / 'loads.json').read_text())
        lines += ['', f'Gemma load (spawn to /health ok): cold {ld["cold_load_s"]:.2f} s [{ld["cold_state"]["mode"]}], '
                      f'warm {ld["warm_load_s"]:.2f} s; gate wait {ld["thermal_start"]["waited_s"]} s'
                      f'{" (warm start)" if ld["thermal_start"]["warm_start"] else ""}', 'server command: ' + ' '.join(ld['cmd'])]
    if (out / 'block_B5_mtp_ram_snapshot.json').exists():
        b5 = json.loads((out / 'block_B5_mtp_ram_snapshot.json').read_text())
        sm = b5['sample']
        errors = {k: sm[k] for k in ('error', 'thermal_error', 'pss_error') if k in sm}
        if errors:
            bad.append(f'B5_mtp_ram_snapshot: sample errors {errors}')
            lines.insert(2, f'INCOMPLETE: {bad[-1]}')
        if why := camera_end_failed(b5['camera_end']):
            bad.append(f'B5_mtp_ram_snapshot: RobotCam end check: {why}')
            lines.insert(2, f'INCOMPLETE: {bad[-1]}')
        lines += ['', f'MTP snapshot (RobotCam + both detector sessions + Gemma with drafter, after one request): '
                      f'MemAvailable {sm["mem_available_mib"]} MiB, swap used {sm["swap_used_mib"]} MiB, '
                      f'PSS llama-server {f(mib(sm.get("pss_kb", {}).get("llama_server")))} MiB, '
                      f'VmHWM {f(mib(b5["server_status_kb"].get("VmHWM")))} MiB, load {b5["load_s"]:.2f} s',
                  'server command: ' + ' '.join(b5['cmd'])]
    lines += ['', 'Notes: detect ms = Detector.detect (resize, inference, NMS); read/decode split read_frame at its '
                  'BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from '
                  'battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines '
                  'matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", '
                  'or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Block limits, checked between '
                  f'frames: Android thermal status >= {STATUS_STOP} ({STATUS_NAMES[STATUS_STOP]}), battery >= '
                  f'{BATTERY_STOP_C} degC, CPU zone >= {CPU_FAULT_C:.0f} degC in {CPU_N} consecutive 1 s samples (fault '
                  'stop); time to limit = when the runner saw it (up to one detection plus the 1 s/5 s reading '
                  'interval late); the block ends there and counts as run; drift is n/a if it ran under '
                  f'{2 * DRIFT_S} s. Skin/status = `dumpsys thermalservice` every {DUMP_S} s; capped = time (each in-block 1 s sample '
                  'counts the time since the previous one) with any policy\'s scaling_max_freq below its '
                  'cpuinfo_max_freq, % of the block length. Gate: skin <= idle + '
                  f'{SKIN_GATE_C} and z9 <= idle + {GATE_MC / 1000:.0f} degC; warm start = not reached in '
                  f'{GATE_MAX_S // 60} min.']
    return '\n'.join(lines), bad


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--smoke', action='store_true', help=f'{SMOKE_S} s blocks, no cooldown gate (functional check)')
    ap.add_argument('--resume', type=Path, help='continue RUN_DIR: completed blocks are kept')
    ap.add_argument('--thermal-log', default=str(OUT_ROOT / 'thermal.log'))
    a = ap.parse_args(argv)
    require_native()
    signal.signal(signal.SIGTERM, exit_on_signal)
    signal.signal(signal.SIGHUP, exit_on_signal)
    if a.resume:
        out = a.resume
        run = json.loads((out / 'run.json').read_text())
        if run['smoke'] != a.smoke:
            raise SystemExit(f'--resume: {out} was a {"smoke" if run["smoke"] else "full"} run')
        if 'block_stops' not in run:  # its blocks lack the limit readings and were stopped on zone9
            raise SystemExit(f'--resume: {out} was made by an older runner (no block limits); start a new run')
    else:
        out = OUT_ROOT / f'run_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}{"_smoke" if a.smoke else ""}'
        out.mkdir(parents=True)
    ctx = {'smoke': a.smoke, 'duration': SMOKE_S if a.smoke else BLOCK_S, 'thermal_log': a.thermal_log,
           'shell': None, 'server': None, 'selector': None, 'cases': load_cases()}
    try:
        rc, who = root('id', 'id')
        if rc != 0:
            raise SystemExit(f'su failed: {who.strip()}')
        wait_cores('start')
        ctx['shell'] = RootShell()
        ctx['layout'] = discover(ctx['shell'])
        ctx['idle'] = {**read_thermal(a.thermal_log), 'skin': read_dump()['skin']}  # after the launcher's idle
        if ctx['idle']['skin'] is None:
            raise SystemExit(f'no {SKIN} reading from dumpsys thermalservice at idle; not started')
        print(f'thermal idle reading: z9 {ctx["idle"]["z9"] / 1000:.1f} degC, skin {ctx["idle"]["skin"]:.1f} degC; '
              f'output {out}', flush=True)
        files = {'coresidency.py': __file__, 'robotcam_reader.py': robotcam_reader.__file__,
                 'detect_person.py': ROBOT / 'detect_person.py', 'detector_size_policy.py': ROBOT / 'detector_size_policy.py',
                 'server_manager.py': server_manager.__file__, 'main.py': ROBOT / 'main.py',
                 'adapters.py': V3 / 'adapters.py',
                 's1/schema.py': S1O_SRC / 's1/schema.py', 'cases': CASES,
                 'yolo11s_320.onnx': ROBOT / 'yolo11s_320.onnx', 'yolo11s_640.onnx': ROBOT / 'yolo11s_640.onnx'}
        meta = {'started': utc(), 'smoke': a.smoke, 'block_s': ctx['duration'], 'idle': ctx['idle'],
                'sha256': {k: sha256(p) for k, p in files.items()},
                'model_bytes': {p: os.path.getsize(p) for p in (MODEL, DRAFT)},
                'layout': ctx['layout'],
                'block_stops': {'status': STATUS_STOP, 'battery_c': BATTERY_STOP_C, 'cpu_fault_c': CPU_FAULT_C,
                                'cpu_consecutive': CPU_N, 'skin_wait_s': SKIN_WAIT_S, 'sensor_wait_s': SENSOR_WAIT_S,
                                'gate_skin_c': SKIN_GATE_C, 'gate_z9_mc': GATE_MC, 'gate_max_s': GATE_MAX_S},
                'server_cmd': server_cmd(), 'mtp_cmd': server_cmd(MTP_ARGS), 'python': sys.version,
                'cpus_allowed': sorted(allowed_cpus())}
        if a.resume:
            (out / f'run.resume_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json').write_text(json.dumps(meta, indent=1))
        else:
            (out / 'run.json').write_text(json.dumps(meta, indent=1))
        ctx['detector'] = Detector()  # both sessions resident, as on the robot
        camera_stop()  # start from a stopped service
        for name, mode, gemma in BLOCKS:
            path = out / f'block_{name}.json'
            if path.exists():
                print(f'[{name}] already completed, skipped (--resume)', flush=True)
                continue
            while gemma and not (out / 'loads.json').exists():
                wait_cores('loads')
                try:
                    (out / 'loads.json').write_text(json.dumps(loads(ctx, out), indent=1))
                except CoresLost as e:
                    print(f'  [loads] {e}: discarding both loads and redoing them', flush=True)
                    if ctx['server'] is not None:
                        ctx['server'].stop()
                        ctx['server'] = None
                    continue
                ld = json.loads((out / 'loads.json').read_text())
                print(f'[loads] cold {ld["cold_load_s"]:.2f} s ({ld["cold_state"]["mode"]}), warm {ld["warm_load_s"]:.2f} s',
                      flush=True)
            while True:
                wait_cores(name)
                started = ensure_server(ctx, out) if gemma else None
                try:
                    b = run_block(name, mode, gemma, ctx)
                    break
                except CoresLost as e:
                    print(f'  [{name}] {e}: discarding this block and redoing it', flush=True)
            b['server_started_for_block_load_s'] = started
            path.write_text(json.dumps(b) + '\n')
            s = block_summary(b)
            print(f'[{name}] done: {s["frames"]} frames, failed {s["failed"]}, sel {s["sel"][0]} calls, '
                  f'z9 {s["z9"][0]:.1f}->{s["z9"][1]:.1f}, {f(s["w"], ".2f")} W', flush=True)
        if ctx['server'] is not None:
            ctx['server'].stop()
            ctx['server'] = None
        b5 = out / 'block_B5_mtp_ram_snapshot.json'
        if not b5.exists():
            b5.write_text(json.dumps(mtp_snapshot(ctx, out), indent=1) + '\n')
        text, bad = report(out)
        (out / 'report.txt').write_text(text + '\n')
        print(text, flush=True)
        if DOWNLOADS.is_dir():
            shutil.copy(out / 'report.txt', DOWNLOADS / f'coresidency_{out.name}_report.txt')
            print(f'copied to {DOWNLOADS / f"coresidency_{out.name}_report.txt"}', flush=True)
        else:
            print(f'WARNING: {DOWNLOADS} missing; report only in {out}', flush=True)
        if bad:
            raise SystemExit('RUN INCOMPLETE: ' + '; '.join(bad))
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        for server in list(LIVE):  # ctx['server'] and any server a signal caught between its start and its owner
            server.stop()
        try:
            camera_stop()
        except (OSError, subprocess.SubprocessError) as e:
            print(f'[CAM] stop failed: {e}', file=sys.stderr)
        if why := camera_end_failed(camera_end_check()):  # cleanup after an abort (each block and B5 check their own)
            print(f'[CAM] end check: {why}', file=sys.stderr)
        if ctx.get('shell'):
            ctx['shell'].close()


if __name__ == '__main__':
    main()
