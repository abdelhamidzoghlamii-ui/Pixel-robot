#!/data/data/com.termux/files/usr/bin/python
"""Co-residency benchmark (DECISIONS #124). Research only, motors off: never imports motors.py or
opens USB/serial. Native Termux only (the robot runs ONNX Runtime and llama-server natively);
started by oneshot.sh in this folder, which runs the thermal logger and answers cache-drop requests.

Blocks, in order (180 s each; --smoke: 20 s, no cooldown gate):
  B1_mix_nollm      RobotCam mode B rate 2 -> read_frame (fixed session, frame must advance) ->
                    Detector at SizePolicy drive mode sizes (320 each frame, 640 every 5 s)
  B2_only640_nollm  same, always 640
  (loads)           Gemma cold load after a page-cache drop (handshake), then warm load by restart
  B3_mix_gemma      B1 + resident llama-server (server_manager setup_q4 flags) + a selector
                    letter-scoring call every 20 s in its own thread (s1o, variant s1o_b1609dp_q40 prompts)
  B4_only640_gemma  B2 + the same
  B5_mtp_ram_snapshot  (not timed) Gemma with the conv_mtp drafter flags, one short request, RAM snapshot
B1-B4 end early at the robot's live thermal pause (zone and threshold read from main.py): a result, not a failure.

Every 5 s: MemAvailable/swap, PSS (root dumpsys meminfo) of the runner, llama-server, RobotCam app and
camera provider, zone9/10/11 from the thermal log, battery power (camera_heat.py's method). Per block:
LMK log lines, RobotCam/llama-server survival. Per read: status, read/decode/detect ms, size, frame age.

Usage: coresidency.py [--smoke] [--resume RUN_DIR] [--thermal-log PATH]
"""
import argparse
import hashlib
import io
import json
import math
import os
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
# Heat stop: the robot's live thermal pause, main.Robot.run_mission `if get_temp() > 80:` with get_temp() reading
# thermal_zone9 in whole degC (millidegC // 1000). Both values are read from main.py, never restated here.
_MAIN = (ROBOT / 'main.py').read_text()
_zone = re.search(r'def get_temp\(\):\n.*?thermal_zone(\d+)/temp', _MAIN, re.S)
_limit = re.search(r'if get_temp\(\) > (\d+):', _MAIN)
if not (_zone and _limit) or _zone.group(1) not in ('9', '10', '11'):
    raise SystemExit('main.py: the live thermal pause (get_temp zone, get_temp() > N) was not found')
PAUSE_ZONE, PAUSE_ABOVE_C = f'z{_zone.group(1)}', int(_limit.group(1))
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
    """ladder.thermal_gate: wait until z9 <= idle + 4 degC; --smoke skips the wait."""
    began = time.monotonic()
    while not smoke and (t := read_thermal(path))['z9'] > idle['z9'] + GATE_MC:
        print(f'  [{label}] waiting: z9 {t["z9"] / 1000:.1f} degC, need <= {(idle["z9"] + GATE_MC) / 1000:.1f}, '
              f'{time.monotonic() - began:.0f} s', flush=True)
        time.sleep(10)
    t = read_thermal(path)
    t['waited_s'] = round(time.monotonic() - began)
    return t


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


def heat_reached(thermal_log):
    """The live pause test on the newest thermal-log line: the reading if it is reached, else None. A line that
    cannot be parsed is retried next frame (the sampler records it as a thermal error); a stale log still stops."""
    try:
        t = read_thermal(thermal_log)
    except (OSError, ValueError, IndexError, KeyError):
        return None
    return t if t[PAUSE_ZONE] // 1000 > PAUSE_ABOVE_C else None


def frame_loop(detector, mode, session, t0, duration, reads, thermal_log):
    """Reads every new frame and detects on it until the block ends or the heat stop is reached.
    Returns (last frame number processed, heat stop record or None)."""
    policy = SizePolicy(LARGE_S, 1/3)  # main.LARGE_FRAME_INTERVAL_S, PERSON_HEIGHT_FRACTION; drive mode
    last = None
    while time.monotonic() < t0 + duration:
        # ponytail: checked between frames on the 5 s thermal log, so the stop (time_to_limit_s = when the runner
        # saw it) can come up to ~5 s plus one detection after the zone crossed; a live su read would cut that
        if (hot := heat_reached(thermal_log)) is not None:
            return last, {'time_to_limit_s': round(time.monotonic() - t0, 2), 'zone_reading': hot}
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

def run_block(name, mode, gemma, ctx):
    """One block; raises CoresLost if cores 4-7 go away (the caller discards it and redoes it)."""
    check_cores('before ' + name)
    therm_start = thermal_gate(ctx['thermal_log'], ctx['idle'], name, ctx['smoke'])
    print(f'[{name}] start: z9 {therm_start["z9"] / 1000:.1f} degC (waited {therm_start["waited_s"]} s)', flush=True)
    server = ctx['server'] if gemma else None
    if gemma and not server.alive():
        raise RuntimeError('llama-server is not healthy at block start')
    lmk_since = time.time()
    cam = camera_start()
    t0 = time.monotonic()
    duration = ctx['duration']
    stop, samples, calls, reads = threading.Event(), [], [], []
    threads = [threading.Thread(target=sampler_loop, args=(t0, ctx['thermal_log'], server, stop, samples), daemon=True)]
    if gemma:
        threads.append(threading.Thread(target=selector_loop, args=(ctx['selector'], ctx['cases'], t0, duration,
                                                                    stop, calls), daemon=True))
    for t in threads:
        t.start()
    try:
        last, heat = frame_loop(ctx['detector'], mode, cam['session'], t0, duration, reads, ctx['thermal_log'])
        elapsed = time.monotonic() - t0
        stop.set()  # a heat stop ends the selector's cadence here too (no slot starts after the block)
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
        camera_stop()
    if any(t.is_alive() for t in threads):  # e.g. a stalled selector request: it would overlap the next block
        raise RuntimeError(f'{name}: a sampler/selector thread still running 120 s after the block; run stopped')
    check_cores('after ' + name)
    therm_end = read_thermal(ctx['thermal_log'])
    return {'block': name, 'mode': mode, 'gemma': gemma, 'duration_s': round(elapsed, 2), 'planned_s': duration,
            'heat_stop': {'zone': PAUSE_ZONE, 'above_c': PAUSE_ABOVE_C, 'reached_limit': heat is not None,
                          **(heat or {'time_to_limit_s': None, 'zone_reading': None})}, 'camera': cam,
            'thermal_start': therm_start, 'thermal_end': therm_end,
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
        return {'block': 'B5_mtp_ram_snapshot', 'cmd': server.cmd, 'load_s': server.load_s,
                'reply': res.get('content'), 'timings': res.get('timings'), 'detected_640_on_frame': r['frame'],
                'server_status_kb': server.status_kb(), 'sample': snap}
    finally:
        if server:
            server.stop()
        camera_stop()


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
    z9 = [b['thermal_start']['z9'], b['thermal_end']['z9']] + [x['z9'] for x in sm if 'z9' in x] + \
         ([b['heat_stop']['zone_reading']['z9']] if b['heat_stop']['reached_limit'] else [])  # the stop reading
    s['z9'] = (b['thermal_start']['z9'] / 1000, b['thermal_end']['z9'] / 1000, max(z9) / 1000)
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
        # a heat stop is a result: rules that need time only apply as far as the block ran (at_start: stopped
        # before its first frame; no_sample: before a second sample was due, so the first may finish after it)
        at_start = heat['reached_limit'] and s['frames'] == 0 and not s['failed']
        no_sample = heat['reached_limit'] and b['duration_s'] < SAMPLE_S
        if b['gemma']:  # the Gemma workload is one call per 20 s slot, each inside the block
            calls = b['selector_calls']
            # slots that can start inside the block: k * SELECTOR_S < planned length (selector_loop's bound; the
            # block itself runs a little past it), or before the heat stop, which also stops the selector; a slot
            # due in the last second before a heat stop is not required (the stop may beat the thread to it)
            end = heat['time_to_limit_s'] - 1 if heat['reached_limit'] else b['planned_s']
            slots = max(0, math.ceil(end / SELECTOR_S))
            late = [c['t'] for c in calls if c['started_s'] - c['t'] > 5]
            # a call in flight at a heat stop ends after it by construction: not an overrun
            over = [] if heat['reached_limit'] else [c['t'] for c in calls if c['ended_s'] > b['duration_s']]
            if slots and s['sel'][0] == 0:
                out.append(f'{b["block"]}: no successful selector call ({s["sel"][1]} failed)')
            elif s['sel'][0] < slots or late or over:
                out.append(f'{b["block"]}: selector cadence missed: {s["sel"][0]}/{slots} successful calls, '
                           f'started >5 s late at slots {late}, ended after the block at slots {over}')
        if s['sample_errors'] or (s['min_avail'] is None and not no_sample):
            out.append(f'{b["block"]}: {s["sample_errors"]} sample(s) with root/PSS/thermal errors, '
                       f'{0 if s["min_avail"] is None else "some"} usable samples')
        sv = b['survived']
        if s['failed'] or (s['frames'] == 0 and not at_start) or \
                not (sv['robotcam_new_frame_at_end'] and sv['robotcam_process']):
            out.append(f'{b["block"]}: RobotCam: {s["frames"]} frames, failed reads {s["failed"] or "none"}, '
                       f'new frame at end {sv["robotcam_new_frame_at_end"]}, process at end {sv["robotcam_process"]}')
        for size in (320, 640) if b['mode'] == 'mix' else (640,):
            if at_start or (size == 640 and b['mode'] == 'mix' and heat['reached_limit'] and
                            heat['time_to_limit_s'] < LARGE_S):
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
        ('gate wait s', None),
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
             'gate wait s': lambda b: str(b['thermal_start']['waited_s']),
             'time to limit s': lambda b: (f'{b["heat_stop"]["time_to_limit_s"]:.1f}' if b['heat_stop']['reached_limit']
                                           else '-')}
    run = json.loads((out / 'run.json').read_text())
    lines = [f'Co-residency benchmark (DECISIONS #124), run {out.name}{"  [SMOKE: not a measurement]" if run["smoke"] else ""}',
             f'block length {run["block_s"]} s; idle z9 {run["idle"]["z9"] / 1000:.1f} degC; runner sha256 {run["sha256"]["coresidency.py"][:12]}']
    lines += [f'INCOMPLETE: {p}' for p in bad] + ['']
    width = 32
    lines.append(f'{"":{width}}' + ''.join(f'{b["block"]:>20}' for b in blocks))
    for label, fn in rows:
        vals = [extra[label](b) if label in extra else fn(s) for b, s in zip(blocks, sums)]
        lines.append(f'{label:{width}}' + ''.join(f'{v:>20}' for v in vals))
    if (out / 'loads.json').exists():
        ld = json.loads((out / 'loads.json').read_text())
        lines += ['', f'Gemma load (spawn to /health ok): cold {ld["cold_load_s"]:.2f} s [{ld["cold_state"]["mode"]}], '
                      f'warm {ld["warm_load_s"]:.2f} s', 'server command: ' + ' '.join(ld['cmd'])]
    if (out / 'block_B5_mtp_ram_snapshot.json').exists():
        b5 = json.loads((out / 'block_B5_mtp_ram_snapshot.json').read_text())
        sm = b5['sample']
        errors = {k: sm[k] for k in ('error', 'thermal_error', 'pss_error') if k in sm}
        if errors:
            bad.append(f'B5_mtp_ram_snapshot: sample errors {errors}')
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
                  'or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Heat stop = the robot\'s '
                  f'live pause ({PAUSE_ZONE} > {PAUSE_ABOVE_C} degC, main.Robot.run_mission), checked between frames '
                  'on the 5 s thermal log; time to limit = when the runner saw it (up to ~5 s plus one detection late); '
                  f'the block ends there and counts as run; drift is n/a if it ran under {2 * DRIFT_S} s.']
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
        if 'heat_stop' not in run:  # its blocks lack heat_stop/planned_s and were judged by the old cadence count
            raise SystemExit(f'--resume: {out} was made by an older runner (no heat stop); start a new run')
    else:
        out = OUT_ROOT / f'run_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}{"_smoke" if a.smoke else ""}'
        out.mkdir(parents=True)
    ctx = {'smoke': a.smoke, 'duration': SMOKE_S if a.smoke else BLOCK_S, 'thermal_log': a.thermal_log,
           'server': None, 'selector': None, 'cases': load_cases()}
    try:
        rc, who = root('id', 'id')
        if rc != 0:
            raise SystemExit(f'su failed: {who.strip()}')
        wait_cores('start')
        ctx['idle'] = read_thermal(a.thermal_log)
        print(f'thermal idle reading: z9 {ctx["idle"]["z9"] / 1000:.1f} degC; output {out}', flush=True)
        files = {'coresidency.py': __file__, 'robotcam_reader.py': robotcam_reader.__file__,
                 'detect_person.py': ROBOT / 'detect_person.py', 'detector_size_policy.py': ROBOT / 'detector_size_policy.py',
                 'server_manager.py': server_manager.__file__, 'main.py': ROBOT / 'main.py',
                 'adapters.py': V3 / 'adapters.py',
                 's1/schema.py': S1O_SRC / 's1/schema.py', 'cases': CASES,
                 'yolo11s_320.onnx': ROBOT / 'yolo11s_320.onnx', 'yolo11s_640.onnx': ROBOT / 'yolo11s_640.onnx'}
        meta = {'started': utc(), 'smoke': a.smoke, 'block_s': ctx['duration'], 'idle': ctx['idle'],
                'sha256': {k: sha256(p) for k, p in files.items()},
                'model_bytes': {p: os.path.getsize(p) for p in (MODEL, DRAFT)},
                'heat_stop': {'zone': PAUSE_ZONE, 'above_c': PAUSE_ABOVE_C},
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


if __name__ == '__main__':
    main()
