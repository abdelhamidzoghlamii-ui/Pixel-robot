"""Mission heat control (DECISIONS #127 rules, #129 planned pauses, adaptive 37/35 degC schedule from #132).

HeatMonitor reads the phone through root (native Termux only; `su` is not available in proot).
decide() is pure and unit-tested. HeatControl ties them to the mission loop and writes the thermal trace.
Code marked "copied from" is the proven campaign code, copied and trimmed (benchmark/ is not deployed).
"""
import collections
import json
import math
import os
import queue
import re
import subprocess
import threading
import time

DUMP_PERIOD_S = 5          # Android thermal status + VIRTUAL-SKIN (dumpsys)
FAST_PERIOD_S = 1          # CPU zones + battery (persistent root shell)
SKIN_PAUSE_C = 37.0        # ACTIVE -> PAUSE at skin >= this ...
SKIN_RESUME_C = 35.0       # ... PAUSE -> ACTIVE at skin <= this
MIN_ACTIVE_S = 60          # counted from camera start, including it
MIN_PAUSE_S = 30           # counted from stop_camera() returning successfully
STATUS_SEVERE = 3
STATUS_CRITICAL = 4
STATUS_RESUME = 2          # MODERATE: a CRITICAL hold ends at status <= this
BATTERY_STOP_C = 45.0
CPU_FAULT_C = 110.0
CPU_FAULT_N = 3            # consecutive 1 s samples
SKIN_STALE_S = 60          # older than this (strictly) = fail closed
FAST_STALE_S = 5
PAUSE_FORWARD_SPEED = 100  # UNCALIBRATED: FORWARD speed in sensor-only pause cycles
DUMP_TIMEOUT_S = 10
START_WAIT_S = 15          # first complete reading must arrive within this, or the mission does not start

THERMAL = '/sys/class/thermal'
CPU_TYPES = ('BIG', 'MID', 'LITTLE')
SKIN = 'VIRTUAL-SKIN'
BATTERY_TEMP = '/sys/class/power_supply/battery/temp'  # tenths of degC; not a thermal zone (zone number unconfirmed)
TRACE_DIR = '/data/data/com.termux/files/home/robot_runs'


# Copied from benchmark/coresidency/coresidency.py: root() (DECISIONS #123 form), trimmed (no thread-local stderr).
def root(cmd, tag, timeout=60):
    """su -c "<cmd> </dev/null >FILE 2>&1", then the file is read. Returns (rc, text)."""
    path = f'/data/local/tmp/heat_control_{tag}.txt'
    r = subprocess.run(['su', '-c', f'{{ {cmd}; }} </dev/null >{path} 2>&1; rc=$?; cat {path}; exit $rc'],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout


# Copied from benchmark/coresidency/coresidency.py: to_int().
def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# Copied from benchmark/coresidency/coresidency.py: RootShell (END marker renamed, last_stderr dropped).
class RootShell:
    """One persistent root shell for the sysfs reads (no new su per sample). Android service calls do not go
    through it: they use root() (DECISIONS #123)."""
    END = '__heat_control_end_'

    def __init__(self):
        self.p = subprocess.Popen(['su'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  bufsize=1, start_new_session=True)
        self.lines, self.errors, self.n = queue.Queue(), queue.Queue(), 0
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._pump, args=(self.p.stderr, self.errors), daemon=True).start()

    def _pump(self, stream=None, lines=None):
        stream = self.p.stdout if stream is None else stream
        lines = self.lines if lines is None else lines
        for line in stream:
            lines.put(line.rstrip('\n'))
        lines.put(None)

    def run(self, script, timeout=5):
        """Output lines of script; raises RuntimeError on timeout or a dead shell."""
        self.n += 1
        end = f'{self.END}{self.n}__'
        try:
            self.p.stdin.write(f'{script}\necho {end}\necho {end} >&2\n')
            self.p.stdin.flush()
        except OSError as e:
            raise RuntimeError(f'root shell: {e}') from e
        deadline = time.monotonic() + timeout
        out = self._until(self.lines, end, deadline, timeout)
        self._until(self.errors, end, deadline, timeout)
        return out

    def _until(self, lines, end, deadline, timeout):
        out = []
        while True:
            try:
                line = lines.get(timeout=max(0.01, deadline - time.monotonic()))
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


# Copied from benchmark/coresidency/coresidency.py: discover(), cut down to the CPU zones (never hardcoded numbers).
def discover(shell):
    """The BIG/MID/LITTLE thermal zone numbers (lowest zone of each type)."""
    script = (f'for f in {THERMAL}/thermal_zone*/type; do [ -e "$f" ] || continue; '
              "v=; read -r v 2>/dev/null <\"$f\"; printf '%s\\t%s\\n' \"$f\" \"$v\"; done")
    zones = {}
    for line in shell.run(script, timeout=30):
        path, _, v = line.partition('\t')
        if m := re.fullmatch(re.escape(THERMAL) + r'/thermal_zone(\d+)/type', path):
            zones[int(m.group(1))] = v
    cpu = {t: next((i for i in sorted(zones) if zones[i] == t), None) for t in CPU_TYPES}
    if None in cpu.values():
        raise RuntimeError(f'CPU thermal zones by type: {cpu}; heat control not started')
    return cpu


# Copied from benchmark/coresidency/coresidency.py: fast_sample(), cut down to CPU zones + battery.
def fast_sample(shell, paths):
    """One 1 s reading: CPU zone files then the battery file. Raises RuntimeError unless every value was read."""
    lines = shell.run(f'for f in {" ".join(paths)}; do v=; read -r v 2>/dev/null <"$f"; echo "$v"; done')
    values = [to_int(v) for v in lines]
    if len(values) != len(paths) or None in values:
        raise RuntimeError(f'incomplete sample: {lines!r}')
    return {'t': time.monotonic(), 'cpu_c': max(values[:-1]) / 1000, 'bat_c': values[-1] / 10}


# Copied from benchmark/coresidency/coresidency.py: TEMP_RE and parse_dump(), trimmed to status and VIRTUAL-SKIN.
TEMP_RE = re.compile(r'Temperature\{mValue=([^,]+), mType=(-?\d+), mName=([^,]+), mStatus=(\d+)\}')


def parse_dump(text):
    """Android thermal status and VIRTUAL-SKIN from 'Current temperatures from HAL' (None when absent or NaN)."""
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
    return {'status': int(status.group(1)) if status else None, 'skin': hal.get(SKIN)}


# Copied from benchmark/coresidency/coresidency.py: read_dump(), timeout 10 s.
def read_dump():
    """`dumpsys thermalservice` (DECISIONS #123 form); t = monotonic s when the reading was in hand."""
    try:
        rc, text = root('dumpsys thermalservice', 'thermalservice', timeout=DUMP_TIMEOUT_S)
        d = {**parse_dump(text), 'error': None if rc == 0 else f'rc {rc}'}
    except (OSError, subprocess.SubprocessError) as e:
        d = {'status': None, 'skin': None, 'error': f'{type(e).__name__}: {e}'}
    return {'t': time.monotonic(), **d}


class HeatMonitor:
    """Real hardware reader: one daemon thread, CPU zones + battery every 1 s, dumpsys every 5 s.
    Only complete readings are published; a failed read leaves the last one to go stale."""

    def __init__(self):
        self.on_dump = None  # HeatControl sets it; called (in this thread) with every dumpsys read
        self.lock = threading.Lock()
        self.dump = None
        self.fast = collections.deque(maxlen=CPU_FAULT_N)
        self.stopping = threading.Event()
        self.shell = self.thread = None

    def start(self):
        """Raises (mission not started) without root, CPU zones, or a complete first reading within 15 s."""
        self.shell = RootShell()
        try:
            self.paths = [f'{THERMAL}/thermal_zone{i}/temp' for i in discover(self.shell).values()] + [BATTERY_TEMP]
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            deadline = time.monotonic() + START_WAIT_S
            while time.monotonic() < deadline:
                r = self.readings()
                if r['dump'] and r['fast']:
                    return
                time.sleep(0.2)
            raise RuntimeError(f'no complete heat reading (status, skin, CPU zones, battery) in {START_WAIT_S} s')
        except BaseException:
            self.stop()
            raise

    def _loop(self):
        next_dump = 0.0
        while not self.stopping.is_set():
            began = time.monotonic()
            try:
                s = fast_sample(self.shell, self.paths)
                with self.lock:
                    self.fast.append(s)
            except Exception:
                pass  # stays stale; decide() fails closed after FAST_STALE_S
            if began >= next_dump:
                next_dump = began + DUMP_PERIOD_S
                d = read_dump()
                if d['status'] is not None and d['skin'] is not None:
                    with self.lock:
                        self.dump = d
                try:
                    if self.on_dump:
                        self.on_dump(d)
                except Exception:
                    pass
            self.stopping.wait(max(0.0, began + FAST_PERIOD_S - time.monotonic()))

    def readings(self):
        with self.lock:
            return {'dump': self.dump, 'fast': list(self.fast)}

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=DUMP_TIMEOUT_S + 5)
        if self.shell:
            self.shell.close()


def decide(now, readings, state):
    """(target, reason): target ACTIVE / PAUSE / HOLD / END. Pure; times are monotonic s.
    readings: {'dump': {'t', 'status', 'skin'} or None, 'fast': [{'t', 'cpu_c', 'bat_c'}, ...] oldest first}.
    state: {'phase': ACTIVE / PAUSE / HOLD, 'since': start of the phase (a HOLD keeps its pause's start)}.
    Staleness: a reading exactly SKIN_STALE_S / FAST_STALE_S old is still fresh; older ends the mission."""
    # Precedence and staleness copied from benchmark/coresidency/coresidency.py: block_limit(); fail-closed first.
    dump, fast = readings['dump'], readings['fast']
    if dump is None or now - dump['t'] > SKIN_STALE_S:
        return 'END', f'fail-closed: no Android status + {SKIN} reading for > {SKIN_STALE_S} s'
    if not fast or now - fast[-1]['t'] > FAST_STALE_S:
        return 'END', f'fail-closed: no CPU zone + battery reading for > {FAST_STALE_S} s'
    if fast[-1]['bat_c'] >= BATTERY_STOP_C:
        return 'END', f'battery {fast[-1]["bat_c"]:.1f} >= {BATTERY_STOP_C} degC'
    if len(fast) >= CPU_FAULT_N and all(s['cpu_c'] >= CPU_FAULT_C for s in fast[-CPU_FAULT_N:]):
        return 'END', (f'CPU zone >= {CPU_FAULT_C:.0f} degC in {CPU_FAULT_N} consecutive 1 s samples '
                       f'({[s["cpu_c"] for s in fast[-CPU_FAULT_N:]]})')
    status, skin, phase = dump['status'], dump['skin'], state['phase']
    if status >= STATUS_CRITICAL or (phase == 'HOLD' and status > STATUS_RESUME):
        return 'HOLD', f'Android thermal status {status}: motors stopped until status <= {STATUS_RESUME}'
    note = (f'Android thermal status SEVERE ({status}): no optional Gemma work in the mission loop, nothing changed'
            if status == STATUS_SEVERE else '')
    # Adaptive rule copied from benchmark/campaign/phase23.py: switch() ('adaptive').
    elapsed = now - state['since']
    if phase == 'ACTIVE':
        if skin >= SKIN_PAUSE_C and elapsed >= MIN_ACTIVE_S:
            return 'PAUSE', f'{SKIN} {skin:.2f} >= {SKIN_PAUSE_C} after {elapsed:.0f} s active'
        return 'ACTIVE', note
    if skin <= SKIN_RESUME_C and elapsed >= MIN_PAUSE_S:
        return 'ACTIVE', f'{SKIN} {skin:.2f} <= {SKIN_RESUME_C} after {elapsed:.0f} s paused'
    return 'PAUSE', note


class HeatControl:
    """The mission's heat controller: monitor readings -> decide(), the phase state, and the thermal trace."""

    def __init__(self, monitor, trace_dir=TRACE_DIR, clock=time.monotonic):
        self.monitor, self.trace_dir, self.clock = monitor, trace_dir, clock
        self.state = {'phase': 'ACTIVE', 'since': None}
        self.note = ''
        self.distance = None
        self.trace_path = None
        self.lock = threading.Lock()
        monitor.on_dump = lambda d: self.event('dumpsys', d['error'], d)

    @property
    def phase(self):
        return self.state['phase']

    def now(self):
        return self.clock()

    def start(self):
        """Starts the monitor (raises if it cannot) and begins ACTIVE; call just before the camera starts."""
        self.trace_path = os.path.join(self.trace_dir, time.strftime('heat_trace_%Y%m%dT%H%M%SZ.jsonl', time.gmtime()))
        try:
            os.makedirs(self.trace_dir, exist_ok=True)
        except OSError as e:
            print(f'[HEAT] trace folder failed: {e}')
        print(f'[HEAT] trace: {self.trace_path}')
        self.monitor.start()
        self.state = {'phase': 'ACTIVE', 'since': self.clock()}
        self.event('start')

    def stop(self):
        self.monitor.stop()

    def decide(self, distance=None):
        self.distance = distance
        target, reason = decide(self.clock(), self.monitor.readings(), self.state)
        if target == 'END':
            self.event('end', reason)
        elif target == self.phase and reason != self.note:
            self.note = reason
            if reason:
                print(f'[HEAT] {reason}')
                self.event('note', reason)
        return target, reason

    def enter(self, phase, since=None):
        """Phase change; since=None keeps the current phase start (PAUSE <-> HOLD)."""
        old = self.phase
        self.state = {'phase': phase, 'since': self.state['since'] if since is None else since}
        self.note = ''
        self.event('hold_start' if phase == 'HOLD' else 'hold_end' if old == 'HOLD' else 'phase_change')

    def summary(self):
        d = self.monitor.readings()['dump'] or {}
        return f"skin {d.get('skin')} degC status {d.get('status')}"

    def event(self, name, detail=None, dump=None):
        """One JSON line in the trace; a write failure is printed, never raised."""
        r = self.monitor.readings()
        d, fast = dump or r['dump'] or {}, r['fast'][-1] if r['fast'] else {}
        row = {'t_mono': self.clock(), 't_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': name,
               'phase': self.phase, 'status': d.get('status'), 'skin_c': d.get('skin'),
               'battery_c': fast.get('bat_c'), 'cpu_max_c': fast.get('cpu_c'), 'distance_cm': self.distance,
               'detail': detail}
        try:
            with self.lock, open(self.trace_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
        except (OSError, TypeError) as e:
            print(f'[HEAT] trace write failed: {e}')
