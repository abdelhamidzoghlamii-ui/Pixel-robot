#!/data/data/com.termux/files/usr/bin/python
"""Thermal characterization run, motors off: at which sensor readings the robot's heaviest realistic load is throttled.
Research only: never imports motors.py or opens USB/serial. Native Termux only; started by oneshot.sh in this folder.

Load, until a stop condition: RobotCam mode B rate 2 -> 640 detection (detect_person.Detector) on every new frame,
plus llama-server with server_manager's setup_q4 flags generating back-to-back (one fixed prompt, n_predict 256,
temperature 0, streamed) in its own thread. Reuses ../coresidency/coresidency.py for RobotCam start/stop, the
llama-server process (flags, start, stop, signal deferral), the cpuset guard and the signal handler.

Every 1 s, one persistent root shell reads all thermal zone temps, scaling_cur_freq/scaling_max_freq per cpufreq
policy, every cooling device's cur_state, battery temp/current/voltage/status. Every 5 s, `dumpsys thermalservice`
in the DECISIONS #123 form, parsed for the Android thermal status and the HAL temperatures.

Stops the load at the first of: VIRTUAL-SKIN >= 48.0 degC; battery >= 45.0 degC; Android status >= EMERGENCY;
a CPU zone (BIG/MID/LITTLE) >= 110 degC in 3 consecutive 1 s samples; 20 min of load. Fail closed: no
VIRTUAL-SKIN + status reading for 60 s (counted from load start), no CPU zone or battery temperature reading
for 5 s (each limit uses only its own readings), no new RobotCam
frame for 10 s, a llama-server failure, or cores 4-7 lost. The thermal stops also apply while llama-server
and RobotCam are starting (then no load runs). After the load, stops RobotCam and llama-server, checks that capture
stopped (no new frame for 3 s) and force-stops the app (it stays cached after STOP), and keeps logging 5 min
(cooldown). --smoke: 60 s load, 30 s cooldown, same stops.

Usage: thermal_char.py [--smoke] [--note TEXT]
"""
import argparse
import bisect
import contextlib
import hashlib
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

sys.dont_write_bytecode = True
HOME = Path('/data/data/com.termux/files/home')
sys.path.insert(0, str(HOME / 'robot/benchmark/coresidency'))
import coresidency as cr  # noqa: E402  (imports robotcam_reader, detect_person, server_manager, measure; no motors)

OUT_ROOT = HOME / 'thermal_char'
DOWNLOADS = HOME / 'storage/downloads'
THERMAL = '/sys/class/thermal'
CPUFREQ = '/sys/devices/system/cpu/cpufreq'
BATTERY = cr.BATTERY
ROOT_TMP = '/data/local/tmp'
LOAD_S, COOL_S = 20 * 60, 5 * 60
SMOKE_LOAD_S, SMOKE_COOL_S = 60, 30
BASELINE_S = 10
SAMPLE_S, DUMP_S, WINDOW_S, TIMELINE_S, MINUTE_S = 1, 5, 10, 30, 60
SKIN_MAX_C, BATTERY_MAX_C, STATUS_MAX, CPU_MAX_C, CPU_N = 48.0, 45.0, 5, 110.0, 3
SKIN_WAIT_S, SENSOR_WAIT_S, FRAME_WAIT_S = 60, 5, 10
SKIN = 'VIRTUAL-SKIN'
CPU_TYPES = ('BIG', 'MID', 'LITTLE')
STATUS_NAMES = ['NONE', 'LIGHT', 'MODERATE', 'SEVERE', 'CRITICAL', 'EMERGENCY', 'SHUTDOWN']
PROMPT = ('The following is a detailed, step-by-step guide to how a small home robot explores an unfamiliar house, '
          'room by room, and what it looks for in each room:\n\n1.')
N_PREDICT = 256
DEVNULL = subprocess.DEVNULL
LOGS = ('samples_1s.jsonl', 'thermalservice_5s.jsonl', 'thermalservice_raw.jsonl', 'frames.jsonl', 'gen.jsonl')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- root access

class RootShell:
    """One persistent root shell for the sysfs reads (no new su per sample). Android service calls do not go
    through it: they use root_file (DECISIONS #123)."""
    END = '__thermal_char_end_'

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


def root_file(cmd, tag, timeout=30):
    """DECISIONS #123 form (as coresidency.root, own file name): su -c "<cmd> </dev/null >FILE 2>&1", file read."""
    path = f'{ROOT_TMP}/thermal_char_{tag}.txt'
    r = subprocess.run(['su', '-c', f'{{ {cmd}; }} </dev/null >{path} 2>&1; rc=$?; cat {path}; exit $rc'],
                       stdin=DEVNULL, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout


# ---------------------------------------------------------------- layout and 1 s samples

def discover(shell):
    """Every thermal zone's type and trip points, every cpufreq policy, every cooling device, as read."""
    globs = [f'{THERMAL}/thermal_zone*/type', f'{THERMAL}/thermal_zone*/trip_point_*_temp',
             f'{THERMAL}/thermal_zone*/trip_point_*_type', f'{THERMAL}/cooling_device*/type',
             f'{THERMAL}/cooling_device*/max_state'] + \
            [f'{CPUFREQ}/policy*/{n}' for n in ('related_cpus', 'cpuinfo_max_freq', 'cpuinfo_min_freq',
                                                'scaling_available_frequencies', 'scaling_max_freq')]
    script = (f'for f in {" ".join(globs)}; do [ -e "$f" ] || continue; v=; read -r v 2>/dev/null <"$f"; '
              "printf '%s\\t%s\\n' \"$f\" \"$v\"; done")
    zones, policies, cooling = {}, {}, {}
    for line in shell.run(script, timeout=30):
        path, _, v = line.partition('\t')
        if m := re.fullmatch(re.escape(THERMAL) + r'/thermal_zone(\d+)/type', path):
            zones.setdefault(m.group(1), {'trips': {}})['type'] = v
        elif m := re.fullmatch(re.escape(THERMAL) + r'/thermal_zone(\d+)/trip_point_(\d+)_(temp|type)', path):
            zones.setdefault(m.group(1), {'trips': {}})['trips'].setdefault(m.group(2), {})[m.group(3)] = \
                to_int(v) if m.group(3) == 'temp' else v
        elif m := re.fullmatch(re.escape(THERMAL) + r'/cooling_device(\d+)/(type|max_state)', path):
            cooling.setdefault(m.group(1), {})[m.group(2)] = to_int(v) if m.group(2) == 'max_state' else v
        elif m := re.fullmatch(re.escape(CPUFREQ) + r'/(policy\d+)/(\w+)', path):
            policies.setdefault(m.group(1), {})[m.group(2)] = to_int(v) if m.group(2).endswith('_freq') else v
    num = lambda d: dict(sorted(d.items(), key=lambda kv: int(re.sub(r'\D', '', kv[0]))))
    for z in zones.values():
        z['trips'] = num(z['trips'])
    layout = {'zones': num(zones), 'policies': num(policies), 'cooling': num(cooling)}
    layout['cpu_zones'] = {t: next((i for i, z in layout['zones'].items() if z.get('type') == t), None)
                           for t in CPU_TYPES}
    missing = [t for t, i in layout['cpu_zones'].items() if i is None]
    bad = [p for p, d in layout['policies'].items() if not d.get('cpuinfo_max_freq')]
    if missing or not layout['policies'] or bad:
        raise SystemExit(f'layout: CPU zone types missing {missing}, policies {list(layout["policies"])}, '
                         f'without cpuinfo_max_freq {bad}; not started')
    return layout


def sample_keys(layout):
    keys = [('zones', i, f'{THERMAL}/thermal_zone{i}/temp') for i in layout['zones']]
    for p in layout['policies']:
        keys += [('cur', p, f'{CPUFREQ}/{p}/scaling_cur_freq'), ('max', p, f'{CPUFREQ}/{p}/scaling_max_freq')]
    keys += [('cd', i, f'{THERMAL}/cooling_device{i}/cur_state') for i in layout['cooling']]
    keys += [('bat', n, f'{BATTERY}/{n}') for n in ('temp', 'current_now', 'voltage_now', 'status')]
    return keys


def sample_script(keys):
    """Shell builtins only (read, echo): no process per file. An unreadable file gives an empty line."""
    return f'for f in {" ".join(k[2] for k in keys)}; do v=; read -r v 2>/dev/null <"$f"; echo "$v"; done'


def parse_sample(keys, lines, layout):
    if len(lines) != len(keys):
        raise RuntimeError(f'expected {len(keys)} values, got {len(lines)}: {lines[:5]!r}')
    s, bat = {'zones': {}, 'cur': {}, 'max': {}, 'cd': {}}, {}
    for (kind, name, _), v in zip(keys, lines):
        if kind == 'bat':
            bat[name] = v.strip()
        else:
            s[kind][name] = to_int(v)
    temp, cur, volt = to_int(bat['temp']), to_int(bat['current_now']), to_int(bat['voltage_now'])
    s['bat_c'] = None if temp is None else temp / 10  # power_supply temp is in tenths of degC
    s['bat_w'] = None if None in (cur, volt) else round(-(cur * volt) / 1e12, 3)  # camera_heat.py's method
    s['bat_status'] = bat['status'] or None
    s['cpu_ok'] = all(s['zones'][i] is not None for i in layout['cpu_zones'].values())  # the CPU stop's inputs
    s['complete'] = s['cpu_ok'] and s['bat_c'] is not None  # every 1 s stop input (power/freqs are data only)
    return s


def take_sample(shell, keys, script, layout, t0):
    start = time.monotonic()
    try:
        lines, err = shell.run(script), None
    except RuntimeError as e:
        lines, err = None, e
    done = time.monotonic()
    # t = when the read was in hand (it ran t_start..t): stops, events, timeline, cooldown and gaps use it
    s = {'t': round(done - t0, 3), 't_start': round(start - t0, 3), 'read_s': round(done - start, 3),
         'utc': cr.utc()}
    try:
        if err:
            raise err
        s.update(parse_sample(keys, lines, layout))
    except RuntimeError as e:
        s.update(complete=False, error=str(e))
    s['cpus_ok'] = set(range(4, 8)) <= cr.allowed_cpus()
    return s


# ---------------------------------------------------------------- 5 s thermalservice

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


def dump_loop(t0, stop, dumps, fh, raw_fh):
    k, last_raw, began = 0, None, time.monotonic()
    while not stop.wait(max(0.0, began + k * DUMP_S - time.monotonic())):
        t_start = round(time.monotonic() - t0, 3)
        try:
            rc, text = root_file('dumpsys thermalservice', 'thermalservice')
            d = {'rc': rc, **parse_dump(text)}
        except (OSError, subprocess.SubprocessError) as e:
            text, d = None, {'rc': None, 'error': f'{type(e).__name__}: {e}',
                             'status': None, 'skin': None, 'hal': {}, 'cached_names': []}
        # t = when the reading was in hand (the dump ran t_start..t): stops, freshness and events use it
        d = {'t': round(time.monotonic() - t0, 3), 't_start': t_start, 'utc': cr.utc(), **d}
        d['raw_kept'] = text is not None and text != last_raw  # the first dump and every changed one
        if d['raw_kept']:
            raw_fh.write(json.dumps({'t': d['t'], 't_start': t_start, 'text': text}) + '\n')
            raw_fh.flush()
            last_raw = text
        dumps.append(d)
        fh.write(json.dumps(d) + '\n')
        fh.flush()
        k = max(k + 1, math.ceil((time.monotonic() - began) / DUMP_S))


def sample_loop(shell, keys, layout, t0, stop, samples, fh):
    script, k, began = sample_script(keys), 0, time.monotonic()
    while not stop.wait(max(0.0, began + k * SAMPLE_S - time.monotonic())):
        s = take_sample(shell, keys, script, layout, t0)
        samples.append(s)
        fh.write(json.dumps(s) + '\n')
        fh.flush()
        k = max(k + 1, math.ceil((time.monotonic() - began) / SAMPLE_S))  # an overrun skips slots, never bunches


# ---------------------------------------------------------------- stops

def stop_reasons(now, load_start, samples, dumps, layout):
    """Every stop condition true at `now` (times in s since the runner's t0). Pure: tested offline."""
    r = []
    skin = next((d for d in reversed(dumps) if d['skin'] is not None), None)
    status = next((d for d in reversed(dumps) if d['status'] is not None), None)
    if skin and skin['skin'] >= SKIN_MAX_C:
        r.append(f'{SKIN} {skin["skin"]:.1f} >= {SKIN_MAX_C} degC')
    if status and status['status'] >= STATUS_MAX:
        r.append(f'Android thermal status {status["status"]} >= {STATUS_MAX} ({STATUS_NAMES[STATUS_MAX]})')
    # each limit from its own readings: a failed read of another sensor neither blocks nor resets it
    bat = [s for s in samples[-SENSOR_WAIT_S * 4:] if s.get('bat_c') is not None]
    if bat and bat[-1]['bat_c'] >= BATTERY_MAX_C:
        r.append(f'battery {bat[-1]["bat_c"]:.1f} >= {BATTERY_MAX_C} degC')
    cpu = list(layout['cpu_zones'].values())
    read = [s for s in samples[-SENSOR_WAIT_S * 4:] if s.get('cpu_ok')]  # samples with every CPU zone readable
    tail = read[-CPU_N:]
    if len(tail) == CPU_N and all(max(s['zones'][z] for z in cpu) >= CPU_MAX_C * 1000 for s in tail):
        r.append(f'CPU zone >= {CPU_MAX_C:.0f} degC in {CPU_N} consecutive 1 s samples '
                 f'(max {[max(s["zones"][z] for z in cpu) / 1000 for s in tail]})')
    fresh = min(skin['t'] if skin else -math.inf, status['t'] if status else -math.inf)
    if now - max(fresh, load_start) > SKIN_WAIT_S:
        r.append(f'fail closed: no {SKIN} and Android status reading for {SKIN_WAIT_S} s')
    for name, rows in (('CPU zone', read), ('battery temperature', bat)):
        if now - max(rows[-1]['t'] if rows else -math.inf, load_start) > SENSOR_WAIT_S:
            r.append(f'fail closed: no {name} reading for {SENSOR_WAIT_S} s')
    return r


class StartupStopped(Exception):
    pass


@contextlib.contextmanager
def stops_checked(check):
    """llama-server and RobotCam start on the main thread (coresidency's signal deferral needs it) and poll
    coresidency.healthy / coresidency.read_frame while they wait; check() runs on every such poll, so a stop
    condition raises inside them (Server then stops its own process; main's finally stops RobotCam)."""
    real_healthy, real_read = cr.healthy, cr.read_frame

    def healthy():
        check()
        return real_healthy()

    def read_frame(*a, **k):
        check()
        return real_read(*a, **k)
    cr.healthy, cr.read_frame = healthy, read_frame
    try:
        yield
    finally:
        cr.healthy, cr.read_frame = real_healthy, real_read


# ---------------------------------------------------------------- load

def frame_loop(detector, session, t0, stop, st, fh):
    """640 detection on every new RobotCam frame (read_frame's fail-closed checks, frame number must advance)."""
    try:
        _frame_loop(detector, session, t0, stop, st, fh)
    except Exception as e:  # the controller stops the load on it (else it would read as a RobotCam stall)
        st['load_error'] = f'frame/detect loop failed: {type(e).__name__}: {e}'


def _frame_loop(detector, session, t0, stop, st, fh):
    last = None
    while not stop.is_set():
        a = time.perf_counter()
        r = cr.read_frame(cr.FRAME_DIR, session=session)
        row = {'t': round(time.monotonic() - t0, 3), 'status': r['status']}
        if r['status'] == 'ok' and last is not None and r['frame'] <= last:
            st['repeats'] += 1
            stop.wait(0.02)
            continue
        if r['status'] != 'ok':
            fh.write(json.dumps(row) + '\n')
            stop.wait(0.1)
            continue
        last = r['frame']
        st['last_frame_t'] = time.monotonic()
        c = time.perf_counter()
        n = len(detector.detect(r['image'], 640))
        row.update(frame=r['frame'], age_s=round(r['age_s'], 3), detect_ms=round((time.perf_counter() - c) * 1000, 2),
                   n_detections=n)
        fh.write(json.dumps(row) + '\n')
        fh.flush()
        stop.wait(max(0.0, a + 0.5 - time.perf_counter()))  # rate 2: next frame about 0.5 s after this one


def generate(t0, arrivals):
    """One streamed /completion; appends token arrival times to `arrivals` as they come (kept if the request is
    cut off), returns llama-server's timings."""
    body = {'prompt': PROMPT, 'n_predict': N_PREDICT, 'temperature': 0, 'cache_prompt': False, 'stream': True}
    req = urllib.request.Request(f'http://127.0.0.1:{cr.PORT}/completion', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    final = None
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data: '):
                continue
            msg = json.loads(line[6:])
            if msg.get('stop'):
                final = msg
                break
            arrivals.append(round(time.monotonic() - t0, 3))
    if not final or 'timings' not in final:
        raise RuntimeError('stream ended without final timings')
    return final['timings']


def gen_loop(t0, stop, st, fh):
    while not stop.is_set():
        began, arrivals, tm, err = round(time.monotonic() - t0, 3), [], {}, None
        try:
            tm = generate(t0, arrivals)
        except Exception as e:  # after the load stop the server is killed under the request: not a failure
            err = f'{type(e).__name__}: {e}'
            if not stop.is_set():
                st['load_error'] = f'llama-server request failed: {err}'
        # an interrupted request keeps its streamed tokens (work rate) but has no timings
        fh.write(json.dumps({'t_start': began, 't_end': round(time.monotonic() - t0, 3), 'streamed': len(arrivals),
                             'arrivals': arrivals, 'interrupted': err, **{k: tm.get(k) for k in (
                                 'prompt_n', 'prompt_ms', 'predicted_n', 'predicted_ms', 'predicted_per_second')}}) + '\n')
        fh.flush()
        if err:
            return


# ---------------------------------------------------------------- report

def jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def med(xs):
    return statistics.median(xs) if xs else None


def f(v, spec='.1f'):
    return 'n/a' if v is None else format(v, spec)


def at_or_before(rows, t, key='t'):
    i = bisect.bisect_right([r[key] for r in rows], t)
    return rows[i - 1] if i else None


def work(frames, gens, a, b):
    """Work in (a, b]: frames detected, 640 median ms, streamed tok/s, timings tok/s of requests ending there."""
    det = [r['detect_ms'] for r in frames if 'detect_ms' in r and a < r['t'] <= b]
    toks = sum(a < x <= b for g in gens for x in g['arrivals'])
    done = [g for g in gens if a < g['t_end'] <= b and g.get('predicted_ms')]
    tim = (sum(g['predicted_n'] for g in done) / sum(g['predicted_ms'] for g in done) * 1000) if done else None
    return {'frames': len(det), 'ms640': med(det), 'tok_s_stream': toks / (b - a), 'tok_s_timings': tim,
            'requests': len(done)}


def report(out):
    """Returns (report text, problems)."""
    run = json.loads((out / 'run.json').read_text())
    lay = run['layout']
    samples, dumps, frames, gens = (jsonl(out / n) for n in ('samples_1s.jsonl', 'thermalservice_5s.jsonl',
                                                             'frames.jsonl', 'gen.jsonl'))
    L0, L1 = run.get('load_start'), run.get('load_stop')
    base = L0 if L0 is not None else (samples[0]['t'] if samples else 0)
    rel = lambda t: f'{t - base:+.1f}'
    ztype = {i: z.get('type', f'zone{i}') for i, z in lay['zones'].items()}
    cz = lay['cpu_zones']
    quiet = next((i for i, t in ztype.items() if t == 'quiet_therm'), None)
    mhz = lambda khz: None if khz is None else khz / 1000
    c = lambda mc: None if mc is None else mc / 1000
    good_skin = [d for d in dumps if d['skin'] is not None]
    good_status = [d for d in dumps if d['status'] is not None]

    def phase(t):
        if L0 is None or t < L0:
            return 'baseline'
        return 'load' if L1 is None or t < L1 else ('stop' if t == L1 else 'cooldown')

    def readings(t):
        s, sk, stt = at_or_before(samples, t), at_or_before(good_skin, t), at_or_before(good_status, t)
        line = (f'skin {f(sk and sk["skin"])} (dump {rel(sk["t"]) if sk else "-"}) status '
                f'{stt["status"] if stt else "n/a"} | ')
        if not s:
            return line + 'no 1 s sample yet'
        line += (f'battery {f(s.get("bat_c"))} degC {f(s.get("bat_w"), ".2f")} W {s.get("bat_status")} | ' +
                 ' '.join(f'{p} {f(mhz(s["cur"].get(p)), ".0f")}/{f(mhz(s["max"].get(p)), ".0f")} MHz'
                          for p in lay['policies']) + f' (sample {rel(s["t"])})')
        if 'zones' in s:
            line += '\n      zones: ' + ' '.join(f'{ztype[i]}={f(c(v))}' for i, v in s['zones'].items())
        if sk:
            line += '\n      HAL: ' + ' '.join(f'{k}={v:.1f}' for k, v in sk['hal'].items())
        return line

    # events
    ev = []
    below = set()
    known = {}  # last readable value per policy max / cooling device: an unreadable sample is no change
    for s in samples:
        for kind in ('max', 'cd'):
            for name, v in s.get(kind, {}).items():
                if v is None:
                    continue
                old = known.get((kind, name))
                known[(kind, name)] = v
                if kind == 'cd':
                    if old is not None and v != old:
                        cd = lay['cooling'].get(name, {})
                        ev.append((s['t'], f'cooling_device{name} {cd.get("type")} cur_state {old} -> {v} '
                                           f'(max {cd.get("max_state")})'))
                    continue
                top = lay['policies'][name]['cpuinfo_max_freq']
                if v == old or (old is None and v >= top):
                    continue
                tag = ''
                if v < top and name not in below:
                    below.add(name)
                    tag = ' FIRST below cpuinfo_max_freq'
                ev.append((s['t'], f'{name} scaling_max_freq {"start" if old is None else f"{mhz(old):.0f}"} -> '
                                   f'{mhz(v):.0f} MHz (cpuinfo_max {mhz(top):.0f}){tag}'))
    last = None
    for d in good_status:
        if d['status'] != last:
            name = STATUS_NAMES[d['status']] if d['status'] < len(STATUS_NAMES) else '?'
            ev.append((d['t'], f'Android thermal status {"start" if last is None else last} -> {d["status"]} ({name}; '
                               f'dump ran {rel(d["t_start"])}..{rel(d["t"])})'))
            last = d['status']
    ev.sort(key=lambda e: e[0])

    bad = []
    reasons = run.get('stop_reasons') or ['(none recorded: run ended abnormally)']
    if not run.get('stop_ok'):
        bad.append('stop reason is not a threshold or the planned duration: ' + '; '.join(reasons))
    load_samples = [s for s in samples if L0 is not None and L0 <= s['t'] < (L1 or math.inf)]
    statuses = sorted({s.get('bat_status') for s in load_samples}, key=str)
    if load_samples and statuses != ['Discharging']:
        bad.append(f'battery status during load {statuses} (W needs Discharging throughout)')
    if any(not s.get('cpus_ok') for s in load_samples):
        bad.append(f'cores 4-7 not allowed in {sum(not s.get("cpus_ok") for s in load_samples)} load samples')
    for k, v in (run.get('after_stop') or {}).items():
        if k == 'robotcam':  # a cached app process is no failure: capture still advancing or a failed force-stop is
            if why := cr.camera_end_failed(v):
                bad.append(f'after the load stop: RobotCam {why}')
        elif v:
            bad.append(f'after the load stop: {k} {v}')
    if L1 is not None and run.get('after_stop') and samples and samples[-1]['t'] - L1 < run['cool_s'] - 2 * SAMPLE_S:
        bad.append(f'cooldown logged {samples[-1]["t"] - L1:.0f} of {run["cool_s"]} s')
    incomplete = [s for s in samples if not s.get('complete')]
    reads = [s['read_s'] for s in samples if 'read_s' in s]
    failed_reads = [r for r in frames if r['status'] != 'ok']

    lines = [f'Thermal characterization run {out.name}{"  [SMOKE: functional check, not a measurement]" if run["smoke"] else ""}',
             f'note: {run.get("note") or "(none)"}',
             f'runner sha256 {run["sha256"]["thermal_char.py"][:12]}; planned load {run["load_s"]} s, cooldown {run["cool_s"]} s',
             f'STOP REASON: {"; ".join(reasons)}' +
             (f' at load {run["load_stop"] - L0:+.1f} s' if L0 is not None and L1 is not None else ''),
             *[f'INCOMPLETE: {p}' for p in bad], '',
             'LAYOUT (discovered at start)']
    for p, d in lay['policies'].items():
        lines.append(f'  {p}: cpus {d.get("related_cpus")}, cpuinfo_max {mhz(d["cpuinfo_max_freq"]):.0f} MHz, '
                     f'min {f(mhz(d.get("cpuinfo_min_freq")), ".0f")} MHz, scaling_max at start '
                     f'{f(mhz(d.get("scaling_max_freq")), ".0f")} MHz')
    for t, i in cz.items():
        trips = lay['zones'][i]['trips']
        lines.append(f'  {t} = thermal_zone{i}; trips ' + ', '.join(
            f'{f(c(v.get("temp")), ".0f")} {v.get("type")}' for v in trips.values()))
    lines.append(f'  quiet_therm = {"thermal_zone" + quiet if quiet else "not found"}; {len(lay["zones"])} zones, '
                 f'{len(lay["cooling"])} cooling devices (all trip points and types in run.json)')
    hal_names = sorted({n for d in dumps for n in d['hal']})
    cached = sorted({n for d in dumps for n in d.get('cached_names', [])})
    lines += [f'  Android HAL temperature names: {", ".join(hal_names) or "none parsed"}',
              f'  {SKIN} in HAL temperatures: {"yes" if SKIN in hal_names else "NO"}; cached names: {", ".join(cached) or "none"}',
              f'  thermalservice dumps {len(dumps)} (with skin {len(good_skin)}, with status {len(good_status)}), '
              f'raw kept {sum(d.get("raw_kept", False) for d in dumps)}; 1 s samples {len(samples)} '
              f'(incomplete {len(incomplete)}, largest gap {f(max((b["t"] - a["t"] for a, b in zip(samples, samples[1:])), default=None))} s); '
              f'failed RobotCam reads {len(failed_reads)}',
              f'  1 s sample read duration (stamped at completion): median {f(med(reads), ".3f")} s, '
              f'max {f(max(reads, default=None), ".3f")} s', '',
              f'EVENTS (t = s from load start; {len(ev)} events)']
    for t, text in ev:
        lines += [f'  {rel(t)} [{phase(t)}] {text}', f'      {readings(t)}']
    if not ev:
        lines.append('  none')

    lines += ['', f'TIMELINE every {TIMELINE_S} s (t from load start; skin/status from the latest dump; 640 ms, '
                  f'tok/s and W over the preceding {TIMELINE_S} s)']
    pol = list(lay['policies'])
    lines.append('  ' + ' '.join(f'{h:>8}' for h in ['t', 'phase', 'skin', 'battery', 'BIG', 'MID', 'LITTLE', 'quiet'] +
                                 [f'p{p[6:]}max' for p in pol] + ['status', '640ms', 'tok/s', 'tok/s_t', 'W']))
    end = samples[-1]['t'] if samples else base
    ts = [base + k * TIMELINE_S for k in range(0, int((end - base) // TIMELINE_S) + 1)] + \
         ([L1] if L1 is not None else []) + ([samples[0]['t']] if samples and samples[0]['t'] < base else [])
    for t in sorted(ts):
        s, sk, stt = at_or_before(samples, t), at_or_before(good_skin, t), at_or_before(good_status, t)
        if not s or 'zones' not in s:
            continue
        w = work(frames, gens, t - TIMELINE_S, t)
        watts = [x['bat_w'] for x in samples if t - TIMELINE_S < x['t'] <= t and x.get('bat_w') is not None]
        lines.append('  ' + ' '.join(f'{v:>8}' for v in [
            rel(t), phase(t)[:8], f(sk and sk['skin']), f(s.get('bat_c')),
            *[f(c(s['zones'].get(cz[k]))) for k in CPU_TYPES], f(c(s['zones'].get(quiet)) if quiet else None),
            *[f(mhz(s['max'].get(p)), '.0f') for p in pol], str(stt['status']) if stt else 'n/a',
            f(w['ms640'], '.0f'), f(w['tok_s_stream']), f(w['tok_s_timings']),
            f(statistics.mean(watts) if watts else None, '.2f')]))

    if L0 is not None:
        stop_t = L1 if L1 is not None else end
        lines += ['', f'WORK RATE per {WINDOW_S} s window of load (frames detected, 640 detect ms median, Gemma tok/s '
                      f'streamed / from timings of requests ending in the window)']
        k = 0
        while L0 + (k + 1) * WINDOW_S <= stop_t + 1e-6:
            w = work(frames, gens, L0 + k * WINDOW_S, L0 + (k + 1) * WINDOW_S)
            lines.append(f'  {k * WINDOW_S:>5}-{(k + 1) * WINDOW_S:<5} frames {w["frames"]:>3}  640ms {f(w["ms640"], ".0f"):>5}  '
                         f'tok/s {f(w["tok_s_stream"]):>5} / {f(w["tok_s_timings"]):>5}')
            k += 1
        dur = stop_t - L0
        a, b = work(frames, gens, L0, L0 + min(MINUTE_S, dur)), work(frames, gens, stop_t - min(MINUTE_S, dur), stop_t)
        lines += ['', f'SLOWDOWN (load ran {dur:.1f} s{"; first and last minute overlap" if dur < 2 * MINUTE_S else ""})']
        for label, w in (('first minute', a), ('last minute', b)):
            lines.append(f'  {label:12}: frames {w["frames"]}, 640 ms median {f(w["ms640"], ".0f")}, tok/s streamed '
                         f'{f(w["tok_s_stream"])}, tok/s timings {f(w["tok_s_timings"])} ({w["requests"]} requests)')

        lines += ['', 'COOLDOWN (start = latest reading before llama-server started, i.e. idle baseline; '
                      'time from the load stop until within 2 degC of it)']
        if L1 is None or not any(s['t'] > L1 for s in samples):
            lines.append('  no cooldown logged (the run ended at the load stop)')
        else:
            B = run.get('baseline_end', L0)
            sk0, s0 = at_or_before(good_skin, B), at_or_before([s for s in samples if s.get('complete')], B)
            after_s = [s for s in samples if s['t'] >= L1 and s.get('complete')]
            after_d = [d for d in good_skin if d['t'] >= L1]
            span = (samples[-1]['t'] - L1) if samples else 0
            for label, start, rows, get in (
                    ('skin', sk0 and sk0['skin'], after_d, lambda r: r['skin']),
                    ('BIG', s0 and c(s0['zones'][cz['BIG']]), after_s, lambda r: c(r['zones'][cz['BIG']]))):
                if start is None:
                    lines.append(f'  {label}: no start value')
                    continue
                hit = next((r for r in rows if get(r) <= start + 2), None)
                lines.append(f'  {label}: start {start:.1f}; ' + (
                    f'within 2 degC after {hit["t"] - L1:.1f} s ({get(hit):.1f})' if hit else
                    f'not reached in {span:.1f} s (last {f(get(rows[-1]) if rows else None)})'))
    lines += ['', f'STOP REASON: {"; ".join(reasons)}',
              '', 'Notes: skin = VIRTUAL-SKIN from `dumpsys thermalservice` "Current temperatures from HAL" (5 s, '
                  'DECISIONS #123 form); battery = power_supply/battery temp; W = -(current_now*voltage_now) as in '
                  'camera_heat.py; zones/freqs/cooling states read every 1 s from one root shell; tok/s streamed = '
                  'streamed tokens / window length, tok/s timings = sum predicted_n / sum predicted_ms of requests '
                  'ending in the window. Measurements only.']
    return '\n'.join(lines), bad


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--smoke', action='store_true', help=f'{SMOKE_LOAD_S} s load, {SMOKE_COOL_S} s cooldown')
    ap.add_argument('--note', default='', help='room temperature, how the phone is mounted')
    a = ap.parse_args(argv)
    cr.require_native()
    signal.signal(signal.SIGTERM, cr.exit_on_signal)  # SystemExit; deferred while a llama-server is being spawned
    signal.signal(signal.SIGHUP, cr.exit_on_signal)
    load_s, cool_s = (SMOKE_LOAD_S, SMOKE_COOL_S) if a.smoke else (LOAD_S, COOL_S)
    out = OUT_ROOT / f'run_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}{"_smoke" if a.smoke else ""}'
    out.mkdir(parents=True)
    t0 = time.monotonic()
    run = {'started': cr.utc(), 'note': a.note, 'smoke': a.smoke, 'argv': sys.argv, 'load_s': load_s, 'cool_s': cool_s,
           'stops': {'skin_c': SKIN_MAX_C, 'battery_c': BATTERY_MAX_C, 'status': STATUS_MAX, 'cpu_c': CPU_MAX_C,
                     'cpu_consecutive': CPU_N, 'skin_wait_s': SKIN_WAIT_S, 'sensor_wait_s': SENSOR_WAIT_S,
                     'frame_wait_s': FRAME_WAIT_S},
           'prompt': PROMPT, 'n_predict': N_PREDICT, 'python': sys.version, 'stop_reasons': None, 'stop_ok': False}
    save = lambda: (out / 'run.json').write_text(json.dumps(run, indent=1))
    fhs = {n: open(out / n, 'a') for n in LOGS}
    log_stop, load_stop = threading.Event(), threading.Event()
    samples, dumps, st = [], [], {'repeats': 0, 'last_frame_t': None, 'load_error': None}
    shell = server = None
    loggers, workers, bad = [], [], []
    try:
        rc, who = root_file('id', 'id')
        if rc != 0:
            raise SystemExit(f'su failed: {who.strip()}')
        cr.wait_cores('start')
        shell = RootShell()
        layout = discover(shell)
        keys = sample_keys(layout)
        run.update(layout=layout, sample_script=sample_script(keys), dump_cmd='dumpsys thermalservice',
                   server_cmd=cr.server_cmd(), model_bytes=os.path.getsize(cr.MODEL),
                   cpus_allowed=sorted(cr.allowed_cpus()),
                   sha256={'thermal_char.py': sha256(__file__), 'coresidency.py': sha256(cr.__file__),
                           'robotcam_reader.py': sha256(cr.robotcam_reader.__file__),
                           'detect_person.py': sha256(cr.ROBOT / 'detect_person.py'),
                           'server_manager.py': sha256(cr.server_manager.__file__),
                           'yolo11s_640.onnx': sha256(cr.ROBOT / 'yolo11s_640.onnx'),
                           'yolo11s_320.onnx': sha256(cr.ROBOT / 'yolo11s_320.onnx')})
        save()
        first = take_sample(shell, keys, sample_script(keys), layout, t0)
        if not first.get('complete'):
            raise SystemExit(f'first 1 s sample incomplete (CPU zones, battery temperature): {first}')
        print(f'output {out}; {len(layout["zones"])} zones, policies {list(layout["policies"])}, '
              f'{len(layout["cooling"])} cooling devices', flush=True)
        detector = cr.Detector()  # both sessions resident, as on the robot
        cr.camera_stop()  # start from a stopped service
        loggers = [threading.Thread(target=sample_loop, args=(shell, keys, layout, t0, log_stop, samples,
                                                              fhs['samples_1s.jsonl']), daemon=True),
                   threading.Thread(target=dump_loop, args=(t0, log_stop, dumps, fhs['thermalservice_5s.jsonl'],
                                                            fhs['thermalservice_raw.jsonl']), daemon=True)]
        for t in loggers:
            t.start()
        log_stop.wait(BASELINE_S)
        now = time.monotonic() - t0
        pre = [r for r in stop_reasons(now, now, samples, dumps, layout) if not r.startswith('fail closed')]
        if pre or not any(s.get('complete') for s in samples):
            run['stop_reasons'] = ['before load: ' + r for r in pre] or ['before load: no complete 1 s sample']
            run['stop_ok'] = False  # no load ran: not a measurement
            print(f'[load] not started: {run["stop_reasons"]}', flush=True)
        else:
            cr.check_cores('before the load')
            run['baseline_end'] = time.monotonic() - t0  # cooldown start values: readings before any load
            B = run['baseline_end']

            def check():  # every stop condition, fail-closed timers counted from the startup
                r = stop_reasons(time.monotonic() - t0, B, samples, dumps, layout)
                if r:
                    raise StartupStopped(r)
            cam = None
            try:
                with stops_checked(check):
                    server = cr.Server(out / 'llama-server.log')
                    cam = cr.camera_start()
            except StartupStopped as e:
                run['stop_reasons'], run['stop_ok'] = ['during startup: ' + r for r in e.args[0]], False
                print(f'[load] not started: {run["stop_reasons"]}', flush=True)
            if cam is not None:
                run['load_start'] = time.monotonic() - t0
                run['camera'] = cam
                st['last_frame_t'] = time.monotonic()
                print(f'[load] started (llama-server load {server.load_s:.2f} s)', flush=True)
                workers = [threading.Thread(target=frame_loop, args=(detector, cam['session'], t0, load_stop, st,
                                                                     fhs['frames.jsonl']), daemon=True),
                           threading.Thread(target=gen_loop, args=(t0, load_stop, st, fhs['gen.jsonl']), daemon=True)]
                for t in workers:
                    t.start()
                while True:
                    now = time.monotonic() - t0
                    reasons = stop_reasons(now, run['load_start'], samples, dumps, layout)
                    if st['load_error']:
                        reasons.append('load failed: ' + st['load_error'])
                    if time.monotonic() - st['last_frame_t'] > FRAME_WAIT_S:
                        reasons.append(f'load failed: no new RobotCam frame for {FRAME_WAIT_S} s')
                    if server.proc.poll() is not None:
                        reasons.append(f'load failed: llama-server exited ({server.proc.poll()})')
                    if not set(range(4, 8)) <= cr.allowed_cpus():
                        reasons.append(f'cores 4-7 lost (allowed {sorted(cr.allowed_cpus())})')
                    if not reasons and now >= run['load_start'] + load_s:
                        reasons, run['stop_ok'] = [f'planned load duration {load_s} s reached'], True
                    elif reasons:
                        run['stop_ok'] = all(not r.startswith(('fail closed', 'load failed', 'cores')) for r in reasons)
                    if reasons:
                        break
                    time.sleep(0.25)
                run['stop_reasons'] = reasons
                run['load_stop'] = time.monotonic() - t0
                print(f'[load] stopped at {run["load_stop"] - run["load_start"]:.1f} s: {"; ".join(reasons)}', flush=True)
                save()
                load_stop.set()
                cr.camera_stop()
                server.stop()
                for t in workers:
                    t.join(timeout=30)
                run['after_stop'] = {'robotcam': cr.camera_end_check(root_file),  # STOP ends capture asynchronously
                                     'llama_server_running': server.proc.poll() is None,
                                     'worker_threads_alive': [t.name for t in workers if t.is_alive()]}
                save()
                print(f'[cooldown] logging {cool_s} s', flush=True)
                log_stop.wait(cool_s)
    except BaseException as e:  # signal (SystemExit 143/129), Ctrl-C or error: recorded, then cleanup below
        now = time.monotonic() - t0
        if run.get('load_start') is not None and run.get('load_stop') is None:
            run['load_stop'] = now
        if run['stop_reasons'] is None or run.get('load_start') is not None and 'after_stop' not in run:
            run['stop_reasons'] = (run['stop_reasons'] or []) + [f'aborted: {type(e).__name__}: {e}']
            run['stop_ok'] = False
        raise
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        load_stop.set()
        log_stop.set()
        for s in list(cr.LIVE):  # any llama-server still running, whoever started it
            s.stop()
        try:
            cr.camera_stop()
        except (OSError, subprocess.SubprocessError) as e:
            print(f'[CAM] stop failed: {e}', file=sys.stderr)
        for t in loggers + workers:
            t.join(timeout=30)
        if run.get('baseline_end') is not None and 'after_stop' not in run:  # stopped or aborted in startup/load
            try:
                run['after_stop'] = {'robotcam': cr.camera_end_check(root_file),
                                     'llama_server_running': bool(cr.LIVE) or bool(server and server.proc.poll() is None),
                                     'worker_threads_alive': [t.name for t in workers if t.is_alive()]}
            except (OSError, subprocess.SubprocessError) as e:
                run['after_stop'] = {'check_failed': str(e)}
        if shell:
            shell.close()
        for fh in fhs.values():
            fh.close()
        run['finished'] = cr.utc()
        save()
        if 'layout' in run:
            try:
                text, bad = report(out)
                (out / 'report.txt').write_text(text + '\n')
                print(text, flush=True)
                dest = DOWNLOADS / f'thermal_char_{out.name}_report.txt'
                if DOWNLOADS.is_dir():
                    shutil.copy(out / 'report.txt', dest)
                    print(f'copied to {dest}', flush=True)
                else:
                    print(f'WARNING: {DOWNLOADS} missing; report only in {out}', flush=True)
            except Exception as e:  # never hide the original error; the raw logs remain
                bad = [f'report failed: {type(e).__name__}: {e}']
                print(bad[0], file=sys.stderr, flush=True)
    if bad:
        raise SystemExit('RUN INCOMPLETE: ' + '; '.join(bad))


if __name__ == '__main__':
    main()
