#!/data/data/com.termux/files/usr/bin/python
"""MIND-B4: timed baselines measured alone (no agent resident): one picker decision, one vision call, RobotCam+YOLO,
idle, and the cost of 5 s monitoring. Research only: no motors (motors.py is never imported), no serial port, no
mission, no MTP. Timings count only from an owner run started by oneshot.sh in native Termux.

Server (P, P0, PT, V, PY): BC's b11716 command, bc.command(<run>/slots) = mem1.command(2048) (setup_q4 flags) with
the b11716 llama-server, --slot-save-path <run>/slots and VIS-3D's projector flags (Q8_0, --image-min/max-tokens 70);
pins bc.PINS. One server per block, started after the gate and stopped at the block end; load and warm-up are not
timed.
Picker: B2 V3 unchanged (b2.preflight / b2.score: rules, examples, reason sentence, two-step scoring), no memory, on
B2 TEST items 1-20 (b2_rules/set/b2_test.json, SHA-256 pinned) in FWD order, repeated. Before the block: the 20
prompts are rendered and checked (b2.preflight), the common prefix of their ids (the system turn) is sent once
(n_predict 1, cache_prompt true) and slot 0 is saved. Before every 5th decision (5th, 10th, ...) slot 0 is restored.
Vision: BV ids FRAME_IDS (frame SHA-256 checked against results.json), bv.body('V70'): BV's prompt and settings, one
/v1/chat/completions call each, in that order, repeated.
Camera: main.Robot(motors=None).start_camera() (CAMERA_RATE 1, PinnedDetector on cpus 4-5, CAM1-MID #136) and
stop_camera(); each cycle = run_cycle's vision step (a new frame of this session, SizePolicy size 640 every 5 s else
320, detect, observe) followed by the dry run's move time (main.CYCLE_MOVE_TIME: motors None sleeps it); no
navigation rules, picker, M2 or heat control. A frame that is not new is recorded as a camera stop (the mission would
stop) and read again after 0.1 s. After stop_camera, capture must show stopped (frame files gone) within
CAMERA_STOP_S.

Blocks, in ORDER, at most BLOCK_S s of work each (a decision or call is not started when the previous one would not
fit): I idle (screen on), P picker, P0 two cold decisions (item 1 after /slots erase = full prefix rebuild, then item
1 after slot restore; seconds only, not a heat block), PT = P plus the root sampler (every SAMPLE_S s: skin, status,
CPU caps, battery; thread asking for cores 0-3 at nice 10 as MEM-1's mem1.pin_sampler, what it got is recorded), V
vision, Y camera+YOLO without llama-server, PY = a P phase, then right away (no gate) a Y phase with the server loaded
and idle.
Before every block: cores 4-7 (cr.wait_cores, then the check), agent check (repro.agents), MEM-1's skin gate
(pm.thermal_gate: VIRTUAL-SKIN <= launch-start skin + 1.0 degC, waiting idle at most pm.GATE_MAX_S; the wait is
recorded), Android status < 3 and battery < 45 degC. A gate not passed, cores not available or an agent found ends
the session before the block (it runs in a later session). Start and end of each block (PY also at the hand-off):
reading(): VIRTUAL-SKIN, Android thermal status, cpufreq caps policy0/4/6, battery %, temp, charge counter and
voltage, PSS/Rss/VmHWM/VmRSS of llama-server, the runner (detector) and RobotCam, MemAvailable.
Labels: block VALID only with every check clean (block_problems: agent, cores, readings, request/camera errors,
sampler); an INVALID block ends the session and is redone in a later one. Server exit code != 0 after SIGTERM is a
FINDING (#145), not a validity input. P vs PT COMPARABLE only when both are VALID with start skin within 1.0 degC and
battery within 10 % (comparable()); other cross-block numbers are descriptive. Heat rate = (end skin - start skin) /
minutes. Energy from the charge counter is INDICATIVE.

Usage: b4.py                 a new run, first session (native Termux, via oneshot.sh)
       b4.py --resume RUN    the next session of RUN (its folder name in ~/robot_runs/mind_b4)
       b4.py --smoke         every block once for SMOKE_S s, no gate, checks recorded but not enforced;
                             SMOKE - NOT A MEASUREMENT (also from proot Debian, where root reads and the camera fail)
Writes ~/robot_runs/mind_b4/<UTC>[_smoke]/: run.json, sessionNN.json, sessionNN_end.json, blocks.jsonl,
requests.jsonl, NN_<block>_server.log, slots/, report.txt, artifacts.sha256 (the last two rewritten per session).
"""
import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE.parent / 'bc_build'), str(REPO / 'benchmark/gemma_vision/bv_compare'), str(REPO)]
import bc  # noqa: E402  server command and pins; imports b0, b0b, b2, mem1, vis3d, coresidency
import bv  # noqa: E402  vision prompt and payload, frames

b0, b0b, b2, mem1, cr, vis3d, sha = bc.b0, bc.b0b, bc.b2, bc.mem1, bc.cr, bc.vis3d, bc.sha
pm, cm, repro = mem1.pm, mem1.cm, mem1.repro

OUT_ROOT = cr.HOME / 'robot_runs' / 'mind_b4'
THERMAL_LOG = OUT_ROOT / 'thermal.log'  # oneshot.sh's logger; the skin gate (pm.thermal_gate) needs it
ORDER = ('I', 'P', 'P0', 'PT', 'V', 'Y', 'PY')
SERVER_BLOCKS = ('P', 'P0', 'PT', 'V', 'PY')
BLOCK_S, SMOKE_S = 180, 20
ITEMS = tuple(range(1, 21))
FRAME_IDS = (1, 5, 6, 7, 9, 10, 16, 24, 30, 33)
RESTORE_EVERY = 5
SAMPLE_S, SAMPLE_MAX_GAP_S = 5, 10
ROOT_TIMEOUT_S = 15
CAMERA_STOP_S = 10
POLICIES = ('policy0', 'policy4', 'policy6')
SKIN_MATCH_C, BATTERY_MATCH_PCT = mem1.SKIN_MATCH_C, mem1.BATTERY_MATCH_PCT
STATUS_STOP, BATTERY_MAX_C = mem1.STATUS_STOP, repro.BATTERY_MAX_C
ROBOTCAM = 'com.pixelrobot.robotcam'
SMOKE_LABEL = 'SMOKE - NOT A MEASUREMENT'


# ---------------------------------------------------------------- readings

def caps():
    out = {}
    for p in POLICIES:
        try:
            out[p] = int(Path(f'/sys/devices/system/cpu/cpufreq/{p}/scaling_max_freq').read_text())
        except (OSError, ValueError) as e:
            out[p] = f'error: {e}'
    return out


def root_script(pids, light):
    """One su call (DECISIONS #123 form via cr.root): its shell asks for cores 0-3 and nice 10; battery, dumpsys
    thermalservice and, unless light, memory of each pid and of every RobotCam process."""
    b = mem1.BATTERY
    s = ('taskset -p f $$ >/dev/null 2>&1; renice -n 10 -p $$ >/dev/null 2>&1; '
         f'echo "=== battery"; cat {b}/capacity {b}/status {b}/temp {b}/charge_counter {b}/voltage_now; echo "rc=$?"; '
         'echo "=== thermalservice"; dumpsys thermalservice; echo "rc=$?"; ')
    if light:
        return s
    mem = 'grep -hE "^(Pss|Rss):" /proc/{0}/smaps_rollup || r=1; grep -hE "^Vm(HWM|RSS):" /proc/{0}/status || r=1; '
    for name, pid in pids.items():
        s += f'echo "=== mem_{name}"; r=0; echo "pid: {int(pid)}"; {mem.format(int(pid))}echo "rc=$r"; '
    return s + (f'echo "=== mem_robotcam"; r=0; for p in $(pidof {ROBOTCAM}); do echo "pid: $p"; {mem.format("$p")}'
                'done; echo "rc=$r"')


def mem_section(text):
    """kB summed over the section's processes, and their pids."""
    m = {'pids': []}
    for line in text.splitlines():
        k, _, v = line.partition(':')
        if k == 'pid':
            m['pids'].append(int(v))
        elif k in ('Pss', 'Rss', 'VmHWM', 'VmRSS') and v.split():
            m[k] = m.get(k, 0) + int(v.split()[0])
    return m


def reading(pids=None, light=False, tag='b4_read'):
    """Skin, status, caps, battery (+ charge counter, voltage); unless light, MemAvailable and process memory."""
    c = {'utc': cr.utc(), 'mono': round(time.monotonic(), 3), 'caps': caps()}
    if not light:
        c['avail_kb'] = next(int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()
                                   if l.startswith('MemAvailable'))
    began = time.monotonic()
    try:
        rc, out = cr.root(root_script(pids or {}, light), tag, timeout=ROOT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as e:
        rc, out = None, ''
        c['root_error'] = f'{type(e).__name__}: {e}'[:300]
    c.update(root_rc=rc, root_s=round(time.monotonic() - began, 3))
    sec = mem1.sections(out)
    text = sec.get('battery', (None, ''))[1]
    try:
        pct, status, temp, charge, volt = text.split()[:5]
        c.update(battery_pct=int(pct), battery_status=status, battery_c=int(temp) / 10, charge_uah=int(charge),
                 voltage_uv=int(volt))
    except ValueError:
        c['battery_error'] = text[:300]
    d = cr.parse_dump(sec.get('thermalservice', (None, ''))[1])
    c.update(status=d['status'], skin=d['skin'])
    for name, (rc_m, text) in sec.items():
        if name.startswith('mem_'):
            c[name] = {**mem_section(text), 'rc': rc_m}
    return c


def reading_problems(c, label, need=()):
    why = []
    if c.get('root_rc') != 0:
        why.append(f'{label}: root rc {c.get("root_rc")} {c.get("root_error", "")}'.strip())
    why += [f'{label}: {k} missing' for k in ('skin', 'status', 'battery_pct', 'charge_uah') if c.get(k) is None]
    why += [f'{label}: {p} cap {v}' for p, v in c.get('caps', {}).items() if not isinstance(v, int)]
    why += [f'{label}: {n} memory missing' for n in need if not c.get(f'mem_{n}', {}).get('Pss')]
    return why


class Sampler:
    """PT's root sampler: a reading(light) every period s from a thread on cores 0-3 at nice 10 (mem1.pin_sampler;
    what it got is recorded). A slow sample skips the missed slots."""

    def __init__(self, read=None, period=SAMPLE_S):
        self.read = read or (lambda: reading(light=True, tag='b4_sample'))
        self.period, self.samples, self.sched, self.error = period, [], None, None
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def loop(self):
        try:
            self.sched = mem1.pin_sampler()
            t0 = time.monotonic()
            while not self.halt.is_set():
                self.samples.append(self.read())
                k = int((time.monotonic() - t0) / self.period) + 1
                self.halt.wait(max(0.0, t0 + k * self.period - time.monotonic()))
        except Exception as e:
            self.error = f'{type(e).__name__}: {e}'

    def stop(self):
        self.halt.set()
        self.thread.join(ROOT_TIMEOUT_S + 5)
        return {'period_s': self.period, 'sched': self.sched, 'error': self.error, 'alive': self.thread.is_alive(),
                'samples': self.samples}


def sampler_problems(s, start_mono, end_mono):
    why = [f'sampler: {s["error"]}'] if s['error'] else []
    why += ['sampler: still running after its join'] if s['alive'] else []
    ok = [x['mono'] for x in s['samples'] if not reading_problems(x, 'sample')]
    if len(ok) < len(s['samples']):
        why.append(f'sampler: {len(s["samples"]) - len(ok)} of {len(s["samples"])} samples failed')
    marks = [start_mono] + [m for m in ok if start_mono <= m <= end_mono] + [end_mono]
    gap = max(b - a for a, b in zip(marks, marks[1:]))
    if gap > SAMPLE_MAX_GAP_S:
        why.append(f'sampler: gap {gap:.1f} s > {SAMPLE_MAX_GAP_S} s')
    return why


# ---------------------------------------------------------------- checks before a block

def cores_ok():
    return set(range(4, 8)) <= cr.allowed_cpus()


def gate(ref_skin, label, smoke):
    """MEM-1's skin gate (pm.thermal_gate with cr.SKIN_GATE_C = 1.0); smoke: not applied."""
    if smoke:
        return {'skipped': SMOKE_LABEL, 'passed': True}
    g = pm.thermal_gate(THERMAL_LOG, {'skin': ref_skin}, label, False)
    return {**g, 'passed': not g['warm_start']}


def comparable(p, pt):
    """P vs PT: (COMPARABLE | NOT COMPARABLE, reasons)."""
    why = [f'{r["block"]} {r["label"]}' for r in (p, pt) if r['label'] != 'VALID']
    if not why:
        a, b = p['start'], pt['start']
        if abs(a['skin'] - b['skin']) > SKIN_MATCH_C:
            why.append(f'start skin {a["skin"]} vs {b["skin"]} degC differ by more than {SKIN_MATCH_C}')
        if abs(a['battery_pct'] - b['battery_pct']) > BATTERY_MATCH_PCT:
            why.append(f'battery {a["battery_pct"]} vs {b["battery_pct"]} % differ by more than {BATTERY_MATCH_PCT}')
    return ('NOT COMPARABLE' if why else 'COMPARABLE'), why


# ---------------------------------------------------------------- server and picker

def post(path, body):
    return cr.post(path, body, timeout=b0.REQUEST_TIMEOUT_S)


def start_server(out, name):
    """(process, log file, load seconds): bc.command, started by this process only."""
    log = open(out / f'{name}_server.log', 'x')
    t0 = time.monotonic()
    proc = subprocess.Popen(bc.command(out / 'slots'), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    while not cr.healthy():
        if proc.poll() is not None or time.monotonic() - t0 > b0.LOAD_TIMEOUT_S:
            stop_server(proc, log)
            raise RuntimeError(f'llama-server not healthy (exit {proc.poll()})')
        time.sleep(0.1)
    return proc, log, round(time.monotonic() - t0, 3)


def stop_server(proc, log):
    """(exit code, SIGKILL needed)."""
    killed = False
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=b0.TERM_WAIT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            killed = True
    log.close()
    return proc.returncode, killed


def slot(action):
    t0 = time.perf_counter()
    res = post(f'/slots/0?action={action}', {'filename': b0b.SLOT_FILE})
    return {'action': action, 'wall_s': round(time.perf_counter() - t0, 3), 'response': res}


def picker_setup(test):
    """(set of items 1-20, prompts, system prefix ids): b2.preflight's V3 FWD prompts; any problem raises."""
    s = {**test, 'items': [it for it in test['items'] if it['id'] in ITEMS]}
    info = {'problems': []}
    prompts = b2.preflight(post, s, [('V3', 'FWD')], info, 'TEST')
    if info['problems'] or len(s['items']) != len(ITEMS):
        raise RuntimeError(f'picker preflight: {info["problems"]} ({len(s["items"])} items)')
    ids = [prompts['V3', 'FWD', it['id']][1] for it in s['items']]
    return s, prompts, ids[0][:min(mem1.lcp(ids[0], x) for x in ids[1:])]


def warm(prefix):
    """The system prefix once (not timed), then slot 0 saved."""
    res = post('/completion', {'prompt': prefix, **b0.DECODE, 'n_predict': 1, 'cache_prompt': True})
    tm = res.get('timings', {})
    return {'prefix_tokens': len(prefix), 'prompt_n': tm.get('prompt_n'), 'cache_n': tm.get('cache_n'),
            'save': slot('save')}


class Recorder:
    """post() for one decision: keeps every /completion response (V3 sends two) and the time spent per path."""

    def __init__(self):
        self.completions, self.seconds = [], {}

    def __call__(self, path, body):
        t0 = time.perf_counter()
        res = post(path, body)
        self.seconds[path] = round(self.seconds.get(path, 0) + time.perf_counter() - t0, 3)
        if path == '/completion':
            self.completions.append(res)
        return res


def decide(s, it, prompt):
    """One B2 V3 decision (b2.score unchanged): wall, reason tokens, step-1 generation tok/s, both steps' prompt_n and
    cache_n; seconds per endpoint (b2's /tokenize checks are harness overhead, reported separately)."""
    rec, row = Recorder(), {'id': it['id']}
    t0 = time.perf_counter()
    try:
        r = b2.score(rec, it, 'V3', 'FWD', prompt)
        t1, t2 = (x.get('timings', {}) for x in rec.completions)
        row.update(top=r['top'], reason=r['reason'], reason_tokens=r['reason_tokens'],
                   gen_tok_s=t1.get('predicted_per_second'), step1_prompt_n=t1.get('prompt_n'),
                   step1_cache_n=t1.get('cache_n'), step1_prompt_ms=t1.get('prompt_ms'),
                   step2_prompt_n=t2.get('prompt_n'), step2_cache_n=t2.get('cache_n'), step2_prompt_ms=t2.get('prompt_ms'))
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'
    row.update(wall_s=round(time.perf_counter() - t0, 3), endpoint_s=rec.seconds)
    return row


def run_picker(ps, dur, emit):
    """Decisions on items 1-20 in order, repeated, until the next would not fit in dur; restore before every 5th."""
    s, prompts = ps[0], ps[1]
    t0, n, last = time.monotonic(), 0, 0.0
    while n == 0 or time.monotonic() - t0 + last <= dur:
        it = s['items'][n % len(s['items'])]
        row = {'kind': 'decision', 'n': n + 1}
        if (n + 1) % RESTORE_EVERY == 0:
            try:
                row['restore'] = slot('restore')
            except Exception as e:
                row['error'] = f'restore: {type(e).__name__}: {e}'
        if 'error' not in row:
            row.update(decide(s, it, prompts['V3', 'FWD', it['id']]))
        row.update(t=round(time.monotonic() - t0, 3), cores_ok=cores_ok())
        emit(row)
        if 'error' in row:
            return
        last = row['wall_s'] + row.get('restore', {}).get('wall_s', 0)
        n += 1


def run_p0(ps, emit):
    """Item 1 after /slots erase (full prefix rebuild), then item 1 after slot restore."""
    s, prompts = ps[0], ps[1]
    it = s['items'][0]
    for kind, action in (('cold_erase', 'erase'), ('cold_restore', 'restore')):
        row = {'kind': kind}
        try:
            row[action] = slot(action)
            row.update(decide(s, it, prompts['V3', 'FWD', it['id']]))
        except Exception as e:
            row['error'] = f'{action}: {type(e).__name__}: {e}'
        row['cores_ok'] = cores_ok()
        emit(row)
        if 'error' in row:
            return


def frames():
    """{id: (results.json record, base64 JPEG)} for FRAME_IDS; a frame whose SHA-256 differs raises."""
    names = sorted(p.name for p in bv.FRAMES_DIR.glob('*.jpg'))
    ph = bv.photos(json.loads(bv.RESULTS.read_bytes()), names)
    out = {}
    for i in FRAME_IDS:
        data = (bv.FRAMES_DIR / Path(ph[i]['frame_file']).name).read_bytes()
        if sha(data) != ph[i]['frame_sha256']:
            raise SystemExit(f'refused: frame id {i} SHA-256 != results.json')
        out[i] = (ph[i], base64.b64encode(data).decode())
    return out


def run_vision(fr, dur, emit):
    t0, n, last = time.monotonic(), 0, 0.0
    while n == 0 or time.monotonic() - t0 + last <= dur:
        i = FRAME_IDS[n % len(FRAME_IDS)]
        row = {'kind': 'vision', 'n': n + 1, 'id': i}
        c0 = time.perf_counter()
        try:
            res = post('/v1/chat/completions', bv.body('V70', *fr[i]))
            tm = res.get('timings', {})
            row.update(content=res['choices'][0]['message'].get('content'), prompt_ms=tm.get('prompt_ms'),
                       prompt_n=tm.get('prompt_n'), cache_n=tm.get('cache_n'), predicted_n=tm.get('predicted_n'),
                       gen_tok_s=tm.get('predicted_per_second'))
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'
        row.update(wall_s=round(time.perf_counter() - c0, 3), t=round(time.monotonic() - t0, 3), cores_ok=cores_ok())
        emit(row)
        if 'error' in row:
            return
        last = row['wall_s']
        n += 1


# ---------------------------------------------------------------- camera

def start_camera(rec):
    """main.Robot(motors=None).start_camera(): pinned detector (created once per process), RobotCam at CAMERA_RATE."""
    import main  # noqa: E402  imported here: never motors.py (main imports it nowhere; run_mission only when live)
    robot = main.Robot(None)
    t0 = time.monotonic()
    robot.start_camera()
    rec['camera'] = {'start_s': round(time.monotonic() - t0, 3), 'rate': main.CAMERA_RATE,
                     'detector': robot.detector.config(), 'affinity_start': robot.detector.affinity()}
    return robot


def stop_camera(robot):
    """main.Robot.stop_camera(), then capture must show stopped (frame files gone) within CAMERA_STOP_S."""
    import main
    t0 = time.monotonic()
    out = {'affinity_end': robot.detector.affinity()}
    try:
        robot.stop_camera()
    except Exception as e:
        return {**out, 'error': f'stop_camera: {type(e).__name__}: {e}'}
    while time.monotonic() - t0 < CAMERA_STOP_S:
        status = main.read_frame()['status']
        if status == 'missing':
            return {**out, 'stopped_s': round(time.monotonic() - t0, 2)}
        time.sleep(0.2)
    return {**out, 'error': f'capture not shown stopped {CAMERA_STOP_S} s after stop_camera (status {status})'}


def run_camera(robot, dur, emit):
    """run_cycle's vision step and the dry run's move time; a frame that is not new is a camera stop."""
    import main
    t0, last = time.monotonic(), None
    while time.monotonic() - t0 < dur:
        c0 = time.monotonic()
        r = main.read_frame(session=robot.camera_session)
        if r['status'] != 'ok' or (last is not None and r['frame'] <= last):
            emit({'kind': 'camera_stop', 'reason': r['status'] if r['status'] != 'ok' else 'no new',
                  't': round(c0 - t0, 3)})
            time.sleep(0.1)
            continue
        last, robot.camera_session = r['frame'], r['session']
        size = robot.size_policy.next_size()
        row = {'kind': 'cycle', 't': round(c0 - t0, 3), 'frame': r['frame'], 'age_s': r['age_s'], 'size': size}
        try:
            d0 = time.perf_counter()
            dets = robot.detector.detect(r['image'], size)
            row.update(detect_ms=round((time.perf_counter() - d0) * 1000, 1), n_det=len(dets))
            robot.size_policy.observe(dets, r['image'].height)
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'
        row['cores_ok'] = cores_ok()
        emit(row)
        if 'error' in row:
            return
        time.sleep(main.CYCLE_MOVE_TIME)


# ---------------------------------------------------------------- one block

def block_problems(rec, rows):
    """Why a block is INVALID (empty: VALID)."""
    need = {'start': ['runner'], 'end': ['runner'], 'handoff': ['runner']}
    if rec['block'] in SERVER_BLOCKS:
        for k in need:
            need[k].append('llama_server')
    if rec['block'] in ('Y', 'PY'):
        need['end'].append('robotcam')
        need['handoff' if rec['block'] == 'PY' else 'start'].append('robotcam')
    why = list(rec['errors'])
    for k in ('start', 'handoff', 'end'):
        if k in rec:
            why += reading_problems(rec[k], k, need[k])
        elif k != 'handoff' or rec['block'] == 'PY':
            why.append(f'{k} reading missing')
    if rec.get('agents_before') or rec.get('agents_after'):
        why.append(f'agent process running: {(rec.get("agents_before") or rec.get("agents_after"))[:120]!r}')
    if not rec.get('cores_before') or not rec.get('cores_after') or any(r.get('cores_ok') is False for r in rows):
        why.append('cores 4-7 not all allowed during the block')
    if n := sum('error' in r for r in rows):
        why.append(f'{n} request/cycle errors')
    if rec['block'] in ('P', 'PT', 'PY') and not any(r['kind'] == 'decision' for r in rows):
        why.append('no decisions')
    if rec['block'] == 'P0' and [r['kind'] for r in rows] != ['cold_erase', 'cold_restore']:
        why.append('P0 needs both cold decisions')
    if rec['block'] == 'V' and not rows:
        why.append('no vision calls')
    if rec['block'] in ('Y', 'PY') and not any(r['kind'] == 'cycle' for r in rows):
        why.append('no camera cycles')
    if rec.get('camera_stop', {}).get('error'):
        why.append(rec['camera_stop']['error'])
    if 'sampler' in rec:
        why += sampler_problems(rec['sampler'], rec.get('start', {}).get('mono', 0), rec.get('end', {}).get('mono', 0))
    return why


def run_block(name, out, ctx, emit):
    """One block; its rows go to requests.jsonl through emit, its record is returned."""
    rec = {'block': name, 'session': ctx['session'], 'utc_start': cr.utc(), 'work_s': ctx['dur'], 'errors': [],
           'cores_before': cores_ok(), 'agents_before': repro.agents()}
    rows = []

    def add(row):
        row = {'block': name, 'session': ctx['session'], **row}
        rows.append(row)
        emit(row)
    proc = log = robot = sampler = None
    pids = {'runner': os.getpid()}
    try:
        if name in SERVER_BLOCKS:
            proc, log, rec['load_s'] = start_server(out, f'{ctx["session"]:02d}_{name}')
            pids['llama_server'] = proc.pid
            if name != 'V':
                ps = picker_setup(ctx['test'])
                rec['warm'] = warm(ps[2])
        if name == 'Y':
            robot = start_camera(rec)
        rec['start'] = reading(pids)
        if name == 'I':
            time.sleep(ctx['dur'])
        elif name in ('P', 'PT'):
            sampler = Sampler().start() if name == 'PT' else None
            run_picker(ps, ctx['dur'], add)
        elif name == 'P0':
            run_p0(ps, add)
        elif name == 'V':
            run_vision(ctx['frames'], ctx['dur'], add)
        elif name == 'Y':
            run_camera(robot, ctx['dur'], add)
        elif name == 'PY':
            run_picker(ps, ctx['dur'], add)
            if not any('error' in r for r in rows):
                robot = start_camera(rec)
                rec['handoff'] = reading(pids)
                run_camera(robot, ctx['dur'], add)
        rec['end'] = reading(pids)
    except BaseException as e:  # recorded; cleanup below; a stop (signal) re-raises after the record is kept
        rec['errors'].append(f'{type(e).__name__}: {e}')
        if not isinstance(e, Exception):
            rec['interrupted'] = True
    finally:
        if sampler:
            rec['sampler'] = sampler.stop()
        if robot:
            rec['camera_stop'] = stop_camera(robot)
        if proc:
            rec['exit_code'], rec['killed'] = stop_server(proc, log)
        rec.update(utc_end=cr.utc(), cores_after=cores_ok(), agents_after=repro.agents())
    rec['problems'] = block_problems(rec, rows)
    rec['label'] = SMOKE_LABEL if ctx['smoke'] else 'VALID' if not rec['problems'] else 'INVALID'
    return rec, rows


# ---------------------------------------------------------------- report

def heat(a, b):
    """(skin change degC, minutes, degC/min) between two readings."""
    if not (a and b) or a.get('skin') is None or b.get('skin') is None:
        return None
    mins = (b['mono'] - a['mono']) / 60
    return round(b['skin'] - a['skin'], 2), round(mins, 2), round((b['skin'] - a['skin']) / mins, 3) if mins else None


def energy(a, b):
    """INDICATIVE: (charge used mAh, J at the mean voltage) from the charge counter."""
    try:
        q = (a['charge_uah'] - b['charge_uah']) / 1000
        return round(q, 1), round(q * 3.6 * (a['voltage_uv'] + b['voltage_uv']) / 2e6, 0)
    except (KeyError, TypeError):
        return None


def stat(xs, nd=0):
    xs = [x for x in xs if x is not None]
    return f'median {round(mem1.med(xs), nd)} p95 {round(mem1.p95(xs), nd)} max {round(max(xs), nd)} (n {len(xs)})' \
        if xs else 'n 0'


def mem_text(c):
    if not c:
        return '-'
    parts = [f'{n[4:]} Pss {m.get("Pss")} HWM {m.get("VmHWM")}' for n, m in c.items() if n.startswith('mem_')]
    return '; '.join(parts) + f'; MemAvailable {c.get("avail_kb")} kB'


def reading_text(c):
    if not c:
        return '-'
    return (f'skin {c.get("skin")} status {c.get("status")} caps {c.get("caps")} battery {c.get("battery_pct")} % '
            f'{c.get("battery_c")} degC charge {c.get("charge_uah")} uAh')


def detail(rec, rows):
    k = rec['block']
    L = []
    dec = [r for r in rows if r['kind'] == 'decision' and 'error' not in r]
    if dec:
        L += [f'  decisions {len(dec)}: wall s {stat([r["wall_s"] for r in dec], 3)}',
              f'    excl. b2 /tokenize checks: {stat([r["wall_s"] - r["endpoint_s"].get("/tokenize", 0) for r in dec], 3)}',
              f'    reason tokens {stat([r["reason_tokens"] for r in dec])}; gen tok/s {stat([r["gen_tok_s"] for r in dec], 2)}',
              f'    step1 prompt_n {stat([r["step1_prompt_n"] for r in dec])} cache_n {stat([r["step1_cache_n"] for r in dec])}',
              f'    step2 prompt_n {stat([r["step2_prompt_n"] for r in dec])} cache_n {stat([r["step2_cache_n"] for r in dec])}',
              f'    restores {len([r for r in dec if "restore" in r])}: wall s '
              f'{stat([r["restore"]["wall_s"] for r in dec if "restore" in r], 3)}']
    if k == 'P0':
        L += [f'  {r["kind"]}: wall {r.get("wall_s")} s, step1 prompt_n {r.get("step1_prompt_n")} cache_n '
              f'{r.get("step1_cache_n")}, slot {r.get("erase", r.get("restore", {})).get("wall_s")} s '
              f'{r.get("erase", r.get("restore", {})).get("response")}' + (f' ERROR {r["error"]}' if 'error' in r else '')
              for r in rows]
    vis = [r for r in rows if r['kind'] == 'vision' and 'error' not in r]
    if vis:
        L += [f'  vision calls {len(vis)}: wall s {stat([r["wall_s"] for r in vis], 3)}',
              f'    prompt ms {stat([r["prompt_ms"] for r in vis])}; gen tok/s {stat([r["gen_tok_s"] for r in vis], 2)}']
    cyc = [r for r in rows if r['kind'] == 'cycle' and 'error' not in r]
    if cyc:
        ts = [r['t'] for r in cyc]
        L += [f'  camera cycles {len(cyc)}: period s {stat([b - a for a, b in zip(ts, ts[1:])], 3)}; max frame age '
              f'{max(r["age_s"] for r in cyc):.2f} s; camera stops {sum(r["kind"] == "camera_stop" for r in rows)}'] + \
             [f'    detect {z} ms {stat([r["detect_ms"] for r in cyc if r["size"] == z])}' for z in (320, 640)]
    if 'camera' in rec:
        L.append(f'  camera start {rec["camera"]["start_s"]} s, affinity start {rec["camera"]["affinity_start"]}; '
                 f'stop {rec.get("camera_stop")}')
    if 'sampler' in rec:
        s = rec['sampler']
        L.append(f'  sampler: {len(s["samples"])} samples, root s {stat([x["root_s"] for x in s["samples"]], 2)}, '
                 f'sched {s["sched"]}, error {s["error"]}')
    if 'warm' in rec:
        L.append(f'  load {rec.get("load_s")} s; warm prefix {rec["warm"]["prefix_tokens"]} tokens (prompt_n '
                 f'{rec["warm"]["prompt_n"]}), save {rec["warm"]["save"]["wall_s"]} s')
    if 'exit_code' in rec:
        L.append(f'  server exit {rec["exit_code"]}, SIGKILL {rec["killed"]}' +
                 ('' if rec['exit_code'] == 0 and not rec['killed'] else '  FINDING (#145)'))
    return L


def report(run, recs, rows):
    L = [f'MIND-B4 {run["utc"]}  code {run["code_sha256"][Path(__file__).name][:12]}  base {run["git_head"]}',
         f'server: {" ".join(run["command"])}', f'launch-start skin {run["start"].get("skin")} degC (gate reference)']
    if run['smoke']:
        L.append(f'*** {SMOKE_LABEL}: {run["work_s"]} s blocks, no gate, checks not enforced ***')
    L += ['timings: VALID blocks only, from the owner run with no agent resident; energy INDICATIVE', '']
    final = {}
    for r in recs:
        if r['label'] in ('VALID', SMOKE_LABEL) or r['block'] not in final:
            final[r['block']] = r
    for k in ORDER:
        tries = [r for r in recs if r['block'] == k]
        if k not in final:
            L += [f'== {k}: not run', '']
            continue
        r = final[k]
        rr = [x for x in rows if x['block'] == k and x['session'] == r['session']]
        phases = [('', r.get('start'), r.get('end'))] if k != 'PY' else \
            [(' P phase', r.get('start'), r.get('handoff')), (' Y phase', r.get('handoff'), r.get('end'))]
        L.append(f'== {k}: {r["label"]} (session {r["session"]}; {len(tries)} attempt(s); gate {r.get("gate")})')
        L += [f'  problems: {"; ".join(r["problems"])}'] if r['problems'] else []
        for ph, a, b in phases:
            L += [f' {ph} start: {reading_text(a)}', f' {ph}   end: {reading_text(b)}',
                  f' {ph} heat (change, min, degC/min): {heat(a, b)}; energy (mAh, J) INDICATIVE: {energy(a, b)}',
                  f' {ph} RAM start: {mem_text(a)}', f' {ph} RAM   end: {mem_text(b)}']
        L += detail(r, rr) + ['']
    if 'P' in final and 'PT' in final:
        lab, why = comparable(final['P'], final['PT'])
        w = {k: mem1.med([x['wall_s'] for x in rows if x['block'] == k and x['session'] == final[k]['session']
                          and x['kind'] == 'decision' and 'error' not in x]) for k in ('P', 'PT')}
        L.append(f'P vs PT (monitoring cost): {lab}' + (f' ({"; ".join(why)})' if why else '') +
                 f'; median decision wall P {w["P"]} s, PT {w["PT"]} s' +
                 (f', PT/P {w["PT"] / w["P"]:.3f}' if w['P'] and w['PT'] else ''))
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- sessions

def next_blocks(recs):
    """Blocks still to run, in ORDER: those without a VALID record (an INVALID one is redone)."""
    done = {r['block'] for r in recs if r['label'] == 'VALID'}
    return [k for k in ORDER if k not in done]


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def code_files():
    import main  # noqa: F401  its file is part of the camera code
    names = ('b0', 'b0b', 'b1', 'b2', 'bc', 'bv', 'mem1', 'vis3d', 'coresidency', 'power_map', 'repro', 'chat_mtp',
             'main', 'detect_person', 'detector_size_policy', 'robotcam_reader')
    return [Path(__file__)] + [Path(sys.modules[n].__file__) for n in names]


def session_loop(out, run, ctx, emit, blog):
    """Runs the remaining blocks in order; returns the end reason. Smoke: every block once, nothing enforced."""
    todo = list(ORDER) if ctx['smoke'] else next_blocks(read_jsonl(out / 'blocks.jsonl'))
    for k in todo:
        if not ctx['smoke']:
            cr.wait_cores(f'block {k}')
            if agents := repro.agents():
                return f'stopped before block {k}: agent processes running: {agents[:120]!r}'
        g = gate(run['start']['skin'], f'block {k}', ctx['smoke'])
        if not g['passed']:
            return (f'deferred before block {k}: skin {g.get("skin")} not <= {run["start"]["skin"]} + '
                    f'{cr.SKIN_GATE_C} degC after {g.get("waited_s")} s')
        if not ctx['smoke']:
            pre = reading(light=True)
            if pre.get('status') is None or pre['status'] >= STATUS_STOP or pre.get('battery_c') is None or \
                    pre['battery_c'] >= BATTERY_MAX_C:
                return f'stopped before block {k}: status {pre.get("status")}, battery {pre.get("battery_c")} degC'
            if not cores_ok():
                return f'deferred before block {k}: cores 4-7 lost after the gate'
        print(f'[block {k}] gate {g}', flush=True)
        rec, rows = run_block(k, out, ctx, emit)
        rec['gate'] = g
        blog.write(json.dumps(rec) + '\n')
        blog.flush()
        print(f'[block {k}] {rec["label"]}' + (f': {"; ".join(rec["problems"])}' if rec['problems'] else '') +
              f'; {len(rows)} rows', flush=True)
        if rec.get('interrupted'):  # a signal: the record is kept, then the run stops
            raise SystemExit(f'interrupted in block {k}')
        if rec['label'] == 'INVALID':
            return f'block {k} INVALID: redo in a later session (--resume {out.name})'
    return 'all blocks done' if ctx['smoke'] or not next_blocks(read_jsonl(out / 'blocks.jsonl')) else 'blocks remain'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', metavar='RUN')
    a = ap.parse_args(argv)
    if a.smoke and a.resume:
        ap.error('--smoke runs alone')
    if not a.smoke:
        cr.require_native()
        cr.SKIN_GATE_C = SKIN_MATCH_C  # pm.thermal_gate reads it (MEM-1)
        if why := cm.refusals():
            raise SystemExit('not started: ' + '; '.join(why))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cr.exit_on_signal)
    test, _ = b2.load_set(bc.TEST_SET, b2.TEST_SHA256, custody=False)
    fr = frames()
    print('checking server, model and projector SHA-256 (about 30 s)', flush=True)
    bad = [f'{p}: SHA-256 != pinned {h[:12]}' for p, h in bc.PINS.items() if not Path(p).is_file() or cr.sha256(p) != h]
    bad += vis3d.env_problems() + ([] if b0.repro.port_free() else ['port 8080 in use'])
    if bad:
        raise SystemExit('refused: ' + '; '.join(bad))
    code = {str(p.name): cr.sha256(p) for p in code_files()}
    if a.resume:
        out = OUT_ROOT / a.resume
        if Path(a.resume).name != a.resume or not (out / 'run.json').is_file():
            raise SystemExit(f'not resumed: {out} is not a run folder')
        run = json.loads((out / 'run.json').read_text())
        if run['smoke'] or run['code_sha256'] != code:
            raise SystemExit('not resumed: a smoke run, or the code differs from the run start')
        if not next_blocks(read_jsonl(out / 'blocks.jsonl')):
            raise SystemExit('not resumed: every block already VALID')
    else:
        start = reading(light=True)
        if not a.smoke and (why := reading_problems(start, 'launch start')):
            raise SystemExit('not started: ' + '; '.join(why))
        utc = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        out = OUT_ROOT / (utc + ('_smoke' if a.smoke else ''))
        out.mkdir(parents=True)  # a new folder; an existing one raises
        (out / 'slots').mkdir()
        run = {'utc': utc, 'smoke': a.smoke, 'work_s': SMOKE_S if a.smoke else BLOCK_S, 'start': start,
               'command': bc.command(out / 'slots'), 'pins': bc.PINS, 'code_sha256': code,
               'git_head': bc.run_out('git', '-C', str(HERE), 'rev-parse', 'HEAD').strip(),
               'test_sha256': b2.TEST_SHA256, 'items': list(ITEMS), 'frame_ids': list(FRAME_IDS),
               'order': list(ORDER), 'restore_every': RESTORE_EVERY, 'sample_s': SAMPLE_S, 'skin_gate_c': SKIN_MATCH_C,
               'gate_max_s': pm.GATE_MAX_S, 'battery_match_pct': BATTERY_MATCH_PCT}
        (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
    k = len(list(out.glob('session??.json'))) + 1
    with open(out / f'session{k:02d}.json', 'x') as fh:
        fh.write(json.dumps({'session': k, 'started': cr.utc(), 'code_sha256': code,
                             'logger_sched': mem1.logger_sched(THERMAL_LOG)}, indent=1) + '\n')
    ctx = {'session': k, 'smoke': a.smoke, 'dur': run['work_s'], 'test': test, 'frames': fr}
    print(f'run folder {out}; session {k}', flush=True)
    reason = None
    try:
        with open(out / 'requests.jsonl', 'a') as rq, open(out / 'blocks.jsonl', 'a') as blog:
            def emit(row):
                rq.write(json.dumps(row) + '\n')
                rq.flush()
            reason = session_loop(out, run, ctx, emit, blog)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        reason = reason or f'ended by {sys.exc_info()[1]!r}'[:300]
        with open(out / f'session{k:02d}_end.json', 'x') as fh:
            fh.write(json.dumps({'session': k, 'ended': cr.utc(), 'end_reason': reason,
                                 'agents_end': repro.agents(), 'port_free_end': b0.repro.port_free()}, indent=1) + '\n')
        text = report(run, read_jsonl(out / 'blocks.jsonl'), read_jsonl(out / 'requests.jsonl'))
        (out / 'report.txt').write_text(text)
        (out / 'artifacts.sha256').write_text(''.join(f'{cr.sha256(p)}  {p.relative_to(out)}\n'
                                                      for p in sorted(out.rglob('*'))
                                                      if p.is_file() and p.name != 'artifacts.sha256'))
        print(text + f'\nsession {k}: {reason}\n{out}', flush=True)


if __name__ == '__main__':
    main()
