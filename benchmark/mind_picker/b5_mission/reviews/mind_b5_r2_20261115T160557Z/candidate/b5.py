#!/data/data/com.termux/files/usr/bin/python
"""MIND-B5: 30-minute motors-off mission. With RobotCam + YOLO running (CAM1-MID) and the deployed adaptive heat pause
in force, how many B2 V3 picker decisions does the phone serve, how fast after an event, how much time do heat pauses
take, and does throttling change the picker's choices? Research only. Timings count only from an owner run started
by oneshot.sh in native Termux with no agent resident.

Motors off: motors.py and pyserial are never imported (main.py imports neither; run_mission.py imports motors only
without --dry and is not imported here); main.Robot(None) has no motors, so its distance is 999 cm and moves are
sleeps. run_arm refuses to start, and the arm is INVALID, if 'motors' or 'serial' is in sys.modules at its start or
end. No serial port is opened and no ESP32 is contacted.

Setup per arm (not timed): main.get_detector(pinned=True) first, before any thread of this process (run_mission's
order); B4's server (b4.server_command(<run>, 'P') = bc.command(<run>/slots): b11716, setup_q4 flags,
--slot-save-path, Q8_0 projector at 70 image tokens) with bc.PINS; the 40 B2 TEST prompts (V3 FWD, b2.preflight); the
common prefix sent once and slot 0 saved (b4.warm); main.Robot(None); heat_control.HeatControl(HeatMonitor()) started
(its own 1 s / 5 s root readings: the essential safety monitoring; its trace goes to the run folder); then
robot.start_camera(heat) (RobotCam 1 frame/s, PinnedDetector cpus 4-5). Then the start reading, B4's watcher, and the
mission phase of ARM_S (30 min; smoke SMOKE_S) with an absolute deadline.

Mission loop (main thread) = Robot.run_mission motors-off: per cycle heat.decide(999); END ends the arm (INVALID);
ACTIVE -> PAUSE/HOLD: picker forbidden, B4's stop_camera (capture shown stopped), heat.enter(target, now), one
reading; PAUSE/HOLD -> ACTIVE: robot.start_camera(heat), heat.enter('ACTIVE', since=before the start), picker
allowed (fresh SEVERE check, as main.py), then one reading (a battery stop or an exception there forbids at once and
ends the arm); PAUSE <-> HOLD: heat.enter. Heat rows: 'intent' when a transition begins, 'done' when it completed; a
pause runs from the intent to leave ACTIVE to the 'done' re-entry (camera stop/start and the pause_start reading
count as pause, the pause_end reading as ACTIVE; a failed restart is no resume). A camera stop or start failure ends
the arm (INVALID). HOLD sleeps 1 s; PAUSE sleeps the move time (pause_cycle motors-off: distance 999 -> rotate ->
move() sleeps CYCLE_MOVE_TIME); ACTIVE runs B4's camera cycle (new frame, SizePolicy, detect, observe, B4's 'no new'
rule) and then sleeps the move time (cut at the deadline). The picker is allowed only while ACTIVE and Android
status < SEVERE (request_pick); a forbid cancels a running decision (checked before and after every HTTP request and
before its result is accepted) for good (picker.Picker's generation rule). Start barrier: the scheduler thread
starts only after the first heat / SEVERE assessment set the permission. No camera cycle starts at or after the
deadline; a person change detected by a cycle that ends after it is counted 'late_not_offered', not offered.
Re-pin (owner decision 2026-11-15): at arm start, at every camera start and whenever a watcher scan finds drift,
every llama-server thread is re-pinned to 4-7 (os.sched_setaffinity per thread) and the detector to 4-5 (its
check_affinity, as main.py; at camera start main.py's own re-pin is recorded). Every re-pin is recorded (time,
trigger, process, threads, before/after). Drift corrected by one re-pin is recorded, not INVALID; a failed re-pin or
the same process drifted in two consecutive scans is INVALID. Each scan also records /proc/<pid>/cpuset and cgroup
of llama-server, the runner and the detector caller thread.
Events: (a) scripted: the 40 B2 TEST items in id order, cycled, item k offered at k * OFFER_S (10 s) while before the
deadline (180 per 30-min arm); (b) real: each ACTIVE cycle's most confident person (scene_from_detections) gives an
area bucket; appear / leave / bucket change becomes an event line in the B2 state format (mission REAL_MISSION,
current activity REAL_CURRENT, memory none, all five actions; logged, served, unscored). Pending queue of one: a new
event replaces an unserved one (that one is 'dropped').
Schedulers (scheduler thread, polls every POLL_S; each decision runs in its own thread):
  A FIXED: at each TICK_S (30 s) mark, if allowed and idle, decide on the pending event (if any).
  B ADAPTIVE: as soon as an event is pending, the previous decision has finished, the picker is allowed and the heat
    control's latest VIRTUAL-SKIN < SKIN_B_C (36.0 degC).
A decision = slot 0 restored before every RESTORE_EVERY-th, then b2.score (V3 FWD, unchanged); a real event's prompt is
rendered first (b2.preflight, counted in its wall). A cancelled decision's event goes back to pending unless a newer
one is pending (then 'cancelled'). Latency = decision end - offer; wait = start - offer. Nothing starts after the
deadline; the decision or cycle in flight finishes; overrun (last of both) <= OVERRUN_MAX_S (15 s).
Readings (b4.reading, per-PID memory checked): start (its t relative to t0, usually negative), end, and one at each
heat-pause start (after the camera stop) and end (after the camera start); robotcam memory required while the camera
is on. Battery < BATTERY_STOP_PCT (30 %) at any reading ends the arm (INVALID) and the session.
Before every arm: cores 4-7, agent check, MEM-1 skin gate (launch-start skin + 1.0 degC, idle wait at most GATE_MAX_S;
smoke: first arm only, pm's default wait), Android status < 3, battery < 45 degC and >= 30 %. A timed run needs
battery >= BATTERY_START_PCT (85 %) at its first start and >= BATTERY_RESUME_PCT (50 %) at each --resume. Smoke only:
the deployed HeatControl runs with lowered thresholds set per arm from that arm's start reading (after server load
and camera start; smoke_thresholds: pause at arm-start skin + SMOKE_PAUSE_DC, resume at pause - SMOKE_RESUME_DC; its
60 s / 30 s rules unchanged) so that a real pause and resume with camera stop/start can happen in SMOKE_S; timed runs
use 37/35 degC. Order A then B; an INVALID arm ends the
session and is redone with --resume. Labels: arm VALID only with every check clean (arm_problems: errors incl. heat
END, readings, agents, cores, request errors, camera stops, overrun, watcher); A vs B is descriptive; energy
INDICATIVE. Agreement: each served scripted decision's action vs the B2 run of record (TEST V3 FWD, RECORD, SHA-256
pinned).
Frozen definition: SHA-256 of oneshot.sh, the run of record, the TEST set and every imported module file under the
repo, and the server command; --resume refuses any difference.

Usage: b5.py | b5.py --resume RUN | b5.py --smoke (each arm SMOKE_S, gate before the first arm only, agent/cores
checks recorded but not enforced; SMOKE - NOT A MEASUREMENT).
Writes ~/robot_runs/mind_b5/<UTC>[_smoke]/: run.json, sessionNN.json, sessionNN_end.json, arms.jsonl,
requests.jsonl, NN_<arm>_server.log, heat_trace_<UTC>.jsonl, slots/, report.txt, artifacts.sha256.
"""
import argparse
import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'b4_baseline'))
import b4  # noqa: E402  B4's readings, watcher, server, gate pieces, camera stop (imports bc, b2, mem1, ...)
import main  # noqa: E402  Robot, read_frame, CYCLE_MOVE_TIME (never motors.py)
import heat_control  # noqa: E402  the deployed adaptive 37/35 degC heat control
from detect_person import scene_from_detections  # noqa: E402

bc, b2, b0, mem1, cr, pm, repro, cm = b4.bc, b4.b2, b4.b0, b4.mem1, b4.cr, b4.pm, b4.repro, b4.cm
REPO = b4.REPO
OUT_ROOT = cr.HOME / 'robot_runs' / 'mind_b5'
THERMAL_LOG = OUT_ROOT / 'thermal.log'  # oneshot.sh's logger; the skin gate reads it
RECORD = HERE.parent / 'b2_rules' / 'runs' / '20261102T143251Z' / 'requests.jsonl'
RECORD_SHA256 = '940930a3723f07ff6fe1639fb4a333be37635cca93fc0536cae1ed6a6d9b8475'
ARMS = ('A', 'B')
ARM_S, SMOKE_S = 30 * 60, 180
OFFER_S, TICK_S, POLL_S = 10, 30, 0.1
SKIN_B_C = 36.0
BATTERY_START_PCT, BATTERY_RESUME_PCT, BATTERY_STOP_PCT = 85, 50, 30
SMOKE_PAUSE_DC, SMOKE_RESUME_DC = 1.0, 0.5  # smoke: arm-start skin + 1.0; pause - 0.5
GATE_MAX_S = 30 * 60
OVERRUN_MAX_S = b4.OVERRUN_MAX_S
JOIN_S = 4 * b4.REQUEST_TIMEOUT_S + 30  # a decision in flight: at most a few requests of 60 s each
RESTORE_EVERY = b4.RESTORE_EVERY
REAL_MISSION, REAL_CURRENT = 'M1', 'explore'
PHRASE = {'very close': 'very close', 'close': 'close', 'medium': 'at medium distance', 'far': 'far away'}
SMOKE_LABEL = b4.SMOKE_LABEL
clock, pause = time.monotonic, time.sleep  # the tests inject a fake clock


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- events and schedulers

def person_event(prev, cur):
    """Event line for a change of the most confident person's area bucket (None: no person), or None."""
    if prev == cur:
        return None
    if prev is None:
        return f'A person appeared {PHRASE[cur]}.'
    if cur is None:
        return 'The person left your view.'
    return f'The person is now {PHRASE[cur]}.'


class Mission:
    """Pending queue of one, both schedulers and every event's fate. Times are arguments (tests use a fake clock);
    every method holds the lock (mission, scheduler and decision threads call it)."""

    def __init__(self, arm, t0, work_s, items):
        self.arm, self.t0, self.deadline, self.items = arm, t0, t0 + work_s, items
        self.lock = threading.Lock()
        self.events, self.pending, self.busy = [], None, None
        self.ok, self.gen, self.k_offer, self.k_tick, self.n_started = False, 0, 0, 0, 0
        self.ticks = {'decided': 0, 'empty': 0, 'paused': 0, 'busy': 0}

    def rel(self, now):
        return round(now - self.t0, 3)

    def offer(self, ev, now):
        with self.lock:
            ev.update(offered=self.rel(now), fate='pending', attempts=0)
            self.events.append(ev)
            if self.pending:
                self.pending['fate'] = 'dropped'
            self.pending = ev

    def offer_due(self, now):
        """Scripted item k at t0 + k * OFFER_S, for every such time before the deadline that has come."""
        while self.t0 + self.k_offer * OFFER_S <= now and self.k_offer * OFFER_S < self.deadline - self.t0:
            it = self.items[self.k_offer % len(self.items)]
            self.offer({'source': 'scripted', 'id': it['id'], 'item': it, 'due': self.k_offer * OFFER_S}, now)
            self.k_offer += 1

    def set_ok(self, ok):
        """The picker allowed (ACTIVE, below SEVERE) or not; a forbid bumps the generation (cancels for good)."""
        with self.lock:
            if self.ok and not ok:
                self.gen += 1
            self.ok = ok

    def allowed(self, gen):
        with self.lock:
            return self.ok and self.gen == gen

    def start(self, now, skin):
        """(event, generation) to decide on now, or None."""
        with self.lock:
            if now >= self.deadline:
                return None
            if self.arm == 'A':
                if now < self.t0 + self.k_tick * TICK_S:
                    return None
                self.k_tick = int((now - self.t0) // TICK_S) + 1
                why = 'busy' if self.busy else 'paused' if not self.ok else 'empty' if not self.pending else 'decided'
                self.ticks[why] += 1
                if why != 'decided':
                    return None
            elif self.busy or not self.ok or not self.pending or skin is None or skin >= SKIN_B_C:
                return None
            ev, self.pending, self.busy = self.pending, None, self.pending
            self.n_started += 1
            ev.update(fate='deciding', started=self.rel(now), attempts=ev['attempts'] + 1, n=self.n_started)
            return ev, self.gen

    def done(self, ev, row, now):
        with self.lock:
            self.busy = None
            ev['ended'] = self.rel(now)
            if row.get('cancelled'):
                if self.pending is None and now < self.deadline:
                    ev['fate'], self.pending = 'pending', ev  # back in the queue; latency still from its offer
                else:
                    ev['fate'] = 'cancelled'
            elif 'error' in row:
                ev['fate'] = 'error'
            else:
                ev.update(fate='served', top=row['top'], latency_s=round(ev['ended'] - ev['offered'], 3),
                          wait_s=round(ev['started'] - ev['offered'], 3))

    def close(self):
        with self.lock:
            if self.pending:
                self.pending['fate'] = 'unserved'
            self.pending = None


# ---------------------------------------------------------------- picker

def picker_setup(test):
    """(prompts of all 40 TEST items, V3 FWD; their common prefix ids); any problem raises."""
    info = {'problems': []}
    prompts = b2.preflight(b4.post, test, [('V3', 'FWD')], info, 'TEST')
    if info['problems']:
        raise RuntimeError(f'picker preflight: {info["problems"]}')
    ids = [prompts['V3', 'FWD', it['id']][1] for it in test['items']]
    return prompts, ids[0][:min(mem1.lcp(ids[0], x) for x in ids[1:])]


class Checked(b4.Recorder):
    """b4.Recorder that checks the picker (its generation and permission) before every request and again after
    every response (picker.Picker's check()): a forbid at any time, even one followed by an allow, cancels."""

    def __init__(self, ok):
        super().__init__()
        self.ok = ok

    def __call__(self, path, body):
        if not self.ok():
            raise Cancelled('picker forbidden (left ACTIVE, SEVERE or stop)')
        res = super().__call__(path, body)
        if not self.ok():
            raise Cancelled(f'picker forbidden while {path} was in flight')
        return res


def real_item(n, text):
    return {'id': f'R{n}', 'mission': REAL_MISSION, 'current': REAL_CURRENT, 'event': text, 'memory': None,
            'allowed': list(b2.NAMES)}


def decide(ev, ok, prompts, test):
    """One decision row: restore before every RESTORE_EVERY-th, a real event's prompt rendered, then b2.score V3 FWD."""
    rec = Checked(ok)
    row = {'kind': 'decision', 'source': ev['source'], 'id': ev['item']['id'], 'n': ev['n']}
    t0 = time.perf_counter()
    try:
        if ev['n'] % RESTORE_EVERY == 0:
            if not ok():
                raise Cancelled('picker forbidden before the restore')
            row['restore'] = b4.slot('restore')
            if not ok():
                raise Cancelled('picker forbidden while the restore was in flight')
        it = ev['item']
        prompt = prompts.get(('V3', 'FWD', it['id']))
        if prompt is None:
            r0, info = time.perf_counter(), {'problems': []}
            prompt = b2.preflight(rec, {**test, 'items': [it]}, [('V3', 'FWD')], info, 'REAL')['V3', 'FWD', it['id']]
            row['render_s'] = round(time.perf_counter() - r0, 3)
            if info['problems']:
                raise RuntimeError(f'real event prompt: {info["problems"]}')
        n0 = len(rec.completions)
        r = b2.score(rec, it, 'V3', 'FWD', prompt)
        if not ok():
            raise Cancelled('picker forbidden before the result was accepted')
        t1, t2 = (x.get('timings', {}) for x in rec.completions[n0:])
        row.update(top=r['top'], reason=r['reason'], reason_tokens=r['reason_tokens'],
                   gen_tok_s=t1.get('predicted_per_second'), step1_prompt_n=t1.get('prompt_n'),
                   step1_cache_n=t1.get('cache_n'), step2_prompt_n=t2.get('prompt_n'), step2_cache_n=t2.get('cache_n'))
    except Cancelled as e:
        row.update(cancelled=True, reason=str(e))
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'
    row.update(wall_s=round(time.perf_counter() - t0, 3), endpoint_s=rec.seconds, cores_ok=b4.cores_ok())
    return row


def scheduler(m, skin, halt, run, out):
    """Offers the scripted items and starts decisions until the deadline has passed with none in flight, or halt."""
    th = None
    try:
        while not halt.is_set():
            now = clock()
            m.offer_due(now)
            if go := m.start(now, skin()):
                th = threading.Thread(target=run, args=go, daemon=True)
                th.start()
            if now >= m.deadline and m.busy is None:
                break
            halt.wait(POLL_S)
    except Exception as e:
        out['error'] = f'scheduler: {type(e).__name__}: {e}'
    finally:
        if th:
            th.join(JOIN_S)
            out['alive'] = th.is_alive()


# ---------------------------------------------------------------- mission loop

def severe(heat):
    d = heat.monitor.readings()['dump']
    return bool(d) and d['status'] >= heat_control.STATUS_SEVERE


def mission_loop(robot, heat, m, rec, add, read, cam, ready=lambda: None, repin=lambda: None):
    """Robot.run_mission motors-off (see the module docstring) until the deadline. Returns at heat END, a camera
    failure or a battery stop, with the reason in rec['errors']. ready() is called once, right after the first heat /
    SEVERE assessment has set the picker's permission (the start barrier: the scheduler starts there). Heat rows:
    stage 'intent' when a transition begins, stage 'done' when it completed (into ACTIVE: camera started and ACTIVE
    entered). Detections of a cycle admitted before the deadline that end after it offer no real event (counted in
    rec['late_not_offered'])."""
    last = prev = None
    no_new = n_real = 0
    rec.setdefault('late_not_offered', 0)
    first = True
    while clock() < m.deadline:
        target, reason = heat.decide(999)  # dry run: get_distance() is 999
        sev = severe(heat)
        if sev:
            m.set_ok(False)
        if target == 'END':
            rec['errors'].append(f'heat END: {reason}')
            return
        if target != heat.phase:
            frm = heat.phase
            add({'kind': 'heat', 'stage': 'intent', 'from': frm, 'to': target, 'reason': reason, 't': m.rel(clock())})
            if heat.phase == 'ACTIVE':
                m.set_ok(False)
                st = b4.stop_camera(robot)
                rec['camera_stops'].append(st)
                if st.get('error') or st.get('affinity_error'):
                    heat.event('camera_stop_fail', st.get('error') or st.get('affinity_error'))
                    rec['errors'].append(f'camera stop at pause: {st}')
                    return
                cam['on'] = False
                heat.event('camera_stop_ok')
                heat.enter(target, since=heat.now())
                add({'kind': 'heat', 'stage': 'done', 'from': frm, 'to': target, 't': m.rel(clock())})
                if not read('pause_start'):
                    return
            elif target == 'ACTIVE':
                started = heat.now()
                cam['on'] = True  # a start attempted: cleanup stops it
                try:
                    robot.start_camera(heat)
                except Exception as e:
                    heat.event('camera_start_fail', str(e))
                    rec['errors'].append(f'camera start after pause: {type(e).__name__}: {e}')
                    return
                heat.event('camera_start_ok')
                repin()  # llama-server threads to 4-7 at every camera start (the detector: main.py's start_camera)
                heat.enter('ACTIVE', since=started)
                add({'kind': 'heat', 'stage': 'done', 'from': frm, 'to': 'ACTIVE', 't': m.rel(clock())})
                last, cam['slot'] = None, cam['slot'] + 1
                sev = severe(heat)
                m.set_ok(not sev)  # main.py: picker.allow() right after ACTIVE entry, before our reading
                try:
                    if not read('pause_end'):
                        m.set_ok(False)
                        return
                except BaseException:
                    m.set_ok(False)
                    raise
            else:
                heat.enter(target)
                add({'kind': 'heat', 'stage': 'done', 'from': frm, 'to': target, 't': m.rel(clock())})
        m.set_ok(heat.phase == 'ACTIVE' and not sev)
        if first:
            first = False
            ready()
        if heat.phase == 'HOLD':
            pause(1)
            continue
        if heat.phase == 'PAUSE':
            pause(min(main.CYCLE_MOVE_TIME, max(0.0, m.deadline - clock())))
            continue
        c0 = clock()
        if c0 >= m.deadline:  # a transition or reading crossed it: no new cycle
            break
        r = main.read_frame(session=robot.camera_session)
        if r['status'] != 'ok' or (last is not None and r['frame'] <= last):
            why = r['status'] if r['status'] != 'ok' else 'no new'
            add({'kind': 'camera_stop', 'reason': why, 't': m.rel(c0)})
            no_new += why == 'no new'
            if why != 'no new' or no_new > b4.NO_NEW_MAX:
                rec['errors'].append(f'camera stop: {why} ({no_new} no new)')
                return
            pause(0.1)
            continue
        last, robot.camera_session = r['frame'], r['session']
        size = robot.size_policy.next_size()
        row = {'kind': 'cycle', 't': m.rel(c0), 'slot': cam['slot'], 'frame': r['frame'], 'age_s': r['age_s'],
               'size': size}
        try:
            d0 = time.perf_counter()
            dets = robot.detector.detect(r['image'], size)
            row.update(detect_ms=round((time.perf_counter() - d0) * 1000, 1), n_det=len(dets))
            robot.size_policy.observe(dets, r['image'].height)
            person = next((x for x in scene_from_detections(dets, *r['image'].size) if x[0] == 'person'), None)
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'
        row['cores_ok'] = b4.cores_ok()
        add(row)
        if 'error' in row:
            rec['errors'].append(f'camera cycle: {row["error"]}')
            return
        cur = person[3] if person else None
        text = person_event(prev, cur)
        if text and clock() >= m.deadline:
            rec['late_not_offered'] += 1  # admitted before the deadline, ended after it
        elif text:
            n_real += 1
            m.offer({'source': 'real', 'id': f'R{n_real}', 'item': real_item(n_real, text), 'text': text}, clock())
        prev = cur
        pause(min(main.CYCLE_MOVE_TIME, max(0.0, m.deadline - clock())))


# ---------------------------------------------------------------- one arm

# ---------------------------------------------------------------- re-pin (owner decision 2026-11-15)

class Repinner:
    """Re-pins like a robot would: every llama-server thread to SERVER_CPUS (os.sched_setaffinity per thread) and the
    detector to 4-5 (its own check_affinity, as main.py). Every re-pin is recorded in events (mono, trigger, process,
    threads, before/after, ok). Called from the mission thread and the watcher thread (lock)."""

    def __init__(self, targets):
        self.targets, self.events, self.lock = targets, [], threading.Lock()

    def add(self, ev):
        with self.lock:
            self.events.append({'mono': round(time.monotonic(), 3), **ev})
        return self.events[-1]

    def server(self, trigger):
        pid = self.targets.get('server_pid')
        before, after, errors = {}, {}, {}
        try:
            tids = sorted(int(t.name) for t in (b4.PROC / str(pid) / 'task').iterdir())
        except OSError as e:
            return self.add({'trigger': trigger, 'process': 'llama_server', 'ok': False, 'errors': {'task': str(e)}})
        for tid in tids:
            try:
                before[tid] = sorted(os.sched_getaffinity(tid))
                if set(before[tid]) != b4.SERVER_CPUS:
                    os.sched_setaffinity(tid, b4.SERVER_CPUS)
                    after[tid] = sorted(os.sched_getaffinity(tid))
                    if set(after[tid]) != b4.SERVER_CPUS:
                        errors[tid] = f'after {after[tid]}'
            except ProcessLookupError:
                before.pop(tid, None)  # the thread ended: nothing to pin
            except OSError as e:
                errors[tid] = f'{type(e).__name__}: {e}'
        return self.add({'trigger': trigger, 'process': 'llama_server', 'threads': sorted(after),
                         'before': before, 'after': after, 'ok': not errors, 'errors': errors})

    def detector(self, trigger, result=None):
        """result: a check_affinity() result already obtained (main.py's at camera start); else one is run now."""
        r = result if result is not None else self.targets['detector'].check_affinity()
        return self.add({'trigger': trigger, 'process': 'detector', 'threads': r['repinned'], 'before': r['before'],
                         'after': r['after'], 'ok': r['ok'], 'errors': r['errors']})


DRIFT = {'llama_server': re.compile(r'llama-server thread \d+ Cpus_allowed_list'),
         'detector': re.compile(r'detector threads \{')}


def groups(targets):
    """/proc/<pid>/cpuset and cgroup of llama-server, the runner and the detector caller thread (error text when
    unreadable): shows whether Android moved Termux between CPU groups."""
    paths = {'runner': b4.PROC / str(os.getpid())}
    if targets.get('server_pid') is not None:
        paths['llama_server'] = b4.PROC / str(targets['server_pid'])
    if targets.get('detector') is not None:
        paths['detector'] = b4.PROC / str(os.getpid()) / 'task' / str(targets['detector'].caller_tid)
    out = {}
    for name, p in paths.items():
        for f in ('cpuset', 'cgroup'):
            try:
                out[f'{name}_{f}'] = (p / f).read_text().strip()
            except OSError as e:
                out[f'{name}_{f}'] = f'{type(e).__name__}: {e}'
    return out


def scan(targets, rp):
    """b4.watch_check, then drift (a llama-server thread off 4-7, a detector thread off 4-5) is split from the other
    hits and re-pinned at once; the CPU groups are recorded."""
    x = b4.watch_check(targets)
    hits, drift = [], {}
    for h in x['hits']:
        p = next((k for k, rx in DRIFT.items() if rx.match(h)), None)
        (drift.setdefault(p, []) if p else hits).append(h)
    x.update(hits=hits, drift=drift, groups=groups(targets))
    x['repins'] = [rp.server('watch') if p == 'llama_server' else rp.detector('watch') for p in sorted(drift)]
    return x


def watch_problems(w, repins):
    """b4's watcher rules, except that drift corrected by one re-pin is recorded, not a problem; INVALID when any
    re-pin fails or the same process is drifted in two consecutive scans."""
    why = b4.thread_problems(w, 'watch')
    why += [f'watch: {h}' for x in w['samples'] for h in x['hits']]
    why += [f're-pin failed ({e["trigger"]}, {e["process"]}): {e.get("errors")}' for e in repins if not e['ok']]
    for a, b in zip(w['samples'], w['samples'][1:]):
        why += [f'watch: {p} drifted in two consecutive scans ({a["mono"]}, {b["mono"]})'
                for p in sorted(set(a.get('drift', {})) & set(b.get('drift', {})))]
    return why if w['samples'] else why + ['watch: no scan']


def record_camera_repins(heat, rp):
    """Records main.py's detector re-pin at every camera start: start_camera passes its check_affinity() result to
    heat.event('detector_affinity', ...); this instance wrapper also hands it to rp (heat_control.py unchanged)."""
    event = heat.event

    def wrapped(name, detail=None, dump=None):
        if name == 'detector_affinity':
            rp.detector('camera_start', detail)
        return event(name, detail, dump)
    heat.event = wrapped


def need(c):
    return ['runner', 'llama_server'] + (['robotcam'] if c['camera'] else [])


def arm_problems(rec, rows):
    """Why an arm is INVALID (empty: VALID)."""
    why = list(rec['errors'])
    for k in ('start', 'end'):
        if not any(c['tag'] == k for c in rec['readings']):
            why.append(f'{k} reading missing')
    for c in rec['readings']:
        why += b4.reading_problems(c, c['tag'], need(c))
    if rec.get('agents_before') or rec.get('agents_after'):
        why.append(f'agent process running: {(rec.get("agents_before") or rec.get("agents_after"))[:120]!r}')
    if not rec.get('cores_before') or not rec.get('cores_after') or any(r.get('cores_ok') is False for r in rows):
        why.append('cores 4-7 not all allowed during the arm')
    if n := sum('error' in r for r in rows):
        why.append(f'{n} request/cycle errors')
    if not any(r['kind'] == 'cycle' for r in rows):
        why.append('no camera cycles')
    stops = [r['reason'] for r in rows if r['kind'] == 'camera_stop']
    why += [f'camera stop: {s}' for s in stops if s != 'no new']
    if stops.count('no new') > b4.NO_NEW_MAX:
        why.append(f'{stops.count("no new")} "no new" frames > {b4.NO_NEW_MAX}')
    if rec.get('overrun_s', 0) > OVERRUN_MAX_S:
        why.append(f'overrun {rec["overrun_s"]} s > {OVERRUN_MAX_S} s')
    why += [f'cleanup camera stop: {rec["camera_end"][k]}' for k in ('affinity_error', 'error')
            if rec.get('camera_end', {}).get(k)]
    if 'watch' in rec:
        why += watch_problems(rec['watch'], rec.get('repins', []))
    else:
        why += [f're-pin failed ({e["trigger"]}, {e["process"]}): {e.get("errors")}'
                for e in rec.get('repins', []) if not e['ok']]
    return why


def run_arm(name, out, ctx, emit):
    """One arm; its rows go to requests.jsonl through emit; returns (record, rows)."""
    rec = {'arm': name, 'session': ctx['session'], 'utc_start': cr.utc(), 'work_s': ctx['dur'], 'errors': [],
           'readings': [], 'camera_stops': [], 'cores_before': b4.cores_ok(), 'agents_before': repro.agents()}
    rows, lock = [], threading.Lock()

    def add(row):
        row = {'arm': name, 'session': ctx['session'], **row}
        with lock:
            rows.append(row)
            emit(row)
    if not ctx['smoke'] and (rec['agents_before'] or not rec['cores_before']):
        rec['errors'].append('initial agent/cores check failed: no workload started')
        rec.update(utc_end=cr.utc(), cores_after=b4.cores_ok(), agents_after=repro.agents())
        rec['problems'], rec['label'] = arm_problems(rec, rows), 'INVALID'
        rec['metrics'] = metrics(rec, rows, ctx['record'], out)  # the same schema as every other arm record
        return rec, rows
    proc = log = robot = heat = watch = th = m = None
    rp = None
    cam = {'on': False, 'slot': 1}
    halt, sched = threading.Event(), {}
    pids, targets = {'runner': os.getpid()}, {}

    def read(tag):
        c = {'tag': tag, 'camera': cam['on'], **b4.reading(pids)}
        c['t'] = m.rel(c['mono']) if m else None
        rec['readings'].append(c)
        if isinstance(c.get('battery_pct'), int) and c['battery_pct'] < BATTERY_STOP_PCT:
            rec['errors'].append(f'battery {c["battery_pct"]} % < {BATTERY_STOP_PCT} %: session stopped')
            rec['battery_stop'] = True
            return False
        return True
    try:
        if bad := [x for x in ('motors', 'serial') if x in sys.modules]:
            raise RuntimeError(f'refused: {bad} imported')
        main.get_detector(pinned=True)  # before any thread of this process (run_mission's order)
        proc, log, rec['load_s'] = b4.start_server(out, f'{ctx["session"]:02d}_{name}', b4.server_command(out, 'P'))
        pids['llama_server'] = targets['server_pid'] = proc.pid
        prompts, prefix = picker_setup(ctx['test'])
        rec['warm'] = b4.warm(prefix)
        robot = main.Robot(None)
        heat = heat_control.HeatControl(heat_control.HeatMonitor(), trace_dir=str(out))
        rp = Repinner(targets)
        record_camera_repins(heat, rp)
        heat.start()
        rec['heat_trace'] = Path(heat.trace_path).name
        cam['on'] = True
        robot.start_camera(heat)  # main.py re-pins the detector (recorded through record_camera_repins)
        targets['detector'] = robot.detector
        rp.server('arm_start')
        rec['camera'] = {'detector': robot.detector.config(), 'affinity_start': robot.detector.affinity()}
        if read('start'):
            if ctx['smoke']:  # smoke only: thresholds from this arm's start skin (timed runs: 37/35 degC)
                if (skin := rec['readings'][-1].get('skin')) is None:
                    raise RuntimeError('smoke: no arm-start skin for the smoke heat thresholds')
                rec['heat_thresholds'] = smoke_thresholds(skin)
            watch = b4.Periodic(lambda: scan(targets, rp), b4.WATCH_S).start()
            m = Mission(name, clock(), ctx['dur'], ctx['test']['items'])
            start = next(c for c in rec['readings'] if c['tag'] == 'start')
            start['t'] = m.rel(start['mono'])  # taken before t0: its real relative time, usually negative

            def run(ev, gen):
                row = {'kind': 'decision', 'error': 'not run'}
                try:
                    row = decide(ev, lambda: m.allowed(gen) and not halt.is_set(), prompts, ctx['test'])
                finally:
                    m.done(ev, row, clock())
                    add({**row, 'offered': ev['offered'], 'started': ev['started'], 'ended': ev['ended'],
                         'fate': ev['fate']})
            skin = lambda: (heat.monitor.readings()['dump'] or {}).get('skin')  # noqa: E731
            def ready():  # start barrier: after the first heat / SEVERE assessment set the permission
                nonlocal th
                th = threading.Thread(target=scheduler, args=(m, skin, halt, run, sched), daemon=True)
                th.start()
            mission_loop(robot, heat, m, rec, add, read, cam, ready, lambda: rp.server('camera_start'))
            rec['mission_end_s'] = min(m.rel(clock()), ctx['dur'])
            if rec['errors']:
                halt.set()  # the arm failed: cancel the decision in flight
            if th:
                th.join(JOIN_S + 5)
            rec['overrun_s'] = round(max(0.0, clock() - m.deadline), 3)
            read('end')
    except BaseException as e:  # recorded; cleanup below; a stop (signal) re-raises after the record is kept
        rec['errors'].append(f'{type(e).__name__}: {e}')
        if not isinstance(e, Exception):
            rec['interrupted'] = True
    finally:
        halt.set()
        if th:
            b4.cleanup(rec, 'scheduler', lambda: th.join(JOIN_S + 5))
            rec['scheduler'] = {**sched, 'thread_alive': th.is_alive()}
            if sched.get('error') or sched.get('alive') or th.is_alive():
                rec['errors'].append(f'scheduler: {rec["scheduler"]}')
        if m:
            m.close()
            rec['ticks'] = m.ticks
            for ev in m.events:
                add({'kind': 'event', **{k: v for k, v in ev.items() if k != 'item'}})
        if watch:
            b4.cleanup(rec, 'watch', lambda: rec.__setitem__('watch', watch.stop()))
        if rp:
            rec['repins'] = list(rp.events)
        if robot and cam['on']:
            b4.cleanup(rec, 'camera', lambda: rec.__setitem__('camera_end', b4.stop_camera(robot)))
        if heat:
            b4.cleanup(rec, 'heat', heat.stop)
        if proc:
            b4.cleanup(rec, 'server', lambda: rec.update(zip(('exit_code', 'killed'), b4.stop_server(proc, log))))
        if bad := [x for x in ('motors', 'serial') if x in sys.modules]:
            rec['errors'].append(f'{bad} imported during the arm')
        rec.update(utc_end=cr.utc(), cores_after=b4.cores_ok(), agents_after=repro.agents())
    rec['problems'] = arm_problems(rec, rows)
    rec['label'] = SMOKE_LABEL if ctx['smoke'] else 'VALID' if not rec['problems'] else 'INVALID'
    rec['metrics'] = metrics(rec, rows, ctx['record'], out)
    return rec, rows


# ---------------------------------------------------------------- metrics and report

def stat(xs, nd=3):
    xs = [x for x in xs if x is not None]
    return {'n': len(xs), 'median': round(mem1.med(xs), nd), 'p95': round(mem1.p95(xs), nd)} if xs else {'n': 0}


def battery_refusal(pct, smoke, resume):
    """Why a session may not start on this battery %, or None: timed first start >= 85 %, --resume >= 50 %."""
    need = None if smoke else BATTERY_RESUME_PCT if resume else BATTERY_START_PCT
    return f'not started: battery {pct} % < {need} %' if need is not None and pct < need else None


def smoke_thresholds(skin):
    """Smoke only: heat_control's pause skin set to the arm-start skin + 1.0 degC and its resume skin to pause - 0.5
    degC (runtime values of the deployed module's constants; its file is unchanged). Returns them."""
    heat_control.SKIN_PAUSE_C = round(skin + SMOKE_PAUSE_DC, 2)
    heat_control.SKIN_RESUME_C = round(heat_control.SKIN_PAUSE_C - SMOKE_RESUME_DC, 2)
    return {'pause_c': heat_control.SKIN_PAUSE_C, 'resume_c': heat_control.SKIN_RESUME_C,
            'min_active_s': heat_control.MIN_ACTIVE_S, 'min_pause_s': heat_control.MIN_PAUSE_S}


def pauses(rows, end_t):
    """(count, time to the first, total s): an interval runs from the intent to leave ACTIVE (picker forbidden,
    camera stop begins) to the successful re-entry into ACTIVE after the camera start (stage 'done'), or to end_t; a
    failed restart is no re-entry. Camera stop and start and the pause_start reading count as pause; the pause_end
    reading (picker allowed, camera on) as ACTIVE."""
    out, began = [], None
    for r in rows:
        if r['kind'] != 'heat':
            continue
        if r['stage'] == 'intent' and r['from'] == 'ACTIVE' and began is None:
            began = r['t']
        elif r['stage'] == 'done' and r['to'] == 'ACTIVE' and began is not None:
            out.append((began, r['t']))
            began = None
    if began is not None:
        out.append((began, max(began, end_t)))
    return len(out), (out[0][0] if out else None), round(sum(b - a for a, b in out), 3)


def metrics(rec, rows, record, out):
    ev = [r for r in rows if r['kind'] == 'event']
    dec = [r for r in rows if r['kind'] == 'decision' and 'top' in r and not r.get('cancelled') and 'error' not in r]
    cyc = [r for r in rows if r['kind'] == 'cycle' and 'error' not in r]
    reads = rec['readings']
    end_t = rec.get('mission_end_s', 0)
    n, first, total = pauses(rows, end_t)
    fates = {s: {f: sum(e['fate'] == f for e in ev if e['source'] == s)
                 for f in ('served', 'dropped', 'cancelled', 'unserved', 'error')} for s in ('scripted', 'real')}
    for s in fates:
        fates[s]['offered'] = sum(e['source'] == s for e in ev)
    fates['real']['late_not_offered'] = rec.get('late_not_offered', 0)
    periods = [b['t'] - a['t'] for a, b in zip(cyc, cyc[1:]) if a['slot'] == b['slot']]
    skins = [c['skin'] for c in reads if c.get('skin') is not None]
    scripted = [r for r in dec if r['source'] == 'scripted']
    differ = sorted({r['id'] for r in scripted if r['top'] != record.get(r['id'])})
    start = next((c for c in reads if c['tag'] == 'start'), None)
    end = next((c for c in reads if c['tag'] == 'end'), None)
    return {
        'decisions_served': len(dec), 'events': fates,
        'latency_s': {s: stat([e['latency_s'] for e in ev if s in ('all', e['source']) and e['fate'] == 'served'])
                      for s in ('scripted', 'real', 'all')},
        'wait_s': stat([e.get('wait_s') for e in ev if e['fate'] == 'served']),
        'decision_wall_s': stat([r['wall_s'] for r in dec]),
        'ticks': rec.get('ticks'),
        'pauses': n, 'resumes': sum(r['kind'] == 'heat' and r['stage'] == 'done' and r['to'] == 'ACTIVE' for r in rows),
        'first_pause_s': first, 'pause_s': total, 'arm_s': round(end_t, 3),
        'active_share': round(1 - total / end_t, 3) if end_t else None,
        'skin': {'start': start and start.get('skin'), 'max_readings': max(skins) if skins else None,
                 'max_trace': trace_max(out / rec['heat_trace']) if rec.get('heat_trace') else None,
                 'end': end and end.get('skin')},
        'caps_min': {p: min([c['caps'][p] for c in reads if isinstance(c.get('caps', {}).get(p), int)], default=None)
                     for p in b4.POLICIES},
        'cycle_period_s': stat(periods), 'detect_ms': {z: stat([r['detect_ms'] for r in cyc if r['size'] == z], 1)
                                                      for z in (320, 640)},
        'mem_available_min_kb': min([c['avail_kb'] for c in reads if 'avail_kb' in c], default=None),
        'energy_indicative': b4.energy(start, end) if start and end else None,
        'repins': repin_summary(rec),
        'agreement': {'compared': len(scripted), 'agree': len(scripted) - sum(r['top'] != record.get(r['id'])
                                                                             for r in scripted), 'ids_differ': differ},
    }


def repin_summary(rec):
    """Re-pins that changed threads or failed, per trigger and process; drifted scans per process; every CPU group
    (cpuset / cgroup) value seen per process across the scans (more than one value: Android moved it)."""
    samples = rec.get('watch', {}).get('samples', [])
    acted = {}
    for e in rec.get('repins', []):
        if e.get('threads') or not e['ok']:
            k = f'{e["trigger"]}/{e["process"]}'
            acted[k] = acted.get(k, 0) + 1
    drift = {}
    for x in samples:
        for p in x.get('drift', {}):
            drift[p] = drift.get(p, 0) + 1
    seen = {}
    for x in samples:
        for k, v in x.get('groups', {}).items():
            seen.setdefault(k, set()).add(v)
    return {'acted': acted, 'failed': sum(not e['ok'] for e in rec.get('repins', [])), 'drifted_scans': drift,
            'groups_seen': {k: sorted(v) for k, v in seen.items()}}


def trace_max(path):
    """Highest VIRTUAL-SKIN in the heat control's own trace (its 5 s dumpsys rows), or None."""
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    except (OSError, ValueError):
        return None
    vals = [r['skin_c'] for r in rows if r.get('event') == 'dumpsys' and r.get('skin_c') is not None]
    return max(vals) if vals else None


def report(run, recs):
    code = run['definition']['files'].get(str(Path(__file__).resolve().relative_to(REPO)), '?')
    L = [f'MIND-B5 {run["utc"]}  code {code[:12]}  base {run["git_head"]}',
         f'server: {run["definition"]["commands"]["server"]}',
         f'launch-start skin {run["start"].get("skin")} degC (gate reference), battery '
         f'{run["start"].get("battery_pct")} %']
    if run['smoke']:
        L.append(f'*** {SMOKE_LABEL}: {run["work_s"]} s arms, gate before the first arm only; SMOKE HEAT THRESHOLDS '
                 '(not the deployed 37/35 degC): pause at arm-start skin + 1.0, resume at pause - 0.5 degC ***')
    L += ['timings: VALID arms only, from the owner run with no agent resident; energy INDICATIVE; '
          'A vs B descriptive', '']
    final = {}
    for r in recs:
        if r['label'] in ('VALID', SMOKE_LABEL) or r['arm'] not in final:
            final[r['arm']] = r
    for k in ARMS:
        if k not in final:
            L += [f'== {k}: not run', '']
            continue
        r, x = final[k], final[k]['metrics']
        L.append(f'== {k}: {r["label"]} (session {r["session"]}; {sum(q["arm"] == k for q in recs)} attempt(s); '
                 f'gate {r.get("gate")})')
        L += [f'  problems: {"; ".join(r["problems"])}'] if r['problems'] else []
        if run['smoke']:
            L.append(f'  smoke heat thresholds {r.get("heat_thresholds")}; smoke heat check: {x["pauses"]} pause(s) '
                     f'with camera stop, {x["resumes"]} resume(s) with camera start' +
                     ('' if x['pauses'] else '; NO PAUSE HAPPENED in this arm') +
                     ('' if x['resumes'] else '; NO RESUME HAPPENED in this arm'))
        L += [f'  decisions served {x["decisions_served"]}; events {x["events"]}; A ticks {x["ticks"]}',
              f'  latency event->decision s {x["latency_s"]}; wait s {x["wait_s"]}',
              f'  decision wall s {x["decision_wall_s"]}',
              f'  heat: pauses {x["pauses"]} (resumes {x["resumes"]}), first after {x["first_pause_s"]} s, pause '
              f'total {x["pause_s"]} s of '
              f'{x["arm_s"]} s, ACTIVE share {x["active_share"]}; skin {x["skin"]}; caps min {x["caps_min"]}',
              f'  camera period s {x["cycle_period_s"]}; detect ms {x["detect_ms"]}',
              f'  MemAvailable min {x["mem_available_min_kb"]} kB; energy (mAh, J) INDICATIVE {x["energy_indicative"]}',
              f'  agreement with B2 run of record: {x["agreement"]}',
              f'  load {r.get("load_s")} s; overrun {r.get("overrun_s")} s; server exit {r.get("exit_code")}, SIGKILL '
              f'{r.get("killed")}' + ('' if r.get('exit_code') in (0, None) and not r.get('killed')
                                      else '  FINDING (#145)'),
              f'  re-pins {x.get("repins")}',
              f'  watch: {len(r.get("watch", {}).get("samples", []))} scans, hits '
              f'{sum(len(s["hits"]) for s in r.get("watch", {}).get("samples", []))}; readings '
              + ', '.join(f'{c["tag"]}@{c.get("t")} skin {c.get("skin")} bat {c.get("battery_pct")} %'
                          for c in r['readings']), '']
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- sessions

def next_arms(recs):
    done = {r['arm'] for r in recs if r['label'] == 'VALID'}
    return [k for k in ARMS if k not in done]


def gate(ref_skin, label):
    """MEM-1's skin gate on this run's thermal log (as b4.gate)."""
    g = pm.thermal_gate(THERMAL_LOG, {'skin': ref_skin}, label, False)
    return {**g, 'passed': not g['warm_start']}


def session_loop(out, run, ctx, emit, alog):
    todo = list(ARMS) if ctx['smoke'] else next_arms(b4.read_jsonl(out / 'arms.jsonl'))
    for k in todo:
        if not ctx['smoke']:
            cr.wait_cores(f'arm {k}')
            if agents := repro.agents():
                return f'stopped before arm {k}: agent processes running: {agents[:120]!r}'
        if ctx['smoke'] and k != todo[0]:
            g = {'passed': True, 'skipped': 'smoke: gate before the first arm only'}
        else:
            g = gate(run['start']['skin'], f'arm {k}')
        if not g['passed']:
            return (f'deferred before arm {k}: skin {g.get("skin")} not <= {run["start"]["skin"]} + '
                    f'{cr.SKIN_GATE_C} degC after {g.get("waited_s")} s')
        pre = b4.reading(light=True)
        if pre.get('status') is None or pre['status'] >= b4.STATUS_STOP or pre.get('battery_c') is None or \
                pre['battery_c'] >= b4.BATTERY_MAX_C or pre.get('battery_pct') is None or \
                pre['battery_pct'] < BATTERY_STOP_PCT:
            return (f'stopped before arm {k}: status {pre.get("status")}, battery {pre.get("battery_c")} degC '
                    f'{pre.get("battery_pct")} %')
        if not ctx['smoke'] and not b4.cores_ok():
            return f'deferred before arm {k}: cores 4-7 lost after the gate'
        print(f'[arm {k}] gate {g}', flush=True)
        rec, rows = run_arm(k, out, ctx, emit)
        rec['gate'] = g
        alog.write(json.dumps(rec) + '\n')
        alog.flush()
        print(f'[arm {k}] {rec["label"]}' + (f': {"; ".join(rec["problems"])}' if rec['problems'] else '') +
              f'; {len(rows)} rows', flush=True)
        if rec.get('interrupted'):
            raise SystemExit(f'interrupted in arm {k}')
        if rec.get('battery_stop'):
            return f'arm {k}: battery below {BATTERY_STOP_PCT} %: session stopped'
        if rec['label'] == 'INVALID':
            return f'arm {k} INVALID: redo in a later session (--resume {out.name})'
    return 'all arms done' if ctx['smoke'] or not next_arms(b4.read_jsonl(out / 'arms.jsonl')) else 'arms remain'


def definition(out):
    files = {Path(x.__file__).resolve() for x in list(sys.modules.values()) if getattr(x, '__file__', None)}
    files = sorted({p for p in files if REPO in p.parents} | {HERE / 'oneshot.sh', RECORD, bc.TEST_SET})
    return {'files': {str(p.relative_to(REPO)): cr.sha256(p) for p in files},
            'commands': {'server': ' '.join(b4.server_command(out, 'P'))}}


def load_record():
    """{TEST id: action} of the B2 run of record, TEST V3 FWD; the file's SHA-256 is pinned."""
    if cr.sha256(RECORD) != RECORD_SHA256:
        raise SystemExit(f'refused: {RECORD} SHA-256 != pinned {RECORD_SHA256[:12]}')
    rows = [r for r in b4.read_jsonl(RECORD)
            if (r.get('set'), r.get('variant'), r.get('order')) == ('TEST', 'V3', 'FWD')]
    out = {r['id']: r['top'] for r in rows}
    if len(rows) != 40 or len(out) != 40:
        raise SystemExit(f'refused: run of record has {len(rows)} TEST V3 FWD rows')
    return out


def main_(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', metavar='RUN')
    a = ap.parse_args(argv)
    if a.smoke and a.resume:
        ap.error('--smoke runs alone')
    cr.SKIN_GATE_C = b4.SKIN_MATCH_C
    if not a.smoke:
        pm.GATE_MAX_S = GATE_MAX_S
        cr.require_native()
        if why := cm.refusals():
            raise SystemExit('not started: ' + '; '.join(why))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cr.exit_on_signal)
    test, _ = b2.load_set(bc.TEST_SET, b2.TEST_SHA256, custody=False)
    record = load_record()
    print('checking server, model and projector SHA-256 (about 30 s)', flush=True)
    bad = [f'{p}: SHA-256 != pinned {h[:12]}' for p, h in bc.PINS.items() if not Path(p).is_file() or cr.sha256(p) != h]
    bad += b4.vis3d.env_problems() + ([] if b0.repro.port_free() else ['port 8080 in use'])
    if bad:
        raise SystemExit('refused: ' + '; '.join(bad))
    start = b4.launch_start()  # every session: complete reading or a clean refusal, before any folder
    if why := battery_refusal(start['battery_pct'], a.smoke, a.resume):
        raise SystemExit(why)
    if a.resume:
        out = OUT_ROOT / a.resume
        if Path(a.resume).name != a.resume or not (out / 'run.json').is_file():
            raise SystemExit(f'not resumed: {out} is not a run folder')
        run = json.loads((out / 'run.json').read_text())
        if run['smoke']:
            raise SystemExit('not resumed: a smoke run')
        if diff := b4.definition_diff(run['definition'], definition(out)):
            raise SystemExit(f'not resumed: the definition differs from the run start: {diff}')
        if not next_arms(b4.read_jsonl(out / 'arms.jsonl')):
            raise SystemExit('not resumed: every arm already VALID')
    else:
        utc = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        out = OUT_ROOT / (utc + ('_smoke' if a.smoke else ''))
        out.mkdir(parents=True)  # a new folder; an existing one raises
        (out / 'slots').mkdir()
        run = {'utc': utc, 'smoke': a.smoke, 'work_s': SMOKE_S if a.smoke else ARM_S, 'start': start,
               'pins': bc.PINS, 'definition': definition(out),
               'git_head': bc.run_out('git', '-C', str(HERE), 'rev-parse', 'HEAD').strip(),
               'test_sha256': b2.TEST_SHA256, 'record_sha256': RECORD_SHA256, 'arms': list(ARMS),
               'offer_s': OFFER_S, 'tick_s': TICK_S, 'poll_s': POLL_S, 'skin_b_c': SKIN_B_C,
               'restore_every': RESTORE_EVERY, 'watch_s': b4.WATCH_S, 'overrun_max_s': OVERRUN_MAX_S,
               'no_new_max': b4.NO_NEW_MAX, 'request_timeout_s': b4.REQUEST_TIMEOUT_S, 'skin_gate_c': b4.SKIN_MATCH_C,
               'gate_max_s': pm.GATE_MAX_S, 'battery_start_pct': BATTERY_START_PCT,
               'battery_resume_pct': BATTERY_RESUME_PCT,
               'heat_thresholds': 'smoke: per arm (arms.jsonl heat_thresholds)' if a.smoke else
               {'pause_c': heat_control.SKIN_PAUSE_C, 'resume_c': heat_control.SKIN_RESUME_C,
                'min_active_s': heat_control.MIN_ACTIVE_S, 'min_pause_s': heat_control.MIN_PAUSE_S},
               'battery_stop_pct': BATTERY_STOP_PCT, 'real_mission': REAL_MISSION, 'real_current': REAL_CURRENT}
        (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
    k = len(list(out.glob('session??.json'))) + 1
    with open(out / f'session{k:02d}.json', 'x') as fh:
        fh.write(json.dumps({'session': k, 'started': cr.utc(), 'start': start,
                             'logger_sched': mem1.logger_sched(THERMAL_LOG)}, indent=1) + '\n')
    ctx = {'session': k, 'smoke': a.smoke, 'dur': run['work_s'], 'test': test, 'record': record}
    print(f'run folder {out}; session {k}', flush=True)
    reason = None
    try:
        with open(out / 'requests.jsonl', 'a') as rq, open(out / 'arms.jsonl', 'a') as alog:
            def emit(row):
                rq.write(json.dumps(row) + '\n')
                rq.flush()
            reason = session_loop(out, run, ctx, emit, alog)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        reason = reason or f'ended by {sys.exc_info()[1]!r}'[:300]
        with open(out / f'session{k:02d}_end.json', 'x') as fh:
            fh.write(json.dumps({'session': k, 'ended': cr.utc(), 'end_reason': reason,
                                 'agents_end': repro.agents(), 'port_free_end': b0.repro.port_free()}, indent=1) + '\n')
        text = report(run, b4.read_jsonl(out / 'arms.jsonl'))
        (out / 'report.txt').write_text(text)
        (out / 'artifacts.sha256').write_text(''.join(f'{cr.sha256(p)}  {p.relative_to(out)}\n'
                                                      for p in sorted(out.rglob('*'))
                                                      if p.is_file() and p.name != 'artifacts.sha256'))
        print(text + f'\nsession {k}: {reason}\n{out}', flush=True)


if __name__ == '__main__':
    main_()
