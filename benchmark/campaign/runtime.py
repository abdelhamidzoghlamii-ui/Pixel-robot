"""Lazy imports of unchanged benchmark runners; native operations owner only."""
import ast
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
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parents[1]
HOME = Path('/data/data/com.termux/files/home')

FALLBACK = ROBOT/'benchmark/relate_anything/desk2'
FALLBACK_HASHES = {
    'speed_photo.jpg': '4da2d789dd9a60405658184de81e9f46b3c55e09024a6f2ce26a934d899d0fd8',
    'speed_input.json': '35694e263f3894597e711bd2c2045bc55b3790f0bcd32d75baae11e60c20e32f',
}


class Blip(RuntimeError):
    """The answer carried nothing (H2, narrowed in FIX4C): no output at all, or an unreadable transient mask readback
    with empty output. Only this, a launch failure (an OSError while starting su/am/pgrep) or an HTTP timeout is re-read; every non-empty answer is
    judged by the reader's own base parser and post-parse checks and is never re-read."""


class ReadFailed(RuntimeError):
    """The single re-read failed again (Blip or launch failure): the read truly failed. Never re-read again."""


READ_PAUSE_S, PID_RECHECK_S, RETRY_CAP, PGREP_TIMEOUT_S, HEARTBEAT_S = .3, .5, 3, 10, 15
# Every H2 re-read of the run, from any thread: t (raw monotonic s), tid, what, first error, recovered.
READ_RETRIES = []


LAUNCH = (subprocess.Popen.__init__.__code__, subprocess.Popen._execute_child.__code__)


def launch_failure(e):
    """An OSError raised while a process was being started (inside subprocess.Popen: pipes, fork, exec), so before the
    transport returned any answer (FIX4D R1). Any other OSError (a later local read, a cleanup kill) is not one."""
    return isinstance(e, OSError) and any(frame.f_code in LAUNCH for frame, _ in traceback.walk_tb(e.__traceback__))


def reread(what, read, empty=None, pause=None):
    """H2 (narrowed, FIX4C/FIX4D): read() once more only when it raised Blip (the answer carried nothing) or a launch
    failure (an OSError from starting su/am/pgrep, before any answer), or returned a value for which empty(value) says the
    answer carried nothing. Every other answer or error, good or bad, is the reader's result and is never re-read."""
    try:
        value = read()
        why = empty(value) if empty else None
    except Blip as e:
        why = f'{type(e).__name__}: {e}'
    except OSError as e:
        if not launch_failure(e):
            raise
        why = f'launch failure: {type(e).__name__}: {e}'
    return retry(what, why, read, empty, pause) if why else value


def retry(what, why, read, empty=None, pause=None):
    """The single re-read after `pause` s (default READ_PAUSE_S), recorded in READ_RETRIES. Its own result stands and is
    judged like any first answer; if it carries nothing again the read fails closed (ReadFailed, or the empty value
    that its reader then rejects)."""
    row = dict(t=time.monotonic(), tid=threading.get_native_id(), what=what, first=why[:300], recovered=False)
    READ_RETRIES.append(row)
    time.sleep(READ_PAUSE_S if pause is None else pause)
    try:
        value = read()
    except (Blip, OSError) as e:  # ReadFailed, not Blip: no caller can re-read it again
        if isinstance(e, OSError) and not launch_failure(e):
            raise  # any other error stands as the reader's own
        raise ReadFailed(f'{what}: {type(e).__name__}: {e} (re-read once; first: {row["first"]})') from e
    row['recovered'] = not (empty and empty(value))
    return value


def retries_since(mark):
    tid = threading.get_native_id()
    return sum(r['tid'] == tid for r in READ_RETRIES[mark:])


def counted(read):
    """read()'s row with 'retries': the re-reads this thread made for it (H2 per-row record)."""
    mark = len(READ_RETRIES)
    row = read()
    row['retries'] = retries_since(mark)
    return row


def retry_check(record, mark):
    """H2 per phase: its re-reads; more than RETRY_CAP make the phase/block NOT VALID, never a session stop."""
    record['read_retries'] = READ_RETRIES[mark:]
    over = len(record['read_retries']) > RETRY_CAP
    record['read_retry_check'] = f'NOT VALID — READ RETRIES ({len(record["read_retries"])} > {RETRY_CAP})' if over else 'OK'
    return over



def blank(text):
    """An answer that carries nothing: no characters other than whitespace."""
    return not str(text).strip()


QUOTED = re.compile(r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")\s*$""")


def blank_repr(message):
    """True when a base parser's error message ends with the repr() of a blank answer (the parsers put the answer there)."""
    m = QUOTED.search(message)
    try:
        return m is not None and blank(ast.literal_eval(m.group(1)))
    except (ValueError, SyntaxError):
        return False


def su_stderr(cr):
    """su's stderr of this thread's last cr.root call (FIX4D: part of the answer); '' for a cr without root_stderr."""
    read = getattr(cr, 'root_stderr', None)
    return read() if callable(read) else ''


def nothing(cr, text):
    """The last cr.root answer carried nothing: its stdout AND su's stderr are blank (FIX4D)."""
    return blank(text) and blank(su_stderr(cr))


def shell_nothing(shell, lines):
    """A persistent-shell answer carried nothing: its stdout lines AND the shell's stderr lines are blank (FIX4D)."""
    return blank(''.join(lines)) and blank(''.join(getattr(shell, 'last_stderr', None) or []))


class Heard:
    """A persistent-shell proxy that remembers its last answer (stdout and su's stderr) only to tell whether it carried
    nothing (FIX4C/FIX4D); no value in it is judged here. `last` is None when the run raised (a shell timeout or a dead
    shell: not an empty answer)."""
    def __init__(self, shell):
        self.shell, self.last = shell, None
    def run(self, script, *args, **kwargs):
        self.last = None
        self.last = self.shell.run(script, *args, **kwargs)
        return self.last
    @property
    def empty(self):
        return self.last is not None and shell_nothing(self.shell, self.last)
    def __getattr__(self, name):
        return getattr(self.shell, name)



def causes(*lists):
    """M4: the first two underlying errors, 300 chars each."""
    return '; '.join(str(e)[:300] for e in [e for errors in lists for e in errors][:2])


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


def parent_pid(pid):
    """ppid from /proc/PID/stat (comm may hold spaces or parentheses, so split after the last ')')."""
    with open(f'/proc/{pid}/stat') as f:
        return int(f.read().rsplit(')', 1)[1].split()[1])


def own_process(pid):
    """True if pid is this runner or its descendant, by ppid ancestry read now; never by command text.

    Our su clients, root shells, pumps and diagnostics are our children, and their command lines can
    match the guard pattern (diagnostics' read_node matched 'node'). None: the matched pid itself
    exited before its ancestry was read, so it is no longer resident. Any other unreadable link
    (including an ancestor hidden by hidepid) raises: fail closed.
    """
    me, chain = os.getpid(), pid
    for _ in range(4096):
        if chain == me:
            return True
        if chain <= 1:
            return False
        try:
            chain = parent_pid(chain)
        except (FileNotFoundError, ProcessLookupError) as e:
            if chain == pid:
                return None
            raise RuntimeError(f'process ancestry unreadable: pid {pid} at {chain}: {e}') from e
        except (OSError, ValueError, IndexError) as e:
            raise RuntimeError(f'process ancestry unreadable: pid {pid} at {chain}: {e}') from e
    raise RuntimeError(f'process ancestry unreadable: pid {pid}: chain too long')


def clear_processes(server_pid=None):
    pattern = (r'claude|agy|node|codex|llama-server|chat\.py|main\.py|run_mission\.py|'
               r'benchmark/.*\.py|phase1\.py|lag_probe\.py|power_map\.py|coresidency\.py|thermal_char\.py|'
               r'duty_cycle\.py|camera_power\.py|speed_block\.py|session\.py|run_.*\.sh|oneshot\.sh')
    def pgrep():
        result = subprocess.run(['pgrep', '-fa', pattern], capture_output=True, text=True, timeout=PGREP_TIMEOUT_S)
        if blank(result.stdout + result.stderr) and result.returncode != 1:  # rc 1 = no match; else no output at all: nothing
            raise Blip(f'process enumeration failed with no output: rc {result.returncode}, {result.stderr[:200]!r}')
        if result.returncode not in (0, 1):
            raise RuntimeError('process enumeration failed: '+result.stderr)
        return result
    result = reread('pgrep', pgrep)
    others = []
    for line in result.stdout.splitlines():
        pid = int(line.split()[0])
        try:
            if pid == server_pid or own_process(pid) is not False:
                continue
        except RuntimeError as e:  # fail closed: unproven ancestry counts as resident
            line += f' [{e}]'
        others.append(line)
    if others:
        raise RuntimeError('agents/robot/other runners resident: '+'\n'.join(others))


def require_mask(mask, measured, allowed, process, how):
    if not mask or set(mask) & set(measured) or not set(mask) <= set(allowed):
        raise RuntimeError('root shell/monitor mask empty or includes inference cores: '
                           f'actual={sorted(mask)} measured={sorted(measured)} allowed={sorted(allowed)} '
                           f'process={process} how={how}')


def root_mask(sb, shell, pid):
    """sb.root_mask (base parser and checks); only an answer without output is a Blip."""
    heard = Heard(shell)
    try:
        return sb.root_mask(heard, pid)
    except RuntimeError as e:
        if heard.empty:
            raise Blip(f'empty answer: {e}') from e
        raise


def verify_monitor(sb, monitor, measured):
    shell, mask, pid, _ = monitor
    require_mask(mask, measured, mask, 'configured monitor', 'prepare result')
    for root_pid, process in ((pid, 'persistent shell'), (shell.p.pid, 'su')):
        observed = reread('root mask', lambda: root_mask(sb, shell, root_pid))
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
    os.sched_setaffinity(sentinel(), requested)
    actual = set(os.sched_getaffinity(0))
    try:
        cr.check_cores('campaign monitor setup after affinity refresh')
    except Exception as e:
        raise RuntimeError(f'campaign setup affinity restricted: before={sorted(before)} '
                           f'requested={sorted(requested)} actual={sorted(actual)} '
                           f'measured={sorted(measured)} allowed={sorted(actual)} '
                           f'process=calling thread how=os.sched_setaffinity/getaffinity: {e}') from e
    return dict(before=sorted(before), requested=sorted(requested), actual=sorted(actual))


_SENTINEL = []


def sentinel():
    """Native id of a parked thread kept at all CPUs by setup_affinity (L7). Android cpuset moves reach it as they
    reached the unpinned main thread, so check_cores keeps its meaning while the main thread is pinned."""
    if not _SENTINEL:
        thread = threading.Thread(target=threading.Event().wait, daemon=True, name='campaign_cores_sentinel')
        thread.start()
        _SENTINEL.append(thread.native_id)
    return _SENTINEL[0]


def check_cores(cr, when):
    """cr.check_cores, read on the sentinel: /proc/self/status is the main thread, pinned during phases (L7)."""
    allowed = os.sched_getaffinity(sentinel())
    if not set(range(4, 8)) <= allowed:
        raise cr.CoresLost(f'cores 4-7 not all allowed {when} (allowed {sorted(allowed)}, sentinel thread)')


def pin_main(mask):
    """L7: after setup the main thread (frame decode, bookkeeping) runs on the monitor mask only."""
    os.sched_setaffinity(0, mask)
    if os.sched_getaffinity(0) != set(mask):
        raise RuntimeError(f'main thread pin failed: actual={sorted(os.sched_getaffinity(0))} requested={sorted(mask)}')


def prepare_monitor(sb, measured):
    """sb.prepare_monitor (base). Its root shell is wrapped in Heard for the call (setup runs on the main thread before
    any monitor thread), so a failure whose last answer carried nothing (no mask line, no `echo $$`, no discover output)
    is a Blip and one fresh setup is tried; a failure on a non-empty answer fails as in the base. It closes its own
    shell on any failure."""
    _, _, _, cr, _ = imports()
    original, shells = cr.RootShell, []
    cr.RootShell = lambda: shells.append(Heard(original())) or shells[-1]
    try:
        shell, *rest = sb.prepare_monitor(measured)
        if isinstance(shell, Heard):  # hand back the real shell, with the attributes setup gave the proxy
            for name, value in vars(shell).items():
                if name not in ('shell', 'last'):  # (the real shell keeps its own last_stderr)
                    setattr(shell.shell, name, value)
            shell = shell.shell
        return (shell, *rest)
    except (RuntimeError, ValueError, IndexError) as e:
        if shells and shells[-1].empty:
            raise Blip(f'empty answer in monitor setup: {type(e).__name__}: {e}') from e
        raise
    except SystemExit as e:  # cr.discover's 'layout:' refusal; a signal's SystemExit is never touched
        if str(e).startswith('layout:') and shells and shells[-1].empty:
            raise Blip(f'empty answer in monitor setup: {e}') from e
        raise
    finally:
        cr.RootShell = original


def prepare(sb, measured, restrict=None):
    _, _, _, cr, _ = imports()
    affinity = setup_affinity(cr, measured)
    monitor = reread('monitor setup', lambda: prepare_monitor(sb, measured))
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


DUMP_ATTEMPTS, DUMP_RETRY_S, DUMP_PAUSE_S, RAW_KEEP = 3, 2.0, 0.5, 1024
# Copy of every complete read that needed a retry, from any caller, t in raw monotonic s (phase1 'thermal_retries').
THERMAL_RETRIES = []


def dump_once(cr, timeout=30):
    """cr.read_dump's read and parse, plus the raw output (key 'raw').

    The tag (file) is per thread: the idle/pause/block thermal worker and the main thread's checks
    shared coresidency_thermalservice.txt, so one read could cat it while the other truncated it
    (rc 0, no skin/status; the likely cause in owner_session_p1_fix3c/fix3c2).
    """
    start = time.monotonic()
    try:
        rc, text = cr.root('dumpsys thermalservice', f'campaign_thermal_{threading.get_native_id()}', timeout=timeout)
        d = {'rc': rc, **{k: v for k, v in cr.parse_dump(text).items() if k in ('status', 'skin')}}
    except (OSError, subprocess.SubprocessError) as e:
        text, d = '', {'rc': None, 'error': f'{type(e).__name__}: {e}', 'status': None, 'skin': None}
    return {'t': time.monotonic(), 't_start': start, **d, 'raw': text}


def read_dump(cr):
    """Checked thermal reader; returns the first complete read with its attempt count.

    Only an incomplete read (rc 0, no error, skin or status missing/unparsable, no stop status) is
    retried: at most DUMP_ATTEMPTS reads, DUMP_PAUSE_S apart, within DUMP_RETRY_S after the first read
    ended. A retry starts and is accepted only inside that window (its su timeout is the time left; a
    retry ending later, e.g. after a slow launch, counts as incomplete). Each incomplete attempt keeps its raw
    output (RAW_KEEP chars) in the row and THERMAL_RETRIES or, when no read completes, in the error.
    rc != 0, an error or a parsed status >= cr.STATUS_STOP raises at once; a persistent incomplete
    read raises when the attempts or the window run out.
    """
    began, incomplete, deadline = time.monotonic(), [], None
    for attempt in range(1, DUMP_ATTEMPTS+1):
        start = time.monotonic()
        d = dump_once(cr) if deadline is None else dump_once(cr, timeout=deadline-start)
        raw = d.pop('raw', '')
        why = thermal_failure(d)
        if not why and deadline is not None and time.monotonic() > deadline:
            why = f'retry ended after the {DUMP_RETRY_S} s window'
        if not why:
            d['attempts'] = attempt
            if incomplete:
                d['incomplete_attempts'] = incomplete
                THERMAL_RETRIES.append(dict(d))
            return d
        incomplete.append(dict(attempt=attempt, start_offset_s=start-began, elapsed_s=time.monotonic()-start,
                               why=why, raw=raw[:RAW_KEEP]))
        status = d.get('status')
        if (d.get('rc') != 0 or d.get('error') or (type(status) is int and status >= cr.STATUS_STOP)
                or attempt == DUMP_ATTEMPTS):
            break
        if deadline is None:
            deadline = time.monotonic() + DUMP_RETRY_S
        time.sleep(min(DUMP_PAUSE_S, max(0., deadline-time.monotonic())))
        if time.monotonic() >= deadline:
            break
    raise RuntimeError(f'thermal read failed: {why} (attempts {len(incomplete)}: {incomplete!r})')


def thermal_row_failures(rows):
    return [f"t={r.get('t')}: {why}" for r in rows if (why := thermal_failure(r))]


def dump_check(cr):
    d = read_dump(cr)
    if d['status'] >= cr.STATUS_STOP:
        raise RuntimeError('skin/status missing or thermal stop')
    return d


def camera_version(cr):
    def read():
        rc, text = cr.root('dumpsys package com.pixelrobot.robotcam', 'campaign_version')
        if nothing(cr, text):
            raise Blip(f'empty dumpsys package answer: rc {rc}')
        code = re.search(r'\bversionCode=(\d+)', text)
        name = re.search(r'\bversionName=([^\s]+)', text)
        if rc or not code or not name:
            raise RuntimeError('installed RobotCam version unreadable')
        return dict(versionCode=code.group(1), versionName=name.group(1))
    return reread('dumpsys package', read)



def camera_start(pm, rate=1):
    """pm.camera_start (base, with its own 3 attempts for a missing frame); only a launch failure of its FIRST `am start`
    (before any answer) is re-read. A later launch failure (after `am` already answered: a retry attempt or the in-loop
    STOP broadcast) fails closed as a RuntimeError; a non-zero `am` exit is judged as in the base (its output is
    discarded, so it cannot be shown empty)."""
    _, _, _, cr, _ = imports()
    def first_start(e):
        frames = [frame for frame, _ in traceback.walk_tb(e.__traceback__)]
        return any(a.f_code is pm.camera_start.__code__ and a.f_locals.get('attempt') == 1 and b.f_code is cr.am.__code__
                   for a, b in zip(frames, frames[1:]))
    def read():
        try:
            return pm.camera_start(rate)
        except OSError as e:
            if launch_failure(e) and not first_start(e):
                raise RuntimeError(f'am launch failure after an answer: {type(e).__name__}: {e}') from e
            raise
    return reread('am start RobotCam', read)


PIDOF = 'pidof com.pixelrobot.robotcam; echo "pidof_rc=$?"'


def robotcam_pidof(cr):
    """camera_end_check's pidof read and parse (same command, tag and fields); an answer without output is a Blip."""
    rc, text = cr.root(PIDOF, 'pidof')
    if nothing(cr, text):
        raise Blip(f'empty pidof answer: rc {rc}')
    status = re.search(r'^pidof_rc=(\d+)$', text, re.M)
    return dict(pidof_root_rc=rc, pidof_rc=int(status.group(1)) if status else None,
                pids_after_force_stop=[w for w in text.split() if w.isdigit()])



def stop_camera(cr):
    """Base: STOP broadcast (an `am` launch failure is re-read), then camera_end_check, which runs ONCE, so its
    capture_stopped result stands (FIX4B R3-3). Its two root reads are recorded and judged each on its own answer (a
    force-stop that ran keeps its answer even if the later pidof fails; camera_end_check's except would overwrite it):
    - force-stop that did not run (launch failure incl. an empty pinned readback, or rc != 0 with no output): the
      force-stop alone is run once more; it never replaces the pidof's own answer (FIX4C R2-1);
    - pidof: read if it never ran; read once more if it did not run or answered nothing; a pid listed by a successful
      pidof (root rc 0, pidof rc 0) is re-checked once after PID_RECHECK_S (owner rule); any other answer stands.
    Every non-empty answer is judged by camera_end_failed as in the base."""
    heard = {}
    def rootf(command, tag, timeout=60):
        heard[tag] = None
        try:
            answer = cr.root(command, tag, timeout=timeout)
        except Blip as e:  # the pinned wrapper read back no mask: the command did not run
            heard[tag] = e
            raise OSError(str(e)) from e  # camera_end_check records it as force_stop_rc None
        except BaseException as e:
            heard[tag] = e
            raise
        heard[tag] = (*answer, su_stderr(cr))  # (rc, text, su's stderr): the whole answer (FIX4D)
        return answer
    def said_nothing(answer):
        return isinstance(answer, tuple) and blank(answer[1]) and blank(answer[2])
    def did_not_run(answer):
        return isinstance(answer, Blip) or launch_failure(answer) or said_nothing(answer) and answer[0] != 0
    try:
        reread('am broadcast STOP', cr.camera_stop)
    finally:
        check = cr.camera_end_check(rootf=rootf)
    forced, pidof = heard.get('forcestop'), heard.get('pidof')
    if isinstance(forced, tuple):  # it ran: its own answer stands
        check.update(force_stop_rc=forced[0], force_stop_output=forced[1].strip()[:200])
    if did_not_run(forced):
        def again():
            rc, text = cr.root('am force-stop com.pixelrobot.robotcam', 'forcestop')
            return dict(check, force_stop_rc=rc, force_stop_output=text.strip()[:200])
        check = retry('am force-stop', f'force-stop did not run: {forced!r}', again)
    if check.get('force_stop_rc') != 0:
        pass  # a failed force-stop fails below; its pidof (if any) is not read again
    elif did_not_run(forced) and 'pidof' not in heard:  # it never ran (the force-stop raised first): its first read
        check = reread('RobotCam pidof after force-stop', lambda: dict(check, **robotcam_pidof(cr)))
    elif did_not_run(pidof) or said_nothing(pidof):
        check = retry('RobotCam pidof after force-stop', f'pidof answered nothing: {pidof!r}', lambda: dict(check, **robotcam_pidof(cr)))
    elif check.get('pidof_root_rc') == 0 and check.get('pidof_rc') == 0 and check.get('pids_after_force_stop'):
        check = retry('RobotCam pidof after force-stop', f"RobotCam pids {check['pids_after_force_stop']} after force-stop",
                      lambda: dict(check, **robotcam_pidof(cr)), pause=PID_RECHECK_S)
    if why := cr.camera_end_failed(check):
        raise RuntimeError('camera cleanup: '+why)
    return check


APP_ONLY = "su rc 0; PSS missing for ['robotcam_app']: "


def memory_sample(cr, server_pid, camera_on):
    """MemAvailable plus one root battery/PSS read.

    coresidency.root_sample expects the RobotCam app in every state. With the camera OFF the
    app is force-stopped, so its absence alone is "not running", not a read failure. Root rc,
    battery fields, runner/server/provider PSS and a missing app with the camera ON still fail.
    H2 (FIX4C): only an answer without output is read again (root_sample's answer_blank: its whole su answer was blank);
    every non-empty answer is judged by the base parser and the phase's memory predicate.
    """
    def empty(s):
        return 'empty root_sample answer' if s.get('answer_blank') else None
    s = reread('root_sample', lambda: dict(t=time.monotonic(), **cr.meminfo_mib(), **cr.root_sample(server_pid)), empty)
    # Exact root_sample message when rc is 0 and the app is the only missing process.
    if not camera_on and s.get('pss_error', '').startswith(APP_ONLY):
        # root_sample discards pidof's status, so confirm absence: pidof exits 1 only when nothing matches.
        def confirm():
            rc, out = cr.root(PIDOF, 'campaign_pidof')
            if nothing(cr, out):
                raise Blip(f'empty pidof answer: rc {rc}')
            return rc, out
        try:
            rc, out = reread('pidof', confirm)
        except ReadFailed as e:  # any other error (e.g. a transient mask mismatch) propagates as before
            s['pss_error'] += f'; app absence unconfirmed: {str(e)[:300]}'
        else:
            if rc == 0 and out.split() == ['pidof_rc=1']:
                del s['pss_error']
                s['pss_kb']['robotcam_app'] = None
                s['robotcam_app'] = 'not running (camera OFF)'
            else:
                s['pss_error'] += f'; app absence unconfirmed: rc {rc}, {out[:100]!r}'
    return s


def battery_sample(sb):
    """sb.battery_sample (base parser and checks); only an answer without output is a Blip (its message ends with the
    answer's repr); a launch failure (an OSError while starting su) is re-read by reread."""
    def read():
        try:
            return sb.battery_sample()
        except RuntimeError as e:
            if str(e).startswith('root battery read failed') and blank_repr(str(e)) and blank(su_stderr(sb.cr)):
                raise Blip(str(e)) from e
            raise
    return reread('battery', read)


def server_ok(server):
    """/health fails only on 2 consecutive failures (H2). server.alive() also checks that the process runs, so
    an exited server fails both reads."""
    return reread('/health', server.alive, lambda ok: None if ok else '/health not ok (1 of 2)')


SCREEN = ("dumpsys power | grep -m3 -E 'mWakefulness=|Display Power|mHoldingDisplay'; echo ===; "
          "dumpsys activity activities | grep -m1 -E 'topResumedActivity|mResumedActivity'")


def screen_state(cr):
    """H5: wakefulness/display lines and the resumed activity, so a lock or a lost foreground shows in the evidence.
    Evidence only: an answer without output is read once more; nothing is ever raised and no phase fails on it."""
    def read():
        rc, text = cr.root(SCREEN, 'campaign_screen', timeout=20)
        if nothing(cr, text):
            raise Blip(f'empty screen state answer: rc {rc}')
        return rc, text
    try:
        rc, text = reread('dumpsys power/activity', read)
    except Exception as e:
        return dict(error=f'{type(e).__name__}: {e}'[:300])
    power, marker, activity = text.partition('===')
    if not marker or blank(power) and blank(activity):  # no separator, or nothing around it: recorded, not re-read
        return dict(rc=rc, error=f'unparsable: {text[:200]!r}')
    return dict(rc=rc, power=[line.strip() for line in power.splitlines() if line.strip()],
                resumed_activity=activity.strip()[:300])


def screen(sb):
    """sb.Screen whose settings commands are read again only when they answered nothing: a su failure with no output
    ('screen setting failed: ' + output) or a `settings get` with no output. Every non-empty answer goes to Screen's own
    checks (timeout value, readback) as in the base."""
    s = sb.Screen()
    setting = s.setting
    def read(command):
        try:
            out = setting(command)
        except RuntimeError as e:
            if str(e).startswith('screen setting failed: ') and nothing(sb.cr, str(e)[len('screen setting failed: '):]):
                raise Blip(str(e)) from e
            raise
        if command.startswith('settings get') and nothing(sb.cr, out):
            raise Blip(f'empty answer to {command!r}')
        return out
    s.setting = lambda command: reread('settings', lambda: read(command))
    return s


def release(caller):
    if caller:
        caller.close()
    gc.collect()


def fast_check(cr, monitor, sb=None, measured=None):
    if sb is not None:
        verify_monitor(sb, monitor, measured)
    def read():
        heard = Heard(monitor[0])
        row = cr.fast_sample(heard, cr.fast_keys(monitor[3]))
        if heard.empty:  # an answer without output: the only re-read (FIX4C)
            raise Blip(f"empty fast answer: {row.get('error')}")
        if row.get('error') or row.get('bat_c') is None or row.get('cpu_c') is None or any(v is None for v in row['max'].values()):
            raise RuntimeError('missing root CPU/battery/policy reading' + (f": {row['error']}" if row.get('error') else ''))
        return row
    row = reread('fast sample', read)
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
            message = (f'transient root shell pin/readback failed: actual={text!r} '
                       f'measured={sorted(measured)} allowed={sorted(mask)} '
                       f'process=transient shell (pid in output) how=taskset -p $$ rc={rc}')
            # A readback that answered nothing (empty actual=) ran no command: a Blip for the caller's one re-read.
            # Any non-empty readback that differs, or a failed pin (97), is an affinity failure: never re-read.
            if rc == 98 and re.fullmatch(r'\s*CAMPAIGN_ROOT_MASK pid=\d+ actual=\s*', text) and blank(su_stderr(cr)):  # the whole answer
                raise Blip(message)
            raise RuntimeError(message)
        text = '\n'.join(line for line in text.splitlines() if not line.startswith('CAMPAIGN_ROOT_MASK '))
        return rc, text
    cr.root = bound
    try:
        yield
    finally:
        cr.root = original
