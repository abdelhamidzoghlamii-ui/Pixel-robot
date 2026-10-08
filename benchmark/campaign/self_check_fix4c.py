"""CAMPAIGN_P1_FIX4C focused offline mocks: the narrowed H2 re-read. For EVERY reader wrapped by the re-read helper,
through its actual (base) parser with only the transport mocked: a non-empty answer (good, partial, malformed or bad)
is read once and judged by the base checks; an answer that carried nothing (empty output, a launch failure, an empty
transient mask readback) is re-read once and fails closed if the re-read is empty or bad. Then every probe of the six
FIX4/FIX4B review rounds, re-run on this code (table). Never runs hardware, root or models."""
import ast
import contextlib
import io
import itertools
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import diagnostics as d
import phase1 as p
import power
import runtime as rt
from self_check_fix3d import FULL, PARTIAL_STOP, fake_cr

v, sb, pm, cr, _ = rt.imports()
HERE = Path(__file__).resolve().parent
BASE = '56296995ea780141d757ba74ee768e07e27c2979'


# ---------------------------------------------------------------- read counting

def launch_error():
    """A real launch failure: the OSError subprocess.Popen raises for a missing executable (its traceback is inside
    Popen, which is what runtime.launch_failure recognises; FIX4D R1)."""
    try:
        subprocess.Popen(['/nonexistent/campaign-launch-failure'])
    except OSError as e:
        return e
    raise AssertionError('missing executable started')


def run(call):
    """call() with the re-read pause skipped: (its value or exception, the re-reads it recorded)."""
    mark = len(rt.READ_RETRIES)
    with patch.object(rt.time, 'sleep'):
        try:out = call()
        except (Exception, SystemExit) as e:out = e
    return out, rt.READ_RETRIES[mark:]


def failed(out):
    return isinstance(out, BaseException)


def outcome(out):
    return (f'{type(out).__name__}: {out}' if failed(out) else 'returned ' + repr(out)).replace('\n', ' ')[:120]


def once(reader, check, reads=1):
    """A non-empty answer: `reads` transport calls, no re-read, never a Blip/ReadFailed; check(out) holds."""
    call, transport = reader
    out, retries = run(call)
    assert not isinstance(out, (rt.Blip, rt.ReadFailed)), outcome(out)
    assert transport.call_count == reads and not retries, (transport.call_count, retries, outcome(out))
    assert check(out), outcome(out)
    return f'{transport.call_count} read(s), no re-read: ' + outcome(out)


def again(reader, check, reads=2, rereads=1):
    """An answer that carried nothing: exactly one recorded re-read (per read that carried nothing), then check(out) holds."""
    call, transport = reader
    out, retries = run(call)
    assert transport.call_count == reads and len(retries) == rereads, (transport.call_count, retries, outcome(out))
    assert check(out), outcome(out)
    return f'{transport.call_count} reads, {rereads} re-read(s): ' + outcome(out)


ok = lambda out: not failed(out)
closed = lambda out: isinstance(out, rt.ReadFailed) and 're-read once' in str(out)


def raises(needle, kind=BaseException):
    return lambda out: isinstance(out, kind) and needle in str(out)


def sweep(name, make, good, nonempty, empty, bad, twice=closed):
    """The invariant for one reader: good and every non-empty answer -> 1 read; each empty answer -> one re-read,
    recovering on a good answer and failing closed on an empty (twice) or bad (bad[1]) re-read."""
    once(make(good, good), ok)
    for answer, check in nonempty:
        once(make(answer, good), check)
    for blank in empty:
        again(make(blank, good), ok)
        again(make(blank, blank), twice)
        again(make(blank, bad[0]), bad[1])
    SWEPT.append(name)
    print(f'PASS {name}: {1+len(nonempty)} non-empty answers read once; {len(empty)} empty answers re-read once '
          '(recovered / failed closed when empty or bad again)')


SWEPT = []


# ---------------------------------------------------------------- readers (actual parsers, transport mocked)

GOOD_POWER = ['-1000000', '4000000', 'Discharging']


def power_reader(*answers, average=False):
    shell = NS(run=MagicMock(side_effect=list(answers)))
    return (lambda: power.sample(shell, '/b', 0, average)), shell.run


LAYOUT = {'cpu_zones': {'BIG': '9', 'MID': '10', 'LITTLE': '11'}, 'policies': {'policy0': 1, 'policy4': 1, 'policy6': 1}}
GOOD_FAST = ['50000', '51000', '52000', '300', '1803000', '2350000', '2850000']  # cr.fast_keys(LAYOUT) order


def fast_reader(*answers):
    shell = NS(run=MagicMock(side_effect=list(answers)))
    return (lambda: rt.fast_check(cr, (shell, {0}, 1, LAYOUT))), shell.run


def fast(**at):
    lines = list(GOOD_FAST)
    for i, value in at.items():lines[int(i[1:])] = value
    return lines


GOOD_MASK = ['Cpus_allowed_list:\t0-1']


def mask_reader(*answers, expected={0, 1}, measured={4, 5}):
    """verify_monitor; the transport counts the persistent shell's (pid 12) reads, the su pid answers GOOD."""
    read = MagicMock(side_effect=list(answers))
    good = [f'Cpus_allowed_list:\t{min(expected)}-{max(expected)}']
    shell = NS(p=NS(pid=10), monitor_tids=set(), run=lambda command, timeout=5: read() if '/proc/12/' in command else good)
    def call():
        with patch.object(rt.os, 'cpu_count', return_value=8):
            return rt.verify_monitor(sb, (shell, set(expected), 12, {}), set(measured))
    return call, read


DISCOVER = ([f'{cr.THERMAL}/thermal_zone{i}/type\t{t}' for i, t in ((9, 'BIG'), (10, 'MID'), (11, 'LITTLE'))] +
            [f'{cr.CPUFREQ}/policy{n}/cpuinfo_max_freq\t1000' for n in (0, 4, 6)])
GOOD_SETUP = {}


def setup_reader(*answers):
    """rt.prepare -> the real sb.prepare_monitor; each answer overrides GOOD (echo/mask/discover) for one root shell;
    the transport counts root shells (one per setup read)."""
    answers = list(answers)
    def make():
        a = dict(dict(echo=['12'], mask=['Cpus_allowed_list:\t0-3'], discover=DISCOVER), **answers.pop(0))
        def run(command, timeout=5):
            key = ('echo' if command == 'echo $$' else 'mask' if 'Cpus_allowed_list' in command else
                   'discover' if 'cpuinfo_max_freq' in command else None)
            if key is None:return []  # taskset
            if isinstance(a[key], BaseException):raise a[key]
            return a[key]
        return NS(p=NS(pid=10), run=run, close=MagicMock())
    shells = MagicMock(side_effect=make)
    def call():
        with patch.object(cr, 'RootShell', shells), patch.object(pm, 'thread_ids', side_effect=itertools.cycle([{1}, {1, 11}])), \
             patch.object(rt.os, 'sched_getaffinity', return_value=set(range(8))), patch.object(rt.os, 'sched_setaffinity'), \
             patch.object(rt.os, 'cpu_count', return_value=8), patch.object(rt, 'setup_affinity', return_value={}), \
             patch.object(rt, 'verify_monitor'):
            return rt.prepare(sb, {4, 5})
    return call, shells


def pgrep_reader(*answers):
    results = [a if isinstance(a, BaseException) else NS(returncode=a[0], stdout=a[1], stderr=a[2] if len(a) > 2 else '')
               for a in answers]
    run = MagicMock(side_effect=results)
    def call():
        with patch.object(rt.subprocess, 'run', run), patch.object(rt, 'own_process', return_value=False):
            return rt.clear_processes()
    return call, run


def root_reader(call_with):
    """A reader over cr.root answers (rc, text) or raised exceptions."""
    def make(*answers):
        root = MagicMock(side_effect=list(answers))
        return (lambda: call_with(root)), root
    return make


def patched_root(root, call, *patches):
    with contextlib.ExitStack() as stack:
        for m in (patch.object(cr, 'root', root), patch.object(cr, 'meminfo_mib', return_value={}), *patches):
            stack.enter_context(m)
        return call()


GOOD_BATTERY = (0, '-1000000\n4000000\nDischarging\n')
battery_reader = root_reader(lambda root: patched_root(root, lambda: rt.battery_sample(sb), patch.object(sb, 'peak_rss_kib', return_value=1)))

FULL_SAMPLE = ('-1000000\n4000000\nDischarging\n=== runner 1\nTOTAL PSS: 100\n=== llama_server 2\nTOTAL PSS: 100\n'
               '=== robotcam_app 3\nTOTAL PSS: 100\n=== camera_provider 4\nTOTAL PSS: 100\n')
sample_reader = root_reader(lambda root: patched_root(root, lambda: rt.memory_sample(cr, 2, True)))
row_bad = lambda out: not failed(out) and bool(p.memory_row_bad(out))
row_good = lambda out: not failed(out) and not p.memory_row_bad(out)

APP_ONLY = dict(root_rc=0, battery_status='Discharging', pss_pids={}, pss_error=rt.APP_ONLY + "'x'")
pidof_reader = root_reader(lambda root: patched_root(root, lambda: rt.memory_sample(cr, 2, False),
                                                     patch.object(cr, 'root_sample', side_effect=lambda pid: dict(APP_ONLY, pss_kb={}))))
unconfirmed = lambda out: not failed(out) and 'app absence unconfirmed' in out.get('pss_error', '')

capacity_reader = root_reader(lambda root: d.battery(NS(BATTERY='/b', root=root), 25))

NODES = '/b/capacity\t85\n/b/temp\t287\n/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq\t1401000\n'
diagnostics_reader = root_reader(lambda root: patched_root(root, lambda: d.snapshot(NS(BATTERY='/b', root=root), True),
                                                           patch.object(rt, 'read_dump', return_value=dict(status=0, skin=30., attempts=1)),
                                                           patch.object(rt, 'screen_state', return_value={})))

version_reader = root_reader(lambda root: rt.camera_version(NS(root=root)))

GOOD_LOGCAT = (0, 'logcat_rc=0\nhead\n\n=== lmk lines\n')
KILL = '0.5 lmkd: Kill com.example\n'
logcat_reader = root_reader(lambda root: patched_root(root, lambda: p.lmk_window(cr, 0, 1)))
lmk_failed = lambda out: not failed(out) and out['ok'] is False

SCREEN_TEXT = '  mWakefulness=Awake\n===\n  topResumedActivity=x\n'
screen_reader = root_reader(lambda root: rt.screen_state(NS(root=root)))
recorded = lambda out: not failed(out) and 'error' in out


def settings_reader(*answers, readback=None):
    """Screen.start (real sb.Screen.setting over cr.root): the transport serves the first `settings get` (old value), or
    with readback= the get after the put; the other commands answer correctly."""
    first, back = MagicMock(side_effect=list(answers)), MagicMock(side_effect=readback)
    value = []
    def root(command, tag, timeout=60):
        if command.startswith('settings put'):
            value.append(command.split()[-1])
            return 0, ''
        return first() if not value else back() if readback else (0, value[-1]+'\n')
    def call():
        with patch.object(cr, 'root', root), patch.object(sb.subprocess, 'run', return_value=NS(returncode=0)), \
             contextlib.redirect_stdout(io.StringIO()):
            screen = rt.screen(sb)
            screen.start()
            return screen.old
    return call, (back if readback else first)


def am_reader(start):
    def make(*answers):
        run = MagicMock(side_effect=[NS(returncode=0) if a == 0 else a for a in answers])
        def call():
            with patch.object(cr.subprocess, 'run', run), patch.object(cr, 'read_frame', return_value=dict(status='ok', session='s', frame=1)), \
                 patch.object(cr, 'camera_end_check', return_value=dict(END_OK)):
                return rt.camera_start(pm) if start else rt.stop_camera(cr)
        return call, run
    return make


END_OK = dict(capture_stopped=True, force_stop_rc=0, pidof_root_rc=0, pidof_rc=1, pids_after_force_stop=[])
READBACK_OK = (0, 'CAMPAIGN_ROOT_MASK pid=78 actual=x\nversionCode=7 versionName=1.0\n')
READBACK_EMPTY = (98, 'CAMPAIGN_ROOT_MASK pid=77 actual=\n')
READBACK_SPLIT = (98, "CAMPAIGN_ROOT_MASK pid=77 actual=\npid 77's current affinity mask: 30\n")  # empty marker, mask after it
READBACK_MISMATCH = (98, "CAMPAIGN_ROOT_MASK pid=77 actual=pid 77's current affinity mask: 30\n")


def readback_reader(*answers):
    root = MagicMock(side_effect=list(answers))
    fake = NS(root=root)
    def call():
        with rt.root_mask_scope(fake, sb, {0, 1}, {4, 5}):
            return rt.camera_version(fake)
    return call, root


def end_reader(*answers, frames=False):
    """stop_camera with the real camera_end_check on a fake clock; the transport is cr.root (force-stop, pidof)."""
    root = MagicMock(side_effect=list(answers))
    clock, frame = [0.], itertools.count(1)
    def read_frame(directory, **k):
        return dict(status='ok', session='s', frame=next(frame)) if frames else dict(status='missing')
    def sleep(dt):
        call.sleeps.append(dt)
        clock[0] += dt
    def call():
        with patch.object(cr, 'camera_stop'), patch.object(cr, 'read_frame', side_effect=read_frame), patch.object(cr, 'root', root), \
             patch.object(cr.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(cr.time, 'sleep', side_effect=sleep):
            return rt.stop_camera(cr)
    call.sleeps = []
    return call, root


# ---------------------------------------------------------------- the invariant, reader by reader

def check_readers():
    assert pm.cr is cr and sb.cr is cr
    sweep('power (power.sample)', power_reader, GOOD_POWER,
          [(['-1000000', '+0', 'Discharging'], raises('invalid battery voltage')), (['-1000000', '0'], raises('power read missing')),
           (['', '0', 'Discharging'], raises('invalid literal')), (['-1', '4000000', 'Charging'], raises('charger connected')),
           (['-1000000', '4000000', '123'], raises('charger connected')), (['x'], raises('power read missing')),
           (RuntimeError('root shell: no answer in 5 s'), raises('no answer in 5 s'))],
          [[], ['', '', '']], (['-1', '4000000', 'Charging'], raises('charger connected')))
    sweep('fast sample (rt.fast_check over cr.fast_sample)', fast_reader, GOOD_FAST,
          [(fast(i3='450'), raises('battery thermal stop')), (fast(i3='450')[:6], raises('missing root CPU/battery/policy reading')),
           (fast(i3='450', i4=''), raises('missing root')), (fast(i0='110000', i4=''), raises('missing root')),
           (fast(i3='1001')[:6], raises('missing root')), (fast(i0='200001')[:6], raises('missing root')),
           (RuntimeError('root shell: no answer in 5 s'), raises('no answer in 5 s'))],
          [[], [''] * 7], (fast(i3='450'), raises('battery thermal stop')))
    sweep('root mask (verify_monitor over sb.root_mask)', mask_reader, GOOD_MASK,
          [(['Cpus_allowed_list:\t0'], raises('root/su affinity changed')), (['Cpus_allowed_list:\t0-5'], raises('includes inference cores')),
           (['Cpus_allowed_list:\t0-5,'], raises('missing/malformed')), (['Cpus_allowed_list:\t0-1', 'Cpus_allowed_list:\t0-5'], raises('missing/malformed')),
           (['Cpus_allowed_list:\t3-0'], raises('invalid root CPU range')), (['Cpus_allowed_list:\t0-8'], raises('invalid root CPU range')),
           (['garbage'], raises('missing/malformed')), (RuntimeError('root shell: no answer in 5 s'), raises('no answer in 5 s'))],
          [[], ['']], (['Cpus_allowed_list:\t0'], raises('root/su affinity changed')))
    sweep('monitor setup (rt.prepare over sb.prepare_monitor)', setup_reader, GOOD_SETUP,
          [(dict(mask=['Cpus_allowed_list:\t4-7']), raises('includes inference cores')), (dict(mask=['Cpus_allowed_list:\t0-3,']), raises('missing/malformed')),
           (dict(mask=['Cpus_allowed_list:\t3-0']), raises('invalid root CPU range')), (dict(mask=['Cpus_allowed_list:\t0-8']), raises('invalid root CPU range')),
           (dict(discover=DISCOVER[:4]), raises('missing required CPU frequency policies')), (dict(discover=['garbage']), raises('layout:')),
           (dict(echo=['x']), raises('invalid literal')), (dict(mask=RuntimeError('root shell: no answer in 5 s')), raises('no answer in 5 s'))],
          [dict(echo=[]), dict(mask=[]), dict(discover=[])], (dict(mask=['Cpus_allowed_list:\t4-7']), raises('includes inference cores')))
    sweep('pgrep (clear_processes)', pgrep_reader, (1, ''),
          [((0, 'garbage line\n'), raises('invalid literal')), ((1, '2001 codex\n'), raises('resident: 2001 codex')),
           ((0, '2001 node /usr/bin/claude\n'), raises('resident: 2001 node')), ((2, 'x\n', 'bad'), raises('process enumeration failed: bad')),
           ((2, '', 'pgrep: permission denied'), raises('process enumeration failed: pgrep: permission denied')),
           ((0, '', 'warning'), ok),  # 5629699 judgement of rc 0 with no process line: nothing resident
           (subprocess.TimeoutExpired('pgrep', 10), raises('timed out'))],
          [(0, ''), (2, '', ''), (0, ' \n', '\n'), launch_error()], ((0, '2001 codex\n'), raises('resident: 2001 codex')))
    sweep('battery (rt.battery_sample over sb.battery_sample)', battery_reader, GOOD_BATTERY,
          [((0, '-1000000\n0\n'), raises('root battery read failed')), ((0, '-1\n4000000\nCharging\n'), raises('charger/battery status: Charging')),
           ((0, '-1\n4000000\n123\n'), raises('charger/battery status: 123')), ((0, '-1\n4000000\n123\nx\n'), raises('root battery read failed')),
           ((1, 'su: denied'), raises('root battery read failed'))],
          [(0, ''), (1, ''), (1, '\n'), launch_error()], ((0, '-1\n4000000\nCharging\n'), raises('Charging')))
    sweep('root_sample (rt.memory_sample over cr.root_sample)', sample_reader, (0, FULL_SAMPLE),
          [((0, '-1000000\n0\nDischarging\n'), row_bad), ((0, 'garbage'), row_bad), ((1, FULL_SAMPLE), row_bad),
           ((0, FULL_SAMPLE.replace('Discharging', 'Charging')), row_bad), ((0, '\n'*200 + '-1000000\n4000000\nCharging\n'), row_bad)],
          [(0, ''), (1, ''), (0, '\n'*300)], ((0, FULL_SAMPLE.replace('Discharging', 'Charging')), row_bad), twice=row_bad)
    again(sample_reader(launch_error(), (0, FULL_SAMPLE)), row_good)
    again(sample_reader(launch_error(), launch_error()), closed)
    sweep('pidof (camera-OFF app absence in rt.memory_sample)', pidof_reader, (0, 'pidof_rc=1\n'),
          [((1, '4321\npidof_rc=0\n'), unconfirmed), ((0, '4321\npidof_rc=0\n'), unconfirmed), ((0, 'garbage'), unconfirmed),
           (RuntimeError('transient root shell pin/readback failed: mismatch'), raises('pin/readback failed'))],
          [(0, ''), (1, ''), launch_error()], ((1, '4321\npidof_rc=0\n'), unconfirmed), twice=unconfirmed)
    sweep('battery capacity (diagnostics.battery)', capacity_reader, (0, '85'),
          [((0, '101\ngarbage'), raises('unreadable')), ((0, '+20\ngarbage'), raises('unreadable')), ((0, '-1\ngarbage'), raises('unreadable')),
           ((0, '+101\ngarbage'), raises('unreadable')), ((0, '101'), raises('unreadable')), ((0, '24'), raises('below 25%', d.BatteryStop)),
           ((1, '85'), raises('unreadable')), ((1, '20'), raises('unreadable'))],
          [(0, ''), (1, ''), launch_error()], ((0, '24'), raises('below 25%', d.BatteryStop)))
    sweep('diagnostics nodes (diagnostics.snapshot)', diagnostics_reader, (0, NODES),
          [((0, '/b/capacity\t85\n'), raises('required diagnostics unreadable')), ((1, NODES), raises('diagnostics root read failed')),
           ((0, NODES.replace('287', 'UNREADABLE')), raises('required diagnostics unreadable'))],
          [(0, ''), (1, ''), launch_error()], ((1, NODES), raises('diagnostics root read failed')))
    sweep('dumpsys package (camera_version)', version_reader, (0, 'versionCode=7 versionName=1.0'),
          [((0, 'versionCode=7'), raises('version unreadable')), ((1, 'versionCode=7 versionName=1.0'), raises('version unreadable')),
           ((1, 'Unable to find package'), raises('version unreadable'))],
          [(0, ''), (1, ''), launch_error()], ((1, 'x'), raises('version unreadable')))
    sweep('logcat (phase1.lmk_window over cr.lmk_lines)', logcat_reader, GOOD_LOGCAT,
          [((0, 'logcat_rc=1\n'+KILL), lmk_failed), ((0, 'logcat_rc=1\nerr\n\n=== lmk lines\n'+KILL), lambda o: lmk_failed(o) and o['n_kills'] == 1),
           ((0, 'logcat_rc=0\n\n=== lmk lines\n'+KILL), lambda o: ok(o) and o['n_kills'] == 1),
           ((0, 'logcat_rc=0\n\n=== lmk lines\nnan lmkd: kill x\n'), lmk_failed), ((0, 'garbage'), lmk_failed),
           ((0, '\n'*400 + 'logcat_rc=1\n'+KILL), lmk_failed), ((0, '=== lmk lines\ngarbage\n'), lmk_failed)],
          [(0, ''), (1, ''), (0, '\n'*500)], ((0, 'logcat_rc=1\n'+KILL), lmk_failed), twice=lmk_failed)
    again(logcat_reader(launch_error(), GOOD_LOGCAT), lambda o: ok(o) and o['ok'])
    again(logcat_reader(launch_error(), launch_error()), closed)
    sweep('settings (rt.screen over sb.Screen.setting)', settings_reader, (0, '60000\n'),
          [((0, '60000\ngarbage'), raises('invalid screen timeout')), ((1, 'error: x'), raises('screen setting failed: error: x')),
           ((0, 'abc'), raises('invalid screen timeout'))],
          [(0, ''), (1, ''), (0, '\n'), launch_error()], ((0, 'abc'), raises('invalid screen timeout')))
    sweep('dumpsys power/activity (screen_state)', screen_reader, (0, SCREEN_TEXT),
          [((1, '===\n'), recorded), ((0, ' \n===\n \n'), recorded), ((0, 'garbage'), recorded), ((1, 'Error: x'), recorded)],
          [(0, ''), (1, ' \n'), launch_error()], ((0, 'garbage'), recorded), twice=recorded)
    for name, start in (('am start RobotCam (camera_start over pm.camera_start)', True), ('am broadcast STOP (stop_camera)', False)):
        sweep(name, am_reader(start), 0,
              [(subprocess.CalledProcessError(1, 'am'), raises('non-zero exit status 1')), (subprocess.TimeoutExpired('am', 10), raises('timed out'))],
              [launch_error()], (subprocess.CalledProcessError(1, 'am'), raises('non-zero exit status 1')))
    sweep('transient mask readback (root_mask_scope)', readback_reader, READBACK_OK,
          [(READBACK_MISMATCH, raises('pin/readback failed')), ((97, 'CAMPAIGN_ROOT_MASK pid=77 actual=\n'), raises('pin/readback failed')),
           ((98, 'CAMPAIGN_ROOT_MASK pid=77 actual=garbage\n'), raises('pin/readback failed')),
           (READBACK_SPLIT, raises('pin/readback failed'))],
          [READBACK_EMPTY, (98, '\nCAMPAIGN_ROOT_MASK pid=77 actual= \n')], (READBACK_MISMATCH, raises('pin/readback failed')))
    check_stop_camera()
    check_health_and_thermal()
    check_all_wrapped()


def check_stop_camera():
    """am force-stop / pidof after STOP (camera_end_check runs once; only what did not run or answered nothing is redone)."""
    pidof_ok, still = (0, 'pidof_rc=1\n'), (0, '4321\npidof_rc=0\n')
    blip = rt.Blip('transient root shell pin/readback failed: actual=...')
    once(end_reader((0, ''), pidof_ok), ok, 2)
    for answers, needle in ((((1, 'Error: unknown package'), pidof_ok), 'am force-stop failed (rc 1'),
                            (((1, 'Error: unknown package'), blip), 'am force-stop failed (rc 1'),  # FIX4B R3-2
                            (((0, ''), (0, 'garbage')), 'absence after force-stop unconfirmed'),
                            (((0, ''), (1, 'su: error')), 'absence after force-stop unconfirmed'),
                            (((0, ''), (1, 'su: error 123\npidof_rc=2\n')), 'RobotCam still running after force-stop (pids'),
                            (((0, ''), (0, '123\npidof_rc=2\n')), 'RobotCam still running after force-stop (pids')):
        once(end_reader(*answers), raises(needle), 2)
    once(end_reader(subprocess.TimeoutExpired('su', 60)), raises('am force-stop failed (rc None'), 1)
    # Did not run (launch failure, empty readback, rc != 0 with no output): force-stop and pidof once more.
    for first in (launch_error(), blip):
        again(end_reader(first, (0, ''), pidof_ok), ok, 3)
        again(end_reader(first, first), raises('am force-stop'), 2)
    # ... alone: the pidof's own first answer is never replaced by the re-run (FIX4C R2-1).
    again(end_reader((1, ''), pidof_ok, (0, ''), still), ok, 3)
    again(end_reader((1, ''), pidof_ok, (1, ''), pidof_ok), raises('am force-stop failed (rc 1'), 3)
    for first_pidof in ((0, 'pidof_rc=2\n'), (0, 'garbage\n'), (1, 'su: error')):
        again(end_reader((1, ''), first_pidof, (0, ''), pidof_ok), raises('absence after force-stop unconfirmed'), 3)
    again(end_reader((1, ''), still, (0, ''), pidof_ok), ok, 4, 2)  # a successful pidof listing pids: owner re-check
    again(end_reader((1, ''), (0, ''), (0, ''), pidof_ok), ok, 4, 2)  # an empty pidof: its own re-read
    # pidof answered nothing or did not run: pidof once more; fails closed if it answers nothing again.
    for first in ((0, ''), launch_error(), blip):
        again(end_reader((0, ''), first, pidof_ok), ok, 3)
    again(end_reader((0, ''), (0, ''), (0, '')), raises('RobotCam pidof after force-stop: Blip: empty pidof answer'), 3)
    again(end_reader((0, ''), launch_error(), launch_error()), raises('RobotCam pidof after force-stop'), 3)
    # Owner rule: a remaining pid is re-checked once after 0.5 s (only a successful pidof listing pids: root rc 0, pidof rc 0).
    reader = end_reader((0, ''), still, pidof_ok)
    again(reader, ok, 3)
    assert rt.PID_RECHECK_S in reader[0].sleeps
    again(end_reader((0, ''), still, still), raises('RobotCam still running after force-stop'), 3)
    # FIX4B R3-3: capture not stopped stays failed through a pidof readback blip (the end check runs once).
    end = MagicMock(wraps=cr.camera_end_check)
    with patch.object(cr, 'camera_end_check', end):
        again(end_reader((0, ''), blip, pidof_ok, frames=True), raises('capture not shown stopped'), 3)
    assert end.call_count == 1
    SWEPT.extend(['am force-stop (stop_camera)', 'RobotCam pidof after force-stop (stop_camera)'])
    print('PASS am force-stop / pidof (stop_camera over the real camera_end_check): non-empty answers read once (rc 1 with '
          'output, malformed pidof, timeout); a force-stop that did not run is re-run once with its pidof; an empty pidof '
          'is read once more; a remaining pid is re-checked once after 0.5 s; capture_stopped=False is kept')


def check_health_and_thermal():
    def health(*answers):
        alive = MagicMock(side_effect=list(answers))
        return (lambda: rt.server_ok(NS(alive=alive))), alive
    once(health(True), lambda out: out is True)
    again(health(False, True), lambda out: out is True)
    again(health(False, False), lambda out: out is False)
    SWEPT.append('/health (server_ok)')
    # Thermal (FIX3D rule, not the H2 helper): the reader's source is unchanged and a partial stop fails at once.
    base = subprocess.run(['git', '-C', str(HERE), 'show', f'{BASE}:benchmark/campaign/runtime.py'],
                          capture_output=True, text=True, check=True).stdout
    def functions(source):
        return {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    old, new = functions(base), functions((HERE/'runtime.py').read_text())
    for name in ('thermal_failure', 'dump_once', 'read_dump', 'thermal_row_failures', 'dump_check'):
        assert old[name] == new[name], name
    tcr, queue = fake_cr([(0, PARTIAL_STOP), (0, FULL)])
    try:rt.read_dump(tcr)
    except RuntimeError as e:assert 'status 4 (attempts 1' in str(e)
    else:raise AssertionError('partial stop status retried away')
    assert len(queue) == 1
    print('PASS /health: one failure read once more (2 of 2 rule), two fail; thermal reader identical to 5629699 (AST), '
          'a partial status-4 answer fails on the first read')


def check_all_wrapped():
    """Every re-read call site in the campaign code is swept above (by its `what` name)."""
    names = set()
    for path in ('runtime.py', 'power.py', 'diagnostics.py', 'phase1.py', 'lag_probe.py'):
        for node in ast.walk(ast.parse((HERE/path).read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, 'attr', getattr(node.func, 'id', None)) in ('reread', 'retry') \
               and node.args and isinstance(node.args[0], ast.Constant):
                names.add(node.args[0].value)
    swept = ' | '.join(SWEPT)
    missing = sorted(n for n in names if n not in swept)
    assert not missing, missing
    print(f'PASS every wrapped reader swept ({len(names)} re-read call names): ' + ', '.join(sorted(names)))


# ---------------------------------------------------------------- probes of the six review rounds

def probes():
    """(round finding, probe, its result on this code). Each is a non-empty answer read once and judged by the base, or
    an answer that carried nothing, re-read once and failing closed when it carries nothing again."""
    blip = rt.Blip('transient root shell pin/readback failed: actual=...')
    lag = dict(expected={0, 1, 2, 3}, measured={6, 7})
    return [
        ('FIX4 R1-2', 'fast: battery 45.0 C next to a missing policy, healthy re-read queued',
         lambda: once(fast_reader(fast(i3='450', i4=''), GOOD_FAST), raises('missing root CPU/battery/policy reading'))),
        ('FIX4 R1-3', 'pgrep rc 0 with empty stdout (twice)', lambda: again(pgrep_reader((0, ''), (0, '')), closed)),
        ('FIX4 R1-3', 'pgrep rc 1 with a listed line', lambda: once(pgrep_reader((1, '2001 codex\n'), (1, '')), raises('resident: 2001 codex'))),
        ('FIX4 R1-3', 'pgrep rc 0 with a malformed line', lambda: once(pgrep_reader((0, 'garbage line\n'), (1, '')), raises('invalid literal'))),
        ('FIX4 R1-5', 'screen state: su answer (1, empty), then nothing again', lambda: again(screen_reader((1, ''), (1, '')), recorded)),
        ('FIX4 R1-5', 'screen-timeout settings get answering nothing, twice', lambda: again(settings_reader((0, ''), (0, '')), closed)),
        ('FIX4 R2-1', "power average ['-1000000','0','Discharging','garbage']",
         lambda: once(power_reader(['-1000000', '0', 'Discharging', 'garbage'], GOOD_POWER+['-1'], average=True), raises('invalid battery voltage'))),
        ('FIX4 R2-1', "power ['', '0', 'Discharging']", lambda: once(power_reader(['', '0', 'Discharging'], GOOD_POWER), raises('invalid literal'))),
        ('FIX4 R2-2', "settings get answering '' (then '')", lambda: again(settings_reader((0, ''), (0, '')), closed)),
        ('FIX4 R2-2', "screen state (1, '===\\n')", lambda: once(screen_reader((1, '===\n'), (0, SCREEN_TEXT)), recorded)),
        ('FIX4 R2-2', 'transient readback empty actual= (twice)', lambda: again(readback_reader(READBACK_EMPTY, READBACK_EMPTY), closed)),
        ('FIX4 R2-2', 'transient readback mismatch', lambda: once(readback_reader(READBACK_MISMATCH, READBACK_OK), raises('pin/readback failed'))),
        ('FIX4 R2-2', 'setup discover answer without the required policies',
         lambda: once(setup_reader(dict(discover=DISCOVER[:4]), {}), raises('missing required CPU frequency policies'))),
        ('FIX4 R3-1', 'fast: 7-field layout, battery 450, final policy line missing',
         lambda: once(fast_reader(fast(i3='450')[:6], GOOD_FAST), raises('missing root CPU/battery/policy reading'))),
        ('FIX4 R3-2', "power ['-1000000','0']", lambda: once(power_reader(['-1000000', '0'], GOOD_POWER), raises('power read missing'))),
        ('FIX4 R3-2', "power average ['-1000000','0','Discharging']",
         lambda: once(power_reader(['-1000000', '0', 'Discharging'], GOOD_POWER+['-1'], average=True), raises('power read missing'))),
        ('FIX4 R3-3', 'capture not stopped, force-stop ok, pidof readback blip',
         lambda: again(end_reader((0, ''), blip, (0, 'pidof_rc=1\n'), frames=True), raises('capture not shown stopped'), 3)),
        ('FIX4B R1-1', 'fast: BIG 110000 with one blank policy', lambda: once(fast_reader(fast(i0='110000', i4=''), GOOD_FAST), raises('missing root'))),
        ('FIX4B R1-1', 'fast: BIG 110000, shortened answer', lambda: once(fast_reader(fast(i0='110000')[:6], GOOD_FAST), raises('missing root'))),
        ('FIX4B R1-2', "su battery (0, '-1000000\\n0\\n')", lambda: once(battery_reader((0, '-1000000\n0\n'), GOOD_BATTERY), raises('root battery read failed'))),
        ('FIX4B R1-2', 'root_sample voltage 0 with provider PSS missing',
         lambda: once(sample_reader((0, FULL_SAMPLE.replace('4000000', '0').split('=== camera_provider')[0]), (0, FULL_SAMPLE)), row_bad)),
        ('FIX4B R1-3', "capacity '101\\ngarbage'", lambda: once(capacity_reader((0, '101\ngarbage'), (0, '90')), raises('unreadable'))),
        ('FIX4B R1-3', "capacity '-1\\ngarbage'", lambda: once(capacity_reader((0, '-1\ngarbage'), (0, '90')), raises('unreadable'))),
        ('FIX4B R1-4', "setup mask '3-0' plus a malformed line",
         lambda: once(setup_reader(dict(mask=['Cpus_allowed_list:\t3-0', 'Cpus_allowed_list:\tx']), {}), raises('missing/malformed'))),
        ('FIX4B R1-4', "setup mask CPU 8 on 8 cores", lambda: once(setup_reader(dict(mask=['Cpus_allowed_list:\t0-8']), {}), raises('invalid root CPU range'))),
        ('FIX4B R1-5', "after put: readback (1, '60000')",
         lambda: once(settings_reader((0, '60000\n'), readback=[(1, '60000'), (0, '2147483647')]), raises('screen setting failed: 60000'))),
        ('FIX4B R1-5', "after put: readback '60000\\ngarbage'",
         lambda: once(settings_reader((0, '60000\n'), readback=[(0, '60000\ngarbage'), (0, '2147483647')]), raises('screen timeout set/readback failed'))),
        ('FIX4B R2-1', 'root_sample voltage 0, complete PSS (base parser has no voltage check)',
         lambda: once(sample_reader((0, FULL_SAMPLE.replace('4000000', '0')), (0, FULL_SAMPLE)), row_good)),
        ('FIX4B R2-2', 'logcat ok=False, rc 1, in-window kill line',
         lambda: once(logcat_reader((0, 'logcat_rc=1\nerr\n\n=== lmk lines\n'+KILL), GOOD_LOGCAT), lambda o: lmk_failed(o) and o['n_kills'] == 1)),
        ('FIX4B R2-3', "power ['-1000000','4000000','123']", lambda: once(power_reader(['-1000000', '4000000', '123'], GOOD_POWER), raises('charger connected'))),
        ('FIX4B R2-3', "su battery status '123'", lambda: once(battery_reader((0, '-1\n4000000\n123\n'), GOOD_BATTERY), raises('status: 123'))),
        ('FIX4B R2-4', "mask '0-5,'", lambda: once(mask_reader(['Cpus_allowed_list:\t0-5,'], GOOD_MASK), raises('missing/malformed'))),
        ('FIX4B R2-4', "mask '3-0,'", lambda: once(mask_reader(['Cpus_allowed_list:\t3-0,'], GOOD_MASK), raises('missing/malformed'))),
        ('FIX4B R3-1', "power ['-1000000','+0','Discharging']",
         lambda: once(power_reader(['-1000000', '+0', 'Discharging'], GOOD_POWER), raises('invalid battery voltage'))),
        ('FIX4B R3-1', "capacity '+20\\ngarbage'", lambda: once(capacity_reader((0, '+20\ngarbage'), (0, '90')), raises('unreadable'))),
        ('FIX4B R3-1', "capacity '+101\\ngarbage'", lambda: once(capacity_reader((0, '+101\ngarbage'), (0, '90')), raises('unreadable'))),
        ('FIX4B R3-1', 'root_sample voltage +0, complete PSS (base parser has no voltage check)',
         lambda: once(sample_reader((0, FULL_SAMPLE.replace('4000000', '+0')), (0, FULL_SAMPLE)), row_good)),
        ('FIX4B R3-2', "force-stop (1, 'Error: unknown package'), then pidof readback blip",
         lambda: once(end_reader((1, 'Error: unknown package'), blip, (0, ''), (0, 'pidof_rc=1\n')), raises('am force-stop failed (rc 1'), 2)),
        ('FIX4B R3-3', 'logcat partial header with a kill, separator missing',
         lambda: once(logcat_reader((0, 'logcat_rc=1\n'+KILL), GOOD_LOGCAT), lmk_failed)),
        ('FIX4B R3-4', 'fast: battery 1001, shortened answer', lambda: once(fast_reader(fast(i3='1001')[:6], GOOD_FAST), raises('missing root'))),
        ('FIX4B R3-4', 'fast: CPU 200001, shortened answer', lambda: once(fast_reader(fast(i0='200001')[:6], GOOD_FAST), raises('missing root'))),
        ('FIX4B R3-5', "lag mask config: '0-4,' then '0-3'",
         lambda: once(mask_reader(['Cpus_allowed_list:\t0-4,'], ['Cpus_allowed_list:\t0-3'], **lag), raises('missing/malformed'))),
        ('FIX4B R3-6', "power average ['-1000000','4000000','123']",
         lambda: once(power_reader(['-1000000', '4000000', '123'], GOOD_POWER+['-1'], average=True), raises('power read missing'))),
        ('FIX4B R3-6', "su battery four tokens with status '123'",
         lambda: once(battery_reader((0, '-1\n4000000\n123\nx\n'), GOOD_BATTERY), raises('root battery read failed'))),
        ('FIX4C R1-1', 'root_sample: 200 newlines, then Charging, PSS missing (blank 200-char excerpt)',
         lambda: once(sample_reader((0, '\n'*200 + '-1000000\n4000000\nCharging\n'), (0, FULL_SAMPLE)), row_bad)),
        ('FIX4C R1-2', 'logcat: 400 newlines, then logcat_rc=1 and a kill, no separator',
         lambda: once(logcat_reader((0, '\n'*400 + 'logcat_rc=1\n'+KILL), GOOD_LOGCAT), lmk_failed)),
        ('FIX4C R1-2', "logcat '=== lmk lines\\ngarbage\\n'", lambda: once(logcat_reader((0, '=== lmk lines\ngarbage\n'), GOOD_LOGCAT), lmk_failed)),
        ('FIX4C R1-3', "force-stop ok, pidof (1, 'su: error 123\\npidof_rc=2\\n')",
         lambda: once(end_reader((0, ''), (1, 'su: error 123\npidof_rc=2\n'), (0, 'pidof_rc=1\n')), raises('RobotCam still running'), 2)),
        ('FIX4C R1-4', 'readback rc 98: empty marker line, then a different mask line',
         lambda: once(readback_reader(READBACK_SPLIT, READBACK_OK), raises('pin/readback failed'))),
        ('FIX4C R2-1', "force-stop (1, ''), pidof (0, 'pidof_rc=2\\n'), re-run force-stop (0, ''), healthy pidof queued",
         lambda: again(end_reader((1, ''), (0, 'pidof_rc=2\n'), (0, ''), (0, 'pidof_rc=1\n')), raises('absence after force-stop unconfirmed'), 3)),
        ('FIX4C R1-5', "pgrep rc 2, stdout empty, stderr 'pgrep: permission denied'",
         lambda: once(pgrep_reader((2, '', 'pgrep: permission denied'), (1, '')), raises('process enumeration failed: pgrep'))),
    ]


NOT_READER_PROBES = {
    'FIX4 R1-1': 'camera counter vs catch-up (H4): self_check_fix4.check_round1',
    'FIX4 R1-4': 'D2 extension power coverage: self_check_fix4.check_round1',
    'FIX4 R1-6': 'memory-row cause in the block error (M4): self_check_fix4.check_round1',
    'FIX4 R1-7': 'screen state on failure paths, setup END line (H5): self_check_fix4.check_round1',
    'FIX4 R2-3': 'lag probe per-row re-reads and cap: self_check_fix4.check_round2 (empty fast answers)',
    'FIX4 R2-4': 'cleanup failures in rehearsal coverage: self_check_fix4.check_round2',
    'FIX4 R2-5': 'preflight keeps an unwinding CoresLost: self_check_fix4.check_round2',
}


def check_probes():
    print('| Finding | Probe (healthy re-read queued) | FIX4C result |\n|---|---|---|')
    for finding, probe, thunk in probes():
        print(f'| {finding} | {probe} | {thunk()} |')
    for finding, where in NOT_READER_PROBES.items():
        print(f'| {finding} | not a re-read probe | regression kept: {where} |')
    print('PASS every review-round probe: each non-empty answer fails closed (or is judged as in 5629699) on its first read; '
          'each answer that carried nothing is re-read once and fails closed when empty again')


if __name__ == '__main__':
    check_readers()
    check_probes()
    print('PASS CAMPAIGN_P1_FIX4C self-check (offline; agents resident; proot; NOT VALID for timing)')
