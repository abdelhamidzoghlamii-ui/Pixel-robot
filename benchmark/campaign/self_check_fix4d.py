"""CAMPAIGN_P1_FIX4D focused offline mocks: su's stderr is part of every answer. Through the REAL cr.root (subprocess
stdout AND stderr mocked) and the REAL RootShell (a fake `su` on PATH): a stderr-only su failure is a non-empty answer,
read once and judged by the base rules; an answer with blank stdout AND blank stderr is re-read once and fails closed if
the re-read is empty or bad. Then every earlier probe table (FIX4, FIX4B, FIX4C R1-R2 via self_check_fix4c, FIX4C R4
here) on this code. Never runs hardware, root or models."""
import contextlib
import io
import itertools
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import diagnostics as d
import phase1 as p
import power
import runtime as rt
import self_check_fix4c as c
from self_check_fix4c import again, closed, failed, ok, once, raises

v, sb, pm, cr, _ = rt.imports()
DENIED = (1, '', 'su: permission denied\n')  # a refused su: no stdout, a readable stderr
EMPTY = (1, '', '')  # nothing at all
GOOD_BATTERY = (0, '-1000000\n4000000\nDischarging\n', '')
NODES = (f'{cr.BATTERY}/capacity\t85\n{cr.BATTERY}/temp\t287\n'
         '/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq\t1401000\n')


def su(call, *patches):
    """A reader over the REAL cr.root: answers are (rc, stdout, stderr) of `su -c` or raised exceptions; the transport
    counts su calls only (other subprocess calls, e.g. termux-wake-lock, answer rc 0)."""
    def make(*answers):
        answers = list(answers)
        transport = MagicMock()
        def run(argv, **kw):
            if argv[0] != 'su':
                return NS(returncode=0, stdout='', stderr='')
            transport()
            a = answers.pop(0)
            if isinstance(a, BaseException):raise a
            return NS(returncode=a[0], stdout=a[1], stderr=a[2])
        def wrapped():
            with contextlib.ExitStack() as stack:
                for m in (patch.object(cr.subprocess, 'run', side_effect=run), patch.object(cr, 'meminfo_mib', return_value={}),
                          patch.object(sb, 'peak_rss_kib', return_value=1), contextlib.redirect_stdout(io.StringIO()), *patches):
                    stack.enter_context(m)
                try:
                    return call()
                finally:
                    cr._root_local.stderr = ''  # no stale stderr for later tests that replace cr.root itself
        return wrapped, transport
    return make


clock = [0.]
def end_patches():
    frame = itertools.count(1)
    return (patch.object(cr, 'camera_stop'), patch.object(cr, 'read_frame', return_value=dict(status='missing')),
            patch.object(cr.time, 'monotonic', side_effect=lambda: clock[0]),
            patch.object(cr.time, 'sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0]+dt)))


battery = su(lambda: rt.battery_sample(sb))
memory = su(lambda: rt.memory_sample(cr, 2, True))
APP_ONLY = dict(root_rc=0, battery_status='Discharging', pss_pids={}, pss_error=rt.APP_ONLY + "'x'", answer_blank=False)
pidof_confirm = su(lambda: rt.memory_sample(cr, 2, False), patch.object(cr, 'root_sample', side_effect=lambda pid: dict(APP_ONLY, pss_kb={})))
version = su(lambda: rt.camera_version(cr))
logcat = su(lambda: p.lmk_window(cr, 0, 1))
capacity = su(lambda: d.battery(cr, 25))
diagnostics = su(lambda: d.snapshot(cr, True), patch.object(rt, 'read_dump', return_value=dict(status=0, skin=30., attempts=1)),
                 patch.object(rt, 'screen_state', return_value={}))
screen_state = su(lambda: rt.screen_state(cr))
def start_screen():
    screen = rt.screen(sb)
    screen.start()
    return screen.old
settings = su(start_screen)
def stop():
    with contextlib.ExitStack() as stack:
        for m in end_patches():stack.enter_context(m)
        return rt.stop_camera(cr)
end = su(stop)
def pinned():
    with rt.root_mask_scope(cr, sb, {0, 1}, {4, 5}):
        return rt.camera_version(cr)
readback = su(pinned)

GOOD_SAMPLE = (0, c.FULL_SAMPLE, '')
GOOD_LOGCAT = (0, 'logcat_rc=0\nhead\n\n=== lmk lines\n', '')
row_bad, row_good, lmk_failed, recorded, unconfirmed = c.row_bad, c.row_good, c.lmk_failed, c.recorded, c.unconfirmed
SU_READERS = [  # name, reader, good, stderr-only check, bad (answer, check), empty-twice check
    ('battery (rt.battery_sample)', battery, GOOD_BATTERY, raises('root battery read failed'),
     ((0, '-1\n4000000\nCharging\n', ''), raises('Charging')), closed),
    ('memory/PSS (rt.memory_sample, cr.root_sample)', memory, GOOD_SAMPLE, row_bad,
     ((0, c.FULL_SAMPLE.replace('Discharging', 'Charging'), ''), row_bad), row_bad),
    ('camera-OFF pidof (rt.memory_sample)', pidof_confirm, (0, 'pidof_rc=1\n', ''), unconfirmed,
     ((1, '4321\npidof_rc=0\n', ''), unconfirmed), unconfirmed),
    ('package version (camera_version)', version, (0, 'versionCode=7 versionName=1.0', ''), raises('version unreadable'),
     ((1, 'x', ''), raises('version unreadable')), closed),
    ('logcat (phase1.lmk_window, cr.lmk_lines)', logcat, GOOD_LOGCAT, lmk_failed,
     ((0, 'logcat_rc=1\n0.5 lmkd: Kill com.example\n', ''), lmk_failed), lmk_failed),
    ('capacity (diagnostics.battery)', capacity, (0, '85', ''), raises('unreadable'),
     ((0, '24', ''), raises('below 25%', d.BatteryStop)), closed),
    ('diagnostics nodes (diagnostics.snapshot)', diagnostics, (0, NODES, ''), raises('diagnostics root read failed'),
     ((1, NODES, ''), raises('diagnostics root read failed')), closed),
    ('screen state (screen_state)', screen_state, (0, c.SCREEN_TEXT, ''), recorded, ((0, 'garbage', ''), recorded), recorded),
]


def check_su_readers():
    for name, make, good, denied, bad, twice in SU_READERS:
        once(make(good, good), ok)
        once(make(DENIED, good), denied)
        once(make((0, '', 'su: warning: x\n'), good), lambda out: True)  # rc 0 + stderr: non-empty, judged by the base
        again(make(EMPTY, good), ok)
        again(make(EMPTY, EMPTY), twice)
        again(make(EMPTY, bad[0]), bad[1])
        print(f'PASS {name}: stderr-only su failure read once and fails closed; blank stdout+stderr re-read once '
              '(recovered / failed closed when empty or bad again)')
    # settings: the first `settings get` (then put and readback answer correctly)
    rest = [(0, '', ''), (0, '2147483647\n', '')]
    once(settings((0, '60000\n', ''), *rest), ok, 3)
    once(settings(DENIED, *rest), raises('screen setting failed: '))
    again(settings(EMPTY, (0, '60000\n', ''), *rest), ok, 4)
    again(settings(EMPTY, EMPTY), closed)
    again(settings(EMPTY, (0, 'abc', '')), raises('invalid screen timeout'))
    print('PASS settings (rt.screen, sb.Screen.setting): stderr-only su failure read once; blank re-read once')
    # force-stop and pidof (real camera_end_check over the real cr.root)
    pidof_ok = (0, 'pidof_rc=1\n', '')
    once(end((0, '', ''), pidof_ok), ok, 2)
    once(end(DENIED, pidof_ok), raises('am force-stop failed (rc 1'), 2)
    once(end((0, '', ''), DENIED), raises('absence after force-stop unconfirmed'), 2)
    again(end(EMPTY, pidof_ok, (0, '', '')), ok, 3)
    again(end(EMPTY, pidof_ok, EMPTY), raises('am force-stop failed (rc 1'), 3)
    again(end((0, '', ''), EMPTY, pidof_ok), ok, 3)
    again(end((0, '', ''), EMPTY, EMPTY), raises('RobotCam pidof after force-stop: Blip'), 3)
    again(end((0, '', ''), EMPTY, DENIED), raises('absence after force-stop unconfirmed'), 3)
    print('PASS force-stop / pidof (rt.stop_camera, real camera_end_check): stderr-only failures read once and fail; '
          'blank ones re-read once')
    # transient readback over the real cr.root: an empty actual= with su stderr is a non-empty answer
    empty_actual = 'CAMPAIGN_ROOT_MASK pid=77 actual=\n'
    good = (0, 'CAMPAIGN_ROOT_MASK pid=78 actual=x\nversionCode=7 versionName=1.0\n', '')
    once(readback((98, empty_actual, 'su: warning\n'), good), raises('pin/readback failed'))
    again(readback((98, empty_actual, ''), good), ok)
    again(readback((98, empty_actual, ''), (98, empty_actual, '')), closed)
    print('PASS transient readback (root_mask_scope over the real cr.root): with su stderr read once; blank re-read once')


@contextlib.contextmanager
def fake_su():
    """The REAL RootShell over a fake `su` (a plain sh) on PATH."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp)/'su').write_text('#!/bin/sh\nexec sh\n')
        (Path(tmp)/'su').chmod(0o755)
        with patch.dict(os.environ, PATH=tmp + os.pathsep + os.environ['PATH']):
            shell = cr.RootShell()
            try:yield shell, Path(tmp)
            finally:shell.close()


class Counted:
    """A RootShell proxy counting run() calls; after() runs after each call (e.g. to change the files)."""
    def __init__(self, shell, after=None):
        self.shell, self.after, self.run_count = shell, after, 0
    def run(self, script, *args, **kwargs):
        self.run_count += 1
        try:return self.shell.run(script, *args, **kwargs)
        finally:
            if self.after:self.after(self.run_count)
    def __getattr__(self, name):
        return getattr(self.shell, name)


def check_rootshell_readers():
    with fake_su() as (shell, tmp):
        battery_dir = tmp/'battery'
        battery_dir.mkdir()
        def write(current, voltage, status):
            for name, value in (('current_now', current), ('voltage_now', voltage), ('status', status)):
                (battery_dir/name).write_text(value)
        # stderr-only: a missing battery directory (each read prints only to stderr)
        proxy = Counted(shell)
        out, retries = c.run(lambda: power.sample(proxy, str(tmp/'missing'), 0))
        assert proxy.run_count == 1 and not retries and failed(out) and 'power read missing' in str(out), (proxy.run_count, out)
        assert shell.last_stderr, 'stderr captured'
        # blank stdout and stderr (empty files): re-read once; recovered when the files fill, fails closed when not
        write('', '', '')
        proxy = Counted(shell, after=lambda n: write('-1000000', '4000000', 'Discharging'))
        out, retries = c.run(lambda: power.sample(proxy, str(battery_dir), 0))
        assert proxy.run_count == 2 and len(retries) == 1 and out['battery_w'] == 4., (proxy.run_count, out)
        write('', '', '')
        proxy = Counted(shell)
        out, retries = c.run(lambda: power.sample(proxy, str(battery_dir), 0))
        assert proxy.run_count == 2 and len(retries) == 1 and closed(out), out
        write('', '', '')
        proxy = Counted(shell, after=lambda n: write('-1', '4000000', 'Charging'))
        out, retries = c.run(lambda: power.sample(proxy, str(battery_dir), 0))
        assert proxy.run_count == 2 and len(retries) == 1 and 'charger' in str(out), out
        print('PASS power (power.sample over the real RootShell): stderr-only answer read once and fails; blank re-read once')
        # mask: a pid whose status cannot be opened (stderr only) -> read once, base 'missing/malformed'
        proxy = Counted(shell)
        proxy.monitor_tids = set()
        out, retries = c.run(lambda: rt.verify_monitor(sb, (proxy, {0, 1}, 999999, {}), {4, 5}))
        assert proxy.run_count == 1 and not retries and failed(out) and 'missing/malformed' in str(out), (proxy.run_count, out)
        print('PASS mask (verify_monitor, sb.root_mask over the real RootShell): stderr-only answer read once and fails')


def setup(*shells_or_errors, close_error=None):
    """rt.prepare over the real sb.prepare_monitor; each item is a mask for one root shell (with su stderr) or a raised error."""
    queue = list(shells_or_errors)
    def make():
        item = queue.pop(0)
        if isinstance(item, BaseException):raise item
        def run(command, timeout=5):
            shell.last_stderr = ['su: warning']
            return (['12'] if command == 'echo $$' else [f'Cpus_allowed_list:\t{item}'] if 'Cpus_allowed_list' in command
                    else c.DISCOVER if 'cpuinfo_max_freq' in command else [])
        shell = NS(p=NS(pid=10), run=run, close=MagicMock(side_effect=close_error), last_stderr=[])
        return shell
    shells = MagicMock(side_effect=make)
    def call():
        with patch.object(cr, 'RootShell', shells), patch.object(pm, 'thread_ids', side_effect=itertools.cycle([{1}, {1, 11}])), \
             patch.object(rt.os, 'sched_getaffinity', return_value=set(range(8))), patch.object(rt.os, 'sched_setaffinity'), \
             patch.object(rt.os, 'cpu_count', return_value=8), patch.object(rt, 'setup_affinity', return_value={}), \
             patch.object(rt, 'verify_monitor'):
            return rt.prepare(sb, {4, 5})
    return call, shells


def check_launch_only():
    """FIX4D R1: only an OSError raised while starting a process is a launch failure (re-read); an OSError after the
    transport answered (a local read, a cleanup kill, a later `am`) stands as the reader's own error."""
    launch = c.launch_error
    # battery: a complete answer (with su stderr), then /proc/meminfo fails -> the OSError stands, 1 su call
    reader = su(lambda: rt.battery_sample(sb), patch.object(cr, 'meminfo_mib', side_effect=OSError('meminfo unreadable')))
    once(reader((0, '-2000000\n1000000\nDischarging\n', 'su: warning\n'), GOOD_BATTERY), raises('meminfo unreadable', OSError))
    # su itself cannot be started -> one re-read
    again(battery(launch(), GOOD_BATTERY), ok)
    again(battery(launch(), launch()), closed)
    # monitor setup: an unsafe mask (with su stderr), then the cleanup kill fails -> the error stands, 1 shell
    once(setup('4-7', '0-3', close_error=PermissionError('kill')), raises('kill', PermissionError))
    once(setup('4-7', '0-3'), raises('includes inference cores'))
    again(setup(launch(), '0-3'), ok)  # su could not be started: one fresh setup
    # camera_start: only the first `am start` launch failure is re-read; a later one (after `am` answered) fails closed
    def camera(*answers, frame='missing'):
        run = MagicMock(side_effect=[NS(returncode=0) if a == 0 else a for a in answers])
        read = dict(status='ok', session='s', frame=1) if frame == 'ok' else dict(status='missing')
        def call():
            with patch.object(cr.subprocess, 'run', run), patch.object(cr, 'read_frame', return_value=read), \
                 patch.object(cr.time, 'monotonic', side_effect=lambda: clock[0]), \
                 patch.object(cr.time, 'sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0]+dt)):
                return rt.camera_start(pm)
        return call, run
    def framed(*answers):  # a frame is published as soon as `am start` answers
        return camera(*answers, frame='ok')
    again(framed(launch(), 0), ok)
    again(framed(launch(), launch()), closed)
    once(camera(0, launch()), raises('am launch failure after an answer'), 2)  # in-loop STOP could not start
    once(framed(OSError('am: not a launch failure')), raises('not a launch failure', OSError))
    # pgrep: a plain OSError (not from starting pgrep) stands
    once(c.pgrep_reader(OSError('pgrep: other'), (1, '')), raises('pgrep: other', OSError))
    print('PASS launch failures only (FIX4D R1): a local OSError after a complete battery answer and a cleanup '
          'PermissionError after an unsafe setup mask stand (1 read); su/RootShell/am that could not be started are re-read '
          'once; camera_start re-reads only a launch failure of its first `am start` (a later one fails closed)')


def r4_probes():
    pidof_ok = (0, 'pidof_rc=1\n', '')
    return [
        ('FIX4C R4-1', "force-stop rc 1, stdout '', stderr 'su: permission denied'; pidof ok; healthy force-stop queued",
         lambda: once(end(DENIED, pidof_ok, (0, '', '')), raises('am force-stop failed (rc 1'), 2)),
        ('FIX4C R4-1', "pidof rc 1, stdout '', stderr 'su: permission denied'; healthy pidof queued",
         lambda: once(end((0, '', ''), DENIED, pidof_ok), raises('absence after force-stop unconfirmed'), 2)),
        ('FIX4C R4-1', 'battery: the same stderr-only failure; healthy answer queued',
         lambda: once(battery(DENIED, GOOD_BATTERY), raises('root battery read failed'))),
        ('FIX4C R4-1', 'package version: the same stderr-only failure; healthy answer queued',
         lambda: once(version(DENIED, (0, 'versionCode=7 versionName=1.0', '')), raises('version unreadable'))),
        ('FIX4D R1-2', 'setup: mask 4-7 (+ su stderr) with measured {4,5}, then the cleanup kill raises PermissionError; healthy setup queued',
         lambda: once(setup('4-7', '0-3', close_error=PermissionError('kill')), raises('kill', PermissionError))),
        ('FIX4D R1-1', 'battery: complete 2 W answer + su stderr, then meminfo raises OSError; healthy 4 W queued',
         lambda: once(su(lambda: rt.battery_sample(sb), patch.object(cr, 'meminfo_mib', side_effect=OSError('meminfo unreadable')))(
             (0, '-2000000\n1000000\nDischarging\n', 'su: warning\n'), GOOD_BATTERY), raises('meminfo unreadable', OSError))),
    ]


def check_probes():
    print('| Finding | Probe (healthy re-read queued) | FIX4D result |\n|---|---|---|')
    for finding, probe, thunk in c.probes() + r4_probes():
        print(f'| {finding} | {probe} | {thunk()} |')
    print('PASS every earlier probe (FIX4, FIX4B, FIX4C R1-R2 from self_check_fix4c, FIX4C R4) on this code; FIX4C R3 '
          'had no findings (sandbox blocker)')


if __name__ == '__main__':
    check_su_readers()
    check_rootshell_readers()
    check_launch_only()
    check_probes()
    print('PASS CAMPAIGN_P1_FIX4D self-check (offline; agents resident; proot; NOT VALID for timing)')
