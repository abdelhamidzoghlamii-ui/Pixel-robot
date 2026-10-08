"""CAMPAIGN_P1_FIX3D focused offline mocks: thermal read retry, raw evidence, per-thread tag. Never run hardware."""
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import phase1 as p
import runtime as rt
from self_check_fix3 import rest_harness, root_text
from self_check_fix3b import GOOD, live_record

REAL_ONCE = rt.dump_once
# `dumpsys thermalservice` as the phone prints it (coresidency test form, shortened).
FULL = ('IsStatusOverride: false\nThermal Status: 0\nCached temperatures:\n'
        '\tTemperature{mValue=40.0, mType=3, mName=VIRTUAL-SKIN, mStatus=1}\n'
        'HAL Ready: true\nCurrent temperatures from HAL:\n'
        '\tTemperature{mValue=29.3, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\n'
        'Current cooling devices from HAL:\n\tCoolingDevice{mValue=0, mType=2, mName=thermal-cpufreq-2}\n')
PARTIAL = 'IsStatusOverride: false\n'  # truncated output: neither status nor skin
PARTIAL_STOP = 'IsStatusOverride: false\nThermal Status: 4\n'  # stop status, skin missing


def fake_cr(outputs, tags=None, timeouts=None):
    """cr with a root() that returns the next (rc, text) or raises it; the real parser."""
    _, _, _, cr, _ = rt.imports()
    queue = list(outputs)
    def root(cmd, tag, timeout=60):
        assert cmd == 'dumpsys thermalservice'
        if tags is not None:tags.append(tag)
        if timeouts is not None:timeouts.append(timeout)
        out = queue.pop(0)
        if isinstance(out, BaseException):raise out
        return out
    return NS(root=root, parse_dump=cr.parse_dump, STATUS_STOP=4), queue


class Clock:
    """Fake monotonic clock; sleep advances it by dt plus `over` (an oversleeping OS)."""
    def __init__(self, over=0.):self.now, self.over = 0., over
    def monotonic(self):return self.now
    def sleep(self, dt):self.now += dt + self.over


def check_reader():
    rt.THERMAL_RETRIES.clear()
    tags, timeouts = [], []
    cr, queue = fake_cr([(0, ''), (0, FULL)], tags, timeouts)
    d = rt.read_dump(cr)
    assert timeouts[0] == 30 and 0 < timeouts[1] <= rt.DUMP_RETRY_S, timeouts
    assert d['attempts'] == 2 and d['skin'] == 29.3 and d['status'] == 0 and not queue, d
    (bad,) = d['incomplete_attempts']
    assert bad['attempt'] == 1 and bad['raw'] == '' and bad['why'] == 'skin None, status None', bad
    assert rt.THERMAL_RETRIES == [d] and rt.THERMAL_RETRIES[0] is not d
    assert tags == [f'campaign_thermal_{threading.get_native_id()}'] * 2, tags
    # Partial and oversized raw outputs are kept, truncated to RAW_KEEP.
    cr, _ = fake_cr([(0, PARTIAL), (0, PARTIAL + 'x' * 5000), (0, FULL)])
    d = rt.read_dump(cr)
    assert d['attempts'] == 3 and [a['raw'] for a in d['incomplete_attempts']] == [PARTIAL, (PARTIAL + 'x' * 5000)[:rt.RAW_KEEP]]
    n = len(rt.THERMAL_RETRIES)
    d = rt.read_dump(fake_cr([(0, FULL)])[0])
    assert d['attempts'] == 1 and 'incomplete_attempts' not in d and len(rt.THERMAL_RETRIES) == n, d
    # Three empty reads fail closed; the raw output of each attempt is in the error.
    sleeps = []
    with patch.object(rt.time, 'sleep', side_effect=sleeps.append):
        cr, queue = fake_cr([(0, '')] * 3 + [(0, FULL)])
        try:rt.read_dump(cr)
        except RuntimeError as e:msg = str(e)
        else:raise AssertionError('three empty reads accepted')
        assert len(queue) == 1 and sleeps == [rt.DUMP_PAUSE_S] * 2, (queue, sleeps)
        assert msg.startswith('thermal read failed: skin None, status None (attempts 3: ') and msg.count("'raw': ''") == 3, msg
        # rc != 0, an error, rc missing: no retry, raised at once.
        for name, out in (('rc 1', (1, FULL)), ('rc 1 empty', (1, '')), ('timeout', subprocess.TimeoutExpired('su', 30)),
                          ('oserror', OSError('su missing'))):
            sleeps.clear()
            cr, queue = fake_cr([out, (0, FULL)])
            try:rt.read_dump(cr)
            except RuntimeError as e:assert 'thermal read failed: rc' in str(e) and '(attempts 1: ' in str(e), (name, e)
            else:raise AssertionError(f'{name} accepted')
            assert len(queue) == 1 and sleeps == [], (name, queue, sleeps)
        cr, _ = fake_cr([(0, FULL.replace('Thermal Status: 0', 'Thermal Status: 4'))])
        try:rt.dump_check(cr)
        except RuntimeError as e:assert 'thermal stop' in str(e)
        else:raise AssertionError('thermal stop accepted')
        # A partial read that parsed a stop status is never retried away (review r1 F1).
        for reader in (rt.read_dump, rt.dump_check):
            sleeps.clear()
            cr, queue = fake_cr([(0, PARTIAL_STOP), (0, FULL)])
            try:reader(cr)
            except RuntimeError as e:assert 'skin None, status 4 (attempts 1: ' in str(e), e
            else:raise AssertionError('partial stop status retried away')
            assert len(queue) == 1 and sleeps == [], (queue, sleeps)
    print('PASS reader: empty then complete -> attempts 2, raw kept (row + THERMAL_RETRIES), per-thread tag; '
          'retry su timeout <= window; raw truncated to RAW_KEEP; three empty -> fails closed with 3 raws; rc!=0/timeout/OSError -> 1 attempt, '
          'no sleep; thermal stop kept; partial read with status 4 -> fails at once, not retried')


def check_timing():
    """A retry starts and is accepted only within DUMP_RETRY_S after the first read ended (its su timeout
    is the time left), also with slow retries, a slow launch the timeout does not cover (review r2) and an
    oversleeping OS; a persistent failure always raises."""
    _, _, _, cr, _ = rt.imports()
    added, typical = [], None
    for first in (.01, .2, 1., 5., 29.):
        for retry in (.01, .2, .9, 1.5, 3., 29.9):
            for over in (0., .3, 3.):
                for launch in (0., 3.):
                    for complete_on in (None, 2, 3):
                        clock, starts, ends = Clock(over), [], []
                        def root(cmd, tag, timeout=60):
                            starts.append(clock.now)
                            if ends:clock.now += launch  # process start before subprocess.run's timeout runs
                            cost = first if not ends else retry
                            if cost > timeout:
                                clock.now += timeout;ends.append(clock.now)
                                raise subprocess.TimeoutExpired('su', timeout)
                            clock.now += cost;ends.append(clock.now)
                            return (0, FULL) if len(ends) == complete_on else (0, '')
                        with patch.object(rt.time, 'monotonic', clock.monotonic), patch.object(rt.time, 'sleep', clock.sleep):
                            try:d = rt.read_dump(NS(root=root, parse_dump=cr.parse_dump, STATUS_STOP=4))
                            except RuntimeError:d = None
                        case = (first, retry, over, launch, complete_on, starts, ends)
                        window = ends[0] + rt.DUMP_RETRY_S + 1e-9
                        # Persistent failure always raises; a success is the planned attempt and ended inside the window.
                        assert d is None if complete_on is None else d is None or d['attempts'] == complete_on == len(ends), case
                        assert d is None or len(ends) == 1 or ends[-1] <= window, case
                        assert len(ends) <= rt.DUMP_ATTEMPTS and all(t < window for t in starts[1:]), case
                        assert clock.now <= window + launch + over, case
                        if not launch:assert all(e <= window for e in ends), case
                        if not (over or launch):added.append(clock.now - ends[0])
                        if (first, retry, over, launch, complete_on) == (.2, .2, 0., 0., None):typical = (clock.now, len(ends))
                        if (first, retry, over, launch, complete_on) == (.2, .2, 0., 3., 2):assert d is None, case  # late complete rejected
    assert max(added) <= rt.DUMP_RETRY_S + 1e-9 and typical[1] == 3 and abs(typical[0] - 1.6) < 1e-9, (max(added), typical)
    # Cadence: monitor_loop keeps the k*period grid. A 2.2 s read (0.2 s + full 2 s window) skips no 4.87 s slot;
    # a read longer than the period (e.g. a slow first read plus retries) skips slots, never bunches.
    for slow, skipped in ((2.2, 0), (5.0, 1)):
        clock, rows, reads = Clock(), [], [0]
        class Stop:
            def wait(self, dt):clock.now += dt;return clock.now > 40
        def read():
            reads[0] += 1;start = clock.now
            clock.now += slow if reads[0] == 3 else .2
            return dict(t_start=start, t=clock.now)
        with patch.object(cr.time, 'monotonic', clock.monotonic):cr.monitor_loop(4.87, read, rows, Stop())
        slots = [round(r['t_start'] / 4.87, 6) for r in rows]
        assert all(k == int(k) for k in slots) and len(rows) == 9 - skipped, (slow, slots)
    print(f'PASS timing (first read 0.01-29 s, retries 0.01-29.9 s, launch 0/3 s outside the su timeout, oversleep '
          f'0/0.3/3 s): retries start and are accepted only <= {rt.DUMP_RETRY_S} s after the first read ended (late '
          f'complete retry -> fails closed); added <= {max(added):.2f} s without launch/oversleep overhead; 3 x 0.2 s '
          f'reads end at {typical[0]:.2f} s; persistent failure always raises; 4.87 s grid kept: a 2.2 s read skips no '
          'slot, a 5.0 s read skips exactly one')


def check_shared_file_race():
    """cr.root's exact su form, fake su/dumpsys, forced interleaving: reader A writes the head of its
    output and waits; reader B starts (its `>FILE` truncates), lets A finish, and writes 0.3 s later.
    With one shared tag A cats a NUL hole plus its tail: rc 0, no skin, no status (the owner signature).
    With per-thread tags both are complete. Shows the mechanism, not the phone's timing."""
    _, _, _, cr, _ = rt.imports()
    real_run = subprocess.run
    results = {}
    for mode, read in (('shared', cr.read_dump), ('per-thread', lambda: REAL_ONCE(cr))):
        with tempfile.TemporaryDirectory() as tmp:
            bin_ = Path(tmp)/'bin';bin_.mkdir()
            cut = FULL.index('\tTemperature{mValue=29.3')
            (Path(tmp)/'head.txt').write_text(FULL[:cut]);(Path(tmp)/'tail.txt').write_text(FULL[cut:])
            dumpsys = (f'#!/bin/sh\ncd {tmp}\nif mkdir first 2>/dev/null; then cat head.txt; touch head_done\n'
                       '  while [ ! -e go ]; do sleep 0.01; done; cat tail.txt\n'
                       'else touch go; sleep 0.3; cat head.txt tail.txt; fi\n')
            for name, body in (('su', '#!/bin/sh\n[ "$1" = -c ] && exec sh -c "$2"\nexit 1\n'), ('dumpsys', dumpsys)):
                (bin_/name).write_text(body);(bin_/name).chmod(stat.S_IRWXU)
            env = dict(os.environ, PATH=f'{bin_}:{os.environ["PATH"]}')
            def run(argv, **k):
                assert argv[:2] == ['su', '-c']
                return real_run([str(bin_/'su'), '-c', argv[2].replace('/data/local/tmp/', tmp+'/')], env=env, **k)
            rows = {}
            def reader(name):
                if name == 'B':
                    while not (Path(tmp)/'head_done').exists():time.sleep(.01)
                rows[name] = read()
            with patch.object(cr.subprocess, 'run', side_effect=run):
                threads = [threading.Thread(target=reader, args=(n,)) for n in 'AB']
                for t in threads:t.start()
                for t in threads:t.join(timeout=30)
            results[mode] = {n: (r['rc'], r['status'], r['skin']) for n, r in rows.items()}
    assert results == {'shared': {'A': (0, None, None), 'B': (0, 0, 29.3)},
                       'per-thread': {'A': (0, 0, 29.3), 'B': (0, 0, 29.3)}}, results
    print('PASS shared-file race (real cr.root su form, fake su/dumpsys, forced interleaving): shared tag '
          '"thermalservice" -> reader A rc 0, skin None, status None; per-thread campaign tags -> both complete')


def check_frame():
    """The live rt.dump_check frame this read belongs to, None for a worker read."""
    f = sys._getframe(2)
    while f and f.f_code.co_name != 'dump_check':
        f = f.f_back
    return f


def scripted(main_outputs, worker_outputs=()):
    """reads= for the harnesses: the real dump_once; the reads of main-thread dump_check number i take
    main_outputs[i] in order (list of (rc, text), then complete); worker reads take worker_outputs in order."""
    calls, live, worker = {'main': 0, 'worker': 0}, [None], list(worker_outputs)
    def reads(_cr, timeout=30):
        f = check_frame()
        if f is None:
            calls['worker'] += 1;plan = worker
        else:
            if f is not live[0]:live[0] = f;calls['main'] += 1  # holding the frame keeps its identity unique
            plan = main_outputs.get(calls['main'])
        cr, _ = fake_cr([plan.pop(0)] if plan else [(0, FULL)])
        return REAL_ONCE(cr, timeout)
    return reads, calls


def check_rest():
    """FIX4 M3 removed rest()'s main-loop dump_check (the owner_session_p1_fix3c2 stop site): the thermal worker
    is the only thermal read in idle phases and pauses, with the same checked reader and retry rule."""
    for camera_on in (False, True):
        text = root_text(app=camera_on)
        reads, calls = scripted({})
        rt.THERMAL_RETRIES.clear()
        error, record = rest_harness(p, camera_on, text, reads=reads)
        assert error is None and calls['main'] == 0 and calls['worker'] >= 1, (error, calls)
        assert all(r['attempts'] == 1 for r in record['dumps']) and not rt.THERMAL_RETRIES
        # The worker's read is empty once, then completes: the phase passes, retry + raw in THERMAL_RETRIES.
        reads, calls = scripted({}, [(0, ''), (0, FULL)])
        with patch.object(rt.time, 'sleep'):error, record = rest_harness(p, camera_on, text, reads=reads)
        assert error is None and record.get('error') is None and calls['main'] == 0, (error, record.get('error'))
        (retry,) = rt.THERMAL_RETRIES
        assert retry['attempts'] == 2 and retry['incomplete_attempts'][0]['raw'] == '' and record['dumps'][0]['attempts'] == 2, retry
        rt.THERMAL_RETRIES.clear()
        # Three empty reads: fails closed, raw in the phase error.
        reads, calls = scripted({}, [(0, '')] * 3)
        with patch.object(rt.time, 'sleep'):error, record = rest_harness(p, camera_on, text, reads=reads)
        assert error and 'thermal read failed: skin None, status None (attempts 3' in str(record['monitor_errors']), (error, record.get('monitor_errors'))
        # rc != 0: fails at once.
        reads, calls = scripted({}, [(1, FULL), (0, FULL)])
        error, record = rest_harness(p, camera_on, text, reads=reads)
        assert error and 'thermal read failed: rc 1' in str(record['monitor_errors']) and '(attempts 1' in str(record['monitor_errors'])
        # Partial stop status: fails at once (review r1 F1).
        reads, calls = scripted({}, [(0, PARTIAL_STOP), (0, FULL)])
        error, record = rest_harness(p, camera_on, text, reads=reads)
        assert error and 'skin None, status 4 (attempts 1' in str(record['monitor_errors']), (error, record.get('monitor_errors'))
    print('PASS rest() camera OFF and ON (M3: no main-loop dump_check, 0 main reads), real dump_once in the thermal worker: '
          'empty once -> phase passes, attempts 2 + raw in THERMAL_RETRIES; three empty -> fails with 3 raws; rc 1 -> fails, '
          '1 attempt; partial status-4 read -> fails at once')


def check_live_block():
    # live_block always runs with the camera ON; skin_start is main check 1, skin_end (end of block) check 2.
    for dry in (False, True):
        reads, _ = scripted({})
        record = live_record(dry, [GOOD], reads=reads)
        assert record.get('error') is None and record['skin_start']['attempts'] == record['skin_end']['attempts'] == 1, record.get('error')
        reads, _ = scripted({2: [(0, PARTIAL), (0, FULL)]})
        with patch.object(rt.time, 'sleep'):record = live_record(dry, [GOOD], reads=reads)
        assert record.get('error') is None and record['validity'] == (p.REHEARSAL if dry else 'VALID'), record.get('error')
        assert record['skin_end']['attempts'] == 2 and record['skin_end']['incomplete_attempts'][0]['raw'] == PARTIAL
        for plan, needle in (([(0, '')] * 3, 'skin None, status None (attempts 3'), ([(1, FULL)], 'rc 1, error None (attempts 1')):
            reads, _ = scripted({2: plan})
            with patch.object(rt.time, 'sleep'):record = live_record(dry, [GOOD], reads=reads)
            assert record['failure_kind'] == 'shared' and needle in record['error'] and 'skin_end' not in record, (needle, record.get('error'))
            assert record['validity'] == (p.REHEARSAL if dry else 'NOT VALID — INCOMPLETE')
    print('PASS live_block session and rehearsal (camera ON), real dump_check: skin_end partial once -> VALID, '
          'attempts 2 + raw in the block row; three empty or rc 1 at skin_end -> NOT VALID / shared')


if __name__ == '__main__':
    check_reader();check_timing();check_shared_file_race();check_rest();check_live_block()
    print('PASS CAMPAIGN_P1_FIX3D self-check (offline; agents resident; proot; NOT VALID for timing)')
