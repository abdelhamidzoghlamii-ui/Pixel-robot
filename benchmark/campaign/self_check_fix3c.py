"""CAMPAIGN_P1_FIX3C focused offline checks: the process guard excludes the runner's own process tree by
ppid ancestry, never by command text. Never runs hardware, root, models or a timed phase."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types
from types import SimpleNamespace as NS
from unittest.mock import patch

import diagnostics as d
import runtime as rt

HERE = Path(__file__).resolve().parent
BASE = '95bb0d858a1a7a9fa32a8575ea1bdba63fb71bb5'
OWNER = json.loads((HERE/'runs/owner_rehearsal_p1_fix3_warmup_L0.json').read_text())['monitor_errors'][0]
LINE = OWNER.split('agents/robot/other runners resident: ', 1)[1]  # the exact pgrep -fa line, pid 1312


def base_runtime():
    src = subprocess.run(['git', 'show', f'{BASE}:benchmark/campaign/runtime.py'], cwd=rt.ROBOT,
                         capture_output=True, text=True, check=True).stdout
    module = types.ModuleType('runtime_base'); module.__file__ = str(HERE/'runtime.py')
    exec(compile(src, 'runtime@base', 'exec'), module.__dict__)
    return module


def pattern():
    with patch.object(rt.subprocess, 'run', return_value=NS(returncode=1, stdout='', stderr='')) as run:
        rt.clear_processes()
    return run.call_args.args[0][-1]


def pgrep(stdout):
    return patch.object(rt.subprocess, 'run', return_value=NS(returncode=0, stdout=stdout, stderr=''))


def refused(module, stdout, table, server_pid=None):
    """Error text, or None; table maps pid -> ppid or an exception raised for that pid's stat."""
    def parent(pid):
        v = table[pid] if pid in table else FileNotFoundError(f'/proc/{pid}/stat')
        if isinstance(v, BaseException):
            raise v
        return v
    with patch.object(module.subprocess, 'run', return_value=NS(returncode=0, stdout=stdout, stderr='')), \
         patch.object(rt, 'parent_pid', side_effect=parent):
        try:module.clear_processes(server_pid)
        except RuntimeError as e:return 'RuntimeError: '+str(e)


def check_owner_failure():
    me = os.getpid()
    assert LINE.startswith('1312 /debug_ramdisk/su -c { taskset -p cf $$') and 'read_node()' in LINE
    hits = sorted({m.group(0) for m in re.finditer(pattern(), LINE)})
    assert hits == ['node'], hits  # read_node x3 and 'power-hint nodes'
    # Before: the base guard refuses the runner's own su diagnostics client with the owner's exact message.
    assert refused(base_runtime(), LINE, {}) == OWNER
    # After: same text, pid 1312 is our child (su wrapper execs /debug_ramdisk/su in place) -> accepted.
    assert refused(rt, LINE, {1312: me}) is None
    assert refused(rt, LINE, {1312: 700, 700: me}) is None  # deeper own descendant (e.g. a pump's child)
    # Identical text, not our descendant -> still refused with the owner's message.
    assert refused(rt, LINE, {1312: 1}) == OWNER
    assert refused(rt, LINE, {1312: 500, 500: 1}) == OWNER
    assert refused(rt, LINE, {1312: os.getppid(), os.getppid(): 1}) == OWNER  # our parent's sibling side
    # Foreign agents/robot scripts/other runners stay refused; the server pid stays allowed.
    for text in ('2001 node /usr/bin/claude', '2001 codex', '2001 python run_mission.py', '2001 python benchmark/campaign/phase1.py'):
        assert refused(rt, text, {2001: 1}) == 'RuntimeError: agents/robot/other runners resident: '+text, text
    assert refused(rt, '2001 llama-server -m x', {2001: 1}, server_pid=2001) is None
    assert refused(rt, f'1312 x\n2001 codex', {1312: me, 2001: 1}) == 'RuntimeError: agents/robot/other runners resident: 2001 codex'
    print('PASS owner monitor_errors reproduced on base (own su diagnostics client, pattern node <- read_node); '
          'fixed: own descendants accepted, identical-text foreign refused, foreign agents refused')


def check_fail_closed():
    me = os.getpid()
    for name, table in (('permission', {1312: PermissionError('denied')}),
                        ('garbage stat', {1312: ValueError('bad')}),
                        ('ancestor gone', {1312: 500}),          # 500 unreadable mid-chain
                        ('ancestor hidden', {1312: 500, 500: PermissionError('hidepid')}),
                        ('cycle', {1312: 500, 500: 1312})):
        e = refused(rt, LINE, table)  # refused as resident, with the ancestry error attached
        assert e and e.startswith(OWNER+' [process ancestry unreadable: pid 1312'), (name, e)
    with pgrep(LINE), patch.object(rt, 'parent_pid', side_effect=FileNotFoundError):
        rt.clear_processes()  # the matched pid itself exited after pgrep: no longer resident
    with patch.object(rt.subprocess, 'run', return_value=NS(returncode=2, stdout='', stderr='bad')):
        try:rt.clear_processes()
        except RuntimeError as e:assert 'process enumeration failed' in str(e)
        else:raise AssertionError('pgrep failure accepted')
    assert refused(rt, LINE, {1312: me}) is None
    print('PASS fail closed: unreadable/garbage/vanished-ancestor/hidden/cyclic ancestry refuse; pgrep failure raises')


def check_real_proc():
    """Real /proc ancestry (no root): an own child and a reparented orphan carry the owner's command text."""
    me = os.getpid()
    assert rt.parent_pid(me) == os.getppid() and rt.own_process(me) is True
    try:assert rt.own_process(os.getppid()) is False
    except RuntimeError as e:assert 'process ancestry unreadable' in str(e)  # hidden (other-uid) ancestor: fail closed
    py = sys.executable
    child = subprocess.Popen([py, '-c', 'import time; time.sleep(30)', LINE], start_new_session=True)
    orphan_pid = None
    try:
        out = subprocess.run([py, '-c', 'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)",sys.argv[1]],'
                              'start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(p.pid)', LINE],
                             capture_output=True, text=True, check=True)
        orphan_pid = int(out.stdout)  # its parent has exited: reparented away from us
        for pid in (child.pid, orphan_pid):
            assert LINE.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
        assert rt.parent_pid(child.pid) == me and rt.own_process(child.pid) is True
        assert rt.parent_pid(orphan_pid) != me
        try:assert rt.own_process(orphan_pid) is False
        except RuntimeError as e:assert 'process ancestry unreadable' in str(e)
        with pgrep(f'{child.pid} {LINE}'):rt.clear_processes()
        with pgrep(f'{child.pid} {LINE}\n{orphan_pid} {LINE}'):
            try:rt.clear_processes()
            except RuntimeError as e:assert str(e).startswith(f'agents/robot/other runners resident: {orphan_pid} {LINE}'), e
            else:raise AssertionError('foreign identical-text orphan accepted')
    finally:
        child.kill(); child.wait()
        if orphan_pid:
            try:os.kill(orphan_pid, 9)
            except ProcessLookupError:pass
    gone = child.pid
    assert rt.own_process(gone) is None
    print(f'PASS real /proc: own child {child.pid} accepted, reparented orphan {orphan_pid} with identical text refused, exited pid skipped')


def check_audit():
    """Every command line the runner spawns itself, tested against the guard pattern (pgrep -f sees argv joined by spaces)."""
    v, sb, pm, cr, _ = rt.imports()
    seen = []
    def record(argv, *a, **k):
        seen.append((source[0], ' '.join(argv)))
        cmd = ' '.join(argv)
        out = '2147483647' if 'screen_off_timeout' in cmd else 'pidof_rc=1' if 'pidof_rc' in cmd else ''
        return NS(returncode=0, stdout=out, stderr='')
    class Popen:
        def __init__(self, argv, *a, **k):
            seen.append((source[0], ' '.join(argv))); self.pid = 0; self.stdin = self.stdout = None
    source = ['']
    calls = {
        'diag.battery': lambda: d.battery(cr, 0),
        'diag.snapshot': lambda: d.snapshot(cr, False),
        'rt.camera_version': lambda: rt.camera_version(cr),
        'rt.memory_sample (root_sample + pidof)': lambda: rt.memory_sample(cr, 4242, False),
        'cr.read_dump': cr.read_dump,
        'cr.lmk_lines': lambda: cr.lmk_lines(0.),
        'cr.camera_end_check': lambda: cr.camera_end_check(rootf=cr.root, quiet_s=0, limit_s=0),
        'sb.battery_sample': sb.battery_sample,
        'sb.Screen start/restore': lambda: (lambda s: (s.start(), s.restore()))(sb.Screen()),
        'pm.camera_start (am)': lambda: pm.camera_start(1),
        'cr.camera_stop (am)': cr.camera_stop,
        'cr.RootShell (pumps, persistent monitor shell)': cr.RootShell,
    }
    ok = {'status': 'ok', 'session': 1, 'frame': 1}
    with patch.object(cr.subprocess, 'run', side_effect=record), patch.object(cr.subprocess, 'Popen', Popen), \
         patch.object(sys.modules[sb.__name__].subprocess, 'run', side_effect=record), \
         patch.object(cr, 'read_frame', return_value=ok), patch.object(pm.cr, 'read_frame', return_value=ok), \
         patch.object(cr.threading, 'Thread'):
        with rt.root_mask_scope(cr, sb, {0, 1, 2, 3}, {4, 5}):
            for name, call in calls.items():
                source[0] = name
                try:call()
                except Exception:pass  # fake outputs; only the spawned command lines matter
    seen.append(('cr.server_cmd (llama-server Popen argv)', ' '.join(map(str, cr.server_cmd(())))))
    pat, report = pattern(), []
    for name in dict.fromkeys(n for n, _ in seen):
        cmds = [c for n, c in seen if n == name]
        hits = sorted({m.group(0) for c in cmds for m in re.finditer(pat, c)})
        report.append(dict(source=name, spawned=len(cmds), matches=hits, example=cmds[0][:160]))
    for r in report:print(json.dumps(r))
    matching = {r['source']: r['matches'] for r in report if r['matches']}
    assert matching == {'diag.snapshot': ['node'], 'cr.server_cmd (llama-server Popen argv)': ['llama-server']}, matching
    assert all(r['spawned'] for r in report), report
    print('PASS audit: of', len(report), 'spawn sources only diagnostics (node) and the owned llama-server match; '
          'both are direct children of the runner (subprocess.run/Popen), excluded by ancestry/server pid')


if __name__ == '__main__':
    check_owner_failure()
    check_fail_closed()
    check_real_proc()
    check_audit()
    print('PASS CAMPAIGN_P1_FIX3C self-check (offline; agents resident; proot; NOT VALID for timing)')
