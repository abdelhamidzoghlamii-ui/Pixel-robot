"""Native tests of fixed quality rules, cluster masks, session preflight/order/failure."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
import variants as v
import session as s
from quality import compare_triplets

# Keep the existing direct-script tests in the same native invocation.
import runpy
runpy.run_path(str(v.dc.HERE / 'self_check.py'), run_name='__main__')

assert v.CLUSTERS == {'MID': ({4, 5}, 2), 'BIG': ({6, 7}, 2), 'LITTLE': ({0, 1, 2, 3}, 4)}
for cluster, (cpus, threads) in v.CLUSTERS.items():
    safe = set(range(8)) - cpus
    v.sb.require_monitor_mask(safe, cpus)
    v.sb.require_monitor_mask({min(safe)}, cpus)
    for mask in (set(), cpus, safe | {min(cpus)}):
        try: v.sb.require_monitor_mask(mask, cpus)
        except RuntimeError: pass
        else: raise AssertionError(cluster)
    with patch.object(v.sb.os, 'sched_getaffinity', return_value=cpus):
        v.sb.check_pinning([100, 101], cpus)

dets = [{'box_xyxy': [i, 0, i+1, 1]} for i in range(12)]
rows = [{'subject_idx': 0, 'object_idx': i+1, 'predicate': 'near', 'score': .6, 'pass': True} for i in range(10)]
assert compare_triplets(rows, rows[:9], dets)['passed']
assert not compare_triplets(rows, rows[:8], dets)['passed']
assert not compare_triplets(rows[:2], rows[:2], dets)['passed']  # owner rule stays literal
changed = [r | {'score': .66, 'pass': False} for r in rows]
assert not compare_triplets(rows, changed, dets)['passed']
assert compare_triplets(rows, changed, dets)['pass_fail_flips'] == 10
other_boxes = [r | {'object_idx': 11} for r in rows]
assert not compare_triplets(rows, other_boxes, dets)['passed']
quality = {'variants': {n: {'passed': n in ('fp32', 'xnnpack'), 'informal_mid_median_ms': ms}
                       for n, ms in zip(v.VARIANTS, [100, 120, 90, 80, 70])}}
assert v.planned_blocks(quality) == [('fp32', 'MID'), ('xnnpack', 'MID'), ('fp32', 'BIG'),
                                    ('fp32', 'LITTLE'), ('xnnpack', 'BIG'), ('fp32', 'MID')]
for message in ('cadence missed/overrun', 'call cadence overran block'):
    assert s.can_continue({'error': 'RuntimeError: ' + message})
    assert not s.can_continue({'error': 'RuntimeError: ' + message, 'cleanup_error': 'alive'})
for message in ('root timeout', 'cores lost', 'charger', 'monitor/limit', 'inference error'):
    assert not s.can_continue({'error': 'RuntimeError: ' + message})

# Root refusal is reached before the standalone runner's idle or skin gate.
with tempfile.TemporaryDirectory() as tmp:
    args = SimpleNamespace(model=v.NAME, dry_run=False, output=Path(tmp) / 'refused.json')
    with contextlib.redirect_stdout(io.StringIO()), patch.object(v.sb, 'require_parity'), \
         patch.object(v.sb.cr, 'require_native'), patch.object(v.sb, 'processes_clear'), \
         patch.object(v.sb.cr, 'check_cores'), patch.object(v.sb, 'battery_sample'), \
         patch.object(v.sb, 'prepare_monitor', side_effect=RuntimeError('unsafe root mask')), \
         patch.object(v.sb.time, 'sleep') as sleep:
        try: v.sb.run_block(args)
        except RuntimeError as error: assert 'unsafe root mask' in str(error)
        else: raise AssertionError('pre-idle root failure ignored')
        sleep.assert_not_called()
    assert json.loads(args.output.read_text())['label'] == 'NOT VALID — INCOMPLETE'

# Session orchestration with fake root/engine/block resources: no hardware/time.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / 'quality.json').write_text(json.dumps(quality))
    events = []
    shell = SimpleNamespace(close=lambda: events.append('root_close'), p=SimpleNamespace(pid=123),
                            run=lambda command: ['Cpus_allowed_list: 0'])
    monitor = (shell, {0, 1}, 123, {'policies': {}})
    def fake_block(args, **kw):
        events.append(('block', args.variant, args.cluster, kw['session']['idle']['skin']))
        return {'label': 'TIMING COMPLETE', 'variant': args.variant, 'cluster': args.cluster}
    def fake_build(variant, cluster):
        events.append(('build', variant, cluster))
        return object(), SimpleNamespace(close=lambda: None), [123]
    def sleep(seconds): events.append(('sleep', seconds))
    with contextlib.redirect_stdout(io.StringIO()), patch.object(s, 'HERE', root), \
         patch.object(s, 'require_quality'), patch.object(s, 'build', side_effect=fake_build), \
         patch.object(s, 'repin_monitor', side_effect=lambda m,c: m), \
         patch.object(s.sb, 'prepare_monitor', return_value=monitor), patch.object(s.sb, 'Screen') as screen, \
         patch.object(s.sb.cr, 'require_native'), patch.object(s.sb, 'processes_clear'), \
         patch.object(s.sb.cr, 'check_cores'), patch.object(s.sb, 'battery_sample'), \
         patch.object(s.sb.cr, 'read_dump', return_value={'skin': 30, 'status': 0}), \
         patch.object(s.sb, 'check_pinning'), patch.object(s.os, 'sched_getaffinity', return_value=set(range(8))), \
         patch.object(s.time, 'sleep', side_effect=sleep), patch.object(s.sb, 'run_block', side_effect=fake_block), \
         patch.object(sys, 'argv', ['session.py', '--session', '--output', str(root/'session.json')]):
        s.main()
    assert events.count(('sleep', 5)) == 60
    assert max(i for i,e in enumerate(events) if isinstance(e, tuple) and e[0] == 'build') < events.index(('sleep', 5))
    recorded = json.loads((root/'session.json').read_text())
    assert len(recorded['blocks']) == 6 and not recorded['unrun_blocks']
    assert recorded['plan'][0]['variant'] == recorded['plan'][-1]['variant'] == 'fp32'
    screen.return_value.start.assert_called_once()
    screen.return_value.restore.assert_called_once()

print('PASS: all cluster/thread rules; literal quality thresholds and box identity; plan/order; continuation fail-closed; pre-idle root refusal; one session idle and cleanup (mocks, NOT VALID TIMING)')

class Root:
    p = SimpleNamespace(pid=123)
    def run(self, command):
        if '/proc/' in command:
            return ['Cpus_allowed_list:\t0']
        return ['234'] if command == 'echo $$' else []
    def close(self): pass
with patch.object(v.sb.cr, 'RootShell', return_value=Root()), \
     patch.object(v.sb.pm, 'thread_ids', side_effect=[{1}, {1, 2}]), \
     patch.object(v.sb.os, 'sched_getaffinity', return_value=set(range(8))), \
     patch.object(v.sb.os, 'sched_setaffinity'), \
     patch.object(v.sb.cr, 'discover', return_value={'policies': dict.fromkeys(v.sb.pm.POLICIES, 1)}):
    monitor = v.sb.prepare_monitor()
    assert monitor[2] == 234 and monitor[0].monitor_tids == {2}
print('PASS: actual prepare_monitor accepts narrower safe root/su masks (mock), before idle')

# Status parsing and validation use the real root reader; foreign affinity syscalls fail.
for text, expected in [('0-3,6', {0,1,2,3,6}), ('0', {0})]:
    shell = SimpleNamespace(run=lambda command: ['Name:\tsh', 'Cpus_allowed_list:\t' + text])
    with patch.object(v.sb.os, 'cpu_count', return_value=8):
        assert v.sb.root_mask(shell, 123) == expected
for text in ('', '0-', '3-0', '0,,6', '0-3 junk', '-1', '8', '0x1'):
    shell = SimpleNamespace(run=lambda command: ['Cpus_allowed_list:\t' + text])
    with patch.object(v.sb.os, 'cpu_count', return_value=8):
        try: v.sb.root_mask(shell, 123)
        except RuntimeError: pass
        else: raise AssertionError(text)
for lines in ([], ['Cpus_allowed_list: 0', 'Cpus_allowed_list: 6']):
    try: v.sb.root_mask(SimpleNamespace(run=lambda command: lines), 123)
    except RuntimeError: pass
    else: raise AssertionError(lines)
for mask in ('4', '0,5'):
    shell = SimpleNamespace(run=lambda command: ['Cpus_allowed_list: ' + mask])
    try: v.sb.require_monitor_mask(v.sb.root_mask(shell, 123))
    except RuntimeError: pass
    else: raise AssertionError(mask)
print('PASS: root status valid/ranges/narrow; empty/missing/duplicate/malformed/overlapping refused')

# Both session and --variant preflight check every planned cluster, without idle/blocks.
for variant in (None, 'fp32'):
    for refusal in (None, 'charger', 'agents', 'root', 'cleanup'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'quality.json').write_text(json.dumps(quality))
            checked = []
            shell = SimpleNamespace(close=lambda: None, p=SimpleNamespace(pid=123),
                                    run=lambda command: ['Cpus_allowed_list: 0'])
            monitor = (shell, {0}, 123, {'policies': {}})
            output = root / 'preflight.json'
            argv = ['session.py', '--preflight', '--output', str(output)]
            if variant: argv += ['--variant', variant, '--cluster', 'BIG']
            managers = (contextlib.redirect_stdout(io.StringIO()), patch.object(s, 'HERE', root),
                 patch.object(s, 'require_quality'), patch.object(s, 'build', side_effect=fake_build),
                 patch.object(s, 'repin_monitor', side_effect=lambda m,c: checked.append(c) or m),
                 patch.object(s.sb, 'prepare_monitor', side_effect=RuntimeError('root') if refusal == 'root' else None, return_value=monitor),
                 patch.object(s.sb, 'Screen'), patch.object(s.sb.cr, 'require_native'),
                 patch.object(s.sb, 'processes_clear', side_effect=RuntimeError('agents') if refusal == 'agents' else None),
                 patch.object(s.sb, 'battery_sample', side_effect=RuntimeError('charger') if refusal == 'charger' else None),
                 patch.object(s.sb.cr, 'check_cores'), patch.object(s.sb.cr, 'read_dump', return_value={'skin':30,'status':0}),
                 patch.object(s.sb, 'check_pinning'), patch.object(s.os, 'sched_getaffinity', return_value=set(range(8))),
                 patch.object(s.time, 'sleep'), patch.object(s.sb, 'run_block'),
                 patch.object(sys, 'argv', argv))
            with contextlib.ExitStack() as stack:
                entered = [stack.enter_context(m) for m in managers]
                screen, no_sleep, no_block = entered[6], entered[-3], entered[-2]

                if refusal == 'cleanup': screen.return_value.restore.side_effect = RuntimeError('cleanup')
                try: s.main()
                except RuntimeError: assert refusal
                else: assert refusal is None
                no_sleep.assert_not_called()
                no_block.assert_not_called()
                screen.return_value.restore.assert_called_once()
            record = json.loads(output.read_text())
            assert not record['blocks']
            if refusal is None:
                assert record['label'] == 'PREFLIGHT ONLY — NO TIMING'
                assert all(v.CLUSTERS[c][0] in checked for _,c in
                           ([('fp32','BIG')] if variant else v.planned_blocks(quality)))
print('PASS: session and single-variant preflight all clusters, charger/agents/root refusal, cleanup failure; no idle/gates/blocks (mocks)')

with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
    args = SimpleNamespace(model=v.NAME, dry_run=False, preflight=True, output=Path(tmp)/'block.json')
    for manager in (contextlib.redirect_stdout(io.StringIO()), patch.object(v.sb, 'require_parity'),
                    patch.object(v.sb.cr, 'require_native'), patch.object(v.sb, 'processes_clear'),
                    patch.object(v.sb.cr, 'check_cores'), patch.object(v.sb, 'battery_sample'),
                    patch.object(v.sb, 'prepare_monitor', return_value=(Root(), {0}, 234, {'policies':{}})),
                    patch.object(v.sb, 'build', return_value=(object(), SimpleNamespace(close=lambda:None), [1])),
                    patch.object(v.sb, 'check_pinning'), patch.object(v.sb.cr, 'read_dump', return_value={'skin':30,'status':0})):
        stack.enter_context(manager)
    screen = stack.enter_context(patch.object(v.sb, 'Screen'))
    sleep = stack.enter_context(patch.object(v.sb.time, 'sleep'))
    result = v.sb.run_block(args)
    assert result['label'] == result['validity'] == 'PREFLIGHT ONLY — NO TIMING'
    assert not result['calls'] and not result['power'] and 'idle' not in result
    sleep.assert_not_called()
    screen.return_value.start.assert_called_once()
    screen.return_value.restore.assert_called_once()
print('PASS: standalone block preflight build/pinning and screen cleanup, no idle/gates/calls (mocks)')

for cluster, (cpus, _) in v.CLUSTERS.items():
    safe = min(set(range(8)) - cpus)
    shell = SimpleNamespace(p=SimpleNamespace(pid=123), monitor_tids={1},
                            run=lambda command: ['Cpus_allowed_list: ' + str(safe)])
    def own_affinity(pid):
        assert pid in (0, 1), 'foreign sched_getaffinity'
        return set(range(8)) if pid == 0 else {safe}
    with patch.object(s.os, 'sched_getaffinity', side_effect=own_affinity), \
         patch.object(s.os, 'sched_setaffinity'):
        s.repin_monitor((shell, {safe}, 234, {}), cpus)
for value in ('', '4', 'bogus'):
    class BadRoot(Root):
        def run(self, command):
            return ['Cpus_allowed_list: ' + value] if '/proc/' in command else super().run(command)
    with patch.object(v.sb.cr, 'RootShell', return_value=BadRoot()), \
         patch.object(v.sb.pm, 'thread_ids', side_effect=[{1}, {1,2}]), \
         patch.object(v.sb.os, 'sched_getaffinity', return_value=set(range(8))), \
         patch.object(v.sb.os, 'sched_setaffinity'):
        try: v.sb.prepare_monitor()
        except RuntimeError: pass
        else: raise AssertionError(value)
try: v.sb.root_mask(SimpleNamespace(run=lambda command: (_ for _ in ()).throw(RuntimeError('read failure'))),123)
except RuntimeError: pass
else: raise AssertionError('read error accepted')
print('PASS: real repin across all clusters avoids foreign affinity syscalls; prepare_monitor and read failures fail closed (mocks)')

# Execute the generated builtin shell command against mock status files, no su/root.
import subprocess
import shlex
with tempfile.TemporaryDirectory() as tmp:
    status = Path(tmp)/'status'
    def mock_status_shell(command):
        assert 'cat ' not in command
        command = command.replace('</proc/123/status', '<' + shlex.quote(str(status)))
        done = subprocess.run(['bash', '-c', command], capture_output=True, text=True)
        return done.stdout.splitlines()
    for text, expected in [('Name:\tsh\nCpus_allowed_list:\t0-3,6\n', {0,1,2,3,6}),
                           ('Cpus_allowed_list:\t0\n', {0})]:
        status.write_text(text)
        assert v.sb.root_mask(SimpleNamespace(run=mock_status_shell), 123) == expected
    status.unlink()
    try: v.sb.root_mask(SimpleNamespace(run=mock_status_shell), 123)
    except RuntimeError: pass
    else: raise AssertionError('missing status read accepted')
print('PASS: generated builtin shell reader handles mock status files and missing file; no external cat, no root')
