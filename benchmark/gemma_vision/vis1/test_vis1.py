"""Offline VIS-1 contract checks: no server, root, USB, network or timings."""
import contextlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('vis1',Path(__file__).with_name('vis1_run.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class Checks(unittest.TestCase):
    def test_order_and_selection(self):
        self.assertEqual(v.blocks(None),list(v.ORDER))
        self.assertEqual(v.blocks('V4,T1,V2'),['T1','V2','V4'])
        for value in ('T3','T1,T1',''):
            if value:
                with self.assertRaises(ValueError):v.blocks(value)

    def test_flags(self):
        for b in v.ORDER:
            c = v.command(b)
            self.assertEqual(c[:4],['taskset','-c','4-7','timeout'])
            self.assertIn('--kill-after=3',c)
            for flag,value in [('--ctx-size','4096'),('--threads','4'),('--threads-batch','4'),('--port','8091')]:
                self.assertEqual(c[c.index(flag)+1],value)
            if b.startswith('V'):
                self.assertNotIn('--model-draft',c)
                self.assertEqual(c[c.index('--reasoning')+1],'off')
                budget = '70' if b in ('V1','V2') else '140'
                self.assertEqual(c[c.index('--image-max-tokens')+1],budget)
                self.assertEqual(c[c.index('--image-min-tokens')+1],budget)
                self.assertIn('BF16' if b in ('V1','V4') else 'Q8_0',c[c.index('--mmproj')+1])
            else:self.assertIn('draft-mtp',c)
        self.assertNotIn('--model-draft',v.command('T2',False))
        self.assertIn('--model-draft',v.command('T1',False))

    def test_payloads(self):
        p = v.text_payload('Hi')
        self.assertEqual(p['prompt'],'<|turn>user\nHi<turn|>\n<|turn>model\n')
        self.assertEqual(p['temperature'],.7)
        self.assertEqual(p['n_predict'],512)
        self.assertNotIn('think',p['prompt'])
        self.assertEqual(p['stop'],['<turn|>'])
        p = v.vision_payload(b'fake','- chair [0], center, far')
        self.assertEqual(p['temperature'],0)
        self.assertEqual(p['max_tokens'],96)
        self.assertFalse(p['cache_prompt'])
        content = p['messages'][0]['content']
        self.assertTrue(content[0]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertIn('Objects detected by YOLO:\n- chair',content[1]['text'])

    def test_gate(self):
        self.assertTrue(v.gate_ok(31.5,30))
        for x in (31.501,None,float('nan'),float('inf')):self.assertFalse(v.gate_ok(x,30))
        clock = [0]
        def sleep(t):clock[0] += t
        with patch.object(v,'telemetry',side_effect=[dict(skin=33,status=0),dict(skin=31,status=0)]):
            g = v.gate(None,30,now=lambda:clock[0],sleep=sleep)
        self.assertEqual(g['waited_s'],5)
        self.assertFalse(g['missed'])
        with patch.object(v,'telemetry',return_value=dict(skin=35,status=0)):
            with self.assertRaisesRegex(RuntimeError,'gate missed'):
                v.gate(None,30,now=lambda:clock[0],sleep=sleep)
        self.assertEqual(clock[0],905)  # second gate started at 5; its timeout stays 900 s

    def test_preflight_returns_before_run_setup(self):
        with patch.object(v,'preflight') as check,patch.object(v,'monitor') as monitor, \
             patch.object(v,'inputs') as inputs,patch.object(v.subprocess,'run') as run:
            v.main(['--preflight'])
        check.assert_called_once_with()
        monitor.assert_not_called()
        inputs.assert_not_called()
        run.assert_not_called()

    def test_telemetry_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            sample = dict(t=100,capacity=50,battery_status='Discharging',agent=False,skin=30,status=0,bat_c=30)
            with patch.object(v.time,'monotonic',return_value=100):
                for value in (50,80):
                    sample['capacity']=value;v.write(p/'latest.json',sample);v.telemetry(p)
                for updates in (dict(capacity=49),dict(capacity=81),dict(agent=True),dict(t=80),dict(status=4)):
                    v.write(p/'latest.json',dict(sample,**updates))
                    with self.assertRaises(RuntimeError):v.telemetry(p)

    def test_dry_run_no_operations(self):
        with patch.object(v,'spawn',side_effect=AssertionError('server')),patch.object(v,'root',side_effect=AssertionError('root')),\
             patch.object(v,'post',side_effect=AssertionError('network')),patch.object(v,'telemetry',side_effect=AssertionError('telemetry')),\
             contextlib.redirect_stdout(io.StringIO()) as out:
            v.main(['--dry-run','--only','V2,T1'])
        self.assertIn('NOT VALID TIMING',out.getvalue())
        self.assertIn('"block_order": [\n    "T1",\n    "V2"',out.getvalue())

    def test_lifecycle_only_own_group(self):
        proc = Mock(pid=12345,returncode=-15)
        proc.poll.return_value=None
        proc.wait.side_effect=[subprocess.TimeoutExpired('fake',3),-9]
        # Server pid unknown: fall back to TERM on timeout; escalation still covers the whole own group.
        with patch.object(v.os,'kill') as term,patch.object(v.os,'killpg') as kill:
            v.own_stop(proc)
        term.assert_called_once_with(12345,signal.SIGTERM)
        kill.assert_called_once_with(12345,signal.SIGKILL)
        with patch.object(v.subprocess,'Popen',return_value=proc) as popen:
            v.spawn(['timeout','900','fake'],io.StringIO())
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertIs(popen.call_args.kwargs['stdin'],subprocess.DEVNULL)
        v.OWNED.clear()

    def test_own_stop_terms_owned_server_exactly_once(self):
        # VIS1-FIX3: GNU timeout re-sends a received TERM to its whole group, so TERM goes to the
        # server itself, once, and never as a group-wide TERM.
        rows=[(12345,1,12345),(12346,12345,12345)]
        proc=Mock(pid=12345,returncode=0);proc.poll.return_value=None;proc.wait.return_value=0
        with patch.object(v,'proc_rows',return_value=rows),patch.object(v.os,'kill') as term, \
             patch.object(v.os,'killpg') as kill:
            self.assertEqual(v.own_stop(proc,12346),0)
        term.assert_called_once_with(12346,signal.SIGTERM)
        kill.assert_not_called()
        # Server hangs: unchanged group-wide SIGKILL escalation, still no group TERM.
        proc=Mock(pid=12345,returncode=-9);proc.poll.return_value=None
        proc.wait.side_effect=[subprocess.TimeoutExpired('fake',3),-9]
        with patch.object(v,'proc_rows',return_value=rows),patch.object(v.os,'kill') as term, \
             patch.object(v.os,'killpg') as kill:
            v.own_stop(proc,12346)
        term.assert_called_once_with(12346,signal.SIGTERM)
        kill.assert_called_once_with(12345,signal.SIGKILL)
        # A remembered pid that is no longer our direct child in our group is never signalled.
        for stale in ([(12345,1,12345)],[(12345,1,12345),(12346,1,12346)],[(12345,1,12345),(12346,12345,999)]):
            proc=Mock(pid=12345,returncode=0);proc.poll.return_value=None;proc.wait.return_value=0
            with patch.object(v,'proc_rows',return_value=stale),patch.object(v.os,'kill') as term, \
                 patch.object(v.os,'killpg'):
                v.own_stop(proc,12346)
            term.assert_called_once_with(12345,signal.SIGTERM)

    def test_blocks_pass_server_pid_to_own_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);bridge=out/'bridge';bridge.mkdir()
            proc=Mock(pid=12345);proc.poll.return_value=None
            with contextlib.ExitStack() as stack:
                for name,value in [('free_ports',None),('telemetry',dict()),('spawn',proc),('server_pid',12346),
                                   ('group_alive',False),('gate',dict(skin=30)),('agent_found',False),
                                   ('lmk_request',dict(ok=True))]:
                    stack.enter_context(patch.object(v,name,return_value=value))
                stop=stack.enter_context(patch.object(v,'own_stop',return_value=0))
                stack.enter_context(patch.object(v,'post',side_effect=OSError('stop here')))
                # Clock advances 5 s per call: the pid is found at once, /health never answers, load times out.
                stack.enter_context(patch.object(v.time,'monotonic',side_effect=itertools.count(0,5)))
                def sleep(t):
                    if (bridge/'drop_request').exists():(bridge/'drop_done').write_text('ok')
                stack.enter_context(patch.object(v.time,'sleep',side_effect=sleep))
                v.smoke_block('T1',out,bridge,True,True,[('A5',{})])
                stop.assert_called_with(proc,12346)
                stop.reset_mock()
                v.run_block('T1',out,bridge,30,[('A5',{})],True)
                stop.assert_called_with(proc,12346)

    def test_spawn_interruption_owned_cleanup(self):
        proc=Mock(pid=12345)
        def popen(*args,**kwargs):
            v.interrupted(signal.SIGTERM,None)
            return proc
        with patch.object(v.subprocess,'Popen',side_effect=popen),patch.object(v,'own_stop') as stop:
            with self.assertRaises(SystemExit):v.spawn(['timeout','900','fake'],io.StringIO())
        stop.assert_called_once_with(proc)
        v.OWNED.clear()

    def test_resume_valid_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name,label in [('T1','VALID'),('T2','NOT VALID'),('V1','INDICATIVE')]:
                v.write(p/f'{name}_fake.json',dict(block=name,label=label))
            self.assertEqual(v.completed(p),{'T1'})
            (p/'partial.jsonl').write_text('{"ok": 1}\n{"torn"')
            self.assertEqual(v.rows(p/'partial.jsonl'),[dict(ok=1)])

    def test_resume_preserves_t1_and_original_records(self):
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as tmp:
                here=Path(tmp);out=here/'runs'/'original';out.mkdir(parents=True)
                bridge=here/'bridge';bridge.mkdir();(bridge/'owner_launch').touch()
                spec=dict(blocks=list(v.ORDER),code_sha256={'wrapper':'original'})
                v.write(out/'run.json',dict(manifest=spec,idle_skin=36.676502))
                v.write(out/'T1_original.json',dict(block='T1',label='VALID',requests=['saved reply']))
                v.write(out/'T2_original.json',dict(block='T2',label='NOT VALID',reason='gate missed'))
                originals={p.name:p.read_bytes() for p in out.iterdir()}
                def attempt(block,dest,source,idle,payloads,t2_draft):
                    self.assertEqual((block,idle),('T2',36.676502))
                    record=dict(block=block,label='NOT VALID',reason='gate missed again')
                    v.write(dest/'T2_new_attempt.json',record)
                    return record
                with contextlib.ExitStack() as stack:
                    for name,value in [('HERE',here),('EXPECTED_HASHES',{Path('fake-weight'):'h'})]:
                        stack.enter_context(patch.object(v,name,value))
                    for name,value in [('agent_found',False),('free_ports',None),('telemetry',{}),
                                       ('check_cores',None),('inputs',([('A5',{})],[])),('sha','h'),
                                       ('manifest',dict(spec,code_sha256={'wrapper':'changed'}) if mismatch else spec)]:
                        stack.enter_context(patch.object(v,name,return_value=value))
                    stack.enter_context(patch.object(v.os,'sched_setaffinity'))
                    stack.enter_context(patch.object(v.signal,'signal'))
                    stack.enter_context(patch.object(v.subprocess,'run',return_value=Mock(returncode=0,
                        stdout='--model-draft --spec-type --spec-draft-n-max draft-mtp',stderr='')))
                    run=stack.enter_context(patch.object(v,'run_block',side_effect=attempt))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    with self.assertRaisesRegex(SystemExit,'manifest mismatch' if mismatch else 'stopped'):
                        v.main(['--resume',str(out),'--bridge',str(bridge)])
                self.assertEqual(run.call_count,0 if mismatch else 1)
                for name,data in originals.items():self.assertEqual((out/name).read_bytes(),data)
                self.assertEqual(v.completed(out),{'T1'})

    def test_gate_interruption_saves_failed_attempt_without_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with patch.object(v,'gate',side_effect=SystemExit('owner aborted wrapper')), \
                 patch.object(v,'spawn') as spawn,patch.object(v,'lmk_request',return_value=dict(ok=True)):
                record=v.run_block('T2',p,p,30,[('A5',{})],True)
            spawn.assert_not_called()
            self.assertEqual(record['label'],'NOT VALID')
            self.assertIn('owner aborted wrapper',record['reason'])
            self.assertEqual(len(list(p.glob('T2_*.json'))),1)

    def test_thermal_parser_ignores_cached(self):
        text=('Thermal Status: 1\nCurrent temperatures from HAL:\n'
              '  Temperature{mValue=31.2, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\n'
              'Cached temperatures:\n  Temperature{mValue=99, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\n')
        self.assertEqual(v.parse_dump(text),dict(status=1,skin=31.2))

    def test_power_and_caps(self):
        samples=[dict(t=t,battery_w=w,skin=30+t/100,status=0,mem_available_kib=3000,
                      vmhwm_kib=4000,pss_kib=3900,max=dict.fromkeys(v.POLICIES,50),
                      cpuinfo_max_khz=dict.fromkeys(v.POLICIES,100)) for t,w in [(-1,2),(0,2),(1,4),(2,6),(3,6)]]
        m=v.metrics(dict(samples=samples,measure_start=0,measure_end=2))
        self.assertEqual(m['power']['mean_battery_w'],4)
        self.assertEqual(m['caps']['policy6']['percent'],100)
        samples[2]['t']=10
        self.assertIsNone(v.metrics(dict(samples=samples[:3],measure_start=0,measure_end=2))['power']['mean_battery_w'])

    def test_block_fake_lifecycle_and_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);bridge=out/'bridge';bridge.mkdir()
            clock=[0.]
            proc=Mock(pid=12345,returncode=-15);proc.poll.return_value=None
            count=[0]
            def post(path,body=None,timeout=30):
                if path == '/health':return dict(status='ok')
                count[0]+=1
                clock[0] += 60
                return dict(content='fake reply',timings=dict(prompt_n=4,prompt_ms=10,predicted_n=2,predicted_per_second=5))
            def sleep(t):
                clock[0]+=t
                if (bridge/'drop_request').exists():(bridge/'drop_done').write_text('ok')
            with contextlib.ExitStack() as stack:
                for name,value in [('gate',dict(skin=30,status=0,waited_s=0,missed=False)),('free_ports',None),
                                   ('agent_found',False),('spawn',proc),('server_pid',12346),('check_cores',None),
                                   ('telemetry',dict()),('lmk_request',dict(ok=True,n_kills=0)),('own_stop',-15),
                                   ('request_record',dict(reply='fake reply'))]:
                    stack.enter_context(patch.object(v,name,return_value=value))
                stack.enter_context(patch.object(v,'post',side_effect=post))
                stack.enter_context(patch.object(v.time,'monotonic',side_effect=lambda:clock[0]))
                stack.enter_context(patch.object(v.time,'sleep',side_effect=sleep))
                record=v.run_block('T1',out,bridge,30,[('A5',{})],True)
            self.assertEqual(record['label'],'VALID')
            self.assertEqual(count[0],4)  # warm-up plus three measured requests
            self.assertEqual(len(record['requests']),3)
            self.assertEqual(record['measure_end']-record['measure_start'],180)
            self.assertEqual(record['server_exit'],-15)
            self.assertIsNone(json.loads((bridge/'target.json').read_text())['pid'])

    def test_missed_gate_never_spawns(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with patch.object(v,'gate',side_effect=RuntimeError('gate missed')),patch.object(v,'spawn') as spawn, \
                 patch.object(v,'lmk_request',return_value=dict(ok=True)):
                record=v.run_block('T1',p,p,30,[('A5',{})],True)
            spawn.assert_not_called()
            self.assertEqual(record['label'],'NOT VALID')
            self.assertIn('gate missed',record['reason'])

    def test_monitor_separates_hal_from_power(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            bridge=Path(tmp);calls={}
            native_is_dir=Path.is_dir
            def is_dir(path):
                return False if str(path)=='/termux-home' else native_is_dir(path)
            def root(script,tag):
                calls[tag]=threading.get_ident()
                if tag=='thermal':
                    return ('Thermal Status: 0\nCurrent temperatures from HAL:\n'
                            '  Temperature{mValue=30, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\n')
                self.assertEqual(tag,'sample')
                (bridge/'stop').touch()
                return '-1000000\n4000000\nDischarging\n75\n300\nCAPS\n50\n100\n50\n100\n50\n100\nCPU\n30000\n30000\n30000\n'
            with patch.object(Path,'is_dir',is_dir),patch.dict(os.environ,PREFIX='/data/data/com.termux/files/usr'), \
                 patch.object(v.os,'sched_setaffinity'),patch.object(v,'agent_matches',return_value=[]), \
                 patch.object(v,'root',side_effect=root):
                v.monitor(bridge)
            self.assertNotEqual(calls['thermal'],calls['sample'])
            sample=json.loads((bridge/'latest.json').read_text())
            self.assertEqual(sample['battery_w'],4)
            self.assertEqual(sample['capacity'],75)
            self.assertEqual(sample['skin'],30)
            self.assertEqual(sample['max']['policy6'],50)
            self.assertFalse((bridge/'errors.jsonl').exists())

    def test_wrapper_contract(self):
        text=Path(v.HERE/'oneshot.sh').read_text()
        for required in ('--preflight','pause 300','--monitor','proot-distro login debian','drop_caches',
                          '--bridge','--smoke'):
            self.assertIn(required,text)
        self.assertNotIn('pkill',text)
        self.assertNotIn('killall',text)
        # Preflight runs in Debian's own interpreter too, before the idle (VIS1-FIX).
        self.assertLess(text.index('--preflight'),text.index('pause 300'))
        self.assertEqual(text.count('proot-distro login debian'),2)
        # Inspect only: never execute this owner/hardware wrapper in offline tests.
        stop=text.index('[ "$PREFLIGHT" = 0 ] || exit 0')
        self.assertIn('[ "$ARG" = --preflight ] && PREFLIGHT=1',text[:stop])
        self.assertLess(text.index('PREFLIGHT=0'),text.index('for ARG'))
        self.assertLess(text.index('|| exit 1',text.index('proot-distro login debian')),stop)
        for action in ('su -c id','mkdir -p','termux-wake-lock','--monitor','pause 300'):
            self.assertLess(stop,text.index(action))

    def test_agent_matcher_ignores_path_substrings(self):
        # A process whose cmdline merely contains "codex" as part of a path, not its own name.
        fake_pid = os.getpid()+90000
        with patch.object(v.subprocess,'run',return_value=Mock(returncode=0,stdout=f'{fake_pid}\n')), \
             patch.object(v.Path,'read_text',return_value='bash\n'), \
             patch.object(v.Path,'read_bytes',return_value=b'/usr/bin/bash\0--config\0/root/.codex/thing\0'):
            self.assertEqual(v.agent_matches(),[])

    def test_agent_matcher_catches_real_agent_process(self):
        fake_pid = os.getpid()+90001
        with patch.object(v.subprocess,'run',return_value=Mock(returncode=0,stdout=f'{fake_pid}\n')), \
             patch.object(v.Path,'read_text',return_value='codex\n'), \
             patch.object(v.Path,'read_bytes',return_value=b'/usr/local/bin/codex\0app-server\0'):
            matches = v.agent_matches()
        self.assertEqual(matches,[dict(pid=fake_pid,comm='codex',cmdline='/usr/local/bin/codex app-server')])

    def test_agent_found_logs_only_when_asked(self):
        with patch.object(v,'agent_matches',return_value=[dict(pid=1,comm='agy',cmdline='agy run')]), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(v.agent_found())
        self.assertEqual(out.getvalue(),'')
        with patch.object(v,'agent_matches',return_value=[dict(pid=1,comm='agy',cmdline='agy run')]), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(v.agent_found(log=True))
        self.assertIn('pid=1 comm=agy cmdline=agy run',out.getvalue())

    def test_telemetry_allow_agent_bypasses_only_the_agent_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            sample=dict(t=100,capacity=60,battery_status='Discharging',agent=True,agent_detail=[],
                        skin=30,status=0,bat_c=30)
            v.write(p/'latest.json',sample)
            with patch.object(v.time,'monotonic',return_value=100):
                with self.assertRaises(RuntimeError):
                    v.telemetry(p)
                v.telemetry(p,allow_agent=True)  # does not raise
                v.write(p/'latest.json',dict(sample,capacity=81))
                with self.assertRaises(RuntimeError):
                    v.telemetry(p,allow_agent=True)  # other checks still enforced

    def test_preflight_reports_missing_file_before_hash(self):
        with patch.object(v,'free_ports'),patch.object(v,'agent_matches',return_value=[]), \
             patch.object(v.Path,'is_file',return_value=False):
            with self.assertRaisesRegex(SystemExit,'missing file'):
                v.preflight()

    def test_preflight_debian_reports_missing_imports(self):
        native_is_dir=Path.is_dir
        def is_dir(path):
            return True if str(path)=='/termux-home' else native_is_dir(path)
        with patch.object(v,'free_ports'),patch.object(v,'agent_matches',return_value=[]), \
             patch.object(v.Path,'is_file',return_value=True),patch.object(v,'sha',return_value='x'), \
             patch.dict(v.EXPECTED_HASHES,{p:'x' for p in v.EXPECTED_HASHES}), \
             patch.object(Path,'is_dir',is_dir), \
             patch.object(v.importlib.util,'find_spec',side_effect=lambda m:None if m=='PIL' else object()):
            with self.assertRaisesRegex(SystemExit,'apt-get install'):
                v.preflight()

    def test_smoke_block_one_request_starts_and_stops_own_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);bridge=out/'bridge';bridge.mkdir()
            proc=Mock(pid=12345,returncode=0);proc.poll.return_value=None
            calls=[]
            def post(path,body=None,timeout=30):
                calls.append(path)
                if path == '/health':return dict(status='ok')
                return dict(content='fake reply',timings=dict(prompt_n=4,prompt_ms=10,predicted_n=2,predicted_per_second=5))
            with contextlib.ExitStack() as stack:
                for name,value in [('free_ports',None),('agent_matches',[]),('spawn',proc),
                                   ('server_pid',12346),('check_cores',None),('group_alive',False),
                                   ('telemetry',dict(skin=30)),('own_stop',0),
                                   ('request_record',dict(reply='fake reply',image_tokens=42))]:
                    stack.enter_context(patch.object(v,name,return_value=value))
                stack.enter_context(patch.object(v,'post',side_effect=post))
                record=v.smoke_block('V1',out,bridge,True,False,[('frame_a',{})])
            self.assertEqual(record['label'],v.SMOKE_LABEL)
            self.assertTrue(record['server_started'])
            self.assertTrue(record['reply_received'])
            self.assertEqual(record['image_tokens'],42)
            self.assertTrue(record['exit_clean'])
            self.assertEqual(calls,['/health','/v1/chat/completions'])
            self.assertIsNone(json.loads((bridge/'target.json').read_text())['pid'])

    def test_server_pid_without_proc_children_file(self):
        # VIS1-FIX2: the kernel has no /proc/<pid>/task/<pid>/children; select by ppid+pgid only.
        proc=Mock(pid=100)
        rows=[(100,1,100),(101,100,100),(102,100,555),(103,101,100),(7,1,7)]
        with patch.object(v,'proc_rows',return_value=rows):
            self.assertEqual(v.server_pid(proc),101)
        for bad in ([(100,1,100)],[(100,1,100),(101,100,100),(104,100,100)]):
            with patch.object(v,'proc_rows',return_value=bad):
                with self.assertRaisesRegex(RuntimeError,'owned timeout child'):v.server_pid(proc)

    def test_proc_rows_parses_stat_with_odd_comm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for pid,stat in [('42','42 (llama server) S 41 42 42 0'),('43','43 (a)b) R 42 40 40 0'),('x','junk')]:
                (root/pid).mkdir();(root/pid/'stat').write_text(stat)
            (root/'44').mkdir()  # exited: no stat file
            real=v.Path
            with patch.object(v,'Path',side_effect=lambda p:real(tmp) if p=='/proc' else real(p)):
                self.assertEqual(sorted(v.proc_rows()),[(42,41,42),(43,42,40)])

    def test_smoke_exit_clean_means_no_kill_and_empty_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);bridge=out/'bridge';bridge.mkdir()
            proc=Mock(pid=12345);proc.poll.return_value=None
            # Strict: exit 0 and nothing left in our group; any other status (1, -15, -9, -6) is unclean.
            for rc,alive,clean in [(0,False,True),(1,False,False),(-15,False,False),(-9,False,False),
                                   (-6,False,False),(0,True,False)]:
                with contextlib.ExitStack() as stack:
                    for name,value in [('free_ports',None),('telemetry',dict()),('spawn',proc),
                                       ('server_pid',12346),('own_stop',rc),('group_alive',alive)]:
                        stack.enter_context(patch.object(v,name,return_value=value))
                    stack.enter_context(patch.object(v,'post',side_effect=OSError('stop here')))
                    # pid found at t=0, then /health never answers and load times out at t=999
                    stack.enter_context(patch.object(v.time,'monotonic',side_effect=itertools.chain([0,0],itertools.repeat(999))))
                    record=v.smoke_block('T1',out,bridge,True,True,[('A5',{})])
                self.assertTrue(record['server_started'])
                self.assertEqual(record['exit_clean'],clean,(rc,alive))
                self.assertEqual(record['server_exit'],rc)

    def test_smoke_block_allow_agent_skips_agent_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);bridge=out/'bridge';bridge.mkdir()
            with patch.object(v,'telemetry',return_value=dict()) as telemetry, \
                 patch.object(v,'agent_matches') as matches,patch.object(v,'free_ports'), \
                 patch.object(v,'spawn',side_effect=RuntimeError('stop after agent check')):
                record=v.smoke_block('T1',out,bridge,True,True,[('A5',{})])
            matches.assert_not_called()
            telemetry.assert_called_once_with(bridge,allow_agent=True)
            self.assertIn('stop after agent check',record['reason'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
