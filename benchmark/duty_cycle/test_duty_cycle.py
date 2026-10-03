#!/usr/bin/env python3
"""Offline assertions only. All Android/root/model calls below are mocked."""
import json
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import duty_cycle as dc
import confirm_view as cv
cr = dc.cr


def check_plans():
    assert [s['name'] for s in dc.blocks_for('PARTS')] == ['P0U','LOAD','P0','CAM','YOLO','YOLO_NOSPIN','SEL']
    assert [s['duration'] for s in dc.blocks_for('PARTS', True)] == [20,60,20,20,20,20,20]
    assert [s['duration'] for s in dc.blocks_for('CYCLE')] == [240,240,240,540]
    assert [s['duration'] for s in dc.blocks_for('CYCLE', True)] == [20,20,20,50]
    for smoke in (False, True):
        for spec in dc.blocks_for('CYCLE', smoke):
            plan = dc.phase_plan(spec, smoke)
            assert sum(p[1] for p in plan) == spec['duration']
            clock, active, slots = 0, 0, []
            for phase, length, _, _ in plan:
                if phase == 'active':
                    for s in dc.selector_slots(active, length):
                        assert 0 <= s < length
                        slots.append((clock+s, active+s))
                    active += length
                clock += length
            assert [s[1] for s in slots] == list(range(0, int(active), 20))
    assert dc.selector_slots(50, 50) == [10,30]
    assert dc.selector_slots(25, 25) == [15]
    print('PARTS/CYCLE/smoke block lists and active-only 20 s selector clock: PASS')


def check_restart():
    clock, thresholds = [100.], []
    def frame(*a, **k):
        thresholds.append(k['min_capture_boot_s'])
        return dict(status='missing') if len(thresholds) == 1 else dict(status='ok', session='new', frame=1)
    with patch.object(cr, 'am'), patch.object(cr, 'read_frame', frame), \
         patch.object(dc.pm.time, 'clock_gettime', lambda k: clock[0]), \
         patch.object(dc.pm.time, 'monotonic', lambda: clock[0]), \
         patch.object(dc.pm.time, 'sleep', lambda s: clock.__setitem__(0, clock[0]+s)):
        result = dc.pm.camera_start(1)
    assert thresholds == [100.,100.] and result['attempts'] == 1
    # Real reader must reject a valid JPEG with capture <= request, even in a new session.
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'frame.jpg'
        Image.new('RGB',(4,4)).save(path, comment=b'robotcam session=abc frame=1 capture_boot_ms=100000 capture_wall_ms=1 clock=arrival')
        with patch.object(cr.robotcam_reader.time, 'clock_gettime', lambda k: 100.1):
            assert cr.read_frame(tmp, min_capture_boot_s=100.)['status'] == 'missing'
            assert cr.read_frame(tmp, min_capture_boot_s=99.)['status'] == 'ok'
    print('restart passes request CLOCK_BOOTTIME; actual reader rejects pre-restart capture: PASS')


def check_energy_cpu():
    rows = [dict(t=i/2, battery_w=2+i/2) for i in range(9)]
    assert dc.energy_above(rows, 2., .25, 3.75) == 7.
    assert dc.energy_above(rows, 2., 1., 2.) == 1.5
    assert dc.energy_above(rows, 2., -.1, 2.) is None
    assert dc.energy_above([rows[0],rows[-1]],2.,0.,4.) is None
    assert dc.energy_above(rows,None,0.,1.) is None
    assert dc.mean_power(rows,[(0,1),(3,4)]) == 4.
    fields=['S']+['0']*40
    fields[11],fields[12],fields[19]='7','3','500'
    assert dc.parse_stat('123 (name (space)) '+' '.join(fields)) == dict(pid=123,ticks=10,born_ticks=500)
    print('energy above baseline and per-call interpolation/missing coverage; proc stat parser: PASS')


def check_camera_pacing():
    from PIL import Image
    # Exercise the actual reader and JPEG decoder at 1 Hz through the shared camera iteration.
    for name in ('CAM', 'YOLO', 'YOLO_NOSPIN', 'Active'):
        clock, times, decodes = [100.], [], []
        class Detector:
            def detect(self, *args):
                clock[0] += .2
                return []
        reads, last = [], None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frame.jpg'
            read = cr.read_frame
            image_open = cr.robotcam_reader.Image.open
            def decoded(*a, **kw):
                decodes.append(clock[0])
                return image_open(*a, **kw)
            def frame(*a, **kw):
                times.append(clock[0])
                number = int(clock[0]-100)+1
                Image.new('RGB',(4,4)).save(path,comment=f'robotcam session=abc frame={number} capture_boot_ms={int(clock[0]*1000)} capture_wall_ms=1 clock=arrival'.encode())
                result = read(tmp, session='abc')
                clock[0] += .03  # read+decode time must consume, not extend, the one-second period
                return result
            policy = dc.pm.size_policy('5')
            with patch.object(cr,'read_frame',frame), patch.object(cr.robotcam_reader.Image,'open',decoded), \
                 patch.object(dc.time,'monotonic',lambda:clock[0]), patch.object(dc.time,'perf_counter',lambda:clock[0]), \
                 patch.object(dc.time,'clock_gettime',lambda k:clock[0]), \
                 patch.object(dc.time,'sleep',lambda n:clock.__setitem__(0,clock[0]+n)):
                while clock[0] < 104.:
                    last = dc.camera_frame(None if name=='CAM' else Detector(),policy,'abc',last,100.,reads,rate=1)
        assert times == [100.,101.,102.,103.] and decodes == times, (name,times,decodes)
        assert last==4 and all(r['status']=='ok' for r in reads)
    print('CAM/YOLO/YOLO_NOSPIN/Active rate 1: exactly one read and JPEG decode per second, work included in period: PASS')


def check_load():
    events=[]
    class Server:
        def __init__(self,path):
            events.append('spawn'); self.load_s=.4; self.proc=type('P',(),{'pid': 123})()
        def stop(self): events.append('stop')
    snapshots=iter([100,110,150])
    def cpu(server):
        return dict(t=0, groups={'llama_server':[dict(ticks=next(snapshots))]}, errors=[])
    ctx=dict(out=Path('/unused'),cases=[dict(id='one',answer='a')],cpu=[])
    with patch.object(cr,'resident_pages', lambda p: 0), patch.object(cr,'meminfo_mib', lambda: {'mem_available_mib':2500}), \
         patch.object(cr,'Server',Server), patch.object(cr,'make_selector',lambda: object()), patch.object(dc,'cpu_snapshot',cpu), \
         patch.object(cr,'select',lambda s,c: dict(ms=20.,correct=True)), patch.object(os,'sysconf',lambda k:100):
        row=dc.load_once(ctx,time.monotonic(),0)
    assert events==['spawn','stop'] and row['cache_state']=='cold' and row['load_cpu_s']==1. and row['call_cpu_s']==.4
    events.clear()
    with patch.object(cr,'resident_pages',lambda p:5), patch.object(cr,'meminfo_mib',lambda:{'mem_available_mib':1}), \
         patch.object(cr,'Server',Server), patch.object(dc,'cpu_snapshot',lambda s:dict(groups={},errors=[])), \
         patch.object(cr,'make_selector',side_effect=SystemExit(143)):
        try: dc.load_once(ctx,time.monotonic(),0)
        except SystemExit: pass
        else: raise AssertionError('interruption lost')
    assert events==['spawn','stop']
    # Supervised interruption waits for registration and cleanup, not just the worker's caller.
    ended=threading.Event()
    def fake_load(*args):
        try: time.sleep(.03)
        finally: ended.set()
    with patch.object(dc,'load_once',fake_load):
        try: dc.supervised_load(ctx,0,0,lambda: (_ for _ in ()).throw(SystemExit(143)))
        except SystemExit: pass
    assert ended.is_set()
    print('LOAD spawn/health/call CPU split, GGUF cold/warm, stop on interruption, worker joined: PASS')


def check_drain_exceptions():
    with tempfile.TemporaryDirectory() as tmp:
        ctx=dict(out=Path(tmp))
        for exception in (RuntimeError('failed limit query'), SystemExit(143)):
            finished=threading.Event()
            thread=threading.Thread(target=lambda:(time.sleep(.05),finished.set()))
            thread.start()
            record=dict(start=0)
            def limit(): raise exception
            try:
                dc.drain_selector(thread,threading.Event(),ctx,record,'CONT',limit,None)
            except BaseException as e: assert e is exception
            else: raise AssertionError('exception swallowed')
            assert finished.is_set() and not thread.is_alive() and record['selector_drain_s']>0
        # An exception already unwinding the phase also drains before it can leave run_block.
        finished.clear()
        thread=threading.Thread(target=lambda:(time.sleep(.05),finished.set()))
        thread.start()
        try:
            try: raise cr.CoresLost('body core loss')
            finally:
                dc.drain_selector(thread,threading.Event(),ctx,dict(start=0),'CONT',lambda:None,None)
        except cr.CoresLost: pass
        assert finished.is_set() and not thread.is_alive()
        # A timeout cancels via the owned server and still joins, rather than leaving a daemon behind.
        cancelled=threading.Event()
        class Server:
            def stop(self): cancelled.set()
        thread=threading.Thread(target=cancelled.wait)
        thread.start()
        clock=iter([0.,121.,122.])
        with patch.object(cr,'LIVE',{Server()}), patch.object(dc.time,'monotonic',lambda:next(clock)):
            try: dc.drain_selector(thread,threading.Event(),ctx,dict(start=0),'CONT',lambda:None,None)
            except RuntimeError as e: assert 'cancelled' in str(e)
            else: raise AssertionError('timeout not reported')
        assert cancelled.is_set() and not thread.is_alive()
    for termination in (SystemExit(143), SystemExit(129), SystemExit(130), KeyboardInterrupt()):
        class Pending:
            def __init__(self): self.joined=False; self.signalled=False
            def is_alive(self): return not self.joined
            def join(self, **kw):
                if not self.signalled:
                    self.signalled=True
                    raise termination
                self.joined=True
        thread=Pending()
        clock=iter([0.,121.,122.])
        with tempfile.TemporaryDirectory() as tmp, patch.object(cr,'LIVE',set()), \
             patch.object(dc.time,'monotonic',lambda:next(clock)):
            try:
                try: raise cr.CoresLost('retained body error')
                finally:
                    dc.drain_selector(thread,threading.Event(),dict(out=Path(tmp)),dict(start=0),'CONT',lambda:None,None)
            except BaseException as e: assert e is termination, type(e)
            else: raise AssertionError('termination swallowed in final join')
        assert thread.joined
    print('SystemExit TERM/HUP/INT and KeyboardInterrupt override retained CoresLost in final join, after joining: PASS')
    print('selector drain: arbitrary error, signal, unwinding CoresLost and timeout cancellation all join before propagation: PASS')


def check_spinning():
    configs=[]
    class Options:
        def __init__(self): self.config={}
        def add_session_config_entry(self,k,v): self.config[k]=v; configs.append((k,v))
        def get_session_config_entry(self,k): return self.config[k]
    class Session:
        def __init__(self,path,sess_options): self.opts=sess_options
        def get_session_options(self): return self.opts
    class Wrapped:
        def __init__(self,d,cpus): self.tid=44
    with patch.object(dc.pm,'thread_ids',side_effect=[set(),{22,33}]), patch.object(dc.dp.ort,'SessionOptions',Options), \
         patch.object(dc.dp.ort,'InferenceSession',Session), patch.object(dc.pm,'PinnedDetector',Wrapped), \
         patch.object(os,'sched_setaffinity'), patch.object(os,'sched_getaffinity',lambda tid:{4,5}):
        _,info=dc.build_detector(True)
    assert all(info['spinning_accepted'].values()) and len(configs)==2
    ctx=dict(smoke=True, thermal_log='', idle={}, server=object(), out=Path('/unused'))
    with patch.object(cr,'check_cores'), patch.object(dc.pm,'thermal_gate',lambda *args:{}), \
         patch.object(dc,'build_detector',side_effect=RuntimeError('option rejected')):
        b=dc.run_block(dict(name='YOLO_NOSPIN',duration=20),ctx)
    assert 'rejected' in dc.block_problems(b)[0] and b['ort']['spinning_accepted'] is False
    print('both ORT spinning options retained; rejection produces INCOMPLETE without fallback: PASS')


def synthetic(spec):
    duration=spec['duration']
    groups={n:[] for n in ('runner','llama_server','robotcam_app','camera_provider')}
    return dict(block=spec['name'],spec={**spec,'threads':'mid'},duration_s=duration,planned_s=duration,
                incomplete=[], heat_stop=dict(limit=None,reached_limit=False), lmk=dict(ok=True,lines=[]),
                power=[dict(t=i*.5,battery_w=2.,battery_status='Discharging') for i in range(2*duration+1)],
                fast=[dict(t=i,max=dict.fromkeys(dc.pm.POLICIES,2000)) for i in range(duration)],
                dumps=[dict(t=i,t_start=i,skin=30+i/100,status=0) for i in range(0,duration,5)],
                cpuinfo_max_khz=dict.fromkeys(dc.pm.POLICIES,2000),cpu_snapshots=[dict(groups=groups,errors=[])],
                cpu_seconds=dict.fromkeys(groups,0),server_alive=True,ort={},reads=[],selector_calls=[],
                loads=[],phases=[],restarts=[],thread_samples=[],memory=[dict(t=0,mem_available_mib=100)],thermal_start=dict(waited_s=0,warm_start=False))


def check_report_resume():
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)
        specs=dc.blocks_for('PARTS',True)[:1]+[dict(name='P0',duration=20)]
        (out/'run.json').write_text(json.dumps(dict(set='PARTS',smoke=True,blocks=specs)))
        for spec in specs: dc.save_block(out,spec,synthetic(spec))
        text,bad=dc.report(out)
        assert not bad,bad
        assert 'SMOKE: not a measurement' in text and 'derived cost above base P0U W: 0.000' in text
        assert 'sampled every 0.5 s' in text and 'CPU-core reading, not a heat state' in text
        saved=(out/'block_P0U.json').read_bytes()
        spec=dict(name='HEATCOOL',duration=50)
        assert not dc.completed(out,spec)
        (out/'block_HEATCOOL.tmp').write_text('interrupted during COOL')
        assert not dc.completed(out,spec)
        dc.save_block(out,spec,synthetic(spec))
        assert dc.completed(out,spec) and (out/'block_P0U.json').read_bytes()==saved
        b=synthetic(specs[0]); b['power'][5]['battery_status']='Charging'
        dc.save_block(out,specs[0],b)
        assert any('Discharging' in p for p in dc.report(out)[1])
        # Missing fields in historical data are never filled from current cases.
        (out/'block_CONFIRM.json').write_text(json.dumps(dict(duration_s=60,selector_calls=[dict(t=20,case_id='x',correct=False,choice='b',ms=12)])))
        view=cv.render(out)
        assert 'x | not recorded | b | 12' in view and 'policy4 first recorded cap s: not recorded' in view
    for prefix in ('','/data/data/com.termux/files/usr'):
        with patch.dict(os.environ,{'PREFIX':prefix}):
            try: cr.require_native()
            except SystemExit as e: assert 'native Termux' in str(e)
            else: raise AssertionError('proot allowed')
    print('synthetic report/invalid battery; atomic resume preserves finished blocks and redoes HEATCOOL; proot refusal: PASS')


def check_main_integration(terminate=False):
    from contextlib import ExitStack
    groups={n:[dict(pid=i+1,ticks=10,born_ticks=0)] for i,n in enumerate(('runner','llama_server','robotcam_app','camera_provider'))}
    class Shell:
        def close(self): pass
    servers=[]
    class Server:
        def __init__(self,path):
            self.running=True
            servers.append(self)
            cr.LIVE.add(self)
        def stop(self):
            self.running=False
            cr.LIVE.discard(self)
        def alive(self): return self.running
    class Selector:
        def decide(self,*a): return {}
    class Detector:
        def detect(self,*a): return []
    spec=[dict(name=n,duration=.35) for n in ('CONT','CYC50','HEATCOOL')]
    real_run=dc.run_block
    frame=[0]
    gates=[]
    pending, finished = threading.Event(), threading.Event()
    state = dict(in_drain=False, lost=False, attempts=0, signalled=False, camera_stopped=False, camera_running=False)
    trace=[]
    def select(*args):
        if not pending.is_set():
            pending.set()
            real_sleep(.55)
            trace.append('call finished')
            finished.set()
        return dict(ms=1,correct=True,case_id='one',choice='a')
    def check_cores(*args):
        if state['in_drain'] and pending.is_set() and not finished.is_set() and not state['lost']:
            state['lost']=True
            raise cr.CoresLost('offline: core loss during pending selector drain')
    real_drain=dc.drain_selector
    def drain(*args):
        state['in_drain']=True
        thread=args[0]
        real_join=thread.join
        def join(*a, **kw):
            if terminate and state['lost'] and not state['signalled']:
                assert pending.is_set() and not finished.is_set()
                state['signalled']=True
                os.kill(os.getpid(), signal.SIGTERM)  # main() installed the real exit-143 handler
            return real_join(*a, **kw)
        try:
            with patch.object(thread,'join',join):
                return real_drain(*args)
        finally: state['in_drain']=False
    def camera_start(rate):
        state['camera_running']=True
        return dict(session='abc',frame=0,attempts=1)
    def camera_stop():
        state['camera_running']=False
        state['camera_stopped']=True
        return dict(capture_stopped=True,force_stop_rc=0)
    def plan(s,smoke):
        return [('active',.15,0,True),('pause',.2,0,True)] if s['name']!='CONT' else [('active',.35,0,True)]
    def read_frame(*a,**kw):
        frame[0]+=1
        return dict(status='ok',session='abc',frame=frame[0],image=None,age_s=.1)
    def thread_sample(t0,workers):
        return dict(t=time.monotonic()-t0,threads=[dict(tid=44,processor=4,allowed='4-5',cpu_ticks=10)])
    def fast(*a):
        t=time.monotonic()
        return dict(t=t,t_start=t,cpu_c=30.,bat_c=30.,max=dict.fromkeys(dc.pm.POLICIES,2000))
    def dump():
        t=time.monotonic()
        return dict(t=t,t_start=t,skin=30.,status=0)
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        root=Path(tmp)
        import contextlib, io
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        patches=[(dc,'OUT_ROOT',root),(cr,'require_native',lambda:None),(cr,'root',lambda *a:(0,'fake')),
                 (cr,'wait_cores',lambda *a:None),(cr,'check_cores',check_cores),(cr,'RootShell',Shell),
                 (cr,'discover',lambda s:dict(policies=dict.fromkeys(dc.pm.POLICIES,2000))),
                 (cr,'read_thermal',lambda p:dict(z9=30000)),(cr,'read_dump',dump),(cr,'Server',Server),
                 (cr,'make_selector',Selector),(cr,'load_cases',lambda:[dict(id='one',answer='a')]),
                 (cr,'select',select),(dc,'drain_selector',drain),
                 (cr,'fast_keys',lambda l:[]),(cr,'fast_sample',fast),(cr,'meminfo_mib',lambda:dict(mem_available_mib=100)),
                 (dc,'blocks_for',lambda *a:spec),(dc,'phase_plan',plan),(dc,'cpu_snapshot',lambda *a:dict(t=time.monotonic(),groups=groups,errors=[])),
                 (dc,'stop_camera',camera_stop),
                 (cr,'read_frame',read_frame),(cr,'robotcam_pids',lambda:[3]),(cr,'lmk_lines',lambda *a:dict(ok=True,lines=[])),
                 (dc.pm,'camera_start',camera_start),
                 (dc.pm,'next_size',lambda p:640 if frame[0]%2 else 320),
                 (dc.pm,'thread_sample',thread_sample),(cr,'DOWNLOADS',root),
                 (dc,'build_detector',lambda *a:(Detector(),dict(worker_tids=[44],caller_tid=None,version='fake'))),
                 (dc,'battery_sample',lambda sh,t0:dict(t=time.monotonic()-t0,battery_w=2,battery_status='Discharging')),
                 (dc.pm,'thermal_gate',lambda p,i,n,s:gates.append(n) or dict(waited_s=0,warm_start=False)),
                 (dc.pm,'skin_slope',lambda *a:0.),(cr,'SELECTOR_S',.1)]
        for obj,key,value in patches: stack.enter_context(patch.object(obj,key,value))
        # Skip warm-up waiting and shorten frame sleeps; rate-1 pacing has a separate decoder check.
        real_sleep=time.sleep
        stack.enter_context(patch.object(dc.time,'sleep',lambda n:None if n==60 else real_sleep(min(n,.05))))
        interrupt=[True]
        def run(s,ctx):
            if s['name']=='CONT':
                state['attempts']+=1
                if state['attempts']>1:
                    assert finished.is_set(), 'retried block started with selector still pending'
                    trace.append('retry started')
            if interrupt[0] and s['name']=='HEATCOOL':
                raise SystemExit(143)
            return real_run(s,ctx)
        stack.enter_context(patch.object(dc,'run_block',run))
        try: dc.main(['--set','CYCLE','--smoke'])
        except SystemExit as e: assert e.code==143
        else: raise AssertionError('interruption not propagated')
        out=next(root.glob('run_*'))
        if terminate:
            assert state['lost'] and state['signalled'] and state['attempts']==1
            assert finished.is_set() and trace==['call finished'], trace
            assert state['camera_stopped'] and not state['camera_running'] and not cr.LIVE
            assert servers and all(not server.running for server in servers)
            assert not (out/'block_CONT.json').exists(), 'interrupted block saved or retried'
            drains=[json.loads(line) for line in (out/'selector_drains.jsonl').read_text().splitlines()]
            assert any(r['error']=='SystemExit: 143' for r in drains)
            return
        saved=(out/'block_CONT.json').read_bytes()
        assert not dc.completed(out,spec[-1]) and not cr.LIVE
        assert state['lost'] and state['attempts']==2 and trace[:2]==['call finished','retry started'], trace
        drains=[json.loads(line) for line in (out/'selector_drains.jsonl').read_text().splitlines()]
        assert any(r['error'] and 'CoresLost' in r['error'] and r['drain_s']>.1 for r in drains)
        interrupt[0]=False
        dc.main(['--set','CYCLE','--smoke','--resume',str(out)])
        assert (out/'block_CONT.json').read_bytes()==saved and dc.completed(out,spec[-1])
        assert gates.count('HEATCOOL')==1 and not any(n=='COOL' for n in gates)
        assert not cr.LIVE
        b=json.loads((out/'block_HEATCOOL.json').read_text())
        assert len(b['phases'])==2 and all(p['completed'] for p in b['phases'])
        assert all(c['started_s']<b['phases'][0]['end'] for c in b['selector_calls'])
    print('core loss during pending selector drain: request finished before retry; discarded-block drain time recorded: PASS')
    print('mocked main + real block loops: interrupt/resume preserves CONT, redoes HEATCOOL, no COOL gate, final cleanup: PASS')


if __name__=='__main__':
    check_plans(); check_restart(); check_energy_cpu(); check_camera_pacing(); check_load(); check_drain_exceptions(); check_spinning(); check_report_resume(); check_main_integration(); check_main_integration(terminate=True)
    print("pending selector + core loss + actual SIGTERM: call finished, exit 143, no retry, RobotCam/server stopped: PASS")
    print('DUTY1 offline checks: PASS')
