"""Offline campaign P23 regression checks. No hardware, root, motors, weights or valid timings."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import phase1 as p
import phase23 as q
import runtime as rt
import power
import diagnostics as diag
from self_check import FastOps
import self_check_fix4c as c
import self_check_fix4d as d

BASE='074d5b92ce32330d6fb3c04d9d0ee7db0556ef3d'


def check_gate():
    for temps,maximum,reached,comparable in (([33.,32.,31.5],10,True,True),([33.,33.,33.],10,False,False),([29.],10,True,True),([28.],10,True,False)):
        clock=[0.];samples=iter(temps);reads=[]
        def read():
            row=dict(t=clock[0],skin=next(samples),status=0);reads.append(row);return row
        last,gate=p.start_gate(read,lambda dt:clock.__setitem__(0,clock[0]+dt),lambda:None,30.,maximum,1.5,now=lambda:clock[0])
        assert last is reads[-1] and gate['reading'] is last and gate['end_skin_c']==last['skin'] and gate['reached']==reached
        block=dict(skin_start=last,validity='VALID')
        p.compare_start(block,30.)
        assert (block['validity']=='VALID')==comparable
    clock=[0.]
    final,gate=p.start_gate(lambda:dict(t=clock[0],skin=32.,status=0),lambda dt:clock.__setitem__(0,clock[0]+dt),lambda:None,30.,900,1.,now=lambda:clock[0])
    assert not gate['reached'] and gate['extra_s']==900 and final['t']==900
    final,gate=p.start_gate(lambda:dict(t=900.,skin=31.,status=0),lambda dt:None,lambda:None,30.,900,1.,now=lambda:900.)
    assert gate['reached'] and gate['extra_s']==0
    # Root parser errors are not turned into new reads by the gate.
    read=MagicMock(side_effect=RuntimeError('non-empty bad thermal'))
    try:p.start_gate(read,lambda dt:None,lambda:None,30,900,1.)
    except RuntimeError:pass
    else:raise AssertionError('bad answer hidden')
    read.assert_called_once()
    print('PASS Part B: same final reading identity for gate and comparability; reached, hot timeout, below-band; P23 1 C / 900 s cold gate; parser failure read once')


def check_replacement():
    original_wait=threading.Event.wait
    def fast_wait(event,timeout=None):return original_wait(event,None if timeout is None else max(0,timeout/10))
    for missing in (set(),{0},{0,1,2}):
        ops=FastOps('L2');ops.now=lambda:(p.time.monotonic()-ops.base)*10;ops.origin=0.;record={}
        number=[0]
        frame=ops.frame
        def read(last):
            n=number[0];number[0]+=1
            if n in missing:return dict(status='missing')
            r=frame(last);r['age_s']=1.
            return r
        ops.frame=read
        with patch.object(threading.Event,'wait',fast_wait):
            try:p.run_cycle('L2',60,ops,record,replacement=True)
            except RuntimeError as e:assert len(missing)==3 and '3 consecutive' in str(e),e
            else:
                assert len(missing)<3
                assert record['frames_due']=={'320':46,'640':11}
                assert len(record['frame_slots'])==57
                planned={r['slot_s']:r['size'] for r in record['frame_slots']}
                assert all(r['size']==planned[r['slot_s']] for r in record['yolo'])
                large=[r['slot_s'] for r in record['yolo'] if r['size']==640]
                assert 5<=large[0]<6.1 and all(5<=b-a<6.1 for a,b in zip(large,large[1:])),large
                assert record['frames_skipped']['320']==len(missing),(missing,record['frames_skipped'])
                assert len(record['yolo'])==57-len(missing)
                kinds=[e[0] for e in ops.events if e[0].startswith('m2') or e[0]=='select']
                assert all(kinds[i:i+3]==['m2_start','m2_end','select'] for i in range(0,len(kinds),3)),kinds
                for m,s in zip(record['m2'],record['selector']):assert m['ended_s']<=s['started_s']
                if missing:assert record['cadence_missed']
    ops=FastOps('L2');ops.now=lambda:(p.time.monotonic()-ops.base)*10;ops.origin=ops.now();record={}
    with patch.object(threading.Event,'wait',fast_wait):p.run_cycle('L2',60,ops,record,replacement=True,stop_when=lambda:ops.now()-ops.origin>=24)
    assert record['controlled_end'] and 24<=record['duration_s']<40
    print('PASS real reused scheduler: SizePolicy 46/11 complete slots in 60 s (640 replaces 320; bounded terminal tail); L2 M2-before-selector; H4 one miss skips / three stop; latched controlled end')


def check_rules():
    s=q.FULL
    assert q.stop_reason('endurance',3,100,1800)=='SEVERE'
    assert q.stop_reason('endurance',0,1800,1800)=='TIME LIMIT'
    assert q.stop_reason('endurance',4,100,1800).startswith('EMERGENCY')
    assert q.stop_reason('endurance',3,100,1800,('battery',45)).startswith('EMERGENCY')
    assert q.stop_reason('fixed',3,100,2160) is None
    assert not q.switch('adaptive',True,59,38,37,s)
    assert q.switch('adaptive',True,60,37,37,s)
    assert not q.switch('adaptive',False,29,34,37,s)
    assert not q.switch('adaptive',False,30,35.1,37,s)
    assert q.switch('adaptive',False,30,35,37,s)
    assert not q.switch('fixed',True,119,30,37,s) and q.switch('fixed',True,120,30,37,s)
    assert not q.switch('fixed',False,59,30,37,s) and q.switch('fixed',False,60,30,37,s)
    retries=[dict(t=t) for t in (0,1,2,180,181,182)]
    assert q.retry_windows(retries,0,360)['ok']
    retries.append(dict(t=3));assert not q.retry_windows(retries,0,360)['ok']
    print('PASS SEVERE / 30 min / emergency precedence, fixed timing, adaptive hysteresis and minimum 60/30 s; burst-limited 3 re-reads per 180 s')


def check_review_regressions():
    original_wait=threading.Event.wait
    scale=10
    def fast_wait(event,timeout=None):return original_wait(event,None if timeout is None else max(0,timeout/scale))
    class CameraOps(FastOps):
        def __init__(self):
            super().__init__('L2');self.clock=1000.;self.clock_lock=threading.Lock()
            self.now=lambda:self.clock;self.origin=self.now()
        def advance(self,seconds):
            with self.clock_lock:self.clock+=seconds
        def frame(self,last):
            elapsed=self.now()-self.origin
            index=max(0 if last is None else last,int(elapsed/1.0007))
            captured=index*1.0007
            self.advance(max(0.,captured+.03-elapsed))
            return dict(status='ok',frame=index+1,age_s=self.now()-self.origin-captured,image=None)
        def detect(self,image,size):
            self.advance(self.large_cost if size==640 else .25)
            return [{},{}]
        def relate(self,image,boxes):
            self.advance(.001);return [],1.
        def select(self,index,text):
            self.advance(.001);return dict(ms=1.,miss=False)
    for duration,large_cost in ((4.6,.65),(8.1,.65),(41.4,.65),(1800.3,.65),(60.,1.01),(60.,1.5),(60.,1.9)):
        session=q.LiveSession.__new__(q.LiveSession)
        session.cr=session.sb=session.detector=session.ort=session.caller=session.tids=session.cam=session.server=session.selector=session.cases=session.fallback=None
        session.guard=lambda:None
        session.now=p.time.monotonic
        ops=CameraOps();ops.large_cost=large_cost;record=dict(origin=ops.origin)
        session.now=ops.now
        calls=[];main_tid=threading.get_native_id();stop_event=[None]
        def end():calls.append(threading.get_native_id());return False
        def virtual_wait(event,timeout=None):
            if timeout is None:return original_wait(event,None)
            if threading.get_native_id()==main_tid or timeout<=0:stop_event[0]=event
            if event.is_set() or timeout<=0:return event.is_set()
            if threading.get_native_id()==main_tid:
                ops.advance(timeout)
                # Hold the clock while a due selector runs; host thread scheduling cannot add simulated latency.
                deadline=p.time.monotonic()+5
                due=[s for s in record.get('slot_plan',{}).get('selector_slots',[]) if s<=ops.now()-ops.origin]
                while any(r['size']==640 for r in record.get('yolo',[])) and any(s not in {r['slot_s'] for r in record['selector']} for s in due) and not event.is_set():
                    assert p.time.monotonic()<deadline,'virtual selector did not finish'
                    original_wait(event,.001)
            else:
                if event is stop_event[0]:
                    completed={r['slot_s'] for r in record['selector']}
                    pending=[s for s in record['slot_plan']['selector_slots'] if s not in completed]
                    target=ops.origin+pending[0] if pending else ops.now()
                    while ops.now()<target and not event.is_set():original_wait(event,.001)
                else:original_wait(event,.001)
            return event.is_set()
        with patch.object(threading.Event,'wait',virtual_wait),patch.object(p,'LiveOps',return_value=ops):
            session.cycle(duration,record,end)
        assert not record.get('cadence_missed'),(duration,record['frames_skipped'],max(r['ms'] for r in record['yolo']),len(record['selector']))
        assert all(r['ended_s']<=duration for key in ('yolo','m2','selector') for r in record[key])
        assert all(r['started_s']>=r['slot_s'] for r in record['selector'])
        assert len(record['selector'])==len(record['m2'])==int(duration//20)
        assert set(calls)=={threading.get_native_id()}
        if duration>1000:
            assert len(record['reads'])>1790
            assert all(b['frame']==a['frame']+1 for a,b in zip(record['reads'],record['reads'][1:]))
    # A stopped selector cannot become valid just because the controller later stops normally.
    class Absent:
        def __init__(self,*args,**kwargs):pass
        def start(self):pass
        def join(self,timeout=None):pass
        def is_alive(self):return False
    ops=FastOps('L2');ops.now=lambda:(p.time.monotonic()-ops.base)*scale;ops.origin=ops.now();record={}
    with patch.object(p.threading,'Thread',Absent),patch.object(threading.Event,'wait',fast_wait):
        p.run_cycle('L2',60,ops,record,replacement=True,stop_when=lambda:ops.now()-ops.origin>=24)
    assert record['controlled_end'] and record['cadence_missed']
    # The actual live_block caller must anchor LMK after a long gate, not before it.
    from self_check_fix4 import block_run
    clock=[100.];seen=[]
    def gate(*args,**kwargs):
        clock[0]+=300
        reading=dict(t=clock[0],skin=30.,status=0)
        return reading,dict(reading=reading,end_skin_c=30.,reached=True,extra_s=300)
    def lmk(*args):seen.append(args);return dict(ok=True,n_kills=0)
    r=block_run(lambda *args:None,(patch.object(p.time,'monotonic',side_effect=lambda:clock[0]),
        patch.object(p.time,'time',side_effect=lambda:1000+clock[0]),patch.object(p,'start_gate',side_effect=gate),patch.object(p,'lmk_window',side_effect=lmk)))
    assert r['cycle_origin']==400. and r['lmk_window_epoch_s']==[1400.,1580.]
    assert seen[0][1:]==(1400.,1580.)
    # An interrupt at an evidence bracket must propagate, with the measured end retained.
    s=q.MockSession(None,{});s.start();row=dict(origin=s.now())
    with patch.object(s,'bracket',side_effect=SystemExit(130)):
        try:q.phase_evidence(s,row,[])
        except SystemExit as e:assert e.code==130 and 'end' in row
        else:raise AssertionError('interrupt swallowed')
    print('PASS slow640 1.01/1.5/1.9s without lost frames remains cadence-valid; terminal budget only')
    print('PASS review regressions: REAL LiveSession.cycle/run_cycle fractional tails and 1800.3 s / 1.0007 s camera drift; stop predicate main-only; missing selector remains invalid; post-gate LMK window; evidence interrupt propagates')


def result():return dict(runs=[],mock_only=True,setup_complete=True,setup={'read_retry_check':'OK'})


def check_controller():
    for mode in q.MODES:
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            data={};session=q.MockSession(None,data);session.start();out=result()
            q.run_mode(mode,session,out,q.FULL,37.,lambda suffix:Path(tmp)/('run'+suffix))
            r=out['runs'][0]
            assert r['cold_start'] and r['checks_validity']=='VALID' and r['result_validity']==q.MOCK_LABEL,(mode,r['issues'])
            assert abs(r['duration_s']-q.FULL[mode])<.5,(mode,r['duration_s'])
            assert r['skin_start']==r['gate']['reading']
            if mode=='fixed':
                assert len(r['active_phases'])==len(r['pauses'])==12
                assert r['totals']['camera_restarts']==11 and r['totals']['duty_fraction']==r['totals']['inference_s']/r['duration_s']
                assert all(not x.get('yolo') and x['camera_off_at']>=x['origin'] for x in r['pauses'])
            if mode=='adaptive':assert r['totals']['pause_count'] and r['totals']['camera_restarts']
            assert len(list(Path(tmp).glob('*_active_*.json')))==len(r['active_phases'])
            assert (Path(tmp)/'run_idle.json').exists() and (Path(tmp)/'run_gate.json').exists()
            assert len(r['sample_checkpoints'])>=r['duration_s']//60
            assert all(json.loads(path.read_text())['label']==q.MOCK_LABEL for path in Path(tmp).glob('run_*.json'))
            if mode=='endurance':assert len(r['windows'])==30 and r['stop_reason']=='TIME LIMIT'
    class Severe(q.MockSession):
        def status(self):return 3 if self.camera_on and self.now()-self.origin>=65 else 0
        def camera(self,on):
            if on:self.origin=self.now()
            return super().camera(on)
    class Emergency(q.MockSession):
        def guard(self):
            if self.camera_on and self.now()-self.origin>=65:raise RuntimeError('EMERGENCY: battery 45 C')
        def camera(self,on):
            if on:self.origin=self.now()
            return super().camera(on)
    class CriticalRace(Severe):
        def status(self):return 4 if self.camera_on and self.now()-self.origin>=65 else 0
    for cls,label in ((Severe,'VALID — STOPPED AT SEVERE'),(Emergency,'NOT VALID')):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            session=cls(None,{});session.start();out=result()
            try:q.run_mode('endurance',session,out,q.FULL,37,lambda suffix:Path(tmp)/('run'+suffix))
            except RuntimeError:assert cls is Emergency
            r=out['runs'][0];assert r['checks_validity'].startswith(label) and r['result_validity']==q.MOCK_LABEL,r
            assert r['duration_s']<100 and not session.camera_on
            assert (Path(tmp)/'run_active_01.json').exists() and (Path(tmp)/'run.json').exists()
    for mode in q.MODES:
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            session=CriticalRace(None,{});session.start();out=result()
            q.run_mode(mode,session,out,q.FULL,37,lambda suffix:Path(tmp)/('run'+suffix))
            r=out['runs'][0]
            assert r['checks_validity'].startswith('NOT VALID') and r['stop_reason'].startswith('EMERGENCY')
    class Hot(q.MockSession):
        reads=0
        def reading(self):
            self.reads+=1
            row=dict(t=self.now(),skin=30. if self.reads==1 else 40.,status=0,rc=0,attempts=1)
            self.data['dumps'].append(row)
            return row
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        session=Hot(None,{});session.start();out=result()
        q.run_mode('endurance',session,out,q.SHORT,37.,lambda suffix:Path(tmp)/('run'+suffix))
        r=out['runs'][0]
        assert r['gate']['extra_s']==q.SHORT['gate'] and not r['cold_start']
        assert r['checks_validity']=='WARM START — heat-timing results NOT VALID' and r['active_phases']
    class LateRestart(q.MockSession):
        def camera(self,on):
            if on:self.active_origin=self.now();self.deadline=self.now()+180
            return super().camera(on)
        def skin(self):
            if self.camera_on:return 37. if self.now()-self.active_origin>=60 else 30.
            return 34.9 if hasattr(self,'deadline') and self.now()>=self.deadline-25 else 36.
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        session=LateRestart(None,{});session.start();out=result()
        spec=dict(q.SHORT,idle=0,gate=0,adaptive=180)
        q.run_mode('adaptive',session,out,spec,37.,lambda suffix:Path(tmp)/('run'+suffix))
        r=out['runs'][0]
        assert len(r['active_phases'])==1 and len(r['pauses'])==1 and r['totals']['camera_restarts']==0
        assert r['duration_s']>=180 and r['stop_reason']=='TIME LIMIT'
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        session=q.MockSession(None,{});session.start();out=result()
        caps={key:dict(percent=50. if key=='policy0' else 0.) for key in ('policy0','policy4','policy6')}
        with patch.object(session.pm,'capped_by_policy',return_value=caps):
            q.run_mode('endurance',session,out,q.SHORT,37.,lambda suffix:Path(tmp)/('run'+suffix))
        assert out['runs'][0]['totals']['capping_while_active']
    print('PASS shared controller: full 30 min endurance and 36 min fixed/adaptive virtual sessions, 12 fixed cycles, camera OFF pauses/restarts, atomic per-phase files, SEVERE valid and emergency invalid with files retained')


def check_windows():
    data=dict(cpuinfo_max_khz={k:1000 for k in ('policy0','policy4','policy6')},
        fast=[dict(t=t,max={'policy0':1000,'policy4':900,'policy6':1000}) for t in range(121)],
        dumps=[dict(t=t,skin=34.+t/30,status=int(t>=60)) for t in range(121)],
        memory=[dict(t=t,mem_available_mib=2500,pss_kb={'llama_server':4000}) for t in range(121)],
        power=[dict(t=t,battery_w=4.) for t in range(-1,122)],read_retries=[dict(t=5),dict(t=70)],
        thermal_retries=[dict(t=10,attempts=2),dict(t=80,attempts=3)])
    v,sb,pm,cr,_=rt.imports()
    work=[dict(origin=0,frame_slots=[dict(slot_s=t,size=320) for t in range(120)],
        yolo=[dict(slot_s=t,size=320,ms=10.) for t in range(120) if t!=20],
        m2=[dict(started_s=5,inference_ms=1000.)],selector=[dict(started_s=6,ms=2000.,miss=False)])]
    windows=q.aggregate(data,0,120,work,pm)
    assert len(windows)==2 and windows[0]['yolo']['320']['due']==60 and windows[0]['yolo']['320']['done']==59
    assert windows[0]['yolo']['320']['skipped']==1 and windows[1]['yolo']['320']['skipped']==0
    assert windows[0]['m2']['median_ms']==1000 and windows[0]['selector']['median_ms']==2000
    assert all(w['power']['mean_battery_w']==4. and w['empty_answer_re_reads']==1 for w in windows)
    assert [w['thermal_re_reads'] for w in windows]==[1,2] and [w['read_re_reads'] for w in windows]==[2,3]
    e=q.events(data['dumps'],windows,0)
    assert [r['at_s'] for r in e if r['event']=='android_status']==[0,60]
    assert [r['at_s'] for r in e if r['event']=='skin_crossing']==[30,90]
    assert any(r['event']=='first_window_capped_10_percent' and r['policy']=='policy4' and r['at_s']==0 for r in e)
    assert not any(r.get('policy')=='policy6' for r in e)
    print('PASS 60 s aggregation: due/done/skipped, timings, receipt-weighted power, sensor/status/caps/PSS/memory/re-read counts and event detection')


def check_m3_timing():
    sample=dict(origin=10.,end=40.,camera_on_at_origin=False,camera_start_at=12.,camera_start_end_at=15.,
        inference_origin=16.,inference_end=30.,camera_stop_at=31.,camera_stop_end_at=34.,camera_off_at=34.)
    q.phase_timing(sample)
    assert [sample[k] for k in q.TIMING_FIELDS]==[3.,14.,3.,8.,8.]
    assert sample['duty_fraction']==14/30
    clipped=dict(sample,origin=14.,end=33.);q.phase_timing(clipped)
    assert [clipped[k] for k in q.TIMING_FIELDS]==[1.,14.,2.,0.,5.]
    Delayed=q.MockSession
    for mode in ('fixed','adaptive'):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            session=Delayed(None,{});session.start();out=result()
            spec=q.FULL if mode=='fixed' else dict(q.SHORT,adaptive=360)
            q.run_mode(mode,session,out,spec,37.,lambda suffix:Path(tmp)/('run'+suffix),dry=mode=='adaptive')
            row=out['runs'][0];totals=row['totals']
            assert abs(row['duration_s']-spec[mode])<1e-6 and row['duty_fraction_basis']=='inference_s / duration_s'
            assert 'camera' in row['duty_description'] and 'exactly 120/60' not in row['duty_description']
            assert totals['duty_fraction']==totals['inference_s']/row['duration_s']
            assert abs(sum(totals[k] for k in ('inference_s','camera_off_s','camera_on_without_inference_s'))-row['duration_s'])<1e-6
            for phase in row['active_phases']+row['pauses']:
                assert all(key in phase for key in q.TIMING_FIELDS)
                assert abs(sum(phase[k] for k in ('inference_s','camera_off_s','camera_on_without_inference_s'))-phase['wall_duration_s'])<1e-6
            for cycle in row['cycles']:
                assert all(key in cycle for key in q.TIMING_FIELDS)
                assert cycle['duty_fraction']==cycle['inference_s']/(cycle['active_s']+cycle['pause_s'])
            if mode=='fixed':
                assert len(row['cycles'])==len(row['pauses'])==12
                assert abs(totals['camera_start_s']-36.)<1e-6 and abs(totals['camera_stop_s']-24.)<1e-6
                assert all(abs(c['active_s']-120)<1e-6 and abs(c['pause_s']-60)<1e-6 and c['camera_off_s']<60 and c['inference_s']<120 for c in row['cycles'])
                assert totals['duty_fraction']<2/3 and row['completion_lag_s']>0
            else:
                assert row['pauses'] and totals['camera_restarts']
                assert all(p['end']-p['camera_start_at']>=60-1e-6 for p in row['active_phases'])
                assert all(p['end']-p['camera_off_at']>=30-1e-6 for p in row['pauses'] if p['end']<row['end'])
    print('PASS OWNER M3: full2160s fixed grid/12cycles with3s starts+2s stops inside slots; phase/cycle/totals camera/inference/OFF fields partition wall time; duty uses inference_s; adaptive minimums include start/confirmedOFF')


def check_fixed_rehearsal():
    original_wait=threading.Event.wait
    def fast_wait(event,timeout=None):return original_wait(event,None if timeout is None else max(0,timeout/10))
    class Session(q.MockSession):
        def cycle(self,duration,row,stop_when,force_fallback=False):
            self.cr=self.sb=self.detector=self.ort=self.caller=self.tids=self.cam=self.server=self.selector=self.cases=self.fallback=None
            ops=FastOps('L2');start=self.now()
            ops.now=lambda:start+(p.time.monotonic()-ops.base)*10
            frame=ops.frame
            def read(last):
                r=frame(last);r['age_s']=1.;return r
            ops.frame=read
            with patch.object(threading.Event,'wait',fast_wait),patch.object(p,'LiveOps',return_value=ops):
                q.LiveSession.cycle(self,duration,row,lambda:False,force_fallback=force_fallback)
            self.wait(duration)
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        session=Session(None,{});session.start();out=result()
        q.run_mode('fixed',session,out,q.SHORT,37.,lambda suffix:Path(tmp)/('run'+suffix),dry=True)
        row=out['runs'][0]
        assert len(row['active_phases'])==3 and row['checks_validity']=='VALID',(row['issues'],[(a['frames_skipped'],len(a['selector'])) for a in row['active_phases']])
        assert [len(a['selector']) for a in row['active_phases']]==[1,1,1]
        assert [a['m2'][0]['scene'] for a in row['active_phases']]==['fallback','live','live']
        assert q.coverage(out)['live_m2_calls']['fixed']==2 and q.coverage(out)['fallback_m2_calls']['fixed']==1
    print('PASS REAL fixed rehearsal scheduler with3s starts:40s slots leave one selector each; first active forced fallback, next two liveM2; all three pauses/restarts run')


def check_live():
    v,sb,pm,cr,_=rt.imports()
    data={};server=NS(proc=NS(pid=7),alive=lambda:True)
    session=q.LiveSession(server,data)
    session.data.update(monitor_errors=[]);session.began=0
    cam=MagicMock(return_value={'session':'real-mock'})
    stop=MagicMock(return_value={'capture_stopped':True})
    with patch.object(rt,'camera_start',cam),patch.object(rt,'stop_camera',stop),patch.object(rt,'memory_sample',return_value=dict(t=1,root_rc=0,battery_status='Discharging',pss_kb={})) as memory:
        session.camera(True);session.memory();assert memory.call_args.args[-1] is True
        session.camera(False);session.memory();assert memory.call_args.args[-1] is False
        assert data['camera_starts']==1;cam.assert_called_once_with(pm,rate=1);stop.assert_called_once_with(cr)
    with patch.object(rt,'camera_start',side_effect=RuntimeError('bad camera')):
        try:session.camera(True)
        except RuntimeError:pass
        else:raise AssertionError('startup failed open')
    assert data['camera_failures']==1
    with patch.object(rt,'camera_start',return_value=dict(session='retry',attempts=2)):
        session.camera(True)
    assert data['camera_failures']==2
    def recovered_launch(*args,**kwargs):
        rt.READ_RETRIES.append(dict(t=1.,what='am start RobotCam',recovered=True))
        return dict(session='launch-retry',attempts=1)
    with patch.object(rt,'camera_start',side_effect=recovered_launch):session.camera(True)
    assert data['camera_failures']==3
    rt.READ_RETRIES.pop()
    session.beat=-100
    session.watchdog_armed=True
    with patch.object(q.os,'kill') as kill:
        try:session.shared()
        except RuntimeError as e:assert 'heartbeat' in str(e)
        else:raise AssertionError('no heartbeat failure')
        kill.assert_called_once_with(q.os.getpid(),signal.SIGTERM)
    session.data['battery_stop']='battery 24% below 25%'
    session.data.update(fast=[],dumps=[])
    with patch.object(rt,'check_cores'),patch.object(cr,'block_limit',return_value=None):
        try:session.guard()
        except diag.BatteryStop:pass
        else:raise AssertionError('battery stop became monitor error')
    del session.data['battery_stop']
    with patch.object(rt,'memory_sample',return_value=dict(root_rc=1,battery_status='Discharging',pss_error='su: denied')):
        try:session.memory()
        except RuntimeError:pass
        else:raise AssertionError('bad nonempty memory accepted')
    print('PASS LiveSession camera state serialized with PSS read, startup failure counted, heartbeat SIGTERM and memory failure propagate')


def check_readers():
    # Shared parser/re-read modules remain byte-identical to FIX4D; no reader is introduced in phase23.
    for name in ('runtime.py','power.py','diagnostics.py'):
        base=subprocess.check_output(['git','show',BASE+':benchmark/campaign/'+name])
        assert base==(rt.HERE/name).read_bytes(),name
    def functions(source):return {n.name:ast.dump(n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    base=functions(subprocess.check_output(['git','show',BASE+':benchmark/campaign/phase1.py']).decode())
    current=functions((rt.HERE/'phase1.py').read_text())
    for name in ('LiveOps','bounded_selector','preflight','memory_row_bad','lmk_window'):assert base[name]==current[name],name
    source=ast.parse((rt.HERE/'phase23.py').read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('root','root_sample','fast_sample','alive','reread','retry') for n in ast.walk(source))
    c.check_readers();d.check_su_readers();d.check_rootshell_readers();d.check_launch_only();d.check_probes()
    print('PASS FIX4D sweep extended to P23: every hardware reader reused unchanged; stdout AND su stderr rule, launch-only, HTTP timeout, /health 2-of-2, FIX3D thermal rule; no new transport/parser/re-read call')


def check_cli():
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        output=Path(tmp)/'r.json'
        # Redirections already exist at program entry, as in RUN.md.
        output.with_suffix('.stdout').write_text('');output.with_suffix('.stderr').write_text('')
        q.main(['--dry-run','--mock','--output',str(output)])
        data=json.loads(output.read_text())
        assert data['label']==q.MOCK_LABEL and data['mock_only'] and not data['rehearsal_coverage']['rehearsal_pass']
        assert data['rehearsal_coverage']['modes']==list(q.MODES)
        assert data['rehearsal_coverage']['fixed_pauses']==3 and data['rehearsal_coverage']['adaptive_high_switch'] and data['rehearsal_coverage']['adaptive_low_restart']
        try:q.require_rehearsal(output)
        except RuntimeError:pass
        else:raise AssertionError('mock accepted as live rehearsal')
        live=copy.deepcopy(data);live['mock_only']=False;live['rehearsal_coverage']['rehearsal_pass']=True
        cov=q.coverage(live);assert cov['rehearsal_pass']
        late=copy.deepcopy(live);late['samples']['monitor_errors'].append('late reader failure')
        assert not q.coverage(late)['rehearsal_pass']
        restart=copy.deepcopy(live);restart['runs'][2]['totals']['camera_restarts']=0
        assert not q.coverage(restart)['rehearsal_pass']
        failed=copy.deepcopy(live);failed['runs'][1]['totals']['camera_failures']=1
        assert not q.coverage(failed)['rehearsal_pass']
        output.write_text(json.dumps(live));q.require_rehearsal(output)
        live['code_hashes']['benchmark/campaign/phase23.py']='old';output.write_text(json.dumps(live))
        try:q.require_rehearsal(output)
        except RuntimeError:pass
        else:raise AssertionError('old code accepted')
        try:q.main(['--dry-run','--mock','--output',str(output)])
        except SystemExit:pass
        else:raise AssertionError('old stem accepted')
    print('PASS CLI mock dry run all three modes, adaptive high/low switch and pause restart; mock NOT VALID, mandatory live/current-code rehearsal proof, redirection files allowed and evidence overwrite refused')
    class LowBattery(q.MockSession):
        def guard(self):
            if self.camera_on and self.temp>31.5:raise diag.BatteryStop('battery 24% below 25%')
            super().guard()
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()),patch.object(q,'MockSession',LowBattery):
        output=Path(tmp)/'battery.json';q.main(['--dry-run','--mock','--output',str(output)])
        data=json.loads(output.read_text())
        assert 'below 25%' in data['error'] and not data['samples']['monitor_errors']
        assert data['runs'][0]['stop_reason']=='BATTERY BELOW 25%' and not data['rehearsal_coverage']['rehearsal_pass']
    class LateBattery(q.MockSession):
        def close(self):
            super().close();self.data['battery_stop']='battery 24% below 25%'
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()),patch.object(q,'MockSession',LateBattery):
        output=Path(tmp)/'late_battery.json';q.main(['--dry-run','--mock','--output',str(output)])
        data=json.loads(output.read_text())
        assert 'below 25%' in data['error'] and not data['samples']['monitor_errors']
        assert not data['rehearsal_coverage']['rehearsal_pass'] and all('BATTERY BELOW 25%' in r['result_validity'] for r in data['runs'])


def check_round2():
    required={'server_manager.py','benchmark/relate_anything/speed1/session.py',
        'benchmark/strategic_selector/ladder/measure.py','benchmark/strategic_selector/ladder/cases/ladder_cases_v1.jsonl'}
    assert required<=q.code_hashes().keys() and len(q.code_hashes())==18
    class BracketEmergency(q.MockSession):
        def bracket(self,end):
            if self.camera_on:raise RuntimeError('EMERGENCY: battery 45 C')
            super().bracket(end)
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        session=BracketEmergency(None,{});session.start();out=result()
        q.run_mode('endurance',session,out,q.SHORT,37.,lambda suffix:Path(tmp)/('run'+suffix))
        row=out['runs'][0]
        assert row['stop_reason'].startswith('EMERGENCY: evidence bracket')
        assert row['events'][-1]['reason']==row['stop_reason'] and row['checks_validity'].startswith('NOT VALID')
        checkpoints=[json.loads((Path(tmp)/name).read_text()) for name in row['sample_checkpoints']]
        assert any(c['completed_work'] and any(w['yolo'] for w in c['completed_work']) for c in checkpoints)
    class NearDeadline(q.MockSession):
        def camera(self,on):
            if on:self.active_start=self.now();self.deadline=self.now()+180
            return super().camera(on)
        def skin(self):
            if self.camera_on:return 37. if self.now()-self.active_start>=60 else 30.
            return 34.9 if hasattr(self,'deadline') and self.now()>=self.deadline-60.5 else 36.
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        session=NearDeadline(None,{});session.start();out=result()
        q.run_mode('adaptive',session,out,dict(q.SHORT,idle=0,gate=0,adaptive=180),37.,lambda suffix:Path(tmp)/('run'+suffix))
        row=out['runs'][0]
        assert len(row['active_phases'])==1 and row['checks_validity']=='VALID'
    # Drain a delayed selector on the main thread without losing heartbeat or BatteryStop type.
    original_wait=threading.Event.wait
    for battery in (False,True):
        ops=FastOps('L2');start=p.time.monotonic();ops.origin=0.;ops.now=lambda:(p.time.monotonic()-start)*10
        beats=[];ops.drain_heartbeat=lambda:beats.append(threading.get_native_id())
        def select(index,text):
            p.time.sleep(.4)
            if battery:raise diag.BatteryStop('battery 24% below 25%')
            return dict(ms=4000.,miss=False)
        ops.select=select
        def fast_wait(event,timeout=None):return original_wait(event,None if timeout is None else max(0,timeout/10))
        with patch.object(threading.Event,'wait',fast_wait):
            try:p.run_cycle('L2',60,ops,{},replacement=True,stop_when=lambda:ops.now()>=6.)
            except diag.BatteryStop:assert battery
            else:assert not battery
        assert beats and set(beats)=={threading.get_native_id()}
    print('PASS R2: complete-slot mock/live counts,18 proof hashes,bracket emergency stop reason,incremental work checkpoints,recovered launch failure count,adaptive restart reserve,main-thread drain heartbeat,BatteryStop worker type')


if __name__=='__main__':
    check_gate();check_replacement();check_rules();check_controller();check_windows();check_m3_timing();check_fixed_rehearsal();check_live();check_cli();check_readers();check_review_regressions();check_round2()
    print('PASS CAMPAIGN_P23 self-check (offline mocks only; agents resident; proot; NOT VALID for timing)')
