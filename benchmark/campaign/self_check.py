"""Offline mocks only. No root, real camera/server/inference, idle or timed benchmark."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch, MagicMock

import power
import phase1 as p
import runtime as rt
import lag_probe as lp


def check_power():
    assert all(abs(k*power.PERIOD-round(k*power.PERIOD/5)*5)>.001 for k in range(1,100))
    rows=[dict(t=t,t_start=t-.02,battery_w=w) for t,w in [(-1,0),(0,1),(1,3),(2,5),(3,7)]]
    s=power.weighted(rows,0,2); assert s['mean_battery_w']==3 and s['covered_s']==2
    assert power.weighted(rows,.5,1.5)['mean_battery_w']==3
    assert power.weighted(rows,0,4)['mean_battery_w'] is None
    assert power.weighted([rows[1],rows[3]],0,2)['mean_battery_w'] is None
    assert power.weighted([],0,1)['mean_battery_w'] is None
    assert power.summary(rows,[dict(started_s=0,ended_s=1)],2)['inside_samples']==1
    for allowed,measured in [({4,5},{4,5}),(set(),{4,5})]:
        try:power.safe_mask(allowed,measured)
        except RuntimeError:pass
        else:raise AssertionError('unsafe mask accepted')
    assert power.safe_mask(range(8),{0,1,2,3,4,5})=={6,7}
    shell=NS(run=lambda script:['-1000000','4000000','Discharging','-900000'])
    with patch.object(power.time,'monotonic',side_effect=[10.,10.1]):
        row=power.sample(shell,'/fake',9.,True)
    assert row['t_start']==1 and abs(row['t']-1.1)<1e-6 and row['battery_w']==4 and row['current_avg_uA']==-900000
    for fields in ([],['0','0','Discharging'],['-1','4000000','Charging'],['bad','4000000','Discharging']):
        try:power.sample(NS(run=lambda script:fields),'/fake',0)
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('bad battery accepted')
    stop=threading.Event(); reads=[]; errors=[]; stamps=iter([100.,100.01,100.02,100.03])
    shell=NS(run=lambda script:['-1','4000000','Discharging'])
    def verify():stop.set()
    # One final end bracket even when stopped; no foreign affinity syscall.
    with patch.object(power.os,'sched_setaffinity'),patch.object(power.os,'sched_getaffinity',return_value={0}),patch.object(power.time,'monotonic',side_effect=lambda:next(stamps)):
        stop.set();power.sampler(shell,'/fake',100.,{0},{4,5},stop,reads,errors,verify)
    assert len(reads)==1 and not errors
    stop.clear(); errors.clear()
    with patch.object(power.os,'sched_setaffinity'),patch.object(power.os,'sched_getaffinity',return_value={4}):
        power.sampler(shell,'/fake',0,{0},{4,5},stop,[],errors,lambda:None)
    assert errors and stop.is_set()
    # Lag known ramp: baseline 100, plateau 200, rise/fall in 2 s.
    rows=[]
    for i in range(501):
        t=i/10
        val=100 if t<20 else 100+50*(t-20) if t<22 else 200 if t<30 else 200-50*(t-30) if t<32 else 100
        rows.append(dict(t=t,current_now_uA=-val,current_avg_uA=-val))
    got=power.lag_response(rows,20,30)
    for k,want in [('rise_50_s',1),('rise_90_s',1.8),('fall_50_s',1),('fall_10_s',1.8)]:assert abs(got[k]-want)<1e-6,(k,got)
    assert power.lag_response([],20,30)['rise_50_s'] is None
    assert power.lag_response([dict(t=t,current_now_uA=-100) for t in range(51)],20,30)['step_uA'] is None
    print('PASS sampler masks, de-phasing, timestamps, weighting/gaps/boundaries, battery refusals, lag crossings')


class FastOps(p.MockOps):
    def __init__(self,name):
        super().__init__(name); self.base=time.monotonic(); self.events=[]
    def now(self):return (time.monotonic()-self.base)*1000
    def detect(self,image,size):
        self.events.append(('yolo',size,self.now()))
        return [{'class_name':'person','box_xyxy':[0,0,1,1]}, {'class_name':'chair','box_xyxy':[1,1,2,2]}]
    def relate(self,image,boxes):
        self.events.append(('m2_start',self.now()))
        if self.name=='L3':time.sleep(.002)
        self.events.append(('m2_end',self.now()))
        return [],1
    def select(self,index,text):
        self.events.append(('select',self.now()))
        return dict(ms=1,prompt_tokens=1,miss=False,timeout=False)


# A virtual clock cooperates with real threads and Event.wait: patch wait to advance wall deadlines rapidly.
def check_cycle():
    assert p.blocks(None)==p.DEFAULT and p.blocks('L4,L0')==['L4','L0']
    for bad in ('','L5','L0,'):
        try:p.blocks(bad)
        except ValueError:pass
        else:raise AssertionError('bad layout')
    assert p.due_frames(3.2,0,180)==([0,1,2],3)
    assert p.due_frames(180,179,180)==([179],180)
    r=[dict(subject='person',subject_idx=0,predicate='next_to',object='chair',object_idx=1,score=.9,threshold=.5,**{'pass':True})]*7
    assert p.context(r).count('- person')==5 and p.context([])=='M2 relations:\n- none above threshold.'
    r[0]=dict(r[0],score=.1);assert p.context(r).count('- person')==5
    assert p.can_continue(dict(failure_kind='cadence'))
    for kind in ('shared','root','thermal','camera','inference','charger'):
        assert not p.can_continue(dict(failure_kind=kind))
    assert not p.can_continue(dict(failure_kind='cadence',cleanup_errors=['bad']))
    assert not p.can_continue(dict(failure_kind='cadence',monitor_errors=['bad']))
    # 21 actual milliseconds model a full 180 s; a real thread Event supplies ordering.
    original_wait=threading.Event.wait
    def fast_wait(event,timeout=None):return original_wait(event,None if timeout is None else max(0,timeout/1000))
    for name in p.LAYOUTS:
        ops=FastOps(name);rec={}
        with patch.object(threading.Event,'wait',fast_wait):p.run_cycle(name,180,ops,rec)
        assert rec['summary']['yolo']['320']['done']+rec['summary']['yolo']['320']['skipped']==180
        assert rec['summary']['yolo']['640']['due']==36
        if name=='L0':assert not rec['m2'] and all(r['context'] is None for r in rec['selector'])
        if name=='L3':assert rec['frames_skipped']['320']>0
        if name=='L2':
            kinds=[e[0] for e in ops.events if e[0].startswith('m2') or e[0]=='select']
            assert kinds and all(kinds[i:i+3]==['m2_start','m2_end','select'] for i in range(0,len(kinds),3)),kinds
        assert all('context' in r for r in rec['selector'])
    ops=FastOps('L0');ops.frame=lambda last:dict(status='old')
    with patch.object(threading.Event,'wait',fast_wait):
        try:p.run_cycle('L0',180,ops,{})
        except RuntimeError:pass
        else:raise AssertionError('camera refusal missing')
    ops=FastOps('L4');ops.relate=lambda *a:(_ for _ in ()).throw(ValueError('inference'))
    with patch.object(threading.Event,'wait',fast_wait):
        try:p.run_cycle('L4',180,ops,{})
        except RuntimeError:pass
        else:raise AssertionError('inference refusal missing')
    print('PASS full-layout scheduling, L3 skips/no queue, L2 M2-before-selector, context, continuation/camera/inference refusals')


def check_runtime():
    shell=NS(p=NS(pid=10),monitor_tids={11},close=MagicMock(),run=MagicMock())
    sb=NS(require_monitor_mask=lambda mask,measured:power.safe_mask(mask,measured),root_mask=lambda shell,pid:{0},
          prepare_monitor=lambda measured:(shell,{0},12,{'policies':{'policy0':1,'policy4':1,'policy6':1},'cpu_zones':{'BIG':'9','MID':'10','LITTLE':'11'}}))
    with patch.object(rt.os,'sched_getaffinity',return_value={0}):rt.verify_monitor(sb,sb.prepare_monitor({4,5}),{4,5})
    with patch.object(rt.os,'sched_getaffinity',return_value={4}):
        try:rt.verify_monitor(sb,sb.prepare_monitor({4,5}),{4,5})
        except RuntimeError:pass
        else:raise AssertionError('pump affinity accepted')
    sb.root_mask=lambda shell,pid:{4}
    try:rt.verify_monitor(sb,sb.prepare_monitor({4,5}),{4,5})
    except RuntimeError:pass
    else:raise AssertionError('root affinity accepted')
    for rc in (0,1):
        with patch.object(rt.subprocess,'run',return_value=NS(returncode=rc,stdout='',stderr='')):rt.clear_processes()
    for output in ('99999 codex','99999 main.py','99999 llama-server','99999 phase1.py'):
        with patch.object(rt.subprocess,'run',return_value=NS(returncode=0,stdout=output,stderr='')):
            try:rt.clear_processes()
            except RuntimeError:pass
            else:raise AssertionError('resident accepted')
    with patch.object(rt.subprocess,'run',return_value=NS(returncode=0,stdout='99999 llama-server',stderr='')):rt.clear_processes(99999)
    cr=NS(read_dump=lambda:dict(t=0,skin=30,status=0),SKIN_GATE_C=1.5,STATUS_STOP=4)
    assert not rt.skin_gate(cr,dict(skin=30),'L0',lambda:None)['warm_start']
    cr.read_dump=lambda:dict(t=0,skin=40,status=0)
    with patch.object(rt.time,'monotonic',side_effect=[0,481]):assert rt.skin_gate(cr,dict(skin=30),'L0',lambda:None)['warm_start']
    for skin,status in ((None,0),(30,None),(30,4)):
        cr.read_dump=lambda:dict(skin=skin,status=status);cr.STATUS_STOP=4
        try:rt.dump_check(cr)
        except RuntimeError:pass
        else:raise AssertionError('thermal missing accepted')
    cr.root=lambda *a:(0,'versionCode=42 versionName=1.2')
    assert rt.camera_version(cr)==dict(versionCode='42',versionName='1.2')
    cr.root=lambda *a:(0,'no version')
    try:rt.camera_version(cr)
    except RuntimeError:pass
    else:raise AssertionError('missing installed version accepted')
    print('PASS process/agent/foreign-server refusals, root/pump masks, gates/warm start, thermal/version refusals')


def check_preflight():
    # Real preflight control flow; hardware calls mocked, refusals before idle or gates.
    v,sb,pm,cr,_=rt.imports()
    for refusal in (None,'charger','agents','root','cores','hashes','camera','server','policies'):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            out=Path(tmp)/'preflight.json'
            bad=lambda name:RuntimeError(name) if refusal==name else None
            monitor=(NS(close=lambda:None),{0},10,{'policies':{'policy0':1,'policy4':1,'policy6':1},'cpu_zones':{'BIG':'9','MID':'10','LITTLE':'11'}})
            mocks=[patch.object(rt,'clear_processes',side_effect=bad('agents')),
                   patch.object(cr,'require_native'),patch.object(cr,'check_cores',side_effect=bad('cores')),
                   patch.object(sb,'battery_sample',side_effect=bad('charger')),
                   patch.object(rt,'require_hashes',side_effect=bad('hashes'),return_value={}),
                   patch.object(rt,'camera_version',return_value={}),patch.object(rt,'dump_check',return_value={'skin':30,'status':0}),
                   patch.object(sb,'Screen'),patch.object(rt,'prepare',side_effect=lambda sb_arg,measured: (_ for _ in ()).throw(bad('root') or bad('policies')) if bad('root') or bad('policies') else (monitor[0],set(range(8))-measured,monitor[2],monitor[3])),
                   patch.object(pm,'build_detector',return_value=(None,{'worker_tids':[],'caller_tid':1})),
                   patch.object(sb,'check_pinning'),patch.object(v,'build',return_value=(None,None,[])),
                   patch.object(cr,'Server'),patch.object(p,'bounded_selector'),patch.object(cr,'load_cases',return_value=[{}]),
                   patch.object(pm,'camera_start',side_effect=bad('camera'),return_value={'session':'s'}),
                   patch.object(cr,'read_frame',return_value={'status':'ok'}),patch.object(rt,'stop_camera'),
                   patch.object(p.time,'sleep'),patch.object(p,'live_block'),patch.object(rt,'skin_gate')]
            handles=[stack.enter_context(m) for m in mocks]
            stack.enter_context(patch.object(rt,'fast_check',return_value={}))
            handles[12].return_value.alive.return_value=refusal!='server'
            handles[12].return_value.cmd=['mock-server'];handles[12].return_value.load_s=0;handles[12].return_value.proc.pid=123
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:p.main(['--preflight','--output',str(out)])
            except RuntimeError:assert refusal
            else:assert refusal is None
            handles[-3].assert_not_called();handles[-2].assert_not_called();handles[-1].assert_not_called()
            result=json.loads(out.read_text())
            assert not result['blocks']
            if not refusal:assert result['label']=='PREFLIGHT ONLY — NO TIMING' and len(result['layout_checks'])==5
    print('PASS real preflight path all layouts, charger/agents/root/cores/hashes/policies/camera/server refusals; no idle/gates/blocks')



class InlineThread:
    def __init__(self,target,args=(),**kw):self.target,self.args=target,args
    def start(self):self.target(*self.args)
    def join(self,timeout=None):pass
    def is_alive(self):return False


def check_lag_lifecycle():
    v,sb,pm,cr,_=rt.imports()
    for fail in (None,'sampler','inference','cleanup'):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            clock=[100.]
            stack.enter_context(patch.object(lp.time,'monotonic',side_effect=lambda:clock[0]))
            stack.enter_context(patch.object(lp.time,'sleep',side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)))
            shell=NS(close=MagicMock())
            monitor=(shell,{0,1,2,3},10,{'policies':{},'cpu_zones':{'BIG':'9','MID':'10','LITTLE':'11'}})
            caller=NS(close=MagicMock())
            def detect(*a):
                clock[0]+=.6
                if fail=='inference':raise RuntimeError('inference')
                return [],600.
            caller.detect=detect
            def sampler(shell,battery,t0,mask,measured,stop,rows,errors,verify,*args):
                assert mask=={0,1,2,3} and measured=={6,7} and args==(0,True)
                if fail=='sampler':errors.append('root failed');return
                rows.extend(dict(t=t,t_start=t-.01,current_now_uA=-100 if t<20 or t>32 else -200,
                                 current_avg_uA=None,battery_w=1,voltage_now_uV=4000000) for t in range(52))
            for m in (patch.object(cr,'require_native'),patch.object(rt,'clear_processes'),patch.object(cr,'check_cores'),
                      patch.object(sb,'battery_sample'),patch.object(rt,'require_hashes',return_value={}),
                      patch.object(rt,'dump_check'),patch.object(sb,'load_speed_input',return_value=({'detections':[{},{}]},None,None)),
                      patch.object(sb,'Screen'),patch.object(rt,'stop_camera'),patch.object(rt,'prepare',return_value=monitor),
                      patch.object(v,'build',return_value=(None,caller,[])),patch.object(sb,'check_pinning'),
                      patch.object(power,'sampler',side_effect=sampler),patch.object(lp.threading,'Thread',InlineThread),
                      patch.object(cr,'monitor_loop',side_effect=lambda period,read,rows,stop:rows.append(read())),
                      patch.object(cr,'fast_sample',return_value={'t':100.,'bat_c':30,'cpu_c':50}),
                      patch.object(cr,'read_dump',return_value={'t':100.,'skin':30,'status':0}),patch.object(cr,'block_limit',return_value=None)):
                stack.enter_context(m)
            if fail=='cleanup':caller.close.side_effect=RuntimeError('cleanup')
            output=Path(tmp)/'lag.json'
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:lp.main(['--output',str(output)])
            except (RuntimeError,SystemExit):assert fail
            else:assert not fail
            result=json.loads(output.read_text())
            assert result['label']==lp.LABEL
            assert result['complete']==(fail is None)
            if not fail:
                assert 10<=result['load_end_s']-result['load_start_s']<11 and result['achieved_hz']>0
                assert result['step_response']['current_avg_uA']['step_uA'] is None
            caller.close.assert_called_once()
    print('PASS lag probe full quiet/load/quiet lifecycle, BIG/0-3 masks, current_avg absent, sampler/inference/cleanup failure (mocks)')


def check_live_block():
    v,sb,pm,cr,_=rt.imports()
    for fail in (None,'camera','inference','cleanup','powergap'):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            shell=NS(close=MagicMock());monitor=(shell,{0},10,{'policies':{'policy0':1,'policy4':1,'policy6':1},'cpu_zones':{'BIG':'9','MID':'10','LITTLE':'11'}})
            detector=NS(close=MagicMock());caller=NS(close=MagicMock());server=NS(proc=NS(pid=123),alive=lambda:True)
            if fail=='cleanup':caller.close.side_effect=RuntimeError('cleanup')
            for m in (patch.object(rt,'prepare',return_value=monitor),patch.object(pm,'build_detector',return_value=(detector,{'worker_tids':[],'caller_tid':1})),
                      patch.object(v,'build',return_value=(None,caller,[])),patch.object(rt,'clear_processes'),patch.object(cr,'check_cores'),
                      patch.object(rt,'skin_gate',return_value={'skin':30,'status':0,'warm_start':False}),
                      patch.object(pm,'camera_start',side_effect=RuntimeError('camera') if fail=='camera' else None,return_value={'session':'s'}),
                      patch.object(rt,'dump_check',return_value={'skin':30,'status':0}),patch.object(p,'bounded_selector'),patch.object(cr,'load_cases',return_value=[{}]),
                      patch.object(power,'sample',return_value={'t':0,'t_start':0,'battery_w':1}),patch.object(p.threading,'Thread',return_value=NS(start=lambda:None,join=lambda timeout=None:None,is_alive=lambda:False)),
                      patch.object(rt,'stop_camera'),patch.object(cr,'lmk_lines',return_value={'ok':True,'n_kills':0}),patch.object(cr,'block_limit',return_value=None),
                      patch.object(pm,'capped_by_policy',return_value={})):
                stack.enter_context(m)
            def cycle(name,duration,ops,record):
                if fail=='inference':raise RuntimeError('inference')
                origin=record['cycle_origin']
                record.update(duration_s=180,summary={},m2=[],selector=[],yolo=[],reads=[])
                record['power']=[{'t':-1,'t_start':-1,'battery_w':1},{'t':181,'t_start':181,'battery_w':1}] if fail=='powergap' else [dict(t=i,t_start=i-.01,battery_w=1) for i in range(-1,182)]
                record['fast']=[{'t':origin,'max':{}}];record['dumps']=[dict(t=origin,skin=30,status=0)]
                record['memory']=[dict(t=origin,mem_available_mib=1000,root_rc=0,pss_kb={'llama_server':100},battery_status='Discharging')]
            stack.enter_context(patch.object(p,'run_cycle',side_effect=cycle))
            output=Path(tmp)/'live.json'
            result=p.live_block('L4',output,server,{'skin':30})
            assert result['validity']==('VALID' if fail is None else 'NOT VALID — INCOMPLETE'),(fail,result)
            assert result.get('failure_kind') is None if fail is None else result['failure_kind']=='shared'
            detector.close.assert_called_once();caller.close.assert_called_once()
            assert output.exists()
    print('PASS live block lifecycle/wiring, camera/inference/cleanup/power coverage failures, boundary accounting and file capture (all mocks)')



def check_live_adapters():
    pins=[]
    sb=NS(check_pinning=lambda tids,cpus:pins.append((tids,cpus)))
    cr=NS(FRAME_DIR='/fake')
    detector=NS(detect=lambda image,size:[{'size':size}])
    caller=NS(detect=lambda image,boxes:([],12.))
    cases=[dict(id='a',situation={'robot':'idle'},options={'wait':''},instruction='pick',answer='wait')]
    captured=[]
    cr.select=lambda sel,case:(captured.append(case) or dict(ms=1,correct=True,prompt_tokens=12))
    ops=p.LiveOps(cr,sb,detector,{'worker_tids':[1],'caller_tid':2},caller,[3],{6,7}, {'session':'s'},None,None,cases,lambda:None)
    assert ops.detect(None,320)==[{'size':320}] and pins[-1]==([1,2],{4,5})
    assert ops.relate(None,[{}])==([],0.)
    assert ops.relate(None,[{},{}])==([],12.)
    try:ops.relate(None,[{}]*33)
    except RuntimeError:pass
    else:raise AssertionError('33 boxes accepted')
    text='M2 relations:\n- person [0] next to chair [1].'
    assert 'correct' not in ops.select(0,text)
    assert captured[-1]['situation']==json.dumps(cases[0]['situation'],ensure_ascii=False,indent=1)+'\n\n'+text
    assert cases[0]['situation']=={'robot':'idle'}
    ops.select(0,None);assert captured[-1]['situation']==cases[0]['situation']
    cr.read_frame=MagicMock(side_effect=[dict(status='ok',frame=1),dict(status='ok',frame=2)])
    with patch.object(p.time,'monotonic',side_effect=[0.,0.1]),patch.object(p.time,'sleep'):
        assert ops.frame(1)['frame']==2
    cr.read_frame=MagicMock(return_value=dict(status='ok',frame=1))
    with patch.object(p.time,'monotonic',side_effect=[0.,2.]):assert ops.frame(1)['frame']==1
    cr.read_frame=lambda *a,**k:dict(status='old');assert ops.frame(1)['status']=='old'
    # Bounded S1O transport: the existing prompt/scoring object, fixed 15 s per request.
    selector=NS(url='http://fake')
    cr.make_selector=lambda:selector
    response=MagicMock();response.__enter__.return_value.read.return_value=b'{"ok":true}'
    with patch('urllib.request.urlopen',return_value=response) as request:
        bound=p.bounded_selector(cr);assert bound._post('/completion',{'prompt':[1]})=={'ok':True}
        assert request.call_args.kwargs['timeout']==15
    with patch('urllib.request.urlopen',side_effect=TimeoutError('timeout')):
        try:bound._post('/completion',{})
        except TimeoutError:pass
        else:raise AssertionError('transport timeout swallowed')
    print('PASS LiveOps frame retry/repeated/old, YOLO/M2 pins/box limits, actual context append/L0, bounded selector transport/timeout')


def check_extra_scheduling():
    original_wait=threading.Event.wait
    def fast_wait(event,timeout=None):return original_wait(event,None if timeout is None else max(0,timeout/1000))
    for name in ('L1','L4'):
        ops=FastOps(name)
        def slow_relate(*a):time.sleep(.008);return [],1
        ops.relate=slow_relate
        record={}
        with patch.object(threading.Event,'wait',fast_wait):p.run_cycle(name,180,ops,record)
        assert record['cadence_missed'] and len(record['m2'])<36
    for error in (TimeoutError('timeout'),OSError('socket')):
        ops=FastOps('L0');record={}
        ops.select=lambda *a:(_ for _ in ()).throw(error)
        with patch.object(threading.Event,'wait',fast_wait):
            try:p.run_cycle('L0',180,ops,record)
            except RuntimeError:pass
            else:raise AssertionError('selector error not fatal')
        assert record['selector'][0]['miss'] and record['selector'][0]['timeout']==isinstance(error,TimeoutError)
    ops=FastOps('L0');record={}
    def delayed(*a):time.sleep(.022);return dict(ms=22,prompt_tokens=1,miss=False,timeout=False)
    ops.select=delayed
    with patch.object(threading.Event,'wait',fast_wait):p.run_cycle('L0',180,ops,record)
    assert record['summary']['selector']['misses']>0
    print('PASS L1/L4 busy-M2 skip branch, selector timeout/OSError refusal, successful late-selector miss counts')


def check_root_adapters():
    commands=[];mask_state={11:{0,1,2,3},10:{0,1,2,3},12:{0,1,2,3}}
    shell=NS(p=NS(pid=10),monitor_tids={11},close=MagicMock())
    def run(command):
        commands.append(command)
        if command.startswith('taskset -p'):
            _,_,bits,pid=command.split();mask_state[int(pid)]={i for i in range(8) if int(bits,16)&(1<<i)}
        return []
    shell.run=run
    sb=NS(require_monitor_mask=lambda mask,measured:power.safe_mask(mask,measured),
          prepare_monitor=lambda measured:(shell,{0,1,2,3},12,{}),root_mask=lambda shell,pid:mask_state[pid])
    with patch.object(rt.os,'sched_setaffinity',side_effect=lambda tid,mask:mask_state.__setitem__(tid,set(mask))),patch.object(rt.os,'sched_getaffinity',side_effect=lambda tid:mask_state[tid]):
        monitor=rt.prepare(sb,{6,7},restrict={0,1})
        assert monitor[1]=={0,1} and mask_state[10]=={0,1} and mask_state[12]=={0,1}
        try:rt.prepare(sb,{6,7},restrict={4})
        except RuntimeError:pass
        else:raise AssertionError('empty restrict accepted')
    shell.close.assert_called_once()
    root_calls=[]
    original=lambda command,tag,timeout=60:(root_calls.append((command,tag,timeout)) or (0,'ok'))
    cr=NS(root=original)
    with rt.root_mask_scope(cr,sb,{0,1},{4,5,6,7}):
        assert cr.root('dumpsys thermalservice','test',timeout=2)==(0,'ok')
        assert 'taskset -p 3 $$' in root_calls[-1][0] and 'exit 98' in root_calls[-1][0]
    assert cr.root is original
    cr.root=lambda *a,**k:(98,'')
    with rt.root_mask_scope(cr,sb,{0},{4,5}):
        try:cr.root('anything','test')
        except RuntimeError:pass
        else:raise AssertionError('transient root readback failure accepted')
    # Periodic sampler runs four de-phased reads plus the final receipt bracket.
    clock=[0.];rows=[];errors=[]
    class Stop:
        done=False;waits=0
        def is_set(self):return self.done
        def set(self):self.done=True
        def wait(self,dt):
            clock[0]+=dt;self.waits+=1
            if self.waits==4:self.done=True
            return self.done
    stop=Stop()
    def sampling(shell,battery,t0,average=False):
        start=clock[0];clock[0]+=.01
        return dict(t_start=start,t=clock[0],battery_w=1)
    with patch.object(power.time,'monotonic',side_effect=lambda:clock[0]),patch.object(power.os,'sched_setaffinity'),patch.object(power.os,'sched_getaffinity',return_value={0}),patch.object(power,'sample',side_effect=sampling):
        power.sampler(None,'/fake',0,{0},{4,5},stop,rows,errors,lambda:None)
    assert len(rows)==5 and not errors
    assert all(abs((b['t']-a['t'])-.37)<1e-6 for a,b in zip(rows[:3],rows[1:4]))
    # Expanded bare-filename process refusal coverage.
    for name in ('power_map.py','coresidency.py','thermal_char.py','duty_cycle.py','camera_power.py','speed_block.py','session.py'):
        with patch.object(rt.subprocess,'run',return_value=NS(returncode=0,stdout='99999 python '+name,stderr='')) as command:
            try:rt.clear_processes()
            except RuntimeError:pass
            else:raise AssertionError('bare runner accepted')
            assert __import__('re').search(command.call_args.args[0][-1],name)
    print('PASS restrict repinning/empty masks, transient-root pin/readback/restore, periodic sampler cadence/final bracket, bare-runner refusals')


def check_session_transitions():
    v,sb,pm,cr,_=rt.imports()
    for failure in ('shared','cadence','warm'):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            server=NS(proc=NS(pid=123),alive=lambda:True,stop=MagicMock())
            def preflight(names,out,result,resources):resources['server']=server
            calls=[]
            def block(name,path,server,idle):
                calls.append(name)
                kind='shared' if failure=='shared' and name=='L1' else 'cadence' if failure=='cadence' and name=='L0' else None
                record=dict(block=name,validity='NOT VALID — WARM START' if failure=='warm' else 'VALID',failure_kind=kind)
                p.write(path,record);return record
            sleep=stack.enter_context(patch.object(p.time,'sleep'))
            for m in (patch.object(p,'preflight',side_effect=preflight),patch.object(p,'live_block',side_effect=block),
                      patch.object(rt,'clear_processes'),patch.object(sb,'battery_sample'),patch.object(cr,'check_cores'),
                      patch.object(rt,'dump_check',return_value={'skin':30,'status':0}),contextlib.redirect_stdout(io.StringIO())):
                stack.enter_context(m)
            output=Path(tmp)/'session.json'
            try:p.main(['--blocks','L0,L1,L2','--output',str(output)])
            except RuntimeError:assert failure=='shared'
            else:assert failure!='shared'
            result=json.loads(output.read_text())
            assert sleep.call_count==60 and all(c.args==(5,) for c in sleep.call_args_list)
            assert calls==(['L0','L1'] if failure=='shared' else ['L0','L1','L2'])
            assert result['unrun_blocks']==(['L2'] if failure=='shared' else [])
            server.stop.assert_called_once()
    print('PASS real session control flow: one mocked 300s idle, cadence continue, warm continue, shared stop/unrun capture, server cleanup')


def check_monitor_workers():
    v,sb,pm,cr,_=rt.imports()
    for bad_mask in (False,True):
        pending=[];waits=[]
        class Stop:
            done=False
            def set(self):self.done=True
            def is_set(self):return self.done
            def wait(self,delay):waits.append(delay);return self.done
        class Deferred:
            def __init__(self,target,args=(),**kw):self.target,self.args=target,args
            def start(self):pending.append(self)
            def join(self,timeout=None):pass
            def is_alive(self):return False
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            shell=NS(close=lambda:None);monitor=(shell,{0},10,{'policies':{'policy0':1,'policy4':1,'policy6':1}})
            for m in (patch.object(rt,'prepare',return_value=monitor),patch.object(pm,'build_detector',return_value=(NS(close=lambda:None),{'worker_tids':[],'caller_tid':1})),
                      patch.object(v,'build',return_value=(None,NS(close=lambda:None),[])),patch.object(rt,'clear_processes'),patch.object(cr,'check_cores'),
                      patch.object(rt,'skin_gate',return_value={'warm_start':False}),patch.object(pm,'camera_start',return_value={'session':'s'}),patch.object(rt,'dump_check',return_value={'skin':30,'status':0}),
                      patch.object(p,'bounded_selector'),patch.object(cr,'load_cases',return_value=[{}]),patch.object(power,'sample',return_value={'t':0,'t_start':0,'battery_w':1}),
                      patch.object(p.threading,'Thread',Deferred),patch.object(p.threading,'Event',Stop),patch.object(rt,'stop_camera'),patch.object(cr,'lmk_lines',return_value={'ok':True}),
                      patch.object(cr,'block_limit',return_value=None),patch.object(rt,'fast_check',return_value={'t':time.monotonic(),'t_start':time.monotonic(),'max':{}}),
                      patch.object(cr,'read_dump',return_value={'t':time.monotonic(),'t_start':time.monotonic(),'skin':30,'status':0}),
                      patch.object(cr,'root_sample',return_value={'root_rc':0,'battery_status':'Discharging','pss_kb':{'llama_server':1}}),
                      patch.object(p.os,'sched_setaffinity'),patch.object(p.os,'sched_getaffinity',return_value={4} if bad_mask else {0}),
                      patch.object(pm,'capped_by_policy',return_value={})):stack.enter_context(m)
            def monitor_loop(period,read,rows,stop):
                rows.append(read());stop.set()
            stack.enter_context(patch.object(cr,'monitor_loop',side_effect=monitor_loop))
            stack.enter_context(patch.object(power,'sampler'))
            def cycle(name,duration,ops,record):
                assert [t.args[-1] for t in pending[:-1]]==[.11,.37,1.31,.71]
                assert [t.args[0] for t in pending[:-1]]==[1,4.87,5.13,5.23]
                for thread in pending:
                    # Reset only the fake stop between one-shot worker executions.
                    for obj in getattr(thread.target,'__closure__',None) or []:
                        try:value=obj.cell_contents
                        except ValueError:continue
                        if isinstance(value,Stop):value.done=False
                    thread.target(*thread.args)
                record.update(duration_s=180,m2=[],selector=[],yolo=[],reads=[])
                record['power']=[dict(t=i,t_start=i-.01,battery_w=1) for i in range(-1,182)]
            stack.enter_context(patch.object(p,'run_cycle',side_effect=cycle))
            record=p.live_block('L4',Path(tmp)/'b.json',NS(proc=NS(pid=123),alive=lambda:True),{'skin':30})
            assert record['validity']==('NOT VALID — INCOMPLETE' if bad_mask else 'VALID'),record.get('error')
            if bad_mask:assert record['monitor_errors']
            else:assert waits and record['shared_checks'] and record['memory'] and record['dumps']
    print('PASS pinned periodic worker startup offsets, health/process checks, read wiring, affinity failure and root-scope cleanup (mocks)')


def check_round3_regressions():
    cr=NS(LMK_KILL_RE=__import__('re').compile('lmkd.*kill'))
    cr.lmk_lines=lambda since:dict(ok=True,lines=['99 lmkd kill before','100 lmkd kill inside','279.9 lmkd kill inside','280 lmkd kill after'])
    got=p.lmk_window(cr,100,280)
    assert got['n_kills']==2 and len(got['outside_block_lines'])==2 and got['ok']
    cr.lmk_lines=lambda since:dict(ok=True,lines=['bad lmkd kill','nan lmkd kill'])
    assert not p.lmk_window(cr,100,280)['ok']
    cr.lmk_lines=lambda since:dict(ok=False,lines=[])
    assert not p.lmk_window(cr,100,280)['ok']
    clock=[0.];rows=[];errors=[];verifies=[]
    class Stop:
        done=False
        def is_set(self):return self.done
        def set(self):self.done=True
    stop=Stop()
    def reading(*args):
        start=clock[0];clock[0]+=.01
        if clock[0]>=3.05:stop.set()
        return dict(t_start=start,t=clock[0],battery_w=1)
    with patch.object(power.time,'monotonic',side_effect=lambda:clock[0]),patch.object(power.os,'sched_setaffinity'),patch.object(power.os,'sched_getaffinity',return_value={0}),patch.object(power,'sample',side_effect=reading):
        power.sampler(None,'/fake',0,{0},{6,7},stop,rows,errors,lambda:verifies.append(clock[0]),period=0,average=True)
    assert len(rows)>300 and len(verifies)==5 and not errors
    assert verifies[:4]==sorted(verifies[:4]) and all(b-a>=1 for a,b in zip(verifies[:3],verifies[1:4]))
    # M2 remains busy at the next slot, but completes during YOLO-640.
    # A stale pre-YOLO busy check would skip that slot and fail this check.
    original_thread=threading.Thread;original_wait=threading.Event.wait
    def fast_wait(event,timeout=None):return original_wait(event,None if timeout is None else max(0,timeout/1000))
    for name in ('L1','L4'):
        active=[]
        class M2Thread:
            def __init__(self,target,args):self.target,self.args,self.busy=target,args,True;active.append(self)
            def start(self):self.target(*self.args)
            def is_alive(self):return self.busy
            def join(self,timeout=None):self.busy=False
        def factory(target,args=(),**kw):
            return original_thread(target=target,args=args,**kw) if target.__name__=='selectors' else M2Thread(target,args)
        ops=FastOps(name)
        def detect(image,size):
            if size==640:
                for thread in active:thread.busy=False
            return [{},{}]
        ops.detect=detect
        record={}
        with patch.object(threading,'Thread',side_effect=factory),patch.object(threading.Event,'wait',fast_wait):p.run_cycle(name,180,ops,record)
        assert len(record['m2'])==sum(r['size']==640 for r in record['yolo'])
    assert all(abs(5/x-round(5/x))>.001 for x in (4.87,5.13,5.23))
    print('PASS de-phased telemetry periods, fastest-probe 1 Hz verification, LMK block window/error handling, busy recheck after YOLO')


if __name__=='__main__':
    check_power();check_cycle();check_runtime();check_preflight();check_lag_lifecycle();check_live_block()
    check_live_adapters();check_extra_scheduling();check_root_adapters();check_session_transitions();check_monitor_workers();check_round3_regressions()
