"""CAMPAIGN_P1_FIX2 focused offline mocks. Never run hardware."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import diagnostics as d
import phase1 as p
import runtime as rt
from self_check import FastOps


def check_fallback():
    assert rt.fallback_hashes()==rt.FALLBACK_HASHES
    image,boxes=rt.fallback_input()
    assert image.size==(482,640) and 2<=len(boxes)<=32
    with patch.object(rt,'FALLBACK_HASHES',{'speed_input.json':'bad'}):
        try:rt.fallback_hashes()
        except RuntimeError:pass
        else:raise AssertionError('bad fallback hash accepted')
    original_wait=threading.Event.wait
    def fast_wait(event,timeout=None):return original_wait(event,None if timeout is None else max(0,timeout/1000))
    for name in p.LAYOUTS:
        for n in (0,1,2,3):
            ops=FastOps(name)
            ops.detect=lambda image,size:[{}]*n
            ops.fallback=MagicMock(return_value=('fixed',[{},{}]))
            inputs=[]
            def relate(image,boxes):
                inputs.append((image,len(boxes)))
                return [],1
            ops.relate=relate
            record={}
            with patch.object(threading.Event,'wait',fast_wait):p.run_cycle(name,21,ops,record)
            calls=len(record['m2'])
            if name=='L0':
                assert calls==0 and not ops.fallback.called
                continue
            assert calls>0
            scene='fallback' if n<2 else 'live'
            assert record['summary']['m2'][scene+'_calls']==calls
            assert record['summary']['m2']['insufficient_box_slots']==0
            assert all(r['scene']==scene and r['live_boxes']==n for r in record['m2'])
            assert all(('(fallback scene)' in r['context'])==(n<2) for r in record['m2'])
            assert all(count>=2 for _,count in inputs)
            if n<2:assert all(image=='fixed' for image,_ in inputs)
            if name=='L2':assert all(('(fallback scene)' in r['context'])==(n<2) for r in record['selector'])
    print('PASS fallback hashes, PIL contract, all layouts with 0/1/2/3 live boxes, real-call counts and context labels')


def check_battery_diagnostics():
    cr=NS(BATTERY='/battery',root=lambda *a:(0,'80'))
    assert d.battery(cr,80)==80
    for value,minimum,kind in [('79',80,RuntimeError),('24',25,d.BatteryStop),('',25,RuntimeError),('101',25,RuntimeError),('nan',25,RuntimeError)]:
        cr.root=lambda *a:(0,value)
        try:d.battery(cr,minimum)
        except kind:pass
        else:raise AssertionError('bad battery accepted')
    cr.root=lambda *a:(0,'25');assert d.battery(cr,25)==25
    text='/battery/capacity\t85\n/battery/temp\t287\n/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq\t1401000\n/sys/class/thermal/thermal_zone99/temp\t23100\n/sys/class/thermal/cooling_device4/cur_state\t2\n/boost/min\tUNREADABLE\n'
    cr.root=lambda *a:(0,text);cr.read_dump=lambda:dict(rc=0,status=0,skin=23.)
    captured=[]
    def capture(command,tag,timeout=60):
        captured.append(command)
        return 0,text
    cr.root=capture
    row=d.snapshot(cr,False)
    assert row['battery_temperature_c']==28.7 and row['battery_level']==85
    assert row['nodes']['/sys/class/thermal/thermal_zone99/temp']=='23100'
    assert row['nodes']['/sys/class/thermal/cooling_device4/cur_state']=='2'
    assert row['nodes']['/boost/min']=='UNREADABLE' and not row['camera_on']
    assert row['read_elapsed_s']>=0 and 'requested' in row['camera_state_source']
    # Exact reused root wrapper, including campaign taskset/readback prefix, syntax only.
    with rt.root_mask_scope(cr,NS(),{0},{4,5}):d.snapshot(cr,False)
    for command in captured:
        assert command==command.rstrip() and '-maxdepth 5' in command
        wrapped=f'{{ {command}; }} </dev/null >/tmp/never_executed 2>&1; rc=$?; cat /tmp/never_executed; exit $rc'
        checked=subprocess.run(['sh','-n'],input=wrapped,text=True,capture_output=True)
        assert checked.returncode==0,checked.stderr
    # The round-1 form really fails this same parser; no hardware commands execute.
    broken=subprocess.run(['sh','-n'],input='{ '+captured[0]+'\n; }',text=True,capture_output=True)
    assert broken.returncode!=0
    cr.root=lambda *a:(0,text.replace('1401000','UNREADABLE'))
    try:d.snapshot(cr,False)
    except RuntimeError:pass
    else:raise AssertionError('missing required diagnostics accepted')
    print('PASS battery refusal/stop, thermal/cooling/cap parsing, bounded/timed diagnostics and exact root wrapper sh -n regression')


def check_session():
    for failure in (None,'pause_battery','block_battery','warm_failure'):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            events=[];server=NS(stop=MagicMock())
            def preflight(names,out,result,resources):
                resources['server']=server
                p.diag.battery(None,80)
            def battery(cr,minimum):
                if minimum==80:return 85
                # Before warm-up, first pause, first result, second pause, second result.
                events.append(('battery',25))
                limit=4 if failure=='pause_battery' else 5 if failure=='block_battery' else None
                if limit and sum(e[0]=='battery' for e in events)==limit:raise d.BatteryStop('battery 24% below 25%')
                return 85
            def rest(label,duration,server,record,camera_on=False,dry=False):
                events.append(('rest',label,duration,camera_on))
                record.update(label=label,planned_s=duration)
            starts=iter([20,30,31.5,31.6])
            def block(name,path,server,idle,*_):
                events.append(('block',name))
                return dict(block=name,validity='VALID',skin_start={'skin':next(starts)},
                            failure_kind='shared' if failure=='warm_failure' else None)
            for m in (patch.object(p,'preflight',side_effect=preflight),patch.object(p,'rest',side_effect=rest),
                      patch.object(p,'live_block',side_effect=block),patch.object(d,'battery',side_effect=battery),
                      contextlib.redirect_stdout(io.StringIO())):stack.enter_context(m)
            output=Path(tmp)/'session.json'
            try:p.main(['--blocks','L0,L1,L2','--output',str(output)])
            except RuntimeError:assert failure=='warm_failure'
            result=json.loads(output.read_text())
            assert [(e[2],e[3]) for e in events if e[0]=='rest'][:3]==[(180,False),(60,False),(60,True)]
            assert result['warmup']['validity']=='WARM-UP — NOT A RESULT'
            if failure is None:
                assert [b['validity'] for b in result['blocks']]==['VALID','VALID','NOT COMPARABLE — START TEMP']
                assert result['T_ref_c']==30 and len(result['pauses'])==3
                assert [e[2] for e in events if e[0]=='rest'][3:]==[600]*3
            elif failure!='warm_failure':
                assert result['label'].startswith('SESSION STOPPED — BATTERY')
                assert len(result['blocks'])==1 and result['unrun_blocks']==['L1','L2']
            else:assert result['unrun_blocks']==['L0','L1','L2']
            server.stop.assert_called_once()
    for temp,expected in [(28.49,False),(28.5,True),(31.5,True),(31.51,False)]:
        row=dict(skin_start={'skin':temp},validity='VALID')
        assert p.compare_start(row,30)==30
        assert (row['validity']=='VALID')==expected
    print('PASS warm-up/subset schedule, six fixed pauses by default, T_ref +/-1.5C, clean battery stop/unrun and server cleanup')


def check_rest():
    v,sb,pm,cr,_=rt.imports()
    for camera_on in (False,True):
        with contextlib.ExitStack() as stack:
            clock=[100.];shell=NS(close=MagicMock());fast_shell=NS(close=MagicMock())
            monitor=(shell,{0},1,{'policies':{'policy0':1803000}})
            fast_monitor=(fast_shell,{0},2,monitor[3])
            class Stop:
                done=False
                def set(self):self.done=True
                def is_set(self):return self.done
                def wait(self,dt):clock[0]+=dt;return self.done
            class Deferred:
                def __init__(self,target,args=(),**kwargs):self.target,self.args=target,args
                def start(self):pending.append(self)
                def join(self,timeout=None):pass
                def is_alive(self):return False
            pending=[]
            # Exercise actual rest worker read wiring once before advancing the fake clock.
            def monitor_loop(period,read,rows,stop):rows.append(read())
            def block_limit(*args):
                if pending:
                    workers=pending[:];pending.clear()
                    for worker in workers:worker.target(*worker.args)
                return None
            def sampler(shell,battery,began,mask,measured,stop,rows,errors,verify):
                rows.extend(dict(t=t,t_start=t-.01,battery_w=1) for t in range(-1,602))
            for m in (patch.object(p.time,'monotonic',side_effect=lambda:clock[0]),patch.object(p.threading,'Event',Stop),
                      patch.object(p.threading,'Thread',Deferred),patch.object(rt,'prepare',side_effect=[monitor,fast_monitor]),
                      patch.object(rt,'root_mask_scope',return_value=contextlib.nullcontext()),patch.object(d,'battery',return_value=85),
                      patch.object(d,'snapshot',side_effect=lambda cr,on:dict(camera_on=on)),patch.object(rt,'stop_camera'),
                      patch.object(pm,'camera_start',return_value={'session':'mock'}),patch.object(rt,'clear_processes'),
                      patch.object(sb,'battery_sample'),patch.object(cr,'check_cores'),patch.object(rt,'dump_check'),
                      patch.object(rt,'fast_check',return_value={'t':100.,'max':{'policy0':1401000}}),
                      patch.object(cr,'read_dump',return_value={'t':100.,'rc':0,'skin':30,'status':0}),
                      patch.object(cr,'meminfo_mib',return_value={'mem_available_mib':2500}),
                      patch.object(cr,'root_sample',return_value={'root_rc':0,'battery_status':'Discharging'}),
                      patch.object(cr,'monitor_loop',side_effect=monitor_loop),patch.object(cr,'block_limit',side_effect=block_limit),
                      patch.object(p.power,'sample',return_value={'t':-1,'t_start':-1,'battery_w':1}),
                      patch.object(p.power,'sampler',side_effect=sampler),patch.object(p.os,'sched_setaffinity'),
                      patch.object(p.os,'sched_getaffinity',return_value={0}),patch.object(pm,'capped_by_policy',return_value={'policy0':{'percent':100}})):
                stack.enter_context(m)
            record={};p.rest('FIXED PAUSE',600,NS(proc=NS(pid=3),alive=lambda:True),record,camera_on)
            assert clock[0]==700 and record['power_summary']['mean_battery_w']==1
            assert record['fast'] and record['dumps'] and record['memory']
            assert record['diagnostics_start']['camera_on']==camera_on
            shell.close.assert_called_once();fast_shell.close.assert_called_once()
    print('PASS actual pause 600s deadline, camera state, power/skin/caps/memory read wiring and both root reader cleanup (mocks)')


if __name__=='__main__':
    check_fallback();check_battery_diagnostics();check_session();check_rest()
