#!/data/data/com.termux/files/usr/bin/python
"""L2 endurance / fixed / adaptive campaign. Owner only, native Termux, motors OFF."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import tempfile
import threading
import time

sys.dont_write_bytecode = True
import phase1 as p
import diagnostics as diag
import runtime as rt
import power

MODES = ('endurance', 'fixed', 'adaptive')
MOCK_LABEL = 'NOT VALID — DRY RUN (all hardware mocked)'
REHEARSAL = 'NOT VALID — OWNER REHEARSAL'
TIMING_FIELDS = ('camera_start_s','inference_s','camera_stop_s','camera_off_s','camera_on_without_inference_s')
FULL = dict(idle=300, gate=900, endurance=1800, fixed=2160, adaptive=2160,
            active=120, pause=60, min_active=60, min_pause=30)
SHORT = dict(idle=30, gate=30, endurance=180, fixed=180, adaptive=240,
             active=40, pause=20, min_active=60, min_pause=30)


def switch(mode, active, elapsed, skin, high, spec):
    if mode == 'fixed':
        return elapsed >= spec['active' if active else 'pause']
    if mode == 'adaptive':
        return elapsed >= spec['min_active' if active else 'min_pause'] and (
            skin >= high if active else skin <= high-2.)
    return False


def stop_reason(mode, status, elapsed, maximum, emergency=None):
    # Emergency precedence; none of the #127 emergency stops is accepted as a valid performance result.
    if emergency:return 'EMERGENCY: '+str(emergency)
    if status >= 4:return 'EMERGENCY: Android CRITICAL or higher'
    if mode == 'endurance' and status >= 3:return 'SEVERE'
    if elapsed >= maximum:return 'TIME LIMIT'
    return None


def retry_windows(retries, origin, duration):
    counts = [0]*max(1, math.ceil(duration/180))
    for row in retries:
        if origin <= row['t'] <= origin+duration:
            counts[min(len(counts)-1, int((row['t']-origin)//180))] += 1
    return dict(cap_per_180_s=rt.RETRY_CAP, counts=counts, ok=all(n <= rt.RETRY_CAP for n in counts))


def relative(rows, origin):
    result = copy.deepcopy(rows)
    for row in result:
        for key in ('t', 't_start'):
            if key in row:row[key] -= origin
    return result


def phase_timing(row):
    """Clip observed intervals to the measured phase; transitions are part of camera-ON time."""
    lo,hi=row['origin'],row['end']
    def seconds(start,end):return max(0.,min(hi,end)-max(lo,start))
    start=row.get('camera_start_at',lo if row.get('camera_on_at_origin') else hi)
    on=seconds(start,row.get('camera_off_at',hi))
    inference_start=row.get('inference_origin',hi)
    inference_end=row.get('inference_end',inference_start+max(
        (r['ended_s'] for key in ('yolo','m2','selector') for r in row.get(key,[]) if 'ended_s' in r),default=0.))
    inference=seconds(inference_start,inference_end)
    row.update(camera_start_s=seconds(row.get('camera_start_at',hi),row.get('camera_start_end_at',hi)),
        inference_s=inference,camera_stop_s=seconds(row.get('camera_stop_at',hi),row.get('camera_stop_end_at',hi)),
        camera_off_s=max(0.,hi-lo-on),camera_on_without_inference_s=max(0.,on-inference),
        duty_fraction=inference/(hi-lo) if hi>lo else 0.)


def aggregate(data, origin, duration, active, pm):
    """60 s windows including pause and transition time; raw evidence remains available."""
    windows = []
    fast, dumps, memory, watts = (relative([r for r in data.get(k, []) if origin-2 <= r['t'] <= origin+duration+2], origin) for k in ('fast','dumps','memory','power'))
    retries = relative(data.get('read_retries', []), origin)
    thermal = relative(data.get('thermal_retries', []), origin)
    for lo in range(0, math.ceil(duration), 60):
        hi = min(duration, lo+60.)
        if hi-lo<1e-6:continue
        inside = lambda row: lo <= row['t'] < hi
        fr = [dict(r, t=r['t']-lo) for r in fast if lo <= r['t'] <= hi]
        caps = pm.capped_by_policy(dict(fast=fr,duration_s=hi-lo,cpuinfo_max_khz=data['cpuinfo_max_khz']))
        skin = [r['skin'] for r in dumps if inside(r)]
        mem = [r for r in memory if inside(r)]
        work = dict(yolo=[],m2=[],selector=[],slots=[])
        for phase in active:
            offset = phase['origin']-origin+phase.get('camera_setup_s',0.)
            for key in work:
                for r in phase.get('frame_slots' if key=='slots' else key,[]):
                    stamp = r['slot_s'] if key in ('slots','yolo') else r['started_s']
                    if lo <= offset+stamp < hi:work[key].append(r)
        windows.append(dict(start_s=lo,end_s=hi,
            yolo={str(size):dict(due=sum(r['size']==size for r in work['slots']),
                done=sum(r['size']==size for r in work['yolo']),
                skipped=sum(r['size']==size for r in work['slots'])-sum(r['size']==size for r in work['yolo']),
                **p.stats([r for r in work['yolo'] if r['size']==size])) for size in (320,640)},
            m2=dict(calls=len(work['m2']),**p.stats(work['m2'],'inference_ms')),
            selector=dict(calls=len(work['selector']),misses=sum(bool(r.get('miss')) for r in work['selector']),**p.stats(work['selector'])),
            power=power.weighted(watts,lo,hi), skin_min_c=min(skin,default=None),skin_max_c=max(skin,default=None),
            android_statuses=[r['status'] for r in dumps if inside(r)],caps_by_policy=caps,
            cap_tail_unknown_s=max(0.,hi-lo-(fr[-1]['t'] if fr else 0.)),
            mem_available_min_mib=min((r['mem_available_mib'] for r in mem),default=None),
            llama_server_pss_kb=[r['pss_kb']['llama_server'] for r in mem],
            empty_answer_re_reads=sum(inside(r) for r in retries),
            thermal_re_reads=sum(r['attempts']-1 for r in thermal if inside(r)),
            read_re_reads=sum(inside(r) for r in retries)+sum(r['attempts']-1 for r in thermal if inside(r))))
    return windows


def events(dumps, windows, origin):
    result, crossed, capped, previous = [], set(), set(), None
    for row in sorted(dumps,key=lambda r:r['t']):
        if row['t'] < origin:continue
        t=row['t']-origin
        if row['status'] != previous:
            result.append(dict(event='android_status',at_s=t,previous=previous,status=row['status']))
            previous=row['status']
        for threshold in (35,37,39,41,43):
            if threshold not in crossed and row['skin'] >= threshold:
                crossed.add(threshold)
                result.append(dict(event='skin_crossing',threshold_c=threshold,at_s=t,skin_c=row['skin'],
                    already_above_at_first_read=not any(r['t']>=origin and r['t']<row['t'] for r in dumps)))
    for window in windows:
        for policy in ('policy4','policy6'):
            if policy not in capped and window['caps_by_policy'][policy]['percent'] >= 10.:
                capped.add(policy)
                result.append(dict(event='first_window_capped_10_percent',policy=policy,at_s=window['start_s'],
                                   window_end_s=window['end_s']))
    return sorted(result,key=lambda r:r['at_s'])


class LiveSession:
    """One setup and continuous monitoring across active, camera transitions and OFF pauses."""
    now = staticmethod(time.monotonic)
    def __init__(self, server, data):
        self.server,self.data=server,data
        self.v,self.sb,self.pm,self.cr,_=rt.imports()
        self.monitors,self.threads=[],[]
        self.detector=self.caller=self.head=self.scope=None
        self.stop=threading.Event()
        self.memory_lock=threading.Lock()
        self.camera_on=False
        self.watchdog_armed=False
        self.beat=self.now()
        self.main_tid=threading.get_native_id()
        self.measured={4,5,6,7}
    def start(self):
        self.monitor=rt.prepare(self.sb,self.measured);self.monitors.append(self.monitor)
        self.fast_monitor=rt.prepare(self.sb,self.measured);self.monitors.append(self.fast_monitor)
        self.mask=self.monitor[1]
        self.scope=rt.root_mask_scope(self.cr,self.sb,self.mask,self.measured);self.scope.__enter__()
        self.detector,self.ort=self.pm.build_detector('mid')
        self.head,self.caller,self.tids=self.v.build('fp32','BIG')
        self.sb.check_pinning(self.tids,{6,7})
        self.selector=p.bounded_selector(self.cr)
        self.cases=self.cr.load_cases()
        self.fallback=rt.fallback_input()
        self.data.update(cpuinfo_max_khz=self.monitor[3]['policies'],ort=self.ort,m2_cluster=dict(cpus=[6,7],threads=2),
            monitor_mask=sorted(self.mask),setup_affinity=getattr(self.monitor[0],'campaign_setup_affinity',None),
            fast=[],dumps=[],memory=[],power=[],shared_checks=[],monitor_errors=[],cleanup_errors=[])
        rt.pin_main(self.mask)
        self.camera(False)
        self.began=self.now()
        self.data['power'].append(rt.counted(lambda:power.sample(self.monitor[0],self.cr.BATTERY,0)))
        self.data['fast'].append(rt.counted(lambda:rt.fast_check(self.cr,self.fast_monitor,self.sb,self.measured)))
        self.data['dumps'].append(rt.read_dump(self.cr))
        self.data['memory'].append(self.memory())
        self.beat=self.now()  # Arm only after model/monitor setup, before the first watchdog worker.
        self.watchdog_armed=True
        for period,read,key in ((1.,lambda:rt.counted(lambda:rt.fast_check(self.cr,self.fast_monitor,self.sb,self.measured)),'fast'),
            (4.87,lambda:rt.read_dump(self.cr),'dumps'),(5.13,self.memory,'memory'),(5.23,self.shared,'shared_checks')):
            thread=threading.Thread(target=self.worker,args=(period,read,self.data[key]),daemon=True)
            self.threads.append(thread);thread.start()
        thread=threading.Thread(target=power.sampler,args=(self.monitor[0],self.cr.BATTERY,0,self.mask,self.measured,
            self.stop,self.data['power'],self.data['monitor_errors'],lambda:rt.verify_monitor(self.sb,self.monitor,self.measured)),daemon=True)
        self.threads.append(thread);thread.start()
    def worker(self,period,read,rows):
        try:
            tid=threading.get_native_id();os.sched_setaffinity(tid,self.mask)
            def checked():
                if os.sched_getaffinity(tid)!=self.mask:raise RuntimeError('monitor affinity changed')
                return read()
            self.cr.monitor_loop(period,checked,rows,self.stop)
        except diag.BatteryStop as e:
            self.data['battery_stop']=str(e);self.stop.set()
        except BaseException as e:
            self.data['monitor_errors'].append(f'{type(e).__name__}: {e}');self.stop.set()
    def memory(self):
        with self.memory_lock:
            row=rt.counted(lambda:rt.memory_sample(self.cr,self.server.proc.pid,self.camera_on))
        if p.memory_row_bad(row):raise RuntimeError('memory/battery read failed: '+str(p.memory_row_cause(row)))
        return row
    def shared(self):
        age=self.now()-self.beat
        if self.watchdog_armed and age>rt.HEARTBEAT_S:
            print(f'main loop heartbeat {age:.1f} s old: SIGTERM, normal cleanup',flush=True)
            os.kill(os.getpid(),signal.SIGTERM)
            raise RuntimeError('heartbeat expired')
        rt.clear_processes(self.server.proc.pid)
        if not rt.server_ok(self.server):raise RuntimeError('llama-server unhealthy')
        diag.battery(self.cr,25)
        return dict(t=self.now(),ok=True)
    def guard(self):
        if threading.get_native_id()==self.main_tid:self.beat=self.now()
        rt.check_cores(self.cr,'campaign P23')
        if self.data['monitor_errors']:raise RuntimeError('monitor failed: '+rt.causes(self.data['monitor_errors']))
        if hit:=self.cr.block_limit(self.now(),self.began,self.data['fast'],self.data['dumps']):
            raise RuntimeError('EMERGENCY: '+str(hit))
        if self.data.get('battery_stop'):raise diag.BatteryStop(self.data['battery_stop'])
        if threading.get_native_id()==self.main_tid and hasattr(self,'checkpoint'):self.checkpoint()
    def wait(self,seconds):
        end=self.now()+seconds
        while self.now()<end:
            self.guard()
            self.stop.wait(min(.2,max(0.,end-self.now())))
        self.guard()
    def reading(self):
        row=rt.read_dump(self.cr)
        self.data['dumps'].append(row)
        return row
    def skin(self):return self.data['dumps'][-1]['skin']
    def status(self):return self.data['dumps'][-1]['status']
    def camera(self,on):
        # The camera wrapper has its own bounded attempts; shared safety checks remain running.
        armed=self.watchdog_armed;self.watchdog_armed=False
        try:
            with self.memory_lock:
                if on:
                    retry_mark=len(rt.READ_RETRIES)
                    try:cam=rt.camera_start(self.pm,rate=1)
                    except BaseException:
                        launch_retries=sum(r['what']=='am start RobotCam' for r in rt.READ_RETRIES[retry_mark:])
                        self.data['camera_failures']=self.data.get('camera_failures',0)+1+launch_retries
                        raise
                    launch_retries=sum(r['what']=='am start RobotCam' for r in rt.READ_RETRIES[retry_mark:])
                    self.data['camera_failures']=self.data.get('camera_failures',0)+cam.get('attempts',1)-1+launch_retries
                    self.data['camera_starts']=self.data.get('camera_starts',0)+1
                    self.cam=cam
                else:rt.stop_camera(self.cr)
                self.camera_on=on
        finally:self.beat=self.now();self.watchdog_armed=armed
        return getattr(self,'cam',None)
    def cycle(self,duration,row,stop_when,force_fallback=False):
        ops=p.LiveOps(self.cr,self.sb,self.detector,self.ort,self.caller,self.tids,{6,7},self.cam,
                      self.server,self.selector,self.cases,self.guard)
        ops.fallback_input=self.fallback
        ops.force_fallback_slot=0 if force_fallback else None
        def drain_heartbeat():
            self.beat=self.now()
            if hasattr(self,'checkpoint'):self.checkpoint()
        ops.drain_heartbeat=drain_heartbeat
        ops.origin=self.now()
        row['camera_setup_s']=ops.origin-row['origin']
        row['inference_origin']=ops.origin
        p.run_cycle('L2',duration,ops,row,replacement=True,stop_when=stop_when)
        row['inference_end']=ops.origin+max((r['ended_s'] for key in ('yolo','m2','selector') for r in row[key]),default=0.)
        row['boundary_idle_s']=max(0.,self.now()-row['inference_end'])
    def bracket(self,end):
        deadline=self.now()+2
        while self.data['power'][-1]['t']<end and self.now()<deadline:self.wait(.05)
    def close(self):
        self.stop.set()
        for thread in self.threads:thread.join(timeout=60)
        if any(t.is_alive() for t in self.threads):self.data.setdefault('cleanup_errors',[]).append('monitor still running')
        for action in (lambda:self.camera(False),lambda:rt.release(self.caller),lambda:rt.release(self.detector),
                       *(lambda m=m:m[0].close() for m in self.monitors)):
            try:action()
            except BaseException as e:self.data.setdefault('cleanup_errors',[]).append(f'{type(e).__name__}: {e}')
        self.head=None
        if self.scope:self.scope.__exit__(None,None,None)


class MockSession:
    """Offline virtual clock; same session controller, with no hardware or waits."""
    def __init__(self,server,data):
        self.data=data;self.clock=100.;self.camera_on=False;self.temp=30.;self.camera_count=0
        self.pm=type('PM',(),{'capped_by_policy':staticmethod(mock_caps)})
        data.update(cpuinfo_max_khz={k:1000 for k in ('policy0','policy4','policy6')},
                    fast=[],dumps=[],memory=[],power=[],monitor_errors=[],cleanup_errors=[])
    def now(self):return self.clock
    def start(self):self.wait(1)
    def guard(self):
        if hasattr(self,'checkpoint'):self.checkpoint()
    def wait(self,seconds):
        end=self.clock+seconds
        while self.clock<end:
            step=min(.37,end-self.clock);self.clock+=step
            self.temp+=step*(.04 if self.camera_on else -.04)
            self.data['power'].append(dict(t=self.clock,t_start=self.clock,battery_w=4. if self.camera_on else 1.))
            self.data['fast'].append(dict(t=self.clock,max={k:1000 for k in self.data['cpuinfo_max_khz']}))
            self.data['dumps'].append(dict(t=self.clock,rc=0,skin=self.temp,status=0,attempts=1))
            self.data['memory'].append(dict(t=self.clock,root_rc=0,battery_status='Discharging',mem_available_mib=2500,pss_kb={'llama_server':4000000}))
    def reading(self):return dict(self.data['dumps'][-1])
    def skin(self):return self.temp
    def status(self):return 0
    def camera(self,on):
        if on:
            self.camera_on=True;self.wait(3.)
        elif self.camera_on:self.wait(2.)
        self.camera_on=on
        if on:
            self.camera_count+=1;self.data['camera_starts']=self.camera_count
        return dict(session=f'mock{self.camera_count}')
    def cycle(self,duration,row,stop_when,force_fallback=False):
        start=self.now()
        row.update(camera_setup_s=start-row['origin'],frame_slots=[],reads=[],yolo=[],m2=[],selector=[],frames_due={'320':0,'640':0},frames_skipped={'320':0,'640':0})
        row['inference_origin']=start
        frame_budget=p.FRAME_WAIT_S+2.
        selector_slots=list(range(0,max(0,math.floor(duration/20))*20,20))
        row['slot_plan']=dict(frame_finish_budget_s=frame_budget,selector_slots=selector_slots,partial_tails_unscheduled=True)
        for slot in range(max(0,math.floor(duration-frame_budget)+1)):
            if stop_when():row['controlled_end']=True;break
            size=640 if slot and slot%5==0 else 320
            row['frame_slots'].append(dict(slot_s=slot,size=size));row['frames_due'][str(size)]+=1
            row['yolo'].append(dict(slot_s=slot,size=size,started_s=slot,ended_s=slot+.01,ms=10,n_detections=2))
            if slot in selector_slots:
                row['m2'].append(dict(slot_s=slot,started_s=slot,ended_s=slot+.01,inference_ms=10,ms=10,inference_executed=True,scene='fallback' if force_fallback and slot==0 else 'live'))
                row['selector'].append(dict(slot_s=slot,started_s=slot+.01,ended_s=slot+.02,ms=10,miss=False,
                    context='(fallback scene)\n' if force_fallback and slot==0 else 'M2 relations'))
            self.wait(min(1.,duration-slot))
        while self.now()<start+duration and not stop_when():self.wait(min(.2,start+duration-self.now()))
        row['duration_s']=self.now()-start
        row['inference_end']=start+max((r['ended_s'] for key in ('yolo','m2','selector') for r in row[key]),default=0.)
        p.summarize(row)
    def bracket(self,end):self.wait(.37)
    def close(self):self.camera(False)


def mock_caps(b):
    return {k:dict(percent=0.,seconds=0.,unknown_s=0.,lowest_mhz=1.) for k in b['cpuinfo_max_khz']}


def phase_evidence(session,row,active,limit=None):
    row['observed_end']=session.now()
    end=min(row['observed_end'],limit) if limit is not None else row['observed_end']
    row['wall_duration_s']=end-row['origin']
    row['end']=end
    phase_timing(row)
    try:session.bracket(end)
    except Exception as e:
        row['evidence_error']=f'{type(e).__name__}: {e}'
        row['evidence_stop_reason']='BATTERY BELOW 25%' if isinstance(e,diag.BatteryStop) else 'EMERGENCY: evidence bracket: '+str(e)
    row['windows']=aggregate(session.data,row['origin'],row['wall_duration_s'],active,session.pm)
    skins=[r['skin'] for r in session.data['dumps'] if row['origin']<=r['t']<=end]
    row['skin_min_c'],row['skin_max_c']=min(skins,default=None),max(skins,default=None)
    row['caps_by_policy']=session.pm.capped_by_policy(dict(fast=relative([r for r in session.data['fast'] if row['origin']<=r['t']<=end],row['origin']),
        duration_s=row['wall_duration_s'],cpuinfo_max_khz=session.data['cpuinfo_max_khz']))
    row['read_retry_windows']=retry_windows(rt.READ_RETRIES,row['origin'],row['wall_duration_s'])


def run_mode(mode,session,result,spec,high,stem,dry=False):
    row=dict(mode=mode,label='IN PROGRESS — NOT VALID',active_phases=[],pauses=[],cycles=[],switches=[])
    camera_starts=session.data.get('camera_starts',0)
    camera_failures=session.data.get('camera_failures',0)
    result['runs'].append(row)
    evidence_label=MOCK_LABEL if result['mock_only'] else REHEARSAL if dry else 'IN PROGRESS — NOT VALID'
    saved=[session.now()-2.];sample_files=[];row['sample_checkpoints']=sample_files
    def checkpoint(force=False):
        if force or session.now()-saved[0]>=60:
            end=session.now();lo=saved[0]-2.
            samples={key:[r for r in list(value) if not isinstance(r,dict) or 't' not in r or lo<=r['t']<=end]
                if isinstance(value,list) else value for key,value in session.data.items()}
            path=stem(f'_samples_{len(sample_files)+1:04d}.json')
            work=[]
            for phase in row['active_phases']:
                origin=phase.get('inference_origin',phase['origin'])
                completed={key:[r for r in list(phase.get(key,[])) if lo<=origin+r.get('ended_s',r.get('t',r.get('slot_s',0.)))<=end]
                    for key in ('frame_slots','reads','yolo','m2','selector','camera_misses')}
                if any(completed.values()):work.append(dict(phase_origin=phase['origin'],inference_origin=origin,**completed))
            p.write(path,dict(label=evidence_label,mode=mode,origin=lo,end=end,overlap_s=2.,samples=samples,completed_work=work))
            sample_files.append(path.name);saved[0]=end
    session.checkpoint=checkpoint
    checkpoint(True)
    idle=dict(label=evidence_label,phase='IDLE CAMERA OFF',origin=session.now())
    row['idle']=idle
    try:
        session.camera(False)
        idle['skin_start']=session.reading()
        session.wait(spec['idle'])
    except BaseException as e:
        idle['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        try:phase_evidence(session,idle,[])
        finally:p.write(stem('_idle.json'),idle)
    row['gate_origin']=session.now()
    gate=dict(label=evidence_label,phase='COLD GATE CAMERA OFF',origin=row['gate_origin'])
    row['gate_phase']=gate
    try:
        reading,cooling=p.start_gate(session.reading,session.wait,session.guard,idle['skin_start']['skin'],spec['gate'],1.,now=session.now)
        row.update(gate=cooling,skin_start=reading,cold_start=cooling['reached'],origin=session.now())
        gate.update(reading=reading,cooling=cooling)
    except BaseException as e:
        gate['error']=f'{type(e).__name__}: {e}'
        raise
    finally:p.write(stem('_gate.json'),gate)
    # Camera startup belongs to active time; no unmeasured setup heats the phone after the gate decision.
    if dry and mode=='adaptive':high=reading['skin']+1.
    row['t_hi_c']=high
    row['gate_read_retry_windows']=retry_windows(rt.READ_RETRIES,row['gate_origin'],row['origin']-row['gate_origin'])
    maximum=spec[mode]
    deadline=row['origin']+maximum
    row.update(planned_duration_s=maximum,duty_fraction_basis='inference_s / duration_s',
        inference_s_definition='camera-ready through last completed inference call, including inter-call waits; clipped to measured slots',
        duty_description=(f"wall-clock grid: {int(maximum/(spec['active']+spec['pause']))} cycles of {spec['active']+spec['pause']} s; {spec['active']} s active slot / {spec['pause']} s pause slot; camera transitions included; actual inference/OFF durations vary"
            if mode=='fixed' else f"wall-clock limit {maximum} s; adaptive minimum active {spec['min_active']} s including camera start, minimum confirmed-OFF pause {spec['min_pause']} s"
            if mode=='adaptive' else f'continuous L2, wall-clock limit {maximum} s; camera start included'))
    reason=None
    active=True
    try:
        while session.now()<deadline:
            session.guard()
            if mode=='adaptive' and active and row['active_phases'] and deadline-session.now()<spec['min_active']+1.:
                if row['switches'] and not row['switches'][-1]['from_active']:row['switches'].pop()
                session.wait(deadline-session.now())
                phase_evidence(session,phase,[],deadline)
                p.write(path,phase)
                break
            previous=(row['pauses'] if active else row['active_phases'])
            phase=dict(origin=previous[-1]['end'] if previous else row['origin'],actual_start_at=session.now(),
                camera_on_at_origin=session.camera_on,label=evidence_label,
                phase=('ACTIVE SLOT L2' if active else 'PAUSE SLOT: STOP CAMERA THEN OFF') if mode=='fixed' else ('ACTIVE L2' if active else 'PAUSE: STOP CAMERA THEN OFF'))
            target=row['active_phases'] if active else row['pauses'];target.append(phase)
            path=stem(f'_{"active" if active else "pause"}_{len(target):02d}.json')
            cycle=len(row['active_phases'])
            if mode=='fixed':phase['origin']=row['origin']+(cycle-1)*(spec['active']+spec['pause'])+(0 if active else spec['active'])
            planned_end=min(deadline,row['origin']+(cycle-1)*(spec['active']+spec['pause'])+spec['active']+(0 if active else spec['pause'])) if mode=='fixed' else deadline
            phase['planned_end']=planned_end
            print(f'START {mode} {phase["phase"]} at +{phase["origin"]-row["origin"]:.1f} s',flush=True)
            try:
                if active:
                    phase['camera_start_at']=session.now()
                    try:phase['camera']=session.camera(True)
                    finally:phase['camera_start_end_at']=session.now()
                    def should_end():
                        nonlocal reason
                        session.guard()
                        elapsed=session.now()-row['origin']
                        reason=reason or stop_reason(mode,session.status(),elapsed,maximum)
                        ending=bool(reason or (session.now()>=planned_end if mode=='fixed' else switch(mode,True,session.now()-phase['camera_start_at'],session.skin(),high,spec)))
                        if ending:phase.setdefault('stop_decision',dict(at_s=elapsed,skin_c=session.skin(),status=session.status(),reason=reason or ('ACTIVE SLOT END' if mode=='fixed' else 'HIGH THRESHOLD')))
                        return ending
                    remaining=planned_end-session.now()
                    if remaining<=0:raise RuntimeError('camera startup consumed active interval')
                    # Later short fixed actives must exercise live M2, rather than repeat the only fallback slot.
                    session.cycle(remaining,phase,should_end,force_fallback=dry and len(row['active_phases'])==1)
                else:
                    phase['inference_origin']=row['active_phases'][-1].get('inference_origin',phase['origin'])
                    phase['inference_end']=row['active_phases'][-1].get('inference_end',phase['origin'])
                    phase['camera_stop_at']=session.now()
                    try:session.camera(False)
                    finally:phase['camera_stop_end_at']=session.now()
                    phase['camera_off_at']=session.now()
                    while session.now()<planned_end:
                        if switch(mode,False,session.now()-phase['camera_off_at'],session.skin(),high,spec) and (mode!='adaptive' or deadline-session.now()>=spec['min_active']+1.):break
                        session.wait(min(.2,planned_end-session.now()))
                phase['skin_end']=session.reading()
                session.guard()
                if mode=='endurance':reason=reason or stop_reason(mode,session.status(),session.now()-row['origin'],maximum)
            except BaseException as e:
                phase['error']=f'{type(e).__name__}: {e}'
                raise
            finally:
                try:phase_evidence(session,phase,[phase] if active else [],planned_end)
                finally:p.write(path,phase)
                print(f'END {mode} {phase["phase"]} {phase["wall_duration_s"]:.1f} s',flush=True)
            if phase.get('evidence_stop_reason'):reason=phase['evidence_stop_reason']
            if reason:break
            if mode=='endurance':break
            if session.now()>=deadline:break
            decision=phase.get('stop_decision',{}) if active else {}
            row['switches'].append(dict(at_s=decision.get('at_s',session.now()-row['origin']),from_active=active,skin_c=decision.get('skin_c',session.skin()),threshold_c=high if active else high-2.))
            active=not active
        reason=reason or 'TIME LIMIT'
        row['stop_reason']=reason
    except BaseException as e:
        row.update(error=f'{type(e).__name__}: {e}',stop_reason='BATTERY BELOW 25%' if isinstance(e,diag.BatteryStop) else 'EMERGENCY OR FAILURE')
        raise
    finally:
        row['end']=phase.get('end',session.now()) if 'phase' in locals() else session.now()
        try:session.camera(False)
        except BaseException as e:
            row.setdefault('error',f'camera cleanup: {type(e).__name__}: {e}')
            session.data['cleanup_errors'].append(row['error'])
        cleanup_evidence=dict(origin=row['origin'])
        phase_evidence(session,cleanup_evidence,[])
        if cleanup_evidence.get('evidence_stop_reason'):
            reason=cleanup_evidence['evidence_stop_reason'];row['stop_reason']=reason
            row['error']=cleanup_evidence['evidence_error']
        row['duration_s']=row['end']-row['origin']
        row['completion_lag_s']=max(0.,session.now()-row['end'])
        row['cleanup_read_retry_windows']=retry_windows(rt.READ_RETRIES,row['end'],max(0.,session.now()-row['end']))
        row['windows']=aggregate(session.data,row['origin'],row['duration_s'],row['active_phases'],session.pm)
        row['events']=events([r for r in session.data['dumps'] if r['t']<=row['end']],row['windows'],row['origin'])
        row['events'].append(dict(event='stop',at_s=row['duration_s'],reason=row.get('stop_reason')))
        row['read_retry_windows']=retry_windows(rt.READ_RETRIES,row['origin'],row['duration_s'])
        for i,a in enumerate(row['active_phases']):
            pause=row['pauses'][i] if i<len(row['pauses']) else {}
            row['cycles'].append(dict(active_s=a.get('wall_duration_s',0.),pause_s=pause.get('wall_duration_s',0.),
                **{key:a.get(key,0.)+pause.get(key,0.) for key in TIMING_FIELDS},
                skin_min_c=min((x for x in (a.get('skin_min_c'),pause.get('skin_min_c')) if x is not None),default=None),
                skin_max_c=max((x for x in (a.get('skin_max_c'),pause.get('skin_max_c')) if x is not None),default=None),caps_during_active=a.get('caps_by_policy')))
        active_s=sum(r.get('wall_duration_s',0.) for r in row['active_phases'])
        pauses=[r['wall_duration_s'] for r in row['pauses'] if 'wall_duration_s' in r]
        for cycle in row['cycles']:cycle['duty_fraction']=cycle['inference_s']/(cycle['active_s']+cycle['pause_s']) if cycle['active_s']+cycle['pause_s'] else 0.
        row['totals']=dict(active_s=active_s,pause_s=sum(pauses),
            **{key:sum(c[key] for c in row['cycles']) for key in TIMING_FIELDS},
            pause_count=len(pauses),mean_pause_s=statistics.mean(pauses) if pauses else 0.,camera_restarts=max(0,session.data.get('camera_starts',0)-camera_starts-1),
            camera_failures=session.data.get('camera_failures',0)-camera_failures,
            capping_while_active=any((c['percent'] or 0)>0 for a in row['active_phases'] for c in a.get('caps_by_policy',{}).values()))
        row['totals']['duty_fraction']=row['totals']['inference_s']/row['duration_s'] if row['duration_s'] else 0.
        issues=[]
        if str(reason).startswith('EMERGENCY'):issues.append(reason)
        if row.get('error'):issues.append(row['error'])
        if any(a.get('evidence_error') for a in [idle]+row['active_phases']+row['pauses']):issues.append('phase evidence failed')
        if result.get('setup',{}).get('read_retry_check')!='OK':issues.append('setup read retry cap')
        if any(a.get('cadence_missed') for a in row['active_phases']):issues.append('cadence misses')
        if mode=='fixed' and reason=='TIME LIMIT' and len(row['pauses'])!=int(maximum/(spec['active']+spec['pause'])):issues.append('incomplete fixed grid')
        if mode=='adaptive' and reason=='TIME LIMIT' and any(a.get('end',a['origin'])-a.get('camera_start_at',a['origin'])<spec['min_active']-1e-6 for a in row['active_phases']):issues.append('adaptive active below minimum')
        if any(w['power']['mean_battery_w'] is None for w in idle['windows']+row['windows']):issues.append('power coverage')
        if not all(r.get('read_retry_windows',{}).get('ok',False) for r in [idle,row]):issues.append('read retry cap')
        if not row['gate_read_retry_windows']['ok']:issues.append('gate read retry cap')
        if not row['cleanup_read_retry_windows']['ok']:issues.append('mode cleanup read retry cap')
        if session.data['monitor_errors'] or session.data['cleanup_errors']:issues.append('monitor/cleanup failure')
        row['issues']=issues
        validity='NOT VALID — '+ '; '.join(issues) if issues else 'WARM START — heat-timing results NOT VALID' if not row['cold_start'] else 'VALID — STOPPED AT SEVERE' if reason=='SEVERE' else 'VALID'
        row['checks_validity']=validity
        row['result_validity']=MOCK_LABEL if result['mock_only'] else REHEARSAL if dry else validity
        row['label']=MOCK_LABEL if result['mock_only'] else REHEARSAL if dry else validity
        p.write(stem('.json'),row)
        checkpoint(True)


def coverage(result):
    rows=result['runs']
    modes=[r['mode'] for r in rows]
    live={r['mode']:sum(m.get('scene')=='live' for a in r['active_phases'] for m in a.get('m2',[])) for r in rows}
    fallback={r['mode']:sum(m.get('scene')=='fallback' for a in r['active_phases'] for m in a.get('m2',[])) for r in rows}
    contexts={r['mode']:sum(str(s.get('context')).startswith('(fallback scene)') for a in r['active_phases'] for s in a.get('selector',[])) for r in rows}
    adaptive=next((r for r in rows if r['mode']=='adaptive'),{})
    switches=adaptive.get('switches',[])
    paths=dict(modes=modes,live_m2_calls=live,fallback_m2_calls=fallback,selector_fallback_contexts=contexts,
        fixed_pauses=len(next((r for r in rows if r['mode']=='fixed'),{}).get('pauses',[])),
        adaptive_high_switch=any(r['from_active'] and r['skin_c']>=adaptive.get('t_hi_c',math.inf) for r in switches),
        adaptive_low_restart=any(not r['from_active'] and r['skin_c']<=adaptive.get('t_hi_c',-math.inf)-2 for r in switches),
        adaptive_camera_restarts=adaptive.get('totals',{}).get('camera_restarts',0),
        camera_restarts=sum(r.get('totals',{}).get('camera_restarts',0) for r in rows),
        camera_failed_attempts=sum(r.get('totals',{}).get('camera_failures',0) for r in rows),
        errors=[r.get('issues') for r in rows if r.get('issues')],setup_complete=result.get('setup_complete',False))
    paths['rehearsal_pass']=bool(not result['mock_only'] and modes==list(MODES) and all(live.values()) and all(fallback.values()) and all(contexts.values()) and
        paths['fixed_pauses']==3 and paths['adaptive_high_switch'] and paths['adaptive_low_restart'] and paths['adaptive_camera_restarts'] and
        not paths['camera_failed_attempts'] and not paths['errors'] and not result.get('error') and not result.get('cleanup_errors') and
        not result.get('samples',{}).get('monitor_errors') and not result.get('samples',{}).get('cleanup_errors') and
        result.get('setup',{}).get('read_retry_check')=='OK' and paths['setup_complete'])
    return paths


def code_hashes():
    names=['benchmark/campaign/'+name for name in ('phase23.py','phase1.py','runtime.py','power.py','diagnostics.py','expected_hashes.json')]
    names+=['benchmark/coresidency/coresidency.py','benchmark/power_map/power_map.py','benchmark/relate_anything/speed1/variants.py',
            'benchmark/relate_anything/desk2/speed_block.py','benchmark/relate_anything/desk2/desk_check.py','detector_size_policy.py','robotcam_reader.py','detect_person.py',
            'server_manager.py','benchmark/relate_anything/speed1/session.py',
            'benchmark/strategic_selector/ladder/measure.py','benchmark/strategic_selector/ladder/cases/ladder_cases_v1.jsonl']
    return {name:hashlib.sha256((rt.ROBOT/name).read_bytes()).hexdigest() for name in names}


def require_rehearsal(path):
    d=json.loads(path.read_bytes())
    if d.get('mock_only') is not False or d.get('code_hashes')!=code_hashes() or d.get('rehearsal_coverage',{}).get('rehearsal_pass') is not True:
        raise RuntimeError('mandatory live rehearsal missing, failed, mocked or from other code')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stem_exists(output):
    return output.with_name(output.stem+'_llama-server.log').exists()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=MODES)
    ap.add_argument('--t-hi',type=float,default=37.)
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true',help='ONE shortened owner rehearsal of all three modes')
    group.add_argument('--preflight',action='store_true')
    ap.add_argument('--mock',action='store_true')
    ap.add_argument('--rehearsal',type=Path,help='passed live rehearsal JSON; mandatory for a session')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv)
    if a.mock and not a.dry_run:ap.error('--mock requires --dry-run')
    if a.dry_run and a.mode:ap.error('--dry-run exercises all three modes; omit --mode')
    if not a.dry_run and not a.mode:ap.error('--mode is required')
    if not math.isfinite(a.t_hi) or not 25<=a.t_hi<=43:ap.error('--t-hi must be finite and within 25..43 C')
    if a.mode!='adaptive' and a.t_hi!=37:ap.error('--t-hi is for adaptive mode')
    if a.output.suffix!='.json':ap.error('--output must end in .json')
    if a.output.exists() or list(a.output.parent.glob(a.output.stem+'_*.json')) or stem_exists(a.output):ap.error('evidence exists; choose a fresh stem')
    proof=None
    if not a.dry_run and not a.preflight:
        if not a.rehearsal:ap.error('--rehearsal is mandatory before a session')
        proof=require_rehearsal(a.rehearsal)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    stem=lambda suffix:a.output.with_name(a.output.stem+suffix)
    result=dict(label='IN PROGRESS — NOT VALID',runs=[],mock_only=a.mock,rehearsal=a.dry_run,
        code_hashes=code_hashes(),rehearsal_proof=proof,mode='rehearsal' if a.dry_run else a.mode,
        read_retries=rt.READ_RETRIES,thermal_retries=rt.THERMAL_RETRIES,cleanup_errors=[],
        detector_note='#128 CONFIRM: 1 frame/s, SizePolicy replaces 320 with 640 every 5 s; MID 4-5 / 2 threads',
        emergency_validity_rule='All emergency stops NOT VALID; only endurance SEVERE is a regular measured stop.',
        context_accuracy='NOT EVALUATED')
    resources={};session=None;data=result['samples']={'read_retries':rt.READ_RETRIES,'thermal_retries':rt.THERMAL_RETRIES}
    lock_path=Path(tempfile.gettempdir())/'campaign_p1_dry.lock' if a.mock else rt.HOME/'.cache/campaign_p1.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p.write(a.output,result)
            result['setup']={};mark=len(rt.READ_RETRIES)
            if not a.mock:p.preflight(['L2'],a.output,result,resources)
            rt.retry_check(result['setup'],mark)
            result['setup_complete']=True
            if a.preflight:
                result['label']='PREFLIGHT ONLY — NO TIMING'
                return
            session=(MockSession if a.mock else LiveSession)(resources.get('server'),data)
            session.start()
            rt.retry_check(result['setup'],mark)
            result['lmk_start_epoch_s']=time.time()
            modes=MODES if a.dry_run else (a.mode,)
            for mode in modes:
                run_mode(mode,session,result,SHORT if a.dry_run else FULL,a.t_hi,
                    lambda suffix,mode=mode:stem('_'+mode+suffix),dry=a.dry_run)
                p.write(a.output,result)
            result['label']=MOCK_LABEL if a.mock else REHEARSAL if a.dry_run else result['runs'][0]['label']
        except diag.BatteryStop as e:
            result.update(error=str(e),label=MOCK_LABEL if a.mock else 'SESSION STOPPED — BATTERY BELOW 25% — NOT VALID')
        except BaseException as e:
            result.update(error=f'{type(e).__name__}: {e}',label=MOCK_LABEL if a.mock else 'NOT VALID — SESSION INCOMPLETE')
            if not a.mock:result['screen_state_at_failure']=rt.screen_state(rt.imports()[3])
            raise
        finally:
            handlers={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
            if session:
                try:session.close()
                except BaseException as e:result['cleanup_errors'].append('session: '+str(e))
            for key,action in (('server',lambda:resources['server'].stop()),('screen',lambda:resources['screen'].restore())):
                if key in resources:
                    try:action()
                    except BaseException as e:result['cleanup_errors'].append(f'{key}: {type(e).__name__}: {e}')
            if not a.mock and session:
                try:
                    cr=rt.imports()[3]
                    result['lmk']=p.lmk_window(cr,result.get('lmk_start_epoch_s',time.time()),time.time())
                    if result['lmk']['ok'] is not True or result['lmk']['n_kills']:result['cleanup_errors'].append('LMK evidence failed or kills present')
                except BaseException as e:result['cleanup_errors'].append('LMK: '+str(e))
            cleanup_end=session.now() if session else time.monotonic()
            tail=result['runs'][-1].get('end',cleanup_end) if result['runs'] else cleanup_end
            result['cleanup_read_retry_windows']=retry_windows(rt.READ_RETRIES,tail,max(0.,cleanup_end-tail))
            if not result['cleanup_read_retry_windows']['ok']:result['cleanup_errors'].append('final cleanup read retry cap')
            if data.get('battery_stop'):
                result.setdefault('error',data['battery_stop'])
                result['label']=MOCK_LABEL if a.mock else 'SESSION STOPPED — BATTERY BELOW 25% — NOT VALID'
                for row in result['runs']:
                    row['result_validity']='NOT VALID — BATTERY BELOW 25%'
                    row['label']=MOCK_LABEL if a.mock else row['result_validity']
                    row.setdefault('issues',[]).append(data['battery_stop'])
                    p.write(stem('_'+row['mode']+'.json'),row)
            if result['cleanup_errors'] or data.get('monitor_errors') or data.get('cleanup_errors'):
                result['label']=MOCK_LABEL if a.mock else 'NOT VALID — CLEANUP OR MONITOR FAILURE'
                for row in result['runs']:
                    row['result_validity']='NOT VALID — CLEANUP OR MONITOR FAILURE'
                    row['label']=MOCK_LABEL if a.mock else row['result_validity']
                    row.setdefault('issues',[]).append('cleanup or monitor failure')
                    p.write(stem('_'+row['mode']+'.json'),row)
            if a.dry_run:result['rehearsal_coverage']=coverage(result)
            p.write(a.output,result)
            print(result['label'],'Evidence:',a.output,flush=True)
            if a.dry_run:print('Coverage:',json.dumps(result['rehearsal_coverage']),flush=True)
            for sig,handler in handlers.items():signal.signal(sig,handler)
            if a.preflight and result['cleanup_errors'] and not result.get('error'):raise RuntimeError('preflight cleanup failed')
    if result['cleanup_errors'] or data.get('monitor_errors') or data.get('cleanup_errors'):raise SystemExit(1)


if __name__=='__main__':
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,lambda signum,frame: (_ for _ in ()).throw(SystemExit(128+signum)))
    main()
