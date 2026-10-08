#!/data/data/com.termux/files/usr/bin/python
"""POWER LAG PROBE — method check, not a benchmark. Owner only, native Termux."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

sys.dont_write_bytecode = True
import power
import runtime as rt
from phase1 import write

LABEL='POWER LAG PROBE — method check, not a benchmark'


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv)
    if a.output.exists():ap.error('output exists; choose a fresh stem')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    mark=len(rt.READ_RETRIES)  # H2: the probe's re-reads from its first read on
    v,sb,pm,cr,_=rt.imports()
    cr.require_native(); rt.clear_processes(); cr.check_cores('lag preflight'); rt.battery_sample(sb)
    hashes=rt.require_hashes(v,cr,pm,gemma=False)
    rt.dump_check(cr)
    datum,_,image=sb.load_speed_input()
    boxes=datum['detections']
    result=dict(label=LABEL,hashes=hashes,power=[],fast=[],dumps=[],errors=[],calls=[],quiet_before_s=20,load_s=10,quiet_after_s=20,root_mask_verification_period_s=1.,
             thermal_retries=rt.THERMAL_RETRIES)
    stop=threading.Event(); threads=[]; monitors=[]; caller=head=None; root_scope=None; screen=rt.screen(sb)
    lock_path=rt.HOME/'.cache/campaign_p1.lock'; lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(a.output,result)
            screen.start()
            rt.stop_camera(cr)
            monitor=rt.prepare(sb,{6,7},restrict={0,1,2,3}); monitors.append(monitor)
            fast_monitor=rt.prepare(sb,{6,7},restrict={0,1,2,3}); monitors.append(fast_monitor)
            root_scope=rt.root_mask_scope(cr,sb,monitor[1],{6,7});root_scope.__enter__()
            head,caller,tids=v.build('fp32','BIG')
            sb.check_pinning(tids,{6,7})
            t0=time.monotonic()
            def worker(period,read,rows):
                try:
                    tid=threading.get_native_id(); os.sched_setaffinity(tid,monitor[1])
                    if os.sched_getaffinity(tid)!=monitor[1]:raise RuntimeError('lag monitor pin failed')
                    cr.monitor_loop(period,read,rows,stop)
                except BaseException as e:
                    result['errors'].append(str(e)); stop.set()
            thread=threading.Thread(target=power.sampler,args=(monitor[0],cr.BATTERY,t0,monitor[1],{6,7},stop,
                result['power'],result['errors'],lambda:rt.verify_monitor(sb,monitor,{6,7}),0,True),daemon=True)
            threads.append(thread); thread.start()
            for period,read,rows in ((1,lambda:rt.counted(lambda:rt.fast_check(cr,fast_monitor)),result['fast']),
                                     (5,lambda:rt.read_dump(cr),result['dumps'])):
                thread=threading.Thread(target=worker,args=(period,read,rows),daemon=True); threads.append(thread); thread.start()
            def guard():
                cr.check_cores('lag probe')
                if result['errors']:raise RuntimeError('lag sampler failed: '+str(result['errors']))
                if hit:=cr.block_limit(time.monotonic(),t0,result['fast'],result['dumps']):raise RuntimeError('lag safety stop: '+str(hit))
                if not result['power'] and time.monotonic()-t0>5:raise RuntimeError('no power samples')
            thread=threading.Thread(target=worker,args=(5.23,lambda:(rt.clear_processes() or {'t':time.monotonic()}),[]),daemon=True)
            threads.append(thread);thread.start()
            def quiet(until):
                while time.monotonic()<until:
                    guard(); time.sleep(.1)
            quiet(t0+20)
            deadline=None
            while deadline is None or time.monotonic()<deadline:
                guard(); sb.check_pinning(tids,{6,7})
                start=time.monotonic()-t0
                if deadline is None:
                    result['load_start_s']=start
                    deadline=t0+start+10
                _,ms=caller.detect(image,boxes)
                result['calls'].append(dict(started_s=start,ended_s=time.monotonic()-t0,ms=ms))
            result['load_end_s']=time.monotonic()-t0  # actual last call end, not nominal t=30
            quiet(t0+result['load_end_s']+20)
            result['complete']=True
        except BaseException as e:
            result.update(complete=False,error=f'{type(e).__name__}: {e}')
            raise
        finally:
            stop.set()
            for thread in threads:thread.join(timeout=60)
            if any(t.is_alive() for t in threads):result['errors'].append('lag monitor still running')
            for action in [lambda:rt.release(caller),*(lambda m=m:m[0].close() for m in monitors),screen.restore]:
                try:action()
                except BaseException as e:result['errors'].append('cleanup: '+str(e))
            if root_scope:root_scope.__exit__(None,None,None)
            rows=result['power']
            span=rows[-1]['t']-rows[0]['t'] if len(rows)>1 else 0
            result['achieved_hz']=(len(rows)-1)/span if span>0 else None
            result['sample_count']=len(rows)
            if 'load_end_s' in result:
                result['step_response']={key:power.lag_response(rows,result['load_start_s'],result['load_end_s'],key)
                                         for key in ('current_now_uA','current_avg_uA')}
            if rt.retry_check(result,mark):result['errors'].append(result['read_retry_check'])  # H2 cap for the probe
            if result['errors']:result['complete']=False
            write(a.output,result)
            print(LABEL,'complete:',result.get('complete',False),'rate:',result['achieved_hz'],'Hz; files:',a.output,flush=True)
    if not result.get('complete'):raise SystemExit(1)


if __name__=='__main__':
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,lambda signum,frame: (_ for _ in ()).throw(SystemExit(128+signum)))
    main()
