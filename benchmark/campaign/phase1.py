#!/data/data/com.termux/files/usr/bin/python
"""CAMPAIGN_P1 screening only. Motors off; no motion API is imported or commanded."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import threading
import tempfile
import time

sys.dont_write_bytecode = True
import power
import runtime as rt
import diagnostics as diag

LAYOUTS = dict(L0=None, L1='LITTLE', L2='BIG', L3='MID', L4='BIG')
DEFAULT = ['L0', 'L1', 'L2', 'L3', 'L4', 'L0']
REHEARSAL = 'NOT VALID — OWNER REHEARSAL (live hardware, shortened phases)'
CONTINUATION = ('Only local cadence misses may continue after successful cleanup and no monitor errors. '
                'Root, sensors, thermal, charger, processes, cores/affinity, camera, inference, server and cleanup failures stop later blocks. '
                'Start-temperature comparability is assessed against the first measured block.')
M2_EVERY = dict(L1=10, L3=5, L4=5)  # s between M2 calls after a 640 frame; L1 every second 640 (owner D1)
M2_MAX_BOXES = 32
FRAME_WAIT_S, CAMERA_MISS_STOP = 1.35, 3  # H4: wait for a new frame, then skip the slot; stop at 3 in a row


def blocks(value):
    names = DEFAULT[:] if value is None else value.split(',')
    if not names or any(n not in LAYOUTS for n in names):
        raise ValueError('--blocks needs comma-separated L0,L1,L2,L3,L4')
    return names


def context(rows):
    chosen = sorted((r for r in rows if r.get('pass') is True and r['score'] >= r['threshold']),
                    key=lambda r: (-r['score'], r['subject_idx'], r['object_idx'], r['predicate']))[:5]
    def clean(s):
        return ' '.join(str(s).replace('_', ' ').split())
    return 'M2 relations:\n' + ('\n'.join(f'- {clean(r["subject"])} [{r["subject_idx"]}] {clean(r["predicate"])} {clean(r["object"])} [{r["object_idx"]}].'
                                   for r in chosen) or '- none above threshold.')


def stats(rows, key='ms'):
    xs = sorted(r[key] for r in rows if key in r)
    return dict(median_ms=statistics.median(xs) if xs else None,
                p95_ms=xs[max(0, math.ceil(.95*len(xs))-1)] if xs else None)


def due_frames(now, slot, duration):
    """Catch up by SKIPPING past slots, never queue them. Returns skip slots and current slot."""
    past = []
    while slot+1 <= now and slot < duration:
        past.append(slot)
        slot += 1
    return past, slot


def can_continue(b):
    return (b.get('failure_kind') == 'cadence' and not b.get('cleanup_errors')
            and not b.get('monitor_errors'))


def run_cycle(name, duration, ops, record):
    """Same scheduling engine for mocks/live. Async M2 after 640; L2 M2 synchronously precedes selection."""
    t0 = getattr(ops, "origin", None)
    if t0 is None:
        t0 = ops.now()
    stop, ready, lock = threading.Event(), threading.Event(), threading.Lock()
    shared = dict(frame=None, boxes=None, triplets=[], scene='live', m2_thread=None)
    errors = record.setdefault('worker_errors', [])
    record.update(reads=[], yolo=[], m2=[], selector=[], frames_due={'320':0,'640':0},
                  frames_skipped={'320':0,'640':0}, camera_late=0, camera_misses=[])
    def stamp():
        return ops.now()-t0
    def m2(image, boxes, frame, slot):
        start = stamp()
        try:
            scene = 'live'
            live_boxes = len(boxes)
            truncated = 0
            # Owner rehearsal only: this slot takes the real fallback path even with live boxes.
            forced = slot == getattr(ops, 'force_fallback_slot', None)
            if live_boxes < 2 or forced:
                image, boxes = ops.fallback()
                scene = 'fallback'
            elif live_boxes > M2_MAX_BOXES:
                # H3: Head takes at most 32 boxes: the 32 highest-confidence live boxes, in detector order.
                keep = sorted(sorted(range(live_boxes), key=lambda i: -boxes[i]['conf'])[:M2_MAX_BOXES])
                boxes, truncated = [boxes[i] for i in keep], live_boxes-M2_MAX_BOXES
            rows, inference_ms = ops.relate(image, boxes)
            if inference_ms <= 0:
                raise RuntimeError('M2 slot did not execute inference')
            end = stamp()
            row = dict(slot_s=slot, frame=frame, started_s=start, ended_s=end,
                       scene=scene, live_boxes=live_boxes, input_boxes=len(boxes), truncated_boxes=truncated,
                       ms=(end-start)*1000, inference_ms=inference_ms, inference_executed=True,
                       **({'fallback_forced': True} if forced else {}),
                       context=('(fallback scene)\n' if scene=='fallback' else '')+context(rows))
            with lock:
                shared['triplets'] = rows
                shared['scene'] = scene
                record['m2'].append(row)
            return rows
        except BaseException as e:
            errors.append(f'M2 {type(e).__name__}: {e}')
            stop.set()
            return None
    def selectors():
        try:
            for slot in range(0, math.ceil(duration), 20):
                if slot >= duration or stop.wait(max(0., t0+slot-ops.now())):
                    return
                # 15 s: a skipped slot-0 frame (H4) is covered by the slot-5 one
                if name == 'L2' and not ready.wait(timeout=15):
                    raise RuntimeError('L2 no 640 frame ready')
                with lock:
                    image, boxes, frame = shared['frame'] or (None,None,None)
                if name == 'L2':
                    if m2(image, boxes, frame, slot) is None:
                        return
                with lock:
                    text = None if name == 'L0' else context(shared['triplets'])
                    if text is not None and shared['scene']=='fallback':
                        text = '(fallback scene)\n'+text
                start = stamp()
                try:
                    row = ops.select(len(record['selector']), text)
                except (TimeoutError, OSError) as e:
                    record['selector'].append(dict(slot_s=slot,started_s=start,ended_s=stamp(),
                        error=str(e),timeout=isinstance(e, TimeoutError) or isinstance(getattr(e,'reason',None),TimeoutError),miss=True,context=text))
                    raise
                end = stamp()
                record['selector'].append(dict(slot_s=slot, started_s=start, ended_s=end,
                    context=text, late_s=max(0., start-slot), **row))
                if end >= slot+20:
                    record['selector'][-1]['miss'] = True
                    record['cadence_missed'] = True
        except BaseException as e:
            errors.append(f'selector {type(e).__name__}: {e}')
            stop.set()
    selector_thread = threading.Thread(target=selectors, daemon=True)
    selector_thread.start()
    slot, last, camera_failed = 0, None, 0  # consecutive slots without a new frame (H4)
    try:
        while slot < duration and not stop.is_set():
            ops.guard()
            now = stamp()
            skipped, slot = due_frames(now, slot, duration)
            for missed in skipped:
                record['frames_due']['320'] += 1
                record['frames_skipped']['320'] += 1
                if missed % 5 == 0:
                    record['frames_due']['640'] += 1
                    record['frames_skipped']['640'] += 1
            if slot >= duration:
                break
            if stop.wait(max(0., t0+slot-ops.now())):
                break
            large = slot % 5 == 0
            record['frames_due']['320'] += 1
            record['frames_due']['640'] += int(large)
            with lock:
                busy = shared['m2_thread'] is not None and shared['m2_thread'].is_alive()
            if name == 'L3' and busy:
                record['frames_skipped']['320'] += 1
                record['frames_skipped']['640'] += int(large)
                slot += 1
                continue
            r = ops.frame(last)
            if r['status'] == 'missing' or r['status'] == 'ok' and last is not None and r['frame'] <= last:
                # H4: no new frame by the deadline: a skipped slot (cadence miss), a dead camera only 3 times in a row.
                camera_failed += 1
                record['camera_misses'].append(dict(slot_s=slot, status='repeat' if r['status'] == 'ok' else r['status']))
                record['camera_late'] += 1
                record['frames_skipped']['320'] += 1
                record['frames_skipped']['640'] += int(large)
                record['cadence_missed'] = True
                if camera_failed >= CAMERA_MISS_STOP:
                    raise RuntimeError(f'camera failed: no new frame in {camera_failed} consecutive slots: {record["camera_misses"][-camera_failed:]}')
                slot += 1
                continue
            if r['status'] != 'ok':
                raise RuntimeError('camera failed/repeated frame: '+r['status'])
            camera_failed = 0
            last = r['frame']
            record['reads'].append(dict(t=stamp(), frame=last, age_s=r['age_s']))
            record['camera_late'] += int(r['age_s'] > 1.35 or stamp()-slot > .35)
            for size in ([320,640] if large else [320]):
                a = stamp()
                boxes = ops.detect(r['image'], size)
                record['yolo'].append(dict(slot_s=slot, started_s=a, ended_s=stamp(),
                    size=size, ms=(stamp()-a)*1000, n_detections=len(boxes)))
                if size == 640:
                    with lock:
                        shared['frame'] = (r['image'], boxes, last)
                    ready.set()
                    if name in M2_EVERY and slot % M2_EVERY[name] == 0:
                        with lock:
                            busy = shared['m2_thread'] is not None and shared['m2_thread'].is_alive()
                        if busy:
                            record['cadence_missed'] = True
                            continue
                        thread = threading.Thread(target=m2, args=(r['image'],boxes,last,slot), daemon=True)
                        with lock:
                            shared['m2_thread'] = thread
                        thread.start()
            slot += 1
        if not stop.is_set():
            stop.wait(max(0., t0+duration-ops.now()))
        record['duration_s'] = stamp()
    finally:
        stop.set()
        selector_thread.join(timeout=25)
        with lock:
            thread = shared['m2_thread']
        if thread:
            thread.join(timeout=120)
        if selector_thread.is_alive() or thread and thread.is_alive():
            raise RuntimeError('worker still running; session must stop')
        record['duration_s'] = stamp()
    if errors:
        raise RuntimeError('; '.join(errors))
    # Calls crossing the block boundary are cadence failures, never silent valid timing.
    if any(r['ended_s'] > duration for r in record['m2']+record['selector']+record['yolo']):
        record['cadence_missed'] = True
    expected_m2 = 0 if name == 'L0' else math.ceil(duration/20) if name == 'L2' else math.ceil(duration/M2_EVERY[name])
    if len(record['m2']) != expected_m2:
        record['cadence_missed'] = True
    if len(record['selector']) != math.ceil(duration/20):
        record['cadence_missed'] = True
    if name != 'L3' and sum(record['frames_skipped'].values()):
        record['cadence_missed'] = True
    summarize(record)


def summarize(record):
    for key in ('reads','yolo','m2','selector'):
        record.setdefault(key,[])
    record.setdefault('frames_due',{'320':0,'640':0})
    record.setdefault('frames_skipped',{'320':0,'640':0})
    record.setdefault('camera_late',0)
    record['summary'] = dict(
        yolo={str(size):dict(due=record['frames_due'][str(size)],
               done=sum(r['size']==size for r in record['yolo']),
               skipped=record['frames_skipped'][str(size)],
               **stats([r for r in record['yolo'] if r['size']==size])) for size in (320,640)},
        camera_frames_delivered=len(record['reads']), camera_frames_late=record['camera_late'],
        camera_slots_skipped=len(record.get('camera_misses',[])),
        max_live_boxes=max((r['n_detections'] for r in record['yolo'] if r['size']==640), default=None),
        m2=dict(calls=sum(r['inference_executed'] for r in record['m2']), slots=len(record['m2']),
                live_calls=sum(r.get('scene')=='live' for r in record['m2']),
                fallback_calls=sum(r.get('scene')=='fallback' for r in record['m2']),
                truncated_slots=sum(r.get('truncated_boxes',0)>0 for r in record['m2']),
                insufficient_box_slots=sum(not r['inference_executed'] for r in record['m2']), **stats(record['m2']), inference_stats=stats([r for r in record['m2'] if r['inference_executed']], 'inference_ms')),
        selector=dict(calls=len(record['selector']), **stats(record['selector']),
                      prompt_tokens=[r.get('prompt_tokens') for r in record['selector']],
                      misses=sum(bool(r.get('miss')) for r in record['selector']),
                      timeouts=sum(bool(r.get('timeout')) for r in record['selector'])))


class MockOps:
    """Portable scheduling dry-run. No root, model, camera, server, idle or gates."""
    now = staticmethod(time.monotonic)
    guard = staticmethod(lambda: None)
    def __init__(self, name):
        self.name, self.number = name, 0
    def frame(self, last):
        self.number += 1
        return dict(status='ok', frame=self.number, age_s=.1, image=None)
    def detect(self, image, size):
        time.sleep(.005)
        return [dict(class_name='person', box_xyxy=[0,0,10,20]),
                dict(class_name='chair', box_xyxy=[20,0,40,20])]
    def fallback(self):
        return None, self.detect(None,640)
    def relate(self, image, boxes):
        time.sleep(1.2 if self.name=='L3' else .01)
        return [dict(subject='person',subject_idx=0,predicate='next_to',object='chair',object_idx=1,
                     score=.9,threshold=.5,**{'pass':True})], 10.
    def select(self, index, text):
        time.sleep(.01)
        return dict(ms=10., prompt_tokens=100, miss=False, timeout=False)


class LiveOps:
    now = staticmethod(time.monotonic)
    def __init__(self, cr, sb, detector, ort, caller, tids, cluster, cam, server, selector, cases, guard):
        self.cr,self.sb,self.detector,self.ort,self.caller,self.tids,self.cluster = cr,sb,detector,ort,caller,tids,cluster
        self.cam,self.server,self.selector,self.cases,self.guard = cam,server,selector,cases,guard
    def frame(self, last):
        # Poll a repeated or missing (absent/older than 2 s) frame until the deadline, as power_map.frame_loop
        # retries repeats (H4); 'bad', 'other_session' or any other status returns at once (fails closed).
        deadline = time.monotonic()+FRAME_WAIT_S
        while True:
            r = self.cr.read_frame(self.cr.FRAME_DIR, session=self.cam['session'])
            new = r['status'] == 'ok' and (last is None or r['frame'] > last)
            if new or r['status'] not in ('ok', 'missing') or time.monotonic() >= deadline:
                return r
            time.sleep(.02)
    def detect(self, image, size):
        self.sb.check_pinning(self.ort['worker_tids']+[self.ort['caller_tid']], {4,5})
        return self.detector.detect(image, size)
    def relate(self, image, boxes):
        self.sb.check_pinning(self.tids, self.cluster)
        # Head requires 2..32 boxes; run_cycle already chose the fallback (< 2) or the top 32 (H3).
        if len(boxes) < 2:
            raise RuntimeError('M2 requires fallback selection before inference')
        if len(boxes) > 32:
            raise RuntimeError('M2 needs at most 32 boxes; never truncate')
        rows, ms = self.caller.detect(image, boxes)
        self.sb.check_pinning(self.tids, self.cluster)
        return rows, ms
    def fallback(self):
        return self.fallback_input
    def select(self, index, text):
        case = copy.deepcopy(self.cases[index % len(self.cases)])
        if text is not None:
            state=case['situation']
            case['situation'] = (state if isinstance(state,str) else json.dumps(state,ensure_ascii=False,indent=1))+'\n\n'+text
        row = self.cr.select(self.selector, case)
        row.pop('correct', None)  # Context accuracy is explicitly not evaluated.
        return dict(**row, miss=False, timeout=False)


def bounded_selector(cr):
    import urllib.request
    def post(path, body):
        req = urllib.request.Request(f'http://127.0.0.1:{cr.PORT}'+path,json.dumps(body).encode(), {'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read())
    # Preserve S1O prompt/scoring; each transport request is bounded, from make_selector's tokenize calls on (M5).
    return cr.make_selector(post)


def live_block(name, out, server, idle, dry=False, duration=180, mark=None):
    """dry=True is the live owner rehearsal: real hardware, shortened, never a result.
    mark: READ_RETRIES index from which re-reads count for this block (main's capacity check just before it)."""
    v,sb,pm,cr,_ = rt.imports()
    measured = {4,5} | (v.CLUSTERS[LAYOUTS[name]][0] if LAYOUTS[name] else set())
    monitor = None
    layout = {'policies':{}}
    record = dict(block=name, validity='NOT VALID — INCOMPLETE', fast=[], dumps=[], power=[], memory=[],
                  monitor_errors=[], cleanup_errors=[], cpuinfo_max_khz=layout['policies'])
    detector=caller=head=None
    fast_monitor=None
    root_scope=None
    stop=threading.Event()
    threads=[]
    mark=len(rt.READ_RETRIES) if mark is None else mark
    beat=[None]  # M5: main-loop heartbeat, armed from the bracket to the end of run_cycle
    def guard():
        if beat[0] is not None:
            beat[0]=time.monotonic()
        rt.check_cores(cr,'campaign')
        if record['monitor_errors']:
            raise RuntimeError('monitor failed: '+rt.causes(record['monitor_errors']))
        if record['fast'] or record['dumps']:
            if hit := cr.block_limit(time.monotonic(), began, record['fast'], record['dumps']):
                raise RuntimeError('thermal/sensor stop: '+str(hit))
    def shared_check():
        if beat[0] is not None and (age:=time.monotonic()-beat[0]) > rt.HEARTBEAT_S:
            message=f'main loop heartbeat {age:.1f} s old (> {rt.HEARTBEAT_S} s): SIGTERM to self, normal cleanup'
            print(message,flush=True)
            os.kill(os.getpid(),signal.SIGTERM)  # the handler raises SystemExit in the main thread
            raise RuntimeError(message)
        rt.clear_processes(server.proc.pid)
        if not rt.server_ok(server):
            raise RuntimeError('llama-server unhealthy')
        return dict(t=time.monotonic(),ok=True)
    def worker(period, read, rows, phase=0.):
        try:
            tid=threading.get_native_id()
            os.sched_setaffinity(tid,mask)
            if os.sched_getaffinity(tid)!=mask:
                raise RuntimeError('monitor pinning failed')
            if stop.wait(max(0., began+phase-time.monotonic())):
                return
            def checked():
                if os.sched_getaffinity(tid)!=mask:
                    raise RuntimeError('monitor affinity changed')
                return read()
            cr.monitor_loop(period,checked,rows,stop)
            if not stop.is_set():
                raise RuntimeError('monitor stopped unexpectedly')
        except BaseException as e:
            record['monitor_errors'].append(f'{type(e).__name__}: {e}')
            stop.set()
    lmk_since=time.time()
    began=time.monotonic()
    try:
        monitor=rt.prepare(sb,measured)
        shell,mask,pid,layout=monitor
        record['setup_affinity']=getattr(shell,'campaign_setup_affinity',None)
        root_scope=rt.root_mask_scope(cr,sb,mask,measured)
        root_scope.__enter__()
        record['root_service_mask']=sorted(mask)
        record['monitor_phases_s']={'fast':.11,'skin_status':.37,'memory_pss':1.31,'shared_checks':.71}
        record['monitor_periods_s']={'fast':1.,'skin_status':4.87,'memory_pss':5.13,'shared_checks':5.23}
        record['shared_checks']=[]
        record['cpuinfo_max_khz']=layout['policies']
        detector,ort=pm.build_detector('mid')
        cluster=v.CLUSTERS[LAYOUTS[name]][0] if LAYOUTS[name] else set()
        tids=[]
        if LAYOUTS[name]:
            head,caller,tids=v.build('fp32',LAYOUTS[name])
        record['ort']=ort
        record['m2_cluster']=dict(cpus=sorted(cluster),threads=v.CLUSTERS[LAYOUTS[name]][1]) if LAYOUTS[name] else None
        # A dedicated root reader owns each shell, so no request interleaving.
        fast_monitor=rt.prepare(sb,measured)
        record['fast_setup_affinity']=getattr(fast_monitor[0],'campaign_setup_affinity',None)
        rt.pin_main(mask)
        guard()
        shared_check()
        rt.battery_sample(sb)
        rt.fast_check(cr,fast_monitor,sb,measured)
        fallback_input = rt.fallback_input() if LAYOUTS[name] else None
        cam=rt.camera_start(pm)
        record['camera']=cam
        record['diagnostics_start']=diag.snapshot(cr,True)
        record['skin_start']=rt.dump_check(cr)
        selector=bounded_selector(cr)
        cases=cr.load_cases()
        t0=time.monotonic()
        began=t0
        record['power'].append(rt.counted(lambda:power.sample(shell,cr.BATTERY,t0)))
        # Give the first power sample a negative receipt timestamp to bracket t=0 exactly.
        t0=time.monotonic()
        offset=t0-began
        for r in record['power']:
            r['t']-=offset; r['t_start']-=offset
        began=t0
        lmk_since=time.time()
        record['lmk_window_epoch_s']=[lmk_since,lmk_since+duration]
        beat[0]=time.monotonic()
        for period,read,rows,phase in (
            (1,lambda:rt.counted(lambda:rt.fast_check(cr,fast_monitor,sb,measured)),record['fast'],.11),
            (4.87,lambda:rt.read_dump(cr),record['dumps'],.37),
            (5.13,lambda:rt.counted(lambda:rt.memory_sample(cr,server.proc.pid,True)),record['memory'],1.31),
            (5.23,lambda:rt.counted(shared_check),record['shared_checks'],.71)):
            thread=threading.Thread(target=worker,args=(period,read,rows,phase),daemon=True)
            threads.append(thread); thread.start()
        thread=threading.Thread(target=power.sampler,args=(shell,cr.BATTERY,t0,mask,measured,stop,
            record['power'],record['monitor_errors'],lambda:rt.verify_monitor(sb,monitor,measured)),daemon=True)
        threads.append(thread); thread.start()
        ops=LiveOps(cr,sb,detector,ort,caller,tids,cluster,cam,server,selector,cases,guard)
        ops.fallback_input=fallback_input
        # Rehearsal: force the M2 slot whose context the 20 s selector reads (L2 runs M2 at each selector slot,
        # L1 at 0/10/20 s, L3/L4 every 5 s).
        ops.force_fallback_slot=(0 if name=='L2' else 10 if name=='L1' else 15) if dry else None
        # Use exactly the same origin for cadence and power receipt accounting.
        ops.now=lambda:time.monotonic()
        ops.origin=t0
        record['cycle_origin']=t0
        run_cycle(name,duration,ops,record)
        beat[0]=None
        guard()
        shared_check()
        record['skin_end']=rt.dump_check(cr)
        record['diagnostics_end']=diag.snapshot(cr,True)
        record['failure_kind']='cadence' if record.get('cadence_missed') else None
        record['validity']='NOT VALID — INCOMPLETE' if record.get('cadence_missed') else 'VALID'
    except BaseException as e:
        record['error']=f'{type(e).__name__}: {e}'
        record['failure_kind']='shared'
    finally:
        beat[0]=None
        stop.set()
        for thread in threads:
            thread.join(timeout=60)
        if any(t.is_alive() for t in threads):
            record['cleanup_errors'].append('monitor still running')
        # A failed thermal read, also one completing during the joins, stops the session.
        if bad:=rt.thermal_row_failures(record['dumps']):
            record['monitor_errors'].append('thermal rows failed: '+'; '.join(bad))
        for action in (lambda:rt.stop_camera(cr), lambda:rt.release(caller), lambda:rt.release(detector),
                       lambda:monitor[0].close() if monitor else None, lambda:fast_monitor[0].close() if fast_monitor else None):
            try: action()
            except BaseException as e: record['cleanup_errors'].append(f'{type(e).__name__}: {e}')
        if record['cleanup_errors'] or record['monitor_errors']:
            record['validity']='NOT VALID — INCOMPLETE'; record['failure_kind']='shared'
        try:
            record['lmk']=lmk_window(cr,lmk_since,lmk_since+duration)
        except BaseException as e:
            record['cleanup_errors'].append('LMK read: '+str(e))
            record['validity']='NOT VALID — INCOMPLETE'; record['failure_kind']='shared'
        if record.get('lmk',{}).get('ok') is not True:
            record['validity']='NOT VALID — INCOMPLETE'; record['failure_kind']='shared'
            record.setdefault('error',f"LMK window not ok: logcat_rc {record.get('lmk',{}).get('logcat_rc')}")
        if root_scope:
            root_scope.__exit__(None,None,None)
        origin=record.get('cycle_origin',began)
        for key in ('gate','skin_start','skin_end'):
            if key in record:
                for stamp_key in ('t','t_start'):
                    if stamp_key in record[key]:record[key][stamp_key]-=origin
        for rows in (record['fast'],record['dumps'],record['memory'],record.get('shared_checks',[])):
            for r in rows:
                r['t']-=origin
                if 't_start' in r:r['t_start']-=origin
        record['power_summary']=power.summary(record['power'],record.get('m2',[]),duration)
        if record['power_summary']['mean_battery_w'] is None:
            record['validity']='NOT VALID — INCOMPLETE'; record['failure_kind']='shared'
            record.setdefault('error',f"power coverage missing: covered {record['power_summary']['covered_s']:.2f} s of {duration} s, "
                                      f"gaps > 1.5 s: {record['power_summary']['gaps']}")
        if record.get('duration_s'):
            record['caps_by_policy']=pm.capped_by_policy({**record,'duration_s':duration})
        summarize(record)
        in_block_memory=[r for r in record['memory'] if 0 <= r['t'] <= duration]
        record['mem_available_min_mib']=min((r['mem_available_mib'] for r in in_block_memory),default=None)
        record['llama_server_pss_kb']=[r.get('pss_kb',{}).get('llama_server') for r in in_block_memory]
        failed=[r for r in record['memory'] if memory_row_bad(r)]
        if not record['memory'] or failed:
            record['validity']='NOT VALID — INCOMPLETE'; record['failure_kind']='shared'
            record.setdefault('error','memory/battery read failed: '+(rt.causes([memory_row_cause(r) for r in failed]) if failed else 'no rows'))
        if rt.retry_check(record,mark) and record['validity']=='VALID':
            record['validity']='NOT VALID — READ RETRIES'  # H2 cap: not a session stop
        if record.get('failure_kind')=='shared':
            record['screen_state_at_failure']=rt.screen_state(cr)  # H5: a lock or lost foreground shows here
        if dry:
            record['rehearsal_block_validity']=record['validity']
            record['validity']=REHEARSAL
        write(out,record)
    return record


def memory_row_bad(r):
    """A memory row that invalidates its phase (the base predicate): root failure, PSS failure, or a battery status other
    than Discharging."""
    return r.get('root_rc') != 0 or r.get('pss_error') or r.get('battery_status') != 'Discharging'


def memory_row_cause(r):
    return r.get('pss_error') or r.get('error') or r.get('battery_status')


def lmk_window(cr, since, until):
    # H2 (FIX4C): only an answer without output (lmk_lines' answer_blank: the whole su answer was blank) is read again;
    # every other answer is judged below as in the base.
    raw = rt.reread('logcat', lambda: cr.lmk_lines(since), lambda r: 'empty logcat answer' if r.get('answer_blank') else None)
    kept, outside, unknown = [], [], []
    for line in raw.get('lines', []):
        try:
            stamp = float(line.split()[0])
            if not math.isfinite(stamp):raise ValueError('nonfinite logcat timestamp')
        except (ValueError, IndexError):
            unknown.append(line)
            continue
        (kept if since <= stamp < until else outside).append(line)
    return dict(**{k:v for k,v in raw.items() if k not in ('lines','n_lines','n_kills','ok')},
                lines=kept,n_lines=len(kept),n_kills=sum(bool(cr.LMK_KILL_RE.search(l)) for l in kept),
                outside_block_lines=outside,unparsed_timestamp_lines=unknown,
                ok=raw.get('ok') is True and not unknown)


def write(path,data):
    """Atomic (M1): a kill mid-write leaves the previous complete file, never a truncated one."""
    text=json.dumps(data,indent=2,allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name,path)


START_BAND_C = 1.5  # start skin comparability band around T_ref (also the D2 cooling target)


def compare_start(record, reference):
    temperature = record['skin_start']['skin']
    reference = temperature if reference is None else reference
    record['T_ref_c'] = reference
    record['start_temperature_delta_c'] = temperature-reference
    if abs(temperature-reference) > START_BAND_C:
        record['start_temperature_comparability']='NOT COMPARABLE — START TEMP'
        if record['validity']=='VALID':
            record['validity']='NOT COMPARABLE — START TEMP'
    else:
        record['start_temperature_comparability']='COMPARABLE'
    return reference


def rest(label, duration, server, record, camera_on=False, dry=False, t_ref=None, cool_max=0, mark=None):
    """No inference objects exist here; only Gemma and measurement remain resident.

    D2 (owner): with t_ref set, a pause whose newest skin is above t_ref+START_BAND_C after `duration` goes on
    (same checks, still sampled) until it is within, for at most cool_max s more; record['cooling'] has the
    extra seconds. A skin below the band never waits (the block is labelled by compare_start).
    mark: READ_RETRIES index from which re-reads count for this phase (main's capacity check just before it).
    """
    record.update(label=label, camera_on=camera_on, planned_s=duration,
                  fast=[], dumps=[], memory=[], power=[], monitor_errors=[], cleanup_errors=[])
    if dry:
        record['label']='NOT VALID — DRY RUN (all hardware mocked): '+label
        time.sleep(duration)
        record['duration_s']=duration
        return record
    v,sb,pm,cr,_=rt.imports()
    mark=len(rt.READ_RETRIES) if mark is None else mark
    monitor=None; fast_monitor=None; scope=None; stop=threading.Event(); threads=[]
    began=time.monotonic()
    target=None if t_ref is None else t_ref+START_BAND_C
    def skin():
        return next((d['skin'] for d in reversed(record['dumps'])), None)
    def skin_age():  # the newest worker row is usually <= 4.87 s old; more if a read overran
        return next((time.monotonic()-d['t'] for d in reversed(record['dumps'])), None)
    def until():
        """End of the current wait; None once the pause, with any D2 extension, is over."""
        elapsed=time.monotonic()-began
        if elapsed<duration:
            return began+duration
        if target is not None and elapsed<duration+cool_max and (now:=skin()) is not None and now>target:
            return began+duration+cool_max
        return None
    def fail(message):  # M4: the cause in the message; H5: screen state in the record
        record['error']='RuntimeError: '+message
        record['screen_state_at_failure']=rt.screen_state(cr)
        return RuntimeError(message)
    try:
        diag.battery(cr,25)
        rt.stop_camera(cr)
        monitor=rt.prepare(sb,{4,5})
        shell,mask,pid,layout=monitor
        # Power shell is owned solely by its sampler; other reads use the scoped root wrapper.
        # A second persistent shell for the reused fast parser, ready before the bracket (H1, as in live_block).
        fast_monitor=rt.prepare(sb,{4,5})
        rt.pin_main(mask)
        scope=rt.root_mask_scope(cr,sb,mask,{4,5});scope.__enter__()
        record['cpuinfo_max_khz']=layout['policies']
        if camera_on:
            record['camera']=rt.camera_start(pm)
        record['diagnostics_start']=diag.snapshot(cr,camera_on)
        def worker(period,read,rows):
            try:
                tid=threading.get_native_id();os.sched_setaffinity(tid,mask)
                def checked():
                    if os.sched_getaffinity(tid)!=mask:
                        raise RuntimeError('pause monitor affinity changed')
                    return read()
                cr.monitor_loop(period,checked,rows,stop)
            except BaseException as e:
                record['monitor_errors'].append(str(e));stop.set()
        def fast():return rt.fast_check(cr,fast_monitor,sb,{4,5})
        record['fast_monitor']=True
        record['power'].append(rt.counted(lambda:power.sample(shell,cr.BATTERY,began)))
        origin=time.monotonic()
        offset=origin-began
        for row in record['power']:
            row['t']-=offset;row['t_start']-=offset
        began=origin
        for period,read,rows in ((1,lambda:rt.counted(fast),record['fast']),
            (4.87,lambda:rt.read_dump(cr),record['dumps']),
            (5.13,lambda:rt.counted(lambda:rt.memory_sample(cr,server.proc.pid,camera_on)),record['memory'])):
            thread=threading.Thread(target=worker,args=(period,read,rows),daemon=True)
            threads.append(thread);thread.start()
        thread=threading.Thread(target=power.sampler,args=(shell,cr.BATTERY,began,mask,{4,5},stop,
            record['power'],record['monitor_errors'],lambda:rt.verify_monitor(sb,monitor,{4,5})),daemon=True)
        threads.append(thread);thread.start()
        # M3: battery status is checked by the power sampler and memory rows, skin/status by the thermal worker
        # (block_limit below); the loop's own battery_sample/dump_check reads were redundant and are gone.
        while (end:=until()) is not None:
            if record['monitor_errors']:raise RuntimeError('pause monitor failed: '+rt.causes(record['monitor_errors']))
            rt.clear_processes(server.proc.pid);rt.check_cores(cr,'pause')
            if not rt.server_ok(server):raise RuntimeError('server unhealthy during pause')
            if hit:=cr.block_limit(time.monotonic(),began,record['fast'],record['dumps']):
                raise RuntimeError('pause thermal/sensor stop: '+str(hit))
            if stop.wait(min(5,max(0,end-time.monotonic()))):
                raise RuntimeError('pause sampler stopped: '+rt.causes(record['monitor_errors']))
        if t_ref is not None:
            now=skin()
            record['cooling']=dict(t_ref_c=t_ref,target_skin_c=target,max_extra_s=cool_max,
                                   extra_s=max(0.,time.monotonic()-began-duration),end_skin_c=now,end_skin_age_s=skin_age(),
                                   reached=now is not None and now<=target,
                                   below_band=now is not None and now<t_ref-START_BAND_C)
        record['diagnostics_end']=diag.snapshot(cr,camera_on)
    except BaseException as e:
        record['error']=f'{type(e).__name__}: {e}'
        record['screen_state_at_failure']=rt.screen_state(cr)
        raise
    finally:
        stop.set()
        for thread in threads:thread.join(timeout=60)
        if any(t.is_alive() for t in threads):record['cleanup_errors'].append('pause monitor still running')
        if bad:=rt.thermal_row_failures(record['dumps']):record['monitor_errors'].append('thermal rows failed: '+'; '.join(bad))
        for action in (lambda:rt.stop_camera(cr),lambda:monitor[0].close() if monitor else None,
                       lambda:fast_monitor[0].close() if fast_monitor else None):
            try:action()
            except BaseException as e:record['cleanup_errors'].append(str(e))
        if scope:scope.__exit__(None,None,None)
        record['duration_s']=time.monotonic()-began
        for rows in (record['fast'],record['dumps'],record['memory']):
            for row in rows:
                row['t']-=began
                if 't_start' in row:row['t_start']-=began
        record['power_summary']=power.summary(record['power'],[],duration)
        sampled=duration+record.get('cooling',{}).get('extra_s',0.)
        if sampled>duration:  # D2 extension: fixed-600 s summary stays comparable; the whole interval is validated too
            record['power_summary_with_cooling']=power.summary(record['power'],[],sampled)
        if monitor:record['caps_by_policy']=pm.capped_by_policy({**record,'duration_s':duration})
        if rt.retry_check(record,mark):
            record['validity']=record['read_retry_check']  # H2 cap: this phase NOT VALID, no session stop
        # L5: a cleanup failure never replaces an error already propagating (BatteryStop, CoresLost, ...).
        if record['cleanup_errors'] and 'error' not in record:
            raise fail('pause cleanup failed: '+rt.causes(record['cleanup_errors']))
    # A reader can fail after the last loop check, while the workers are joined.
    if record['monitor_errors']:
        raise fail('pause monitor failed after deadline: '+rt.causes(record['monitor_errors']))
    for key in ('power_summary','power_summary_with_cooling'):
        if key in record and record[key]['mean_battery_w'] is None:
            summary=record[key]
            raise fail(f"pause power coverage missing ({key}): covered {summary['covered_s']:.2f} s, gaps > 1.5 s: {summary['gaps']}")
    failed=[r for r in record['memory'] if memory_row_bad(r)]
    if not record['memory'] or failed:
        raise fail('pause memory/battery read failed: '+(rt.causes([memory_row_cause(r) for r in failed]) if failed else 'no rows'))
    return record


def preflight(names, out, result, resources):
    v,sb,pm,cr,_=rt.imports()
    cr.require_native()
    rt.clear_processes()
    cr.check_cores('preflight')
    rt.battery_sample(sb)
    result['battery_start_percent']=diag.battery(cr,80)
    result['fallback_hashes']=rt.fallback_hashes()
    fallback_image,fallback_boxes=rt.fallback_input()
    result['fallback_input_check']=dict(size=list(fallback_image.size),boxes=len(fallback_boxes),loaded=True)
    del fallback_image,fallback_boxes
    result['hashes']=rt.require_hashes(v,cr,pm)
    result['installed_robotcam']=rt.camera_version(cr)
    rt.dump_check(cr)
    screen=rt.screen(sb); resources['screen']=screen; screen.start()
    result['layout_checks']=[]
    # Subsets still execute the L0 warm-up, so validate that configuration too.
    for name in dict.fromkeys(['L0',*names]):
        detector=caller=head=None; monitor=None
        try:
            measured={4,5}|(v.CLUSTERS[LAYOUTS[name]][0] if LAYOUTS[name] else set())
            monitor=rt.prepare(sb,measured)
            rt.fast_check(cr,monitor,sb,measured)
            with rt.root_mask_scope(cr,sb,monitor[1],measured):
                rt.dump_check(cr)
                rt.battery_sample(sb)
                diagnostics=diag.snapshot(cr,False)
            detector,ort=pm.build_detector('mid')
            sb.check_pinning(ort['worker_tids']+[ort['caller_tid']],{4,5})
            if LAYOUTS[name]:
                head,caller,tids=v.build('fp32',LAYOUTS[name])
                sb.check_pinning(tids,v.CLUSTERS[LAYOUTS[name]][0])
            result['layout_checks'].append(dict(layout=name,safe_cpus=sorted(monitor[1]),policies=monitor[3]['policies'],ort=ort,
                diagnostics=diagnostics,
                setup_affinity=getattr(monitor[0],'campaign_setup_affinity',None)))
        finally:
            rt.release(caller); head=None
            rt.release(detector)
            if monitor:monitor[0].close()
    # Own exactly one server for this session, using the existing flags; no attached foreign server.
    server=cr.Server(out.with_name(out.stem+'_llama-server.log')); resources['server']=server
    if not rt.server_ok(server):raise RuntimeError('server unhealthy')
    result['server_command']=server.cmd
    result['server_load_s']=server.load_s
    selector=bounded_selector(cr)
    selector.decide(*cr.WARMUP)
    if not cr.load_cases():raise RuntimeError('no selector cases')
    try:
        result['camera_check']=rt.camera_start(pm)
        r=cr.read_frame(cr.FRAME_DIR,session=result['camera_check']['session'])
        if r['status']!='ok':raise RuntimeError('camera not ready')
    finally:
        rt.stop_camera(cr)
    rt.clear_processes(server.proc.pid)
    rt.battery_sample(sb); cr.check_cores('pre-idle'); rt.dump_check(cr)


# Phase seconds: idle sub-phases (camera OFF, OFF, ON), warm-up, each pause, each block, D2 extra cooling at most.
FULL = dict(idle=(180,60,60), warm=180, pause=600, block=180, cool=300)
OWNER_REHEARSAL = dict(idle=(6,6,6), warm=20, pause=6, block=25, cool=6)
MOCK = dict(idle=(.1,.1,.1), warm=1, pause=.1, block=6, cool=.1)


def coverage(result):
    """Which session paths the rehearsal actually reached; read from the evidence itself."""
    runs=([result['warmup']] if 'warmup' in result else [])+result['blocks']
    done=lambda rows:sum('duration_s' in r and not (r.get('error') or r.get('monitor_errors') or r.get('cleanup_errors')) for r in rows)
    per_block=[dict(block=b['block'],m2_live_calls=b.get('summary',{}).get('m2',{}).get('live_calls',0),
                    m2_fallback_calls=b.get('summary',{}).get('m2',{}).get('fallback_calls',0),
                    selector_calls=len(b.get('selector',[])),
                    selector_fallback_contexts=sum(str(r.get('context')).startswith('(fallback scene)') for r in b.get('selector',[])),
                    max_live_boxes=b.get('summary',{}).get('max_live_boxes'),read_retries=len(b.get('read_retries',[])))
               for b in result['blocks']]
    phases=[('setup',result.get('setup',{}))]+[(r.get('label') or r.get('block'),r) for r in result['idle_phases']+result['pauses']+runs]
    cov=dict(setup_complete=result.get('setup_complete',False),idle_phases_done=done(result['idle_phases']),
             warmup_done='warmup' in result,pauses_done=done(result['pauses']),blocks=per_block,
             read_retries=len(result.get('read_retries',[])),
             read_retry_over_cap=[label for label,r in phases if r.get('read_retry_check','OK')!='OK'],
             errors=[f"{b['block']}: {b.get('error') or b.get('monitor_errors') or b.get('cleanup_errors') or b.get('failure_kind')}"
                     for b in runs if b.get('error') or b.get('monitor_errors') or b.get('cleanup_errors') or b.get('failure_kind')=='shared']
                    +([f"session: {result['error']}"] if result.get('error') else [])
                    +([f"cleanup: {result['cleanup_error']}"] if result.get('cleanup_error') else []))
    # Every phase ran; each M2 layout ran live M2, fallback M2 and a selector with fallback context; every block
    # reported its largest YOLO-640 box count (H3); no phase exceeded the H2 re-read cap.
    cov['rehearsal_pass']=bool(cov['setup_complete'] and cov['idle_phases_done']==len(result['idle_phases'])==3 and cov['warmup_done']
        and cov['pauses_done']==len(result['plan']) and [b['block'] for b in per_block]==result['plan'] and not cov['errors']
        and not cov['read_retry_over_cap']
        and all(b['selector_calls'] and b['max_live_boxes'] is not None
                and (LAYOUTS[b['block']] is None or b['m2_live_calls'] and b['m2_fallback_calls'] and b['selector_fallback_contexts'])
                for b in per_block))
    return cov


def rest_status(r):
    if 'error' in r:
        return f"FAILED: {r['error'][:300]}; screen: {json.dumps(r.get('screen_state_at_failure'),ensure_ascii=False)[:300]}"
    cooling=r.get('cooling')
    return (f"{r.get('validity','ok')}, {r.get('duration_s',0):.1f} s, read re-reads {len(r.get('read_retries',[]))}"
            +(f", D2 extra cooling {cooling['extra_s']:.0f} s, end skin {cooling['end_skin_c']} C (target {cooling['target_skin_c']:.2f} C)" if cooling else ''))


def block_status(b):
    """L3: camera lateness and live/fallback M2 next to the label; H3: largest YOLO-640 box count."""
    s=b.get('summary',{})
    return (f"{b.get('result_validity',b['validity'])}; camera late {s.get('camera_frames_late')}, camera slots skipped "
            f"{s.get('camera_slots_skipped')}, M2 live {s.get('m2',{}).get('live_calls')} / fallback {s.get('m2',{}).get('fallback_calls')}, "
            f"max_live_boxes {s.get('max_live_boxes')}, read re-reads {len(b.get('read_retries',[]))}"
            +(f"; error: {b['error'][:300]}" if b.get('error') else '')
            +(f"; screen: {json.dumps(b['screen_state_at_failure'],ensure_ascii=False)[:300]}" if 'screen_state_at_failure' in b else ''))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--blocks')
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true',help='owner rehearsal on the phone: full live setup and every phase, shortened; NOT VALID')
    group.add_argument('--preflight',action='store_true')
    ap.add_argument('--mock',action='store_true',help='with --dry-run: portable mocks, no hardware (proot); NOT VALID')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv)
    if a.mock and not a.dry_run:ap.error('--mock requires --dry-run')
    mock=a.mock
    rehearsal=a.dry_run and not mock
    times=MOCK if mock else OWNER_REHEARSAL if rehearsal else FULL
    try:names=blocks(a.blocks)
    except ValueError as e:ap.error(str(e))
    stem=lambda suffix:a.output.with_name(a.output.stem+suffix)
    paths=[stem(f'_block_{i:02d}_{n}.json') for i,n in enumerate(names,1)]
    pause_paths=[stem(f'_pause_{i:02d}_{n}.json') for i,n in enumerate(names,1)]
    idle_paths=[stem(f'_idle_{i}.json') for i in (1,2,3)]
    warm_path=stem('_warmup_L0.json')
    if any(p.exists() for p in [a.output,stem('_llama-server.log'),warm_path,*paths,*pause_paths,*idle_paths]):ap.error('evidence exists; choose a fresh stem')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    idle=times['idle']
    minimum=sum(idle)+times['warm']+(times['pause']+times['block'])*len(names)
    cooling=(len(names)-1)*times['cool']  # D2: only pauses after the first measured block (T_ref) may extend
    print(f'Plan: {sum(idle):g} s idle ({idle[0]:g} s camera OFF + {idle[1]:g} s camera OFF + {idle[2]:g} s camera ON without inference), WARM-UP L0 {times["warm"]:g} s,',
          ','.join(f'{times["pause"]:g} s pause + {n} {times["block"]:g} s' for n in names),'; motors OFF.',flush=True)
    if rehearsal:print(f'{REHEARSAL}: {minimum/60:.1f} min scheduled (+ up to {cooling:g} s D2 cooling); about 12 min including setup, model builds, camera transitions, diagnostics and cleanup.',flush=True)
    else:print(f'Estimate {minimum/60:.1f} min scheduled; full default about 95–100 min including setup/loads/cleanup, plus up to {cooling/60:.0f} min of D2 cooling.',flush=True)
    result=dict(label='NOT VALID — DRY RUN (all hardware mocked)' if mock else REHEARSAL if rehearsal else 'IN PROGRESS — NOT VALID',
        plan=names,minimum_s=minimum,continuation_rule=CONTINUATION,blocks=[],pauses=[],idle_phases=[],unrun_blocks=names[:],
        detector_note='Explicit task: 320 EVERY frame plus 640 every 5 s; #128 SizePolicy replaced 320 on 640 frames.',
        context_example='M2 relations:\n- person [0] next to chair [1].',context_accuracy='NOT EVALUATED',
        thermal_retries=rt.THERMAL_RETRIES,read_retries=rt.READ_RETRIES)
    prefix='NOT VALID — REHEARSAL: ' if rehearsal else ''
    resources={}
    t_start=time.monotonic()
    def say(*words):
        print(f'[{time.monotonic()-t_start:7.1f} s]',*words,
              f'| read re-reads {len(rt.READ_RETRIES)}, thermal retries {len(rt.THERMAL_RETRIES)}',flush=True)
    def pause_phase(label,path,row,*args,**kw):
        """M4 START/END lines; M1: each idle/pause phase also in its own file, written on failure too."""
        say('START',label)
        try:
            rest(label,*args,row,**kw)
        finally:
            write(path,row)
            say('END',label,rest_status(row))
    lock_path=rt.HOME/'.cache/campaign_p1.lock' if not mock else Path(tempfile.gettempdir())/'campaign_p1_dry.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(a.output,result)
            if not mock:
                # Preflight, rehearsal and session execute this exact setup, with no mode argument or early branch.
                say('START','setup (preflight)')
                setup=result['setup']={};mark=len(rt.READ_RETRIES)
                try:
                    preflight(names,a.output,result,resources)
                except BaseException as e:
                    rt.retry_check(setup,mark)
                    say('END','setup (preflight)',f'FAILED: {type(e).__name__}: {e}'[:300])
                    raise
                rt.retry_check(setup,mark)
                result['setup_complete']=True
                say('END','setup (preflight)',setup['read_retry_check'])
                if a.preflight:
                    result['label']='PREFLIGHT ONLY — NO TIMING'
                    return
                v,sb,pm,cr,_=rt.imports()
                server=resources['server']
            server=resources.get('server')
            for label,seconds,on,path in zip(('IDLE','IDLE CAMERA OFF','IDLE CAMERA ON — NO INFERENCE'),idle,(False,False,True),idle_paths):
                row={};result['idle_phases'].append(row)
                pause_phase(prefix+label,path,row,seconds,server,camera_on=on,dry=mock)
                write(a.output,result)
            # Each between-phase capacity check counts for the phase it guards (H2: from `mark` on).
            mark=len(rt.READ_RETRIES)
            if not mock:diag.battery(cr,25)
            say('START','WARM-UP L0')
            warm=dict(block='L0',validity='NOT VALID — DRY RUN (all hardware mocked)',mock_only=True) if mock else live_block('L0',warm_path,server,None,rehearsal,times['warm'],mark=mark)
            if mock:run_cycle('L0',times['warm'],MockOps('L0'),warm)
            warm['result_validity']=warm['validity'];warm['validity']='WARM-UP — NOT A RESULT'
            result['warmup']=warm;write(warm_path,warm);write(a.output,result)
            say('END','WARM-UP L0',block_status(warm))
            if warm.get('failure_kind') and not can_continue(warm):
                raise RuntimeError('warm-up invalidates later blocks: '+rt.causes([warm.get('error')] if warm.get('error') else [],warm.get('monitor_errors',[]),warm.get('cleanup_errors',[])))
            reference=None
            for i,(name,path,pause_path) in enumerate(zip(names,paths,pause_paths),1):
                mark=len(rt.READ_RETRIES)
                if not mock:diag.battery(cr,25)
                row={};result['pauses'].append(row)
                # D2: once T_ref exists, a pause ending above T_ref+1.5 C extends by at most times['cool'] s.
                pause_phase(prefix+'FIXED PAUSE',pause_path,row,times['pause'],server,dry=mock,t_ref=reference,cool_max=times['cool'],mark=mark)
                write(a.output,result)
                mark=len(rt.READ_RETRIES)
                if not mock:diag.battery(cr,25)
                say('START',f'block {i:02d} {name}')
                if mock:
                    b=dict(block=name,validity='NOT VALID — DRY RUN (all hardware mocked)',mock_only=True)
                    run_cycle(name,times['block'],MockOps(name),b)
                    b['power_summary']=power.summary([],[],times['block'])
                    write(path,b)
                else:
                    b=live_block(name,path,server,None,rehearsal,times['block'],mark=mark)
                    if 'skin_start' in b:reference=compare_start(b,reference)
                    result['T_ref_c']=reference
                    write(path,b)
                result['blocks'].append(b); result['unrun_blocks']=names[len(result['blocks']):]; write(a.output,result)
                say('END',f'block {i:02d} {name}',block_status(b))
                if b.get('failure_kind') and not can_continue(b):
                    raise RuntimeError('block invalidates later blocks: '+rt.causes([b.get('error')] if b.get('error') else [],b.get('monitor_errors',[]),b.get('cleanup_errors',[])))
            result['label']=('NOT VALID — DRY RUN (all hardware mocked)' if mock else REHEARSAL+' COMPLETE; inspect rehearsal_coverage'
                             if rehearsal else 'SESSION COMPLETE; inspect individual validity')
        except diag.BatteryStop as e:
            result.update(label='SESSION STOPPED — BATTERY BELOW 25%; remaining blocks UNRUN',error=str(e))
            if not mock:result['screen_state_at_failure']=rt.screen_state(rt.imports()[3])  # H5
        except BaseException as e:
            result.update(label='NOT VALID — SESSION INCOMPLETE',error=f'{type(e).__name__}: {e}')
            if not mock:result['screen_state_at_failure']=rt.screen_state(rt.imports()[3])  # H5
            raise
        finally:
            # L4: a second signal must not skip the server stop, the screen restore or the final write.
            handlers={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
            unwinding='error' in result  # a cause is recorded: an exception is propagating, or the handled battery stop
            failures=[]
            for key,action in (('server',lambda:resources['server'].stop()),('screen',lambda:resources['screen'].restore())):
                if key in resources:
                    try:action()
                    except BaseException as e:failures.append(f'{key}: {type(e).__name__}: {e}')
            if failures:
                result['cleanup_error']=rt.causes(failures)  # M4: the first two, each kept
                if not unwinding:  # L5: never replaces the cause already recorded (battery, cores, ...)
                    result.update(label='NOT VALID — SESSION INCOMPLETE',error='cleanup failed: '+result['cleanup_error'])
                if not mock and 'screen_state_at_failure' not in result:
                    result['screen_state_at_failure']=rt.screen_state(rt.imports()[3])  # H5
            if a.dry_run:
                result['rehearsal_coverage']=coverage(result)
                print('Coverage:',json.dumps(result['rehearsal_coverage'],ensure_ascii=False),flush=True)
            write(a.output,result)
            print(result['label'],'Evidence:',a.output,flush=True)
            for sig,handler in handlers.items():
                signal.signal(sig,handler)
            if a.preflight and failures and not unwinding:  # L5: a propagating exception stays the one raised
                raise RuntimeError('preflight cleanup failed: '+result['cleanup_error'])
    if result.get('cleanup_error'):raise SystemExit(1)


if __name__=='__main__':
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,lambda signum,frame: (_ for _ in ()).throw(SystemExit(128+signum)))
    main()
