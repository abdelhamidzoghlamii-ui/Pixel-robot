"""Recompute campaign results from archived JSON; never starts hardware."""
import math
import statistics

import campaign_data as data
import power


CAVEATS = [
    'Start skins differ; direct thermal comparisons are conditional on those recorded starts.',
    'Picker accuracy NOT EVALUATED; latency, execution and cadence misses are not accuracy.',
    'Single measured runs at one ambient temperature; no repeatability or ambient sweep. Ambient is not recorded in the P23 JSON.',
    'Fixed had a short rest: observed monotonic gap below the owner >=30 min rule.',
    'Monotonic rest gaps use increasing device timestamps; no boot ID is recorded, so cross-invocation clock continuity is UNVERIFIED.',
    'Phase-1 fixed validity labels are retained, including NOT COMPARABLE — START TEMP and rehearsals.',
]


def intervals_power(samples, intervals):
    parts = [data.weighted(samples.get('power', []), lo, hi) for lo, hi in intervals if hi > lo]
    seconds = sum(hi-lo for lo, hi in intervals if hi > lo)
    complete = bool(parts) and all(p['mean_battery_w'] is not None for p in parts)
    return dict(mean_w=sum(p['mean_battery_w']*p['covered_s'] for p in parts)/seconds if complete else None,
                duration_s=seconds, covered_s=sum(p['covered_s'] for p in parts), gaps=sum(p['gaps'] for p in parts))


def metrics(calls):
    result = {}
    for name, kind, size in [('yolo_320','YOLO',320),('yolo_640','YOLO',640),('m2','M2',None),('picker','picker',None)]:
        selected = [c for c in calls if c['kind']==kind and (size is None or c['size']==size)]
        result[name] = dict(data.stat(c['duration_ms'] for c in selected), calls=len(selected))
    result['picker_misses'] = sum(bool(c['miss']) for c in calls if c['kind']=='picker')
    return result


def summarize(runs, calls):
    result = dict(methods=dict(power='power.weighted on raw current_now receipt samples: clipped trapezoids, <=1.5 s gaps, no extrapolation.',
                              p95='Nearest rank ceil(.95*n); median is statistics.median.',
                              m2='Inference-only duration; calls.csv also keeps full wall duration.',
                              active_minutes='Cumulative active-phase wall time, including camera start and boundary waits; pauses excluded.',
                              skin_crossings='First recorded skin receipt >= threshold; no interpolation.',
                              caps='Later sample owns preceding interval; no tail credit, identical to power_map.capped_by_policy.',
                              scope='P23 measured origin through end; idle, gate and cleanup excluded from mode totals. Phase1 comparison excludes warmup.'),
                  caveats=CAVEATS, phase1=[], modes=[], rest_gaps=[], comparison_only=[],
                  dataset_call_counts={})
    for c in calls:
        counts = result['dataset_call_counts'].setdefault(c['run'], {})
        counts[c['kind']] = counts.get(c['kind'], 0)+1
    measured = []
    for stem, d, segments in runs:
        if 'runs' not in d:
            for s in segments:
                if s['state'] != 'active':
                    continue
                p = s['row']
                chosen = [c for c in calls if c['run']==stem and c['phase_id']==s['phase_id']]
                result['phase1'].append(dict(run=stem, phase_id=s['phase_id'], layout=s['layout'], label=s['label'],
                    source='runs/'+stem+'.json', duration_s=s['duration'], power=intervals_power(s['samples'],[(0,s['duration'])]),
                    calls=metrics(chosen), yolo=p.get('summary',{}).get('yolo'), caps=p.get('caps_by_policy'),
                    skin_start_c=p.get('skin_start',{}).get('skin'),skin_end_c=p.get('skin_end',{}).get('skin'),
                    memory_available_min_mib=p.get('mem_available_min_mib'), rereads=len(p.get('read_retries',[])),
                    start_temperature_comparability=p.get('start_temperature_comparability')))
            continue
        for run in d['runs']:
            name = stem+'_'+run['mode'] if d.get('rehearsal') else stem
            ss = [s for s in segments if s['run']==name and s['state'] in ('active','pause')]
            active = sorted([s for s in ss if s['state']=='active'], key=lambda s:s['origin'])
            chosen = [c for c in calls if c['run']==name and c['state']=='active']
            lo, hi = run['origin'], run['end']
            samples = d['samples']
            dumps = [r for r in samples['dumps'] if lo <= r['t'] <= hi]
            skin = [r['skin'] for r in dumps if r.get('skin') is not None]
            mem = [r for r in samples['memory'] if lo <= r['t'] <= hi]
            active_duration = sum(s['duration'] for s in active)
            offsets, elapsed = {}, 0.
            for s in active:
                offsets[s['phase_id']] = elapsed
                elapsed += s['duration']
            active_at = lambda c: offsets[c['phase_id']]+c['phase_started_s']
            totals = run.get('totals', {})
            item = dict(run=name, mode=run['mode'], label=run['label'], checks_validity=run.get('checks_validity'),
                        result_validity=run.get('result_validity'), source='runs/'+stem+'.json', duration_s=hi-lo,
                        start_skin_c=run.get('skin_start',{}).get('skin'),idle_start_skin_c=run.get('idle',{}).get('skin_start',{}).get('skin'),
                        max_skin_c=max(skin,default=None), android_status_max=max((r['status'] for r in dumps),default=None),
                        time_to_skin_s={str(t):next((r['t']-lo for r in dumps if r['skin']>=t),None) for t in (35,37,39)},
                        power=dict(whole=intervals_power(samples,[(lo,hi)]),
                                   active=intervals_power(samples,[(s['origin'],s['origin']+s['duration']) for s in active]),
                                   pause=intervals_power(samples,[(s['origin'],s['origin']+s['duration']) for s in ss if s['state']=='pause'])),
                        totals=totals, inference_s=sum(s['row'].get('inference_s',0.) for s in ss),
                        calls=dict(overall=metrics(chosen),first_5_active_minutes=metrics([c for c in chosen if active_at(c)<300]),
                                   last_5_active_minutes=metrics([c for c in chosen if active_at(c)>=max(0,active_duration-300)])),
                        yolo_due={str(size):sum(s['row'].get('frames_due',{}).get(str(size),0) for s in active) for size in (320,640)},
                        yolo_done={str(size):sum(c['kind']=='YOLO' and c['size']==size for c in chosen) for size in (320,640)},
                        caps_whole=data.caps(samples,lo,hi),
                        caps_active={p:dict(seconds=sum(s['row'].get('caps_by_policy',{}).get(p,{}).get('seconds',0) for s in active),
                            percent=100*sum(s['row'].get('caps_by_policy',{}).get(p,{}).get('seconds',0) for s in active)/active_duration if active_duration else None,
                            lowest_mhz=min((s['row']['caps_by_policy'][p]['lowest_mhz'] for s in active if s['row'].get('caps_by_policy',{}).get(p,{}).get('lowest_mhz') is not None),default=None)) for p in data.POLICIES},
                        memory=dict(available_min_mib=min((r['mem_available_mib'] for r in mem if r.get('mem_available_mib') is not None),default=None),
                                    swap_used_max_mib=max((r.get('swap_used_mib',0) for r in mem),default=None),
                                    llama_server_pss_max_kib=max((r.get('pss_kb',{}).get('llama_server',0) or 0 for r in mem),default=None),
                                    lmk=d.get('lmk')),
                        rereads=sum(lo<=r['t']<=hi for r in d.get('read_retries',[])),
                        thermal_rereads=sum(r['attempts']-1 for r in d.get('thermal_retries',[]) if lo<=r['t']<=hi),
                        failures=dict(issues=run.get('issues'),error=d.get('error'),cleanup_errors=d.get('cleanup_errors'),
                                      camera_failures=totals.get('camera_failures'),camera_restarts=totals.get('camera_restarts')),
                        cycles=run.get('cycles',[]), phases=[], steady_state_duty=None)
            for s in sorted(ss,key=lambda s:s['origin']):
                p = s['row']
                item['phases'].append(dict(phase_id=s['phase_id'],state=s['state'],start_s=s['origin']-lo,
                    duration_s=s['duration'],inference_s=p.get('inference_s'),duty_fraction=p.get('duty_fraction'),
                    camera_start_s=p.get('camera_start_s'),camera_stop_s=p.get('camera_stop_s'),
                    camera_off_s=p.get('camera_off_s'),camera_on_without_inference_s=p.get('camera_on_without_inference_s'),
                    low_threshold_c=p.get('low_threshold_c'),skin_min_c=p.get('skin_min_c'),skin_max_c=p.get('skin_max_c'),
                    power=intervals_power(samples,[(s['origin'],s['origin']+s['duration'])])))
            if run['pauses']:
                first = run['pauses'][0]['origin']
                inference = sum(max(0.,min(hi,s['row'].get('inference_end',s['origin']))-max(first,s['row'].get('inference_origin',hi))) for s in active)
                item['steady_state_duty'] = dict(from_s=first-lo,duration_s=hi-first,inference_s=inference,fraction=inference/(hi-first))
            result['modes'].append(item)
            if not d.get('rehearsal'):
                measured.append((stem,d,run,item))
    measured.sort(key=lambda x:x[2]['origin'])
    for previous,current in zip(measured, measured[1:]):
        stem,d,run,item = current
        # Both endpoints are recorded device monotonic values. Boot identity is not available.
        gap=run['idle']['origin']-previous[2]['end']
        result['rest_gaps'].append(dict(previous_run=previous[0],run=stem,gap_s=gap if gap>=0 else None,
            minutes=gap/60 if gap>=0 else None,owner_minimum_s=1800,meets_owner_rule=gap>=1800 if gap>=0 else None,
            previous_end=previous[2]['end'],next_idle_origin=run['idle']['origin'],
            basis='Recorded device monotonic end to next idle origin. No boot ID; continuity UNVERIFIED.' if gap>=0 else 'unknown: clocks not comparable (negative gap).'))
    targets={'endurance':(4.40,.997,40.76,89),'fixed':(3.30,.629,38.02,60),'adaptive':(3.54,.629,37.43,70)}
    for _,_,_,item in measured:
        for field, actual, expected in zip(('mean_w','duty_fraction','max_skin_c','picker_calls'),
                (item['power']['whole']['mean_w'],item['totals'].get('duty_fraction'),item['max_skin_c'],item['calls']['overall']['picker']['calls']),targets[item['mode']]):
            result['comparison_only'].append(dict(mode=item['mode'],metric=field,planner=expected,recomputed=actual,difference=actual-expected if actual is not None else None))
    for gap,expected in zip(result['rest_gaps'],(14,31)):
        result['comparison_only'].append(dict(mode=gap['run'],metric='rest_minutes',planner=expected,recomputed=gap['minutes'],difference=gap['minutes']-expected if gap['minutes'] is not None else None))
    return result


def fmt(value):
    return 'null' if value is None else f'{value:.6f}' if isinstance(value,float) else str(value)


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows]


def markdown(result):
    lines=['# Campaign results','', 'Generated offline by `python3 -B benchmark/campaign/summarize_campaign.py`; exact numeric values and full phase/cycle/memory details: [CAMPAIGN_RESULTS.json](CAMPAIGN_RESULTS.json). Raw invocations are cited in each table. No hardware was rerun.','', '## Caveats','']+['- '+s for s in result['caveats']]
    lines+=['','## Methods','']+['- '+k+': '+v for k,v in result['methods'].items()]
    lines+=['','## Phase 1 — all archived blocks, original labels','']
    lines+=table(['Run / phase (source)','Layout','Label','Mean W','M2 median / p95 ms','Picker median / p95 ms'],[(f"[{r['run']} / {r['phase_id']}]({r['source']})",r['layout'],r['label'],r['power']['mean_w'],f"{fmt(r['calls']['m2']['median_ms'])} / {fmt(r['calls']['m2']['p95_ms'])}",f"{fmt(r['calls']['picker']['median_ms'])} / {fmt(r['calls']['picker']['p95_ms'])}") for r in result['phase1']])
    for r in result['modes']:
        lines+=['',f"## {r['run']} — {r['mode']}",'',f"Label: **{r['label']}**; checks: {r['checks_validity']}; results: {r['result_validity']}. [Raw invocation]({r['source']}).",'']
        values=[('duration s',r['duration_s']),('start skin C',r['start_skin_c']),('idle start skin C',r['idle_start_skin_c']),('max skin C',r['max_skin_c']),('max Android status',r['android_status_max']),('time to 35 / 37 / 39 C s',' / '.join(fmt(r['time_to_skin_s'][str(t)]) for t in (35,37,39))),('whole / active / pause mean W',' / '.join(fmt(r['power'][k]['mean_w']) for k in ('whole','active','pause')))]
        values += list(r['totals'].items())
        values += [('memory available min MiB',r['memory']['available_min_mib']),('rereads',r['rereads']),('thermal rereads',r['thermal_rereads']),('issues',str(r['failures']))]
        lines+=table(['Metric','Value'],values)
        lines+=['','Call latencies (M2 inference only):','']
        lines+=table(['Slice','Kind','Calls','Median ms','p95 ms'],[(scope,kind,m['calls'],m['median_ms'],m['p95_ms']) for scope,cs in r['calls'].items() for kind,m in cs.items() if isinstance(m,dict)])
        lines+=['']+table(['Policy','Whole cap %','Active cap %','Whole lowest MHz','Active lowest MHz'],[(p,r['caps_whole'][p]['percent'],r['caps_active'][p]['percent'],r['caps_whole'][p]['lowest_mhz'],r['caps_active'][p]['lowest_mhz']) for p in data.POLICIES])
        lines+=['',f"YOLO due / done: `{r['yolo_due']}` / `{r['yolo_done']}`."]
        if r['mode']=='fixed':
            lines+=['','Per-cycle table:','']+table(['Cycle','Active s','Pause s','Inference s','Duty','Skin min C','Skin max C','Camera start s','Camera stop s'],[(i,c['active_s'],c['pause_s'],c['inference_s'],c['duty_fraction'],c.get('skin_min_c'),c.get('skin_max_c'),c['camera_start_s'],c['camera_stop_s']) for i,c in enumerate(r['cycles'],1)])
        if r['mode']=='adaptive':
            lines+=['',f"Steady-state duty from first pause: `{r['steady_state_duty']}`.",'']+table(['Phase','State','Start s','Duration s','Inference s','Mean W','Skin min C','Skin max C'],[(p['phase_id'],p['state'],p['start_s'],p['duration_s'],p['inference_s'],p['power']['mean_w'],p['skin_min_c'],p['skin_max_c']) for p in r['phases']])
    lines+=['','## Rest gaps','']+table(['Previous','Next','Minutes','>=30 min','Basis'],[(g['previous_run'],g['run'],g['minutes'],g['meets_owner_rule'],g['basis']) for g in result['rest_gaps']])
    lines+=['','## Planner comparison only — every difference','', 'These approximate hand numbers never feed generation, labels or checks. Differences below retain full precision in JSON.','']+table(['Mode','Metric','Planner','Recomputed','Difference'],[(r['mode'],r['metric'],r['planner'],r['recomputed'],r['difference']) for r in result['comparison_only']])
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    for name, content in data.rebuild().items():
        path=data.HERE/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
