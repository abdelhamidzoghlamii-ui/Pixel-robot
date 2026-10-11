"""Offline, deterministic campaign archive views. Only the Python standard library is needed."""
import ast
from bisect import bisect_left, bisect_right
import csv
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess

import power

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
POLICIES = ('policy0', 'policy4', 'policy6')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n').encode()


def load_card(stem):
    return json.loads((HERE/'runs/cards'/f'{stem}.json').read_bytes())


def _rows(name):
    with (HERE/'data'/f'{name}.csv').open(newline='') as stream:
        yield from csv.DictReader(stream)


def iter_windows():
    yield from _rows('windows')


def iter_calls():
    yield from _rows('calls')


def iter_thermal():
    yield from _rows('thermal')


def stat(values):
    values = sorted(v for v in values if v is not None)
    return dict(count=len(values), median_ms=statistics.median(values) if values else None,
                p95_ms=values[math.ceil(.95*len(values))-1] if values else None)


def last(rows, at):
    # Receipt timestamps are sorted once by segments(); never use a future reading.
    i = bisect_right(rows, at, key=lambda r:r['t'])-1
    return rows[i] if i >= 0 else None


def weighted(rows, lo, hi):
    a = max(0, bisect_left(rows, lo, key=lambda r:r['t'])-1)
    b = bisect_right(rows, hi, key=lambda r:r['t'])+1
    return power.weighted(rows[a:b], lo, hi)


def caps(samples, lo, hi):
    result = {}
    rows = [r for r in samples.get('fast', []) if lo <= r['t'] <= hi]
    for policy in POLICIES:
        seconds = unknown = 0.
        previous = lo
        values = []
        for r in rows:
            value, top = r.get('max', {}).get(policy), samples.get('cpuinfo_max_khz', {}).get(policy)
            dt, previous = r['t']-previous, r['t']
            if value is None or top is None:
                unknown += dt
            else:
                values.append(value)
                if value < top:
                    seconds += dt
        result[policy] = dict(seconds=seconds, percent=100*seconds/(hi-lo) if hi > lo else None,
                              unknown_s=unknown, tail_unknown_s=hi-previous,
                              lowest_mhz=min(values)/1000 if values else None)
    return result


def archive_runs():
    """Root invocations only; embedded phases/checkpoints are not duplicate invocations."""
    for path in sorted((HERE/'runs').glob('owner_*.json')):
        d = json.loads(path.read_bytes())
        if 'runs' in d or 'plan' in d or 'step_response' in d:
            yield path, d


def segments(stem, d):
    """Each record carries an explicit clock origin; old rest phases lack a shared clock."""
    if 'runs' in d:
        for run in d['runs']:
            name = stem+'_'+run['mode'] if d.get('rehearsal') else stem
            for state, phases in [('idle', [run.get('idle', {})]), ('active', run.get('active_phases', [])), ('pause', run.get('pauses', []))]:
                for i, phase in enumerate(phases, 1):
                    if 'end' not in phase or 'origin' not in phase:
                        continue
                    yield dict(run=name, source=stem+'.json', phase='2/3', mode=run['mode'], layout='L2',
                               label=run.get('label', d['label']), checks=run.get('checks_validity'),
                               result=run.get('result_validity'), phase_id=f'{state}_{i:02d}', state=state,
                               origin=phase['origin'], run_origin=run['origin'], duration=phase['end']-phase['origin'],
                               call_origin=phase.get('inference_origin', phase['origin']),
                               samples=d['samples'], row=phase, run_row=run, shared_clock=True)
    else:
        groups = [('idle', d.get('idle_phases', [])), ('pause', d.get('pauses', [])), ('warmup', [d['warmup']] if d.get('warmup') else []), ('active', d.get('blocks', []))]
        if 'step_response' in d:
            groups = [('probe', [d])]
        for state, phases in groups:
            for i, phase in enumerate(phases, 1):
                if not phase.get('duration_s'):
                    continue
                label = phase.get('validity', d['label'])
                if state == 'warmup':
                    label = 'NOT VALID — WARMUP'
                elif 'REHEARSAL' in d['label']:
                    label = d['label']
                samples = {k:sorted(phase.get(k, []), key=lambda r:r['t']) for k in ('fast','dumps','memory','power')}
                for key in ('skin_start','skin_end'):
                    reading=phase.get(key)
                    if reading and 't' in reading and not any(r['t']==reading['t'] for r in samples['dumps']):
                        samples['dumps'].append(reading)
                samples['dumps'].sort(key=lambda r:r['t'])
                samples['cpuinfo_max_khz'] = phase.get('cpuinfo_max_khz', {})
                samples['read_retries'] = phase.get('read_retries', [])
                samples['thermal_retries'] = phase.get('thermal_retries', [])
                yield dict(run=stem, source=stem+'.json', phase='1' if state != 'probe' else 'probe', mode='layout_screen',
                           layout=phase.get('block'), label=label, checks=None, result=phase.get('validity'),
                           phase_id=f'{state}_{i:02d}', state=state, origin=0., run_origin=0., call_origin=0.,
                           duration=phase['duration_s'], samples=samples, row=phase, run_row=d, shared_clock=False)


def call_rows(s):
    phase, samples = s['row'], s['samples']
    for raw_kind, kind in [('yolo','YOLO'), ('m2','M2'), ('selector','picker')]:
        for index, call in enumerate(phase.get(raw_kind, [])):
            if raw_kind == 'm2' and not call.get('inference_executed', False):
                continue
            if 'started_s' not in call:
                continue
            at = s['call_origin']+call['started_s']
            fast, skin = last(samples.get('fast', []), at), last(samples.get('dumps', []), at)
            context = call.get('scene') if raw_kind == 'm2' else ('fallback' if str(call.get('context')).startswith('(fallback scene)') else 'live' if call.get('context') else 'none')
            row = dict(run=s['run'], label=s['label'], phase=s['phase'], mode=s['mode'], layout=s['layout'],
                       phase_id=s['phase_id'], state=s['state'], kind=kind, size=call.get('size'), call_index=index,
                       started_s=at-s['run_origin'], phase_started_s=at-s['origin'],
                       duration_ms=call.get('inference_ms') if kind == 'M2' else call.get('ms'),
                       wall_duration_ms=(call['ended_s']-call['started_s'])*1000 if 'ended_s' in call else None,
                       skin_c=skin.get('skin') if skin else None, skin_age_s=at-skin['t'] if skin else None,
                       frequency_age_s=at-fast['t'] if fast else None,
                       context_type=context if kind != 'YOLO' else 'none', miss=call.get('miss'),
                       clock='run' if s['shared_clock'] else 'phase', source=s['source'])
            row.update({p+'_max_mhz':fast.get('max', {}).get(p)/1000 if fast and fast.get('max', {}).get(p) is not None else None for p in POLICIES})
            yield row


def window_rows(s, calls):
    phase, samples = s['row'], s['samples']
    for lo in range(0, math.ceil(s['duration']), 60):
        hi = min(lo+60., s['duration'])
        if hi-lo < 1e-6:
            continue
        # P23 archive contains exact per-phase 60 s windows, including its scheduler's slot convention.
        saved = next((w for w in phase.get('windows', []) if w['start_s']==lo), {})
        selected = [c for c in calls if lo <= c['phase_started_s'] < hi]
        dumps = [r for r in samples.get('dumps', []) if s['origin']+lo <= r['t'] < s['origin']+hi]
        mem = [r.get('mem_available_mib') for r in samples.get('memory', []) if s['origin']+lo <= r['t'] < s['origin']+hi and r.get('mem_available_mib') is not None]
        skin = [r['skin'] for r in dumps if r.get('skin') is not None]
        cap = saved.get('caps_by_policy') or caps(samples, s['origin']+lo, s['origin']+hi)
        rereads = saved.get('read_re_reads')
        if rereads is None:
            rereads = sum(s['origin']+lo <= r['t'] < s['origin']+hi for r in samples.get('read_retries', []))+sum(r['attempts']-1 for r in samples.get('thermal_retries', []) if s['origin']+lo <= r['t'] < s['origin']+hi)
        row = dict(run=s['run'], label=s['label'], phase=s['phase'], mode=s['mode'], layout=s['layout'],
                   phase_id=s['phase_id'], state=s['state'], start_s=s['origin']-s['run_origin']+lo, phase_start_s=lo,
                   active_s=hi-lo if s['state']=='active' else 0., pause_s=hi-lo if s['state']=='pause' else 0., idle_s=hi-lo if s['state']=='idle' else 0.,
                   duration_s=hi-lo, mean_w=weighted(samples.get('power', []), s['origin']+lo, s['origin']+hi)['mean_battery_w'],
                   skin_min_c=min(skin, default=None), skin_max_c=max(skin, default=None),
                   android_status_max=max((r['status'] for r in dumps if r.get('status') is not None), default=None),
                   memory_available_min_mib=min(mem, default=None), rereads=rereads,
                   clock='run' if s['shared_clock'] else 'phase',
                   missing_reason=None if saved else 'Older archive lacks per-window scheduled slots; due/skipped are null. Phase clock for old rest records cannot be aligned to run.')
        if s.get('state_intervals'):
            for state in ('active','pause'):
                row[state+'_s'] = sum(max(0., min(s['origin']+hi,b)-max(s['origin']+lo,a)) for st,a,b in s['state_intervals'] if st==state)
            states = [state for state in ('active','pause') if row[state+'_s']>1e-6]
            row['state'] = states[0] if len(states)==1 else 'mixed'
        for p in POLICIES:
            row[p+'_cap_percent'] = cap.get(p, {}).get('percent')
            row[p+'_lowest_mhz'] = cap.get(p, {}).get('lowest_mhz')
        for size in (320, 640):
            counts = saved.get('yolo', {}).get(str(size), {})
            metrics = stat(c['duration_ms'] for c in selected if c['kind']=='YOLO' and c['size']==size)
            for field in ('due','done','skipped','median_ms','p95_ms'):
                row[f'yolo_{size}_{field}'] = counts.get(field) if saved else (metrics['count'] if field=='done' else metrics.get(field))
        for kind, raw in [('M2','m2'),('picker','selector')]:
            metrics = stat(c['duration_ms'] for c in selected if c['kind']==kind)
            counts = saved.get(raw, {})
            row[raw+'_calls'] = counts.get('calls', metrics['count'])
            row[raw+'_median_ms'] = counts.get('median_ms', metrics['median_ms'])
        row['picker_misses'] = saved.get('selector', {}).get('misses', sum(bool(c['miss']) for c in selected if c['kind']=='picker'))
        yield row


def thermal_rows(s):
    samples = s['samples']
    base = dict(run=s['run'], label=s['label'], phase=s['phase'], mode=s['mode'], layout=s['layout'], phase_id=s['phase_id'],
                state=s['state'], clock='run' if s['shared_clock'] else 'phase')
    def annotation(at):
        p = s['row']
        if s['state']=='active':
            if at < p.get('camera_start_at', -math.inf):
                return False, 'camera-OFF before start'
            return True, 'heating '+str(s['layout'])
        if s['state']=='pause':
            if at < p.get('camera_off_at', -math.inf):
                return True, 'camera stop transition'
            return False, 'camera-OFF cooling'
        return p.get('camera_on', s['state'] in ('warmup','probe')), 'camera-ON idle' if p.get('camera_on') else 'camera-OFF cooling'
    for r in samples.get('dumps', []):
        if s['origin'] <= r['t'] < s['origin']+s['duration']:
            on, segment = annotation(r['t'])
            yield dict(base, kind='skin', time_s=r['t']-s['run_origin'], duration_s=None, skin_c=r.get('skin'),
                       android_status=r.get('status'), mean_w=None, covered_s=None, camera_on=on, segment_type=segment)
    for lo in range(math.ceil(s['duration'])):
        hi = min(lo+1., s['duration'])
        value = weighted(samples.get('power', []), s['origin']+lo, s['origin']+hi)
        on, segment = annotation(s['origin']+lo)
        yield dict(base, kind='power_1s', time_s=s['origin']-s['run_origin']+lo, duration_s=hi-lo, skin_c=None,
                   android_status=None, mean_w=value['mean_battery_w'], covered_s=value['covered_s'], camera_on=on, segment_type=segment)


def matching_code(hashes):
    if not hashes:
        return 'unknown'
    commits = subprocess.check_output(['git','rev-list','c0ba141dc1f6abce558f578cbae361a8d9cc16d1'], cwd=ROOT, text=True).splitlines()
    blob_hash = {}
    for commit in commits:
        matches = True
        for name, expected in hashes.items():
            blob = subprocess.run(['git','rev-parse',f'{commit}:{name}'], cwd=ROOT, capture_output=True, text=True)
            if blob.returncode:
                matches = False
                break
            key = blob.stdout.strip()
            if key not in blob_hash:
                blob_hash[key] = hashlib.sha256(subprocess.check_output(['git','cat-file','blob',key], cwd=ROOT)).hexdigest()
            if blob_hash[key] != expected:
                matches = False
                break
        if matches:
            return commit
    return 'unknown'


def card(stem, d, evidence, commit):
    missing = {}
    def recorded(key, value, reason='Not recorded in archived invocation JSON.'):
        if value is None:
            missing[key] = reason
        return value
    commands = None
    config = None
    if commit != 'unknown':
        text = subprocess.check_output(['git','show',f'{commit}:benchmark/campaign/RUN.md'], cwd=ROOT, text=True)
        commands = [line for line in text.splitlines() if line.startswith('(set -C;') and '--output ~/storage/downloads/campaign/'+stem+'.json' in line]
        src = subprocess.check_output(['git','show',f'{commit}:benchmark/campaign/phase23.py'], cwd=ROOT, text=True)
        tree = ast.parse(src)
        schedule = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('FULL','SHORT') for t in node.targets):
                schedule[node.targets[0].id] = {k.arg:ast.literal_eval(k.value) for k in node.value.keywords}
        config = dict(schedule=schedule.get('SHORT' if d.get('rehearsal') else 'FULL'),
                      source='phase23.py at '+commit, t_hi_c=[r.get('t_hi_c') for r in d.get('runs', [])])
    samples = d.get('samples', {})
    return dict(stem=stem, phase='2/3' if 'runs' in d else '1' if 'plan' in d else 'probe',
                mode=d.get('mode', [r['mode'] for r in d.get('runs', [])] or None),
                layout='L2' if 'runs' in d else [r.get('block') for r in d.get('blocks', [])], label=d.get('label'),
                checks_validity=d.get('rehearsal_coverage', [r.get('checks_validity') for r in d.get('runs', [])] or None),
                result_validity=[r.get('result_validity') for r in d.get('runs', [])] or [r.get('validity') for r in d.get('blocks', [])],
                code_commit=commit, code_commit_basis='Newest matching snapshot through P23D base c0ba141; matching hashes do not prove the executed commit.' if commit!='unknown' else 'No recorded code hashes; executed commit unknown.', code_hashes=recorded('code_hashes',d.get('code_hashes')),
                model_files_and_hashes=recorded('model_files_and_hashes',d.get('hashes')),
                llama_server_command=recorded('llama_server_command',d.get('server_command')),
                config=dict(recorded_schedule=recorded('schedule',config), detector_note=recorded('detector_note',d.get('detector_note')),
                            ort=samples.get('ort'), m2_cluster=samples.get('m2_cluster'), monitor_mask=samples.get('monitor_mask'),
                            layout_checks=d.get('layout_checks'),
                            by_phase=[dict(phase_id=f'active_{i:02d}',layout=b.get('block'),ort=b.get('ort'),m2_cluster=b.get('m2_cluster'),setup_affinity=b.get('setup_affinity')) for i,b in enumerate(d.get('blocks',[]),1)],
                            size_policy=recorded('SizePolicy',d.get('detector_note')),
                            cadence_s=1 if 'runs' in d else recorded('cadence_s',None),
                            selector_slot_s=20 if 'runs' in d else recorded('selector_slot_s',None),
                            hysteresis_c=2.0 if 'runs' in d and commit != 'unknown' else recorded('hysteresis_c',None)),
                exact_owner_command=recorded('exact_owner_command',commands or None,'No matching code commit / exact stem command in that commit RUN.md.'),
                environment=dict(battery_start_percent=recorded('battery_start_percent',d.get('battery_start_percent')),
                                 installed_robotcam=recorded('installed_robotcam',d.get('installed_robotcam')),
                                 device=recorded('device',None,'No device identity recorded by invocation; phone model from docs is not run metadata.'),
                                 idle_start_skin=[r.get('idle',{}).get('skin_start') for r in d.get('runs', [])] or recorded('idle_start_skin',None),
                                 gate_skin=[r.get('gate',{}).get('reading') for r in d.get('runs', [])] or recorded('gate_skin',None),
                                 rest_gap_s=None, rest_gap_reason='No comparable preceding invocation established; see summary rest_gaps for P23 sessions.'),
                evidence=evidence, missing_reasons=missing)


def csv_bytes(rows):
    if not rows:
        return b''
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def rebuild():
    """Return derived bytes without writing them; used by both generator and self-check."""
    output, all_calls, windows, thermals, runs, cards = {}, [], [], [], [], {}
    commit_cache = {}
    invocations = list(archive_runs())
    stems = [path.stem for path, _ in invocations]
    for path, d in invocations:
        stem = path.stem
        evidence = [dict(path=str(p.relative_to(HERE)), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted((HERE/'runs').glob(stem+'*')) if p.is_file() and stem==max((name for name in stems if p.name.startswith(name+'.') or p.name.startswith(name+'_')), key=len)]
        key = json.dumps(d.get('code_hashes', {}), sort_keys=True)
        if key not in commit_cache:
            commit_cache[key] = matching_code(d.get('code_hashes'))
        cards[stem] = card(stem, d, evidence, commit_cache[key])
        ss = list(segments(stem, d))
        for s in ss:
            for k in ('fast','dumps','memory','power'):
                s['samples'][k] = sorted(s['samples'].get(k, []), key=lambda r:r['t'])
            calls = list(call_rows(s))
            all_calls.extend(calls)
            if 'runs' not in d or s['state']=='idle':
                windows.extend(window_rows(s, calls))
            thermals.extend(thermal_rows(s))
        for run in d.get('runs', []):
            name = stem+'_'+run['mode'] if d.get('rehearsal') else stem
            related = [s for s in ss if s['run']==name and s['state'] in ('active','pause')]
            base = dict(related[0], row=run, state='mixed', phase_id='measured', origin=run['origin'],
                        duration=run['duration_s'], state_intervals=[(s['state'],s['origin'],s['origin']+s['duration']) for s in related])
            aligned = [dict(c, phase_started_s=c['started_s']) for c in all_calls if c['run']==name and c['state']=='active']
            windows.extend(window_rows(base, aligned))
            idle_end=run['idle']['end']
            if run['origin']>idle_end:
                gate=dict(base, row={}, state='idle', phase_id='gate', origin=idle_end, duration=run['origin']-idle_end)
                thermals.extend(thermal_rows(gate))
        runs.append((stem, d, ss))
    from summarize_campaign import summarize, markdown
    summary = summarize(runs, all_calls)
    for gap in summary['rest_gaps']:
        cards[gap['run']]['environment'].update(rest_gap_s=gap['gap_s'], rest_gap_reason=gap['basis'])
    # Rehearsals are one invocation; mode cards are useful as direct data-row lookup keys.
    for stem, d, ss in runs:
        for run in d.get('runs', []):
            if d.get('rehearsal'):
                name=stem+'_'+run['mode']
                cards[name] = dict(copy.deepcopy(cards[stem]), stem=name, mode=run['mode'], label=run['label'],
                                   checks_validity=run.get('checks_validity'),result_validity=run.get('result_validity'), parent_stem=stem)
                child=cards[name]
                if child['config']['recorded_schedule'] is not None:
                    child['config']['recorded_schedule']['t_hi_c']=[run.get('t_hi_c')]
                child['environment']['idle_start_skin']=[run.get('idle',{}).get('skin_start')]
                child['environment']['gate_skin']=[run.get('gate',{}).get('reading')]
                child['parent_owner_command']=child['exact_owner_command']
                child['exact_owner_command']=None
                child['missing_reasons']['exact_owner_command']='No separate mode invocation: produced by the parent all-mode rehearsal command in parent_owner_command.'
    for stem, c in cards.items():
        def missing(value, prefix=''):
            if value is None:
                c['missing_reasons'].setdefault(prefix, 'Recorded null or absent in archived evidence; not inferred.')
            elif isinstance(value, dict):
                for key, child in list(value.items()):
                    if key != 'missing_reasons':
                        missing(child, prefix+'.'+key if prefix else key)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    missing(child, prefix+f'[{index}]')
        missing(c)
        output[f'runs/cards/{stem}.json'] = json_bytes(c)
    for name, rows in [('windows',windows), ('calls',all_calls), ('thermal',thermals)]:
        output[f'data/{name}.csv'] = csv_bytes(rows)
    output['CAMPAIGN_RESULTS.json'] = json_bytes(summary)
    output['CAMPAIGN_RESULTS.md'] = markdown(summary).encode()
    return output


if __name__ == '__main__':
    for name, data in rebuild().items():
        path = HERE/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
