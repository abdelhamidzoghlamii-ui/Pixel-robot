"""Offline archive/derivation integrity and meaningful cross-checks; no hardware calls."""
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

import campaign_data as data
import power


def main():
    derived = data.rebuild()
    for name, content in derived.items():
        assert (data.HERE/name).read_bytes()==content, 'Rebuild differs: '+name
    calls = list(data.iter_calls())
    summary = json.loads(derived['CAMPAIGN_RESULTS.json'])
    actual = collections.defaultdict(collections.Counter)
    for c in calls:
        actual[c['run']][c['kind']] += 1
        if c['skin_age_s']:
            assert float(c['skin_age_s'])>=0
        if c['frequency_age_s']:
            assert float(c['frequency_age_s'])>=0
    assert dict(actual)==summary['dataset_call_counts']
    for mode in summary['modes']:
        if mode['label']=='VALID':
            assert mode['yolo_due']==mode['yolo_done'], mode['run']
        assert math.isclose(sum(p['inference_s'] for p in mode['phases']),mode['totals']['inference_s'],abs_tol=1e-6)
        chosen = [c for c in calls if c['run']==mode['run'] and c['state']=='active']
        assert len([c for c in chosen if c['kind']=='picker'])==mode['calls']['overall']['picker']['calls']
        assert len([c for c in chosen if c['kind']=='M2'])==mode['calls']['overall']['m2']['calls']
    windows = list(data.iter_windows())
    for path, d in data.archive_runs():
        if 'runs' in d:
            for run in d['runs']:
                stem=path.stem+'_'+run['mode'] if d.get('rehearsal') else path.stem
                selected=[w for w in windows if w['run']==stem and w['phase_id']=='measured']
                assert len(selected)==math.ceil((run['duration_s']-1e-6)/60)
                assert math.isclose(sum(float(w['duration_s']) for w in selected),run['duration_s'],abs_tol=1e-6)
                assert all(math.isclose(float(w['active_s'])+float(w['pause_s']),float(w['duration_s']),abs_tol=1e-6) for w in selected)
                idle=[w for w in windows if w['run']==stem and w['phase_id']=='idle_01']
                assert len(idle)==math.ceil((run['idle']['end']-run['idle']['origin']-1e-6)/60)
        else:
            for s in data.segments(path.stem,d):
                selected=[w for w in windows if w['run']==s['run'] and w['phase_id']==s['phase_id']]
                assert len(selected)==math.ceil((s['duration']-1e-6)/60)
                assert math.isclose(sum(float(w['duration_s']) for w in selected),s['duration'],abs_tol=1e-6)
    for card in (data.HERE/'runs/cards').glob('*.json'):
        d=json.loads(card.read_bytes())
        for row in d['evidence']:
            path=data.HERE/row['path']
            assert path.stat().st_size==row['bytes'] and data.digest(path)==row['sha256'],str(card)
    # Hash every entry in the campaign ARTIFACTS and each other manifest updated by P23D.
    manifest_list=json.loads((data.HERE/'checks/p23d/artifact_manifests.json').read_bytes())
    artifacts=0
    for name in manifest_list:
        manifest=data.ROOT/name
        old=data.subprocess.run(['git','show','HEAD:'+name],cwd=data.ROOT,capture_output=True)
        if old.returncode==0:
            assert manifest.read_bytes().startswith(old.stdout), 'Historical manifest rewritten: '+name
        text=manifest.read_text().split('## P23D current inventory',1)[-1]
        for line in text.splitlines():
            cols=line.split('|')
            if len(cols)>=5 and re.fullmatch(r'`[a-f0-9]{64}`',cols[-2].strip()):
                name=cols[1].strip().strip('`')
                if Path(name).is_absolute():
                    continue  # Historical manifests also list omitted external sources.
                path=manifest.parent/name
                assert path.stat().st_size==int(cols[-3]) and data.digest(path)==cols[-2].strip().strip('`'),str(path)
                artifacts+=1
    frozen=json.loads((data.HERE/'frozen_manifest_campaign_p23d.json').read_bytes())
    frozen_paths={r['path'] for r in frozen['files']}
    required={str((data.HERE/name).relative_to(data.ROOT)) for name in derived}
    required.update(str(Path(name).parent/doc) for name in manifest_list for doc in ('README.md','RUN_INDEX.md'))
    tracked=set(data.subprocess.check_output(['git','ls-files'],cwd=data.ROOT,text=True).splitlines())
    decisions=json.loads((data.HERE/'checks/p23d/sweep_decisions.json').read_bytes())
    for row in decisions:
        if row['decision']=='archived into folder':
            path=data.ROOT/row['destination']
            assert data.digest(path)==row['sha256'] and path.stat().st_size==row['bytes']
            required.add(row['destination'])
        elif row['decision']=='already archived':
            assert any((name in tracked or name in frozen_paths) and (data.ROOT/name).is_file() and data.digest(data.ROOT/name)==row['sha256'] for name in row['destinations'])
    required.update(manifest_list)
    required.add('benchmark/UNARCHIVED_INVENTORY.md')
    base_paths=set(data.subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],cwd=data.ROOT,text=True).splitlines())
    assert required<=frozen_paths|base_paths, 'Missing from frozen candidate: '+str(sorted(required-frozen_paths-base_paths))
    tracked=set(data.subprocess.check_output(['git','ls-files'],cwd=data.ROOT,text=True).splitlines())
    ignored=data.subprocess.check_output(['git','ls-files','--others','--ignored','--exclude-standard'],cwd=data.ROOT,text=True).splitlines()
    assert not (required & set(ignored)), 'Required files remain ignored and untracked'
    assert all(name in tracked or name in frozen_paths for name in required)
    # Phase1 start/end readings are recorded separately from the periodic worker samples.
    fixture=dict(label='SESSION COMPLETE',plan=['L0'],blocks=[dict(block='L0',validity='VALID',duration_s=2.,
        skin_start=dict(t=-.1,skin=20.),dumps=[dict(t=1.,skin=21.)],yolo=[dict(started_s=.1,ended_s=.2,ms=100.,size=320)])])
    segment=next(data.segments('fixture',fixture))
    call=next(data.call_rows(segment))
    assert call['skin_c']==20. and math.isclose(call['skin_age_s'],.2)
    # Preceding-read boundaries, absent history, median/p95, and integration subset equivalence.
    points=[dict(t=1,skin=20),dict(t=2,skin=21)]
    assert data.last(points,.5) is None and data.last(points,1) is points[0] and data.last(points,1.99) is points[0]
    assert data.last(points,2) is points[1]
    assert data.stat([1,2,3,4])==dict(count=4,median_ms=2.5,p95_ms=4)
    raw=[dict(t=i/2,battery_w=float(i*i)) for i in range(12)]
    for lo,hi in [(.1,2.7),(0,5.5),(-1,2),(4,8),(1,1.1)]:
        assert data.weighted(raw,lo,hi)==power.weighted(raw,lo,hi)
    gap=[dict(t=0,battery_w=1),dict(t=2,battery_w=1)]
    assert data.weighted(gap,0,1)['mean_battery_w'] is None
    print(f'PASS P23D: {len(derived)} byte-identical derived files; {len(calls)} calls; {len(windows)} windows; {artifacts} artifact entries; cards, validity/counts, phase totals, durations, preceding samples, power coverage.')


if __name__=='__main__':
    main()
