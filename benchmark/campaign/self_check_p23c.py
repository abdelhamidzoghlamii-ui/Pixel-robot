"""P23C offline setup settle and post-origin controller differential checks."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import phase23 as q
import runtime as rt
from self_check_p23 import result

BASE = 'd5d1e87c66db86848cc93495046cde8f6e3d75f6'


class Scheduled(q.MockSession):
    """Independent power receipts; polling never manufactures extra samples."""
    def __init__(self, times):
        super().__init__(None,{})
        self.times=times
        self.data['power']=[dict(t=t,battery_w=1.) for t in times if t<=self.clock]
    def wait(self, seconds):
        super().wait(seconds)
        self.data['power']=[dict(t=t,battery_w=1.) for t in self.times if t<=self.clock]


def check_settle():
    for first,gap in ((100.,.37),(100.,2.47),(94751.921060669,2.874435548)):
        times=[first]+[first+gap+i*.37 for i in range(1000)]
        s=Scheduled(times);s.clock=first;s.data['power']=[dict(t=first,battery_w=1.)]
        rec={};origin=q.settle_power(s,rec)
        expected=first+(9*.37 if gap==.37 else gap+9*.37)
        assert expected-1e-8<=origin<=expected+.050001,(origin,expected)
        assert rec['settled'] and rec['sample_count']==10 and rec['max_interval_s']<=1.5
        assert rec['last_sample_age_s']<=1.5
        # Drive the actual first idle and actual weighted window; ready gate adds no delay.
        out=result();out['setup_complete']=False
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            q.run_mode('endurance',s,out,dict(q.SHORT,endurance=30),37.,lambda x:Path(tmp)/('run'+x),dry=True)
        idle=out['runs'][0]['idle'];power=idle['windows'][0]['power']
        assert idle['origin']==origin and out['setup']['power_settle']['settle_s']==0
        assert power['mean_battery_w']==1. and power['gaps']==0
        assert abs(power['covered_s']-idle['wall_duration_s'])<1e-6
        print('PASS start-up gap',gap,'settle_s',rec['settle_s'],'first idle full coverage')
    # Ten stale samples, nine samples only, no samples, and a continuing 2 s cadence all fail at 60 s.
    for times in ([96+i*.1 for i in range(10)],[100+i*.37 for i in range(9)],[],[100+i*2 for i in range(100)]):
        s=Scheduled(times);rec={}
        try:q.settle_power(s,rec)
        except RuntimeError as e:assert 'within 60 s' in str(e)
        else:raise AssertionError('unsettled sampler accepted')
        assert abs(s.now()-160)<1e-8 and abs(rec['settle_s']-60)<1e-8 and not rec['settled']
    # Exactly 1.5 s is accepted; immediately stale history must wait for ten new receipts.
    s=Scheduled([100+1.5*i for i in range(50)]);rec={};q.settle_power(s,rec)
    assert 13.5<=rec['settle_s']<=13.550001 and rec['max_interval_s']==1.5
    s=Scheduled([96+i*.1 for i in range(10)]+[102+i*.37 for i in range(500)])
    rec={};q.settle_power(s,rec);assert s.now()>=105.33
    print('PASS steady cadence: no delay beyond tenth receipt + <=50 ms polling; stale/absent/slow/short streams timeout at 60 s')


def check_main_failure():
    class Never(Scheduled):
        def __init__(self,server,data):
            super().__init__([100+i*2 for i in range(100)])
            data.update(self.data);self.data=data;self.closed=False
            sessions.append(self)
        def start(self):pass
        def close(self):self.closed=True
    sessions=[]
    rt.READ_RETRIES.clear();rt.THERMAL_RETRIES.clear()
    with tempfile.TemporaryDirectory() as tmp,patch.object(q,'MockSession',Never),contextlib.redirect_stdout(io.StringIO()):
        output=Path(tmp)/'timeout.json'
        try:q.main(['--dry-run','--mock','--output',str(output)])
        except RuntimeError as e:assert 'within 60 s' in str(e)
        else:raise AssertionError('setup timeout did not fail invocation')
        d=json.loads(output.read_bytes())
        assert d['failure_kind']=='setup' and not d['setup_complete']
        assert not d['rehearsal_coverage']['rehearsal_pass'] and d['mock_only']
        assert d['setup']['power_settle']['settle_s']==60
        assert all('idle' not in r and not r['active_phases'] and not r['pauses'] for r in d['runs'])
        assert not list(Path(tmp).glob('*_idle.json')) and sessions[0].closed
    # Native-session exception classification, with all external operations mocked.
    with tempfile.TemporaryDirectory() as tmp,patch.object(q,'LiveSession',Never),patch.object(q.p,'preflight'),patch.object(q,'require_rehearsal',return_value={}),patch.object(rt,'screen_state',return_value={}),patch.object(rt,'imports',return_value=(None,None,None,None,None)),patch.object(q.p,'lmk_window',return_value={'ok':True,'n_kills':0}),contextlib.redirect_stdout(io.StringIO()):
        output=Path(tmp)/'timeout.json'
        try:q.main(['--mode','endurance','--rehearsal','unused.json','--output',str(output)])
        except RuntimeError as e:assert 'within 60 s' in str(e)
        else:raise AssertionError('native setup timeout accepted')
        d=json.loads(output.read_bytes())
        assert d['label']=='NOT VALID — SETUP FAILURE' and d['failure_kind']=='setup' and not d['setup_complete']
    print('PASS main: 60 s timeout is setup failure, no idle/active measurement, cleanup, NOT VALID')


def check_post_origin_identical():
    base=NS(__name__='phase23_base',__file__=q.__file__)
    source=subprocess.check_output(['git','show',BASE+':benchmark/campaign/phase23.py'])
    exec(compile(source,'phase23_base.py','exec'),base.__dict__)
    assert q.FULL==base.FULL and q.SHORT==base.SHORT
    for dry in (False,True):
        for mode in q.MODES:
            evidence=[]
            for module in (base,q):
                with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                    session=module.MockSession(None,{});session.start()
                    # Identical, already settled input at the common origin; new setup gate executes.
                    session.data['power']=[dict(t=session.now()-(9-i)*.37,battery_w=1.) for i in range(10)]
                    out=result();out['setup_complete']=module is base
                    module.run_mode(mode,session,out,module.SHORT if dry else module.FULL,37.,lambda x:Path(tmp)/('run'+x),dry=dry)
                    if module is q:assert out['setup']['power_settle']['settle_s']==0
                    evidence.append((out['runs'],{p.name:p.read_bytes() for p in Path(tmp).glob('*.json')}))
            assert evidence[0]==evidence[1],(mode,dry,'post-origin output differs')
    print('PASS differential d5d1e87: all three modes FULL and SHORT, run objects and every phase/checkpoint JSON byte identical at common origin; setup gate executed')


def check_setup_retries():
    class RetryDuringSettle(q.MockSession):
        def guard(self):
            if getattr(self,'retry_due',False):
                rt.READ_RETRIES.append(dict(t=90.,what='during settle'))
                self.retry_due=False
            super().guard()
    rt.READ_RETRIES.clear()
    rt.READ_RETRIES.extend(dict(t=90.,what=what) for what in ('previous invocation','setup1','setup2'))
    out=result();out['setup_complete']=False
    rt.retry_check(out['setup'],1)
    rt.READ_RETRIES.append(dict(t=90.,what='after setup snapshot'))
    s=RetryDuringSettle(None,{});s.start();s.retry_due=True
    s.data['power']=[dict(t=s.now()-(9-i)*.37,battery_w=1.) for i in range(10)]
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        q.run_mode('endurance',s,out,dict(q.SHORT,idle=0,gate=0,endurance=30),37.,lambda x:Path(tmp)/('run'+x),setup_mark=1)
    assert [r['what'] for r in out['setup']['read_retries']]==['setup1','setup2','after setup snapshot','during settle']
    assert out['setup']['read_retry_check'].startswith('NOT VALID — READ RETRIES (4 > 3)')
    assert 'setup read retry cap' in out['runs'][0]['issues']
    rt.READ_RETRIES.clear()
    print('PASS exact setup mark: retries after snapshot and during settle retained; prior invocation excluded; cap4>3 preserved')


if __name__=='__main__':
    check_settle();check_main_failure();check_post_origin_identical();check_setup_retries()
    print('PASS CAMPAIGN_P23C (offline mocks only; NOT VALID for hardware timing)')
