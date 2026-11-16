"""Offline VIS-2 checks: no servers, no root, no telemetry. Run: python3 -B test_vis2.py"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vis2_run as v

MM = ['--mmproj', str(v.v1.Q8), '--image-min-tokens', '70', '--image-max-tokens', '70']


def test_commands():
    a, b, c, d, m = (v.command(x) for x in 'ABCDM')
    assert a + MM == b and c + MM == d and m == b, 'only the projector flags may differ'
    assert c + v.v1.DRAFT_ARGS == a, 'only the drafter flags may differ'
    assert str(v.BIN).endswith('llama.cpp-vis/build/bin/llama-server') and a.count(str(v.BIN)) == 1
    for flag in ('--swa-full', '--cache-ram', '--reasoning', '-m'):
        assert flag in a
    assert '-lv' not in b, 'verbose logging would differ from the baseline cost'


def test_blocks():
    assert v.blocks('A,B,B,A,M') == ['A', 'B', 'B', 'A', 'M']
    for bad in ('A,X', '', 'T2'):
        try:
            v.blocks(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_payloads_and_sequence():
    gen, unc, cached = v.payloads()
    assert len(gen) == v.ROUNDS == len(cached)
    for _, body in gen:
        assert body['temperature'] == 0 and body['n_predict'] == v.GEN_N and body['cache_prompt'] is False
    assert unc[1]['cache_prompt'] is False and all(b['cache_prompt'] for _, b in cached)
    log_part = unc[1]['prompt'].split('\nSummarize')[0]
    assert all(b['prompt'].startswith(log_part) for _, b in cached), 'cached requests must share the U prefix'
    assert len(log_part) > 1500
    sa, sb = (v.sequence(x, False, gen, unc, cached, {}) for x in 'AB')
    assert sa == sb and len(sa) == 3 * v.ROUNDS
    assert [k for k, *_ in sa[:3]] == ['uncached', 'cached', 'gen']
    assert len(v.sequence('A', True, gen, unc, cached, {})) == 3
    sm = v.sequence('M', False, gen, unc, cached, {'img': 1})
    assert [k for k, *_ in sm] == ['gen', 'uncached', 'image', 'gen', 'uncached']
    assert sm[2][2] == '/v1/chat/completions' and sm[0][3] is sm[3][3]


def test_verify_config():
    real = v.get
    try:
        v.get = lambda p: {'modalities': {'vision': True}} if p == '/props' else [{'speculative': True}]
        assert v.verify_config('B') == dict(vision=True, speculative=[True])
        assert v.verify_config('M')
        for wrong in 'ACD':
            try:
                v.verify_config(wrong)
            except RuntimeError as e:
                assert 'configuration mismatch' in str(e)
                continue
            raise AssertionError(wrong)
    finally:
        v.get = real


def test_fresh_pss():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp, 'samples.jsonl')
        f.write_text(json.dumps(dict(t_start=0., t=.1, pid=7, pss_kib=1)) + '\n')
        assert v.fresh_pss(Path(tmp), 7, wait=.3)['pss_kib'] is None, 'stale sample must not count'
        import threading, time
        def later():
            time.sleep(.3)
            with f.open('a') as h:
                h.write(json.dumps(dict(t_start=time.monotonic(), t=time.monotonic(), pid=8, pss_kib=5)) + '\n')
                h.write(json.dumps(dict(t_start=time.monotonic(), t=time.monotonic(), pid=7, pss_kib=9)) + '\n')
        threading.Thread(target=later).start()
        assert v.fresh_pss(Path(tmp), 7, wait=3)['pss_kib'] == 9


def test_guarded_post_stops_in_flight():
    import time
    real = v.v1.post, v.telemetry, v.v1.check_cores
    calls = []
    def telemetry(bridge):
        calls.append(1)
        if len(calls) >= 2:
            raise RuntimeError('thermal stop')
    try:
        v.v1.post = lambda *a, **k: time.sleep(5)
        v.telemetry, v.v1.check_cores = telemetry, lambda pid: None
        t = time.monotonic()
        try:
            v.guarded_post(None, 1, '/completion', {}, 120)
        except RuntimeError as e:
            assert 'thermal stop' in str(e) and time.monotonic() - t < 2
        else:
            raise AssertionError('no stop')
        v.v1.post = lambda *a, **k: {'ok': 1}
        assert v.guarded_post(None, 1, '/completion', {}, 120) == {'ok': 1}
    finally:
        v.v1.post, v.telemetry, v.v1.check_cores = real


def test_sample_violations():
    good = dict(agent=False, battery_status='Discharging', capacity=70, cores_ok=True, status=1, bat_c=30)
    assert v.sample_violations([good]) == []
    assert v.sample_violations([dict(good, status=4)]) == ['thermal stop threshold']
    assert v.sample_violations([dict(good, bat_c=45)]) == ['thermal stop threshold']
    assert v.sample_violations([dict(good, capacity=49)]) == ['battery band']
    assert v.sample_violations([dict(good, capacity=100)]) == [], 'no upper battery bound (VIS2-BATTERY-FIX)'
    assert v.sample_violations([dict(good, battery_status='Charging')]) == ['battery band']


def test_telemetry_battery_and_stops():
    import time
    good = dict(t=time.monotonic(), capacity=100, battery_status='Discharging', agent=False, skin=30., status=0,
                bat_c=30., voltage_now_uV=4400000)
    bad = [dict(capacity=49), dict(battery_status='Charging'), dict(battery_status='Full'), dict(status=4),
           dict(bat_c=45.), dict(agent=True), dict(skin=None), dict(t=time.monotonic() - 13), dict(error='x')]
    with tempfile.TemporaryDirectory() as tmp:
        b = Path(tmp)
        for cap in (50, 81, 100):
            (b/'latest.json').write_text(json.dumps(dict(good, capacity=cap)))
            assert v.telemetry(b)['capacity'] == cap
            assert v.gate(b, 30.)['skin'] == 30.
        for change in bad:
            (b/'latest.json').write_text(json.dumps(dict(good, **change)))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    v.telemetry(b)
            except RuntimeError:
                continue
            raise AssertionError(change)
        (b/'latest.json').write_text(json.dumps(good))
        (b/'abort').touch()
        try:
            v.telemetry(b)
        except SystemExit:
            pass
        else:
            raise AssertionError('abort ignored')


def test_summarize():
    req = lambda kind, **k: dict(dict(kind=kind, prompt_per_second=None, prompt_tokens=10, prompt_ms=5., cache_n=0,
                                      predicted_per_second=None, generated_tokens=5, wall_ms=100., draft_n=None,
                                      draft_n_accepted=None, reply='x'), **k)
    rec = dict(index=1, block='B', label='VALID', reason='ok', load_s=9.5, reported={}, lmk={'ok': True},
               battery_start=dict(capacity=92, voltage_now_uV=4300000, t=1.),
               requests=[req('uncached', prompt_per_second=80.), req('cached', cache_n=590),
                         req('gen', predicted_per_second=9., draft_n=30, draft_n_accepted=12),
                         req('gen', predicted_per_second=11., draft_n=10, draft_n_accepted=8)],
               pss_after_load=dict(pss_kib=4 * 1024 * 1024), metrics=dict(peak_pss_kib=None))
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, '01_B_20261022T000000Z.json').write_text(json.dumps(rec))
        rec2 = dict(rec, index=2, requests=rec['requests'][:3])
        Path(tmp, '02_A_20261022T000001Z.json').write_text(json.dumps(rec2))
        Path(tmp, '01_B_20261022T000000Z.server.log').write_text('x')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            line, _ = v.summarize(tmp)
    assert line['gen_tok_s'] == 10 and line['draft_acceptance'] == .5 and line['uncached_pp_tok_s'] == 80
    assert line['cached_cache_n'] == [590] and line['pss_after_load_mib'] == 4096 and line['peak_pss_mib'] is None
    assert all(line['checks'].values()) and line['battery_start']['capacity'] == 92
    comp = json.loads(out.getvalue().splitlines()[-1])['comparability']
    assert comp['gen_tokens_identical'] is False and comp['uncached_prompt_n_identical'] is True


def test_dry_run():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        v.main(['--dry-run', '--blocks', 'A,B,B,A,M'])
    text = out.getvalue()
    assert text.startswith(v.DRY_LABEL) and '"M"' in text


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
