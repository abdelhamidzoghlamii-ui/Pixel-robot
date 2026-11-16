"""Offline VIS-3B checks: no servers, no root, no telemetry. Run: python3 -B test_vis3.py"""
import contextlib
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vis3_run as v

v1, v2 = v.v1, v.v2
REPO = v.REPO
T1_LOG = next((REPO/'benchmark/gemma_vision/vis1/runs/20261019T222505Z').glob('T1_*.server.log'))
A_LOG = next((REPO/'benchmark/gemma_vision/vis2/runs/20261022T114634Z').glob('00_A_*.server.log'))


@contextlib.contextmanager
def patched(*triples):
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in triples]
    try:
        for obj, name, value in triples:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in saved:
            setattr(obj, name, value)


def deployed_command(setup):
    """The exact shell command server_manager.start_setup builds; os.system and waits are stubbed."""
    sys.path.insert(0, str(REPO))
    import server_manager as sm
    calls = []
    with patched((sm.os, 'system', calls.append), (sm.time, 'sleep', lambda s: None),
                 (sm, 'wait_for_server', lambda port=8080, timeout=60: True), (sm, 'mtp_draft_ok', lambda: True)):
        with contextlib.redirect_stdout(io.StringIO()):
            sm.start_setup(setup)
    return shlex.split(calls[-1].replace(' 2>/dev/null &', ''))


def test_config():
    assert v.config('gen', 'old') == dict(test='gen', build='old', mtp=False, q8=False)
    assert v.config('mtp', 'new')['mtp'] and v.config('vision', 'new')['q8']
    assert v.config('prompt', 'new', mtp=True, q8=True) == dict(test='prompt', build='new', mtp=True, q8=True)
    for bad in (('gen', 'old', True, False), ('vision', 'old', False, False), ('cold_load', 'old', False, True),
                ('x', 'old', False, False), ('gen', 'b1609', False, False)):
        try:
            v.config(*bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_commands_match_deployed_flags():
    prefix = ['taskset', '-c', '4-7', 'timeout', '--signal=TERM', '--kill-after=3', '900']
    for cfg, setup in ((v.config('gen', 'old'), 'setup_q4'), (v.config('mtp', 'old'), 'setup_q4_mtp')):
        cmd = v.command(cfg)
        want = deployed_command(setup)
        want[want.index('--port') + 1] = '8091'
        assert cmd[:7] == prefix and cmd[7:] == want, (cmd[7:], want)
    old, new = v.command(v.config('gen', 'old')), v.command(v.config('gen', 'new'))
    assert old[7] == str(v1.BINS['b1609-dotprod']) and new[7] == str(v1.BINS['vis'])
    assert old[:7] + old[8:] == new[:7] + new[8:], 'only the binary may differ between old and new'
    for test in v.TESTS[:5]:
        assert v.command(v.config(test, 'new')) == new, 'every plain test launches the same server'
    vis = v.command(v.config('vision', 'new'))
    assert vis == new + ['--mmproj', str(v1.Q8), '--image-min-tokens', '70', '--image-max-tokens', '70']
    assert v.command(v.config('mtp', 'new')) == new + v1.DRAFT_ARGS


def test_payloads_per_test():
    seqs = {t: v.payloads(v.config(t, 'new')) for t in v.TESTS}
    assert seqs['cold_load'] == seqs['warm_load'] == (None, [])
    warm, seq = seqs['first_response']
    assert warm is None and len(seq) == 1 and seq[0][0] == 'first' and seq[0][3]['n_predict'] == v.FIRST_N
    warm, seq = seqs['prompt']
    assert warm[0] == 'first' and [k for k, *_ in seq] == ['uncached', 'cached'] * v.ROUNDS
    assert all(b['cache_prompt'] is False for k, _, _, b in seq if k == 'uncached')
    assert all(b['cache_prompt'] for k, _, _, b in seq if k == 'cached')
    assert seqs['gen'] == seqs['mtp'] and len(seqs['gen'][1]) == len(v2.GEN_IDS)
    for _, _, _, body in seqs['gen'][1]:
        assert body['temperature'] == 0 and body['n_predict'] == v2.GEN_N and body['cache_prompt'] is False
    assert seqs['gen'][1][0][3]['n_predict'] == v2.GEN_N, 'first-response body must not mutate gen[0]'
    warm, seq = seqs['vision']
    assert warm[1] == v.WARMUP_FRAME and [n for _, n, _, _ in seq] == list(v.VISION_FRAMES)
    assert v.WARMUP_FRAME not in v.VISION_FRAMES and {e for _, _, e, _ in seq} == {'/v1/chat/completions'}


def test_log_markers_on_archived_logs():
    old, new = (v.log_markers(p.read_text(errors='replace')) for p in (T1_LOG, A_LOG))
    for m in (old, new):
        assert set(m) == set(v.MARKERS), m
        assert m['load_start'] < m['vocab'] < m['kv_cache'] < m['draft_start'] < m['loaded']
    assert abs(old['loaded'] - 9.557) < .01 and abs(new['loaded'] - 8.689) < .01
    assert v.log_markers('1.02.003.004 I srv  llama_server: model loaded\n') == dict(loaded=62.003)


def test_freq():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for p, rows in (('policy4', '500000 10\n2348000 30\n'), ('policy6', '500000 0\n2850000 5\n')):
            (root/p/'stats').mkdir(parents=True)
            (root/p/'stats/time_in_state').write_text(rows)
        a = v.freq_state(root)
        (root/'policy4/stats/time_in_state').write_text('500000 10\n2348000 40\n')
        (root/'policy6/stats/time_in_state').write_text('500000 10\n2850000 15\n')
        d = v.freq_delta(a, v.freq_state(root))
    assert d['policy4'] == dict(ticks=10, mean_mhz=2348, top_share=1.0)
    assert d['policy6'] == dict(ticks=20, mean_mhz=1675, top_share=.5)
    assert v.freq_delta(a, a)['policy6']['mean_mhz'] is None


def test_mapped_libs():
    own = str(v1.BINS['vis'].parent)
    maps = (f'7a00-7b00 r-xp 0 fd:01 1 {own}/libggml-cpu.so.0.27.0\n'
            f'7b00-7c00 r--p 0 fd:01 2 {own}/libllama.so.0.7.0\n'
            '7c00-7d00 r-xp 0 fd:01 3 /system/lib64/libc.so\n')
    got = v.mapped_libs(1, 'vis', maps)
    assert got['ok'] and len(got['libs']) == 2
    assert not v.mapped_libs(1, 'b1609-dotprod', maps)['ok'], 'new libraries in an old server must fail'
    assert not v.mapped_libs(1, 'vis', '')['ok']


def test_verify():
    def fake(vision, spec, n_ctx=2048):
        return lambda p: ({'modalities': {'vision': vision}, 'build_info': 'b1'} if p == '/props'
                          else [{'speculative': spec, 'n_ctx': n_ctx}])
    with patched((v2, 'get', fake(False, False))):
        assert v.verify(v.config('gen', 'old'))['speculative'] == [False]
    with patched((v2, 'get', fake(True, True))):
        assert v.verify(v.config('prompt', 'new', True, True))
    for get, cfg in ((fake(False, True), v.config('gen', 'old')), (fake(False, False), v.config('vision', 'new')),
                     (fake(False, False), v.config('mtp', 'new')), (fake(False, False, 4096), v.config('gen', 'new')),
                     (fake(None, False), v.config('gen', 'new'))):
        with patched((v2, 'get', get)):
            try:
                v.verify(cfg)
            except RuntimeError:
                continue
            raise AssertionError(cfg)


def test_env_overrides_refused():
    v.check_env({'HOME': '/root', 'PATH': '/bin', 'LD_PRELOADX': '1'})
    for key in ('LLAMA_ARG_UBATCH', 'LLAMA_ARG_MMAP', 'GGML_NTHREADS', 'OMP_WAIT_POLICY', 'KMP_BLOCKTIME',
                'LD_LIBRARY_PATH', 'LD_PRELOAD'):
        try:
            v.check_env({'HOME': '/root', key: '1'})
        except SystemExit as e:
            assert key in str(e)
        else:
            raise AssertionError(key)


def test_prime_reads_every_byte():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp, 'a'), Path(tmp, 'b')
        a.write_bytes(b'x' * ((8 << 20) + 5))
        b.write_bytes(b'y' * 7)
        assert v.prime([a, b])['bytes'] == (8 << 20) + 12


class Proc:
    pid = 4242

    def poll(self):
        return None


def fake_run(tmp, cfg, fail_verify=False, libs_ok=True):
    out = Path(tmp)
    calls = dict(drop=0, prime=0, posts=[], stops=[])
    tele = dict(skin=30., status=0, bat_c=28., capacity=70, voltage_now_uV=4e6, mem_available_kib=1,
                max=dict(policy4=2348000, policy6=2850000), t=1.)

    def launch(cmd, logpath, bridge, live):
        live.update(proc=Proc(), pid=99)
        logpath.write_text(T1_LOG.read_text(errors='replace'))
        return 1.5

    def timed(bridge, pid, logpath, kind, name, endpoint, body, timeout):
        return dict(kind=kind, input=name, wall_ms=100., prompt_ms=50., prompt_tokens=10, generated_tokens=5,
                    predicted_per_second=10., prompt_per_second=200., cache_n=0 if kind != 'cached' else 9,
                    reply='ok', draft_n=None, draft_n_accepted=None, freq=None)

    def verify(c):
        if fail_verify:
            raise RuntimeError('configuration mismatch')
        return dict(vision=c['q8'], speculative=[c['mtp']], n_ctx=[2048], build_info='b')
    freq = dict(policy4={2348000: 10}, policy6={2850000: 10})
    with patched((v, 'launch', launch), (v, 'timed', timed), (v, 'verify', verify),
                 (v, 'mapped_libs', lambda pid, b: dict(libs=['/elsewhere/libllama.so'], ok=libs_ok)),
                 (v, 'drop_caches', lambda b: calls.__setitem__('drop', calls['drop'] + 1)),
                 (v, 'prime', lambda paths: calls.__setitem__('prime', calls['prime'] + 1) or dict(bytes=1, s=0)),
                 (v, 'freq_state', lambda root=None: freq),
                 (v2, 'telemetry', lambda b: tele), (v2, 'fresh_pss', lambda b, p: dict(pss_kib=2048)),
                 (v2, 'guarded_post', lambda b, p, e, body, t: calls['posts'].append(e) or {}),
                 (v1, 'check_cores', lambda pid: None), (v1, 'agent_found', lambda log=False: False),
                 (v1, 'own_stop', lambda proc, pid=None: calls['stops'].append(pid) or 0),
                 (v1, 'group_alive', lambda g: False), (v1, 'lmk_request', lambda b, e: dict(ok=True)),
                 (v1, 'metrics', lambda r: {})):
        rec = v.run_test(cfg, out, out)
        saved = json.loads((out/'result.json').read_text())
        target = json.loads((out/'target.json').read_text())
    return rec, calls, saved, target


def test_cold_load_measures_only_the_load():
    with tempfile.TemporaryDirectory() as tmp:
        rec, calls, saved, target = fake_run(tmp, v.config('cold_load', 'old'))
    assert rec['label'] == 'VALID', rec['reason']
    assert calls['drop'] == 1 and calls['prime'] == 0 and not calls['posts'] and not rec['requests']
    assert rec['load_s'] == 1.5 and rec['measure_start'] <= rec['measure_end'] and rec['end']['skin'] == 30.
    assert rec['freq']['policy6']['ticks'] == 0 and set(rec['markers']) == set(v.MARKERS)
    assert calls['stops'] == [99] and rec['exit_clean'] and saved['label'] == 'VALID' and target == dict(pid=None)


def test_gen_primes_warms_up_and_measures():
    with tempfile.TemporaryDirectory() as tmp:
        rec, calls, saved, _ = fake_run(tmp, v.config('gen', 'new'))
    assert rec['label'] == 'VALID', rec['reason']
    assert calls['drop'] == 0 and calls['prime'] == 1 and calls['posts'] == ['/completion'], 'one discarded warm-up'
    assert [r['kind'] for r in rec['requests']] == ['gen'] * len(v2.GEN_IDS)
    assert 'measure_start_snapshot' in rec and rec['stop_start'] >= rec['measure_end']
    with tempfile.TemporaryDirectory() as tmp:
        rec, calls, _, _ = fake_run(tmp, v.config('first_response', 'old'))
    assert not calls['posts'] and [r['kind'] for r in rec['requests']] == ['first'], 'no warm-up before first response'


def test_failures_stop_the_server():
    for kw, text in ((dict(fail_verify=True), 'mismatch'), (dict(libs_ok=False), 'outside its build')):
        with tempfile.TemporaryDirectory() as tmp:
            rec, calls, saved, _ = fake_run(tmp, v.config('gen', 'new'), **kw)
        assert rec['label'] == saved['label'] == 'NOT VALID' and text in rec['reason']
        assert calls['stops'] == [99] and not rec['requests'] and not calls['posts']


def test_done_before():
    spec = dict(config=dict(test='gen'), command=['x'], payload_sha256={}, code_sha256={'a': 1}, weight_sha256={},
                binary_sha256={}, constants={}, git_status='?? other')
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        assert v.done_before(spec, runs) is None
        for name, label, man in (('r1', 'NOT VALID', spec), ('r2', 'VALID', dict(spec, code_sha256={'a': 2})),
                                 ('old_format', 'VALID', None)):
            (runs/name).mkdir()
            if man:
                (runs/name/'run.json').write_text(json.dumps(dict(manifest=man)))
                (runs/name/'result.json').write_text(json.dumps(dict(label=label)))
            else:
                (runs/name/'00_O_x.json').write_text('{}')
        assert v.done_before(spec, runs) is None, 'invalid, different code and old-format folders do not count'
        (runs/'r3').mkdir()
        (runs/'r3/run.json').write_text(json.dumps(dict(manifest=dict(spec, git_status=''))))
        (runs/'r3/result.json').write_text(json.dumps(dict(label='VALID')))
        assert v.done_before(spec, runs) == runs/'r3', 'git status is not part of the fingerprint'


def test_preflight_refusals():
    cfg = v.config('gen', 'new')
    spec = dict(weight_sha256={str(p): h for p, h in v.WEIGHTS.items()})
    real = v.importlib.util.find_spec
    base = ((v1, 'free_ports', lambda: None), (v1, 'agent_matches', lambda: []), (v, 'check_env', lambda: None),
            (v, 'manifest', lambda c: spec))
    with patched(*base, (v.importlib.util, 'find_spec', lambda m: None if m == 'requests' else real(m))):
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                v.preflight(cfg)
        except SystemExit as e:
            assert 'requests' in str(e) and 'PREFLIGHT OK' not in out.getvalue()
        else:
            raise AssertionError('preflight must refuse without requests')
    with patched(*base, (v, 'done_before', lambda s: Path('runs/prior'))):
        try:
            v.preflight(cfg)
        except SystemExit as e:
            assert 'already measured' in str(e)
        else:
            raise AssertionError('preflight must refuse a repeated VALID test')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            v.preflight(cfg, repeat=True)
        assert 'PREFLIGHT OK' in out.getvalue()


def result(build, skin, gen_tps, capacity=70, caps=2850000, mhz=2000, tokens=5):
    req = dict(kind='gen', input='A5', wall_ms=100., prompt_ms=50., prompt_tokens=10, generated_tokens=tokens,
               predicted_per_second=gen_tps, prompt_per_second=100., cache_n=0, reply='ok', draft_n=None,
               draft_n_accepted=None)
    start = dict(skin=skin, status=0, bat_c=28., capacity=capacity, max=dict(policy0=1, policy4=2348000, policy6=caps))
    return dict(test='gen', build=build, mtp=False, q8=False, label='VALID', reason='', load_s=4., requests=[req],
                start=start, measure_start_snapshot=start, end=dict(skin=skin + 3), libs=dict(ok=True),
                pss_after_load=dict(pss_kib=3 << 20), exit_clean=True,
                freq={p: dict(ticks=10, mean_mhz=mhz, top_share=1.) for p in v.POLICIES},
                metrics=dict(peak_pss_kib=4 << 20, min_mem_available_kib=2 << 20, power=dict(mean_battery_w=4.)))


def manifest_of(rec, **over):
    cfg = dict(test=rec['test'], build=rec['build'], mtp=rec['mtp'], q8=rec['q8'])
    return dict(dict(config=cfg, command=v.command(cfg), payload_sha256={'A5': 'p'}, weight_sha256={'m': 'w'},
                     code_sha256={'c': 1}, constants=dict(rounds=4)), **over)


def report_of(a, b, ma=None, mb=None):
    with tempfile.TemporaryDirectory() as tmp:
        for name, rec, man in (('a', a, ma), ('b', b, mb)):
            Path(tmp, name).mkdir()
            Path(tmp, name, 'result.json').write_text(json.dumps(rec))
            Path(tmp, name, 'run.json').write_text(json.dumps(dict(manifest=man or manifest_of(rec))))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lines = v.report(Path(tmp, 'a'), Path(tmp, 'b'))
    return lines, json.loads(out.getvalue().splitlines()[-1])


def test_report_states_whether_conditions_match():
    lines, last = report_of(result('old', 30., 12.), result('new', 30.6, 9.))
    assert lines[0]['gen_tok_s'] == 12 and lines[1]['pss_after_load_mib'] == 3072 and lines[0]['skin_end'] == 33.
    assert last['second_over_first']['gen_tok_s'] == .75 and last['prompt_n_identical'] and last['gen_tokens_identical']
    assert last['configuration']['comparable'] and last['configuration']['builds'] == ['old', 'new']
    c = last['conditions']
    assert c['matched'] and c['measured_freq_within'] and c['start_skin'] == [30., 30.6]
    for b, failed in ((result('new', 31.5, 9.), 'start_skin_within'), (result('new', 30., 9., capacity=55), 'battery_within'),
                      (result('new', 30., 9., caps=1582000), 'start_caps_equal')):
        _, last = report_of(result('old', 30., 12.), b)
        assert not last['conditions']['matched'] and not last['conditions'][failed], failed
    _, last = report_of(result('old', 30., 12.), result('new', 30., 9., mhz=1800))
    assert last['conditions']['matched'] and not last['conditions']['measured_freq_within']
    other = result('new', 30., 9.)
    other['measure_start_snapshot'] = dict(other['start'], max=dict(other['start']['max'], policy0=5))
    assert report_of(result('old', 30., 12.), other)[1]['conditions']['start_caps_equal'], 'little cores ignored'


def test_report_checks_configuration_from_manifests():
    a, b = result('old', 30., 12.), result('new', 30., 9.)
    mtp = dict(b, mtp=True)
    for mb, failed in ((manifest_of(mtp), 'same_configuration'),
                       (manifest_of(b, payload_sha256={'A5': 'other'}), 'same_payloads'),
                       (manifest_of(b, code_sha256={'c': 2}), 'same_harness_code'),
                       (manifest_of(b, weight_sha256={'m': 'x'}), 'same_weights'),
                       (manifest_of(b, command=v.command(v.config('gen', 'new')) + ['-ub', '256']),
                        'same_command_except_binary')):
        cfg = report_of(a, b, mb=mb)[1]['configuration']
        assert not cfg['comparable'] and not cfg[failed], failed
        also = ('same_command_except_binary',) if failed == 'same_configuration' else ()  # the drafter flags
        assert all(val for k, val in cfg.items() if k not in (failed, 'comparable', 'builds') + also), (failed, cfg)


def test_manifest_hashes_only_selected_artifacts():
    cfg = v.config('gen', 'old')
    files = [str(p) for p in v.files_of(cfg)]
    assert str(v1.MODEL) in files and str(v1.Q8) not in files and str(v1.DRAFT) not in files
    assert not [f for f in files if 'llama.cpp-vis' in f], 'an old test must not depend on the new build'
    assert str(v1.Q8) in map(str, v.files_of(v.config('vision', 'new')))
    hashed = []
    with patched((v1, 'sha', lambda p: hashed.append(str(p)) or 'h'), (v, 'payloads', lambda c: (None, []))):
        spec = v.manifest(cfg)
    assert set(hashed) - {str(REPO/p) for p in v.code_paths()} == set(files)
    assert list(spec['weight_sha256']) == [str(v1.MODEL)] and str(v1.BINS['vis']) not in spec['binary_sha256']


def test_folder_suffix_is_normalized():
    def name(*argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            v.main(['--name', *argv])
        return out.getvalue().strip()
    assert name('--test', 'gen', '--build', 'new') == name('--build=new', '--test=gen') == '_gen_new'
    assert name('--test', 'cold_load', '--test', 'gen', '--build', 'old') == '_gen_old', 'last option wins, as argparse'
    assert name('--test', 'mtp', '--build', 'old') == '_mtp_old' and name('--test', 'vision', '--build', 'new') == '_vision_new'
    assert name('--test', 'prompt', '--build', 'new', '--q8', '--mtp', '--repeat') == '_prompt_new_mtp_q8_repeat'
    for bad in (['--test', 'vision', '--build', 'old'], ['--test', 'gen']):
        try:
            name(*bad)
        except SystemExit:
            continue
        raise AssertionError(bad)


def test_dry_run():
    for test in v.TESTS:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            v.main(['--dry-run', '--test', test, '--build', 'new'])
        text = out.getvalue()
        assert text.startswith(v.DRY_LABEL)
        assert json.loads(text.split('\n', 1)[1])['config']['test'] == test
    try:
        v.main(['--dry-run', '--test', 'vision', '--build', 'old'])
    except SystemExit as e:
        assert 'new build only' in str(e)
    else:
        raise AssertionError('vision on old must be refused')


def test_oneshot_single_test_launch():
    text = (Path(__file__).parent/'oneshot.sh').read_text()
    assert 'vis3_run.py' in text and 'vis2_run.py' not in text and 'gemma_vision/vis2/runs/.lock' in text
    assert 'pause 300' not in text and 'idle.json' not in text and '--preflight "${PARGS[@]}"' in text
    assert 'SUFFIX=$("$PY" -B "$R/vis3_run.py" --name "${PARGS[@]}") || exit 1' in text
    assert 'drop_caches' in text and 'settings put system screen_off_timeout' in text and 'check_battery' in text
    assert 'NAME=$(date -u +%Y%m%dT%H%M%SZ)$SUFFIX' in text


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
