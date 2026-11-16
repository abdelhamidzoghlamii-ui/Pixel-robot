"""VIS-3B: one independent measurement per launch, b1609-dotprod (old) or b11716 (new); owner-only, motors off.

Reuses VIS-2 (vis2_run.py, imported unchanged) and through it VIS-1. --dry-run is NOT VALID TIMING.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_spec = importlib.util.spec_from_file_location('vis2_run', REPO/'benchmark/gemma_vision/vis2/vis2_run.py')
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)
v1 = v2.v1

BUILDS = dict(old='b1609-dotprod', new='vis')
TESTS = ('cold_load', 'warm_load', 'first_response', 'prompt', 'gen', 'mtp', 'vision')
LOAD_TESTS = ('cold_load', 'warm_load')
WEIGHTS = {p: v1.EXPECTED_HASHES[p] for p in (v1.MODEL, v1.DRAFT, v1.Q8)}
ROUNDS = 4            # prompt: 4 x (uncached, cached); vision: 4 frames; gen/mtp: the 6 VIS-2 prompts
FIRST_N = 16          # first-response request: gen[0] prompt, 16 tokens
IMAGE_TOKENS = 70     # Q8 projector budget, as VIS-2
VISION_FRAMES = v1.FRAMES[:ROUNDS]
WARMUP_FRAME = v1.FRAMES[ROUNDS]
TEST_CAP_S = 600
REQ_TIMEOUT_S = 120
LOAD_TIMEOUT_S = 180
# Cross-run comparison: start conditions count as matched within these limits (reported, never enforced).
MATCH_SKIN_C = 1.0
MATCH_BATTERY_PCT = 10
MATCH_FREQ_REL = .05
POLICIES = ('policy4', 'policy6')  # cores 4-5 (A78) and 6-7 (X1) = taskset 4-7
CPUFREQ = Path('/sys/devices/system/cpu/cpufreq')
DRY_LABEL = v2.DRY_LABEL
# First occurrence of each line in the default-verbosity server log; both builds print all of them.
MARKERS = dict(load_start='load_model: loading model', vocab="control-looking token", kv_cache='llama_kv_cache_iswa:',
               draft_start='common_speculative_init_result: loading draft model', loaded='llama_server: model loaded')
STAMP = re.compile(r'^(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) ')
LIB = re.compile(r'(/\S*/lib(?:ggml|llama|mtmd)[^/\s]*\.so[.\d]*)$')
ENV_OVERRIDE = re.compile(r'^(LLAMA_|GGML_|OMP_|KMP_|LD_LIBRARY_PATH$|LD_PRELOAD$)')


def config(test, build, mtp=False, q8=False):
    """mtp = 'gen' with the drafter; vision needs the projector, which only the new build serves."""
    if test not in TESTS or build not in BUILDS:
        raise ValueError(f'--test one of {TESTS}, --build old|new')
    if test == 'gen' and mtp:
        raise ValueError('generation with the drafter is --test mtp')
    q8 = q8 or test == 'vision'
    if q8 and build == 'old':
        raise ValueError('the Q8 projector is measured on the new build only')
    return dict(test=test, build=build, mtp=mtp or test == 'mtp', q8=q8)


def suffix(cfg, repeat=False):
    """Run-folder label from the validated configuration, independent of argument spelling or order."""
    return (f"_{cfg['test']}_{cfg['build']}" + ('_mtp' if cfg['mtp'] and cfg['test'] != 'mtp' else '')
            + ('_q8' if cfg['q8'] and cfg['test'] != 'vision' else '') + ('_repeat' if repeat else ''))


def command(cfg):
    """server_manager.start_setup('setup_q4') flags (ctx 2048), plus taskset/timeout/port; drafter/projector as asked."""
    cmd = ['taskset', '-c', '4-7', 'timeout', '--signal=TERM', '--kill-after=3', '900', str(v1.BINS[BUILDS[cfg['build']]]),
           '-m', str(v1.MODEL), '--port', '8091', '--ctx-size', '2048', '--threads', '4', '--threads-batch', '4',
           '--parallel', '1', '--swa-full', '--cache-ram', '0', '--host', '127.0.0.1']
    if cfg['mtp']:
        cmd += v1.DRAFT_ARGS
    if cfg['q8']:
        cmd += ['--mmproj', str(v1.Q8), '--image-min-tokens', str(IMAGE_TOKENS), '--image-max-tokens', str(IMAGE_TOKENS)]
    return cmd


def weights_of(cfg):
    return [v1.MODEL] + ([v1.DRAFT] if cfg['mtp'] else []) + ([v1.Q8] if cfg['q8'] else [])


def binaries_of(cfg):
    return [v1.BINS[BUILDS[cfg['build']]]] + libs_of(BUILDS[cfg['build']])


def files_of(cfg):
    """Everything the server reads at load: weights it uses plus its build's launcher and libraries."""
    return weights_of(cfg) + binaries_of(cfg)


def libs_of(build):
    return sorted(v1.BINS[build].parent.glob('lib*.so*'))


def payloads(cfg):
    """(warm-up, [(kind, name, endpoint, body)]) for the test; load tests send nothing."""
    gen, unc, cached = v2.payloads()
    first = ('F_' + gen[0][0], dict(gen[0][1], n_predict=FIRST_N))
    test = cfg['test']
    if test in LOAD_TESTS:
        return None, []
    if test == 'vision':
        images = dict(v1.inputs()[1])
        return (('image', WARMUP_FRAME, '/v1/chat/completions', images[WARMUP_FRAME]),
                [('image', f, '/v1/chat/completions', images[f]) for f in VISION_FRAMES])
    warm = ('first', *first[:1], '/completion', first[1])
    if test == 'first_response':
        return None, [warm]
    if test == 'prompt':
        seq = []
        for k in range(ROUNDS):
            seq += [('uncached', *unc), ('cached', *cached[k])]
        return warm, [(kind, name, '/completion', body) for kind, name, body in seq]
    return warm, [('gen', name, '/completion', body) for name, body in gen]


def stamp_s(m):
    mi, se, ms, us = map(int, m.groups())
    return round(mi*60 + se + ms/1e3 + us/1e6, 3)


def log_markers(text):
    out = {}
    for line in text.splitlines():
        m = STAMP.match(line)
        if m:
            for key, needle in MARKERS.items():
                if key not in out and needle in line:
                    out[key] = stamp_s(m)
    return out


def freq_state(root=CPUFREQ):
    """cpufreq time_in_state (10 ms ticks per frequency); readable without root."""
    state = {}
    for p in POLICIES:
        rows = (line.split() for line in (root/p/'stats/time_in_state').read_text().splitlines() if line.strip())
        state[p] = {int(f): int(t) for f, t in rows}
    return state


def freq_delta(a, b):
    """Mean policy frequency (MHz) and share of time at its top frequency between two snapshots."""
    out = {}
    for p in POLICIES:
        d = {f: b[p][f] - a[p].get(f, 0) for f in b[p]}
        total = sum(d.values())
        out[p] = dict(ticks=total, mean_mhz=round(sum(f*t for f, t in d.items())/total/1000) if total > 0 else None,
                      top_share=round(d[max(d)]/total, 3) if total > 0 else None)
    return out


def mapped_libs(pid, build, maps=None):
    """llama/ggml/mtmd libraries the running server actually mapped; all must come from its own build."""
    text = maps if maps is not None else Path(f'/proc/{pid}/maps').read_text()
    libs = sorted({m.group(1) for line in text.splitlines() if (m := LIB.search(line))})
    want = str(v1.BINS[build].parent) + '/'
    return dict(libs=libs, ok=bool(libs) and all(x.startswith(want) for x in libs))


def verify(cfg):
    """The running server must report the intended projector and MTP state and ctx 2048; never adjust flags."""
    props, slots = v2.get('/props'), v2.get('/slots')
    got = dict(vision=props.get('modalities', {}).get('vision'), speculative=[s.get('speculative') for s in slots],
               n_ctx=[s.get('n_ctx') for s in slots], build_info=props.get('build_info'))
    if got['vision'] is not cfg['q8'] or got['speculative'] != [cfg['mtp']] or got['n_ctx'] != [2048]:
        raise RuntimeError(f"configuration mismatch: expected vision={cfg['q8']} mtp={cfg['mtp']} n_ctx=2048, "
                           f'server reports {got}')
    return got


def snapshot(bridge):
    """Operating conditions now: VIS-2 telemetry (with all its stops) plus cpufreq residency counters."""
    d = v2.telemetry(bridge)
    keep = ('skin', 'status', 'bat_c', 'capacity', 'voltage_now_uV', 'battery_w', 'mem_available_kib', 'max',
            'cpuinfo_max_khz', 'cpu_mc', 't')
    return dict({k: d.get(k) for k in keep}, freq=freq_state())


def drop_caches(bridge):
    """Same native cache-drop handshake as VIS-1/VIS-2."""
    (bridge/'drop_done').unlink(missing_ok=True)
    (bridge/'drop_request').touch()
    began = time.monotonic()
    while not (bridge/'drop_done').exists():
        if time.monotonic()-began > 30:
            raise RuntimeError('cache-drop handshake timeout')
        time.sleep(.1)
    if (bridge/'drop_done').read_text().strip() != 'ok':
        raise RuntimeError('cache drop failed')
    (bridge/'drop_request').unlink()
    (bridge/'drop_done').unlink()


def prime(paths):
    """Read every file once so a non-cold test does not depend on what ran before (page cache warm)."""
    began, n = time.monotonic(), 0
    for p in paths:
        with Path(p).open('rb') as f:
            while chunk := f.read(8 << 20):
                n += len(chunk)
    return dict(bytes=n, s=round(time.monotonic()-began, 2))


def launch(cmd, logpath, bridge, live):
    """Spawn the owned server and wait for /health; load_s = spawn to /health ok (server readiness)."""
    v1.free_ports()
    with logpath.open('w') as log:
        began = time.monotonic()
        live['proc'] = proc = v1.spawn(cmd, log)
    while live.get('pid') is None:
        if proc.poll() is not None or time.monotonic()-began > 10:
            raise RuntimeError('server crashed or owned child not found')
        try:
            live['pid'] = v1.server_pid(proc)
        except (OSError, RuntimeError):
            time.sleep(.05)
    v1.write(bridge/'target.json', dict(pid=live['pid']))
    while True:
        if proc.poll() is not None:
            raise RuntimeError('server crashed during load')
        try:
            if v1.post('/health', timeout=1).get('status') == 'ok':
                break
        except (OSError, ValueError):
            pass
        v2.telemetry(bridge)
        if time.monotonic()-began >= LOAD_TIMEOUT_S:
            raise RuntimeError('server load timeout')
        time.sleep(.05)
    return time.monotonic()-began


def stop(live, bridge, record):
    v1.write(bridge/'target.json', dict(pid=None))
    proc, pid = live.pop('proc', None), live.pop('pid', None)
    if proc is not None:
        began = time.monotonic()
        record['server_exit'] = v1.own_stop(proc, pid)
        record['stop_s'] = time.monotonic()-began
        record['exit_clean'] = record['server_exit'] == 0 and not v1.group_alive(proc.pid)


def timed(bridge, pid, logpath, kind, name, endpoint, body, timeout):
    offset = logpath.stat().st_size
    f0, t = freq_state(), time.monotonic()
    data = v2.guarded_post(bridge, pid, endpoint, body, timeout)
    end, f1 = time.monotonic(), freq_state()
    with logpath.open() as f:
        f.seek(offset)
        delta = f.read()
    row = v1.request_record(name, data, t, end, delta, 'V2' if kind == 'image' else 'T2')
    row.update(kind=kind, cache_n=row['timings'].get('cache_n'),
               prompt_per_second=row['timings'].get('prompt_per_second'), freq=freq_delta(f0, f1))
    return row


def run_test(cfg, out, bridge):
    record = dict(cfg, phase_start=time.monotonic(), label='NOT VALID', reason='interrupted', requests=[],
                  command=command(cfg), started_epoch=time.time())
    path, logpath = out/'result.json', out/'server.log'
    live = {}
    warm, seq = payloads(cfg)
    try:
        if v1.agent_found(log=True):
            raise RuntimeError('agent found before launch')
        if cfg['test'] == 'cold_load':
            drop_caches(bridge)
        else:
            record['primed'] = prime(files_of(cfg))
        record['start'] = snapshot(bridge)
        if cfg['test'] in LOAD_TESTS:
            record['measure_start'] = time.monotonic()
        record['load_s'] = launch(record['command'], logpath, bridge, live)
        if cfg['test'] in LOAD_TESTS:
            record['measure_end'] = time.monotonic()
            record['end'] = snapshot(bridge)
        pid = live['pid']
        v1.check_cores(pid)
        record['reported'] = verify(cfg)
        record['libs'] = mapped_libs(pid, BUILDS[cfg['build']])
        if not record['libs']['ok']:
            raise RuntimeError(f"server mapped libraries outside its build: {record['libs']['libs']}")
        record['pss_after_load'] = v2.fresh_pss(bridge, pid)
        if warm:  # discarded; first_response has none, its single request is the measurement
            v2.guarded_post(bridge, pid, warm[2], warm[3], REQ_TIMEOUT_S)
        if seq:
            record['measure_start'] = start = time.monotonic()
            record['measure_start_snapshot'] = snapshot(bridge)
            for kind, name, endpoint, body in seq:
                v2.telemetry(bridge)
                v1.check_cores(pid)
                left = TEST_CAP_S - (time.monotonic()-start)
                if left <= 0:
                    raise RuntimeError(f'test cap {TEST_CAP_S} s exceeded')
                record['requests'].append(timed(bridge, pid, logpath, kind, name, endpoint, body, min(REQ_TIMEOUT_S, left)))
                v1.write(path, record)
            record['measure_end'] = time.monotonic()
            record['end'] = snapshot(bridge)
        record['freq'] = freq_delta((record.get('measure_start_snapshot') or record['start'])['freq'], record['end']['freq'])
        record['pss_end'] = v2.fresh_pss(bridge, pid)
        if live['proc'].poll() is not None:
            raise RuntimeError('server crashed during the test')
        v1.check_cores(pid)
        record['stop_start'] = time.monotonic()  # monitor errors from the server's own exit come after this
        stop(live, bridge, record)
        if not record['exit_clean']:
            raise RuntimeError(f"server cleanup not clean (exit {record['server_exit']})")
        record.update(label='VALID', reason='completed ' + cfg['test'])
    except (Exception, SystemExit) as e:
        record.update(label='NOT VALID', reason=str(e))
    finally:
        record.setdefault('stop_start', time.monotonic())
        stop(live, bridge, record)
        record['stopped'] = time.monotonic()
        if logpath.exists():
            record['markers'] = log_markers(logpath.read_text(errors='replace'))
        record['samples'] = [s for s in v1.rows(bridge/'samples.jsonl')
                             if record['phase_start']-1 <= s['t'] <= record['stopped']]
        record['lmk'] = v1.lmk_request(bridge, record['started_epoch'])
        record['metrics'] = v1.metrics(record)
        if record['label'] == 'VALID':
            bad = [] if record['lmk'].get('ok') else ['LMK query failed']
            if v2.sample_violations(record['samples']):
                bad.append('sample violation: ' + ', '.join(v2.sample_violations(record['samples'])))
            if [e for e in v1.rows(bridge/'errors.jsonl') if record['phase_start'] <= e['t'] <= record['stop_start']]:
                bad.append('sampling failed')
            if bad:
                record.update(label='NOT VALID', reason='; '.join(bad))
        v1.write(path, record)
    return record


def code_paths():
    return ['benchmark/gemma_vision/vis3/vis3_run.py', 'benchmark/gemma_vision/vis3/oneshot.sh'] + v2.code_paths()


def manifest(cfg):
    warm, seq = payloads(cfg)
    return dict(config=cfg, command=command(cfg),
                payload_sha256={n: hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()
                                for _, n, _, b in ([warm] if warm else []) + seq},
                code_sha256={p: v1.sha(REPO/p) for p in code_paths()},
                weight_sha256={str(p): v1.sha(p) for p in weights_of(cfg)},
                binary_sha256={str(p): v1.sha(p) for p in binaries_of(cfg)},
                constants=dict(rounds=ROUNDS, first_n=FIRST_N, image_tokens=IMAGE_TOKENS, test_cap_s=TEST_CAP_S),
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=REPO, text=True))


FINGERPRINT = ('config', 'command', 'payload_sha256', 'code_sha256', 'weight_sha256', 'binary_sha256', 'constants')


def done_before(spec, runs=HERE/'runs'):
    """A VALID result for the same test, configuration, code, payloads, weights and binaries already exists."""
    want = {k: spec[k] for k in FINGERPRINT}
    for result in sorted(runs.glob('*/result.json')):
        try:
            prior = json.loads((result.parent/'run.json').read_text())['manifest']
            if json.loads(result.read_text()).get('label') == 'VALID' and {k: prior.get(k) for k in FINGERPRINT} == want:
                return result.parent
        except (OSError, ValueError, KeyError):
            continue
    return None


def check_env(env=os.environ):
    """Both servers read LLAMA_ARG_* (batch, ubatch, flash attention, mmap, ...) and OpenMP variables from the
    environment, which the command line would not show; refuse any so both run on their defaults."""
    found = sorted(k for k in env if ENV_OVERRIDE.match(k))
    if found:
        raise SystemExit('environment overrides server settings; unset: ' + ', '.join(found))


def preflight(cfg, repeat=False):
    """Before launch, native and Debian: ports, agents, files/hashes, no shadowing libs, not already done,
    plus Debian env/imports or native root/battery/cpuset."""
    v1.free_ports()
    matches = v1.agent_matches()
    if matches:
        v1.log_agent_matches(matches)
        raise SystemExit('agent found; close Claude/AGY/Codex (see pid/comm/cmdline above)')
    missing = [str(p) for p in files_of(cfg) if not p.is_file()]
    if missing or not libs_of(BUILDS[cfg['build']]):
        raise SystemExit('missing file(s): ' + ', '.join(missing or ['build shared libraries']))
    # Both launchers list $PREFIX/lib before their own build dir in RUNPATH: a packaged libllama would shadow both.
    shadow = sorted(p.name for p in (v1.NATIVE.parent/'usr/lib').glob('lib*') if re.match(r'lib(ggml|llama|mtmd)', p.name))
    if shadow:
        raise SystemExit('Termux $PREFIX/lib would shadow the build libraries: ' + ', '.join(shadow))
    freq_state()
    if Path('/termux-home').is_dir():  # Debian: the side that builds payloads and runs the servers
        check_env()
        needed = ['requests'] + (['PIL'] if cfg['test'] == 'vision' else [])
        if [m for m in needed if importlib.util.find_spec(m) is None]:
            raise SystemExit('Debian python3 is missing ' + ', '.join(needed) + ' — run: apt-get install python3-requests python3-pil')
        spec = manifest(cfg)
        bad = [p for p in weights_of(cfg) if spec['weight_sha256'][str(p)] != WEIGHTS[p]]
        if bad:
            raise SystemExit('SHA-256 mismatch: ' + ', '.join(map(str, bad)))
        if not repeat and (prior := done_before(spec)):
            raise SystemExit(f'already measured VALID with identical code and configuration: {prior} (--repeat to redo)')
    else:
        if subprocess.run(['su', '-c', 'id'], capture_output=True, timeout=10).returncode != 0:
            raise SystemExit('no root (su) access; telemetry needs root')
        cap, status = v1.root(f'cat {v1.BAT}/capacity {v1.BAT}/status', 'preflight_battery').split()[:2]
        if not (cap.isdigit() and int(cap) >= 50 and status == 'Discharging'):
            raise SystemExit('battery below 50% or not Discharging; unplug the charger')
        before = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {4, 5, 6, 7})
        stuck = os.sched_getaffinity(0) == {4, 5, 6, 7}
        os.sched_setaffinity(0, before)
        if not stuck:
            raise SystemExit('Termux cpuset blocks cores 4-7; bring Termux to the foreground, screen on')
    print('PREFLIGHT OK', flush=True)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def summary(folder):
    """One result, from saved files only; missing fields are None (unavailable)."""
    d = json.loads((Path(folder)/'result.json').read_text())
    r, m = d['requests'], d.get('metrics', {})
    by = lambda k: [x for x in r if x['kind'] == k]
    dn = sum(x.get('draft_n') or 0 for x in r)
    start = d.get('measure_start_snapshot') or d.get('start') or {}
    return dict(
        folder=str(folder), test=d['test'], build=d['build'], mtp=d['mtp'], q8=d['q8'], label=d['label'],
        reason=d['reason'], load_s=med([d.get('load_s')]), markers=d.get('markers'),
        first_ms=med([x['wall_ms'] for x in by('first')]),
        uncached_pp_tok_s=med([x['prompt_per_second'] for x in by('uncached')]),
        cached_ms=med([x['prompt_ms'] for x in by('cached')]),
        gen_tok_s=med([x['predicted_per_second'] for x in by('gen')]),
        draft_acceptance=round(sum(x.get('draft_n_accepted') or 0 for x in r)/dn, 3) if dn else None,
        image_wall_ms=med([x['wall_ms'] for x in by('image')]),
        image_tokens=sorted({x.get('image_tokens') for x in by('image')}, key=str),
        prompt_n={x['input']: x['prompt_tokens'] for x in r}, gen_tokens={x['input']: x['generated_tokens'] for x in r},
        checks=dict(uncached_cache_n_zero=all(x['cache_n'] == 0 for x in by('uncached')),
                    cached_reuse=all((x['cache_n'] or 0) > 0 for x in by('cached')),
                    replies_nonempty=all((x.get('reply') or '').strip() for x in r),
                    libs_ok=(d.get('libs') or {}).get('ok')),
        pss_after_load_mib=(d.get('pss_after_load') or {}).get('pss_kib') and round(d['pss_after_load']['pss_kib']/1024),
        peak_pss_mib=m.get('peak_pss_kib') and round(m['peak_pss_kib']/1024),
        min_mem_available_mib=m.get('min_mem_available_kib') and round(m['min_mem_available_kib']/1024),
        mean_battery_w=(m.get('power') or {}).get('mean_battery_w'),
        start=dict(skin=start.get('skin'), status=start.get('status'), bat_c=start.get('bat_c'),
                   capacity=start.get('capacity'), max=start.get('max')),
        skin_end=(d.get('end') or {}).get('skin'), status_max=m.get('status_max'),
        caps={k: round(v['percent']) for k, v in m.get('caps', {}).items()},
        freq=d.get('freq'), lmk=d.get('lmk'), exit_clean=d.get('exit_clean'))


def conditions(a, b):
    """Whether two results ran under matched operating conditions; each criterion reported separately."""
    sa, sb = a['start'], b['start']
    known = lambda *xs: all(x is not None for x in xs)
    c = dict(
        same_test=a['test'] == b['test'],
        both_valid=a['label'] == b['label'] == 'VALID',
        start_skin_within=known(sa['skin'], sb['skin']) and abs(sa['skin']-sb['skin']) <= MATCH_SKIN_C,
        start_status_normal=sa['status'] == sb['status'] == 0,
        battery_within=known(sa['capacity'], sb['capacity']) and abs(sa['capacity']-sb['capacity']) <= MATCH_BATTERY_PCT,
        start_caps_equal=bool(sa['max'] and sb['max']) and all(sa['max'].get(p) == sb['max'].get(p) for p in POLICIES))
    fa, fb = (x['freq'] or {} for x in (a, b))
    mhz = {p: [(f.get(p) or {}).get('mean_mhz') for f in (fa, fb)] for p in POLICIES}
    measured = all(known(*v) and abs(v[0]-v[1]) <= MATCH_FREQ_REL*max(v) for v in mhz.values())
    return dict(c, matched=all(c.values()), start_skin=[sa['skin'], sb['skin']], measured_mean_mhz=mhz,
                measured_freq_within=measured,
                note='matched = start conditions only; measured frequency also reflects each build\'s own load')


def comparable(ma, mb):
    """Whether two runs measured the same thing; only the build (binary path and its hashes) may differ."""
    same_but_build = lambda c: {k: v for k, v in c.items() if k != 'build'}
    ca, cb = ma['command'], mb['command']
    c = dict(same_configuration=same_but_build(ma['config']) == same_but_build(mb['config']),
             same_command_except_binary=len(ca) == len(cb) and ca[:7] + ca[8:] == cb[:7] + cb[8:],
             same_payloads=ma['payload_sha256'] == mb['payload_sha256'],
             same_weights=ma['weight_sha256'] == mb['weight_sha256'],
             same_harness_code=ma['code_sha256'] == mb['code_sha256'],
             same_constants=ma['constants'] == mb['constants'])
    return dict(c, comparable=all(c.values()), builds=[ma['config']['build'], mb['config']['build']])


def report(*folders):
    lines = [summary(f) for f in folders]
    for line in lines:
        print(json.dumps(line), flush=True)
    if len(lines) == 2:
        a, b = lines
        num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
        ratio = {k: round(b[k]/a[k], 3) for k in a if num(a[k]) and num(b[k]) and a[k]}
        ma, mb = (json.loads((Path(f)/'run.json').read_text())['manifest'] for f in folders)
        print(json.dumps(dict(second_over_first=ratio, configuration=comparable(ma, mb), conditions=conditions(a, b),
                              prompt_n_identical=a['prompt_n'] == b['prompt_n'],
                              gen_tokens_identical=a['gen_tokens'] == b['gen_tokens'])), flush=True)
    return lines


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--test', choices=TESTS)
    ap.add_argument('--build', choices=tuple(BUILDS))
    ap.add_argument('--mtp', action='store_true', help='load the MTP drafter (implied by --test mtp)')
    ap.add_argument('--q8', action='store_true', help='load the Q8 projector, new build only (implied by --test vision)')
    ap.add_argument('--repeat', action='store_true', help='measure again although an identical VALID result exists')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--report', type=Path, nargs='+', help='print one result, or two plus their comparison')
    ap.add_argument('--bridge', type=Path)
    ap.add_argument('--preflight', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--monitor', type=Path, help=argparse.SUPPRESS)
    ap.add_argument('--name', action='store_true', help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.monitor:
        return v1.monitor(a.monitor)
    if a.report:
        return report(*a.report)
    if not (a.test and a.build):
        raise SystemExit('--test and --build are required')
    try:
        cfg = config(a.test, a.build, a.mtp, a.q8)
    except ValueError as e:
        raise SystemExit(str(e))
    if a.name:
        return print(suffix(cfg, a.repeat))
    if a.preflight:
        return preflight(cfg, a.repeat)
    if a.dry_run:
        warm, seq = payloads(cfg)
        print(DRY_LABEL)
        print(json.dumps(dict(config=cfg, command=command(cfg), warmup=warm and warm[1],
                              sequence=[(k, n, e) for k, n, e, _ in seq]), indent=1))
        return
    if a.bridge is None or not Path('/termux-home').is_dir():
        raise SystemExit('launch only through native vis3/oneshot.sh into Debian')
    out = bridge = a.bridge.resolve()
    if not (bridge/'owner_launch').is_file():
        raise SystemExit('missing owner wrapper receipt')
    if v1.agent_found(log=True):
        raise SystemExit('agent found; no server started')
    check_env()
    v1.free_ports()
    v2.telemetry(bridge)
    os.sched_setaffinity(0, {4, 5, 6, 7})
    v1.check_cores(os.getpid())
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, v1.interrupted)
    spec = manifest(cfg)
    if any(spec['weight_sha256'][str(p)] != WEIGHTS[p] for p in weights_of(cfg)):
        raise SystemExit('model/drafter/projector SHA-256 mismatch')
    if not a.repeat and (prior := done_before(spec)):
        raise SystemExit(f'already measured VALID: {prior} (--repeat to redo)')
    v1.write(out/'run.json', dict(manifest=spec, repeat=a.repeat,
             note='One independent measurement; no thermal-equivalence gate. Compare with --report A B, '
                  'which states whether start conditions matched.'))
    print('RUN_FOLDER='+str(out), flush=True)
    result = run_test(cfg, out, bridge)
    print(f"{cfg['test']} {cfg['build']}{' mtp' if cfg['mtp'] else ''}{' q8' if cfg['q8'] else ''}: "
          f"{result['label']} — {result['reason']}", flush=True)
    report(out)
    if result['label'] != 'VALID':
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    finally:
        for proc in v1.OWNED:
            v1.own_stop(proc)
