"""VIS-2: Q8 projector residency vs Gemma text speed, MTP on/off; owner-only, motors off.

Reuses VIS-1 helpers (vis1_run.py, imported unchanged). --dry-run and --smoke are NOT VALID TIMING.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_spec = importlib.util.spec_from_file_location('vis1_run', REPO/'benchmark/gemma_vision/vis1/vis1_run.py')
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

NATIVE = v1.NATIVE
BIN = v1.BINS['vis']
LIBS = sorted((BIN.parent).glob('lib*.so*'))  # llama-server is a 6 KB launcher; the code is in these
WEIGHTS = {p: v1.EXPECTED_HASHES[p] for p in (v1.MODEL, v1.DRAFT, v1.Q8)}
CONFIGS = dict(A=dict(mtp=True, q8=False), B=dict(mtp=True, q8=True),
               C=dict(mtp=False, q8=False), D=dict(mtp=False, q8=True))
BLOCKS = ('A', 'B', 'C', 'D', 'M')  # M = mixed text/image check on config B
ROUNDS = 6           # timed: 6 x (uncached, cached, generation) per block; ABBA gives 12 per config
BLOCK_CAP_S = 600    # measurement cap per block; the server itself is capped at 900 s by timeout
REQ_TIMEOUT_S = 120
GEN_N = 128
GEN_IDS = v1.IDS[:ROUNDS]
IMAGE_FRAME = 'bench_photos__person_facing_150cm'
SMOKE_LABEL = 'SMOKE — NOT VALID TIMING'
DRY_LABEL = 'NOT VALID TIMING (offline dry-run; no servers, gates or measurements)'
# Fixed ~600-token robot log: a deterministic long prompt for prompt-processing throughput.
LOG = '\n'.join(f'Cycle {i}: front {40 + 7*i % 160} cm clear, left wall {60 + 11*i % 90} cm, '
                f'right door {"open" if i % 3 else "closed"}, person {"none" if i % 4 else "far left"}, '
                f'battery {80 - i // 4}%.' for i in range(30))


def config_of(block):
    return CONFIGS['B' if block == 'M' else block]


def command(block):
    c = config_of(block)
    cmd = ['taskset', '-c', '4-7', 'timeout', '--signal=TERM', '--kill-after=3', '900', str(BIN),
           '-m', str(v1.MODEL), '--host', '127.0.0.1', '--port', '8091', '--ctx-size', '4096',
           '--threads', '4', '--threads-batch', '4', '--parallel', '1', '--swa-full', '--cache-ram', '0',
           '--reasoning', 'off']
    if c['mtp']:
        cmd += v1.DRAFT_ARGS
    if c['q8']:
        cmd += ['--mmproj', str(v1.Q8), '--image-min-tokens', '70', '--image-max-tokens', '70']
    return cmd


def text(user, n_predict, cache):
    """Gemma raw prompt via chat.py (TPL-FIX, thinking off); greedy so A/B outputs can match."""
    body = v1.text_payload(user)
    body.update(temperature=0, top_k=1, seed=0, n_predict=n_predict, cache_prompt=cache)
    return body


def payloads():
    bench = v1.module('vis2_bench', 'benchmark/llm_objective_setting/bench.py')
    by_id = {p['id']: p for p in bench.PROMPTS_A + bench.PROMPTS_B + bench.PROMPTS_C}
    gen = [(i, text(' '.join(t['user'] for t in by_id[i]['text']) if by_id[i].get('multi')
                    else by_id[i]['text'], GEN_N, False)) for i in GEN_IDS]
    unc = ('U', text(LOG + '\nSummarize the log in one sentence.', 8, False))
    cached = [(f'K{k}', text(LOG + f'\nIn one sentence: what happened at cycle {k}?', 8, True))
              for k in range(ROUNDS)]
    return gen, unc, cached


def image_payload():
    _, images = v1.inputs()
    return dict(images)[IMAGE_FRAME]


def sequence(block, smoke, gen, unc, cached, image=None):
    """(kind, name, endpoint, body). Fixed order; identical between matched configs."""
    if block == 'M':
        g = gen[0]
        seq = [('gen', *g), ('uncached', *unc), ('image', IMAGE_FRAME, image), ('gen', *g), ('uncached', *unc)]
    else:
        seq = []
        for k in range(1 if smoke else ROUNDS):
            seq += [('uncached', *unc), ('cached', *cached[k]), ('gen', *gen[k])]
    return [(kind, name, '/v1/chat/completions' if kind == 'image' else '/completion', body)
            for kind, name, body in seq]


def get(path):
    with urllib.request.urlopen('http://127.0.0.1:8091'+path, timeout=10) as r:
        return json.load(r)


def verify_config(block):
    """The running server must report the intended projector and MTP state; never adjust flags."""
    c = config_of(block)
    props, slots = get('/props'), get('/slots')
    vision = props.get('modalities', {}).get('vision')
    spec = [s.get('speculative') for s in slots]
    got = dict(vision=vision, speculative=spec)
    if vision is not c['q8'] or spec != [c['mtp']]:
        raise RuntimeError(f'configuration mismatch: expected q8={c["q8"]} mtp={c["mtp"]}, server reports {got}')
    return got


def fresh_pss(bridge, pid, wait=5):
    """PSS of this server from a monitor sample begun after now, taken while the server is idle."""
    after, deadline = time.monotonic(), time.monotonic() + wait
    while time.monotonic() < deadline:
        for s in reversed(v1.rows(bridge/'samples.jsonl')):
            if s.get('t_start', -1) <= after:
                break
            if s.get('pid') == pid:
                return dict(pss_kib=s.get('pss_kib'), t=s['t'], after=after)
        time.sleep(.2)
    return dict(pss_kib=None, t=None, after=after, unavailable=f'no fresh sample within {wait} s')


def telemetry(bridge):
    """Marked copy of vis1_run.telemetry (VIS2-BATTERY-FIX): no 80 % upper bound; all other stops kept."""
    if (bridge/'abort').exists():
        raise SystemExit('owner aborted wrapper')
    d = json.loads((bridge/'latest.json').read_text())
    if time.monotonic() - d['t'] > 12:
        raise RuntimeError('native telemetry stale')
    if d.get('error'):
        raise RuntimeError('native telemetry failed: ' + d['error'])
    if d['capacity'] < 50 or d['battery_status'] != 'Discharging':
        raise RuntimeError('battery below 50% or not Discharging')
    if d['agent']:
        v1.log_agent_matches(d.get('agent_detail', []))
        raise RuntimeError('agent found')
    if d['skin'] is None or d['status'] is None:
        raise RuntimeError('missing VIRTUAL-SKIN or Android status')
    if d['status'] >= 4 or d['bat_c'] >= 45:
        raise RuntimeError('thermal stop (Android CRITICAL or battery >=45 C)')
    return d


def gate(bridge, idle):
    """Marked copy of vis1_run.gate using the local telemetry()."""
    start = time.monotonic()
    while True:
        d = telemetry(bridge)
        if v1.gate_ok(d['skin'], idle):
            return dict(skin=d['skin'], status=d['status'], waited_s=time.monotonic()-start, missed=False)
        if time.monotonic()-start >= 900:
            raise RuntimeError('gate missed: not cool after 15 minutes; no warm-start block')
        time.sleep(5)


def guarded_post(bridge, pid, endpoint, body, timeout):
    """Request in a worker; telemetry/abort/affinity stay enforced every 0.5 s while it runs.
    A violation raises here and run_block's finally stops the owned server, ending the request."""
    result = {}

    def work():
        try:
            result['data'] = v1.post(endpoint, body, timeout=timeout)
        except BaseException as e:
            result['error'] = e
    th = threading.Thread(target=work, daemon=True)
    th.start()
    while th.is_alive():
        th.join(.5)
        if th.is_alive():
            telemetry(bridge)
            v1.check_cores(pid)
    if 'error' in result:
        raise result['error']
    return result['data']


def sample_violations(samples):
    """Every recorded sample must satisfy the same stops telemetry() enforces live."""
    bad = []
    if any(s.get('agent') for s in samples):
        bad.append('agent')
    if any(s.get('battery_status') != 'Discharging' or s.get('capacity', -1) < 50 for s in samples):
        bad.append('battery band')
    if any(s.get('cores_ok') is False for s in samples):
        bad.append('affinity')
    if any((s.get('status') or 0) >= 4 or (s.get('bat_c') or 0) >= 45 for s in samples):
        bad.append('thermal stop threshold')
    return bad


def run_block(block, index, out, bridge, idle, seq, smoke):
    record = dict(phase_start=time.monotonic(), block=block, index=index, config=config_of(block),
                  label=SMOKE_LABEL if smoke else 'NOT VALID', reason='interrupted', requests=[],
                  command=command(block), started_epoch=time.time(), smoke=smoke)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = out/f'{index:02d}_{block}_{attempt}.json'
    logpath = out/f'{index:02d}_{block}_{attempt}.server.log'
    proc = pid = None
    try:
        if not smoke:
            record['gate'] = gate(bridge, idle)
        v1.free_ports()
        if v1.agent_found(log=True):
            raise RuntimeError('agent found before launch')
        if not smoke:  # cold load: same native cache-drop handshake as VIS-1
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
            record['gate_after_drop'] = gate(bridge, idle)
        d = telemetry(bridge)
        record['battery_start'] = dict(capacity=d['capacity'], voltage_now_uV=d['voltage_now_uV'], t=d['t'])
        with logpath.open('w') as log:
            began = time.monotonic()
            proc = v1.spawn(record['command'], log)
            while pid is None:
                if proc.poll() is not None or time.monotonic()-began > 10:
                    raise RuntimeError('server crashed or owned child not found')
                try:
                    pid = v1.server_pid(proc)
                except (OSError, RuntimeError):
                    time.sleep(.05)
            v1.write(bridge/'target.json', dict(pid=pid))
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('server crashed during load')
                try:
                    if v1.post('/health', timeout=1).get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                telemetry(bridge)
                if time.monotonic()-began >= 180:
                    raise RuntimeError('server load timeout')
                time.sleep(.2)
            record.update(server_pid=pid, load_s=time.monotonic()-began, loaded_at=time.monotonic())
            v1.check_cores(pid)
            record['reported'] = verify_config(block)
            record['pss_after_load'] = fresh_pss(bridge, pid)
            warm = next(x for x in seq if x[0] == 'gen')
            guarded_post(bridge, pid, warm[2], warm[3], REQ_TIMEOUT_S)  # discarded warm-up (generation)
            record['measure_start'] = start = time.monotonic()
            for kind, name, endpoint, body in seq:
                telemetry(bridge)
                v1.check_cores(pid)
                left = BLOCK_CAP_S - (time.monotonic()-start)
                if left <= 0:
                    raise RuntimeError(f'block cap {BLOCK_CAP_S} s exceeded')
                if kind == 'image':
                    record['pss_before_image'] = fresh_pss(bridge, pid)
                offset = logpath.stat().st_size
                t = time.monotonic()
                data = guarded_post(bridge, pid, endpoint, body, min(REQ_TIMEOUT_S, left))
                end = time.monotonic()
                if end-start > BLOCK_CAP_S:
                    raise RuntimeError(f'block cap {BLOCK_CAP_S} s exceeded')
                if kind == 'image':
                    record['pss_after_image'] = fresh_pss(bridge, pid)
                with logpath.open() as f:
                    f.seek(offset)
                    delta = f.read()
                row = v1.request_record(name, data, t, end, delta, 'V2' if kind == 'image' else 'T2')
                row.update(kind=kind, cache_n=row['timings'].get('cache_n'),
                           prompt_per_second=row['timings'].get('prompt_per_second'))
                record['requests'].append(row)
                v1.write(path, record)
            record['measure_end'] = time.monotonic()
            if proc.poll() is not None:
                raise RuntimeError('server crashed during measurement')
            v1.check_cores(pid)
            telemetry(bridge)
            record.update(label=SMOKE_LABEL if smoke else 'VALID', reason='completed fixed request sequence')
    except (Exception, SystemExit) as e:
        record.update(label='NOT VALID', reason=str(e))
    finally:
        record['stop_start'] = time.monotonic()
        v1.write(bridge/'target.json', dict(pid=None))
        if proc is not None:
            record['server_exit'] = v1.own_stop(proc, pid)
            record['exit_clean'] = record['server_exit'] == 0 and not v1.group_alive(proc.pid)
        record['stopped'] = time.monotonic()
        record['samples'] = [s for s in v1.rows(bridge/'samples.jsonl')
                             if record['phase_start']-1 <= s['t'] <= record['stopped']]
        record['lmk'] = v1.lmk_request(bridge, record['started_epoch'])
        record['metrics'] = v1.metrics(record)
        if record['label'] != 'NOT VALID':  # VALID and SMOKE both need a clean finish
            bad = []
            if proc is not None and not record['exit_clean']:
                bad.append(f"server cleanup not clean (exit {record['server_exit']})")
            if not record['lmk'].get('ok'):
                bad.append('LMK query failed')
            if sample_violations(record['samples']):
                bad.append('sample violation: ' + ', '.join(sample_violations(record['samples'])))
            if [e for e in v1.rows(bridge/'errors.jsonl') if record['phase_start'] <= e['t'] <= record['stop_start']]:
                bad.append('sampling failed')
            if bad:
                record.update(label='NOT VALID', reason='; '.join(bad))
        v1.write(path, record)
    return record


def blocks(arg):
    sel = arg.split(',')
    if not sel or any(b not in BLOCKS for b in sel):
        raise ValueError('--blocks takes a comma list of A,B,C,D,M (repeats allowed, e.g. A,B,B,A)')
    return sel


def code_paths():
    return ['benchmark/gemma_vision/vis2/vis2_run.py', 'benchmark/gemma_vision/vis2/oneshot.sh',
            'benchmark/gemma_vision/vis1/vis1_run.py', 'benchmark/gemma_vision/vis0/vis0.py',
            'benchmark/llm_objective_setting/bench.py', 'benchmark/campaign/power.py',
            'benchmark/campaign/runtime.py', 'chat.py', 'server_manager.py']


def manifest(selected, smoke, gen, unc, cached, image):
    allp = gen + [unc] + cached + ([(IMAGE_FRAME, image)] if image else [])
    return dict(blocks=selected, smoke=smoke, rounds=ROUNDS, gen_n=GEN_N, block_cap_s=BLOCK_CAP_S,
                commands={b: command(b) for b in sorted(set(selected))},
                payload_sha256={n: hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() for n, p in allp},
                code_sha256={p: v1.sha(REPO/p) for p in code_paths()},
                weight_sha256={str(p): v1.sha(p) for p in WEIGHTS},
                binary_sha256={str(p): v1.sha(p) for p in [BIN] + LIBS},
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=REPO, text=True))


def preflight():
    """Before and after the 5-minute idle: ports, agents, files/hashes, plus Debian imports or native root/battery/cpuset."""
    v1.free_ports()
    matches = v1.agent_matches()
    if matches:
        v1.log_agent_matches(matches)
        raise SystemExit('agent found; close Claude/AGY/Codex (see pid/comm/cmdline above)')
    missing = [str(p) for p in (*WEIGHTS, BIN) if not p.is_file()]
    if missing or not LIBS:
        raise SystemExit('missing file(s): ' + ', '.join(missing or ['llama.cpp-vis shared libraries']))
    bad = [str(p) for p, h in WEIGHTS.items() if v1.sha(p) != h]
    if bad:
        raise SystemExit('SHA-256 mismatch: ' + ', '.join(bad))
    if Path('/termux-home').is_dir():
        missing_mods = [m for m in v1.DEBIAN_IMPORTS if importlib.util.find_spec(m) is None]
        if missing_mods:
            raise SystemExit('Debian python3 is missing ' + ', '.join(missing_mods))
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


def mib(snap):
    return snap['pss_kib'] and round(snap['pss_kib']/1024) if snap else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def summarize(folder):
    """Per-block medians from saved records only; missing fields print as None (unavailable)."""
    lines = []
    for p in sorted(Path(folder).glob('[0-9][0-9]_[A-DM]_*Z.json')):
        d = json.loads(p.read_text())
        r, m = d['requests'], d.get('metrics', {})
        by = lambda k: [x for x in r if x['kind'] == k]
        dn = sum(x.get('draft_n') or 0 for x in by('gen'))
        da = sum(x.get('draft_n_accepted') or 0 for x in by('gen'))
        lines.append(dict(
            block=f"{d['index']:02d}_{d['block']}", label=d['label'], reason=d['reason'], load_s=med([d.get('load_s')]),
            battery_start=d.get('battery_start'),
            reported=d.get('reported'), uncached_pp_tok_s=med([x['prompt_per_second'] for x in by('uncached')]),
            uncached_prompt_n=sorted({x['prompt_tokens'] for x in by('uncached')}),
            cached_ms=med([x['prompt_ms'] for x in by('cached')]),
            cached_cache_n=sorted({x['cache_n'] for x in by('cached')}),
            gen_tok_s=med([x['predicted_per_second'] for x in by('gen')]),
            gen_tokens=[x['generated_tokens'] for x in by('gen')],
            gen_wall_ms=med([x['wall_ms'] for x in by('gen')]),
            draft_acceptance=round(da/dn, 3) if dn else None,
            image=[dict(wall_ms=round(x['wall_ms']), prompt_tokens=x['prompt_tokens'], usage=x.get('usage'))
                   for x in by('image')],
            pss_after_load_mib=mib(d.get('pss_after_load')),
            peak_pss_mib=m.get('peak_pss_kib') and round(m['peak_pss_kib']/1024),
            pss_image_before_after_mib=[mib(d.get('pss_before_image')), mib(d.get('pss_after_image'))],
            checks=dict(uncached_cache_n_zero=all(x['cache_n'] == 0 for x in by('uncached')),
                        cached_reuse=bool(by('cached')) and all((x['cache_n'] or 0) > 0 for x in by('cached')),
                        replies_nonempty=all((x.get('reply') or '').strip() for x in r)),
            min_mem_available_mib=m.get('min_mem_available_kib') and round(m['min_mem_available_kib']/1024),
            lmk=d.get('lmk'), server_exit=d.get('server_exit'), exit_clean=d.get('exit_clean'),
            skin=[m.get('skin_start'), m.get('skin_end')], status_max=m.get('status_max'),
            caps={k: round(v['percent']) for k, v in m.get('caps', {}).items()}, power=m.get('power')))
    for line in lines:
        print(json.dumps(line), flush=True)
    # VALID covers execution conditions only; these say whether blocks are comparable at all.
    timed = [x for x in lines if not x['block'].endswith('M') and x['label'] == 'VALID']
    print(json.dumps(dict(comparability=dict(
        gen_tokens_identical=len({tuple(x['gen_tokens']) for x in timed}) <= 1,
        uncached_prompt_n_identical=len({tuple(x['uncached_prompt_n']) for x in timed}) <= 1,
        all_checks_pass=all(all(x['checks'].values()) for x in lines)))), flush=True)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--blocks', default='A,B,B,A')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--summarize', type=Path)
    ap.add_argument('--bridge', type=Path)
    ap.add_argument('--preflight', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--monitor', type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.preflight:
        return preflight()
    if a.monitor:
        return v1.monitor(a.monitor)
    if a.summarize:
        return summarize(a.summarize)
    selected = blocks(a.blocks)
    if a.dry_run:
        gen, unc, cached = payloads()
        print(DRY_LABEL)
        print(json.dumps(dict(blocks=selected, commands={b: command(b) for b in sorted(set(selected))},
                              sequence=[(b, [(k, n, e) for k, n, e, _ in sequence(b, a.smoke, gen, unc, cached, {})])
                                        for b in sorted(set(selected))],
                              uncached_chars=len(unc[1]['prompt'])), indent=1))
        return
    if a.bridge is None or not Path('/termux-home').is_dir():
        raise SystemExit('launch only through native vis2/oneshot.sh into Debian')
    out = bridge = a.bridge.resolve()
    if not (bridge/'owner_launch').is_file():
        raise SystemExit('missing owner wrapper receipt')
    if v1.agent_found(log=True):
        raise SystemExit('agent found; no server started')
    v1.free_ports()
    telemetry(bridge)
    os.sched_setaffinity(0, {4, 5, 6, 7})
    v1.check_cores(os.getpid())
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, v1.interrupted)
    gen, unc, cached = payloads()
    image = image_payload() if 'M' in selected else None
    if any(v1.sha(p) != h for p, h in WEIGHTS.items()):
        raise SystemExit('model/projector/drafter SHA-256 mismatch')
    idle = None if a.smoke else json.loads((bridge/'idle.json').read_text())['skin']
    v1.write(out/'run.json', dict(manifest=manifest(selected, a.smoke, gen, unc, cached, image), idle_skin=idle,
             note='Short blocks; thermal equivalence between blocks is not established (#93). '
                  + (SMOKE_LABEL if a.smoke else 'ABBA order controls linear drift only.')))
    print('RUN_FOLDER='+str(out), flush=True)
    for i, b in enumerate(selected):
        result = run_block(b, i, out, bridge, idle, sequence(b, a.smoke, gen, unc, cached, image), a.smoke)
        print(f"{i:02d} {b}: {result['label']} — {result['reason']}", flush=True)
        if result['label'] == 'NOT VALID':
            raise SystemExit('stopped; later blocks not run. See README recovery.')
    summarize(out)
    print('Finished. Package RUN_FOLDER (see README); do not rerun for analysis.', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        for proc in v1.OWNED:
            v1.own_stop(proc)
