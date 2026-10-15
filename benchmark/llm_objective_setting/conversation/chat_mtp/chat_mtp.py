#!/data/data/com.termux/files/usr/bin/python
"""robot-chat speed with and without the MTP drafter (task MTP-A). Research only: no motors, no camera, no main.py.
Native Termux only; started by oneshot.sh in this folder (thermal logger, 5 min idle, screen on, no agent).

Configs: OFF = chat.py's Gemma 4 E2B Q4_0 command without the drafter, ON = with it, both from chat.server_cmd and
pinned like chat.py (taskset -c 4-7); every request is chat.chat_payload(chat.build_prompt('gemma', [one user
message])): ctx 4096, temperature 0.7, n_predict 512, stream false. Prompts: the 13 DECISIONS #120 ids from bench.py
(the source conv_speed_mtp.py used), user text only, a fresh history per prompt; A5's three user turns are joined
into one message.

Blocks (server spawn to exit within 180 s): skin gate (power_map.thermal_gate: VIRTUAL-SKIN <= idle + 1.5 degC, at
most 8 min, else WARM START), start the server (load s = spawn to /health ok), then the config's next prompts; no
prompt starts after 150 s. A watchdog KILLs the server 175 s after the spawn, which ends any wait; a reply not in by
then is dropped and redone later. Server stopped after every block (TERM, KILL after 3 s). Block order
OFF, ON, ON, OFF, OFF, ON, ... (a finished config's turn is skipped) until both configs ran every prompt once.
Losing cores 4-7 drops the running prompt, ends the block and waits for the cores.
Every 5 s (coresidency.sampler_loop): MemAvailable, llama-server PSS and battery power (coresidency.root_sample:
-(current_now * voltage_now) / 1e12, camera_heat.py's method). Per block: VmHWM, peak PSS, minimum MemAvailable,
mean power, energy per generated token, skin and Android thermal status at start and end, the server's
Cpus_allowed_list. Draft acceptance = sum(draft_n_accepted) / sum(draft_n) from the server's timings
(conv_speed_mtp.py). TTFT is not measurable (chat's payload is not streamed): prompt_s = the server's prompt_ms.

Refuses to start: not native Termux, a claude/codex/agy process, battery < 80 %, not discharging, port 8080 in use,
model or drafter SHA-256 not the conv_mtp/ARTIFACTS.md values.

Usage: chat_mtp.py [--smoke] [--resume RUN_DIR] [--thermal-log PATH]
  --smoke   one prompt per config, gate at most 20 s
Writes ~/robot_runs/chat_mtp_<UTC>[_smoke]/: run.json, prompts.jsonl, blocks.jsonl, report.txt, server logs.
"""
import argparse
import hashlib
import http.client
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2]  # robot/benchmark
sys.path[:0] = [str(BENCH.parent), str(BENCH / 'coresidency'), str(BENCH / 'power_map'),
                str(BENCH / 'llm_objective_setting')]
import chat  # noqa: E402  the command and payload under test
import server_manager as sm  # noqa: E402  drafter path (MTP_DRAFT)
import coresidency as cr  # noqa: E402  native check, root, sampling, cpuset guard, signal deferral
import power_map as pm  # noqa: E402  the skin gate
import bench  # noqa: E402  the #110 prompt texts

OUT_ROOT = cr.HOME / 'robot_runs'
THERMAL_LOG = OUT_ROOT / 'chat_mtp' / 'thermal.log'  # oneshot.sh's logger
MODEL = str(cr.HOME / 'models' / chat.MTP_MODEL)
SHA256 = {MODEL: '8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52',  # conv_mtp/ARTIFACTS.md
          sm.MTP_DRAFT: '9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919'}
IDS = 'A5 A6 B1 B2 B6 B7 C1 C6 C7 C8 C9 C10 C11'.split()  # conv_speed_mtp.IDS
ORDER = ('OFF', 'ON', 'ON', 'OFF')  # block n runs ORDER[n % 4]
BLOCK_S, LAST_START_S, SMOKE_GATE_S = 180, 150, 20
TERM_WAIT_S = 3  # stop(): KILL when TERM has not ended the server after this
DEADLINE_S = BLOCK_S - 5  # the watchdog KILLs the server here; a reply not in by then is dropped
MIN_BATTERY = 80
AGENTS = 'claude|codex|agy'


def prompt_texts():
    """id -> the user message chat would send: bench.py's text; A5's turns joined into one message."""
    by_id = {p['id']: p for p in bench.PROMPTS_A + bench.PROMPTS_B + bench.PROMPTS_C}
    return {i: ' '.join(t['user'] for t in by_id[i]['text']) if by_id[i].get('multi') else by_id[i]['text']
            for i in IDS}


def model_key():
    key = next((k for k, m in chat.MODELS.items() if m['file'] == chat.MTP_MODEL), None)
    if key is None:
        raise SystemExit(f'{MODEL} is not in chat.py\'s menu (missing on disk?)')
    return key


def command(config):
    """What chat.start_server launches for the E2B Q4_0 entry when cores 4-7 are allowed."""
    return ['taskset', '-c', '4-7', *chat.server_cmd(model_key(), draft=config == 'ON')]


def payload(text):
    return chat.chat_payload(chat.build_prompt('gemma', [{'role': 'user', 'content': text}]), 'gemma')


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------- refusals

def refusals():
    """Reasons not to start (cr.require_native runs before this)."""
    why = []
    agents = subprocess.run(['pgrep', '-fa', AGENTS], capture_output=True, text=True).stdout.strip()
    if agents:
        why.append(f'agent process running: {agents}')
    with socket.socket() as s:
        s.settimeout(2)
        if s.connect_ex(('127.0.0.1', 8080)) == 0:
            why.append('port 8080 in use')
    rc, out = cr.root('cat /sys/class/power_supply/battery/capacity /sys/class/power_supply/battery/status', 'battery')
    try:
        level, status = out.split()[:2]
        if int(level) < MIN_BATTERY:
            why.append(f'battery {level} % < {MIN_BATTERY} %')
        if status != 'Discharging':
            why.append(f'battery status {status}, not Discharging: unplug the charger')
    except ValueError:
        why.append(f'battery unreadable (su rc {rc}: {out[:100]!r})')
    return why


def check_hashes():
    got = {p: cr.sha256(p) if os.path.isfile(p) else 'missing' for p in SHA256}
    bad = [f'{p}: {got[p]} != {want}' for p, want in SHA256.items() if got[p] != want]
    if bad:
        raise SystemExit('SHA-256 mismatch: ' + '; '.join(bad))
    return got


# ---------------------------------------------------------------- server

LIVE = []  # started servers until stopped; main's finally stops them all


def stop(proc):
    """cr.Server.stop with a shorter grace: TERM the process group, KILL after TERM_WAIT_S."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=TERM_WAIT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass
    if proc in LIVE:
        LIVE.remove(proc)


def spawn(cmd, log):
    """Popen with cr.Server's signal deferral: a signal during the spawn is raised once the process is in LIVE."""
    cr._deferred.update(on=True, signum=None)
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        LIVE.append(proc)
    finally:
        cr._deferred['on'] = False
    if cr._deferred['signum'] is not None:
        stop(proc)
        raise SystemExit(128 + cr._deferred['signum'])
    return proc


def watchdog(proc, delay):
    """After delay s, KILL the server's process group: that ends any wait on it (health poll or request) at an
    absolute deadline. Not TERM: on TERM llama-server starts cleaning up but keeps a running request's connection
    open (untimed proot check, 2026-10-14). Returns (timer, fired); cancel the timer once the server is stopped."""
    fired = threading.Event()

    def fire():
        fired.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    timer = threading.Timer(delay, fire)
    timer.daemon = True
    timer.start()
    return timer, fired


def wait_healthy(proc, t0, log):
    """Seconds from spawn (t0) to /health ok, which must come before DEADLINE_S."""
    while not cr.healthy():
        if proc.poll() is not None or time.monotonic() - t0 > DEADLINE_S:
            raise RuntimeError(f'llama-server not healthy (exit {proc.poll()}); see {log.name}')
        time.sleep(0.05)
    load_s = time.monotonic() - t0
    if load_s > DEADLINE_S:
        raise RuntimeError(f'llama-server healthy only after {load_s:.0f} s')
    return load_s


def proc_status(pid):
    try:
        return {l.split(':')[0]: l.split(':', 1)[1].strip() for l in open(f'/proc/{pid}/status')
                if l.startswith(('VmHWM', 'Cpus_allowed_list'))}
    except OSError as e:
        return {'error': str(e)}


# ---------------------------------------------------------------- one block

def run_block(seq, config, todo, texts, ctx):
    """Gate, start, run todo's prompts in order until the time rules stop it, kill the server.
    Calls ctx['emit'](row) for every finished prompt; returns the block record."""
    gate = pm.thermal_gate(ctx['thermal_log'], ctx['idle'], f'block {seq} {config}', False)
    print(f'[block {seq} {config}] skin {gate["skin"]} degC, status {gate["status"]}, waited {gate["waited_s"]} s'
          f'{", WARM START" if gate["warm_start"] else ""}', flush=True)
    cr.check_cores(f'before block {seq}')
    rows, end_reason, hwm = [], 'all prompts done', None
    stop_sampling, samples, sampler, timer = threading.Event(), [], None, None
    with open(ctx['out'] / f'server_block{seq:02d}_{config}.log', 'w') as log:
        t0 = time.monotonic()
        proc = spawn(command(config), log)
        try:  # sampling covers the load too; it stops when the server stop begins
            timer, fired = watchdog(proc, DEADLINE_S - (time.monotonic() - t0))
            sampler = threading.Thread(target=cr.sampler_loop, args=(t0, ctx['thermal_log'], SimpleNamespace(proc=proc),
                                                                     stop_sampling, samples), daemon=True)
            sampler.start()
            load_s = wait_healthy(proc, t0, log)
            status_start = proc_status(proc.pid)
            for pid in todo:
                started = time.monotonic() - t0
                if started > LAST_START_S:
                    if not rows:  # load alone took the start window: the block would be redone forever
                        raise RuntimeError(f'load took {started:.0f} s; no prompt can start by {LAST_START_S} s')
                    end_reason = f'no prompt starts after {LAST_START_S} s'
                    break
                try:  # the watchdog ends the wait at DEADLINE_S; the socket timeout is a second bound
                    data = cr.post('/completion', payload(texts[pid]), timeout=DEADLINE_S - started)
                    ended = time.monotonic() - t0
                    if ended > DEADLINE_S:  # arrived just past the deadline: rejected like a cut-off reply
                        raise TimeoutError(f'reply at {ended:.1f} s')
                except (OSError, http.client.HTTPException) as e:
                    if not (fired.is_set() or isinstance(e, TimeoutError)):
                        raise
                    if not rows:  # a reply that cannot finish inside a fresh block would be redone forever
                        raise RuntimeError(f'{pid}: no reply by {DEADLINE_S} s of a block start ({e!r})') from e
                    end_reason = f'{pid} not finished at {DEADLINE_S} s ({e!r}): dropped, redone later'
                    break
                try:
                    cr.check_cores(f'after {pid}')
                except cr.CoresLost as e:
                    end_reason = f'{e}: {pid} dropped, redone later'
                    break
                tm = data.get('timings', {})
                row = {'config': config, 'block': seq, 'prompt_id': pid, 't_start': round(started, 3),
                       't_end': round(ended, 3), 'wall_s': round(ended - started, 3),
                       'ttft_s': None,  # not measurable: chat's payload is not streamed (see prompt_s)
                       'prompt_s': tm['prompt_ms'] / 1000 if tm.get('prompt_ms') is not None else None,
                       'prompt_n': tm.get('prompt_n'), 'predicted_n': tm.get('predicted_n'),
                       'predicted_ms': tm.get('predicted_ms'), 'gen_tok_s': tm.get('predicted_per_second'),
                       'draft_n': tm.get('draft_n'), 'draft_n_accepted': tm.get('draft_n_accepted'),
                       'stop_type': data.get('stop_type'), 'truncated': data.get('truncated'),
                       'reply_chars': len(data.get('content', '').strip()), 'reply': data.get('content'),
                       'timings': tm}
                rows.append(row)
                ctx['emit'](row)
                hwm = proc_status(proc.pid).get('VmHWM', hwm)  # kept: a server killed by the watchdog has none
            requests_s = time.monotonic() - t0
            status_end = proc_status(proc.pid)
            if 'VmHWM' not in status_end and hwm:
                status_end.update(VmHWM=hwm, VmHWM_source='last completed prompt')
        finally:
            if timer:
                timer.cancel()
            stop_sampling.set()
            stop(proc)
            server_s = time.monotonic() - t0  # spawn to server exit
            if sampler:
                sampler.join(timeout=60)
    dump_end = cr.read_dump()  # after the stop: it takes seconds and must not extend the server's block
    return block_record(seq, config, gate, load_s, requests_s, server_s, end_reason, status_start, status_end,
                        dump_end, samples, rows)


def block_record(seq, config, gate, load_s, requests_s, server_s, end_reason, status_start, status_end, dump_end,
                 samples, rows):
    watts = [s['battery_w'] for s in samples if s.get('battery_w') is not None]
    in_req = [s['battery_w'] for s in samples if s.get('battery_w') is not None
              and any(r['t_start'] <= s['t'] <= r['t_end'] for r in rows)]
    pss = [s['pss_kb']['llama_server'] for s in samples if 'llama_server' in s.get('pss_kb', {})]
    mem = [s['mem_available_mib'] for s in samples if 'mem_available_mib' in s]
    tokens = sum(r['predicted_n'] or 0 for r in rows)
    req_s = sum(r['wall_s'] for r in rows)
    return {'block': seq, 'config': config, 'prompts': [r['prompt_id'] for r in rows], 'end_reason': end_reason,
            'load_s': round(load_s, 3), 'requests_s': round(requests_s, 3), 'server_s': round(server_s, 3),
            'gate': gate,
            'end': {'skin': dump_end.get('skin'), 'status': dump_end.get('status')},
            'server_start': status_start, 'server_end': status_end,
            'vm_hwm_kb': int(status_end['VmHWM'].split()[0]) if 'VmHWM' in status_end else None,
            'peak_pss_kb': max(pss) if pss else None, 'min_mem_available_mib': min(mem) if mem else None,
            'mean_w': statistics.mean(watts) if watts else None,
            'mean_w_requests': statistics.mean(in_req) if in_req else None,
            'energy_j_per_token': statistics.mean(in_req) * req_s / tokens if in_req and tokens else None,
            'samples': samples}


# ---------------------------------------------------------------- order, resume, report

def next_block(seq, remaining):
    """(seq, config) of the next block from seq on: ORDER, skipping a config with nothing left; None when done."""
    if not any(remaining.values()):
        return None
    while not remaining[ORDER[seq % 4]]:
        seq += 1
    return seq, ORDER[seq % 4]


def read_jsonl(p):
    """Rows of a JSONL file; a torn last line (a write cut off by a kill) is dropped (conv_speed_mtp.read_jsonl)."""
    lines = p.read_text().splitlines() if p.exists() else []
    rows = [json.loads(l) for l in lines[:-1]]
    try:
        rows += [json.loads(l) for l in lines[-1:]]
    except json.JSONDecodeError:
        print(f'{p}: dropping torn last line', flush=True)
    return rows


def state(out, ids):
    """Finished prompts, block records and the next block number, from the run folder."""
    prompts, blocks = read_jsonl(out / 'prompts.jsonl'), read_jsonl(out / 'blocks.jsonl')
    remaining = {c: [i for i in ids if (c, i) not in {(r['config'], r['prompt_id']) for r in prompts}]
                 for c in ('OFF', 'ON')}
    seq = 1 + max([r['block'] for r in prompts + blocks], default=-1)
    return prompts, blocks, remaining, seq


def tok_s(rows):
    """Token-weighted generation tok/s over replies with >= 10 tokens (conv_speed.summary)."""
    gen = [r for r in rows if (r['predicted_n'] or 0) >= 10]
    ms = sum(r['predicted_ms'] for r in gen)
    return sum(r['predicted_n'] for r in gen) / ms * 1000 if ms else None


def acceptance(rows):
    drafted = sum(r['draft_n'] or 0 for r in rows)
    return sum(r['draft_n_accepted'] or 0 for r in rows) / drafted if drafted else None


def report(prompts, blocks, ids):
    f = lambda v, spec: format(v, spec) if v is not None else 'n/a'
    med = lambda xs: statistics.median(xs) if xs else None
    L = ['CHAT MTP REPORT  (chat.py E2B Q4_0 command and payload: ctx 4096, temperature 0.7, n_predict 512, cores 4-7)',
         'prompt s = server prompt_ms, median (TTFT not measurable: payload not streamed); gen tok/s token-weighted',
         '(>= 10 tokens);',
         'accept = accepted / drafted; W = mean battery power over all block samples; J/tok = mean W during requests',
         '  x request time / generated tokens (whole phone); RAM: max VmHWM / max PSS, min MemAvailable', '']
    L.append(f"{'config':<7}{'prompts':>8}{'load s':>8}{'prompt s':>9}{'gen t/s':>9}{'accept':>8}{'reply ch':>9}"
             f"{'HWM MiB':>9}{'PSS MiB':>9}{'avail MiB':>10}{'W':>6}{'J/tok':>7}{'blocks':>7}{'warm':>5}")
    for c in ('OFF', 'ON'):
        rows = [r for r in prompts if r['config'] == c]
        bs = [b for b in blocks if b['config'] == c]
        top = lambda k: max((b[k] for b in bs if b[k] is not None), default=None)
        mean = lambda k: statistics.mean([b[k] for b in bs if b[k] is not None]) if any(b[k] is not None for b in bs) else None
        hwm, pss = top('vm_hwm_kb'), top('peak_pss_kb')
        L.append(f"{c:<7}{f'{len(rows)}/{len(ids)}':>8}{f(med([b['load_s'] for b in bs]), '.1f'):>8}"
                 f"{f(med([r['prompt_s'] for r in rows if r['prompt_s'] is not None]), '.2f'):>9}{f(tok_s(rows), '.2f'):>9}"
                 f"{f(acceptance(rows), '.1%'):>8}{f(med([r['reply_chars'] for r in rows]), '.0f'):>9}"
                 f"{f(hwm and hwm / 1024, '.0f'):>9}{f(pss and pss / 1024, '.0f'):>9}"
                 f"{f(min((b['min_mem_available_mib'] for b in bs if b['min_mem_available_mib'] is not None), default=None), 'd'):>10}"
                 f"{f(mean('mean_w'), '.2f'):>6}{f(mean('energy_j_per_token'), '.2f'):>7}{len(bs):>7}"
                 f"{sum(b['gate']['warm_start'] for b in bs):>5}")
    L += ['', 'blocks: seq config prompts | load s, last reply s, server exit s (from spawn) | skin degC/status start -> end'
          ' | server Cpus_allowed_list | end']
    for b in blocks:
        L.append(f"  {b['block']:>2} {b['config']:<3} {','.join(b['prompts']) or '-'} | {b['load_s']:.1f}, {b['requests_s']:.0f}, {b['server_s']:.0f}"
                 f" | {b['gate']['skin']}/{b['gate']['status']} -> {b['end']['skin']}/{b['end']['status']}"
                 f"{' WARM START' if b['gate']['warm_start'] else ''} | {b['server_start'].get('Cpus_allowed_list')}"
                 f" -> {b['server_end'].get('Cpus_allowed_list')} | {b['end_reason']}")
    orphans = sorted({r['block'] for r in prompts} - {b['block'] for b in blocks})
    if orphans:
        L.append(f'  prompts without a block record (interrupted blocks, no block metrics): blocks {orphans}')
    return '\n'.join(L)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--thermal-log', default=str(THERMAL_LOG))
    a = ap.parse_args(argv)
    cr.require_native()  # before any root command or server
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cr.exit_on_signal)  # Ctrl-C too: SystemExit, so the finally below stops the server
    try:
        why = refusals()
        if why:
            raise SystemExit('not started: ' + '; '.join(why))
        ids = IDS[:1] if a.smoke else IDS
        texts = prompt_texts()
        manifest = {'smoke': a.smoke, 'ids': ids, 'commands': {c: command(c) for c in ('OFF', 'ON')},
                    'payload_example': payload('<text>'),
                    'prompt_text_sha256': {i: sha256_text(texts[i]) for i in ids}}
        if a.smoke:
            pm.GATE_MAX_S = SMOKE_GATE_S
        if a.resume:
            out = a.resume
            run = json.loads((out / 'run.json').read_text())
            if any(run[k] != v for k, v in manifest.items()):
                raise SystemExit('--resume: smoke flag, prompts, commands or payload differ from run.json')
        else:
            out = OUT_ROOT / f'chat_mtp_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}{"_smoke" if a.smoke else ""}'
            out.mkdir(parents=True)
        print(f'checking model and drafter SHA-256 (about 30 s); run folder {out}', flush=True)
        hashes = check_hashes()
        cr.wait_cores('start')
        idle = run['idle'] if a.resume else {**cr.read_thermal(a.thermal_log), 'skin': cr.read_dump()['skin']}
        if idle['skin'] is None:
            raise SystemExit('no VIRTUAL-SKIN reading at idle')
        meta = {**manifest, 'started': cr.utc(), 'idle': idle, 'model_sha256': hashes,
                'bench_sha256': bench.SOURCE_SHA256, 'prompt_texts': {i: texts[i] for i in ids},
                'code_sha256': {str(p): cr.sha256(p) for p in (Path(__file__), Path(chat.__file__), Path(sm.__file__),
                                                               Path(cr.__file__), Path(pm.__file__))},
                'llama_server': chat.SERVER, 'libggml_cpu_sha256': cr.sha256(Path(chat.SERVER).parent / 'libggml-cpu.so'),
                'block_s': BLOCK_S, 'deadline_s': DEADLINE_S, 'term_wait_s': TERM_WAIT_S, 'last_start_s': LAST_START_S,
                'gate_max_s': pm.GATE_MAX_S}
        (out / (f'resume_{time.time_ns()}.json' if a.resume else 'run.json')).write_text(json.dumps(meta, indent=1) + '\n')
        prompts, blocks, remaining, seq = state(out, ids)
        if a.resume:
            # conv_speed_mtp.py: rewrite both logs with the parsed rows (no torn line to append onto); the old files
            # are kept as *.before_resume_<ns>, the new ones replace them atomically
            for name, rows in (('prompts.jsonl', prompts), ('blocks.jsonl', blocks)):
                if (out / name).exists():
                    shutil.copy2(out / name, out / f'{name}.before_resume_{time.time_ns()}')
                (out / f'{name}.tmp').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                os.replace(out / f'{name}.tmp', out / name)
            print(f'resume: {len(prompts)} prompts done, next block {seq}', flush=True)
        with open(out / 'prompts.jsonl', 'a') as plog, open(out / 'blocks.jsonl', 'a') as blog:
            def emit(row):
                plog.write(json.dumps(row) + '\n')
                plog.flush()
                prompts.append(row)
                remaining[row['config']].remove(row['prompt_id'])
            ctx = {'out': out, 'thermal_log': a.thermal_log, 'idle': idle, 'emit': emit}
            while (nxt := next_block(seq, remaining)) is not None:
                seq, config = nxt
                while True:
                    cr.wait_cores(f'block {seq}')
                    try:  # raised only before the server starts; later losses end the block inside run_block
                        b = run_block(seq, config, list(remaining[config]), texts, ctx)
                        break
                    except cr.CoresLost as e:
                        print(f'[block {seq} {config}] {e}; redoing the gate', flush=True)
                blog.write(json.dumps(b) + '\n')
                blog.flush()
                blocks.append(b)
                print(f'[block {seq} {config}] {",".join(b["prompts"]) or "no prompt"}, server {b["server_s"]:.0f} s; '
                      f'{b["end_reason"]}', flush=True)
                seq += 1
        text = report(prompts, blocks, ids)
        (out / 'report.txt').write_text(text + '\n')
        print(text, flush=True)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        for proc in list(LIVE):
            stop(proc)


if __name__ == '__main__':
    main()
