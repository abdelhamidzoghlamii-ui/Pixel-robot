#!/data/data/com.termux/files/usr/bin/python
"""MIND-B0b: keeping picker and chat contexts warm on the robot's llama-server: slot save/restore (L1) vs two slots (L2),
against the B0 reference (L0). Research only: no motors, camera, YOLO, M2, main.py or run_mission.py; no MTP, no
projector. Counters, probabilities, slot file sizes and RAM snapshots; timings are NOT MEASUREMENTS and RAM is
INDICATIVE (an agent may be resident).

Reuses B0 unchanged (b0_cache/b0.py, b0_set.json, SHA-256 checked): same prompts, scoring, picker tail, decoding.
Launches (one per invocation, setup_q4 flags = mem1.command(ctx) plus only the named change):
  L0  exactly B0 (one slot, ctx 2048): R1..R8, then FRESH R1/R3/R5/R7/R8.
  L1  L0 + --slot-save-path <run>/slots: R1..R5, SAVE slot 0, R6 (separate chat), RESTORE slot 0, R7, R8, FRESH.
  L2  --parallel 2 (replaces 1), ctx 4096: pickers id_slot 0, chats id_slot 1; R1..R8, R9 = chat on slot 1
      continuing R6 (speech, R6 reply, CHAT2), R10 = R8's ids again on slot 0, FRESH on slot 0.
Per request as B0 plus the slot asked for and the slot the server reports; LCP against that slot's previous content
(prompt + generated ids; after RESTORE the saved content truncated to n_restored). RAM (llama-server Pss, VmRSS,
VmHWM; MemAvailable) after load and at the end.
Criteria (criteria()): K0 L0 rows identical to the B0 run of record in all 13 rows (K0_FIELDS); K1 L1 restore ok and
R7 cache_n >= min(LCP with the restored content, n - 1); K2 L2 R7 cache_n >= min(LCP with slot 0's previous content,
n - 1); K3 L2 R9 cache_n >= R6 prompt tokens; K4 cached vs fresh: B0's C4 (same top 5/5, max |dp| <= 0.02); K5 one
server PID, all planned rows, 0 request errors, no exception, the reported slot = the requested one, exit code 0
without SIGKILL. Exceptions keep the rows collected so far and fail the criteria they affect (B0's known
interrupt/cleanup limitation is carried over).

Usage: b0b.py L0|L1|L2   (Termux python; from proot Debian or native Termux)
Writes ~/robot_runs/mind_b0b/<UTC>_<L>/: run.json, requests.jsonl, server.log, [slots/], report.txt, artifacts.sha256.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'b0_cache'))
import b0  # noqa: E402  B0's prompts, checks, scoring, criteria; imports mem1

mem1, cr, repro, sha = b0.mem1, b0.cr, b0.repro, b0.sha
SET_SHA256 = 'dddb288ace1642b1e93a6e268cb68ff1a028121b6853dfa837ac7849cfaa187b'
B0_RECORD = HERE.parent / 'b0_cache' / 'runs' / '20261102T110006Z' / 'requests.jsonl'
OUT_ROOT = cr.HOME / 'robot_runs' / 'mind_b0b'
CHAT2 = 'And what will you do next?'
SLOT_FILE = 'picker.bin'
LAUNCHES = {'L0': {'ctx': 2048, 'parallel': 1, 'save': False}, 'L1': {'ctx': 2048, 'parallel': 1, 'save': True},
            'L2': {'ctx': 4096, 'parallel': 2, 'save': False}}
HELP_FLAGS = ('--slot-save-path', '--parallel', '--kv-unified', '--slot-prompt-similarity', '--no-slots', '--slots')
K0_FIELDS = ('id', 'prompt_tokens', 'prompt_ids_sha256', 'lcp_prev_slot', 'prompt_n', 'cache_n', 'generated_ids',
             'content', 'probs_raw', 'probs', 'top')
SLOT_CTX = 2048  # every launch aims at 2048 per slot


def command(launch, slot_dir=None):
    L = LAUNCHES[launch]
    cmd = mem1.command(L['ctx'])
    cmd[cmd.index('--parallel') + 1] = str(L['parallel'])
    return cmd + (['--slot-save-path', str(slot_dir)] if L['save'] else [])


def plan(launch):
    """[(step, rid, kind, k)]: 'req' a B0 request, 'save'/'restore' slot 0, 'chat2' R9, 'again' R10 (R8's ids)."""
    seq = [('req', *x) for x in b0.SEQUENCE]
    if launch == 'L1':
        seq = seq[:5] + [('save', None, None, None), seq[5], ('restore', None, None, None)] + seq[6:]
    if launch == 'L2':
        seq += [('chat2', 'R9', 'chat2', None), ('again', 'R10', 'picker', 4)]
    return seq + [('fresh', r, None, None) for r in b0.FRESH]


def n_rows(launch):
    return sum(step not in ('save', 'restore') for step, *_ in plan(launch))


def slot_of(launch, kind):
    return 1 if launch == 'L2' and kind != 'picker' else 0


def run_sequence(post, s, launch, pid, emit):
    """B0's checks, then the launch's plan; returns (rows, info). An exception is recorded, the rows are kept."""
    rows, info = [], {'problems': [], 'slot_actions': []}
    try:
        sequence(post, s, launch, pid, emit, rows, info)
    except Exception as e:
        info['exception'] = f'{type(e).__name__}: {e}'
        info['stopped'] = f'exception after {len(rows)} requests: {info["exception"]}'
    return rows, info


def sequence(post, s, launch, pid, emit, rows, info):
    p0, ids0, bad = b0.render(post, b0.messages(s, 'picker', 0))
    info['name_ids'], why = b0.name_ids(post, p0, ids0, s['names'])
    info['problems'] += why + ([bad] if bad else [])
    info['t1_prefix_tokens'] = len(b0.tokenize(post, '<|turn>system\n' + b0.transcript(s, 1)))
    info['slots_n_ctx'] = [x.get('n_ctx') for x in post('/slots', None)]
    if info['slots_n_ctx'] != [SLOT_CTX] * LAUNCHES[launch]['parallel']:
        info['problems'].append(f'per-slot n_ctx {info["slots_n_ctx"]} != {SLOT_CTX} x {LAUNCHES[launch]["parallel"]}')
    if info['problems']:
        info['stopped'] = 'STOP before any inference: ' + '; '.join(info['problems'])
        return
    content, reply, r6_reply, sent, saved = {}, '', '', {}, None
    for step, rid, kind, k in plan(launch):
        if step == 'save':
            saved = list(content[0])
            info['slot_actions'].append(slot_action(post, 'save', info))
            continue
        if step == 'restore':
            a = slot_action(post, 'restore', info)
            info['slot_actions'].append(a)
            content[0] = saved[:a['response'].get('n_restored', 0)]
            continue
        cache, row_id = step != 'fresh', rid if step != 'fresh' else rid + '_FRESH'
        if step in ('req', 'chat2'):
            msgs = b0.messages(s, kind, k, reply) if step == 'req' else [
                {'role': 'system', 'content': s['companion_sys']}, {'role': 'user', 'content': s['speech']},
                {'role': 'assistant', 'content': r6_reply}, {'role': 'user', 'content': CHAT2}]
            _, ids, bad = b0.render(post, msgs)
            if bad:
                info['stopped'] = f'{rid}: {bad}'
                return
            sent[rid] = (kind, k, ids)
        else:
            kind, k, ids = sent['R8' if step == 'again' else rid]
            sent.setdefault(rid, (kind, k, ids))
        slot = slot_of(launch, kind)
        body = {'prompt': ids, **b0.DECODE, **(b0.PICK if kind == 'picker' else b0.CHAT), 'cache_prompt': cache}
        if launch == 'L2':
            body['id_slot'] = slot
        row = {'id': row_id, 'kind': kind, 'k': k, 'cache_prompt': cache, 'slot': slot,
               'prompt_tokens': len(ids), 'prompt_ids_sha256': sha(json.dumps(ids)),
               'lcp_prev_slot': mem1.lcp(ids, content[slot]) if slot in content else None,
               'pid': pid, 'pid_start': b0.pid_start(pid)}
        t0 = time.perf_counter()
        try:
            res = post('/completion', body)
        except Exception as e:  # fail closed: recorded, the sequence ends
            row.update(error=f'{type(e).__name__}: {e}', wall_s=round(time.perf_counter() - t0, 3))
            rows.append(row)
            emit(row)
            info['stopped'] = f'{row_id} failed: {e!r}'
            return
        tm, gen = res.get('timings', {}), res.get('tokens') or []
        row.update(wall_s=round(time.perf_counter() - t0, 3), slot_reported=res.get('id_slot'),
                   prompt_n=tm.get('prompt_n'), cache_n=tm.get('cache_n'), prompt_ms=tm.get('prompt_ms'),
                   generated_ids=gen, content=res.get('content'), tokens_cached=res.get('tokens_cached'), timings=tm)
        if kind == 'picker':
            row['probs_raw'], row['probs'] = b0.probs(res, info['name_ids'])
            row['top'] = max(row['probs'], key=row['probs'].get) if row['probs'] else None
        if row_id == 'R4':
            reply = info['reply'] = res.get('content') or ''
        if row_id == 'R6':
            r6_reply = info['r6_reply'] = res.get('content') or ''
        if row_id == 'R9':
            info['r9_reply'] = res.get('content') or ''
        content[slot] = ids + gen
        rows.append(row)
        emit(row)


def slot_action(post, action, info):
    t0 = time.perf_counter()
    res = post(f'/slots/0?action={action}', {'filename': SLOT_FILE})
    return {'action': action, 'wall_s': round(time.perf_counter() - t0, 3), 'response': res}


def criteria(launch, rows, info, exit_code, killed, exception=None, reference=None):
    """[(name, PASS|FAIL, numbers)] for this launch, from the recorded rows alone."""
    r = {x['id']: x for x in rows}
    out = []

    def got(rid, key):
        return r.get(rid, {}).get(key)

    def reuse(rid):  # lcp_prev_slot is already against the right content (restored, or the request's own slot)
        n, lcp, cn = got(rid, 'prompt_tokens'), got(rid, 'lcp_prev_slot'), got(rid, 'cache_n')
        need = min(lcp, n - 1) if None not in (n, lcp) else None
        return cn, need, None not in (cn, need) and cn >= need
    if launch == 'L0':
        ref = {x['id']: x for x in reference or []}
        diff = [(i, f) for i in [x['id'] for x in reference or []] for f in K0_FIELDS
                if r.get(i, {}).get(f) != ref[i].get(f)]
        ok = len(ref) == 13 and len(rows) == 13 and not diff
        out.append(('K0', 'PASS' if ok else 'FAIL', f'{len(rows)} rows vs {len(ref)} B0 rows; differing (id, field): '
                    f'{diff[:8] or "none"}'))
    if launch == 'L1':
        acts = info.get('slot_actions', [])
        ok_act = [a['action'] for a in acts] == ['save', 'restore'] and \
            acts[1]['response'].get('n_restored', 0) > 0
        cn, need, ok = reuse('R7')
        out.append(('K1', 'PASS' if ok and ok_act else 'FAIL', f'R7 cache_n {cn} vs min(LCP with restored, n-1) {need}; '
                    f'save/restore ok {ok_act}'))
    if launch == 'L2':
        cn, need, ok = reuse('R7')
        out.append(('K2', 'PASS' if ok and got('R7', 'slot') == 0 else 'FAIL',
                    f'R7 cache_n {cn} vs min(LCP with slot 0, n-1) {need}'))
        cn9, n6 = got('R9', 'cache_n'), got('R6', 'prompt_tokens')
        out.append(('K3', 'PASS' if None not in (cn9, n6) and cn9 >= n6 else 'FAIL',
                    f'R9 cache_n {cn9} vs R6 prompt tokens {n6}'))
    c4 = b0.criteria(rows, info, exit_code, killed)[3]
    out.append(('K4', c4[1], c4[2]))
    pids = {(x['pid'], x['pid_start']) for x in rows}
    errors = sum('error' in x for x in rows)
    exc = exception or info.get('exception')
    misrouted = [x['id'] for x in rows if 'error' not in x and launch == 'L2' and x.get('slot_reported') != x['slot']]
    ok5 = len(pids) == 1 and errors == 0 and not exc and not misrouted and len(rows) == n_rows(launch) and \
        exit_code == 0 and not killed
    out.append(('K5', 'PASS' if ok5 else 'FAIL', f'PIDs {sorted(pids)}; {len(rows)}/{n_rows(launch)} requests, '
                f'{errors} errors; misrouted {misrouted}; exit code {exit_code}, SIGKILL {killed}; exception {exc}'))
    return out


def ram(pid):
    """INDICATIVE: llama-server Pss (smaps_rollup), VmRSS, VmHWM (mem1.proc_mem) and MemAvailable, in kB."""
    m = mem1.proc_mem(pid)
    try:
        m['Pss'] = next(l.split(':', 1)[1].strip() for l in open(f'/proc/{pid}/smaps_rollup') if l.startswith('Pss:'))
    except (OSError, StopIteration) as e:
        m['Pss'] = f'error: {e}'
    m['MemAvailable'] = next(l.split(':', 1)[1].strip() for l in open('/proc/meminfo') if l.startswith('MemAvailable'))
    return m


def report(run, rows, info, crit):
    L = [f'MIND-B0b {run["launch"]} {run["utc"]}  code {run["code_sha256"][:12]}  b0 {run["b0_sha256"][:12]}  '
         f'set {run["set_sha256"][:12]}  base {run["git_head"]}', f'server: {" ".join(run["command"])}',
         f'--help flags: {run["help_flags"]}', f'per-slot n_ctx {info.get("slots_n_ctx")}; T(1) without tail '
         f'{info.get("t1_prefix_tokens")}; name ids {info.get("name_ids")}',
         f'R4 reply: {info.get("reply")!r}', f'R6 reply: {info.get("r6_reply")!r}']
    if 'r9_reply' in info:
        L.append(f'R9 reply: {info["r9_reply"]!r}')
    for a in info.get('slot_actions', []):
        L.append(f'slot {a["action"]}: wall {a["wall_s"]} s* response {a["response"]}')
    if run.get('slot_file_bytes') is not None:
        L.append(f'slot file {SLOT_FILE}: {run["slot_file_bytes"]} bytes')
    if info.get('stopped'):
        L.append(f'STOPPED: {info["stopped"]}')
    if run.get('exception'):
        L.append(f'EXCEPTION (runner): {run["exception"]}')
    L += ['', 'id        kind      cache slot n    LCP  prompt_n cache_n  top      p(top)  prompt_ms* wall_s*']
    for x in rows:
        p = x.get('probs') or {}
        L.append(f'{x["id"]:<9} {x["kind"]:<9} {str(x["cache_prompt"])[0]:<5} {x["slot"]}/{x.get("slot_reported")!s:<3}'
                 f'{x["prompt_tokens"]:<4} {str(x["lcp_prev_slot"]):<4} {str(x.get("prompt_n")):<8} '
                 f'{str(x.get("cache_n")):<8} {str(x.get("top")):<8} {p.get(x.get("top"), float("nan")):.4f}  '
                 f'{str(x.get("prompt_ms")):<10.10} {x.get("wall_s")}' + (f'  ERROR {x["error"]}' if 'error' in x
                                                                           else ''))
    L += ['* timings NOT MEASUREMENTS (agent resident; no thermal or core control); slot = asked/reported', '',
          'probabilities (renormalised over the five; raw mass in brackets):']
    for x in rows:
        if x.get('probs_raw'):
            mass = sum(v for v in x['probs_raw'].values() if v is not None)
            L.append(f'  {x["id"]:<9} ' + ' '.join(f'{n}={v:.4f}' for n, v in (x['probs'] or {}).items()) +
                     f'  [{mass:.4f}]')
    L += ['', 'RAM (INDICATIVE, kB):'] + [f'  {k}: {v}' for k, v in run.get('ram', {}).items()]
    L += ['', 'criteria:'] + [f'  {c}  {v:<4}  {t}' for c, v, t in crit]
    L.append(f'server exit code {run.get("exit_code")}; SIGKILL needed {run.get("killed")}')
    return '\n'.join(L) + '\n'


def finish(out, run, rows, info, reference):
    run['criteria'] = criteria(run['launch'], rows, info, run.get('exit_code'), run.get('killed'),
                               run.get('exception'), reference)
    (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
    text = report(run, rows, info, run['criteria'])
    (out / 'report.txt').write_text(text)
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'artifacts.sha256')
    (out / 'artifacts.sha256').write_text(''.join(f'{sha(p.read_bytes())}  {p.relative_to(out)}\n' for p in files))
    return text


def main():
    launch = sys.argv[1] if len(sys.argv) == 2 and sys.argv[1] in LAUNCHES else None
    if launch is None:
        raise SystemExit(f'usage: b0b.py {"|".join(LAUNCHES)}')
    s_bytes = b0.SET.read_bytes()
    if sha(s_bytes) != SET_SHA256:
        raise SystemExit(f'refused: {b0.SET} SHA-256 {sha(s_bytes)[:12]} is not {SET_SHA256[:12]}')
    s = json.loads(s_bytes)
    if bad := mem1.pinned_problems()[0]:
        raise SystemExit(f'refused: {bad}')
    if not repro.port_free():
        raise SystemExit('refused: port 8080 in use (never stops a server it did not start)')
    reference = [json.loads(l) for l in B0_RECORD.read_text().splitlines()]
    utc = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = OUT_ROOT / f'{utc}_{launch}'
    out.mkdir(parents=True)
    slot_dir = out / 'slots'
    if LAUNCHES[launch]['save']:
        slot_dir.mkdir()  # llama-server refuses a --slot-save-path that is not a directory
    help_text = subprocess.run([mem1.SERVER, '--help'], capture_output=True, text=True, timeout=60).stdout
    run = {'launch': launch, 'utc': utc, 'set_sha256': sha(s_bytes), 'code_sha256': sha(Path(__file__).read_bytes()),
           'b0_sha256': sha(Path(b0.__file__).read_bytes()), 'mem1_sha256': sha(Path(mem1.__file__).read_bytes()),
           'b0_record_sha256': sha(B0_RECORD.read_bytes()), 'pinned_sha256': mem1.PINNED,
           'git_head': subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True,
                                      text=True).stdout.strip(),
           'command': command(launch, slot_dir), 'help_flags': {f: f in help_text.split() or f + ',' in help_text
                                                                for f in HELP_FLAGS},
           'timings': 'NOT MEASUREMENTS', 'ram': {}}
    (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')

    def post(path, body):
        if body is None:  # GET
            with urllib.request.urlopen(f'http://127.0.0.1:{cr.PORT}{path}', timeout=30) as r:
                return json.loads(r.read())
        return cr.post(path, body, timeout=b0.REQUEST_TIMEOUT_S)

    with open(out / 'requests.jsonl', 'x') as rq, open(out / 'server.log', 'x') as log:
        def emit(row):
            rq.write(json.dumps(row) + '\n')
            rq.flush()
        killed, rows, info, proc, exc = False, [], {}, None, None
        try:
            proc = subprocess.Popen(run['command'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            t0 = time.monotonic()
            while not cr.healthy():
                if proc.poll() is not None or time.monotonic() - t0 > b0.LOAD_TIMEOUT_S:
                    raise RuntimeError(f'llama-server not healthy (exit {proc.poll()})')
                time.sleep(0.1)
            run['server_pid'], run['server_comm'] = proc.pid, Path(f'/proc/{proc.pid}/comm').read_text().strip()
            run['ram']['after_load'] = ram(proc.pid)
            rows, info = run_sequence(post, s, launch, proc.pid, emit)
            run['ram']['end'] = ram(proc.pid)
        except BaseException as e:  # recorded; the report and manifest are still written, then a stop re-raises
            exc = e
            run['exception'] = f'{type(e).__name__}: {e}'
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=b0.TERM_WAIT_S)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    killed = True
            run.update(exit_code=proc.returncode if proc else None, killed=killed, info=info)
    if (slot_dir / SLOT_FILE).is_file():
        run['slot_file_bytes'] = (slot_dir / SLOT_FILE).stat().st_size
    print(finish(out, run, rows, info, reference) + f'\n{out}')
    if exc is not None and not isinstance(exc, Exception):
        raise exc


if __name__ == '__main__':
    main()
