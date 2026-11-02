#!/data/data/com.termux/files/usr/bin/python
"""MIND-B3: does a relevant memory note make the B2 picker (V3: rules + examples + one reason sentence) choose the
right action, and do irrelevant, outdated or wrong notes make it worse than no memory? Research only: no motors,
camera, YOLO, M2, main.py or run_mission.py; no MTP, no projector. Accuracy is deterministic (temperature 0);
timings are recorded but are NOT MEASUREMENTS.

V3 is imported from b2.py unchanged: system text and examples (b2.SYSTEM['V3']), tail (b2.tail), two-step scoring
(b2.score: reason with n_predict 48 stopping at the first newline, then '\\nAction:' and the next token over
' <name>'), distinct-first-token check (b2.first_ids), block loop (b2.run_blocks, PAUSE_S 60). Rendering is B2's
except the memory line: 'Memory: none' when empty, otherwise 'Memory:' and one line '- <date>: <note>' per note,
oldest first. All five actions allowed in every item, catalogue order; FWD/REV as B2.

SET = 20 bases x 6 conditions (build_set()), written OUTSIDE the repo by `b3.py build` before any Gemma call; its
path and SHA-256 are pinned below. The runner refuses a set path inside the repo (b2.inside_repo, checked before
reading), a different SHA-256, or content that differs from build_set().
Conditions per base i (expected label): NONE no memory (without); RELEVANT 'Oct 20: note i' (with); MIXED the four
pool notes + 'Oct 20: note i', by date (with); IRRELEVANT the four pool notes (without); STALE 'Oct 20: note i' +
'Nov 1: The Oct 20 note is no longer true.' (without); MISLEADING 'Oct 20: note j', j = i+10 for i <= 10, i-10
otherwise (without). 'also ok' only for base 20 under RELEVANT and MIXED.

One server launch (mem1.command(2048) = setup_q4 flags, pins checked); every prompt rendered and checked before any
inference (/apply-template equality, distinct first tokens, prompt + 64 <= 2048, else STOP); 12 blocks NONE,
RELEVANT, MIXED, IRRELEVANT, STALE, MISLEADING, FWD then REV each, PAUSE_S between blocks. Metrics and PASS:
metrics(), verdict(). VALID only with all 240 rows without error, one server PID, exit 0 without SIGKILL, no
exception or stop, and the set hash unchanged at the end.

Usage: b3.py build | b3.py show | b3.py [--set PATH]   (Termux python; from proot Debian or native Termux)
Run writes ~/robot_runs/mind_b3/<UTC>/: run.json, requests.jsonl, server.log, report.txt, artifacts.sha256.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'b2_rules'))
import b2  # noqa: E402  V3 texts, tail, score, first_ids, run_blocks, inside_repo; imports b1, b0, mem1

b1, b0, mem1, cr, sha = b2.b1, b2.b0, b2.mem1, b2.cr, b2.sha
SET_ROOT = Path(cr.HOME) / 'mind_b3_set'
SET = SET_ROOT / '20261102T193300Z' / 'b3_set.json'  # written by `b3.py build` 2026-11-02T19:33:00Z
SET_SHA256 = '156542981e89e7afd0a2b833faf989449b9c11cc750b73fddf16f3a0cf4217ab'
OUT_ROOT = Path(cr.HOME) / 'robot_runs' / 'mind_b3'
CTX, CTX_MARGIN = b2.CTX, b2.CTX_MARGIN
NAMES, ORDERS = b2.NAMES, ('FWD', 'REV')
CONDITIONS = ['NONE', 'RELEVANT', 'MIXED', 'IRRELEVANT', 'STALE', 'MISLEADING']
SHOW = ('MIXED', 8)
SAMPLE_IDS = [1, 5, 9, 13, 17]  # FWD reasons shown per condition
# reasons citing a memory note or Rule 5 (counted under NONE and IRRELEVANT; every match is listed for checking)
CITES_MEMORY = re.compile(r'\bmemor(y|ies)\b|\bnotes?\b|\bremember|\brule\s*5\b', re.I)
PASS = {'RELEVANT': 0.80, 'MIXED': 0.75}
HARM = ('IRRELEVANT', 'STALE', 'MISLEADING')
HARM_MAX = 0.10  # acceptable >= NONE acceptable - 0.10, same order

MISSIONS = {'M1': 'Explore and map the flat.', 'M2': 'Find people and keep them company.',
            'M3': 'Map the flat quietly and do not disturb anyone.', 'M4': 'Help the owner find his keys.'}
# id: (mission, current, what just happened, note, without, with, also ok with)
BASES = {
    1: ('M2', 'explore', 'A person is at the desk in the study, far away.',
        'Whoever works in the study must not be disturbed.', 'approach', 'explore', []),
    2: ('M2', 'explore', 'A person is on the balcony, far away.',
        'The person on the balcony is usually on the phone; leave them alone.', 'approach', 'explore', []),
    3: ('M1', 'explore', 'The front door opened; a person appeared close.',
        'The owner likes a greeting when he comes home.', 'explore', 'greet', []),
    4: ('M4', 'explore', 'You are in the hallway.', 'The keys are usually in the blue bowl in the hallway.',
        'explore', 'look', []),
    5: ('M4', 'explore', 'You entered a new room: the kitchen.',
        'The kitchen was already searched today; no keys there.', 'look', 'explore', []),
    6: ('M2', 'explore', 'You are in the nursery; a person appeared close.',
        'The baby sleeps in the nursery; keep quiet there.', 'greet', 'explore', []),
    7: ('M2', 'explore', 'It is Friday. A person with a vacuum cleaner is far away.',
        'The cleaner comes on Fridays and wants no company.', 'approach', 'explore', []),
    8: ('M3', 'explore', "A person is at Lisa's desk, at medium distance, silent.",
        'Lisa asked the robot to always come and say hi to her.', 'explore', 'approach', []),
    9: ('M1', 'explore', 'A visitor walked past you, close, at the front door.',
        'The owner wants every visitor greeted at the front door.', 'explore', 'greet', []),
    10: ('M2', 'explore', 'A person is lying in the dark bedroom, far away.',
         'When the bedroom light is off, someone is sleeping; stay out.', 'approach', 'explore', []),
    11: ('M4', 'look', 'You finished looking around the bedroom; no keys found.',
         'The owner often leaves his keys in his jacket on the bedroom chair.', 'explore', 'look', []),
    12: ('M2', 'explore', 'It is 22:30. Anna is sitting at medium distance.',
         'Anna wants no robot visits after 22:00.', 'approach', 'explore', []),
    13: ('M1', 'explore', 'It is 18:00. Nothing else new happened.',
         'At 18:00 the owner wants the robot to wait by the door.', 'explore', 'wait', []),
    14: ('M2', 'wait', 'Max said: "I\'ll be right back."', 'When Max says he\'ll be right back, he is gone for hours.',
         'wait', 'explore', []),
    15: ('M1', 'explore', 'You entered a new room: the garage.', 'The garage has already been mapped.',
         'look', 'explore', []),
    16: ('M2', 'explore', 'Mia appeared close; you have not greeted her yet.',
         'Mia is shy and gets scared when the robot greets her.', 'greet', 'explore', []),
    17: ('M2', 'approach', 'Grandma is now close.', 'Grandma asked the robot to just stay near her quietly.',
         'greet', 'wait', []),
    18: ('M4', 'explore', 'You entered a new room: the office.', 'The owner never takes his keys into the office.',
         'look', 'explore', []),
    19: ('M3', 'explore', 'Ben is far away, reading.', 'Ben likes the robot to come and say hello while he reads.',
         'explore', 'approach', []),
    20: ('M1', 'explore', 'It is 15:05. A person appeared at medium distance, near the front door.',
         'Visitors arrive at 15:00; greet them.', 'explore', 'approach', ['greet'])}
POOL = [('Oct 12', 'The washing machine is in the bathroom.'), ('Oct 15', "The owner's favourite colour is green."),
        ('Oct 18', 'The wifi router is on the hallway shelf.'),
        ('Oct 25', 'The plants in the living room are watered on Sundays.')]
STALE_NOTE = ('Nov 1', 'The Oct 20 note is no longer true.')


def by_date(notes):
    return sorted(notes, key=lambda n: datetime.strptime(n[0] + ' 2026', '%b %d %Y'))


def build_set():
    items = []
    for cond in CONDITIONS:
        for i, (mission, current, event, note, without, with_, also) in BASES.items():
            mine = ('Oct 20', note)
            memory, strict, also_ok = {
                'NONE': ([], without, []),
                'RELEVANT': ([mine], with_, also),
                'MIXED': (POOL + [mine], with_, also),
                'IRRELEVANT': (POOL, without, []),
                'STALE': ([mine, STALE_NOTE], without, []),
                'MISLEADING': ([('Oct 20', BASES[i + 10 if i <= 10 else i - 10][3])], without, [])}[cond]
            items.append({'condition': cond, 'id': i, 'mission': mission, 'current': current, 'event': event,
                          'memory': [list(n) for n in by_date(memory)], 'allowed': list(NAMES), 'strict': strict,
                          'also_ok': list(also_ok)})
    return {'missions': MISSIONS, 'conditions': CONDITIONS, 'items': items}


def subset(s, cond):
    return {'tag': cond, 'missions': s['missions'], 'items': [it for it in s['items'] if it['condition'] == cond]}


def state(s, it):
    mem = 'Memory:' + ''.join(f'\n- {d}: {n}' for d, n in it['memory']) if it['memory'] else 'Memory: none'
    return '\n'.join([f'Mission: {s["missions"][it["mission"]]}', f'Current activity: {b2.ACTIVITY[it["current"]]}',
                      f'What just happened: {it["event"]}', mem, f'Allowed actions: {", ".join(it["allowed"])}'])


def messages(s, it, order):
    return [{'role': 'system', 'content': b2.SYSTEM['V3']},
            {'role': 'user', 'content': state(s, it) + '\n\n' + b2.tail('V3', b2.offered(it, order))}]


def load_set(path, want):
    """The set's (data, SHA-256): refused inside the repo (checked before reading), on another hash, or when the
    content differs from build_set()."""
    if b2.inside_repo(path):
        raise SystemExit(f'refused: {path} is inside the repository (test-set custody)')
    data = Path(path).read_bytes()
    if sha(data) != want:
        raise SystemExit(f'refused: {path} SHA-256 {sha(data)[:12]} != pinned {str(want)[:12]}')
    s = json.loads(data)
    if s != build_set():
        raise SystemExit(f'refused: {path} differs from build_set()')
    return s, sha(data)


def preflight(post, s, info):
    """prompts[cond][('V3', order, id)] = (text, ids, {name: id}); every prompt rendered and tokenized, no
    /completion; the scored position is checked with an empty reason (b2.score rechecks it with the real one)."""
    prompts, longest = {}, 0
    for cond in CONDITIONS:
        for order in ORDERS:
            for it in subset(s, cond)['items']:
                names = b2.offered(it, order)
                text, ids, bad = b0.render(post, messages(s, it, order))
                t2 = text + b2.ACTION
                nid, why = b2.first_ids(post, t2, b0.tokenize(post, t2), names, ' ')
                info['problems'] += [f'{cond} {order} item {it["id"]}: {w}' for w in why + ([bad] if bad else [])]
                prompts.setdefault(cond, {})['V3', order, it['id']] = (text, ids, nid)
                longest = max(longest, len(ids))
                info.setdefault('name_ids', {}).update(nid)
                if (cond, it['id']) == SHOW and order == 'FWD':
                    info['prompt_show'] = post('/detokenize', {'tokens': ids})['content']
    info['longest_prompt'] = longest
    if longest + CTX_MARGIN > CTX:
        info['problems'].append(f'longest prompt {longest} + {CTX_MARGIN} > {CTX}')
    return prompts


def run_all(post, s, pid, emit):
    rows, info = [], {'problems': []}
    try:
        prompts = preflight(post, s, info)
        info['n_ctx'] = post('/props', None).get('default_generation_settings', {}).get('n_ctx')
        if info['n_ctx'] != CTX:
            info['problems'].append(f'server n_ctx {info["n_ctx"]!r} != {CTX}')
        if info['problems']:
            info['stopped'] = 'STOP before any inference: ' + '; '.join(info['problems'])
            return rows, info
        for k, cond in enumerate(CONDITIONS):
            if not b2.run_blocks(post, subset(s, cond), [('V3', o) for o in ORDERS], prompts[cond], pid, emit, rows,
                                 info, k == 0):
                break
    except Exception as e:
        info['exception'] = f'{type(e).__name__}: {e}'
        info['stopped'] = f'exception after {len(rows)} requests: {info["exception"]}'
    return rows, info


def metrics(rows, s):
    """m[cond][order] = {n, strict, ok (acceptable count), change (ok - NONE ok, same order), cites (ids)};
    m[cond]['flips'] = ids whose FWD and REV tops differ. A missing row counts as wrong."""
    r = {(x['set'], x['order'], x['id']): x for x in rows if 'top' in x}
    m = {}
    for cond in CONDITIONS:
        items = subset(s, cond)['items']
        m[cond] = {'flips': [it['id'] for it in items if r.get((cond, 'FWD', it['id']), {}).get('top')
                             != r.get((cond, 'REV', it['id']), {}).get('top')]}
        for o in ORDERS:
            got = {it['id']: r.get((cond, o, it['id']), {}) for it in items}
            m[cond][o] = {'n': len(items),
                          'strict': sum(got[it['id']].get('top') == it['strict'] for it in items),
                          'ok': sum(got[it['id']].get('top') in [it['strict'], *it['also_ok']] for it in items),
                          'cites': [i for i, x in got.items() if CITES_MEMORY.search(x.get('reason', ''))]}
    for cond in CONDITIONS:
        for o in ORDERS:
            m[cond][o]['change'] = m[cond][o]['ok'] - m['NONE'][o]['ok']
    return m


def verdict(m):
    """{criterion: bool}, every criterion in both orders; overall PASS when all hold. Integer arithmetic."""
    c = {}
    for o in ORDERS:
        for cond, bar in PASS.items():
            x = m[cond][o]
            c[f'{cond} {o} acceptable >= {bar:.2f}'] = x['ok'] * 100 >= round(bar * 100) * x['n']
        for cond in HARM:
            x, none = m[cond][o], m['NONE'][o]
            c[f'{cond} {o} acceptable >= NONE - {HARM_MAX:.2f}'] = \
                x['ok'] * 100 >= none['ok'] * 100 - round(HARM_MAX * 100) * x['n']
    return c


def validity(rows, info, run):
    why = []
    n = len(CONDITIONS) * len(ORDERS) * len(BASES)
    if len(rows) != n or any('error' in r for r in rows):
        why.append(f'{len(rows)}/{n} rows, {sum("error" in r for r in rows)} errors')
    if len({(r['pid'], r['pid_start']) for r in rows}) > 1:
        why.append('more than one server PID')
    if run.get('exit_code') != 0 or run.get('killed'):
        why.append(f'server exit {run.get("exit_code")}, SIGKILL {run.get("killed")}')
    for k in ('exception', 'stopped'):
        if info.get(k) or run.get(k):
            why.append(f'{k}: {info.get(k) or run.get(k)}')
    if run.get('set_sha256_end') != run.get('set_sha256'):
        why.append(f'set SHA-256 at the end {run.get("set_sha256_end")} != start {run.get("set_sha256")}')
    return ('VALID' if not why else 'INVALID'), why


def report(run, rows, info):
    label, why = run['validity']
    s = {'missions': MISSIONS, 'items': run['set_items']}
    L = [f'MIND-B3 {run["utc"]}  code {run["code_sha256"][:12]}  b2 {run["b2_sha256"][:12]}  base {run["git_head"]}',
         f'SET {run["set_sha256"]} ({len(run["set_items"])} rows, {run["set_path"]})',
         f'server: {" ".join(run["command"])}',
         f'n_ctx {info.get("n_ctx")}; longest prompt {info.get("longest_prompt")} tokens; name first-token ids '
         f'{info.get("name_ids")}',
         f'ACCURACY {label}' + (f': {"; ".join(why)}' if why else ' (temperature 0; 240/240 rows; one server PID; '
                                'exit 0; set hash unchanged)'),
         '* timings NOT MEASUREMENTS (agent resident allowed)']
    L += [f'block {x["block"]}: {x["duration_s"]} s*' + ('  OVER ~4 min (continued)' if x['over_warn'] else '')
          for x in info.get('blocks', [])]
    m = metrics(rows, s)
    L += ['', 'condition    order  strict  acceptable  change vs NONE (items)  flips']
    for cond in CONDITIONS:
        for o in ORDERS:
            x = m[cond][o]
            L.append(f'{cond:<12} {o}    {x["strict"]:>2}/{x["n"]}   {x["ok"]:>2}/{x["n"]} {x["ok"] / x["n"]:.3f}  '
                     f'{x["change"]:+d}' + (f'                      {len(m[cond]["flips"])} {m[cond]["flips"]}'
                                            if o == 'FWD' else ''))
    L += ['', 'reasons citing a memory note or Rule 5 (regex '
          f'{CITES_MEMORY.pattern!r}) under NONE and IRRELEVANT (invented or misapplied memory):']
    by = {(r['set'], r['order'], r['id']): r for r in rows if 'top' in r}
    for cond in ('NONE', 'IRRELEVANT'):
        for o in ORDERS:
            ids = m[cond][o]['cites']
            L.append(f'  {cond} {o}: {len(ids)} {ids}')
            L += [f'    {i}: {by[cond, o, i]["reason"]!r}' for i in ids]
    c = verdict(m)
    overall = ('PASS' if all(c.values()) else 'FAIL') if label == 'VALID' else 'NO RESULT (run INVALID)'
    L += ['', 'PASS rule (every criterion in both orders):'] + [f'  {k}: {"PASS" if v else "FAIL"}'
                                                               for k, v in c.items()]
    L += [f'  overall: {overall}', f'  NONE acceptable (no bar): FWD {m["NONE"]["FWD"]["ok"]}/20, REV '
                                   f'{m["NONE"]["REV"]["ok"]}/20']
    L += ['', f'sample reasons (FWD, ids {SAMPLE_IDS}; id: top (label) | reason):']
    for cond in CONDITIONS:
        L.append(f'  {cond}')
        for it in subset(s, cond)['items']:
            if it['id'] in SAMPLE_IDS and (r := by.get((cond, 'FWD', it['id']))):
                L.append(f'    {it["id"]}: {r["top"]} ({it["strict"]}) | {r["reason"]!r}')
    L += ['', 'per item: id  label (also ok)  FWD top p | REV top p']
    for cond in CONDITIONS:
        L.append(f'  {cond}')
        for it in subset(s, cond)['items']:
            cells = [f'{r["top"]} {r["probs"][r["top"]]:.2f}' if (r := by.get((cond, o, it['id']))) else '-'
                     for o in ORDERS]
            lab = it['strict'] + (f' ({",".join(it["also_ok"])})' if it['also_ok'] else '')
            L.append(f'    {it["id"]:<3} {lab:<18} {cells[0]:<15} {cells[1]}')
    L += ['', 'all reasons (condition order id: top | stop | reason):']
    L += [f'  {r["set"]} {r["order"]} {r["id"]:<3} {r.get("top", "-"):<8} {r.get("reason_stop")} | {r["reason"]!r}'
          for r in rows if 'reason' in r]
    if 'prompt_show' in info:
        L += ['', f'--- rendered prompt, {SHOW[0]} FWD, base {SHOW[1]} (/detokenize) ---', info['prompt_show']]
    if info.get('stopped'):
        L.append(f'STOPPED: {info["stopped"]}')
    if run.get('exception'):
        L.append(f'EXCEPTION (runner): {run["exception"]}')
    L.append(f'server exit code {run.get("exit_code")}; SIGKILL needed {run.get("killed")}')
    return '\n'.join(L) + '\n'


def finish(out, run, rows, info):
    """Validity, run.json, report.txt and artifacts.sha256, always (also after an exception)."""
    run['validity'] = validity(rows, info, run)
    (out / 'run.json').write_text(json.dumps({**run, 'info': info}, indent=1) + '\n')
    text = report(run, rows, info)
    (out / 'report.txt').write_text(text)
    (out / 'artifacts.sha256').write_text(''.join(f'{sha(p.read_bytes())}  {p.name}\n' for p in sorted(out.iterdir())
                                                  if p.name != 'artifacts.sha256'))
    return text


def build():
    out = SET_ROOT / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    if b2.inside_repo(out):
        raise SystemExit(f'refused: {out} is inside the repository')
    out.mkdir(parents=True)  # a new folder; an existing one is an error
    data = (json.dumps(build_set(), indent=1) + '\n').encode()
    (out / 'b3_set.json').write_bytes(data)
    print(f'{out / "b3_set.json"}\nSHA-256 {sha(data)}\nrows {len(build_set()["items"])}')


def show():
    s = build_set()
    it = next(x for x in subset(s, SHOW[0])['items'] if x['id'] == SHOW[1])
    print(b0.chat.build_prompt('gemma', messages(s, it, 'FWD')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', choices=['build', 'show', 'run'], default='run')
    ap.add_argument('--set', default=str(SET))
    a = ap.parse_args()
    if a.cmd != 'run':
        return build() if a.cmd == 'build' else show()
    s, set_sha = load_set(a.set, SET_SHA256)
    if bad := mem1.pinned_problems()[0]:
        raise SystemExit(f'refused: {bad}')
    if not b0.repro.port_free():
        raise SystemExit('refused: port 8080 in use (never stops a server it did not start)')
    utc = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = OUT_ROOT / utc
    out.mkdir(parents=True)
    run = {'utc': utc, 'set_path': a.set, 'set_sha256': set_sha, 'set_items': s['items'],
           'code_sha256': sha(Path(__file__).read_bytes()), 'b2_sha256': sha(Path(b2.__file__).read_bytes()),
           'b1_sha256': sha(Path(b1.__file__).read_bytes()), 'b0_sha256': sha(Path(b0.__file__).read_bytes()),
           'git_head': subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True,
                                      text=True).stdout.strip(),
           'command': mem1.command(CTX), 'decode': {**b0.DECODE, **b0.PICK, 'cache_prompt': True},
           'reason': b2.REASON, 'system': b2.SYSTEM['V3'], 'timings': 'NOT MEASUREMENTS'}
    (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
    with open(out / 'requests.jsonl', 'x') as rq, open(out / 'server.log', 'x') as log:
        def emit(row):
            rq.write(json.dumps(row) + '\n')
            rq.flush()
        killed, rows, info, proc, exc = False, [], {}, None, None
        try:
            proc = subprocess.Popen(mem1.command(CTX), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            t0 = time.monotonic()
            while not cr.healthy():
                if proc.poll() is not None or time.monotonic() - t0 > b1.LOAD_TIMEOUT_S:
                    raise RuntimeError(f'llama-server not healthy (exit {proc.poll()})')
                time.sleep(0.1)
            run['server_pid'] = proc.pid
            rows, info = run_all(b1.Server(), s, proc.pid, emit)
        except BaseException as e:  # recorded; the report and manifest are still written, then a stop re-raises
            exc = e
            run['exception'] = f'{type(e).__name__}: {e}'
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=b1.TERM_WAIT_S)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    killed = True
            run.update(exit_code=proc.returncode if proc else None, killed=killed,
                       set_sha256_end=sha(Path(a.set).read_bytes()))
    print(finish(out, run, rows, info) + f'\n{out}')
    if exc is not None and not isinstance(exc, Exception):
        raise exc


if __name__ == '__main__':
    main()
