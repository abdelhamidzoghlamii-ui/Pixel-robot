#!/data/data/com.termux/files/usr/bin/python
"""MIND-B1: picker decision quality, old layout (A) vs new layout (B), each in two option orders. Research only: no
motors, camera, YOLO, M2, main.py or run_mission.py; no MTP, no projector. Accuracy is deterministic (temperature 0);
timings are recorded but are NOT MEASUREMENTS (an agent may be resident).

Set: 40 owner-labelled items kept OUTSIDE the repo (WORKFLOW test-set custody) at ~/mind_b1_set/b1_set.json, SHA-256
pinned below (recorded before any Gemma call); a set path inside the repo is refused before it is read. Imports B0
(b0_cache/b0.py, b0_set.json SHA-256 checked) and picker.py unchanged.
State (both arms, same facts): Mission, Current activity (neutral gerund, never an option phrase), What just happened,
Memory (or none), Allowed actions (catalogue order). Python removed the forbidden actions: only allowed ones are offered
and scored, renormalised over them.
  A  picker.Gemma exactly: raw <state> block, letters, 'Answer: (', no chat template, cache_prompt false; options =
     the allowed b0_set.json catalogue lines ('<name>: <description>'); five letters A-E.
  B  B0 layout: Gemma 4 template (b0.render, checked against /apply-template, thinking off); system = SYS + catalogue
     (as b0.transcript, without mission/memory/events); user = state, blank line, B0 tail with the allowed names;
     first-token scoring over the names (b0.name_ids, b0.probs); cache_prompt true (shared system prefix); cache_n kept.
  FWD = catalogue order; REV: A reverses the options, B the names in the tail. Ties: first offered (as picker).
One server launch (mem1.command(2048) = setup_q4 flags, pins checked), preflight (all 160 prompts rendered and
checked, no /completion), then blocks A-FWD, A-REV, B-FWD, B-REV (40 items each, set order), PAUSE_S between blocks;
a block over BLOCK_WARN_S is noted and the run continues. The first request error stops the run (INVALID).
Metrics (metrics()) and the fixed PASS rule (PASS): see the task; the result is VALID only when all 160 rows exist
without error, one server PID, exit code 0 without SIGKILL, no exception, and the set hash is unchanged at the end.

Usage: b1.py [--set PATH]   (Termux python; from proot Debian or native Termux)
Writes ~/robot_runs/mind_b1/<UTC>/: run.json, requests.jsonl, server.log, report.txt, artifacts.sha256.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'b0_cache'))
sys.path.insert(0, str(REPO))
import b0  # noqa: E402  render(), name_ids(), probs(), DECODE, PICK; imports mem1
import picker  # noqa: E402  Gemma (layout A), LETTERS, http()

mem1, cr, sha = b0.mem1, b0.cr, b0.sha
SET = Path(cr.HOME) / 'mind_b1_set' / 'b1_set.json'
SET_SHA256 = 'b1abba5daa3cb87c30a225af40d3e19e6f0feaf82ba95af48f8b48c4a23461eb'
B0_SET_SHA256 = 'dddb288ace1642b1e93a6e268cb68ff1a028121b6853dfa837ac7849cfaa187b'
OUT_ROOT = Path(cr.HOME) / 'robot_runs' / 'mind_b1'
CTX = 2048
ACTIVITY = {'explore': 'exploring', 'visit': 'visiting', 'greet': 'greeting', 'look': 'looking', 'wait': 'waiting'}
BLOCKS = [('A', 'FWD'), ('A', 'REV'), ('B', 'FWD'), ('B', 'REV')]
PAUSE_S, BLOCK_WARN_S = 60, 240
NEAR_TIE = 0.05
PASS = {'acceptable': 0.80, 'strict': 0.65, 'max_flips': 4}
SHOW_ITEM = 11
REQUEST_TIMEOUT_S, LOAD_TIMEOUT_S, TERM_WAIT_S = 300, 120, 30


def inside_repo(path):
    p = Path(path).absolute()
    return any(q.exists() and os.path.samefile(q, REPO) for q in [p, *p.parents])  # bind mounts: same inode


def load_set(path, want=SET_SHA256):
    """(set, bytes SHA-256): refused when the path is inside the repo (checked before reading) or the hash differs."""
    if inside_repo(path):
        raise SystemExit(f'refused: {path} is inside the repository (test-set custody)')
    data = Path(path).read_bytes()
    if want is not None and sha(data) != want:
        raise SystemExit(f'refused: {path} SHA-256 {sha(data)[:12]} != pinned {want[:12]}')
    return json.loads(data), sha(data)


def state(s, it):
    return '\n'.join([f'Mission: {s["missions"][it["mission"]]}', f'Current activity: {ACTIVITY[it["current"]]}',
                      f'What just happened: {it["event"]}', f'Memory: {it["memory"] or "none"}',
                      f'Allowed actions: {", ".join(it["allowed"])}'])


def offered(it, order):
    return it['allowed'] if order == 'FWD' else it['allowed'][::-1]


def tail(names):  # B0's tail for the given names (equal to b0_set.json's tail for all five)
    return f'Which action next? Answer with one word: {", ".join(names[:-1])} or {names[-1]}.'


def options(b, names):  # layout A: the catalogue lines of the offered names, in their order
    line = {c.split(':')[0]: c for c in b['catalogue']}
    return [line[n] for n in names]


def messages(s, b, it, order):  # layout B
    return [{'role': 'system', 'content': '\n'.join([b['sys'], '', 'Actions:', *b['catalogue']])},
            {'role': 'user', 'content': state(s, it) + '\n\n' + tail(offered(it, order))}]


class Server:
    """POST with B0's timeout, GET for body None (picker.http); keeps the last /completion body and response."""

    def __init__(self, post=None):
        self.post = post or (lambda path, body: picker.http(path) if body is None else
                             cr.post(path, body, timeout=REQUEST_TIMEOUT_S))
        self.last = None

    def __call__(self, path, body=None):
        res = self.post(path, body)
        if path == '/completion':
            self.last = (body, res)
        return res


def preflight(post, s, b):
    """(gemma, prompts, info): every prompt of every block rendered, tokenized and checked; no /completion.
    prompts[(arm, order, id)] = ids (A) or (ids, {name: [id]}) (B). info['problems'] non-empty = STOP."""
    info = {'problems': []}
    g = picker.Gemma(post)
    letters = picker.LETTERS[:len(b['names'])]
    g.letter_ids = [g.encode(L)[0] for L in letters]  # picker builds four (its MODES); five actions here
    info['letter_ids'] = dict(zip(letters, g.letter_ids))
    if any(len(g.encode(L)) != 1 for L in letters) or len(set(g.letter_ids)) != len(letters):
        info['problems'].append(f'letters not single distinct tokens: {info["letter_ids"]}')
    prompts, sizes = {}, {'A': [], 'B': []}
    for arm, order in BLOCKS:
        for it in s['items']:
            names = offered(it, order)
            if arm == 'A':
                ids = g.prompt_ids(state(s, it), options(b, names))
                prompts[arm, order, it['id']] = ids
            else:
                text, ids, bad = b0.render(post, messages(s, b, it, order))
                name_ids, why = b0.name_ids(post, text, ids, names)
                info['problems'] += [f'B {order} item {it["id"]}: {w}' for w in why + ([bad] if bad else [])]
                prompts[arm, order, it['id']] = (ids, name_ids)
                info.setdefault('name_ids', {}).update(name_ids)
            sizes[arm].append(len(ids))
            if it['id'] == SHOW_ITEM and order == 'FWD':
                info[f'prompt_{arm}_item{SHOW_ITEM}'] = post('/detokenize', {'tokens': ids})['content']
    info['prompt_tokens'] = {a: (min(v), max(v)) for a, v in sizes.items()}
    info['n_ctx'] = post('/props', None).get('default_generation_settings', {}).get('n_ctx')
    if info['n_ctx'] != CTX:
        info['problems'].append(f'server n_ctx {info["n_ctx"]!r} != {CTX}')
    if max(max(v) for v in sizes.values()) >= CTX:
        info['problems'].append(f'a prompt does not fit ctx {CTX}: {info["prompt_tokens"]}')
    return g, prompts, info


def score(post, g, s, b, it, arm, order, prompt):
    """One row: probabilities over the offered names (raw and renormalised), top, margin, counters."""
    names = offered(it, order)
    t0 = time.perf_counter()
    if arm == 'A':
        dist = g.decide(state(s, it), options(b, names))  # picker's own request (cache_prompt false)
        body, res = post.last
        if body['prompt'] != prompt:
            raise RuntimeError('picker.Gemma sent ids differing from the preflight')
        raw, norm = b0.probs(res, {n: [g.letter_ids[j]] for j, n in enumerate(names)})
        if max(abs(norm[n] - dist[c]) for n, c in zip(names, options(b, names))) > 1e-9:
            raise RuntimeError('letter probabilities differ from picker.Gemma.decide')
    else:
        ids, name_ids = prompt
        body = {'prompt': ids, **b0.DECODE, **b0.PICK, 'cache_prompt': True}
        res = post('/completion', body)
        raw, norm = b0.probs(res, {n: name_ids[n] for n in names})
        if norm is None:
            raise RuntimeError(f'item {it["id"]}: an offered name is outside the top {b0.PICK["n_probs"]}: {raw}')
    top = max(norm, key=norm.get)  # ties: first offered
    second = sorted(norm.values())[-2] if len(norm) > 1 else 0.0
    tm = res.get('timings', {})
    return {'arm': arm, 'order': order, 'id': it['id'], 'offered': names, 'prompt_tokens': len(body['prompt']),
            'prompt_ids_sha256': sha(json.dumps(body['prompt'])), 'cache_prompt': body['cache_prompt'],
            'prompt_n': tm.get('prompt_n'), 'cache_n': tm.get('cache_n'), 'prompt_ms': tm.get('prompt_ms'),
            'wall_s': round(time.perf_counter() - t0, 3), 'probs_raw': raw, 'probs': norm, 'top': top,
            'margin': norm[top] - second}


def run_blocks(post, s, b, pid, emit):
    """Preflight, then the four blocks; returns (rows, info). Stops at the first problem or request error; an
    exception is recorded in info['exception'] and the rows collected so far are kept."""
    rows, info = [], {}
    try:
        g, prompts, info = preflight(post, s, b)
        if info['problems']:
            info['stopped'] = 'STOP before any inference: ' + '; '.join(info['problems'])
            return rows, info
        info['blocks'] = []
        for k, (arm, order) in enumerate(BLOCKS):
            if k:
                time.sleep(PAUSE_S)
            t0 = time.monotonic()
            for it in s['items']:
                try:
                    row = score(post, g, s, b, it, arm, order, prompts[arm, order, it['id']])
                except Exception as e:  # fail closed: recorded, the run ends
                    row = {'arm': arm, 'order': order, 'id': it['id'], 'error': f'{type(e).__name__}: {e}'}
                row['pid'], row['pid_start'] = pid, b0.pid_start(pid)
                rows.append(row)
                emit(row)
                if 'error' in row:
                    info['stopped'] = f'{arm}-{order} item {it["id"]} failed: {row["error"]}'
                    return rows, info
            dur = round(time.monotonic() - t0, 1)
            info['blocks'].append({'block': f'{arm}-{order}', 'duration_s': dur, 'over_warn': dur > BLOCK_WARN_S})
    except Exception as e:
        info['exception'] = f'{type(e).__name__}: {e}'
        info['stopped'] = f'exception after {len(rows)} requests: {info["exception"]}'
    return rows, info


def metrics(rows, s):
    """{'blocks': {'A-FWD': {...}}, 'arms': {'A': {...}}} from the recorded rows alone; denominators are the set's
    items (a missing row counts as wrong)."""
    items, n = {it['id']: it for it in s['items']}, len(s['items'])
    top = {(r['arm'], r['order'], r['id']): r['top'] for r in rows if 'top' in r}
    margin = {(r['arm'], r['order'], r['id']): r['margin'] for r in rows if 'top' in r}
    blocks, arms = {}, {}
    for arm, order in BLOCKS:
        got = {i: top.get((arm, order, i)) for i in items}
        strict = sum(got[i] == it['strict'] for i, it in items.items())
        ok = sum(got[i] in [it['strict'], *it['also_ok']] for i, it in items.items())
        recall, confusion = {}, {}
        for i, it in items.items():
            h, m = recall.get(it['strict'], (0, 0))
            recall[it['strict']] = (h + (got[i] == it['strict']), m + 1)
            row = confusion.setdefault(it['strict'], {})
            row[str(got[i])] = row.get(str(got[i]), 0) + 1
        copy_ids = [i for i, it in items.items() if it['strict'] != it['current']]
        copied = [i for i in copy_ids if got[i] == items[i]['current']]
        blocks[f'{arm}-{order}'] = {
            'n': n, 'rows': sum(v is not None for v in got.values()), 'strict': strict / n, 'acceptable': ok / n,
            'recall': {k: list(v) for k, v in sorted(recall.items())}, 'confusion': confusion,
            'copy_rate': len(copied) / len(copy_ids) if copy_ids else None, 'copy': [len(copied), len(copy_ids)],
            'copied_ids': copied,
            'near_ties': [i for i in items if margin.get((arm, order, i), 1.0) < NEAR_TIE]}
    for arm in ('A', 'B'):
        flips = [i for i in items if top.get((arm, 'FWD', i)) != top.get((arm, 'REV', i))]
        F, R = blocks[f'{arm}-FWD'], blocks[f'{arm}-REV']
        ok = min(F['acceptable'], R['acceptable']) >= PASS['acceptable'] and \
            min(F['strict'], R['strict']) >= PASS['strict'] and len(flips) <= PASS['max_flips']
        arms[arm] = {'flips': flips, 'verdict': 'PASS' if ok else 'FAIL'}
    return {'blocks': blocks, 'arms': arms}


def validity(rows, info, run):
    why = []
    if len(rows) != len(BLOCKS) * run['set_rows'] or any('error' in r for r in rows):
        why.append(f'{len(rows)} rows, {sum("error" in r for r in rows)} errors')
    if len({(r['pid'], r['pid_start']) for r in rows}) > 1:
        why.append('more than one server PID')
    if run.get('exit_code') != 0 or run.get('killed'):
        why.append(f'server exit {run.get("exit_code")}, SIGKILL {run.get("killed")}')
    for k in ('exception', 'stopped'):
        if info.get(k) or run.get(k):
            why.append(f'{k}: {info.get(k) or run.get(k)}')
    if run.get('set_sha256_end') != run['set_sha256']:
        why.append(f'set SHA-256 at the end {run.get("set_sha256_end")} != start')
    return ('VALID' if not why else 'INVALID'), why


def report(run, rows, info, m):
    s_label, why = run['validity']
    L = [f'MIND-B1 {run["utc"]}  code {run["code_sha256"][:12]}  set {run["set_sha256"][:12]} ({run["set_rows"]} '
         f'rows, {run["set_path"]})  base {run["git_head"]}',
         f'server: {" ".join(run["command"])}',
         f'n_ctx {info.get("n_ctx")}; prompt tokens (min, max) {info.get("prompt_tokens")}',
         f'letter ids {info.get("letter_ids")}; name ids {info.get("name_ids")}',
         f'ACCURACY {s_label}' + (f': {"; ".join(why)}' if why else ' (temperature 0; 160/160 rows; one server PID; '
                                  'exit 0; set hash unchanged)'),
         '* timings NOT MEASUREMENTS (agent resident allowed)']
    L += [f'block {x["block"]}: {x["duration_s"]} s*' + ('  OVER ~4 min (continued)' if x['over_warn'] else '')
          for x in info.get('blocks', [])]
    L += ['', 'block   strict        acceptable    copy rate      near-ties (margin < 0.05)']
    for k, v in m['blocks'].items():
        L.append(f'{k:<7} {v["strict"]:.3f} {round(v["strict"] * v["n"]):>2}/{v["n"]}  '
                 f'{v["acceptable"]:.3f} {round(v["acceptable"] * v["n"]):>2}/{v["n"]}  '
                 f'{(v["copy_rate"] or 0):.3f} {v["copy"][0]:>2}/{v["copy"][1]}  {v["near_ties"]}')
    L += ['', 'recall per strict label (hits/n):']
    L += [f'  {k:<7} ' + '  '.join(f'{a} {h}/{n}' for a, (h, n) in v['recall'].items()) for k, v in m['blocks'].items()]
    L += ['', 'confusion (rows strict, columns chosen):']
    for k, v in m['blocks'].items():
        L.append(f'  {k}  ' + '; '.join(f'{a} -> ' + ', '.join(f'{c} {n}' for c, n in sorted(r.items()))
                                         for a, r in sorted(v['confusion'].items())))
    L += ['', f'copied current activity (ids): ' + '; '.join(f'{k} {v["copied_ids"]}' for k, v in m['blocks'].items())]
    L += ['', f'PASS rule: acceptable >= {PASS["acceptable"]} and strict >= {PASS["strict"]} in both orders, '
              f'flips <= {PASS["max_flips"]}/40']
    L += [f'  arm {a}: flips {len(v["flips"])} {v["flips"]}  -> {v["verdict"]}' for a, v in m['arms'].items()]
    L += ['', 'id  strict (also ok)       A-FWD          A-REV          B-FWD          B-REV']
    by = {(r['arm'], r['order'], r['id']): r for r in rows}
    for it in run['items']:
        cells = []
        for arm, order in BLOCKS:
            r = by.get((arm, order, it['id']), {})
            cells.append(f'{r["top"]} {r["probs"][r["top"]]:.2f}' if 'top' in r else r.get('error', '-')[:14])
        lab = it['strict'] + (f' ({",".join(it["also_ok"])})' if it['also_ok'] else '')
        L.append(f'{it["id"]:<3} {lab:<22} ' + ''.join(f'{c:<15}' for c in cells))
    for arm in ('A', 'B'):
        if f'prompt_{arm}_item{SHOW_ITEM}' in info:
            L += ['', f'--- rendered prompt, arm {arm} FWD, item {SHOW_ITEM} (/detokenize) ---',
                  info[f'prompt_{arm}_item{SHOW_ITEM}']]
    if info.get('stopped'):
        L.append(f'STOPPED: {info["stopped"]}')
    if run.get('exception'):
        L.append(f'EXCEPTION (runner): {run["exception"]}')
    L.append(f'server exit code {run.get("exit_code")}; SIGKILL needed {run.get("killed")}')
    return '\n'.join(L) + '\n'


def finish(out, run, rows, info):
    """Metrics, validity, run.json, report.txt and artifacts.sha256, always (also after an exception)."""
    run['metrics'] = metrics(rows, {'items': run['items']})
    run['validity'] = validity(rows, info, run)
    (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
    text = report(run, rows, info, run['metrics'])
    (out / 'report.txt').write_text(text)
    (out / 'artifacts.sha256').write_text(''.join(f'{sha(p.read_bytes())}  {p.name}\n' for p in sorted(out.iterdir())
                                                  if p.name != 'artifacts.sha256'))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default=str(SET))
    a = ap.parse_args()
    s, set_sha = load_set(a.set)
    b_bytes = b0.SET.read_bytes()
    if sha(b_bytes) != B0_SET_SHA256:
        raise SystemExit(f'refused: {b0.SET} SHA-256 changed')
    b = json.loads(b_bytes)
    if bad := mem1.pinned_problems()[0]:
        raise SystemExit(f'refused: {bad}')
    if not b0.repro.port_free():
        raise SystemExit('refused: port 8080 in use (never stops a server it did not start)')
    utc = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = OUT_ROOT / utc
    out.mkdir(parents=True)
    run = {'utc': utc, 'set_path': str(a.set), 'set_sha256': set_sha, 'set_rows': len(s['items']),
           'b0_set_sha256': B0_SET_SHA256, 'code_sha256': sha(Path(__file__).read_bytes()),
           'picker_sha256': sha(Path(picker.__file__).read_bytes()), 'b0_sha256': sha(Path(b0.__file__).read_bytes()),
           'git_head': subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True,
                                      text=True).stdout.strip(),
           'command': mem1.command(CTX), 'decode_b': {**b0.DECODE, **b0.PICK, 'cache_prompt': True},
           'missions': s['missions'], 'items': s['items'], 'timings': 'NOT MEASUREMENTS'}
    (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')  # set SHA-256 on disk before the server starts
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
                if proc.poll() is not None or time.monotonic() - t0 > LOAD_TIMEOUT_S:
                    raise RuntimeError(f'llama-server not healthy (exit {proc.poll()})')
                time.sleep(0.1)
            run['server_pid'] = proc.pid
            rows, info = run_blocks(Server(), s, b, proc.pid, emit)
        except BaseException as e:  # recorded; the report and manifest are still written, then a stop re-raises
            exc = e
            run['exception'] = f'{type(e).__name__}: {e}'
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=TERM_WAIT_S)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    killed = True
            run.update(exit_code=proc.returncode if proc else None, killed=killed, info=info,
                       set_sha256_end=sha(Path(a.set).read_bytes()))
    print(finish(out, run, rows, info) + f'\n{out}')
    if exc is not None and not isinstance(exc, Exception):
        raise exc


if __name__ == '__main__':
    main()
