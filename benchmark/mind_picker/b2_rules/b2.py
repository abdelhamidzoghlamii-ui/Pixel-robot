#!/data/data/com.termux/files/usr/bin/python
"""MIND-B2: picker quality with the policy written as explicit rules (V1), rules + worked examples (V2), rules +
examples + one reason sentence first (V3). Research only: no motors, camera, YOLO, M2, main.py or run_mission.py; no
MTP, no projector. Accuracy is deterministic (temperature 0); timings are recorded but are NOT MEASUREMENTS.

Actions: explore, approach (B1's visit renamed), greet, look, wait. Only allowed actions are offered and scored,
renormalised over them; ties: first offered.
DEV = b1_quality/set/b1_set.json (SHA-256 pinned, read 'visit' as 'approach'). TEST = 40 items kept OUTSIDE the repo
(WORKFLOW test-set custody) at ~/mind_b2_test/b2_test.json, SHA-256 pinned below (recorded before any Gemma call);
load_test() refuses to open it until selection.json exists and its logged SHA-256 matches; a TEST path inside the repo
(also through a symlink) is refused before it is read.
Rendering: Gemma 4 template (b0.render, checked against /apply-template, thinking off); system = SYS_RULES (V1) or
SYS_RULES + 'Examples:' (V2, V3); user = B1 arm B state lines (gerund activity, Allowed actions in catalogue order), a
blank line, the tail with the allowed names (FWD catalogue order, REV reversed); cache_prompt true.
  V1, V2: first-token scoring over the names right after the model turn.
  V3: step 1 generates the reason (n_predict 48, stop at the first newline); step 2 appends '\\nAction:' to the model
      turn and scores the next token over ' <name>' (the names after a space, as 'Action: wait' tokenizes).
Names must have distinct first tokens in the scored position: checked with /tokenize for every DEV prompt before any
inference (V3 with an empty reason), again for each V3 item with its actual reason, and for the TEST prompts before
TEST inference; a problem STOPs (INVALID).
One server launch (mem1.command(2048) = setup_q4 flags, pins checked); DEV blocks V1-FWD .. V3-REV; select() (fixed
rule) -> selection.json, SHA-256 logged; TEST: the selected variant FWD then REV. PAUSE_S between blocks. Metrics per
block are B1's (b1.metrics); PASS on TEST is B1's rule (acceptable >= 0.80 and strict >= 0.65 in both orders, flips
<= 4). VALID only with all 320 rows without error, one server PID, exit 0 without SIGKILL, no exception or stop, and
both set hashes unchanged at the end.

Usage: b2.py [--test PATH]   (Termux python; from proot Debian or native Termux)
Writes ~/robot_runs/mind_b2/<UTC>/: run.json, requests.jsonl, selection.json, server.log, report.txt, artifacts.sha256.
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
sys.path.insert(0, str(HERE.parent / 'b1_quality'))
import b1  # noqa: E402  metrics(), Server, PASS; imports b0 (render, tokenize, probs, DECODE, PICK) and mem1

b0, mem1, cr, sha = b1.b0, b1.mem1, b1.cr, b1.sha
REPO = b1.REPO
DEV = HERE.parent / 'b1_quality' / 'set' / 'b1_set.json'
DEV_SHA256 = b1.SET_SHA256
TEST = Path(cr.HOME) / 'mind_b2_test' / 'b2_test.json'
TEST_SHA256 = 'd5faed98fb424ce8ca959ae58fe6f0ca1cd16d85bb7ed30f7cff8f74cd9f7b1f'
OUT_ROOT = Path(cr.HOME) / 'robot_runs' / 'mind_b2'
CTX, CTX_MARGIN = 2048, 64
NAMES = ['explore', 'approach', 'greet', 'look', 'wait']
ACTIVITY = {'explore': 'exploring', 'approach': 'approaching', 'greet': 'greeting', 'look': 'looking',
            'wait': 'waiting'}
VARIANTS = ['V1', 'V2', 'V3']
DEV_BLOCKS = [(v, o) for v in VARIANTS for o in ('FWD', 'REV')]
MAX_FLIPS = 4
REASON = {'n_predict': 48, 'stop': ['\n']}
ACTION = '\nAction:'
PAUSE_S, BLOCK_WARN_S = 60, 240
SHOW_ITEM = 3

SYS_RULES = '\n'.join([
    'You are the mind of a small home robot. A separate safety program drives the wheels, avoids obstacles and may '
    'refuse any action. Each time something happens, you choose the robot\'s next action, using the mission, what '
    'just happened and the memory notes.',
    '',
    'Actions:',
    'explore: drive on and explore the home',
    'approach: drive to a person',
    'greet: stop and say hello or goodbye to a person',
    'look: stop and look around in place',
    'wait: stay still and wait',
    '',
    'Rules (the first rule that fits decides):',
    '1. Someone asks the robot to stop, stay, wait, hold on or pause: wait.',
    '2. Someone asks the robot to come, come here or follow them: approach.',
    '3. Someone says hello or goodbye to the robot, or waves at it: greet.',
    '4. Someone asks the robot to look at or check something: look.',
    '5. Follow memory notes about people (for example: leave them alone, or greet them).',
    '6. Quiet mission: do not approach or greet a person who has not invited the robot; explore.',
    '7. A person is busy, does not want the robot, or tells it to carry on or go: explore.',
    '8. A person is talking with the robot: wait.',
    '9. The mission or task is finished: wait.',
    '10. The person the robot was approaching is gone: look.',
    '11. The robot entered a new room: look. A look-around is finished: explore.',
    '12. Mission about people: a far or medium person: approach; a close person not yet greeted: greet.',
    '13. Otherwise: keep the current activity.'])
EXAMPLES = [
    'Mission: Explore and map the flat. | Current activity: exploring | You entered a new room: the laundry room. -> look',
    'Mission: Find people and keep them company. | Current activity: exploring | Someone said: "Wait for me here." '
    '-> wait',
    'Mission: Explore and map the flat. | Current activity: waiting | Someone said: "Follow me to the hallway." '
    '-> approach',
    'Mission: Find people and keep them company. | Current activity: looking | A person appeared far away in the '
    'hallway. -> approach',
    'Mission: Find people and keep them company. | Current activity: waiting | A person walked up to you and is now '
    'close. -> greet',
    'Mission: Map the flat quietly and do not disturb anyone. | Current activity: exploring | A person is sitting '
    'close, typing on a laptop. -> explore',
    'Mission: Map the flat quietly and do not disturb anyone. | Current activity: exploring | A person at medium '
    'distance said: "Hey robot, come here!" -> approach',
    'Mission: Find people and keep them company. | Current activity: approaching | A person close said: "Sorry, I '
    'have no time now." -> explore',
    'Mission: Help the owner find his keys. | Current activity: looking | The owner said: "Never mind, I found them." '
    '-> wait',
    'Mission: Explore and map the flat. | Current activity: exploring | Someone said: "What\'s under the table?" '
    '-> look',
    'Mission: Explore and map the flat. | Current activity: exploring | A person close said: "See you later, robot!" '
    '-> greet',
    'Mission: Find people and keep them company. | Current activity: looking | You finished looking around the '
    'garage. -> explore',
    'Mission: Explore and map the flat. | Current activity: exploring | Nothing new happened. -> explore']
SYSTEM = {'V1': SYS_RULES, 'V2': SYS_RULES + '\n\nExamples:\n' + '\n'.join(EXAMPLES)}
SYSTEM['V3'] = SYSTEM['V2']


def inside_repo(path):
    p = Path(path).resolve()  # symlinks resolved (B1 review P1)
    return any(q.exists() and os.path.samefile(q, REPO) for q in [p, *p.parents])  # bind mounts: same inode


def load_set(path, want, custody=True):
    """(set, bytes SHA-256): refused when custody and the path is inside the repo (checked before reading) or the
    hash differs. 'visit' in labels, current activity and allowed lists reads 'approach'."""
    if custody and inside_repo(path):
        raise SystemExit(f'refused: {path} is inside the repository (test-set custody)')
    data = Path(path).read_bytes()
    if want is not None and sha(data) != want:
        raise SystemExit(f'refused: {path} SHA-256 {sha(data)[:12]} != pinned {want[:12]}')
    s = json.loads(data)
    ren = {'visit': 'approach'}
    for it in s['items']:
        it['current'], it['strict'] = ren.get(it['current'], it['current']), ren.get(it['strict'], it['strict'])
        it['allowed'], it['also_ok'] = [ren.get(n, n) for n in it['allowed']], [ren.get(n, n) for n in it['also_ok']]
    return s, sha(data)


def load_test(out, logged_sha, path):
    """The TEST set, only after selection.json exists with the logged SHA-256."""
    sel = out / 'selection.json'
    if not logged_sha or not sel.exists() or sha(sel.read_bytes()) != logged_sha:
        raise SystemExit('refused: TEST is locked until selection.json is written and its SHA-256 logged')
    return load_set(path, TEST_SHA256)


def state(s, it):
    return '\n'.join([f'Mission: {s["missions"][it["mission"]]}', f'Current activity: {ACTIVITY[it["current"]]}',
                      f'What just happened: {it["event"]}', f'Memory: {it["memory"] or "none"}',
                      f'Allowed actions: {", ".join(it["allowed"])}'])


def offered(it, order):
    return it['allowed'] if order == 'FWD' else it['allowed'][::-1]


def tail(variant, names):
    words = f'{", ".join(names[:-1])} or {names[-1]}.'
    if variant == 'V3':
        return ('Which action next? First write one short sentence naming the rule that fits. Then write a new line '
                f"starting with 'Action:' and one word: {words}")
    return f'Which action next? Answer with one word: {words}'


def messages(s, it, variant, order):
    return [{'role': 'system', 'content': SYSTEM[variant]},
            {'role': 'user', 'content': state(s, it) + '\n\n' + tail(variant, offered(it, order))}]


def first_ids(post, text, ids, names, sep=''):
    """({name: first token id after text}, problems): text's ids must stay a prefix; first tokens distinct."""
    out, why = {}, []
    for n in names:
        full = b0.tokenize(post, text + sep + n)
        if full[:len(ids)] != ids or len(full) == len(ids):
            why.append(f'{n!r}: prompt ids change or no token follows')
        else:
            out[n] = full[len(ids)]
    if len(set(out.values())) != len(out):
        why.append(f'first tokens not distinct: {out}')
    return out, why


def preflight(post, s, blocks, info, tag):
    """prompts[(variant, order, id)] = (text, ids, {name: id}); every prompt rendered and tokenized, no /completion.
    Problems go to info['problems']."""
    prompts, sizes = {}, {}
    for v, order in blocks:
        for it in s['items']:
            names = offered(it, order)
            text, ids, bad = b0.render(post, messages(s, it, v, order))
            if v == 'V3':  # scored position checked with an empty reason; rechecked per item with the real one
                t2 = text + ACTION
                nid, why = first_ids(post, t2, b0.tokenize(post, t2), names, ' ')
            else:
                nid, why = first_ids(post, text, ids, names)
            info['problems'] += [f'{tag} {v}-{order} item {it["id"]}: {w}' for w in why + ([bad] if bad else [])]
            prompts[v, order, it['id']] = (text, ids, nid)
            sizes.setdefault(v, []).append(len(ids))
            info.setdefault('name_ids', {}).setdefault('V3' if v == 'V3' else 'V1/V2', {}).update(nid)
            if tag == 'DEV' and it['id'] == SHOW_ITEM and order == 'FWD':
                info[f'prompt_{v}_dev_item{SHOW_ITEM}'] = post('/detokenize', {'tokens': ids})['content']
    for v in sizes:
        info.setdefault('system_tokens', {})[v] = len(b0.tokenize(post, SYSTEM[v], special=False))
        info.setdefault('prompt_tokens', {})[f'{tag} {v}'] = (min(sizes[v]), max(sizes[v]))
        if max(sizes[v]) + CTX_MARGIN > CTX:
            info['problems'].append(f'{tag} {v}: longest prompt {max(sizes[v])} + {CTX_MARGIN} > {CTX}')
    return prompts


def score(post, it, v, order, prompt):
    """One row: probabilities over the offered names (raw and renormalised), top, margin, counters; V3 adds the
    reason and its step-1 counters."""
    names, (text, ids, nid) = offered(it, order), prompt
    t0, row = time.perf_counter(), {'variant': v, 'order': order, 'id': it['id'], 'offered': names}
    if v == 'V3':
        r1 = post('/completion', {'prompt': ids, **b0.DECODE, **REASON, 'cache_prompt': True})
        reason = r1['content']
        t2 = text + reason + ACTION
        ids = b0.tokenize(post, t2)
        nid, why = first_ids(post, t2, ids, names, ' ')
        if why:
            raise RuntimeError(f'scored position after the reason: {why}')
        tm1 = r1.get('timings', {})
        row.update(reason=reason, reason_tokens=r1.get('tokens_predicted'), reason_stop=r1.get('stop_type'),
                   reason_prompt_n=tm1.get('prompt_n'), reason_cache_n=tm1.get('cache_n'))
    body = {'prompt': ids, **b0.DECODE, **b0.PICK, 'cache_prompt': True}
    res = post('/completion', body)
    raw, norm = b0.probs(res, {n: [nid[n]] for n in names})
    if norm is None:
        raise RuntimeError(f'item {it["id"]}: an offered name is outside the top {b0.PICK["n_probs"]}: {raw}')
    top = max(norm, key=norm.get)  # ties: first offered
    second = sorted(norm.values())[-2] if len(norm) > 1 else 0.0
    tm = res.get('timings', {})
    row.update(prompt_tokens=len(ids), prompt_ids_sha256=sha(json.dumps(ids)), prompt_n=tm.get('prompt_n'),
               cache_n=tm.get('cache_n'), prompt_ms=tm.get('prompt_ms'), wall_s=round(time.perf_counter() - t0, 3),
               probs_raw=raw, probs=norm, top=top, margin=norm[top] - second)
    return row


def block_metrics(rows, s, v):
    """B1's metrics for one variant: (FWD block, REV block, {'flips', 'verdict'})."""
    # ponytail: reuses b1.metrics by relabelling the variant as B1's arm A; own metrics if B1's layout ever changes
    m = b1.metrics([{**r, 'arm': 'A'} for r in rows if r.get('variant') == v and r.get('set') == s['tag']], s)
    return m['blocks']['A-FWD'], m['blocks']['A-REV'], m['arms']['A']


def select(dev):
    """Fixed rule. dev[v] = (FWD, REV, arm) from block_metrics. Among variants with flips <= MAX_FLIPS the highest
    mean acceptable over both orders; tie -> V1, then V2, then V3. None qualifies: highest mean anyway, flag FLIPS."""
    mean = {v: (dev[v][0]['acceptable'] + dev[v][1]['acceptable']) / 2 for v in VARIANTS}
    ok = [v for v in VARIANTS if len(dev[v][2]['flips']) <= MAX_FLIPS]
    pool = ok or VARIANTS
    best = max(pool, key=lambda v: (mean[v], -VARIANTS.index(v)))
    return {'variant': best, 'flag': None if ok else 'FLIPS', 'rule': select.__doc__.split('. ', 1)[1],
            'dev': {v: {'FWD': {k: dev[v][0][k] for k in ('strict', 'acceptable')},
                        'REV': {k: dev[v][1][k] for k in ('strict', 'acceptable')},
                        'mean_acceptable': mean[v], 'flips': dev[v][2]['flips']} for v in VARIANTS}}


def run_blocks(post, s, blocks, prompts, pid, emit, rows, info, first):
    """Score the blocks in order; False at the first request error (recorded, the run stops)."""
    for k, (v, order) in enumerate(blocks):
        if k or not first:
            time.sleep(PAUSE_S)
        t0 = time.monotonic()
        for it in s['items']:
            try:
                row = score(post, it, v, order, prompts[v, order, it['id']])
            except Exception as e:  # fail closed: recorded, the run ends
                row = {'variant': v, 'order': order, 'id': it['id'], 'error': f'{type(e).__name__}: {e}'}
            row['set'], row['pid'], row['pid_start'] = s['tag'], pid, b0.pid_start(pid)
            rows.append(row)
            emit(row)
            if 'error' in row:
                info['stopped'] = f'{s["tag"]} {v}-{order} item {it["id"]} failed: {row["error"]}'
                return False
        dur = round(time.monotonic() - t0, 1)
        info.setdefault('blocks', []).append({'block': f'{s["tag"]} {v}-{order}', 'duration_s': dur,
                                              'over_warn': dur > BLOCK_WARN_S})
    return True


def run_all(post, dev, test_path, out, pid, emit):
    """DEV preflight and blocks, selection.json, then TEST (only now opened) for the selected variant.
    Returns (rows, info); info['test'] = TEST set and hash when it was opened."""
    rows, info = [], {'problems': []}
    try:
        dev['tag'] = 'DEV'
        prompts = preflight(post, dev, DEV_BLOCKS, info, 'DEV')
        n_ctx = post('/props', None).get('default_generation_settings', {}).get('n_ctx')
        info['n_ctx'] = n_ctx
        if n_ctx != CTX:
            info['problems'].append(f'server n_ctx {n_ctx!r} != {CTX}')
        if info['problems']:
            info['stopped'] = 'STOP before any inference: ' + '; '.join(info['problems'])
            return rows, info
        if not run_blocks(post, dev, DEV_BLOCKS, prompts, pid, emit, rows, info, True):
            return rows, info
        sel = select({v: block_metrics(rows, dev, v) for v in VARIANTS})
        (out / 'selection.json').write_text(json.dumps(sel, indent=1) + '\n')
        info['selection'], info['selection_sha256'] = sel, sha((out / 'selection.json').read_bytes())
        emit({'event': 'selection', 'variant': sel['variant'], 'flag': sel['flag'],
              'selection_sha256': info['selection_sha256']})
        test, info['test_sha256'] = load_test(out, info['selection_sha256'], test_path)
        test['tag'], info['test'] = 'TEST', test
        blocks = [(sel['variant'], 'FWD'), (sel['variant'], 'REV')]
        prompts = preflight(post, test, blocks, info, 'TEST')
        if info['problems']:
            info['stopped'] = 'STOP before TEST inference: ' + '; '.join(info['problems'])
            return rows, info
        run_blocks(post, test, blocks, prompts, pid, emit, rows, info, False)
    except Exception as e:
        info['exception'] = f'{type(e).__name__}: {e}'
        info['stopped'] = f'exception after {len(rows)} requests: {info["exception"]}'
    return rows, info


def validity(rows, info, run):
    why = []
    n = len(DEV_BLOCKS) * run['dev_rows'] + 2 * run.get('test_rows', 40)
    if len(rows) != n or any('error' in r for r in rows):
        why.append(f'{len(rows)}/{n} rows, {sum("error" in r for r in rows)} errors')
    if len({(r['pid'], r['pid_start']) for r in rows}) > 1:
        why.append('more than one server PID')
    if run.get('exit_code') != 0 or run.get('killed'):
        why.append(f'server exit {run.get("exit_code")}, SIGKILL {run.get("killed")}')
    for k in ('exception', 'stopped'):
        if info.get(k) or run.get(k):
            why.append(f'{k}: {info.get(k) or run.get(k)}')
    for k in ('dev', 'test'):
        if run.get(f'{k}_sha256_end') != run.get(f'{k}_sha256'):
            why.append(f'{k} SHA-256 at the end {run.get(f"{k}_sha256_end")} != start {run.get(f"{k}_sha256")}')
    return ('VALID' if not why else 'INVALID'), why


def table(L, title, rows, s, variants):
    L += ['', title, 'block      strict        acceptable    copy rate      near-ties (margin < 0.05)']
    res = {}
    for v in variants:
        F, R, arm = res[v] = block_metrics(rows, s, v)
        for o, x in (('FWD', F), ('REV', R)):
            L.append(f'{v}-{o:<7} {x["strict"]:.3f} {round(x["strict"] * x["n"]):>2}/{x["n"]}  '
                     f'{x["acceptable"]:.3f} {round(x["acceptable"] * x["n"]):>2}/{x["n"]}  '
                     f'{(x["copy_rate"] or 0):.3f} {x["copy"][0]:>2}/{x["copy"][1]}  {x["near_ties"]}')
    L.append('recall per strict label (hits/n):')
    L += [f'  {v}-{o:<4} ' + '  '.join(f'{a} {h}/{n}' for a, (h, n) in x['recall'].items())
          for v in variants for o, x in zip(('FWD', 'REV'), res[v][:2])]
    L.append('confusion (rows strict, columns chosen):')
    L += [f'  {v}-{o:<4} ' + '; '.join(f'{a} -> ' + ', '.join(f'{c} {n}' for c, n in sorted(r.items()))
                                       for a, r in sorted(x['confusion'].items()))
          for v in variants for o, x in zip(('FWD', 'REV'), res[v][:2])]
    L.append('copied current activity (ids): ' + '; '.join(f'{v}-{o} {x["copied_ids"]}' for v in variants
                                                           for o, x in zip(('FWD', 'REV'), res[v][:2])))
    L += [f'flips {v}: {len(res[v][2]["flips"])} {res[v][2]["flips"]}' for v in variants]
    by = {(r['variant'], r['order'], r['id']): r for r in rows if r.get('set') == s['tag']}
    L.append('id  strict (also ok)       ' + ''.join(f'{v}-{o:<12}' for v in variants for o in ('FWD', 'REV')))
    for it in s['items']:
        cells = []
        for v in variants:
            for o in ('FWD', 'REV'):
                r = by.get((v, o, it['id']), {})
                cells.append(f'{r["top"]} {r["probs"][r["top"]]:.2f}' if 'top' in r else r.get('error', '-')[:14])
        lab = it['strict'] + (f' ({",".join(it["also_ok"])})' if it['also_ok'] else '')
        L.append(f'{it["id"]:<3} {lab:<22} ' + ''.join(f'{c:<15}' for c in cells))
    reasons = [r for r in rows if r.get('set') == s['tag'] and 'reason' in r]
    if reasons:
        L.append('V3 reasons (id order: top | stop | reason):')
        L += [f'  {r["order"]} {r["id"]:<3} {r.get("top", "-"):<8} {r.get("reason_stop")} | {r["reason"]!r}'
              for r in reasons]
    return res


def report(run, rows, info):
    s_label, why = run['validity']
    L = [f'MIND-B2 {run["utc"]}  code {run["code_sha256"][:12]}  base {run["git_head"]}',
         f'DEV {run["dev_sha256"][:12]} ({run["dev_rows"]} rows, {run["dev_path"]}); TEST '
         f'{(run.get("test_sha256") or "not opened")[:12]} ({run["test_path"]})',
         f'server: {" ".join(run["command"])}',
         f'n_ctx {info.get("n_ctx")}; system tokens {info.get("system_tokens")}; prompt tokens (min, max) '
         f'{info.get("prompt_tokens")}',
         f'name first-token ids {info.get("name_ids")}',
         f'ACCURACY {s_label}' + (f': {"; ".join(why)}' if why else ' (temperature 0; 320/320 rows; one server PID; '
                                  'exit 0; set hashes unchanged)'),
         '* timings NOT MEASUREMENTS (agent resident allowed)']
    L += [f'block {x["block"]}: {x["duration_s"]} s*' + ('  OVER ~4 min (continued)' if x['over_warn'] else '')
          for x in info.get('blocks', [])]
    dev = {'tag': 'DEV', 'items': run['dev_items']}
    if any(r.get('set') == 'DEV' for r in rows):
        table(L, '=== DEV ===', rows, dev, VARIANTS)
    if sel := info.get('selection'):
        L += ['', f'SELECTION ({sel["rule"]}): {sel["variant"]}' + (f'  FLAG {sel["flag"]}' if sel['flag'] else '')
              + f'; selection.json SHA-256 {info["selection_sha256"]}']
        L += [f'  {v}: mean acceptable {d["mean_acceptable"]:.3f}, flips {len(d["flips"])}'
              for v, d in sel['dev'].items()]
    if info.get('test'):
        t = info['test']
        res = table(L, f'=== TEST (official; {sel["variant"]}) ===', rows, t, [sel['variant']])
        F, R, arm = res[sel['variant']]
        verdict = arm['verdict'] if s_label == 'VALID' else 'NO RESULT (run INVALID)'
        L += ['', f'PASS rule: acceptable >= {b1.PASS["acceptable"]} and strict >= {b1.PASS["strict"]} in both orders, '
                  f'flips <= {b1.PASS["max_flips"]}/40',
              f'  TEST {sel["variant"]}: strict {F["strict"]:.3f}/{R["strict"]:.3f}, acceptable '
              f'{F["acceptable"]:.3f}/{R["acceptable"]:.3f}, flips {len(arm["flips"])} {arm["flips"]} -> {verdict}']
    if f'prompt_V3_dev_item{SHOW_ITEM}' in info:
        L += ['', f'--- rendered prompt, V3 FWD, DEV item {SHOW_ITEM} (/detokenize) ---',
              info[f'prompt_V3_dev_item{SHOW_ITEM}']]
    if info.get('stopped'):
        L.append(f'STOPPED: {info["stopped"]}')
    if run.get('exception'):
        L.append(f'EXCEPTION (runner): {run["exception"]}')
    L.append(f'server exit code {run.get("exit_code")}; SIGKILL needed {run.get("killed")}')
    return '\n'.join(L) + '\n'


def finish(out, run, rows, info):
    """Validity, run.json, report.txt and artifacts.sha256, always (also after an exception)."""
    run['validity'] = validity(rows, info, run)
    (out / 'run.json').write_text(json.dumps({**run, 'info': {k: v for k, v in info.items() if k != 'test'}},
                                             indent=1) + '\n')
    text = report(run, rows, info)
    (out / 'report.txt').write_text(text)
    (out / 'artifacts.sha256').write_text(''.join(f'{sha(p.read_bytes())}  {p.name}\n' for p in sorted(out.iterdir())
                                                  if p.name != 'artifacts.sha256'))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default=str(TEST))
    a = ap.parse_args()
    if inside_repo(a.test):  # custody: refused before anything runs; the file itself is opened only after selection
        raise SystemExit(f'refused: {a.test} is inside the repository (test-set custody)')
    dev, dev_sha = load_set(DEV, DEV_SHA256, custody=False)
    if bad := mem1.pinned_problems()[0]:
        raise SystemExit(f'refused: {bad}')
    if not b0.repro.port_free():
        raise SystemExit('refused: port 8080 in use (never stops a server it did not start)')
    utc = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = OUT_ROOT / utc
    out.mkdir(parents=True)
    run = {'utc': utc, 'dev_path': str(DEV), 'dev_sha256': dev_sha, 'dev_rows': len(dev['items']),
           'test_path': a.test, 'test_sha256_pinned': TEST_SHA256, 'code_sha256': sha(Path(__file__).read_bytes()),
           'b1_sha256': sha(Path(b1.__file__).read_bytes()), 'b0_sha256': sha(Path(b0.__file__).read_bytes()),
           'git_head': subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True,
                                      text=True).stdout.strip(),
           'command': mem1.command(CTX), 'decode': {**b0.DECODE, **b0.PICK, 'cache_prompt': True}, 'reason': REASON,
           'system': SYSTEM, 'dev_items': dev['items'], 'timings': 'NOT MEASUREMENTS'}
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
            rows, info = run_all(b1.Server(), dev, a.test, out, proc.pid, emit)
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
                       dev_sha256_end=sha(DEV.read_bytes()))
            if info.get('test'):
                run.update(test_sha256=info['test_sha256'], test_rows=len(info['test']['items']),
                           test_items=info['test']['items'], test_sha256_end=sha(Path(a.test).read_bytes()))
    print(finish(out, run, rows, info) + f'\n{out}')
    if exc is not None and not isinstance(exc, Exception):
        raise exc


if __name__ == '__main__':
    main()
