#!/data/data/com.termux/files/usr/bin/python
"""MEM-1: memory-context length vs Gemma speed, recall, RAM and heat. Research only: no motors, camera, microphone,
LiDAR, main.py or run_mission.py. Native Termux only; started by oneshot.sh in this folder, ONE launch = one
experiment at one context size; nothing chains launches.

Runtime: the robot's server_manager command (b1609 dotprod llama-server, Gemma 4 E2B Q4_0, --threads 4
--threads-batch 4 --parallel 1 --swa-full --cache-ram 0) with --ctx-size from --ctx, pinned with taskset -c 4-7 like
the chat benchmarks; no MTP drafter, no projector. Prompts: chat.build_prompt('gemma', [system: instructions + memory,
user: question]) from the frozen memory_set.json (SHA-256 pinned). Decoding: temperature 0, top_k 1, seed 0,
n_predict 48, stop <turn|>, not streamed.

Experiments (--exp):
  primary   ctx 2048; memory 0/128/256/512/1024 tokens
  capacity  ctx 2048 or 4096 (one per launch); memory 1024 only: same prompts, different allocation
  long      ctx 4096; memory 2048 (optional longer-memory condition)
  freetext  ctx 2048; memory 0/128/256/512/1024; the one-sentence instructions: replies kept verbatim, UNSCORED
            (MANUAL REVIEW REQUIRED); speed and resources as in primary
Tasks: primary, capacity and long use the scored structured-recall task: the reply must be one JSON object
{"answer": VALUE} (VALUE copied from the memory, or UNKNOWN), parsed strictly (parse_answer) and scored by exact
match against the frozen key (score); format compliance and recall accuracy are reported separately.
A block = one server start, one memory length, one cache mode. Requests in order: the fixed warm-up (no memory,
cache_prompt false, unscored); the frozen PRIME prompt (instructions + memory + 'Reply with the single word: ready.',
unscored); then all six scored questions, in the same order in every block of a round (rotated by round only).
Modes: FRESH = cache_prompt false on every request (server n_past = 0): the prime is class PRIME, the questions class
FRESH. CACHED = cache_prompt true: the prime is class FIRST (the cold-first-request cost: only the warm-up's few
tokens can be reused) and fills the slot with the instructions + memory prefix, so every scored question is class
CACHED. FRESH and CACHED therefore score the same questions. Reuse is read from the server's timings.cache_n and
checked against the token LCP with the slot's previous prompt; the label never comes from the request flag alone.
Round order: FRESH ascending interleaved with CACHED descending, the next round reversed (position-balanced over each
round pair). Before every block: cores 4-7, skin gate (power_map: VIRTUAL-SKIN <= launch start + 1.0 degC, at most
8 min, else WARM START), a preflight observation. Inside: load, template and token checks (no generation), warm-up,
prime, then each question; no sensor command runs on the request path. After the server stop: a postflight
observation. Telemetry (observe, sampler_loop): one asynchronous sampler thread, every SAMPLE_PERIOD_S (15 s) from the
server start to the end of the requests, pinned to cores 0-3 at nice 10 (requested, then read back and recorded);
each sample = cheap reads (/proc/meminfo, two cpufreq caps, the thermal log's last 4 KiB) + ONE su call (battery,
dumpsys thermalservice, llama-server smaps_rollup) whose root shell also asks for cores 0-3 and nice 10 and reports
what it got. Samples stay in memory (at most MAX_SAMPLES; a dropped sample is counted, never reconstructed) and are
written with the block record after the server stop. The sampler measures its own cost (duration, su time, thread and
root CPU time, missed slots, actual intervals). Before the warm-up and before every request the runner only checks,
in memory, whether the sampler failed or latched a su timeout, its latest sample is missing or older than
SAFETY_MAX_AGE_S, failed its safety reads, or crossed a safety stop (each ends the block, fail closed); after a su
timeout no further observation starts (no postflight) and the launch ends, as when the sampler outlives its join.
Validity (fail closed; validate_block, the one contract of the runner and the report): the pre/postflight readings and
every sample are checked field by field (observation_problems: return codes, finite values within physical limits,
thermal log fresh by its own stamp and age against the reading's UTC, caps unchanged); every reading's UTC must agree
with its monotonic time and every sample's mono with mono_t0 + t; valid samples must cover the block from the server
start to the sampling end, which must not precede the last request's end, with gaps <= SAMPLE_MAX_GAP_S; a failed,
stopped or overflowing sampler, or one whose attempt count disagrees with its records, makes the block INVALID. The
report recomputes block validity and every request's validity and recall outcome from the recorded evidence, with
the expected question, token counts and cache_n taken from the frozen set and the plan (expected), never from the
request record: stored labels are kept for audit and never trusted.
Environmental matching (condition_diffs) is separate: UNMATCHED. A comparison (compare) is COMPARABLE only if its
configuration, planned and actual questions and repeats, completeness, validity and matching all agree; otherwise NOT
COMPARABLE with reason codes. Artifacts are written as strict JSON (no NaN/Infinity: a non-finite value is stored as
a string, i.e. as evidence that fails validation) and read strictly.

Refuses: not native Termux, an agent process, port 8080 in use, battery < 80 % or charging, MemAvailable < 3000 MiB,
Android status not 0, battery >= 45 degC, pinned SHA-256 mismatch, a prompt that would not fit the context with
n_predict, a prompt or template that differs from the frozen set, an invalid launch-start reading. Stops the launch
before a block at Android status >= 3 or battery >= 45 degC, or when either is unreadable (fail closed). Only
servers this process started are stopped. A run is COMPLETE only when every scheduled block finished with every
planned request; anything else is reported INCOMPLETE (or INVALID when run.json is unusable). Partial artifacts
are read defensively (load_run): torn or malformed lines and records missing keys are listed and left out, requests
without a block record are listed as orphans; nothing missing is filled in.

Usage: mem1.py --exp primary|capacity|long|freetext --ctx 2048|4096 [--rounds N] [--smoke] [--thermal-log PATH]
       mem1.py --report RUN_DIR [RUN_DIR ...] [--json PATH]    (report only; starts nothing)
Writes ~/robot_runs/mem1/<UTC>_<exp>_ctx<ctx>[_smoke]/: run.json, blocks.jsonl, requests.jsonl, server logs,
result.json, report.txt, comparisons.json, artifacts.sha256.
"""
import argparse
import hashlib
import http.client
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'llm_objective_setting/conversation/chat_mtp_repro'))
import repro  # noqa: E402  cpufreq(), top_share(), BATTERY_MAX_C; imports chat_mtp and its harness

cm, cr, pm = repro.cm, repro.cr, repro.pm
chat, sm = cm.chat, cm.sm

SET = HERE / 'memory_set.json'
SET_SHA256 = '8feb74f97429417e8e504be1abce979b958950e1b80692b332b7d88cde8f8df4'
SERVER, MODEL = sm.LLAMA_SERVER, repro.MODEL
PINNED = {MODEL: repro.SHA256[MODEL], repro.LIBGGML_CPU: repro.SHA256[repro.LIBGGML_CPU],
          SERVER: '77eb35a0e716dff307e0d33b0b45ce0ed91e216a20db9411a7b9dc94e6624053'}
EXPERIMENTS = {'primary': {'ctx': (2048,), 'lengths': (0, 128, 256, 512, 1024), 'task': 'recall'},
               'capacity': {'ctx': (2048, 4096), 'lengths': (1024,), 'task': 'recall'},
               'long': {'ctx': (4096,), 'lengths': (2048,), 'task': 'recall'},
               'freetext': {'ctx': (2048,), 'lengths': (0, 128, 256, 512, 1024), 'task': 'freetext'}}
DECODING = {'n_predict': 48, 'temperature': 0, 'top_k': 1, 'seed': 0, 'stop': chat.get_stop_tokens('gemma'),
            'stream': False}
WARMUP = chat.build_prompt('gemma', [{'role': 'user', 'content': 'Hi'}])
OUT_ROOT = cr.HOME / 'robot_runs' / 'mem1'
THERMAL_LOG = OUT_ROOT / 'thermal.log'  # oneshot.sh's logger
MIN_MEM_AVAILABLE_MIB, STATUS_STOP, LOAD_TIMEOUT_S, REQUEST_TIMEOUT_S = 3000, 3, 120, 240
SKIN_MATCH_C, BATTERY_MATCH_PCT = 1.0, 10  # matched conditions (repro.py's thresholds); also the skin gate
SERVER_POLICIES = ('policy4', 'policy6')  # cpufreq policies of cores 4-7; caps compared only there
GEN_MIN_TOKENS = 5  # gen tok/s only from replies with at least this many tokens
ENV_MAX_GAP_S = 60  # preflight reading at most this long before the server start, postflight after the stop
SAMPLE_PERIOD_S = 15  # asynchronous telemetry period (rounds 6-8: 5 s root sampling + 2 su calls after every request)
SAMPLE_MAX_GAP_S = 20  # longest gap in valid samples (block start, between, sampling end): one period + 5 s of
#                        sample-duration jitter; a missed slot (30 s) makes the block INVALID (too sparse)
MAX_SAMPLES = 256  # per block (a block is at most ~30 min = 120 slots); more are dropped, counted, INVALID
ROOT_TIMEOUT_S = 10  # one telemetry su call; a slower call is a failed sample
JOIN_S = ROOT_TIMEOUT_S + 5  # the sampler join after the server stop; also the longest possible observation
SAMPLER_CPUS, SAMPLER_NICE, ROOT_CPU_MASK = (0, 1, 2, 3), 10, 'f'  # telemetry off the server's cores 4-7
DIAG_CHARS = 300  # bounded raw diagnostics of a failed root or dumpsys read
ROUND_S = 0.002  # times are stored rounded to 1 ms: mono_t0 + server_s can exceed a later reading's mono by < 2 ms
CLOCK_TOL_S = 1.0  # a sample's mono vs mono_t0 + t; a reading's UTC vs its mono (both clocks) and its thermal age
SAFETY_MAX_AGE_S = SAMPLE_MAX_GAP_S + ROOT_TIMEOUT_S  # between requests: the latest safety sample is at most this old
JSON_INT_MAX = 2 ** 63  # a larger JSON integer is malformed evidence (rejected on load like NaN)
BATTERY_MIN_C, SKIN_RANGE_C = -10, (0, 80)  # plausible ranges; outside = a malformed reading
BATTERY_W_MAX, MEM_MIB_MAX, PSS_KB_MAX = 30, 65536, 16 << 20  # physical limits of a sample (Pixel 7: 8 GB RAM)
BATTERY = '/sys/class/power_supply/battery'
UNKNOWN = 'unknown'  # normalize('UNKNOWN'): the structured abstention
FENCE = re.compile(r'```(?:json)?[ \t]*\n(.*?)\n?[ \t]*```', re.S)


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _no_constant(name):
    raise ValueError(f'non-finite number {name}')


def _finite_float(text):
    v = float(text)
    if not math.isfinite(v):
        raise ValueError(f'non-finite number {text}')
    return v


def _bounded_int(text):
    v = int(text)
    if abs(v) > JSON_INT_MAX:
        raise ValueError(f'integer out of range {text[:20]}...')
    return v


def strict_loads(text):
    """json.loads without NaN, Infinity, -Infinity, an overflowing literal (1e999) or an integer beyond JSON_INT_MAX:
    ValueError instead."""
    return json.loads(text, parse_constant=_no_constant, parse_float=_finite_float, parse_int=_bounded_int)


def finite(o):
    """o with every non-finite float replaced by its string ('nan', 'inf', '-inf'), and every integer beyond
    JSON_INT_MAX by its digits: kept as evidence that fails validation, never written as a number strict_loads
    rejects."""
    if (isinstance(o, float) and not math.isfinite(o)) or (type(o) is int and abs(o) > JSON_INT_MAX):
        return str(o)
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def dumps(o, **kw):
    """Strict JSON for every artifact (allow_nan=False after finite())."""
    return json.dumps(finite(o), allow_nan=False, **kw)


def num(v, lo=None, hi=None):
    """A finite int or float (never a bool) within [lo, hi] when given (an int of any size never raises)."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool) and (type(v) is int or math.isfinite(v))
            and (lo is None or v >= lo) and (hi is None or v <= hi))


def zero(v):
    """An int return code of exactly 0 (False, 0.0, None and strings are not)."""
    return type(v) is int and v == 0


def load_set():
    text = SET.read_text()
    if sha256_text(text) != SET_SHA256:
        raise SystemExit(f'{SET} SHA-256 {sha256_text(text)[:12]} is not the pinned {SET_SHA256[:12]}')
    return json.loads(text)


def prompt_for(s, length, q):
    return chat.build_prompt('gemma', [{'role': 'system', 'content': s['system'] + '\n\n' +
                                        s['lengths'][str(length)]['memory_text']},
                                       {'role': 'user', 'content': q['text']}])


def task_view(s, task):
    """The set as one task sees it: 'recall' (the scored structured instructions) or 'freetext' (the one-sentence
    instructions, unscored): system text and prompt token data swapped; memory, questions and key unchanged."""
    if task == 'recall':
        return {**s, 'task': 'recall'}
    return {**s, 'task': 'freetext', 'system': s['freetext_system'],
            'lengths': {k: {**v, **v['freetext']} for k, v in s['lengths'].items()}}


def messages_for(s, length, q):  # what /apply-template renders into prompt_for's text
    return [{'role': 'system', 'content': s['system'] + '\n\n' + s['lengths'][str(length)]['memory_text']},
            {'role': 'user', 'content': q['text']}]


def command(ctx):
    """server_manager.start_server's argv in its order (coresidency.server_cmd), --ctx-size from ctx, cores 4-7."""
    return ['taskset', '-c', '4-7', SERVER, '-m', MODEL, '--port', str(cr.PORT), '--ctx-size', str(ctx),
            '--threads', sm.THREADS, '--threads-batch', sm.BATCH_THREADS, '--parallel', '1', '--swa-full',
            '--cache-ram', '0', '--host', '127.0.0.1']


def payload(prompt, cache):
    return {'prompt': prompt, **DECODING, 'cache_prompt': cache}


def plan_problems(s, exp, ctx):
    """Reasons the frozen prompts cannot run as planned at this context size (no silent truncation)."""
    why = [] if ctx in EXPERIMENTS[exp]['ctx'] else [f'--exp {exp} runs at ctx {EXPERIMENTS[exp]["ctx"]}, not {ctx}']
    for length in EXPERIMENTS[exp]['lengths']:
        frozen = {**s['lengths'][str(length)]['prompts'], 'prime': s['lengths'][str(length)]['prime_prompt']}
        for q in s['questions'] + [s['prime']]:
            p = frozen[q['id']]
            if p['tokens'] + DECODING['n_predict'] > ctx:
                why.append(f'memory {length} {q["id"]}: {p["tokens"]} prompt + {DECODING["n_predict"]} output tokens '
                           f'> ctx {ctx}')
            if sha256_text(prompt_for(s, length, q)) != p['text_sha256']:
                why.append(f'memory {length} {q["id"]}: prompt text differs from the frozen set (chat.build_prompt?)')
    return why


def schedule(exp, rounds, smoke):
    """[(round, length, mode)]: FRESH ascending interleaved with CACHED descending; odd rounds reversed."""
    lengths = sorted(EXPERIMENTS[exp]['lengths'])
    if smoke:
        lengths = sorted({lengths[0], lengths[-1]})
    base = [b for pair in zip([(l, 'FRESH') for l in lengths], [(l, 'CACHED') for l in reversed(lengths)])
            for b in pair]
    return [(r, *b) for r in range(rounds) for b in (base if r % 2 == 0 else base[::-1])]


def question_order(questions, rnd, smoke):
    """The scored questions of every block in round rnd: all of them, rotated by round only, so every condition of a
    round asks the same questions in the same order."""
    k = rnd % len(questions)
    order = questions[k:] + questions[:k]
    return order[:2] if smoke else order


def matrix(s, plan, smoke):
    """The expected work of a launch: every block with its scored questions (stored in run.json)."""
    return [{'block': seq, 'round': rnd, 'length': length, 'mode': mode,
             'questions': [q['id'] for q in question_order(s['questions'], rnd, smoke)]}
            for seq, (rnd, length, mode) in enumerate(plan)]


def lcp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def expected(s, length, mode, rnd, position):
    """Request `position` of a block, from the frozen set alone (the runner's and the report's only source): its
    question id, prompt tokens, shared prefix and the server's expected cache_n (FRESH: 0; CACHED: the frozen token
    LCP with the request before it: warm-up -> prime, prime -> first question, question -> next). None when the
    memory length or the position is not in the plan (a smoke block asks a prefix of the same order)."""
    L, order = s['lengths'].get(str(length)), question_order(s['questions'], rnd, False)
    if L is None or not 0 <= position <= len(order):
        return None
    if position == 0:
        qid, p, cache = s['prime']['id'], L['prime_prompt'], L['prime_prompt']['lcp_warmup']
    else:
        qid = order[position - 1]['id']
        p = L['prompts'][qid]
        cache = (L['prime_prompt'] if position == 1 else L['prompts'][order[position - 2]['id']])['lcp'][qid]
    return {'question': qid, 'tokens_expected': p['tokens'], 'expected_cache_n': 0 if mode == 'FRESH' else cache,
            'shared_prefix_tokens': L['shared_prefix_tokens']}


# ---------------------------------------------------------------- structured recall (deterministic, frozen key)

def _no_duplicate_keys(pairs):
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError('duplicate key')
    return dict(pairs)


def parse_answer(reply):
    """(format, value). STRICT: the reply (outer whitespace aside) is exactly one JSON object whose only key is
    "answer" with a string value; FENCED: that object is the only content of one Markdown code fence; MALFORMED:
    anything else (prose, extra keys or text, a non-string value, duplicate keys); EMPTY: a blank reply. value is
    None unless STRICT or FENCED. No prose is interpreted; a reply that is not a string (malformed evidence) is
    MALFORMED."""
    if reply is not None and not isinstance(reply, str):
        return 'MALFORMED', None
    text = (reply or '').strip()
    if not text:
        return 'EMPTY', None
    fenced = FENCE.fullmatch(text)
    for fmt, body in (('STRICT', text), ('FENCED', fenced.group(1) if fenced else None)):
        if body is None:
            continue
        try:
            obj = json.loads(body, object_pairs_hook=_no_duplicate_keys)
        except ValueError:
            continue
        if isinstance(obj, dict) and list(obj) == ['answer'] and isinstance(obj['answer'], str):
            return fmt, obj['answer']
    return 'MALFORMED', None


def normalize(value):
    """NFKC, casefolded, whitespace collapsed, one trailing period dropped: the only normalization before the
    exact match."""
    v = ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
    return v[:-1].rstrip() if v.endswith('.') else v


def score(q, fmt, value, present):
    """The outcome of one structured reply. Unparsed (MALFORMED, EMPTY): that format, a failure. A fact the memory
    lacks (absent questions, every question at memory 0): ABSTAINED for UNKNOWN, else UNSUPPORTED. A present fact:
    CORRECT (an accept value), MISSED (UNKNOWN), STALE (the superseded value), DISTRACTOR (the near-miss) or WRONG
    (anything else). Right = CORRECT or ABSTAINED."""
    if value is None:
        return fmt
    v = normalize(value)
    if not present:
        return 'ABSTAINED' if v == UNKNOWN else 'UNSUPPORTED'
    if v in q['accept']:
        return 'CORRECT'
    if v == UNKNOWN:
        return 'MISSED'
    return 'STALE' if v in q['stale'] else 'DISTRACTOR' if v in q['distractor'] else 'WRONG'


def scored_fields(s, q, reply, present):
    """Row fields of a scored reply: recall = format, parsed answer, outcome; freetext = UNSCORED."""
    if s['task'] == 'freetext':
        return {'outcome': 'UNSCORED', 'review': 'MANUAL REVIEW REQUIRED'}
    fmt, value = parse_answer(reply)
    return {'format': fmt, 'answer': value, 'outcome': score(q, fmt, value, present)}


# ---------------------------------------------------------------- request checks

def request_class(mode, prime):
    """The prime request: FIRST in a CACHED block (cold first request with memory), PRIME in a FRESH block; a scored
    question: its block's mode."""
    return ('FIRST' if mode == 'CACHED' else 'PRIME') if prime else mode


def row_problems(cls, data, expected_tokens, expected_cache, shared):
    """Why a reply is not a valid observation of its class; cache reuse only from the server's own counts."""
    tm = data.get('timings', {})
    cache_n, prompt_n = tm.get('cache_n'), tm.get('prompt_n')
    why = []
    if data.get('tokens_evaluated') != expected_tokens:
        why.append(f'tokens_evaluated {data.get("tokens_evaluated")} != {expected_tokens}')
    if data.get('truncated') is not False:  # missing or null is no evidence against a context shift
        why.append(f'truncated {data.get("truncated")!r} (context shift or overflow, or not recorded)')
    if data.get('stop_type') not in ('eos', 'word', 'limit'):
        why.append(f'stop_type {data.get("stop_type")!r}')
    if cache_n != expected_cache:
        why.append(f'cache_n {cache_n} != expected {expected_cache}')
    counts = num(cache_n) and num(prompt_n)  # arithmetic only on checked numbers (else a problem below)
    if counts and prompt_n + cache_n != expected_tokens:
        why.append(f'prompt_n {prompt_n} + cache_n {cache_n} != {expected_tokens}')
    if cls == 'CACHED' and not (num(cache_n) and cache_n >= shared):
        why.append(f'prefix not reused (cache_n {cache_n!r} < {shared})')
    if cls != 'CACHED' and num(cache_n) and cache_n >= shared:
        why.append(f'prefix reused in an uncached request (cache_n {cache_n})')
    for k in ('prompt_ms', 'predicted_ms'):
        if not num(tm.get(k), 0):
            why.append(f'timings.{k} missing or not a finite number >= 0')
    for k, hi in (('prompt_n', expected_tokens), ('cache_n', expected_tokens), ('predicted_n', DECODING['n_predict'])):
        if not (type(tm.get(k)) is int and 0 <= tm[k] <= hi):
            why.append(f'timings.{k} {tm.get(k)!r} not an integer in 0..{hi}')
    for kn, km in (('prompt_n', 'prompt_ms'), ('predicted_n', 'predicted_ms')):  # work takes time
        if type(tm.get(kn)) is int and tm[kn] > 0 and num(tm.get(km)) and tm[km] <= 0:
            why.append(f'timings.{km} {tm[km]} for {kn} {tm[kn]} > 0 (no duration)')
    pps, n, ms = tm.get('predicted_per_second'), tm.get('predicted_n'), tm.get('predicted_ms')
    if pps is not None and not num(pps, 0):
        why.append('timings.predicted_per_second not a finite number >= 0')
    elif pps is not None and type(n) is int and num(ms) and abs(pps * ms - 1000 * n) > 0.01 * 1000 * n:
        why.append(f'timings.predicted_per_second {pps} != 1000 * predicted_n / predicted_ms (server formula)')
    if data.get('stop_type') == 'limit' and n != DECODING['n_predict']:
        why.append(f'stop_type limit with predicted_n {n!r} != n_predict {DECODING["n_predict"]}')
    return why


def proc_mem(pid):
    try:
        return {l.split(':')[0]: l.split(':', 1)[1].strip() for l in open(f'/proc/{pid}/status')
                if l.startswith(('VmRSS', 'VmHWM', 'VmSwap', 'Cpus_allowed_list'))}
    except OSError as e:
        return {'error': str(e)}


def checks(s, length, qs, ctx):
    """No generation: GGUF template render, server tokenization (prompts and the warm-up's LCP with the prime) and
    n_ctx against the frozen set. Returns (record, problems)."""
    frozen = {**s['lengths'][str(length)]['prompts'], 'prime': s['lengths'][str(length)]['prime_prompt']}
    rec, why, ids = {'template': {}, 'tokens': {}}, [], {}
    for q in [s['prime']] + qs:
        p = prompt_for(s, length, q)
        t = cr.post('/apply-template', {'messages': messages_for(s, length, q),
                                        'chat_template_kwargs': {'enable_thinking': False}}, timeout=30)
        rec['template'][q['id']] = t.get('prompt') == p
        if t.get('prompt') != p:
            why.append(f'{q["id"]}: GGUF template render differs from chat.build_prompt')
        ids[q['id']] = cr.post('/tokenize', {'content': p, 'add_special': True, 'parse_special': True}, 30)['tokens']
        rec['tokens'][q['id']] = len(ids[q['id']])
        if len(ids[q['id']]) != frozen[q['id']]['tokens'] or \
                sha256_text(json.dumps(ids[q['id']])) != frozen[q['id']]['ids_sha256']:
            why.append(f'{q["id"]}: server tokens {len(ids[q["id"]])} differ from the frozen '
                       f'{frozen[q["id"]]["tokens"]} (or their ids)')
    warm = cr.post('/tokenize', {'content': WARMUP, 'add_special': True, 'parse_special': True}, 30)['tokens']
    if lcp(warm, ids[s['prime']['id']]) != s['lengths'][str(length)]['prime_prompt']['lcp_warmup']:
        why.append('warm-up tokens: LCP with the prime differs from the frozen set')
    props = props_json()
    rec['chat_template_sha256'] = sha256_text(props.get('chat_template', '')) if props.get('chat_template') else None
    rec['n_ctx'] = props.get('default_generation_settings', {}).get('n_ctx')
    if rec['n_ctx'] != ctx:
        why.append(f'server n_ctx {rec["n_ctx"]!r} != {ctx} (or unknown)')
    return rec, why


def props_json():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{cr.PORT}/props', timeout=10) as r:
            return json.loads(r.read())
    except (OSError, ValueError) as e:
        return {'error': str(e)}


# ---------------------------------------------------------------- telemetry (asynchronous, self-measured)

def thermal_tail(path):
    """coresidency.read_thermal's reading (last line 'UTC z9=.. z10=.. z11=..', stale after THERMAL_MAX_AGE_S) from
    the log's last 4 KiB only: the log grows during a launch and is never read in full here."""
    with open(path, 'rb') as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - 4096))
        last = fh.read().decode(errors='replace').strip().splitlines()[-1].split()
    stamp = datetime.strptime(last[0], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if not -2 <= age <= cr.THERMAL_MAX_AGE_S:
        raise ValueError(f'thermal log {path} last line is {age:.0f} s old (allowed 0..{cr.THERMAL_MAX_AGE_S} s)')
    z = dict(kv.split('=', 1) for kv in last[1:])
    return {'at': last[0], 'thermal_age_s': round(age, 1), **{k: int(z[k]) for k in ('z9', 'z10', 'z11')}}


def server_caps():
    """scaling_max_freq of the server cores' policies (cpufreq sysfs, no su): the per-sample cap check."""
    out = {}
    for p in SERVER_POLICIES:
        try:
            out[p] = {'max': int(Path(f'/sys/devices/system/cpu/cpufreq/{p}/scaling_max_freq').read_text())}
        except (OSError, ValueError) as e:
            out[p] = {'error': str(e)[:DIAG_CHARS]}
    return out


def root_script(pid):
    """The one root command of an observation: the root shell first asks for cores 0-3 and nice 10 for itself (its
    children inherit both), then reads the battery sysfs, dumpsys thermalservice and, for a sample, llama-server's
    smaps_rollup; each section closes with its own return code; finally the shell's actual CPU list, nice and CPU
    time (/proc/$$/stat: utime stime cutime cstime)."""
    b = BATTERY
    s = (f'taskset -p {ROOT_CPU_MASK} $$ >/dev/null 2>&1; renice -n {SAMPLER_NICE} -p $$ >/dev/null 2>&1; '
         f'echo "=== battery"; cat {b}/capacity {b}/status {b}/temp {b}/current_now {b}/voltage_now; echo "rc=$?"; '
         'echo "=== thermalservice"; dumpsys thermalservice; echo "rc=$?"; ')
    if pid is not None:
        s += f'echo "=== smaps_rollup"; cat /proc/{int(pid)}/smaps_rollup; echo "rc=$?"; '
    return s + 'echo "=== sched"; grep Cpus_allowed_list /proc/$$/status; cat /proc/$$/stat; echo "rc=$?"'


def sections(out):
    """{name: (rc, text)} of root_script's '=== name' sections; rc from the closing 'rc=N' line, None when the
    section did not finish."""
    parts = re.split(r'^=== (\w+)\n', out, flags=re.M)
    got = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        m = re.search(r'^rc=(\d+)\s*\Z', body, re.M)
        got[name] = (int(m.group(1)), body[:m.start()]) if m else (None, body)
    return got


def diag(c, op, rc, raw, kind=None):
    """Bounded evidence of a failed root or dumpsys read: operation, kind, return code, su's stderr and the start of
    the raw output (DIAG_CHARS each), with its time."""
    kind = kind or ('missing' if rc is None else 'non-zero return code' if rc != 0 else 'unparseable')
    c.setdefault('diag', []).append({'op': op, 'kind': kind, 'rc': rc, 'stderr': cr.root_stderr()[:DIAG_CHARS],
                                     'raw': str(raw)[:DIAG_CHARS], 'utc': cr.utc(), 'mono': round(time.monotonic(), 3)})


def observe(thermal_log, pid=None):
    """One environment observation. pid None: a pre/postflight reading outside the block (full cpufreq stats, for the
    X1 top share); pid: a telemetry sample of that llama-server (the two server caps, MemAvailable, its PSS). Cheap
    reads in Python; everything needing root in ONE su call (root_script). Every return code is kept; a failure keeps
    bounded diagnostics (diag). Validity is judged by observation_problems, never here."""
    c = {'mono': round(time.monotonic(), 3), 'utc': cr.utc(),
         'cpufreq': repro.cpufreq() if pid is None else server_caps()}
    if pid is not None:
        try:
            c.update(cr.meminfo_mib())
        except (OSError, ValueError, KeyError, IndexError) as e:
            c['meminfo_error'] = f'{type(e).__name__}: {e}'[:DIAG_CHARS]
    try:
        c.update(thermal_tail(thermal_log))
    except (OSError, ValueError, IndexError, KeyError) as e:
        c['thermal_error'] = f'{type(e).__name__}: {e}'[:DIAG_CHARS]
    began = time.monotonic()
    try:
        rc, out = cr.root(root_script(pid), 'mem1_observe', timeout=ROOT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as e:
        rc, out = None, ''
        diag(c, 'su', None, f'{type(e).__name__}: {e}; partial output: {getattr(e, "output", None)!r}',
             'timeout' if isinstance(e, subprocess.TimeoutExpired) else 'exception')
    c.update(root_rc=rc, root_s=round(time.monotonic() - began, 3))
    if rc is not None and rc != 0:
        diag(c, 'su', rc, out)
    sec = sections(out)
    rc_b, text = sec.get('battery', (None, ''))
    c['battery_rc'] = rc_b
    try:
        pct, status, temp, cur, volt = text.split()[:5]
        c.update(battery_pct=int(pct), battery_status=status, battery_c=int(temp) / 10,
                 battery_w=-(int(cur) * int(volt)) / 1e12)
    except ValueError:
        diag(c, 'battery', rc_b, text, 'unparseable')
    if rc_b != 0:
        diag(c, 'battery', rc_b, text)
    rc_d, text = sec.get('thermalservice', (None, ''))
    d = cr.parse_dump(text)
    c.update(dump_rc=rc_d, status=d['status'], skin=d['skin'])
    if rc_d != 0 or d['status'] is None or d['skin'] is None:
        diag(c, 'dumpsys thermalservice', rc_d, text)
    if pid is not None:
        rc_s, text = sec.get('smaps_rollup', (None, ''))
        pss, rss = (re.search(rf'^{k}:\s+(\d+) kB', text, re.M) for k in ('Pss', 'Rss'))
        c.update(smaps_rc=rc_s, pss_pid=pid, pss_kb=int(pss.group(1)) if pss else None,
                 rss_kb=int(rss.group(1)) if rss else None)
        if rc_s != 0 or not pss:
            diag(c, 'smaps_rollup', rc_s, text)
    _, text = sec.get('sched', (None, ''))  # overhead evidence only, not a validity input
    m = re.search(r'^Cpus_allowed_list:\s*(\S+)', text, re.M)
    stat = text.rsplit(')', 1)[1].split() if ')' in text else []
    c['root_cpus'] = m.group(1) if m else None
    try:
        c.update(root_nice=int(stat[16]), root_cpu_s=sum(int(x) for x in stat[11:15]) / os.sysconf('SC_CLK_TCK'))
    except (IndexError, ValueError):
        c['root_sched_raw'] = text[:DIAG_CHARS]
    return c


def pin_sampler():
    """Ask for the calling (sampler) thread on SAMPLER_CPUS at nice SAMPLER_NICE (both per thread on Linux; its su
    clients inherit them) and record what it actually got: a refused or ignored request is recorded, never assumed."""
    tid = threading.get_native_id()
    st = {'tid': tid, 'requested_cpus': list(SAMPLER_CPUS), 'requested_nice': SAMPLER_NICE}
    try:
        os.sched_setaffinity(0, SAMPLER_CPUS)
    except OSError as e:
        st['affinity_error'] = str(e)
    try:
        os.setpriority(os.PRIO_PROCESS, tid, SAMPLER_NICE)
    except OSError as e:
        st['nice_error'] = str(e)
    st.update(cpus=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, tid))
    return st


def timed_out(x):
    """An observation whose su call timed out (its diagnostics say so)."""
    return isinstance(x, dict) and any(isinstance(d, dict) and d.get('kind') == 'timeout'
                                       for d in (x.get('diag') if isinstance(x.get('diag'), list) else []))


def sampler_loop(t0, thermal_log, pid, stop, samples, state):
    """The block's telemetry thread: slot k at t0 + k * SAMPLE_PERIOD_S (slot 0 = the server start) until stop; an
    overrun skips slots (counted in state['missed']), never bunches; no busy wait (Event.wait). Each sample records
    its slot, scheduled and actual start, end, and its own thread CPU time; samples beyond MAX_SAMPLES are dropped
    and counted (state['dropped']), never reconstructed. state['attempts'] counts every observation begun, also one
    that raised or never returned (attempts = recorded + dropped otherwise). A su call that timed out latches
    state['root_timeout'] and ends the loop at once: no further observation may share the root output file with a
    root shell that may still run (the block then fails closed between requests)."""
    state.update(pin_sampler())
    k = 0
    while not stop.wait(max(0.0, t0 + k * SAMPLE_PERIOD_S - time.monotonic())):
        state['attempts'] += 1
        cpu, began = time.thread_time(), time.monotonic()
        x = observe(thermal_log, pid)
        x.update(slot=k, t_sched=k * SAMPLE_PERIOD_S, t=round(began - t0, 3), t_end=round(time.monotonic() - t0, 3),
                 cpu_s=round(time.thread_time() - cpu, 4))
        if len(samples) < MAX_SAMPLES:
            samples.append(x)
        else:
            state['dropped'] += 1
        if timed_out(x):
            state['root_timeout'] = True
            return
        nxt = max(k + 1, math.ceil((time.monotonic() - t0) / SAMPLE_PERIOD_S))
        state['missed'] += nxt - k - 1
        k = nxt


def safety_stop(x):
    """Why observation x allows no (further) inference, else None: its safety reads failed (su, battery or dumpsys
    return code not exactly 0, or Android status / battery temperature not a finite number: fail closed), or it is
    at or over a safety stop (Android status >= STATUS_STOP, battery >= BATTERY_MAX_C). Used before every block and
    before every request."""
    st, bat = x.get('status'), x.get('battery_c')
    if not (num(st) and num(bat) and all(zero(x.get(k)) for k in ('root_rc', 'battery_rc', 'dump_rc'))):
        return (f'Android status {st!r}, battery {bat!r} degC, rc su/battery/dumpsys {x.get("root_rc")!r}/'
                f'{x.get("battery_rc")!r}/{x.get("dump_rc")!r}: safety reads failed, fail closed')
    if st >= STATUS_STOP or bat >= repro.BATTERY_MAX_C:
        return f'safety stop: Android status {st}, battery {bat} degC'
    return None


def telemetry_stop(sampler, state, samples, now_s):
    """Checked before the warm-up and before every request (now_s = block time), in memory only (no sensor read): why
    no further request may start, else None. Fail closed: the sampler failed, stopped or latched a su timeout; no
    sample yet, or the latest one begun more than SAFETY_MAX_AGE_S ago (stuck or late: the safety evidence is stale);
    else safety_stop of the latest sample (failed safety reads, or a safety stop)."""
    if 'error' in state or state.get('root_timeout') or not sampler.is_alive():
        why = state.get('error') or ('a su call timed out' if state.get('root_timeout') else 'sampler stopped')
        return f'telemetry failed ({why}): fail closed'
    x = samples[-1] if samples else {}
    if now_s - (x['t'] if num(x.get('t')) else 0.0) > SAFETY_MAX_AGE_S:
        return f'no safety sample begun in the last {SAFETY_MAX_AGE_S} s (stuck or late telemetry): fail closed'
    if not x:
        return None  # the first sample (slot 0, at the server start) is still running, within SAFETY_MAX_AGE_S
    return safety_stop(x)


# ---------------------------------------------------------------- one block

def run_block(seq, rnd, length, mode, s, ctx, out, emit, thermal_log, smoke):
    """Start the server, check, warm up, prime, ask the round's questions, stop it. Returns the block record (main adds
    the gate and the pre/postflight readings)."""
    qs = question_order(s['questions'], rnd, smoke)
    rows, end_reason, samples, stop_sampling, sampler = [], 'all questions done', [], threading.Event(), None
    rec, why, status_start, status_end, hwm, load_s = None, [], {}, {}, None, None
    sampler_state = {'attempts': 0, 'dropped': 0, 'missed': 0}
    with open(Path(out) / f'server_block{seq:02d}_L{length}_{mode}.log', 'w') as log:
        t0 = time.monotonic()
        proc = cm.spawn(command(ctx), log)

        def sampling():  # an exception ends the thread: recorded (the block is then INVALID) and re-raised
            try:
                sampler_loop(t0, thermal_log, proc.pid, stop_sampling, samples, sampler_state)
            except Exception as e:
                sampler_state['error'] = f'{type(e).__name__}: {e}'
                raise
            finally:
                sampler_state['ended_s'] = round(time.monotonic() - t0, 3)
        try:  # sampling covers the load; it stops when the server stop begins
            sampler = threading.Thread(target=sampling, daemon=True)
            sampler.start()
            while not cr.healthy():
                if proc.poll() is not None or time.monotonic() - t0 > LOAD_TIMEOUT_S:
                    raise RuntimeError(f'llama-server not healthy (exit {proc.poll()}); see {log.name}')
                time.sleep(0.05)
            load_s = time.monotonic() - t0
            status_start = proc_mem(proc.pid)
            rec, why = checks(s, length, qs, ctx)
            if why:
                end_reason = 'refused: ' + '; '.join(why)
            elif stop_why := telemetry_stop(sampler, sampler_state, samples, time.monotonic() - t0):
                end_reason = f'{stop_why}: block ended before the warm-up'
            else:
                w0 = time.perf_counter()
                warm = cr.post('/completion', {**payload(WARMUP, False), 'n_predict': 4}, timeout=REQUEST_TIMEOUT_S)
                rec['warmup'] = {'wall_s': round(time.perf_counter() - w0, 3), 'timings': warm.get('timings')}
                for i, q in enumerate([s['prime']] + qs):  # position 0 = the unscored prime
                    if stop_why := telemetry_stop(sampler, sampler_state, samples, time.monotonic() - t0):
                        end_reason = f'{stop_why}: block ended before {q["id"]}'
                        break
                    cls, p = request_class(mode, i == 0), prompt_for(s, length, q)
                    want = expected(s, length, mode, rnd, i)  # == the server's own token LCPs (checks)
                    row = {'block': seq, 'round': rnd, 'length': length, 'mode': mode, 'class': cls, 'position': i,
                           'scored': i > 0, **want, 't_start': round(time.monotonic() - t0, 3), 'utc': cr.utc()}
                    c0 = time.perf_counter()
                    try:
                        data = cr.post('/completion', payload(p, mode == 'CACHED'), timeout=REQUEST_TIMEOUT_S)
                        wall_s, t_end = round(time.perf_counter() - c0, 3), round(time.monotonic() - t0, 3)
                    except (OSError, http.client.HTTPException, ValueError) as e:
                        row.update(error=f'{type(e).__name__}: {e}', problems=['request failed'], valid=False,
                                   outcome='ERROR' if i > 0 else None, t_end=round(time.monotonic() - t0, 3))
                        rows.append(row)
                        emit(row)
                        end_reason = f'{q["id"]} failed: {e!r}'
                        break
                    tm = data.get('timings', {})
                    present = length > 0 and q.get('kind') == 'present'
                    graded = scored_fields(s, q, data.get('content'), present) if i > 0 else {'outcome': None}
                    problems = row_problems(cls, data, want['tokens_expected'], want['expected_cache_n'],
                                            want['shared_prefix_tokens'])
                    row.update(wall_s=wall_s, t_end=t_end,
                               tokens_evaluated=data.get('tokens_evaluated'), tokens_cached=data.get('tokens_cached'),
                               cache_n=tm.get('cache_n'), prompt_n=tm.get('prompt_n'), prompt_ms=tm.get('prompt_ms'),
                               ttft_proxy_s=tm['prompt_ms'] / 1000 if num(tm.get('prompt_ms')) else None,
                               predicted_n=tm.get('predicted_n'), predicted_ms=tm.get('predicted_ms'),
                               gen_tok_s=tm.get('predicted_per_second'), stop_type=data.get('stop_type'),
                               truncated=data.get('truncated'), reply=data.get('content'), present=present,
                               **graded, problems=problems, valid=not problems, timings=tm)
                    rows.append(row)
                    emit(row)
                    hwm = proc_mem(proc.pid).get('VmHWM', hwm)
                    try:
                        cr.check_cores(f'after {q["id"]}')
                    except cr.CoresLost as e:
                        end_reason = f'{e}: block ended after {q["id"]}'
                        break
            status_end = proc_mem(proc.pid)
            if 'VmHWM' not in status_end and hwm:
                status_end.update(VmHWM=hwm, VmHWM_source='last completed request')
        finally:
            sampling_end = round(time.monotonic() - t0, 3)
            stop_sampling.set()
            cm.stop(proc)
            server_s = time.monotonic() - t0
            if sampler:
                sampler.join(timeout=JOIN_S)  # bounded: a stuck su call cannot hold the launch
                sampler_state['alive_after_join'] = sampler.is_alive()
    return {'block': seq, 'round': rnd, 'length': length, 'mode': mode, 'ctx': ctx, 'questions': [q['id'] for q in qs],
            'prime': s['prime']['id'], 'end_reason': end_reason, 'refused': bool(why), 'checks': rec,
            'load_s': round(load_s, 3) if load_s is not None else None, 'server_s': round(server_s, 3),
            'mono_t0': round(t0, 3), 'sampling_end_s': sampling_end, 'sample_period_s': SAMPLE_PERIOD_S,
            'sampler': dict(sampler_state), 'server_pid': proc.pid, 'server_start': status_start,
            'server_end': status_end, 'runner_end': proc_mem('self'), 'n_rows': len(rows), 'samples': list(samples)}


# ---------------------------------------------------------------- validity (one contract: runner and report)

def cap_of(e, policy):
    """The policy's int scaling_max_freq in an observation; None when missing or not an int."""
    c = e.get('cpufreq')
    v = c[policy].get('max') if isinstance(c, dict) and isinstance(c.get(policy), dict) else None
    return v if type(v) is int else None


def env_readings(b):
    """The observations behind a block's environment, in time order: preflight, samples, postflight (non-objects left
    out)."""
    xs = b.get('samples') if isinstance(b.get('samples'), list) else []
    return [x for x in [b.get('start'), *xs, b.get('end')] if isinstance(x, dict)]


def utc_of(v):
    """An aware ISO time string as a datetime; None when missing, malformed or without a time zone."""
    try:
        d = datetime.fromisoformat(v) if isinstance(v, str) else None
    except ValueError:
        return None
    return d if d is not None and d.tzinfo is not None else None


def utc_ok(v):
    return utc_of(v) is not None


def log_age(e):
    """Problems of the thermal-log freshness evidence of one observation: its last line's stamp 'at' (thermal_tail's
    form), the recorded age within thermal_tail's limits, and that age consistent with the observation's own UTC
    (re-derived, never trusted)."""
    try:
        at = datetime.strptime(e.get('at'), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return [f'thermal log stamp {e.get("at")!r} missing or malformed']
    age, now = e.get('thermal_age_s'), utc_of(e.get('utc'))
    if not num(age, -2, cr.THERMAL_MAX_AGE_S):
        return [f'thermal log age {age!r} s (missing, or outside -2..{cr.THERMAL_MAX_AGE_S}: stale)']
    if now is None or abs((now - at).total_seconds() - age) > 1 + CLOCK_TOL_S:  # UTC and the stamp: whole seconds
        return [f'thermal log stamp {e.get("at")} and age {age} s disagree with the reading time {e.get("utc")!r}']
    return []


def observation_problems(e, label, caps0, pid=None):
    """Field-level validity of one observation (a pre/postflight reading, or with pid a sample of that server). Every
    field is required: a missing, failed (non-zero su, battery, dumpsys or smaps return code, parse error),
    non-finite, mistyped (a bool is not a number), out-of-range or stale (thermal log) value is a problem, never a
    healthy reading; diagnostics stay in the observation."""
    why = []
    w = lambda text: why.append(f'{label}: {text}')
    if not (num(e.get('mono'), 0) and utc_ok(e.get('utc'))):
        w('timestamp missing or invalid')
    for k, what in (('root_rc', 'su'), ('battery_rc', 'battery read'), ('dump_rc', 'dumpsys thermalservice')):
        if not zero(e.get(k)):
            w(f'{what} rc {e.get(k)!r}')
    pct, bat_c = e.get('battery_pct'), e.get('battery_c')
    if not (type(pct) is int and 0 <= pct <= 100):
        w(f'battery % {pct!r} (missing or out of range)')
    if not (num(bat_c, BATTERY_MIN_C) and bat_c < repro.BATTERY_MAX_C):
        w(f'battery {bat_c!r} degC (missing, or outside {BATTERY_MIN_C}..<{repro.BATTERY_MAX_C})')
    if e.get('battery_status') != 'Discharging':
        w(f'battery status {e.get("battery_status")!r}')
    if not zero(e.get('status')):
        w(f'Android status {e.get("status")!r}')
    if not (num(e.get('skin')) and SKIN_RANGE_C[0] < e['skin'] < SKIN_RANGE_C[1]):
        w(f'VIRTUAL-SKIN {e.get("skin")!r} (missing or out of range)')
    if 'thermal_error' in e:
        w(f'thermal log: {e["thermal_error"]}')
    for text in log_age(e):
        w(text)
    for p in SERVER_POLICIES:
        c = cap_of(e, p)
        if not (c is not None and c > 0):
            w(f'{p} cap {c!r} (missing or not a positive int)')
        elif caps0.get(p) is not None and c != caps0[p]:
            w(f'{p} cap {caps0[p]} -> {c}')
    if pid is not None:
        t, te = e.get('t'), e.get('t_end')
        if not (num(t, 0) and num(te) and te >= t):
            w(f'sample times {t!r}..{te!r} missing or invalid')
        if not num(e.get('battery_w'), -BATTERY_W_MAX, BATTERY_W_MAX):
            w(f'battery power {e.get("battery_w")!r} W (missing or out of range)')
        if 'meminfo_error' in e or not num(e.get('mem_available_mib'), 0, MEM_MIB_MAX):
            w(f'MemAvailable {e.get("mem_available_mib")!r} (missing or out of range)')
        if not num(e.get('swap_used_mib'), 0, MEM_MIB_MAX):
            w(f'swap used {e.get("swap_used_mib")!r} MiB (missing or out of range)')
        pss = e.get('pss_kb')
        if not (zero(e.get('smaps_rc')) and type(e.get('pss_pid')) is int and e['pss_pid'] == pid
                and type(pss) is int and 0 < pss <= PSS_KB_MAX):
            w(f'llama-server PSS missing or invalid (pid {pid}: rc {e.get("smaps_rc")!r}, pss {pss!r})')
    return why


def sample_pid(b):
    """The server pid samples are checked against; a missing one is -1, so the samples are still checked as samples
    (times and all) and every PSS check fails."""
    return b.get('server_pid') if b.get('server_pid') is not None else -1


def gaps_over(ts, limit):
    """Gaps between consecutive times that exceed limit or run backwards."""
    return [round(y - x, 1) for x, y in zip(ts, ts[1:]) if y - x > limit or y < x]


def checks_problems(b, rows, ctx, s, smoke):
    """The block's recorded no-generation checks and warm-up, against the plan and the frozen set: the block's
    questions are exactly the planned questions of its round (question_order with the launch's smoke flag); a
    GGUF-template result (exactly True) and a server token count (an int equal to the frozen count) for exactly the
    prime and those questions, and for every recorded request's question; server n_ctx = ctx (an int); a chat
    template hash; the warm-up recorded (uncached, 1..4 generated tokens, client wall time)."""
    c = b.get('checks') if isinstance(b.get('checks'), dict) else {}
    L, why = s['lengths'].get(str(b.get('length'))), []
    qs = b.get('questions') if isinstance(b.get('questions'), list) else []
    order = [q['id'] for q in question_order(s['questions'], b['round'], smoke)] if type(b.get('round')) is int else []
    if not (L and qs == order):
        return [f'block questions {qs!r:.80} are not the planned questions of round {b.get("round")!r} '
                f'(smoke {smoke}), or its memory length {b.get("length")!r} is not in the set']
    want = {s['prime']['id']: L['prime_prompt']['tokens'], **{q: L['prompts'][q]['tokens'] for q in qs}}
    tpl, tok = c.get('template'), c.get('tokens')
    if not (isinstance(tpl, dict) and set(tpl) == set(want) and all(v is True for v in tpl.values())):
        why.append(f'template checks {tpl!r:.120} are not a passed check (true) for each of {sorted(want)}')
    if not (isinstance(tok, dict) and tok == want and all(type(v) is int for v in tok.values())):
        why.append(f'token checks {tok!r:.120} are not the frozen counts {want} (ints)')
    if unchecked := sorted({r.get('question') for r in rows} - set(want), key=str):
        why.append(f'requests {unchecked} have no recorded no-generation check')
    if not (type(c.get('n_ctx')) is int and c['n_ctx'] == ctx):
        why.append(f'server n_ctx {c.get("n_ctx")!r} is not the launch ctx {ctx!r}')
    if not (type(c.get('chat_template_sha256')) is str and HEX64.fullmatch(c['chat_template_sha256'])):
        why.append('chat template hash missing')
    w = c.get('warmup') if isinstance(c.get('warmup'), dict) else {}
    tm = w.get('timings') if isinstance(w.get('timings'), dict) else {}
    if not (num(w.get('wall_s'), 0) and zero(tm.get('cache_n')) and type(tm.get('prompt_n')) is int and
            tm['prompt_n'] > 0 and type(tm.get('predicted_n')) is int and 1 <= tm['predicted_n'] <= 4):
        why.append(f'warm-up evidence missing or not an uncached 1..4-token request: {w!r:.120}')
    return why


def validate_block(b, rows, ctx, s, smoke):
    """THE validity contract of the runner and the report: every reason the block's recorded evidence is not a valid
    measurement, recomputed from that evidence each time (a stored label is never read). rows = the block's requests
    with recomputed validity (recheck_row); ctx = the launch's context size (run.json / --ctx), which the block's own
    ctx and the server's n_ctx must equal; s = the frozen set (task view) and smoke = the launch's smoke flag, against
    which the block's planned questions and no-generation checks are verified (checks_problems). Reasons are
    {'code', 'detail'}; none = VALID. Tolerates any JSON shape."""
    why = []
    add = lambda code, detail: why.append({'code': code, 'detail': detail})
    obj = lambda k: b.get(k) if isinstance(b.get(k), dict) else {}
    gate, start, end, sampler = obj('gate'), obj('start'), obj('end'), obj('sampler')
    if gate.get('warm_start') is not False:
        add('WARM_START', 'warm start' if gate.get('warm_start') is True else 'gate record missing')
    if b.get('end_reason') != 'all questions done':
        add('ENDED_EARLY', str(b.get('end_reason'))[:200])
    caps0 = {p: cap_of(start, p) for p in SERVER_POLICIES}
    for label, e in (('start', start), ('end', end)):
        for text in observation_problems(e, label, caps0):
            add('READING_INVALID', text)
    t0, server_s, send = b.get('mono_t0'), b.get('server_s'), b.get('sampling_end_s')
    times = num(t0, 0) and num(server_s, 0) and num(send, 0) and send <= server_s + ROUND_S
    if not times:
        add('TIMES_INVALID', f'block times mono_t0 {t0!r}, server_s {server_s!r}, sampling_end_s {send!r}')
    else:
        a, z = start.get('mono'), end.get('mono')
        if not (num(a) and t0 - ENV_MAX_GAP_S <= a <= t0):
            add('READING_ORDER', f'preflight reading {a!r} not within {ENV_MAX_GAP_S} s before the server start')
        if not (num(z) and t0 + server_s - ROUND_S <= z <= t0 + server_s + ENV_MAX_GAP_S):
            add('READING_ORDER', f'postflight reading {z!r} not within {ENV_MAX_GAP_S} s after the server stop')
        ends = [r['t_end'] for r in rows if num(r.get('t_end'))]
        if ends and max(ends) > send + ROUND_S:
            add('TIMES_INVALID', f'sampling ended at {send} s, before the last request ended ({max(ends)} s)')
        timed = sorted((r['position'], r['t_start'], r['t_end']) for r in rows
                       if type(r.get('position')) is int and num(r.get('t_start')) and num(r.get('t_end')))
        load = b.get('load_s')
        if timed and ((num(load) and timed[0][1] < load - ROUND_S) or any(
                y[1] < x[2] - ROUND_S for x, y in zip(timed, timed[1:]))):
            add('TIMES_INVALID', 'requests overlap, run out of position order or begin before the model load ended')
    samples = b.get('samples') if isinstance(b.get('samples'), list) else []
    pid, ok, ended, prev_end = sample_pid(b), [], sampler.get('ended_s'), None
    if not (type(b.get('server_pid')) is int and b['server_pid'] > 0):
        add('TIMES_INVALID', f'server pid {b.get("server_pid")!r}')
    for x in samples:  # in recorded order: the sampler is serial, so each begins after the previous one ended
        bad = observation_problems(x, f'sample t {x.get("t")!r}', caps0, pid) if isinstance(x, dict) else \
            [f'sample is not an object ({type(x).__name__})']
        if not bad and times and abs(x['mono'] - (t0 + x['t'])) > CLOCK_TOL_S:
            bad = [f'sample t {x["t"]}: mono {x["mono"]} is not mono_t0 + t ({t0} + {x["t"]})']
        if not bad and x['t_end'] - x['t'] > JOIN_S:
            bad = [f'sample t {x["t"]}: took {x["t_end"] - x["t"]:.1f} s (an observation is at most {JOIN_S} s)']
        if not bad and num(ended) and x['t_end'] > ended + ROUND_S:
            bad = [f'sample t {x["t"]}: ended at {x["t_end"]} s, after the sampler ended ({ended} s)']
        if not bad and prev_end is not None and x['t'] < prev_end - ROUND_S:
            bad = [f'sample t {x["t"]}: begun before the previous sample ended ({prev_end} s; the sampler is serial)']
        if isinstance(x, dict) and num(x.get('t_end')):
            prev_end = x['t_end']
        if bad:
            add('SAMPLE_INVALID', '; '.join(bad))
        elif times and x['t'] <= send:  # a sample begun after the sampling end (stop race) covers nothing
            ok.append(x['t'])
    if not ok:
        add('SAMPLES_EMPTY', f'no valid samples within the sampling interval ({len(samples)} recorded)')
    elif gaps := gaps_over([0.0] + ok + [send], SAMPLE_MAX_GAP_S):
        add('SAMPLE_GAP', f'valid samples leave gaps {gaps} s (> {SAMPLE_MAX_GAP_S} s or out of order; block start to '
                          f'first sample, between samples, last sample to sampling end)')
    readings = [e for e in env_readings(b) if num(e.get('mono')) and utc_of(e.get('utc'))]
    if readings and (skew := [e['utc'] for e in readings if abs(
            (utc_of(e['utc']) - utc_of(readings[0]['utc'])).total_seconds() - (e['mono'] - readings[0]['mono']))
            > 1 + CLOCK_TOL_S]):
        add('TIMES_INVALID', f'reading UTC times {skew[:3]} disagree with their monotonic times (clock change?)')
    attempts, dropped = sampler.get('attempts'), sampler.get('dropped')
    if not (type(attempts) is int and type(dropped) is int and attempts == len(samples) + dropped):
        add('SAMPLER_STATE', f'{attempts!r} sample attempts != {len(samples)} recorded + {dropped!r} dropped '
                             '(an observation raised or never returned, or the count is missing)')
    if 'error' in sampler:
        add('SAMPLER_FAILED', str(sampler['error'])[:DIAG_CHARS])
    if sampler.get('alive_after_join') is not False:
        add('SAMPLER_STATE', 'sampler did not stop (or its state is missing)')
    if not (num(sampler.get('ended_s')) and num(send) and sampler['ended_s'] >= send):
        add('SAMPLER_STATE', f'sampler ended at {sampler.get("ended_s")!r} s, before sampling end {send!r} s '
                             '(unexpected termination)')
    elif times and ended > server_s + JOIN_S + ROUND_S:
        add('SAMPLER_STATE', f'sampler ended at {ended} s, after its bounded join ({server_s} + {JOIN_S} s)')
    elif times and not (num(end.get('mono')) and end['mono'] >= t0 + ended - ROUND_S):
        add('READING_ORDER', f'postflight reading {end.get("mono")!r} not after the sampler ended ({t0} + {ended})')
    if not zero(sampler.get('dropped')):
        add('SAMPLES_LOST', f'{sampler.get("dropped")!r} samples dropped (buffer of {MAX_SAMPLES} full) or count '
                            'missing')
    load = b.get('load_s')
    if not num(load, 0, LOAD_TIMEOUT_S):
        add('TIMES_INVALID', f'load_s {load!r} not a number in 0..{LOAD_TIMEOUT_S}')
    for text in checks_problems(b, rows, ctx, s, smoke):
        add('CHECKS_FAILED', text)
    if b.get('ctx') != ctx:
        add('CHECKS_FAILED', f'block ctx {b.get("ctx")!r} is not the launch ctx {ctx!r}')
    if mixed := sorted({r.get('question') for r in rows if (r.get('round'), r.get('length'), r.get('mode')) !=
                        (b.get('round'), b.get('length'), b.get('mode'))}, key=str):
        add('REQUESTS_INVALID', f'requests {mixed} record another round, memory length or mode than their block')
    ag = obj('agents')
    if not (ag.get('before') == '' and ag.get('after') == ''):
        add('AGENTS_PRESENT', f'agent processes before/after the block (or not recorded): {ag!r:.160}')
    for when in ('server_start', 'server_end'):
        if obj(when).get('Cpus_allowed_list') != '4-7':
            add('SERVER_CORES', f'{when} Cpus_allowed_list {obj(when).get("Cpus_allowed_list")!r}')
    if bad := [r.get('question') for r in rows if r.get('valid') is not True]:
        add('REQUESTS_INVALID', f'invalid requests {bad}')
    return why


def reasons_text(why):
    return '; '.join(f"{w['code']} {w['detail']}" if isinstance(w, dict) else str(w) for w in why)


# ---------------------------------------------------------------- artifacts (read strictly and defensively)

NUM, OPT = (int, float), (int, float, type(None))
RUN_SCHEMA = {'task': str, 'exp': str, 'ctx': int, 'rounds': int, 'smoke': bool, 'set_sha256': str, 'prime': dict,
              'command': list, 'decoding': dict, 'warmup': str, 'model_sha256': dict, 'pinned_sha256': dict,
              'build_sha256': dict, 'code_sha256': dict, 'thresholds': dict, 'telemetry': dict}
HEX64 = re.compile(r'[0-9a-f]{64}')
MATRIX_SCHEMA = {'block': int, 'round': int, 'length': int, 'mode': str, 'questions': list}
BLOCK_SCHEMA = {'block': int, 'round': int, 'length': int, 'mode': str, 'questions': list, 'end_reason': str,
                'problems': list, 'gate': dict, 'start': dict, 'end': dict, 'samples': list, 'sampler': dict,
                'load_s': OPT}
ROW_SCHEMA = {'block': int, 'round': int, 'length': int, 'mode': str, 'class': str, 'question': str, 'scored': bool,
              'position': int, 'valid': bool, 'problems': list}
RESULT_SCHEMA = {'end_reason': str, 'skipped': list, 'agents_end': str, 'port_free_end': bool}
METRICS = {'prompt_ms': 'prompt_ms', 'prompt_n': 'prompt_n', 'cache_n': 'cache_n', 'predicted_n': 'predicted_n',
           'predicted_ms': 'predicted_ms', 'gen_tok_s': 'predicted_per_second'}  # row field -> server timings key


def typed(v, t):
    """isinstance, except that a bool is never an int or a float."""
    ts = t if isinstance(t, tuple) else (t,)
    return isinstance(v, ts) and not (isinstance(v, bool) and bool not in ts)


def shape(rec, schema):
    """Missing or mistyped required keys of one artifact record."""
    if not isinstance(rec, dict):
        return [f'not a JSON object ({type(rec).__name__})']
    return [f'{k} missing' if k not in rec else f'{k} is {type(rec[k]).__name__}'
            for k, t in schema.items() if k not in rec or not typed(rec[k], t)]


def configuration():
    """What main records in run.json about this code's measurement configuration (decoding, warm-up, thresholds,
    telemetry), as JSON reads it back; a run must carry exactly these to be this benchmark."""
    return json.loads(json.dumps({
        'decoding': DECODING, 'warmup': WARMUP,
        'thresholds': {'min_mem_available_mib': MIN_MEM_AVAILABLE_MIB, 'status_stop': STATUS_STOP,
                       'battery_max_c': repro.BATTERY_MAX_C, 'request_timeout_s': REQUEST_TIMEOUT_S},
        'telemetry': {'sample_period_s': SAMPLE_PERIOD_S, 'sample_max_gap_s': SAMPLE_MAX_GAP_S,
                      'max_samples': MAX_SAMPLES, 'root_timeout_s': ROOT_TIMEOUT_S,
                      'sampler_cpus': list(SAMPLER_CPUS), 'sampler_nice': SAMPLER_NICE}}))


def run_problems(run):
    """Nested run.json fields the report relies on (after shape(run, RUN_SCHEMA)), including the provenance every
    comparison assumes: decoding, warm-up, thresholds and telemetry equal to this code's (configuration()); the model
    and pinned hashes equal to PINNED; the build hashes include the pinned server and CPU library; code hashes
    present. Missing or different provenance makes run.json INVALID (not a measurement of this benchmark)."""
    why = []
    if run['task'] not in ('recall', 'freetext') or run['task'] != EXPERIMENTS.get(run['exp'], {}).get('task'):
        why.append(f'task {run["task"]!r} (not the task of exp {run["exp"]!r})')
    why += [f'{k} differs from this code\'s configuration' for k, v in configuration().items() if run[k] != v]
    if run['model_sha256'] != {MODEL: PINNED[MODEL]} or run['pinned_sha256'] != PINNED:
        why.append('model_sha256 / pinned_sha256 are not the pinned model, server and CPU library')
    bs = run['build_sha256']
    if not (all(type(v) is str and HEX64.fullmatch(v) for v in bs.values()) and
            bs.get(Path(SERVER).name) == PINNED[SERVER] and bs.get(Path(repro.LIBGGML_CPU).name) == PINNED[
                repro.LIBGGML_CPU]):
        why.append('build_sha256 missing, malformed or without the pinned llama-server and libggml-cpu.so')
    if set(run['code_sha256']) != {str(p) for p in code_files()} or not all(
            type(v) is str and HEX64.fullmatch(v) for v in run['code_sha256'].values()):
        why.append('code_sha256 is not one well-formed hash for each of the benchmark\'s source files')
    if not isinstance(run['prime'].get('id'), str):
        why.append('prime.id missing')
    if not run['command'] or not all(isinstance(x, str) for x in run['command']):
        why.append('command malformed')
    elif run['command'] != command(run['ctx']):
        why.append(f'command is not this benchmark\'s server command at ctx {run["ctx"]} (model, binary, threads, '
                   'cache flags, affinity, --ctx-size; no drafter or projector)')
    if run['exp'] not in EXPERIMENTS or run['ctx'] not in EXPERIMENTS[run['exp']]['ctx']:
        why.append(f'exp {run["exp"]!r} at ctx {run["ctx"]} is not a configured experiment')
    if not (run['rounds'] == 1 if run['smoke'] else 2 <= run['rounds'] <= 64 and run['rounds'] % 2 == 0):
        why.append(f'rounds {run["rounds"]} not allowed (smoke: 1; else even, 2..64)')
    if run['set_sha256'] != SET_SHA256:
        why.append('set_sha256 is not the frozen set of this code: outcomes cannot be recomputed')
    return why


def read_json(path):
    """(object, issue); a missing file is (None, None): the caller decides what absence means."""
    if not path.exists():
        return None, None
    text = path.read_text()
    try:
        return strict_loads(text), None
    except ValueError as e:
        return None, f'{path.name} unreadable ({e}; truncated or non-finite?): {text[:120]!r}'


def read_records(path, schema):
    """(records, issues) of a JSONL artifact. A torn last line (a write cut off by a kill), a malformed line (or one
    with NaN/Infinity) or a record missing required keys is listed with its raw text and left out; nothing is filled
    in."""
    good, issues = [], []
    lines = path.read_text().splitlines() if path.exists() else []
    for i, line in enumerate(lines, 1):
        try:
            rec = strict_loads(line)
        except ValueError:
            kind = 'torn last line (interrupted write)' if i == len(lines) else 'not JSON (or a non-finite number)'
            issues.append(f'{path.name} line {i}: {kind}: {line[:80]!r}')
            continue
        if bad := shape(rec, schema):
            issues.append(f'{path.name} line {i} left out: {", ".join(bad)}')
        else:
            good.append(rec)
    return good, issues


def recheck_row(r, s):
    """The request's validity and recall outcome recomputed from its recorded evidence (server timings, token counts,
    reply) with the runner's own functions; the stored labels are kept under 'stored' for audit and never used. The
    question and the expected token counts (prompt tokens, shared prefix, cache_n) come from the frozen set and the
    plan (expected): a row's own copies are only compared. The statistics' fields are taken from the server timings
    (a top-level copy that differs makes the row invalid)."""
    q = {x['id']: x for x in s['questions'] + [s['prime']]}.get(r['question'])
    out = {**r, 'stored': {k: r.get(k) for k in ('valid', 'problems', 'outcome', 'format', 'answer')}}
    why, cls = [], request_class(r['mode'], r['position'] == 0)
    want = expected(s, r['length'], r['mode'], r['round'], r['position'])
    if r['mode'] not in ('FRESH', 'CACHED') or want is None or want['question'] != r['question'] or \
            r['scored'] != (r['position'] > 0):
        why.append('mode, memory length, question, position and scored flag disagree with the plan')
    elif any(r.get(k) != v for k, v in want.items()):
        why.append('recorded expected token counts differ from the frozen set')
    if r['class'] != cls:
        why.append(f'class {r["class"]!r} != {cls!r} (mode, position)')
    if not utc_ok(r.get('utc')):
        why.append(f'request utc {r.get("utc")!r} missing or invalid')
    if 'error' in r:
        why.append('request failed')
    else:
        tm = r.get('timings') if isinstance(r.get('timings'), dict) else {}
        if want is not None:
            why += row_problems(cls, {'tokens_evaluated': r.get('tokens_evaluated'), 'truncated': r.get('truncated'),
                                      'stop_type': r.get('stop_type'), 'timings': tm}, want['tokens_expected'],
                                want['expected_cache_n'], want['shared_prefix_tokens'])
        if not num(r.get('wall_s'), 0):
            why.append(f'wall_s {r.get("wall_s")!r} not a finite number >= 0')
        if not (num(r.get('t_start'), 0) and num(r.get('t_end')) and r['t_end'] >= r['t_start']):
            why.append('request times missing or invalid')
        elif num(r.get('wall_s')) and r['wall_s'] > r['t_end'] - r['t_start'] + ROUND_S:
            why.append(f'wall_s {r["wall_s"]} longer than the request interval {r["t_start"]}..{r["t_end"]}')
        server = [tm.get('prompt_ms'), tm.get('predicted_ms')]
        if all(num(v, 0) for v in server) and num(r.get('wall_s')) and sum(server) / 1000 > r['wall_s'] + ROUND_S:
            why.append(f'server prompt + generation {sum(server) / 1000:.3f} s longer than the client wall time')
        metrics = {k: tm.get(v) for k, v in METRICS.items()}
        if any(k in r and r[k] != v for k, v in metrics.items()):
            why.append('row fields differ from the server timings')
        out.update(metrics)
    if not r['scored']:
        graded = {'outcome': None}
    elif 'error' in r or q is None:
        graded = {'outcome': 'ERROR'}
    else:
        graded = scored_fields(s, q, r.get('reply'), r['length'] > 0 and q.get('kind') == 'present')
    out.update(graded, problems=why, valid=not why)
    return out


def artifact_problems(folder):
    """When the run wrote artifacts.sha256 (its last step), every file it lists must still have that hash: a file
    changed or removed after the run is listed (the run is then INCOMPLETE). Absent (an interrupted or running
    launch, already INCOMPLETE): nothing to check."""
    p = Path(folder) / 'artifacts.sha256'
    if not p.is_file():
        return []
    why = []
    for line in p.read_text().splitlines():
        want, _, name = line.partition('  ')
        f = Path(folder) / name
        if not (HEX64.fullmatch(want) and name and '/' not in name and f.is_file()):
            why.append(f'artifacts.sha256: {line[:80]!r} malformed or its file missing')
        elif hashlib.sha256(f.read_bytes()).hexdigest() != want:
            why.append(f'artifacts.sha256: {name} changed after the run')
    return why


def load_run(folder):
    """A run folder as recorded, with every validity label RECOMPUTED: requests by recheck_row, blocks by
    validate_block (stored labels kept for audit: 'stored', 'stored_problems'; a disagreement is listed). run = None
    when run.json is missing or unusable (the run is then INVALID); issues = recognized incomplete or malformed
    artifacts; orphans = requests whose block has no usable record (a block interrupted mid-execution), kept out of
    every aggregate."""
    folder = Path(folder)
    run, issue = read_json(folder / 'run.json')
    issues = [issue or 'run.json missing'] if run is None else []
    if run is not None and (bad := shape(run, RUN_SCHEMA) or run_problems(run)):
        issues.append('run.json: ' + ', '.join(bad))
        run = None
    if run is None:
        return SimpleNamespace(run=None, blocks=[], rows=[], orphans=[], res=None, issues=issues)
    if 'matrix' in run:
        if not isinstance(run['matrix'], list) or any(shape(m, MATRIX_SCHEMA) or not all(
                isinstance(q, str) for q in m['questions']) for m in run['matrix']):
            issues.append('run.json matrix malformed: completeness cannot be verified')
            run = {k: v for k, v in run.items() if k != 'matrix'}
    s = task_view(load_set(), run['task'])
    if run['prime'] != s['prime']:
        issues.append('run.json prime is not the frozen prime: run INVALID')
        return SimpleNamespace(run=None, blocks=[], rows=[], orphans=[], res=None, issues=issues)
    if 'matrix' in run and run['matrix'] != matrix(s, schedule(run['exp'], run['rounds'], run['smoke']), run['smoke']):
        issues.append('run.json matrix differs from the configured schedule (exp, rounds, smoke, frozen questions): '
                      'the planned work is not the configured experiment')
    blocks, more = read_records(folder / 'blocks.jsonl', BLOCK_SCHEMA)
    issues += more
    seen = [b['block'] for b in blocks]
    if dup := sorted({x for x in seen if seen.count(x) > 1}):
        issues.append(f'blocks {dup} recorded more than once: left out')
        blocks = [b for b in blocks if b['block'] not in dup]
    rows, more = read_records(folder / 'requests.jsonl', ROW_SCHEMA)
    issues += more
    keys = [(r['block'], r['position']) for r in rows]
    if dup := sorted({k for k in keys if keys.count(k) > 1}):
        issues.append(f'requests (block, position) {dup} recorded more than once: left out')
        rows = [r for r in rows if (r['block'], r['position']) not in dup]
    rows = [recheck_row(r, s) for r in rows]
    if relabeled := sum((r['stored']['valid'], r['stored']['outcome']) != (r['valid'], r['outcome']) for r in rows):
        issues.append(f'{relabeled} requests whose stored validity or outcome differs from the recomputation '
                      '(stored labels ignored)')
    for b in blocks:
        b['stored_problems'], b['problems'] = b['problems'], validate_block(
            b, [r for r in rows if r['block'] == b['block']], run['ctx'], s, run['smoke'])
    if relabeled := [b['block'] for b in blocks if b['stored_problems'] != b['problems']]:
        issues.append(f'blocks {relabeled}: stored validity differs from the recomputation (stored labels ignored)')
    ids = {b['block'] for b in blocks}
    orphans = [r for r in rows if r['block'] not in ids]
    issues += artifact_problems(folder)
    res, issue = read_json(folder / 'result.json')
    if issue:
        issues.append(issue)
    elif res is not None and (bad := shape(res, RESULT_SCHEMA) or [
            f'skipped entry {x!r:.40} is not {{"block": int, ...}}' for x in res['skipped']
            if not (isinstance(x, dict) and type(x.get('block')) is int)][:3]):
        issues.append('result.json partial: ' + ', '.join(bad))
        res = None
    return SimpleNamespace(run=run, blocks=blocks, rows=[r for r in rows if r['block'] in ids], orphans=orphans,
                           res=res, issues=issues)


# ---------------------------------------------------------------- report

def med(xs):
    return statistics.median(xs) if xs else None


def p95(xs):
    """Nearest-rank 95th percentile."""
    xs = sorted(xs)
    return xs[max(0, -(-95 * len(xs) // 100) - 1)] if xs else None


def f(v, spec='.0f'):
    """A number in spec; 'n/a' for None; anything else (malformed evidence) as its short repr, never formatted."""
    return format(v, spec) if num(v) else 'n/a' if v is None else repr(v)[:20]


def ratio(a, b):
    return a / b if a is not None and b else None


def cap(b, policy='policy6'):
    return cap_of(b['start'], policy)


def top_share(start, end, policy='policy6'):
    """repro.top_share on checked input only: None unless both readings carry the policy's time_in_state (int values)
    and the end reading its hw_max."""
    get = lambda e: e['cpufreq'].get(policy) if isinstance(e.get('cpufreq'), dict) else None
    ok = lambda p: isinstance(p, dict) and isinstance(p.get('time_in_state'), dict) and \
        all(type(v) is int for v in p['time_in_state'].values())
    a, b = get(start), get(end)
    return repro.top_share(start, end, policy) if ok(a) and ok(b) and type(b.get('hw_max')) is int else None


def kb(status, key):
    """The kB figure of one /proc/<pid>/status line ('3000000 kB'), None when missing or malformed."""
    v = status.get(key) if isinstance(status, dict) else None
    return int(v.split()[0]) if isinstance(v, str) and v.split() and v.split()[0].isdigit() else None


def resources(b):
    """A block's resource figures, recomputed from its samples (finite values only) and the server's /proc status
    at the end of the block."""
    xs = [x for x in b['samples'] if isinstance(x, dict)]
    vals = lambda k: [x[k] for x in xs if num(x.get(k))]
    return {'peak_pss_kb': max(vals('pss_kb'), default=None), 'min_mem_available_mib': min(vals('mem_available_mib'),
                                                                                          default=None),
            'max_swap_used_mib': max(vals('swap_used_mib'), default=None),
            'mean_w': statistics.mean(vals('battery_w')) if vals('battery_w') else None,
            'vm_hwm_kb': kb(b.get('server_end'), 'VmHWM'), 'vm_rss_end_kb': kb(b.get('server_end'), 'VmRSS')}


def telemetry(b, rows):
    """A block's telemetry cost and timing, recomputed from its raw samples and sampler state: attempts (every
    observation begun, from the sampler's count; None when missing), failures (failed recorded samples plus attempts
    that left no record), dropped and missed slots, actual intervals, sample duration, su wall time, sampler thread
    CPU and root-shell CPU, overlap of sampling with the requests and with the model load, and the CPU list / nice
    actually obtained. A cost total covers only the attempts that measured it: 'unmeasured' counts the attempts
    without that figure (unknown, never zero)."""
    xs = [x for x in b['samples'] if isinstance(x, dict)]
    sp = b['sampler']
    caps0 = {p: cap_of(b['start'], p) for p in SERVER_POLICIES}
    vals = lambda k: [x[k] for x in xs if num(x.get(k))]
    t = vals('t')
    spans = [(x['t'], x['t_end']) for x in xs if num(x.get('t')) and num(x.get('t_end'))]
    reqs = [(r['t_start'], r['t_end']) for r in rows if num(r.get('t_start')) and num(r.get('t_end'))]
    ov = lambda a, z: max(0.0, min(a[1], z[1]) - max(a[0], z[0]))
    iv, dur = [y - x for x, y in zip(t, t[1:])], [e - s for s, e in spans]
    dropped = sp.get('dropped') if type(sp.get('dropped')) is int else None
    attempts = sp.get('attempts') if type(sp.get('attempts')) is int else None
    unrecorded = attempts - len(xs) - (dropped or 0) if attempts is not None else None
    failed = sum(bool(observation_problems(x, '', caps0, sample_pid(b))) for x in xs)
    unmeasured = {k: (attempts - len(vals(k)) if attempts is not None else None) for k in ('root_s', 'cpu_s',
                                                                                          'root_cpu_s')}
    return {'attempts': attempts, 'failed': failed + max(0, unrecorded or 0), 'unrecorded': unrecorded,
            'dropped': dropped, 'missed': sp.get('missed'), 'interval_med': med(iv),
            'interval_max': max(iv, default=None),
            'dur_med': med(dur), 'dur_max': max(dur, default=None), 'su_s': sum(vals('root_s')),
            'su_s_max': max(vals('root_s'), default=None), 'thread_cpu_s': sum(vals('cpu_s')),
            'root_cpu_s': sum(vals('root_cpu_s')), 'unmeasured': unmeasured,
            'overlap_s': sum(ov(s, r) for s in spans for r in reqs),
            'requests_overlapped': sum(any(ov(s, r) > 0 for s in spans) for r in reqs), 'requests': len(reqs),
            'load_overlap_s': sum(ov(s, (0.0, b['load_s'])) for s in spans) if num(b['load_s']) else None,
            'thread': f"{sp.get('cpus')}/{sp.get('nice')}",
            'root_shell': sorted({f"{x.get('root_cpus')}/{x.get('root_nice')}" for x in xs})}


def telemetry_lines(blocks, rows):
    L = ['', f'telemetry (asynchronous sampler, {SAMPLE_PERIOD_S} s period, requested cores {list(SAMPLER_CPUS)} nice '
         f'{SAMPLER_NICE}; its own measured cost, NOT a measurement of its effect on inference):',
         '  seq | samples failed/attempts, dropped, missed slots | interval med/max s | duration med/max s | su s '
         'total/max | CPU s sampler thread + root shells | sampling overlap with requests s (requests touched/all) | '
         'with the model load s | thread cpus/nice obtained | root shell cpus/nice obtained']
    tot = {'su_s': 0.0, 'thread_cpu_s': 0.0, 'root_cpu_s': 0.0, 'server_s': 0.0}
    gaps = set()  # cost figures not measured for every attempt (or the attempt count missing): totals are lower bounds
    for b in blocks:
        tm = telemetry(b, [r for r in rows if r['block'] == b['block']])
        for k in ('su_s', 'thread_cpu_s', 'root_cpu_s'):
            tot[k] += tm[k]
        lb = {k: '' if tm['unmeasured'][src] == 0 else '+?'
              for k, src in (('su_s', 'root_s'), ('thread_cpu_s', 'cpu_s'), ('root_cpu_s', 'root_cpu_s'))}
        gaps |= {k for k, v in lb.items() if v}
        tot['server_s'] += b.get('server_s') if num(b.get('server_s')) else 0.0
        L.append(f"  {b['block']:>2} | {tm['failed']}/{f(tm['attempts'])}, {tm['dropped']}, {tm['missed']} | "
                 f"{f(tm['interval_med'], '.1f')}/{f(tm['interval_max'], '.1f')} | {f(tm['dur_med'], '.2f')}/"
                 f"{f(tm['dur_max'], '.2f')} | {tm['su_s']:.2f}{lb['su_s']}/{f(tm['su_s_max'], '.2f')} | "
                 f"{tm['thread_cpu_s']:.3f}{lb['thread_cpu_s']} + "
                 f"{tm['root_cpu_s']:.2f}{lb['root_cpu_s']} | {tm['overlap_s']:.2f} ({tm['requests_overlapped']}/"
                 f"{tm['requests']}) | {f(tm['load_overlap_s'], '.2f')} | {tm['thread']} | "
                 f"{', '.join(tm['root_shell'])}")
    if blocks:
        lb = {k: '+?' if k in gaps else '' for k in tot}
        L.append(f"  all blocks: su {tot['su_s']:.1f}{lb['su_s']} s wall, sampler CPU {tot['thread_cpu_s']:.2f}"
                 f"{lb['thread_cpu_s']} s + root shells {tot['root_cpu_s']:.1f}{lb['root_cpu_s']} s (system_server's "
                 f"dumpsys work not attributable) over {tot['server_s']:.0f} s of server time")
        if gaps:
            L.append('  +? = a lower bound: some attempts did not measure that cost (failed, raised or never returned) '
                     'or the attempt count is missing; the unmeasured part is unknown, not zero')
    return L


def launch_problems(res):
    """Reasons a launch's timings are not eligible although it may be complete: an agent process found at its end
    (result.json agents_end; the runner refuses to start with one and checks before and after every block), or port
    8080 not free at its end (server cleanup unconfirmed). [(code, detail)]; none without a usable result.json
    (that run is INCOMPLETE anyway)."""
    why = []
    if res is not None and res['agents_end'] != '':
        why.append(('AGENTS_PRESENT', f'agent processes at the end of the launch: {res["agents_end"][:120]!r}'))
    if res is not None and res['port_free_end'] is not True:
        why.append(('PORT_NOT_FREE', 'port 8080 not free at the end of the launch (server cleanup unconfirmed)'))
    return why


def coverage(run, blocks, rows, res):
    """Planned work (run.json 'matrix') against what was recorded. Returns ({block: COMPLETED | FAILED | SKIPPED |
    MISSING}, reasons the run is not COMPLETE); no reasons = COMPLETE. A result.json alone proves nothing."""
    if 'matrix' not in run:
        return {}, ['run.json has no planned matrix: completeness cannot be verified']
    got = {b['block']: b for b in blocks}
    skipped = {x.get('block') for x in (res or {}).get('skipped', []) if isinstance(x, dict)}
    state = {}
    for m in run['matrix']:
        b, seq = got.get(m['block']), m['block']
        if b is None:
            state[seq] = 'SKIPPED' if seq in skipped else 'MISSING'
            continue
        rs = [r for r in rows if r['block'] == seq and 'error' not in r]
        done = ((b['round'], b['length'], b['mode']) == (m['round'], m['length'], m['mode'])
                and b['end_reason'] == 'all questions done' and sum(not r['scored'] for r in rs) == 1
                and [r['question'] for r in rs if r['scored']] == m['questions'])
        state[seq] = 'COMPLETED' if done else 'FAILED'
    why = []
    if res is None:
        why.append('no usable result.json (interrupted, still running, or partial)')
    elif res['end_reason'] != 'all blocks done':
        why.append(f'ended: {res["end_reason"]}')
    for k in ('FAILED', 'SKIPPED', 'MISSING'):
        if k in state.values():
            why.append(f'blocks {k}: {sorted(s for s, v in state.items() if v == k)}')
    if extra := sorted(set(got) - set(state)):
        why.append(f'unplanned blocks {extra}')
    planned = sum(len(m['questions']) for m in run['matrix'])
    answered = sum(1 for r in rows if r['scored'] and 'error' not in r)
    if answered != planned:
        why.append(f'{answered}/{planned} planned scored requests answered')
    return state, why


def condition_diffs(blocks):
    """Environmental differences among the blocks behind one comparison: UNMATCHED when non-empty. Uses every
    condition reading of every block (start, between requests, end) and the gate skin; a missing value is a
    difference, never a match. Validity is separate (validate_block)."""
    if not blocks:
        return ['no recorded block behind the comparison']
    why = []
    readings = [e for b in blocks for e in env_readings(b)]
    skins = [b['gate'].get('skin') for b in blocks] + [e.get('skin') for e in readings]
    if not all(num(x) for x in skins):
        why.append('a skin reading is missing')
    elif max(skins) - min(skins) > SKIN_MATCH_C:
        why.append(f'skin {min(skins):.2f}-{max(skins):.2f} degC over the blocks (> {SKIN_MATCH_C} apart)')
    bats = [e.get('battery_pct') for e in readings]
    if not all(num(x) for x in bats):
        why.append('a battery % reading is missing')
    elif max(bats) - min(bats) > BATTERY_MATCH_PCT:
        why.append(f'battery {min(bats)}-{max(bats)} % (> {BATTERY_MATCH_PCT} apart)')
    caps = {p: [cap_of(e, p) for e in readings] for p in SERVER_POLICIES}
    if any(None in v for v in caps.values()):
        why.append('a CPU cap reading is missing')
    elif any(len(set(v)) > 1 for v in caps.values()):
        why.append(f'CPU caps differ {({p: sorted(set(v)) for p, v in caps.items()})}')
    return why


def stats(rows):
    ok = [r for r in rows if r['valid']]
    gen = [r['gen_tok_s'] for r in ok if (r.get('predicted_n') or 0) >= GEN_MIN_TOKENS and r.get('gen_tok_s')]
    return {'n': len(ok), 'all': len(rows), 'prompt_ms': med([r['prompt_ms'] for r in ok]),
            'prompt_ms_p95': p95([r['prompt_ms'] for r in ok]), 'wall_s': med([r['wall_s'] for r in ok]),
            'wall_s_p95': p95([r['wall_s'] for r in ok]), 'prompt_n': med([r['prompt_n'] for r in ok]),
            'cache_n': med([r['cache_n'] for r in ok]), 'gen_tok_s': med(gen)}


OUTCOMES = ('CORRECT', 'ABSTAINED', 'MISSED', 'STALE', 'DISTRACTOR', 'WRONG', 'UNSUPPORTED', 'MALFORMED', 'EMPTY',
            'ERROR')
RIGHT = ('CORRECT', 'ABSTAINED')
FORMATS = ('STRICT', 'FENCED', 'MALFORMED', 'EMPTY', 'ERROR')
CLASSES = ('FRESH', 'PRIME', 'FIRST', 'CACHED')
# configuration a comparison across launches requires to be identical (capacity: only ctx may differ)
CONFIG_KEYS = ('task', 'exp', 'rounds', 'smoke', 'decoding', 'warmup', 'prime', 'set_sha256', 'model_sha256',
               'pinned_sha256', 'build_sha256', 'code_sha256', 'thresholds', 'telemetry')


def recall_section(rows):
    """Structured-output compliance and recall accuracy, separately, over every scored request (N): unparsed and
    failed requests stay in N, never dropped."""
    acc = {}
    for r in rows:
        if r['scored']:
            acc.setdefault((r['length'], r['mode']), []).append(r)
    L = ['', 'format compliance (structured replies; ERROR = request failed): strict % = STRICT / N, parsed % = '
         '(STRICT + FENCED) / N', f"{'mem':>5}{'mode':>7}{'N':>4}  " + ' '.join(f'{x:>9}' for x in FORMATS) +
         f"{'strict':>8}{'parsed':>8}"]
    for (length, mode), rs in sorted(acc.items()):
        fm = ['ERROR' if 'error' in r else r.get('format') for r in rs]
        L.append(f"{length:>5}{mode:>7}{len(rs):>4}  " + ' '.join(f'{fm.count(x):>9}' for x in FORMATS) +
                 f"{f(ratio(fm.count('STRICT'), len(rs)), '.0%'):>8}"
                 f"{f(ratio(fm.count('STRICT') + fm.count('FENCED'), len(rs)), '.0%'):>8}")
    L += ['', 'recall accuracy (exact match against the frozen key; right = CORRECT + ABSTAINED): accuracy = right / N;'
          ' parsed-only = right / parsed replies (conditional, shown separately). memory 0 = unknown-answer baseline: '
          'nothing to recall, not comparable to recall at memory > 0',
          f"{'mem':>5}{'mode':>7}{'N':>4}  " + ' '.join(f'{x[:9]:>9}' for x in OUTCOMES) + f"{'acc':>6}{'parsed':>8}"]
    for (length, mode), rs in sorted(acc.items()):
        oc = ['ERROR' if 'error' in r else r['outcome'] for r in rs]
        right, parsed = sum(oc.count(x) for x in RIGHT), sum(r.get('format') in ('STRICT', 'FENCED') for r in rs)
        L.append(f"{length:>5}{mode:>7}{len(rs):>4}  " + ' '.join(f'{oc.count(x):>9}' for x in OUTCOMES) +
                 f"{f(ratio(right, len(rs)), '.0%'):>6}{f(ratio(right, parsed), '.0%'):>8}" +
                 ('  baseline' if length == 0 else ''))
    return L


def run_report(folder):
    """(text, comparison records) of one run folder."""
    name, R = Path(folder).name, load_run(folder)
    if R.run is None:
        return f'MEM-1 REPORT {name} [INVALID]: ' + '; '.join(R.issues), []
    run, blocks, rows, res = R.run, R.blocks, R.rows, R.res
    state, missing = coverage(run, blocks, rows, res)
    missing = R.issues + ([f'{len(R.orphans)} requests of blocks {sorted({r["block"] for r in R.orphans})} '
                           'without a block record (interrupted mid-block): kept out of every aggregate']
                          if R.orphans else []) + missing
    launch = launch_problems(res)
    tag = 'SMOKE - NOT A MEASUREMENT' if run['smoke'] else 'INCOMPLETE' if missing else \
        'COMPLETE, TIMINGS NOT ELIGIBLE: ' + '; '.join(f'{c} {d}' for c, d in launch) if launch else 'COMPLETE'
    L = [f'MEM-1 REPORT {name}: exp {run["exp"]}, task {run["task"]}, ctx {run["ctx"]}, rounds {run["rounds"]} [{tag}]',
         'coverage: ' + ', '.join(f'{k} {list(state.values()).count(k)}' for k in
                                  ('COMPLETED', 'FAILED', 'SKIPPED', 'MISSING')) +
         f' of {len(run.get("matrix", []))} planned blocks' + ('' if not missing else '; NOT COMPLETE: ' +
                                                               '; '.join(missing)),
         'times: prompt ms = server prompt_ms (TTFT PROXY: not streamed, no true TTFT); wall = client request time;',
         f'monitoring: measured with asynchronous telemetry running ({SAMPLE_PERIOD_S} s period, see the telemetry '
         'section); its effect on inference is NOT quantified: not an unmonitored baseline;',
         'gen t/s = median server predicted_per_second (replies >= 5 tokens); n = valid / all requests; P95 nearest rank',
         'classes: FRESH / CACHED = the scored questions (cache_prompt false / true, CACHED = server cache_n >= shared '
         'prefix); FIRST = the unscored prime in a CACHED block (cold first request with memory); PRIME = the same '
         'prime in a FRESH block', '',
         f"{'mem':>5}{'class':>7}{'n':>7}{'prompt ms':>10}{'P95':>8}{'wall s':>8}{'P95':>7}{'prompt_n':>9}"
         f"{'cache_n':>8}{'gen t/s':>8}"]
    by = {}
    for r in rows:
        by.setdefault((r['length'], r['class']), []).append(r)
    for (length, cls) in sorted(by, key=lambda k: (k[0], CLASSES.index(k[1]) if k[1] in CLASSES else 9)):
        st = stats(by[(length, cls)])
        L.append(f"{length:>5}{cls:>7}{st['n']:>4}/{st['all']:<2}{f(st['prompt_ms']):>10}{f(st['prompt_ms_p95']):>8}"
                 f"{f(st['wall_s'], '.2f'):>8}{f(st['wall_s_p95'], '.2f'):>7}{f(st['prompt_n']):>9}{f(st['cache_n']):>8}"
                 f"{f(st['gen_tok_s'], '.2f'):>8}")
    if run['task'] == 'freetext':
        n = sum(r['scored'] for r in rows)
        L += ['', f'free-text replies: UNSCORED - MANUAL REVIEW REQUIRED. {n} scored-position replies are kept '
              'verbatim in requests.jsonl (field reply); no automatic grade, not part of any recall accuracy.']
    else:
        L += recall_section(rows)
    L += ['', 'blocks: seq rnd mem mode | load s | PSS peak / HWM / RSS end MiB | min avail MiB, max swap MiB | '
          'skin degC/status start -> end | battery % degC | X1 cap MHz, X1 top | VALID or INVALID (measurement)']
    for b in blocks:
        mib, rs = (lambda v: f(v and v / 1024)), resources(b)
        L.append(f"  {b['block']:>2} {b['round']} {b['length']:>4} {b['mode']:<6} | {f(b['load_s'], '.1f')} | "
                 f"{mib(rs['peak_pss_kb'])}/{mib(rs['vm_hwm_kb'])}/{mib(rs['vm_rss_end_kb'])} | "
                 f"{f(rs['min_mem_available_mib'])}, {f(rs['max_swap_used_mib'])} | {f(b['gate'].get('skin'), '.2f')}/"
                 f"{b['start'].get('status')} -> {f(b['end'].get('skin'), '.2f')}/{b['end'].get('status')} | "
                 f"{b['start'].get('battery_pct')} {b['start'].get('battery_c')} | {f(cap(b) and cap(b) / 1000)}, "
                 f"{f(top_share(b['start'], b['end']), '.0%')} | "
                 f"{'VALID' if not b['problems'] else 'INVALID: ' + reasons_text(b['problems'])}")
    if blocks:
        L.append(f"  model load: first block {f(blocks[0]['load_s'], '.1f')} s (page cache not dropped), later median "
                 f"{f(med([b['load_s'] for b in blocks[1:] if b['load_s'] is not None]), '.1f')} s")
    L += telemetry_lines(blocks, rows)
    recs = comparisons(run, blocks, rows, missing, launch) if 'matrix' in run else []
    L += ['', 'comparisons (median prompt ms, wall s and gen t/s ratios, shown only when COMPARABLE):']
    L += render(recs) if recs else ['  none (no planned matrix)']
    if res:
        L += ['', f"end: {res['end_reason']}; agents at end: {res['agents_end'] or 'none'}; port 8080 free: "
              f"{res['port_free_end']}"]
    return '\n'.join(L), [{'run': name, **r} for r in recs]


def planned_side(matrix, length, cls, prime_id):
    """(sorted expected question ids, expected repeats) of one comparison side from the planned matrix."""
    mode = 'CACHED' if cls in ('CACHED', 'FIRST') else 'FRESH'
    ms = [m for m in matrix if m['length'] == length and m['mode'] == mode]
    qs = [prime_id] * len(ms) if cls in ('FIRST', 'PRIME') else [q for m in ms for q in m['questions']]
    return sorted(qs), len(ms)


def compare(name, variable, allowed, identical, sides, run_why, smoke=(), launch=()):
    """One comparison record. sides: two dicts (rows, blocks, expected, repeats). COMPARABLE only when the
    configuration agrees except the independent variable, the run(s) are complete, both sides cover exactly the
    planned questions and repeats (and the same ones), every block behind them is VALID and their conditions are
    MATCHED; otherwise NOT COMPARABLE with every reason. A favourable match never overrides another reason.
    smoke: the smoke launches behind it (a pipeline check, never a measurement); launch: (code, detail) of
    launch_problems behind it (agents or a busy port at a launch's end)."""
    why = [('RUN_NOT_COMPLETE', w) for w in run_why] + [('SMOKE_LAUNCH', w) for w in smoke] + list(launch)
    why += [('CONFIG_DIFFERS', k) for k, same in identical.items() if not same]
    valid = [[r for r in s['rows'] if r['valid']] for s in sides]
    actual = [sorted(r['question'] for r in v) for v in valid]
    reps = [len({(r.get('run'), r['block']) for r in v}) for v in valid]
    if not all(valid):
        why.append(('NO_VALID_REQUESTS', [i for i, v in enumerate(valid) if not v]))
    if sides[0]['expected'] != sides[1]['expected']:
        why.append(('QUESTIONS_DIFFER', 'the planned questions of the two sides differ'))
    for i, s in enumerate(sides):
        if actual[i] != s['expected']:
            why.append(('QUESTIONS_NOT_AS_PLANNED',
                        f'side {i}: {len(actual[i])} valid of {len(s["expected"])} planned'))
        if reps[i] != s['repeats']:
            why.append(('REPEATS_NOT_AS_PLANNED', f'side {i}: {reps[i]} blocks of {s["repeats"]} planned'))
    if sides[0]['repeats'] != sides[1]['repeats']:
        why.append(('REPEATS_DIFFER', f'{sides[0]["repeats"]} vs {sides[1]["repeats"]} planned blocks'))
    blocks = sides[0]['blocks'] + sides[1]['blocks']
    invalid = sorted(f"{b.get('run', '')}{b['block']}" for b in blocks if b['problems'])
    if invalid or not blocks:
        why.append(('ENV_INVALID', f'blocks {invalid} INVALID' if invalid else 'no recorded blocks'))
    unmatched = condition_diffs(blocks)
    why += [('ENV_UNMATCHED', w) for w in unmatched]
    ok = not why
    st = [stats(s['rows']) for s in sides]
    return {'name': name, 'independent_variable': variable, 'allowed_to_differ': allowed,
            'required_identical': identical, 'expected_questions': [s['expected'] for s in sides],
            'actual_questions': actual, 'expected_repeats': [s['repeats'] for s in sides], 'actual_repeats': reps,
            'complete': not run_why, 'environment_valid': not invalid and bool(blocks),
            'environment_matched': not unmatched, 'verdict': 'COMPARABLE' if ok else 'NOT COMPARABLE',
            'reasons': [{'code': c, 'detail': d} for c, d in why],
            'ratios': {k: ratio(st[0][k], st[1][k]) for k in ('prompt_ms', 'wall_s', 'gen_tok_s')} if ok else None}


def render(recs):
    out = []
    for c in recs:
        if c['verdict'] == 'COMPARABLE':
            rt = c['ratios']
            out.append(f"  {c['name']}: COMPARABLE (varies {c['independent_variable']}) prompt "
                       f"{f(rt['prompt_ms'], '.3f')}, wall {f(rt['wall_s'], '.3f')}, gen {f(rt['gen_tok_s'], '.3f')}")
        else:
            codes = sorted({r['code'] for r in c['reasons']})
            out.append(f"  {c['name']}: NOT COMPARABLE {codes}: " +
                       '; '.join(f"{r['code']} {r['detail']}" for r in c['reasons']))
    return out


def comparisons(run, blocks, rows, run_why, launch=()):
    """The within-launch comparisons, from the planned matrix (a side that recorded nothing is still listed):
    memory length per class against the shortest memory, and CACHED against FRESH per memory. One launch = one
    run.json, so every configuration dimension is identical by construction (and run_problems has checked that
    run.json records this code's configuration and pinned hashes)."""
    matrix, prime = run['matrix'], run['prime']['id']
    lengths = sorted({m['length'] for m in matrix})
    smoke = ['smoke launch'] if run['smoke'] else []
    same = {k: True for k in ('one launch: one run.json (model, build, decoding, command, set, code, ctx)',)}

    def side(length, cls):
        expected, repeats = planned_side(matrix, length, cls, prime)
        mode = 'CACHED' if cls in ('CACHED', 'FIRST') else 'FRESH'
        return {'rows': [r for r in rows if r['length'] == length and r['class'] == cls], 'expected': expected,
                'repeats': repeats, 'blocks': [b for b in blocks if b['length'] == length and b['mode'] == mode]}
    out = []
    for cls in ('FRESH', 'FIRST', 'CACHED'):
        for length in lengths[1:]:
            out.append(compare(f'{cls} mem {length} / mem {lengths[0]}', 'memory length',
                               ['memory text and its tokens'], same, [side(length, cls), side(lengths[0], cls)],
                               run_why, smoke, launch))
    for length in lengths:
        out.append(compare(f'mem {length} CACHED / FRESH', 'cache mode', ['cache_prompt and server prefix reuse'],
                           same, [side(length, 'CACHED'), side(length, 'FRESH')], run_why, smoke, launch))
    return out


def capacity_report(folders):
    """(lines, records): ctx 4096 vs 2048 for exactly two capacity launches. Only ctx (and --ctx-size) may differ;
    any other configuration difference, an incomplete or smoke launch, missing work or telemetry makes it NOT
    COMPARABLE."""
    if len(folders) < 2:
        return [], []
    runs = [(load_run(x), Path(x).name) for x in folders]
    caps = {R.run['ctx']: (R, n) for R, n in runs if R.run is not None and R.run['exp'] == 'capacity'}
    if set(caps) != {2048, 4096} or len(runs) != 2:
        return ['', 'capacity comparison: needs exactly two capacity launches, one at ctx 2048 and one at 4096'], []
    (A, na), (B, nb) = caps[2048], caps[4096]
    run_why, launch = [], []
    for R, n in ((A, na), (B, nb)):
        launch += [(c, f'{n}: {d}') for c, d in launch_problems(R.res)]
        _, miss = coverage(R.run, R.blocks, R.rows, R.res) if 'matrix' in R.run else (None, ['no planned matrix'])
        run_why += [f'{n}: {w}' for w in R.issues + miss + ([f'{len(R.orphans)} orphan requests'] if R.orphans else [])]
    a, b = A.run, B.run
    identical = {k: k in a and k in b and a[k] == b[k] for k in CONFIG_KEYS}
    ca, cb = list(a.get('command', [])), list(b.get('command', []))
    for c in (ca, cb):
        if '--ctx-size' in c and c.index('--ctx-size') + 1 < len(c):
            c[c.index('--ctx-size') + 1] = 'X'
    identical['command except --ctx-size'] = bool(ca) and ca == cb
    identical['planned matrix (questions, rounds, modes, memory)'] = 'matrix' in a and \
        a.get('matrix') == b.get('matrix')
    prime = a['prime']['id']
    tagged = lambda R, n: ([{**r, 'run': n} for r in R.rows], [{**x, 'run': n} for x in R.blocks])
    (ra, ba), (rb, bb) = tagged(A, na), tagged(B, nb)
    recs = []
    for cls in ('FRESH', 'FIRST', 'CACHED'):
        mode = 'CACHED' if cls in ('CACHED', 'FIRST') else 'FRESH'
        sides = []
        for R, rows, blocks in ((B, rb, bb), (A, ra, ba)):  # ratio = ctx 4096 / ctx 2048
            exp, reps = planned_side(R.run.get('matrix', []), 1024, cls, prime)
            sides.append({'rows': [r for r in rows if r['class'] == cls], 'expected': exp, 'repeats': reps,
                          'blocks': [x for x in blocks if x['mode'] == mode]})
        recs.append(compare(f'capacity {cls}: ctx 4096 / ctx 2048', 'context allocation (ctx 2048 -> 4096)',
                            ['ctx', '--ctx-size', 'run folder and times'], identical, sides, run_why,
                            [f'{n}: smoke launch' for R, n in ((A, na), (B, nb)) if R.run['smoke']], launch))
    hwm = lambda bs: [resources(x)['vm_hwm_kb'] for x in bs]
    hw = lambda bs: 'n/a (missing)' if not bs or None in hwm(bs) else f"{max(hwm(bs)) / 1024:.0f}"
    out = ['', 'capacity (memory 1024, same prompts; only ctx may differ): configuration ' +
           ('SAME except ctx' if all(identical.values()) else
            'DIFFERENT: ' + '; '.join(k for k, v in identical.items() if not v)),
           f'  peak VmHWM MiB {hw(A.blocks)} -> {hw(B.blocks)}'] + render(recs)
    return out, [{'runs': [na, nb], **r} for r in recs]


def report(folders):
    """(text, comparison records) for run folders, plus the capacity pair when two folders are given."""
    parts = [run_report(x) for x in folders]
    lines, recs = capacity_report(folders)
    return '\n\n'.join(t for t, _ in parts) + '\n'.join([''] + lines), [r for _, rs in parts for r in rs] + recs


# ---------------------------------------------------------------- main

def logger_sched(thermal_log):
    """What oneshot.sh's root thermal logger reported about its own CPU list and nice (thermal_logger_sched.txt next
    to the log; None when absent). Recorded, never assumed."""
    p = Path(thermal_log).parent / 'thermal_logger_sched.txt'
    return p.read_text()[:DIAG_CHARS] if p.is_file() else None


def root_left_running(b):
    """Why no further observation (the postflight, a later block) may start: the sampler thread outlived its bounded
    join, or a telemetry su call timed out (sampler latch, a sample or the postflight; killing the su client does not
    prove its root shell ended, and every observation writes the same root output file). None otherwise."""
    if b['sampler'].get('alive_after_join') is not False:
        return 'the telemetry sampler did not stop: fail closed (it may still be running)'
    if b['sampler'].get('root_timeout') or any(timed_out(x) for x in [b.get('end'), *b['samples']]):
        return 'a telemetry su call timed out: fail closed (its root shell may still be running)'
    return None


def code_files():
    """The source files whose SHA-256 every run.json records (code_sha256 keys)."""
    return [Path(__file__).resolve(), HERE / 'oneshot.sh', Path(repro.__file__), Path(cm.__file__), Path(cr.__file__),
            Path(pm.__file__), Path(chat.__file__), Path(sm.__file__)]


def pinned_problems():
    got = {p: cr.sha256(p) if Path(p).is_file() else 'missing' for p in PINNED}
    return [f'{p}: SHA-256 {got[p][:12]} != pinned {want[:12]}' for p, want in PINNED.items() if got[p] != want], got


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', choices=tuple(EXPERIMENTS))
    ap.add_argument('--ctx', type=int, choices=(2048, 4096))
    ap.add_argument('--rounds', type=int, default=2, help='even, >= 2 (default 2); --smoke runs 1')
    ap.add_argument('--smoke', action='store_true', help='1 round, shortest and longest memory, 2 questions: '
                    'checks the pipeline, NOT a measurement')
    ap.add_argument('--report', nargs='+', type=Path, metavar='RUN_DIR')
    ap.add_argument('--json', type=Path, metavar='PATH', help='with --report: write the comparison records here')
    ap.add_argument('--thermal-log', default=str(THERMAL_LOG))
    a = ap.parse_args(argv)
    if a.report:
        if a.exp or a.ctx or a.smoke:
            ap.error('--report takes only run folders')
        text, recs = report(a.report)
        print(text)
        if a.json:
            a.json.write_text(dumps(recs, indent=1) + '\n')
        return
    if a.json:
        ap.error('--json goes with --report')
    if not (a.exp and a.ctx):
        ap.error('give --exp and --ctx (or --report)')
    rounds = 1 if a.smoke else a.rounds
    if not a.smoke and (rounds < 2 or rounds % 2):
        ap.error('--rounds must be even and >= 2 (each round pair balances the order)')
    s = task_view(load_set(), EXPERIMENTS[a.exp]['task'])
    if why := plan_problems(s, a.exp, a.ctx):
        raise SystemExit('not started: ' + '; '.join(why))
    cr.require_native()  # before any root command or server
    cr.SKIN_GATE_C = SKIN_MATCH_C  # power_map.thermal_gate reads it: a gated block starts <= launch start + 1.0 degC
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cr.exit_on_signal)
    try:
        why = cm.refusals()
        avail = cr.meminfo_mib()['mem_available_mib']
        if avail < MIN_MEM_AVAILABLE_MIB:
            why.append(f'MemAvailable {avail} MiB < {MIN_MEM_AVAILABLE_MIB}')
        if why:
            raise SystemExit('not started: ' + '; '.join(why))
        print('checking model and build SHA-256 (about 30 s)', flush=True)
        bad, hashes = pinned_problems()
        if bad:
            raise SystemExit('not started: ' + '; '.join(bad))
        cr.wait_cores('start')
        start = observe(a.thermal_log)
        if why := observation_problems(start, 'start', {}):  # every field valid, as inside a block
            raise SystemExit('not started: ' + '; '.join(why))
        out = OUT_ROOT / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{a.exp}_ctx{a.ctx}{"_smoke" if a.smoke else ""}'
        out.mkdir(parents=True)
        plan = schedule(a.exp, rounds, a.smoke)
        code = code_files()
        (out / 'run.json').write_text(dumps({
            'task': s['task'], 'benchmark': 'MEM-1', 'exp': a.exp, 'ctx': a.ctx, 'rounds': rounds, 'smoke': a.smoke,
            'started': cr.utc(),
            'schedule': plan, 'matrix': matrix(s, plan, a.smoke), 'prime': s['prime'], 'command': command(a.ctx), 'decoding': DECODING, 'warmup': WARMUP,
            'set_sha256': SET_SHA256, 'model_sha256': {MODEL: hashes[MODEL]}, 'pinned_sha256': hashes,
            'build_sha256': repro.build_sha256(), 'code_sha256': {str(p): cr.sha256(p) for p in code},
            'start': start, 'mem_available_start_mib': avail, 'gate_max_s': pm.GATE_MAX_S,
            'telemetry': {'sample_period_s': SAMPLE_PERIOD_S, 'sample_max_gap_s': SAMPLE_MAX_GAP_S,
                          'max_samples': MAX_SAMPLES, 'root_timeout_s': ROOT_TIMEOUT_S,
                          'sampler_cpus': list(SAMPLER_CPUS), 'sampler_nice': SAMPLER_NICE},
            'thermal_logger_sched': logger_sched(a.thermal_log),
            'skin_gate_c': cr.SKIN_GATE_C, 'thresholds': {'min_mem_available_mib': MIN_MEM_AVAILABLE_MIB,
                                                          'status_stop': STATUS_STOP,
                                                          'battery_max_c': repro.BATTERY_MAX_C,
                                                          'request_timeout_s': REQUEST_TIMEOUT_S}}, indent=1) + '\n')
        print(f'run folder {out}; {len(plan)} blocks', flush=True)
        idle, end_reason, skipped = {'skin': start['skin']}, 'all blocks done', []
        with open(out / 'requests.jsonl', 'a') as rlog, open(out / 'blocks.jsonl', 'a') as blog:
            def emit(row):
                rlog.write(dumps(row) + '\n')
                rlog.flush()
            for seq, (rnd, length, mode) in enumerate(plan):
                cr.wait_cores(f'block {seq}')
                if agents_before := repro.agents():
                    end_reason = f'stopped before block {seq}: agent processes running: {agents_before[:120]!r}'
                    print(end_reason, flush=True)
                    break
                gate = pm.thermal_gate(a.thermal_log, idle, f'block {seq}', a.smoke)
                cond = observe(a.thermal_log)
                if why := safety_stop(cond):
                    end_reason = f'stopped before block {seq}: {why}'
                    print(end_reason, flush=True)
                    break
                print(f'[block {seq} r{rnd} mem {length} {mode}] skin {gate["skin"]} degC, status {cond.get("status")}'
                      f', waited {gate["waited_s"]} s{", WARM START" if gate["warm_start"] else ""}', flush=True)
                try:
                    cr.check_cores(f'before block {seq}')
                    b = run_block(seq, rnd, length, mode, s, a.ctx, out, emit, a.thermal_log, a.smoke)
                except cr.CoresLost as e:
                    print(f'[block {seq}] {e}; block skipped: the run will be INCOMPLETE', flush=True)
                    skipped.append({'block': seq, 'reason': str(e)})
                    continue
                lingering = root_left_running(b)  # checked before the postflight: no observation while one may run
                b.update(gate=gate, start=cond, end={'skipped': lingering} if lingering else observe(a.thermal_log),
                         agents={'before': agents_before, 'after': repro.agents()})
                lingering = lingering or root_left_running(b)
                rows = [recheck_row(r, s) for r in read_records(out / 'requests.jsonl', ROW_SCHEMA)[0]
                        if r['block'] == seq]  # the report's strict loader
                b['problems'] = validate_block(b, rows, a.ctx, s, a.smoke)  # the report recomputes it the same way
                blog.write(dumps(b) + '\n')
                blog.flush()
                print(f'[block {seq}] {b["n_rows"]} requests, load {b["load_s"]} s; {b["end_reason"]}; '
                      f'{"VALID" if not b["problems"] else "INVALID: " + reasons_text(b["problems"])}', flush=True)
                if b['refused']:
                    end_reason = f'block {seq} {b["end_reason"]}'
                    break
                if lingering:
                    end_reason = f'stopped after block {seq}: {lingering}'
                    print(end_reason, flush=True)
                    break
        res = {'end_reason': end_reason, 'ended': cr.utc(), 'skipped': skipped, 'agents_end': repro.agents(),
               'port_free_end': repro.port_free()}
        R = load_run(out)
        state, missing = coverage(R.run, R.blocks, R.rows, res)
        missing = R.issues + ([f'{len(R.orphans)} orphan requests'] if R.orphans else []) + missing
        res.update(complete=not missing, coverage=state, not_complete=missing)  # the report recomputes both
        (out / 'result.json').write_text(dumps(res) + '\n')
        text, recs = report([out])
        (out / 'report.txt').write_text(text + '\n')
        (out / 'comparisons.json').write_text(dumps(recs, indent=1) + '\n')
        print(text, flush=True)
        (out / 'artifacts.sha256').write_text(''.join(f'{cr.sha256(p)}  {p.name}\n' for p in sorted(out.iterdir())
                                                      if p.is_file() and p.name != 'artifacts.sha256'))
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        for proc in list(cm.LIVE):
            cm.stop(proc)


if __name__ == '__main__':
    main()
