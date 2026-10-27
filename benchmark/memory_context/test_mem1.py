"""Offline tests for MEM-1: the frozen set, fakes and temporary folders only; nothing is launched, no root, no
inference. Every interface that reaches the phone (su, dumpsys, sysfs/proc readings, processes, sockets, the server)
is replaced by a guard for the whole module, and run folders go to a temporary OUT_ROOT: an unexpected call fails
the test in Debian proot and in native Termux alike. Run from this folder: python3 -B -m unittest -v test_mem1"""
import copy
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mem1  # noqa: E402
from mem1 import num  # noqa: E402

cr, cm, chat, repro = mem1.cr, mem1.cm, mem1.chat, mem1.repro
ORIGINAL_TAIL = mem1.thermal_tail  # before the guard replaces it
S = mem1.load_set()
Q = {q['id']: q for q in S['questions']}


# ---------------------------------------------------------------- hardware / environment guard

TOUCHED = []
GUARDED = [(cr, 'root'), (cr, 'read_dump'), (cr, 'require_native'), (cr, 'read_thermal'), (cr, 'meminfo_mib'),
           (cr, 'allowed_cpus'), (cr, 'post'), (cr, 'healthy'), (cr, 'sha256'), (cm, 'spawn'), (cm, 'stop'),
           (cm, 'refusals'), (repro, 'cpufreq'), (repro, 'agents'), (repro, 'port_free'), (repro, 'conditions'),
           (mem1, 'props_json'), (mem1, 'proc_mem'), (mem1, 'server_caps'), (mem1, 'thermal_tail'),
           (subprocess, 'Popen'), (os, 'killpg'), (os, 'sched_setaffinity'), (os, 'setpriority'),
           (socket, 'create_connection'), (socket.socket, 'connect'), (socket.socket, 'connect_ex')]
_patches = []


def guard(name):
    def refuse(*a, **k):
        TOUCHED.append(name)
        raise AssertionError(f'offline test touched {name}')
    return refuse


def setUpModule():
    for obj, attr in GUARDED:
        p = mock.patch.object(obj, attr, guard(f'{getattr(obj, "__name__", obj)}.{attr}'))
        p.start()
        _patches.append(p)
    p = mock.patch.object(mem1, 'OUT_ROOT', Path(tempfile.mkdtemp()))  # a launch never writes to ~/robot_runs
    p.start()
    _patches.append(p)


def tearDownModule():
    for p in reversed(_patches):
        p.stop()


class Offline(unittest.TestCase):
    """Fails any test that reached a guarded interface, even if the code under test swallowed the error."""
    def setUp(self):
        TOUCHED.clear()

    def tearDown(self):
        self.assertEqual(TOUCHED, [], 'a hardware or environment interface was reached')


# ---------------------------------------------------------------- fakes

def fake_ids(text):
    """Deterministic fake tokenizer: BOS + one id per whitespace/newline piece (enough for prefix arithmetic)."""
    return [2] + [hash(w) % 50000 + 10 for w in text.replace('\n', ' \n ').split(' ') if w]


def fake_raw():
    """The frozen set with the token data of both tasks recomputed by fake_ids, so checks() can pass against a fake
    server."""
    raw = copy.deepcopy(S)
    for task in ('recall', 'freetext'):
        view = mem1.task_view(raw, task)
        for length, L in view['lengths'].items():
            ids = {q['id']: fake_ids(mem1.prompt_for(view, int(length), q)) for q in view['questions']}
            for qid, v in ids.items():
                L['prompts'][qid].update(tokens=len(v), ids_sha256=mem1.sha256_text(json.dumps(v)))
            for qid, v in ids.items():
                L['prompts'][qid]['lcp'] = {k: mem1.lcp(v, w) for k, w in ids.items() if k != qid}
            prime = fake_ids(mem1.prompt_for(view, int(length), view['prime']))
            L['prime_prompt'].update(tokens=len(prime), ids_sha256=mem1.sha256_text(json.dumps(prime)),
                                     lcp_warmup=mem1.lcp(fake_ids(mem1.WARMUP), prime),
                                     lcp={k: mem1.lcp(prime, w) for k, w in ids.items()})
            shared = min(mem1.lcp(prime, v) for v in ids.values())
            (raw['lengths'][length] if task == 'recall' else raw['lengths'][length]['freetext'])[
                'shared_prefix_tokens'] = shared
    return raw


FAKE_RAW = fake_raw()  # the frozen set the fake server's tokens belong to (the report's set in report tests)
FAKE = mem1.task_view(FAKE_RAW, 'recall')


def good_checks(length, qs, s=FAKE):
    """A block's passed no-generation checks and warm-up, as checks()/run_block record them."""
    L = s['lengths'][str(length)]
    tokens = {s['prime']['id']: L['prime_prompt']['tokens'], **{q: L['prompts'][q]['tokens'] for q in qs}}
    return {'template': {k: True for k in tokens}, 'tokens': tokens, 'n_ctx': 2048, 'chat_template_sha256': 'a' * 64,
            'warmup': {'wall_s': 0.01, 'timings': {'cache_n': 0, 'prompt_n': 9, 'predicted_n': 4}}}


def fake_set(task='recall'):
    return mem1.task_view(fake_raw(), task)


class FakeFrozenSet:
    """Report tests: load_set() returns FAKE_RAW, so the report recomputes every expectation from the set the fake
    rows were produced with (the real set's token counts differ from the fake tokenizer's)."""
    def setUp(self):
        super().setUp()
        p = mock.patch.object(mem1, 'load_set', return_value=FAKE_RAW)
        p.start()
        self.addCleanup(p.stop)


PROVENANCE = {'model_sha256': {mem1.MODEL: mem1.PINNED[mem1.MODEL]}, 'pinned_sha256': dict(mem1.PINNED),
              'build_sha256': {Path(mem1.SERVER).name: mem1.PINNED[mem1.SERVER],
                               Path(repro.LIBGGML_CPU).name: mem1.PINNED[repro.LIBGGML_CPU]},
              'code_sha256': {str(p): 'c' * 64 for p in mem1.code_files()}}  # what main records
REPRO_BUILD = {Path(mem1.SERVER).name: mem1.PINNED[mem1.SERVER],
               'libggml-cpu.so.0.18.0': mem1.PINNED[repro.LIBGGML_CPU]}  # repro.build_sha256: regular files only
DROP = object()
PID = 4242
CLOCK0 = datetime(2026, 10, 25, tzinfo=timezone.utc)  # the fixtures' UTC at mono 0: UTC follows mono


def stamps(mono, age=3.0):
    """utc (whole seconds, as cr.utc) at monotonic time mono, and the thermal log stamp age s before it."""
    now = (CLOCK0 + timedelta(seconds=mono)).replace(microsecond=0)
    return {'utc': now.isoformat(), 'at': f'{now - timedelta(seconds=age):%Y-%m-%dT%H:%M:%SZ}', 'thermal_age_s': age}


def cond(status=0, skin=25.0, pct=90, bstat='Discharging', bat_c=30.0, cap6=2850000, cap4=2350000, tis=0, mono=0.0,
         **kw):
    """A healthy mem1.observe pre/postflight reading unless told otherwise; a keyword set to DROP removes that
    field."""
    caps = {'policy0': 1800000, 'policy4': cap4, 'policy6': cap6}
    c = {'mono': mono, **stamps(mono if num(mono) else 0.0), 'root_rc': 0, 'root_s': 0.2, 'battery_rc': 0,
         'dump_rc': 0, 'status': status, 'skin': skin, 'battery_pct': pct, 'battery_status': bstat,
         'battery_c': bat_c, 'battery_w': 3.1, 'z9': 30000,
         'z10': 30000, 'z11': 30000, 'root_cpus': '0-3', 'root_nice': 10, 'root_cpu_s': 0.02,
         'cpufreq': {p: ({'max': c, 'hw_max': 2850000, 'time_in_state': {'2850000': tis}} if c is not None else
                         {'error': 'unreadable'}) for p, c in caps.items()}}
    c.update(kw)
    return {k: v for k, v in c.items() if v is not DROP}


def obs(pid=PID, mono=0.0, **kw):
    """A healthy mem1.observe telemetry sample of llama-server pid (before sampler_loop adds its times)."""
    x = {**cond(mono=mono), 'cpufreq': {'policy4': {'max': 2350000}, 'policy6': {'max': 2850000}},
         'mem_available_mib': 5000, 'swap_used_mib': 0, 'cached_mib': 900, 'smaps_rc': 0, 'pss_pid': pid,
         'pss_kb': 3000000, 'rss_kb': 3100000}
    x.update(kw)
    return {k: v for k, v in x.items() if v is not DROP}


def sample(t, t0=100.0, **kw):
    """A healthy recorded sample at block time t of a block whose server started at mono t0 (as sampler_loop stores
    it)."""
    x = {**obs(mono=t0 + t), 'slot': int(t // 15), 't_sched': t, 't': t, 't_end': t + 0.3, 'cpu_s': 0.004}
    x.update(kw)
    return {k: v for k, v in x.items() if v is not DROP}


SAMPLER_OK = {'tid': 9, 'requested_cpus': [0, 1, 2, 3], 'requested_nice': 10, 'cpus': [0, 1, 2, 3], 'nice': 10}


class FakeTelemetry:
    """observe() and pin_sampler() faked for run_block: records which thread observed; the first sample releases the
    fake health check (so a block always has its slot-0 sample, as a real model load guarantees)."""

    def __init__(self, status=0, fail_after=None, block=None):
        self.threads, self.first, self.status, self.fail_after, self.block = [], threading.Event(), status, \
            fail_after, block

    def observe(self, log, pid=None):
        self.threads.append(threading.current_thread())
        if pid is None:
            return cond(mono=round(time.monotonic(), 3))
        if self.fail_after is not None and len(self.threads) > self.fail_after:
            raise RuntimeError('sensor exploded')
        self.first.set()
        if self.block is not None:
            self.block.wait(30)
        return obs(pid, mono=round(time.monotonic(), 3), status=self.status)

    def healthy(self):
        return self.first.wait(5)


def answer(v):
    return json.dumps({'answer': v})


class FakeServer:
    """llama.cpp b1609 slot semantics for one slot: cache_prompt false -> n_past 0; true -> LCP with the slot's
    previous prompt + reply (server-context.cpp around get_common_prefix)."""

    def __init__(self, s, length, answers=None, template_ok=True, n_ctx=2048, cache_bug=False):
        self.s, self.length, self.slot, self.calls = s, length, [], []
        self.answers, self.template_ok, self.n_ctx, self.cache_bug = answers or {}, template_ok, n_ctx, cache_bug
        self.threads = []

    def post(self, path, body, timeout=600):
        self.calls.append(path)
        self.threads.append(threading.current_thread())
        if path == '/apply-template':
            p = chat.build_prompt('gemma', body['messages'])
            return {'prompt': p if self.template_ok else p.replace('<|turn>', '<start_of_turn>')}
        if path == '/tokenize':
            return {'tokens': fake_ids(body['content'])}
        assert path == '/completion'
        ids = fake_ids(body['prompt'])
        n_past = mem1.lcp(self.slot, ids) if body['cache_prompt'] else 0
        if self.cache_bug:
            n_past = 0
        q = next((q for q in self.s['questions'] if body['prompt'].endswith(q['text'] + '<turn|>\n<|turn>model\n')),
                 None)
        reply = self.answers.get(q['id'], answer('UNKNOWN')) if q else 'Ready.'
        self.slot = ids + [7] * 5
        n = min(6, body['n_predict'])  # the warm-up asks for 4
        return {'content': reply, 'tokens_evaluated': len(ids), 'tokens_cached': len(ids) + n, 'truncated': False,
                'stop_type': 'eos', 'timings': {'cache_n': n_past, 'prompt_n': len(ids) - n_past,
                                                'prompt_ms': 0.0004 * (len(ids) - n_past) + 0.01, 'predicted_n': n,
                                                'predicted_ms': 0.3, 'predicted_per_second': n / 0.3 * 1000}}


def slowed(server, s=0.01):
    post = server.post
    server.post = lambda *a, **k: (time.sleep(s), post(*a, **k))[1]
    return server


class FakeProc:
    pid = PID

    def poll(self):
        return None


PROC_MEM = {'VmRSS': '3000000 kB', 'VmHWM': '3100000 kB', 'VmSwap': '0 kB', 'Cpus_allowed_list': '4-7'}


def run_fake_block(server, mode, seq=0, rnd=0, smoke=False, out=None, tel=None, sampler=None, period=0.02, ctx=2048):
    """run_block against a fake server with the real asynchronous sampler (fast period) over a faked observe(), or a
    replacement sampler_loop."""
    rows, stopped, tel = [], [], tel or FakeTelemetry()
    out = out or tempfile.mkdtemp()
    with mock.patch.object(cm, 'spawn', return_value=FakeProc()), \
            mock.patch.object(cm, 'stop', side_effect=lambda p: stopped.append(p)), \
            mock.patch.object(cr, 'healthy', tel.healthy), mock.patch.object(cr, 'post', server.post), \
            mock.patch.object(mem1, 'observe', tel.observe), mock.patch.object(mem1, 'pin_sampler',
                                                                              return_value=dict(SAMPLER_OK)), \
            mock.patch.object(mem1, 'SAMPLE_PERIOD_S', period), mock.patch.object(cr, 'check_cores'), \
            mock.patch.object(mem1, 'props_json', return_value={
                'chat_template': 'T', 'default_generation_settings': {'n_ctx': server.n_ctx}}), \
            mock.patch.object(mem1, 'proc_mem', return_value=PROC_MEM), ExitStack() as st:
        if sampler:
            st.enter_context(mock.patch.object(mem1, 'sampler_loop', sampler))
        b = mem1.run_block(seq, rnd, server.length, mode, server.s, ctx, out, rows.append, '/nonexistent', smoke)
    return b, rows, stopped


RIGHT = {'q_begin': answer('Pepper'), 'q_mid': answer('4417'), 'q_end': answer('honey'),
         'q_update': answer('hallway shelf')}


def recheck(rows, s=None):
    return [mem1.recheck_row(json.loads(mem1.dumps(r)), s or mem1.task_view(FAKE_RAW, 'recall')) for r in rows]


def finish(b, rows, seq, skin=25.0, warm=False, s=None, smoke=False, **kw):
    """The block record as main completes it: gate, preflight reading before the server start, postflight after."""
    b.update(gate={'skin': skin, 'status': 0, 'waited_s': 0, 'warm_start': warm},
             start=cond(skin=skin, tis=seq, mono=b['mono_t0'] - 1),
             end=cond(skin=skin, tis=seq + 1, mono=b['mono_t0'] + b['server_s'] + 0.5),
             agents={'before': '', 'after': ''})
    b.update(kw)
    b['problems'] = mem1.validate_block(b, recheck(rows, s), b.get('ctx'), s or mem1.task_view(FAKE_RAW, 'recall'),
                                        smoke)
    return b


def codes(why):
    return {w['code'] for w in why}


def details(why):
    return ' | '.join(f"{w['code']} {w['detail']}" for w in why)


# ---------------------------------------------------------------- tests

class FrozenSet(Offline):
    def test_pinned_and_tamper_refused(self):
        self.assertEqual(mem1.sha256_text(mem1.SET.read_text()), mem1.SET_SHA256)
        with mock.patch.object(mem1, 'SET_SHA256', '0' * 64), self.assertRaises(SystemExit):
            mem1.load_set()

    def test_token_accounting(self):
        for task in ('recall', 'freetext'):
            v = mem1.task_view(S, task)
            base = v['lengths']['0']['prompts']
            for length, L in v['lengths'].items():
                for qid, p in L['prompts'].items():  # the memory adds the same token count to every question
                    self.assertEqual(p['tokens'] - base[qid]['tokens'], L['memory_tokens'], (task, length, qid))
                self.assertEqual(L['deviation_tokens'], L['memory_tokens'] - L['target_tokens'])
                if L['target_tokens']:
                    self.assertLessEqual(abs(L['deviation_pct']), 10, length)  # recorded; reported in the README
                self.assertGreater(L['shared_prefix_tokens'], L['memory_tokens'])  # instructions + memory shared
                self.assertLessEqual(L['shared_prefix_tokens'], min(p['tokens'] for p in L['prompts'].values()))

    def test_prime_shares_the_scored_prefix_and_is_not_scored(self):
        self.assertNotIn(S['prime']['id'], Q)
        for task in ('recall', 'freetext'):
            v = mem1.task_view(S, task)
            for length, L in v['lengths'].items():
                self.assertEqual(L['prime_prompt']['lcp_with_scored'], L['shared_prefix_tokens'], (task, length))
                self.assertEqual(mem1.sha256_text(mem1.prompt_for(v, int(length), S['prime'])),
                                 L['prime_prompt']['text_sha256'])
                for q in S['questions']:
                    self.assertEqual(mem1.sha256_text(mem1.prompt_for(v, int(length), q)),
                                     L['prompts'][q['id']]['text_sha256'])

    def test_frozen_cache_expectations(self):
        """Round 9 finding 2: cache_n expectations come from the frozen LCP tables (built from the same warm-up
        prompt), never from a run's own records."""
        import build_set
        self.assertEqual(build_set.WARMUP, mem1.WARMUP)
        for task in ('recall', 'freetext'):
            for length, L in mem1.task_view(S, task)['lengths'].items():
                P, shared = L['prime_prompt'], L['shared_prefix_tokens']
                self.assertEqual(min(P['lcp'].values()), P['lcp_with_scored'], (task, length))
                self.assertLess(P['lcp_warmup'], shared)
                for a, pa in L['prompts'].items():
                    for b, n in pa['lcp'].items():
                        self.assertEqual(n, L['prompts'][b]['lcp'][a])
                        self.assertTrue(shared <= n < min(pa['tokens'], L['prompts'][b]['tokens']), (task, a, b))
        order = mem1.question_order(S['questions'], 1, False)
        e = [mem1.expected(S, 1024, 'CACHED', 1, i) for i in range(7)]
        L = S['lengths']['1024']
        self.assertEqual([x['question'] for x in e], ['prime'] + [q['id'] for q in order])
        self.assertEqual([x['expected_cache_n'] for x in e[:3]],
                         [L['prime_prompt']['lcp_warmup'], L['prime_prompt']['lcp'][order[0]['id']],
                          L['prompts'][order[0]['id']]['lcp'][order[1]['id']]])
        self.assertEqual({x['tokens_expected'] for x in e[1:]}, {L['prompts'][q['id']]['tokens'] for q in order})
        self.assertEqual({mem1.expected(S, 1024, 'FRESH', 1, i)['expected_cache_n'] for i in range(7)}, {0})
        self.assertIsNone(mem1.expected(S, 1024, 'CACHED', 0, 7))
        self.assertIsNone(mem1.expected(S, 999, 'CACHED', 0, 1))

    def test_probe_positions_dates_and_nesting(self):
        order = ['begin', 'stale', 'mid', 'near', 'update', 'end']
        prev = []
        for length, L in S['lengths'].items():
            e = L['entries']
            if not L['target_tokens']:
                self.assertEqual(e, [])
                self.assertIn(S['empty'], L['memory_text'])
                continue
            self.assertEqual([x['probe'] for x in e if x['probe']], order)
            self.assertEqual((e[0]['probe'], e[-1]['probe']), ('begin', 'end'))
            days = [datetime.strptime(x['date'] + ' 2026', '%b %d %Y') for x in e]
            self.assertTrue(all(x < y for x, y in zip(days, days[1:])), length)  # strictly rising
            fill = [x['text'] for x in e if not x['probe']]
            self.assertEqual(fill, S['filler_pool'][:len(fill)])
            self.assertTrue(set(prev) <= set(fill))
            prev = fill

    def test_same_relevant_facts_at_every_memory_length(self):
        """Lengths differ only by fillers (controlled distractors): every probe fact, the update and the near-miss are
        in every non-zero memory exactly once and in the same order; memory 0 has none (unknown-answer baseline)."""
        probes = [text for _, _, text in S['probes']]
        for length, L in S['lengths'].items():
            facts = [x['text'] for x in L['entries'] if x['probe']]
            self.assertEqual(facts, [] if length == '0' else probes, length)
            for text in probes:
                self.assertEqual(L['memory_text'].count(text), 0 if length == '0' else 1, (length, text))

    def test_key_is_frozen_normalized_and_never_leaked(self):
        """The answer key: normalized, disjoint per question, UNKNOWN never a value; no key value appears in the
        instructions, a question or the prime, and fillers never contain one or a probe name."""
        values = set()
        for q in S['questions']:
            lists = q['accept'] + q['stale'] + q['distractor']
            self.assertEqual(len(lists), len(set(lists)), q['id'])
            self.assertTrue(all(mem1.normalize(v) == v for v in lists), q['id'])
            self.assertNotIn(mem1.UNKNOWN, lists)
            self.assertEqual(bool(q['accept']), q['kind'] == 'present', q['id'])
            values |= set(lists)
        leak = lambda text: [v for v in values if re.search(r'\b' + re.escape(v) + r'\b', text.lower())]
        for text in [S['system'], S['freetext_system'], S['prime']['text']] + [q['text'] for q in S['questions']]:
            self.assertEqual(leak(text), [], text)
        names = ('nadia', 'oskar', 'leo ', 'charger', 'tea', 'dentist', 'dog', 'shelf', 'drawer')
        for text in S['filler_pool']:
            self.assertEqual(leak(text), [], text)
            self.assertFalse([n for n in names if n in text.lower()], text)

    def test_questions_identical_across_lengths_and_tasks(self):
        for task in ('recall', 'freetext'):
            v = mem1.task_view(S, task)
            for length, L in v['lengths'].items():
                self.assertEqual(set(L['prompts']), set(Q))
                for q in S['questions'] + [S['prime']]:
                    self.assertTrue(mem1.prompt_for(v, int(length), q).endswith(
                        '<|turn>user\n' + q['text'] + '<turn|>\n<|turn>model\n'))
        self.assertIn('{"answer": "VALUE"}', S['system'])
        self.assertNotIn('JSON', S['freetext_system'])


class Plan(Offline):
    def test_command_is_server_manager_setup_q4(self):
        self.assertEqual(mem1.command(2048), ['taskset', '-c', '4-7', *cr.server_cmd()])
        a, b = mem1.command(2048), mem1.command(4096)
        self.assertEqual([x for x in a if x != '2048'], [x for x in b if x != '4096'])
        for flag in ('--swa-full', '--cache-ram', '--parallel'):
            self.assertIn(flag, a)
        self.assertFalse({'--model-draft', '--mmproj', '--spec-type'} & set(a))

    def test_decoding_fixed_and_template_gemma4(self):
        self.assertEqual(mem1.DECODING, {'n_predict': 48, 'temperature': 0, 'top_k': 1, 'seed': 0,
                                         'stop': ['<turn|>'], 'stream': False})
        p = mem1.prompt_for(S, 128, Q['q_begin'])
        self.assertTrue(p.startswith('<|turn>system\n'))
        self.assertNotIn('<start_of_turn>', p)
        self.assertNotIn('<|think|>', p)
        self.assertEqual(mem1.payload(p, False)['cache_prompt'], False)

    def test_fit_and_combinations(self):
        for exp, ctx in (('primary', 2048), ('capacity', 2048), ('capacity', 4096), ('long', 4096),
                         ('freetext', 2048)):
            self.assertEqual(mem1.plan_problems(mem1.task_view(S, mem1.EXPERIMENTS[exp]['task']), exp, ctx), [])
        why = mem1.plan_problems(S, 'long', 2048)
        self.assertTrue(any('runs at ctx' in w for w in why))
        self.assertTrue(any('> ctx 2048' in w for w in why))  # 2048-token memory cannot fit: refused, not truncated

    def test_overflow_refused(self):
        s = copy.deepcopy(S)
        s['lengths']['1024']['prompts']['q_mid']['tokens'] = 2048 - 47
        why = mem1.plan_problems(s, 'primary', 2048)
        self.assertEqual(len(why), 1)
        self.assertIn('q_mid', why[0])
        s['lengths']['1024']['prompts']['q_mid']['tokens'] = 2048 - 48  # exactly fits
        self.assertEqual(mem1.plan_problems(s, 'primary', 2048), [])
        s['lengths']['1024']['prime_prompt']['tokens'] = 2048 - 47  # the unscored prime must fit too
        self.assertTrue(any('prime' in w for w in mem1.plan_problems(s, 'primary', 2048)))

    def test_prompt_drift_refused(self):
        with mock.patch.object(chat, 'build_prompt', lambda t, m: 'x'):
            self.assertTrue(all('differs from the frozen set' in w for w in mem1.plan_problems(S, 'capacity', 2048)))
        # the recall token data cannot pass for the free-text prompts (or the reverse)
        mixed = {**mem1.task_view(S, 'freetext'), 'lengths': S['lengths']}
        self.assertTrue(mem1.plan_problems(mixed, 'freetext', 2048))

    def test_schedule_balanced(self):
        plan = mem1.schedule('primary', 2, False)
        self.assertEqual(len(plan), 20)
        pos = {}
        for i, (r, l, m) in enumerate(plan):
            pos.setdefault((l, m), []).append(i % 10)
        self.assertEqual(len(pos), 10)
        self.assertTrue(all(len(v) == 2 and sum(v) == 9 for v in pos.values()))  # mean position 4.5 for all
        self.assertEqual([m for _, _, m in plan[:10]], ['FRESH', 'CACHED'] * 5)
        self.assertEqual(mem1.schedule('capacity', 2, False),
                         [(0, 1024, 'FRESH'), (0, 1024, 'CACHED'), (1, 1024, 'CACHED'), (1, 1024, 'FRESH')])
        self.assertEqual({(l, m) for _, l, m in mem1.schedule('primary', 1, True)},
                         {(0, 'FRESH'), (0, 'CACHED'), (1024, 'FRESH'), (1024, 'CACHED')})

    def test_scored_questions_identical_across_the_matrix(self):
        for exp, rounds in (('primary', 2), ('primary', 4), ('capacity', 2), ('long', 2), ('freetext', 2)):
            m = mem1.matrix(S, mem1.schedule(exp, rounds, False), False)
            for b in m:  # every block scores every question once
                self.assertEqual(sorted(b['questions']), sorted(Q), (exp, b))
            for r in range(rounds):  # within a round: the same order in every condition
                self.assertEqual(len({tuple(b['questions']) for b in m if b['round'] == r}), 1, (exp, r))
            per = Counter((b['length'], b['mode'], q) for b in m for q in b['questions'])
            for (length, mode, q), n in per.items():  # FRESH and CACHED: the same questions, as often
                self.assertEqual(n, per[(length, 'CACHED' if mode == 'FRESH' else 'FRESH', q)])
                self.assertEqual(n, rounds)
        firsts = {mem1.question_order(S['questions'], r, False)[0]['id'] for r in range(6)}
        self.assertEqual(firsts, set(Q))  # rotation by round, recorded in the matrix
        smoke = mem1.matrix(S, mem1.schedule('primary', 1, True), True)
        self.assertEqual({tuple(b['questions']) for b in smoke}, {('q_begin', 'q_mid')})

    def test_main_refusals_before_anything_starts(self):
        with self.assertRaises(SystemExit) as e:  # plan check: before any phone interface
            mem1.main(['--exp', 'long', '--ctx', '2048'])
        self.assertIn('not started', str(e.exception))
        with mock.patch.object(cr, 'require_native', side_effect=SystemExit('not native Termux')), \
                self.assertRaises(SystemExit) as e:
            mem1.main(['--exp', 'primary', '--ctx', '2048'])
        self.assertIn('native Termux', str(e.exception))
        with mock.patch.object(cr, 'require_native'), mock.patch.object(cm, 'refusals', return_value=['agent']), \
                mock.patch.object(cr, 'meminfo_mib', return_value={'mem_available_mib': 9000}), \
                mock.patch.object(signal_module(), 'signal'), self.assertRaises(SystemExit) as e:
            mem1.main(['--exp', 'primary', '--ctx', '2048'])
        self.assertIn('not started: agent', str(e.exception))
        for bad in (['--exp', 'primary', '--ctx', '2048', '--rounds', '3'], ['--exp', 'primary'],
                    ['--report', '/x', '--exp', 'primary'], ['--exp', 'primary', '--ctx', '2048', '--json', '/x']):
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), mock.patch('sys.stderr', io.StringIO()):
                mem1.main(bad)

    def test_start_refused_on_any_invalid_reading(self):
        """The launch-start reading is judged field by field like every block reading (a failed su refuses)."""
        with ExitStack() as st:
            for obj, attr, kw in ((cr, 'require_native', {}), (signal_module(), 'signal', {}),
                                  (cm, 'refusals', {'return_value': []}),
                                  (cr, 'meminfo_mib', {'return_value': {'mem_available_mib': 9000}}),
                                  (mem1, 'pinned_problems', {'return_value': ([], {})}), (cr, 'wait_cores', {}),
                                  (mem1, 'observe', {'return_value': cond(battery_rc=1)})):
                st.enter_context(mock.patch.object(obj, attr, **kw))
            st.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises(SystemExit) as e:
                mem1.main(['--exp', 'capacity', '--ctx', '2048'])
        self.assertIn('start: battery read rc 1', str(e.exception))

    def test_unexpected_phone_access_fails(self):
        """The guard itself: a launch that is not stopped by a mock reaches a guarded interface and fails."""
        with mock.patch.object(cr, 'require_native'), mock.patch.object(signal_module(), 'signal'), \
                self.assertRaises(AssertionError):
            mem1.main(['--exp', 'primary', '--ctx', '2048'])
        self.assertEqual(TOUCHED, ['chat_mtp.refusals'])
        for f, args in ((cr.root, ('id', 't')), (cr.read_dump, ()), (repro.cpufreq, ()), (cr.meminfo_mib, ()),
                        (cm.spawn, (['x'], None)), (cr.post, ('/x', {})), (subprocess.run, (['true'],)),
                        (repro.conditions, ('/x',)), (mem1.observe, ('/x',)), (mem1.server_caps, ()),
                        (mem1.thermal_tail, ('/x',)), (os.sched_setaffinity, (0, {0})), (os.setpriority, (0, 0, 10))):
            with self.assertRaises(AssertionError):
                f(*args)
        with socket.socket() as sk, self.assertRaises(AssertionError):
            sk.connect_ex(('127.0.0.1', 8080))
        self.assertGreaterEqual(len(TOUCHED), 10)
        TOUCHED.clear()


def signal_module():
    return mem1.signal


class Recall(Offline):
    """Repair B: structured recall, parsed strictly and scored by exact match; no prose is interpreted."""

    def o(self, qid, reply, present=True):
        return mem1.score(Q[qid], *mem1.parse_answer(reply), present)

    def test_round7_examples_are_never_correct(self):
        for qid, reply, present in (('q_begin', "Nadia's cat is named Pepper and Max.", True),
                                    ('q_mid', 'The code is not actually supposed to be 4417.', True),
                                    ('q_update', 'Hallway was wrong; now kitchen.', True),
                                    ('q_absent_time', "I don't know; the appointment is in the morning.", False)):
            self.assertEqual(mem1.parse_answer(reply), ('MALFORMED', None))
            self.assertEqual(self.o(qid, reply, present), 'MALFORMED')
            self.assertEqual(self.o(qid, answer(reply), present), 'WRONG' if present else 'UNSUPPORTED', reply)
        self.assertEqual(self.o('q_begin', answer('Pepper and Max')), 'WRONG')
        self.assertEqual(self.o('q_mid', answer('not 4417')), 'WRONG')
        self.assertEqual(self.o('q_update', answer('kitchen')), 'STALE')
        self.assertEqual(self.o('q_absent_time', answer('in the morning'), False), 'UNSUPPORTED')

    def test_correct_structured_answers(self):
        self.assertEqual(mem1.parse_answer(answer('Pepper')), ('STRICT', 'Pepper'))
        self.assertEqual(self.o('q_begin', '  {"answer": "PEPPER."}\n'), 'CORRECT')  # case, period, outer space
        self.assertEqual(mem1.parse_answer('```json\n{"answer": "4417"}\n```'), ('FENCED', '4417'))
        self.assertEqual(self.o('q_mid', '```\n{"answer": "4417"}\n```'), 'CORRECT')
        self.assertEqual(self.o('q_update', answer('On the  hallway shelf')), 'CORRECT')
        self.assertEqual(self.o('q_end', answer('honey and no milk')), 'CORRECT')
        self.assertEqual(self.o('q_begin', answer('Ｐｅｐｐｅｒ')), 'CORRECT')  # NFKC

    def test_wrong_structured_answers(self):
        self.assertEqual(self.o('q_mid', answer('9032')), 'DISTRACTOR')
        self.assertEqual(self.o('q_mid', answer('4417 or 9032')), 'WRONG')
        self.assertEqual(self.o('q_mid', answer('44 17')), 'WRONG')
        self.assertEqual(self.o('q_update', answer('the kitchen drawer')), 'STALE')
        self.assertEqual(self.o('q_update', answer('kitchen drawer, now hallway shelf')), 'WRONG')
        self.assertEqual(self.o('q_end', answer('milk')), 'DISTRACTOR')
        self.assertEqual(self.o('q_end', answer('honey and milk')), 'DISTRACTOR')
        self.assertEqual(self.o('q_end', answer('no honey')), 'WRONG')
        self.assertEqual(self.o('q_begin', answer('')), 'WRONG')
        self.assertEqual(self.o('q_begin', answer('Pepper (probably)')), 'WRONG')

    def test_unknown(self):
        self.assertEqual(self.o('q_absent_pet', answer('UNKNOWN'), False), 'ABSTAINED')
        self.assertEqual(self.o('q_absent_pet', answer('unknown.'), False), 'ABSTAINED')
        self.assertEqual(self.o('q_absent_pet', answer('Pepper'), False), 'UNSUPPORTED')  # the cat, not a dog
        self.assertEqual(self.o('q_absent_time', answer("I don't know"), False), 'UNSUPPORTED')  # not the token
        self.assertEqual(self.o('q_begin', answer('UNKNOWN')), 'MISSED')
        self.assertEqual(self.o('q_begin', answer('UNKNOWN'), False), 'ABSTAINED')  # memory 0 baseline
        self.assertEqual(self.o('q_begin', answer('Pepper'), False), 'UNSUPPORTED')  # memory 0: a guess

    def test_malformed_json(self):
        for reply in ('{"answer": "Pepper"', '{"answer": 4417}', '{"answer": null}', '{"answer": ["Pepper"]}',
                      '{"answer": "Pepper", "confidence": 1}', '{"answer": "Max", "answer": "Pepper"}',
                      'Answer: {"answer": "Pepper"}', '{"answer": "Pepper"} I think.', '{"answer": "Pepper"}\n'
                      '{"answer": "Max"}', '["Pepper"]', '"Pepper"', 'null', 'Pepper', '{"Answer": "Pepper"}',
                      '```json\n{"answer": "Pepper"}\n``` extra', 'Sure!\n```json\n{"answer": "Pepper"}\n```'):
            self.assertEqual(mem1.parse_answer(reply), ('MALFORMED', None), reply)
            self.assertEqual(self.o('q_begin', reply), 'MALFORMED', reply)
        for reply in ('', '   \n', None):
            self.assertEqual(self.o('q_begin', reply), 'EMPTY')

    def test_freetext_is_unscored(self):
        s = mem1.task_view(S, 'freetext')
        self.assertEqual(mem1.scored_fields(s, Q['q_begin'], 'Pepper.', True),
                         {'outcome': 'UNSCORED', 'review': 'MANUAL REVIEW REQUIRED'})
        self.assertEqual(mem1.scored_fields(mem1.task_view(S, 'recall'), Q['q_begin'], answer('Pepper'), True),
                         {'format': 'STRICT', 'answer': 'Pepper', 'outcome': 'CORRECT'})


class RequestChecks(Offline):
    def data(self, cache_n, n=100, **kw):
        d = {'tokens_evaluated': n, 'truncated': False, 'stop_type': 'eos',
             'timings': {'cache_n': cache_n, 'prompt_n': n - cache_n, 'prompt_ms': 50.0, 'predicted_n': 6,
                         'predicted_ms': 400.0}}
        d.update(kw)
        return d

    def test_classes(self):
        self.assertEqual([mem1.request_class('FRESH', p) for p in (True, False)], ['PRIME', 'FRESH'])
        self.assertEqual([mem1.request_class('CACHED', p) for p in (True, False)], ['FIRST', 'CACHED'])

    def test_valid_and_invalid(self):
        rp = mem1.row_problems
        self.assertEqual(rp('FRESH', self.data(0), 100, 0, 80), [])
        self.assertEqual(rp('FIRST', self.data(2), 100, 2, 80), [])
        self.assertEqual(rp('CACHED', self.data(85), 100, 85, 80), [])
        self.assertTrue(rp('FRESH', self.data(85), 100, 0, 80))  # reuse despite cache_prompt false
        self.assertTrue(any('prefix not reused' in w for w in rp('CACHED', self.data(0), 100, 85, 80)))
        self.assertTrue(rp('CACHED', self.data(84), 100, 85, 80))  # partial reuse is not the expected reuse
        self.assertTrue(any('truncated' in w for w in rp('FRESH', self.data(0, truncated=True), 100, 0, 80)))
        self.assertTrue(any('tokens_evaluated' in w for w in rp('FRESH', self.data(0, n=90), 100, 0, 80)))
        self.assertTrue(any('stop_type' in w for w in rp('FRESH', self.data(0, stop_type='none'), 100, 0, 80)))
        bad = self.data(0)
        bad['timings']['prompt_n'] = 50
        self.assertTrue(any('prompt_n' in w for w in rp('FRESH', bad, 100, 0, 80)))
        limit = self.data(0, stop_type='limit')
        limit['timings']['predicted_n'] = 48
        self.assertEqual(rp('FRESH', limit, 100, 0, 80), [])  # recorded, not invalid
        self.assertTrue(any('stop_type limit' in w for w in rp('FRESH', self.data(0, stop_type='limit'), 100, 0, 80)))
        fast = self.data(0)
        fast['timings'].update(predicted_per_second=1e6)  # 6 tokens in 400 ms is 15 tok/s
        self.assertTrue(any('server formula' in w for w in rp('FRESH', fast, 100, 0, 80)))
        fast['timings'].update(predicted_per_second=15.0)
        self.assertEqual(rp('FRESH', fast, 100, 0, 80), [])
        gone = self.data(0)
        del gone['timings']['prompt_ms']
        self.assertIn('timings.prompt_ms missing or not a finite number >= 0', rp('FRESH', gone, 100, 0, 80))
        for bad in (float('nan'), float('inf'), -1.0, True, '5'):
            d = self.data(0)
            d['timings']['prompt_ms'] = bad
            self.assertTrue(rp('FRESH', d, 100, 0, 80), bad)
        d = self.data(0)
        d['timings']['predicted_per_second'] = float('inf')
        self.assertTrue(rp('FRESH', d, 100, 0, 80))


    def test_request_interval_wall_and_server_times_agree(self):
        """A request's server time fits in its client wall time, which fits in its block-time interval."""
        _, rows, _ = run_fake_block(FakeServer(fake_set(), 128, RIGHT), 'FRESH')
        r = rows[2]
        self.assertTrue(recheck([r])[0]['valid'], recheck([r])[0]['problems'])
        long = {**r, 'wall_s': round(r['t_end'] - r['t_start'] + 1, 3)}
        self.assertIn('longer than the request interval', ' '.join(recheck([long])[0]['problems']))
        slow = {**r, 'timings': {**r['timings'], 'prompt_ms': 5000.0}, 'prompt_ms': 5000.0}
        self.assertIn('longer than the client wall time', ' '.join(recheck([slow])[0]['problems']))


class Block(Offline):
    def test_cached_block_primes_then_reuses_for_every_scored_question(self):
        s = fake_set()
        b, rows, stopped = run_fake_block(FakeServer(s, 512, RIGHT), 'CACHED', seq=1)
        self.assertEqual(b['end_reason'], 'all questions done')
        self.assertEqual([r['class'] for r in rows], ['FIRST'] + ['CACHED'] * 6)
        self.assertEqual([r['scored'] for r in rows], [False] + [True] * 6)
        self.assertTrue(all(r['valid'] for r in rows), [r['problems'] for r in rows])
        shared = s['lengths']['512']['shared_prefix_tokens']
        self.assertTrue(all(r['cache_n'] >= shared for r in rows[1:]))
        self.assertLess(rows[0]['cache_n'], shared)
        self.assertEqual([r['question'] for r in rows[1:]], [q['id'] for q in S['questions']])  # round 0 order
        self.assertIsNone(rows[0]['outcome'])
        self.assertEqual({r['question']: r['outcome'] for r in rows[1:]},
                         {**{q: 'CORRECT' for q in RIGHT}, 'q_absent_pet': 'ABSTAINED', 'q_absent_time': 'ABSTAINED'})
        self.assertTrue(all(r['format'] == 'STRICT' for r in rows[1:]))
        self.assertEqual(stopped, [stopped[0]])
        self.assertEqual(mem1.resources(b)['vm_hwm_kb'], 3100000)
        self.assertTrue(all(b['checks']['template'].values()))
        self.assertIn('prime', b['checks']['tokens'])
        self.assertNotIn('env', b)  # no synchronous readings between requests any more
        self.assertEqual(finish(b, rows, 1)['problems'], [])  # a healthy fake block is VALID end to end
        self.assertEqual((b['sampler']['alive_after_join'], b['sampler']['dropped']), (False, 0))
        self.assertGreaterEqual(b['sampler']['ended_s'], b['sampling_end_s'])
        self.assertEqual(b['sampler']['cpus'], [0, 1, 2, 3])
        self.assertTrue(all(x['pss_pid'] == PID and 'cpu_s' in x and 'slot' in x for x in b['samples']))

    def test_fresh_block_scores_the_same_questions(self):
        s = fake_set()
        b, rows, _ = run_fake_block(FakeServer(s, 0, RIGHT), 'FRESH', seq=4, rnd=1)
        self.assertEqual([r['class'] for r in rows], ['PRIME'] + ['FRESH'] * 6)
        self.assertTrue(all(r['valid'] and r['cache_n'] == 0 for r in rows))
        self.assertTrue(all(not r['present'] for r in rows))
        self.assertEqual([r['question'] for r in rows[1:]], [q['id'] for q in S['questions'][1:] + S['questions'][:1]])
        self.assertEqual(sum(r['outcome'] == 'UNSUPPORTED' for r in rows[1:]), 4)  # right answers are guesses at 0
        _, cached, _ = run_fake_block(FakeServer(s, 0, RIGHT), 'CACHED', seq=5, rnd=1)
        self.assertEqual([r['question'] for r in rows[1:]], [r['question'] for r in cached[1:]])

    def test_freetext_block_keeps_replies_verbatim_unscored(self):
        prose = {'q_begin': "Nadia's cat is named Pepper and Max.", 'q_mid': 'The code is not actually 4417.'}
        b, rows, _ = run_fake_block(FakeServer(fake_set('freetext'), 128, prose), 'FRESH')
        self.assertEqual(b['end_reason'], 'all questions done')
        by = {r['question']: r for r in rows}
        self.assertEqual(by['q_begin']['reply'], prose['q_begin'])
        self.assertTrue(all(r['outcome'] == 'UNSCORED' and 'format' not in r for r in rows[1:]))
        self.assertTrue(all(r['valid'] for r in rows))
        again = recheck(rows, mem1.task_view(FAKE_RAW, 'freetext'))
        self.assertTrue(all(r['outcome'] == 'UNSCORED' and r['valid'] for r in again[1:]))

    def test_cache_not_reused_is_flagged(self):
        _, rows, _ = run_fake_block(FakeServer(fake_set(), 256, cache_bug=True), 'CACHED')
        self.assertTrue(all(not r['valid'] for r in rows[1:]))

    def test_template_or_token_mismatch_refuses_without_generation(self):
        srv = FakeServer(fake_set(), 128, template_ok=False)
        b, rows, stopped = run_fake_block(srv, 'FRESH')
        self.assertTrue(b['refused'] and 'template' in b['end_reason'])
        self.assertEqual((rows, srv.calls.count('/completion'), len(stopped)), ([], 0, 1))
        srv = FakeServer(S, 128)  # real frozen ids vs the fake tokenizer
        b, rows, _ = run_fake_block(srv, 'FRESH')
        self.assertTrue(b['refused'] and 'server tokens' in b['end_reason'])
        b, _, _ = run_fake_block(FakeServer(fake_set(), 128, n_ctx=4096), 'FRESH')
        self.assertIn('n_ctx', b['end_reason'])
        b, rows, _ = run_fake_block(FakeServer(fake_set(), 128, n_ctx=None), 'FRESH')  # /props without n_ctx
        self.assertTrue(b['refused'] and 'server n_ctx None != 2048 (or unknown)' in b['end_reason'] and rows == [])
        s = fake_set()
        s['lengths']['128']['prime_prompt']['lcp_warmup'] += 1
        b, rows, _ = run_fake_block(FakeServer(s, 128), 'CACHED')
        self.assertTrue(b['refused'] and 'warm-up tokens' in b['end_reason'] and rows == [], b['end_reason'])

    def test_wall_time_is_the_request_alone(self):
        """wall_s ends when the reply arrives: grading and validation come after it."""
        grade = mem1.scored_fields
        with mock.patch.object(mem1, 'scored_fields', lambda *a: (time.sleep(0.2), grade(*a))[1]):
            b, rows, _ = run_fake_block(FakeServer(fake_set(), 128, RIGHT), 'FRESH')
        self.assertEqual(b['end_reason'], 'all questions done')
        self.assertTrue(all(r['wall_s'] < 0.2 for r in rows), [r['wall_s'] for r in rows])

    def test_server_stopped_and_sampler_cleaned_up_on_error(self):
        srv = FakeServer(fake_set(), 128)
        srv.post = mock.Mock(side_effect=RuntimeError('boom'))
        stopped, threads = [], []

        def sampler(t0, log, pid, stop, samples, state):
            threads.append(threading.current_thread())
            stop.wait(5)
        with mock.patch.object(cm, 'spawn', return_value=FakeProc()), \
                mock.patch.object(cm, 'stop', side_effect=stopped.append), \
                mock.patch.object(cr, 'healthy', return_value=True), mock.patch.object(cr, 'post', srv.post), \
                mock.patch.object(mem1, 'sampler_loop', sampler), \
                mock.patch.object(mem1, 'proc_mem', return_value={}), self.assertRaises(RuntimeError):
            mem1.run_block(0, 0, 128, 'FRESH', srv.s, 2048, tempfile.mkdtemp(), lambda r: None, '/x', False)
        self.assertEqual(len(stopped), 1)
        threads[0].join(1)
        self.assertFalse(threads[0].is_alive())  # the stop event ended the sampler despite the exception

    def test_request_failure_ends_block(self):
        srv = FakeServer(fake_set(), 128)
        real = srv.post

        def post(path, body, timeout=600):
            if path == '/completion' and 'Oskar' in body['prompt'][-80:]:
                raise TimeoutError('timed out')
            return real(path, body, timeout)
        srv.post = post
        b, rows, _ = run_fake_block(srv, 'FRESH', seq=0)
        self.assertIn('failed', b['end_reason'])
        self.assertFalse(rows[-1]['valid'])
        self.assertEqual(rows[-1]['outcome'], 'ERROR')
        self.assertIn('error', rows[-1])


class Telemetry(Offline):
    """Asynchronous, bounded, self-measured telemetry; nothing on the request path."""

    def test_no_sensor_work_on_the_request_path(self):
        """Every observation runs on the sampler thread; every request on the runner's thread; between requests
        only in-memory checks."""
        tel, srv = FakeTelemetry(), FakeServer(fake_set(), 256, RIGHT)
        b, rows, _ = run_fake_block(srv, 'CACHED', tel=tel)
        main = threading.current_thread()
        self.assertTrue(tel.threads and all(t is not main for t in tel.threads))
        self.assertTrue(srv.threads and all(t is main for t in srv.threads))
        self.assertEqual(finish(b, rows, 0)['problems'], [])

    def test_requests_never_wait_for_a_stuck_sample_and_cleanup_is_bounded(self):
        release = threading.Event()
        tel = FakeTelemetry(block=release)
        with mock.patch.object(mem1, 'ROOT_TIMEOUT_S', 0.1), mock.patch.object(mem1, 'JOIN_S', 5.1):
            began = time.monotonic()
            b, rows, _ = run_fake_block(FakeServer(fake_set(), 128, RIGHT), 'FRESH', tel=tel)
            took = time.monotonic() - began
        release.set()
        self.assertEqual(b['end_reason'], 'all questions done')  # every request ran while the sample was stuck
        self.assertEqual(len(rows), 7)
        self.assertLess(took, 7)  # join bounded at JOIN_S (patched to 0.1 + 5); the sample is stuck 30 s
        self.assertTrue(b['sampler']['alive_after_join'])
        why = finish(b, rows, 0)['problems']
        self.assertTrue({'SAMPLES_EMPTY', 'SAMPLER_STATE'} <= codes(why), details(why))  # never a valid block

    def test_telemetry_failure_ends_the_block_fail_closed(self):
        b, rows, _ = run_fake_block(slowed(FakeServer(fake_set(), 128, RIGHT)), 'FRESH',
                                    tel=FakeTelemetry(fail_after=1), period=0.001)
        self.assertIn('telemetry failed (RuntimeError: sensor exploded): fail closed', b['end_reason'])
        why = finish(b, rows, 0)['problems']
        self.assertTrue({'SAMPLER_FAILED', 'ENDED_EARLY'} <= codes(why), details(why))

    def test_safety_stop_from_the_latest_sample(self):
        b, rows, _ = run_fake_block(FakeServer(fake_set(), 128, RIGHT), 'FRESH', tel=FakeTelemetry(status=3))
        self.assertIn('safety stop: Android status 3', b['end_reason'])
        self.assertLess(len(rows), 7)

    def test_between_requests_failed_missing_or_stale_safety_reads_fail_closed(self):
        """Round 9 finding 3: observe returns a failed read (it never raises for one); such a sample, no sample
        after SAFETY_MAX_AGE_S or a stale latest sample ends the block."""
        alive, dead, st = mock.Mock(**{'is_alive.return_value': True}), \
            mock.Mock(**{'is_alive.return_value': False}), {'attempts': 1, 'dropped': 0, 'missed': 0}
        ok, stop = sample(15.0), mem1.telemetry_stop
        self.assertEqual(mem1.SAFETY_MAX_AGE_S, 30)
        self.assertIsNone(stop(alive, st, [ok], 20.0))
        self.assertIsNone(stop(alive, st, [], 10.0))  # the slot-0 sample is still running
        self.assertIn('no safety sample begun in the last 30 s', stop(alive, st, [], 31.0))
        self.assertIn('no safety sample begun in the last 30 s', stop(alive, st, [ok], 46.0))  # stuck since t 15
        timed_out = {**ok, 'root_rc': None, 'battery_rc': None, 'dump_rc': None, 'status': None, 'battery_c': None}
        for x in (timed_out, {**ok, 'status': None}, {**ok, 'battery_c': 'nan'}, {**ok, 'battery_rc': 1},
                  {**ok, 'dump_rc': 255}):
            self.assertIn('safety reads failed', stop(alive, st, [x], 20.0), x)
        self.assertIn('safety stop', stop(alive, st, [{**ok, 'battery_c': 45.0}], 20.0))
        self.assertIn('telemetry failed', stop(dead, st, [ok], 20.0))

    def test_a_su_timeout_latches_and_stops_the_sampler_at_once(self):
        """Round 10 finding 1: after a timed-out su call no further observation starts in that block."""
        samples, state, stop, calls = [], {'attempts': 0, 'dropped': 0, 'missed': 0}, threading.Event(), []

        def timeout(log, pid=None):
            calls.append(pid)
            return obs(pid, root_rc=None, status=None, diag=[{'op': 'su', 'kind': 'timeout'}])
        with mock.patch.object(mem1, 'observe', timeout), mock.patch.object(mem1, 'pin_sampler', return_value={}), \
                mock.patch.object(mem1, 'SAMPLE_PERIOD_S', 0.01):
            th = threading.Thread(target=mem1.sampler_loop, args=(time.monotonic(), '/x', PID, stop, samples, state))
            th.start()
            th.join(2)
        self.assertFalse(th.is_alive())  # ended by itself: stop was never set
        self.assertEqual((len(calls), state['attempts'], state['root_timeout']), (1, 1, True))
        alive = mock.Mock(**{'is_alive.return_value': True})
        self.assertIn('a su call timed out', mem1.telemetry_stop(alive, state, samples, 1.0))

    def test_known_failures_allow_no_further_request(self):
        """Round 10 finding 2: the in-memory check runs before the warm-up and before every request."""
        srv = FakeServer(fake_set(), 128, RIGHT)

        def failed(t0, log, pid, stop, samples, state):
            state['error'] = 'RuntimeError: sensor exploded'  # failed during the load and checks
        tel = FakeTelemetry()
        tel.first.set()  # healthy at once: no sample ever arrives
        b, rows, _ = run_fake_block(srv, 'FRESH', sampler=failed, tel=tel)
        self.assertIn('telemetry failed (RuntimeError: sensor exploded): fail closed: block ended before the warm-up',
                      b['end_reason'])
        self.assertEqual((rows, srv.calls.count('/completion')), ([], 0))
        st, ok = {'attempts': 1, 'dropped': 0, 'missed': 0}, sample(15.0)
        alive = mock.Mock(**{'is_alive.return_value': True})
        self.assertIsNone(mem1.safety_stop(ok))
        for x in ({**ok, 'dump_rc': 255}, {**ok, 'root_rc': 1}, {**ok, 'battery_c': None}):
            self.assertIn('safety reads failed', mem1.safety_stop(x))
            self.assertEqual(mem1.telemetry_stop(alive, st, [x], 20.0), mem1.safety_stop(x))

    def test_failed_safety_reads_end_the_block(self):
        b, rows, _ = run_fake_block(slowed(FakeServer(fake_set(), 128, RIGHT)), 'FRESH',
                                    tel=FakeTelemetry(status=None))
        self.assertIn('safety reads failed', b['end_reason'])
        self.assertLess(len(rows), 7)

    def test_failed_attempts_and_unmeasured_cost_are_counted(self):
        """Round 9 finding 5: an observation that raised counts as an attempt and a failure; a cost figure missing
        from an attempt makes its total a marked lower bound, never a measured zero."""
        b, rows, _ = run_fake_block(slowed(FakeServer(fake_set(), 128, RIGHT)), 'FRESH',
                                    tel=FakeTelemetry(fail_after=1), period=0.001)
        sp = b['sampler']
        self.assertEqual(sp['attempts'], len(b['samples']) + sp['dropped'] + 1)
        why = finish(b, rows, 0)['problems']
        self.assertIn('raised or never returned', details(why))
        tm = mem1.telemetry(b, rows)
        self.assertEqual((tm['attempts'], tm['unrecorded'], tm['failed']), (sp['attempts'], 1, 1))
        lines = '\n'.join(mem1.telemetry_lines([b], rows))
        self.assertIn('+?', lines)
        self.assertIn('unknown, not zero', lines)
        env = Environment()
        self.assertNotIn('+?', '\n'.join(mem1.telemetry_lines([env.block()], [])))
        gap = env.block(samples=env.with_sample(1, root_cpu_s=DROP))
        self.assertIn('+?', '\n'.join(mem1.telemetry_lines([gap], [])))
        self.assertIsNone(mem1.telemetry(env.block(sampler={'dropped': 0}), [])['attempts'])

    def test_buffer_overflow_is_counted_never_reconstructed(self):
        with mock.patch.object(mem1, 'MAX_SAMPLES', 2):
            b, rows, _ = run_fake_block(slowed(FakeServer(fake_set(), 128, RIGHT)), 'FRESH', period=0.001)
        self.assertEqual(len(b['samples']), 2)
        self.assertGreater(b['sampler']['dropped'], 0)
        self.assertIn('SAMPLES_LOST', codes(finish(b, rows, 0)['problems']))

    def test_sampler_loop_schedule_measures_itself_and_skips_overrun_slots(self):
        samples, state, stop = [], {'attempts': 0, 'dropped': 0, 'missed': 0}, threading.Event()

        def slow(log, pid=None):
            time.sleep(0.12)  # longer than two periods
            return obs(pid)
        with mock.patch.object(mem1, 'observe', slow), mock.patch.object(mem1, 'pin_sampler', return_value={}), \
                mock.patch.object(mem1, 'SAMPLE_PERIOD_S', 0.05):
            th = threading.Thread(target=mem1.sampler_loop, args=(time.monotonic(), '/x', PID, stop, samples, state))
            th.start()
            time.sleep(0.45)
            stop.set()
            th.join(2)
        self.assertFalse(th.is_alive())
        slots = [x['slot'] for x in samples]
        self.assertEqual(slots, sorted(set(slots)))  # never bunched, never repeated
        self.assertGreater(state['missed'], 0)
        self.assertTrue(all(x['t_end'] >= x['t'] >= x['t_sched'] - 0.01 and 'cpu_s' in x for x in samples))

    def test_pin_sampler_records_what_it_got(self):
        with mock.patch.object(os, 'sched_setaffinity', side_effect=OSError('EINVAL')), \
                mock.patch.object(os, 'setpriority'), mock.patch.object(os, 'getpriority', return_value=0), \
                mock.patch.object(os, 'sched_getaffinity', return_value={0, 1, 2, 3, 4, 5}):
            st = mem1.pin_sampler()
        self.assertEqual((st['affinity_error'], st['cpus'], st['nice']), ('EINVAL', [0, 1, 2, 3, 4, 5], 0))
        self.assertEqual((st['requested_cpus'], st['requested_nice']), ([0, 1, 2, 3], 10))

    def root_out(self, battery='90\nDischarging\n300\n-500000\n4000000\n', brc=0, dump='Thermal Status: 0\n',
                 drc=0, smaps='Rss:  3100000 kB\nPss:  3000000 kB\n', src=0):
        return (f'=== battery\n{battery}rc={brc}\n=== thermalservice\n{dump}'
                'Current temperatures from HAL:\n\tTemperature{mValue=25.5, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\n'
                f'rc={drc}\n=== smaps_rollup\n{smaps}rc={src}\n=== sched\nCpus_allowed_list:\t0-3\n'
                '1234 (sh) S 1 1 0 0 -1 0 0 0 0 0 3 2 4 1 30 10 1 0\nrc=0\n')

    def observe(self, root, thermal=None, pid=PID):
        with mock.patch.object(mem1, 'server_caps', return_value={'policy4': {'max': 2350000},
                                                                  'policy6': {'max': 2850000}}), \
                mock.patch.object(repro, 'cpufreq', return_value=cond()['cpufreq']), \
                mock.patch.object(cr, 'meminfo_mib', return_value={'mem_available_mib': 5000, 'swap_used_mib': 0,
                                                                   'cached_mib': 1}), \
                mock.patch.object(mem1, 'thermal_tail', **(thermal or {'return_value': {
                    'at': f'{datetime.now(timezone.utc) - timedelta(seconds=3):%Y-%m-%dT%H:%M:%SZ}',
                    'thermal_age_s': 3.0, 'z9': 1, 'z10': 1, 'z11': 1}})), \
                mock.patch.object(cr, 'root', **root) as r, mock.patch.object(cr, 'root_stderr',
                                                                             return_value='su: denied'):
            return mem1.observe('/x', pid), r

    def test_observe_one_root_call_pinned_and_parsed(self):
        x, r = self.observe({'return_value': (0, self.root_out())})
        script = r.call_args[0][0]
        self.assertEqual(r.call_count, 1)
        self.assertTrue(script.startswith('taskset -p f $$') and 'renice -n 10 -p $$' in script)
        self.assertIn(f'cat /proc/{PID}/smaps_rollup', script)
        self.assertEqual(r.call_args[1]['timeout'], mem1.ROOT_TIMEOUT_S)
        self.assertEqual((x['battery_pct'], x['battery_c'], x['battery_w'], x['status'], x['skin'], x['pss_kb']),
                         (90, 30.0, 2.0, 0, 25.5, 3000000))
        self.assertEqual((x['root_cpus'], x['root_nice']), ('0-3', 10))
        self.assertAlmostEqual(x['root_cpu_s'], 10 / os.sysconf('SC_CLK_TCK'))
        self.assertNotIn('diag', x)
        self.assertEqual(mem1.observation_problems({**x, 't': 1.0, 't_end': 1.2}, 's', {}, PID), [])

    def test_failed_reads_keep_bounded_diagnostics(self):
        x, _ = self.observe({'return_value': (1, self.root_out(brc=1, drc=255, dump='x' * 5000, src=1, smaps=''))},
                            {'side_effect': ValueError('thermal log x last line is 95 s old')})
        kinds = {(d['op'], d['kind']) for d in x['diag']}
        self.assertTrue({('su', 'non-zero return code'), ('battery', 'non-zero return code'),
                         ('dumpsys thermalservice', 'non-zero return code'),
                         ('smaps_rollup', 'non-zero return code')} <= kinds, kinds)
        self.assertTrue(all(len(d['raw']) <= mem1.DIAG_CHARS and d['stderr'] == 'su: denied' and d['utc']
                            and d['rc'] is not None for d in x['diag']))
        why = mem1.observation_problems({**x, 't': 1.0, 't_end': 1.2}, 's', {}, PID)
        for text in ('s: su rc 1', 's: battery read rc 1', 's: dumpsys thermalservice rc 255',
                     's: thermal log: ValueError: thermal log x last line is 95 s old'):
            self.assertIn(text, why)
        self.assertTrue(any('PSS missing or invalid' in w for w in why))
        x, _ = self.observe({'side_effect': subprocess.TimeoutExpired('su', 10)})
        self.assertEqual([(d['op'], d['kind']) for d in x['diag']][0], ('su', 'timeout'))
        self.assertIsNone(x['root_rc'])
        self.assertIn('s: su rc None', mem1.observation_problems(x, 's', {}))
        x, _ = self.observe({'return_value': (0, self.root_out(battery='90\nDischarging\n'))}, pid=None)
        self.assertIn(('battery', 'unparseable'), {(d['op'], d['kind']) for d in x['diag']})
        self.assertNotIn('pss_kb', x)  # a pre/postflight reading reads no server

    def test_thermal_tail_reads_only_the_end_and_rejects_stale(self):
        log = Path(tempfile.mkdtemp()) / 'thermal.log'
        now = datetime.now(timezone.utc)
        old = (now - timedelta(seconds=95)).strftime('%Y-%m-%dT%H:%M:%SZ')
        log.write_text('garbage\n' * 2000 + f'{now:%Y-%m-%dT%H:%M:%SZ} z9=31000 z10=32000 z11=33000 x=1\n')
        with mock.patch.object(mem1, 'thermal_tail', ORIGINAL_TAIL):
            self.assertEqual(mem1.thermal_tail(log)['z11'], 33000)
            log.write_text(f'{old} z9=31000 z10=32000 z11=33000\n')
            with self.assertRaises(ValueError):
                mem1.thermal_tail(log)

    def test_report_shows_measured_cost_and_overlap(self):
        b = Environment().block()
        rows = [{'block': 0, 't_start': 14.0, 't_end': 16.0}, {'block': 0, 't_start': 20.0, 't_end': 21.0}]
        tm = mem1.telemetry(b, rows)
        self.assertEqual((tm['attempts'], tm['failed'], tm['dropped'], tm['missed']), (4, 0, 0, 0))
        self.assertEqual((tm['interval_med'], tm['interval_max']), (15.0, 15.0))
        self.assertAlmostEqual(tm['overlap_s'], 0.3)  # the t 15 sample (15.0-15.3) inside the first request
        self.assertEqual((tm['requests_overlapped'], tm['requests']), (1, 2))
        self.assertAlmostEqual(tm['load_overlap_s'], 0.3)  # the slot-0 sample during the 5 s load
        self.assertEqual(tm['root_shell'], ['0-3/10'])
        self.assertIn('NOT a measurement of its effect on inference', '\n'.join(mem1.telemetry_lines([b], rows)))


class Environment(Offline):
    """validate_block: field-level validity, full-block coverage of the samples, INVALID vs UNMATCHED."""

    def block(self, seq=0, samples=None, sampler=None, start=None, end=None, skin=25.0, rows=(), **kw):
        """A synthetic recorded block: server t0 = mono 100 s, 50 s long, loaded at 5 s; samples every 15 s from
        t 0; sampling ended at t 49."""
        samples = [sample(float(t), skin=skin) for t in (0, 15, 30, 45)] if samples is None else samples
        b = {'block': seq, 'round': 0, 'length': 128, 'mode': 'FRESH', 'prime': 'prime',
             'questions': [q['id'] for q in S['questions']], 'gate': {'skin': skin, 'warm_start': False},
             'end_reason': 'all questions done', 'start': cond(skin=skin, mono=99.0) if start is None else start,
             'end': cond(skin=skin, mono=151.0) if end is None else end, 'mono_t0': 100.0, 'server_s': 50.0,
             'sampling_end_s': 49.0,
             'load_s': 5.0, 'samples': samples, 'sampler': {'ended_s': 49.1, 'alive_after_join': False, 'dropped': 0,
                                                            'missed': 0, 'attempts': len(samples)}
             if sampler is None else sampler, 'server_pid': PID,
             'server_start': {'Cpus_allowed_list': '4-7'}, 'server_end': {'Cpus_allowed_list': '4-7'}, 'ctx': 2048,
             'checks': good_checks(128, [q['id'] for q in S['questions']]),
             'agents': {'before': '', 'after': ''}}
        b.update(kw)
        b['problems'] = mem1.validate_block(b, list(rows), 2048, FAKE, False)
        return b

    def with_sample(self, i, **kw):
        xs = self.block()['samples']
        xs[i] = {**xs[i], **kw}
        return xs

    def test_healthy_block_has_no_problems(self):
        self.assertEqual(self.block()['problems'], [])

    def test_reasons_are_machine_readable(self):
        why = self.block(samples=[])['problems']
        self.assertTrue(all(set(w) == {'code', 'detail'} for w in why))
        self.assertEqual(codes(why), {'SAMPLES_EMPTY'})

    def test_empty_samples_and_first_last_and_middle_gaps(self):
        self.assertIn('SAMPLES_EMPTY', codes(self.block(samples=[])['problems']))
        for ts, gap in (((21.0, 30.0, 45.0), '[21.0]'), ((0.0, 15.0, 40.0, 45.0), '[25.0]'), ((0.0, 15.0), '[34.0]')):
            why = self.block(samples=[sample(t) for t in ts])['problems']
            self.assertTrue(any(w['code'] == 'SAMPLE_GAP' and gap in w['detail'] for w in why), (ts, why))
        why = self.block(samples=[sample(t) for t in (0.0, 15.0, 30.0, 45.0, 50.0)])['problems']  # after the end
        self.assertIn('after the sampler ended', details(why))
        why = self.block(samples=[sample(t) for t in (0.0, 30.0, 15.0, 45.0)])['problems']  # out of order
        self.assertIn('SAMPLE_GAP', codes(why))
        why = self.block(samples=[sample(0.0), sample(15.0, root_rc=None, diag=[{'op': 'su', 'kind': 'timeout'}]),
                                  sample(30.0), sample(45.0)])['problems']  # a failed sample leaves a 30 s gap
        self.assertTrue({'SAMPLE_INVALID', 'SAMPLE_GAP'} <= codes(why), details(why))
        self.assertEqual(mem1.SAMPLE_MAX_GAP_S, 20)  # a missed 15 s slot is never covered

    def test_sampler_failure_termination_and_lost_samples(self):
        why = self.block(sampler={'error': 'KeyError: x', 'ended_s': 12.0, 'alive_after_join': False,
                                  'dropped': 0})['problems']
        self.assertTrue({'SAMPLER_FAILED', 'SAMPLER_STATE'} <= codes(why))
        self.assertIn('unexpected termination', details(why))
        why = self.block(sampler={'ended_s': 49.1, 'alive_after_join': True, 'dropped': 0})['problems']
        self.assertIn('did not stop', details(why))
        self.assertTrue({'SAMPLER_STATE', 'SAMPLES_LOST'} <= codes(self.block(sampler={})['problems']))
        why = self.block(sampler={'ended_s': 49.1, 'alive_after_join': False, 'dropped': 3, 'attempts': 7})['problems']
        self.assertEqual(codes(why), {'SAMPLES_LOST'})

    def test_failed_reads_are_rejected_even_with_parseable_output(self):
        for kw, text in (({'root_rc': 1}, 'su rc 1'), ({'battery_rc': 1}, 'battery read rc 1'),
                         ({'dump_rc': 255}, 'dumpsys thermalservice rc 255'), ({'smaps_rc': 1}, 'PSS missing'),
                         ({'pss_pid': 999}, 'PSS missing'), ({'pss_kb': DROP}, 'PSS missing'),
                         ({'meminfo_error': 'KeyError'}, 'MemAvailable')):
            why = self.block(samples=self.with_sample(1, **kw))['problems']
            self.assertIn('SAMPLE_INVALID', codes(why), kw)
            self.assertIn(text, details(why), kw)
        why = self.block(start=cond(mono=99.0, battery_rc=1))['problems']
        self.assertIn('READING_INVALID start: battery read rc 1', details(why))

    def test_non_finite_and_mistyped_values_are_never_valid(self):
        for bad in (float('nan'), float('inf'), float('-inf'), True, 'nan', None):
            for field in ('t', 't_end', 'battery_w', 'mem_available_mib', 'skin', 'battery_c', 'mono'):
                why = self.block(samples=self.with_sample(2, **{field: bad}))['problems']
                self.assertIn('SAMPLE_INVALID', codes(why), (field, bad))
            for field in ('mono', 'skin', 'battery_c'):
                why = self.block(end=cond(**{'mono': 151.0, field: bad}))['problems']
                self.assertTrue(codes(why) & {'READING_INVALID', 'READING_ORDER'}, (field, bad))
            for field in ('mono_t0', 'server_s', 'sampling_end_s'):
                self.assertIn('TIMES_INVALID', codes(self.block(**{field: bad})['problems']), (field, bad))
        for field, v in (('root_rc', False), ('status', False), ('battery_pct', True), ('pss_kb', True),
                         ('pss_kb', 3000000.0), ('battery_pct', 90.0), ('utc', 'yesterday'), ('utc', 5)):
            self.assertIn('SAMPLE_INVALID', codes(self.block(samples=self.with_sample(2, **{field: v}))['problems']),
                          (field, v))
        for field, v in (('battery_w', 99.0), ('mem_available_mib', -1), ('pss_kb', 0), ('skin', 85.0),
                         ('battery_pct', 101)):
            self.assertIn('SAMPLE_INVALID', codes(self.block(samples=self.with_sample(2, **{field: v}))['problems']),
                          (field, v))

    def test_stale_missing_and_malformed_observations(self):
        why = self.block(samples=self.with_sample(1, thermal_error='ValueError: last line is 95 s old',
                                                  at=DROP))['problems']
        self.assertIn('thermal log: ValueError: last line is 95 s old', details(why))
        why = self.block(samples=self.block()['samples'] + ['oops', 7])['problems']
        self.assertIn('sample is not an object (str)', details(why))
        bare = self.block(start={}, end={}, gate={}, sampler={}, samples=[{}], server_start={}, server_end={},
                          server_pid=None, mono_t0=None)['problems']  # every nested key missing: reasons, no crash
        self.assertTrue({'WARM_START', 'READING_INVALID', 'TIMES_INVALID', 'SAMPLE_INVALID', 'SAMPLES_EMPTY',
                         'SAMPLER_STATE', 'SAMPLES_LOST', 'SERVER_CORES'} <= codes(bare), codes(bare))
        why = self.block(start={'mono': 99.0})['problems']
        for text in ('start: su rc None', 'start: battery % None', 'start: Android status None',
                     'start: VIRTUAL-SKIN None', 'start: policy4 cap None (missing or not a positive int)',
                     'start: thermal log stamp None missing or malformed'):
            self.assertIn(text, details(why))

    def test_thermal_freshness_is_recomputed_from_its_evidence(self):
        """Round 9 finding 1: a stale or malformed thermal-log stamp is invalid without any stored error marker."""
        for kw, text in (({'thermal_age_s': 95.0}, 'stale'), ({'thermal_age_s': 'nan'}, 'stale'),
                         ({'thermal_age_s': DROP}, 'stale'), ({'at': 'garbage'}, 'missing or malformed'),
                         ({'at': DROP}, 'missing or malformed'),
                         ({'at': '2026-10-24T23:00:00Z'}, 'disagree with the reading time')):
            why = self.block(samples=self.with_sample(1, **kw))['problems']
            self.assertIn('SAMPLE_INVALID', codes(why), kw)
            self.assertIn(text, details(why), kw)
        why = self.block(start=cond(mono=99.0, **stamps(99.0, age=95.0)))['problems']
        self.assertIn('READING_INVALID', codes(why))
        self.assertEqual(self.block(start=cond(mono=99.0, **stamps(99.0, age=29.0)))['problems'], [])

    def test_coverage_is_tied_to_requests_sampler_and_clocks(self):
        """Round 9 finding 1: sampling must reach the last request; every sample's mono must be mono_t0 + t, and
        end before the sampler ended; every reading's UTC must follow its monotonic time."""
        at = {'round': 0, 'length': 128, 'mode': 'FRESH', 'valid': True}  # the block's
        rows = [{**at, 'question': 'q_begin', 't_start': 40.0, 't_end': 48.0}]
        self.assertEqual(self.block(rows=rows)['problems'], [])
        why = self.block(rows=[{**rows[0], 'length': 0}])['problems']
        self.assertIn('record another round, memory length or mode than their block', details(why))
        ids = [q['id'] for q in S['questions']]
        for kw, text in (({'load_s': None}, 'load_s None'),
                         ({'checks': {}}, 'template checks None are not a passed check'),
                         ({'checks': {**good_checks(128, ids), 'template': {'prime': False}}}, 'CHECKS_FAILED'),
                         ({'checks': {**good_checks(128, ids), 'n_ctx': 4096}}, 'CHECKS_FAILED')):
            self.assertIn(text, details(self.block(**kw)['problems']), kw)
        why = self.block(samples=self.with_sample(1, swap_used_mib=-5))['problems']
        self.assertIn('swap used -5 MiB', details(why))
        short = self.block(rows=rows, sampling_end_s=30.0, samples=[sample(t) for t in (0.0, 15.0, 29.0)],
                           sampler={'ended_s': 30.1, 'alive_after_join': False, 'dropped': 0, 'attempts': 3})
        self.assertIn('before the last request ended (48.0 s)', details(short['problems']))
        why = self.block(samples=self.with_sample(1, mono=50.0))['problems']
        self.assertIn('is not mono_t0 + t', details(why))
        why = self.block(samples=self.with_sample(3, t_end=60.0))['problems']
        self.assertIn('after the sampler ended', details(why))
        why = self.block(samples=self.with_sample(2, **stamps(250.0)))['problems']  # UTC 120 s ahead of its mono
        self.assertIn('disagree with their monotonic times', details(why))
        late = [sample(t, t_end=500.0) for t in (0.0, 15.0, 30.0, 45.0)]  # Round 10 finding 3
        why = details(self.block(samples=late, sampler={'ended_s': 500.0, 'alive_after_join': False, 'dropped': 0,
                                                        'attempts': 4}, end=cond(mono=601.0))['problems'])
        for text in ('took 500.0 s (an observation is at most 15 s)', 'after its bounded join (50.0 + 15 s)'):
            self.assertIn(text, why)
        why = details(self.block(samples=self.with_sample(1, t_end=31.0))['problems'])
        self.assertIn('begun before the previous sample ended (31.0 s; the sampler is serial)', why)
        why = details(self.block(sampler={'ended_s': 52.0, 'alive_after_join': False, 'dropped': 0, 'attempts': 4},
                                 end=cond(mono=151.0))['problems'])
        self.assertIn('not after the sampler ended', why)
        race = self.block(samples=[sample(t) for t in (0.0, 15.0, 30.0, 45.0, 49.5)],
                          sampler={'ended_s': 50.0, 'alive_after_join': False, 'dropped': 0, 'attempts': 5})
        self.assertEqual(race['problems'], [])  # begun after the sampling end (stop race): valid, covers nothing
        overlap = [{**at, 'question': 'q_begin', 'position': 1, 't_start': 10.0, 't_end': 20.0},
                   {**at, 'question': 'q_mid', 'position': 2, 't_start': 15.0, 't_end': 25.0}]
        self.assertIn('requests overlap', details(self.block(rows=overlap)['problems']))
        early = [{**at, 'question': 'prime', 'position': 0, 't_start': 2.0, 't_end': 3.0}]
        self.assertIn('before the model load ended', details(self.block(rows=early)['problems']))
        why = self.block(sampler={'ended_s': 49.1, 'alive_after_join': False, 'dropped': 0, 'attempts': 5})
        self.assertIn('5 sample attempts != 4 recorded + 0 dropped', details(why['problems']))

    def test_reading_order_and_rounding_allowance(self):
        self.assertIn('READING_ORDER', codes(self.block(start=cond(mono=101.0))['problems']))
        self.assertIn('READING_ORDER', codes(self.block(start=cond(mono=30.0))['problems']))  # stale preflight
        self.assertIn('READING_ORDER', codes(self.block(end=cond(mono=149.997))['problems']))  # 3 ms early
        self.assertIn('READING_ORDER', codes(self.block(end=cond(mono=215.0))['problems']))  # stale postflight
        self.assertEqual(self.block(end=cond(mono=150.001), mono_t0=100.001, server_s=50.001)['problems'], [])

    def test_cap_change_and_mid_block_thermal_change(self):
        why = self.block(samples=self.with_sample(2, cpufreq={'policy4': {'max': 2350000},
                                                              'policy6': {'max': 2000000}}))['problems']
        self.assertIn('policy6 cap 2850000 -> 2000000', details(why))
        self.assertIn('Android status 1', details(self.block(samples=self.with_sample(2, status=1))['problems']))
        why = self.block(end=cond(bstat='Charging', bat_c=45.0, mono=151.0))['problems']
        self.assertIn("end: battery status 'Charging'", details(why))
        self.assertIn('end: battery 45.0 degC', details(why))
        warm = self.block(1, samples=self.with_sample(2, skin=26.6))  # valid, but the skin rose mid-block
        self.assertEqual(warm['problems'], [])
        self.assertTrue(any('skin 25.00-26.60' in w for w in mem1.condition_diffs([self.block(0), warm])))
        self.assertTrue(any('skin 25.00-26.60' in w for w in mem1.condition_diffs([warm])))  # within one block

    def test_invalid_or_missing_telemetry_never_comparable(self):
        good, bad = self.block(0), self.block(1, samples=[])
        side = lambda b: {'rows': [{'block': b['block'], 'question': 'q', 'valid': True, 'prompt_ms': 1.0,
                                    'wall_s': 1.0, 'prompt_n': 1, 'cache_n': 0, 'gen_tok_s': None}],
                          'expected': ['q'], 'repeats': 1, 'blocks': [b]}
        self.assertEqual(mem1.compare('x', 'v', [], {}, [side(good), side(self.block(1))], [])['verdict'],
                         'COMPARABLE')
        c = mem1.compare('x', 'v', [], {}, [side(good), side(bad)], [])
        self.assertEqual((c['verdict'], c['environment_valid'], c['ratios']), ('NOT COMPARABLE', False, None))
        self.assertIn('ENV_INVALID', [r['code'] for r in c['reasons']])
        missing = self.block(1, samples=self.with_sample(2, skin=None))
        c = mem1.compare('x', 'v', [], {}, [side(good), side(missing)], [])
        self.assertTrue({'ENV_INVALID', 'ENV_UNMATCHED'} <= {r['code'] for r in c['reasons']})
        self.assertIn('a skin reading is missing', mem1.condition_diffs([good, missing]))
        caps = self.block(1, start=cond(cap6=None, mono=99.0))
        self.assertIn('a CPU cap reading is missing', mem1.condition_diffs([good, caps]))

    def test_battery_threshold_kept_at_10_percent(self):
        self.assertEqual(mem1.BATTERY_MATCH_PCT, 10)
        a = self.block(0, start=cond(pct=90, mono=99.0))
        self.assertEqual(mem1.condition_diffs([a, self.block(1, end=cond(pct=80, mono=151.0))]), [])
        why = mem1.condition_diffs([a, self.block(1, end=cond(pct=79, mono=151.0))])
        self.assertIn('battery 79-90 % (> 10 apart)', why)
        self.assertIn('a battery % reading is missing',
                      mem1.condition_diffs([a, self.block(1, end=cond(pct=None, mono=151.0))]))

    def test_caps_compared_on_server_cores_only(self):
        a, b = self.block(0), self.block(1)
        b['start']['cpufreq']['policy0']['max'] = 1000000
        self.assertEqual(mem1.condition_diffs([a, b]), [])

    def test_old_block_reasons_kept(self):
        why = self.block(start=cond(status=1, mono=99.0), gate={'skin': 25.0, 'warm_start': True},
                         end_reason='cores lost', server_start={'Cpus_allowed_list': '0-7'},
                         rows=[{'question': 'q_mid', 'valid': False},
                               {'question': 'q_end', 'valid': 'yes'}])['problems']
        for text in ('WARM_START warm start', 'ENDED_EARLY cores lost', 'start: Android status 1',
                     "SERVER_CORES server_start Cpus_allowed_list '0-7'", "invalid requests ['q_mid', 'q_end']"):
            self.assertIn(text, details(why))


def write_run(folder, s, exp, ctx, plan, smoke=False, skins=None, result=True, warm=(), drop_blocks=(),
              drop_requests=(), end_reason='all blocks done', skipped=(), answers=None, **run_kw):
    """A run folder from fake blocks, as main writes it; drop_* simulate lost work (a dropped block keeps no
    requests unless drop_blocks is combined with keep_orphans=True in run_kw)."""
    keep_orphans = run_kw.pop('keep_orphans', False)
    folder.mkdir(parents=True)
    rows_all, blocks = [], []
    for seq, (rnd, length, mode) in enumerate(plan):
        b, rows, _ = run_fake_block(FakeServer(s, length, answers or RIGHT, n_ctx=ctx), mode, seq=seq, rnd=rnd,
                                    smoke=smoke, out=folder, ctx=ctx)
        rows = [r for r in rows if (seq, r['question']) not in drop_requests]
        if seq in drop_blocks:
            rows_all += rows if keep_orphans else []
            continue
        blocks.append(finish(b, rows, seq, (skins or {}).get(seq, 25.0), seq in warm, s, smoke,
                             **({'samples': [{**x, 'skin': skins[seq]} for x in b['samples']]} if seq in (skins or {})
                                else {})))
        rows_all += rows
    run = {'task': s['task'], 'benchmark': 'MEM-1', 'exp': exp, 'ctx': ctx, 'rounds': 1 if smoke else 2,
           'smoke': smoke, 'schedule': plan, 'matrix': mem1.matrix(S, plan, smoke), 'prime': S['prime'],
           'command': mem1.command(ctx), **mem1.configuration(), 'set_sha256': mem1.SET_SHA256, **PROVENANCE}
    run.update(run_kw)
    (folder / 'run.json').write_text(mem1.dumps(run))
    (folder / 'requests.jsonl').write_text(''.join(mem1.dumps(r) + '\n' for r in rows_all))
    (folder / 'blocks.jsonl').write_text(''.join(mem1.dumps(b) + '\n' for b in blocks))
    if result:
        (folder / 'result.json').write_text(json.dumps({'end_reason': end_reason, 'agents_end': '',
                                                       'port_free_end': True,
                                                       'skipped': [{'block': x, 'reason': 'cores'} for x in skipped]}))
    return rows_all, blocks


def comparison_lines(text):
    return [l for l in text.splitlines() if 'COMPARABLE' in l and l.startswith('  ')]


class Report(FakeFrozenSet, Offline):
    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())
        self.s = fake_set()

    def test_primary_complete_comparable_and_consistent(self):
        rows, blocks = write_run(self.tmp / 'a', self.s, 'primary', 2048, mem1.schedule('primary', 2, False))
        text, recs = mem1.report([self.tmp / 'a'])
        self.assertIn('[COMPLETE]', text)
        self.assertIn('COMPLETED 20, FAILED 0, SKIPPED 0, MISSING 0 of 20 planned blocks', text)
        self.assertIn('TTFT PROXY', text)
        self.assertEqual(len(recs), 3 * 4 + 5)
        self.assertTrue(all(r['verdict'] == 'COMPARABLE' for r in recs), [r['reasons'] for r in recs])
        self.assertNotIn('NOT COMPARABLE', text)
        self.assertIn('FRESH mem 1024 / mem 0: COMPARABLE (varies memory length) prompt', text)
        self.assertIn('mem 512 CACHED / FRESH: COMPARABLE (varies cache mode)', text)
        r = next(x for x in recs if x['name'] == 'FIRST mem 1024 / mem 0')
        self.assertEqual((r['expected_questions'], r['actual_repeats']), ([['prime'] * 2] * 2, [2, 2]))
        for key in ('independent_variable', 'allowed_to_differ', 'required_identical', 'expected_questions',
                    'actual_questions', 'expected_repeats', 'actual_repeats', 'complete', 'environment_valid',
                    'environment_matched', 'verdict', 'reasons'):
            self.assertIn(key, r)
        n_req = sum(int(line.split()[2].split('/')[1]) for line in text.splitlines()
                    if line.strip() and line.split()[0].isdigit() and len(line.split()) > 2
                    and line.split()[1] in mem1.CLASSES and '/' in line.split()[2])
        self.assertEqual(n_req, len(rows))  # every logged request appears once in the speed table
        self.assertEqual(sum(1 for line in text.splitlines() if line.endswith('| VALID')), len(blocks))

    def test_compliance_and_accuracy_reported_separately(self):
        answers = {'q_begin': answer('Pepper'), 'q_mid': '```json\n{"answer": "4417"}\n```',
                   'q_end': 'Leo takes honey and no milk.', 'q_update': answer('kitchen drawer')}
        write_run(self.tmp / 'r', self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False), answers=answers)
        text = mem1.report([self.tmp / 'r'])[0]
        fmt = next(l for l in text.splitlines() if l.split()[:2] == ['1024', 'FRESH'] and l.split()[2] == '12'
                   and l.rstrip().endswith('%') and len(l.split()) == 10)
        self.assertEqual(fmt.split()[3:], ['8', '2', '2', '0', '0', '67%', '83%'])  # 4 UNKNOWN + 2 pepper STRICT
        acc = next(l for l in text.splitlines() if l.split()[:2] == ['1024', 'FRESH'] and len(l.split()) == 15)
        # CORRECT 4 (2 pepper + 2 fenced 4417), ABSTAINED 4, STALE 2, MALFORMED 2; 8/12 right, 8/10 parsed
        self.assertEqual(acc.split()[3:], ['4', '4', '0', '2', '0', '0', '0', '2', '0', '0', '67%', '80%'])

    def test_freetext_report_is_unscored(self):
        write_run(self.tmp / 'ft', fake_set('freetext'), 'freetext', 2048, mem1.schedule('freetext', 1, True),
                  smoke=True)
        text = mem1.report([self.tmp / 'ft'])[0]
        self.assertIn('UNSCORED - MANUAL REVIEW REQUIRED', text)
        self.assertNotIn('format compliance', text)
        self.assertTrue(all('NOT COMPARABLE' in l and 'SMOKE_LAUNCH' in l for l in comparison_lines(text)))

    def test_drift_and_invalid_blocks_not_comparable(self):
        write_run(self.tmp / 'b', self.s, 'primary', 2048, mem1.schedule('primary', 2, False),
                  skins={12: 27.5}, warm=(3,))
        text = mem1.report([self.tmp / 'b'])[0]
        self.assertIn('NOT COMPARABLE', text)
        self.assertIn('INVALID: WARM_START warm start', text)
        self.assertIn('ENV_UNMATCHED skin 25.00-27.50 degC over the blocks', text)
        self.assertIn('ENV_INVALID', text)

    def test_interrupted_run_without_result_is_incomplete(self):
        write_run(self.tmp / 'd', self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False), result=False)
        text = mem1.report([self.tmp / 'd'])[0]
        self.assertIn('[INCOMPLETE]', text)
        self.assertIn('no usable result.json', text)
        self.assertTrue(all('RUN_NOT_COMPLETE' in l for l in comparison_lines(text)))

    def test_missing_block_is_incomplete_despite_result_json(self):
        write_run(self.tmp / 'e', self.s, 'primary', 2048, mem1.schedule('primary', 2, False), drop_blocks=(7,))
        text = mem1.report([self.tmp / 'e'])[0]
        self.assertIn('[INCOMPLETE]', text)
        self.assertIn('blocks MISSING: [7]', text)
        self.assertIn('114/120 planned scored requests answered', text)
        self.assertNotIn(': COMPARABLE', text)

    def test_missing_request_and_partial_result_are_incomplete(self):
        write_run(self.tmp / 'f', self.s, 'primary', 2048, mem1.schedule('primary', 2, False),
                  drop_requests={(3, 'q_end')})
        text = mem1.report([self.tmp / 'f'])[0]
        self.assertIn('blocks FAILED: [3]', text)
        self.assertIn('[INCOMPLETE]', text)
        plan = mem1.schedule('primary', 2, False)
        write_run(self.tmp / 'g', self.s, 'primary', 2048, plan, drop_blocks=range(5, 20),
                  end_reason='stopped before block 5: Android status 3, battery 31 degC')
        text = mem1.report([self.tmp / 'g'])[0]
        self.assertIn('ended: stopped before block 5', text)
        self.assertIn('COMPLETED 5, FAILED 0, SKIPPED 0, MISSING 15 of 20', text)
        self.assertIn('[INCOMPLETE]', text)
        write_run(self.tmp / 'h', self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False), drop_blocks=(1,),
                  skipped=(1,))
        self.assertIn('blocks SKIPPED: [1]', mem1.report([self.tmp / 'h'])[0])

    def test_run_without_matrix_cannot_be_complete(self):
        write_run(self.tmp / 'i', self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        run = json.loads((self.tmp / 'i' / 'run.json').read_text())
        del run['matrix']
        (self.tmp / 'i' / 'run.json').write_text(json.dumps(run))
        text = mem1.report([self.tmp / 'i'])[0]
        self.assertIn('completeness cannot be verified', text)
        self.assertIn('[INCOMPLETE]', text)

    def test_smoke_label(self):
        write_run(self.tmp / 'c', self.s, 'primary', 2048, mem1.schedule('primary', 1, True), smoke=True)
        self.assertIn('SMOKE - NOT A MEASUREMENT', mem1.report([self.tmp / 'c'])[0])

    def test_unequal_question_sides_not_comparable(self):
        rows, blocks = write_run(self.tmp / 'j', self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        for r in rows:  # one invalid scored request leaves the sides with different questions
            if r['block'] == 0 and r['question'] == 'q_mid':
                r['valid'] = False
        for b in blocks:
            b['problems'] = []
        recs = mem1.comparisons(json.loads((self.tmp / 'j' / 'run.json').read_text()), blocks, rows, [])
        c = next(x for x in recs if 'CACHED / FRESH' in x['name'])
        self.assertEqual(c['verdict'], 'NOT COMPARABLE')
        self.assertIn('QUESTIONS_NOT_AS_PLANNED', [r['code'] for r in c['reasons']])

    def test_report_cli_starts_nothing_and_writes_json(self):
        write_run(self.tmp / 'k', self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        with redirect_stdout(io.StringIO()) as out:
            mem1.main(['--report', str(self.tmp / 'k'), '--json', str(self.tmp / 'k.json')])
        self.assertIn('MEM-1 REPORT', out.getvalue())
        recs = json.loads((self.tmp / 'k.json').read_text())
        self.assertEqual({r['verdict'] for r in recs}, {'COMPARABLE'})


class Capacity(FakeFrozenSet, Offline):
    """Repair C: only the deliberate ctx change is allowed between the two capacity launches."""

    def setUp(self):
        super().setUp()
        self.tmp, self.s, self.plan = Path(tempfile.mkdtemp()), fake_set(), mem1.schedule('capacity', 2, False)

    def pair(self, b_kw=None, b_plan=None, **a_kw):
        a, b = self.tmp / 'a', self.tmp / 'b'
        write_run(a, self.s, 'capacity', 2048, self.plan, **a_kw)
        write_run(b, self.s, 'capacity', 4096, b_plan or self.plan, **(b_kw or {}))
        return mem1.report([a, b])

    def codes(self, recs):
        return {r['code'] for c in recs if 'runs' in c for r in c['reasons']}

    def test_deliberate_context_change_is_comparable(self):
        text, recs = self.pair()
        cap = [r for r in recs if 'runs' in r]
        self.assertEqual(len(cap), 3)
        self.assertTrue(all(r['verdict'] == 'COMPARABLE' for r in cap), [r['reasons'] for r in cap])
        self.assertEqual(cap[0]['independent_variable'], 'context allocation (ctx 2048 -> 4096)')
        self.assertTrue(all(cap[0]['required_identical'].values()))
        self.assertIn('configuration SAME except ctx', text)
        self.assertIn('capacity FRESH: ctx 4096 / ctx 2048: COMPARABLE', text)

    def test_unintended_configuration_differences_reject(self):
        for kw, dim in (({'code_sha256': {str(p): 'd' * 64 for p in mem1.code_files()}}, 'code_sha256'),):
            shutil.rmtree(self.tmp)
            self.tmp.mkdir()
            text, recs = self.pair(b_kw=kw)
            cap = [r for r in recs if 'runs' in r]
            self.assertTrue(all(r['verdict'] == 'NOT COMPARABLE' and r['environment_matched'] for r in cap), dim)
            self.assertIn({'code': 'CONFIG_DIFFERS', 'detail': dim}, cap[0]['reasons'])
            self.assertIn('configuration DIFFERENT', text)
            self.assertNotIn(': COMPARABLE', text.split('capacity (memory 1024')[1])

    def test_missing_or_foreign_provenance_is_an_invalid_run(self):
        """Round 13 finding 1: every configuration and provenance dimension is required and must be this code's;
        removing or changing any one makes run.json INVALID, so no comparison (within or across launches) uses it."""
        folder = self.tmp / 'prov'
        write_run(folder, self.s, 'primary', 2048, mem1.schedule('primary', 2, False))
        run = json.loads((folder / 'run.json').read_text())
        self.assertIn('[COMPLETE]', mem1.report([folder])[0])
        changed = {'decoding': {**run['decoding'], 'temperature': 0.7}, 'warmup': 'Hello',
                   'model_sha256': {mem1.MODEL: 'f' * 64}, 'pinned_sha256': {}, 'build_sha256': {'llama-server': 'x'},
                   'code_sha256': {'mem1.py': 'h'}, 'thresholds': {**run['thresholds'], 'status_stop': 4},
                   'telemetry': {**run['telemetry'], 'sample_period_s': 5}, 'task': 'freetext'}
        for k in mem1.RUN_SCHEMA:
            for how, value in (('removed', None), ('changed', changed.get(k))):
                if how == 'changed' and value is None:
                    continue
                (folder / 'run.json').write_text(json.dumps(
                    {x: v for x, v in run.items() if x != k} if how == 'removed' else {**run, k: value}))
                text, recs = mem1.report([folder])
                self.assertIn('[INVALID]', text, (k, how))
                self.assertEqual(recs, [], (k, how))
        (folder / 'run.json').write_text(json.dumps({**run, 'prime': {**run['prime'], 'text': 'Say hi.'}}))
        self.assertIn('run.json prime is not the frozen prime', mem1.report([folder])[0])
        shutil.rmtree(self.tmp)
        self.tmp.mkdir()
        text, recs = self.pair(b_kw={'thresholds': None})
        self.assertIn('needs exactly two capacity launches', text)
        self.assertFalse([r for r in recs if 'runs' in r])

    def test_other_question_set_is_an_invalid_run(self):
        text, recs = self.pair(b_kw={'set_sha256': 'y'})
        self.assertIn('[INVALID]: run.json: set_sha256 is not the frozen set of this code', text)
        self.assertFalse([r for r in recs if 'runs' in r])

    def test_mismatched_questions_and_unequal_repetitions(self):
        _, recs = self.pair(b_kw={'rounds': 4})
        self.assertIn('CONFIG_DIFFERS', self.codes(recs))
        shutil.rmtree(self.tmp)
        self.tmp.mkdir()
        _, recs = self.pair(b_plan=mem1.schedule('capacity', 4, False), b_kw={'rounds': 2})
        self.assertTrue({'REPEATS_DIFFER', 'QUESTIONS_DIFFER', 'CONFIG_DIFFERS'} <= self.codes(recs), self.codes(recs))
        shutil.rmtree(self.tmp)
        self.tmp.mkdir()
        _, recs = self.pair(b_kw={'drop_requests': {(1, 'q_end')}, 'result': True})
        self.assertTrue({'QUESTIONS_NOT_AS_PLANNED', 'RUN_NOT_COMPLETE'} <= self.codes(recs), self.codes(recs))

    def test_incomplete_or_smoke_launch_rejects(self):
        _, recs = self.pair(b_kw={'drop_blocks': (3,)})
        self.assertIn('RUN_NOT_COMPLETE', self.codes(recs))
        shutil.rmtree(self.tmp)
        self.tmp.mkdir()
        _, recs = self.pair(b_kw={'smoke': True})
        self.assertTrue({'SMOKE_LAUNCH', 'CONFIG_DIFFERS'} <= self.codes(recs), self.codes(recs))
        shutil.rmtree(self.tmp)
        self.tmp.mkdir()
        _, recs = self.pair(b_kw={'result': False})
        self.assertIn('RUN_NOT_COMPLETE', self.codes(recs))

    def test_missing_telemetry_rejects_despite_match(self):
        a, b = self.tmp / 'a', self.tmp / 'b'
        write_run(a, self.s, 'capacity', 2048, self.plan)
        write_run(b, self.s, 'capacity', 4096, self.plan)
        lines = (b / 'blocks.jsonl').read_text().splitlines()
        blk = json.loads(lines[0])
        blk['samples'] = []
        blk['problems'] = []  # a falsified label: the report recomputes it
        lines[0] = json.dumps(blk)
        (b / 'blocks.jsonl').write_text('\n'.join(lines) + '\n')
        _, recs = mem1.report([a, b])
        cap = [r for r in recs if 'runs' in r]
        self.assertTrue(any(not r['environment_valid'] and r['verdict'] == 'NOT COMPARABLE' for r in cap))
        self.assertIn('ENV_INVALID', self.codes(recs))

    def test_block_context_must_be_the_launch_context(self):
        """Round 12 finding 1: blocks claiming another ctx (stored VALID) cannot make a false capacity comparison."""
        a, b = self.tmp / 'a', self.tmp / 'b'
        write_run(a, self.s, 'capacity', 2048, self.plan)
        write_run(b, self.s, 'capacity', 4096, self.plan)
        blocks = [json.loads(l) for l in (a / 'blocks.jsonl').read_text().splitlines()]
        for x in blocks:
            x.update(ctx=4096, problems=[])
            x['checks']['n_ctx'] = 4096
        del blocks[1]['ctx']
        (a / 'blocks.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in blocks))
        text, recs = mem1.report([a, b])
        self.assertIn('block ctx 4096 is not the launch ctx 2048', text)
        self.assertIn('block ctx None is not the launch ctx 2048', text)
        self.assertTrue(all(r['verdict'] == 'NOT COMPARABLE' for r in recs if 'runs' in r))
        self.assertIn('ENV_INVALID', self.codes(recs))
        run = json.loads((a / 'run.json').read_text())
        (a / 'run.json').write_text(json.dumps({**run, 'command': mem1.command(4096)}))
        self.assertIn("command is not this benchmark's server command at ctx 2048", mem1.report([a])[0])

    def test_recorded_commands_and_code_files_are_this_benchmarks(self):
        """Round 14 finding 1: the full recorded argv and the set of hashed source files."""
        a = self.tmp / 'a'
        write_run(a, self.s, 'capacity', 2048, self.plan)
        run = json.loads((a / 'run.json').read_text())
        cmd = mem1.command(2048)
        at = lambda flag: cmd.index(flag) + 1
        variants = {'drafter': cmd + ['--model-draft', '/x/drafter.gguf'], 'projector': cmd + ['--mmproj', '/x/p.gguf'],
                    'threads': cmd[:at('--threads')] + ['1'] + cmd[at('--threads') + 1:],
                    'model': cmd[:at('-m')] + ['/x/other.gguf'] + cmd[at('-m') + 1:],
                    'affinity': ['taskset', '-c', '0-7'] + cmd[3:], 'binary': cmd[:3] + ['/x/llama-server'] + cmd[4:],
                    'cache': [x for x in cmd if x != '--swa-full']}
        for name, c in variants.items():
            (a / 'run.json').write_text(json.dumps({**run, 'command': c}))
            text, recs = mem1.report([a])
            self.assertIn("command is not this benchmark's server command", text, name)
            self.assertEqual(recs, [], name)
        code = dict(run['code_sha256'])
        code.pop(next(iter(code)))
        (a / 'run.json').write_text(json.dumps({**run, 'code_sha256': {**code, '/x/extra.py': 'c' * 64}}))
        self.assertIn('code_sha256 is not one well-formed hash for each', mem1.report([a])[0])

    def test_wrong_folder_sets_are_refused(self):
        a = self.tmp / 'a'
        write_run(a, self.s, 'capacity', 2048, self.plan)
        text, recs = mem1.report([a, a])
        self.assertIn('needs exactly two capacity launches', text)
        self.assertFalse([r for r in recs if 'runs' in r])


class Artifacts(FakeFrozenSet, Offline):
    """Repair D: partial and malformed artifacts never crash the report or look complete; programming errors are
    not swallowed."""

    def setUp(self):
        super().setUp()
        self.tmp, self.s = Path(tempfile.mkdtemp()), fake_set()
        self.plan = mem1.schedule('primary', 2, False)

    def test_orphan_requests_of_an_interrupted_block_do_not_crash(self):
        """Round-7 KeyError: requests flushed, block record never written, valid rows on both sides."""
        write_run(self.tmp / 'o', self.s, 'primary', 2048, self.plan, drop_blocks=(7,), keep_orphans=True,
                  result=False)
        text, recs = mem1.report([self.tmp / 'o'])
        self.assertIn('[INCOMPLETE]', text)
        self.assertIn('7 requests of blocks [7] without a block record (interrupted mid-block)', text)
        self.assertTrue(all(r['verdict'] == 'NOT COMPARABLE' for r in recs))
        self.assertNotIn(' 7 ', ''.join(l for l in text.splitlines() if l.startswith('   7 ')))  # no block line 7

    def test_block_interrupted_mid_execution_with_torn_line(self):
        write_run(self.tmp / 'm', self.s, 'primary', 2048, self.plan, drop_blocks=range(4, 20), keep_orphans=True,
                  result=False)
        path = self.tmp / 'm' / 'requests.jsonl'
        lines = path.read_text().splitlines()
        lines = lines[:4 * 7 + 3] + [lines[4 * 7 + 3][:40]]  # block 4: 3 requests, then a write cut off by a kill
        path.write_text('\n'.join(lines))
        text, recs = mem1.report([self.tmp / 'm'])
        self.assertIn('torn last line (interrupted write)', text)
        self.assertIn('requests of blocks [4] without a block record', text)
        self.assertIn('[INCOMPLETE]', text)

    def test_missing_keys_and_malformed_records_left_out(self):
        rows, blocks = write_run(self.tmp / 'k', self.s, 'primary', 2048, self.plan)
        path = self.tmp / 'k' / 'blocks.jsonl'
        lines = path.read_text().splitlines()
        b = json.loads(lines[2])
        del b['end_reason']
        lines[2] = json.dumps(b)
        lines[5] = '{"block": 5, "round"'  # malformed middle line
        lines.append('[1, 2]')
        path.write_text('\n'.join(lines) + '\n')
        text = mem1.report([self.tmp / 'k'])[0]
        self.assertIn('blocks.jsonl line 3 left out: end_reason missing', text)
        self.assertIn("blocks.jsonl line 6: not JSON", text)
        self.assertIn('blocks.jsonl line 21 left out: not a JSON object (list)', text)
        self.assertIn('requests of blocks [2, 5] without a block record', text)
        self.assertIn('[INCOMPLETE]', text)
        rp = self.tmp / 'k' / 'requests.jsonl'
        lines = rp.read_text().splitlines()
        r = json.loads(lines[0])
        del r['wall_s']
        rp.write_text('\n'.join([json.dumps(r)] + lines[1:]) + '\n')
        rel = '{} requests whose stored validity or outcome differs from the recomputation (stored labels ignored)'
        self.assertIn(rel.format(1), mem1.report([self.tmp / 'k'])[0])  # kept, recomputed INVALID: wall_s missing
        lines = rp.read_text().splitlines()
        r = json.loads(lines[1])
        r['timings'], r['prompt_ms'] = {}, None  # a row claiming VALID without the timings the statistics read
        rp.write_text('\n'.join(lines[:1] + [json.dumps(r)] + lines[2:]) + '\n')
        self.assertIn(rel.format(2), mem1.report([self.tmp / 'k'])[0])

    def test_partial_and_truncated_result_json(self):
        for content, expect in (('{}', 'result.json partial: end_reason missing'),
                                ('{"end_reason": "all blocks done"}', 'result.json partial: skipped missing'),
                                ('{"end_reason": "all blo', 'result.json unreadable'),
                                ('', 'result.json unreadable'), ('[]', 'result.json partial: not a JSON object')):
            folder = self.tmp / f'p{len(list(self.tmp.iterdir()))}'
            write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
            (folder / 'result.json').write_text(content)
            text, recs = mem1.report([folder])
            self.assertIn(expect, text, content)
            self.assertIn('[INCOMPLETE]', text)
            self.assertNotIn('end: ', text)
            self.assertTrue(all(r['verdict'] == 'NOT COMPARABLE' for r in recs))

    def test_unusable_run_json_is_invalid(self):
        for content, expect in ((None, 'run.json missing'), ('{"exp": "prim', 'run.json unreadable'),
                                ('{"exp": "primary"}', 'run.json: task missing'),
                                (json.dumps({'task': 'recall', 'exp': 'primary', 'ctx': '2048', 'rounds': 2,
                                             'smoke': False}), 'run.json: ctx is str')):
            folder = self.tmp / f'v{len(list(self.tmp.iterdir()))}'
            write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
            if content is None:
                (folder / 'run.json').unlink()
            else:
                (folder / 'run.json').write_text(content)
            text, recs = mem1.report([folder])
            self.assertIn('[INVALID]', text)
            self.assertIn(expect, text)
            self.assertEqual(recs, [])
        matrix = self.tmp / 'mx'
        write_run(matrix, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False), matrix=[{'block': 0}])
        self.assertIn('matrix malformed', mem1.report([matrix])[0])

    def test_duplicate_block_records_left_out(self):
        write_run(self.tmp / 'u', self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        path = self.tmp / 'u' / 'blocks.jsonl'
        path.write_text(path.read_text() + path.read_text().splitlines()[1] + '\n')
        text = mem1.report([self.tmp / 'u'])[0]
        self.assertIn('blocks [1] recorded more than once', text)
        self.assertIn('[INCOMPLETE]', text)

    def test_falsified_labels_are_recomputed_never_trusted(self):
        folder = self.tmp / 'lie'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        bp, rp = folder / 'blocks.jsonl', folder / 'requests.jsonl'
        blocks = [json.loads(l) for l in bp.read_text().splitlines()]
        blocks[0].update(samples=[], problems=[])  # no telemetry at all, stored as VALID
        bp.write_text(''.join(mem1.dumps(b) + '\n' for b in blocks))
        rows = [json.loads(l) for l in rp.read_text().splitlines()]
        liar = next(r for r in rows if r['block'] == 1 and r['scored'])
        liar.update(reply=answer('wrong'), outcome='CORRECT', valid=True, problems=[])  # wrong reply labelled CORRECT
        bad = next(r for r in rows if r['block'] == 2 and r['scored'])
        bad.update(tokens_evaluated=1, valid=True, problems=[])  # token mismatch labelled VALID
        rp.write_text(''.join(mem1.dumps(r) + '\n' for r in rows))
        run = mem1.load_run(folder)
        b0 = next(b for b in run.blocks if b['block'] == 0)
        self.assertEqual(b0['stored_problems'], [])
        self.assertIn('SAMPLES_EMPTY', codes(b0['problems']))
        by = {(r['block'], r['question']): r for r in run.rows}
        self.assertNotEqual(by[1, liar['question']]['outcome'], 'CORRECT')
        self.assertEqual(by[1, liar['question']]['stored']['outcome'], 'CORRECT')  # kept for audit only
        self.assertFalse(by[2, bad['question']]['valid'])
        self.assertIn('blocks [0, 2]: stored validity differs from the recomputation (stored labels ignored)',
                      run.issues)
        self.assertIn('2 requests whose stored validity or outcome differs from the recomputation (stored labels '
                      'ignored)', run.issues)
        text, recs = mem1.report([folder])
        self.assertIn('SAMPLES_EMPTY', text)
        self.assertNotIn('COMPARABLE', {r['verdict'] for r in recs})

    def test_missing_or_out_of_range_request_evidence_is_never_valid(self):
        """Round 11 finding 1: no-truncation evidence, request UTC and integer token counts within their limits are
        required; stored VALID labels do not rescue a row."""
        folder = self.tmp / 'req'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        rp = folder / 'requests.jsonl'
        rows = [json.loads(l) for l in rp.read_text().splitlines()]
        scored = [x for x in rows if x['scored'] and x['block'] == 0]
        del scored[0]['truncated']
        scored[1]['truncated'] = None
        scored[2]['utc'] = 'garbage'
        scored[3]['timings']['predicted_n'] = scored[3]['predicted_n'] = 1000
        scored[4]['timings']['predicted_n'] = scored[4]['predicted_n'] = 6.0
        scored[5]['timings']['prompt_n'] = scored[5]['prompt_n'] = True
        for x in scored:
            x.update(valid=True, problems=[])
        rp.write_text(''.join(mem1.dumps(x) + '\n' for x in rows))
        got = {x['question']: x for x in mem1.load_run(folder).rows if x['block'] == 0}
        for x, text in zip(scored, ('truncated None', 'truncated None', "request utc 'garbage'",
                                    'timings.predicted_n 1000 not an integer in 0..48',
                                    'timings.predicted_n 6.0 not an integer', 'timings.prompt_n True not an integer')):
            self.assertFalse(got[x['question']]['valid'], text)
            self.assertIn(text, ' '.join(got[x['question']]['problems']))
        self.assertNotIn(': COMPARABLE', mem1.report([folder])[0])

    def test_work_without_duration_is_never_valid(self):
        """Round 12 finding 2: generated (or evaluated) tokens with zero duration and an asserted rate."""
        folder = self.tmp / 'zero'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        rp = folder / 'requests.jsonl'
        rows = [json.loads(l) for l in rp.read_text().splitlines()]
        g, q = [x for x in rows if x['scored'] and x['block'] == 0][:2]
        g['timings'].update(predicted_ms=0, predicted_per_second=1e6)
        g.update(predicted_ms=0, gen_tok_s=1e6, valid=True, problems=[])
        q['timings']['prompt_ms'] = q['prompt_ms'] = 0
        q.update(ttft_proxy_s=0, valid=True, problems=[])
        rp.write_text(''.join(mem1.dumps(x) + '\n' for x in rows))
        got = {x['question']: x for x in mem1.load_run(folder).rows if x['block'] == 0}
        self.assertIn('timings.predicted_ms 0 for predicted_n 6 > 0 (no duration)', got[g['question']]['problems'])
        self.assertTrue(any('server formula' in w for w in got[g['question']]['problems']))
        self.assertTrue(any('timings.prompt_ms 0 for prompt_n' in w for w in got[q['question']]['problems']))
        self.assertNotIn(': COMPARABLE', mem1.report([folder])[0])

    def test_shortened_or_unconfigured_plans_are_never_complete(self):
        """Round 11 finding 2: the recorded matrix must be the configured schedule's."""
        folder = self.tmp / 'short'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        run = json.loads((folder / 'run.json').read_text())
        run['matrix'] = run['matrix'][:2]  # round 1 only, although rounds is 2
        (folder / 'run.json').write_text(json.dumps(run))
        for name in ('blocks.jsonl', 'requests.jsonl'):
            lines = [l for l in (folder / name).read_text().splitlines() if json.loads(l)['block'] < 2]
            (folder / name).write_text('\n'.join(lines) + '\n')
        text, recs = mem1.report([folder])
        self.assertIn('run.json matrix differs from the configured schedule', text)
        self.assertIn('[INCOMPLETE]', text)
        self.assertTrue(recs and all(r['verdict'] == 'NOT COMPARABLE' for r in recs))
        for kw, expect in (({'rounds': 3}, 'rounds 3 not allowed'), ({'exp': 'bogus'}, "exp 'bogus'"),
                           ({'exp': 'long'}, "exp 'long' at ctx 2048 is not a configured experiment")):
            (folder / 'run.json').write_text(json.dumps({**run, **kw}))
            text = mem1.report([folder])[0]
            self.assertIn('[INVALID]', text, kw)
            self.assertIn(expect, text, kw)

    def test_agents_ports_duplicates_and_changed_artifacts_are_never_eligible(self):
        """Round 13 finding 2 and the eligibility audit: agent processes (at the launch end, or before/after a block),
        a busy port at the end, duplicated request records and artifacts changed after the run each keep every
        comparison NOT COMPARABLE, whatever the stored labels say."""
        def fresh(name, **res):
            f = self.tmp / name
            write_run(f, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
            if res:
                r = json.loads((f / 'result.json').read_text())
                (f / 'result.json').write_text(json.dumps({**r, **res}))
            return f
        ok = fresh('ok')
        self.assertIn('[COMPLETE]', mem1.report([ok])[0])
        for res, code in (({'agents_end': '4242 claude --resume'}, 'AGENTS_PRESENT'),
                          ({'port_free_end': False}, 'PORT_NOT_FREE')):
            text, recs = mem1.report([fresh(code, **res)])
            self.assertIn('TIMINGS NOT ELIGIBLE: ' + code, text)
            self.assertTrue(recs and all(r['verdict'] == 'NOT COMPARABLE' and code in {x['code'] for x in r['reasons']}
                                         for r in recs), code)
        b4096 = self.tmp / 'b4096'
        write_run(b4096, self.s, 'capacity', 4096, mem1.schedule('capacity', 2, False))
        r = json.loads((b4096 / 'result.json').read_text())
        (b4096 / 'result.json').write_text(json.dumps({**r, 'agents_end': '99 codex exec'}))
        _, recs = mem1.report([ok, b4096])
        cap = [x for x in recs if 'runs' in x]
        self.assertTrue(cap and all('AGENTS_PRESENT' in {y['code'] for y in x['reasons']} for x in cap))
        f = fresh('blockagents')
        blocks = [json.loads(l) for l in (f / 'blocks.jsonl').read_text().splitlines()]
        blocks[0]['agents'] = {'before': '', 'after': '77 claude'}
        del blocks[1]['agents']
        (f / 'blocks.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in blocks))
        text, recs = mem1.report([f])
        self.assertIn("AGENTS_PRESENT agent processes before/after the block", text)
        self.assertNotIn(': COMPARABLE', text)
        f = fresh('dup')
        lines = (f / 'requests.jsonl').read_text().splitlines()
        (f / 'requests.jsonl').write_text('\n'.join(lines + lines[3:4]) + '\n')
        text = mem1.report([f])[0]
        self.assertIn('recorded more than once: left out', text)
        self.assertIn('[INCOMPLETE]', text)
        f = fresh('hashes')
        (f / 'artifacts.sha256').write_text(''.join(
            f'{hashlib.sha256((f / n).read_bytes()).hexdigest()}  {n}\n' for n in ('blocks.jsonl', 'requests.jsonl')))
        self.assertIn('[COMPLETE]', mem1.report([f])[0])
        (f / 'blocks.jsonl').write_text((f / 'blocks.jsonl').read_text() + '\n')
        text = mem1.report([f])[0]
        self.assertIn('artifacts.sha256: blocks.jsonl changed after the run', text)
        self.assertNotIn(': COMPARABLE', text)

    def test_missing_or_contradicted_checks_and_warm_up_are_never_eligible(self):
        """Round 14 finding 2: the no-generation checks and the warm-up must be recorded completely and agree with
        the plan and the frozen set, whatever the stored labels say."""
        folder = self.tmp / 'chk'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        bp = folder / 'blocks.jsonl'
        good = [json.loads(l) for l in bp.read_text().splitlines()]
        self.assertIn('[COMPLETE]', mem1.report([folder])[0])
        c0 = good[0]['checks']
        bad_tokens = {**c0['tokens'], 'q_mid': c0['tokens']['q_mid'] + 1}
        for name, change, expect in (
                ('reviewer', {'checks': {'template': {'prime': True}}}, 'template checks'),
                ('tokens', {'checks': {**c0, 'tokens': bad_tokens}}, 'are not the frozen counts'),
                ('template', {'checks': {**c0, 'template': {**c0['template'], 'q_end': False}}}, 'template checks'),
                ('n_ctx', {'checks': {**c0, 'n_ctx': None}}, 'server n_ctx None'),
                ('chat template', {'checks': {**c0, 'chat_template_sha256': None}}, 'chat template hash missing'),
                ('no warm-up', {'checks': {k: v for k, v in c0.items() if k != 'warmup'}}, 'warm-up evidence'),
                ('cached warm-up', {'checks': {**c0, 'warmup': {**c0['warmup'], 'timings': {
                    **c0['warmup']['timings'], 'cache_n': 5}}}}, 'warm-up evidence'),
                ('order', {'questions': good[0]['questions'][::-1]}, 'not the planned questions'),
                ('ints', {'checks': {**c0, 'n_ctx': 2048.0}}, 'server n_ctx 2048.0'),
                ('bools', {'checks': {**c0, 'template': {k: 1 for k in c0['template']}}}, 'passed check (true)'),
                ('floats', {'checks': {**c0, 'tokens': {k: float(v) for k, v in c0['tokens'].items()}}}, '(ints)')):
            blocks = copy.deepcopy(good)
            for x in blocks:
                x.update(copy.deepcopy(change), problems=[])
            bp.write_text(''.join(json.dumps(x) + '\n' for x in blocks))
            text, recs = mem1.report([folder])
            self.assertIn('CHECKS_FAILED', text, name)
            self.assertTrue(any(w['code'] == 'CHECKS_FAILED' and expect in w['detail']
                                for b in mem1.load_run(folder).blocks for w in b['problems']), name)
            self.assertTrue(recs and all(r['verdict'] == 'NOT COMPARABLE' for r in recs), name)

    def test_shortened_verification_coverage_is_never_eligible(self):
        """Round 15 finding 1: every block keeps only its first question in questions, template and tokens (all
        consistent with each other) while all six scored requests remain: CHECKS_FAILED, nothing COMPARABLE."""
        folder = self.tmp / 'shortcov'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        bp = folder / 'blocks.jsonl'
        blocks = [json.loads(l) for l in bp.read_text().splitlines()]
        for x in blocks:
            keep = ['prime', x['questions'][0]]
            x['questions'] = x['questions'][:1]
            for k in ('template', 'tokens'):
                x['checks'][k] = {q: v for q, v in x['checks'][k].items() if q in keep}
            x['problems'] = []
        bp.write_text(''.join(json.dumps(x) + '\n' for x in blocks))
        text, recs = mem1.report([folder])
        details_ = [w['detail'] for b in mem1.load_run(folder).blocks for w in b['problems']]
        self.assertTrue(any('not the planned questions of round' in d for d in details_), details_[:3])
        self.assertTrue(recs and all(r['verdict'] == 'NOT COMPARABLE' for r in recs))
        b0 = copy.deepcopy(blocks[0])
        b0['questions'] = mem1.question_order(FAKE['questions'], b0['round'], False)[0:6]
        b0['questions'] = [q['id'] for q in b0['questions']]
        rows = [{'question': 'q_other', 'round': b0['round'], 'length': b0['length'], 'mode': b0['mode']}]
        self.assertIn("requests ['q_other'] have no recorded no-generation check",
                      ' | '.join(mem1.checks_problems(b0, rows, 2048, FAKE, False)))

    def test_falsified_cache_expectations_are_recomputed_from_the_frozen_set(self):
        """Round 9 finding 2: a CACHED row claiming a 1-token shared prefix and cache expectation, with consistent
        counts and copied metrics, and a row moved to another position, are INVALID."""
        folder = self.tmp / 'cache'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        rp = folder / 'requests.jsonl'
        rows = [json.loads(l) for l in rp.read_text().splitlines()]
        r = next(x for x in rows if x['class'] == 'CACHED')
        n = r['tokens_expected']
        r.update(shared_prefix_tokens=1, expected_cache_n=1, cache_n=1, prompt_n=n - 1)
        r['timings'].update(cache_n=1, prompt_n=n - 1)
        moved = next(x for x in rows if x['class'] == 'FRESH' and x['position'] == 2)
        moved['question'] = next(x['question'] for x in rows if x['block'] == moved['block'] and x['position'] == 3)
        rp.write_text(''.join(mem1.dumps(x) + '\n' for x in rows))
        got = {(x['block'], x['position']): x for x in mem1.load_run(folder).rows}
        bad = got[r['block'], r['position']]
        self.assertFalse(bad['valid'])
        self.assertIn('recorded expected token counts differ from the frozen set', bad['problems'])
        self.assertTrue(any('prefix not reused' in w for w in bad['problems']), bad['problems'])
        self.assertIn('mode, memory length, question, position and scored flag disagree with the plan',
                      got[moved['block'], 2]['problems'])
        self.assertNotIn(': COMPARABLE', mem1.report([folder])[0])

    def test_malformed_values_report_without_crashing(self):
        """Round 9 finding 4: malformed operands, replies, huge integers and caps are INVALID evidence, not
        crashes."""
        folder = self.tmp / 'mal'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        rp, bp = folder / 'requests.jsonl', folder / 'blocks.jsonl'
        rows = [json.loads(l) for l in rp.read_text().splitlines()]
        a, b = [x for x in rows if x['scored'] and x['block'] == 1][:2]
        a['timings']['cache_n'] = a['cache_n'] = '0'
        b['reply'] = ['Pepper']
        rp.write_text(''.join(mem1.dumps(x) + '\n' for x in rows))
        blocks = [json.loads(l) for l in bp.read_text().splitlines()]
        blocks[0]['start']['cpufreq']['policy6']['max'] = [2850000]
        blocks[0]['gate']['skin'] = 'hot'
        blocks[2]['samples'][0]['pss_kb'] = 10 ** 400
        bp.write_text(''.join(json.dumps(x) + '\n' for x in blocks))
        text, recs = mem1.report([folder])
        self.assertIn('blocks.jsonl line 3: not JSON (or a non-finite number)', text)  # the huge integer
        got = {(x['block'], x['question']): x for x in mem1.load_run(folder).rows}
        self.assertFalse(got[1, a['question']]['valid'])
        self.assertEqual((got[1, b['question']]['format'], got[1, b['question']]['outcome']),
                         ('MALFORMED', 'MALFORMED'))
        self.assertIn('policy6 cap None (missing or not a positive int)', text)
        self.assertIn("'hot'", text)
        self.assertNotIn(': COMPARABLE', text)
        self.assertTrue(mem1.num(10 ** 400) and not mem1.num(10 ** 400, 0, 100))  # never OverflowError
        self.assertEqual(mem1.parse_answer({'answer': 'Pepper'}), ('MALFORMED', None))
        self.assertTrue(mem1.row_problems('CACHED', {'tokens_evaluated': 10, 'stop_type': 'eos', 'timings': {
            'cache_n': '5', 'prompt_n': 5}}, 10, 5, 4))

    def test_malformed_skipped_entries_and_missing_pid_report_without_crashing(self):
        """Round 10 finding 4."""
        folder = self.tmp / 'skip'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        (folder / 'result.json').write_text(json.dumps({'end_reason': 'all blocks done', 'agents_end': '',
                                                       'port_free_end': True, 'skipped': [{'block': []}, 'x']}))
        bp = folder / 'blocks.jsonl'
        blocks = [json.loads(l) for l in bp.read_text().splitlines()]
        blocks[0]['server_pid'] = None
        del blocks[0]['samples'][0]['t']
        bp.write_text(''.join(json.dumps(x) + '\n' for x in blocks))
        text, recs = mem1.report([folder])
        self.assertIn('result.json partial: skipped entry', text)
        self.assertIn('[INCOMPLETE]', text)
        b0 = next(b for b in mem1.load_run(folder).blocks if b['block'] == 0)
        self.assertTrue({'TIMES_INVALID', 'SAMPLE_INVALID'} <= codes(b0['problems']), details(b0['problems']))
        self.assertNotIn('COMPARABLE', {r['verdict'] for r in recs})

    def test_non_finite_numbers_are_rejected_on_load_and_never_written(self):
        for text in ('NaN', '-Infinity', '{"a": Infinity}', '[1e999]', '{"a": -1e999}', '[1' + '0' * 30 + ']',
                     '{"a": -' + '9' * 400 + '}'):
            with self.assertRaises(ValueError):
                mem1.strict_loads(text)
        self.assertEqual(mem1.strict_loads(str(2 ** 63)), 2 ** 63)
        self.assertEqual(mem1.strict_loads(mem1.dumps({'n': 10 ** 30})), {'n': str(10 ** 30)})  # round trip
        written = mem1.dumps({'a': float('nan'), 'b': [float('inf'), float('-inf')], 'c': 1.5})
        self.assertEqual(mem1.strict_loads(written), {'a': 'nan', 'b': ['inf', '-inf'], 'c': 1.5})
        folder = self.tmp / 'nan'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        bp, rp = folder / 'blocks.jsonl', folder / 'requests.jsonl'
        bl, rl = bp.read_text().splitlines(), rp.read_text().splitlines()
        b = json.loads(bl[1])
        b['load_s'] = float('nan')
        bl[1] = json.dumps(b)  # what a non-strict writer produces: NaN
        r = json.loads(rl[0])
        r['timings']['prompt_ms'] = float('inf')
        rl[0] = json.dumps(r)
        bp.write_text('\n'.join(bl) + '\n')
        rp.write_text('\n'.join(rl) + '\n')
        text = mem1.report([folder])[0]
        self.assertIn('blocks.jsonl line 2: not JSON (or a non-finite number)', text)
        self.assertIn('requests.jsonl line 1: not JSON (or a non-finite number)', text)
        self.assertIn('[INCOMPLETE]', text)
        run = json.loads((folder / 'run.json').read_text())
        (folder / 'run.json').write_text(json.dumps(run).replace('"ctx": 2048', '"ctx": 2048, "x": 1e999'))
        self.assertIn('run.json unreadable', mem1.report([folder])[0])
        nan = self.tmp / 'nan2'  # written by the runner: the string 'nan' is kept as evidence and fails validation
        write_run(nan, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        blocks = [json.loads(l) for l in (nan / 'blocks.jsonl').read_text().splitlines()]
        blocks[0]['samples'][0]['battery_c'] = float('nan')
        (nan / 'blocks.jsonl').write_text(''.join(mem1.dumps(x) + '\n' for x in blocks))
        b0 = next(b for b in mem1.load_run(nan).blocks if b['block'] == 0)
        self.assertEqual(b0['samples'][0]['battery_c'], 'nan')
        self.assertIn('SAMPLE_INVALID', codes(b0['problems']))

    def test_shallow_nested_records_report_without_crashing(self):
        for kw, expect in (({'prime': None}, 'run.json: prime is NoneType'),
                           ({'command': None}, 'run.json: command is NoneType'),
                           ({'prime': {}}, 'run.json: prime.id missing'), ({'command': []}, 'command malformed'),
                           ({'task': 'other'}, "task 'other'")):
            folder = self.tmp / f'r{len(list(self.tmp.iterdir()))}'
            write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False), **kw)
            text, recs = mem1.report([folder])
            self.assertIn('[INVALID]', text, kw)
            self.assertIn(expect, text, kw)
        folder = self.tmp / 'shallow'
        write_run(folder, self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        bp = folder / 'blocks.jsonl'
        blocks = [json.loads(l) for l in bp.read_text().splitlines()]
        blocks[0].update(start={}, end={}, gate={}, sampler={}, samples=[{}, 'x'], server_end={})
        for k in ('mono_t0', 'server_s', 'server_start', 'sampling_end_s'):
            del blocks[1][k]
        bp.write_text(''.join(mem1.dumps(b) + '\n' for b in blocks))
        text, recs = mem1.report([folder])
        for seq in (0, 1):
            b = next(b for b in mem1.load_run(folder).blocks if b['block'] == seq)
            self.assertTrue(b['problems'], seq)
        self.assertIn('TIMES_INVALID', text)
        self.assertIn('READING_INVALID', text)
        self.assertNotIn('COMPARABLE', {r['verdict'] for r in recs})

    def test_programming_errors_are_not_swallowed(self):
        write_run(self.tmp / 'z', self.s, 'capacity', 2048, mem1.schedule('capacity', 2, False))
        with mock.patch.object(mem1, 'stats', side_effect=ZeroDivisionError('bug')), \
                self.assertRaises(ZeroDivisionError):
            mem1.report([self.tmp / 'z'])


class Launch(Offline):
    """main() end to end with every phone interface faked: run.json matrix, skipped blocks, result.json coverage."""

    def launch(self, exp='capacity', lost_before=(), tel=None, agents=None, extra=()):
        tmp, raw, stopped, tel = Path(tempfile.mkdtemp()), fake_raw(), [], tel or FakeTelemetry()
        post = FakeServer(raw, 1024, RIGHT).post

        def check_cores(when):
            if any(when == f'before block {x}' for x in lost_before):
                raise cr.CoresLost(f'cores 4-7 not all allowed {when}')

        patches = [(mem1, 'load_set', {'return_value': raw}), (mem1, 'OUT_ROOT', tmp),
                   (cr, 'SKIN_GATE_C', cr.SKIN_GATE_C), (cr, 'require_native', {}), (signal_module(), 'signal', {}),
                   (cm, 'refusals', {'return_value': []}),
                   (cr, 'meminfo_mib', {'return_value': {'mem_available_mib': 9000}}),
                   (mem1, 'pinned_problems', {'return_value': ([], dict(mem1.PINNED))}),
                   (cr, 'wait_cores', {}), (cr, 'check_cores', {'side_effect': check_cores}),
                   (mem1, 'observe', tel.observe), (mem1, 'pin_sampler', {'return_value': dict(SAMPLER_OK)}),
                   (mem1, 'SAMPLE_PERIOD_S', 0.02),
                   (mem1.pm, 'thermal_gate', {'return_value': {'skin': 25.0, 'status': 0, 'waited_s': 0,
                                                               'warm_start': False}}),
                   (repro, 'build_sha256', {'return_value': REPRO_BUILD}),
                   (cr, 'sha256', {'return_value': 'c' * 64}),
                   (repro, 'agents', agents or {'return_value': ''}), (repro, 'port_free', {'return_value': True}),
                   (cm, 'spawn', {'return_value': FakeProc()}), (cm, 'stop', {'side_effect': stopped.append}),
                   (cr, 'healthy', tel.healthy), (cr, 'post', post),
                   (mem1, 'props_json', {'return_value': {'chat_template': 'T',
                                                          'default_generation_settings': {'n_ctx': 2048}}}),
                   (mem1, 'proc_mem', {'return_value': PROC_MEM}), *extra]
        with ExitStack() as stack:
            for obj, attr, how in patches:  # a dict configures a Mock; anything else replaces the attribute
                stack.enter_context(mock.patch.object(obj, attr, **how) if isinstance(how, dict)
                                    else mock.patch.object(obj, attr, how))
            stack.enter_context(redirect_stdout(io.StringIO()))
            mem1.main(['--exp', exp, '--ctx', '2048'] + (['--smoke'] if exp == 'freetext' else []))
        out = next(tmp.iterdir())
        return out, json.loads((out / 'run.json').read_text()), json.loads((out / 'result.json').read_text()), stopped

    def test_complete_launch(self):
        out, run, res, stopped = self.launch()
        self.assertEqual(len(run['matrix']), 4)
        self.assertEqual((run['prime'], run['task']), (S['prime'], 'recall'))
        self.assertTrue(res['complete'], res['not_complete'])
        self.assertEqual(len(stopped), 4)  # every server this launch started was stopped
        self.assertIn('[COMPLETE]', (out / 'report.txt').read_text())
        recs = json.loads((out / 'comparisons.json').read_text())
        self.assertEqual({r['verdict'] for r in recs}, {'COMPARABLE'})
        blocks = [json.loads(l) for l in (out / 'blocks.jsonl').read_text().splitlines()]
        self.assertTrue(all(b['problems'] == [] for b in blocks), [b['problems'] for b in blocks])

    def test_build_sha256_adds_the_symlinked_cpu_library(self):
        """Smoke 20261027T091311Z: repro.build_sha256 skips symlinks, so libggml-cpu.so was missing and run.json
        INVALID. main adds it under its own name with the verified pinned hash; the validation rule is unchanged."""
        out, run, res, _ = self.launch()
        self.assertNotIn(Path(repro.LIBGGML_CPU).name, REPRO_BUILD)
        self.assertEqual(run['build_sha256'], {**REPRO_BUILD, 'libggml-cpu.so': mem1.PINNED[repro.LIBGGML_CPU]})
        self.assertTrue(res['complete'], res['not_complete'])  # main's load_run applied run_problems
        self.assertIn('build_sha256 missing, malformed or without the pinned llama-server and libggml-cpu.so',
                      mem1.run_problems({**run, 'build_sha256': REPRO_BUILD}))  # the rule is unchanged

    def test_an_invalid_run_json_is_reported_without_a_crash(self):
        """Smoke 20261027T091311Z: load_run().run None made coverage raise TypeError before result.json, the report
        and artifacts.sha256 were written."""
        out, run, res, _ = self.launch(extra=[(mem1, 'run_problems', {'return_value': ['forced invalid']})])
        self.assertFalse(res['complete'])
        self.assertEqual(res['coverage'], {})
        self.assertIn('run.json: forced invalid', res['not_complete'])
        self.assertIn('run.json missing or INVALID: completeness cannot be verified', res['not_complete'])
        self.assertIn('[INVALID]: run.json: forced invalid', (out / 'report.txt').read_text())
        self.assertEqual(json.loads((out / 'comparisons.json').read_text()), [])
        self.assertIn('result.json', (out / 'artifacts.sha256').read_text())

    def test_freetext_launch(self):
        out, run, res, _ = self.launch('freetext')
        self.assertEqual(run['task'], 'freetext')
        rows = [json.loads(l) for l in (out / 'requests.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['outcome'] == 'UNSCORED' for r in rows if r['scored']))
        self.assertIn('UNSCORED - MANUAL REVIEW REQUIRED', (out / 'report.txt').read_text())

    def test_skipped_block_makes_the_launch_incomplete(self):
        out, run, res, _ = self.launch(lost_before=(2,))
        self.assertEqual(res['skipped'][0]['block'], 2)
        self.assertFalse(res['complete'])
        self.assertIn('blocks SKIPPED: [2]', res['not_complete'])
        text = (out / 'report.txt').read_text()
        self.assertIn('[INCOMPLETE]', text)
        self.assertIn('NOT COMPARABLE', text)

    def test_a_lingering_root_shell_ends_the_launch(self):
        """A timed-out su call (its root shell may live on and share the output file) or a sampler that outlived its
        join: no further observation or block."""
        class TimedOut(FakeTelemetry):
            readings = 0

            def observe(self, log, pid=None):
                x = super().observe(log, pid)
                self.readings += pid is None
                return {**x, 'diag': [{'op': 'su', 'kind': 'timeout'}]} if pid is not None else x
        tel = TimedOut()
        out, run, res, stopped = self.launch(tel=tel)
        self.assertIn('stopped after block 0: a telemetry su call timed out', res['end_reason'])
        self.assertEqual(len(stopped), 1)
        self.assertFalse(res['complete'])
        self.assertEqual(tel.readings, 2)  # launch start and before block 0: no postflight while a shell may run
        b0 = json.loads((out / 'blocks.jsonl').read_text().splitlines()[0])
        self.assertIn('timed out', b0['end']['skipped'])
        self.assertIn('READING_INVALID', codes(b0['problems']))
        self.assertIn('sampler did not stop', mem1.root_left_running({'sampler': {'alive_after_join': True},
                                                                     'end': {}, 'samples': []}))
        self.assertIsNone(mem1.root_left_running({'sampler': {'alive_after_join': False}, 'end': {}, 'samples': []}))

    def test_an_agent_before_a_block_stops_the_launch(self):
        calls = []

        def agents():
            calls.append(1)
            return '55 claude' if len(calls) == 3 else ''  # block 0: before, after; block 1: before -> found
        out, run, res, stopped = self.launch(agents=agents)
        self.assertIn("stopped before block 1: agent processes running: '55 claude'", res['end_reason'])
        self.assertEqual(len(stopped), 1)
        self.assertFalse(res['complete'])

    def test_unreadable_conditions_stop_the_launch_fail_closed(self):
        class Unreadable(FakeTelemetry):
            n = 0

            def observe(self, log, pid=None):
                if pid is None:  # 1 launch start, 2 before block 0, 3 after it, 4 before block 1
                    self.n += 1
                    if self.n == 4:
                        return cond(mono=round(time.monotonic(), 3), status=None)
                return super().observe(log, pid)
        out, run, res, _ = self.launch(tel=Unreadable())
        self.assertIn('stopped before block 1: Android status None', res['end_reason'])
        self.assertIn('fail closed', res['end_reason'])
        self.assertFalse(res['complete'])

        class FailedRead(Unreadable):  # Round 10 finding 2: parseable values beside a failed dumpsys read
            def observe(self, log, pid=None):
                x = super().observe(log, pid)
                return {**x, 'status': 0, 'dump_rc': 255} if pid is None and self.n == 4 else x
        out, run, res, _ = self.launch(tel=FailedRead())
        self.assertIn('stopped before block 1: Android status 0', res['end_reason'])
        self.assertIn('safety reads failed', res['end_reason'])


if __name__ == '__main__':
    unittest.main()
