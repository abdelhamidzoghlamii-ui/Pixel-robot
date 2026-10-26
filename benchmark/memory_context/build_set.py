#!/usr/bin/env python3
"""MEM-1: builds the frozen memory/question set memory_set.json. No server, no inference: token counts come from
llama-tokenize of the original b1609 build (same commit e1a1abb7 as the robot's dotprod build, vocabulary only).

Every fact, name and preference is invented. Six fixed probe entries sit at the same relative positions in every
memory length (index floor(f * (n - 1)) for f = 0, 1/4, 1/2, 3/4, 1; the near-miss right after 1/2); filler distractors from one frozen shuffled list fill the rest, so a
longer memory contains every shorter memory's fillers. Dates (2026, written 'Jan 5') rise strictly with position (oldest first).
Two tasks share the memory and the questions: 'recall' (scored: the reply is one JSON object {"answer": VALUE},
graded by exact match against the frozen accept/stale/distractor values) and 'freetext' (the earlier one-sentence
instructions; replies kept verbatim, UNSCORED, for manual review).

Usage (Debian or Termux): python3 -B build_set.py [--check]   --check: rebuild in memory, compare with the file
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import chat  # noqa: E402  the deployed Gemma 4 prompt builder (TPL-FIX)

OUT = HERE / 'memory_set.json'
HOME = Path('/data/data/com.termux/files/home')
if not HOME.is_dir():
    HOME = Path('/termux-home')  # the same files from Debian
TOKENIZER = HOME / 'llama.cpp/build/bin/llama-tokenize'
MODEL = HOME / 'models/gemma-4-E2B-it-Q4_0.gguf'
TARGETS = (0, 128, 256, 512, 1024, 2048)  # 2048: only the optional ctx 4096 long-memory condition
START, SPAN_DAYS = date(2026, 1, 5), 175

SYSTEM = ("You are the memory assistant of a small home robot. Answer the user's question using only the memory "
          "entries below. Entries are listed oldest first; if two entries conflict, the newer entry is correct. "
          'Reply with exactly one line of JSON and nothing else, in the form {"answer": "VALUE"}. VALUE is the '
          "shortest exact name, number, word or phrase from the memory that answers the question. If the memory "
          "does not contain the answer, VALUE is UNKNOWN.")
FREETEXT_SYSTEM = ("You are the memory assistant of a small home robot. Answer the user's question using only the "
                   "memory entries below. Entries are listed oldest first; if two entries conflict, the newer entry "
                   "is correct. Answer in one short sentence. If the memory does not contain the answer, reply "
                   "exactly: I don't know.")
HEADER = 'Memory (all dates in 2026, oldest first):'
EMPTY = '(no entries)'
WARMUP = chat.build_prompt('gemma', [{'role': 'user', 'content': 'Hi'}])  # mem1.WARMUP, the block's first request

# position -> probe entry; positions are fractions of the last index
PROBES = [('begin', 0.0, "Nadia's cat is named Pepper."),
          ('stale', 0.25, 'The spare robot charger is kept in the kitchen drawer.'),
          ('mid', 0.5, "Oskar's bicycle lock code is 4417."),
          ('near', None, "Oskar's gym locker code is 9032."),  # placed right after mid
          ('update', 0.75, 'Update: the spare robot charger is now kept on the hallway shelf.'),
          ('end', 1.0, 'Leo takes his tea with honey and no milk.')]

# frozen answer key, in mem1.normalize form (NFKC, casefolded, single spaces, no trailing period): accept = the
# answers counted CORRECT; stale = the superseded entry; distractor = a near-miss in the memory. Any other value of a
# present question is WRONG; for an absent question (and every question at memory 0) only UNKNOWN is right.
QUESTIONS = [
    {'id': 'q_begin', 'text': "What is the name of Nadia's cat?", 'kind': 'present', 'probe': 'begin',
     'accept': ['pepper'], 'stale': [], 'distractor': []},
    {'id': 'q_mid', 'text': "What is Oskar's bicycle lock code?", 'kind': 'present', 'probe': 'mid',
     'accept': ['4417'], 'stale': [], 'distractor': ['9032']},
    {'id': 'q_end', 'text': 'What does Leo take in his tea?', 'kind': 'present', 'probe': 'end',
     'accept': ['honey', 'honey and no milk', 'honey, no milk'], 'stale': [], 'distractor': ['milk', 'honey and milk']},
    {'id': 'q_update', 'text': 'Where is the spare robot charger kept now?', 'kind': 'present', 'probe': 'update',
     'accept': ['hallway shelf', 'the hallway shelf', 'on the hallway shelf', 'hallway'],
     'stale': ['kitchen drawer', 'the kitchen drawer', 'in the kitchen drawer', 'kitchen'], 'distractor': []},
    {'id': 'q_absent_pet', 'text': "What is the name of Nadia's dog?", 'kind': 'absent', 'probe': None,
     'accept': [], 'stale': [], 'distractor': ['pepper']},
    {'id': 'q_absent_time', 'text': "At what time is Oskar's dentist appointment?", 'kind': 'absent', 'probe': None,
     'accept': [], 'stale': [], 'distractor': []},
]
# unscored request that every block sends before its scored questions: in a CACHED block it fills the slot with the
# instructions + memory prefix (class FIRST: the cold-first-request cost), so every scored question is a reuse
PRIME = {'id': 'prime', 'text': 'Reply with the single word: ready.'}

NAMES = ['Ines', 'Marta', 'Yusuf', 'Petra', 'Hugo', 'Selma', 'Arjun', 'Clara', 'Tobias', 'Wen', 'Rafael', 'Greta',
         'Milan', 'Esra']
FILLERS = [  # (template, values per name); no probe person, charger, tea, lock code, dog or dentist
    ('{n} watered the {v} on the balcony.', 'basil mint ferns lavender tomatoes chillies orchids rosemary thyme '
     'strawberries geraniums parsley cactus succulents'),
    ("{n}'s favourite board game is {v}.", 'chess Go checkers backgammon Scrabble Risk Catan Carcassonne Dominion '
     'Azul Pandemic Ticket-to-Ride Mahjong Monopoly'),
    ('{n} asked the robot to play {v} music in the evening.', 'jazz folk piano reggae classical blues swing salsa '
     'ambient opera soul funk tango gospel'),
    ('{n} left a {v} umbrella by the front door.', 'blue red yellow black grey white striped dotted orange purple '
     'pink brown silver beige'),
    ('{n} borrowed the {v} from the garage.', 'ladder drill hammer saw wrench pliers shovel rake hose toolbox '
     'stapler sander broom bucket'),
    ('{n} said the {v} window sticks when it rains.', 'bathroom bedroom attic study pantry office laundry dining '
     'nursery guest stairwell porch cellar loft'),
    ("{n}'s rabbit is named {v}.", 'Clover Biscuit Hazel Juniper Maple Nutmeg Poppy Thistle Willow Bramble '
     'Saffron Truffle Pebble Sorrel'),
    ('{n} prefers the living room lights at {v} percent.', '20 25 30 35 40 45 50 55 60 65 70 75 80 85'),
    ('{n} is learning to bake {v}.', 'sourdough bagels croissants brioche focaccia pretzels scones muffins '
     'baguettes cornbread pita naan rye challah'),
    ("{n}'s spin class starts at {v} on Saturdays.", '7:00 7:30 8:00 8:30 9:00 9:30 10:00 10:30 11:00 11:30 '
     '12:00 12:30 13:00 13:30'),
    ('{n} reminded the robot that the {v} recycling goes out on Thursdays.', 'glass paper plastic cardboard metal '
     'battery textile compost glass-bottle carton tin can bottle newspaper'),
    ('{n} hung a {v} painting in the corridor.', 'harbour mountain forest lighthouse desert river meadow canyon '
     'glacier vineyard island windmill sunset waterfall'),
]
SEED = 20261024


def fillers():
    rows = [t.format(n=n, v=v) for t, vals in FILLERS for n, v in zip(NAMES, vals.split())]
    assert len(rows) == len(NAMES) * len(FILLERS) == len(set(rows))
    random.Random(SEED).shuffle(rows)
    return rows


def entries(k, pool):
    """Probe entries at their fixed relative positions plus the first k fillers, each with a rising date."""
    n = k + len(PROBES)
    at = {pid: int((f or 0) * (n - 1)) for pid, f, _ in PROBES}
    at['near'] = at['mid'] + 1  # the near-miss distractor always directly follows its target
    at['update'] = max(at['update'], at['mid'] + 2)
    slots = {at[pid]: (pid, text) for pid, _, text in PROBES}
    if len(slots) != len(PROBES):
        raise ValueError(f'{n} entries are too few for distinct probe positions')
    it = iter(pool[:k])
    days = [round(i * SPAN_DAYS / (n - 1)) for i in range(n)]
    assert all(a < b for a, b in zip(days, days[1:])), 'dates must rise strictly'
    return [{'date': f'{START + timedelta(d):%b} {(START + timedelta(d)).day}', 'probe': slots[i][0] if i in slots else None,
             'text': slots[i][1] if i in slots else next(it)} for i, d in enumerate(days)]


def valid(k, pool):
    try:
        entries(k, pool)
        return True
    except ValueError:
        return False


def memory_text(ents):
    return HEADER + '\n' + ('\n'.join(f"- {e['date']}: {e['text']}" for e in ents) if ents else EMPTY)


def prompt(mem, question, system=SYSTEM):
    """The exact /completion prompt: chat.build_prompt with a system turn (instructions + memory) and the question."""
    return chat.build_prompt('gemma', [{'role': 'system', 'content': system + '\n\n' + mem},
                                       {'role': 'user', 'content': question}])


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def tokenize(text):
    """Token ids as /completion sees them (BOS added, special tokens parsed), from llama-tokenize."""
    with tempfile.NamedTemporaryFile('w', suffix='.txt') as f:
        f.write(text)
        f.flush()
        out = subprocess.run([str(TOKENIZER), '-m', str(MODEL), '-f', f.name, '--ids', '--log-disable'],
                             capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def lcp(lists):
    n = 0
    while all(len(l) > n and l[n] == lists[0][n] for l in lists):
        n += 1
    return n


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def task_prompts(mem, system, target, warm_ids):
    """Token data of one task's prompts at one memory: every question, the prime and their shared prefix, and the
    token LCP of every pair a CACHED block can send in a row (warm-up -> prime, prime -> question, question ->
    question): the server's expected cache_n, frozen so the report never takes it from a run's own records."""
    ids = {q['id']: tokenize(prompt(mem, q['text'], system)) for q in QUESTIONS}
    pre = prompt(mem, QUESTIONS[0]['text'], system).rsplit(QUESTIONS[0]['text'], 1)[0]  # same for every question
    pre_ids = tokenize(pre)
    prime_ids = tokenize(prompt(mem, PRIME['text'], system))
    for q in QUESTIONS:  # the shared prefix must tokenize the same alone as inside the prompt
        assert ids[q['id']][:len(pre_ids)] == pre_ids, (target, q['id'])
    assert prime_ids[:len(pre_ids)] == pre_ids, (target, 'prime')
    return {'shared_prefix_tokens': lcp(list(ids.values())), 'prefix_text_tokens': len(pre_ids),
            'prompts': {q['id']: {'tokens': len(ids[q['id']]), 'ids_sha256': sha(json.dumps(ids[q['id']])),
                                  'text_sha256': sha(prompt(mem, q['text'], system)),
                                  'lcp': {k: lcp([ids[q['id']], v]) for k, v in ids.items() if k != q['id']}}
                        for q in QUESTIONS},
            'prime_prompt': {'tokens': len(prime_ids), 'ids_sha256': sha(json.dumps(prime_ids)),
                             'text_sha256': sha(prompt(mem, PRIME['text'], system)),
                             'lcp_with_scored': min(lcp([prime_ids, v]) for v in ids.values()),
                             'lcp_warmup': lcp([warm_ids, prime_ids]),
                             'lcp': {k: lcp([prime_ids, v]) for k, v in ids.items()}}}, ids


def build():
    pool = fillers()
    q0 = QUESTIONS[0]['text']
    base = len(tokenize(prompt(memory_text([]), q0)))
    warm_ids = tokenize(WARMUP)
    ks = [k for k in range(1, len(pool) + 1) if valid(k, pool)]  # filler counts with distinct probe slots
    memo = {}

    def mem_tokens(k):  # memory tokens grow with k
        if k not in memo:
            memo[k] = len(tokenize(prompt(memory_text(entries(k, pool)), q0))) - base
        return memo[k]
    lengths = {}
    for target in TARGETS:
        if target == 0:
            ents = []
        else:  # smallest valid k reaching the target (bisection over ks), or the valid k below if that is closer
            lo, hi = 0, len(ks) - 1
            if mem_tokens(ks[hi]) < target:
                raise SystemExit(f'filler pool too small for {target} tokens ({mem_tokens(ks[hi])})')
            while lo < hi:
                mid = (lo + hi) // 2
                lo, hi = (mid + 1, hi) if mem_tokens(ks[mid]) < target else (lo, mid)
            if lo > 0 and target - mem_tokens(ks[lo - 1]) < mem_tokens(ks[lo]) - target:
                lo -= 1
            ents = entries(ks[lo], pool)
        mem = memory_text(ents)
        recall, ids = task_prompts(mem, SYSTEM, target, warm_ids)
        freetext, _ = task_prompts(mem, FREETEXT_SYSTEM, target, warm_ids)
        mem_tok = len(ids[QUESTIONS[0]['id']]) - base
        lengths[str(target)] = {
            'target_tokens': target, 'fillers': len(ents) - len(PROBES) if ents else 0, 'entries': ents,
            'memory_text': mem, 'memory_tokens': mem_tok, 'deviation_tokens': mem_tok - target,
            'deviation_pct': round(100 * (mem_tok - target) / target, 2) if target else 0.0,
            **recall, 'freetext': freetext}
        print(f'target {target:>5}: {lengths[str(target)]["fillers"]:>3} fillers, memory {mem_tok} tokens '
              f'({mem_tok - target:+d}), prompts {min(len(v) for v in ids.values())}-{max(len(v) for v in ids.values())}'
              f', shared prefix {lengths[str(target)]["shared_prefix_tokens"]}', flush=True)
    return {'task': 'MEM-1', 'note': 'synthetic, invented facts only; frozen answer key; built by build_set.py',
            'system': SYSTEM, 'freetext_system': FREETEXT_SYSTEM, 'header': HEADER, 'empty': EMPTY,
            'probes': [list(p) for p in PROBES],
            'questions': QUESTIONS, 'prime': PRIME, 'filler_pool': pool, 'seed': SEED, 'lengths': lengths,
            'tokenizer': {'binary': str(TOKENIZER), 'binary_sha256': file_sha(TOKENIZER), 'commit': 'e1a1abb7',
                          'model': MODEL.name, 'model_sha256': file_sha(MODEL),
                          'prompt_builder': "chat.build_prompt('gemma', [system, user])",
                          'counting': 'BOS added, special tokens parsed (as /completion and /tokenize add_special)'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    data = json.dumps(build(), indent=1) + '\n'
    if a.check:
        same = OUT.read_text() == data
        print('memory_set.json', 'matches a fresh build' if same else 'DIFFERS from a fresh build')
        sys.exit(0 if same else 1)
    OUT.write_text(data)
    print(f'wrote {OUT} sha256 {sha(data)}')


if __name__ == '__main__':
    main()
