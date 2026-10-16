"""Gemma picker (DECISIONS #109, #116, #125, #131): Gemma chooses one behaviour mode by letter scoring; Python
computes which modes are eligible and keeps every motor decision (#104). Gemma never outputs a motor command.

Copied, so the prompt and scoring are the campaign's (benchmark/ is not deployed):
- benchmark/strategic_selector/v3/adapters.py: S1O.encode(), completion_probs(), prompt_ids(), decide();
- benchmark/coresidency/coresidency.py: make_selector() (connection setup), select() (choice), healthy();
- system-one-open s1/schema.py at 77f1f7cc (github.com/mithalouni/system-one-open): LETTERS, render_option(),
  truncate_tokens() and build(), trimmed to one choice question without shuffling. That code is MIT licensed:

  MIT License

  Copyright (c) 2026 Rami Mithalouni

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
  the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import json
import math
import queue
import threading
import time
import urllib.request

URL = 'http://127.0.0.1:8080'   # setup_q4's llama-server (the owner starts it before the mission)
TIMEOUT_S = 10                  # per HTTP request
N_PROBS = 1000
MIN_GAP_S = 5                   # an event call starts no sooner than this after the previous call started
HEARTBEAT_S = 20                # otherwise one call every 20 s
INSTRUCTION = 'Choose the single best next action for the robot.'
# Offered in this order; APPROACH and GREET only while a person is in the latest detections.
MODES = {'EXPLORE': 'keep exploring', 'APPROACH': 'go toward the person',
         'GREET': 'stop and greet the person', 'LOOK': 'look around in place'}
PERSON_MODES = ('APPROACH', 'GREET')
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'  # s1/schema.py
MAX_STATE_TOKENS = 1536                                          # s1/schema.py build() default


def eligible(person_visible):
    return [m for m in MODES if person_visible or m not in PERSON_MODES]


def situation(mission, results, distance, mode, mode_s, last_moves, rooms):
    """The picker's state text without the M2 block (the worker appends it). results: legacy detection tuples."""
    scene = ', '.join(f'{r[0]} ({r[2]}, {r[3]})' for r in results) or 'nothing detected'
    return '\n'.join([
        f'Mission: {mission}',
        f'Scene: {scene}',
        f"Ultrasonic distance ahead: {'no reading' if distance == 999 else f'{distance} cm'}",
        f'Current action: {MODES[mode]} (for {mode_s:.0f} s)',
        f"Last moves: {', '.join(last_moves[-3:]) or 'none'}",
        f"Rooms seen: {', '.join(rooms) or 'none'}"])


# Copied from benchmark/coresidency/coresidency.py: post(), with the 10 s timeout; GET when body is None.
def http(path, body=None, url=URL):
    req = urllib.request.Request(url + path, None if body is None else json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


class Gemma:
    """adapters.S1O pointed at setup_q4's server (as coresidency.make_selector). Raises unless /health is ok."""

    def __init__(self, post=http):
        self.post = post
        self.check = lambda: None  # the Picker sets it: raises Cancelled before any request it must not send
        if post('/health').get('status') != 'ok':
            raise RuntimeError(f'{URL}/health not ok')
        with_bos = post('/tokenize', {'content': '', 'add_special': True})['tokens']
        self.bos_token_id = with_bos[0] if with_bos else None
        self.letter_ids = [self.encode(L)[0] for L in LETTERS[:len(MODES)]]
        assert all(len(self.encode(L)) == 1 for L in LETTERS[:len(MODES)])
        self.prompt_tokens = None

    def request(self, path, body):
        self.check()
        return self.post(path, body)

    def encode(self, text):
        return self.request('/tokenize', {'content': text, 'add_special': False})['tokens']

    def prompt_ids(self, state, phrases):
        """s1/schema.py build() for one 'choice' question, options in the given order, no descriptions."""
        ids = ([self.bos_token_id] if self.bos_token_id is not None else []) + self.encode('<state>\n')
        state_ids = self.encode(state)
        if len(state_ids) > MAX_STATE_TOKENS:  # truncate_tokens(): keep head 70 % and tail
            head = int(MAX_STATE_TOKENS * 0.7)
            tail = MAX_STATE_TOKENS - head - 8
            state_ids = state_ids[:head] + self.encode('\n[... truncated ...]\n') + state_ids[-tail:]
        ids += state_ids + self.encode('\n</state>\n')
        ids += self.encode(''.join([f'\nQuestion (choice): {INSTRUCTION.strip()}', '\nOptions:',
                                    *(f'\n({LETTERS[j]}) {p.strip()}' for j, p in enumerate(phrases)),
                                    '\nAnswer: (']))
        return ids

    def decide(self, state, phrases):
        """{phrase: probability}: the letter probabilities at the 'Answer: (' slot, renormalised (S1O.decide)."""
        ids = self.prompt_ids(state, phrases)
        res = self.request('/completion', {'prompt': ids, 'n_predict': 1, 'n_probs': N_PROBS, 'post_sampling_probs': False,
                                        'temperature': 0, 'cache_prompt': False})
        by_id = {e['id']: (e['prob'] if 'prob' in e else math.exp(e['logprob']))
                 for e in res['completion_probabilities'][0]['top_logprobs']}
        raw = {}
        for j, phrase in enumerate(phrases):
            p = by_id.get(self.letter_ids[j])
            if p is None:
                raise RuntimeError(f'letter {LETTERS[j]} outside top-{N_PROBS}')
            raw[phrase] = p
        mass = sum(raw.values())
        self.prompt_tokens = len(ids)
        return {k: v / mass for k, v in raw.items()}


class Cadence:
    """When to call (owner D2). An event (person appears or disappears; the ultrasonic becomes blocked) calls at
    once, but not sooner than MIN_GAP_S after the previous call started; otherwise every HEARTBEAT_S. An event that
    arrives while a call is not allowed stays pending (the latest one wins). The first call is due at once."""

    def __init__(self):
        self.last_start = None
        self.person = None
        self.blocked = False
        self.pending = 'start'

    def due(self, now, person, blocked, ready):
        """The trigger name when a call starts now, else None. ready: False while a call is in flight, at SEVERE or
        without a 640 frame; the observed state is still updated."""
        if self.person is not None and person != self.person:
            self.pending = 'person_appeared' if person else 'person_left'
        if blocked and not self.blocked:
            self.pending = 'blocked'
        self.person, self.blocked = person, blocked
        if not ready:
            return None
        if self.last_start is not None and now - self.last_start < MIN_GAP_S:
            return None
        trigger = self.pending or ('heartbeat' if now - self.last_start >= HEARTBEAT_S else None)
        if trigger:
            self.pending, self.last_start = None, now
        return trigger


class Cancelled(Exception):
    """The mission withdrew permission (left ACTIVE, Android status SEVERE, or stop) while a pick was running."""


class Picker:
    """One background worker thread runs each pick: M2 (relate.Relate) on the latest 640 frame, then Gemma. It gets
    plain data in a request and returns a result; it never touches motors, camera, heat state or the robot. Any
    failure becomes an EXPLORE result with the reason (a fallback, never a stop). The mission thread calls allow()
    while ACTIVE and below SEVERE and forbid() otherwise; a running pick checks before M2, after M2 and before every
    HTTP request, and a forbid() cancels it for good (a later allow() does not revive it). Only a request already
    sent (<= TIMEOUT_S) or an M2 inference already started can still finish."""

    def __init__(self, gemma, relate):
        self.gemma, self.relate = gemma, relate
        self.requests, self.results = queue.Queue(), queue.Queue()
        self.allowed = threading.Event()
        self.generation = 0  # forbid() bumps it; a pick runs only while it is unchanged
        ready = queue.Queue()
        self.thread = threading.Thread(target=self.loop, args=(ready,), daemon=True)
        self.thread.start()
        error = ready.get()
        if error:
            self.thread.join()
            raise error

    def loop(self, ready):
        try:
            self.relate.pin_caller()  # this thread runs M2: BIG 6-7, verified
        except BaseException as e:
            ready.put(e)
            return
        ready.put(None)
        while (request := self.requests.get()) is not None:
            self.results.put(self.pick(request))

    def allow(self):
        self.allowed.set()

    def forbid(self):
        if self.allowed.is_set():
            self.generation += 1
            self.allowed.clear()

    def pick(self, request):
        out = {'slot': request['slot'], 'trigger': request['trigger'],
               'eligible': [MODES[m] for m in request['eligible']], 'mode': 'EXPLORE', 'probs': None,
               'm2_ms': None, 'm2_boxes': None, 'm2_relations': None, 'm2_repinned': None,
               'picker_ms': None, 'prompt_tokens': None, 'fallback': None, 'cancelled': False}
        generation = self.generation

        def check():
            if generation != self.generation or not self.allowed.is_set():
                raise Cancelled('optional work not allowed (left ACTIVE, SEVERE or stop)')
        self.gemma.check = check
        stage = 'M2'
        try:
            check()
            affinity = self.relate.check_affinity()
            out['m2_repinned'] = affinity['repinned']
            if not affinity['ok']:
                raise RuntimeError(f"affinity: {affinity['errors']}")
            check()  # a forbid() during the affinity check: no M2 run
            text, out['m2_boxes'], out['m2_relations'], out['m2_ms'] = self.relate.relations(*request['frame_640'])
            check()
            stage = 'Gemma'
            phrases = out['eligible']
            started = time.perf_counter()
            dist = self.gemma.decide(request['situation'] + '\n' + text, phrases)
            check()  # a forbid() while /completion was in flight cancels it too
            out['picker_ms'] = (time.perf_counter() - started) * 1000
            out['probs'] = dist
            out['prompt_tokens'] = self.gemma.prompt_tokens
            choice = max(phrases, key=lambda k: dist[k])  # ties: first offered, as in ladder.py
            out['mode'] = request['eligible'][phrases.index(choice)]
        except Exception as e:
            out['mode'] = 'EXPLORE'
            out['fallback'] = f'{stage}: {type(e).__name__}: {e}'
            out['cancelled'] = isinstance(e, Cancelled)
        return out

    def submit(self, request):
        self.requests.put(request)

    def result(self):
        """A finished pick, or None (never waits)."""
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def stop(self, timeout=TIMEOUT_S + 5):
        """Cancels the running pick (after its current request) and ends the worker; True when it has ended."""
        self.forbid()
        self.requests.put(None)
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def config(self):
        return {'url': URL, 'options': list(MODES.values()), 'instruction': INSTRUCTION, 'timeout_s': TIMEOUT_S,
                'cadence': {'events': ['person_appeared', 'person_left', 'blocked'], 'min_gap_s': MIN_GAP_S,
                            'heartbeat_s': HEARTBEAT_S, 'phases': ['ACTIVE'], 'not_at_status': 'SEVERE or above'},
                'm2': {'cpus': sorted(self.relate.cpus), 'threads': 2, 'worker_tids': self.relate.worker_tids,
                       'caller_tid': self.relate.caller_tid, 'affinity': self.relate.affinity(),
                       'parity': 'graph SHA-256 equals the parity-passed graph (#130)'}}
