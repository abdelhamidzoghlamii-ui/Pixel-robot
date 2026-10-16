"""picker.py and relate.py offline: fake HTTP, ORT head, /proc thread lists, os.sched_* and clock. No server, model
or pinning of real threads."""
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import picker as P
import relate as RL

S1O_SRC = '/data/data/com.termux/files/home/sel-candidates/s1o/src'  # outside the repo; parity test only


def fake_server(probs=None, health='ok'):
    """A /health, /tokenize, /completion stand-in. Tokens: one per character (ord + 10); bos = 2.
    probs: {letter: probability} at the answer slot."""
    calls = []

    def post(path, body=None):
        calls.append((path, body))
        if path == '/health':
            return {'status': health}
        if path == '/tokenize':
            return {'tokens': ([2] if body['add_special'] else []) + [ord(c) + 10 for c in body['content']]}
        if path == '/completion':
            top = [{'id': ord(L) + 10, 'logprob': __import__('math').log(p)} for L, p in (probs or {}).items()]
            return {'completion_probabilities': [{'top_logprobs': top + [{'id': 5, 'logprob': -0.1}]}]}
        raise AssertionError(path)
    post.calls = calls
    return post


def ids(text):
    return [ord(c) + 10 for c in text]


class TextTest(unittest.TestCase):
    def test_eligibility_fixed_order(self):
        self.assertEqual(P.eligible(False), ['EXPLORE', 'LOOK'])
        self.assertEqual(P.eligible(True), ['EXPLORE', 'APPROACH', 'GREET', 'LOOK'])
        self.assertEqual(list(P.MODES.values()), ['keep exploring', 'go toward the person',
                                                  'stop and greet the person', 'look around in place'])

    def test_situation_template(self):
        results = [('person', 0.9, 'center', 'close', 320, 300, 200, 400), ('chair', 0.5, 'left', 'far', 1, 1, 1, 1)]
        self.assertEqual(P.situation('explore and map the rooms', results, 37, 'APPROACH', 12.4,
                                     ['LEFT', 'FORWARD', 'HOLD', 'FORWARD'], ['kitchen', 'bedroom']),
                         'Mission: explore and map the rooms\n'
                         'Scene: person (center, close), chair (left, far)\n'
                         'Ultrasonic distance ahead: 37 cm\n'
                         'Current action: go toward the person (for 12 s)\n'
                         'Last moves: FORWARD, HOLD, FORWARD\n'
                         'Rooms seen: kitchen, bedroom')
        self.assertEqual(P.situation('m', [], 999, 'EXPLORE', 0, [], []),
                         'Mission: m\nScene: nothing detected\nUltrasonic distance ahead: no reading\n'
                         'Current action: keep exploring (for 0 s)\nLast moves: none\nRooms seen: none')


class GemmaTest(unittest.TestCase):
    def test_health_not_ok_raises(self):
        with self.assertRaises(RuntimeError):
            P.Gemma(fake_server(health='loading'))

    def test_http_timeout_and_methods(self):
        response = Mock()
        response.__enter__ = Mock(return_value=SimpleNamespace(read=lambda: b'{"status": "ok"}'))
        response.__exit__ = Mock(return_value=False)
        with patch.object(P.urllib.request, 'urlopen', return_value=response) as urlopen:
            self.assertEqual(P.http('/health'), {'status': 'ok'})
            P.http('/tokenize', {'content': 'x'})
        get, post = [c.args[0] for c in urlopen.call_args_list]
        self.assertEqual((get.full_url, get.get_method(), get.data), ('http://127.0.0.1:8080/health', 'GET', None))
        self.assertEqual((post.get_method(), post.data), ('POST', b'{"content": "x"}'))
        self.assertTrue(all(c.kwargs['timeout'] == 10 for c in urlopen.call_args_list))

    def test_prompt_ids_exact(self):
        g = P.Gemma(fake_server())
        self.assertEqual(g.letter_ids, ids('ABCD'))
        got = g.prompt_ids('state text', ['keep exploring', 'look around in place'])
        self.assertEqual(got, [2] + ids('<state>\n') + ids('state text') + ids('\n</state>\n') + ids(
            '\nQuestion (choice): Choose the single best next action for the robot.\nOptions:'
            '\n(A) keep exploring\n(B) look around in place\nAnswer: ('))

    @unittest.skipUnless(os.path.isdir(S1O_SRC), 'system-one-open source not on this phone')
    def test_prompt_ids_equal_system_one_open_build(self):
        """The copy against the original s1/schema.build as adapters.S1O.prompt_ids calls it (test only)."""
        with patch.object(sys, 'path', [S1O_SRC] + sys.path), patch.dict(sys.modules):
            from s1.schema import Example, Q, build
        g = P.Gemma(fake_server())
        tok = SimpleNamespace(bos_token_id=2, encode=lambda text, add_special_tokens=False: ids(text))
        for state in ('short state', 'x' * 2000):  # the second one is truncated (> 1536 tokens)
            for phrases in (P.eligible(False), [P.MODES[m] for m in P.eligible(True)]):
                q = Q(P.INSTRUCTION, phrases, -1, kind='choice', descs=[''] * len(phrases))
                original = build(Example(state, [q]), tok, shuffle_choice=False, train=False)['ids']
                self.assertEqual(g.prompt_ids(state, phrases), original)

    def test_decide_scores_letters_and_renormalises(self):
        post = fake_server({'A': 0.2, 'B': 0.6, 'C': 0.1, 'D': 0.1})
        g = P.Gemma(post)
        phrases = [P.MODES[m] for m in P.eligible(True)]
        dist = g.decide('s', phrases)
        self.assertAlmostEqual(dist['go toward the person'], 0.6)
        self.assertAlmostEqual(sum(dist.values()), 1.0)
        body = post.calls[-1][1]
        self.assertEqual({k: v for k, v in body.items() if k != 'prompt'},
                         {'n_predict': 1, 'n_probs': 1000, 'post_sampling_probs': False, 'temperature': 0,
                          'cache_prompt': False})
        self.assertEqual(g.prompt_tokens, len(body['prompt']))

    def test_letter_outside_top_raises(self):
        g = P.Gemma(fake_server({'A': 0.9}))
        with self.assertRaises(RuntimeError):
            g.decide('s', ['keep exploring', 'look around in place'])


class CadenceTest(unittest.TestCase):
    def test_first_call_heartbeat_and_minimum_gap(self):
        c = P.Cadence()
        self.assertEqual(c.due(0, False, False, True), 'start')
        self.assertIsNone(c.due(19.9, False, False, True))
        self.assertEqual(c.due(20, False, False, True), 'heartbeat')
        self.assertIsNone(c.due(24, True, False, True))            # person appeared 4 s after: pending
        self.assertEqual(c.due(25, True, False, True), 'person_appeared')
        self.assertIsNone(c.due(30, True, False, True))
        self.assertEqual(c.due(31, False, False, True), 'person_left')  # >= 5 s: at once
        self.assertEqual(c.due(40, False, True, True), 'blocked')
        self.assertIsNone(c.due(46, False, True, True))             # still blocked: no new event
        self.assertEqual(c.due(60, False, True, True), 'heartbeat')

    def test_not_ready_keeps_event_pending_and_one_in_flight(self):
        c = P.Cadence()
        self.assertEqual(c.due(0, False, False, True), 'start')
        self.assertIsNone(c.due(10, True, False, False))   # in flight / SEVERE / no 640 frame
        self.assertIsNone(c.due(30, True, False, False))   # also no heartbeat while not ready
        self.assertEqual(c.due(31, True, False, True), 'person_appeared')

    def test_latest_event_wins_and_first_observation_is_not_an_event(self):
        c = P.Cadence()
        self.assertEqual(c.due(0, True, False, True), 'start')
        self.assertIsNone(c.due(1, True, True, True))      # blocked inside the 5 s gap
        self.assertIsNone(c.due(2, False, True, True))     # then the person left
        self.assertEqual(c.due(5, False, True, True), 'person_left')


class FakeRelate:
    cpus = {6, 7}
    worker_tids = [11]

    def __init__(self, fail_pin=False):
        self.fail_pin = fail_pin
        self.caller_tid = None
        self.check = {'ok': True, 'repinned': [], 'errors': {}}
        self.relations = Mock(return_value=('M2 relations:\n- person [0] next to chair [1].', 2, 1, 640.0))

    def pin_caller(self):
        if self.fail_pin:
            raise RuntimeError('refusing to pin')
        self.caller_tid = threading.get_native_id()

    def check_affinity(self):
        return self.check

    def affinity(self):
        return {11: [6, 7], self.caller_tid: [6, 7]}


def request(eligible=('EXPLORE', 'APPROACH', 'GREET', 'LOOK'), slot=1):
    return {'slot': slot, 'trigger': 'start', 'eligible': list(eligible), 'frame_640': ('image', ['boxes']),
            'situation': 'Mission: m'}


class PickerTest(unittest.TestCase):
    def make(self, probs=None, relate=None):
        gemma = SimpleNamespace(prompt_tokens=99, decide=Mock(return_value=probs or {
            'keep exploring': .1, 'go toward the person': .2, 'stop and greet the person': .6,
            'look around in place': .1}))
        p = P.Picker(gemma, relate or FakeRelate())
        self.addCleanup(p.stop)
        p.allow()
        return p, gemma

    def run_pick(self, p, req):
        p.submit(req)
        for _ in range(500):
            r = p.result()
            if r:
                return r
            threading.Event().wait(0.01)
        self.fail('no result')

    def test_pick_runs_m2_then_gemma_in_the_pinned_worker(self):
        p, gemma = self.make()
        self.assertNotEqual(p.relate.caller_tid, threading.get_native_id())  # the worker, not the main thread
        r = self.run_pick(p, request())
        self.assertEqual((r['mode'], r['fallback'], r['m2_boxes'], r['m2_relations'], r['m2_ms']),
                         ('GREET', None, 2, 1, 640.0))
        self.assertEqual(r['eligible'], list(P.MODES.values()))
        self.assertEqual(r['prompt_tokens'], 99)
        p.relate.relations.assert_called_once_with('image', ['boxes'])
        gemma.decide.assert_called_once_with(
            'Mission: m\nM2 relations:\n- person [0] next to chair [1].', list(P.MODES.values()))

    def test_ties_go_to_the_first_offered(self):
        p, _ = self.make({'keep exploring': .5, 'look around in place': .5})
        self.assertEqual(self.run_pick(p, request(('EXPLORE', 'LOOK')))['mode'], 'EXPLORE')

    def test_failures_fall_back_to_explore(self):
        for name in ('m2', 'affinity', 'gemma', 'phase'):
            with self.subTest(name=name):
                p, gemma = self.make()
                if name == 'm2':
                    p.relate.relations.side_effect = RuntimeError('ort')
                elif name == 'affinity':
                    p.relate.check = {'ok': False, 'repinned': [11], 'errors': {11: 'gone'}}
                elif name == 'gemma':
                    gemma.decide.side_effect = TimeoutError('timed out')
                else:
                    p.forbid()
                r = self.run_pick(p, request())
                self.assertEqual(r['mode'], 'EXPLORE')
                self.assertTrue(r['fallback'].startswith({'m2': 'M2: RuntimeError', 'affinity': 'M2: RuntimeError: affinity',
                                                          'gemma': 'Gemma: TimeoutError', 'phase': 'M2: Cancelled'}[name]))
                self.assertEqual(r['cancelled'], name == 'phase')
                if name in ('m2', 'affinity', 'phase'):
                    gemma.decide.assert_not_called()
                if name == 'phase':
                    p.relate.relations.assert_not_called()

    def test_phase_left_during_m2_skips_gemma(self):
        p, gemma = self.make()
        p.relate.relations.side_effect = lambda *a: (p.forbid(), ('M2 relations:', 2, 0, 1.0))[1]
        r = self.run_pick(p, request())
        self.assertEqual((r['mode'], r['cancelled']), ('EXPLORE', True))
        self.assertTrue(r['fallback'].startswith('M2: Cancelled'))
        gemma.decide.assert_not_called()

    def real_gemma(self, during_tokenize):
        """A real Gemma on the fake server; during_tokenize(n) runs inside the n-th /tokenize after setup."""
        post = fake_server({'A': .5, 'B': .5})
        g = P.Gemma(post)
        count = [0]

        def hooked(path, body=None):
            if path == '/tokenize':
                count[0] += 1
                during_tokenize(count[0])
            return post(path, body)
        g.post = hooked
        return g, post

    def test_forbid_during_tokenization_sends_no_further_request(self):
        holder = {}
        g, post = self.real_gemma(lambda n: n == 2 and holder['p'].forbid())
        p = P.Picker(g, FakeRelate())
        self.addCleanup(p.stop)
        holder['p'] = p
        p.allow()
        before = len(post.calls)
        r = self.run_pick(p, request(('EXPLORE', 'LOOK')))
        self.assertEqual((r['mode'], r['cancelled'], r['probs']), ('EXPLORE', True, None))
        self.assertTrue(r['fallback'].startswith('Gemma: Cancelled'))
        self.assertEqual([c[0] for c in post.calls[before:]], ['/tokenize', '/tokenize'])  # no /completion
        p.allow()  # a later allow() does not revive it; the next pick runs normally
        self.assertIsNone(p.result())

    def test_forbid_then_allow_during_completion_stays_cancelled(self):
        entered, release = threading.Event(), threading.Event()
        g, post = self.real_gemma(lambda n: None)
        inner = g.post

        def blocking(path, body=None):
            if path == '/completion':
                entered.set()
                release.wait(2)
            return inner(path, body)
        g.post = blocking
        p = P.Picker(g, FakeRelate())
        self.addCleanup(p.stop)
        p.allow()
        p.submit(request(('EXPLORE', 'LOOK')))
        self.assertTrue(entered.wait(2))
        p.forbid()
        p.allow()
        release.set()
        for _ in range(200):
            r = p.result()
            if r:
                break
            threading.Event().wait(0.01)
        self.assertEqual((r['mode'], r['cancelled']), ('EXPLORE', True))
        self.assertTrue(r['fallback'].startswith('Gemma: Cancelled'))

    def test_stop_during_an_http_sequence_ends_the_worker(self):
        entered, release = threading.Event(), threading.Event()

        def slow(n):
            if n == 1:
                entered.set()
                release.wait(2)
        g, post = self.real_gemma(slow)
        p = P.Picker(g, FakeRelate())
        p.allow()
        before = len(post.calls)
        p.submit(request(('EXPLORE', 'LOOK')))
        self.assertTrue(entered.wait(2))
        threading.Timer(0.2, release.set).start()
        self.assertTrue(p.stop())  # returns only once the worker has ended
        self.assertFalse(p.thread.is_alive())
        self.assertEqual([c[0] for c in post.calls[before:]], ['/tokenize'])  # nothing after the stop
        self.assertTrue(p.result()['cancelled'])

    def test_forbid_during_affinity_check_runs_no_m2(self):
        p, gemma = self.make()
        ok = p.relate.check
        p.relate.check_affinity = lambda: (p.forbid(), ok)[1]
        r = self.run_pick(p, request())
        self.assertEqual((r['mode'], r['cancelled']), ('EXPLORE', True))
        p.relate.relations.assert_not_called()
        gemma.decide.assert_not_called()

    def test_worker_pin_failure_raises(self):
        with self.assertRaises(RuntimeError):
            P.Picker(SimpleNamespace(), FakeRelate(fail_pin=True))

    def test_stop_ends_worker(self):
        p, _ = self.make()
        self.assertTrue(p.stop())
        self.assertFalse(p.thread.is_alive())
        self.assertFalse(p.allowed.is_set())

    def test_config(self):
        p, _ = self.make()
        c = p.config()
        self.assertEqual((c['url'], c['timeout_s'], c['options'], c['m2']['cpus'], c['m2']['threads']),
                         ('http://127.0.0.1:8080', 10, list(P.MODES.values()), [6, 7], 2))
        self.assertEqual((c['cadence']['min_gap_s'], c['cadence']['heartbeat_s']), (5, 20))


def boxes(n):
    return [{'class_name': f'c{i}', 'conf': (i % 7) / 10 + i / 1000, 'box_xyxy': (i, i, i + 10, i + 10)} for i in range(n)]


class RelateTest(unittest.TestCase):
    def relate(self, head=None):
        r = RL.Relate.__new__(RL.Relate)
        r.head = head or SimpleNamespace(infer=Mock(return_value=[]))
        return r

    def test_fewer_than_two_boxes_runs_no_m2(self):
        r = self.relate()
        for n in (0, 1):
            self.assertEqual(r.relations('img', boxes(n)), (RL.TOO_FEW, n, 0, None))
        r.head.infer.assert_not_called()
        self.assertEqual(RL.TOO_FEW, 'M2 relations:\n- none (fewer than 2 objects).')

    def test_more_than_32_keeps_the_32_most_confident_in_detector_order(self):
        r = self.relate()
        b = boxes(40)
        r.relations('img', b)
        used = r.head.infer.call_args.args[1]
        expected = sorted(sorted(range(40), key=lambda i: -b[i]['conf'])[:32])
        self.assertEqual(used, [b[i] for i in expected])
        self.assertEqual(r.relations('img', b)[1], 32)
        r.relations('img', boxes(32))
        self.assertEqual(len(r.head.infer.call_args.args[1]), 32)

    def test_context_text(self):
        rows = [{'subject': 'person', 'subject_idx': 0, 'predicate': 'sitting_on', 'object': 'dining table',
                 'object_idx': 1, 'score': .8, 'threshold': .5, 'pass': True},
                {'subject': 'cup', 'subject_idx': 2, 'predicate': 'on', 'object': 'dining table', 'object_idx': 1,
                 'score': .3, 'threshold': .5, 'pass': False}]
        r = self.relate(SimpleNamespace(infer=Mock(return_value=rows)))
        text, n_boxes, n_rel, ms = r.relations('img', boxes(3))
        self.assertEqual((text, n_boxes, n_rel), ('M2 relations:\n- person [0] sitting on dining table [1].', 3, 1))
        self.assertIsInstance(ms, float)
        self.assertEqual(RL.context(rows[1:]), 'M2 relations:\n- none above threshold.')

    def test_build_pins_exactly_the_new_ort_worker(self):
        lists = iter([{1, 2}, {1, 2, 9}])
        with patch.object(RL, 'thread_ids', side_effect=lambda: next(lists)), patch.object(RL, 'pin') as pin:
            r = RL.Relate(head=lambda: 'head')
        self.assertEqual((r.worker_tids, r.head), ([9], 'head'))
        pin.assert_called_once_with(9, {6, 7})

    def test_build_refuses_an_unexpected_thread_diff(self):
        for after in ({1, 2}, {1, 2, 8, 9}):
            lists = iter([{1, 2}, after])
            with patch.object(RL, 'thread_ids', side_effect=lambda: next(lists)), patch.object(RL, 'pin') as pin:
                with self.assertRaises(RuntimeError):
                    RL.Relate(head=lambda: 'head')
            pin.assert_not_called()

    def test_caller_pin_never_on_the_main_thread(self):
        r = self.relate()
        r.cpus = {6, 7}
        with patch.object(os, 'sched_setaffinity') as setaff:  # detect_person.pin refuses before any syscall
            with self.assertRaises(RuntimeError):
                r.pin_caller()  # this test runs on the main thread: tid == pid
        setaff.assert_not_called()

    def test_hash_gate_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            for name in RL.HASHES:
                with open(os.path.join(d, name), 'wb') as f:
                    f.write(b'not the parity-checked file')
            with patch.object(RL.np, 'load') as load, patch.object(RL.ort, 'InferenceSession') as session:
                with self.assertRaises(RuntimeError) as e:
                    RL.Head(d)
            self.assertIn('STOP M2', str(e.exception))
            load.assert_not_called()
            session.assert_not_called()


if __name__ == '__main__':
    unittest.main()
