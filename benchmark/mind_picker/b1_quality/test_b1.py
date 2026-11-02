"""Offline tests for b1.py with a fake server: rendering of both arms, eligibility renormalisation, metrics, custody.
No server, no phone. Uses a synthetic set (not the frozen one, which stays outside the repo)."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1  # noqa: E402
import test_b0  # noqa: E402  B0's word-level fake server (b0_cache is on the path via b1)

B = json.loads(b1.b0.SET.read_text())
NAMES = B['names']
S = {'missions': {'M1': 'Explore and map the flat.', 'M2': 'Find people and keep them company.'}, 'items': [
    {'id': 1, 'mission': 'M1', 'current': 'explore', 'event': 'Someone said: "Stay right there."', 'memory': None,
     'allowed': NAMES, 'strict': 'wait', 'also_ok': []},
    {'id': 2, 'mission': 'M2', 'current': 'wait', 'event': 'A person is close.', 'memory': None,
     'allowed': ['explore', 'visit', 'look', 'wait'], 'strict': 'wait', 'also_ok': ['visit']},
    {'id': 11, 'mission': 'M2', 'current': 'explore', 'event': 'A person is lying on the couch, close.',
     'memory': 'Leo sleeps on the couch; do not wake him.', 'allowed': NAMES, 'strict': 'explore',
     'also_ok': ['look']}]}
LABEL = {it['event']: it['strict'] for it in S['items']}


class Fake(test_b0.Fake):
    """Adds /health, /detokenize and a /completion whose favourite action is chosen by `pick(text, offered)`
    (letters for layout A, names for B); a token not offered gets the highest raw probability."""

    def __init__(self, pick):
        super().__init__()
        self.pick, self.bodies = pick, []

    def text(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return ''.join(inv.get(i, '') for i in ids)

    def __call__(self, path, body=None):
        if path == '/health':
            return {'status': 'ok'}
        if path == '/detokenize':
            return {'content': self.text(body['tokens'])}
        if path != '/completion':
            return super().__call__(path, body)
        self.bodies.append(body)
        ids, text = body['prompt'], self.text(body['prompt'])
        cn = min(b1.mem1.lcp(ids, self.slot), len(ids) - 1) if body['cache_prompt'] else 0
        if text.endswith('Answer: ('):  # A: letter -> name from the options
            opts = re.findall(r'\n\((\w)\) (\w+):', text)
            fav = dict((n, L) for L, n in opts)[self.pick(text, [n for _, n in opts])]
            cand, extra = 'ABCDE', 'ABCDE'[len(opts):]
        else:
            names = re.search(r'one word: (.*)\.<turn\|>', text).group(1).replace(' or ', ', ').split(', ')
            fav, cand, extra = self.pick(text, names), NAMES, [n for n in NAMES if n not in names]
        p = {k: (0.5 if k in extra else 0.4 if k == fav else 0.05) for k in cand}  # extra: not offered
        tid = self.vocab.setdefault(fav, len(self.vocab) + 10)
        self.slot = ids + [tid]
        return {'content': fav, 'tokens': [tid], 'timings': {'prompt_n': len(ids) - cn, 'cache_n': cn},
                'completion_probabilities': [{'top_logprobs': [{'id': self.vocab.setdefault(k, len(self.vocab) + 10),
                                                                'prob': v} for k, v in p.items()]}]}


def by_label(text, offered):  # the item's strict label, found by its event text
    return next(v for k, v in LABEL.items() if k in text)


def first(text, offered):  # order-sensitive: always the first offered
    return offered[0]


def run(pick):
    b1.PAUSE_S = 0
    fake = Fake(pick)
    rows, info = b1.run_blocks(b1.Server(fake), S, B, 123, lambda row: None)
    return fake, rows, info


class T(unittest.TestCase):
    def test_tail_equals_b0(self):
        self.assertEqual(b1.tail(NAMES), B['tail'])

    def test_rendering(self):
        fake, rows, info = run(by_label)
        a, b = info['prompt_A_item11'], info['prompt_B_item11']
        st = ('Mission: Find people and keep them company.\nCurrent activity: exploring\n'
              'What just happened: A person is lying on the couch, close.\n'
              'Memory: Leo sleeps on the couch; do not wake him.\nAllowed actions: explore, visit, greet, look, wait')
        self.assertEqual(a, '<state>\n' + st + '\n</state>\n\nQuestion (choice): Choose the single best next action '
                         'for the robot.\nOptions:\n' + ''.join(f'({L}) {c}\n' for L, c in zip('ABCDE', B['catalogue']))
                         + 'Answer: (')
        self.assertEqual(b, b1.b0.chat.build_prompt('gemma', [
            {'role': 'system', 'content': B['sys'] + '\n\nActions:\n' + '\n'.join(B['catalogue'])},
            {'role': 'user', 'content': st + '\n\n' + B['tail']}]))
        rev = {(r['arm'], r['order'], r['id']): r for r in rows}
        a_rev = fake.text(fake.bodies[rows.index(rev['A', 'REV', 11])]['prompt'])
        self.assertIn('(A) wait: stay still and wait\n(B) look', a_rev)
        self.assertIn('Allowed actions: explore, visit, greet, look, wait', a_rev)  # the state line keeps its order
        self.assertEqual(rev['B', 'REV', 11]['offered'], NAMES[::-1])
        self.assertNotIn('Current activity: explore', a + b)  # neutral, never the option phrase
        self.assertTrue(all(fake.bodies[i]['cache_prompt'] == (r['arm'] == 'B') for i, r in enumerate(rows)))

    def test_eligibility_renormalised(self):
        _, rows, info = run(by_label)
        self.assertNotIn('stopped', info)
        self.assertEqual(len(rows), 12)
        for r in rows:
            self.assertAlmostEqual(sum(r['probs'].values()), 1.0)
            self.assertEqual(list(r['probs']), r['offered'])
        r2 = [r for r in rows if r['id'] == 2]
        self.assertTrue(all('greet' not in r['probs'] and r['top'] == 'wait' for r in r2))
        self.assertTrue(all(max(v for v in r['probs_raw'].values()) == 0.4 for r in r2))  # 0.5 went to greet / E

    def test_metrics_pass_and_fail(self):
        _, rows, _ = run(by_label)
        m = b1.metrics(rows, S)
        self.assertEqual({k: (v['strict'], v['acceptable']) for k, v in m['blocks'].items()},
                         {k: (1.0, 1.0) for k in ('A-FWD', 'A-REV', 'B-FWD', 'B-REV')})
        self.assertEqual({a: (v['flips'], v['verdict']) for a, v in m['arms'].items()},
                         {'A': ([], 'PASS'), 'B': ([], 'PASS')})
        _, rows, _ = run(first)  # FWD -> explore, REV -> wait
        m = b1.metrics(rows, S)
        f, r = m['blocks']['A-FWD'], m['blocks']['A-REV']
        self.assertEqual((f['strict'], f['acceptable'], r['strict']), (1 / 3, 1 / 3, 2 / 3))
        self.assertEqual(f['recall'], {'explore': [1, 1], 'wait': [0, 2]})
        self.assertEqual(f['confusion'], {'wait': {'explore': 2}, 'explore': {'explore': 1}})
        self.assertEqual((f['copy'], f['copied_ids']), ([1, 1], [1]))  # only item 1 has strict != current; it chose explore
        self.assertEqual(m['arms']['A'], {'flips': [1, 2, 11], 'verdict': 'FAIL'})
        self.assertEqual(f['near_ties'], [])
        rows[0]['margin'] = 0.01
        self.assertEqual(b1.metrics(rows, S)['blocks']['A-FWD']['near_ties'], [1])
        self.assertEqual(b1.metrics(rows[:-1], S)['blocks']['B-REV']['rows'], 2)  # a missing row counts as wrong

    def test_letter_stop(self):
        fake = Fake(by_label)
        fake.split = ('E',)
        b1.PAUSE_S = 0
        rows, info = b1.run_blocks(b1.Server(fake), S, B, 1, lambda r: None)
        self.assertEqual((rows, fake.bodies), ([], []))
        self.assertIn('STOP before any inference', info['stopped'])

    def test_finish_and_validity(self):
        _, rows, info = run(by_label)
        run_rec = {'utc': 'T', 'code_sha256': 'c' * 64, 'set_sha256': 's' * 64, 'set_sha256_end': 's' * 64,
                   'set_rows': 3, 'set_path': '/x', 'git_head': 'h', 'command': ['x'], 'items': S['items'],
                   'exit_code': 0, 'killed': False}
        with tempfile.TemporaryDirectory() as d:
            text = b1.finish(Path(d), dict(run_rec), rows, info)
            self.assertIn('ACCURACY VALID', text)
            self.assertIn('arm A: flips 0 []  -> PASS', text)
            text = b1.finish(Path(d), {**run_rec, 'set_sha256_end': 'z'}, rows[:-1], info)
            self.assertIn('ACCURACY INVALID: 11 rows, 0 errors; set SHA-256', text)

    def test_custody(self):
        with self.assertRaisesRegex(SystemExit, 'inside the repository'):
            b1.load_set(b1.HERE / 'b1_set.json')  # refused before reading: the file need not exist
        with self.assertRaisesRegex(SystemExit, 'inside the repository'):
            b1.load_set(b1.REPO)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'set.json'
            p.write_text(json.dumps(S))
            with self.assertRaisesRegex(SystemExit, 'pinned'):
                b1.load_set(p)
            self.assertEqual(b1.load_set(p, want=None)[0], S)


if __name__ == '__main__':
    unittest.main()
