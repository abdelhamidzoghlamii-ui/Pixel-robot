"""Offline tests for b2.py with a fake server: rendering of the three variants, two-step V3 scoring, the selection rule
(tie and FLIPS), the TEST lock, symlink custody. No server, no phone. Synthetic sets (the real TEST stays outside the
repo)."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b2  # noqa: E402
import test_b0  # noqa: E402  B0's word-level fake server (b0_cache is on the path via b1)

N = b2.NAMES
MISSIONS = {'M1': 'Explore and map the flat.', 'M2': 'Find people and keep them company.'}
DEV = {'missions': MISSIONS, 'items': [
    {'id': 1, 'mission': 'M1', 'current': 'explore', 'event': 'Someone said: "Stay right there."', 'memory': None,
     'allowed': ['explore', 'visit', 'greet', 'look', 'wait'], 'strict': 'wait', 'also_ok': []},
    {'id': 2, 'mission': 'M2', 'current': 'visit', 'event': 'A person is close.', 'memory': None,
     'allowed': ['explore', 'visit', 'look', 'wait'], 'strict': 'visit', 'also_ok': ['wait']},
    {'id': 3, 'mission': 'M2', 'current': 'explore', 'event': 'A person is lying on the couch, close.',
     'memory': 'Leo sleeps on the couch; do not wake him.', 'allowed': ['explore', 'visit', 'greet', 'look', 'wait'],
     'strict': 'explore', 'also_ok': ['look']}]}
TEST = {'missions': MISSIONS, 'items': [
    {'id': 1, 'mission': 'M1', 'current': 'look', 'event': 'You finished looking around the hallway.', 'memory': None,
     'allowed': N, 'strict': 'explore', 'also_ok': []},
    {'id': 2, 'mission': 'M1', 'current': 'explore', 'event': 'Someone said: "Robot, over here!"', 'memory': None,
     'allowed': N, 'strict': 'approach', 'also_ok': []}]}


class Fake(test_b0.Fake):
    """Tokens keep their leading space (' explore'); /completion answers a reason (n_predict 48) or the favourite
    action chosen by `pick(user_text, offered)`; names not offered get the highest raw probability."""

    def __init__(self, pick, alias=None):
        super().__init__()
        self.pick, self.alias, self.bodies = pick, alias or {}, []

    def ids(self, text):
        return [self.vocab.setdefault(self.alias.get(w, w), len(self.vocab) + 10)
                for w in re.findall(r'<\|turn>|<turn\|>| ?\w+|\s|[^\w\s]', text)]

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
        text = self.text(body['prompt'])
        if body['n_predict'] == 48:
            return {'content': 'Rule 1 fits', 'tokens_predicted': 3, 'stop_type': 'word',
                    'timings': {'prompt_n': 5, 'cache_n': 0}}
        offered = re.search(r'one word: (.*)\.<turn\|>', text).group(1).replace(' or ', ', ').split(', ')
        fav = self.pick(text.split('<|turn>user')[1], offered)
        sp = ' ' if text.endswith('Action:') else ''
        p = {self.vocab.setdefault(sp + k, len(self.vocab) + 10): 0.5 if k not in offered else 0.4 if k == fav else 0.05
             for k in N}
        return {'content': fav, 'timings': {'prompt_n': 3, 'cache_n': len(body['prompt']) - 3},
                'completion_probabilities': [{'top_logprobs': [{'id': i, 'prob': v} for i, v in p.items()]}]}


def by_label(user, offered):  # the strict label of the item whose event is in the user turn
    items = DEV['items'] + TEST['items']
    lab = next(it['strict'] for it in items if it['event'] in user)
    return {'visit': 'approach'}.get(lab, lab)


def first(user, offered):
    return offered[0]


def sets():
    d = json.loads(json.dumps(DEV))
    for it in d['items']:  # load_set's renaming, without a file
        for k in ('current', 'strict'):
            it[k] = {'visit': 'approach'}.get(it[k], it[k])
        it['allowed'] = [{'visit': 'approach'}.get(n, n) for n in it['allowed']]
        it['also_ok'] = [{'visit': 'approach'}.get(n, n) for n in it['also_ok']]
    return d


def run(pick, alias=None):
    b2.PAUSE_S = 0
    fake, events = Fake(pick, alias), []
    d = tempfile.mkdtemp()
    tp = Path(d) / 'test.json'
    tp.write_text(json.dumps(TEST))
    b2.TEST_SHA256 = b2.sha(tp.read_bytes())
    rows, info = b2.run_all(b2.b1.Server(fake), sets(), tp, Path(d), 7, events.append)
    return fake, rows, info, events, Path(d)


class T(unittest.TestCase):
    def test_rendering(self):
        s = sets()
        self.assertEqual(s['items'][1]['allowed'], ['explore', 'approach', 'look', 'wait'])
        st = b2.state(s, s['items'][1])
        self.assertEqual(st, 'Mission: Find people and keep them company.\nCurrent activity: approaching\n'
                             'What just happened: A person is close.\nMemory: none\n'
                             'Allowed actions: explore, approach, look, wait')
        m1, m2, m3 = (b2.messages(s, s['items'][1], v, 'REV') for v in b2.VARIANTS)
        self.assertEqual(m1[0]['content'], b2.SYS_RULES)
        self.assertTrue(b2.SYS_RULES.endswith('13. Otherwise: keep the current activity.'))
        self.assertEqual(m2[0]['content'], b2.SYS_RULES + '\n\nExamples:\n' + '\n'.join(b2.EXAMPLES))
        self.assertEqual((len(b2.EXAMPLES), m3[0]), (13, m2[0]))
        self.assertEqual(m1[1]['content'], st + '\n\nWhich action next? Answer with one word: wait, look, approach '
                                                'or explore.')
        self.assertEqual(m2[1], m1[1])
        self.assertEqual(m3[1]['content'], st + "\n\nWhich action next? First write one short sentence naming the rule"
                         " that fits. Then write a new line starting with 'Action:' and one word: wait, look, "
                         "approach or explore.")
        fwd = b2.messages(s, s['items'][0], 'V1', 'FWD')[1]['content']
        self.assertTrue(fwd.endswith('one word: explore, approach, greet, look or wait.'))

    def test_v3_two_steps(self):
        fake, rows, info, _, _ = run(by_label)
        self.assertNotIn('stopped', info)
        v3 = [r for r in rows if r['variant'] == 'V3' and r['set'] == 'DEV']
        self.assertEqual(len(v3), 6)
        k = rows.index(v3[0])
        r1, r2 = fake.bodies[12], fake.bodies[13]  # V1 and V2 DEV blocks: 12 single requests, then V3 pairs
        self.assertEqual((r1['n_predict'], r1['stop'], r1['cache_prompt']), (48, ['\n'], True))
        self.assertEqual(r2['n_predict'], 1)
        self.assertEqual(fake.text(r2['prompt']), fake.text(r1['prompt']) + 'Rule 1 fits\nAction:')
        self.assertEqual(rows[k]['reason'], 'Rule 1 fits')
        self.assertEqual(info['name_ids']['V3'], {n: fake.vocab[' ' + n] for n in N})
        self.assertEqual(info['name_ids']['V1/V2'], {n: fake.vocab[n] for n in N})
        for r in rows:
            it = next(it for it in (DEV if r['set'] == 'DEV' else TEST)['items'] if it['id'] == r['id'])
            self.assertAlmostEqual(sum(r['probs'].values()), 1.0)
            self.assertEqual((list(r['probs']), r['top']), (r['offered'], by_label(it['event'], None)))

    def test_full_run_and_report(self):
        fake, rows, info, events, d = run(by_label)
        self.assertEqual(len(rows), 6 * 3 + 2 * 2)
        sel = json.loads((d / 'selection.json').read_text())
        self.assertEqual((sel['variant'], sel['flag']), ('V1', None))  # all 1.0: tie -> V1
        k = next(i for i, e in enumerate(events) if e.get('event') == 'selection')
        self.assertTrue(all(e['set'] == 'DEV' for e in events[:k]) and all(e['set'] == 'TEST' for e in events[k + 1:]))
        self.assertEqual({(e['variant'], e['order']) for e in events[k + 1:]}, {('V1', 'FWD'), ('V1', 'REV')})
        run_rec = {'utc': 'T', 'code_sha256': 'c' * 64, 'git_head': 'h', 'dev_path': '/d', 'dev_sha256': 'd' * 64,
                   'dev_sha256_end': 'd' * 64, 'dev_rows': 3, 'dev_items': sets()['items'], 'test_path': '/t',
                   'test_sha256': 't' * 64, 'test_sha256_end': 't' * 64, 'test_rows': 2, 'command': ['x'],
                   'exit_code': 0, 'killed': False}
        text = b2.finish(d, dict(run_rec), rows, info)
        self.assertIn('ACCURACY VALID', text)
        self.assertIn('-> PASS', text)
        self.assertIn("'Rule 1 fits'", text)
        text = b2.finish(d, {**run_rec, 'test_sha256_end': 'z'}, rows[:-1], info)
        self.assertIn('ACCURACY INVALID: 21/22 rows', text)
        self.assertIn('NO RESULT', text)

    def test_selection(self):
        def dev(acc, flips):  # acc[v] = (FWD acceptable, REV acceptable)
            return {v: ({'strict': 0, 'acceptable': acc[v][0]}, {'strict': 0, 'acceptable': acc[v][1]},
                        {'flips': list(range(flips[v]))}) for v in b2.VARIANTS}
        sel = b2.select(dev({'V1': (.5, .6), 'V2': (.7, .7), 'V3': (.9, .9)}, {'V1': 0, 'V2': 4, 'V3': 5}))
        self.assertEqual((sel['variant'], sel['flag']), ('V2', None))  # V3 excluded by flips
        sel = b2.select(dev({'V1': (.5, .6), 'V2': (.6, .5), 'V3': (.55, .55)}, {'V1': 1, 'V2': 0, 'V3': 0}))
        self.assertEqual(sel['variant'], 'V1')  # three-way tie on the mean -> V1
        sel = b2.select(dev({'V1': (.5, .6), 'V2': (.7, .7), 'V3': (.7, .7)}, {'V1': 9, 'V2': 0, 'V3': 0}))
        self.assertEqual(sel['variant'], 'V2')  # tie V2/V3 -> V2
        sel = b2.select(dev({'V1': (.5, .6), 'V2': (.7, .7), 'V3': (.8, .7)}, {'V1': 5, 'V2': 6, 'V3': 9}))
        self.assertEqual((sel['variant'], sel['flag']), ('V3', 'FLIPS'))
        self.assertEqual(sel['dev']['V3']['mean_acceptable'], .75)

    def test_flips_in_run(self):
        _, rows, info, _, _ = run(first)  # FWD -> first catalogue name, REV -> wait: every variant flips 3/3
        self.assertEqual(info['selection']['flag'], None)  # 3 <= 4
        self.assertEqual(info['selection']['dev']['V1']['flips'], [1, 2, 3])

    def test_test_lock(self):
        with tempfile.TemporaryDirectory() as d:
            d, tp = Path(d), Path(d) / 'test.json'
            tp.write_text(json.dumps(TEST))
            b2.TEST_SHA256 = b2.sha(tp.read_bytes())
            with self.assertRaisesRegex(SystemExit, 'locked'):
                b2.load_test(d, None, tp)
            (d / 'selection.json').write_text('{}')
            with self.assertRaisesRegex(SystemExit, 'locked'):
                b2.load_test(d, 'f' * 64, tp)  # logged hash differs
            self.assertEqual(b2.load_test(d, b2.sha(b'{}'), tp)[0], TEST)
            b2.TEST_SHA256 = '0' * 64
            with self.assertRaisesRegex(SystemExit, 'pinned'):
                b2.load_test(d, b2.sha(b'{}'), tp)

    def test_dev_stop_never_opens_test(self):
        fake, rows, info, events, d = run(by_label, alias={'look': 'wait'})  # look and wait share a first token
        self.assertIn('STOP before any inference', info['stopped'])
        self.assertEqual((rows, fake.bodies, events, 'test' in info), ([], [], [], False))
        self.assertFalse((d / 'selection.json').exists())

    def test_custody_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            link, dlink = Path(d) / 'set.json', Path(d) / 'repo'
            os.symlink(b2.DEV, link)  # outside symlink to a file inside the repo
            os.symlink(b2.REPO / 'benchmark', dlink)  # outside symlink to a directory inside the repo
            self.assertTrue(b2.inside_repo(link))
            self.assertTrue(b2.inside_repo(dlink / 'nonexistent.json'))
            with self.assertRaisesRegex(SystemExit, 'inside the repository'):
                b2.load_set(link, None)
            real = Path(d) / 'real.json'
            real.write_text('{"items": []}')
            self.assertFalse(b2.inside_repo(real))
        self.assertTrue(b2.inside_repo(b2.HERE / 'x.json'))
        bind = Path('/termux-home/robot')  # the same repo through the proot bind mount: inode check
        if bind.exists():
            self.assertTrue(b2.inside_repo(bind / 'benchmark' / 'x.json'))


if __name__ == '__main__':
    unittest.main()
