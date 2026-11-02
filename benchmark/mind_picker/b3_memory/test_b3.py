"""Offline tests for b3.py with B2's fake server: condition generation (MISLEADING pairing, date sorting, labels),
memory rendering, a full fake run, metrics, the PASS rule, the invented-memory count, custody and the context STOP.
No server, no phone."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b3  # noqa: E402
sys.path.insert(0, str(b3.HERE.parent / 'b2_rules'))
import test_b2  # noqa: E402  B2's fake server (word-level tokens with leading space, V3 two steps)

S = b3.build_set()


def item(cond, i):
    return next(it for it in S['items'] if it['condition'] == cond and it['id'] == i)


class Fake(test_b2.Fake):
    """Answers the reason with `reason(item)` and the action with `pick(item, offered)`; the item is the one whose
    rendered state is in the user turn."""

    def __init__(self, pick, reason=lambda it: 'Rule 13 fits'):
        super().__init__(lambda user, offered: pick(self.item(user), offered))
        self.reason = reason

    def item(self, user):
        return next(it for it in S['items'] if b3.state(S, it) + '\n' in user)

    def __call__(self, path, body=None):
        if path == '/completion' and body['n_predict'] == 48:
            self.bodies.append(body)
            user = self.text(body['prompt']).split('<|turn>user')[1]
            return {'content': self.reason(self.item(user)), 'tokens_predicted': 3, 'stop_type': 'word',
                    'timings': {'prompt_n': 5, 'cache_n': 0}}
        return super().__call__(path, body)


def label(it, offered):
    return it['strict']


def run(fake):
    b3.b2.PAUSE_S = 0
    events = []
    rows, info = b3.run_all(b3.b1.Server(fake), json.loads(json.dumps(S)), 7, events.append)
    return rows, info, events


def rows_with(ok):  # ok[cond] = acceptable count, same in both orders; the first ok items right, the rest 'look'
    rows = []
    for cond in b3.CONDITIONS:
        for o in b3.ORDERS:
            for it in b3.subset(S, cond)['items']:
                top = it['strict'] if it['id'] <= ok[cond] else 'look' if it['strict'] != 'look' else 'wait'
                rows.append({'set': cond, 'order': o, 'id': it['id'], 'top': top, 'reason': 'Rule 13 fits'})
    return rows


class T(unittest.TestCase):
    def test_set(self):
        self.assertEqual(len(S['items']), 120)
        self.assertEqual([it['condition'] for it in S['items'][::20]], b3.CONDITIONS)
        self.assertTrue(all(it['allowed'] == b3.NAMES for it in S['items']))
        n = {i: b[3] for i, b in b3.BASES.items()}
        for i in b3.BASES:
            self.assertEqual(item('NONE', i)['memory'], [])
            self.assertEqual(item('RELEVANT', i)['memory'], [['Oct 20', n[i]]])
            self.assertEqual([d for d, _ in item('MIXED', i)['memory']], ['Oct 12', 'Oct 15', 'Oct 18', 'Oct 20',
                                                                          'Oct 25'])
            self.assertEqual(item('MIXED', i)['memory'][3], ['Oct 20', n[i]])
            self.assertEqual(item('IRRELEVANT', i)['memory'], [list(p) for p in b3.POOL])
            self.assertEqual(item('STALE', i)['memory'], [['Oct 20', n[i]], ['Nov 1', 'The Oct 20 note is no longer '
                                                                                      'true.']])
            j = i + 10 if i <= 10 else i - 10
            self.assertEqual(item('MISLEADING', i)['memory'], [['Oct 20', n[j]]])
            self.assertNotEqual(j, i)
            b = b3.BASES[i]
            for cond in b3.CONDITIONS:
                it = item(cond, i)
                self.assertEqual(it['strict'], b[5] if cond in ('RELEVANT', 'MIXED') else b[4])
                self.assertEqual(it['also_ok'], ['greet'] if i == 20 and cond in ('RELEVANT', 'MIXED') else [])
        self.assertEqual(item('MISLEADING', 1)['memory'][0][1], n[11])
        self.assertEqual(item('MISLEADING', 20)['memory'][0][1], n[10])
        self.assertEqual(b3.by_date([('Nov 1', 'a'), ('Oct 25', 'b'), ('Oct 3', 'c')]),
                         [('Oct 3', 'c'), ('Oct 25', 'b'), ('Nov 1', 'a')])  # by date, not by text

    def test_rendering(self):
        self.assertEqual(b3.state(S, item('NONE', 8)), "Mission: Map the flat quietly and do not disturb anyone.\n"
                         "Current activity: exploring\nWhat just happened: A person is at Lisa's desk, at medium "
                         "distance, silent.\nMemory: none\nAllowed actions: explore, approach, greet, look, wait")
        st = b3.state(S, item('STALE', 14))
        self.assertIn('Current activity: waiting\n', st)
        self.assertIn("\nMemory:\n- Oct 20: When Max says he'll be right back, he is gone for hours.\n"
                      '- Nov 1: The Oct 20 note is no longer true.\nAllowed actions:', st)
        m = b3.messages(S, item('MIXED', 8), 'REV')
        self.assertEqual(m[0]['content'], b3.b2.SYSTEM['V3'])
        self.assertEqual(m[1]['content'], b3.state(S, item('MIXED', 8)) + '\n\n' + b3.b2.tail(
            'V3', ['wait', 'look', 'greet', 'approach', 'explore']))

    def test_full_run_and_report(self):
        fake = Fake(label, lambda it: 'Rule 5: the memory note decides' if it['condition'] == 'IRRELEVANT'
                    and it['id'] == 3 else 'Rule 12 fits')
        rows, info, events = run(fake)
        self.assertNotIn('stopped', info)
        self.assertEqual(len(rows), 240)
        self.assertEqual([(r['set'], r['order']) for r in rows[::20]], [(c, o) for c in b3.CONDITIONS
                                                                        for o in b3.ORDERS])
        self.assertEqual(len(fake.bodies), 480)  # two steps per row
        self.assertIn('- Oct 25: The plants', info['prompt_show'])
        m = b3.metrics(rows, S)
        self.assertEqual(m['IRRELEVANT']['FWD']['cites'], [3])
        self.assertEqual(m['NONE']['FWD']['cites'], [])
        with tempfile.TemporaryDirectory() as d:
            run_rec = {'utc': 'T', 'code_sha256': 'c' * 64, 'b2_sha256': 'b' * 64, 'git_head': 'h', 'set_path': '/s',
                       'set_sha256': 's' * 64, 'set_sha256_end': 's' * 64, 'set_items': S['items'], 'command': ['x'],
                       'exit_code': 0, 'killed': False}
            text = b3.finish(Path(d), dict(run_rec), rows, info)
            self.assertIn('ACCURACY VALID', text)
            self.assertIn('overall: PASS', text)
            self.assertIn("IRRELEVANT FWD: 1 [3]\n    3: 'Rule 5: the memory note decides'", text)
            self.assertIn('rendered prompt, MIXED FWD, base 8', text)
            text = b3.finish(Path(d), {**run_rec, 'set_sha256_end': 'z'}, rows[:-1], info)
            self.assertIn('ACCURACY INVALID: 239/240 rows', text)
            self.assertIn('NO RESULT', text)

    def test_metrics_and_pass(self):
        ok = {'NONE': 18, 'RELEVANT': 16, 'MIXED': 15, 'IRRELEVANT': 16, 'STALE': 17, 'MISLEADING': 20}
        m = b3.metrics(rows_with(ok), S)
        self.assertEqual((m['NONE']['FWD']['ok'], m['IRRELEVANT']['REV']['change']), (18, -2))
        self.assertEqual(m['MIXED']['flips'], [])
        c = b3.verdict(m)
        self.assertEqual(len(c), 10)
        self.assertTrue(all(c.values()))  # all on their bars
        for cond, v in (('RELEVANT', 15), ('MIXED', 14), ('IRRELEVANT', 15), ('STALE', 15), ('MISLEADING', 15)):
            c = b3.verdict(b3.metrics(rows_with({**ok, cond: v}), S))
            self.assertEqual([k for k, x in c.items() if not x], [k for k in c if k.startswith(cond + ' ')])
        rows = rows_with(ok)
        rows[0]['top'] = 'greet'  # NONE FWD item 1: wrong in FWD only
        m = b3.metrics(rows, S)
        self.assertEqual((m['NONE']['flips'], m['NONE']['FWD']['ok'], m['NONE']['REV']['ok']), ([1], 17, 18))
        self.assertEqual(m['IRRELEVANT']['FWD']['change'], -1)
        self.assertEqual(b3.metrics([r for r in rows if r['id'] != 2], S)['NONE']['FWD']['ok'], 16)  # missing = wrong
        ok20 = {**ok, 'RELEVANT': 20}
        rows = rows_with(ok20)
        next(r for r in rows if (r['set'], r['id']) == ('RELEVANT', 20))['top'] = 'greet'
        self.assertEqual(b3.metrics(rows, S)['RELEVANT']['FWD']['strict'], 19)
        self.assertEqual(b3.metrics(rows, S)['RELEVANT']['FWD']['ok'], 20)  # also ok

    def test_cites_memory(self):
        hit = ['Rule 5 fits.', 'rule 5 applies', 'The memory says to greet.', 'A note says so.', 'I remember that.']
        miss = ['Rule 12 fits.', 'Rule 15 fits.', 'Nothing new happened; rule 13.', 'I notice a person.']
        self.assertTrue(all(b3.CITES_MEMORY.search(r) for r in hit))
        self.assertFalse(any(b3.CITES_MEMORY.search(r) for r in miss))

    def test_custody(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'b3_set.json'
            data = (json.dumps(S, indent=1) + '\n').encode()
            p.write_bytes(data)
            self.assertEqual(b3.load_set(p, b3.sha(data)), (S, b3.sha(data)))
            with self.assertRaisesRegex(SystemExit, 'pinned'):
                b3.load_set(p, '0' * 64)
            bad = json.loads(data)
            bad['items'][0]['strict'] = 'wait'
            p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(SystemExit, 'differs'):
                b3.load_set(p, b3.sha(p.read_bytes()))
            link = Path(d) / 'link.json'
            link.symlink_to(b3.HERE / 'b3.py')
            with self.assertRaisesRegex(SystemExit, 'inside the repository'):
                b3.load_set(link, None)

    def test_context_stop(self):
        fake = Fake(label)
        b3.CTX = 100
        try:
            rows, info, events = run(fake)
        finally:
            b3.CTX = 2048
        self.assertIn('STOP before any inference', info['stopped'])
        self.assertIn('+ 64 > 100', info['stopped'])
        self.assertEqual((rows, fake.bodies, events), ([], [], []))


if __name__ == '__main__':
    unittest.main()
