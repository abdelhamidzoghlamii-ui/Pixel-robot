"""Offline tests for b0.py with a fake server: LCP math, criteria, the token-uniqueness stop. No server, no phone."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b0  # noqa: E402

S = json.loads(b0.SET.read_text())


class Fake:
    """Word-level tokenizer; /completion follows llama-server's rule cache_n = min(LCP, n - 1) when cache_prompt."""

    def __init__(self, split=(), cache=True, fresh_shift=0.0, no_probs_at=None):
        self.vocab, self.slot, self.split, self.cache, self.shift, self.completions = {}, [], split, cache, \
            fresh_shift, 0
        self.no_probs_at = no_probs_at  # this completion (1-based) comes without completion_probabilities

    def ids(self, text):
        out = []
        for w in re.findall(r'<\|turn>|<turn\|>|\w+|\s|[^\w\s]', text):
            for piece in ([w[:2], w[2:]] if w in self.split else [w]):
                out.append(self.vocab.setdefault(piece, len(self.vocab) + 10))
        return out

    def __call__(self, path, body):
        if path == '/tokenize':
            return {'tokens': ([2] if body['add_special'] else []) + self.ids(body['content'])}
        if path == '/apply-template':
            return {'prompt': b0.chat.build_prompt('gemma', body['messages'])}
        if path == '/props':
            return {'default_generation_settings': {'n_ctx': 2048}}
        assert path == '/completion'
        self.completions += 1
        ids = body['prompt']
        cn = min(b0.mem1.lcp(ids, self.slot), len(ids) - 1) if body['cache_prompt'] and self.cache else 0
        gen = [self.vocab.setdefault(w, len(self.vocab) + 10) for w in (['look'] if body['n_predict'] == 1 else
                                                                          ['I', 'am', 'exploring'])]
        self.slot = ids + gen
        shift = 0 if body['cache_prompt'] else self.shift
        p = {'explore': 0.1, 'visit': 0.2, 'greet': 0.3 - shift, 'look': 0.35 + shift, 'wait': 0.05}
        if self.completions == self.no_probs_at:
            return {'content': 'look', 'tokens': gen, 'timings': {'prompt_n': len(ids) - cn, 'cache_n': cn}}
        return {'content': 'I am exploring', 'tokens': gen,
                'timings': {'prompt_n': len(ids) - cn, 'cache_n': cn, 'prompt_ms': 1.0},
                'completion_probabilities': [{'top_logprobs': [{'id': self.vocab[n], 'prob': v}
                                                               for n, v in p.items()]}]}


def run(fake):
    b0.SIZE = (1, 10 ** 6)  # the fake tokenizer is word-level
    return b0.run_sequence(fake, S, 123, lambda row: None)


class T(unittest.TestCase):
    def test_lcp(self):
        self.assertEqual(b0.mem1.lcp([1, 2, 3], [1, 2, 4, 5]), 2)
        self.assertEqual(b0.mem1.lcp([1, 2], [1, 2, 3]), 2)

    def test_full_sequence_passes(self):
        rows, info = run(Fake())
        self.assertNotIn('stopped', info)
        self.assertEqual([r['id'] for r in rows], [x[0] for x in b0.SEQUENCE] + [r + '_FRESH' for r in b0.FRESH])
        r = {x['id']: x for x in rows}
        self.assertEqual(r['R2']['cache_n'], r['R2']['prompt_tokens'] - 1)  # repeat: n - 1
        self.assertEqual(r['R2']['lcp_prev_slot'], r['R2']['prompt_tokens'])
        self.assertIn('I am exploring', b0.transcript(S, 2, info['reply']))
        self.assertEqual(r['R5_FRESH']['prompt_ids_sha256'], r['R5']['prompt_ids_sha256'])
        self.assertLess(r['R7']['cache_n'], 10)  # after the separate chat only BOS, turn opening, 'You are ' survive
        crit = dict((c, v) for c, v, _ in b0.criteria(rows, info, 0, False))
        self.assertEqual(crit, {'C1': 'PASS', 'C2': 'PASS', 'C3': 'INFO', 'C4': 'PASS', 'C5': 'PASS'})

    def test_criteria_fail(self):
        rows, info = run(Fake(cache=False, fresh_shift=0.1))
        crit = dict((c, v) for c, v, _ in b0.criteria(rows, info, 0, False))
        self.assertEqual((crit['C1'], crit['C2'], crit['C4']), ('FAIL', 'FAIL', 'FAIL'))
        rows, info = run(Fake(fresh_shift=0.03))  # same top, |dp| 0.03 > 0.02
        self.assertEqual(dict((c, v) for c, v, _ in b0.criteria(rows, info, 0, False))['C4'], 'FAIL')
        for code, killed in ((1, False), (-15, False), (0, True)):
            self.assertEqual(b0.criteria(rows, info, code, killed)[4][1], 'FAIL')
        self.assertEqual(b0.criteria(rows[:-1], info, 0, False)[4][1], 'FAIL')  # a request missing

    def test_exception_keeps_rows_and_writes_report(self):
        rows, info = run(Fake(no_probs_at=5))  # R5's reply lacks completion_probabilities
        self.assertEqual([r['id'] for r in rows], ['R1', 'R2', 'R3', 'R4'])
        self.assertIn("KeyError: 'completion_probabilities'", info['exception'])
        run_rec = {'utc': 'T', 'code_sha256': 'c' * 64, 'set_sha256': 's' * 64, 'git_head': 'h', 'command': ['x'],
                   'slot_save_path_listed': True, 'exit_code': 0, 'killed': False, 'info': info}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / 'requests.jsonl').write_text('')
            text = b0.finish(out, run_rec, rows, info)
            saved = json.loads((out / 'run.json').read_text())
            self.assertIn('completion_probabilities', saved['info']['exception'])
            self.assertEqual({c: v for c, v, _ in saved['criteria']},
                             {'C1': 'FAIL', 'C2': 'FAIL', 'C3': 'INFO', 'C4': 'FAIL', 'C5': 'FAIL'})
            self.assertEqual((out / 'report.txt').read_text(), text)
            self.assertIn('STOPPED: exception after 4 requests', text)
            names = [l.split()[1] for l in (out / 'artifacts.sha256').read_text().splitlines()]
            self.assertEqual(names, ['report.txt', 'requests.jsonl', 'run.json'])
        crit = b0.criteria(*run(Fake()), 0, False, exception='RuntimeError: x')  # a runner exception alone
        self.assertEqual(crit[4][1], 'FAIL')

    def test_name_token_stop(self):
        fake = Fake(split=('greet',))
        rows, info = run(fake)
        self.assertEqual((rows, fake.completions), ([], 0))
        self.assertIn('STOP before any inference', info['stopped'])
        self.assertIn("'greet': 2 tokens", info['stopped'])

    def test_name_duplicate_stop(self):
        fake = Fake()
        fake.ids('look')  # fix 'look' first, then give 'wait' the same id
        fake.vocab['wait'] = fake.vocab['look']
        rows, info = run(fake)
        self.assertEqual((rows, fake.completions), ([], 0))
        self.assertIn('not distinct', info['stopped'])


if __name__ == '__main__':
    unittest.main()
