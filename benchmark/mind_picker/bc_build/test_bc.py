"""Offline tests for bc.py with fake servers (B0b's slot fake for K1/K2, B2's fake for K3, a chat fake for K4): the
command override, the record diffs, the criteria and the full write path. No server, no phone."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bc  # noqa: E402  puts b0b_slots, b2_rules (and b0_cache) on the path
import test_b0b  # noqa: E402
import test_b2  # noqa: E402

S = json.loads(bc.b0.SET.read_text())
REFS = bc.references()
TEST = REFS['test']
bc.b0.SIZE = (1, 10 ** 6)  # the fake tokenizers are word-level
bc.b2.PAUSE_S = 0


class ChatFake:
    def __init__(self, content='A table with a laptop.'):
        self.content, self.bodies = content, []

    def __call__(self, path, body):
        assert path == '/v1/chat/completions'
        self.bodies.append(body)
        return {'choices': [{'message': {'content': self.content}, 'finish_reason': 'stop'}], 'usage': {}}


def by_label(user, offered):
    return next(it['strict'] for it in TEST['items'] if it['event'] in user)


def phases(chat=None, pid=123):
    rows = []
    slot = test_b0b.SlotFake()  # one server for K1 and K2, as in the run
    ph = {'K1': bc.k1(slot, S, pid, rows.append), 'K2': bc.k2(slot, S, pid, rows.append),
          'K3': bc.k3(test_b2.Fake(by_label), TEST, pid, rows.append), 'K4': bc.k4(chat or ChatFake(), pid, rows.append,
                                                                                 'AAAA')}
    return ph, rows


def refs_for(ph):
    """References that the fake run reproduces exactly: B0 = a fresh K1 run, B2 = the K3 rows."""
    return {**REFS, 'b0': bc.k1(test_b0b.SlotFake(), S, 123, lambda r: None)[0],
            'b2': [{**x, 'set': 'TEST'} for x in ph['K3'][0]]}


def crit(ph, refs, code=0, killed=False):
    return {c: v for c, v, _ in bc.criteria(ph, {'exit_code': code, 'killed': killed}, TEST, refs)}


class T(unittest.TestCase):
    def test_command(self):
        cmd = bc.command(Path('/r/slots'))
        base = bc.mem1.command(2048)
        self.assertNotIn(bc.mem1.SERVER, cmd)
        self.assertEqual(cmd[:len(base)], [bc.SERVER if x == bc.mem1.SERVER else x for x in base])
        self.assertEqual(cmd[len(base):], ['--slot-save-path', '/r/slots', '--mmproj', bc.vis3d.MMPROJ,
                                           '--image-min-tokens', '70', '--image-max-tokens', '70'])
        self.assertEqual(bc.SERVER, f'{bc.HOME}/llama.cpp-vis/build/bin/llama-server')
        for flag in ('--model-draft', '-md', '--spec-type', '--reasoning'):
            self.assertNotIn(flag, cmd)
        self.assertEqual(TEST['items'][0]['id'], 1)
        self.assertEqual(len(TEST['items']), 40)

    def test_all_pass(self):
        ph, rows = phases()
        self.assertEqual(len(rows), 13 + 13 + 80 + 1)
        self.assertEqual({r['phase'] for r in rows}, {'K1', 'K2', 'K3', 'K4'})
        c = crit(ph, refs_for(ph))
        self.assertEqual({k: v for k, v in c.items() if v != 'INFO'},
                         {'K1 C1': 'PASS', 'K1 C2': 'PASS', 'K1 C4': 'PASS', 'K1 C5': 'PASS', 'K2': 'PASS',
                          'K3': 'PASS', 'K4': 'PASS', 'K5': 'PASS'})
        text = dict((a, t) for a, _, t in bc.criteria(ph, {'exit_code': 0}, TEST, refs_for(ph)))
        self.assertIn('0 of 13 rows differ', text['K1 diff'])
        self.assertIn('0 of 80 items differ', text['K3 agreement'])

    def test_diff_and_agreement(self):
        ph, _ = phases()
        ref = refs_for(ph)
        b0r = copy.deepcopy(ref['b0'])
        b0r[6]['cache_n'] += 5
        d = bc.diff_rows(ph['K1'][0], b0r)
        self.assertEqual(d, [('R7', {'cache_n': (b0r[6]['cache_n'], ph['K1'][0][6]['cache_n'])})])
        self.assertEqual(bc.diff_rows(ph['K1'][0][:-1], ref['b0'])[0][0], 'R8_FRESH')  # a missing row differs
        b2r = copy.deepcopy(ref['b2'])
        b2r[3]['top'] = 'wait' if b2r[3]['top'] != 'wait' else 'look'
        a = bc.agreement(ph['K3'][0], b2r)
        self.assertEqual(a, [(b2r[3]['order'], b2r[3]['id'], b2r[3]['top'], ph['K3'][0][3]['top'])])

    def test_failures(self):
        ph, _ = phases()
        ref = refs_for(ph)
        c = crit(ph, ref, code=-6)  # #145's SIGABRT: a finding, K5 unaffected, B0's C5 fails as in B0
        self.assertEqual((c['exit'], c['K5'], c['K1 C5']), ('FINDING', 'PASS', 'FAIL'))
        ph4, _ = phases(ChatFake(content='  '))
        self.assertEqual(crit(ph4, refs_for(ph4))['K4'], 'FAIL')
        bad = copy.deepcopy(ph)
        bad['K3'][0][5] = {**bad['K3'][0][5], 'error': 'URLError: x'}
        del bad['K3'][0][5]['top']
        c = crit(bad, ref)
        self.assertEqual((c['K3'], c['K5']), ('FAIL', 'FAIL'))
        bad = copy.deepcopy(ph)
        bad['K4'][0][0]['pid'] = 999  # a restarted server
        self.assertEqual(crit(bad, ref)['K5'], 'FAIL')
        wrong = test_b2.Fake(lambda user, offered: offered[0])  # always the first offered name
        bad = {**ph, 'K3': bc.k3(wrong, TEST, 123, lambda r: None)}
        self.assertEqual(crit(bad, ref)['K3'], 'FAIL')

    def test_k4_payload(self):
        chat = ChatFake()
        rows, body = bc.k4(chat, 123, lambda r: None, 'AAAA')
        b = chat.bodies[0]
        self.assertEqual({k: b[k] for k in ('temperature', 'top_k', 'seed', 'max_tokens', 'cache_prompt')},
                         {'temperature': 0, 'top_k': 1, 'seed': 0, 'max_tokens': 96, 'cache_prompt': False})
        self.assertEqual(b['chat_template_kwargs'], {'enable_thinking': False})
        content = b['messages'][0]['content']
        self.assertEqual((content[0]['image_url']['url'], content[1]['text']),
                         ('data:image/jpeg;base64,AAAA', bc.vis3d.PROMPT))
        self.assertEqual(rows[0]['content'], 'A table with a laptop.')
        self.assertNotIn('messages', body)

    def test_finish_writes_everything(self):
        ph, _ = phases()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / 'slots').mkdir()
            (out / 'slots' / 'picker.bin').write_bytes(b'x')
            run = {'utc': 'T', 'code_sha256': 'c' * 64, 'git_head': 'h', 'command': ['x'], 'pins': {},
                   'provenance': 'p\n', 'exit_code': 0, 'killed': False, 'ram': {}}
            text = bc.finish(out, run, ph, refs_for(ph))
            self.assertIn('counters and accuracy VALID', text)
            self.assertIn("reply (verbatim): 'A table with a laptop.'", text)
            names = [l.split('  ', 1)[1] for l in (out / 'artifacts.sha256').read_text().splitlines()]
            self.assertEqual(names, ['report.txt', 'run.json', 'slots/picker.bin'])


if __name__ == '__main__':
    unittest.main()
