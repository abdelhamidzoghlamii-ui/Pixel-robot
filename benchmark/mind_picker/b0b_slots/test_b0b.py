"""Offline tests for b0b.py with a fake slot server: slot routing, restore counters, criteria. No server, no phone."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'b0_cache')]
import b0b  # noqa: E402
import test_b0  # noqa: E402  B0's fake server (word tokenizer, llama-server's cache_n rule)

S = json.loads(b0b.b0.SET.read_text())


class SlotFake(test_b0.Fake):
    """B0's fake with per-slot content, id_slot routing (or ignoring it) and slot save/restore."""

    def __init__(self, n_slots=1, honour_id=True, restore_ok=True, n_ctx=2048):
        super().__init__()
        self.slots, self.honour_id, self.restore_ok, self.n_ctx, self.saved = \
            {i: [] for i in range(n_slots)}, honour_id, restore_ok, n_ctx, None

    def __call__(self, path, body):
        if path == '/slots':
            return [{'id': i, 'n_ctx': self.n_ctx} for i in self.slots]
        if path == '/slots/0?action=save':
            self.saved = list(self.slots[0])
            return {'id_slot': 0, 'n_saved': len(self.saved), 'n_written': 100}
        if path == '/slots/0?action=restore':
            if self.restore_ok:
                self.slots[0] = list(self.saved)
            return {'id_slot': 0, 'n_restored': len(self.saved), 'n_read': 100}
        if path != '/completion':
            return super().__call__(path, body)
        sid = body.get('id_slot', 0) if self.honour_id else 0
        self.slot = self.slots[sid]
        res = super().__call__(path, body)
        self.slots[sid] = self.slot
        return {**res, 'id_slot': sid}


def run(launch, fake):
    return b0b.run_sequence(fake, S, launch, 123, lambda row: None)


def crit(launch, rows, info, ref=None, code=0):
    return {c: v for c, v, _ in b0b.criteria(launch, rows, info, code, False, None, ref)}


class T(unittest.TestCase):
    def test_plans(self):
        self.assertEqual([x[1] for x in b0b.plan('L1') if x[0] in ('save', 'restore')], [None, None])
        self.assertEqual([x[0] for x in b0b.plan('L1')][4:8], ['req', 'save', 'req', 'restore'])
        self.assertEqual((b0b.n_rows('L0'), b0b.n_rows('L1'), b0b.n_rows('L2')), (13, 13, 15))
        cmd = b0b.command('L2')
        self.assertEqual((cmd[cmd.index('--parallel') + 1], cmd[cmd.index('--ctx-size') + 1], cmd.count('--parallel')),
                         ('2', '4096', 1))
        self.assertEqual(b0b.command('L0'), b0b.mem1.command(2048))
        self.assertEqual(b0b.command('L1', Path('/x/slots'))[-2:], ['--slot-save-path', '/x/slots'])

    def test_l0_matches_reference(self):
        ref, _ = run('L0', SlotFake())
        rows, info = run('L0', SlotFake())
        self.assertEqual(crit('L0', rows, info, ref), {'K0': 'PASS', 'K4': 'PASS', 'K5': 'PASS'})
        bad = copy.deepcopy(ref)
        bad[6]['cache_n'] += 1
        self.assertEqual(crit('L0', rows, info, bad)['K0'], 'FAIL')
        self.assertEqual(crit('L0', rows[:-1], info, ref)['K0'], 'FAIL')
        self.assertNotIn('id_slot', str(rows))  # L0 sends no id_slot (as B0)

    def test_l1_restore(self):
        rows, info = run('L1', SlotFake())
        r = {x['id']: x for x in rows}
        self.assertEqual(r['R7']['lcp_prev_slot'], r['R7']['cache_n'])  # against the restored R5 content
        self.assertGreater(r['R7']['cache_n'], 300)
        self.assertEqual([a['action'] for a in info['slot_actions']], ['save', 'restore'])
        self.assertEqual(crit('L1', rows, info), {'K1': 'PASS', 'K4': 'PASS', 'K5': 'PASS'})
        rows, info = run('L1', SlotFake(restore_ok=False))  # the server keeps R6: R7 rebuilds
        self.assertLess({x['id']: x for x in rows}['R7']['cache_n'], 20)
        self.assertEqual(crit('L1', rows, info)['K1'], 'FAIL')

    def test_l2_routing(self):
        rows, info = run('L2', SlotFake(n_slots=2))
        r = {x['id']: x for x in rows}
        self.assertEqual({x['id'] for x in rows if x['slot'] == 1}, {'R4', 'R6', 'R9'})
        self.assertEqual(r['R7']['lcp_prev_slot'], r['R7']['cache_n'])
        self.assertGreaterEqual(r['R9']['cache_n'], r['R6']['prompt_tokens'])
        self.assertEqual(r['R10']['cache_n'], r['R10']['prompt_tokens'] - 1)
        self.assertEqual(r['R10']['prompt_ids_sha256'], r['R8']['prompt_ids_sha256'])
        self.assertEqual(crit('L2', rows, info), {'K2': 'PASS', 'K3': 'PASS', 'K4': 'PASS', 'K5': 'PASS'})
        rows, info = run('L2', SlotFake(n_slots=2, honour_id=False))  # one shared slot: chats evict the picker
        c = crit('L2', rows, info)
        self.assertEqual((c['K3'], c['K5']), ('FAIL', 'FAIL'))
        self.assertLess({x['id']: x for x in rows}['R7']['cache_n'], 20)

    def test_k5_and_stop(self):
        rows, info = run('L0', SlotFake())
        self.assertEqual(crit('L0', rows, info, rows, code=1)['K5'], 'FAIL')
        fake = SlotFake(n_slots=2, n_ctx=1024)
        rows, info = run('L2', fake)
        self.assertEqual((rows, fake.completions), ([], 0))
        self.assertIn('per-slot n_ctx', info['stopped'])

    def test_exception_keeps_rows(self):
        fake = SlotFake()
        fake.no_probs_at = 5
        rows, info = run('L1', fake)
        self.assertEqual([x['id'] for x in rows], ['R1', 'R2', 'R3', 'R4'])
        self.assertIn('completion_probabilities', info['exception'])
        self.assertEqual(crit('L1', rows, info)['K5'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
