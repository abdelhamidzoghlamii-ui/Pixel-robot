import ast
import copy
import json
import statistics
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h


class HarnessTests(unittest.TestCase):
    def situation(self, **changes):
        s = dict(mission=h.MISSIONS[0], scene=[], person_greeted_s_ago=None,
                 new_room_s_ago=0, last_look_s_ago=None, ultrasonic_cm=999,
                 current_mode='EXPLORE', mode_s=20, last_moves=[], rooms_seen=[],
                 m2='M2 relations:\n- none above threshold.')
        s.update(changes)
        return s

    def test_policy(self):
        for mission in h.MISSIONS:
            for bucket in ('far', 'medium', 'close', 'very close'):
                for greeted in (None, 0, 299, 300, 301):
                    s = self.situation(mission=mission, scene=[['person', 'center', bucket]],
                                       person_greeted_s_ago=greeted)
                    expected = ('EXPLORE', 'R1') if mission == h.MISSIONS[2] else (
                        ('EXPLORE', 'R4') if greeted is not None and greeted < 300 else
                        ('GREET', 'R2') if bucket in ('close', 'very close') else ('APPROACH', 'R3'))
                    before = copy.deepcopy(s)
                    self.assertEqual(h.label(s), expected)
                    self.assertEqual(s, before)
        for mission, threshold in zip(h.MISSIONS, (60, 30, 60)):
            for room in (threshold - 1, threshold, threshold + 1):
                for look in (None, 59, 60, 61):
                    expected = ('LOOK', 'R5') if room >= threshold and (look is None or look >= 60) else ('EXPLORE', 'R6')
                    self.assertEqual(h.label(self.situation(mission=mission, new_room_s_ago=room, last_look_s_ago=look)), expected)
        # A close person wins over a far person; person rules win over idle rules.
        self.assertEqual(h.label(self.situation(scene=[['person', 'left', 'far'], ['person', 'right', 'close']],
                                               new_room_s_ago=600)), ('GREET', 'R2'))

    def test_v0_and_static_token_parity(self):
        calls = []
        def post(path, body=None):
            calls.append((path, body))
            if path == '/health':
                return {'status': 'ok'}
            self.assertEqual(path, '/tokenize')
            # Per-call sentinel exposes differences in piece boundaries.
            return {'tokens': [1] if body['add_special'] else [2] + [ord(c) + 10 for c in body['content']] if len(body['content']) != 1 else [ord(body['content']) + 10]}
        ref = h.picker.Gemma(post)
        fast = h.StaticGemma(post)
        for person in (False, True):
            s = self.situation(scene=[['person', 'left', 'far']] if person else [])
            expected = h.picker.situation(s['mission'], [('person', 1., 'left', 'far')] if person else [],
                                          999, 'EXPLORE', 20, [], []) + '\n' + s['m2']
            self.assertEqual(h.state(s, 'V0'), expected)
            for v in ('V0', 'V1', 'V2', 'V3'):
                keys, phrases = h.options(s, v)
                text = h.state(s, v)
                self.assertEqual(fast.prompt_ids(text, phrases), ref.prompt_ids(text, phrases))
                calls.clear()
                fast.prompt_ids(text, phrases)
                self.assertEqual(len(calls), 1)
                calls.clear()
                ref.prompt_ids(text, phrases)
                self.assertEqual(len(calls), 4)

    def test_saved_results(self):
        # The isolated review copy contains the frozen run here; tests only read it.
        folder = h.ROOT / 'picker_prompt_results'
        if not folder.is_dir():
            self.skipTest('run artifacts supplied in isolated review copy only')
        summary = json.loads((folder / 'summary.json').read_text())
        frozen = json.loads((folder / 'freeze.json').read_text())
        self.assertEqual(h.sha(folder / 'set.jsonl'), frozen['set_sha256'])
        resumed = folder / 'scored_harness_resume.py'
        if resumed.exists():
            self.assertEqual(h.sha(folder / 'scored_harness_initial.py'), frozen['harness_sha256'])
            self.assertEqual(h.sha(h.__file__), h.sha(resumed))
            def fixed_definitions(path):
                return [ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                        if not isinstance(n, ast.FunctionDef) or n.name not in ('run', 'main')]
            self.assertEqual(fixed_definitions(folder / 'scored_harness_initial.py'), fixed_definitions(resumed))
        else:
            self.assertEqual(h.sha(h.__file__), frozen['harness_sha256'])
        prefixes = folder / 'pre_recovery_results.json'
        if prefixes.exists():
            for entry in json.loads(prefixes.read_text()):
                data = (folder / entry['path']).read_bytes()[:entry['bytes']]
                self.assertEqual(h.hashlib.sha256(data).hexdigest(), entry['sha256'])
        items = [json.loads(s) for s in (folder / 'set.jsonl').read_text().splitlines()]
        by_id = {s['id']: s for s in items}
        self.assertEqual(len(by_id), 120)
        self.assertEqual(h.Counter(s['label'] for s in items), dict.fromkeys(h.MODES, 30))
        for split, count in [('DEV', 18), ('TEST', 12)]:
            self.assertEqual(h.Counter(s['label'] for s in items if s['split'] == split), dict.fromkeys(h.MODES, count))
        for s in items:
            self.assertEqual((s['label'], s['rule']), h.label(s))
        calibration = json.loads((folder / 'calibration.json').read_text())
        rows = {}
        for path in sorted(folder.glob('V*_*.jsonl')):
            data = [json.loads(s) for s in path.read_text().splitlines()]
            variant, split = path.stem.split('_')
            expected_ids = {s['id'] for s in items if s['split'] == split}
            self.assertEqual({r['id'] for r in data}, expected_ids)
            self.assertEqual(len(data), len(expected_ids))
            self.assertEqual(h.metrics(data), summary['dev'][variant] if split == 'DEV' else summary['test'])
            for r in data:
                self.assertEqual(r['label'], by_id[r['id']]['label'])
                self.assertEqual(r['rule'], by_id[r['id']]['rule'])
                self.assertEqual(r['choice'], max(r['eligible'], key=r['probabilities'].get))
                self.assertAlmostEqual(sum(r['probabilities'].values()), 1.)
                if variant in ('V4', 'V5'):
                    baseline = calibration[variant][','.join(r['eligible'])]['probabilities']
                    adjusted = {k: r['raw_probabilities'][k] / baseline[k] for k in r['eligible']}
                    for k in adjusted:
                        self.assertAlmostEqual(r['probabilities'][k], adjusted[k] / sum(adjusted.values()))
            rows[path.stem] = data
        self.assertEqual(summary['selected'], h.select(summary['dev']))
        self.assertEqual(summary['calibration_base'], h.best(('V1', 'V2', 'V3'), summary['dev']))
        self.assertEqual(summary['v5_changed'], sum(a['choice'] != b['choice'] for a, b in zip(rows['V4_DEV'], rows['V5_DEV'])))
        test_files = sorted(p.name for p in folder.glob('*_TEST.jsonl'))
        self.assertEqual(test_files, [summary['selected'] + '_TEST.jsonl'] if summary['selected'] else [])
        events = [json.loads(s) for s in (folder / 'events.jsonl').read_text().splitlines()]
        selection_index = next(i for i, e in enumerate(events) if e['event'] == 'selection_frozen')
        self.assertTrue(all(i > selection_index for i, e in enumerate(events) if e.get('split') == 'TEST'))
        self.assertEqual(events[0]['set_sha256'], frozen['set_sha256'])
        self.assertEqual(events[-1]['event'], 'finished')
        speed = json.loads((folder / 'speed.json').read_text())
        self.assertEqual(len(speed), 72)
        self.assertEqual(summary['speed']['standard_median_ms'], statistics.median(r['standard_ms'] for r in speed))
        self.assertEqual(summary['speed']['static_median_ms'], statistics.median(r['static_ms'] for r in speed))

    def test_selection_and_calibration_state(self):
        def score(macro, tokens=100, violations=0, lowest=.6):
            return dict(macro_recall=macro, median_tokens=tokens, r1_violations=violations,
                        recall={m: lowest for m in h.MODES})
        scores = {'V0': score(.9, violations=1), 'V1': score(.8), 'V2': score(.8, 90),
                  'V3': score(.95, lowest=.59), 'V4': score(.7), 'V5': score(1.)}
        self.assertEqual(h.select(scores), 'V2')
        self.assertIsNone(h.select({'V0': score(.9, violations=1)}))
        a = self.situation()
        b = self.situation(mission=h.MISSIONS[1], scene=[['person', 'left', 'close']],
                           new_room_s_ago=600, m2='M2 relations:\n- person [0] near chair [1].')
        for variant in ('V1', 'V2', 'V3'):
            self.assertEqual(h.state(a, variant, True), h.state(b, variant, True))
            self.assertIn('Mission: N/A', h.state(a, variant, True))


if __name__ == '__main__':
    unittest.main()
