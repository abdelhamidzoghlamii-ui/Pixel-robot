"""PICKPROMPT-2 offline tests: fake model/HTTP/server; optional saved-result audit."""
from contextlib import contextmanager, redirect_stdout
import ast
import copy
import io
import itertools
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness2 as h2

h = h2.h


def item(mode='EXPLORE', **changes):
    s = dict(id='S0', mission=h.MISSIONS[0], scene=[], person_greeted_s_ago=None,
             new_room_s_ago=0, last_look_s_ago=None, ultrasonic_cm=999,
             current_mode='EXPLORE', mode_s=20, last_moves=[], rooms_seen=[],
             m2='M2 relations:\n- none above threshold.', split='DEV')
    if mode in ('APPROACH', 'GREET'):
        s['scene'] = [['person', 'center', 'close' if mode == 'GREET' else 'far']]
    elif mode == 'LOOK':
        s['new_room_s_ago'] = 60
    s.update(changes)
    s['label'], s['rule'] = h.label(s)
    return s


class FactsAndSchemaTest(unittest.TestCase):
    def test_fact_truths_and_boundaries_are_from_items(self):
        for mission, threshold in zip(h.MISSIONS, (60, 30, 60)):
            for greeted in (None, 0, 299, 300):
                for room in (threshold - 1, threshold):
                    for look in (None, 59, 60):
                        s = item(mission=mission, person_greeted_s_ago=greeted,
                                 new_room_s_ago=room, last_look_s_ago=look)
                        before = copy.deepcopy(s)
                        facts = h2.truth(s)
                        self.assertEqual(facts['greeted_in_last_5_min'], greeted is not None and greeted < 300)
                        self.assertEqual(facts['mission_allows_people'], mission != h.MISSIONS[2])
                        self.assertEqual(facts['nothing_new_for_a_while'], room >= threshold and (look is None or look >= 60))
                        self.assertEqual(h2.action_from_facts(facts), h.label(s)[0])
                        self.assertEqual(s, before)
        for mission in h.MISSIONS:
            for bucket in ('far', 'medium', 'close', 'very close'):
                for greeted in (None, 299, 300):
                    s = item(mission=mission, scene=[['person', 'left', bucket]],
                             person_greeted_s_ago=greeted, new_room_s_ago=600)
                    facts = h2.truth(s)
                    self.assertTrue(facts['person_visible'])
                    self.assertEqual(facts['person_close'], bucket in ('close', 'very close'))
                    self.assertFalse(facts['nothing_new_for_a_while'])
                    self.assertEqual(h2.action_from_facts(facts), h.label(s)[0])

    def test_b3_uses_only_gemma_booleans(self):
        for bits in itertools.product((False, True), repeat=5):
            facts = dict(zip(h2.FIELDS[:-1], bits))
            visible, close, recent, allows, idle = bits
            expected = ('EXPLORE' if visible and (not allows or recent) else
                        'GREET' if visible and close else 'APPROACH' if visible else
                        'LOOK' if idle else 'EXPLORE')
            self.assertEqual(h2.action_from_facts(facts), expected)

    def test_fixed_grammar_prompt_and_payload(self):
        for person in (False, True):
            s = item('APPROACH' if person else 'EXPLORE')
            for variant in ('B1', 'B2'):
                body = h2.json_payload(s, variant)
                expected = '<start_of_turn>user\n' + h.state(s, 'V1' if variant == 'B1' else 'V2')
                expected += '\n' + h2.INSTRUCTION + '<end_of_turn>\n<start_of_turn>model\n'
                self.assertEqual(body['prompt'], expected)
                self.assertEqual((body['n_predict'], body['temperature'], body['cache_prompt']), (96, 0, False))
                self.assertEqual(set(body), {'prompt', 'grammar', 'temperature', 'n_predict', 'cache_prompt', 'stream', 'stop'})
                grammar = body['grammar']
                positions = [grammar.index(json.dumps(json.dumps(k))) for k in h2.FIELDS]
                self.assertEqual(positions, sorted(positions))
                self.assertEqual('APPROACH' in grammar, person)
                self.assertEqual('GREET' in grammar, person)
                self.assertIn('boolean ::= "true" | "false"', grammar)

    def test_invalid_json_is_wrong_and_never_retried(self):
        s = item()
        good = {**h2.truth(s), 'action': 'EXPLORE'}
        self.assertEqual(h2.parse_json(json.dumps(good), False), (good, True))
        invalid = ['bad', '{}', '[]', json.dumps(dict(reversed(list(good.items())))),
                   json.dumps({**good, 'person_visible': 0}), json.dumps({**good, 'extra': True}),
                   json.dumps({**good, 'action': 'APPROACH'}),
                   json.dumps(good)[:-1] + ', "action":"LOOK"}']
        for raw in invalid:
            transport = mock.Mock(return_value={'content': raw, 'timings': {'prompt_n': 100, 'predicted_n': 96}})
            rows = list(h2.json_rows([s], 'B1', 'E2B', transport))
            transport.assert_called_once()
            self.assertFalse(rows[0]['valid'])
            self.assertIsNone(rows[0]['choice'])
            self.assertEqual(h2.metrics(rows)['accuracy'], 0)
            self.assertEqual(h2.metrics(rows)['recall']['EXPLORE'], 0)


class SelectionTest(unittest.TestCase):
    def score(self, macro=.8, tokens=100, violations=0, lowest=.6, validity=1):
        return dict(macro_recall=macro, median_total_tokens=tokens, r1_violations=violations,
                    recall=dict.fromkeys(h.MODES, lowest), json_validity=validity)

    def test_selection_thresholds_token_ties_and_diagnostics(self):
        scores = {v: self.score(violations=1) for v in h2.CANDIDATES}
        scores.update(A5=self.score(1), **{'B3-E4B': self.score(1)})
        self.assertIsNone(h2.select(scores))
        scores['A0'] = self.score(.7, 120)
        scores['B1-E2B'] = self.score(.8, 110, validity=.949)
        scores['A2'] = self.score(.9, lowest=.599)
        self.assertEqual(h2.select(scores), 'A0')
        scores['B1-E2B']['json_validity'] = .95
        self.assertEqual(h2.select(scores), 'B1-E2B')
        scores['A4'] = self.score(.8, 109)
        self.assertEqual(h2.select(scores), 'A4')
        scores['A0'] = self.score(.8, 109)
        self.assertEqual(h2.select(scores), 'A0')

    def test_metrics_include_invalid_and_use_median_of_token_sums(self):
        rows = []
        for i, mode in enumerate(h.MODES):
            rows.append(dict(label=mode, rule='R1' if i == 0 else 'R6', choice=mode if i else None,
                             valid=bool(i), prompt_tokens=(10, 30, 50, 70)[i],
                             generated_tokens=(90, 0, 90, 0)[i], ms=i))
        m = h2.metrics(rows)
        self.assertEqual((m['accuracy'], m['macro_recall'], m['json_validity']), (.75, .75, .75))
        self.assertEqual(m['confusion']['EXPLORE']['INVALID'], 1)
        self.assertEqual(m['median_total_tokens'], 85)

    def test_equal_mathematical_macro_recalls_use_token_tie_break(self):
        def rows(counts, tokens):
            return [dict(label=mode, rule='R6', choice=mode if i < count else h.MODES[(j + 1) % 4],
                         valid=True, prompt_tokens=tokens, generated_tokens=1, ms=0)
                    for j, (mode, count) in enumerate(zip(h.MODES, counts)) for i in range(18)]
        a = h2.metrics(rows((11, 11, 11, 13), 110))
        b = h2.metrics(rows((11, 11, 12, 12), 100))
        self.assertEqual(a['macro_recall'], b['macro_recall'])
        self.assertEqual(a['macro_recall'], 46 / 72)
        scores = {v: self.score(violations=1) for v in h2.CANDIDATES}
        scores.update(A0=a, A2=b)
        self.assertEqual(h2.select(scores), 'A2')

    def test_server_argv_equals_named_setups_without_mtp(self):
        for model, setup in [('E2B', 'setup_q4'), ('E4B', 'setup_e4b_q4')]:
            calls = []
            with mock.patch.object(h2.sm.os, 'system', calls.append), mock.patch.object(h2.sm.time, 'sleep'), \
                    mock.patch.object(h2.sm, 'wait_for_server', return_value=True), redirect_stdout(io.StringIO()):
                self.assertTrue(h2.sm.start_setup(setup))
            self.assertEqual(h2.server_command(model), shlex.split(calls[-1])[:-2])
            self.assertNotIn('--model-draft', h2.server_command(model))


class ExecutionTest(unittest.TestCase):
    def test_server_cleanup_on_scoring_failure_and_load_failure(self):
        for load_failed in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                proc = mock.Mock(pid=123, returncode=1 if load_failed else 0)
                proc.poll.side_effect = [1, 1] if load_failed else [None, None]
                transport = mock.Mock(return_value={'status': 'ok'})
                with mock.patch.object(h2, 'servers_left', return_value=''), \
                        mock.patch.object(h2, 'mem_available', return_value='MemAvailable: 123 kB'), \
                        mock.patch.object(h2, 'DeskHTTP', return_value=transport), \
                        mock.patch.object(h2.subprocess, 'Popen', return_value=proc), \
                        mock.patch.object(h2.os, 'killpg') as kill, redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        with h2.server(Path(temp), 'E4B', 'DEV'):
                            raise RuntimeError('scoring failed')
                    self.assertEqual(kill.call_count, 0 if load_failed else 1)
                    record = h2.read_rows(Path(temp) / 'servers.jsonl')[0]
                    self.assertEqual(record['servers_left'], '')
                    self.assertEqual(record['exit'], proc.returncode)

    def test_existing_server_is_never_stopped(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(h2, 'servers_left', return_value='123'), \
                mock.patch.object(h2.subprocess, 'Popen') as spawn, mock.patch.object(h2.os, 'killpg') as kill:
            with self.assertRaises(RuntimeError):
                with h2.server(Path(temp), 'E2B', 'DEV'):
                    self.fail('must refuse existing server')
            spawn.assert_not_called()
            kill.assert_not_called()

    def test_desk_transport_has_120_second_timeout_without_changing_production(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"status":"ok"}'
        timeout = h.picker.TIMEOUT_S
        with mock.patch.object(h2.urllib.request, 'urlopen', return_value=response) as opened:
            self.assertEqual(h2.DeskHTTP()('/health'), {'status': 'ok'})
        self.assertEqual(opened.call_args.kwargs['timeout'], 120)
        self.assertEqual(h.picker.TIMEOUT_S, timeout)

    def source(self, directory):
        source = directory / 'source'
        source.mkdir()
        items = [item(mode, id=f'{mode}{i}', split='DEV' if i < 18 else 'TEST')
                 for mode in h.MODES for i in range(30)]
        (source / 'set.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in items))
        digest = h.sha(source / 'set.jsonl')
        (source / 'set_sha256.txt').write_text(digest + '  set.jsonl\n')
        h.write_json(source / 'freeze.json', {'set_sha256': digest})
        return source, items, digest

    def execute(self, qualifies):
        with tempfile.TemporaryDirectory() as temp:
            source, items, digest = self.source(Path(temp))
            out = Path(temp) / 'out'
            calls, starts, calibrations = [], [], []
            by_prompt = {h2.json_payload(s, v)['prompt']: s for s in items for v in ('B1', 'B2')}
            class FakeGemma:
                def __init__(self, transport):
                    self.transport, self.prompt_tokens = transport, 100
                def decide(self, text, phrases):
                    if 'N/A' in text:
                        calibrations.append(text)
                    self.transport.last = {'content': 'A', 'timings': {'predicted_n': 1}}
                    return {p: (2 if i == 0 else 1) / (len(phrases) + 1) for i, p in enumerate(phrases)}
            @contextmanager
            def fake_server(out, model, phase):
                starts.append((model, phase))
                class Transport:
                    def __call__(self, path, body):
                        calls.append((model, phase, body))
                        s = by_prompt[body['prompt']]
                        raw = json.dumps({**h2.truth(s), 'action': h.label(s)[0]}) if qualifies else 'INVALID'
                        return {'content': raw, 'timings': {'prompt_n': len(body['prompt']), 'predicted_n': 80}}
                yield Transport()
            with mock.patch.object(h2, 'SOURCE', source), mock.patch.object(h2, 'SET_SHA', digest), \
                    mock.patch.object(h2, 'server', side_effect=fake_server), \
                    mock.patch.object(h2.h, 'StaticGemma', FakeGemma), redirect_stdout(io.StringIO()):
                h2.run(out)
                summary = json.loads((out / 'summary.json').read_text())
                self.assertEqual(summary['dev'], h2.summarize(out)['dev'])
                self.assertEqual(len(calibrations), 4)
                self.assertTrue(all(t == h.state(items[0], 'V2', True) for t in calibrations))
                self.assertEqual(starts[:2], [('E4B', 'DEV'), ('E2B', 'DEV')])
                self.assertEqual(len(calls), 288 + (48 if qualifies else 0))
                self.assertEqual(len(list(out.glob('*_TEST.jsonl'))), int(qualifies))
                self.assertEqual(h.sha(out / 'set.jsonl'), h.sha(source / 'set.jsonl'))
                events = h2.read_rows(out / 'events.jsonl')
                selection = next(i for i, e in enumerate(events) if e['event'] == 'selection_frozen')
                if qualifies:
                    self.assertEqual(summary['selected'], 'B1-E2B')
                    self.assertEqual(summary['test']['n'], 48)
                    self.assertEqual(summary['test']['accuracy'], 1)
                    self.assertEqual(starts[-1], ('E2B', 'TEST'))
                    self.assertGreater(next(i for i, e in enumerate(events) if e['event'] == 'test_started'), selection)
                    self.assertTrue(all(v == 1 for fields in summary['fact_accuracy'].values() for v in fields.values()))
                else:
                    self.assertIsNone(summary['selected'])
                    self.assertEqual(summary['outcome'], 'NO VARIANT QUALIFIES')
                    self.assertEqual(summary['dev']['B1-E2B']['accuracy'], 0)
                    self.assertTrue(all(v == 0 for fields in summary['fact_accuracy'].values() for v in fields.values()))
                before = len(calls)
                with self.assertRaises((FileExistsError, AssertionError)):
                    h2.run(out)
                self.assertEqual(len(calls), before)

    def test_complete_run_without_qualifier_never_touches_test(self):
        self.execute(False)

    def test_complete_run_selects_one_winner_and_scores_test_once(self):
        self.execute(True)

    def test_saved_results(self):
        out = h.ROOT / 'picker_prompt2_results'
        if not out.is_dir():
            self.skipTest('external run artifacts supplied in isolated review copy only')
        frozen = json.loads((out / 'freeze2.json').read_text())
        self.assertEqual(h.sha(out / 'set.jsonl'), h2.SET_SHA)
        for name, digest in frozen['copied_sha256'].items():
            self.assertEqual(h.sha(out / name), digest)
        for name, digest in frozen['code_sha256'].items():
            self.assertEqual(h.sha(Path(h2.__file__).parent / name), digest)
        summary = json.loads((out / 'summary.json').read_text())
        rebuilt = h2.summarize(out)
        for field in ('dev', 'fact_accuracy', 'selected', 'a5_changed'):
            self.assertEqual(summary[field], rebuilt[field])
        chosen = summary['selected']
        self.assertEqual(sorted(p.name for p in out.glob('*_TEST.jsonl')), [chosen + '_TEST.jsonl'] if chosen else [])
        events = h2.read_rows(out / 'events.jsonl')
        selection = next(i for i, e in enumerate(events) if e['event'] == 'selection_frozen')
        if chosen:
            test = h2.read_rows(out / (chosen + '_TEST.jsonl'))
            items = {s['id']: s for s in h2.read_rows(out / 'set.jsonl') if s['split'] == 'TEST'}
            self.assertEqual(len(test), 48)
            self.assertEqual({r['id'] for r in test}, set(items))
            self.assertEqual(summary['test'], h2.metrics(test))
            for r in test:
                self.assertEqual((r['label'], r['rule']), h.label(items[r['id']]))
            self.assertEqual(sum(e['event'] == 'test_started' for e in events), 1)
            self.assertGreater(next(i for i, e in enumerate(events) if e['event'] == 'test_started'), selection)
        else:
            self.assertFalse(any(e['event'] == 'test_started' for e in events))
        self.assertEqual(events[-1]['event'], 'finished')
        self.assertTrue(all(not s['servers_left'] for s in h2.read_rows(out / 'servers.jsonl')))
        attempt = h.ROOT / 'interrupted_dev_attempt'
        if attempt.exists():
            self.assertFalse(list(attempt.glob('B*_DEV.jsonl')))
            self.assertFalse(list(attempt.glob('*_TEST.jsonl')))
            def prompt_definitions(path):
                return [ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                        if not isinstance(n, (ast.Import, ast.ImportFrom))
                        and not (isinstance(n, ast.FunctionDef) and n.name == 'metrics')]
            self.assertEqual(prompt_definitions(attempt / 'attempt_source/harness2.py'),
                             prompt_definitions(Path(h2.__file__)))
            old_freeze = json.loads((attempt / 'freeze2.json').read_text())
            for name in ('harness2.py', 'test_harness2.py', 'README2.md'):
                self.assertEqual(h.sha(attempt / 'attempt_source' / name), old_freeze['code_sha256'][name])


if __name__ == '__main__':
    unittest.main()
