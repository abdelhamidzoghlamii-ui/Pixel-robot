"""Offline tests for repro.py: fakes and the archived Oct 14 run only; nothing is launched, no root, no inference.
Run from this folder with Termux python: python3 -B -m unittest -v test_repro"""
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro  # noqa: E402

cm, cr = repro.cm, repro.cr
sys.path.insert(0, str(repro.CHAT_MTP))
import test_chat_mtp_runner as old_tests  # noqa: E402  its run_block fakes; module import, so not rediscovered here

ARCH_RUN = json.loads((repro.ARCHIVE / 'run.json').read_text())
ARCH_ROWS = cm.read_jsonl(repro.ARCHIVE / 'prompts.jsonl')
ARCH_BLOCKS = cm.read_jsonl(repro.ARCHIVE / 'blocks.jsonl')


def freq(cap=2850000):
    tis = {'500000': 100, '2850000': 900}
    return {'policy0': {'max': 1803000, 'hw_max': 1803000, 'time_in_state': {'1803000': 5}},
            'policy4': {'max': 2348000, 'hw_max': 2348000, 'time_in_state': {'2348000': 5}},
            'policy6': {'max': cap, 'hw_max': 2850000, 'time_in_state': tis}}


def conditions(skin=25.0, status=0, pct=90, cap=2850000):
    return {'utc': 'u', 'cpufreq': freq(cap), 'battery_pct': pct, 'battery_status': 'Discharging', 'battery_c': 28.0,
            'skin': skin, 'status': status, 'z9': 26000}


def make_folder(root, config, skin=25.0, agents_end='', result=True):
    """A repro folder built from the archived Oct 14 rows and block of that config."""
    out = Path(root) / f'X_{config}'
    out.mkdir()
    run = {'config': config, 'ids': cm.IDS, 'command': repro.COMMANDS[config], 'payload_example': repro.payload('<text>'),
           'prompt_texts': ARCH_RUN['prompt_texts'], 'model_sha256': ARCH_RUN['model_sha256'], 'build_sha256': {'b': 1},
           'code_sha256': {'c': 1}}
    (out / 'run.json').write_text(json.dumps(run))
    rows = [r for r in ARCH_ROWS if r['config'] == config]
    (out / 'prompts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    block = copy.deepcopy(next(b for b in ARCH_BLOCKS if b['config'] == config))
    block['gate']['skin'] = skin
    end = conditions(skin + 7)
    end['cpufreq']['policy6']['time_in_state'] = {'500000': 150, '2850000': 950}
    if result:
        (out / 'result.json').write_text(json.dumps({'block': block, 'start': conditions(skin), 'end': end,
                                                     'agents_end': agents_end, 'port_free_end': True}))
    return out


class FrozenWorkloadTest(unittest.TestCase):
    def test_frozen_values_equal_the_archived_run(self):
        self.assertEqual(repro.frozen_problems(), [])
        self.assertEqual(repro.COMMANDS, ARCH_RUN['commands'])
        self.assertEqual(repro.payload('<text>'), ARCH_RUN['payload_example'])
        self.assertEqual(cm.prompt_texts(), ARCH_RUN['prompt_texts'])

    def test_a_changed_command_payload_or_code_hash_is_refused(self):
        with mock.patch.dict(repro.COMMANDS, {'ON': repro.OFF + ['--spec-draft-n-max', '4']}):
            self.assertEqual(repro.frozen_problems(), ['commands differ from the Oct 14 run.json'])
        with mock.patch.object(repro, 'payload', lambda t: {'prompt': t}):
            self.assertEqual(repro.frozen_problems(), ['payload differ from the Oct 14 run.json'])
        p = next(iter(repro.CODE_SHA256))
        with mock.patch.dict(repro.CODE_SHA256, {p: '0' * 64}):
            self.assertEqual(len(repro.frozen_problems()), 1)
        with mock.patch.object(repro, 'ARCHIVE_RUN_SHA256', '0' * 64):
            self.assertIn('not the archived file', repro.frozen_problems()[0])

    def test_payload_is_not_todays_chat_template(self):  # TPL-FIX changed chat.py after Oct 14
        today = cm.chat.chat_payload(cm.chat.build_prompt('gemma', [{'role': 'user', 'content': 'x'}]), 'gemma')
        self.assertNotEqual(repro.payload('x')['prompt'], today['prompt'])
        self.assertEqual(repro.payload('x')['prompt'], '<start_of_turn>user\nx<end_of_turn>\n<start_of_turn>model\n')

    def test_run_block_spawns_and_posts_the_frozen_values(self):
        helper = old_tests.BlockTest('test_deadline_constants')
        b, posts, stops, emitted, events, proc = helper.run_block([4] * 13)
        self.assertEqual(events[0], ('spawn', repro.COMMANDS['ON']))
        self.assertEqual([p[1] for p in posts], [repro.payload(f'text {i}') for i in cm.IDS])
        self.assertEqual((b['end_reason'], len(stops)), ('all prompts done', 1))


class ReportTest(unittest.TestCase):
    def test_archive_rows_reproduce_the_oct_14_figures(self):
        text = repro.report([])
        off = next(l for l in text.splitlines() if l.startswith('Oct14') and ' OFF ' in l).split()
        on = next(l for l in text.splitlines() if l.startswith('Oct14') and ' ON ' in l).split()
        self.assertEqual((off[4], on[4], on[6]), ('13.37', '19.18', '39.0%'))  # name is two words
        self.assertEqual((off[7], off[8], on[8]), ('675', '74.7', '114.0'))

    def test_matched_pair_is_valid_same_and_matched(self):
        with tempfile.TemporaryDirectory() as d:
            text = repro.report([make_folder(d, 'OFF', 25.0), make_folder(d, 'ON', 25.6)])
        lines = text.splitlines()
        self.assertEqual(sum(l.startswith('X_') and l.endswith('  VALID') for l in lines), 2)
        self.assertIn('ON/OFF gen t/s: 1.435 (Oct 14: 19.18 / 13.37 = 1.435)', lines)
        self.assertIn('configuration: SAME', lines)
        self.assertIn('conditions: MATCHED', lines)
        x = next(l for l in lines if l.startswith('X_ON')).split()
        self.assertEqual((x[14], x[15]), ('2850', '50%'))  # cap; (950-900)/((150-100)+(950-900)) at max

    def test_unmatched_and_not_valid_are_labelled(self):
        with tempfile.TemporaryDirectory() as d:
            off, on = make_folder(d, 'OFF', 25.0), make_folder(d, 'ON', 27.0, agents_end='123 claude')
            run = json.loads((on / 'run.json').read_text())
            run['code_sha256'] = {'c': 2}
            (on / 'run.json').write_text(json.dumps(run))
            text = repro.report([off, on])
        self.assertIn('conditions: UNMATCHED: start skin 25.0 vs 27.0', text)
        self.assertIn('configuration: DIFFERENT: code_sha256 differs', text)
        self.assertIn('NOT VALID: agent process at end: 123 claude', text)

    def test_incomplete_folder(self):
        with tempfile.TemporaryDirectory() as d:
            text = repro.report([make_folder(d, 'OFF'), make_folder(d, 'ON', result=False)])
        self.assertIn('INCOMPLETE: no result.json', text)
        self.assertIn('comparison: needs exactly one OFF and one ON folder', text)

    def test_missing_prompt_is_not_valid(self):
        with tempfile.TemporaryDirectory() as d:
            off = make_folder(d, 'OFF')
            rows = (off / 'prompts.jsonl').read_text().splitlines()
            (off / 'prompts.jsonl').write_text('\n'.join(rows[:-1]) + '\n')
            self.assertIn('12/13 prompts', repro.load(off)[3])


class MainTest(unittest.TestCase):
    def run_main(self, config, start=None, refusals=()):
        calls = []

        def run_block(seq, cfg, todo, texts, ctx):
            calls.append((seq, cfg, todo, repro.pm.GATE_MAX_S, ctx['idle']))
            for r in ARCH_ROWS:
                if r['config'] == cfg:
                    ctx['emit'](r)
            return next(b for b in ARCH_BLOCKS if b['config'] == cfg)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(repro, 'OUT_ROOT', Path(d)), mock.patch.object(cr, 'require_native'), \
                mock.patch.object(repro.signal, 'signal'), \
                mock.patch.object(cm, 'refusals', return_value=list(refusals)), \
                mock.patch.object(repro, 'check_hashes', return_value=dict(repro.SHA256)), \
                mock.patch.object(repro, 'build_sha256', return_value={'llama-server': 'h'}), \
                mock.patch.object(cr, 'wait_cores'), \
                mock.patch.object(repro, 'conditions', return_value=start or conditions()), \
                mock.patch.object(repro, 'agents', return_value=''), mock.patch.object(repro, 'port_free', return_value=True), \
                mock.patch.object(cm, 'run_block', run_block), mock.patch.object(repro.pm, 'GATE_MAX_S', 480), \
                mock.patch.object(cm, 'LIVE', ['p']), mock.patch.object(cm, 'stop') as stop, \
                redirect_stdout(io.StringIO()):
            try:
                repro.main(['--config', config])
                err = None
            except SystemExit as e:
                err = str(e)
            folders = list(Path(d).iterdir())
            files = {p.name: p.read_text() for p in folders[0].iterdir()} if folders else {}
        return calls, files, err, stop

    def test_one_launch_runs_only_its_config_once(self):
        for config in ('OFF', 'ON'):
            calls, files, err, stop = self.run_main(config)
            self.assertIsNone(err)
            self.assertEqual(calls, [(0, config, cm.IDS, 0, {'skin': 25.0})])
            run = json.loads(files['run.json'])
            self.assertEqual((run['config'], run['command']), (config, repro.COMMANDS[config]))
            self.assertEqual(len(files['prompts.jsonl'].splitlines()), 13)
            self.assertIn('  VALID', files['report.txt'])
            stop.assert_called_once_with('p')  # main's finally stops any server left

    def test_thermal_and_refusals_start_nothing(self):
        for start, refusals, want in ((conditions(status=1), (), 'Android thermal status 1'),
                                      (dict(conditions(), battery_c=45.0), (), 'battery 45.0 degC'),
                                      (None, ('battery 70 % < 80 %',), 'battery 70 %')):
            calls, files, err, stop = self.run_main('ON', start, refusals)
            self.assertEqual(calls, [])
            self.assertIn(want, err)

    def test_config_xor_report(self):
        for argv in ([], ['--config', 'ON', '--report', 'a', 'b']):
            with self.assertRaises(SystemExit), mock.patch('sys.stderr', io.StringIO()):
                repro.main(argv)


if __name__ == '__main__':
    unittest.main()
