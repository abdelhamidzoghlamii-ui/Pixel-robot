"""Offline tests for vis3d.py: fakes, the tracked Oct 14 archive and tracked M2-Q1 frames only; nothing is launched,
no root, no inference. Run from this folder with Termux python: python3 -B -m unittest -v test_vis3d"""
import base64
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vis3d  # noqa: E402

repro, cm, cr = vis3d.repro, vis3d.cm, vis3d.cr
sys.path.insert(0, str(vis3d.REPRO_DIR))
import test_repro  # noqa: E402  its Oct 14 fixtures (make_folder, conditions); module import, not rediscovered here
sys.path.insert(0, str(repro.CHAT_MTP))
import test_chat_mtp_runner as old_tests  # noqa: E402  run_block fakes

ARCH_ROWS, ARCH_BLOCKS = test_repro.ARCH_ROWS, test_repro.ARCH_BLOCKS
# Two image requests from VIS-1 V2 (Q8 projector, 70-token budget, -lv 4, b11716), copied verbatim
LOG = """0.08.241.813 I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
0.08.241.823 I slot   operator(): id  0 | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 268
0.08.490.400 I slot encode_mtmd_: id  0 | task 0 | encoding mtmd batch from idx = 5, n_chunks = 1
0.10.649.720 I decoding image batch 1/1, n_tokens_batch = 63
0.11.926.720 I image decoded (batch 1/1) in 1275 ms
0.21.322.589 I slot print_timing: id  0 | task 0 | prompt eval time =    8401.84 ms /   268 tokens (   31.35 ms per token,    31.90 tokens per second)
0.21.390.451 I slot launch_slot_: id  0 | task 40 | processing task, is_child = 0
0.21.649.817 I slot encode_mtmd_: id  0 | task 40 | encoding mtmd batch from idx = 5, n_chunks = 1
0.23.767.736 I decoding image batch 1/1, n_tokens_batch = 63
0.25.070.858 I image decoded (batch 1/1) in 1302 ms
"""
LIBS = [f'{vis3d.BIN}/libllama.so.0.7.0', f'{vis3d.BIN}/libggml-cpu.so.0.27.0', f'{vis3d.BIN}/libmtmd.so.0.7.0',
        '/system/lib64/libc.so']


def server_log(test):
    vision, spec = vis3d.INTENDED[test]
    return ('I srv  llama_server: model loaded\n' +
            (f"I srv load_model: loaded multimodal model, '{vis3d.MMPROJ}'\n" if vision else '') +
            (f"I common_speculative_init_result: loading draft model '{vis3d.DRAFT}'\n" if spec else ''))


def make(root, test, skin=25.0, first=None, every=None, log=None, n=None, name=None):
    """A VIS-3D folder. A, B: the Oct 14 OFF rows; C: the ON rows; D: 8 image rows. `first`/`every` override probe
    fields of the first / every row."""
    out = Path(root) / (name or f'X_{test}')
    out.mkdir()
    vision, spec = vis3d.INTENDED[test]
    if test == 'D':
        ids = vis3d.FRAMES
        rows = [{'config': 'D', 'prompt_id': f, 'prompt_s': 8.4, 'prompt_n': 268, 'predicted_n': 30,
                 'predicted_ms': 4000.0, 'wall_s': 12.5, 'draft_n': None, 'draft_n_accepted': None,
                 'stop_type': 'stop', 'reply': f'reply {f}'} for f in ids]
        texts, example = {f: vis3d.PROMPT for f in ids}, vis3d.image_payload('<frame>', jpeg='<base64 jpeg>')
    else:
        ids, texts, example = cm.IDS, test_repro.ARCH_RUN['prompt_texts'], repro.payload('<text>')
        rows = copy.deepcopy([r for r in ARCH_ROWS if r['config'] == ('ON' if spec else 'OFF')])
    rows = rows[:n]
    total = 0
    for i, r in enumerate(rows):
        r['config'] = test
        total += (r['predicted_n'] - (r['draft_n_accepted'] or 0)) if spec else 0
        r['probe'] = {'metrics': {'llamacpp:spec_decode_num_drafts_total': float(total)}, **(every or {})}
        if i == 0:
            r['probe'].update(props={'modalities': {'vision': vision}}, slots=[{'speculative': spec}], libs=LIBS)
            if test == 'D':
                r['probe']['template'] = {'prompt': f'<|turn>user\n{vis3d.PROMPT}<turn|>\n<|turn>model\n'}
            r['probe'].update(first or {})
    run = {'test': test, 'ids': ids, 'command': vis3d.COMMANDS[test], 'payload_example': example, 'prompt_texts': texts,
           'model_sha256': dict(vis3d.SHA256), 'build_sha256': dict(vis3d.BUILD_SHA256), 'code_sha256': {'c': 1}}
    (out / 'run.json').write_text(json.dumps(run))
    (out / 'prompts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / f'server_block00_{test}.log').write_text(server_log(test) + (LOG * 4 if test == 'D' else '')
                                                    if log is None else log)
    block = copy.deepcopy(next(b for b in ARCH_BLOCKS if b['config'] == ('ON' if spec else 'OFF')))
    block['gate']['skin'] = skin
    if n is not None:
        block['end_reason'] = f'{ids[n]} not finished at 175 s: dropped, redone later'
    end = test_repro.conditions(skin + 7)
    end['cpufreq']['policy6']['time_in_state'] = {'500000': 150, '2850000': 950}
    (out / 'result.json').write_text(json.dumps({'block': block, 'start': test_repro.conditions(skin), 'end': end,
                                                 'agents_end': '', 'port_free_end': True}))
    return out


def baseline(d):
    return {c: test_repro.make_folder(d, c) for c in ('OFF', 'ON')}  # VIS-3C folders from the Oct 14 rows


class CommandTest(unittest.TestCase):
    def test_a_is_the_frozen_off_command_on_b11716_plus_metrics(self):
        old = repro.COMMANDS['OFF']
        self.assertEqual(vis3d.COMMANDS['A'], old[:3] + [vis3d.SERVER] + old[4:] + ['--metrics'])
        self.assertEqual(old[3], f'{repro.HOME}/llama.cpp-b1609-dotprod/build/bin/llama-server')
        for k in ('--ctx-size 4096', '--threads 4', '--threads-batch 4', '--parallel 1', '--port 8080'):
            self.assertIn(k, ' '.join(vis3d.COMMANDS['A']))

    def test_each_test_adds_only_its_flags(self):
        c = vis3d.COMMANDS
        self.assertEqual(c['B'], c['A'] + ['--mmproj', vis3d.MMPROJ, '--image-min-tokens', '70', '--image-max-tokens', '70'])
        self.assertEqual(c['C'], c['B'] + ['--model-draft', vis3d.DRAFT, '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3'])
        self.assertEqual(c['C'][len(c['B']):], repro.COMMANDS['ON'][len(repro.OFF):])  # the VIS-3C ON flags
        self.assertEqual(c['D'], c['B'] + ['--reasoning', 'off', '-lv', '4'])
        self.assertEqual(vis3d.MMPROJ, f'{repro.HOME}/models/vis0/mmproj-gemma-4-E2B-it-Q8_0.gguf')
        self.assertNotIn('BF16', ' '.join(sum(c.values(), [])))

    def test_run_block_spawns_the_test_command_and_posts_the_frozen_payload(self):
        with mock.patch.object(cm, 'command', lambda config, model='e2b': list(vis3d.COMMANDS['C'])):
            b, posts, stops, emitted, events, proc = old_tests.BlockTest('test_deadline_constants').run_block([4] * 13)
        self.assertEqual(events[0], ('spawn', vis3d.COMMANDS['C']))
        self.assertEqual([p[1] for p in posts], [repro.payload(f'text {i}') for i in cm.IDS])
        self.assertEqual((b['end_reason'], len(stops)), ('all prompts done', 1))

    def test_timeout_ends_the_block_and_the_run_is_not_valid(self):
        with mock.patch.object(cm, 'command', lambda config, model='e2b': list(vis3d.COMMANDS['A'])):
            b, posts, stops, *_ = old_tests.BlockTest('test_deadline_constants').run_block([100] * 13)
        self.assertIn('not finished at 175 s', b['end_reason'])
        self.assertEqual(stops, [175.0])  # the watchdog's deadline; the server is stopped once
        with tempfile.TemporaryDirectory() as d:
            self.assertIn('not finished at 175 s', ' '.join(vis3d.check_folder(make(d, 'A', n=2))[3]))


class WorkloadTest(unittest.TestCase):
    def test_text_workload_is_the_oct_14_archive(self):
        self.assertEqual(repro.frozen_problems(), [])  # payload, prompt texts, ids, hashes = the archived run.json
        self.assertEqual(repro.payload('x')['prompt'], '<start_of_turn>user\nx<end_of_turn>\n<start_of_turn>model\n')
        self.assertEqual(vis3d.code_problems(), [])

    def test_image_request_serialization(self):
        f = vis3d.FRAMES[0]
        p = vis3d.image_payload(f)
        image, text = p['messages'][0]['content']
        self.assertEqual(p['messages'][0]['role'], 'user')
        self.assertEqual(image['type'], 'image_url')
        self.assertTrue(image['image_url']['url'].startswith('data:image/jpeg;base64,'))
        jpeg = base64.b64decode(image['image_url']['url'].split(',', 1)[1])
        self.assertEqual(jpeg, (vis3d.FRAMES_DIR / f'{f}.jpg').read_bytes())
        self.assertEqual(text, {'type': 'text', 'text': vis3d.PROMPT})
        self.assertEqual({k: p[k] for k in ('temperature', 'top_k', 'seed', 'max_tokens', 'cache_prompt', 'stream')},
                         {'temperature': 0, 'top_k': 1, 'seed': 0, 'max_tokens': 96, 'cache_prompt': False, 'stream': False})
        self.assertNotIn('prompt', p)  # the server applies the GGUF template; no hand-written turn markers
        json.dumps(p)

    def test_chat_post_uses_the_chat_endpoint_and_returns_the_reply(self):
        calls = []
        reply = {'choices': [{'message': {'content': 'A table.'}, 'finish_reason': 'length'}], 'timings': {'prompt_n': 9}}
        with mock.patch.object(vis3d, '_post', lambda path, body, timeout: calls.append((path, timeout)) or reply):
            data = vis3d.chat_post('/completion', {'messages': []}, 30)
        self.assertEqual(calls, [('/v1/chat/completions', 30)])
        self.assertEqual((data['content'], data['stop_type'], data['timings']), ('A table.', 'length', {'prompt_n': 9}))
        with self.assertRaises(AssertionError):
            vis3d.chat_post('/other', {}, 1)

    def test_references_match_the_tracked_m2_q1_frames(self):
        src = json.loads((vis3d.FRAMES_DIR.parent / 'results.json').read_text())
        by = {Path(p['frame_file']).stem: p for p in src['photos']}
        for f, r in vis3d.REFS['frames'].items():
            self.assertEqual((r['frame_sha256'], r['source_sha256'], r['photo']),
                             (by[f]['frame_sha256'], by[f]['source_sha256'], by[f]['photo']))
            self.assertTrue(r['reference'] and r['key_objects'] and r['robot_context'])
        self.assertEqual(vis3d.frame_problems(), [])
        self.assertEqual(len(vis3d.FRAMES), 8)


class HashTest(unittest.TestCase):
    def test_changed_weight_or_library_is_refused(self):
        for bad in (vis3d.MMPROJ, f'{vis3d.BIN}/libmtmd.so.0.7.0'):
            want = {**{str(p): h for p, h in vis3d.SHA256.items()},
                    **{f'{vis3d.BIN}/{n}': h for n, h in vis3d.BUILD_SHA256.items()}}
            with mock.patch.object(cr, 'sha256', lambda p: '0' * 64 if str(p) == bad else want[str(p)]), \
                    mock.patch.object(vis3d.Path, 'is_file', return_value=True):
                with self.assertRaises(SystemExit) as e:
                    vis3d.check_hashes()
            self.assertIn(Path(bad).name, str(e.exception))

    def test_all_pinned_files_pass(self):
        want = {**{str(p): h for p, h in vis3d.SHA256.items()},
                **{f'{vis3d.BIN}/{n}': h for n, h in vis3d.BUILD_SHA256.items()}}
        with mock.patch.object(cr, 'sha256', lambda p: want[str(p)]), \
                mock.patch.object(vis3d.Path, 'is_file', return_value=True):
            got, build = vis3d.check_hashes()
        self.assertEqual((got, build), (vis3d.SHA256, vis3d.BUILD_SHA256))

    def test_changed_frame_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for f in vis3d.FRAMES:
                (Path(d) / f'{f}.jpg').write_bytes((vis3d.FRAMES_DIR / f'{f}.jpg').read_bytes())
            (Path(d) / f'{vis3d.FRAMES[3]}.jpg').write_bytes(b'x')
            with mock.patch.object(vis3d, 'FRAMES_DIR', Path(d)):
                why = vis3d.frame_problems()
        self.assertEqual(len(why), 1)
        self.assertIn(vis3d.FRAMES[3], why[0])

    def test_environment_overrides_are_refused(self):
        with mock.patch.dict(os.environ, {'LLAMA_ARG_THREADS': '8', 'OMP_NUM_THREADS': '2'}):
            self.assertIn('LLAMA_ARG_THREADS, OMP_NUM_THREADS', vis3d.env_problems()[0])
        clean = {k: v for k, v in os.environ.items() if not k.startswith(vis3d.ENV_REFUSED)}
        with mock.patch.dict(os.environ, {**clean, 'LD_PRELOAD': 'libtermux-exec.so'}, clear=True):
            self.assertEqual(vis3d.env_problems(), [])  # termux-exec's preload is recorded, not refused


class ParseTest(unittest.TestCase):
    def test_image_log(self):
        self.assertEqual(vis3d.image_log(LOG), [{'encode_ms': 2159.3, 'image_tokens': 63, 'image_decode_ms': 1275},
                                                {'encode_ms': 2117.9, 'image_tokens': 63, 'image_decode_ms': 1302}])
        self.assertEqual(vis3d.image_log('x\n0.0.0.0 I slot launch_slot_: | task 1 | processing task\nno image\n'),
                         [{'encode_ms': None, 'image_tokens': None, 'image_decode_ms': None}])

    def test_metrics(self):
        text = ('# HELP llamacpp:spec_decode_num_drafts_total Speculative: ...\n# TYPE x counter\n'
                'llamacpp:spec_decode_num_drafts_total 240\nllamacpp:tokens_predicted_total 674\n')
        self.assertEqual(vis3d.parse_metrics(text), {'llamacpp:spec_decode_num_drafts_total': 240.0,
                                                     'llamacpp:tokens_predicted_total': 674.0})

    def test_probe_reads_once_and_records_errors(self):
        pages = {'/metrics': 'llamacpp:spec_decode_num_drafts_total 3\n', '/props': '{"modalities": {"vision": true}}',
                 '/slots': '[{"speculative": false}]'}

        def get(path, timeout=10):
            if path not in pages:
                raise OSError('refused')
            return pages[path]
        with mock.patch.object(vis3d, 'get', get), mock.patch.object(vis3d, 'mapped_libs', return_value=LIBS), \
                mock.patch.object(cm, 'LIVE', [mock.Mock(pid=7)]), \
                mock.patch.object(vis3d, '_post', side_effect=OSError('down')):
            later, first = vis3d.probe('D', first=False), vis3d.probe('D', first=True)
        self.assertEqual(later, {'metrics': {'llamacpp:spec_decode_num_drafts_total': 3.0}})
        self.assertEqual((first['props'], first['slots'], first['libs']), ({'modalities': {'vision': True}},
                                                                           [{'speculative': False}], LIBS))
        self.assertEqual(first['template'], {'error': 'OSError: down'})


class ValidityTest(unittest.TestCase):
    def test_intended_folders_are_valid(self):
        with tempfile.TemporaryDirectory() as d:
            for t in vis3d.COMMANDS:
                self.assertEqual(vis3d.check_folder(make(d, t))[3], [], t)

    def test_configuration_not_as_intended_is_not_valid(self):
        cases = (('B', {'props': {'modalities': {'vision': False}}}, None, '/props vision is not True'),
                 ('A', {'slots': [{'speculative': True}]}, None, '/slots speculative is not [False]'),
                 ('A', {'libs': ['/data/data/com.termux/files/usr/lib/libllama.so']}, None, 'not all from'),
                 ('A', {'props': {'error': 'URLError'}}, None, '/props vision'),
                 ('B', None, {'metrics': {'error': 'timeout'}}, '/metrics not read after every reply'),
                 ('D', {'template': {'prompt': '<start_of_turn>user\nx'}}, None, 'not Gemma 4'))
        with tempfile.TemporaryDirectory() as d:
            for i, (t, first, every, want) in enumerate(cases):
                why = vis3d.check_folder(make(d, t, first=first, every=every, name=f'{i}'))[3]
                self.assertIn(want, ' '.join(why), (t, why))

    def test_mtp_is_judged_from_the_server_not_the_command(self):
        with tempfile.TemporaryDirectory() as d:
            c = make(d, 'C', log='x\n')  # the command has the drafter; the log has neither load line
            rows = [dict(r, draft_n=0) for r in cm.read_jsonl(c / 'prompts.jsonl')]
            for r in rows:
                r['probe']['metrics'] = {'llamacpp:spec_decode_num_drafts_total': 0.0}
            (c / 'prompts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            why = ' '.join(vis3d.check_folder(c)[3])
        self.assertIn('MTP requested but nothing drafted or verified', why)
        self.assertIn('server log: drafter not loaded', why)
        self.assertIn('server log: projector not loaded', why)

    def test_incomplete_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as d:
            a = make(d, 'A')
            (a / 'result.json').unlink()
            self.assertEqual(vis3d.check_folder(a)[3], ['INCOMPLETE: no result.json (interrupted or failed)'])
            dd = make(d, 'D', n=5, log=server_log('D') + LOG * 3)
            why = vis3d.check_folder(dd)[3]
            self.assertIn('5/8 requests', why)
            text = vis3d.report([dd])
        self.assertIn('NO RESPONSE', text)
        self.assertIn('5/8 frames answered', text)
        self.assertIn('server log image sections do not match the requests', text)


class ReportTest(unittest.TestCase):
    def report(self, folders, d):
        b = baseline(d)
        with mock.patch.object(vis3d, 'BASELINE', b):
            return vis3d.report(folders)

    def test_text_rows_comparisons_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            text = self.report([make(d, 'A'), make(d, 'B', skin=25.5), make(d, 'C')], d)
        lines = text.splitlines()
        row = {}
        for l in lines:  # the first table; the second has the same names
            if l.startswith(('X_A', 'X_B', 'X_C')):
                row.setdefault(l.split()[0], l.split())
        self.assertEqual((row['X_A'][3], row['X_C'][3], row['X_C'][5]), ('13.37', '19.18', '39.0%'))  # Oct 14 rule
        # steps: exact counter; ms/step = decode ms / steps; ms/pass* = proxy (equal here because the fixture's
        # counter is built as generated - accepted)
        self.assertEqual((row['X_C'][7], row['X_C'][8], row['X_C'][9]), ('326', '114.0', '114.0'))
        self.assertEqual((row['X_A'][7], row['X_A'][8]), ('0', 'n/a'))
        self.assertIn('B/A gen t/s 1.000  (projector residency)', lines)
        self.assertIn('  configuration: SAME apart from projector residency', lines)
        self.assertIn('  configuration: SAME apart from MTP with the projector resident', lines)
        self.assertIn('  conditions: MATCHED', lines)
        self.assertIn('A/OFF gen t/s 1.000  (build, text only (b1609 OFF -> b11716 A))', lines)
        self.assertIn('  configuration: same workload and weights; differs by design: binary b1609-dotprod -> b11716,'
                      ' --metrics added, Q8 projector resident', lines)
        self.assertTrue(any(l.startswith('C/ON gen t/s') and 'projector residency' in l for l in lines))
        self.assertEqual(sum(l.endswith('  VALID') for l in lines), 3)
        self.assertIn('PROXY', text)

    def test_unmatched_different_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = make(d, 'A'), make(d, 'B', skin=27.0)
            run = json.loads((b / 'run.json').read_text())
            run['code_sha256'] = {'c': 2}
            (b / 'run.json').write_text(json.dumps(run))
            text = self.report([a, b, make(d, 'A', name='X_A2')], d)
        self.assertIn('conditions: UNMATCHED: start skin 25.0 vs 27.0', text)
        self.assertIn('configuration: DIFFERENT: code_sha256 differs', text)
        self.assertIn('X_A2: a second A folder, not compared', text)

    def test_image_report(self):
        with tempfile.TemporaryDirectory() as d:
            text = self.report([make(d, 'D')], d)
        self.assertIn('IMAGES  X_D  D  VALID', text)
        line = next(l for l in text.splitlines() if l.strip().startswith(vis3d.FRAMES[0]))
        # prompt 8400 ms - encode 2159 - image decode 1275 = 4966 text (derived); 30 tokens / 4 s = 7.50 t/s
        self.assertEqual(line.split()[1:9], ['63', '2159', '1275', '8400', '4966', '30', '7.50', '12.50'])
        self.assertIn('responses 8/8 complete', text)
        self.assertIn('GRADING SHEET', text)
        self.assertIn(vis3d.REFS['frames'][vis3d.FRAMES[1]]['reference'], text)

    def test_response_flags(self):
        self.assertEqual(vis3d.response_flag({'reply': ' ', 'stop_type': 'stop'}), 'EMPTY')
        self.assertEqual(vis3d.response_flag({'reply': 'x', 'stop_type': 'length'}), 'INCOMPLETE (length)')
        self.assertEqual(vis3d.response_flag({'reply': 'x', 'stop_type': 'stop'}), 'OK')


class MainTest(unittest.TestCase):
    def run_main(self, test, refusals=(), fail=None):
        calls = []

        def run_block(seq, cfg, todo, texts, ctx):
            calls.append((seq, cfg, todo, repro.pm.GATE_MAX_S, cm.command(cfg), cm.payload(texts[todo[0]]), cr.post))
            if fail:
                raise fail
            for r, pid in zip([r for r in ARCH_ROWS if r['config'] == 'OFF'], todo):
                ctx['emit'](dict(r, config=cfg, prompt_id=pid))
            return next(b for b in ARCH_BLOCKS if b['config'] == 'OFF')
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(vis3d, 'OUT_ROOT', Path(d)), mock.patch.object(cr, 'require_native'), \
                mock.patch.object(vis3d.signal, 'signal'), \
                mock.patch.object(cm, 'refusals', return_value=list(refusals)), \
                mock.patch.object(vis3d, 'check_hashes', return_value=(dict(vis3d.SHA256), dict(vis3d.BUILD_SHA256))), \
                mock.patch.object(cr, 'wait_cores'), \
                mock.patch.object(repro, 'conditions', return_value=test_repro.conditions()), \
                mock.patch.object(repro, 'agents', return_value=''), mock.patch.object(repro, 'port_free', return_value=True), \
                mock.patch.object(vis3d, 'probe', return_value={'metrics': {}}), \
                mock.patch.object(cm, 'run_block', run_block), mock.patch.object(repro.pm, 'GATE_MAX_S', 480), \
                mock.patch.object(cm, 'command', cm.command), mock.patch.object(cm, 'payload', cm.payload), \
                mock.patch.object(cr, 'post', cr.post), \
                mock.patch.object(cm, 'LIVE', ['p']), mock.patch.object(cm, 'stop') as stop, \
                redirect_stdout(io.StringIO()):
            try:
                vis3d.main(['--test', test])
                err = None
            except (SystemExit, RuntimeError) as e:
                err = str(e)
            folders = list(Path(d).iterdir())
            files = {p.name: p.read_text() for p in folders[0].iterdir()} if folders else {}
        return calls, files, err, stop

    def test_one_launch_runs_only_its_test_once(self):
        for test in ('A', 'B', 'C'):
            calls, files, err, stop = self.run_main(test)
            self.assertIsNone(err)
            self.assertEqual([c[:5] for c in calls], [(0, test, cm.IDS, 0, vis3d.COMMANDS[test])])
            self.assertEqual(calls[0][5], repro.payload(test_repro.ARCH_RUN['prompt_texts']['A5']))
            run = json.loads(files['run.json'])
            self.assertEqual((run['test'], run['command'], run['build_info']['source_commit_expected']),
                             (test, vis3d.COMMANDS[test], vis3d.SOURCE_COMMIT))
            self.assertEqual(len(files['prompts.jsonl'].splitlines()), 13)
            stop.assert_called_once_with('p')  # main's finally stops any server left

    def test_d_sends_prebuilt_image_requests_to_the_chat_endpoint(self):
        calls, files, err, stop = self.run_main('D')
        self.assertIsNone(err)
        seq, cfg, todo, gate, cmd, payload, post = calls[0]
        self.assertEqual((cfg, todo, cmd, post), ('D', vis3d.FRAMES, vis3d.COMMANDS['D'], vis3d.chat_post))
        self.assertEqual(payload, vis3d.image_payload(vis3d.FRAMES[0]))
        run = json.loads(files['run.json'])
        self.assertEqual(run['payload_example']['messages'][0]['content'][0]['image_url']['url'],
                         'data:image/jpeg;base64,<base64 jpeg>')
        self.assertEqual(run['frames_sha256'], {f: r['frame_sha256'] for f, r in vis3d.REFS['frames'].items()})

    def test_refusals_and_failures_start_nothing_or_clean_up(self):
        calls, files, err, stop = self.run_main('B', refusals=('battery 70 % < 80 %',))
        self.assertEqual((calls, files), ([], {}))
        self.assertIn('battery 70 %', err)
        calls, files, err, stop = self.run_main('C', fail=RuntimeError('llama-server not healthy'))
        self.assertIn('not healthy', err)
        self.assertNotIn('result.json', files)  # reported INCOMPLETE
        stop.assert_called_once_with('p')

    def test_launcher_contract(self):
        sh = (vis3d.HERE / 'oneshot.sh').read_text()
        old = (vis3d.REPRO_DIR / 'oneshot.sh').read_text()
        self.assertIn('case "$*" in "--test A"|"--test B"|"--test C"|"--test D") ;;', sh)
        self.assertIn('python -u "$R/vis3d.py" "$@"', sh)
        self.assertIn('R=$H/robot/benchmark/gemma_vision/vis3d\nS=/data/local/tmp/vis3d_screen.txt', sh)
        for kept in ("pgrep -fa 'claude|agy|node|codex'", 'check_battery\n\ntermux-wake-lock', 'mkdir "$L/.lock"',
                     'settings put system screen_off_timeout $OLD_TIMEOUT', 'thermal_zone9/temp', 'sleep 5; done'):
            self.assertIn(kept, sh)
            self.assertIn(kept, old)
        self.assertEqual(len(sh.splitlines()), len(old.splitlines()) - 1)  # one usage comment line fewer

    def test_test_xor_report(self):
        for argv in ([], ['--test', 'A', '--report', 'x'], ['--test', 'E']):
            with self.assertRaises(SystemExit), mock.patch('sys.stderr', io.StringIO()):
                vis3d.main(argv)


if __name__ == '__main__':
    unittest.main()
