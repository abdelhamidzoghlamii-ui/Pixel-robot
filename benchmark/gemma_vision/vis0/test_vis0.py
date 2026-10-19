"""Offline unit tests for benchmark/gemma_vision/vis0/vis0.py (no network, no llama-server)."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock
import urllib.error

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PIL import Image

import vis0


class FakeServer:
    def __init__(self, budget, label, fail_start=False, alive_sequence=None, log_text=''):
        self.budget = budget
        self.label = label
        self.fail_start = fail_start
        self.alive_sequence = list(alive_sequence or [True])
        self.log_text = log_text
        self.proc = type('FakeProc', (), {'pid': 99999, 'poll': lambda self_: None if self.is_alive() else 139})()
        self.started_utc = f'2026-10-18T00:00:00Z_{label}'
        self.ended_utc = None
        self.stopped = False

    def start(self):
        if self.fail_start:
            raise RuntimeError('simulated start failure')

    def stop(self):
        self.stopped = True
        self.ended_utc = f'2026-10-18T00:01:00Z_{self.label}'

    def is_alive(self):
        if len(self.alive_sequence) > 1:
            return self.alive_sequence.pop(0)
        return self.alive_sequence[0]

    def sample_vmhwm(self):
        return 2900000

    def log_size(self):
        return 0

    def log_delta_from(self, offset):
        return self.log_text[offset:]

    def log_tail(self, n_lines=25):
        return '\n'.join(self.log_text.splitlines()[-n_lines:])

    def summary(self):
        return {
            'label': self.label,
            'budget': self.budget,
            'command': vis0.build_server_cmd(budget=self.budget),
            'started_utc': self.started_utc,
            'ended_utc': self.ended_utc,
            'startup_s_not_a_measurement': 1.0,
            'server_pid': 99999,
            'server_name': 'llama-server',
            'peak_vmhwm_kib_not_a_measurement': 2900000,
            'exit_code': 0 if self.is_alive() else 139,
            'timing_label': vis0.TIMING_LABEL,
        }


class TestVis0Offline(unittest.TestCase):
    def test_labels_and_timing_constants(self):
        self.assertEqual(
            vis0.REPORT_LABELS,
            'Template check VALID (deterministic); '
            'vision replies VALID as outputs (temperature 0); '
            'timings and memory NOT measurements (agent resident); '
            'quality grading pending (Local AI)',
        )
        self.assertEqual(vis0.TIMING_LABEL, 'NOT A MEASUREMENT (agent resident)')

    def test_verify_and_hash_imports_matches_checkout(self):
        hashes = vis0.verify_and_hash_imports(vis0.ROBOT)
        self.assertIn('benchmark/gemma_vision/vis0/vis0.py', hashes)
        self.assertIn('chat.py', hashes)
        self.assertIn('server_manager.py', hashes)
        for rel_p, digest in hashes.items():
            self.assertEqual(len(digest), 64, rel_p)

    def test_verify_and_hash_imports_rejects_foreign_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                vis0.verify_and_hash_imports(Path(tmp))

    def test_encode_image_data_uri_roundtrip(self):
        raw = b'\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9'
        uri = vis0.encode_image_data_uri(raw)
        self.assertTrue(uri.startswith('data:image/jpeg;base64,'))
        b64_part = uri.split(',', 1)[1]
        self.assertEqual(base64.b64decode(b64_part), raw)

    def test_build_server_cmd_flags_and_port_8090(self):
        cmd = vis0.build_server_cmd(budget=140)
        self.assertEqual(cmd[:2], ['timeout', '3600'])
        self.assertIn('--host', cmd)
        self.assertEqual(cmd[cmd.index('--host') + 1], '127.0.0.1')
        self.assertEqual(cmd[cmd.index('--port') + 1], '8090')
        self.assertEqual(cmd[cmd.index('--ctx-size') + 1], '4096')
        self.assertEqual(cmd[cmd.index('--threads') + 1], '4')
        self.assertEqual(cmd[cmd.index('--threads-batch') + 1], '4')
        self.assertEqual(cmd[cmd.index('--parallel') + 1], '1')
        self.assertIn('--swa-full', cmd)
        self.assertEqual(cmd[cmd.index('--cache-ram') + 1], '0')
        self.assertEqual(cmd[cmd.index('--reasoning') + 1], 'off')
        self.assertEqual(cmd[cmd.index('-lv') + 1], '4')
        self.assertEqual(cmd[cmd.index('--image-min-tokens') + 1], '140')
        self.assertEqual(cmd[cmd.index('--image-max-tokens') + 1], '140')
        self.assertNotIn('taskset', cmd)

    def test_request_body_builders(self):
        tok_body = vis0.build_tokenize_payload('<|turn>user')
        self.assertEqual(tok_body, {'content': '<|turn>user', 'with_pieces': True})

        tpl_body = vis0.build_apply_template_payload()
        self.assertEqual(tpl_body, {'messages': [{'role': 'user', 'content': 'Hi'}]})

        old_body = vis0.build_old_completion_payload('<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n')
        self.assertEqual(old_body['temperature'], 0)
        self.assertEqual(old_body['top_k'], 1)
        self.assertEqual(old_body['seed'], 0)
        self.assertEqual(old_body['n_predict'], 96)
        self.assertIs(old_body['cache_prompt'], False)
        self.assertEqual(old_body['stop'], ['<end_of_turn>'])

        new_body = vis0.build_new_chat_text_payload('Hello world')
        self.assertEqual(new_body['messages'], [{'role': 'user', 'content': 'Hello world'}])
        self.assertEqual(new_body['temperature'], 0)
        self.assertEqual(new_body['top_k'], 1)
        self.assertEqual(new_body['seed'], 0)
        self.assertEqual(new_body['max_tokens'], 96)
        self.assertIs(new_body['cache_prompt'], False)

        vis_body = vis0.build_vision_chat_payload(b'1234')
        msg = vis_body['messages'][0]
        self.assertEqual(msg['role'], 'user')
        self.assertEqual(len(msg['content']), 2)
        self.assertEqual(msg['content'][0]['type'], 'image_url')
        self.assertTrue(msg['content'][0]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(msg['content'][1], {'type': 'text', 'text': vis0.VISION_PROMPT})
        self.assertEqual(vis_body['temperature'], 0)
        self.assertEqual(vis_body['top_k'], 1)
        self.assertEqual(vis_body['seed'], 0)
        self.assertEqual(vis_body['max_tokens'], 96)
        self.assertIs(vis_body['cache_prompt'], False)

    def test_classify_tokenized_marker_and_verdict(self):
        sample = {
            '<start_of_turn>user': [
                {'id': 236820, 'piece': '<'},
                {'id': 4159, 'piece': 'start'},
                {'id': 236779, 'piece': '_'},
                {'id': 1142, 'piece': 'of'},
                {'id': 236779, 'piece': '_'},
                {'id': 11438, 'piece': 'turn'},
                {'id': 236813, 'piece': '>'},
                {'id': 4186, 'piece': 'user'},
            ],
            '<end_of_turn>': [
                {'id': 236820, 'piece': '<'},
                {'id': 1711, 'piece': 'end'},
                {'id': 236779, 'piece': '_'},
                {'id': 1142, 'piece': 'of'},
                {'id': 236779, 'piece': '_'},
                {'id': 11438, 'piece': 'turn'},
                {'id': 236813, 'piece': '>'},
            ],
            '<|turn>user': [
                {'id': 105, 'piece': '<|turn>'},
                {'id': 2364, 'piece': 'user'},
            ],
            '<turn|>': [
                {'id': 106, 'piece': '<turn|>'},
            ],
            '<|turn>model': [
                {'id': 105, 'piece': '<|turn>'},
                {'id': 4368, 'piece': 'model'},
            ],
        }
        checks = [vis0.classify_tokenized_marker(k, v) for k, v in sample.items()]
        by_in = {c['input']: c for c in checks}
        self.assertFalse(by_in['<start_of_turn>user']['tag_is_single_special_token'])
        self.assertFalse(by_in['<end_of_turn>']['tag_is_single_special_token'])
        self.assertTrue(by_in['<|turn>user']['tag_is_single_special_token'])
        self.assertTrue(by_in['<turn|>']['whole_input_is_single_token'])
        self.assertTrue(by_in['<|turn>model']['tag_is_single_special_token'])

        verdict = vis0.evaluate_template_verdict(
            chat_template="{{ '<|turn>user\\n' + message['content'] + '<turn|>\\n<|turn>model\\n' }}",
            tokenize_results=checks,
            rendered_hi='<|bos|><|turn>user\nHi<turn|>\n<|turn>model\n<|thought\n<|channel|>',
        )
        self.assertEqual(verdict, 'CONFIRMED')

    def test_parse_server_log_delta(self):
        log_slice = (
            '0.08.100.000 D slot encode_mtmd_: id  0 | task 0 | encoding mtmd batch from idx = 7, n_chunks = 1\n'
            '0.10.600.000 I decoding image batch 1/1, n_tokens_batch = 154\n'
        )
        parsed = vis0.parse_server_log_delta(log_slice)
        self.assertEqual(parsed['image_tokens_logged'], 154)
        self.assertEqual(parsed['image_tokens_batches'], [154])
        self.assertAlmostEqual(parsed['image_encode_ms_not_a_measurement'], 2500.0)

    def test_vision_crash_restarts_at_most_once_and_stops_on_two_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = []
            for i in range(3):
                p = Path(tmp) / f'f{i}.jpg'
                Image.new('RGB', (32, 32), 'blue').save(p, format='JPEG')
                frames.append({'frame_stem': f'f{i}', 'photo': f'f{i}.jpg', 'frame_path': str(p), 'yolo_lines': ''})

            created = []

            def factory(budget, label):
                srv = FakeServer(budget=budget, label=label, alive_sequence=[False], log_text='SIGSEGV\n')
                created.append(srv)
                return srv

            def crashing_post(path, payload):
                raise ConnectionResetError('connection reset by peer')

            res = vis0.run_vision_budget_pass(
                budget=280,
                target_frames=frames,
                server_factory=factory,
                post_fn=crashing_post,
                log_fn=lambda _: None,
            )
            self.assertEqual(res['crashes'], 2)
            self.assertTrue(res['stopped_early'])
            self.assertEqual(res['stop_reason'], 'two_consecutive_crashes_at_budget')
            self.assertEqual(len(res['calls']), 2)
            self.assertEqual(len(created), 2)

    def test_build_vis0_sheet_pdf_under_20mb(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / 'frame0.jpg'
            Image.new('RGB', (640, 480), 'gray').save(img_path, format='JPEG')
            frames = [
                {
                    'requested_stem': 'bench_photos__blocked_1',
                    'frame_stem': 'bench_photos__blocked_1',
                    'match_note': None,
                    'photo': 'bench_photos/blocked_1.jpg',
                    'frame_path': str(img_path),
                    'frame_sha256': '0' * 64,
                    'frame_bytes': img_path.stat().st_size,
                    'frame_size': [640, 480],
                    'detection_count': 1,
                    'yolo_lines': '- chair [0], ahead, 1.0m',
                }
            ]
            results_data = {
                'template_check': {
                    'verdict': 'CONFIRMED',
                    'verdict_statement': 'CONFIRMED: Gemma 3 markers are plain text.',
                    'chat_template': "<|turn>user\n...\n<turn|>\n<|turn>model\n",
                    'tokenize_checks': [
                        vis0.classify_tokenized_marker('<turn|>', [{'id': 106, 'piece': '<turn|>'}]),
                    ],
                    'apply_template_hi': {
                        'messages': vis0.APPLY_TEMPLATE_MESSAGES,
                        'rendered_prompt': '<|bos|><|turn>user\nHi<turn|>\n<|turn>model\n',
                    },
                    'text_ab': [
                        {
                            'frame_stem': 'bench_photos__blocked_1',
                            'photo': 'bench_photos/blocked_1.jpg',
                            'yolo_lines': '- chair [0], ahead, 1.0m',
                            'old_completion': {'reply': '<|turn>thought', 'raw_content': '<|turn>thought', 'tokens_evaluated': 60},
                            'new_chat_completions': {'reply': 'A chair is ahead.', 'prompt_tokens': 54},
                        }
                    ],
                },
                'vision_budgets': {
                    str(b): {
                        'crashes': 0,
                        'calls': [
                            {
                                'frame_stem': 'bench_photos__blocked_1',
                                'status': 'ok',
                                'reply': f'Reply at budget {b}.',
                                'prompt_tokens': b + 30,
                                'image_tokens_logged': b,
                            }
                        ],
                    }
                    for b in vis0.VISION_BUDGETS
                },
                'optional_140_yolo': None,
            }
            pdf_path = Path(tmp) / 'vis0_sheet.pdf'
            vis0.build_vis0_sheet_pdf(results_data, frames, pdf_path)
            self.assertTrue(pdf_path.is_file())
            self.assertLess(pdf_path.stat().st_size, 20 * 1024 * 1024)

    def test_resolve_and_validate_out_dir_enforces_boundary_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / 'runs'
            valid_dir = vis0.resolve_and_validate_out_dir(runs_root / '20261018T000000Z', runs_root=runs_root)
            self.assertEqual(valid_dir, (runs_root / '20261018T000000Z').resolve())

            with self.assertRaises(ValueError):
                vis0.resolve_and_validate_out_dir(Path(tmp) / 'outside', runs_root=runs_root)

            with self.assertRaises(ValueError):
                vis0.resolve_and_validate_out_dir(runs_root, runs_root=runs_root)

            valid_dir.mkdir(parents=True, exist_ok=True)
            (valid_dir / 'results.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                vis0.resolve_and_validate_out_dir(valid_dir, runs_root=runs_root)

    def test_http_error_on_live_server_is_not_counted_as_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = []
            for i in range(3):
                p = Path(tmp) / f'f{i}.jpg'
                Image.new('RGB', (32, 32), 'green').save(p, format='JPEG')
                frames.append({'frame_stem': f'f{i}', 'photo': f'f{i}.jpg', 'frame_path': str(p), 'yolo_lines': ''})

            created = []

            def factory(budget, label):
                srv = FakeServer(budget=budget, label=label, alive_sequence=[True], log_text='')
                created.append(srv)
                return srv

            call_idx = {'n': 0}

            def flaky_http_post(path, payload):
                call_idx['n'] += 1
                if call_idx['n'] <= 2:
                    raise urllib.error.HTTPError(
                        url='http://127.0.0.1:8090/chat/completions',
                        code=400,
                        msg='Bad Request',
                        hdrs=None,
                        fp=None,
                    )
                return {
                    'choices': [{'message': {'content': 'Recovered on frame 2.'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 96, 'completion_tokens': 5},
                    'timings': {'prompt_ms': 100.0, 'predicted_ms': 50.0, 'predicted_per_second': 10.0},
                }

            res = vis0.run_vision_budget_pass(
                budget=70,
                target_frames=frames,
                server_factory=factory,
                post_fn=flaky_http_post,
                log_fn=lambda _: None,
            )
            self.assertEqual(res['crashes'], 0)
            self.assertFalse(res['stopped_early'])
            self.assertEqual(len(created), 1)
            self.assertEqual([c['status'] for c in res['calls']], ['error', 'error', 'ok'])
            self.assertEqual([c['crashed'] for c in res['calls']], [False, False, False])

    def test_load_target_frames_validates_640px_dimensions_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp) / 'frames'
            frames_dir.mkdir()
            stem = 'bench_photos__blocked_1'
            img_path = frames_dir / f'{stem}.jpg'
            Image.new('RGB', (480, 640), 'white').save(img_path, format='JPEG')
            yolo_lines = '- chair [0], center, close'
            user_text = f'{vis0.TEXT_INSTRUCTION}\n\n{yolo_lines}'
            old_prompt = vis0.build_prompt('gemma', [{'role': 'user', 'content': user_text}])
            results_path = Path(tmp) / 'results.json'
            results_path.write_text(
                json.dumps({
                    'photos': [
                        {
                            'photo': 'bench_photos/blocked_1.jpg',
                            'frame_file': f'frames/{stem}.jpg',
                            'frame_size': [480, 640],
                            'detection_count': 1,
                            'm2_detections': [{'class_name': 'chair', 'position': 'center', 'distance': 'close'}],
                            'gemma_prompts': {'Y-ONLY': old_prompt},
                            'replies': {'X': 'a', 'Y': 'b'},
                        }
                    ]
                }),
                encoding='utf-8',
            )
            loaded = vis0.load_target_frames(
                results_json_path=results_path,
                frames_dir=frames_dir,
                stems=(stem,),
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]['frame_size'], [480, 640])

            with self.assertRaises(RuntimeError):
                vis0.load_target_frames(
                    results_json_path=results_path,
                    frames_dir=frames_dir,
                    stems=('bench_photos__missing_frame',),
                )

            Image.new('RGB', (320, 240), 'white').save(img_path, format='JPEG')
            with self.assertRaises(RuntimeError):
                vis0.load_target_frames(
                    results_json_path=results_path,
                    frames_dir=frames_dir,
                    stems=(stem,),
                )

    def test_format_server_log_banner_includes_timing_and_report_labels(self):
        banner = vis0.format_server_log_banner(
            '2026-10-18T00:00:00Z',
            'vision_budget_70',
            70,
            ['timeout', '3600', 'llama-server'],
        )
        self.assertIn(vis0.TIMING_LABEL, banner)
        self.assertIn(vis0.REPORT_LABELS, banner)

    def test_check_port_8080_stopped_rejects_http_503_and_accepts_refused(self):
        with unittest.mock.patch('urllib.request.urlopen') as mock_open, \
                unittest.mock.patch.object(vis0, 'check_port_free') as mock_free:
            mock_open.side_effect = urllib.error.HTTPError(
                url='http://127.0.0.1:8080/health',
                code=503,
                msg='Service Unavailable',
                hdrs=None,
                fp=io.BytesIO(b'{"error":"loading model"}'),
            )
            with self.assertRaises(RuntimeError) as ctx503:
                vis0.check_port_8080_stopped(8080)
            self.assertIn('503', str(ctx503.exception))
            mock_free.assert_not_called()

            mock_open.side_effect = urllib.error.URLError( TimeoutError('timed out') )
            with self.assertRaises(RuntimeError) as ctxtmo:
                vis0.check_port_8080_stopped(8080)
            self.assertIn('unexpected URLError', str(ctxtmo.exception))
            mock_free.assert_not_called()

            mock_open.side_effect = urllib.error.URLError(ConnectionRefusedError(111, 'Connection refused'))
            self.assertTrue(vis0.check_port_8080_stopped(8080))
            mock_free.assert_called_once_with(8080)


if __name__ == '__main__':
    unittest.main()

