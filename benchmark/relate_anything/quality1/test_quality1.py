"""Offline unit tests for benchmark/relate_anything/quality1/quality1.py.

Uses fakes only: no network requests, no ONNX sessions, no GGUF models loaded.
"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parents[2]
for p in (str(HERE), str(ROBOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from PIL import Image, ImageDraw

import quality1
import relate


class FakeHead:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows if rows is not None else [
            {
                'subject_idx': 0,
                'subject': 'person',
                'predicate': 'near',
                'object_idx': 1,
                'object': 'chair',
                'score': 0.82,
                'threshold': 0.40,
                'pass': True,
            },
            {
                'subject_idx': 1,
                'subject': 'chair',
                'predicate': 'on',
                'object_idx': 0,
                'object': 'person',
                'score': 0.21,
                'threshold': 0.45,
                'pass': False,
            },
        ]

    def infer(self, image, detections):
        self.calls.append((image.size, list(detections)))
        return [dict(r) for r in self.rows]


class Quality1Test(unittest.TestCase):
    def test_xy_assignment_reproducible_and_order_independent(self):
        photos_a = ['test_photos/b.jpg', 'bench_photos/a.jpg', 'bench_photos/c.jpg', 'test_photos/d.jpg']
        photos_b = list(reversed(photos_a))
        key1 = quality1.assign_blind_key(photos_a)
        key2 = quality1.assign_blind_key(photos_b)
        self.assertEqual(key1, key2)
        self.assertEqual(list(key1.keys()), sorted(photos_a))
        for mapping in key1.values():
            self.assertEqual(set(mapping.keys()), {'X', 'Y'})
            self.assertEqual(set(mapping.values()), {'Y-ONLY', 'Y+M2'})

    def test_too_few_handling(self):
        img = Image.new('RGB', (640, 480), 'gray')
        for count in (0, 1):
            head = FakeHead()
            dets = [
                {'class_name': 'person', 'conf': 0.9, 'box_xyxy': [10, 10, 100, 200],
                 'position': 'left', 'distance': 'close'}
            ][:count]
            res = quality1.run_m2_once(head, img, dets)
            self.assertFalse(res['ran'])
            self.assertEqual(head.calls, [])
            self.assertEqual(res['boxes_used'], count)
            self.assertEqual(res['rows'], [])
            self.assertEqual(res['shown_rows'], [])
            self.assertEqual(res['shown_text'], relate.TOO_FEW)

    def test_2_to_32_and_over_32_rule_matches_relate(self):
        img = Image.new('RGB', (640, 480), 'gray')
        head = FakeHead()
        # 40 detections with mixed confidence values; must keep top 32 in original detector order.
        confs = [(i * 17) % 40 / 40.0 + 0.01 for i in range(40)]
        dets = [
            {
                'id': i,
                'class_name': f'obj_{i}',
                'conf': confs[i],
                'box_xyxy': [float(i), float(i), float(i + 20), float(i + 20)],
                'position': 'center',
                'distance': 'far',
            }
            for i in range(40)
        ]
        res = quality1.run_m2_once(head, img, dets)
        self.assertTrue(res['ran'])
        self.assertEqual(res['boxes_used'], 32)
        self.assertEqual(len(head.calls), 1)
        passed_dets = head.calls[0][1]
        self.assertEqual(len(passed_dets), 32)
        # Check detector order is preserved and matches relate.py's exact formula
        expected_keep = sorted(sorted(range(len(dets)), key=lambda i: -dets[i]['conf'])[:relate.MAX_BOXES])
        self.assertEqual([d['id'] for d in passed_dets], expected_keep)
        self.assertEqual(res['shown_rows'], relate.chosen(res['rows']))
        self.assertEqual(res['shown_text'], relate.context(res['rows']))

    def test_prompts_call_gemma_and_strip_tokens(self):
        used = [
            {'class_name': 'person', 'conf': 0.91, 'box_xyxy': [0, 0, 50, 50], 'position': 'left', 'distance': 'close'},
            {'class_name': 'chair', 'conf': 0.65, 'box_xyxy': [60, 60, 120, 120], 'position': 'right', 'distance': 'medium'},
        ]
        m2_text = 'M2 relations:\n- person [0] near chair [1].'
        prompts = quality1.build_variant_prompts(used, m2_text)
        self.assertIn(quality1.INSTRUCTION, prompts['Y-ONLY'])
        self.assertIn('- person [0], left, close\n- chair [1], right, medium', prompts['Y-ONLY'])
        self.assertNotIn('M2 relations:', prompts['Y-ONLY'])
        self.assertIn(m2_text, prompts['Y+M2'])
        self.assertTrue(prompts['Y-ONLY'].startswith('<start_of_turn>user\n'))
        self.assertTrue(prompts['Y-ONLY'].endswith('<end_of_turn>\n<start_of_turn>model\n'))

        captured = []
        def fake_post(path, body):
            captured.append((path, body))
            return {'content': ' A person is on the left near a chair.<start_of_of_turn></start_of_turn><end_of_turn><start_of_turn'}

        reply, _ = quality1.call_gemma(prompts['Y+M2'], post=fake_post)
        self.assertEqual(reply, 'A person is on the left near a chair.')
        self.assertEqual(
            quality1.strip_template_tokens('The robot is in a room.\n<start_of_of_turn>laptop [0], center, close\n<'),
            'The robot is in a room.\nlaptop [0], center, close',
        )
        self.assertEqual(len(captured), 1)
        path, payload = captured[0]
        self.assertEqual(path, '/completion')
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['top_k'], 1)
        self.assertEqual(payload['n_predict'], 96)
        self.assertIs(payload['cache_prompt'], False)
        self.assertEqual(payload['seed'], 0)

    def test_pdf_builders_and_key_never_leaked_into_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            frame1 = Image.new('RGB', (640, 480), (200, 210, 220))
            frame2 = Image.new('RGB', (640, 480), (180, 190, 200))
            frames = {
                'bench_photos/p1.jpg': frame1,
                'test_photos/p2.jpg': frame2,
            }
            records = [
                {
                    'photo': 'bench_photos/p1.jpg',
                    'frame_size': [640, 480],
                    'detection_count': 2,
                    'detections': [
                        {'class_name': 'person', 'conf': 0.9, 'box_xyxy': [20, 30, 200, 400],
                         'position': 'left', 'distance': 'close'},
                        {'class_name': 'couch', 'conf': 0.8, 'box_xyxy': [220, 100, 580, 420],
                         'position': 'right', 'distance': 'close'},
                    ],
                    'm2_ran': True,
                    'm2_boxes_used': 2,
                    'm2_raw_top10': [
                        {'subject_idx': 0, 'subject': 'person', 'predicate': 'sitting_on',
                         'object_idx': 1, 'object': 'couch', 'score': 0.75, 'threshold': 0.35, 'pass': True}
                    ],
                    'm2_shown_rows': [
                        {'subject_idx': 0, 'subject': 'person', 'predicate': 'sitting_on',
                         'object_idx': 1, 'object': 'couch', 'score': 0.75, 'threshold': 0.35, 'pass': True}
                    ],
                    'm2_shown_text': 'M2 relations:\n- person [0] sitting on couch [1].',
                    'm2_identical_twice': True,
                    'replies': {
                        'X': 'A person is on the left and a couch is on the right.',
                        'Y': 'A person is sitting on a couch in front of the robot.',
                    },
                    'replies_identical_twice': {'X': True, 'Y': True},
                },
                {
                    'photo': 'test_photos/p2.jpg',
                    'frame_size': [640, 480],
                    'detection_count': 1,
                    'classes': ['chair'],
                    'm2_ran': False,
                    'm2_shown_text': relate.TOO_FEW,
                },
            ]
            m2_pdf = tmp_dir / 'sheet_m2.pdf'
            ab_pdf = tmp_dir / 'sheet_ab.pdf'
            m2_drawn_texts = []
            ab_drawn_texts = []
            orig_text = ImageDraw.ImageDraw.text

            def capture_m2_text(draw_self, xy, text, *args, **kwargs):
                m2_drawn_texts.append(str(text))
                return orig_text(draw_self, xy, text, *args, **kwargs)

            def capture_ab_text(draw_self, xy, text, *args, **kwargs):
                ab_drawn_texts.append(str(text))
                return orig_text(draw_self, xy, text, *args, **kwargs)

            with mock.patch.object(ImageDraw.ImageDraw, 'text', capture_m2_text):
                quality1.build_sheet_m2_pdf(records, frames, m2_pdf)
            with mock.patch.object(ImageDraw.ImageDraw, 'text', capture_ab_text):
                quality1.build_sheet_ab_pdf(records, frames, ab_pdf)

            self.assertTrue(m2_pdf.is_file())
            self.assertTrue(ab_pdf.is_file())
            m2_bytes = m2_pdf.read_bytes()
            ab_bytes = ab_pdf.read_bytes()
            self.assertTrue(m2_bytes.startswith(b'%PDF-'))
            self.assertTrue(ab_bytes.startswith(b'%PDF-'))
            self.assertLess(len(m2_bytes), 20 * 1024 * 1024)
            self.assertLess(len(ab_bytes), 20 * 1024 * 1024)
            self.assertIn(b'/Count 2', m2_bytes)
            self.assertIn(b'/Count 1', ab_bytes)

            m2_joined = '\n'.join(m2_drawn_texts)
            ab_joined = '\n'.join(ab_drawn_texts)
            for forbidden in ('Y-ONLY', 'Y+M2', 'key.json'):
                self.assertNotIn(forbidden, m2_joined)
                self.assertNotIn(forbidden, ab_joined)
                self.assertNotIn(forbidden.encode(), m2_bytes)
                self.assertNotIn(forbidden.encode(), ab_bytes)

            # Verify sheet_ab drawn text contains ONLY header, numbered box labels, Reply X/Y headers, and replies.
            self.assertEqual(
                ab_drawn_texts,
                [
                    'bench_photos/p1.jpg  --  Page 1 / 1',
                    '0: person',
                    '1: couch',
                    'Reply X:',
                    'A person is on the left and a couch is on the right.',
                    'Reply Y:',
                    'A person is sitting on a couch in front of the robot.',
                ],
            )

            # Verify build_sheet_ab_pdf refuses any leaked variant name in replies.
            bad_records = [dict(records[0], replies={'X': 'Variant Y-ONLY reply', 'Y': 'ok'})]
            with self.assertRaises(RuntimeError):
                quality1.build_sheet_ab_pdf(bad_records, frames, tmp_dir / 'bad.pdf')

    def test_module_origins_and_provenance(self):
        self.assertEqual(Path(sys.path[0]).resolve(), quality1.ROBOT.resolve())
        hashes = quality1.verify_and_hash_imports(quality1.ROBOT)
        self.assertEqual(
            set(hashes.keys()),
            {
                'benchmark/relate_anything/quality1/quality1.py',
                'relate.py',
                'detect_person.py',
                'picker.py',
                'chat.py',
                'server_manager.py',
            },
        )
        with tempfile.TemporaryDirectory() as fake_root:
            with self.assertRaises(RuntimeError):
                quality1.verify_and_hash_imports(fake_root)

    def test_prepare_frame_and_server_check_fakes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            src = tmp_dir / 'src.jpg'
            dst = tmp_dir / 'dst.jpg'
            Image.new('RGB', (1200, 900), (10, 20, 30)).save(src, format='JPEG', quality=95)
            reopened = quality1.prepare_frame(src, dst)
            self.assertEqual(reopened.size, (640, 480))
            self.assertTrue(dst.is_file())

        def ok_get(path):
            if path == '/health':
                return {'status': 'ok'}
            return {
                'model_path': '/data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf',
                'model_ftype': 'Q4_0',
                'total_slots': 1,
                'default_generation_settings': {'n_ctx': 2048},
            }

        info = quality1.check_server(get=ok_get)
        self.assertTrue(info['model_path'].endswith('gemma-4-E2B-it-Q4_0.gguf'))

        def wrong_model_get(path):
            if path == '/health':
                return {'status': 'ok'}
            return {'model_path': '/models/other.gguf'}

        with self.assertRaises(RuntimeError) as ctx:
            quality1.check_server(get=wrong_model_get)
        self.assertIn('python3 server_manager.py setup_q4', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
