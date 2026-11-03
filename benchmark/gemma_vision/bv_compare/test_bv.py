"""Offline tests for bv.py with a fake server: Y rules (room priority, largest person box, one-box photos), the
detector line, parsing incl. UNPARSED, metrics, the PASS rule, key custody and the run path. No server, no phone."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bv  # noqa: E402

bv.PAUSE_S = 0
FRAMES = sorted(p.name for p in bv.FRAMES_DIR.glob('*.jpg'))
PH = bv.photos(json.loads(bv.RESULTS.read_text()), FRAMES)


def det(cls, box=(0, 0, 10, 10), dist='far', pos='center'):
    return {'class_name': cls, 'box_xyxy': list(box), 'distance': dist, 'position': pos}


def key_from(pred_fn):
    return {i: {'frame': FRAMES[i - 1], **pred_fn(i)} for i in bv.IDS}


class FakeServer:
    def __init__(self, reply):
        self.reply, self.bodies = reply, []

    def __call__(self, path, body):
        assert path == '/v1/chat/completions'
        self.bodies.append(body)
        return {'choices': [{'message': {'content': self.reply(body)}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100}}


class T(unittest.TestCase):
    def test_room_priority(self):
        y = lambda *c: bv.y_pred({'detections': [det(x) for x in c]})['room']  # noqa: E731
        self.assertEqual(y('couch', 'toilet', 'bed'), 'bathroom')
        self.assertEqual(y('couch', 'oven', 'bed'), 'bedroom')
        self.assertEqual(y('tv', 'microwave'), 'kitchen')
        self.assertEqual(y('dining table', 'cup'), 'living room')
        self.assertEqual(y('cup', 'chair'), 'unknown')
        self.assertEqual(bv.y_pred({'detections': None, 'classes': ['toilet']})['room'], 'bathroom')

    def test_person_largest_box(self):
        p = {'detections': [det('person', (0, 0, 10, 10), 'far'), det('person', (0, 0, 100, 100), 'very close'),
                            det('couch', (0, 0, 500, 500), 'medium')]}
        self.assertEqual(bv.y_pred(p), {'room': 'living room', 'person': 'yes', 'distance': 'close'})
        p['detections'][1]['distance'] = 'medium'
        self.assertEqual(bv.y_pred(p)['distance'], 'medium')
        self.assertEqual(bv.y_pred({'detections': [det('cup')]}), {'room': 'unknown', 'person': 'no', 'distance': 'none'})
        one = {'detections': None, 'classes': ['person']}
        self.assertEqual(bv.y_pred(one), {'room': 'unknown', 'person': 'yes', 'distance': bv.UNAVAILABLE})

    def test_real_photos(self):
        self.assertEqual(bv.y_pred(PH[10])['distance'], bv.UNAVAILABLE)  # one-box person photo
        self.assertEqual(bv.y_pred(PH[33])['room'], 'bathroom')
        self.assertEqual(bv.detector_line(PH[10]), "Objects the robot's detector found: person (position unknown, "
                                                   "distance unknown)")
        self.assertTrue(bv.detector_line(PH[1]).startswith("Objects the robot's detector found: cup (center, far), "
                                                           "chair (right, medium), "))
        self.assertEqual(bv.detector_line({'detections': [], 'classes': []}), "Objects the robot's detector found: none")

    def test_prompt(self):
        v, vy = bv.prompt('V70', PH[1]).splitlines(), bv.prompt('VY70', PH[1]).splitlines()
        self.assertEqual(len(v), 6)
        self.assertEqual(vy[:1] + vy[2:], v)
        self.assertTrue(vy[1].startswith("Objects the robot's detector found: ") and vy[2].startswith('Answer'))
        b = bv.body('V140', PH[1], 'AAAA')
        self.assertEqual((b['temperature'], b['top_k'], b['seed'], b['max_tokens'], b['chat_template_kwargs']),
                         (0, 1, 0, 32, {'enable_thinking': False}))
        self.assertEqual(b['messages'][0]['content'][0]['image_url']['url'], 'data:image/jpeg;base64,AAAA')

    def test_command(self):
        for t in (70, 140):
            c = bv.command(t)
            base = bv.mem1.command(2048)
            self.assertEqual(c[:len(base)], [bv.bc.SERVER if x == bv.mem1.SERVER else x for x in base])
            self.assertEqual(c[len(base):], ['--mmproj', bv.vis3d.MMPROJ, '--image-min-tokens', str(t),
                                             '--image-max-tokens', str(t)])
            self.assertNotIn('--slot-save-path', c)

    def test_parse(self):
        ok = {'room': 'living room', 'person': 'yes', 'distance': 'close'}
        self.assertEqual(bv.parse('Room: Living Room\nPerson: Yes\nDistance: close'), ok)
        self.assertEqual(bv.parse('**Room:** living room.\n**Person:** yes\nDistance: close (under 1 m)'), ok)
        self.assertEqual(bv.parse('Room: lounge\nPerson: maybe\n'),
                         {'room': 'UNPARSED', 'person': 'UNPARSED', 'distance': 'UNPARSED'})
        self.assertEqual(bv.parse('Room: kitchen\nRoom: bedroom\nPerson: no\nDistance: none')['room'], 'UNPARSED')
        self.assertEqual(bv.parse('Room: close (under 1 m)')['room'], 'UNPARSED')
        self.assertEqual(bv.parse(None)['person'], 'UNPARSED')

    def test_metrics(self):
        key = key_from(lambda i: {'room': 'kitchen', 'person': 'yes' if i <= 10 else 'no',
                                  'distance': 'far' if i <= 10 else 'none'})
        pred = {i: dict(key[i]) for i in bv.IDS}
        pred[1]['person'] = 'no'              # person wrong -> distance wrong too
        pred[2]['distance'] = 'close'         # bucket wrong
        pred[3]['room'] = bv.UNPARSED
        pred[20]['distance'] = 'far'          # distance on a no-person photo: not in distance accuracy, not all3
        m = bv.metrics(pred, key)
        self.assertEqual(m['n'], {'room': 61, 'person': 61, 'distance': 10})
        self.assertEqual(m['correct'], {'room': 60, 'person': 60, 'distance': 8})
        self.assertEqual(m['wrong'], {'room': [3], 'person': [1], 'distance': [1, 2]})
        self.assertEqual(m['all3'], 61 - 4)
        self.assertEqual(m['unparsed']['room'], 1)
        self.assertEqual(m['confusion']['room'][('kitchen', bv.UNPARSED)], 1)

    def test_verdict(self):
        def mets(room, person):
            return {'acc': {'room': room, 'person': person, 'distance': 0}}
        y = mets(0.5, 0.9)
        self.assertEqual(bv.verdict({'Y': y, 'V70': mets(0.85, 0.9), 'V140': mets(0.7, 1), 'VY70': mets(0.8, 1)})[0],
                         'PASS')
        self.assertEqual(bv.verdict({'Y': y, 'V70': mets(0.85, 0.8), 'V140': mets(0.7, 1), 'VY70': mets(0.8, 1)})[0],
                         'FAIL')  # the best room method loses on person
        self.assertEqual(bv.verdict({'Y': y, 'V70': mets(0.79, 1), 'V140': mets(0.7, 1), 'VY70': mets(0.6, 1)})[0],
                         'FAIL')
        self.assertEqual(bv.verdict({'Y': y, 'V70': mets(0.9, 0.8), 'V140': mets(0.9, 0.95), 'VY70': mets(0.6, 1)})[0],
                         'PASS')  # tie on room: one of the tied methods meets both

    def test_custody(self):
        with self.assertRaises(SystemExit):
            bv.load_key(Path(__file__))  # inside the repo: refused before reading
        with tempfile.TemporaryDirectory() as d:
            k = Path(d) / 'k.json'
            k.write_text(json.dumps({str(i): {} for i in bv.IDS}))
            with self.assertRaises(SystemExit):
                bv.load_key(k)  # hash differs from the pin
            self.assertEqual(sorted(bv.load_key(k, bv.sha(k.read_bytes()))), bv.IDS)

    def test_frames(self):
        key = key_from(lambda i: {})
        self.assertEqual(bv.frame_problems(PH, key), [])
        key[5]['frame'] = FRAMES[61]
        self.assertEqual(len(bv.frame_problems(PH, key)), 1)

    def test_run_path(self):
        key = key_from(lambda i: {'room': 'living room', 'person': 'yes' if i in (10, 11) else 'no',
                                  'distance': 'far' if i in (10, 11) else 'none'})
        url = {'data:image/jpeg;base64,' + bv.base64.b64encode((bv.FRAMES_DIR / FRAMES[i - 1]).read_bytes()).decode(): i
               for i in bv.IDS}

        def reply(b):  # VY70 answers as the key; V70/V140 omit the distance line
            k = key[url[b['messages'][0]['content'][0]['image_url']['url']]]
            if 'detector' in b['messages'][0]['content'][1]['text']:
                return f'Room: {k["room"]}\nPerson: {k["person"]}\nDistance: {k["distance"]}'
            return 'Room: kitchen\nPerson: no'
        fake = FakeServer(reply)
        rows = []
        for launch, _, methods in bv.LAUNCHES:
            for j, m in enumerate(methods):
                rows += bv.run_method(fake, launch, m, PH, 123, lambda r: None, first_block=j == 0)
        self.assertEqual(len(fake.bodies), 183)
        self.assertEqual([x['block'] for x in rows[:62:30]], [0, 0, 1])  # 31 + 30 per method
        with tempfile.TemporaryDirectory() as d:
            run = {'utc': 'T', 'code_sha256': 'c' * 64, 'git_head': 'h', 'key': {'path': 'k', 'sha256': 's'},
                   'pins': {}, 'detector': {}, 'commands': {'A': ['x'], 'B': ['y']},
                   'exits': {'A': {'exit_code': 0, 'killed': False}, 'B': {'exit_code': -11, 'killed': False}}}
            text = bv.finish(Path(d), run, rows, PH, key)
            out = json.loads((Path(d) / 'run.json').read_text())
            self.assertEqual(out['validity'], 'VALID')
            self.assertEqual(out['metrics']['VY70']['all3'], 61)
            self.assertEqual(out['metrics']['V70']['unparsed']['distance'], 61)
            self.assertEqual(out['verdict'][0], 'PASS')
            self.assertIn('FINDING (#145)', text)
            self.assertEqual(len((Path(d) / 'artifacts.sha256').read_text().splitlines()), 2)
            rows[5]['error'] = 'x'
            self.assertEqual(bv.validity(run, rows[:-1])[0], 'INVALID')


if __name__ == '__main__':
    unittest.main()
