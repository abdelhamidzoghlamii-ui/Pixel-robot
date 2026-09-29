import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from detector_size_policy import SizePolicy
from detect_person import Detector, detect_scene, get_detector, scene_from_detections
from robotcam_reader import FRAME_DIR, read_frame
import main


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, 'frame.jpg')

    def frame(self, boot_ms=100000, comment=None):
        if comment is None:
            comment = (f'robotcam session=abcdef frame=1 capture_boot_ms={boot_ms} '
                       'capture_wall_ms=123 clock=sensor').encode()
        Image.new('RGB', (640, 480)).save(self.path, comment=comment)

    def read(self, session=None, min_capture_boot_s=None):
        with patch('robotcam_reader.time.clock_gettime', return_value=101):
            return read_frame(self.tmp.name, session,
                              min_capture_boot_s=min_capture_boot_s)

    def test_reader_statuses(self):
        self.assertEqual(self.read()['status'], 'missing')
        self.frame()
        good = self.read()
        self.assertEqual((good['status'], good['session'], good['image'].size),
                         ('ok', 'abcdef', (640, 480)))
        self.assertEqual(self.read('123456')['status'], 'other_session')
        self.assertEqual(self.read(min_capture_boot_s=100)['status'], 'missing')
        self.assertEqual(self.read(min_capture_boot_s=99.999)['status'], 'ok')
        self.frame(98000)
        self.assertEqual(self.read()['status'], 'missing')
        self.frame(101201)
        self.assertEqual(self.read()['status'], 'bad')
        self.frame(comment=b'wrong')
        self.assertEqual(self.read()['status'], 'bad')
        self.frame(comment=b'')
        self.assertEqual(self.read()['status'], 'bad')
        with open(self.path, 'wb') as f: f.write(b'not a jpeg')
        self.assertEqual(self.read()['status'], 'bad')

    def test_root_independent_path(self):
        self.assertEqual(FRAME_DIR, '/storage/emulated/0/Download/robotcam')
        self.assertNotIn('~', FRAME_DIR)


class SizePolicyTest(unittest.TestCase):
    def test_drive_and_search(self):
        p = SizePolicy(interval_s=5, person_height_fraction=1/3)
        p.last_large = 0
        self.assertEqual([p.next_size(t) for t in (0, 4.9, 5, 5.1, 10)],
                         [320, 320, 640, 320, 640])
        p.set_mode('person_search')
        self.assertEqual(p.next_size(11), 640)
        p.observe([{'class_name': 'person', 'box_xyxy': (0, 0, 20, 159)}], 480)
        self.assertEqual(p.mode, 'person_search')
        p.observe([{'class_name': 'person', 'box_xyxy': (0, 0, 20, 160)}], 480)
        self.assertEqual(p.mode, 'drive')
        self.assertEqual(p.next_size(), 320)


class DetectorTest(unittest.TestCase):
    def test_legacy_detect_scene_defaults_to_640(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'photo.jpg')
            Image.new('RGB', (640, 480)).save(path)
            detector = SimpleNamespace(detect=Mock(return_value=[]))
            with patch('detect_person.get_detector', return_value=detector):
                self.assertEqual(detect_scene(path), [])
            self.assertEqual(detector.detect.call_args.args[1], 640)

    def test_detection_dict_and_legacy_view(self):
        class Session:
            def get_inputs(self):
                return [type('Input', (), {'name': 'images'})()]

            def run(self, unused, inputs):
                size = inputs['images'].shape[-1]
                assert inputs['images'].shape == (1, 3, size, size)
                out = np.zeros((1, 84, 1), dtype=np.float32)
                out[0, :4, 0] = [size/2, size/2, size/2, size/2]
                out[0, 4, 0] = 0.8
                return [out]

        with patch('detect_person.ort.InferenceSession', return_value=Session()) as load:
            detector = Detector()
        self.assertEqual(load.call_count, 2)
        found = detector.detect(Image.new('RGB', (640, 480)), 320)
        self.assertEqual(len(found), 1)
        d = found[0]
        self.assertEqual(set(d), {'class_name', 'class_id', 'conf', 'box_xyxy', 'size_used'})
        self.assertEqual((d['class_name'], d['class_id'], d['box_xyxy'], d['size_used']),
                         ('person', 0, (160, 120, 480, 360), 320))
        self.assertEqual(detector.detect(Image.new('RGB', (640, 480)), 640)[0]['size_used'], 640)
        self.assertEqual(scene_from_detections(found, 640, 480)[0],
                         ('person', 0.8, 'center', 'close', 320, 320, 320, 320))

    def test_shared_sessions_loaded_once(self):
        with patch('detect_person._detector', None), patch(
                'detect_person.ort.InferenceSession') as load:
            self.assertIs(get_detector(), get_detector())
            self.assertEqual(load.call_count, 2)


class PhotoRollbackTest(unittest.TestCase):
    def test_failed_capture_cannot_reuse_previous_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'photo.jpg')
            Image.new('RGB', (640, 480)).save(path)
            with patch('main.subprocess.run', return_value=SimpleNamespace(returncode=1)), patch(
                    'main.time.sleep'):
                self.assertFalse(main.take_photo(path))
            self.assertFalse(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
