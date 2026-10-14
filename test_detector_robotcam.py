import os
import signal
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
from PIL import Image

from detector_size_policy import SizePolicy
import detect_person as dp
from detect_person import Detector, detect_scene, get_detector, scene_from_detections
from robotcam_reader import FRAME_DIR, read_frame
import main
import run_mission
from run_cycle_safety_test import isolated_robot


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


class FakeORT:
    """Fake onnxruntime sessions, /proc task list and os.sched_*: each session adds `workers` fake ORT worker tids.
    Nothing is really pinned."""
    REAL_THREAD_IDS = staticmethod(dp.thread_ids)

    def __init__(self, workers=1, wrong_get=(), wrong_caller=False):
        self.workers, self.wrong_get, self.wrong_caller = workers, set(wrong_get), wrong_caller
        self.tasks, self.next_tid, self.loaded, self.affinity, self.set_calls, self.fail_set = set(), 90000, [], {}, [], set()

    def InferenceSession(self, path, sess_options=None):
        self.loaded.append((path, sess_options))
        for _ in range(self.workers):
            self.tasks.add(self.next_tid)
            self.next_tid += 1
        return SimpleNamespace(get_session_options=lambda: sess_options)

    def thread_ids(self):
        return self.REAL_THREAD_IDS() | self.tasks

    def setaffinity(self, tid, cpus):
        self.set_calls.append((tid, set(cpus)))
        if tid in self.fail_set:
            raise OSError(22, 'Invalid argument')
        self.affinity[tid] = set(cpus)

    def getaffinity(self, tid):
        if tid in self.wrong_get or (self.wrong_caller and tid not in self.tasks):
            return set(range(8))
        return self.affinity.get(tid, set(range(8)))


class PinnedDetectorTest(unittest.TestCase):
    def fake(self, **kw):
        fake = FakeORT(**kw)
        for p in (patch.object(dp, 'thread_ids', fake.thread_ids),
                  patch.object(dp.ort, 'InferenceSession', fake.InferenceSession),
                  patch.object(dp.ort, 'SessionOptions', SimpleNamespace),
                  patch.object(dp.os, 'sched_setaffinity', fake.setaffinity),
                  patch.object(dp.os, 'sched_getaffinity', fake.getaffinity),
                  patch.object(dp, '_detector', None)):
            p.start()
            self.addCleanup(p.stop)
        return fake

    def test_default_detector_and_get_detector_unchanged(self):
        with patch('detect_person.ort.InferenceSession') as load, patch('detect_person._detector', None), patch(
                'detect_person.os.sched_setaffinity') as setaffinity:
            threads = threading.active_count()
            self.assertIs(type(Detector()), Detector)
            self.assertIs(type(get_detector()), Detector)
            self.assertIs(get_detector(), get_detector())
        self.assertEqual(load.call_args_list, [call(dp.MODELS[320]), call(dp.MODELS[640])] * 2)
        setaffinity.assert_not_called()
        self.assertEqual(threading.active_count(), threads)

    def test_threads_options_and_affinity(self):
        fake = self.fake()
        pd = dp.PinnedDetector()
        self.assertEqual([path for path, _ in fake.loaded], [dp.MODELS[320], dp.MODELS[640]])
        for _, opts in fake.loaded:
            self.assertEqual((opts.intra_op_num_threads, opts.inter_op_num_threads), (2, 1))
        self.assertEqual(sorted(fake.tasks), pd.worker_tids)
        self.assertEqual(len(pd.worker_tids), 2)
        main_tid = threading.main_thread().native_id
        self.assertNotIn(pd.caller_tid, (0, os.getpid(), main_tid, threading.get_native_id()))
        self.assertEqual(sorted(tid for tid, _ in fake.set_calls), sorted(pd.worker_tids + [pd.caller_tid]))
        self.assertTrue(all(cpus == {4, 5} for _, cpus in fake.set_calls))
        config = pd.config()
        self.assertEqual(set(config), {'ort_version', 'sessions', 'worker_tids', 'caller_tid', 'requested_cpus',
                                       'affinity'})
        self.assertEqual(config['sessions'], {size: {'intra_op_num_threads': 2, 'inter_op_num_threads': 1}
                                              for size in (320, 640)})
        self.assertEqual((config['requested_cpus'], config['caller_tid']), ([4, 5], pd.caller_tid))
        self.assertEqual(config['affinity'], {tid: [4, 5] for tid in [pd.caller_tid] + pd.worker_tids})

    def test_pin_refuses_process_and_main_thread(self):
        fake = self.fake()
        for tid in (0, os.getpid()):
            with self.assertRaises(RuntimeError):
                dp.pin(tid, {4, 5})
        self.assertEqual(fake.set_calls, [])

    def test_worker_count_mismatch_raises(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                fake = self.fake(workers=workers)
                with self.assertRaises(RuntimeError):
                    dp.PinnedDetector()
                self.assertEqual(fake.set_calls, [])

    def test_affinity_not_taking_effect_raises(self):
        fake = self.fake(wrong_get={90001})
        with self.assertRaises(RuntimeError):
            dp.PinnedDetector()
        fake = self.fake(wrong_caller=True)
        with self.assertRaises(RuntimeError):
            dp.PinnedDetector()

    def test_detect_runs_on_calling_thread_and_errors_propagate(self):
        self.fake()
        pd = dp.PinnedDetector()
        seen = []

        def detect(frame, size):
            seen.append(threading.get_native_id())
            if frame == 'bad':
                raise ValueError('boom')
            return [frame, size]
        pd.detector = SimpleNamespace(detect=detect)
        self.assertEqual(pd.detect('img', 640), ['img', 640])
        with self.assertRaises(ValueError):
            pd.detect('bad', 320)
        self.assertEqual(seen, [pd.caller_tid] * 2)

        R, motors, robot = isolated_robot(distance=400)  # run_cycle turns that exception into STOP
        robot.camera_source, robot.detector = 'robotcam', pd
        R.read_frame = Mock(return_value={'status': 'ok', 'session': 's', 'frame': 1, 'age_s': 0.4,
                                          'image': 'bad'})
        with patch('builtins.print'):
            self.assertEqual(robot.run_cycle(), 'STOP')
        self.assertEqual(motors.actions, [('stop', ())])
        self.assertEqual(seen, [pd.caller_tid] * 3)  # the cycle reached detect on the calling thread

    def test_signal_while_waiting_and_late_answer_never_reaches_a_later_request(self):
        self.fake()
        pd = dp.PinnedDetector()
        release = threading.Event()

        def detect(frame, size):
            if frame == 'slow':
                release.wait(5)
            return frame
        pd.detector = SimpleNamespace(detect=detect)
        old = signal.signal(signal.SIGALRM, run_mission.exit_on_signal)
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.2)
            with self.assertRaises(SystemExit):
                pd.detect('slow', 320)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        release.set()
        self.assertEqual(pd.detect('fresh', 320), 'fresh')

    def test_drift_is_repinned_and_failures_reported(self):
        fake = self.fake()
        pd = dp.PinnedDetector()
        w = pd.worker_tids[0]
        ok = pd.check_affinity()
        self.assertEqual((ok['repinned'], ok['ok'], ok['errors']), ([], True, {}))
        fake.affinity[w] = set(range(8))  # e.g. an Android cpuset change reset it
        r = pd.check_affinity()
        self.assertEqual((r['repinned'], r['ok'], r['before'][w], r['after'][w]), ([w], True, list(range(8)), [4, 5]))
        fake.affinity[w], fake.fail_set = set(range(8)), {w}
        r = pd.check_affinity()
        self.assertEqual((r['repinned'], r['ok']), ([w], False))
        self.assertIn('OSError', r['errors'][w])
        fake.tasks.discard(w)  # a vanished tid is reported and never touched
        calls = len(fake.set_calls)
        r = pd.check_affinity()
        self.assertFalse(r['ok'])
        self.assertIn('gone', r['errors'][w])
        self.assertEqual(len(fake.set_calls), calls)

    def test_robot_process_uses_one_detector(self):
        fake = self.fake()
        pd = get_detector(pinned=True)
        self.assertIs(get_detector(), pd)
        self.assertIs(get_detector(pinned=True), pd)
        pd.detector = SimpleNamespace(detect=Mock(return_value=[]))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'photo.jpg')
            Image.new('RGB', (640, 480)).save(path)
            self.assertEqual(detect_scene(path), [])
        self.assertEqual(pd.detector.detect.call_args.args[1], 640)
        self.assertEqual(len(fake.loaded), 2)  # no second session pair

    def test_pinned_after_a_default_detector_raises(self):
        self.fake()
        get_detector()
        with self.assertRaises(RuntimeError):
            get_detector(pinned=True)


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
