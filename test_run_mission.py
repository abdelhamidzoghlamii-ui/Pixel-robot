"""Mission cleanup tests; all camera, USB and signal operations are faked."""
import signal
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_mission


class MissionExitTest(unittest.TestCase):
    def run_dry(self, failure, signum=None):
        events = []
        handlers = {}

        def register(number, handler):
            handlers[number] = handler

        class Robot:
            def __init__(self, motors, camera_source):
                self.assert_no_motors(motors)
                self.assert_robotcam(camera_source)
                self.size_policy = SimpleNamespace(set_mode=Mock())
                self.known_rooms = {}
                self.last_moves = []

            def assert_no_motors(self, motors):
                assert motors is None

            def assert_robotcam(self, source):
                assert source == 'robotcam'

            def start_camera(self):
                events.append('start')
                assert handlers[signal.SIGTERM] is run_mission.exit_on_signal
                assert handlers[signal.SIGHUP] is run_mission.exit_on_signal
                if failure == 'start':
                    raise RuntimeError('start failed')

            def run_mission(self, mission):
                events.append('run')
                if signum:
                    handlers[signum](signum, None)
                if failure == 'run':
                    raise RuntimeError('run failed')

            def stop_camera(self):
                events.append('stop_camera')

        with patch.object(run_mission.R, 'Robot', Robot), patch.object(
                run_mission.signal, 'signal', side_effect=register), patch.object(
                run_mission.R, 'warm_up', side_effect=AssertionError('dry warmup')), patch.dict(
                sys.modules, {'motors': None}):
            if signum:
                with self.assertRaises(SystemExit):
                    run_mission.main(['--dry'])
            else:
                with self.assertRaises(RuntimeError):
                    run_mission.main(['--dry'])
        return events

    def test_stop_camera_on_start_and_run_failure(self):
        self.assertEqual(self.run_dry('start'), ['start', 'stop_camera'])
        self.assertEqual(self.run_dry('run'), ['start', 'run', 'stop_camera'])

    def test_sigterm_and_sighup_stop_camera(self):
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signum=signum):
                self.assertEqual(self.run_dry(None, signum),
                                 ['start', 'run', 'stop_camera'])

    def test_motor_stop_precedes_camera_stop(self):
        events = []

        class Motors:
            def connect(self): events.append('connect')
            def ping(self): return True
            def get_distance(self): return 400
            def stop(self): events.append('motor_stop')
            def disconnect(self): events.append('disconnect')

        class Robot:
            def __init__(self, motors, camera_source):
                self.size_policy = SimpleNamespace(set_mode=Mock())
                self.known_rooms = {}
                self.last_moves = []

            def start_camera(self): pass
            def run_mission(self, mission): raise RuntimeError('run failed')
            def stop_camera(self): events.append('camera_stop')

        with patch.dict(sys.modules, {'motors': SimpleNamespace(Motors=Motors)}), patch.object(
                run_mission.R, 'Robot', Robot), patch.object(run_mission.R, 'warm_up'), patch.object(
                run_mission.signal, 'signal'), patch.object(run_mission.time, 'sleep'):
            with self.assertRaises(RuntimeError):
                run_mission.main([])
        self.assertEqual(events, ['connect', 'motor_stop', 'camera_stop', 'disconnect'])


if __name__ == '__main__':
    unittest.main()
