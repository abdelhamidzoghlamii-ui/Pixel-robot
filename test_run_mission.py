"""Mission cleanup tests; all camera, USB and signal operations are faked."""
import signal
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_mission


def fake_heat(events, failure=None):
    """HeatControl/HeatMonitor stand-ins for run_mission.heat_control; no su."""
    class Heat:
        def __init__(self, monitor):
            assert monitor == 'monitor'

        def start(self):
            events.append('heat_start')
            if failure == 'heat':
                raise RuntimeError('no root')

        def stop(self):
            events.append('heat_stop')

    return SimpleNamespace(HeatControl=Heat, HeatMonitor=lambda: 'monitor')


class MissionExitTest(unittest.TestCase):
    def run_dry(self, failure, signum=None):
        events = []
        handlers = {}

        def detector(pinned):
            events.append('detector')
            assert pinned is True
            if failure == 'detector':
                raise RuntimeError('cannot identify ORT workers')

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

            def start_camera(self, heat):
                events.append('start')
                assert heat is not None
                assert handlers[signal.SIGTERM] is run_mission.exit_on_signal
                assert handlers[signal.SIGHUP] is run_mission.exit_on_signal
                if failure == 'start':
                    raise RuntimeError('start failed')

            def run_mission(self, mission, heat):
                events.append('run')
                assert heat is not None
                if signum:
                    handlers[signum](signum, None)
                if failure == 'run':
                    raise RuntimeError('run failed')

            def stop_camera(self):
                events.append('stop_camera')

        with patch.object(run_mission.R, 'Robot', Robot), patch.object(
                run_mission.R, 'get_detector', side_effect=detector), patch.object(
                run_mission.signal, 'signal', side_effect=register), patch.object(
                run_mission.R, 'warm_up', side_effect=AssertionError('dry warmup')), patch.dict(
                sys.modules, {'motors': None}), patch.object(run_mission, 'heat_control', fake_heat(events, failure)):
            if signum:
                with self.assertRaises(SystemExit):
                    run_mission.main(['--dry'])
            else:
                with self.assertRaises(RuntimeError):
                    run_mission.main(['--dry'])
        return events

    def test_stop_camera_on_start_and_run_failure(self):
        self.assertEqual(self.run_dry('start'), ['detector', 'heat_start', 'start', 'stop_camera', 'heat_stop'])
        self.assertEqual(self.run_dry('run'), ['detector', 'heat_start', 'start', 'run', 'stop_camera', 'heat_stop'])

    def test_heat_start_failure_never_starts_camera_or_mission(self):
        self.assertEqual(self.run_dry('heat'), ['detector', 'heat_start', 'stop_camera', 'heat_stop'])

    def test_detector_failure_starts_nothing(self):
        """Created first (before motor and heat threads); a failure raises before anything else starts."""
        self.assertEqual(self.run_dry('detector'), ['detector'])

    def test_sigterm_and_sighup_stop_camera(self):
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signum=signum):
                self.assertEqual(self.run_dry(None, signum),
                                 ['detector', 'heat_start', 'start', 'run', 'stop_camera', 'heat_stop'])

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

            def start_camera(self, heat): pass
            def run_mission(self, mission, heat): raise RuntimeError('run failed')
            def stop_camera(self): events.append('camera_stop')

        with patch.dict(sys.modules, {'motors': SimpleNamespace(Motors=Motors)}), patch.object(
                run_mission.R, 'Robot', Robot), patch.object(run_mission.R, 'warm_up'), patch.object(
                run_mission.R, 'get_detector', side_effect=lambda pinned: events.append('detector')), patch.object(
                run_mission.signal, 'signal'), patch.object(run_mission.time, 'sleep'), patch.object(
                run_mission, 'heat_control', fake_heat(events)):
            with self.assertRaises(RuntimeError):
                run_mission.main([])
        self.assertEqual(events, ['detector', 'connect', 'heat_start', 'motor_stop', 'camera_stop', 'heat_stop',
                                  'disconnect'])


if __name__ == '__main__':
    unittest.main()
