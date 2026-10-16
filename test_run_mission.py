"""Mission cleanup tests; all camera, USB and signal operations are faked."""
import signal
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_mission


def fake_heat(events, failure=None, trace=None):
    """HeatControl/HeatMonitor stand-ins for run_mission.heat_control; no su. trace: gets (name, detail) rows."""
    class Heat:
        def __init__(self, monitor):
            assert monitor == 'monitor'

        def start(self):
            events.append('heat_start')
            if failure == 'heat':
                raise RuntimeError('no root')

        def stop(self):
            events.append('heat_stop')

        def event(self, name, detail=None):
            if trace is not None:
                trace.append((name, detail))

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

            def set_mode(self, mode, reason):
                self.mode = mode

        with patch.object(run_mission, 'build_picker', return_value=(None, 'server down')), patch.object(
                run_mission.R, 'Robot', Robot), patch.object(
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
            def set_mode(self, mode, reason): pass

        with patch.object(run_mission, 'build_picker', return_value=(None, 'server down')), patch.dict(sys.modules, {'motors': SimpleNamespace(Motors=Motors)}), patch.object(
                run_mission.R, 'Robot', Robot), patch.object(run_mission.R, 'warm_up'), patch.object(
                run_mission.R, 'get_detector', side_effect=lambda pinned: events.append('detector')), patch.object(
                run_mission.signal, 'signal'), patch.object(run_mission.time, 'sleep'), patch.object(
                run_mission, 'heat_control', fake_heat(events)):
            with self.assertRaises(RuntimeError):
                run_mission.main([])
        self.assertEqual(events, ['detector', 'connect', 'heat_start', 'motor_stop', 'camera_stop', 'heat_stop',
                                  'disconnect'])



class PickerStartTest(unittest.TestCase):
    """--no-picker, the start fallback, and the worker stop on every exit; motors, camera and heat are faked."""

    def run_main(self, argv, build=None, run=None):
        events, trace, robots = [], [], []

        class Robot:
            def __init__(self, motors, camera_source):
                self.size_policy = SimpleNamespace(set_mode=Mock())
                self.known_rooms, self.last_moves = {}, []
                self.picker = self.mode = None
                robots.append(self)

            def set_mode(self, mode, reason): self.mode = mode
            def collect_pick(self): events.append('collect_pick')
            def start_camera(self, heat): events.append('start')
            def stop_camera(self): events.append('stop_camera')

            def run_mission(self, mission, heat):
                events.append('run')
                if run:
                    run()

        build = build or Mock(side_effect=AssertionError('picker built'))
        with patch.object(run_mission, 'build_picker', build), patch.object(run_mission.R, 'Robot', Robot), \
                patch.object(run_mission.R, 'get_detector', side_effect=lambda pinned: events.append('detector')), \
                patch.object(run_mission.signal, 'signal'), patch.object(run_mission.R, 'warm_up'), \
                patch.object(run_mission, 'heat_control', fake_heat(events, trace=trace)), \
                patch('builtins.print') as out:
            try:
                run_mission.main(argv)
            except BaseException as e:
                events.append(type(e).__name__)
        return events, trace, robots[0] if robots else None, out

    def picker(self, events):
        return SimpleNamespace(stop=Mock(side_effect=lambda: events.append('picker_stop') or True),
                               config=Mock(return_value={'url': 'u'}))

    def test_no_picker_loads_nothing_and_keeps_the_legacy_rules(self):
        events, trace, robot, _ = self.run_main(['--dry', '--no-picker'])
        self.assertEqual((robot.picker, robot.mode), (None, None))  # mode None: navigate_rules with person approach
        self.assertEqual(trace, [('picker_config', {'enabled': False, 'reason': '--no-picker'})])
        self.assertEqual(events, ['detector', 'heat_start', 'start', 'run', 'stop_camera', 'heat_stop'])

    def test_server_down_is_a_fallback_not_a_stop(self):
        build = Mock(return_value=(None, 'llama-server not ready'))
        events, trace, robot, out = self.run_main(['--dry'], build)
        self.assertEqual((robot.picker, robot.mode), (None, 'EXPLORE'))
        self.assertIn('run', events)
        self.assertEqual(trace, [('picker_config', {'enabled': False, 'reason': 'llama-server not ready'})])
        self.assertTrue(any('[PICKER] disabled' in str(c.args[0]) for c in out.call_args_list))

    def test_picker_built_before_motor_and_heat_threads_and_stopped_on_every_exit(self):
        def keyboard():
            raise KeyboardInterrupt

        def sigterm():
            run_mission.exit_on_signal(15, None)

        def error():
            raise RuntimeError('boom')
        for run in (None, keyboard, sigterm, error):
            with self.subTest(run=getattr(run, '__name__', None)):
                stops = []
                p = SimpleNamespace(stop=Mock(side_effect=lambda: stops.append('picker_stop') or True),
                                    config=Mock(return_value={'url': 'u'}))
                events, trace, robot, _ = self.run_main(['--dry'], Mock(return_value=(p, None)), run)
                p.stop.assert_called_once_with()
                self.assertIs(robot.picker, p)
                self.assertEqual(robot.mode, 'EXPLORE')
                self.assertEqual(trace[0], ('picker_config', {'enabled': True, 'url': 'u'}))
                # stopped after the camera, then the last pick is traced, then heat control stops
                self.assertEqual(events[events.index('stop_camera') + 1:events.index('stop_camera') + 3],
                                 ['collect_pick', 'heat_stop'])

    def test_picker_is_built_after_the_detector_and_before_motors_and_heat(self):
        order = []
        build = Mock(side_effect=lambda: order.append('picker') or (None, 'x'))

        class Motors:
            def connect(self): order.append('motors')
            def ping(self): return True
            def get_distance(self): return 400
            def stop(self): pass
            def disconnect(self): pass
        with patch.dict(sys.modules, {'motors': SimpleNamespace(Motors=Motors)}), \
                patch.object(run_mission.time, 'sleep'):
            events, *_ = self.run_main([], build)
        self.assertEqual(order, ['picker', 'motors'])
        self.assertLess(events.index('detector'), events.index('heat_start'))

    def test_build_picker_fallback_reasons(self):
        import picker
        import relate
        with patch.object(picker, 'Gemma', side_effect=OSError('refused')):
            p, reason = run_mission.build_picker()
        self.assertIsNone(p)
        self.assertIn('run setup_q4 first', reason)
        with patch.object(picker, 'Gemma'), patch.object(relate, 'Relate', side_effect=RuntimeError('STOP M2: x')):
            p, reason = run_mission.build_picker()
        self.assertEqual((p, reason), (None, 'M2 not available (RuntimeError: STOP M2: x)'))


if __name__ == '__main__':
    unittest.main()
