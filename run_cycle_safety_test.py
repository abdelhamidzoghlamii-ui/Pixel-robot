"""Decision #104 regression: real run_cycle/rules/move, fake external operations.

run_cycle() no longer consults an LLM. The move navigate_rules() returns is the
move executed, on every path; a failed capture fails closed to STOP (#108).
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from PIL import Image


def forbidden(*args, **kwargs):
    raise AssertionError('Unexpected external operation')


def load_robot_module():
    # Replace dependencies before importing main; never load vision or HTTP code.
    dependencies = {}
    for name, attributes in {
        'requests': ('post',),
        'detect_person': ('get_detector', 'detect_scene', 'scene_from_detections',
                          'scene_to_text', 'person_direction'),
        'stereo_depth': ('stereo_scan', 'scene_with_depth',
                         'estimate_distance_single'),
    }.items():
        module = ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, Mock(side_effect=forbidden))
        dependencies[name] = module
    spec = importlib.util.spec_from_file_location(
        'robot_under_test', Path(__file__).with_name('main.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies), patch.object(sys, 'path', sys.path[:]):
        spec.loader.exec_module(module)
    return module


class FakeMotors:
    def __init__(self, distance, now):
        self.state = {'dist_at': now}
        self.get_distance = Mock(return_value=distance)
        self.actions = []
        for name in ('forward', 'backward', 'rotate_left', 'rotate_right',
                     'strafe_left', 'strafe_right', 'stop'):
            setattr(self, name, Mock(side_effect=self.recorder(name)))

    def recorder(self, name):
        def record(*args):
            self.actions.append((name, args))
        return record


def isolated_robot(distance, now=100.0):
    """A Robot whose every external operation is faked or forbidden."""
    R = load_robot_module()
    R.time = SimpleNamespace(time=Mock(return_value=now),
                             sleep=Mock(side_effect=forbidden))
    R.os = SimpleNamespace(system=forbidden, popen=forbidden)
    R.get_temp = Mock(return_value=35)
    R.detect_scene = Mock(return_value=[])
    R.scene_to_text = Mock(return_value='empty room')
    R.take_photo = Mock(side_effect=forbidden)
    R.speak = Mock(side_effect=forbidden)
    R.listen = Mock(side_effect=forbidden)
    motors = FakeMotors(distance=distance, now=now)
    return R, motors, R.Robot(motors=motors, camera_source='termux_photo')


def recording_rules(robot, recorded):
    """Patch navigate_rules so its real return value is recorded, not replaced."""
    real = robot.navigate_rules

    def wrapper(*args, **kwargs):
        move = real(*args, **kwargs)
        recorded.append(move)
        return move

    return patch.object(robot, 'navigate_rules', Mock(side_effect=wrapper))


class RunCycleSafetyTest(unittest.TestCase):
    def test_rule_move_is_executed_verbatim_in_both_avoidance_directions(self):
        for initial_side, expected_move, expected_motor in (
            ('LEFT', 'RIGHT', 'rotate_right'),
            ('RIGHT', 'LEFT', 'rotate_left'),
        ):
            with self.subTest(initial_side=initial_side):
                now = 100.0
                R, motors, robot = isolated_robot(distance=20, now=now)
                robot.mission = 'explore and map the rooms'
                robot.avoid_side = initial_side
                robot.blocked_n = 6
                robot.blocked_since = now - 6
                robot.asked_gemma = False
                self.assertLess(robot.get_distance(), R.OBSTACLE_DIST)

                # A supplied fake path bypasses capture; detection never opens it.
                recorded = []
                with recording_rules(robot, recorded) as rules:
                    result = robot.run_cycle(photo_path='fake-photo.jpg')
                    rules.assert_called_once_with([], 20)

                # The executed move is exactly the move the rules returned.
                self.assertEqual(recorded, [expected_move])
                self.assertEqual(
                    (result, motors.actions),
                    (recorded[0], [(expected_motor, (120, R.CYCLE_MOVE_TIME))]))
                self.assertEqual(robot.last_moves, [recorded[0]])
                motors.forward.assert_not_called()

                # No outbound LLM call, and no other external operation.
                R.requests.post.assert_not_called()
                R.take_photo.assert_not_called()
                R.time.sleep.assert_not_called()
                R.speak.assert_not_called()

                # The side flip still happens exactly once, at n == 7.
                self.assertTrue(robot.nav_stuck)
                self.assertTrue(robot.asked_gemma)
                self.assertEqual(robot.blocked_n, 7)
                self.assertEqual(robot.avoid_side, expected_move)

                # The next blocked cycle is unchanged: same move, no second flip.
                recorded = []
                with recording_rules(robot, recorded):
                    self.assertEqual(robot.run_cycle('fake-photo.jpg'),
                                     expected_move)
                self.assertEqual(recorded, [expected_move])
                self.assertFalse(robot.nav_stuck)
                self.assertEqual(robot.blocked_n, 8)
                self.assertEqual(robot.avoid_side, expected_move)
                self.assertEqual(motors.actions,
                                 [(expected_motor, (120, R.CYCLE_MOVE_TIME))] * 2)
                R.requests.post.assert_not_called()
                motors.forward.assert_not_called()

    def test_failed_capture_fails_closed_to_stop(self):
        R, motors, robot = isolated_robot(distance=400)
        robot.mission = 'explore and map the rooms'
        R.take_photo = Mock(return_value=False)

        self.assertEqual(robot.run_cycle(), 'STOP')

        R.take_photo.assert_called_once_with(R.PHOTO_A)
        self.assertEqual(motors.actions, [('stop', ())])
        self.assertEqual(robot.last_moves, [])
        R.detect_scene.assert_not_called()
        R.requests.post.assert_not_called()

    def test_unusable_robotcam_frames_stop_motors(self):
        for status in ('missing', 'bad', 'other_session'):
            with self.subTest(status=status):
                R, motors, robot = isolated_robot(distance=400)
                robot.camera_source = 'robotcam'
                R.read_frame = Mock(return_value={'status': status})

                self.assertEqual(robot.run_cycle(), 'STOP')
                R.read_frame.assert_called_once_with(session=None)
                self.assertEqual(motors.actions, [('stop', ())])
                R.detect_scene.assert_not_called()

    def test_vision_exception_stops_motors(self):
        R, motors, robot = isolated_robot(distance=400)
        robot.camera_source = 'robotcam'
        R.read_frame = Mock(return_value={'status': 'ok', 'session': 'abc', 'frame': 1,
                                          'image': SimpleNamespace(height=480, size=(640, 480))})
        R.get_detector = Mock(return_value=SimpleNamespace(
            detect=Mock(side_effect=Exception('inference failure'))))
        self.assertEqual(robot.run_cycle(), 'STOP')
        self.assertEqual(motors.actions, [('stop', ())])

    def test_start_camera_pins_new_frame(self):
        R, motors, robot = isolated_robot(distance=400)
        robot.camera_source = 'robotcam'
        R.get_detector = Mock(return_value=object())
        R.time = SimpleNamespace(clock_gettime=Mock(return_value=100.0),
                                 CLOCK_BOOTTIME=7, monotonic=Mock(side_effect=[0, 0, 1]),
                                 sleep=Mock())
        R.read_frame = Mock(side_effect=[{'status': 'missing'},
                                         {'status': 'ok', 'session': 'abc'}])
        with patch.object(R.subprocess, 'run') as am:
            robot.start_camera()
        self.assertEqual(robot.camera_session, 'abc')
        self.assertIsNotNone(robot.detector)
        self.assertEqual(R.read_frame.call_count, 2)
        R.read_frame.assert_called_with(min_capture_boot_s=100.0)
        self.assertEqual(am.call_args.args[0][0:2], ['am', 'start'])
        self.assertEqual(am.call_args.kwargs['timeout'], 10)

    def test_robotcam_cycle_uses_pinned_session(self):
        R, motors, robot = isolated_robot(distance=400)
        robot.camera_source = 'robotcam'
        robot.camera_session = 'abc'
        frame = SimpleNamespace(height=480, size=(640, 480))
        R.read_frame = Mock(return_value={'status': 'ok', 'session': 'abc',
                                          'frame': 1, 'image': frame})
        detector = SimpleNamespace(detect=Mock(return_value=[]))
        R.get_detector = Mock(return_value=detector)
        R.scene_from_detections = Mock(return_value=[])
        self.assertEqual(robot.run_cycle(), 'FORWARD')
        R.read_frame.assert_called_once_with(session='abc')
        detector.detect.assert_called_once_with(frame, 320)
        self.assertEqual(motors.actions, [('forward', (R.MOTOR_SPEED, R.CYCLE_MOVE_TIME))])

    def test_robotcam_repeated_or_older_frame_stops_before_detection(self):
        for next_frame in (2, 1):
            with self.subTest(next_frame=next_frame):
                R, motors, robot = isolated_robot(distance=400)
                robot.camera_source = 'robotcam'
                frame = SimpleNamespace(height=480, size=(640, 480))
                R.read_frame = Mock(side_effect=[
                    {'status': 'ok', 'session': 'abc', 'frame': 2, 'image': frame},
                    {'status': 'ok', 'session': 'abc', 'frame': next_frame, 'image': frame}])
                detector = SimpleNamespace(detect=Mock(return_value=[]))
                R.get_detector = Mock(return_value=detector)
                R.scene_from_detections = Mock(return_value=[])
                self.assertEqual(robot.run_cycle(), 'FORWARD')
                self.assertEqual(robot.run_cycle(), 'STOP')
                self.assertEqual(robot.camera_frame, 2)
                detector.detect.assert_called_once_with(frame, 320)
                self.assertEqual(motors.actions,
                                 [('forward', (R.MOTOR_SPEED, R.CYCLE_MOVE_TIME)),
                                  ('stop', ())])

    def test_termux_photo_rollback_cycle(self):
        R, motors, robot = isolated_robot(distance=400)
        with tempfile.TemporaryDirectory() as directory:
            R.PHOTO_A = str(Path(directory) / 'photo.jpg')
            Image.new('RGB', (640, 480)).save(R.PHOTO_A)
            R.take_photo = Mock(return_value=True)
            detector = SimpleNamespace(detect=Mock(return_value=[]))
            R.get_detector = Mock(return_value=detector)
            R.scene_from_detections = Mock(return_value=[])
            self.assertEqual(robot.run_cycle(), 'FORWARD')
        R.take_photo.assert_called_once_with(R.PHOTO_A)
        self.assertEqual(motors.actions, [('forward', (R.MOTOR_SPEED, R.CYCLE_MOVE_TIME))])


if __name__ == '__main__':
    unittest.main(verbosity=2)
