"""Decision #104 regression: real run_cycle/rules/move, fake external operations.

run_cycle() no longer consults an LLM. The move navigate_rules() returns is the
move executed, on every path; a failed capture fails closed to STOP (#108).
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def forbidden(*args, **kwargs):
    raise AssertionError('Unexpected external operation')


def load_robot_module():
    # Replace dependencies before importing main; never load vision or HTTP code.
    dependencies = {}
    for name, attributes in {
        'requests': ('post',),
        'detect_person': ('detect_scene', 'scene_to_text', 'person_direction'),
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
    return R, motors, R.Robot(motors=motors)


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
        self.assertEqual(motors.actions, [])
        self.assertEqual(robot.last_moves, [])
        R.detect_scene.assert_not_called()
        R.requests.post.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
