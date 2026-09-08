"""Decision #81 regression: real run_cycle/rules/move, fake external operations."""
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


class RunCycleSafetyTest(unittest.TestCase):
    def test_stuck_forward_cannot_replace_either_rule_selected_turn(self):
        for initial_side, expected_move, expected_motor in (
            ('LEFT', 'RIGHT', 'rotate_right'),
            ('RIGHT', 'LEFT', 'rotate_left'),
        ):
            with self.subTest(initial_side=initial_side):
                R = load_robot_module()
                now = 100.0
                R.time = SimpleNamespace(time=Mock(return_value=now),
                                         sleep=Mock(side_effect=forbidden))
                R.os = SimpleNamespace(system=forbidden, popen=forbidden)
                R.get_temp = Mock(return_value=35)
                R.detect_scene = Mock(return_value=[])
                R.scene_to_text = Mock(return_value='empty room')
                R.take_photo = Mock(side_effect=forbidden)
                R.speak = Mock(side_effect=forbidden)
                R.listen = Mock(side_effect=forbidden)
                R.gemma_decide = Mock(return_value='FORWARD')
                motors = FakeMotors(distance=20, now=now)
                robot = R.Robot(motors=motors)
                robot.mission = 'explore and map the rooms'
                robot.avoid_side = initial_side
                robot.blocked_n = 6
                robot.blocked_since = now - 6
                robot.asked_gemma = False
                self.assertLess(robot.get_distance(), R.OBSTACLE_DIST)

                # A supplied fake path bypasses capture; detection never opens it.
                with patch.object(robot, 'navigate_rules',
                                  wraps=robot.navigate_rules) as rules:
                    result = robot.run_cycle(photo_path='fake-photo.jpg')
                    rules.assert_called_once_with([], 20)
                R.gemma_decide.assert_called_once()
                context = R.gemma_decide.call_args.args[0]
                self.assertIn('Distance ahead: 20cm', context)
                self.assertIn(robot.mission, context)
                self.assertEqual(R.gemma_decide.call_args.kwargs,
                                 {'image_path': None})
                self.assertTrue(robot.nav_stuck)
                self.assertTrue(robot.asked_gemma)
                self.assertEqual(robot.blocked_n, 7)
                self.assertEqual(
                    (result, motors.actions),
                    (expected_move, [(expected_motor, (120, R.CYCLE_MOVE_TIME))]))
                motors.forward.assert_not_called()
                self.assertEqual(robot.last_moves, [expected_move])
                R.take_photo.assert_not_called()
                R.time.sleep.assert_not_called()
                R.speak.assert_not_called()

                # The next blocked cycle must not consult Gemma again.
                self.assertEqual(robot.run_cycle('fake-photo.jpg'), expected_move)
                self.assertFalse(robot.nav_stuck)
                self.assertEqual(robot.blocked_n, 8)
                self.assertEqual(R.gemma_decide.call_count, 1)
                motors.forward.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
