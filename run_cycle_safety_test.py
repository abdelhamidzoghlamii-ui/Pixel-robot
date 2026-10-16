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
                             sleep=Mock(side_effect=forbidden), perf_counter=Mock(return_value=0.0))
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
                    rules.assert_called_once_with([], 20, approach=True)

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
        R.read_frame = Mock(return_value={'status': 'ok', 'session': 'abc', 'frame': 1, 'age_s': 0.4,
                                          'image': SimpleNamespace(height=480, size=(640, 480))})
        R.get_detector = Mock(return_value=SimpleNamespace(
            detect=Mock(side_effect=Exception('inference failure'))))
        self.assertEqual(robot.run_cycle(), 'STOP')
        self.assertEqual(motors.actions, [('stop', ())])

    def test_start_camera_pins_new_frame(self):
        R, motors, robot = isolated_robot(distance=400)
        robot.camera_source = 'robotcam'
        R.get_detector = Mock(return_value=SimpleNamespace(check_affinity=Mock(return_value={'ok': True})))
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
        self.assertEqual(am.call_args.args[0][-6:], ['--es', 'mode', 'B', '--ei', 'rate', '1'])
        self.assertEqual(am.call_args.kwargs['timeout'], 10)
        R.get_detector.assert_called_once_with(pinned=True)

    def test_detector_creation_failure_never_starts_camera(self):
        R, motors, robot = isolated_robot(distance=400)
        robot.camera_source = 'robotcam'
        R.get_detector = Mock(side_effect=RuntimeError('cannot identify ORT workers'))
        with patch.object(R.subprocess, 'run') as am:
            with self.assertRaises(RuntimeError):
                robot.start_camera()
        am.assert_not_called()
        self.assertIsNone(robot.detector)

    def test_robotcam_cycle_uses_pinned_session(self):
        R, motors, robot = isolated_robot(distance=400)
        robot.camera_source = 'robotcam'
        robot.camera_session = 'abc'
        frame = SimpleNamespace(height=480, size=(640, 480))
        R.read_frame = Mock(return_value={'status': 'ok', 'session': 'abc',
                                          'frame': 1, 'age_s': 0.4, 'image': frame})
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
                    {'status': 'ok', 'session': 'abc', 'frame': 2, 'age_s': 0.4, 'image': frame},
                    {'status': 'ok', 'session': 'abc', 'frame': next_frame, 'age_s': 0.4, 'image': frame}])
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


def person_direction(results):
    """detect_person.person_direction (the harness keeps vision code unloaded)."""
    for r in results:
        if r[0] == 'person':
            return {'left': 'LEFT', 'right': 'RIGHT', 'center': 'FORWARD'}[r[2]]
    return None


def person(pos='center', bucket='far'):
    return ('person', 0.9, pos, bucket, 320, 320, 100, 300)


MODES = (None, 'EXPLORE', 'APPROACH', 'GREET', 'LOOK')  # None = --no-picker
FORWARD_MOTORS = ('forward',)  # Robot.move has no diagonal; FWD_L/FWD_R are checked on choose_move directly


def mode_robot(mode, distance=400, results=()):
    """A robot in `mode` whose next run_cycle sees `results` (legacy tuples) on a fake 320 frame."""
    R, motors, robot = isolated_robot(distance=distance)
    R.time.monotonic = Mock(return_value=50.0)
    R.time.sleep = Mock()  # HOLD waits out the cycle
    R.time.strftime = Mock(return_value='12:00')
    R.person_direction = person_direction
    R.speak = Mock()
    robot.camera_source = 'robotcam'
    R.read_frame = Mock(side_effect=lambda session=None: {
        'status': 'ok', 'session': 'abc', 'frame': robot.camera_frame + 1 if robot.camera_frame else 1,
        'age_s': 0.4, 'image': SimpleNamespace(height=480, size=(640, 480))})
    R.get_detector = Mock(return_value=SimpleNamespace(detect=Mock(return_value=[])))
    R.scene_from_detections = Mock(return_value=list(results))
    robot.heat = SimpleNamespace(event=Mock(), monitor=SimpleNamespace(readings=lambda: {'dump': None}))
    if mode:
        robot.set_mode(mode, 'test')
    return R, motors, robot


def traced(robot, name):
    return [c.args[1] for c in robot.heat.event.call_args_list if c.args[0] == name]


class ModeTest(unittest.TestCase):
    """Each mode only shapes the Python rules; Python keeps every move (#104/#109)."""

    def test_explore_ignores_people_approach_follows_them(self):
        for mode, pos, move in (('EXPLORE', 'left', 'FORWARD'), ('EXPLORE', 'right', 'FORWARD'),
                                ('APPROACH', 'left', 'LEFT'), ('APPROACH', 'right', 'RIGHT'),
                                ('APPROACH', 'center', 'FORWARD'), (None, 'left', 'LEFT'), (None, 'center', 'FORWARD')):
            with self.subTest(mode=mode, pos=pos):
                R, motors, robot = mode_robot(mode, results=[person(pos)])
                self.assertEqual(robot.run_cycle(), move)

    def test_explore_logs_rooms_with_a_person_in_view(self):
        R, motors, robot = mode_robot('EXPLORE', results=[person('left'), ('bed', 0.8, 'right', 'medium', 0, 0, 0, 0)])
        self.assertEqual(robot.run_cycle(), 'FORWARD')
        self.assertIn('bedroom', robot.known_rooms)

    def test_greet_holds_and_speaks_once(self):
        R, motors, robot = mode_robot('GREET', results=[person()])
        self.assertEqual([robot.run_cycle() for _ in range(3)], ['HOLD'] * 3)
        R.speak.assert_called_once_with(R.GREETING)
        self.assertEqual(motors.actions, [('stop', ())] * 3)
        self.assertEqual(R.time.sleep.call_args_list, [((R.CYCLE_MOVE_TIME,),)] * 3)  # the cycle keeps its length
        self.assertNotEqual(robot.state, 'found')
        robot.set_mode('EXPLORE', 'x')
        robot.set_mode('GREET', 'x')
        robot.run_cycle()
        self.assertEqual(R.speak.call_count, 2)  # once per GREET

    def test_look_rotates_four_cycles_then_explores(self):
        R, motors, robot = mode_robot('LOOK')
        self.assertEqual([robot.run_cycle() for _ in range(5)], ['LEFT'] * 4 + ['FORWARD'])
        self.assertEqual(robot.mode, 'EXPLORE')
        self.assertEqual(traced(robot, 'mode')[-1]['to'], 'EXPLORE')

    def test_hold_is_not_a_mission_end(self):
        R, motors, robot = mode_robot('GREET')
        robot.move('HOLD')
        self.assertEqual((robot.last_moves, robot.state), (['HOLD'], 'idle'))


class PersonStopTest(unittest.TestCase):
    """#79 area bucket, owner D4: every forward move holds while a person is 'close' or 'very close'."""

    def test_every_mode_holds_forward_at_close_and_very_close(self):
        for mode in MODES:
            for bucket in ('close', 'very close'):
                with self.subTest(mode=mode, bucket=bucket):
                    R, motors, robot = mode_robot(mode, results=[person('center', bucket)])
                    # LOOK turns the clear-path FORWARD into an in-place LEFT: nothing forward to hold
                    self.assertEqual(robot.run_cycle(), 'LEFT' if mode == 'LOOK' else 'HOLD')
                    self.assertFalse([a for a, _ in motors.actions if a in FORWARD_MOTORS])
                    if mode not in ('GREET', 'LOOK'):
                        self.assertEqual(traced(robot, 'person_stop'), [{'move': 'FORWARD', 'mode': mode,
                                                                         'bucket': bucket}])

    def test_a_closer_less_confident_second_person_still_stops(self):
        """scene_from_detections keeps only the most confident person; person-stop must see every person."""
        from detect_person import scene_from_detections
        for mode in MODES:
            with self.subTest(mode=mode):
                R, motors, robot = mode_robot(mode)
                R.scene_from_detections = scene_from_detections
                R.get_detector = Mock(return_value=SimpleNamespace(detect=Mock(return_value=[
                    {'class_name': 'person', 'conf': 0.9, 'box_xyxy': (300, 200, 340, 260)},     # far, centre
                    {'class_name': 'person', 'conf': 0.5, 'box_xyxy': (0, 0, 400, 300)}])))      # close (0.39)
                self.assertEqual([r[3] for r in R.scene_from_detections(robot.detector.detect(None) if robot.detector
                                  else R.get_detector().detect(None), 640, 480)], ['far'])
                move = robot.run_cycle()
                self.assertNotIn(move, ('FORWARD', 'FWD_L', 'FWD_R'))
                self.assertFalse([a for a, _ in motors.actions if a in FORWARD_MOTORS])

    def test_medium_and_far_do_not_stop(self):
        for bucket in ('medium', 'far'):
            R, motors, robot = mode_robot('APPROACH', results=[person('center', bucket)])
            self.assertEqual(robot.run_cycle(), 'FORWARD')
            self.assertEqual(traced(robot, 'person_stop'), [])

    def test_diagonal_forward_moves_hold_in_every_mode(self):
        for mode in MODES:
            for move in ('FWD_L', 'FWD_R', 'FORWARD'):
                with self.subTest(mode=mode, move=move):
                    R, motors, robot = mode_robot(mode)
                    robot.navigate_rules = Mock(return_value=move)
                    expected = 'LEFT' if (mode, move) == ('LOOK', 'FORWARD') else 'HOLD'
                    self.assertEqual(robot.choose_move([person('right', 'close')], 400), expected)

    def test_turns_away_from_a_close_person_are_not_held(self):
        R, motors, robot = mode_robot(None, results=[person('left', 'close')])
        self.assertEqual(robot.run_cycle(), 'LEFT')


class SafetyInEveryModeTest(unittest.TestCase):
    def test_obstacle_rules_win_in_every_mode(self):
        for mode in MODES:
            for distance in (10, 20):
                with self.subTest(mode=mode, distance=distance):
                    R, motors, robot = mode_robot(mode, distance=distance, results=[person('center', 'far')])
                    move = robot.run_cycle()
                    self.assertNotIn(move, ('FORWARD', 'FWD_L', 'FWD_R'))
                    if mode == 'GREET':
                        self.assertEqual(move, 'HOLD')
                    else:  # exactly the ladder's move, whatever the mode
                        ladder = isolated_robot(distance=distance)[2].navigate_rules([], distance)
                        self.assertEqual(move, ladder)

    def test_stale_frame_stops_before_any_mode_logic(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                R, motors, robot = mode_robot(mode, results=[person('center', 'far')])
                robot.camera_frame = 5
                R.read_frame = Mock(return_value={'status': 'ok', 'session': 'abc', 'frame': 5, 'age_s': 0.4,
                                                  'image': None})
                self.assertEqual(robot.run_cycle(), 'STOP')
                self.assertEqual(motors.actions, [('stop', ())])
                R.speak.assert_not_called()

    def test_picks_never_reach_the_motors(self):
        """A pick only sets a mode name; even an unknown name cannot produce a motor command."""
        R, motors, robot = mode_robot('EXPLORE', distance=20)
        robot.picker = SimpleNamespace(result=Mock(return_value={
            'slot': robot.slot_n, 'trigger': 'start', 'mode': 'FORWARD', 'fallback': None, 'picker_ms': 1.0,
            'm2_ms': None, 'cancelled': False}), submit=Mock(), allow=Mock(), forbid=Mock())
        robot.pick_inflight = True
        robot.last_640 = None
        move = robot.run_cycle()
        self.assertEqual(robot.mode, 'FORWARD')
        self.assertEqual(move, isolated_robot(distance=20)[2].navigate_rules([], 20))  # obstacle ladder still decides


if __name__ == '__main__':
    unittest.main(verbosity=2)
