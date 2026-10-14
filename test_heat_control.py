"""Heat control (#127, #129, adaptive 37/35 degC) offline: no su, camera, motors or network."""
import json
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import heat_control as H
from run_cycle_safety_test import isolated_robot

# Real record, trimmed: benchmark/thermal_char/runs/run_20260930T233208Z/thermalservice_raw.jsonl line 1, "text".
# The cached section's VIRTUAL-SKIN (36.986206) differs from the HAL one (31.4255): only HAL must be read.
DUMP_TEXT = """IsStatusOverride: false
ThermalStatusListeners:
\tcallbacks: 8
\tkilled: false
\tbroadcasts count: -1
Thermal Status: 0
Cached temperatures:
\tTemperature{mValue=79.0, mType=0, mName=LITTLE, mStatus=0}
\tTemperature{mValue=36.712074, mType=-1, mName=VIRTUAL-SKIN-CHARGE, mStatus=1}
\tTemperature{mValue=86.00001, mType=0, mName=BIG, mStatus=0}
\tTemperature{mValue=36.986206, mType=3, mName=VIRTUAL-SKIN, mStatus=0}
HAL Ready: true
HAL connection:
\tThermalHAL AIDL 3  connected: yes
Current temperatures from HAL:
\tTemperature{mValue=37.0, mType=9, mName=TPU, mStatus=0}
\tTemperature{mValue=37.0, mType=1, mName=G3D, mStatus=0}
\tTemperature{mValue=42.000004, mType=0, mName=BIG, mStatus=0}
\tTemperature{mValue=43.000004, mType=0, mName=LITTLE, mStatus=0}
\tTemperature{mValue=42.000004, mType=0, mName=MID, mStatus=0}
\tTemperature{mValue=32.173, mType=-1, mName=neutral_therm, mStatus=0}
\tTemperature{mValue=31.448252, mType=-1, mName=VIRTUAL-SKIN-CHARGE, mStatus=0}
\tTemperature{mValue=31.4255, mType=3, mName=VIRTUAL-SKIN, mStatus=0}
\tTemperature{mValue=0.0, mType=4, mName=VIRTUAL-USB-THROTTLING, mStatus=0}
\tTemperature{mValue=31.7, mType=2, mName=battery, mStatus=0}
Current cooling devices from HAL:
\tCoolingDevice{mValue=0, mType=5, mName=gxp-cooling}
"""
HAL_SKIN = '\tTemperature{mValue=31.4255, mType=3, mName=VIRTUAL-SKIN, mStatus=0}\n'


def readings(now, status=0, skin=30.0, bat=30.0, cpu=(60.0,), dump_age=0.0, fast_age=0.0):
    """Fresh readings at `now`; cpu lists consecutive 1 s samples, the last one fast_age old."""
    fast = [{'t': now - fast_age - (len(cpu) - 1 - i), 'cpu_c': c, 'bat_c': bat} for i, c in enumerate(cpu)]
    return {'dump': {'t': now - dump_age, 'status': status, 'skin': skin}, 'fast': fast}


def state(phase='ACTIVE', elapsed=0.0, now=1000.0):
    return {'phase': phase, 'since': now - elapsed}


def decide(phase='ACTIVE', elapsed=0.0, **kw):
    return H.decide(1000.0, readings(1000.0, **kw), state(phase, elapsed))[0]


class ParseTest(unittest.TestCase):
    def test_real_dump_status_and_hal_skin(self):
        self.assertEqual(H.parse_dump(DUMP_TEXT), {'status': 0, 'skin': 31.4255})

    def test_missing_or_nan_skin_is_none(self):
        self.assertIsNone(H.parse_dump(DUMP_TEXT.replace(HAL_SKIN, ''))['skin'])
        self.assertIsNone(H.parse_dump(DUMP_TEXT.replace('mValue=31.4255', 'mValue=NaN'))['skin'])
        self.assertIsNone(H.parse_dump('')['status'])


class DecideTest(unittest.TestCase):
    def test_each_rule_alone(self):
        self.assertEqual(decide(dump_age=61), 'END')                          # 1 fail-closed skin/status
        self.assertEqual(decide(fast_age=6), 'END')                           # 1 fail-closed CPU/battery
        self.assertEqual(H.decide(0, {'dump': None, 'fast': []}, state())[0], 'END')
        self.assertEqual(decide(bat=45.0), 'END')                             # 2 battery
        self.assertEqual(decide(cpu=(110, 110, 110)), 'END')                  # 3 CPU fault
        self.assertEqual(decide(status=4), 'HOLD')                            # 4 CRITICAL
        target, reason = H.decide(1000.0, readings(1000.0, status=3), state())
        self.assertEqual((target, 'SEVERE' in reason), ('ACTIVE', True))      # 5 SEVERE: logged only
        self.assertEqual(decide(skin=38, elapsed=60), 'PAUSE')                # 6 adaptive
        self.assertEqual(decide('PAUSE', skin=34, elapsed=30), 'ACTIVE')

    def test_precedence(self):
        self.assertEqual(decide(status=4, bat=45.0), 'END')
        self.assertEqual(decide(status=4, cpu=(110, 110, 110)), 'END')
        self.assertEqual(decide(status=4, dump_age=61), 'END')
        self.assertEqual(decide(status=4, skin=30), 'HOLD')
        self.assertEqual(decide('PAUSE', status=4, skin=30, elapsed=100), 'HOLD')
        self.assertEqual(decide(status=3, skin=38, elapsed=60), 'PAUSE')
        target, reason = H.decide(1000.0, readings(1000.0, bat=45.0, fast_age=6), state())
        self.assertIn('fail-closed', reason)

    def test_boundaries(self):
        self.assertEqual(decide(skin=36.99, elapsed=100), 'ACTIVE')
        self.assertEqual(decide(skin=37.0, elapsed=60), 'PAUSE')
        self.assertEqual(decide(skin=40, elapsed=59.9), 'ACTIVE')
        self.assertEqual(decide('PAUSE', skin=35.01, elapsed=100), 'PAUSE')
        self.assertEqual(decide('PAUSE', skin=35.0, elapsed=30), 'ACTIVE')
        self.assertEqual(decide('PAUSE', skin=30, elapsed=29.9), 'PAUSE')
        self.assertEqual(decide(bat=44.9), 'ACTIVE')
        self.assertEqual(decide(bat=45.0), 'END')
        self.assertEqual(decide(cpu=(60, 110, 110)), 'ACTIVE')
        self.assertEqual(decide(cpu=(110, 110)), 'ACTIVE')
        self.assertEqual(decide(cpu=(110, 110, 110)), 'END')
        # Staleness: exactly the limit is still fresh (inclusive); beyond it ends the mission.
        self.assertEqual(decide(dump_age=60.0), 'ACTIVE')
        self.assertEqual(decide(dump_age=60.1), 'END')
        self.assertEqual(decide(fast_age=5.0), 'ACTIVE')
        self.assertEqual(decide(fast_age=5.1), 'END')

    def test_hold_releases_into_pause(self):
        self.assertEqual(decide('HOLD', status=3, skin=44, elapsed=100), 'HOLD')
        self.assertEqual(decide('HOLD', status=2, skin=42, elapsed=100), 'PAUSE')
        self.assertEqual(decide('HOLD', status=2, skin=35.5, elapsed=100), 'PAUSE')
        self.assertEqual(decide('HOLD', status=2, skin=34, elapsed=100), 'ACTIVE')


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class FakeMonitor:
    """values = (status, skin, battery, cpu) read fresh at the clock's time; None = no reading at all."""
    def __init__(self, clock):
        self.clock, self.values, self.on_dump = clock, (0, 30.0, 30.0, 60.0), None
        self.calls = []

    def start(self):
        self.calls.append('start')

    def stop(self):
        self.calls.append('stop')

    def readings(self):
        if self.values is None:
            return {'dump': None, 'fast': []}
        status, skin, bat, cpu = self.values
        t = self.clock.t
        return {'dump': {'t': t, 'status': status, 'skin': skin}, 'fast': [{'t': t, 'cpu_c': cpu, 'bat_c': bat}]}


class MissionTest(unittest.TestCase):
    """Real Robot.run_mission/run_cycle/pause_cycle/start_camera/stop_camera; fake clock, monitor, motors, camera."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        R, motors, robot = isolated_robot(distance=400)
        robot.camera_source = 'robotcam'
        ticks = iter(range(10**6))  # perf_counter: +0.1 s per call, so each detect takes 100 ms
        R.time = SimpleNamespace(time=Mock(return_value=100.0), sleep=Mock(), monotonic=Mock(return_value=0.0),
                                 clock_gettime=Mock(return_value=0.0), CLOCK_BOOTTIME=7,
                                 perf_counter=lambda: next(ticks) / 10)
        R.subprocess = SimpleNamespace(DEVNULL=subprocess.DEVNULL, run=Mock(side_effect=self.am))
        self.am_args = []
        self.affinity = {'before': {7: [4, 5]}, 'repinned': [], 'after': {7: [4, 5]}, 'ok': True, 'errors': {}}
        self.detector = SimpleNamespace(detect=Mock(return_value=[]), check_affinity=Mock(side_effect=lambda: self.affinity),
                                        affinity=Mock(return_value={7: [4, 5], 8: [4, 5], 9: [4, 5]}),
                                        config=Mock(return_value={'ort_version': '1.25.1', 'caller_tid': 7}))
        R.get_detector = Mock(return_value=self.detector)
        R.scene_from_detections = Mock(return_value=[])
        self.frames = [50]
        R.read_frame = Mock(side_effect=self.read_frame)
        robot.start_camera = Mock(wraps=robot.start_camera)
        robot.stop_camera = Mock(wraps=robot.stop_camera)
        robot.camera_session, robot.camera_frame = 's1', 49
        self.R, self.motors, self.robot = R, motors, robot
        self.clock = Clock()
        self.monitor = FakeMonitor(self.clock)
        self.heat = H.HeatControl(self.monitor, trace_dir=self.dir.name, clock=self.clock)
        self.am_fail = None
        with patch('builtins.print'):
            self.heat.start()

    def am(self, args, **kw):
        self.am_args.append(args)
        self.motors.actions.append(('am', args[1]))
        if args[1] == self.am_fail:
            raise subprocess.CalledProcessError(1, args)

    def read_frame(self, session=None, min_capture_boot_s=None):
        if min_capture_boot_s is not None:  # start_camera waiting for its first frame
            return {'status': 'ok', 'session': 's2'}
        return {'status': 'ok', 'session': session, 'frame': self.frames.pop(0), 'age_s': 0.5,
                'image': SimpleNamespace(height=480, size=(640, 480))}

    def run_script(self, steps):
        """Each step (t, values, distance) applies just before one decide(); then readings vanish -> END."""
        real = self.heat.decide
        steps = list(steps)

        def scripted(distance=None):
            if steps:
                self.clock.t, self.monitor.values, d = steps.pop(0)
                self.motors.get_distance.return_value = d
                self.motors.state['dist_at'] = 100.0 if d is not None else 0.0  # None = stale -> 999
            else:
                self.monitor.values = None
            return real(distance)

        self.heat.decide = scripted
        with patch('builtins.print'):
            self.robot.run_mission('explore', heat=self.heat)
        self.assertEqual(self.motors.actions[-1], ('stop', ()))  # every end stops the motors

    def trace(self):
        with open(self.heat.trace_path) as f:
            return [json.loads(line) for line in f]

    def test_pause_cycle_and_resume_with_lower_frame_number(self):
        R, m = self.R, self.motors
        self.frames = [50, 1]
        self.run_script([
            (0, (0, 36.0, 30, 60), 400),     # ACTIVE: one YOLO cycle on frame 50
            (61, (0, 37.0, 30, 60), 400),    # -> PAUSE: motors stop, camera stop; sensor-only FORWARD at 100
            (70, (0, 36.0, 30, 60), 20),     # PAUSE, obstacle: navigate_rules([], 20) ladder
            (80, (0, 36.0, 30, 60), None),   # PAUSE, stale distance (999): rotate only
            (91, (0, 35.0, 30, 60), 400),    # -> ACTIVE (30 s after stop): camera start, frame 1 accepted
        ])
        self.assertEqual(m.actions, [
            ('forward', (R.MOTOR_SPEED, R.CYCLE_MOVE_TIME)),
            ('stop', ()), ('am', 'broadcast'),
            ('forward', (H.PAUSE_FORWARD_SPEED, R.CYCLE_MOVE_TIME)),
            ('strafe_left', (100, R.CYCLE_MOVE_TIME)),
            ('rotate_left', (120, R.CYCLE_MOVE_TIME)),
            ('am', 'start'),
            ('forward', (R.MOTOR_SPEED, R.CYCLE_MOVE_TIME)),
            ('stop', ())])
        self.robot.stop_camera.assert_called_once_with()
        self.robot.start_camera.assert_called_once_with(self.heat)
        self.assertEqual(self.detector.detect.call_count, 2)  # only the two ACTIVE cycles
        self.assertEqual((self.robot.camera_session, self.robot.camera_frame), ('s2', 1))
        self.assertEqual(self.robot.last_moves, ['FORWARD', 'FORWARD', 'STRAFE_LEFT', 'LEFT', 'FORWARD'])
        events = [(r['event'], r['phase']) for r in self.trace()]
        self.assertEqual(events, [('start', 'ACTIVE'), ('camera_stop_ok', 'ACTIVE'), ('active_slot', 'ACTIVE'),
                                  ('phase_change', 'PAUSE'), ('detector_affinity', 'PAUSE'),
                                  ('camera_start_ok', 'PAUSE'), ('phase_change', 'ACTIVE'), ('end', 'ACTIVE'),
                                  ('active_slot', 'ACTIVE')])
        row = self.trace()[3]
        self.assertEqual(set(row), {'t_mono', 't_utc', 'event', 'phase', 'status', 'skin_c', 'battery_c',
                                    'cpu_max_c', 'distance_cm', 'detail'})
        self.assertEqual((row['t_mono'], row['skin_c'], row['distance_cm']), (61, 37.0, 400))

    def test_pause_rules_use_no_yolo_and_stale_distance_never_translates(self):
        rules = Mock(wraps=self.robot.navigate_rules)
        self.robot.navigate_rules = rules
        self.run_script([(61, (0, 38.0, 30, 60), 400)] +
                        [(61 + i, (0, 38.0, 30, 60), None) for i in range(1, 6)] +
                        [(70, (0, 38.0, 30, 60), 400)])
        self.assertEqual([c.args for c in rules.call_args_list], [([], 400), ([], 400)])
        moved = [a for a, _ in self.motors.actions if a not in ('stop', 'am')]
        self.assertEqual(moved, ['forward'] + ['rotate_left'] * 5 + ['forward'])
        self.detector.detect.assert_not_called()

    def test_camera_start_failure_ends_mission(self):
        self.am_fail = 'start'
        self.run_script([(61, (0, 38.0, 30, 60), 400), (91, (0, 34.0, 30, 60), 400)])
        self.assertEqual([a for a, _ in self.motors.actions][-3:], ['forward', 'am', 'stop'])
        self.assertEqual(self.trace()[-1]['event'], 'camera_start_fail')
        self.detector.detect.assert_not_called()

    def test_hold_stops_motors_until_moderate_then_pauses(self):
        self.run_script([
            (0, (0, 36.0, 30, 60), 400),     # ACTIVE cycle
            (10, (4, 47.0, 30, 60), 400),    # CRITICAL -> HOLD (camera off), no moves
            (11, (3, 45.0, 30, 60), 400),    # SEVERE keeps HOLD
            (12, (2, 42.0, 30, 60), 400),    # MODERATE -> PAUSE, not ACTIVE (skin > 35): sensor-only move
        ])
        R, m = self.R, self.motors
        self.assertEqual(m.actions, [
            ('forward', (R.MOTOR_SPEED, R.CYCLE_MOVE_TIME)),
            ('stop', ()), ('am', 'broadcast'), ('stop', ()), ('stop', ()),
            ('forward', (H.PAUSE_FORWARD_SPEED, R.CYCLE_MOVE_TIME)),
            ('stop', ())])
        self.assertEqual(R.time.sleep.call_args_list, [((1,),)] * 2)
        self.robot.start_camera.assert_not_called()
        self.assertEqual([(r['event'], r['phase']) for r in self.trace()][2:6],
                         [('active_slot', 'ACTIVE'), ('hold_start', 'HOLD'), ('note', 'HOLD'), ('hold_end', 'PAUSE')])
        self.assertEqual(self.heat.state['since'], 10)  # pause timer from the camera stop, through the HOLD

    def test_each_end_reason_stops_motors_and_ends(self):
        for fake, words in ((lambda: readings(0, bat=45.0), 'battery'),
                            (lambda: readings(0, cpu=(110, 110, 110)), 'CPU zone >='),
                            (lambda: readings(0, dump_age=60.1), 'fail-closed: no Android status'),
                            (lambda: readings(0, fast_age=5.1), 'fail-closed: no CPU zone')):
            with self.subTest(words=words):
                self.setUp()
                self.monitor.readings = fake
                self.run_script([])
                self.assertEqual(self.motors.actions, [('stop', ())])
                self.assertEqual([r['event'] for r in self.trace()][-2:], ['end', 'active_slot'])
                self.assertIn(words, self.trace()[-2]['detail'])
                self.detector.detect.assert_not_called()

    def test_camera_rate_1_and_detector_trace(self):
        """First start (as run_mission.py does) and a PAUSE -> ACTIVE restart: rate 1 and the detector lines."""
        with patch('builtins.print'):
            self.robot.start_camera(self.heat)
        self.R.get_detector.assert_called_with(pinned=True)
        self.robot.camera_frame = 49
        self.frames = [50, 51, 52, 1, 1]
        self.robot.size_policy.last_large = -100  # first cycle uses 640, then 320 (SizePolicy's real clock)
        self.run_script([
            (0, (0, 36.0, 30, 60), 400), (1, (0, 36.0, 30, 60), 400), (2, (0, 36.0, 30, 60), 400),
            (61, (0, 37.0, 30, 60), 400),     # -> PAUSE: one active_slot
            (91, (0, 35.0, 30, 60), 400),     # -> ACTIVE: restart, frame 1
            (92, (0, 35.0, 30, 60), 400),     # frame 1 again: fail-closed 'no new' stop ends the mission
        ])
        starts = [a for a in self.am_args if a[1] == 'start']
        self.assertEqual(len(starts), 2)
        for a in starts:
            self.assertEqual(a[-6:], ['--es', 'mode', 'B', '--ei', 'rate', '1'])
        rows = self.trace()
        self.assertEqual([r['event'] for r in rows if r['event'].startswith(('detector', 'active'))],
                         ['detector_config', 'detector_affinity', 'active_slot', 'detector_affinity', 'active_slot'])
        config = next(r for r in rows if r['event'] == 'detector_config')['detail']
        self.assertEqual(config, {'ort_version': '1.25.1', 'caller_tid': 7, 'camera_rate': 1})
        affinity = next(r for r in rows if r['event'] == 'detector_affinity')['detail']
        self.assertEqual(set(affinity), {'before', 'repinned', 'after', 'ok', 'errors'})
        first, last = [r['detail'] for r in rows if r['event'] == 'active_slot']
        self.assertEqual(set(first), {'cycles', 'frames', 'detections', 'detect_ms', 'age_s', 'camera_stops',
                                      'affinity'})
        self.assertEqual((first['cycles'], first['frames'], first['detections']), (3, 3, {'320': 2, '640': 1}))
        self.assertEqual(first['detect_ms'], {'320': {'median': 100.0, 'max': 100.0},
                                              '640': {'median': 100.0, 'max': 100.0}})
        self.assertEqual((first['age_s'], first['camera_stops']), ({'median': 0.5, 'max': 0.5}, {}))
        self.assertEqual(first['affinity'], {'7': [4, 5], '8': [4, 5], '9': [4, 5]})
        self.assertEqual((last['cycles'], last['frames'], last['camera_stops']), (2, 1, {'no new': 1}))
        self.assertEqual(self.detector.check_affinity.call_count, 2)
        self.assertEqual(self.detector.config.call_count, 1)

    def test_failed_repin_is_logged_and_mission_continues(self):
        self.affinity = {'before': {7: [0, 1, 2, 3]}, 'repinned': [7], 'after': {7: [0, 1, 2, 3]}, 'ok': False,
                         'errors': {7: 'OSError: [Errno 22] Invalid argument'}}
        with patch('builtins.print') as out:
            self.robot.start_camera(self.heat)
        self.assertIn('affinity re-pin failed', out.call_args_list[0].args[0])
        self.robot.camera_frame, self.frames = 49, [50, 51]
        self.run_script([(0, (0, 36.0, 30, 60), 400), (1, (0, 36.0, 30, 60), 400)])
        self.assertEqual(self.detector.detect.call_count, 2)  # the mission went on
        row = next(r for r in self.trace() if r['event'] == 'detector_affinity')
        self.assertEqual((row['detail']['ok'], row['detail']['errors']), (False, {'7': 'OSError: [Errno 22] Invalid argument'}))

    def test_interrupted_active_slot_is_written_once(self):
        """Ctrl-C, SIGTERM/SIGHUP (SystemExit via run_mission.exit_on_signal) or an error inside an ACTIVE cycle:
        the slot line is still written, exactly once, and the exception reaches the caller."""
        import run_mission

        def keyboard(*a):
            raise KeyboardInterrupt

        def sigterm(*a):
            run_mission.exit_on_signal(15, None)  # what SIGTERM/SIGHUP do in a mission

        for exc, raiser in ((KeyboardInterrupt, keyboard), (SystemExit, sigterm), (RuntimeError, None)):
            with self.subTest(exc=exc.__name__):
                self.setUp()
                self.frames = [50, 51]
                calls = []

                def detect(frame, size):
                    calls.append(size)
                    if len(calls) == 2 and raiser:
                        raiser()
                    return []
                self.detector.detect.side_effect = detect
                if raiser is None:  # an error outside run_cycle's vision try (not turned into STOP)
                    self.R.scene_to_text = Mock(side_effect=['empty room', RuntimeError('boom')])
                with self.assertRaises(exc):
                    self.run_script([(0, (0, 36.0, 30, 60), 400), (1, (0, 36.0, 30, 60), 400)])
                slots = [r['detail'] for r in self.trace() if r['event'] == 'active_slot']
                self.assertEqual(len(slots), 1)
                self.assertEqual((slots[0]['cycles'], slots[0]['frames']), (2, 2))

    def test_slot_not_written_twice_when_pause_entry_fails(self):
        self.heat.enter = Mock(side_effect=RuntimeError('enter failed'))
        with self.assertRaises(RuntimeError):
            self.run_script([(0, (0, 36.0, 30, 60), 400), (61, (0, 37.0, 30, 60), 400)])
        self.assertEqual([r['event'] for r in self.trace()].count('active_slot'), 1)

    def test_run_mission_without_heat_raises(self):
        with self.assertRaises(ValueError):
            self.robot.run_mission('explore')
        self.assertEqual(self.motors.actions, [])

    def test_trace_write_failure_is_not_fatal(self):
        self.heat.trace_path = os.path.join(self.dir.name, 'missing', 'trace.jsonl')
        with patch('builtins.print') as out:
            self.heat.event('x')
        self.assertIn('trace write failed', out.call_args.args[0])


class MonitorTest(unittest.TestCase):
    """HeatMonitor with a fake root shell and a fake root() (no su)."""
    TYPES = ['/sys/class/thermal/thermal_zone9/type\tBIG', '/sys/class/thermal/thermal_zone10/type\tMID',
             '/sys/class/thermal/thermal_zone11/type\tLITTLE', '/sys/class/thermal/thermal_zone25/type\tbattery']

    def shell(self, types, values):
        shell = SimpleNamespace(close=Mock(), scripts=[])

        def run(script, timeout=5):
            shell.scripts.append(script)
            return types if '/type' in script else values
        shell.run = run
        return shell

    def test_start_reads_zones_by_type_and_stops_cleanly(self):
        shell = self.shell(self.TYPES, ['61000', '55000', '50000', '312'])
        with patch.object(H, 'RootShell', return_value=shell), patch.object(H, 'root', return_value=(0, DUMP_TEXT)):
            m = H.HeatMonitor()
            m.start()
            r = m.readings()
            m.stop()
        self.assertEqual((r['dump']['status'], r['dump']['skin']), (0, 31.4255))
        self.assertEqual((r['fast'][-1]['cpu_c'], r['fast'][-1]['bat_c']), (61.0, 31.2))
        self.assertIn(' '.join(f'/sys/class/thermal/thermal_zone{i}/temp' for i in (9, 10, 11)) + ' ' + H.BATTERY_TEMP,
                      shell.scripts[1])
        self.assertFalse(m.thread.is_alive())
        shell.close.assert_called_once_with()

    def test_start_fails_without_cpu_zones_or_first_reading(self):
        for types, values, dump in ((self.TYPES[:2], [], DUMP_TEXT),
                                    (self.TYPES, ['61000', '', '50000', '312'], DUMP_TEXT),
                                    (self.TYPES, ['61000', '55000', '50000', '312'], DUMP_TEXT.replace(HAL_SKIN, ''))):
            with self.subTest(types=len(types), values=values):
                shell = self.shell(types, values)
                with patch.object(H, 'RootShell', return_value=shell), patch.object(
                        H, 'root', return_value=(0, dump)), patch.object(H, 'START_WAIT_S', 0.5):
                    with self.assertRaises(RuntimeError):
                        H.HeatMonitor().start()
                shell.close.assert_called_once_with()

    def test_dump_errors_are_traced_not_raised(self):
        with patch.object(H, 'root', side_effect=subprocess.TimeoutExpired('su', 10)):
            d = H.read_dump()
        self.assertEqual((d['status'], d['skin']), (None, None))
        self.assertIn('TimeoutExpired', d['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
