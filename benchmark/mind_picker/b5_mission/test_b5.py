"""Offline tests for b5.py with fakes and a fake clock: scripted offers, the queue of one, both schedulers (A ticks,
B skin), cancel and requeue, latency accounting, the mission loop's heat transitions (pause start/resume, SEVERE,
END, camera failures, battery stop), real person events, pause accounting, labels, agreement, sessions, and one short
real-time arm with every resource faked (deployed HeatControl over a fake monitor). No server, camera, root or phone."""
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b5  # noqa: E402
import test_b4  # noqa: E402  B4's fixtures: good() readings, watch(), FakeClock

b4, main, hc = b5.b4, b5.main, b5.heat_control
ITEMS = [{'id': i, 'mission': 'M1', 'current': 'explore', 'event': f'e{i}', 'memory': None,
          'allowed': list(b5.b2.NAMES)} for i in range(1, 41)]


def mission(arm='A', work_s=1800, t0=0.0):
    m = b5.Mission(arm, t0, work_s, ITEMS)
    m.set_ok(True)
    return m


class Offers(unittest.TestCase):
    def test_180_offers_in_fixed_cycled_order(self):
        m = mission()
        for t in range(0, 2000):
            m.offer_due(float(t))
        ids = [e['id'] for e in m.events]
        self.assertEqual(len(ids), 180)
        self.assertEqual(ids[:41], list(range(1, 41)) + [1])
        self.assertEqual([e['due'] for e in m.events][-1], 1790)

    def test_smoke_180_s_offers_18(self):
        m = mission(work_s=b5.SMOKE_S)
        m.offer_due(10 ** 6)
        self.assertEqual((b5.SMOKE_S, len(m.events)), (180, 18))

    def test_queue_of_one_drops_the_older(self):
        m = mission('B')
        m.offer({'source': 'scripted', 'id': 1}, 1.0)
        m.offer({'source': 'real', 'id': 'R1'}, 2.0)
        self.assertEqual([e['fate'] for e in m.events], ['dropped', 'pending'])
        ev, _ = m.start(2.0, 30.0)
        self.assertEqual(ev['id'], 'R1')
        m.close()
        self.assertEqual(m.events[1]['fate'], 'deciding')


class Schedulers(unittest.TestCase):
    def test_a_decides_only_at_30_s_marks(self):
        m = mission('A')
        m.offer_due(0.0)
        self.assertEqual(m.start(0.0, 30.0)[0]['id'], 1)
        m.done(m.events[0], {'top': 'explore'}, 4.0)
        m.offer_due(10.0)
        self.assertIsNone(m.start(10.0, 30.0))  # pending, but no mark
        m.offer_due(20.0)
        self.assertIsNone(m.start(29.9, 30.0))
        m.offer_due(30.0)
        ev, _ = m.start(30.05, 30.0)
        self.assertEqual(ev['id'], 4)  # 2 and 3 were replaced
        self.assertEqual([e['fate'] for e in m.events[1:3]], ['dropped', 'dropped'])
        self.assertIsNone(m.start(60.0, 30.0))  # busy at the mark
        m.done(ev, {'top': 'wait'}, 61.0)
        self.assertEqual(m.ticks, {'decided': 2, 'empty': 0, 'paused': 0, 'busy': 1})

    def test_a_tick_in_pause_keeps_the_event_pending(self):
        m = mission('A')
        m.offer_due(0.0)
        m.set_ok(False)
        self.assertIsNone(m.start(0.0, 30.0))
        self.assertEqual(m.pending['id'], 1)
        m.set_ok(True)
        self.assertIsNone(m.start(5.0, 30.0))  # waits for the next mark
        self.assertEqual(m.start(30.0, 30.0)[0]['id'], 1)
        self.assertEqual(m.ticks['paused'], 1)

    def test_a_empty_tick(self):
        m = mission('A')
        self.assertIsNone(m.start(0.0, 30.0))
        self.assertEqual(m.ticks['empty'], 1)

    def test_b_at_once_while_allowed_idle_and_skin_below_36(self):
        m = mission('B')
        m.offer_due(0.0)
        self.assertIsNone(m.start(0.0, 36.0))
        self.assertIsNone(m.start(0.0, None))
        m.set_ok(False)
        self.assertIsNone(m.start(0.1, 35.9))
        m.set_ok(True)
        ev, gen = m.start(0.2, 35.9)
        self.assertEqual(ev['id'], 1)
        m.offer_due(10.0)
        self.assertIsNone(m.start(10.0, 30.0))  # previous decision not finished
        m.done(ev, {'top': 'explore'}, 10.5)
        self.assertEqual(m.start(10.6, 30.0)[0]['id'], 2)

    def test_nothing_starts_after_the_deadline(self):
        m = mission('B', work_s=60)
        m.offer({'source': 'real', 'id': 'R1'}, 59.0)
        self.assertIsNone(m.start(60.0, 30.0))
        m.close()
        self.assertEqual(m.events[0]['fate'], 'unserved')


class Cancel(unittest.TestCase):
    def test_forbid_cancels_for_good_and_requeues(self):
        m = mission('B')
        m.offer({'source': 'scripted', 'id': 1}, 0.0)
        ev, gen = m.start(0.0, 30.0)
        self.assertTrue(m.allowed(gen))
        m.set_ok(False)
        m.set_ok(True)
        self.assertFalse(m.allowed(gen))  # a later allow does not revive it
        m.done(ev, {'cancelled': True}, 2.0)
        self.assertIs(m.pending, ev)
        ev2, gen2 = m.start(40.0, 30.0)
        m.done(ev2, {'top': 'look'}, 44.0)
        self.assertEqual((ev['fate'], ev['latency_s'], ev['wait_s'], ev['attempts']), ('served', 44.0, 40.0, 2))

    def test_cancelled_with_a_newer_event_pending(self):
        m = mission('B')
        m.offer({'source': 'scripted', 'id': 1}, 0.0)
        ev, _ = m.start(0.0, 30.0)
        m.offer({'source': 'real', 'id': 'R1'}, 1.0)
        m.done(ev, {'cancelled': True}, 2.0)
        self.assertEqual((ev['fate'], m.pending['id']), ('cancelled', 'R1'))

    def test_latency_and_errors(self):
        m = mission('B', t0=100.0)
        m.offer({'source': 'scripted', 'id': 1}, 105.0)
        ev, _ = m.start(107.0, 30.0)
        m.done(ev, {'top': 'wait'}, 111.0)
        self.assertEqual((ev['offered'], ev['started'], ev['ended'], ev['latency_s'], ev['wait_s']),
                         (5.0, 7.0, 11.0, 6.0, 2.0))
        m.offer({'source': 'scripted', 'id': 2}, 112.0)
        ev, _ = m.start(112.0, 30.0)
        m.done(ev, {'error': 'x'}, 113.0)
        self.assertEqual(ev['fate'], 'error')


class Decide(unittest.TestCase):
    def setUp(self):
        self.saved = b5.b2.score, b5.b2.preflight, b4.slot
        self.calls = []
        b4.slot = lambda action: self.calls.append(action) or {'action': action, 'wall_s': 0.01}

        def score(post, it, v, order, prompt):
            post('/completion', {})
            post('/completion', {})
            return {'top': 'explore', 'reason': 'r', 'reason_tokens': 3}
        b5.b2.score = score
        b5.b2.preflight = lambda post, s, blocks, info, tag: {('V3', 'FWD', it['id']): ('t', [1], {})
                                                               for it in s['items']}
        self.saved_post = b4.post
        b4.post = lambda path, body: {'timings': {'prompt_n': 5}}

    def tearDown(self):
        b5.b2.score, b5.b2.preflight, b4.slot = self.saved
        b4.post = self.saved_post

    def ev(self, n, it):
        return {'source': 'scripted' if isinstance(it['id'], int) else 'real', 'item': it, 'n': n}

    def test_restore_every_5th_and_real_render(self):
        prompts = {('V3', 'FWD', 1): ('t', [1], {})}
        r = b5.decide(self.ev(5, ITEMS[0]), lambda: True, prompts, {'missions': {}})
        self.assertEqual((r['top'], self.calls), ('explore', ['restore']))
        r = b5.decide(self.ev(6, b5.real_item(1, 'A person appeared far away.')), lambda: True, prompts,
                      {'missions': {}})
        self.assertIn('render_s', r)
        self.assertEqual(self.calls, ['restore'])

    def flip_on_final(self, action):
        """b4.post that runs action() while the second (final) /completion is in flight."""
        n = []

        def post(path, body):
            if path == '/completion':
                n.append(1)
                if len(n) == 2:
                    action()
            return {'timings': {'prompt_n': 5}}
        b4.post = post

    def test_forbid_during_the_final_completion_cancels(self):
        allowed = [True]
        self.flip_on_final(lambda: allowed.__setitem__(0, False))
        r = b5.decide(self.ev(1, ITEMS[0]), lambda: allowed[0], {('V3', 'FWD', 1): ('t', [1], {})}, {})
        self.assertTrue(r['cancelled'])
        self.assertNotIn('top', r)

    def test_forbid_then_allow_before_the_response_still_cancels(self):
        m = mission('B')
        m.offer({'source': 'scripted', 'id': 1, 'item': ITEMS[0]}, 0.0)
        ev, gen = m.start(0.0, 30.0)
        self.flip_on_final(lambda: (m.set_ok(False), m.set_ok(True)))
        r = b5.decide(ev, lambda: m.allowed(gen), {('V3', 'FWD', 1): ('t', [1], {})}, {})
        self.assertTrue(r['cancelled'])
        m.done(ev, r, 5.0)
        self.assertIs(m.pending, ev)  # the existing requeue rule

    def test_forbid_after_the_last_response_while_scoring_cancels(self):
        allowed = [True]

        def score(post, it, v, order, prompt):  # b2.score's own work after its last request
            post('/completion', {})
            post('/completion', {})
            allowed[0] = False
            return {'top': 'explore', 'reason': 'r', 'reason_tokens': 3}
        b5.b2.score = score
        r = b5.decide(self.ev(1, ITEMS[0]), lambda: allowed[0], {('V3', 'FWD', 1): ('t', [1], {})}, {})
        self.assertTrue(r['cancelled'])

    def test_forbid_during_the_restore_cancels(self):
        allowed = [True]
        b4.slot = lambda action: allowed.__setitem__(0, False) or {'action': action, 'wall_s': 0.01}
        r = b5.decide(self.ev(5, ITEMS[0]), lambda: allowed[0], {('V3', 'FWD', 1): ('t', [1], {})}, {})
        self.assertTrue(r['cancelled'])

    def test_forbidden_before_a_request_is_cancelled(self):
        prompts = {('V3', 'FWD', 1): ('t', [1], {})}
        r = b5.decide(self.ev(1, ITEMS[0]), lambda: False, prompts, {})
        self.assertTrue(r['cancelled'])
        self.assertNotIn('error', r)
        r = b5.decide(self.ev(5, ITEMS[0]), lambda: False, prompts, {})
        self.assertTrue(r['cancelled'])
        self.assertEqual(self.calls, [])


# ---------------------------------------------------------------- mission loop

class FakeHeat:
    """Scripted heat.decide() targets by fake time: [(from_t, target)]; status by time too."""

    def __init__(self, clock, plan, status=()):
        self.clock, self.plan, self.status, self.phase, self.since = clock, plan, list(status), 'ACTIVE', clock()
        self.events, self.monitor = [], self

    def decide(self, distance):
        target = [t for at, t in self.plan if self.clock() >= at][-1]
        return target, f'plan {target}'

    def readings(self):
        st = [s for at, s in self.status if self.clock() >= at]
        return {'dump': {'status': st[-1] if st else 0, 'skin': 30.0}, 'fast': []}

    def enter(self, phase, since=None):
        self.phase = phase

    def now(self):
        return self.clock()

    def event(self, name, detail=None):
        self.events.append(name)


class Image:
    size, height = (640, 480), 480


class FakeRobot:
    def __init__(self, clock, persons=None, fail_start=False, start_s=0.0):
        self.clock, self.persons, self.fail_start, self.start_s = clock, persons or {}, fail_start, start_s
        self.camera_session, self.capturing, self.starts, self.stops, self.frame = 's', True, 0, 0, 0
        self.detector, self.size_policy = self, self

    def start_camera(self, heat=None):
        self.starts += 1
        self.capturing = True
        self.clock.sleep(self.start_s)  # a restart takes time
        if self.fail_start:
            raise RuntimeError('RobotCam did not publish a usable frame')

    def stop_camera(self):
        self.stops += 1
        self.capturing = False
        if getattr(self, 'm', None):
            self.ok_at_stop = self.m.ok

    def read(self, **kw):
        if not self.capturing:
            return {'status': 'missing'}
        self.frame += 1
        return {'status': 'ok', 'session': 's', 'frame': self.frame, 'age_s': 0.5, 'image': Image()}

    def next_size(self):
        return 640 if self.frame % 5 == 1 else 320

    def observe(self, dets, h):
        pass

    def detect(self, image, size):
        self.clock.sleep(0.2)
        area = [b for at, b in self.persons.items() if self.clock() >= at]
        side = {None: 0, 'far': 50, 'medium': 200, 'close': 300, 'very close': 500}[area[-1] if area else None]
        return [{'class_name': 'person', 'conf': 0.9, 'box_xyxy': (0, 0, side, side)}] if side else []

    def affinity(self):
        return {1: [4, 5]}

    def config(self):
        return {}


class Loop(unittest.TestCase):
    def setUp(self):
        self.c = test_b4.FakeClock()
        self.saved = b5.clock, b5.pause, b4.cores_ok, main.read_frame
        b5.clock, b5.pause, b4.cores_ok = self.c, self.c.sleep, lambda: True

    def tearDown(self):
        b5.clock, b5.pause, b4.cores_ok, main.read_frame = self.saved

    def run_loop(self, plan, arm='B', work_s=120, status=(), persons=None, fail_start=False, battery=90, start_s=0.0,
                 read_s=0.0, hook=None, ready=None):
        t0 = self.c()
        heat = FakeHeat(self.c, [(t0 + a, x) for a, x in plan], [(t0 + a, s) for a, s in status])
        robot = FakeRobot(self.c, {t0 + a: b for a, b in (persons or {}).items()}, fail_start, start_s)
        main.read_frame = robot.read
        m = b5.Mission(arm, t0, work_s, ITEMS)
        robot.m = m
        rec, rows, oks = {'errors': [], 'camera_stops': [], 'readings': []}, [], []
        cam = {'on': True, 'slot': 1}

        def read(tag):
            rec['readings'].append({'tag': tag, 'camera': cam['on'], 't': m.rel(self.c()), 'ok': m.ok})
            self.c.sleep(read_s)  # a root reading takes time
            if hook:
                hook(tag, m)
            if battery < b5.BATTERY_STOP_PCT:
                rec['errors'].append('battery')
                return False
            return True
        real_set_ok = m.set_ok
        m.set_ok = lambda ok: (oks.append((m.rel(self.c()), ok)), real_set_ok(ok))
        self.last_m = m
        b5.mission_loop(robot, heat, m, rec, rows.append, read, cam, ready or (lambda: None))
        return m, rec, rows, robot, heat, oks, cam

    def test_pause_start_and_resume(self):
        m, rec, rows, robot, heat, oks, cam = self.run_loop([(0, 'ACTIVE'), (20, 'PAUSE'), (60, 'ACTIVE')])
        self.assertEqual(rec['errors'], [])
        self.assertEqual([(r['stage'], r['from'], r['to']) for r in rows if r['kind'] == 'heat'],
                         [('intent', 'ACTIVE', 'PAUSE'), ('done', 'ACTIVE', 'PAUSE'), ('intent', 'PAUSE', 'ACTIVE'),
                          ('done', 'PAUSE', 'ACTIVE')])
        self.assertEqual((robot.stops, robot.starts, cam), (1, 1, {'on': True, 'slot': 2}))
        self.assertIs(robot.ok_at_stop, False)  # picker forbidden before the camera stops (main.py order)
        self.assertEqual([r['tag'] for r in rec['readings']], ['pause_start', 'pause_end'])
        self.assertEqual([r['camera'] for r in rec['readings']], [False, True])
        self.assertEqual(heat.events, ['camera_stop_ok', 'camera_start_ok'])
        cyc = [r['t'] for r in rows if r['kind'] == 'cycle']
        self.assertFalse([t for t in cyc if 20 < t < 60])  # no YOLO in the pause
        self.assertEqual({r['slot'] for r in rows if r['kind'] == 'cycle' and r['t'] > 60}, {2})
        paused = [t for t, ok in oks if not ok]
        self.assertTrue(paused and all(20 <= t < 62 for t in paused))
        self.assertTrue(m.ok)
        self.assertLess(self.c() - m.deadline, 2.0)  # the last move is cut at the deadline
        n, first, total = b5.pauses(rows, 120)
        self.assertTrue(n == 1 and 20 <= first < 22 and 38 < total < 42)

    def test_start_barrier_initially_active_serves_the_0_s_mark(self):
        got = []

        def ready():  # what the scheduler's first poll does once it is started here
            m = self.last_m
            got.append((m.ok, m.offer_due(self.c()), m.start(self.c(), 30.0)))
        m, *_ = self.run_loop([(0, 'ACTIVE')], arm='A', work_s=10, ready=ready)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0][0])
        self.assertEqual(got[0][2][0]['id'], 1)
        self.assertEqual(m.ticks['decided'], 1)

    def test_start_barrier_initially_forbidden_does_not(self):
        got = []

        def ready():
            m = self.last_m
            got.append((m.ok, m.offer_due(self.c()), m.start(self.c(), 30.0)))
        m, *_ = self.run_loop([(0, 'ACTIVE')], arm='A', work_s=10, status=[(0, 3)], ready=ready)
        self.assertEqual((got[0][0], got[0][2], m.ticks['paused']), (False, None, 1))

    def test_end_before_the_first_assessment_never_starts_the_scheduler(self):
        got = []
        self.run_loop([(0, 'END')], ready=lambda: got.append(1))
        self.assertEqual(got, [])

    def test_resume_allows_before_the_reading_a_mark_inside_it_is_served(self):
        seen = {}

        def hook(tag, m):  # the scheduler polling while the pause_end reading runs
            if tag == 'pause_end':
                m.offer_due(self.c())
                seen['go'] = m.start(self.c(), 30.0)
        m, rec, rows, robot, heat, oks, cam = self.run_loop([(0, 'ACTIVE'), (20, 'PAUSE'), (87, 'ACTIVE')], arm='A',
                                                            start_s=2.0, read_s=3.0, hook=hook)
        end = next(r for r in rec['readings'] if r['tag'] == 'pause_end')
        self.assertTrue(end['ok'])  # allowed before the reading
        self.assertTrue(87 < end['t'] < 90 < end['t'] + 3.0)  # the reading spans the 90 s mark
        self.assertIsNotNone(seen['go'])
        self.assertEqual(m.ticks['decided'], 1)

    def test_resume_with_b_event_pending(self):
        seen = {}

        def hook(tag, m):
            m.offer_due(self.c())
            seen[tag] = m.start(self.c(), 30.0)
        m, *_ = self.run_loop([(0, 'ACTIVE'), (20, 'PAUSE'), (60, 'ACTIVE')], start_s=2.0, read_s=3.0, hook=hook)
        self.assertIsNone(seen['pause_start'])  # pending, forbidden
        self.assertEqual(seen['pause_end'][0]['id'], 7)  # the newest pending event (offered at 60 s), at resume

    def test_reading_failure_at_resume_forbids_at_once(self):
        m, rec, *_ = self.run_loop([(0, 'ACTIVE'), (10, 'PAUSE')], battery=20)
        self.assertFalse(m.ok)

        def boom(tag, m):
            if tag == 'pause_end':
                raise RuntimeError('root read')
        with self.assertRaises(RuntimeError):
            self.run_loop([(0, 'ACTIVE'), (10, 'PAUSE'), (50, 'ACTIVE')], hook=boom)
        self.assertFalse(self.last_m.ok)

    def test_pause_counts_restart_time_and_failed_restart_is_no_resume(self):
        m, rec, rows, *_ = self.run_loop([(0, 'ACTIVE'), (20, 'PAUSE'), (60, 'ACTIVE')], start_s=2.0, read_s=3.0)
        intent = [r for r in rows if r['kind'] == 'heat' and r['stage'] == 'intent']
        done = [r for r in rows if r['kind'] == 'heat' and r['stage'] == 'done' and r['to'] == 'ACTIVE']
        self.assertAlmostEqual(done[0]['t'] - intent[1]['t'], 2.0, places=3)  # the restart is pause time
        n, first, total = b5.pauses(rows, 120)
        self.assertAlmostEqual(total, done[0]['t'] - intent[0]['t'], places=3)
        with tempfile.TemporaryDirectory() as d:
            x = b5.metrics({**arm_record(), 'mission_end_s': 120}, rows, {}, Path(d))
        self.assertEqual((x['pauses'], x['resumes']), (1, 1))
        m, rec, rows, *_ = self.run_loop([(0, 'ACTIVE'), (10, 'PAUSE'), (50, 'ACTIVE')], fail_start=True, start_s=2.0)
        n, first, total = b5.pauses(rows, 52.0)
        self.assertEqual((n, round(total, 3)), (1, round(52.0 - first, 3)))  # open to the arm end
        with tempfile.TemporaryDirectory() as d:
            x = b5.metrics({**arm_record(), 'mission_end_s': 52.0}, rows, {}, Path(d))
        self.assertEqual(x['resumes'], 0)

    def test_transition_crossing_the_deadline_starts_no_cycle(self):
        m, rec, rows, robot, *_ = self.run_loop([(0, 'ACTIVE'), (100, 'PAUSE'), (114, 'ACTIVE')], work_s=118,
                                                start_s=2.0, read_s=3.0)
        self.assertEqual((rec['errors'], robot.starts), ([], 1))
        self.assertTrue([r for r in rec['readings'] if r['tag'] == 'pause_end'])  # the transition finished
        self.assertFalse([r for r in rows if r['kind'] == 'cycle' and r['t'] >= 100])

    def test_restart_completing_after_the_deadline_counts_only_mission_time(self):
        m, rec, rows, robot, *_ = self.run_loop([(0, 'ACTIVE'), (100, 'PAUSE'), (115, 'ACTIVE')], work_s=118,
                                                start_s=5.0)
        intent = next(r for r in rows if r['kind'] == 'heat' and r['stage'] == 'intent')
        done = next(r for r in rows if r['kind'] == 'heat' and r['stage'] == 'done' and r['to'] == 'ACTIVE')
        self.assertGreater(done['t'], 118)  # the real completion time is kept
        end_t = min(m.rel(self.c()), 118)
        n, first, total = b5.pauses(rows, end_t)
        self.assertAlmostEqual(total, 118 - intent['t'], places=3)  # inside mission time only
        with tempfile.TemporaryDirectory() as d:
            x = b5.metrics({**arm_record(), 'mission_end_s': end_t}, rows, {}, Path(d))
        self.assertEqual((x['pauses'], x['resumes'], x['pause_s']), (1, 1, round(118 - intent['t'], 3)))
        self.assertAlmostEqual(x['active_share'], round(1 - (118 - intent['t']) / 118, 3), places=3)
        self.assertEqual(b5.pauses([hrow('intent', 'ACTIVE', 'PAUSE', 100.0), hrow('done', 'PAUSE', 'ACTIVE', 1802.0)],
                                   1800.0), (1, 100.0, 1700.0))

    def test_detection_ending_after_the_deadline_offers_no_event(self):
        m, rec, rows, *_ = self.run_loop([(0, 'ACTIVE')], work_s=5.2, persons={5.0: 'far'})
        self.assertEqual(max(r['t'] for r in rows if r['kind'] == 'cycle'), 5.1)
        self.assertEqual((rec['late_not_offered'], [e for e in m.events if e['source'] == 'real']), (1, []))
        with tempfile.TemporaryDirectory() as d:
            x = b5.metrics({**arm_record(), 'late_not_offered': 1}, rows, {}, Path(d))
        self.assertEqual(x['events']['real']['late_not_offered'], 1)

    def test_pause_open_at_the_end_counts_to_the_end(self):
        rows = [hrow('intent', 'ACTIVE', 'PAUSE', 100.0), hrow('done', 'ACTIVE', 'PAUSE', 100.4),
                hrow('intent', 'PAUSE', 'HOLD', 120.0), hrow('done', 'PAUSE', 'HOLD', 120.0),
                hrow('intent', 'HOLD', 'ACTIVE', 150.0),
                hrow('done', 'HOLD', 'ACTIVE', 152.0), hrow('intent', 'ACTIVE', 'PAUSE', 1700.0),
                hrow('done', 'ACTIVE', 'PAUSE', 1700.4)]
        self.assertEqual(b5.pauses(rows, 1800.0), (2, 100.0, 152.0))  # 100 -> 152 (restart done), 1700 -> end

    def test_severe_forbids_without_a_camera_change(self):
        m, rec, rows, robot, *_ = self.run_loop([(0, 'ACTIVE')], status=[(30, 3), (50, 0)], work_s=80)
        self.assertEqual((robot.stops, rec['errors']), (0, []))
        self.assertTrue(any(r['kind'] == 'cycle' and 31 < r['t'] < 49 for r in rows))
        self.assertTrue(m.ok)

    def test_end_stops_the_arm(self):
        m, rec, rows, *_ = self.run_loop([(0, 'ACTIVE'), (10, 'END')])
        self.assertEqual(len(rec['errors']), 1)
        self.assertIn('heat END', rec['errors'][0])
        self.assertLess(m.rel(self.c()), 15)

    def test_camera_start_failure_ends_and_leaves_camera_owned(self):
        m, rec, rows, robot, heat, oks, cam = self.run_loop([(0, 'ACTIVE'), (10, 'PAUSE'), (50, 'ACTIVE')],
                                                            fail_start=True)
        self.assertIn('camera start after pause', rec['errors'][0])
        self.assertTrue(cam['on'])
        self.assertIn('camera_start_fail', heat.events)

    def test_battery_stop_at_a_pause_reading(self):
        m, rec, *_ = self.run_loop([(0, 'ACTIVE'), (10, 'PAUSE')], battery=20)
        self.assertEqual(rec['errors'], ['battery'])

    def test_server_repinned_at_every_camera_start(self):
        calls = []
        t0 = self.c()
        heat = FakeHeat(self.c, [(t0, 'ACTIVE'), (t0 + 10, 'PAUSE'), (t0 + 50, 'ACTIVE'), (t0 + 70, 'PAUSE'),
                                 (t0 + 110, 'ACTIVE')])
        robot = FakeRobot(self.c)
        main.read_frame = robot.read
        m = b5.Mission('B', t0, 120, ITEMS)
        rec = {'errors': [], 'camera_stops': [], 'readings': []}
        b5.mission_loop(robot, heat, m, rec, lambda r: None, lambda t: True, {'on': True, 'slot': 1},
                        repin=lambda: calls.append(robot.capturing))
        self.assertEqual(calls, [True, True])  # after each successful camera start
        robot = FakeRobot(self.c, fail_start=True)
        main.read_frame = robot.read
        calls.clear()
        t0 = self.c()
        heat = FakeHeat(self.c, [(t0, 'ACTIVE'), (t0 + 10, 'PAUSE'), (t0 + 50, 'ACTIVE')])
        b5.mission_loop(robot, heat, b5.Mission('B', t0, 120, ITEMS), {'errors': [], 'camera_stops': [],
                        'readings': []}, lambda r: None, lambda t: True, {'on': True, 'slot': 1},
                        repin=lambda: calls.append(1))
        self.assertEqual(calls, [])  # a failed start is not followed by a re-pin

    def test_hold_and_pause_hold(self):
        m, rec, rows, robot, *_ = self.run_loop([(0, 'ACTIVE'), (10, 'HOLD'), (20, 'PAUSE'), (40, 'ACTIVE')])
        self.assertEqual([(r['from'], r['to']) for r in rows if r['kind'] == 'heat' and r['stage'] == 'intent'],
                         [('ACTIVE', 'HOLD'), ('HOLD', 'PAUSE'), ('PAUSE', 'ACTIVE')])
        self.assertEqual((robot.stops, robot.starts), (1, 1))

    def test_real_person_events(self):
        m, rec, rows, *_ = self.run_loop([(0, 'ACTIVE')], persons={5: 'far', 12: 'close', 20: None}, work_s=30)
        texts = [e['text'] for e in m.events if e['source'] == 'real']
        self.assertEqual(texts, ['A person appeared far away.', 'The person is now close.',
                                 'The person left your view.'])
        self.assertEqual(m.events[0]['item']['event'], 'A person appeared far away.')
        self.assertEqual(m.events[0]['item']['memory'], None)

    def test_no_new_frames_limit(self):
        t0 = self.c()
        heat = FakeHeat(self.c, [(t0, 'ACTIVE')])
        robot = FakeRobot(self.c)
        main.read_frame = lambda **kw: {'status': 'ok', 'session': 's', 'frame': 7, 'age_s': 0.1, 'image': Image()}
        m = b5.Mission('A', t0, 60, ITEMS)
        rec, rows = {'errors': [], 'camera_stops': [], 'readings': []}, []
        b5.mission_loop(robot, heat, m, rec, rows.append, lambda t: True, {'on': True, 'slot': 1})
        self.assertEqual([r['reason'] for r in rows if r['kind'] == 'camera_stop'], ['no new'] * 4)
        self.assertIn('camera stop: no new', rec['errors'][0])


# ---------------------------------------------------------------- labels, metrics

def arm_record(**kw):
    good = test_b4.good
    rec = {'arm': 'A', 'errors': [], 'cores_before': True, 'cores_after': True, 'agents_before': '',
           'agents_after': '', 'overrun_s': 3.0, 'watch': test_b4.watch(), 'work_s': 1800, 'mission_end_s': 1800,
           'readings': [{**good(mem=('runner', 'llama_server', 'robotcam')), 'tag': 'start', 'camera': True, 't': 0},
                        {**good(mem=('runner', 'llama_server')), 'tag': 'pause_start', 'camera': False, 't': 400},
                        {**good(mem=('runner', 'llama_server', 'robotcam')), 'tag': 'pause_end', 'camera': True,
                         't': 500},
                        {**good(mem=('runner', 'llama_server', 'robotcam'), mono=1800), 'tag': 'end',
                         'camera': True, 't': 1800}]}
    return {**rec, **kw}


CYC = {'kind': 'cycle', 'cores_ok': True, 'slot': 1}


def hrow(stage, frm, to, t):
    return {'kind': 'heat', 'stage': stage, 'from': frm, 'to': to, 't': t}


class Repin(unittest.TestCase):
    """Owner decision 2026-11-15: re-pin like a robot; drift corrected by one re-pin is VALID and recorded."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.proc = Path(self.d.name)
        for tid in (11, 12, 13):
            (self.proc / '4242' / 'task' / str(tid)).mkdir(parents=True)
        self.aff = {11: {0, 1, 2, 3, 4, 5, 6, 7}, 12: {4, 5, 6, 7}, 13: {0, 1, 2, 3, 4, 5, 6, 7}}
        self.patches = [unittest.mock.patch.object(b4, 'PROC', self.proc),
                        unittest.mock.patch.object(b5.os, 'sched_getaffinity', self.get),
                        unittest.mock.patch.object(b5.os, 'sched_setaffinity', self.set)]
        for p in self.patches:
            p.start()
        self.fail = set()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.d.cleanup()

    def get(self, tid):
        if tid not in self.aff:
            raise ProcessLookupError(tid)
        return set(self.aff[tid])

    def set(self, tid, cpus):
        if tid in self.fail:
            raise PermissionError('EPERM')
        self.aff[tid] = set(cpus)

    def test_server_repin_records_before_after(self):
        rp = b5.Repinner({'server_pid': 4242})
        e = rp.server('watch')
        self.assertTrue(e['ok'])
        self.assertEqual((e['threads'], e['before'][11], e['after'][11]), ([11, 13], [0, 1, 2, 3, 4, 5, 6, 7],
                                                                           [4, 5, 6, 7]))
        self.assertEqual({t: sorted(c) for t, c in self.aff.items()}, {t: [4, 5, 6, 7] for t in (11, 12, 13)})
        self.assertEqual(rp.events, [e])

    def test_server_repin_failure_and_vanished_thread(self):
        self.fail.add(13)
        del self.aff[12]  # ended after the listing: skipped
        e = b5.Repinner({'server_pid': 4242}).server('arm_start')
        self.assertFalse(e['ok'])
        self.assertIn(13, e['errors'])
        self.assertNotIn(12, e['before'])

    def test_concurrent_repins_each_caller_gets_its_own_event(self):
        class HookLock:  # releases, then lets the mission thread re-pin before the watcher's add() returns
            def __init__(self):
                self.lock, self.hook = threading.Lock(), None

            def __enter__(self):
                self.lock.acquire()

            def __exit__(self, *exc):
                self.lock.release()
                hook, self.hook = self.hook, None
                if hook:
                    t = threading.Thread(target=hook)
                    t.start()
                    t.join()
        rp = b5.Repinner({'server_pid': 4242})
        rp.lock, got = HookLock(), {}
        rp.lock.hook = lambda: got.__setitem__('camera', rp.server('camera_start'))
        hit = 'llama-server thread 11 Cpus_allowed_list:\t0-7'
        with unittest.mock.patch.object(b4, 'watch_check', lambda t: {'mono': 1.0, 'hits': [hit], 'vanished': 0}):
            x = b5.scan({'server_pid': 4242}, rp)  # the watcher
        self.assertEqual([(e['trigger'], e['process']) for e in x['repins']], [('watch', 'llama_server')])
        self.assertEqual(x['repins'][0]['threads'], [11, 13])  # the watcher's correction, not the later no-op
        self.assertEqual(got['camera']['trigger'], 'camera_start')
        self.assertEqual([e['trigger'] for e in rp.events], ['watch', 'camera_start'])
        self.assertIs(rp.events[0], x['repins'][0])

    def test_detector_repin_uses_check_affinity(self):
        det = unittest.mock.Mock()
        det.check_affinity.return_value = {'before': {1: [0, 1]}, 'repinned': [1], 'after': {1: [4, 5]}, 'ok': True,
                                           'errors': {}}
        e = b5.Repinner({'detector': det}).detector('watch')
        self.assertEqual((e['process'], e['threads'], e['ok']), ('detector', [1], True))

    def test_camera_start_detector_repin_is_recorded(self):
        seen = []
        heat = unittest.mock.Mock()
        heat.event = lambda name, detail=None, dump=None: seen.append(name)
        rp = b5.Repinner({})
        b5.record_camera_repins(heat, rp)
        heat.event('detector_affinity', {'before': {}, 'repinned': [7], 'after': {}, 'ok': False, 'errors': {7: 'x'}})
        heat.event('camera_start_ok')
        self.assertEqual(seen, ['detector_affinity', 'camera_start_ok'])
        self.assertEqual([(e['trigger'], e['ok']) for e in rp.events], [('camera_start', False)])

    def test_groups_recorded(self):
        (self.proc / '4242' / 'cpuset').write_text('/top-app\n')
        (self.proc / '4242' / 'cgroup').write_text('0::/uid_1/pid_4242\n')
        g = b5.groups({'server_pid': 4242})
        self.assertEqual((g['llama_server_cpuset'], g['llama_server_cgroup']), ('/top-app', '0::/uid_1/pid_4242'))
        self.assertIn('Error', g['runner_cpuset'])  # not in the fake tree: error text, not a crash

    def test_scan_splits_drift_and_repins_it(self):
        hits = ['agent pid 9: claude', 'llama-server thread 11 Cpus_allowed_list:\t0-7',
                'llama-server thread 13 affinity unreadable: x', "detector threads {5: [0, 1, 2, 3, 4, 5, 6, 7]}"]
        rp = unittest.mock.Mock()
        rp.server.return_value, rp.detector.return_value = {'p': 's'}, {'p': 'd'}
        with unittest.mock.patch.object(b4, 'watch_check', lambda t: {'mono': 1.0, 'hits': list(hits), 'vanished': 0}):
            x = b5.scan({'server_pid': 4242}, rp)
        self.assertEqual(x['hits'], [hits[0], hits[2]])  # not drift: still hits
        self.assertEqual(sorted(x['drift']), ['detector', 'llama_server'])
        self.assertEqual(x['repins'], [{'p': 'd'}, {'p': 's'}])
        self.assertIn('llama_server_cpuset', x['groups'])

    def watch(self, drifts):
        return {**test_b4.watch(), 'samples': [{'mono': 30.0 * i, 'hits': [], 'drift': {p: ['x'] for p in d}}
                                               for i, d in enumerate(drifts)]}

    def test_summary(self):
        rec = arm_record(watch={**self.watch([['llama_server'], []]), 'samples': [
            {'mono': 0, 'hits': [], 'drift': {'llama_server': ['x']}, 'groups': {'llama_server_cpuset': '/top-app'}},
            {'mono': 30, 'hits': [], 'drift': {}, 'groups': {'llama_server_cpuset': '/foreground'}}]},
            repins=[{'trigger': 'watch', 'process': 'llama_server', 'threads': [11], 'ok': True},
                    {'trigger': 'arm_start', 'process': 'llama_server', 'threads': [], 'ok': True}])
        x = b5.repin_summary(rec)
        self.assertEqual(x, {'acted': {'watch/llama_server': 1}, 'failed': 0, 'drifted_scans': {'llama_server': 1},
                             'groups_seen': {'llama_server_cpuset': ['/foreground', '/top-app']}})

    def test_validity(self):
        ok = {'trigger': 'watch', 'process': 'llama_server', 'ok': True}
        bad = {**ok, 'ok': False, 'errors': {13: 'EPERM'}}
        # drift corrected by one re-pin: VALID, the event recorded
        rec = arm_record(watch=self.watch([[], ['llama_server'], [], ['detector'], []]), repins=[ok, ok])
        self.assertEqual(b5.arm_problems(rec, [CYC]), [])
        # the same process drifted in two consecutive scans: INVALID
        rec = arm_record(watch=self.watch([[], ['llama_server'], ['llama_server'], []]), repins=[ok, ok])
        self.assertTrue(any('two consecutive scans' in p for p in b5.arm_problems(rec, [CYC])))
        # different processes in consecutive scans: each corrected once, VALID
        rec = arm_record(watch=self.watch([['llama_server'], ['detector']]), repins=[ok, ok])
        self.assertEqual(b5.arm_problems(rec, [CYC]), [])
        # a failed re-pin (watch, arm start or camera start): INVALID
        for trig in ('watch', 'arm_start', 'camera_start'):
            rec = arm_record(watch=self.watch([[]]), repins=[{**bad, 'trigger': trig}])
            self.assertTrue(any('re-pin failed' in p for p in b5.arm_problems(rec, [CYC])), trig)
        rec = arm_record(repins=[bad])
        del rec['watch']
        self.assertTrue(any('re-pin failed' in p for p in b5.arm_problems(rec, [CYC])))


class Labels(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(b5.arm_problems(arm_record(), [CYC]), [])

    def test_invalid(self):
        bad = [arm_record(errors=['heat END: x']), arm_record(agents_after='1 claude'), arm_record(cores_before=False),
               arm_record(overrun_s=15.5), arm_record(watch=test_b4.watch(['agent pid 3'])),
               arm_record(readings=arm_record()['readings'][1:]), arm_record(camera_end={'error': 'not stopped'})]
        for rec in bad:
            self.assertTrue(b5.arm_problems(rec, [CYC]), rec)
        self.assertTrue(b5.arm_problems(arm_record(), []))  # no camera cycles
        self.assertTrue(b5.arm_problems(arm_record(), [CYC, {**CYC, 'cores_ok': False}]))
        self.assertTrue(b5.arm_problems(arm_record(), [CYC, {'kind': 'decision', 'error': 'x'}]))
        self.assertTrue(b5.arm_problems(arm_record(), [CYC, {'kind': 'camera_stop', 'reason': 'stale'}]))
        self.assertEqual(b5.arm_problems(arm_record(), [CYC] + [{'kind': 'camera_stop', 'reason': 'no new'}] * 3), [])
        self.assertTrue(b5.arm_problems(arm_record(), [CYC] + [{'kind': 'camera_stop', 'reason': 'no new'}] * 4))

    def test_robotcam_memory_needed_while_the_camera_is_on(self):
        r = arm_record()
        r['readings'][2] = {**test_b4.good(mem=('runner', 'llama_server')), 'tag': 'pause_end', 'camera': True}
        self.assertTrue(any('robotcam' in p for p in b5.arm_problems(r, [CYC])))
        r['readings'][1]['mem_runner'] = {'Pss': 1, 'pids': [1], 'rc': 0, 'missing': ['pid 1 VmHWM']}
        self.assertTrue(any('pause_start' in p for p in b5.arm_problems(r, [CYC])))


class Metrics(unittest.TestCase):
    def test_counts_latency_agreement(self):
        ev = [{'kind': 'event', 'source': 'scripted', 'fate': 'served', 'latency_s': 4.0, 'wait_s': 0.1},
              {'kind': 'event', 'source': 'scripted', 'fate': 'dropped'},
              {'kind': 'event', 'source': 'real', 'fate': 'served', 'latency_s': 8.0, 'wait_s': 4.0},
              {'kind': 'event', 'source': 'real', 'fate': 'unserved'}]
        dec = [{'kind': 'decision', 'source': 'scripted', 'id': 2, 'top': 'explore', 'wall_s': 3.9},
               {'kind': 'decision', 'source': 'real', 'id': 'R1', 'top': 'approach', 'wall_s': 4.2},
               {'kind': 'decision', 'source': 'scripted', 'id': 3, 'cancelled': True, 'wall_s': 1.0}]
        cyc = [{**CYC, 't': 0.0, 'size': 640, 'detect_ms': 500}, {**CYC, 't': 1.7, 'size': 320, 'detect_ms': 140},
               {**CYC, 't': 900.0, 'slot': 2, 'size': 640, 'detect_ms': 520},
               {**CYC, 't': 901.8, 'slot': 2, 'size': 320, 'detect_ms': 150},
               hrow('intent', 'ACTIVE', 'PAUSE', 600.0), hrow('done', 'PAUSE', 'ACTIVE', 899.0)]
        with tempfile.TemporaryDirectory() as d:
            x = b5.metrics(arm_record(), ev + dec + cyc, {2: 'wait'}, Path(d))
        self.assertEqual(x['decisions_served'], 2)
        self.assertEqual(x['events']['scripted'], {'served': 1, 'dropped': 1, 'cancelled': 0, 'unserved': 0, 'error': 0,
                                                   'offered': 2})
        self.assertEqual(x['latency_s']['all']['n'], 2)
        self.assertEqual(x['agreement'], {'compared': 1, 'agree': 0, 'ids_differ': [2]})
        self.assertEqual((x['pauses'], x['first_pause_s'], x['pause_s'], x['active_share']), (1, 600.0, 299.0, 0.834))
        self.assertEqual(x['cycle_period_s']['n'], 2)
        self.assertEqual(x['detect_ms'][640]['n'], 2)
        self.assertEqual(x['mem_available_min_kb'], None)

    def test_trace_max(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 't.jsonl'
            p.write_text('\n'.join(json.dumps(r) for r in [{'event': 'dumpsys', 'skin_c': 36.1},
                                                           {'event': 'dumpsys', 'skin_c': 37.4},
                                                           {'event': 'start', 'skin_c': 40.0}]))
            self.assertEqual(b5.trace_max(p), 37.4)

    def test_record_pinned(self):
        r = b5.load_record()
        self.assertEqual((len(r), r[2], r[7]), (40, 'wait', 'approach'))


# ---------------------------------------------------------------- sessions

class Sessions(unittest.TestCase):
    def setUp(self):
        self.saved = b5.gate, b5.run_arm, b4.reading, b4.cores_ok, b5.repro.agents, b5.cr.wait_cores
        self.ran = []
        b4.reading = lambda **kw: test_b4.good()
        b4.cores_ok = lambda: True
        b5.repro.agents = lambda: ''
        b5.cr.wait_cores = lambda label: None
        b5.gate = lambda ref, label: {'passed': True}

    def tearDown(self):
        b5.gate, b5.run_arm, b4.reading, b4.cores_ok, b5.repro.agents, b5.cr.wait_cores = self.saved

    def fake_arms(self, labels, battery_stop=False):
        def run_arm(k, out, ctx, emit):
            self.ran.append(k)
            return {'arm': k, 'label': labels.get(k, 'VALID'), 'problems': [], 'battery_stop': battery_stop}, []
        b5.run_arm = run_arm

    def session(self, out, smoke=False):
        with open(out / 'arms.jsonl', 'a') as alog, unittest.mock.patch('sys.stdout', io.StringIO()):
            return b5.session_loop(out, {'start': {'skin': 30.0}}, {'smoke': smoke, 'session': 1}, lambda r: None, alog)

    def test_order_invalid_stops_and_resume_redoes_it(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.fake_arms({'A': 'INVALID'})
            self.assertIn('arm A INVALID', self.session(out))
            self.assertEqual(self.ran, ['A'])
            self.fake_arms({})
            self.assertEqual(self.session(out), 'all arms done')
            self.assertEqual(self.ran, ['A', 'A', 'B'])

    def test_smoke_gates_only_the_first_arm(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake_arms({k: b5.SMOKE_LABEL for k in b5.ARMS})
            gated = []
            b5.gate = lambda ref, label: gated.append(label) or {'passed': True}
            b5.repro.agents = lambda: '42 claude'
            self.assertEqual(self.session(Path(d), smoke=True), 'all arms done')
            self.assertEqual((self.ran, gated), (['A', 'B'], ['arm A']))

    def test_timed_gates_every_arm_and_safety_before_the_arm(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake_arms({})
            gated = []
            b5.gate = lambda ref, label: gated.append(label) or {'passed': True}
            self.assertEqual(self.session(Path(d)), 'all arms done')
            self.assertEqual(gated, ['arm A', 'arm B'])
        with tempfile.TemporaryDirectory() as d:
            self.ran.clear()
            b4.reading = lambda **kw: test_b4.good(battery=29)
            self.assertIn('stopped before arm A', self.session(Path(d)))
            b4.reading = lambda **kw: test_b4.good()
            b5.repro.agents = lambda: '42 claude'
            self.assertIn('agent processes running', self.session(Path(d)))
            self.assertEqual(self.ran, [])

    def test_battery_stop_in_an_arm_ends_the_session(self):
        with tempfile.TemporaryDirectory() as d:
            self.fake_arms({'A': 'INVALID'}, battery_stop=True)
            self.assertIn('battery below 30 %', self.session(Path(d)))
            self.assertEqual(self.ran, ['A'])

    def test_definition_and_resume_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            old = b5.definition(Path(d))
            self.assertIn('benchmark/mind_picker/b5_mission/oneshot.sh', old['files'])
            self.assertIn('benchmark/mind_picker/b2_rules/runs/20261102T143251Z/requests.jsonl', old['files'])
            self.assertIn('heat_control.py', old['files'])
            self.assertIn('--slot-save-path', old['commands']['server'])
            new = json.loads(json.dumps(old))
            new['files']['heat_control.py'] = '0' * 64
            self.assertEqual(len(b4.definition_diff(old, new)), 1)

    def test_battery_first_start_85_resume_50_smoke_none(self):
        self.assertIsNone(b5.battery_refusal(85, False, None))
        self.assertIn('< 85 %', b5.battery_refusal(84, False, None))
        self.assertIsNone(b5.battery_refusal(50, False, '20261110T000000Z'))
        self.assertIn('< 50 %', b5.battery_refusal(49, False, '20261110T000000Z'))
        self.assertIsNone(b5.battery_refusal(40, True, None))

    def test_smoke_thresholds_lower_the_deployed_constants_at_runtime(self):
        saved = hc.SKIN_PAUSE_C, hc.SKIN_RESUME_C
        try:
            t = b5.smoke_thresholds(31.24)  # the arm-start skin
            self.assertEqual((t['pause_c'], t['resume_c'], t['min_active_s'], t['min_pause_s']), (32.24, 31.74, 60, 30))
            st = {'phase': 'ACTIVE', 'since': 0.0}
            now = 100.0
            rd = {'dump': {'t': now, 'status': 0, 'skin': 32.3}, 'fast': [{'t': now, 'cpu_c': 50.0, 'bat_c': 30.0}]}
            self.assertEqual(hc.decide(now, rd, st)[0], 'PAUSE')
            self.assertEqual(hc.decide(59.0, {**rd, 'dump': {**rd['dump'], 't': 59.0},
                                              'fast': [{'t': 59.0, 'cpu_c': 50.0, 'bat_c': 30.0}]}, st)[0], 'ACTIVE')
            rd['dump']['skin'] = 31.8
            self.assertEqual(hc.decide(now, rd, {'phase': 'PAUSE', 'since': 60.0})[0], 'PAUSE')
            rd['dump']['skin'] = 31.74
            self.assertEqual(hc.decide(now, rd, {'phase': 'PAUSE', 'since': 60.0})[0], 'ACTIVE')
        finally:
            hc.SKIN_PAUSE_C, hc.SKIN_RESUME_C = saved
        self.assertEqual((hc.SKIN_PAUSE_C, hc.SKIN_RESUME_C), (37.0, 35.0))

    def test_smoke_report_says_whether_a_pause_happened(self):
        run = {'utc': 'U', 'smoke': True, 'work_s': 180, 'git_head': 'h', 'start': test_b4.good(),
               'definition': {'files': {}, 'commands': {'server': 'srv'}}}
        rec = {**arm_record(label=b5.SMOKE_LABEL, session=1, problems=[], heat_thresholds={'pause_c': 31.0}),
               'metrics': {}}
        with tempfile.TemporaryDirectory() as d:
            rec['metrics'] = b5.metrics(rec, [{**CYC, 't': 0.0, 'size': 320, 'detect_ms': 140}], {}, Path(d))
        text = b5.report(run, [rec])
        self.assertIn('SMOKE HEAT THRESHOLDS', text)
        self.assertIn('NO PAUSE HAPPENED', text)
        self.assertIn("{'pause_c': 31.0}", text)
        rows = [{**CYC, 't': 0.0, 'size': 320, 'detect_ms': 140},
                hrow('intent', 'ACTIVE', 'PAUSE', 61.0)]
        with tempfile.TemporaryDirectory() as d:
            rec['metrics'] = b5.metrics(rec, rows, {}, Path(d))
        text = b5.report(run, [rec])
        self.assertIn('1 pause(s) with camera stop, 0 resume(s)', text)
        self.assertIn('NO RESUME HAPPENED', text)
        rows.append(hrow('intent', 'PAUSE', 'ACTIVE', 93.0))  # an attempt is no resume
        with tempfile.TemporaryDirectory() as d:
            rec['metrics'] = b5.metrics(rec, rows, {}, Path(d))
        self.assertIn('NO RESUME HAPPENED', b5.report(run, [rec]))
        rows.append(hrow('done', 'PAUSE', 'ACTIVE', 95.0))
        with tempfile.TemporaryDirectory() as d:
            rec['metrics'] = b5.metrics(rec, rows, {}, Path(d))
        text = b5.report(run, [rec])
        self.assertIn('1 pause(s) with camera stop, 1 resume(s)', text)
        self.assertNotIn('HAPPENED', text)

    def test_early_refusal_record_saves_and_reports(self):
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.object(b4, 'cores_ok', lambda: False), \
                unittest.mock.patch.object(b5.repro, 'agents', lambda: ''):
            out = Path(d)
            rec, rows = b5.run_arm('A', out, {'session': 1, 'smoke': False, 'dur': 1800, 'test': {}, 'record': {}},
                                   lambda r: None)
            with open(out / 'arms.jsonl', 'a') as f:
                f.write(json.dumps(rec) + '\n')
            run = {'utc': 'U', 'smoke': False, 'work_s': 1800, 'git_head': 'h', 'start': test_b4.good(),
                   'definition': {'files': {}, 'commands': {'server': 'srv'}}}
            text = b5.report(run, b4.read_jsonl(out / 'arms.jsonl'))
        self.assertEqual((rec['label'], rows), ('INVALID', []))
        self.assertEqual(set(rec['metrics']), set(b5.metrics(arm_record(), [], {}, Path('.'))))
        self.assertIn('== A: INVALID', text)
        self.assertIn('initial agent/cores check failed', text)

    def test_motors_and_serial_never_imported(self):
        self.assertNotIn('motors', sys.modules)
        self.assertNotIn('serial', sys.modules)


# ---------------------------------------------------------------- one arm, real time, everything faked

class FakeMonitor:
    """heat_control.HeatMonitor's interface; skin follows a plan by seconds since start()."""
    plan = [(0, 30.0)]

    def __init__(self):
        self.on_dump, self.t0, self.stopped = None, None, False

    def start(self):
        self.t0 = time.monotonic()

    def stop(self):
        self.stopped = True

    status = [(0, 0)]

    def readings(self):
        now = time.monotonic()
        skin = [s for at, s in self.plan if now - self.t0 >= at][-1]
        status = [s for at, s in self.status if now - self.t0 >= at][-1]
        return {'dump': {'t': now, 'status': status, 'skin': skin}, 'fast': [{'t': now, 'cpu_c': 50.0, 'bat_c': 30.0}]}


class Watcher:
    def __init__(self, targets):
        self.targets = targets

    def start(self):
        return self

    def stop(self):
        return test_b4.watch()


class RealTimeArm(unittest.TestCase):
    def setUp(self):
        self.patches = [unittest.mock.patch.object(*p) for p in [
            (b5, 'OFFER_S', 0.2), (b5, 'TICK_S', 0.5), (b5, 'POLL_S', 0.01), (main, 'CYCLE_MOVE_TIME', 0.05),
            (hc, 'MIN_ACTIVE_S', 0.6), (hc, 'MIN_PAUSE_S', 0.4), (hc, 'HeatMonitor', FakeMonitor),
            (hc, 'SKIN_PAUSE_C', hc.SKIN_PAUSE_C), (hc, 'SKIN_RESUME_C', hc.SKIN_RESUME_C),  # restored after
            (main, 'get_detector', lambda pinned: None), (b4, 'server_command', lambda out, k: ['srv']),
            (b4, 'start_server', lambda out, name, cmd: (unittest.mock.Mock(pid=4242), None, 0.1)),
            (b4, 'stop_server', lambda proc, log: (0, False)), (b4, 'warm', lambda prefix: {'save': {}}),
            (b5, 'picker_setup', lambda test: ({('V3', 'FWD', it['id']): ('t', [1], {}) for it in ITEMS}, [1])),
            (b4, 'reading', lambda pids=None, light=False, tag='': {**test_b4.good(
                mem=('runner', 'llama_server', 'robotcam')), 'mono': time.monotonic()}),
            (b4, 'Periodic', lambda fn, period: Watcher(None)), (b4, 'cores_ok', lambda: True),
            (b5.Repinner, 'server', lambda self, trigger: self.add({'trigger': trigger, 'process': 'llama_server',
                                                                    'ok': True})), (b5.repro, 'agents', lambda: ''),
            (b4, 'slot', lambda action: {'action': action, 'wall_s': 0.0}),
            (b5.b2, 'score', self.score), (b5.b2, 'preflight', lambda post, s, blocks, info, tag: {
                ('V3', 'FWD', it['id']): ('t', [1], {}) for it in s['items']}),
            (main, 'Robot', lambda motors: self.robot)]]
        for p in self.patches:
            p.start()
        self.robot = FakeRobot(test_b4.FakeClock(), {})
        self.robot.clock.sleep = lambda s: time.sleep(0.01)
        self.saved_read = main.read_frame
        main.read_frame = self.robot.read
        self.decisions = []

    def tearDown(self):
        for p in self.patches:
            p.stop()
        main.read_frame = self.saved_read

    def score(self, post, it, v, order, prompt):
        self.decisions.append((time.monotonic(), it['id']))
        for _ in range(2):  # V3: reason, then the action
            time.sleep(0.045)
            post('/completion', {})
        return {'top': 'explore', 'reason': 'r', 'reason_tokens': 3}

    def run_arm(self, arm, skin_plan, smoke=False, status=((0, 0),)):
        FakeMonitor.plan, FakeMonitor.status = skin_plan, list(status)
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch('sys.stdout', io.StringIO()), \
                unittest.mock.patch.object(b4, 'post', lambda path, body: {'timings': {}}):
            ctx = {'session': 1, 'smoke': smoke, 'dur': 4.0, 'test': {'missions': {'M1': 'm'}, 'items': ITEMS},
                   'record': {i: 'explore' for i in range(1, 41)}}
            rows = []
            rec, rows = b5.run_arm(arm, Path(d), ctx, rows.append)
            trace = (Path(d) / rec['heat_trace']).read_text()
        return rec, rows, trace

    def check_common(self, rec, rows, label='VALID'):
        self.assertEqual(rec['problems'], [])
        self.assertEqual(rec['label'], label)
        x = rec['metrics']
        self.assertEqual(x['events']['scripted']['offered'], 20)
        for s, f in x['events'].items():
            self.assertEqual(f['offered'], sum(v for k, v in f.items() if k not in ('offered', 'late_not_offered')))
        self.assertLessEqual(rec['readings'][0]['t'], 0)  # the start reading's real relative time (before t0)
        self.assertEqual(x['decisions_served'], sum(f['served'] for f in x['events'].values()))
        self.assertGreater(x['decisions_served'], 0)
        self.assertEqual(x['agreement']['ids_differ'], [])
        self.assertLessEqual(rec['overrun_s'], 1.0)
        self.assertEqual([r['tag'] for r in rec['readings']][0], 'start')
        self.assertEqual([r['tag'] for r in rec['readings']][-1], 'end')
        run = {'utc': 'U', 'smoke': False, 'work_s': 4, 'git_head': 'h', 'start': test_b4.good(),
               'definition': {'files': {}, 'commands': {'server': 'srv'}}}
        text = b5.report(run, json.loads(json.dumps([rec])))  # as read back from arms.jsonl
        self.assertIn(f'== {rec["arm"]}: {label}', text)
        self.assertIn('agreement with B2 run of record', text)

    def test_arm_a_with_a_heat_pause(self):
        rec, rows, trace = self.run_arm('A', [(0, 30.0), (0.7, 38.0), (1.5, 34.0)])
        self.check_common(rec, rows)
        x = rec['metrics']
        self.assertEqual(x['pauses'], 1)
        self.assertEqual([r['tag'] for r in rec['readings']], ['start', 'pause_start', 'pause_end', 'end'])
        t_off = next(r['t'] for r in rows if r['kind'] == 'heat' and r['stage'] == 'intent')
        t_on = next(r['t'] for r in rows if r['kind'] == 'heat' and r['stage'] == 'done' and r['to'] == 'ACTIVE')
        dec = [r for r in rows if r['kind'] == 'decision']
        self.assertFalse([r for r in dec if t_off + 0.05 < r['started'] < t_on], 'a decision started in the pause')
        self.assertLessEqual(len(dec), 9)  # one per 0.5 s mark at most
        self.assertIn('"camera_stop_ok"', trace)
        self.assertEqual((self.robot.stops, self.robot.starts), (2, 2))  # pause + end; arm start + resume
        self.assertEqual([(e['trigger'], e['process']) for e in rec['repins']],
                         [('arm_start', 'llama_server'), ('camera_start', 'llama_server')])

    def test_smoke_thresholds_from_the_arm_start_skin_pause_and_resume(self):
        # the fake start reading's skin is 30.0: pause at 31.0, resume at 30.5; the deployed 37/35 never trigger here
        rec, rows, trace = self.run_arm('B', [(0, 30.0), (0.7, 31.2), (1.5, 30.4)], smoke=True)
        self.check_common(rec, rows, b5.SMOKE_LABEL)
        self.assertEqual((rec['heat_thresholds']['pause_c'], rec['heat_thresholds']['resume_c']), (31.0, 30.5))
        self.assertEqual((rec['metrics']['pauses'], rec['metrics']['resumes']), (1, 1))
        self.assertEqual((self.robot.stops, self.robot.starts), (2, 2))
        self.assertIn('"camera_start_ok"', trace)

    def test_timed_arm_keeps_the_deployed_thresholds(self):
        rec, rows, trace = self.run_arm('B', [(0, 30.0), (0.7, 31.2), (1.5, 30.4)])
        self.assertNotIn('heat_thresholds', rec)
        self.assertEqual(rec['metrics']['pauses'], 0)
        self.assertEqual((hc.SKIN_PAUSE_C, hc.SKIN_RESUME_C), (37.0, 35.0))

    def test_arm_a_serves_every_eligible_mark(self):
        rec, rows, trace = self.run_arm('A', [(0, 30.0)])
        self.check_common(rec, rows)
        self.assertEqual(rec['metrics']['ticks'], {'decided': 8, 'empty': 0, 'paused': 0, 'busy': 0})  # 0 .. 3.5 s
        first = min((r for r in rows if r['kind'] == 'decision'), key=lambda r: r['started'])
        self.assertEqual(first['id'], 1)
        self.assertLess(first['started'], 0.1)

    def test_arm_a_initially_severe_skips_the_forbidden_marks(self):
        rec, rows, trace = self.run_arm('A', [(0, 30.0)], status=[(0, 3), (0.75, 0)])
        self.check_common(rec, rows)
        self.assertEqual(rec['metrics']['ticks'], {'decided': 6, 'empty': 0, 'paused': 2, 'busy': 0})
        self.assertFalse([r for r in rows if r['kind'] == 'decision' and r['started'] < 0.7])

    def test_arm_b_skin_above_36_holds_events(self):
        rec, rows, trace = self.run_arm('B', [(0, 30.0), (2.0, 36.5)])
        self.check_common(rec, rows)
        dec = [r for r in rows if r['kind'] == 'decision' and r['fate'] == 'served']
        self.assertFalse([r for r in dec if r['started'] > 2.1], 'a decision started at skin >= 36')
        self.assertEqual(rec['metrics']['pauses'], 0)
        self.assertEqual(rec['metrics']['events']['scripted']['unserved'], 1)
        early = [r for r in dec if r['offered'] < 1.8]
        self.assertTrue(early and max(r['started'] - r['offered'] for r in early) < 0.15)


if __name__ == '__main__':
    unittest.main(verbosity=1)
