"""Offline tests for chat_mtp.py: fakes for the server, HTTP, clock, root and /proc; nothing is launched.
Run from this folder: python3 -B -m unittest -v test_chat_mtp_runner"""
import http.client
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_mtp as cm  # noqa: E402

chat, cr = cm.chat, cm.cr


def setUpModule():  # the menu as if every model file existed: no dependence on the phone's ~/models
    patch = mock.patch.object(chat, 'MODELS', {str(i): {'name': n, 'file': f, 'type': t, 'ctx': c}
                                               for i, (n, f, t, c) in enumerate(chat.MODEL_FILES, 1)})
    patch.start()
    unittest.addModuleCleanup(patch.stop)
G1 = ['--model-draft', '/data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf',
      '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']


class CommandPayloadTest(unittest.TestCase):
    def test_commands_come_from_chat_server_cmd(self):
        key = cm.model_key()
        with mock.patch.object(chat, 'server_cmd', return_value=['X']) as f:
            self.assertEqual(cm.command('OFF'), ['taskset', '-c', '4-7', 'X'])
            self.assertEqual(cm.command('ON'), ['taskset', '-c', '4-7', 'X'])
        self.assertEqual(f.call_args_list, [mock.call(key, draft=False), mock.call(key, draft=True)])
        off, on = cm.command('OFF'), cm.command('ON')
        self.assertEqual(off, ['taskset', '-c', '4-7', *chat.server_cmd(key)])
        self.assertEqual(on, off + G1)
        self.assertEqual(off[off.index('--ctx-size') + 1], '4096')
        self.assertEqual(off[off.index('-m') + 1], cm.MODEL)

    def test_payload_is_chat_payload_of_a_single_user_turn(self):
        p = cm.payload('How far can you see?')
        self.assertEqual(p, chat.chat_payload(chat.build_prompt('gemma', [{'role': 'user', 'content': 'How far can you see?'}]),
                                              'gemma'))
        self.assertEqual(p, {'prompt': '<start_of_turn>user\nHow far can you see?<end_of_turn>\n<start_of_turn>model\n',
                             'n_predict': 512, 'temperature': 0.7, 'stop': ['<end_of_turn>'], 'stream': False})
        with mock.patch.object(chat, 'chat_payload', return_value='P') as f:
            self.assertEqual(cm.payload('x'), 'P')
        f.assert_called_once()

    def test_prompts_are_the_13_ids_from_bench(self):
        t = cm.prompt_texts()
        self.assertEqual(list(t), 'A5 A6 B1 B2 B6 B7 C1 C6 C7 C8 C9 C10 C11'.split())
        self.assertEqual(t['B1'], 'How far can you see?')
        self.assertEqual(t['C11'], 'Find Abdel')
        self.assertEqual(t['A5'], "I'm looking for something. It's small and black. So where should I start?")


class OrderTest(unittest.TestCase):
    def test_abba_order_and_skip_of_a_finished_config(self):
        rem = {'OFF': ['a'], 'ON': ['a']}
        self.assertEqual([cm.next_block(s, rem)[1] for s in range(8)], ['OFF', 'ON', 'ON', 'OFF', 'OFF', 'ON', 'ON', 'OFF'])
        self.assertEqual(cm.next_block(1, {'OFF': ['a'], 'ON': []}), (3, 'OFF'))
        self.assertEqual(cm.next_block(0, {'OFF': [], 'ON': ['a']}), (1, 'ON'))
        self.assertIsNone(cm.next_block(5, {'OFF': [], 'ON': []}))

    def test_full_run_order_with_fake_blocks(self):
        """Each fake block finishes 5 prompts: OFF, ON, ON, OFF, OFF, ON until both ran all 13 once."""
        rem, seq, ran = {'OFF': list(cm.IDS), 'ON': list(cm.IDS)}, 0, []
        while (nxt := cm.next_block(seq, rem)) is not None:
            seq, config = nxt
            ran.append(config)
            del rem[config][:5]
            seq += 1
        self.assertEqual(ran, ['OFF', 'ON', 'ON', 'OFF', 'OFF', 'ON'])


class Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


def timings(n=100, ms=5000.0, draft=None):
    tm = {'prompt_ms': 300.0, 'prompt_n': 20, 'predicted_n': n, 'predicted_ms': ms, 'predicted_per_second': n / ms * 1000}
    if draft:
        tm.update(draft_n=draft[0], draft_n_accepted=draft[1])
    return tm


class BlockTest(unittest.TestCase):
    """run_block with a fake clock: load and each fake request advance it by their durations. The fake watchdog
    behaves like the real one: a request still running at its deadline ends there with the server's disconnect
    (http.client.RemoteDisconnected, as the untimed proot check of the KILL showed)."""

    def run_block(self, durations, cores_lost_after=None, post_exc=None, late=False, load_s=6.5, samples=(),
                  join_s=0, status=None):
        clock, posts, events, emitted = Clock(), [], [], []
        proc = SimpleNamespace(pid=4242)
        texts = {i: f'text {i}' for i in cm.IDS}
        fired, dog = threading.Event(), {}

        def post(path, body, timeout):
            i = len(posts)
            posts.append((path, body, timeout, clock.t - 1000.0))
            if post_exc and i in post_exc:
                raise post_exc[i]
            d = durations[i]
            if clock.t + d > dog['at'] and not late:  # the watchdog KILLs the server; the request sees a disconnect
                clock.t = dog['at']
                fired.set()
                raise http.client.RemoteDisconnected('Remote end closed connection without response')
            clock.t += d  # late=True: the whole reply arrives after the deadline and nothing fired
            return {'content': ' reply ', 'stop_type': 'eos', 'timings': timings()}

        def check_cores(when):
            if cores_lost_after is not None and when == f'after {cm.IDS[cores_lost_after]}':
                raise cr.CoresLost('cores 4-7 not all allowed')

        def spawn(cmd, log):
            events.append(('spawn', cmd))
            return proc

        def watchdog(p, delay):
            dog['at'] = clock.t + delay
            events.append(('watchdog', p, delay))
            return SimpleNamespace(cancel=lambda: events.append('cancel')), fired

        def wait_healthy(p, t0, log):
            events.append('healthy')
            clock.t += load_s
            return clock.t - t0

        def sampler_loop(t0, log, server, stop, rows):
            events.append(('sampler', server.proc))
            rows.extend(samples)

        class Thread:  # runs the target at start(); join() takes join_s (a su call still in flight)
            def __init__(self, target, args, daemon):
                self.target, self.args = target, args

            def start(self):
                self.target(*self.args)

            def join(self, timeout=None):
                events.append('join')
                clock.t += join_s

        gate = {'skin': 30.0, 'status': 0, 'waited_s': 0, 'warm_start': False, 'z9': 1}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(cm, 'time', SimpleNamespace(monotonic=clock.monotonic)), \
                mock.patch.object(cm, 'threading', SimpleNamespace(Event=threading.Event, Thread=Thread)), \
                mock.patch.object(cm.pm, 'thermal_gate', return_value=gate), \
                mock.patch.object(cm, 'spawn', spawn), mock.patch.object(cm, 'watchdog', watchdog), \
                mock.patch.object(cm, 'wait_healthy', wait_healthy), \
                mock.patch.object(cm, 'stop', side_effect=lambda p: events.append(('stop', p, clock.t - 1000.0))), \
                mock.patch.object(cm, 'proc_status', side_effect=status and (lambda pid: status(fired)) or
                                  (lambda pid: {'VmHWM': '4000000 kB', 'Cpus_allowed_list': '4-7'})), \
                mock.patch.object(cr, 'sampler_loop', sampler_loop), \
                mock.patch.object(cr, 'check_cores', check_cores), \
                mock.patch.object(cr, 'read_dump', side_effect=lambda: events.append('read_dump') or
                                  {'skin': 33.0, 'status': 1}), \
                mock.patch.object(cr, 'post', post), redirect_stdout(io.StringIO()):
            ctx = {'out': Path(d), 'thermal_log': 'log', 'idle': {}, 'emit': emitted.append}
            try:
                b = cm.run_block(3, 'ON', list(cm.IDS), texts, ctx)
            except BaseException as e:
                b = e
        stops = [e[2] for e in events if isinstance(e, tuple) and e[0] == 'stop']
        return b, posts, stops, emitted, events, proc

    def test_deadline_constants(self):
        self.assertEqual((cm.BLOCK_S, cm.DEADLINE_S, cm.TERM_WAIT_S, cm.LAST_START_S), (180, 175, 3, 150))

    def test_no_prompt_starts_after_150_s(self):
        b, posts, stops, emitted, events, proc = self.run_block([30] * 13)
        self.assertEqual([p[3] for p in posts], [6.5, 36.5, 66.5, 96.5, 126.5])  # the next would start at 156.5 s
        self.assertEqual(b['end_reason'], 'no prompt starts after 150 s')
        self.assertEqual((b['prompts'], b['requests_s'], stops), (cm.IDS[:5], 156.5, [156.5]))
        self.assertEqual(events[:2], [('spawn', cm.command('ON')), ('watchdog', proc, 175)])
        self.assertEqual(posts[0][:3], ('/completion', cm.payload('text A5'), 175 - 6.5))

    def test_request_blocked_at_the_deadline_is_cut_there(self):
        b, posts, stops, emitted, events, proc = self.run_block([35] * 13)  # fifth starts 146.5 s, would end 181.5 s
        self.assertEqual(b['prompts'], cm.IDS[:4])
        self.assertIn('not finished at 175 s', b['end_reason'])
        self.assertIn('RemoteDisconnected', b['end_reason'])
        self.assertEqual(stops, [175])  # shutdown begins at the deadline, not when the reply would have come
        self.assertEqual(b['server_s'], 175)
        self.assertLess(events.index('cancel'), events.index(('stop', proc, 175)))
        b, posts, stops, *_ = self.run_block([70, 70, 70], load_s=0)  # third starts 140 s
        self.assertEqual((posts[2][2:], b['prompts'], stops), ((35, 140), cm.IDS[:2], [175]))

    def test_vm_hwm_kept_when_the_watchdog_kill_removes_it(self):
        reads = []

        def status(fired):  # VmHWM grows per read; a killed (zombie) server's status has no memory fields
            reads.append(fired.is_set())
            return {'Cpus_allowed_list': '4-7'} if fired.is_set() else \
                {'VmHWM': f'{4000000 + 1000 * len(reads)} kB', 'Cpus_allowed_list': '4-7'}

        b, posts, stops, emitted, *_ = self.run_block([35] * 13, status=status)  # fifth cut by the watchdog
        self.assertEqual(reads, [False] * 5 + [True])  # start, after each of the 4 prompts, end
        self.assertEqual(b['prompts'], cm.IDS[:4])
        self.assertEqual(b['vm_hwm_kb'], 4005000)  # the read after the fourth prompt
        self.assertEqual(b['server_end'], {'Cpus_allowed_list': '4-7', 'VmHWM': '4005000 kB',
                                           'VmHWM_source': 'last completed prompt'})

    def test_late_reply_without_disconnect_is_rejected(self):
        b, posts, stops, emitted, *_ = self.run_block([70, 70, 70], late=True, load_s=0)  # ends at 210 s, no raise
        self.assertEqual(b['prompts'], cm.IDS[:2])
        self.assertIn('reply at 210.0 s', b['end_reason'])
        self.assertEqual([r['t_end'] for r in emitted], [70, 140])
        self.assertEqual(len(stops), 1)

    def test_other_request_errors_are_not_swallowed(self):
        exc = ConnectionRefusedError('refused')  # an OSError, but the watchdog did not fire
        b, posts, stops, emitted, _, proc = self.run_block([10] * 13, post_exc={1: exc})
        self.assertIs(b, exc)
        self.assertEqual(len(stops), 1)

    def test_slow_load_shrinks_the_budget_and_too_slow_stops_the_run(self):
        b, posts, *_ = self.run_block([60] * 13, load_s=100)
        self.assertEqual([p[2:] for p in posts], [(75, 100)])  # 60 s reply ends at 160 s: no further start
        self.assertEqual((b['prompts'], b['end_reason']), (cm.IDS[:1], 'no prompt starts after 150 s'))
        b, posts, stops, emitted, _, proc = self.run_block([10] * 13, load_s=151)
        self.assertIsInstance(b, RuntimeError)
        self.assertEqual((posts, emitted, len(stops)), ([], [], 1))

    def test_first_reply_cut_by_the_deadline_stops_the_run(self):
        b, posts, stops, emitted, _, proc = self.run_block([200])
        self.assertIsInstance(b, RuntimeError)
        self.assertEqual((emitted, stops), ([], [175]))

    def test_lost_cores_drop_the_prompt_and_end_the_block(self):
        b, posts, stops, emitted, *_ = self.run_block([10] * 13, cores_lost_after=2)
        self.assertEqual(b['prompts'], cm.IDS[:2])
        self.assertIn(f'{cm.IDS[2]} dropped', b['end_reason'])
        self.assertEqual((len(emitted), len(stops)), (2, 1))

    def test_server_killed_on_ctrl_c_and_errors(self):
        for exc in (KeyboardInterrupt(), SystemExit(130), ValueError('bad json')):
            b, posts, stops, emitted, events, proc = self.run_block([10] * 13, post_exc={1: exc})
            self.assertIs(b, exc)
            self.assertEqual(len(stops), 1)
            self.assertIn('cancel', events)
            self.assertEqual(len(emitted), 1)

    def test_sampling_covers_the_load_and_end_reading_follows_the_stop(self):
        samples = [{'t': 2.0, 'battery_w': 9.0, 'mem_available_mib': 2000, 'pss_kb': {'llama_server': 4900}},  # load
                   {'t': 10.0, 'battery_w': 3.0, 'mem_available_mib': 2600, 'pss_kb': {'llama_server': 4100}}]
        b, posts, stops, emitted, events, proc = self.run_block([10] * 13, samples=samples, join_s=30)
        self.assertEqual(events[:4], [('spawn', cm.command('ON')), ('watchdog', proc, 175), ('sampler', proc), 'healthy'])
        self.assertEqual(events[-4:], ['cancel', ('stop', proc, 136.5), 'join', 'read_dump'])
        self.assertEqual(b['server_s'], 136.5)  # taken at the stop, not after the 30 s sampler join
        self.assertEqual((b['mean_w'], b['peak_pss_kb'], b['min_mem_available_mib']), (6.0, 4900, 2000))
        self.assertEqual(b['mean_w_requests'], 3.0)  # only the sample taken during a request

    def test_row_fields(self):
        b, posts, stops, emitted, *_ = self.run_block([10] * 13, load_s=0)
        r = emitted[0]
        self.assertEqual((r['config'], r['block'], r['prompt_id'], r['t_start'], r['t_end'], r['wall_s']),
                         ('ON', 3, 'A5', 0, 10, 10))
        self.assertEqual((r['ttft_s'], r['prompt_s'], r['prompt_n'], r['predicted_n'], r['reply_chars']),
                         (None, 0.3, 20, 100, 5))
        self.assertEqual((b['vm_hwm_kb'], b['server_start']['Cpus_allowed_list'], b['end']), (4000000, '4-7',
                                                                                               {'skin': 33.0, 'status': 1}))
        self.assertEqual((b['load_s'], len(b['prompts']), b['requests_s'], b['server_s']), (0, 13, 130, 130))


class StartStopTest(unittest.TestCase):
    def test_wait_healthy_limits_the_load_to_the_block(self):
        clock = Clock()
        proc = SimpleNamespace(poll=lambda: None)

        def sleep(s):
            clock.t += 30

        with mock.patch.object(cm, 'time', SimpleNamespace(monotonic=clock.monotonic, sleep=sleep)), \
                mock.patch.object(cr, 'healthy', side_effect=lambda: clock.t >= 1060):
            self.assertEqual(cm.wait_healthy(proc, 1000.0, SimpleNamespace(name='log')), 60)
            clock.t = 1000.0
            with mock.patch.object(cr, 'healthy', return_value=False), self.assertRaises(RuntimeError):
                cm.wait_healthy(proc, 1000.0, SimpleNamespace(name='log'))
            self.assertEqual(clock.t, 1180)  # first poll past 175 s
            clock.t = 1176.0  # healthy on the first poll, but already past the deadline
            with self.assertRaises(RuntimeError):
                cm.wait_healthy(proc, 1000.0, SimpleNamespace(name='log'))
        dead = SimpleNamespace(poll=lambda: -9)  # KILLed by the watchdog: the loop ends at once
        with mock.patch.object(cr, 'healthy', return_value=False), self.assertRaises(RuntimeError):
            cm.wait_healthy(dead, time.monotonic(), SimpleNamespace(name='log'))

    def test_watchdog_kills_the_group_once_and_can_be_cancelled(self):
        kills = []
        with mock.patch.object(cm.os, 'killpg', lambda pid, sig: kills.append((pid, sig))):
            timer, fired = cm.watchdog(SimpleNamespace(pid=11), 0.05)
            timer.join(2)
            self.assertTrue(fired.is_set())
            self.assertEqual(kills, [(11, cm.signal.SIGKILL)])
            timer, fired = cm.watchdog(SimpleNamespace(pid=12), 0.5)
            timer.cancel()
            timer.join(2)
            self.assertFalse(fired.is_set())
            self.assertEqual(kills, [(11, cm.signal.SIGKILL)])

    def test_signal_during_spawn_kills_the_new_server(self):
        killed = []

        class Popen:
            def __init__(self, cmd, **kw):
                self.pid = 77
                cr._deferred['signum'] = 2  # SIGINT arrived while spawning (exit_on_signal deferred it)

        with mock.patch.object(cm.subprocess, 'Popen', Popen), mock.patch.object(cm, 'stop', killed.append), \
                tempfile.TemporaryFile('w') as log:
            with self.assertRaises(SystemExit) as e:
                cm.spawn(['x'], log)
        self.assertEqual(e.exception.code, 130)
        self.assertEqual([p.pid for p in killed], [77])

    def test_stop_terms_the_group_then_kills(self):
        calls = []

        class Proc:
            pid = 9

            def poll(self):
                return None

            def wait(self, timeout=None):
                calls.append(('wait', timeout))
                if timeout:
                    raise cm.subprocess.TimeoutExpired('x', timeout)

        p = Proc()
        cm.LIVE.append(p)
        with mock.patch.object(cm.os, 'killpg', lambda pid, sig: calls.append(('kill', pid, sig))):
            cm.stop(p)
        self.assertEqual(calls, [('kill', 9, cm.signal.SIGTERM), ('wait', 3), ('kill', 9, cm.signal.SIGKILL), ('wait', None)])
        self.assertNotIn(p, cm.LIVE)


class RefusalTest(unittest.TestCase):
    def refusals(self, agents='', port_rc=111, battery='85\nDischarging\n'):
        class Sock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def settimeout(self, t):
                pass

            def connect_ex(self, addr):
                assert addr == ('127.0.0.1', 8080)
                return port_rc

        with mock.patch.object(cm.subprocess, 'run', return_value=SimpleNamespace(stdout=agents)) as run, \
                mock.patch.object(cm.socket, 'socket', Sock), \
                mock.patch.object(cr, 'root', return_value=(0, battery)):
            why = cm.refusals()
        self.assertEqual(run.call_args[0][0], ['pgrep', '-fa', 'claude|codex|agy'])
        return why

    def test_each_condition(self):
        self.assertEqual(self.refusals(), [])
        self.assertIn('agent process running', self.refusals(agents='123 node /usr/bin/claude')[0])
        self.assertEqual(self.refusals(port_rc=0), ['port 8080 in use'])
        self.assertEqual(self.refusals(battery='79\nDischarging\n'), ['battery 79 % < 80 %'])
        self.assertEqual(self.refusals(battery='80\nDischarging\n'), [])
        self.assertIn('not Discharging', self.refusals(battery='95\nCharging\n')[0])
        self.assertIn('not Discharging', self.refusals(battery='100\nFull\n')[0])
        self.assertIn('battery unreadable', self.refusals(battery='')[0])

    def test_proot_is_refused_before_anything_else(self):
        with mock.patch.object(cm, 'refusals') as r, mock.patch.object(cm.signal, 'signal') as s, \
                mock.patch.object(cm.os, 'environ', {'PREFIX': '/usr'}):
            with self.assertRaises(SystemExit) as e:
                cm.main([])
        self.assertIn('native Termux', str(e.exception))
        r.assert_not_called()
        s.assert_not_called()

    def test_hash_check(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d, 'model'), Path(d, 'draft')
            a.write_bytes(b'model')
            b.write_bytes(b'draft')
            good = {str(a): cr.sha256(a), str(b): cr.sha256(b)}
            with mock.patch.object(cm, 'SHA256', good):
                self.assertEqual(cm.check_hashes(), good)
            for bad in ({**good, str(b): '0' * 64}, {**good, str(Path(d, 'missing')): good[str(a)]}):
                with mock.patch.object(cm, 'SHA256', bad), self.assertRaises(SystemExit):
                    cm.check_hashes()

    def test_g2_hashes(self):
        self.assertEqual(cm.SHA256, {
            '/data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf':
                '8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52',
            '/data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf':
                '9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919'})


class MathTest(unittest.TestCase):
    def test_token_weighted_tok_s_and_acceptance(self):
        rows = [{'predicted_n': 100, 'predicted_ms': 5000.0, 'draft_n': 30, 'draft_n_accepted': 12},
                {'predicted_n': 300, 'predicted_ms': 10000.0, 'draft_n': 10, 'draft_n_accepted': 8},
                {'predicted_n': 5, 'predicted_ms': 100.0, 'draft_n': None, 'draft_n_accepted': None}]
        self.assertAlmostEqual(cm.tok_s(rows), 400 / 15000 * 1000)  # the 5-token reply is left out
        self.assertAlmostEqual(cm.acceptance(rows), 20 / 40)
        self.assertIsNone(cm.acceptance(rows[2:]))
        self.assertIsNone(cm.tok_s(rows[2:]))

    def test_block_power_energy_and_memory(self):
        rows = [{'prompt_id': 'A5', 't_start': 10.0, 't_end': 20.0, 'wall_s': 10.0, 'predicted_n': 50},
                {'prompt_id': 'A6', 't_start': 20.0, 't_end': 30.0, 'wall_s': 10.0, 'predicted_n': 150}]
        samples = [{'t': 0.0, 'battery_w': 1.0, 'mem_available_mib': 3000, 'pss_kb': {'runner': 1}},
                   {'t': 12.0, 'battery_w': 4.0, 'mem_available_mib': 2500, 'pss_kb': {'llama_server': 4000}},
                   {'t': 25.0, 'battery_w': 6.0, 'mem_available_mib': 2600, 'pss_kb': {'llama_server': 4200}},
                   {'t': 35.0, 'error': 'su failed'}]
        b = cm.block_record(1, 'OFF', {'warm_start': False}, 6.0, 31.0, 32.0, 'done', {}, {'VmHWM': '4300000 kB'}, {},
                            samples, rows)
        self.assertAlmostEqual(b['mean_w'], 11 / 3)
        self.assertAlmostEqual(b['mean_w_requests'], 5.0)
        self.assertAlmostEqual(b['energy_j_per_token'], 5.0 * 20.0 / 200)
        self.assertEqual((b['peak_pss_kb'], b['min_mem_available_mib'], b['vm_hwm_kb']), (4200, 2500, 4300000))


class ResumeTest(unittest.TestCase):
    def test_state_and_main_resume_skip_finished_prompts(self):
        row = lambda c, i, blk: {'config': c, 'prompt_id': i, 'block': blk, 'predicted_n': 50, 'predicted_ms': 4000.0,
                                 'prompt_s': 0.3, 'reply_chars': 10, 'draft_n': None, 'draft_n_accepted': None}
        block = {'block': 0, 'config': 'OFF', 'prompts': ['A5', 'A6'], 'end_reason': 'x', 'load_s': 6.0,
                 'requests_s': 100, 'server_s': 101,
                 'gate': {'skin': 30, 'status': 0, 'warm_start': False}, 'end': {'skin': 31, 'status': 0},
                 'server_start': {}, 'server_end': {}, 'vm_hwm_kb': 1, 'peak_pss_kb': 1, 'min_mem_available_mib': 1,
                 'mean_w': 1.0, 'energy_j_per_token': 1.0}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            # block 0 (OFF) complete; block 1 (ON) interrupted after A5, with a torn last line
            (out / 'prompts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in
                                               (row('OFF', 'A5', 0), row('OFF', 'A6', 0), row('ON', 'A5', 1))) + '{"torn')
            (out / 'blocks.jsonl').write_text(json.dumps(block) + '\n')
            with redirect_stdout(io.StringIO()):
                prompts, blocks, remaining, seq = cm.state(out, cm.IDS)
            self.assertEqual((len(prompts), len(blocks), seq), (3, 1, 2))
            self.assertEqual(remaining['OFF'], cm.IDS[2:])
            self.assertEqual(remaining['ON'], cm.IDS[1:])

            texts = cm.prompt_texts()
            manifest = {'smoke': False, 'ids': cm.IDS, 'commands': {c: cm.command(c) for c in ('OFF', 'ON')},
                        'payload_example': cm.payload('<text>'),
                        'prompt_text_sha256': {i: cm.sha256_text(texts[i]) for i in cm.IDS}}
            (out / 'run.json').write_text(json.dumps({**manifest, 'idle': {'skin': 29.0, 'z9': 1}}))
            calls = []

            def fake_block(seq, config, todo, texts, ctx):
                calls.append((seq, config, list(todo)))
                for i in todo:
                    ctx['emit'](row(config, i, seq))
                return {**block, 'block': seq, 'config': config, 'prompts': list(todo)}

            with mock.patch.object(cr, 'require_native'), mock.patch.object(cm, 'refusals', return_value=[]), \
                    mock.patch.object(cm, 'check_hashes', return_value={}), mock.patch.object(cr, 'wait_cores'), \
                    mock.patch.object(cm.signal, 'signal'), mock.patch.object(cm, 'run_block', fake_block), \
                    mock.patch.object(cr, 'sha256', return_value='h'), redirect_stdout(io.StringIO()):
                cm.main(['--resume', str(out)])
            self.assertEqual(calls, [(2, 'ON', cm.IDS[1:]), (3, 'OFF', cm.IDS[2:])])
            done = [(r['config'], r['prompt_id']) for r in cm.read_jsonl(out / 'prompts.jsonl')]
            self.assertEqual(sorted(done), sorted({(c, i) for c in ('OFF', 'ON') for i in cm.IDS}))
            self.assertEqual(len(done), 26, 'no prompt repeated')
            self.assertIn('13/13', (out / 'report.txt').read_text())
            self.assertTrue(list(out.glob('resume_*.json')))
            self.assertTrue((out / 'prompts.jsonl').read_text().endswith('\n'))
            kept = list(out.glob('prompts.jsonl.before_resume_*'))
            self.assertEqual(len(kept), 1)
            self.assertTrue(kept[0].read_text().endswith('{"torn'))

            (out / 'run.json').write_text(json.dumps({**manifest, 'smoke': True, 'idle': {'skin': 29.0}}))
            with mock.patch.object(cr, 'require_native'), mock.patch.object(cm, 'refusals', return_value=[]), \
                    mock.patch.object(cm.signal, 'signal'), self.assertRaises(SystemExit) as e:
                cm.main(['--resume', str(out)])
            self.assertIn('differ from run.json', str(e.exception))

    def test_main_stops_live_servers_on_ctrl_c(self):
        proc, stopped = SimpleNamespace(pid=5), []

        def fake_block(*a):
            cm.LIVE.append(proc)
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as d, mock.patch.object(cm, 'OUT_ROOT', Path(d)), \
                mock.patch.object(cr, 'require_native'), mock.patch.object(cm, 'refusals', return_value=[]), \
                mock.patch.object(cm, 'check_hashes', return_value={}), mock.patch.object(cr, 'wait_cores'), \
                mock.patch.object(cr, 'read_thermal', return_value={'z9': 1}), \
                mock.patch.object(cr, 'read_dump', return_value={'skin': 29.0}), \
                mock.patch.object(cr, 'sha256', return_value='h'), mock.patch.object(cm.signal, 'signal'), \
                mock.patch.object(cm, 'run_block', fake_block), mock.patch.object(cm.pm, 'GATE_MAX_S', 480), \
                mock.patch.object(cm, 'stop', side_effect=lambda p: (stopped.append(p), cm.LIVE.remove(p))), \
                redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            cm.main(['--smoke'])
        self.assertEqual(stopped, [proc])
        self.assertEqual(cm.LIVE, [])

    def test_cores_lost_before_start_redoes_the_block(self):
        calls = []

        def fake_block(seq, config, todo, texts, ctx):
            calls.append(seq)
            if len(calls) == 1:
                raise cr.CoresLost('cores 4-7 not all allowed before block 0')
            for i in todo:
                ctx['emit']({'config': config, 'prompt_id': i, 'block': seq, 'predicted_n': 50, 'predicted_ms': 4000.0,
                             'prompt_s': 0.3, 'reply_chars': 10, 'draft_n': None, 'draft_n_accepted': None})
            return {'block': seq, 'config': config, 'prompts': list(todo), 'end_reason': 'x', 'load_s': 6.0,
                    'requests_s': 100, 'server_s': 101, 'gate': {'skin': 30, 'status': 0, 'warm_start': False}, 'end': {'skin': 31, 'status': 0},
                    'server_start': {}, 'server_end': {}, 'vm_hwm_kb': 1, 'peak_pss_kb': 1, 'min_mem_available_mib': 1,
                    'mean_w': 1.0, 'energy_j_per_token': 1.0}

        with tempfile.TemporaryDirectory() as d, mock.patch.object(cm, 'OUT_ROOT', Path(d)), \
                mock.patch.object(cr, 'require_native'), mock.patch.object(cm, 'refusals', return_value=[]), \
                mock.patch.object(cm, 'check_hashes', return_value={}), mock.patch.object(cr, 'wait_cores') as wait, \
                mock.patch.object(cr, 'read_thermal', return_value={'z9': 1}), \
                mock.patch.object(cr, 'read_dump', return_value={'skin': 29.0}), \
                mock.patch.object(cr, 'sha256', return_value='h'), mock.patch.object(cm.signal, 'signal'), \
                mock.patch.object(cm, 'run_block', fake_block), mock.patch.object(cm.pm, 'GATE_MAX_S', 480), \
                redirect_stdout(io.StringIO()):
            cm.main(['--smoke'])
            self.assertEqual(cm.pm.GATE_MAX_S, cm.SMOKE_GATE_S)
        self.assertEqual(calls, [0, 0, 1])
        self.assertEqual(wait.call_count, 1 + 3)  # start, then before every attempt

    def test_refusal_stops_main(self):
        with mock.patch.object(cr, 'require_native'), mock.patch.object(cm.signal, 'signal'), \
                mock.patch.object(cm, 'refusals', return_value=['port 8080 in use']), \
                mock.patch.object(cm, 'check_hashes') as h, self.assertRaises(SystemExit) as e:
            cm.main([])
        self.assertIn('port 8080 in use', str(e.exception))
        h.assert_not_called()


if __name__ == '__main__':
    unittest.main()
