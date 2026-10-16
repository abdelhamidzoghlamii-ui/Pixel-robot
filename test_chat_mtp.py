"""MTP-A offline tests: setup_q4 unchanged, opt-in setup_q4_mtp, chat.py's drafter flags, fallback and payload.
Fakes only (os.system, subprocess.Popen, os.path, HTTP); nothing is launched."""
import io
import os
import shlex
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import chat
import server_manager as sm

H = '/data/data/com.termux/files/home'
SERVER = H + '/llama.cpp-b1609-dotprod/build/bin/llama-server'
DRAFT = H + '/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf'
G1 = ['--model-draft', DRAFT, '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']  # DECISIONS #120 run json
# server_manager.start_setup('setup_q4') at e208aaf, as passed to os.system
BASE_SETUP_Q4 = (SERVER + ' -m ' + H + '/models/gemma-4-E2B-it-Q4_0.gguf --port 8080 --ctx-size 2048 --threads 4'
                 ' --threads-batch 4 --parallel 1 --swa-full --cache-ram 0 --host 127.0.0.1  2>/dev/null &')


def base_chat_cmd(file, ctx):
    """chat.start_server's argv at e208aaf (before taskset)."""
    return [SERVER, '-m', f'{H}/models/{file}', '--port', '8080', '--ctx-size', str(ctx), '--host', '127.0.0.1',
            '--threads', '4', '--threads-batch', '4', '--parallel', '1', '--swa-full', '--cache-ram', '0']


def fake_fs(draft_size, e4b_size=sm.E4B_MTP_DRAFT_SIZE):
    """os.path fakes: every model file is 2 GB; the drafter has draft_size bytes, or is missing when None."""
    def exists(p):
        if p == sm.E4B_MTP_DRAFT:
            return e4b_size is not None
        return p != DRAFT or draft_size is not None
    def getsize(p):
        if p == sm.E4B_MTP_DRAFT:
            if e4b_size is None:
                raise FileNotFoundError(p)
            return e4b_size
        if p == DRAFT:
            if draft_size is None:
                raise FileNotFoundError(p)
            return draft_size
        return 2 << 30
    return [mock.patch('os.path.exists', exists), mock.patch('os.path.isfile', exists),
            mock.patch('os.path.getsize', getsize)]


class Patches:
    def __init__(self, patches):
        self.patches = patches

    def __enter__(self):
        return [p.start() for p in self.patches]

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


class ServerManagerTest(unittest.TestCase):
    def launch(self, setup, draft_size=sm.MTP_DRAFT_SIZE):
        calls = []
        with Patches(fake_fs(draft_size) + [mock.patch.object(sm.os, 'system', calls.append),
                                            mock.patch.object(sm.time, 'sleep'),
                                            mock.patch.object(sm, 'wait_for_server', return_value=True)]), \
                redirect_stdout(io.StringIO()) as out:
            ok = sm.start_setup(setup)
        return ok, [c for c in calls if 'pkill' not in c], out.getvalue()

    def test_setup_q4_command_unchanged(self):
        for size in (sm.MTP_DRAFT_SIZE, None, 1):  # the drafter's state never touches setup_q4
            ok, cmds, _ = self.launch('setup_q4', size)
            self.assertTrue(ok)
            self.assertEqual(cmds, [BASE_SETUP_Q4])

    def test_setup_q4_mtp_is_setup_q4_plus_exactly_g1(self):
        ok, cmds, _ = self.launch('setup_q4_mtp')
        self.assertTrue(ok)
        self.assertEqual(cmds, [BASE_SETUP_Q4.replace(' --host 127.0.0.1  2>/dev/null &',
                                                      ' --host 127.0.0.1 ' + ' '.join(G1) + ' 2>/dev/null &')])
        self.assertEqual(shlex.split(cmds[0])[:-2], shlex.split(BASE_SETUP_Q4)[:-2] + G1)

    def test_setup_q4_mtp_refuses_missing_or_wrong_size_drafter(self):
        for size in (None, sm.MTP_DRAFT_SIZE - 1, sm.MTP_DRAFT_SIZE + 1, 0):
            calls = []
            with Patches(fake_fs(size) + [mock.patch.object(sm.os, 'system', calls.append)]), \
                    redirect_stdout(io.StringIO()) as out:
                self.assertIs(sm.start_setup('setup_q4_mtp'), False)
            self.assertEqual(calls, [], 'nothing killed or launched')
            self.assertIn('ERROR: MTP drafter', out.getvalue())

    def test_mtp_draft_ok_on_a_real_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'draft.gguf')
            with mock.patch.object(sm, 'MTP_DRAFT', path):
                self.assertFalse(sm.mtp_draft_ok())
                with open(path, 'wb') as f:
                    f.truncate(sm.MTP_DRAFT_SIZE)
                self.assertTrue(sm.mtp_draft_ok())
                with open(path, 'ab') as f:
                    f.write(b'x')
                self.assertFalse(sm.mtp_draft_ok())

    def test_constants_and_usage(self):
        self.assertEqual(sm.MTP_ARGS, G1)
        self.assertEqual(sm.MTP_DRAFT_SIZE, 97817664)
        ok, cmds, out = self.launch('nonsense')
        self.assertIs(ok, False)
        self.assertEqual(cmds, [])
        self.assertIn('setup_q4, setup_q4_mtp, setup_e4b, stop', out)


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ChatTest(unittest.TestCase):
    def setUp(self):
        models = {str(i): {'name': n, 'file': f, 'type': t, 'ctx': c}
                  for i, (n, f, t, c) in enumerate(chat.MODEL_FILES, 1)}
        self.models = mock.patch.object(chat, 'MODELS', models)
        self.models.start()
        self.addCleanup(self.models.stop)
        self.keys = {m['name']: k for k, m in models.items()}

    def test_server_cmd_is_base_for_every_entry_and_draft_adds_g1(self):
        for name, file, _, ctx in chat.MODEL_FILES:
            key = self.keys[name]
            self.assertEqual(chat.server_cmd(key), base_chat_cmd(file, ctx), name)
            args = sm.E4B_MTP_ARGS if file == chat.E4B_MTP_MODEL else G1
            self.assertEqual(chat.server_cmd(key, draft=True), base_chat_cmd(file, ctx) + args, name)

    def start(self, key, draft_size=sm.MTP_DRAFT_SIZE, e4b_size=sm.E4B_MTP_DRAFT_SIZE, affinity=frozenset(range(8))):
        launched = []

        class Popen:
            def __init__(self, cmd, **kw):
                launched.append(cmd)
                self.returncode = None

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as d, \
                Patches(fake_fs(draft_size, e4b_size) + [mock.patch.object(chat, 'SERVER_LOG', d + '/log'),
                                               mock.patch.object(chat, 'kill_server'),
                                               mock.patch.object(chat, 'server_running', return_value=False),
                                               mock.patch.object(chat, 'get_temp', return_value=''),
                                               mock.patch.object(chat.time, 'sleep'),
                                               mock.patch.object(chat.shutil, 'which', return_value='/usr/bin/taskset'),
                                               mock.patch.object(chat.os, 'sched_getaffinity', return_value=affinity),
                                               mock.patch.object(chat.subprocess, 'Popen', Popen),
                                               mock.patch('urllib.request.urlopen', return_value=Response())]), \
                redirect_stdout(io.StringIO()) as out:
            ok = chat.start_server(key)
        self.assertTrue(ok)
        return launched, out.getvalue()

    def test_start_server_e2b_q4_0_gets_drafter_others_unchanged(self):
        for name, file, _, ctx in chat.MODEL_FILES:
            launched, out = self.start(self.keys[name])
            mtp = name in ('Gemma 4 E2B Q4_0', 'Gemma 4 E4B Q4_0')
            args = sm.E4B_MTP_ARGS if file == chat.E4B_MTP_MODEL else G1
            self.assertEqual(launched, [['taskset', '-c', '4-7'] + base_chat_cmd(file, ctx) + args * mtp], name)
            self.assertIn('MTP on' if mtp else 'MTP off', out)
            self.assertIs(chat.mtp_on, mtp)

    def test_missing_or_wrong_size_drafter_warns_and_starts_without_it(self):
        key = self.keys['Gemma 4 E2B Q4_0']
        for size in (None, sm.MTP_DRAFT_SIZE - 1):
            launched, out = self.start(key, size)
            self.assertEqual(launched, [['taskset', '-c', '4-7'] + base_chat_cmd('gemma-4-E2B-it-Q4_0.gguf', 4096)])
            self.assertIn(chat.C.YELLOW + '  MTP drafter missing or wrong size', out)
            self.assertIn('MTP off', out)
            self.assertIs(chat.mtp_on, False)

    def test_header_shows_mtp_state(self):
        for on in (True, False):
            with mock.patch.object(chat, 'current_model', self.keys['Gemma 4 E2B Q4_0']), \
                    mock.patch.object(chat, 'mtp_on', on), mock.patch.object(chat.os, 'system'), \
                    mock.patch.object(chat, 'get_temp', return_value=''), redirect_stdout(io.StringIO()) as out:
                chat.show_header()
            self.assertIn('MTP on' if on else 'MTP off', out.getvalue())

    def test_chat_payload_identical_to_base(self):
        for name, _, family, _ in chat.MODEL_FILES:
            sent = []

            class Resp:
                def json(self):
                    return {'content': ' hi ', 'timings': {'predicted_per_second': 10.0}}

            with mock.patch.object(chat, 'current_model', self.keys[name]), mock.patch.object(chat, 'history', []), \
                    mock.patch.object(chat.requests, 'post', lambda url, json, timeout: sent.append((url, json, timeout)) or Resp()):
                self.assertEqual(chat.chat('Hello')[0], 'hi')
            base = {'prompt': chat.build_prompt(family, [{'role': 'user', 'content': 'Hello'}]), 'n_predict': 512,
                    'temperature': 0.7, 'stop': chat.get_stop_tokens(family), 'stream': False}
            self.assertEqual(sent, [('http://127.0.0.1:8080/completion', base, 120)], name)
            self.assertEqual(list(sent[0][1]), list(base), 'key order')


if __name__ == '__main__':
    unittest.main()
