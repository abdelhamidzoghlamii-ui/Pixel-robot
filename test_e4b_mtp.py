"""E4B-UP regression checks; server, filesystem and HTTP calls are faked."""
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import chat
import server_manager as sm
import test_chat_mtp as old

E4B = 'gemma-4-E4B-it-Q4_0.gguf'
DRAFT = old.H + '/models/mtp/mtp-gemma-4-E4B-it-Q8_0.gguf'
ARGS = ['--model-draft', DRAFT, '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']
BASE_ENTRIES = (
    ('Gemma 4 E2B Q4_0', 'gemma-4-E2B-it-Q4_0.gguf', 'gemma', 4096),
    ('Gemma 4 E2B Q4_K_M', 'gemma-4-e2b-it-q4_k_m.gguf', 'gemma', 4096),
    ('Gemma 4 E2B Q8_0', 'gemma-4-e2b-it-q8_0.gguf', 'gemma', 2048),
    ('Gemma 4 E4B Q4_K_M', 'gemma-4-e4b-it-q4_k_m.gguf', 'gemma', 2048),
    ('Qwen3.5 2B Q4_K_M', 'qwen35/Qwen3.5-2B-Q4_K_M.gguf', 'qwen', 4096),
    ('Qwen3.5 4B Q4_K_M', 'qwen35/Qwen3.5-4B-Q4_K_M.gguf', 'qwen', 2048),
    ('Qwen3.5 4B Q5_K_M', 'qwen35/Qwen3.5-4B-Q5_K_M.gguf', 'qwen', 2048),
    ('Qwen3.5 4B Q6_K', 'qwen35/Qwen3.5-4B-Q6_K.gguf', 'qwen', 2048),
    ('Mistral 7B Q4_K_M', 'mistral-7b-instruct-v0.2.Q4_K_M.gguf', 'mistral', 2048),
)


class E4BChatTest(unittest.TestCase):
    setUp = old.ChatTest.setUp
    start = old.ChatTest.start

    def test_menu_and_commands_preserve_old_entries(self):
        self.assertEqual(chat.MODEL_FILES, BASE_ENTRIES + (('Gemma 4 E4B Q4_0', E4B, 'gemma', 4096),))
        e2b = chat.server_cmd(self.keys['Gemma 4 E2B Q4_0'], True)
        e4b = chat.server_cmd(self.keys['Gemma 4 E4B Q4_0'], True)
        self.assertEqual(e4b, [s.replace('E2B', 'E4B') for s in e2b])
        self.assertEqual(sm.E4B_MTP_ARGS, ARGS)
        self.assertEqual(sm.E4B_MTP_DRAFT_SIZE, 98653248)

    def test_chat_missing_or_wrong_draft_warns_and_loads_without_it(self):
        key = self.keys['Gemma 4 E4B Q4_0']
        for size in (None, 0, 98653247, 98653249):
            with self.subTest(size=size):
                launched, text = self.start(key, e4b_size=size)
                self.assertEqual(launched, [['taskset', '-c', '4-7'] + old.base_chat_cmd(E4B, 4096)])
                self.assertIn(chat.C.YELLOW + '  MTP drafter missing or wrong size', text)
                self.assertIn(DRAFT, text)
                self.assertIn('MTP off', text)
                self.assertFalse(chat.mtp_on)

    def test_chat_valid_draft_and_taskset_rule(self):
        key = self.keys['Gemma 4 E4B Q4_0']
        expected = old.base_chat_cmd(E4B, 4096) + ARGS
        launched, text = self.start(key)
        self.assertEqual(launched, [['taskset', '-c', '4-7'] + expected])
        self.assertIn('MTP on', text)
        launched, text = self.start(key, affinity={0, 1})
        self.assertEqual(launched, [expected])
        self.assertIn('Cores 4-7 unavailable', text)


class E4BSetupTest(unittest.TestCase):
    def launch(self, setup, size=98653248):
        calls = []
        with old.Patches(old.fake_fs(sm.MTP_DRAFT_SIZE, size) + [
                mock.patch.object(sm.os, 'system', calls.append), mock.patch.object(sm.time, 'sleep'),
                mock.patch.object(sm, 'wait_for_server', return_value=True)]), redirect_stdout(io.StringIO()) as out:
            result = sm.start_setup(setup)
        return result, calls, out.getvalue()

    def test_new_commands_and_unchanged_old_setups(self):
        e4b = old.BASE_SETUP_Q4.replace('gemma-4-E2B-it-Q4_0.gguf', E4B)
        mtp = lambda cmd, args: cmd.replace(' --host 127.0.0.1  2>/dev/null &',
                                          ' --host 127.0.0.1 ' + ' '.join(args) + ' 2>/dev/null &')
        expected = {'setup_e4b_q4': e4b, 'setup_e4b_q4_mtp': mtp(e4b, ARGS),
                    'setup_q4': old.BASE_SETUP_Q4, 'setup_q4_mtp': mtp(old.BASE_SETUP_Q4, old.G1),
                    'setup_e4b': old.BASE_SETUP_Q4.replace('gemma-4-E2B-it-Q4_0.gguf', 'gemma-4-e4b-it-q4_k_m.gguf')}
        for setup, cmd in expected.items():
            result, calls, _ = self.launch(setup)
            self.assertTrue(result)
            self.assertEqual(calls, ['pkill -f llama-server 2>/dev/null', cmd])

    def test_robot_refuses_bad_draft_without_killing_or_starting(self):
        for size in (None, 0, 98653247, 98653249):
            result, calls, text = self.launch('setup_e4b_q4_mtp', size)
            self.assertIs(result, False)
            self.assertEqual(calls, [])
            self.assertIn('ERROR: MTP drafter', text)
            self.assertIn('nothing started', text)
            self.assertTrue(self.launch('setup_e4b_q4', size)[0])

    def test_usage_lists_new_setups(self):
        self.assertIn('setup_e4b_q4, setup_e4b_q4_mtp', self.launch('unknown')[2])


if __name__ == '__main__':
    unittest.main()
