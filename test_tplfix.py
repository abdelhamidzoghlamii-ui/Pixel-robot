"""TPL-FIX: captured robot-server templates and the voice parser contract; no hardware."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import chat
from run_cycle_safety_test import load_robot_module


class GemmaTemplateTest(unittest.TestCase):
    def test_matches_robot_apply_template_captures(self):
        fixture = json.loads((Path(__file__).parent / 'tests' / 'gemma4_apply_template.json').read_text())
        self.assertEqual(len(fixture['cases']), 3)
        for case in fixture['cases']:
            with self.subTest(messages=case['request']['messages']):
                self.assertEqual(chat.build_prompt('gemma', case['request']['messages']), case['prompt'])
                self.assertNotIn('<|think|>', case['prompt'])

    def test_gemma_trims_content_like_the_template(self):
        messages = [{'role': 'system', 'content': ' S\n'}, {'role': 'user', 'content': ' Hi '},
                    {'role': 'assistant', 'content': '\nYo '}, {'role': 'user', 'content': ' Again\n'}]
        self.assertEqual(chat.build_prompt('gemma', messages),
                         '<|turn>system\nS<turn|>\n<|turn>user\nHi<turn|>\n'
                         '<|turn>model\nYo<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n')

    def test_stop_tokens(self):
        self.assertEqual(chat.get_stop_tokens('gemma'), ['<turn|>'])
        self.assertEqual(chat.get_stop_tokens('qwen'), ['<|im_end|>'])
        self.assertEqual(chat.get_stop_tokens('mistral'), ['</s>', '[INST]'])

    def test_other_families_unchanged(self):
        messages = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'Hi'},
                    {'role': 'assistant', 'content': 'Yo'}, {'role': 'user', 'content': 'Again'}]
        self.assertEqual(chat.build_prompt('qwen', messages),
                         '<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n'
                         '<|im_start|>assistant\nYo<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n'
                         '<|im_start|>assistant\n<think>\n\n</think>\n\n')
        self.assertEqual(chat.build_prompt('mistral', messages),
                         '[INST] Hi [/INST] Yo</s>[INST] Again [/INST]')

    def test_voice_prefill_payload_and_json_filter(self):
        robot = load_robot_module()
        robot.requests.post = Mock(return_value=Mock(json=lambda: {
            'content': '{"type":"say","message":"hello"},{"missing":"type"},7] ignored'}))
        self.assertEqual(robot.parse_command('Say hello'), [{'type': 'say', 'message': 'hello'}])
        robot.requests.post.assert_called_once_with(robot.GEMMA_URL, json={
            'prompt': '<|turn>user\n' + robot.PARSE_SYS + '\nCommand: "Say hello"'
                      '<turn|>\n<|turn>model\n[',
            'n_predict': 300, 'temperature': 0.05, 'stop': ['<turn|>', '\n\n'],
        }, timeout=40)

    def test_voice_invalid_json_still_fails_closed(self):
        robot = load_robot_module()
        robot.requests.post = Mock(return_value=Mock(json=lambda: {'content': 'not JSON'}))
        self.assertEqual(robot.parse_command('Unparseable', timeout=12), [])
        self.assertEqual(robot.requests.post.call_args.kwargs['timeout'], 12)


if __name__ == '__main__':
    unittest.main()
