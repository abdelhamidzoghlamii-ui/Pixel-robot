"""TPL-FIX motors-off OLD/NEW HTTP comparison. Never manages servers or instantiates Robot.

Run with native Termux Python: python3 -B benchmark/tplfix/compare.py BASE_SHA
Offline check: python3 -B benchmark/tplfix/compare.py --self-check
"""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import requests

ROOT = Path(__file__).resolve().parents[2]
CHAT_PROMPTS = (
    'Hello! Introduce yourself in one sentence.',
    'What is two plus three? Answer briefly.',
    'Name three common household objects.',
    'Explain why a robot should stop near an obstacle in one sentence.',
    'Write a friendly good morning greeting.',
)
TIMING_LABEL = 'NOT A MEASUREMENT (agent resident)'
CHAT_SYSTEM = 'You are a small home robot. Answer in English, briefly.'
TOKEN = re.compile(r'<(?:\|[^<>\n]*[>|]|[^<>\n]*\|>|start_of_turn>|end_of_turn>)')
FILES = ('chat.py', 'main.py', 'test_tplfix.py', 'tests/gemma4_apply_template.json',
         'benchmark/tplfix/compare.py',
         'benchmark/llm_objective_setting/conversation/chat_mtp/test_chat_mtp_runner.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parsed_ok(raw):
    if not isinstance(raw, str):
        return False
    raw = '[' + raw
    try:
        return isinstance(json.loads(raw[raw.find('['):raw.rfind(']') + 1]), list)
    except (ValueError, TypeError):
        return False


def repeated(reply):
    words = reply.split()
    return any(words[i:i+8] == words[j:j+8]
               for i in range(len(words)-15) for j in range(i+8, len(words)-7))


def voice(module, command):
    captured = {}

    def post(url, **kwargs):
        response = requests.post(url, **kwargs)
        captured['raw_reply'] = response.json().get('content')
        return response

    module.requests = SimpleNamespace(post=post)
    actions = module.parse_command(command)
    raw = captured.get('raw_reply')
    return dict(raw_reply=raw, actions=actions, parse_succeeded=parsed_ok(raw),
                leaks=TOKEN.findall(raw or ''))


def chat_reply(module, prompt, system=None):
    messages = ([{'role': 'system', 'content': system}] if system else [])
    messages.append({'role': 'user', 'content': prompt})
    payload = module.chat_payload(module.build_prompt('gemma', messages), 'gemma')
    payload.update(temperature=0, n_predict=128)
    response = requests.post(module.URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    reply = data['content']
    return dict(raw_reply=reply, leaks=TOKEN.findall(reply), repeated_eight_words=repeated(reply),
                truncated=data.get('truncated'), stop_type=data.get('stop_type'),
                hit_token_cap=data.get('stop_type') == 'limit')


def props():
    response = requests.get('http://127.0.0.1:8080/props', timeout=10)
    response.raise_for_status()
    data = response.json()
    return dict(build_info=data.get('build_info'), model=Path(data['model_path']).name,
                total_slots=data.get('total_slots'),
                generation=data.get('default_generation_settings'),
                chat_template_sha256=hashlib.sha256(data['chat_template'].encode()).hexdigest())


def self_check():
    from unittest.mock import patch

    assert parsed_ok('{"type":"come_back"}]') and parsed_ok(']')
    assert not parsed_ok('bad') and not parsed_ok(None)
    assert TOKEN.findall('hello <turn|> <|channel>thought <start_of_turn>') == [
        '<turn|>', '<|channel>', '<start_of_turn>']
    assert not TOKEN.findall('ordinary answer')
    assert repeated('one two three four five six seven eight ' * 2)
    assert not repeated('A short friendly reply.')
    fake = SimpleNamespace(parse_command=lambda text: [])
    assert voice(fake, 'ignored')['parse_succeeded'] is False
    assert len(CHAT_PROMPTS) == 5
    builder = SimpleNamespace(URL='unused', build_prompt=lambda family, messages: messages,
                              chat_payload=lambda prompt, family: {'prompt': prompt})
    response = SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {'content': '5', 'stop_type': 'limit'})
    with patch.object(requests, 'post', return_value=response) as post:
        for system in (None, CHAT_SYSTEM):
            assert chat_reply(builder, 'Hi', system)['hit_token_cap']
            messages = ([{'role': 'system', 'content': system}] if system else [])
            messages.append({'role': 'user', 'content': 'Hi'})
            post.assert_called_with('unused', json={'prompt': messages, 'temperature': 0, 'n_predict': 128}, timeout=120)
    print('TPL-FIX runner self-check: PASS')


def main(base):
    sys.path.insert(0, str(ROOT))
    before = props()
    assert before['build_info'] == 'b1609-e1a1abb7', before['build_info']
    assert before['model'] == 'gemma-4-E2B-it-Q4_0.gguf', before['model']
    assert before['generation']['n_ctx'] == 2048, 'owner must start setup_q4'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == base
    source = ROOT / 'benchmark/llama_dotprod_rebuild/checks/voice_gen.py'
    commands = next(ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COMMANDS' for t in n.targets))
    assert len(commands) == 15
    out = ROOT / 'benchmark/tplfix/runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    results = dict(timing_label=TIMING_LABEL, voice=[], chat=[])
    manifest = dict(base_commit=base, timing_label=TIMING_LABEL, server_props_before=before,
                    candidate_sha256={f: sha(ROOT / f) for f in FILES},
                    voice_source=str(source.relative_to(ROOT)), voice_source_sha256=sha(source),
                    chat_prompts=list(CHAT_PROMPTS), chat_system=CHAT_SYSTEM, complete=False)

    def save():
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    save()
    with tempfile.TemporaryDirectory(prefix='tplfix_old_') as directory:
        old = Path(directory)
        for filename in ('chat.py', 'main.py'):
            (old / filename).write_bytes(subprocess.check_output(['git', 'show', f'{base}:{filename}'], cwd=ROOT))
        manifest['base_file_sha256'] = {f: sha(old / f) for f in ('chat.py', 'main.py')}
        variants = [(label, load(folder / 'main.py', label+'_voice'), load(folder / 'chat.py', label+'_chat'))
                    for label, folder in [('OLD', old), ('NEW', ROOT)]]
        for command in commands:
            row = dict(command=command)
            for label, parser, _ in variants:
                row[label] = voice(parser, command)
            row['same_actions'] = row['OLD']['actions'] == row['NEW']['actions']
            results['voice'].append(row)
            save()
            print('voice', len(results['voice']), 'saved', flush=True)
        for system in (None, CHAT_SYSTEM):
            for prompt in CHAT_PROMPTS:
                row = dict(prompt=prompt, system=system)
                for label, _, builder in variants:
                    row[label] = chat_reply(builder, prompt, system)
                results['chat'].append(row)
                save()
                print('chat', len(results['chat']), 'saved', flush=True)
    manifest['server_props_after'] = props()
    assert manifest['server_props_after'] == before, 'server props changed; investigate'
    assert manifest['candidate_sha256'] == {f: sha(ROOT / f) for f in FILES}, 'candidate changed'
    manifest['complete'] = True
    save()
    lines = ['# TPL-FIX OLD vs NEW', '', TIMING_LABEL, '',
             '| Voice command | OLD parsed | NEW parsed | Same actions | OLD leaks | NEW leaks |',
             '|---|---|---|---|---|---|']
    for row in results['voice']:
        lines.append('| ' + ' | '.join([row['command'], str(row['OLD']['parse_succeeded']),
                     str(row['NEW']['parse_succeeded']), str(row['same_actions']),
                     str(len(row['OLD']['leaks'])), str(len(row['NEW']['leaks']))]) + ' |')
    lines += ['', '| System | Chat prompt | OLD leaks | NEW leaks | OLD cap | NEW cap |', '|---|---|---|---|---|---|']
    for row in results['chat']:
        lines.append('| ' + ' | '.join([str(bool(row['system'])), row['prompt'],
                     str(len(row['OLD']['leaks'])), str(len(row['NEW']['leaks'])),
                     str(row['OLD']['hit_token_cap']), str(row['NEW']['hit_token_cap'])]) + ' |')
    lines += ['', 'Token cap means stop_type=limit; answer presence is assessed from raw replies.', '']
    (out / 'results.md').write_text('\n'.join(lines))
    print('Saved', out)


if __name__ == '__main__':
    self_check()
    if sys.argv[1:] != ['--self-check']:
        if len(sys.argv) != 2 or not re.fullmatch(r'[0-9a-f]{40}', sys.argv[1]):
            raise SystemExit('usage: compare.py BASE_SHA | --self-check')
        main(sys.argv[1])
