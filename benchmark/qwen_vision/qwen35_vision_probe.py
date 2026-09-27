"""File-only Qwen 2B probe: real robot client, scratch server, retained evidence.

Run with native Termux Python; no camera, detector, motor or production launcher.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
import urllib.request

from run_cycle_safety_test import load_robot_module

ROOT = Path(__file__).resolve().parent
PHONE = Path('/data/data/com.termux/files/home')
BIN = PHONE / 'llama.cpp-upstream/build/bin/llama-server'
MODEL = PHONE / 'models/qwen35/Qwen3.5-2B-Q4_K_M.gguf'
PROJECTOR = PHONE / 'models/qwen35/mmproj-F16.gguf'
URL = 'http://127.0.0.1:8088'
# Expected positions refer to raw encoded pixels, as sent unchanged to mtmd.
# Visually checked before inference; NONE means none of the four queried classes.
CASES = [
    ('person_facing_060cm.jpg', {'PERSON': 'L', 'TV': 'L'}),
    ('roomsig_refrigerator_1.jpg', {'REFRIGERATOR': 'C'}),
    ('roomsig_toilet_3.jpg', {'TOILET': 'C'}),
    ('empty_2.jpg', {}),
]
QUESTION = (
    '\n\nFor this visual diagnostic only, override the navigation reply format. '
    'Inspect the attached image only; do not infer objects from the navigation rules. '
    'Report which of PERSON, TV, REFRIGERATOR, TOILET are visible. '
    'For each visible object, give its horizontal position in the image as encoded, '
    'without mentally rotating it: L for left third, C for center third, R for right third. '
    'Use the center of its visible bounding box. Reply only CLASS=POSITION pairs '
    'separated by commas. If none are visible or no image is supplied, reply NONE.'
)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def get(path):
    with urllib.request.urlopen(URL + path, timeout=3) as response:
        return json.load(response)


def memory(pid):
    info = Path('/proc/meminfo').read_text()
    available = int(re.search(r'^MemAvailable:\s+(\d+)', info, re.M)[1])
    try:
        status = Path(f'/proc/{pid}/status').read_text()
        rss = int(re.search(r'^VmRSS:\s+(\d+)', status, re.M)[1])
    except (OSError, TypeError):
        rss = None
    return {'time': time.time(), 'available_kib': available, 'rss_kib': rss}


def adapt(payload):
    """Adapt only the transport to Qwen ChatML and this build's image API."""
    payload = copy.deepcopy(payload)
    prompt = payload['prompt']
    text = prompt['prompt_string'] if isinstance(prompt, dict) else prompt
    text = text.replace('<start_of_turn>user\n', '<|im_start|>user\n')
    text = text.replace('<end_of_turn>', '<|im_end|>')
    text = text.replace('<start_of_turn>model\n',
                        '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    if isinstance(prompt, dict):
        prompt['prompt_string'] = text
    elif 'image_data' in payload:
        prompt = {'prompt_string': text,
                  'multimodal_data': [item['data'] for item in payload.pop('image_data')]}
    else:
        prompt = text
    payload.update(prompt=prompt, stop=['<|im_end|>', '<|endoftext|>'],
                   cache_prompt=False, seed=123)
    return payload


def self_check():
    import base64
    robot = load_robot_module()
    seen = []
    def capture(url, json, timeout):
        seen.append(adapt(json))
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {'content': 'NONE'})
    robot.requests.post = capture
    photo = ROOT / 'bench_photos' / CASES[0][0]
    robot.gemma_decide('probe', str(photo))
    robot.gemma_decide('probe')
    image_prompt = seen[0]['prompt']
    assert base64.b64decode(image_prompt['multimodal_data'][0]) == photo.read_bytes()
    assert image_prompt['prompt_string'].count('[img-1]') == 1
    assert '<start_of_turn>' not in image_prompt['prompt_string']
    assert image_prompt['prompt_string'].endswith('<think>\n\n</think>\n\n')
    assert isinstance(seen[1]['prompt'], str) and '[img-1]' not in seen[1]['prompt']
    assert seen[0]['n_predict'] == 40 and seen[0]['cache_prompt'] is False
    assert seen[0]['stop'][0] == '<|im_end|>'
    print('PASS: real client, unchanged image bytes, ChatML, image/control separation')


def run(budget, output, http_timeout, single_object):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 8088))
    output.mkdir(parents=True, exist_ok=False)
    for name in ('main.py', 'server_manager.py', 'run_cycle_safety_test.py',
                 'qwen35_vision_probe.py'):
        (output / name).write_bytes((ROOT / name).read_bytes())
    version = subprocess.check_output([str(BIN), '--version'], stderr=subprocess.STDOUT).decode()
    assert 'build 2351, commit 790cf51a' in version, version
    assert digest(MODEL) == 'aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223'
    assert digest(PROJECTOR) == '7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7'
    command = [str(BIN), '-m', str(MODEL), '--mmproj', str(PROJECTOR),
               '--port', '8088', '--host', '127.0.0.1', '--ctx-size', '2048',
               '--threads', '4', '--threads-batch', '4', '--parallel', '1',
               '--swa-full', '--verbosity', '4', '--image-min-tokens', str(budget),
               '--image-max-tokens', str(budget)]
    environment = dict(os.environ, LLAMA_MEDIA_MARKER='[img-1]')
    manifest = {'version': version, 'command': command, 'marker': '[img-1]',
                'budget': budget, 'effective_timeout_s': http_timeout,
                'single_object': single_object, 'expected': CASES,
                'files': {str(p): digest(p) for p in [MODEL, PROJECTOR, ROOT / 'main.py']},
                'photos': {name: digest(ROOT / 'bench_photos' / name) for name, _ in CASES}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    samples = []
    done = threading.Event()
    results = []
    log = (output / 'server.log').open('wb')
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    def monitor():
        while not done.is_set():
            samples.append(memory(process.pid))
            done.wait(0.25)
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        start = time.monotonic()
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'server exited while loading: {process.returncode}')
            try:
                if get('/health').get('status') == 'ok':
                    break
            except Exception:
                pass
            if time.monotonic() - start > 120:
                raise TimeoutError('server startup exceeded 120s')
            time.sleep(1)
        props, models = get('/props'), get('/v1/models')
        (output / 'props.json').write_text(json.dumps(props, indent=2))
        (output / 'models.json').write_text(json.dumps(models, indent=2))
        assert props['media_marker'] == '[img-1]'
        assert 'vision' in json.dumps(models).lower() or 'multimodal' in json.dumps(models).lower()
        robot_module = load_robot_module()
        robot_module.GEMMA_URL = URL + '/completion'
        robot = robot_module.Robot()
        robot.mission = 'explore'
        robot.target = None
        robot.scene_log = ['unavailable'] * 5
        robot.last_moves = ['STOP'] * 5
        context = robot.gemma_context('unavailable') + QUESTION
        (output / 'context.txt').write_text(context)
        for name, expected in CASES:
            case_context = context
            if single_object:
                target = next(iter(expected), 'PERSON')
                expected = {target: expected[target]} if expected else {}
                case_context = robot.gemma_context('unavailable') + (
                    '\n\nVisual test only; ignore navigation actions for this reply. '
                    f'Is a {target} visible in the attached image? '
                    'Reply YES LEFT, YES CENTER, or YES RIGHT for the center of its '
                    'visible bounding box in the raw image without rotating it. '
                    'Reply NO if absent, UNKNOWN if no image is attached. '
                    'Output only that answer.')
            for with_image in (False, True):
                if process.poll() is not None:
                    break
                row = {'image': name, 'with_image': with_image, 'expected': expected,
                       'pid': process.pid, 'log_offset': (output / 'server.log').stat().st_size}
                sample_start = len(samples)
                def post(url, json, timeout):
                    payload = adapt(json)
                    stored = copy.deepcopy(payload)
                    if isinstance(stored['prompt'], dict):
                        stored['prompt']['multimodal_data'] = ['UNCHANGED_BYTES: ' + name]
                    row['request'] = stored
                    row['timeout_s'] = timeout
                    row['effective_timeout_s'] = http_timeout
                    request = urllib.request.Request(url, data=__import__('json').dumps(payload).encode(),
                                                     headers={'Content-Type': 'application/json'})
                    try:
                        with urllib.request.urlopen(request, timeout=http_timeout) as response:
                            row['http_status'] = response.status
                            row['response'] = __import__('json').load(response)
                    except Exception as error:
                        row['transport_error'] = repr(error)
                        raise
                    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: row['response'])
                robot_module.requests.post = post
                started = time.monotonic()
                # gemma_decide itself catches network failures and returns STOP.
                row['client_result'] = robot_module.gemma_decide(
                    case_context, str(ROOT / 'bench_photos' / name) if with_image else None)
                row['elapsed_s'] = time.monotonic() - started
                row['exit_code'] = process.poll()
                try:
                    row['health_after'] = get('/health')
                except Exception as error:
                    row['health_error'] = repr(error)
                call_samples = samples[sample_start:] or [memory(process.pid)]
                row['min_available_kib'] = min(s['available_kib'] for s in call_samples)
                response = row.get('response', {})
                print(json.dumps({k: row[k] for k in ('image', 'with_image', 'expected',
                       'client_result', 'exit_code', 'min_available_kib')}),
                      'tokens_evaluated=', response.get('tokens_evaluated'), flush=True)
                results.append(row)
                (output / 'results.json').write_text(json.dumps(results, indent=2))
                if 'health_error' in row:
                    raise RuntimeError('server did not survive request')
        manifest['survived_all_calls'] = len(results) == 8 and process.poll() is None
    except Exception as error:
        manifest['error'] = repr(error)
        print('PROBE_ERROR', repr(error), flush=True)
    finally:
        manifest['exit_before_cleanup'] = process.poll()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        done.set()
        worker.join(timeout=2)
        log.close()
        manifest['exit_after_cleanup'] = process.returncode
        manifest['min_available_kib'] = min(s['available_kib'] for s in samples)
        manifest['max_rss_kib'] = max((s['rss_kib'] or 0) for s in samples)
        manifest['main_unchanged'] = digest(ROOT / 'main.py') == manifest['files'][str(ROOT / 'main.py')]
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (output / 'memory.json').write_text(json.dumps(samples))
        print('ARTIFACTS', output, 'SURVIVED', manifest.get('survived_all_calls', False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=45)
    parser.add_argument('--single-object', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--budget', type=int, default=128)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if args.output is None or args.budget < 1:
            parser.error('--output and a positive --budget are required')
        run(args.budget, args.output, args.timeout, args.single_object)
