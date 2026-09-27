#!/usr/bin/env python3
"""Frozen, stdlib-only Qwen VL benchmark. Run in native Termux with Codex closed.

  python qwen_vision_bench.py --check
  python qwen_vision_bench.py --output /absolute/new/results-directory

Each image gets a fresh server, including in the 10s pass. Latency excludes
startup; there is no client warmup or prompt reuse. Unbounded means no HTTP
deadline (Ctrl+C saves partial results and stops our child). No robot imports.
"""
import argparse
import base64
import csv
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
PHONE = Path('/data/data/com.termux/files/home')
MODEL_DIR = PHONE / 'models/qwen35'
BIN = PHONE / 'llama.cpp-upstream/build/bin/llama-server'
BUILD = 'build 2351, commit 790cf51a'
PORT = 8088
BUDGETS = (128, 256, 512, 1024)
PASSES = (('unbounded', None), ('10s', 10.0))
COOLDOWN = 60  # seconds after hashing and between models; fixed, not thermal equilibrium
MARKER = '[img-1]'
LABELS_SHA = '841f3d93f6b1f0d0ca43f2214ca073ad4e2672bc5857b634ef101621b456f70b'
MODELS = [
    ('0.8B-Q4_K_M', 'bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517', '0.8B'),
    ('2B-Q4_K_M', 'aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223', '2B'),
    ('4B-Q4_K_M', '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4', '4B'),
    ('4B-Q5_K_M', '8814232b85594dcd46c50e5b8b29324a7efe9e746edbe8a3d1df3d3fce7aad39', '4B'),
    ('4B-Q6_K', 'fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66', '4B'),
]
PROJECTORS = {
    '0.8B': ('mmproj-Qwen3.5-0.8B-F16.gguf', '56e4c6cfe73b0c82e3e82bc518d7591997e61d81f723fc41a586f4fa69ea2453'),
    '2B': ('mmproj-F16.gguf', '7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7'),
    '4B': ('mmproj-Qwen3.5-4B-F16.gguf', 'cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864'),
}
# CSV supplies object/category; positions supplement it by visual inspection of
# raw pixels (EXIF orientation 6 is NOT applied). See qwen_vision_bench_report.md.
CASES = [
    ('person_facing_060cm.jpg', {'PERSON': 'L', 'TV': 'L'}, '6e7882317f0620a383a2da279d58a3f82c357a978bc47221947af3460e67bcdb'),
    ('roomsig_refrigerator_1.jpg', {'REFRIGERATOR': 'C'}, 'ba8abe530b043baf8d0d1b10ddfb0e2493f562cdd8a9817044c5b128bda0317f'),
    ('roomsig_toilet_3.jpg', {'TOILET': 'C'}, 'dca6d9a6228670a13547b808a75fe3f59aa9f9528d2c696e08a191ea5ddaa77e'),
    ('empty_2.jpg', {}, '68c2ed00e7635d6884377b4c34a90617e238e63749318dc602d9f49423246f2a'),
]
QUESTION = (
    'Inspect the attached image only. Report every visible object from these four '
    'classes: PERSON, TV, REFRIGERATOR, TOILET. For each, use the center of its '
    'visible bounding box in the raw image as encoded, without rotating it: '
    'L=left third, C=center third, R=right third. Reply only CLASS=POSITION pairs '
    'separated by commas, or NONE if none of these four classes are visible. '
    'Do not report other classes.'
)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def payload(image_bytes):
    return {
        'prompt': {'prompt_string': '<|im_start|>user\n' + MARKER + '\n' + QUESTION
                   + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
                   'multimodal_data': [base64.b64encode(image_bytes).decode('ascii')]},
        'n_predict': 64, 'temperature': 0, 'seed': 123, 'cache_prompt': False,
        'stream': False, 'stop': ['<|im_end|>', '<|endoftext|>'],
    }


def parse_answer(text):
    text = text.strip().upper()
    if text == 'NONE':
        return {}
    pairs = text.split(',')
    result = {}
    for pair in pairs:
        match = re.fullmatch(r'\s*(PERSON|TV|REFRIGERATOR|TOILET)\s*=\s*([LCR])\s*', pair)
        if not match or match[1] in result:
            return None
        result[match[1]] = match[2]
    return result


def mem_available():
    return int(re.search(r'^MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text(), re.M)[1])


def http_json(port, path, data=None, timeout=3):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        connection.request('GET' if data is None else 'POST', path, body=data,
                           headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def completion(port, body, deadline):
    """SIGALRM imposes a total wall deadline, including the entire response body.

    Socket timeouts alone reset across operations and aren't a 10s total limit.
    Called only on the main thread, never during an existing interval timer.
    """
    row = dict(http_completed=False, http_success=False, timeout=False, inference_response=False,
               http_status=None, actual=None, parse_valid=False)
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=None)
    def expired(signum, frame):
        raise TimeoutError('client wall deadline expired')
    previous = signal.signal(signal.SIGALRM, expired)
    started = time.monotonic()
    try:
        if deadline is not None:
            signal.setitimer(signal.ITIMER_REAL, deadline)
        connection.request('POST', '/completion', body=body,
                           headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        row['http_status'] = response.status
        raw = response.read()
        row['http_completed'] = True
        row['http_success'] = 200 <= response.status < 300
        row['raw_response'] = raw.decode('utf-8', errors='replace')
        value = json.loads(raw)
        row['response'] = value
        if row['http_success'] and isinstance(value, dict) and isinstance(value.get('content'), str):
            row['inference_response'] = True
            row['actual_text'] = value['content']
            row['actual'] = parse_answer(value['content'])
            row['parse_valid'] = row['actual'] is not None
    except TimeoutError as error:
        row.update(timeout=True, error=str(error), http_success=False)
    except (OSError, http.client.HTTPException, ValueError) as error:
        row['error'] = repr(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        row['latency_s'] = time.monotonic() - started
        connection.close()
        signal.signal(signal.SIGALRM, previous)
    return row


def stop_server(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()  # only the child we created, never a discovered PID
            process.wait(timeout=10)
            return 'killed_after_10s'
        return 'terminated'
    process.wait()
    return 'already_exited'


def measure(model, projector, budget, body, deadline, log_path):
    row = dict(projector_loaded=False, http_completed=False, http_success=False,
               timeout=False, inference_response=False, actual=None, parse_valid=False, latency_s=None,
               process_alive_after=None, exit_before_cleanup=None)
    process = None
    done = threading.Event()
    samples = []
    started = time.monotonic()
    def sample():
        while not done.is_set():
            samples.append((time.monotonic() - started, mem_available()))
            done.wait(0.25)
    worker = threading.Thread(target=sample, daemon=True)
    row['mem_available_before_launch_kib'] = mem_available()
    command = [str(BIN), '-m', str(model), '--mmproj', str(projector),
               '--host', '127.0.0.1', '--port', str(PORT), '--ctx-size', '2048',
               '--threads', '4', '--threads-batch', '4', '--parallel', '1',
               '--swa-full', '--verbosity', '4', '--image-min-tokens', str(budget),
               '--image-max-tokens', str(budget)]
    row['command'] = command
    with log_path.open('xb') as log:
        try:
            # Check every launch; never send measurements to an existing listener.
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', PORT))
            worker.start()
            env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
            env['LLAMA_MEDIA_MARKER'] = MARKER
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            limit = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited at startup: {process.returncode}')
                try:
                    status, health = http_json(PORT, '/health', timeout=120)
                    if status == 200 and health.get('status') == 'ok':
                        break
                except (OSError, ValueError, http.client.HTTPException):
                    pass
                if time.monotonic() > limit:
                    raise TimeoutError('startup exceeded 180s (not an inference timeout)')
                time.sleep(0.25)
            status, props = http_json(PORT, '/props')
            row['props'] = props
            row['projector_loaded'] = status == 200 and props.get('modalities', {}).get('vision') is True
            if not row['projector_loaded'] or props.get('media_marker') != MARKER:
                raise RuntimeError('vision capability or media marker verification failed')
            if process.poll() is not None:
                raise RuntimeError('child exited before inference')
            row['startup_s'] = time.monotonic() - started
            row['mem_available_before_request_kib'] = mem_available()
            sample_index = len(samples)
            row.update(completion(PORT, body, deadline))
            request_samples = [row['mem_available_before_request_kib'], mem_available()]
            request_samples.extend(v for _, v in samples[sample_index:])
            row['mem_available_request_min_kib'] = min(request_samples)
        except (OSError, RuntimeError, ValueError, http.client.HTTPException) as error:
            row['setup_error'] = repr(error)
        finally:
            if process is not None:
                row['exit_before_cleanup'] = process.poll()
                row['process_alive_after'] = row['exit_before_cleanup'] is None
                # Cleanup failures propagate: starting another model then is unsafe.
                row['cleanup'] = stop_server(process)
                row['exit_after_cleanup'] = process.returncode
            done.set()
            if worker.ident is not None:
                worker.join()
            row['mem_available_under_load_min_kib'] = min(
                [row['mem_available_before_launch_kib']] + [v for _, v in samples])
            row['memory_samples_elapsed_s_kib'] = samples
            row['log'] = str(log_path)
    log_text = log_path.read_text(errors='replace')
    row['image_tokens_decoding_batches'] = [int(n) for n in re.findall(
        r'decoding image batch \d+/\d+, n_tokens_batch = (\d+)', log_text)]
    return row


def check_inputs():
    version = subprocess.check_output([str(BIN), '--version'], stderr=subprocess.STDOUT,
                                      text=True, timeout=10)
    if BUILD not in version:
        raise ValueError('wrong server build: ' + version)
    labels = ROOT / 'bench_photos/labels.csv'
    if digest(labels) != LABELS_SHA:
        raise ValueError('labels.csv changed; re-verify before running')
    label_rows = list(csv.DictReader(io.StringIO('\n'.join(
        line for line in labels.read_text().splitlines() if not line.startswith('#')))))
    photos = []
    for name, expected, sha in CASES:
        path = ROOT / 'bench_photos' / name
        if digest(path) != sha:
            raise ValueError('photo changed: ' + name)
        source = next(r for r in label_rows if r['filename'] == name)
        photos.append(dict(name=name, sha256=sha, expected=expected, csv_row=source))
    checked = {}
    models = []
    for name, sha, family in MODELS:
        model = MODEL_DIR / ('Qwen3.5-' + name + '.gguf')
        projector_name, projector_sha = PROJECTORS[family]
        projector = MODEL_DIR / projector_name
        item = dict(name=name, model=str(model), projector=str(projector), files={})
        for path, wanted in ((model, sha), (projector, projector_sha)):
            try:
                if path not in checked:
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError('missing or zero-byte file')
                    checked[path] = digest(path)
                actual = checked[path]
                item['files'][str(path)] = actual
                if actual != wanted:
                    raise ValueError('SHA-256 mismatch')
            except (OSError, ValueError) as error:
                item['skip'] = f'{path}: {error}'
        models.append(item)
    return dict(version=version, script_sha256=digest(Path(__file__)), models=models,
                photos=photos, labels_sha256=LABELS_SHA, budgets=BUDGETS, passes=PASSES,
                cooldown_s=COOLDOWN, memory_sampling_s=0.25,
                method='Fresh server per image. No client warmup. Raw JPEG. Exact set scoring.')


def battery():
    value = json.loads(subprocess.check_output(
        ['/data/data/com.termux/files/usr/bin/termux-battery-status'], text=True, timeout=15))
    if not (value.get('plugged') == 'UNPLUGGED' and value.get('status') == 'DISCHARGING'
            and 50 <= value.get('percentage', -1) <= 80):
        raise RuntimeError('benchmark requires battery, discharging, 50-80%: ' + str(value))
    return value


def save(output, result):
    temp = output / 'results.json.tmp'
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(output / 'results.json')


def summary(result):
    lines = ['model | budget | fits | unbounded median s, correct | 10s completed, correct']
    for model in result['manifest']['models']:
        for budget in BUDGETS:
            rows = [r for r in result['rows'] if r['model'] == model['name'] and r['budget'] == budget]
            a = [r for r in rows if r['pass'] == 'unbounded']
            b = [r for r in rows if r['pass'] == '10s']
            lat = [r['latency_s'] for r in a if r.get('http_success')]
            # Fits requires successful image HTTP and vision capability, not merely PID survival.
            fits = 'SKIP' if model.get('skip') else ('YES' if len(a) == 4 and all(
                r.get('projector_loaded') and r.get('http_success') and r.get('inference_response')
                and r.get('process_alive_after')
                for r in a) else 'UNPROVEN')
            latency = f'{statistics.median(lat):.2f} ({len(lat)}/4)' if lat else 'n/a (0/4)'
            lines.append(f"{model['name']} | {budget} | {fits} | {latency}, "
                         f"{sum(r.get('correct', False) for r in a)}/4 | "
                         f"{sum(r.get('http_success', False) for r in b)}/4, "
                         f"{sum(r.get('correct', False) for r in b)}/4")
    return '\n'.join(lines)


def run(output, manifest):
    output.mkdir(parents=True, exist_ok=False)
    result = dict(manifest=manifest, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), rows=[])
    save(output, result)
    try:
        existing = []
        for comm in Path('/proc').glob('[0-9]*/comm'):
            try:
                if comm.read_text().strip() == 'llama-server':
                    existing.append(comm.parent.name)
            except OSError:
                pass
        if existing:
            raise RuntimeError('stop existing llama-server processes before benchmarking: ' + ', '.join(existing))
        result['initial_battery'] = battery()
        print(f'Initial cooldown: {COOLDOWN}s', flush=True)
        time.sleep(COOLDOWN)
        for model in manifest['models']:
            for budget in BUDGETS:
                for pass_name, deadline in PASSES:
                    for name, expected, _ in CASES:
                        row = dict(model=model['name'], budget=budget, **{'pass': pass_name},
                                   image=name, expected=expected, deadline_s=deadline, correct=False)
                        result['rows'].append(row)
                        if model.get('skip'):
                            row['skip'] = model['skip']
                        else:
                            row['battery'] = battery()
                            body = json.dumps(payload((ROOT / 'bench_photos' / name).read_bytes())).encode()
                            # Persist the pending identity so interruptions don't lose this cell.
                            row['state'] = 'running'
                            save(output, result)
                            log = output / f'{model["name"]}-{budget}-{pass_name}-{name}.log'
                            row.update(measure(Path(model['model']), Path(model['projector']),
                                               budget, body, deadline, log))
                            row['correct'] = (row['http_success'] and row['parse_valid']
                                              and row['actual'] == expected and not row['timeout'])
                            row['state'] = 'recorded'
                        save(output, result)
                        print(model['name'], budget, pass_name, name,
                              'HTTP_OK=', row.get('http_success', False), 'timeout=', row.get('timeout', False),
                              'alive=', row.get('process_alive_after'), 'correct=', row['correct'], flush=True)
                        if not model.get('skip'):
                            time.sleep(2)
            if not model.get('skip'):
                print(f'Model stopped; cooldown {COOLDOWN}s', flush=True)
                time.sleep(COOLDOWN)
        result['complete'] = True
    except KeyboardInterrupt:
        result['interrupted'] = True
    except Exception as error:
        result['run_error'] = repr(error)
    finally:
        result['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save(output, result)
        table = summary(result)
        (output / 'summary.txt').write_text(table + '\n')
        print(table, flush=True)
    return 0 if result.get('complete') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='hash/build/image checks only; no model loads')
    parser.add_argument('--output', type=Path, help='new directory for JSON, summary and server logs')
    args = parser.parse_args()
    if not args.check and args.output is None:
        parser.error('--output is required to run; use --check for preflight only')
    manifest = check_inputs()
    if args.check:
        print(json.dumps(manifest, indent=2))
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        raise SystemExit(run(args.output, manifest))
