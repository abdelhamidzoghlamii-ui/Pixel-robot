"""TASK VIS-0 — Gemma 4 template check + Gemma vision feasibility on the Pixel (motors off).

Template check VALID (deterministic); vision replies VALID as outputs (temperature 0);
timings and memory NOT measurements (agent resident); quality grading pending (Local AI).
"""
import argparse
import base64
from datetime import datetime, timezone
import glob
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parents[2]
LIVE_TERMUX_ROBOT = '/data/data/com.termux/files/home/robot'
sys.path[:] = [str(ROBOT)] + [p for p in sys.path if p != str(ROBOT)]

from PIL import Image, ImageDraw, ImageFont, ImageOps

import server_manager
import chat
from chat import build_prompt, get_stop_tokens

# chat.py prepends LIVE_TERMUX_ROBOT to sys.path at import time; keep ROBOT first and remove
# any injected foreign checkout path when running from an isolated copy.
sys.path[:] = [str(ROBOT)] + [
    p for p in sys.path
    if p != str(ROBOT) and (str(ROBOT.resolve()) == LIVE_TERMUX_ROBOT or p != LIVE_TERMUX_ROBOT)
]

IMPORTED_MODULES = (
    ('benchmark/gemma_vision/vis0/vis0.py', lambda: __file__),
    ('chat.py', lambda: chat.__file__),
    ('server_manager.py', lambda: server_manager.__file__),
)

REPORT_LABELS = (
    'Template check VALID (deterministic); '
    'vision replies VALID as outputs (temperature 0); '
    'timings and memory NOT measurements (agent resident); '
    'quality grading pending (Local AI)'
)
TIMING_LABEL = 'NOT A MEASUREMENT (agent resident)'

HOME = Path('/data/data/com.termux/files/home')
ALT_HOME = Path('/termux-home')
LLAMA_VIS_DIR = HOME / 'llama.cpp-vis'
LLAMA_SERVER_BIN = LLAMA_VIS_DIR / 'build/bin/llama-server'
LLAMA_MTMD_CLI_BIN = LLAMA_VIS_DIR / 'build/bin/llama-mtmd-cli'
LIBGGML_CPU_SO = LLAMA_VIS_DIR / 'build/bin/libggml-cpu.so'

MODEL_GGUF = HOME / 'models/gemma-4-E2B-it-Q4_0.gguf'
MODEL_SOURCE_REPO = 'ggml-org/gemma-4-E2B-it-GGUF'
MODEL_SOURCE_REV = 'b4243c156154b6dca9324415f8c7ccc098b4aed1'
MODEL_EXPECTED_SHA256 = '8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52'
MODEL_EXPECTED_BYTES = 2841481184

MMPROJ_GGUF = HOME / 'models/vis0/mmproj-gemma-4-E2B-it-BF16.gguf'
MMPROJ_SOURCE_REPO = 'ggml-org/gemma-4-E2B-it-GGUF'
MMPROJ_SOURCE_REV = 'b4243c156154b6dca9324415f8c7ccc098b4aed1'
MMPROJ_EXPECTED_SHA256 = '711e1e8f43fa0664adbac493129be1e6c25b81af4b4cdea97c7d798b25c0a3a4'
MMPROJ_EXPECTED_BYTES = 986833664

BUILD_INFO_RECORD = {
    'source_repo': 'https://github.com/ggml-org/llama.cpp',
    'tag': 'b11716',
    'commit': '797de63b2fa7396dcc8caca76834c699ac84e1c3',
    'cmake_command': (
        '/data/data/com.termux/files/usr/bin/cmake -S . -B build -G "Unix Makefiles" '
        '-DCMAKE_BUILD_TYPE=Release '
        '-DCMAKE_C_COMPILER=/data/data/com.termux/files/usr/bin/cc '
        '-DCMAKE_CXX_COMPILER=/data/data/com.termux/files/usr/bin/c++ '
        '-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENCL=OFF'
    ),
    'make_command': 'make -C build -j6 llama-server llama-mtmd-cli',
    'compiler': 'Clang 21.1.8 (Target: aarch64-unknown-linux-android24)',
    'libggml_cpu_sdot_count': 1117,
    'libggml_cpu_fmla_8h_count': 456,
    'help_lines_relied_on': [
        '-mm,   --mmproj FILE                    path to a multimodal projector file. see tools/mtmd/README.md',
        '--image-min-tokens N                    minimum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)',
        '--image-max-tokens N                    maximum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)',
        '--swa-full                              use full-size SWA cache (default: false)',
        '-cram, --cache-ram N                    set the maximum cache size in MiB (default: 8192, 0 for decision models, -1 - no limit, 0 - disable)',
        '-rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat (\'on\', \'off\', or \'auto\', default: \'auto\' (detect from template))',
        '-lv,   --verbosity, --log-verbosity N   Set the verbosity threshold. Messages with a higher verbosity will be ignored.',
    ],
}

ROBOT_PORT = 8080
TEST_PORT = 8090
TEST_HOST = '127.0.0.1'
TEST_URL = f'http://{TEST_HOST}:{TEST_PORT}'
SERVER_TIMEOUT_S = 3600
STARTUP_TIMEOUT_S = 180
HTTP_TIMEOUT_S = 180

M2Q1_RUN_DIR = ROBOT / 'benchmark/relate_anything/quality1/runs/20261017T213604Z'
M2Q1_RESULTS_JSON = M2Q1_RUN_DIR / 'results.json'
M2Q1_FRAMES_DIR = M2Q1_RUN_DIR / 'frames'

FRAME_STEMS = (
    'bench_photos__blocked_1',
    'bench_photos__empty_2',
    'bench_photos__person_facing_150cm',
    'bench_photos__person_sideon_300cm',
    'bench_photos__roomsig_couch_1',
    'bench_photos__roomsig_refrigerator_1',
    'bench_photos__roomsig_toilet_1',
    'bench_photos__roomsig_toilet_2',
    'test_photos__chiara_close_back',
    'test_photos__abdel_close_right',
)

TOKENIZE_MARKERS = (
    '<start_of_turn>user',
    '<end_of_turn>',
    '<|turn>user',
    '<turn|>',
    '<|turn>model',
)

APPLY_TEMPLATE_MESSAGES = [{'role': 'user', 'content': 'Hi'}]

TEXT_INSTRUCTION = (
    "You are the robot's eyes. Using only the information below, describe in one or two short sentences "
    "what is in front of the robot. Do not mention anything that is not listed."
)

VISION_PROMPT = (
    'Describe in one or two short sentences what is in front of the robot. '
    'Mention only what you can see.'
)

VISION_BUDGETS = (70, 140, 280)
DOWNLOAD_PDF_DIR = Path('/storage/emulated/0/Download/vis0')


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_and_hash_imports(root=ROBOT):
    """Verify every imported project module resolved inside root and return {rel_path: sha256}."""
    root = Path(root).resolve()
    hashes = {}
    for rel_p, get_file in IMPORTED_MODULES:
        expected = (root / rel_p).resolve()
        actual = Path(get_file()).resolve()
        if actual != expected:
            raise RuntimeError(f'import origin mismatch for {rel_p}: loaded {actual}, expected {expected}')
        hashes[rel_p] = sha256_file(actual)
    return hashes


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def encode_image_data_uri(image_bytes):
    """Encode raw JPEG bytes as a base64 data URI for /chat/completions image_url."""
    b64 = base64.b64encode(bytes(image_bytes)).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def build_server_cmd(
    budget=None,
    server_bin=LLAMA_SERVER_BIN,
    model_path=MODEL_GGUF,
    mmproj_path=MMPROJ_GGUF,
    port=TEST_PORT,
    ctx_size=4096,
    timeout_s=SERVER_TIMEOUT_S,
):
    """Build the timeout-wrapped llama-server command on port 8090."""
    cmd = [
        'timeout',
        str(timeout_s),
        str(server_bin),
        '-m', str(model_path),
        '--mmproj', str(mmproj_path),
        '--host', TEST_HOST,
        '--port', str(port),
        '--ctx-size', str(ctx_size),
        '--threads', str(server_manager.THREADS),
        '--threads-batch', str(server_manager.BATCH_THREADS),
        '--parallel', '1',
        '--swa-full',
        '--cache-ram', '0',
        '--reasoning', 'off',
        '-lv', '4',
    ]
    if budget is not None:
        cmd.extend(['--image-min-tokens', str(budget), '--image-max-tokens', str(budget)])
    return cmd


def build_tokenize_payload(marker_text):
    return {'content': str(marker_text), 'with_pieces': True}


def build_apply_template_payload(messages=None):
    if messages is None:
        messages = APPLY_TEMPLATE_MESSAGES
    return {'messages': list(messages)}


def build_old_completion_payload(prompt_str):
    return {
        'prompt': str(prompt_str),
        'temperature': 0,
        'top_k': 1,
        'seed': 0,
        'n_predict': 96,
        'cache_prompt': False,
        'stop': get_stop_tokens('gemma'),
        'stream': False,
    }


def build_new_chat_text_payload(user_text):
    return {
        'messages': [{'role': 'user', 'content': str(user_text)}],
        'temperature': 0,
        'top_k': 1,
        'seed': 0,
        'max_tokens': 96,
        'n_predict': 96,
        'cache_prompt': False,
        'stream': False,
    }


def build_vision_chat_payload(image_bytes, text_prompt=VISION_PROMPT):
    data_uri = encode_image_data_uri(image_bytes)
    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': data_uri}},
                    {'type': 'text', 'text': str(text_prompt)},
                ],
            }
        ],
        'temperature': 0,
        'top_k': 1,
        'seed': 0,
        'max_tokens': 96,
        'n_predict': 96,
        'cache_prompt': False,
        'stream': False,
    }


def build_optional_yolo_vision_text(yolo_lines):
    return f'{VISION_PROMPT}\n\nObjects detected by YOLO:\n{yolo_lines}'


def classify_tokenized_marker(marker_str, tokens):
    """Determine whether the turn marker tag in marker_str tokenized as ONE special token or several pieces."""
    tag = marker_str[:marker_str.index('>') + 1] if '>' in marker_str else marker_str
    pieces = [t.get('piece') for t in tokens]
    tag_is_single = bool(tokens) and pieces[0] == tag
    whole_is_single = len(tokens) == 1 and tag_is_single
    if whole_is_single:
        summary = 'ONE special token'
    elif tag_is_single:
        summary = f'ONE special token ({tag}) + role token ({len(tokens)} tokens total)'
    else:
        summary = f'several plain-text pieces ({len(tokens)} pieces)'
    return {
        'input': marker_str,
        'tag': tag,
        'token_count': len(tokens),
        'tokens': tokens,
        'pieces': pieces,
        'tag_is_single_special_token': tag_is_single,
        'whole_input_is_single_token': whole_is_single,
        'classification': summary,
    }


def evaluate_template_verdict(chat_template, tokenize_results, rendered_hi):
    """Return CONFIRMED / REFUTED / UNCLEAR that Gemma 3 markers are wrong for this model."""
    by_input = {r['input']: r for r in tokenize_results}
    gemma3_plain = (
        not by_input['<start_of_turn>user']['tag_is_single_special_token']
        and not by_input['<end_of_turn>']['tag_is_single_special_token']
    )
    gemma4_special = (
        by_input['<|turn>user']['tag_is_single_special_token']
        and by_input['<turn|>']['tag_is_single_special_token']
        and by_input['<|turn>model']['tag_is_single_special_token']
    )
    template_uses_gemma4 = (
        '<|turn>' in str(chat_template)
        and '<turn|>' in str(chat_template)
        and '<start_of_turn>' not in str(chat_template)
        and '<end_of_turn>' not in str(chat_template)
    )
    rendered_uses_gemma4 = '<|turn>user\nHi<turn|>\n' in str(rendered_hi) and '<|turn>model\n' in str(rendered_hi)
    if gemma3_plain and gemma4_special and template_uses_gemma4 and rendered_uses_gemma4:
        return 'CONFIRMED'
    if (
        by_input['<start_of_turn>user']['tag_is_single_special_token']
        and by_input['<end_of_turn>']['tag_is_single_special_token']
    ):
        return 'REFUTED'
    return 'UNCLEAR'


def extract_yolo_lines_from_m2q1_record(rec):
    """Extract the Y-ONLY YOLO lines and verify chat.build_prompt reproduces M2-Q1's exact Y-ONLY prompt."""
    old_prompt = rec['gemma_prompts']['Y-ONLY']
    used_dets = rec.get('m2_detections') or rec.get('detections') or []
    yolo_lines = '\n'.join(
        f"- {d['class_name']} [{i}], {d['position']}, {d['distance']}"
        for i, d in enumerate(used_dets)
    )
    user_text = f'{TEXT_INSTRUCTION}\n\n{yolo_lines}'
    rebuilt = build_prompt('gemma', [{'role': 'user', 'content': user_text}])
    if rebuilt != old_prompt:
        raise RuntimeError(f"M2-Q1 Y-ONLY prompt mismatch for {rec.get('photo')}")
    return yolo_lines, user_text, old_prompt


def load_target_frames(
    results_json_path=M2Q1_RESULTS_JSON,
    frames_dir=M2Q1_FRAMES_DIR,
    limit=None,
    stems=FRAME_STEMS,
):
    """Load the target frames, enforcing exact frame identity and 640 px JPEG resolution."""
    with open(results_json_path, 'r', encoding='utf-8') as f:
        m2q1 = json.load(f)
    by_stem = {}
    for rec in m2q1.get('photos', []):
        frame_name = Path(rec['frame_file']).name
        stem = frame_name[:-4] if frame_name.endswith('.jpg') else frame_name
        by_stem[stem] = rec

    selected_stems = stems[:limit] if limit is not None else stems
    items = []
    for wanted in selected_stems:
        if wanted not in by_stem:
            raise RuntimeError(f'Required frame {wanted!r} not found in {results_json_path}')
        matched_stem = wanted
        rec = by_stem[matched_stem]
        frame_path = Path(frames_dir) / f'{matched_stem}.jpg'
        if not frame_path.is_file():
            raise RuntimeError(f'Frame file missing: {frame_path}')
        yolo_lines, user_text, old_prompt = extract_yolo_lines_from_m2q1_record(rec)
        frame_bytes = frame_path.read_bytes()
        with Image.open(io.BytesIO(frame_bytes)) as img_check:
            actual_size = [int(img_check.width), int(img_check.height)]
        if max(actual_size) != 640 or min(actual_size) <= 0:
            raise RuntimeError(
                f'Frame {frame_path} has dimensions {actual_size}, expected 640 px longest edge'
            )
        recorded_size = rec.get('frame_size')
        if recorded_size is not None and [int(recorded_size[0]), int(recorded_size[1])] != actual_size:
            raise RuntimeError(
                f'Frame size mismatch for {matched_stem}: results.json has {recorded_size}, JPEG is {actual_size}'
            )
        frame_sha = hashlib.sha256(frame_bytes).hexdigest()
        items.append({
            'requested_stem': wanted,
            'frame_stem': matched_stem,
            'match_note': 'exact match',
            'photo': rec['photo'],
            'frame_path': str(frame_path),
            'frame_sha256': frame_sha,
            'frame_bytes': len(frame_bytes),
            'frame_size': actual_size,
            'detection_count': rec.get('detection_count'),
            'yolo_lines': yolo_lines,
            'user_text': user_text,
            'old_prompt': old_prompt,
            'm2q1_replies': rec.get('replies'),
        })
    return items


def _parse_log_ts_ms(ts_str):
    parts = [int(p) for p in ts_str.split('.')]
    if len(parts) != 4:
        return None
    mins, secs, ms, us = parts
    return ((mins * 60 + secs) * 1000.0) + ms + (us / 1000.0)


def parse_server_log_delta(log_delta):
    """Extract logged image token counts and image-encode time (if printed) from a server log slice."""
    batch_tokens = [
        int(m) for m in re.findall(r'decoding image batch \d+/\d+, n_tokens_batch = (\d+)', log_delta)
    ]
    enc_batch_tokens = [
        int(m) for m in re.findall(r'encoding batch with \d+ entries and total (\d+) tokens', log_delta)
    ]
    sampler_tokens = [
        int(tot) - int(txt)
        for txt, tot in re.findall(r'init sampler, took [0-9.]+ ms, tokens: text = (\d+), total = (\d+)', log_delta)
        if int(tot) > int(txt)
    ]
    if batch_tokens:
        image_tokens = sum(batch_tokens)
    elif enc_batch_tokens:
        image_tokens = enc_batch_tokens[-1]
    elif sampler_tokens:
        image_tokens = sampler_tokens[-1]
    else:
        image_tokens = None

    encode_ms = None
    for pat in (
        r'image_encode[^0-9\n]*([0-9]+(?:\.[0-9]+)?)\s*ms',
        r'slice\s+encoded\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*ms',
        r'encoded\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*ms',
        r'encode_mtmd_chunk[^0-9\n]*([0-9]+(?:\.[0-9]+)?)\s*ms',
    ):
        m = re.search(pat, log_delta, flags=re.IGNORECASE)
        if m:
            encode_ms = float(m.group(1))
            break

    if encode_ms is None:
        m_enc = re.search(r'(\d+\.\d{2}\.\d{3}\.\d{3})\s+.*encoding mtmd batch from idx', log_delta)
        m_dec = re.search(r'(\d+\.\d{2}\.\d{3}\.\d{3})\s+.*decoding image batch 1/', log_delta)
        if m_enc and m_dec:
            t1 = _parse_log_ts_ms(m_enc.group(1))
            t2 = _parse_log_ts_ms(m_dec.group(1))
            if t1 is not None and t2 is not None and t2 >= t1:
                encode_ms = round(t2 - t1, 3)

    return {
        'image_tokens_logged': image_tokens,
        'image_tokens_batches': batch_tokens,
        'image_encode_ms_not_a_measurement': encode_ms,
    }


def read_vmhwm_kib_for_tree(timeout_pid):
    """Read VmHWM (in KiB) from /proc/<pid>/status for timeout_pid and any direct child (llama-server)."""
    best_kib = None
    best_pid = None
    best_name = None
    for status_path in glob.glob('/proc/[0-9]*/status'):
        try:
            pid = int(status_path.split('/')[2])
            text = Path(status_path).read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        ppid = None
        name = None
        hwm_kib = None
        for line in text.splitlines():
            if line.startswith('Name:'):
                name = line.split(':', 1)[1].strip()
            elif line.startswith('PPid:'):
                try:
                    ppid = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('VmHWM:'):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    hwm_kib = int(parts[1])
        if pid == timeout_pid or ppid == timeout_pid:
            if hwm_kib is not None and (best_kib is None or hwm_kib > best_kib):
                best_kib = hwm_kib
                best_pid = pid
                best_name = name
    return {'pid': best_pid, 'name': best_name, 'vmhwm_kib_not_a_measurement': best_kib}


def check_port_8080_stopped(port=ROBOT_PORT):
    """Ensure the robot server on port 8080 is stopped before starting a test server on 8090."""
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as resp:
            body = resp.read().decode('utf-8', errors='replace')
            raise RuntimeError(
                f'STOP: robot server on 127.0.0.1:{port} is still running ({resp.status}: {body}). '
                'Owner must stop it before running VIS-0.'
            )
    except urllib.error.HTTPError as e:
        body = ''
        if e.fp is not None:
            try:
                body = e.fp.read().decode('utf-8', errors='replace')
            except Exception:
                body = ''
        raise RuntimeError(
            f'STOP: robot server on 127.0.0.1:{port} responded with HTTP {e.code} ({e.reason}: {body}). '
            'Owner must stop it before running VIS-0.'
        ) from e
    except urllib.error.URLError as e:
        reason = getattr(e, 'reason', e)
        if not isinstance(reason, ConnectionRefusedError):
            raise RuntimeError(
                f'STOP: could not confirm 127.0.0.1:{port} is stopped (unexpected URLError: {e}). '
                'Owner must ensure port 8080 is down before running VIS-0.'
            ) from e
    check_port_free(port)
    return True


def check_port_free(port=TEST_PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((TEST_HOST, port))
        except OSError as e:
            raise RuntimeError(f'Port {TEST_HOST}:{port} is already in use: {e}') from e


def http_json(path, body=None, base_url=TEST_URL, timeout=HTTP_TIMEOUT_S):
    data = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(
        base_url + path,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='GET' if data is None else 'POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def extract_chat_reply(resp_json):
    """Extract assistant reply content, reasoning_content, finish_reason, usage, and timings from /chat/completions."""
    choices = resp_json.get('choices') or []
    if not choices:
        raise RuntimeError(f'No choices in /chat/completions response: {resp_json}')
    choice0 = choices[0]
    msg = choice0.get('message') or {}
    raw_content = msg.get('content')
    reasoning_content = msg.get('reasoning_content')
    content = '' if raw_content is None else str(raw_content).strip()
    finish_reason = choice0.get('finish_reason')
    usage = resp_json.get('usage') or {}
    timings = resp_json.get('timings') or {}
    prompt_tokens = usage.get('prompt_tokens')
    if prompt_tokens is None:
        prompt_tokens = timings.get('prompt_n')
    completion_tokens = usage.get('completion_tokens')
    if completion_tokens is None:
        completion_tokens = timings.get('predicted_n')
    return {
        'reply': content,
        'raw_content': raw_content,
        'reasoning_content': reasoning_content,
        'finish_reason': finish_reason,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'prompt_ms_not_a_measurement': timings.get('prompt_ms'),
        'predicted_ms_not_a_measurement': timings.get('predicted_ms'),
        'predicted_per_second_not_a_measurement': timings.get('predicted_per_second'),
        'timing_label': TIMING_LABEL,
    }


def extract_completion_reply(resp_json):
    """Extract reply content and timings from /completion."""
    if 'content' not in resp_json:
        raise RuntimeError(f'Unexpected /completion response: {resp_json}')
    raw_content = str(resp_json['content'])
    timings = resp_json.get('timings') or {}
    return {
        'reply': raw_content.strip(),
        'raw_content': raw_content,
        'stop_type': resp_json.get('stop_type'),
        'stopping_word': resp_json.get('stopping_word'),
        'tokens_evaluated': resp_json.get('tokens_evaluated', timings.get('prompt_n')),
        'tokens_predicted': resp_json.get('tokens_predicted', timings.get('predicted_n')),
        'prompt_ms_not_a_measurement': timings.get('prompt_ms'),
        'predicted_ms_not_a_measurement': timings.get('predicted_ms'),
        'predicted_per_second_not_a_measurement': timings.get('predicted_per_second'),
        'timing_label': TIMING_LABEL,
    }


def format_server_log_banner(started_utc, label, budget, cmd):
    return (
        f"\n{'=' * 72}\n"
        f"[{started_utc}] START {label} (budget={budget})\n"
        f"LABELS: {REPORT_LABELS}\n"
        f"TIMINGS/MEMORY: {TIMING_LABEL}\n"
        f"CMD: {' '.join(cmd)}\n"
        f"{'=' * 72}\n"
    )


class ManagedTestServer:
    """Manages a single timeout-wrapped llama-server instance on port 8090."""

    def __init__(self, budget, server_log_path, label=''):
        self.budget = budget
        self.server_log_path = Path(server_log_path)
        self.label = label or f'budget_{budget}'
        self.cmd = build_server_cmd(budget=budget)
        self.proc = None
        self.log_handle = None
        self.peak_vmhwm_kib = None
        self.server_pid = None
        self.server_name = None
        self.started_utc = None
        self.ended_utc = None
        self.startup_s_not_a_measurement = None
        self.exit_code = None

    def sample_vmhwm(self):
        if self.proc is None:
            return None
        info = read_vmhwm_kib_for_tree(self.proc.pid)
        hwm = info['vmhwm_kib_not_a_measurement']
        if hwm is not None and (self.peak_vmhwm_kib is None or hwm > self.peak_vmhwm_kib):
            self.peak_vmhwm_kib = hwm
            self.server_pid = info['pid']
            self.server_name = info['name']
        return self.peak_vmhwm_kib

    def start(self):
        check_port_8080_stopped(ROBOT_PORT)
        check_port_free(TEST_PORT)
        self.server_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.started_utc = utc_now()
        banner = format_server_log_banner(self.started_utc, self.label, self.budget, self.cmd)
        with open(self.server_log_path, 'a', encoding='utf-8') as lf:
            lf.write(banner)
        self.log_handle = open(self.server_log_path, 'ab')
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            self.cmd,
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
        )
        deadline = time.perf_counter() + STARTUP_TIMEOUT_S
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                self.exit_code = self.proc.returncode
                raise RuntimeError(
                    f'llama-server exited during startup with code {self.exit_code}; '
                    f'tail:\n{self.log_tail(25)}'
                )
            try:
                health = http_json('/health', timeout=2)
                if health.get('status') == 'ok':
                    self.startup_s_not_a_measurement = time.perf_counter() - t0
                    self.sample_vmhwm()
                    return
            except Exception:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f'llama-server did not become healthy within {STARTUP_TIMEOUT_S}s')

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_size(self):
        try:
            if self.log_handle:
                self.log_handle.flush()
            return self.server_log_path.stat().st_size
        except OSError:
            return 0

    def log_delta_from(self, offset):
        try:
            if self.log_handle:
                self.log_handle.flush()
            with open(self.server_log_path, 'rb') as f:
                f.seek(offset)
                return f.read().decode('utf-8', errors='replace')
        except OSError:
            return ''

    def log_tail(self, n_lines=25):
        try:
            if self.log_handle:
                self.log_handle.flush()
            lines = self.server_log_path.read_text(encoding='utf-8', errors='replace').splitlines()
            return '\n'.join(lines[-n_lines:])
        except OSError:
            return ''

    def stop(self):
        self.sample_vmhwm()
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
            self.exit_code = self.proc.returncode
        if self.log_handle is not None:
            try:
                self.log_handle.flush()
                self.log_handle.close()
            except OSError:
                pass
            self.log_handle = None
        self.ended_utc = utc_now()

    def summary(self):
        return {
            'label': self.label,
            'budget': self.budget,
            'command': self.cmd,
            'started_utc': self.started_utc,
            'ended_utc': self.ended_utc,
            'startup_s_not_a_measurement': self.startup_s_not_a_measurement,
            'server_pid': self.server_pid,
            'server_name': self.server_name,
            'peak_vmhwm_kib_not_a_measurement': self.peak_vmhwm_kib,
            'exit_code': self.exit_code,
            'timing_label': TIMING_LABEL,
        }


def run_step_a_template_check(target_frames, post_fn=http_json, get_fn=http_json, sample_hwm_fn=None):
    """Run Step A (template check + text A/B on the target frames) against the server on 8090."""
    props = get_fn('/props')
    chat_template = props.get('chat_template', '')
    dgs = props.get('default_generation_settings') or {}
    props_summary = {
        'model_alias': props.get('model_alias'),
        'model_path': props.get('model_path'),
        'model_ftype': props.get('model_ftype'),
        'total_slots': props.get('total_slots'),
        'n_ctx': dgs.get('n_ctx'),
        'build_info': props.get('build_info'),
        'bos_token': props.get('bos_token'),
        'eos_token': props.get('eos_token'),
        'modalities': props.get('modalities'),
    }

    tokenize_results = []
    for marker in TOKENIZE_MARKERS:
        res = post_fn('/tokenize', build_tokenize_payload(marker))
        tokens = res.get('tokens') or []
        tokenize_results.append(classify_tokenized_marker(marker, tokens))

    apply_res = post_fn('/apply-template', build_apply_template_payload(APPLY_TEMPLATE_MESSAGES))
    rendered_hi = apply_res.get('prompt', '')

    verdict = evaluate_template_verdict(chat_template, tokenize_results, rendered_hi)

    text_ab = []
    for item in target_frames:
        t0 = time.perf_counter()
        old_raw = post_fn('/completion', build_old_completion_payload(item['old_prompt']))
        old_wall_ms = (time.perf_counter() - t0) * 1000.0
        old_parsed = extract_completion_reply(old_raw)
        old_parsed['wall_ms_not_a_measurement'] = old_wall_ms
        if sample_hwm_fn:
            sample_hwm_fn()

        t1 = time.perf_counter()
        new_raw = post_fn('/chat/completions', build_new_chat_text_payload(item['user_text']))
        new_wall_ms = (time.perf_counter() - t1) * 1000.0
        new_parsed = extract_chat_reply(new_raw)
        new_parsed['wall_ms_not_a_measurement'] = new_wall_ms
        if sample_hwm_fn:
            sample_hwm_fn()

        text_ab.append({
            'frame_stem': item['frame_stem'],
            'photo': item['photo'],
            'yolo_lines': item['yolo_lines'],
            'old_completion': old_parsed,
            'new_chat_completions': new_parsed,
            'm2q1_saved_replies': item.get('m2q1_replies'),
            'timing_label': TIMING_LABEL,
        })

    return {
        'verdict': verdict,
        'verdict_statement': (
            f'{verdict}: Gemma 3 markers (<start_of_turn>, <end_of_turn>) are tokenized as multiple '
            'plain-text pieces, whereas Gemma 4 markers (<|turn>, <turn|>) are single special tokens '
            'and are used by the GGUF chat_template.'
            if verdict == 'CONFIRMED'
            else f'{verdict}: see tokenize_checks and rendered_hi.'
        ),
        'props_summary': props_summary,
        'chat_template': chat_template,
        'tokenize_checks': tokenize_results,
        'apply_template_hi': {
            'messages': APPLY_TEMPLATE_MESSAGES,
            'rendered_prompt': rendered_hi,
        },
        'text_ab': text_ab,
        'timing_label': TIMING_LABEL,
    }


def run_vision_budget_pass(
    budget,
    target_frames,
    server_factory,
    post_fn=http_json,
    with_yolo=False,
    log_fn=print,
):
    """Run one vision budget pass across target_frames with crash handling (restart at most once, stop on 2 consecutive crashes)."""
    label = f'vision_budget_{budget}_yolo' if with_yolo else f'vision_budget_{budget}'
    server = server_factory(budget=budget, label=label)
    server_summaries = []
    calls = []
    crashes = 0
    consecutive_crashes = 0
    restarted_once = False
    stopped_early = False
    stop_reason = None

    try:
        server.start()
    except Exception as e:
        server.stop()
        server_summaries.append(server.summary())
        return {
            'budget': budget,
            'with_yolo': with_yolo,
            'optional': bool(with_yolo),
            'crashes': 1,
            'stopped_early': True,
            'stop_reason': f'startup_failed: {type(e).__name__}: {e}',
            'servers': server_summaries,
            'calls': [],
            'timing_label': TIMING_LABEL,
        }

    try:
        for idx, item in enumerate(target_frames):
            img_bytes = Path(item['frame_path']).read_bytes()
            prompt_text = (
                build_optional_yolo_vision_text(item['yolo_lines'])
                if with_yolo
                else VISION_PROMPT
            )
            payload = build_vision_chat_payload(img_bytes, text_prompt=prompt_text)
            offset = server.log_size()
            t0 = time.perf_counter()
            try:
                resp = post_fn('/chat/completions', payload)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                server.sample_vmhwm()
                delta = server.log_delta_from(offset)
                log_info = parse_server_log_delta(delta)
                parsed = extract_chat_reply(resp)
                call_rec = {
                    'frame_stem': item['frame_stem'],
                    'photo': item['photo'],
                    'budget': budget,
                    'with_yolo': with_yolo,
                    'status': 'ok',
                    'crashed': False,
                    'reply': parsed['reply'],
                    'raw_content': parsed['raw_content'],
                    'reasoning_content': parsed['reasoning_content'],
                    'finish_reason': parsed['finish_reason'],
                    'prompt_tokens': parsed['prompt_tokens'],
                    'completion_tokens': parsed['completion_tokens'],
                    'image_tokens_logged': log_info['image_tokens_logged'],
                    'image_tokens_batches': log_info['image_tokens_batches'],
                    'prompt_ms_not_a_measurement': parsed['prompt_ms_not_a_measurement'],
                    'predicted_ms_not_a_measurement': parsed['predicted_ms_not_a_measurement'],
                    'predicted_per_second_not_a_measurement': parsed['predicted_per_second_not_a_measurement'],
                    'wall_ms_not_a_measurement': wall_ms,
                    'image_encode_ms_not_a_measurement': log_info['image_encode_ms_not_a_measurement'],
                    'timing_label': TIMING_LABEL,
                }
                calls.append(call_rec)
                consecutive_crashes = 0
                log_fn(
                    f"[{label}] {item['frame_stem']}: ok, img_tok={log_info['image_tokens_logged']}, "
                    f"prompt_tok={parsed['prompt_tokens']}, wall_ms={wall_ms:.1f} ({TIMING_LABEL})"
                )
            except Exception as e:
                wall_ms = (time.perf_counter() - t0) * 1000.0
                time.sleep(0.2)
                alive = server.is_alive()
                delta = server.log_delta_from(offset)
                log_info = parse_server_log_delta(delta)
                tail = server.log_tail(30)
                is_http_error = isinstance(e, urllib.error.HTTPError)
                is_crash = (not alive) or (
                    not is_http_error
                    and isinstance(e, (http.client.RemoteDisconnected, ConnectionResetError, urllib.error.URLError))
                )
                if is_crash:
                    crashes += 1
                    consecutive_crashes += 1
                call_rec = {
                    'frame_stem': item['frame_stem'],
                    'photo': item['photo'],
                    'budget': budget,
                    'with_yolo': with_yolo,
                    'status': 'crash' if is_crash else 'error',
                    'crashed': is_crash,
                    'error': f'{type(e).__name__}: {e}',
                    'server_exit_code': server.proc.poll() if server.proc is not None else None,
                    'reply': None,
                    'prompt_tokens': None,
                    'image_tokens_logged': log_info['image_tokens_logged'],
                    'prompt_ms_not_a_measurement': None,
                    'predicted_per_second_not_a_measurement': None,
                    'wall_ms_not_a_measurement': wall_ms,
                    'image_encode_ms_not_a_measurement': log_info['image_encode_ms_not_a_measurement'],
                    'log_tail': tail,
                    'timing_label': TIMING_LABEL,
                }
                calls.append(call_rec)
                log_fn(f"[{label}] {item['frame_stem']}: {'CRASH' if is_crash else 'ERROR'} ({e})")

                if is_crash:
                    server.stop()
                    server_summaries.append(server.summary())
                    if consecutive_crashes >= 2:
                        stopped_early = True
                        stop_reason = 'two_consecutive_crashes_at_budget'
                        break
                    if restarted_once:
                        stopped_early = True
                        stop_reason = 'already_restarted_once_at_budget'
                        break
                    if idx + 1 < len(target_frames):
                        restarted_once = True
                        server = server_factory(budget=budget, label=f'{label}_restart1')
                        try:
                            server.start()
                        except Exception as re_err:
                            crashes += 1
                            consecutive_crashes += 1
                            server.stop()
                            server_summaries.append(server.summary())
                            stopped_early = True
                            stop_reason = f'restart_failed: {type(re_err).__name__}: {re_err}'
                            break
    finally:
        if server.proc is not None and (not server_summaries or server_summaries[-1]['started_utc'] != server.started_utc):
            server.stop()
            server_summaries.append(server.summary())

    return {
        'budget': budget,
        'with_yolo': with_yolo,
        'optional': bool(with_yolo),
        'crashes': crashes,
        'stopped_early': stopped_early,
        'stop_reason': stop_reason,
        'servers': server_summaries,
        'calls': calls,
        'timing_label': TIMING_LABEL,
    }


def _draw_wrapped_lines(draw, x, y, lines, width_chars=100, font_size=13, line_gap=4, fill='#111111'):
    font = ImageFont.load_default(size=font_size)
    for raw_line in lines:
        text = str(raw_line)
        wrapped = textwrap.wrap(text, width=width_chars) if text else ['']
        for part in wrapped:
            draw.text((x, y), part, fill=fill, font=font)
            y += font_size + line_gap
    return y


def build_vis0_sheet_pdf(results_data, target_frames, pdf_path):
    """Build vis0_sheet.pdf (PIL only, < 20 MB): Page 1 = template check; then 1 page per frame."""
    page_w, page_h = 960, 1480
    total_pages = 1 + len(target_frames)
    pages = []

    # Page 1: Template Check Results
    p1 = Image.new('RGB', (page_w, page_h), 'white')
    draw1 = ImageDraw.Draw(p1)
    draw1.text(
        (24, 18),
        f'VIS-0 Template Check & Vision Feasibility  --  Page 1 / {total_pages}',
        fill='#000000',
        font=ImageFont.load_default(size=18),
    )
    y = 48
    y = _draw_wrapped_lines(
        draw1, 24, y, [f'Labels: {REPORT_LABELS}'], width_chars=106, font_size=12, fill='#333333'
    )
    y += 10

    tc = results_data['template_check']
    draw1.text(
        (24, y),
        f"Step A Verdict: {tc['verdict']}",
        fill='#000000',
        font=ImageFont.load_default(size=16),
    )
    y += 24
    y = _draw_wrapped_lines(draw1, 32, y, [tc['verdict_statement']], width_chars=102, font_size=13)
    y += 12

    draw1.text(
        (24, y),
        '1. /tokenize with "with_pieces": true:',
        fill='#000000',
        font=ImageFont.load_default(size=15),
    )
    y += 22
    tok_lines = []
    for t_row in tc['tokenize_checks']:
        pieces_str = ', '.join(
            f"{t['id']}:{t['piece']!r}" for t in t_row['tokens']
        )
        tok_lines.append(
            f"- {t_row['input']!r} -> {t_row['classification']}  [{pieces_str}]"
        )
    y = _draw_wrapped_lines(draw1, 32, y, tok_lines, width_chars=102, font_size=13)
    y += 12

    draw1.text(
        (24, y),
        '2. /apply-template with [{"role": "user", "content": "Hi"}] (verbatim repr):',
        fill='#000000',
        font=ImageFont.load_default(size=15),
    )
    y += 22
    rendered_hi = tc['apply_template_hi']['rendered_prompt']
    y = _draw_wrapped_lines(draw1, 32, y, [repr(rendered_hi)], width_chars=102, font_size=13)
    y += 12

    draw1.text(
        (24, y),
        '3. /props chat_template header & turn-token lines:',
        fill='#000000',
        font=ImageFont.load_default(size=15),
    )
    y += 22
    ct_lines = tc['chat_template'].splitlines()
    ct_excerpt = ct_lines[:6] + ['...'] + [
        ln.strip() for ln in ct_lines if '<|turn>' in ln or '<turn|>' in ln
    ][:8]
    y = _draw_wrapped_lines(draw1, 32, y, ct_excerpt, width_chars=102, font_size=12, fill='#222222')
    y += 12

    draw1.text(
        (24, y),
        '4. Vision Budgets Summary (NOT measurements; agent resident):',
        fill='#000000',
        font=ImageFont.load_default(size=15),
    )
    y += 22
    vis_summary_lines = []
    for b_key, b_pass in results_data.get('vision_budgets', {}).items():
        ok_calls = [c for c in b_pass['calls'] if c['status'] == 'ok']
        img_toks = sorted({c['image_tokens_logged'] for c in ok_calls if c.get('image_tokens_logged') is not None})
        vis_summary_lines.append(
            f"- Budget {b_key}: {len(ok_calls)}/{len(b_pass['calls'])} ok, "
            f"crashes={b_pass['crashes']}, logged_image_tokens={img_toks}"
        )
    opt_pass = results_data.get('optional_140_yolo')
    if opt_pass:
        ok_opt = [c for c in opt_pass['calls'] if c['status'] == 'ok']
        vis_summary_lines.append(
            f"- Optional 140+YOLO: {len(ok_opt)}/{len(opt_pass['calls'])} ok, crashes={opt_pass['crashes']}"
        )
    y = _draw_wrapped_lines(draw1, 32, y, vis_summary_lines, width_chars=102, font_size=13)
    pages.append(p1)

    # Index lookup for per-frame pages
    text_by_stem = {r['frame_stem']: r for r in tc['text_ab']}
    vis_by_budget_stem = {}
    for b_key, b_pass in results_data.get('vision_budgets', {}).items():
        vis_by_budget_stem[str(b_key)] = {c['frame_stem']: c for c in b_pass['calls']}
    opt_by_stem = {}
    if opt_pass:
        opt_by_stem = {c['frame_stem']: c for c in opt_pass['calls']}

    for idx, item in enumerate(target_frames, 2):
        stem = item['frame_stem']
        page = Image.new('RGB', (page_w, page_h), 'white')
        draw = ImageDraw.Draw(page)
        draw.text(
            (24, 16),
            f"{stem} ({item['photo']})  --  Page {idx} / {total_pages}",
            fill='#000000',
            font=ImageFont.load_default(size=17),
        )

        with Image.open(item['frame_path']) as raw_img:
            disp_img = ImageOps.exif_transpose(raw_img).convert('RGB')
        disp_img.thumbnail((560, 420), Image.Resampling.BILINEAR)
        img_x = (page_w - disp_img.width) // 2
        img_y = 44
        page.paste(disp_img, (img_x, img_y))
        y = img_y + disp_img.height + 12

        draw.text((24, y), 'YOLO detections (Y-ONLY input lines):', fill='#000000', font=ImageFont.load_default(size=14))
        y += 18
        yolo_preview = item['yolo_lines'].splitlines()
        if len(yolo_preview) > 6:
            yolo_preview = ['; '.join(yolo_preview[:6]), '; '.join(yolo_preview[6:])]
        else:
            yolo_preview = ['; '.join(yolo_preview)] if yolo_preview else ['(none)']
        y = _draw_wrapped_lines(draw, 32, y, yolo_preview, width_chars=106, font_size=12, fill='#333333')
        y += 8

        t_rec = text_by_stem.get(stem, {})
        old_c = t_rec.get('old_completion', {})
        new_c = t_rec.get('new_chat_completions', {})

        draw.text(
            (24, y),
            f"Text OLD (/completion, Gemma 3 markers) [prompt_tok={old_c.get('tokens_evaluated')}]:",
            fill='#8b0000',
            font=ImageFont.load_default(size=14),
        )
        y += 18
        old_text = old_c.get('raw_content', old_c.get('reply', ''))
        y = _draw_wrapped_lines(
            draw, 32, y, [repr(old_text) if ('<' in str(old_text) or '\n' in str(old_text)) else str(old_text)],
            width_chars=104, font_size=13,
        )
        y += 8

        draw.text(
            (24, y),
            f"Text NEW (/chat/completions, GGUF template) [prompt_tok={new_c.get('prompt_tokens')}]:",
            fill='#004d00',
            font=ImageFont.load_default(size=14),
        )
        y += 18
        new_reply_text = new_c.get('reply') or ''
        if not new_reply_text and new_c.get('reasoning_content'):
            new_reply_text = f"[empty content; reasoning_content]: {new_c['reasoning_content']}"
        y = _draw_wrapped_lines(draw, 32, y, new_reply_text.splitlines() or ['(empty reply)'], width_chars=104, font_size=13)
        y += 10

        for b in VISION_BUDGETS:
            b_str = str(b)
            v_call = vis_by_budget_stem.get(b_str, {}).get(stem)
            if v_call is None:
                hdr = f'Vision budget {b}: (not run — budget stopped early)'
                body_lines = ['(not run)']
            elif v_call['status'] != 'ok':
                hdr = f"Vision budget {b}: {v_call['status'].upper()} ({v_call.get('error', '')})"
                body_lines = (v_call.get('log_tail') or '').splitlines()[-3:] or ['(no output)']
            else:
                hdr = (
                    f"Vision budget {b} [prompt_tokens={v_call.get('prompt_tokens')}, "
                    f"image_tokens_logged={v_call.get('image_tokens_logged')}]:"
                )
                v_txt = v_call.get('reply') or ''
                if not v_txt and v_call.get('reasoning_content'):
                    v_txt = f"[empty content; reasoning_content]: {v_call['reasoning_content']}"
                body_lines = v_txt.splitlines() or ['(empty reply)']
            draw.text((24, y), hdr, fill='#003366', font=ImageFont.load_default(size=14))
            y += 18
            y = _draw_wrapped_lines(draw, 32, y, body_lines, width_chars=104, font_size=13)
            y += 8

        if stem in opt_by_stem:
            o_call = opt_by_stem[stem]
            hdr = (
                f"Optional Vision 140 + YOLO [prompt_tokens={o_call.get('prompt_tokens')}, "
                f"image_tokens_logged={o_call.get('image_tokens_logged')}]:"
            )
            o_txt = o_call.get('reply') or ''
            if not o_txt and o_call.get('reasoning_content'):
                o_txt = f"[empty content; reasoning_content]: {o_call['reasoning_content']}"
            draw.text((24, y), hdr, fill='#4b0082', font=ImageFont.load_default(size=14))
            y += 18
            y = _draw_wrapped_lines(draw, 32, y, o_txt.splitlines() or ['(empty reply)'], width_chars=104, font_size=13)

        pages.append(page)

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, format='PDF', resolution=100.0, save_all=True, append_images=pages[1:])
    if pdf_path.stat().st_size >= 20 * 1024 * 1024:
        raise RuntimeError(f'{pdf_path} exceeds 20 MB limit')


def git_base_commit(root=ROBOT):
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


RUN_OUTPUT_FILES = ('results.json', 'manifest.json', 'server.log', 'run.log', 'vis0_sheet.pdf')


def resolve_and_validate_out_dir(out_dir=None, runs_root=None):
    """Resolve out_dir under benchmark/gemma_vision/vis0/runs/ and reject outside paths or existing outputs."""
    allowed_root = Path(HERE / 'runs' if runs_root is None else runs_root).resolve()
    if out_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        resolved = (allowed_root / stamp).resolve()
    else:
        resolved = Path(out_dir).resolve()
    if resolved == allowed_root or not resolved.is_relative_to(allowed_root):
        raise ValueError(
            f'--out must resolve to a subdirectory inside {allowed_root}, got {resolved}'
        )
    for fname in RUN_OUTPUT_FILES:
        existing = resolved / fname
        if existing.exists():
            raise FileExistsError(f'Refusing to overwrite existing output artifact: {existing}')
    return resolved


def run_harness(limit=None, out_dir=None, skip_optional=False):
    start_utc = utc_now()
    t0 = time.perf_counter()
    out_dir = resolve_and_validate_out_dir(out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = out_dir / 'run.log'
    server_log_path = out_dir / 'server.log'

    def log(msg):
        line = f'[{utc_now()}] {msg}'
        print(line, flush=True)
        with open(run_log_path, 'a', encoding='utf-8') as lf:
            lf.write(line + '\n')

    log(f'START VIS-0 harness ({REPORT_LABELS})')
    check_port_8080_stopped(ROBOT_PORT)

    for req_file, exp_bytes, exp_sha in (
        (MODEL_GGUF, MODEL_EXPECTED_BYTES, MODEL_EXPECTED_SHA256),
        (MMPROJ_GGUF, MMPROJ_EXPECTED_BYTES, MMPROJ_EXPECTED_SHA256),
    ):
        if not req_file.is_file():
            raise RuntimeError(f'Missing required weight file: {req_file}')
        actual_bytes = req_file.stat().st_size
        actual_sha = sha256_file(req_file)
        if actual_bytes != exp_bytes or actual_sha != exp_sha:
            raise RuntimeError(
                f'Weight verification failed for {req_file}: '
                f'bytes={actual_bytes} (expected {exp_bytes}), sha256={actual_sha} (expected {exp_sha})'
            )

    target_frames = load_target_frames(limit=limit)
    log(f'Loaded {len(target_frames)} target frames from {M2Q1_RUN_DIR} (limit={limit})')

    all_server_summaries = []

    # Step A runs on the first server instance (started with budget 70), then we run Step B budget 70 on it,
    # or we run Step A on a dedicated budget=70 server and then run budgets 70, 140, 280 cleanly.
    # Starting one server for Step A (with --image-min-tokens 70 --image-max-tokens 70) and stopping it cleanly
    # before the Step B budget loop keeps Step A and each Step B budget completely isolated.
    step_a_server = ManagedTestServer(budget=VISION_BUDGETS[0], server_log_path=server_log_path, label='step_a_template_check')
    try:
        step_a_server.start()
        log(f'Step A server ready on {TEST_URL} (pid={step_a_server.server_pid})')
        template_check = run_step_a_template_check(
            target_frames,
            post_fn=http_json,
            get_fn=http_json,
            sample_hwm_fn=step_a_server.sample_vmhwm,
        )
        log(f"Step A complete: verdict={template_check['verdict']}")
    finally:
        step_a_server.stop()
        all_server_summaries.append(step_a_server.summary())

    def server_factory(budget, label):
        return ManagedTestServer(budget=budget, server_log_path=server_log_path, label=label)

    vision_budgets = {}
    for budget in VISION_BUDGETS:
        log(f'Starting Step B vision pass for budget={budget}')
        b_res = run_vision_budget_pass(
            budget=budget,
            target_frames=target_frames,
            server_factory=server_factory,
            post_fn=http_json,
            with_yolo=False,
            log_fn=log,
        )
        vision_budgets[str(budget)] = b_res
        all_server_summaries.extend(b_res['servers'])

    all_budgets_ok = all(
        b_res['crashes'] == 0
        and not b_res['stopped_early']
        and len(b_res['calls']) == len(target_frames)
        and all(c['status'] == 'ok' for c in b_res['calls'])
        for b_res in vision_budgets.values()
    )

    optional_140_yolo = None
    if all_budgets_ok and not skip_optional:
        log('All budgets (70, 140, 280) succeeded without crash; running optional 140 + YOLO pass')
        optional_140_yolo = run_vision_budget_pass(
            budget=140,
            target_frames=target_frames,
            server_factory=server_factory,
            post_fn=http_json,
            with_yolo=True,
            log_fn=log,
        )
        all_server_summaries.extend(optional_140_yolo['servers'])

    peak_vmhwm_kib = max(
        (s['peak_vmhwm_kib_not_a_measurement'] for s in all_server_summaries if s['peak_vmhwm_kib_not_a_measurement'] is not None),
        default=None,
    )

    results_payload = {
        'task': 'VIS-0',
        'labels': REPORT_LABELS,
        'timing_label': TIMING_LABEL,
        'limit': limit,
        'template_check': template_check,
        'vision_budgets': vision_budgets,
        'optional_140_yolo': optional_140_yolo,
        'peak_vmhwm_kib_not_a_measurement': peak_vmhwm_kib,
        'servers': all_server_summaries,
    }
    write_json(out_dir / 'results.json', results_payload)

    pdf_path = out_dir / 'vis0_sheet.pdf'
    build_vis0_sheet_pdf(results_payload, target_frames, pdf_path)
    DOWNLOAD_PDF_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, DOWNLOAD_PDF_DIR / 'vis0_sheet.pdf')
    shutil.copy2(pdf_path, DOWNLOAD_PDF_DIR / f'vis0_sheet_{out_dir.name}.pdf')
    log(f'Built {pdf_path.name} ({pdf_path.stat().st_size} B) and copied to {DOWNLOAD_PDF_DIR}')

    end_utc = utc_now()
    elapsed_s = time.perf_counter() - t0
    code_hashes = verify_and_hash_imports(ROBOT)

    manifest = {
        'task': 'VIS-0',
        'labels': REPORT_LABELS,
        'timing_label': TIMING_LABEL,
        'base_commit': git_base_commit(ROBOT),
        'start_utc': start_utc,
        'end_utc': end_utc,
        'elapsed_s_not_a_measurement': elapsed_s,
        'limit': limit,
        'code_sha256': code_hashes,
        'build': {
            **BUILD_INFO_RECORD,
            'llama_server_bin': str(LLAMA_SERVER_BIN),
            'llama_server_sha256': sha256_file(LLAMA_SERVER_BIN),
            'llama_mtmd_cli_bin': str(LLAMA_MTMD_CLI_BIN),
            'llama_mtmd_cli_sha256': sha256_file(LLAMA_MTMD_CLI_BIN),
            'libggml_cpu_so': str(LIBGGML_CPU_SO),
            'libggml_cpu_so_sha256': sha256_file(LIBGGML_CPU_SO),
        },
        'weights': {
            'model_gguf': {
                'path': str(MODEL_GGUF),
                'source_repo': MODEL_SOURCE_REPO,
                'source_revision': MODEL_SOURCE_REV,
                'file_name': MODEL_GGUF.name,
                'bytes': MODEL_EXPECTED_BYTES,
                'sha256': MODEL_EXPECTED_SHA256,
            },
            'mmproj_gguf': {
                'path': str(MMPROJ_GGUF),
                'source_repo': MMPROJ_SOURCE_REPO,
                'source_revision': MMPROJ_SOURCE_REV,
                'file_name': MMPROJ_GGUF.name,
                'bytes': MMPROJ_EXPECTED_BYTES,
                'sha256': MMPROJ_EXPECTED_SHA256,
            },
        },
        'server_commands': {
            str(b): build_server_cmd(budget=b) for b in VISION_BUDGETS
        },
        'server_props': template_check['props_summary'],
        'peak_vmhwm_kib_not_a_measurement': peak_vmhwm_kib,
        'interpreter': {
            'executable': sys.executable,
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
            'pil_version': Image.__version__,
        },
        'frames': [
            {
                'requested_stem': item['requested_stem'],
                'frame_stem': item['frame_stem'],
                'match_note': item['match_note'],
                'photo': item['photo'],
                'frame_path': item['frame_path'],
                'frame_sha256': item['frame_sha256'],
                'frame_bytes': item['frame_bytes'],
                'frame_size': item['frame_size'],
                'detection_count': item['detection_count'],
            }
            for item in target_frames
        ],
    }
    write_json(out_dir / 'manifest.json', manifest)
    log(f'DONE VIS-0 harness in {elapsed_s:.2f}s ({TIMING_LABEL}) -> {out_dir}')
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N frames (smoke run)')
    parser.add_argument('--out', type=Path, default=None, help='Output directory (default: runs/<UTC>)')
    parser.add_argument('--skip-optional', action='store_true', help='Skip optional 140+YOLO pass')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be >= 1')
    run_harness(limit=args.limit, out_dir=args.out, skip_optional=args.skip_optional)


if __name__ == '__main__':
    main()
