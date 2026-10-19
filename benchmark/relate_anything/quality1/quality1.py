"""TASK M2-Q1 — Motors-off RelateAnything M2 quality & Gemma A/B desk test harness.

Accuracy VALID (temperature 0); timings NOT measurements (agent resident); grading pending (Local AI, blind).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import textwrap
import time
import urllib.request

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parents[2]
LIVE_TERMUX_ROBOT = '/data/data/com.termux/files/home/robot'
sys.path[:] = [str(ROBOT)] + [p for p in sys.path if p != str(ROBOT)]

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps

import detect_person
import picker
from picker import URL
import relate
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
    ('benchmark/relate_anything/quality1/quality1.py', lambda: __file__),
    ('relate.py', lambda: relate.__file__),
    ('detect_person.py', lambda: detect_person.__file__),
    ('picker.py', lambda: picker.__file__),
    ('chat.py', lambda: chat.__file__),
    ('server_manager.py', lambda: server_manager.__file__),
)

TIMING_LABEL = 'NOT A MEASUREMENT (agent resident)'
ACCURACY_LABEL = 'Accuracy VALID (temperature 0); timings NOT measurements (agent resident); grading pending (Local AI, blind).'
BLIND_SEED = 20261017
FALLBACK_M2_DIR = '/termux-home/models/relate_anything/relsgg-vits16'
EXPECTED_GGUF_NAME = 'gemma-4-E2B-it-Q4_0.gguf'
HTTP_TIMEOUT_S = 60

INSTRUCTION = (
    "You are the robot's eyes. Using only the information below, describe in one or two short sentences "
    "what is in front of the robot. Do not mention anything that is not listed."
)

TEMPLATE_TOKEN_RE = re.compile(
    r'</?(?:start|end)(?:_of)+_turn>(?:user|model)?|<(?:eos|bos)>|'
    r'</?(?:start|end)(?:_of)*_?(?:turn)?(?:user|model)?$|'
    r'(?:^|\n)\s*</?(?:s(?:t(?:a(?:r(?:t)?)?)?)?|e(?:n(?:d)?)?)?\s*$'
)


def verify_and_hash_imports(root=ROBOT):
    """Verify every imported project module resolved to root and return {rel_path: sha256}."""
    root = Path(root).resolve()
    hashes = {}
    for rel_p, get_file in IMPORTED_MODULES:
        expected = (root / rel_p).resolve()
        actual = Path(get_file()).resolve()
        if actual != expected:
            raise RuntimeError(f'import origin mismatch for {rel_p}: loaded {actual}, expected {expected}')
        hashes[rel_p] = relate.sha256(actual)
    return hashes


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def http_json(path, body=None, url=URL, timeout=HTTP_TIMEOUT_S):
    data = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url + path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def discover_photos(root=ROBOT):
    root = Path(root)
    paths = [p for folder in ('bench_photos', 'test_photos') for p in (root / folder).glob('*.jpg')]
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def prepare_frame(src_path, dst_path):
    """Open with PIL, exif_transpose, RGB, thumbnail((640, 640), BILINEAR), save JPEG q=85, then re-open."""
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as src:
        img = ImageOps.exif_transpose(src).convert('RGB')
    img.thumbnail((640, 640), Image.Resampling.BILINEAR)
    img.save(dst_path, format='JPEG', quality=85)
    with Image.open(dst_path) as saved:
        return ImageOps.exif_transpose(saved).convert('RGB')


def enrich_detections(raw_detections, frame_size):
    """Attach position and coarse distance bucket from detect_person.scene_from_detections to each detection."""
    width, height = frame_size
    enriched = []
    for d in raw_detections:
        _, _, pos, dist, *_ = detect_person.scene_from_detections([d], width, height)[0]
        enriched.append({
            'class_name': str(d['class_name']),
            'conf': float(d['conf']),
            'box_xyxy': [float(x) for x in d['box_xyxy']],
            'position': pos,
            'distance': dist,
        })
    return enriched


def select_m2_detections(detections):
    """Reproduce Relate.relations() >32 rule: keep 32 highest-confidence boxes in detector order."""
    if len(detections) > relate.MAX_BOXES:
        keep = sorted(sorted(range(len(detections)), key=lambda i: -detections[i]['conf'])[:relate.MAX_BOXES])
        detections = [detections[i] for i in keep]
    return detections


def run_m2_once(head, image, detections):
    """Reproduce Relate.relations() without thread pinning."""
    if len(detections) < 2:
        return {
            'ran': False,
            'boxes_used': len(detections),
            'used_detections': list(detections),
            'rows': [],
            'shown_rows': [],
            'shown_text': relate.TOO_FEW,
            'infer_ms_not_a_measurement': None,
            'timing_label': TIMING_LABEL,
        }
    used = select_m2_detections(detections)
    started = time.perf_counter()
    rows = head.infer(image, used)
    ms = (time.perf_counter() - started) * 1000.0
    return {
        'ran': True,
        'boxes_used': len(used),
        'used_detections': used,
        'rows': rows,
        'shown_rows': relate.chosen(rows),
        'shown_text': relate.context(rows),
        'infer_ms_not_a_measurement': ms,
        'timing_label': TIMING_LABEL,
    }


def rows_identical(rows_a, rows_b):
    if len(rows_a) != len(rows_b):
        return False
    for a, b in zip(rows_a, rows_b):
        if (a['subject_idx'], a['object_idx'], a['predicate'], a['pass']) != (
            b['subject_idx'], b['object_idx'], b['predicate'], b['pass']
        ):
            return False
        if a['score'] != b['score'] or a['threshold'] != b['threshold']:
            return False
    return True


def format_yolo_lines(used_detections):
    return '\n'.join(
        f"- {d['class_name']} [{i}], {d['position']}, {d['distance']}"
        for i, d in enumerate(used_detections)
    )


def build_variant_prompts(used_detections, m2_context_text):
    yolo_lines = format_yolo_lines(used_detections)
    body_y_only = f"{INSTRUCTION}\n\n{yolo_lines}"
    body_y_m2 = f"{INSTRUCTION}\n\n{yolo_lines}\n{m2_context_text}"
    return {
        'Y-ONLY': build_prompt('gemma', [{'role': 'user', 'content': body_y_only}]),
        'Y+M2': build_prompt('gemma', [{'role': 'user', 'content': body_y_m2}]),
    }


def strip_template_tokens(text):
    return TEMPLATE_TOKEN_RE.sub('', str(text)).strip()


def call_gemma(prompt, post=http_json):
    payload = {
        'prompt': prompt,
        'temperature': 0,
        'top_k': 1,
        'n_predict': 96,
        'cache_prompt': False,
        'seed': 0,
        'stop': get_stop_tokens('gemma'),
        'stream': False,
    }
    started = time.perf_counter()
    res = post('/completion', payload)
    ms = (time.perf_counter() - started) * 1000.0
    if 'content' not in res:
        raise RuntimeError(f'unexpected /completion response: {res}')
    return strip_template_tokens(res['content']), ms


def check_server(get=http_json):
    try:
        health = get('/health')
        props = get('/props')
    except Exception as e:
        raise RuntimeError(
            f'STOP: server on 127.0.0.1:8080 unreachable ({e}). '
            'Ask the owner to start it in native Termux: cd ~/robot && python3 server_manager.py setup_q4'
        ) from e
    if health.get('status') != 'ok':
        raise RuntimeError(
            f'STOP: /health returned {health}. '
            'Ask the owner to start it in native Termux: cd ~/robot && python3 server_manager.py setup_q4'
        )
    model_path = str(props.get('model_path', ''))
    if Path(model_path).name != EXPECTED_GGUF_NAME:
        raise RuntimeError(
            f'STOP: loaded model is {model_path!r}, expected {EXPECTED_GGUF_NAME}. '
            'Ask the owner to start it in native Termux: cd ~/robot && python3 server_manager.py setup_q4'
        )
    gguf_sha256 = None
    for candidate in (model_path, model_path.replace('/data/data/com.termux/files/home', '/termux-home')):
        if os.path.isfile(candidate):
            gguf_sha256 = relate.sha256(candidate)
            break
    dgs = props.get('default_generation_settings', {})
    summary = {
        'model_alias': props.get('model_alias'),
        'model_path': model_path,
        'model_ftype': props.get('model_ftype'),
        'total_slots': props.get('total_slots'),
        'n_ctx': dgs.get('n_ctx'),
        'build_info': props.get('build_info'),
        'bos_token': props.get('bos_token'),
        'eos_token': props.get('eos_token'),
        'modalities': props.get('modalities'),
        'is_sleeping': props.get('is_sleeping'),
    }
    return {'model_path': model_path, 'gguf_sha256': gguf_sha256, 'props_summary': summary}


def assign_blind_key(qualifying_photos, seed=BLIND_SEED):
    """Assign Y-ONLY and Y+M2 to X and Y with random.Random(20261017), one coin per photo in sorted photo order."""
    rng = random.Random(seed)
    key = {}
    for photo in sorted(qualifying_photos):
        first = rng.choice(('Y-ONLY', 'Y+M2'))
        second = 'Y+M2' if first == 'Y-ONLY' else 'Y-ONLY'
        key[photo] = {'X': first, 'Y': second}
    return key


def annotate_frame(image, used_detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    palette = ('#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#17becf', '#e377c2', '#8c564b')
    for i, det in enumerate(used_detections):
        color = palette[i % len(palette)]
        x1, y1, x2, y2 = det['box_xyxy']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tx = max(2, min(annotated.width - 80, x1 + 2))
        ty = max(2, min(annotated.height - 18, y1 + 2))
        draw.text((tx, ty), f"{i}: {det['class_name']}", fill=color, stroke_width=2, stroke_fill='white', font_size=14)
    return annotated


def _draw_wrapped_lines(draw, x, y, lines, width_chars=96, font_size=14, line_gap=4, fill='#111111'):
    font = ImageFont.load_default(size=font_size)
    for raw_line in lines:
        wrapped = textwrap.wrap(str(raw_line), width=width_chars) if raw_line else ['']
        for part in wrapped:
            draw.text((x, y), part, fill=fill, font=font)
            y += font_size + line_gap
    return y


def build_sheet_m2_pdf(photo_records, frames_by_photo, pdf_path):
    """Build sheet_m2.pdf with PIL only: 1 page per >=2-box photo, plus 1 summary page for <2-box photos."""
    qualifying = [r for r in photo_records if r['m2_ran']]
    too_few = [r for r in photo_records if not r['m2_ran']]
    total_pages = len(qualifying) + 1
    pages = []
    page_w, page_h = 900, 1380

    for idx, rec in enumerate(qualifying, 1):
        page = Image.new('RGB', (page_w, page_h), 'white')
        draw = ImageDraw.Draw(page)
        header = f"{rec['photo']}  --  Page {idx} / {total_pages}"
        sub = (
            f"Frame: {rec['frame_size'][0]}x{rec['frame_size'][1]}  |  "
            f"Detections: {rec['detection_count']} (M2 boxes: {rec['m2_boxes_used']})  |  "
            f"M2 2x identical: {rec['m2_identical_twice']}"
        )
        draw.text((24, 18), header, fill='#000000', font=ImageFont.load_default(size=18))
        draw.text((24, 44), sub, fill='#444444', font=ImageFont.load_default(size=13))

        used_dets = rec.get('m2_detections', rec['detections'][:rec['m2_boxes_used']])
        annotated = annotate_frame(frames_by_photo[rec['photo']], used_dets)
        img_x = (page_w - annotated.width) // 2
        img_y = 68
        page.paste(annotated, (img_x, img_y))
        y = img_y + annotated.height + 16

        draw.text((24, y), 'Shown M2 text (relate.context):', fill='#000000', font=ImageFont.load_default(size=15))
        y += 22
        y = _draw_wrapped_lines(draw, 32, y, rec['m2_shown_text'].splitlines(), width_chars=96, font_size=14)
        y += 10

        draw.text((24, y), 'Raw top-10 M2 rows (score / threshold / pass):', fill='#000000',
                  font=ImageFont.load_default(size=15))
        y += 22
        if rec['m2_raw_top10']:
            raw_lines = [
                f"{rk:2d}. {r['subject']} [{r['subject_idx']}]  --({r['predicate']})-->  "
                f"{r['object']} [{r['object_idx']}]   score={r['score']:.4f}   thr={r['threshold']:.4f}   pass={r['pass']}"
                for rk, r in enumerate(rec['m2_raw_top10'], 1)
            ]
        else:
            raw_lines = ['(no candidate pairs)']
        _draw_wrapped_lines(draw, 32, y, raw_lines, width_chars=100, font_size=13)
        pages.append(page)

    summary_page = Image.new('RGB', (page_w, page_h), 'white')
    draw = ImageDraw.Draw(summary_page)
    draw.text(
        (24, 18),
        f"Summary: Photos with < 2 boxes ({len(too_few)})  --  Page {total_pages} / {total_pages}",
        fill='#000000',
        font=ImageFont.load_default(size=18),
    )
    y = 56
    if too_few:
        lines = [
            f"- {r['photo']}: {r['detection_count']} box(es)"
            + (f" ({', '.join(r['classes'])})" if r.get('classes') else '')
            for r in too_few
        ]
    else:
        lines = ['None (all processed photos had >= 2 boxes).']
    _draw_wrapped_lines(draw, 32, y, lines, width_chars=96, font_size=14)
    pages.append(summary_page)

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, format='PDF', resolution=100.0, save_all=True, append_images=pages[1:])
    if pdf_path.stat().st_size >= 20 * 1024 * 1024:
        raise RuntimeError(f'{pdf_path} exceeds 20 MB limit')


def build_sheet_ab_pdf(photo_records, frames_by_photo, pdf_path):
    """Build sheet_ab.pdf with PIL only: 1 page per >=2-box photo, numbered boxes + ONLY replies X and Y."""
    qualifying = [r for r in photo_records if r['m2_ran']]
    pages = []
    page_w, page_h = 900, 1200
    total_pages = max(1, len(qualifying))

    for idx, rec in enumerate(qualifying, 1):
        for forbidden in ('Y-ONLY', 'Y+M2'):
            if forbidden in rec['replies']['X'] or forbidden in rec['replies']['Y']:
                raise RuntimeError(f'variant label leaked into reply for {rec["photo"]}')
        page = Image.new('RGB', (page_w, page_h), 'white')
        draw = ImageDraw.Draw(page)
        header = f"{rec['photo']}  --  Page {idx} / {total_pages}"
        draw.text((24, 18), header, fill='#000000', font=ImageFont.load_default(size=18))

        used_dets = rec.get('m2_detections', rec['detections'][:rec['m2_boxes_used']])
        annotated = annotate_frame(frames_by_photo[rec['photo']], used_dets)
        img_x = (page_w - annotated.width) // 2
        img_y = 54
        page.paste(annotated, (img_x, img_y))
        y = img_y + annotated.height + 24

        draw.text((24, y), 'Reply X:', fill='#000000', font=ImageFont.load_default(size=16))
        y += 24
        y = _draw_wrapped_lines(draw, 32, y, rec['replies']['X'].splitlines() or ['(empty reply)'],
                                width_chars=90, font_size=15)
        y += 20

        draw.text((24, y), 'Reply Y:', fill='#000000', font=ImageFont.load_default(size=16))
        y += 24
        _draw_wrapped_lines(draw, 32, y, rec['replies']['Y'].splitlines() or ['(empty reply)'],
                            width_chars=90, font_size=15)
        pages.append(page)

    if not pages:
        page = Image.new('RGB', (page_w, page_h), 'white')
        draw = ImageDraw.Draw(page)
        draw.text((24, 18), 'No qualifying photos (>= 2 boxes)  --  Page 1 / 1', fill='#000000',
                  font=ImageFont.load_default(size=18))
        pages.append(page)

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, format='PDF', resolution=100.0, save_all=True, append_images=pages[1:])
    if pdf_path.stat().st_size >= 20 * 1024 * 1024:
        raise RuntimeError(f'{pdf_path} exceeds 20 MB limit')


def resolve_m2_dir():
    if os.path.isdir(relate.DIRECTORY):
        return relate.DIRECTORY
    if os.path.isdir(FALLBACK_M2_DIR):
        return FALLBACK_M2_DIR
    raise RuntimeError(f'M2 model directory not found at {relate.DIRECTORY} or {FALLBACK_M2_DIR}')


def git_base_commit(root=ROBOT):
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def run_harness(limit=None, out_dir=None):
    start_utc = utc_now()
    t0 = time.perf_counter()
    if out_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out_dir = HERE / 'runs' / stamp
    out_dir = Path(out_dir)
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'run.log'

    def log(msg):
        line = f'[{utc_now()}] {msg}'
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(line + '\n')

    log(f'START M2-Q1 harness ({TIMING_LABEL})')
    server_info = check_server()
    log(f"Server OK: {server_info['model_path']} (sha256={server_info['gguf_sha256']})")

    m2_dir = resolve_m2_dir()
    head = relate.Head(directory=m2_dir)
    detector = detect_person.get_detector(pinned=False)
    log(f'Loaded YOLO Detector and M2 Head from {m2_dir}')

    all_paths = discover_photos(ROBOT)
    all_photo_meta = [
        {
            'photo': p.relative_to(ROBOT).as_posix(),
            'sha256': relate.sha256(p),
            'bytes': p.stat().st_size,
        }
        for p in all_paths
    ]
    selected_paths = all_paths[:limit] if limit is not None else all_paths
    log(f'Discovered {len(all_paths)} total photos; processing {len(selected_paths)} (limit={limit})')

    staged = []
    frames_by_photo = {}
    failed_photos = []

    for p in selected_paths:
        rel = p.relative_to(ROBOT).as_posix()
        try:
            src_sha = relate.sha256(p)
            src_bytes = p.stat().st_size
            frame_filename = rel.replace('/', '__')
            frame_path = frames_dir / frame_filename
            decoded = prepare_frame(p, frame_path)
            frames_by_photo[rel] = decoded

            raw_dets = detector.detect(decoded, 640)
            detections = enrich_detections(raw_dets, decoded.size)
            staged.append({
                'photo': rel,
                'source_sha256': src_sha,
                'source_bytes': src_bytes,
                'frame_file': f'frames/{frame_filename}',
                'frame_sha256': relate.sha256(frame_path),
                'frame_bytes': frame_path.stat().st_size,
                'frame_size': list(decoded.size),
                'detections': detections,
            })
        except Exception as e:
            failed_photos.append({'photo': rel, 'stage': 'frame_or_yolo', 'error': f'{type(e).__name__}: {e}'})
            log(f'ERROR on {rel}: {type(e).__name__}: {e}')

    qualifying_names = [item['photo'] for item in staged if len(item['detections']) >= 2]
    blind_key = assign_blind_key(qualifying_names, seed=BLIND_SEED)

    photo_records = []
    for item in staged:
        rel = item['photo']
        detections = item['detections']
        decoded = frames_by_photo[rel]
        if len(detections) < 2:
            m2_res = run_m2_once(head, decoded, detections)
            photo_records.append({
                'photo': rel,
                'source_sha256': item['source_sha256'],
                'source_bytes': item['source_bytes'],
                'frame_file': item['frame_file'],
                'frame_sha256': item['frame_sha256'],
                'frame_bytes': item['frame_bytes'],
                'frame_size': item['frame_size'],
                'detection_count': len(detections),
                'classes': [d['class_name'] for d in detections],
                'm2_ran': False,
                'm2_shown_text': m2_res['shown_text'],
                'timing_label': TIMING_LABEL,
            })
            log(f'{rel}: <2 boxes ({len(detections)}), skipped M2 and Gemma')
            continue

        m2_1 = run_m2_once(head, decoded, detections)
        m2_2 = run_m2_once(head, decoded, detections)
        m2_same = rows_identical(m2_1['rows'], m2_2['rows'])
        used_dets = m2_1['used_detections']

        prompts = build_variant_prompts(used_dets, m2_1['shown_text'])
        reply_y_only_1, ms_yo_1 = call_gemma(prompts['Y-ONLY'])
        reply_y_only_2, ms_yo_2 = call_gemma(prompts['Y-ONLY'])
        reply_y_m2_1, ms_ym_1 = call_gemma(prompts['Y+M2'])
        reply_y_m2_2, ms_ym_2 = call_gemma(prompts['Y+M2'])

        by_variant_1 = {'Y-ONLY': reply_y_only_1, 'Y+M2': reply_y_m2_1}
        by_variant_2 = {'Y-ONLY': reply_y_only_2, 'Y+M2': reply_y_m2_2}
        same_by_variant = {
            'Y-ONLY': reply_y_only_1 == reply_y_only_2,
            'Y+M2': reply_y_m2_1 == reply_y_m2_2,
        }

        coin = blind_key[rel]
        replies_xy = {'X': by_variant_1[coin['X']], 'Y': by_variant_1[coin['Y']]}
        identical_xy = {'X': same_by_variant[coin['X']], 'Y': same_by_variant[coin['Y']]}

        rec = {
            'photo': rel,
            'source_sha256': item['source_sha256'],
            'source_bytes': item['source_bytes'],
            'frame_file': item['frame_file'],
            'frame_sha256': item['frame_sha256'],
            'frame_bytes': item['frame_bytes'],
            'frame_size': item['frame_size'],
            'detection_count': len(detections),
            'detections': detections,
            'm2_detections': [{**d, 'm2_index': i} for i, d in enumerate(used_dets)],
            'm2_ran': True,
            'm2_boxes_used': m2_1['boxes_used'],
            'm2_raw_top10': m2_1['rows'],
            'm2_shown_rows': m2_1['shown_rows'],
            'm2_shown_text': m2_1['shown_text'],
            'm2_identical_twice': m2_same,
            'gemma_prompts': prompts,
            'replies': replies_xy,
            'replies_identical_twice': identical_xy,
            'timings_not_a_measurement': {
                'timing_label': TIMING_LABEL,
                'm2_ms_run1': m2_1['infer_ms_not_a_measurement'],
                'm2_ms_run2': m2_2['infer_ms_not_a_measurement'],
                'gemma_total_ms_not_a_measurement': ms_yo_1 + ms_yo_2 + ms_ym_1 + ms_ym_2,
            },
            'timing_label': TIMING_LABEL,
        }
        if not m2_same:
            rec['m2_raw_top10_run2'] = m2_2['rows']
        if not all(identical_xy.values()):
            rec['replies_run2'] = {'X': by_variant_2[coin['X']], 'Y': by_variant_2[coin['Y']]}

        photo_records.append(rec)
        log(
            f"{rel}: boxes={len(detections)} (used={m2_1['boxes_used']}), "
            f"chosen={len(m2_1['shown_rows'])}, m2_2x={m2_same}, "
            f"gemma_2x={all(identical_xy.values())} ({TIMING_LABEL})"
        )

    sheet_m2_path = out_dir / 'sheet_m2.pdf'
    sheet_ab_path = out_dir / 'sheet_ab.pdf'
    build_sheet_m2_pdf(photo_records, frames_by_photo, sheet_m2_path)
    build_sheet_ab_pdf(photo_records, frames_by_photo, sheet_ab_path)
    log(f'Built PDFs: {sheet_m2_path.name} ({sheet_m2_path.stat().st_size} B), {sheet_ab_path.name} ({sheet_ab_path.stat().st_size} B)')

    end_utc = utc_now()
    elapsed_s = time.perf_counter() - t0

    qualifying_records = [r for r in photo_records if r['m2_ran']]
    counts = {
        'total_discovered': len(all_paths),
        'processed': len(photo_records),
        'ge_2_boxes': len(qualifying_records),
        'lt_2_boxes': sum(1 for r in photo_records if not r['m2_ran']),
        'gt_32_boxes': sum(1 for r in qualifying_records if r['detection_count'] > relate.MAX_BOXES),
        'm2_identical_twice': sum(1 for r in qualifying_records if r['m2_identical_twice']),
        'gemma_x_identical_twice': sum(1 for r in qualifying_records if r['replies_identical_twice']['X']),
        'gemma_y_identical_twice': sum(1 for r in qualifying_records if r['replies_identical_twice']['Y']),
        'gemma_both_identical_twice': sum(
            1 for r in qualifying_records if all(r['replies_identical_twice'].values())
        ),
        'failed': len(failed_photos),
    }

    results_payload = {
        'task': 'M2-Q1',
        'accuracy_label': ACCURACY_LABEL,
        'timing_label': TIMING_LABEL,
        'instruction': INSTRUCTION,
        'counts': counts,
        'failed_photos': failed_photos,
        'photos': photo_records,
    }
    write_json(out_dir / 'results.json', results_payload)
    write_json(out_dir / 'key.json', blind_key)

    code_hashes = verify_and_hash_imports(ROBOT)
    manifest = {
        'task': 'M2-Q1',
        'accuracy_label': ACCURACY_LABEL,
        'timing_label': TIMING_LABEL,
        'base_commit': git_base_commit(ROBOT),
        'start_utc': start_utc,
        'end_utc': end_utc,
        'elapsed_s_not_a_measurement': elapsed_s,
        'limit': limit,
        'code_sha256': code_hashes,
        'm2_files': {
            'directory': m2_dir,
            'sha256': {name: relate.sha256(os.path.join(m2_dir, name)) for name in sorted(relate.HASHES)},
        },
        'yolo_files': {
            str(size): {'path': path, 'sha256': relate.sha256(path)}
            for size, path in sorted(detect_person.MODELS.items())
        },
        'gguf': {
            'path': server_info['model_path'],
            'sha256': server_info['gguf_sha256'],
        },
        'server_props': server_info['props_summary'],
        'interpreter': {
            'executable': sys.executable,
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
            'onnxruntime_version': ort.__version__,
            'numpy_version': np.__version__,
            'pil_version': Image.__version__,
        },
        'counts': counts,
        'failed_photos': failed_photos,
        'processed_photos': [
            {
                'photo': r['photo'],
                'source_sha256': r['source_sha256'],
                'source_bytes': r['source_bytes'],
                'frame_file': r['frame_file'],
                'frame_sha256': r['frame_sha256'],
                'frame_bytes': r['frame_bytes'],
                'frame_size': r['frame_size'],
            }
            for r in photo_records
        ],
        'all_discovered_photos': all_photo_meta,
    }
    write_json(out_dir / 'manifest.json', manifest)
    log(f'DONE M2-Q1 harness in {elapsed_s:.2f}s ({TIMING_LABEL}) -> {out_dir}')
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N sorted photos (smoke run)')
    parser.add_argument('--out', type=Path, default=None, help='Output directory (default: runs/<UTC>)')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be >= 1')
    run_harness(limit=args.limit, out_dir=args.out)


if __name__ == '__main__':
    main()
