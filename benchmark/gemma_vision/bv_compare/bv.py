#!/data/data/com.termux/files/usr/bin/python
"""MIND-BV: on the robot's photos, how well does Gemma 4 E2B vision name the room, see a person and judge the person's
distance, compared with the robot's YOLO detections plus simple rules? Research only: no motors, camera, YOLO run,
mission or MTP. Accuracy is deterministic (temperature 0; an agent may be resident); timings NOT MEASUREMENTS.

Photos: the M2-Q1 frames (relate_anything/quality1/runs/20261017T213604Z/frames/), sorted by name and numbered 1-70;
ids 1-60 and 66 are used (61-65 and 67-70 are duplicates). Each frame's SHA-256 is checked against that run's
results.json before any call. Key: the owner's bv_key.json, outside the repo (refused inside it, checked before
reading), SHA-256 pinned in KEY_SHA256. Fields room, person (yes/no), distance (close/medium/far/none).

Methods (fixed before the first Gemma call):
  Y     YOLO + rules from results.json's detections (no new YOLO run). Person = any "person" detection. Distance =
        the robot's bucket of the person with the largest box (very close/close -> close, medium, far); none
        without a person. Photos with one detection keep only 'classes' in results.json (no box, no bucket): a
        person there gets distance UNAVAILABLE, scored wrong (owner decision 2026-11-03). Room, first match in
        ROOM_RULES, else unknown (no hallway cue).
  V70 / V140   Gemma vision with 70 / 140 image tokens, PROMPT.
  VY70  V70 plus a line listing Y's detections (detector_line) inserted before "Answer".
Server: BC's b11716 command (mem1.command(2048), setup_q4 flags, b11716 llama-server, BC's pins) plus --mmproj Q8_0
and --image-min-tokens = --image-max-tokens = 70 or 140; no --slot-save-path. Requests: /v1/chat/completions (GGUF
template), VIS-3D's image payload (frame bytes unchanged) with PROMPT, max_tokens 32, temperature 0, top_k 1, seed 0,
chat_template_kwargs enable_thinking false (as BC K4).
Launch A (70 tokens): V70 then VY70; launch B (140): V140. Blocks of at most 31 photos in id order, a 60 s pause
between blocks of one launch. The server's exit code after SIGTERM is reported; nonzero is a FINDING (#145).

Parsing (parse): per line, '*' removed, split at the first ':'; label and value lowercased and stripped, one
trailing '.' dropped. A field needs exactly one line with its label and a value in ALLOWED (distance also accepts
the option text as offered, e.g. "close (under 1 m)"); anything else is UNPARSED (counted wrong, reported).
Metrics per method: room and person accuracy (61), distance accuracy on the key's person photos (a "no" or a wrong
bucket is wrong), all three correct, confusion matrices (distance over all 61), UNPARSED counts, wrong ids.
PASS: a vision method with the highest room accuracy has room >= 0.80 and person accuracy >= Y's.

Usage: bv.py --key PATH   (Termux python; from proot Debian or native Termux)
Writes ~/robot_runs/mind_bv/<UTC>/: run.json, requests.jsonl, server_A.log, server_B.log, report.txt,
artifacts.sha256. Known limitation carried over from B0: an interrupt or a server-stop error can skip report.txt
and artifacts.sha256 (completed rows stay in requests.jsonl).
"""
import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'benchmark/mind_picker/bc_build'))
import bc  # noqa: E402  BC's command, pins, VIS-3D payload; imports b0, mem1, vis3d

b0, mem1, cr, vis3d, sha = bc.b0, bc.mem1, bc.cr, bc.vis3d, bc.sha
detect_person = sys.modules['detect_person']  # imported by coresidency (cr); read for CONF/IOU/MODELS only
inside_repo = bc.b2.b1.inside_repo
Q1 = REPO / 'benchmark/relate_anything/quality1/runs/20261017T213604Z'
FRAMES_DIR, RESULTS, MANIFEST = Q1 / 'frames', Q1 / 'results.json', Q1 / 'manifest.json'
IDS = list(range(1, 61)) + [66]
KEY_SHA256 = '456f633bd9145edc9a4259171ad23f5957cc66ce11e3bd30983fcd92ad470edb'  # owner key, 2026-11-03
OUT_ROOT = Path(cr.HOME) / 'robot_runs' / 'mind_bv'
CTX = 2048
LAUNCHES = [('A', 70, ['V70', 'VY70']), ('B', 140, ['V140'])]
BLOCK, PAUSE_S = 31, 60
FIELDS = ('room', 'person', 'distance')
ALLOWED = {'room': ('living room', 'kitchen', 'bedroom', 'bathroom', 'hallway', 'unknown'),
           'person': ('yes', 'no'), 'distance': ('close', 'medium', 'far', 'none')}
OFFERED = {'close (under 1 m)': 'close', 'medium (1 to 2 m)': 'medium', 'far (over 2 m)': 'far'}
ROOM_RULES = [({'toilet'}, 'bathroom'), ({'bed'}, 'bedroom'), ({'refrigerator', 'oven', 'microwave'}, 'kitchen'),
              ({'couch', 'tv', 'dining table'}, 'living room')]
BUCKET = {'very close': 'close', 'close': 'close', 'medium': 'medium', 'far': 'far'}
UNPARSED, UNAVAILABLE = 'UNPARSED', 'UNAVAILABLE'
PROMPT_HEAD = "Look at this photo from a small home robot's camera."
PROMPT_TAIL = ('Answer in exactly three lines:\n'
               'Room: living room, kitchen, bedroom, bathroom, hallway or unknown\n'
               'Person: yes or no\n'
               'Distance: close (under 1 m), medium (1 to 2 m), far (over 2 m) or none\n'
               'Give only these three lines.')
VISION_METHODS = ('V70', 'V140', 'VY70')
METHODS = ('Y',) + VISION_METHODS


# ---------------------------------------------------------------- inputs

def load_key(path, want=KEY_SHA256):
    """{id: {frame, room, person, distance}}; refused inside the repo (checked before reading) or on a hash change."""
    if inside_repo(path):
        raise SystemExit(f'refused: {path} is inside the repository (key custody)')
    data = Path(path).read_bytes()
    if sha(data) != want:
        raise SystemExit(f'refused: {path} SHA-256 {sha(data)[:12]} != pinned {want[:12]}')
    key = {int(k): v for k, v in json.loads(data).items()}
    if sorted(key) != IDS:
        raise SystemExit(f'refused: key ids are not {IDS}')
    return key


def photos(results, frames):
    """{id: results.json record} for IDS; frames = sorted frame names (1-based numbering)."""
    by_file = {Path(p['frame_file']).name: p for p in results['photos']}
    return {i: by_file[frames[i - 1]] for i in IDS}


def frame_problems(ph, key, frames_dir=FRAMES_DIR):
    bad = []
    for i, p in ph.items():
        f = frames_dir / Path(p['frame_file']).name
        if key[i]['frame'] != f.name:
            bad.append(f'id {i}: key frame {key[i]["frame"]} != {f.name}')
        elif not f.is_file() or sha(f.read_bytes()) != p['frame_sha256']:
            bad.append(f'id {i}: {f.name} SHA-256 != results.json {p["frame_sha256"][:12]}')
    return bad


# ---------------------------------------------------------------- method Y

def classes(p):
    return [d['class_name'] for d in p['detections']] if p.get('detections') else list(p.get('classes') or [])


def area(d):
    x1, y1, x2, y2 = d['box_xyxy']
    return (x2 - x1) * (y2 - y1)


def y_pred(p):
    cls = classes(p)
    room = next((r for names, r in ROOM_RULES if names & set(cls)), 'unknown')
    if 'person' not in cls:
        return {'room': room, 'person': 'no', 'distance': 'none'}
    boxes = [d for d in p.get('detections') or [] if d['class_name'] == 'person']
    dist = BUCKET[max(boxes, key=area)['distance']] if boxes else UNAVAILABLE
    return {'room': room, 'person': 'yes', 'distance': dist}


def detector_line(p):
    if p.get('detections'):
        items = [f'{d["class_name"]} ({d["position"]}, {d["distance"]})' for d in p['detections']]
    else:  # one-box photos: class only (owner decision 2026-11-03)
        items = [f'{c} (position unknown, distance unknown)' for c in classes(p)]
    return "Objects the robot's detector found: " + (', '.join(items) or 'none')


# ---------------------------------------------------------------- vision requests

def prompt(method, p):
    return '\n'.join([PROMPT_HEAD] + ([detector_line(p)] if method == 'VY70' else []) + [PROMPT_TAIL])


def body(method, p, jpeg_b64):
    b = vis3d.image_payload(None, jpeg_b64)  # VIS-3D: image then text, greedy, nothing cached
    b['messages'][0]['content'][1]['text'] = prompt(method, p)
    return {**b, 'max_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False}}


def command(tokens):
    cmd = mem1.command(CTX)
    cmd[cmd.index(mem1.SERVER)] = bc.SERVER
    return cmd + ['--mmproj', vis3d.MMPROJ, '--image-min-tokens', str(tokens), '--image-max-tokens', str(tokens)]


def run_method(post, launch, method, ph, pid, emit, frames_dir=FRAMES_DIR, first_block=True):
    """Blocks of BLOCK photos in id order, PAUSE_S between blocks (not before the launch's first). Returns rows."""
    rows = []
    for n, k in enumerate(range(0, len(IDS), BLOCK)):
        if n or not first_block:
            time.sleep(PAUSE_S)
        for i in IDS[k:k + BLOCK]:
            p = ph[i]
            b = body(method, p, base64.b64encode((frames_dir / Path(p['frame_file']).name).read_bytes()).decode())
            row = {'launch': launch, 'method': method, 'block': n, 'id': i, 'frame': Path(p['frame_file']).name,
                   'pid': pid, 'pid_start': b0.pid_start(pid), 'prompt': b['messages'][0]['content'][1]['text']}
            t0 = time.perf_counter()
            try:
                res = post('/v1/chat/completions', b)
                msg = res['choices'][0]['message']
                row.update(content=msg.get('content'), reasoning_content=msg.get('reasoning_content'),
                           finish_reason=res['choices'][0].get('finish_reason'), usage=res.get('usage'))
            except Exception as e:  # recorded; the run is INVALID
                row['error'] = f'{type(e).__name__}: {e}'
            row['wall_s'] = round(time.perf_counter() - t0, 3)
            emit(row)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- scoring

def parse(reply):
    found = {f: [] for f in FIELDS}
    for line in (reply or '').splitlines():
        label, sep, value = line.replace('*', '').partition(':')
        label, value = label.strip().lower(), value.strip().lower()
        if sep and label in found:
            found[label].append(value[:-1].strip() if value.endswith('.') else value)
    out = {}
    for f, vals in found.items():
        v = OFFERED.get(vals[0], vals[0]) if len(vals) == 1 else None
        out[f] = v if v in ALLOWED[f] else UNPARSED
    return out


def metrics(pred, key):
    pids = [i for i in IDS if key[i]['person'] == 'yes']
    ok = {f: [i for i in IDS if pred[i][f] == key[i][f]] for f in ('room', 'person')}
    ok['distance'] = [i for i in pids if pred[i]['person'] == 'yes' and pred[i]['distance'] == key[i]['distance']]
    n = {'room': len(IDS), 'person': len(IDS), 'distance': len(pids)}
    return {'acc': {f: len(ok[f]) / n[f] for f in FIELDS}, 'correct': {f: len(ok[f]) for f in FIELDS}, 'n': n,
            'all3': sum(all(pred[i][f] == key[i][f] for f in FIELDS) for i in IDS),
            'wrong': {f: [i for i in (pids if f == 'distance' else IDS) if i not in ok[f]] for f in FIELDS},
            'unparsed': {f: sum(pred[i][f] == UNPARSED for i in IDS) for f in FIELDS},
            'unavailable': sum(pred[i]['distance'] == UNAVAILABLE for i in IDS),
            'confusion': {f: Counter((key[i][f], pred[i][f]) for i in IDS) for f in FIELDS}}


def preds(rows, ph):
    """{method: {id: prediction}}; a vision method needs one row per id, else it is missing."""
    out = {'Y': {i: y_pred(p) for i, p in ph.items()}}
    for m in VISION_METHODS:
        got = {x['id']: x for x in rows if x['method'] == m}
        if sorted(got) == IDS:
            out[m] = {i: parse(got[i].get('content')) for i in IDS}
    return out


def verdict(mets):
    vis = [m for m in VISION_METHODS if m in mets]
    if not vis or 'Y' not in mets:
        return 'FAIL', 'a method is missing'
    best = max(mets[m]['acc']['room'] for m in vis)
    top = [m for m in vis if mets[m]['acc']['room'] == best]
    win = [m for m in top if best >= 0.80 and mets[m]['acc']['person'] >= mets['Y']['acc']['person']]
    return ('PASS' if win else 'FAIL'), (f'best room {best:.3f} by {top}; person {[round(mets[m]["acc"]["person"], 3) for m in top]}'
                                         f' vs Y {mets["Y"]["acc"]["person"]:.3f}; meets both: {win or "none"}')


def validity(run, rows):
    probs = []
    for launch, _, methods in LAUNCHES:
        lr = [x for x in rows if x['launch'] == launch]
        if len(lr) != len(methods) * len(IDS):
            probs.append(f'launch {launch}: {len(lr)} of {len(methods) * len(IDS)} rows')
        if len({(x['pid'], x['pid_start']) for x in lr}) > 1:
            probs.append(f'launch {launch}: more than one server PID')
    errors = sum('error' in x for x in rows)
    probs += [f'{errors} request errors'] if errors else []
    probs += [f'exception: {run["exception"]}'] if run.get('exception') else []
    probs += [f'end check: {p}' for p in run.get('end_problems', [])]
    return ('INVALID' if probs else 'VALID'), probs


# ---------------------------------------------------------------- report

def confusion_text(c, rows_lab, cols_lab):
    w = max(12, *(len(s) + 1 for s in cols_lab))
    L = ['    key \\ pred'.ljust(16) + ''.join(s.rjust(w) for s in cols_lab)]
    return L + [f'    {r:<12}' + ''.join(str(c.get((r, s), 0)).rjust(w) for s in cols_lab) for r in rows_lab]


def report(run, rows, ph, key, mets):
    L = [f'MIND-BV {run["utc"]}  code {run["code_sha256"][:12]}  base {run["git_head"]}',
         f'results: accuracy {run["validity"]} (temperature 0); timings NOT MEASUREMENTS (agent resident)',
         f'key {run["key"]["path"]} sha256 {run["key"]["sha256"]}', f'pins: {run["pins"]}', f'detector behind Y: '
         f'{run["detector"]}'] + [f'launch {l}: {" ".join(c)}' for l, c in run['commands'].items()] + \
        [f'launch {l}: exit code {e["exit_code"]} after SIGTERM, SIGKILL {e["killed"]}' +
         ('' if e['exit_code'] == 0 and not e['killed'] else '  FINDING (#145)') for l, e in run['exits'].items()]
    if run['problems']:
        L.append('validity problems: ' + '; '.join(run['problems']))
    L += ['', f'{"method":<6} {"room":>11} {"person":>11} {"distance":>11} {"all3":>6}  UNPARSED (room/person/dist)']
    for m, x in mets.items():
        c, n = x['correct'], x['n']
        L.append(f'{m:<6} ' + ' '.join(f'{c[f]:>2}/{n[f]} {x["acc"][f]:.3f}'.rjust(11) for f in FIELDS) +
                 f' {x["all3"]:>3}/61  {x["unparsed"]["room"]}/{x["unparsed"]["person"]}/{x["unparsed"]["distance"]}' +
                 (f'  (distance UNAVAILABLE {x["unavailable"]})' if x['unavailable'] else ''))
    L += ['', f'VERDICT: {run["verdict"][0]}  ({run["verdict"][1]})']
    for m, x in mets.items():
        L += ['', f'--- {m}'] + [f'  wrong {f}: {x["wrong"][f]}' for f in FIELDS]
        for f in FIELDS:
            cols = list(ALLOWED[f]) + [s for s in (UNPARSED, UNAVAILABLE) if any(k[1] == s for k in x['confusion'][f])]
            rws = [r for r in ALLOWED[f] if any(k[0] == r for k in x['confusion'][f])]
            L += [f'  confusion {f}' + (' (all 61)' if f == 'distance' else '')] + confusion_text(x['confusion'][f], rws, cols)
    L += ['', '--- prompt tokens per method (usage; min-max)']
    for m in VISION_METHODS:
        t = [x['usage']['prompt_tokens'] for x in rows if x['method'] == m and (x.get('usage') or {}).get('prompt_tokens')]
        L.append(f'  {m}: {min(t)}-{max(t)}' if t else f'  {m}: none')
    L += ['', '--- every reply (id, method, key, parsed, verbatim; * = field wrong)']
    for x in rows:
        p = parse(x.get('content'))
        k = key[x['id']]
        marks = ' '.join(f'{p[f]}{"*" if p[f] != k[f] else ""}' for f in FIELDS)
        L.append(f'{x["id"]:>2} {x["method"]:<5} key {k["room"]}|{k["person"]}|{k["distance"]}  got {marks}  '
                 f'{x.get("content")!r}' + (f'  ERROR {x["error"]}' if 'error' in x else ''))
    L += ['', '--- Y predictions (id: room|person|distance; classes)']
    L += [f'{i:>2} {"|".join(y_pred(ph[i])[f] for f in FIELDS)}  {classes(ph[i])}' for i in IDS]
    return '\n'.join(L) + '\n'


def detector_info():
    m = json.loads(MANIFEST.read_text())
    dp = Path(detect_person.__file__)
    return {'model': m['yolo_files']['640'], 'input': '640x640 (quality1.py: detector.detect(frame, 640); plain resize)',
            'conf': f'> {detect_person.CONF}', 'iou_nms': detect_person.IOU,
            'detect_person_sha256': sha(dp.read_bytes()), 'detect_person_sha256_in_run': m['code_sha256']['detect_person.py']}


def finish(out, run, rows, ph, key):
    run['end_problems'] = frame_problems(ph, key)
    run['validity'], run['problems'] = validity(run, rows)
    mets = {m: metrics(p, key) for m, p in preds(rows, ph).items()}
    run['verdict'] = verdict(mets)
    run['metrics'] = {m: {**x, 'confusion': {f: [[*k, v] for k, v in sorted(c.items())] for f, c in x['confusion'].items()}}
                      for m, x in mets.items()}
    (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
    text = report(run, rows, ph, key, mets)
    (out / 'report.txt').write_text(text)
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'artifacts.sha256')
    (out / 'artifacts.sha256').write_text(''.join(f'{sha(p.read_bytes())}  {p.relative_to(out)}\n' for p in files))
    return text


def stop(proc):
    killed = False
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=b0.TERM_WAIT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            killed = True
    return {'exit_code': proc.returncode, 'killed': killed}


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('--key', required=True)
    args = a.parse_args(argv)
    key = load_key(args.key)
    frames = sorted(p.name for p in FRAMES_DIR.glob('*.jpg'))
    results_bytes = RESULTS.read_bytes()
    ph = photos(json.loads(results_bytes), frames)
    bad = frame_problems(ph, key)
    bad += [f'{p}: SHA-256 != pinned {h[:12]}' for p, h in bc.PINS.items() if not Path(p).is_file() or cr.sha256(p) != h]
    bad += vis3d.env_problems()
    if bad:
        raise SystemExit('refused: ' + '; '.join(bad))
    if not b0.repro.port_free():
        raise SystemExit('refused: port 8080 in use (never stops a server it did not start)')
    utc = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = OUT_ROOT / utc
    out.mkdir(parents=True)  # a new folder; an existing one raises
    run = {'utc': utc, 'code_sha256': sha(Path(__file__).read_bytes()),
           'imports_sha256': {m.__name__: sha(Path(m.__file__).read_bytes()) for m in (bc, b0, mem1, vis3d)},
           'git_head': bc.run_out('git', '-C', str(HERE), 'rev-parse', 'HEAD').strip(), 'pins': bc.PINS,
           'key': {'path': str(Path(args.key).resolve()), 'sha256': KEY_SHA256},
           'results_sha256': sha(results_bytes), 'detector': detector_info(),
           'commands': {l: command(t) for l, t, _ in LAUNCHES}, 'exits': {}, 'timings': 'NOT MEASUREMENTS'}
    (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
    rows, exc = [], None
    with open(out / 'requests.jsonl', 'x') as rq:
        def emit(row):
            rq.write(json.dumps(row) + '\n')
            rq.flush()
        for launch, tokens, methods in LAUNCHES:
            proc = None
            with open(out / f'server_{launch}.log', 'x') as log:
                try:
                    proc = subprocess.Popen(run['commands'][launch], stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    t0 = time.monotonic()
                    while not cr.healthy():
                        if proc.poll() is not None or time.monotonic() - t0 > b0.LOAD_TIMEOUT_S:
                            raise RuntimeError(f'llama-server not healthy (exit {proc.poll()})')
                        time.sleep(0.1)
                    post = lambda path, b: cr.post(path, b, timeout=b0.REQUEST_TIMEOUT_S)  # noqa: E731
                    for j, m in enumerate(methods):
                        rows += run_method(post, launch, m, ph, proc.pid, emit, first_block=j == 0)
                except BaseException as e:  # recorded; the report is still written, then a stop re-raises
                    exc = e
                    run['exception'] = f'launch {launch}: {type(e).__name__}: {e}'
                finally:
                    if proc is not None:
                        run['exits'][launch] = stop(proc)
            if exc is not None:
                break
    print(finish(out, run, rows, ph, key) + f'\n{out}')
    if exc is not None and not isinstance(exc, Exception):
        raise exc


if __name__ == '__main__':
    main()
