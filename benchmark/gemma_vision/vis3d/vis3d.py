#!/data/data/com.termux/files/usr/bin/python
"""VIS-3D: Gemma 4 E2B Q4_0 on the b11716 build (~/llama.cpp-vis), one test per launch. Research only: no motors,
no camera, no main.py. Native Termux only; started by oneshot.sh in this folder.

  A  b11716, text only, no projector, MTP off   } the frozen Oct 14 workload of VIS-3C (repro.py): same 13 prompts,
  B  A + Q8_0 projector resident, MTP off       } payload (legacy <start_of_turn> /completion), ctx 4096, temperature
  C  B + Q8_0 MTP drafter                       } 0.7, n_predict 512, cores 4-7; no image is sent
  D  B + --reasoning off -lv 4: 8 fixed frames through /v1/chat/completions (GGUF Gemma 4 template), 70-token images

The block is chat_mtp.run_block unchanged (5 s coresidency sampler, 175 s watchdog, TERM then KILL, cores 4-7 checks),
as in VIS-3C; only the command, payload and (D) the endpoint are replaced. After every reply the runner reads
/metrics (exact speculative verification steps); after the first it also reads /props, /slots, the mapped libraries
and (D) the applied template. Usage:
  vis3d.py --test A|B|C|D [--thermal-log PATH]
  vis3d.py --report DIR [DIR ...]     (report only; reads folders, starts nothing)
Writes ~/robot_runs/vis3d/<UTC>_<test>/: run.json, prompts.jsonl, server_block00_<test>.log, result.json, report.txt.
"""
import argparse
import base64
import json
import os
import re
import signal
import statistics
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPRO_DIR = REPO / 'benchmark/llm_objective_setting/conversation/chat_mtp_repro'
sys.path.insert(0, str(REPRO_DIR))
import repro  # noqa: E402  VIS-3C: frozen Oct 14 workload, refusals, conditions, summary; block machinery via cm

cm, cr, pm = repro.cm, repro.cr, repro.pm
HOME = cr.HOME
BIN = f'{HOME}/llama.cpp-vis/build/bin'
SERVER = f'{BIN}/llama-server'
MODEL, DRAFT = repro.MODEL, repro.DRAFT
MMPROJ = f'{HOME}/models/vis0/mmproj-gemma-4-E2B-it-Q8_0.gguf'  # vis1/mmproj.json: ggml-org rev b4243c15
SHA256 = {MODEL: repro.SHA256[MODEL], DRAFT: repro.SHA256[DRAFT],
          MMPROJ: '9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292'}  # hashed on the phone 2026-10-23
SOURCE_COMMIT = '797de63b2fa7396dcc8caca76834c699ac84e1c3'  # tag b11716 (llama.cpp-vis/.git/HEAD)
BUILD_SHA256 = {  # = the VIS-2 manifest; rehashed on the phone 2026-10-23
    'llama-server': '0674f5c28aaf23068220fa4290c7a4199aa590127724d2b766bdbb961b768fdf',
    'libllama-server-impl.so': 'a9b6e74c8fc906325f91470e335d77a9b32257dd056102913ee8fcd6aa9fa7aa',
    'libllama-common.so.0.7.0': '805cfe637a6f8eba771d039e06480a1e21789c176cb1fbbe8c88b6b4fd2d53ef',
    'libllama.so.0.7.0': '18041e7c9182ce1534e0f9ab33abb3e12f9fb4d84a773dbc86297ebaf7a84578',
    'libmtmd.so.0.7.0': 'f2813411696515a08de553d17ba13b0e5a758329a4fabd82a14e1a726eea0fc8',
    'libggml.so.0.27.0': '50477f04f6c57176aaea22d6ef3fcabcb21652d26cc58288ce82d174c6d7d79d',
    'libggml-base.so.0.27.0': '8eadd421b15ac463c6be41d97cba5ff23fb02ea6d2217ca81020125bc352477e',
    'libggml-cpu.so.0.27.0': '9e4e11cf979ef78e1aa201ecaed67308520ffa9a6e3f19d028cfb1586b8ab333'}
CMAKE_KEYS = ('CMAKE_BUILD_TYPE', 'GGML_CPU_ARM_ARCH', 'GGML_NATIVE', 'GGML_OPENMP', 'GGML_CPU_REPACK',
              'GGML_LLAMAFILE', 'GGML_CPU_KLEIDIAI', 'BUILD_SHARED_LIBS')  # recorded, not judged: the hashes pin the build
CODE_SHA256 = {REPRO_DIR / 'repro.py': '29276a35618e66833a214fba77c8ce0770bbe37d4fa9e2b4b9de105c13a00a46'}  # = Oct 23 runs

# repro.OFF is the frozen b1609 OFF command; only the binary changes, and --metrics exposes the step counter
BASE = ['taskset', '-c', '4-7', SERVER, *repro.OFF[4:], '--metrics']
IMAGE_TOKENS = 70  # min = max; VIS-0/VIS-1 logged 63 tokens per 482x640 frame (Q8 encode about 2.5 s, VIS-1 V2)
VISION = ['--mmproj', MMPROJ, '--image-min-tokens', str(IMAGE_TOKENS), '--image-max-tokens', str(IMAGE_TOKENS)]
MTP = repro.COMMANDS['ON'][len(repro.OFF):]  # the VIS-3C ON drafter flags
COMMANDS = {'A': BASE, 'B': BASE + VISION, 'C': BASE + VISION + MTP, 'D': BASE + VISION + ['--reasoning', 'off', '-lv', '4']}
INTENDED = {'A': (False, False), 'B': (True, False), 'C': (True, True), 'D': (True, False)}  # (vision, speculative)
EXPECTED_DIFF = {('A', 'B'): VISION, ('B', 'C'): MTP}  # report: the only command difference allowed per pair

REFS = json.loads((HERE / 'references.json').read_text())
FRAMES_DIR = REPO / REFS['frames_dir']
FRAMES = list(REFS['frames'])
PROMPT = ('Describe in one or two short sentences what is in front of the robot. '
          'Mention only what you can see.')  # VIS-0 photo-only prompt (no YOLO list: VIS-0 saw it copy YOLO errors)
ENV_REFUSED = ('LLAMA_', 'GGML_', 'OMP_', 'KMP_', 'LD_LIBRARY_PATH')  # would change the server (LLAMA_ARG_* = flags)
OUT_ROOT = HOME / 'robot_runs' / 'vis3d'
THERMAL_LOG = OUT_ROOT / 'thermal.log'  # oneshot.sh's logger
BASELINE = {'OFF': HOME / 'robot_runs/chat_mtp_repro/20261023T124846Z_OFF',  # VIS-3C b1609-dotprod, both VALID,
            'ON': HOME / 'robot_runs/chat_mtp_repro/20261023T132032Z_ON'}   # SAME, MATCHED: 13.35 / 18.55 tok/s


def image_payload(frame, jpeg=None):
    """VIS-0 vision request: image then text in one user message; greedy, at most 96 tokens, nothing cached."""
    url = 'data:image/jpeg;base64,' + (jpeg or base64.b64encode((FRAMES_DIR / f'{frame}.jpg').read_bytes()).decode())
    return {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}},
                                                      {'type': 'text', 'text': PROMPT}]}],
            'temperature': 0, 'top_k': 1, 'seed': 0, 'max_tokens': 96, 'cache_prompt': False, 'stream': False}


_post = cr.post


def chat_post(path, body, timeout):
    """D: run_block posts to /completion; send the body to the chat endpoint instead (the server applies the GGUF
    Gemma 4 template) and return the reply and finish reason where run_block reads them."""
    assert path == '/completion', path
    data = _post('/v1/chat/completions', body, timeout)
    choice = data['choices'][0]
    return {**data, 'content': choice['message'].get('content') or '', 'stop_type': choice.get('finish_reason')}


# ---------------------------------------------------------------- preflight

def code_problems():
    return [f'{p} SHA-256 is not the pinned {h[:12]}' for p, h in CODE_SHA256.items() if cr.sha256(p) != h]


def env_problems():
    bad = sorted(k for k in os.environ if k.startswith(ENV_REFUSED))
    return [f'environment sets {", ".join(bad)}: unset before the run'] if bad else []


def frame_problems():
    return [f'{f}.jpg is not the M2-Q1 frame {r["frame_sha256"][:12]}' for f, r in REFS['frames'].items()
            if not (FRAMES_DIR / f'{f}.jpg').is_file() or cr.sha256(FRAMES_DIR / f'{f}.jpg') != r['frame_sha256']]


def check_hashes():
    got = {p: cr.sha256(p) if Path(p).is_file() else 'missing' for p in SHA256}
    build = {n: cr.sha256(Path(BIN) / n) if (Path(BIN) / n).is_file() else 'missing' for n in BUILD_SHA256}
    bad = [f'{p}: {got[p]} != {w}' for p, w in SHA256.items() if got[p] != w]
    bad += [f'{n}: {build[n]} != {w}' for n, w in BUILD_SHA256.items() if build[n] != w]
    if bad:
        raise SystemExit('SHA-256 mismatch: ' + '; '.join(bad))
    return got, build


def build_info():
    """Source commit and CMake flags of the build, read from disk (recorded only)."""
    root = Path(BIN).parents[1]
    try:
        head = (root / '.git/HEAD').read_text().strip()
    except OSError as e:
        head = f'error: {e}'
    try:
        lines = (root / 'build/CMakeCache.txt').read_text().splitlines()
        flags = {k: v for k, _, v in (l.partition('=') for l in lines) if k.split(':')[0] in CMAKE_KEYS}
    except OSError as e:
        flags = {'error': str(e)}
    return {'git_head': head, 'source_commit_expected': SOURCE_COMMIT, 'cmake': flags}


# ---------------------------------------------------------------- probes between replies (outside every request)

def get(path, timeout=10):
    with urllib.request.urlopen(f'http://127.0.0.1:{cr.PORT}{path}', timeout=timeout) as r:
        return r.read().decode()


def parse_metrics(text):
    return {k: float(v) for k, v in re.findall(r'^(llamacpp:\w+) (\S+)$', text, re.M)}


def mapped_libs(pid):
    return sorted({l.split()[-1] for l in open(f'/proc/{pid}/maps') if '.so' in l.split()[-1]})


def probe(test, first):
    """/metrics after every reply; on the first also /props, /slots, the mapped libraries and (D) the template.
    A failed read is recorded as an error string (it makes the run NOT VALID, it does not stop the block)."""
    out = {}

    def take(key, fn):
        try:
            out[key] = fn()
        except Exception as e:  # noqa: BLE001  recorded, judged by the report
            out[key] = {'error': f'{type(e).__name__}: {e}'}
    take('metrics', lambda: parse_metrics(get('/metrics')))
    if first:
        take('props', lambda: json.loads(get('/props')))
        take('slots', lambda: json.loads(get('/slots')))
        take('libs', lambda: mapped_libs(cm.LIVE[-1].pid))
        if test == 'D':
            take('template', lambda: _post('/apply-template', {'messages': [{'role': 'user', 'content': PROMPT}]}, 10))
    return out


# ---------------------------------------------------------------- server-log image timings (D, -lv 4)

TS = r'(\d+\.\d{2}\.\d{3}\.\d{3})'


def log_ms(ts):
    m, s, ms, us = map(int, ts.split('.'))
    return (m * 60 + s) * 1000 + ms + us / 1000


def image_log(text):
    """Per server task in order: image encode ms (log-timestamp interval 'encoding mtmd batch' -> first 'decoding
    image batch'), image tokens and image-token decode ms (sum of 'image decoded ... in N ms'). None when absent."""
    out = []
    for sec in re.split(r'^.*\| processing task.*$', text, flags=re.M)[1:]:
        enc, dec = re.search(TS + r' .*encoding mtmd batch', sec), re.search(TS + r' .*decoding image batch', sec)
        toks = [int(n) for n in re.findall(r'decoding image batch \d+/\d+, n_tokens_batch = (\d+)', sec)]
        done = [int(n) for n in re.findall(r'image decoded \(batch \d+/\d+\) in (\d+) ms', sec)]
        out.append({'encode_ms': round(log_ms(dec[1]) - log_ms(enc[1]), 1) if enc and dec else None,
                    'image_tokens': sum(toks) if toks else None, 'image_decode_ms': sum(done) if done else None})
    return out


# ---------------------------------------------------------------- report

def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def check_folder(folder):
    """(run, rows, result, problems): problems make the run NOT VALID (repro.load's checks plus this test's)."""
    folder = Path(folder)
    run = json.loads((folder / 'run.json').read_text())
    rows = cm.read_jsonl(folder / 'prompts.jsonl')
    if not (folder / 'result.json').exists():
        return run, rows, None, ['INCOMPLETE: no result.json (interrupted or failed)']
    res = json.loads((folder / 'result.json').read_text())
    b, why, test = res['block'], [], run['test']
    if [r['prompt_id'] for r in rows] != run['ids']:
        why.append(f'{len(rows)}/{len(run["ids"])} requests')
    if b['end_reason'] != 'all prompts done':
        why.append(b['end_reason'])
    if b['gate']['status'] != 0 or res['start']['status'] != 0:
        why.append('Android thermal status not 0 at start')
    if b['server_start'].get('Cpus_allowed_list') != '4-7' or b['server_end'].get('Cpus_allowed_list') != '4-7':
        why.append('server not on cores 4-7 at start and end')
    if res['agents_end']:
        why.append(f'agent process at end: {res["agents_end"][:100]}')
    if not res['port_free_end']:
        why.append('port 8080 still in use after the stop')
    if any(s.get('battery_status') not in (None, 'Discharging') for s in b['samples']):
        why.append('battery not Discharging in a sample')
    if run['command'] != COMMANDS[test]:
        why.append('command is not this test\'s')
    if not rows:
        return run, rows, res, why
    p = rows[0].get('probe', {})
    vision, spec = INTENDED[test]
    props, slots, libs = p.get('props', {}), p.get('slots', []), p.get('libs', [])
    if not isinstance(props, dict) or props.get('modalities', {}).get('vision') is not vision:
        why.append(f'/props vision is not {vision}')
    if not isinstance(slots, list) or [s.get('speculative') for s in slots] != [spec]:
        why.append(f'/slots speculative is not [{spec}]')
    ours = [l for l in libs if re.search(r'/lib(ggml|llama|mtmd)', l)] if isinstance(libs, list) else []
    if not ours or any(not l.startswith(BIN + '/') for l in ours):
        why.append(f'mapped llama/ggml libraries not all from {BIN}: {ours or libs}')
    if any('error' in r.get('probe', {}).get('metrics', {}) for r in rows):
        why.append('/metrics not read after every reply')
    drafted = sum(r['draft_n'] or 0 for r in rows)
    if spec and not (drafted and steps(rows)):
        why.append('MTP requested but nothing drafted or verified')
    if not spec and drafted:
        why.append('drafting without the drafter')
    log = server_log(folder, test)
    if not log:
        why.append('no server log')
    if (f"loaded multimodal model, '{MMPROJ}'" in log) != vision:
        why.append(f'server log: projector {"not " if vision else ""}loaded')
    if (f"loading draft model '{DRAFT}'" in log) != spec:
        why.append(f'server log: drafter {"not " if spec else ""}loaded')
    if test == 'D':
        tpl = p.get('template', {}).get('prompt', '')
        if not tpl.startswith('<|turn>user\n') or '<turn|>' not in tpl or '<start_of_turn>' in tpl:
            why.append(f'applied template is not Gemma 4: {tpl[:60]!r}')
        if len(image_log(log)) != len(rows):
            why.append('server log image sections do not match the requests')
    return run, rows, res, why


def server_log(folder, test):
    p = Path(folder) / f'server_block00_{test}.log'
    return p.read_text(errors='replace') if p.exists() else ''


def steps(rows):
    """Exact speculative verification steps of the run: the server's cumulative counter after the last reply."""
    return rows[-1].get('probe', {}).get('metrics', {}).get('llamacpp:spec_decode_num_drafts_total') if rows else None


def resources(b):
    return {'hwm': b.get('vm_hwm_kb'), 'pss': b.get('peak_pss_kb'), 'avail': b.get('min_mem_available_mib'),
            'mean_w': b.get('mean_w'), 'end_cond': b.get('end', {}), 'samples': len(b['samples']), 'load_s': b['load_s']}


def text_entry(name, test, rows, res, run, label):
    s = repro.summary(rows, res['block'])
    n = steps(rows) if run.get('test') else None  # the b1609 baseline has no step counter
    s.update(steps=n, ms_per_step=s['decode_ms'] / n if n else None, **resources(res['block']))
    return (name, test, s, res['block']['gate'], res.get('start'), res.get('end'), [], run, label)


def config_diffs(a, b, extra):
    """Two b11716 text runs: commands differ by exactly `extra` appended; workload, weights, build, code equal."""
    why = [] if b['command'] == a['command'] + extra else [f'commands differ beyond {" ".join(extra)}']
    return why + [f'{k} differs' for k in ('payload_example', 'prompt_texts', 'ids', 'model_sha256', 'build_sha256',
                                           'code_sha256') if a[k] != b[k]]


def baseline_diffs(base, mine):
    """b1609 baseline vs a b11716 run: same workload and weights; what else differs, listed."""
    why = [f'{k} differs' for k in ('payload_example', 'prompt_texts', 'ids') if base[k] != mine[k]]
    why += [f'{p} differs' for p in base['model_sha256'] if p in mine['model_sha256'] and base['model_sha256'][p] != mine['model_sha256'][p]]
    return why


def report(folders):
    f = lambda v, spec: format(v, spec) if v is not None else 'n/a'
    L = ['VIS-3D REPORT  (A-C: Oct 14 text workload on b11716, ctx 4096, temperature 0.7, n_predict 512, cores 4-7;'
         ' D: Q8 projector images)', '']
    entries, images, notes = {}, [], []
    for cfg, folder in BASELINE.items():
        if not (folder / 'result.json').exists():
            notes.append(f'baseline {cfg}: {folder} not found')
            continue
        run, rows, res, why = repro.load(folder)
        entries[cfg] = text_entry('VIS-3C ' + folder.name, cfg, rows, res, run,
                                  'BASELINE b1609 ' + ('VALID' if not why else 'NOT VALID: ' + '; '.join(why)))
    for folder in folders:
        if not (Path(folder) / 'run.json').exists():
            notes.append(f'{Path(folder).name}: no run.json (refused before the start, or not a VIS-3D folder)')
            continue
        run, rows, res, why = check_folder(folder)
        name, test = Path(folder).name, run['test']
        label = 'VALID' if not why else 'NOT VALID: ' + '; '.join(why)
        if res is None:
            notes.append(f'{name} {test}: {why[0]}')
        elif test == 'D':
            images.append((name, run, rows, res, label, folder))
        elif test in entries:
            notes.append(f'{name}: a second {test} folder, not compared ({entries[test][0]} kept)')
        else:
            entries[test] = text_entry(name, test, rows, res, run, label)
    L += ['TEXT  gen t/s = token-weighted over replies >= 10 tokens (Oct 14 rule); pooled = all tokens; accept = accepted'
          ' / drafted;', '  steps = exact verification steps (server counter, b11716 only); ms/step = decode ms / steps;'
          ' ms/pass* = decode ms /', '  (generated - accepted), a PROXY, not a step count; prompt s = median prompt_ms;'
          ' load s = spawn to /health ok', '',
          f"{'run':<30}{'test':<5}{'n':>3}{'gen t/s':>8}{'pooled':>7}{'accept':>7}{'gen tok':>8}{'steps':>6}{'ms/step':>8}"
          f"{'ms/pass*':>9}{'prompt s':>9}{'load s':>7}{'req s':>6}  label"]
    order = [k for k in ('OFF', 'ON', 'A', 'B', 'C') if k in entries]
    for k in order:
        name, test, s, *_, label = entries[k]
        L.append(f"{name:<30}{test:<5}{s['prompts']:>3}{f(s['tok_s'], '.2f'):>8}{f(s['pooled_tok_s'], '.2f'):>7}"
                 f"{f(s['accept'], '.1%'):>7}{s['gen_tokens']:>8}{f(s['steps'], '.0f'):>6}{f(s['ms_per_step'], '.1f'):>8}"
                 f"{f(s['ms_per_pass'], '.1f'):>9}{f(s['prompt_s'], '.2f'):>9}{s['load_s']:>7.1f}{s['requests_s']:>6.0f}  {label}")
    L += ['', 'MEMORY AND CONDITIONS  HWM/PSS MiB = server VmHWM / peak 5 s PSS; avail = min MemAvailable; skin/st at the'
          ' spawn -> after the stop;', '  X1 cap = policy6 max MHz at start; X1 top = block share at the X1 maximum;'
          ' samples = 5 s root samples', '',
          f"{'run':<30}{'test':<5}{'prompt tok':>11}{'HWM':>6}{'PSS':>6}{'avail':>6}{'W':>6}{'skin start->end':>17}{'st':>3}"
          f"{'bat%':>5}{'X1 cap':>7}{'X1 top':>7}{'samples':>8}"]
    for k in order:
        name, test, s, gate, start, end, *_ = entries[k]
        L.append(resource_line(name, test, s, gate, start, end, f))
    for name, run, rows, res, label, folder in images:
        L += [''] + image_report(name, rows, res, label, folder, f)
    L += ['', 'COMPARISONS  (configuration and conditions are judged separately; neither is a cause)']
    for x, y, what in (('A', 'B', 'projector residency'), ('B', 'C', 'MTP with the projector resident'),
                       ('OFF', 'A', 'build, text only (b1609 OFF -> b11716 A)'),
                       ('ON', 'C', 'build AND projector residency (b1609 ON has no projector; C has it)')):
        if x not in entries or y not in entries:
            continue
        ex, ey = entries[x], entries[y]
        ratio = ey[2]['tok_s'] / ex[2]['tok_s'] if ey[2]['tok_s'] and ex[2]['tok_s'] else None
        if (x, y) in EXPECTED_DIFF:
            d = config_diffs(ex[7], ey[7], EXPECTED_DIFF[(x, y)])
            conf = 'SAME apart from ' + what if not d else 'DIFFERENT: ' + '; '.join(d)
        else:
            d = baseline_diffs(ex[7], ey[7])
            conf = ('same workload and weights; differs by design: binary b1609-dotprod -> b11716, --metrics added'
                    + (', Q8 projector resident' if y == 'C' else '') if not d else 'DIFFERENT: ' + '; '.join(d))
        cond = repro.condition_diffs(ex, ey)
        L += [f'{y}/{x} gen t/s {f(ratio, ".3f")}  ({what})', f'  configuration: {conf}',
              f'  conditions: {"MATCHED" if not cond else "UNMATCHED: " + "; ".join(cond)}',
              f'  validity: {x} {ex[8].split(":")[0]}, {y} {ey[8].split(":")[0]}']
    return '\n'.join(L + ([''] + notes if notes else []))


def resource_line(name, test, s, gate, start, end, f):
    mib = lambda kb: kb and kb / 1024
    cap = start and start['cpufreq'].get('policy6', {}).get('max')
    skin = f"{f(gate['skin'], '.1f')}->{f(s['end_cond'].get('skin'), '.1f')}"
    return (f"{name:<30}{test:<5}{s['prompt_tokens']:>11}{f(mib(s['hwm']), '.0f'):>6}{f(mib(s['pss']), '.0f'):>6}"
            f"{f(s['avail'], 'd'):>6}{f(s['mean_w'], '.2f'):>6}{skin:>17}{f(gate['status'], 'd'):>3}"
            f"{f(start and start.get('battery_pct'), 'd'):>5}{f(cap and cap // 1000, 'd'):>7}"
            f"{f(start and end and repro.top_share(start, end, 'policy6'), '.0%'):>7}{s['samples']:>8}")


def response_flag(r):
    if not (r['reply'] or '').strip():
        return 'EMPTY'
    return 'OK' if r['stop_type'] == 'stop' else f'INCOMPLETE ({r["stop_type"]})'


def image_report(name, rows, res, label, folder, f):
    """D: per-image timings (server timings + -lv 4 log), block resources, then the grading sheet."""
    tim = image_log(server_log(folder, 'D'))
    tim = tim if len(tim) == len(rows) else [{}] * len(rows)  # unmatched sections: nothing attributed (NOT VALID)
    L = [f'IMAGES  {name}  D  {label}',
         '  encode = vision encoder (log-timestamp interval); img dec = image-token decode (logged); prompt = server',
         '  prompt_ms (encode + image decode + text prefill); text* = prompt - encode - img dec, DERIVED; wall = client',
         '  request time; TTFT not measured (not streamed)', '',
         f"  {'frame':<38}{'img tok':>8}{'encode':>8}{'img dec':>8}{'prompt':>8}{'text*':>7}{'gen tok':>8}{'gen t/s':>8}"
         f"{'wall s':>7}  reply"]
    for r, t in zip(rows, tim):
        pm_ = r['prompt_s'] and r['prompt_s'] * 1000
        text = pm_ - t['encode_ms'] - t['image_decode_ms'] if pm_ and t.get('encode_ms') is not None and \
            t.get('image_decode_ms') is not None else None
        gen = r['predicted_n'] / r['predicted_ms'] * 1000 if r['predicted_n'] and r['predicted_ms'] else None
        L.append(f"  {r['prompt_id']:<38}{f(t.get('image_tokens'), 'd'):>8}{f(t.get('encode_ms'), '.0f'):>8}"
                 f"{f(t.get('image_decode_ms'), 'd'):>8}{f(pm_, '.0f'):>8}{f(text, '.0f'):>7}{f(r['predicted_n'], 'd'):>8}"
                 f"{f(gen, '.2f'):>8}{r['wall_s']:>7.2f}  {response_flag(r)}")
    rest, trest = rows[1:], tim[1:]
    b = res['block']
    s = {'prompt_tokens': sum(r['prompt_n'] or 0 for r in rows), **resources(b)}
    L += ['', f"  first image: wall {f(rows[0]['wall_s'] if rows else None, '.2f')} s, encode "
              f"{f(tim[0].get('encode_ms') if tim else None, '.0f')} ms (includes any first-use cost)",
          f"  images 2..{len(rows)} median: wall {f(med([r['wall_s'] for r in rest]), '.2f')} s, encode "
          f"{f(med([t.get('encode_ms') for t in trest]), '.0f')} ms, image decode "
          f"{f(med([t.get('image_decode_ms') for t in trest]), '.0f')} ms, prompt "
          f"{f(med([r['prompt_s'] and r['prompt_s'] * 1000 for r in rest]), '.0f')} ms, gen t/s (token-weighted) "
          f"{f(cm.tok_s(rest), '.2f')}",
          f"  load {b['load_s']:.1f} s; responses {sum(response_flag(r) == 'OK' for r in rows)}/{len(rows)} complete; "
          f"{len(rows)}/{len(FRAMES)} frames answered", '',
          f"  {'':<28}{'test':<5}{'prompt tok':>11}{'HWM':>6}{'PSS':>6}{'avail':>6}{'W':>6}{'skin start->end':>17}{'st':>3}"
          f"{'bat%':>5}{'X1 cap':>7}{'X1 top':>7}{'samples':>8}",
          resource_line(name[:28], 'D', s, b['gate'], res['start'], res['end'], f),
          '', '  GRADING SHEET (references are Coder-drafted, owner to confirm; grade by hand, not by filename)']
    for r in rows:
        ref = REFS['frames'].get(r['prompt_id'], {'reference': 'NO REFERENCE', 'key_objects': [], 'robot_context': ''})
        L += [f"  [{r['prompt_id']}]", f"    reference: {ref['reference']}", f"    key objects: {', '.join(ref['key_objects'])}",
              f"    robot context: {ref['robot_context']}", f"    reply ({response_flag(r)}): {(r['reply'] or '').strip()}",
              '    grade: key objects named __/' + str(len(ref['key_objects'])) + '; hallucinated objects: __; '
              'robot-context errors (person, obstacle, room): __']
    L += [f"  [{fr}] NO RESPONSE: {b['end_reason']}" for fr in FRAMES[len(rows):]]
    return L


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', choices=tuple(COMMANDS))
    ap.add_argument('--report', nargs='+', type=Path, metavar='DIR')
    ap.add_argument('--thermal-log', default=str(THERMAL_LOG))
    a = ap.parse_args(argv)
    if bool(a.test) == bool(a.report):
        ap.error('give exactly one of --test and --report')
    if a.report:
        print(report(a.report))
        return
    cr.require_native()  # before any root command or server
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cr.exit_on_signal)
    try:
        why = cm.refusals() + repro.frozen_problems() + code_problems() + env_problems() + frame_problems()
        if why:
            raise SystemExit('not started: ' + '; '.join(why))
        out = OUT_ROOT / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{a.test}'
        out.mkdir(parents=True)
        print(f'checking model, drafter, projector and build SHA-256 (about 40 s); run folder {out}', flush=True)
        hashes, build = check_hashes()
        cr.wait_cores('start')
        start = repro.conditions(a.thermal_log)
        if why := repro.thermal_problems(start):
            raise SystemExit('not started: ' + '; '.join(why))
        if a.test == 'D':
            ids, texts = FRAMES, {f: f for f in FRAMES}
            payloads = {f: image_payload(f) for f in FRAMES}  # encoded before the block: no file reads in a request
            cm.payload, cr.post = payloads.__getitem__, chat_post
            example = image_payload('<frame>', jpeg='<base64 jpeg>')
        else:
            ids, texts, example = list(cm.IDS), cm.prompt_texts(), repro.payload('<text>')
        cm.command = lambda config, model='e2b': list(COMMANDS[a.test])  # run_block looks it up at call time
        (out / 'run.json').write_text(json.dumps({
            'test': a.test, 'ids': ids, 'command': COMMANDS[a.test], 'payload_example': example,
            'prompt_texts': texts if a.test != 'D' else {f: PROMPT for f in FRAMES},
            'frames_sha256': {f: r['frame_sha256'] for f, r in REFS['frames'].items()} if a.test == 'D' else None,
            'started': cr.utc(), 'model_sha256': hashes, 'build_sha256': build, 'build_info': build_info(),
            'env': {k: v for k, v in os.environ.items() if k.startswith(ENV_REFUSED + ('LD_PRELOAD',))},
            'code_sha256': {str(p): cr.sha256(p) for p in (Path(__file__), HERE / 'references.json', *CODE_SHA256,
                                                            *repro.CODE_SHA256)},
            'baseline': {k: str(v) for k, v in BASELINE.items()}, 'image_tokens': IMAGE_TOKENS,
            'block_s': cm.BLOCK_S, 'deadline_s': cm.DEADLINE_S, 'gate_max_s': 0, 'sample_s': cr.SAMPLE_S},
            indent=1) + '\n')
        pm.GATE_MAX_S = 0  # one skin reading, no wait (as VIS-3C)
        rows = []
        with open(out / 'prompts.jsonl', 'a') as plog:
            def emit(row):
                row['probe'] = probe(a.test, first=not rows)
                plog.write(json.dumps(row) + '\n')
                plog.flush()
                rows.append(row)
            ctx = {'out': out, 'thermal_log': a.thermal_log, 'idle': {'skin': start['skin']}, 'emit': emit}
            try:
                block = cm.run_block(0, a.test, ids, texts, ctx)
            except cr.CoresLost as e:
                raise SystemExit(f'not started: {e}; nothing measured, relaunch')
        result = {'block': block, 'start': start, 'end': repro.conditions(a.thermal_log), 'agents_end': repro.agents(),
                  'port_free_end': repro.port_free()}
        (out / 'result.json').write_text(json.dumps(result) + '\n')
        text = report([out])
        (out / 'report.txt').write_text(text + '\n')
        print(text, flush=True)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        for proc in list(cm.LIVE):
            cm.stop(proc)


if __name__ == '__main__':
    main()
