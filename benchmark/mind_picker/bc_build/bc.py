#!/data/data/com.termux/files/usr/bin/python
"""MIND-BC: can the vision build (b11716 + Q8 projector, ~/llama.cpp-vis) replace the robot build ("b1609", upstream
tag b10194, ~/llama.cpp-b1609-dotprod, #119) for the picker? Research only: no motors, camera, YOLO, M2, main.py or
run_mission.py; no MTP. Counters and accuracy are deterministic (temperature 0); timings NOT MEASUREMENTS; RAM
INDICATIVE (an agent may be resident).

Imports B0 (b0_cache), B0b (b0b_slots), B2 (b2_rules) and VIS-3D (gemma_vision/vis3d) unchanged; only the server
command differs: mem1.command(2048) (setup_q4 flags) with the b11716 llama-server, plus --slot-save-path <run>/slots
and VIS-3D's projector flags (--mmproj Q8_0, --image-min-tokens 70 --image-max-tokens 70). Same model file and pin.
provenance.txt (read-only, both trees): source commit and tag, CMakeCache CPU flags, SHA-256 of llama-server and
libggml-cpu, --help presence of the flags in HELP_FLAGS and the --cache-ram / --kv-unified-per-slot help text.
One launch, in order:
  K1 B0's sequence (b0.run_sequence, b0_set.json): B0's C1-C5, plus a row diff (DIFF_FIELDS) vs the B0 run of record.
  K2 B0b's L1 sequence (b0b.run_sequence 'L1'): B0b's K1 (restore ok, R7 cache_n >= min(LCP restored, n - 1)); slot
     file size.
  K3 B2 V3 on B2's TEST set (b2_rules/set/b2_test.json, SHA-256 pinned; custody ended), FWD then REV, 60 s pause
     before each block (b2.run_blocks): B2's PASS rule (b1.PASS: acceptable >= 0.80 and strict >= 0.65 in both
     orders, flips <= 4); items whose top differs from the B2 run of record.
  K4 one image request, VIS-3D frame 1 (vis3d.image_payload: prompt, temperature 0, top_k 1, seed 0, max_tokens 96)
     plus chat_template_kwargs enable_thinking false (the per-request form of VIS-3D D's --reasoning off, which the
     fixed server flags do not include): PASS when the reply is non-empty.
  K5 one server PID over all rows, all rows present, 0 request errors, no exception. The server exit code after
     SIGTERM is reported; nonzero is a FINDING (#145), not part of K5 (B0's C5 still requires 0, as in B0).
RAM (llama-server Pss, VmRSS, VmHWM; MemAvailable) after load and at the end, against B0b L0 (b1609).
Exceptions keep the rows collected so far and fail the criteria they affect (B0's interrupt/cleanup limitation is
carried over).

Usage: bc.py   (Termux python; from proot Debian or native Termux)
Writes ~/robot_runs/mind_bc/<UTC>/: run.json, requests.jsonl, provenance.txt, server.log, slots/, report.txt,
artifacts.sha256.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE.parent / 'b0b_slots'), str(HERE.parent / 'b2_rules'), str(REPO / 'benchmark/gemma_vision/vis3d')]
import b0b  # noqa: E402  imports b0 and mem1
import b2  # noqa: E402  imports b1
import vis3d  # noqa: E402  b11716 paths, pins, projector flags, image payload

b0, mem1, cr, sha = b0b.b0, b0b.mem1, b0b.cr, b0b.sha
HOME = Path(cr.HOME)
TREES = {'b1609 (upstream b10194)': HOME / 'llama.cpp-b1609-dotprod', 'b11716': HOME / 'llama.cpp-vis'}
SERVER = vis3d.SERVER
LIBCPU = f'{vis3d.BIN}/libggml-cpu.so.0.27.0'
PINS = {SERVER: vis3d.BUILD_SHA256['llama-server'], LIBCPU: vis3d.BUILD_SHA256['libggml-cpu.so.0.27.0'],
        mem1.MODEL: mem1.PINNED[mem1.MODEL], vis3d.MMPROJ: vis3d.SHA256[vis3d.MMPROJ]}
TEST_SET = HERE.parent / 'b2_rules' / 'set' / 'b2_test.json'
B0_RECORD = b0b.B0_RECORD
B0B_L0 = HERE.parent / 'b0b_slots' / 'runs' / '20261102T123725Z_L0' / 'run.json'
B0B_L1 = HERE.parent / 'b0b_slots' / 'runs' / '20261102T123830Z_L1' / 'run.json'
B2_RECORD = HERE.parent / 'b2_rules' / 'runs' / '20261102T143251Z' / 'requests.jsonl'
VIS3D_D = vis3d.HERE / 'runs' / '20261023T163445Z_D' / 'prompts.jsonl'
FRAME = vis3d.FRAMES[0]
OUT_ROOT = HOME / 'robot_runs' / 'mind_bc'
CTX = 2048
K3_BLOCKS = [('V3', 'FWD'), ('V3', 'REV')]
DIFF_FIELDS = ('prompt_tokens', 'prompt_ids_sha256', 'prompt_n', 'cache_n', 'top')
HELP_FLAGS = ('--swa-full', '--cache-ram', '--parallel', '--slot-save-path', '--kv-unified', '--mmproj',
              '--image-min-tokens', '--image-max-tokens')
CMAKE = ('CMAKE_BUILD_TYPE', 'GGML_NATIVE', 'GGML_CPU_ARM_ARCH', 'HAVE_DOTPROD', 'CMAKE_C_FLAGS', 'CMAKE_CXX_FLAGS',
         'BUILD_SHARED_LIBS', 'GGML_BACKEND_DL', 'GGML_CPU_ALL_VARIANTS')
N_ROWS = {'K1': len(b0.SEQUENCE) + len(b0.FRESH), 'K2': b0b.n_rows('L1'), 'K4': 1}


def command(slot_dir):
    cmd = mem1.command(CTX)
    cmd[cmd.index(mem1.SERVER)] = SERVER
    return cmd + ['--slot-save-path', str(slot_dir), *vis3d.VISION]


def run_out(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout


def help_has(text, flag):
    return flag in text.split() or flag + ',' in text


def provenance():
    """Read-only facts of both trees, as text."""
    L = []
    for name, t in TREES.items():
        cache = (t / 'build' / 'CMakeCache.txt').read_text().splitlines()
        keys = {k: next((l.split('=', 1)[1] for l in cache if l.split(':', 1)[0] == k), 'absent') for k in CMAKE}
        i8mm = [l for l in cache if 'i8mm' in l.lower() and not l.startswith('//')]
        server = t / 'build' / 'bin' / 'llama-server'
        helptext = run_out(str(server), '--help')
        L += [f'== {name}: {t}', f'source commit {run_out("git", "-C", str(t), "rev-parse", "HEAD").strip()}  '
              f'describe {run_out("git", "-C", str(t), "describe", "--tags").strip()}  '
              f'uncommitted {len(run_out("git", "-C", str(t), "status", "--porcelain").splitlines())} lines',
              *(f'  {k} = {v}' for k, v in keys.items()), f'  i8mm entries: {i8mm or "none"}',
              *(f'  sha256 {sha(p.read_bytes())}  {p.name}'
                for p in [server, *sorted((t / 'build' / 'bin').glob('libggml-cpu*.so*'))] if p.is_file()
                and not p.is_symlink()),
              '  --help: ' + ' '.join(f'{f}={"yes" if help_has(helptext, f) else "NO"}' for f in HELP_FLAGS)]
        lines = helptext.splitlines()
        for flag in ('--cache-ram', '--kv-unified-per-slot'):
            i = next((j for j, l in enumerate(lines) if flag in l), None)
            L += [f'  help {flag}: ' + (' '.join(x.strip() for x in lines[i:i + 3]) if i is not None else 'absent')]
    return '\n'.join(L) + '\n'


def tagged(emit, phase):
    return lambda row: emit({'phase': phase, **row})


def k1(post, s, pid, emit):
    return b0.run_sequence(post, s, pid, tagged(emit, 'K1'))


def k2(post, s, pid, emit):
    return b0b.run_sequence(post, s, 'L1', pid, tagged(emit, 'K2'))


def k3(post, test, pid, emit):
    """B2's TEST preflight and the two V3 blocks (b2.run_blocks pauses 60 s before each). Returns (rows, info)."""
    rows, info = [], {'problems': []}
    try:
        prompts = b2.preflight(post, test, K3_BLOCKS, info, 'TEST')
        if info['problems']:
            info['stopped'] = 'STOP before K3 inference: ' + '; '.join(info['problems'])
            return rows, info
        b2.run_blocks(post, test, K3_BLOCKS, prompts, pid, tagged(emit, 'K3'), rows, info, False)
    except Exception as e:
        info['exception'] = f'{type(e).__name__}: {e}'
        info['stopped'] = f'exception after {len(rows)} K3 requests: {info["exception"]}'
    return rows, info


def k4(post, pid, emit, jpeg=None):
    body = {**vis3d.image_payload(FRAME, jpeg), 'chat_template_kwargs': {'enable_thinking': False}}
    row = {'phase': 'K4', 'id': FRAME, 'pid': pid, 'pid_start': b0.pid_start(pid)}
    t0 = time.perf_counter()
    try:
        res = post('/v1/chat/completions', body)
        msg = res['choices'][0]['message']
        row.update(content=msg.get('content'), reasoning_content=msg.get('reasoning_content'),
                   finish_reason=res['choices'][0].get('finish_reason'), usage=res.get('usage'),
                   timings=res.get('timings'))
    except Exception as e:  # recorded, K4 and K5 fail
        row['error'] = f'{type(e).__name__}: {e}'
    row['wall_s'] = round(time.perf_counter() - t0, 3)
    emit(row)
    return [row], {k: v for k, v in body.items() if k != 'messages'}


def diff_rows(rows, ref, fields=DIFF_FIELDS):
    """[(id, {field: (record, now)})] for every record row that is missing or differs in a field."""
    got = {x['id']: x for x in rows}
    out = []
    for r in ref:
        x = got.get(r['id'])
        d = {f: (r.get(f), None if x is None else x.get(f)) for f in fields if x is None or x.get(f) != r.get(f)}
        if d:
            out.append((r['id'], d))
    return out


def agreement(rows, ref):
    """[(order, id, record top, now top)] for TEST V3 items whose choice differs (or is missing) from the record."""
    now = {(x['order'], x['id']): x.get('top') for x in rows}
    return [(o, i, t, now.get((o, i))) for (o, i), t in
            sorted(((r['order'], r['id']), r['top']) for r in ref if r.get('set') == 'TEST' and r['variant'] == 'V3')
            if now.get((o, i)) != t]


def criteria(ph, run, test, refs):
    """[(name, PASS|FAIL|INFO|FINDING, text)] from the recorded rows alone. ph[phase] = (rows, info)."""
    code, killed, exc = run.get('exit_code'), run.get('killed'), run.get('exception')
    rows1, info1 = ph.get('K1', ([], {}))
    out = [(f'K1 {c}', v, t) for c, v, t in b0.criteria(rows1, info1, code, killed, exc)]
    d = diff_rows(rows1, refs['b0'])
    out.append(('K1 diff', 'INFO', f'{len(d)} of {len(refs["b0"])} rows differ from the B0 run of record in '
                f'{DIFF_FIELDS}: ' + ('; '.join(f'{i} {v}' for i, v in d) or 'none')))
    rows2, info2 = ph.get('K2', ([], {}))
    c = next(x for x in b0b.criteria('L1', rows2, info2, code, killed, exc) if x[0] == 'K1')
    out.append(('K2', c[1], c[2] + f'; slot file {run.get("slot_file_bytes")} bytes (b1609 B0b L1: '
                f'{refs["b0b_l1_bytes"]})'))
    rows3, info3 = ph.get('K3', ([], {}))
    F, R, arm = b2.block_metrics(rows3, test, 'V3')
    full3 = len(rows3) == 2 * len(test['items']) and not any('error' in x for x in rows3) and not info3.get('stopped')
    out.append(('K3', arm['verdict'] if full3 else 'FAIL', f'strict {F["strict"]:.3f}/{R["strict"]:.3f}, acceptable '
                f'{F["acceptable"]:.3f}/{R["acceptable"]:.3f} (FWD/REV), flips {len(arm["flips"])} {arm["flips"]}; '
                f'{len(rows3)}/{2 * len(test["items"])} rows'))
    a = agreement(rows3, refs['b2'])
    out.append(('K3 agreement', 'INFO', f'{len(a)} of {2 * len(test["items"])} items differ from the B2 run of '
                f'record (order, id, b1609, b11716): {a or "none"}'))
    rows4 = ph.get('K4', ([], {}))[0]
    ok4 = len(rows4) == 1 and 'error' not in rows4[0] and bool((rows4[0].get('content') or '').strip())
    out.append(('K4', 'PASS' if ok4 else 'FAIL', f'reply {rows4[0].get("content") if rows4 else None!r}'))
    every = [x for p in ('K1', 'K2', 'K3', 'K4') for x in ph.get(p, ([], {}))[0]]
    pids = {(x.get('pid'), x.get('pid_start')) for x in every}
    errors = sum('error' in x for x in every)
    counts = {p: len(ph.get(p, ([], {}))[0]) for p in ('K1', 'K2', 'K3', 'K4')}
    want = {**N_ROWS, 'K3': 2 * len(test['items'])}
    stops = [f'{p}: {ph[p][1].get("stopped")}' for p in ph if ph[p][1].get('stopped')]
    ok5 = len(pids) == 1 and errors == 0 and counts == {p: want[p] for p in counts} and not exc and not stops
    out.append(('K5', 'PASS' if ok5 else 'FAIL', f'PIDs {sorted(pids, key=str)}; rows {counts} of {want}; '
                f'{errors} errors; exception {exc}; stops {stops or "none"}'))
    out.append(('exit', 'INFO' if code == 0 and not killed else 'FINDING',
                f'server exit code {code} after SIGTERM, SIGKILL {killed}'))
    return out


def ram(pid):
    return b0b.ram(pid)


def report(run, ph, crit, refs):
    L = [f'MIND-BC {run["utc"]}  code {run["code_sha256"][:12]}  base {run["git_head"]}',
         f'server: {" ".join(run["command"])}', f'pins: {run["pins"]}',
         f'label: counters and accuracy {run["validity"]}; timings NOT MEASUREMENTS; RAM INDICATIVE', '',
         '--- provenance (read-only) ---', run['provenance'].rstrip(), '']
    if run.get('exception'):
        L.append(f'EXCEPTION (runner): {run["exception"]}')
    ref0 = {x['id']: x for x in refs['b0']}
    for p in ('K1', 'K2'):
        rows, info = ph.get(p, ([], {}))
        L += [f'--- {p} rows (now | B0 run of record, b1609) ---',
              'id        kind      n    prompt_n cache_n  top      | n    prompt_n cache_n  top']
        for x in rows:
            r = ref0.get(x['id'], {})
            L.append(f'{x["id"]:<9} {x["kind"]:<9} {x["prompt_tokens"]:<4} {str(x.get("prompt_n")):<8} '
                     f'{str(x.get("cache_n")):<8} {str(x.get("top")):<8} | {str(r.get("prompt_tokens")):<4} '
                     f'{str(r.get("prompt_n")):<8} {str(r.get("cache_n")):<8} {r.get("top")}' +
                     (f'  ERROR {x["error"]}' if 'error' in x else ''))
        L += [f'  reply R4 {info.get("reply")!r}; R6 {info.get("r6_reply")!r}'] + \
             [f'  slot {a["action"]}: {a["response"]}' for a in info.get('slot_actions', [])]
        if info.get('stopped'):
            L.append(f'  STOPPED: {info["stopped"]}')
    rows3, info3 = ph.get('K3', ([], {}))
    L += ['', '--- K3 (B2 V3, TEST) ---', f'prompt tokens {info3.get("prompt_tokens")}; name ids '
          f'{info3.get("name_ids")}'] + [f'block {x["block"]}: {x["duration_s"]} s*' for x in info3.get('blocks', [])]
    F, R, arm = b2.block_metrics([x for x in refs['b2'] if x.get('set') == 'TEST'], refs['test'], 'V3')
    L.append(f'B2 run of record (b1609): strict {F["strict"]:.3f}/{R["strict"]:.3f}, acceptable '
             f'{F["acceptable"]:.3f}/{R["acceptable"]:.3f}, flips {len(arm["flips"])} {arm["flips"]} -> {arm["verdict"]}')
    if info3.get('stopped'):
        L.append(f'STOPPED: {info3["stopped"]}')
    rows4, body4 = ph.get('K4', ([], {}))
    L += ['', '--- K4 (image, VIS-3D frame 1) ---', f'frame {FRAME}; settings {body4}']
    for x in rows4:
        L += [f'reply (verbatim): {x.get("content")!r}', f'reasoning_content: {x.get("reasoning_content")!r}; finish '
              f'{x.get("finish_reason")}; usage {x.get("usage")}; wall {x.get("wall_s")} s*'] + \
             ([f'ERROR {x["error"]}'] if 'error' in x else [])
    L.append(f'VIS-3D D reply for this frame (server --reasoning off, ctx 4096): {refs["vis3d_reply"]!r}')
    L += ['', 'RAM (INDICATIVE, kB): now b11716 + projector | b1609 B0b L0']
    for when in ('after_load', 'end'):
        a, b = run.get('ram', {}).get(when, {}), refs['b0b_l0_ram'].get(when, {})
        L.append(f'  {when}: ' + '; '.join(f'{k} {a.get(k)} | {b.get(k)}' for k in ('Pss', 'VmRSS', 'VmHWM',
                                                                                     'MemAvailable')))
    L += ['* timings NOT MEASUREMENTS (agent resident; no thermal or core control)', '', 'criteria:'] + \
         [f'  {c:<13} {v:<7} {t}' for c, v, t in crit]
    return '\n'.join(L) + '\n'


def references():
    test, _ = b2.load_set(TEST_SET, b2.TEST_SHA256, custody=False)
    test['tag'] = 'TEST'
    d = [json.loads(l) for l in VIS3D_D.read_text().splitlines()]
    return {'b0': [json.loads(l) for l in B0_RECORD.read_text().splitlines()],
            'b2': [json.loads(l) for l in B2_RECORD.read_text().splitlines()], 'test': test,
            'b0b_l0_ram': json.loads(B0B_L0.read_text())['ram'],
            'b0b_l1_bytes': json.loads(B0B_L1.read_text()).get('slot_file_bytes'),
            'vis3d_reply': next((x.get('reply') for x in d if x.get('prompt_id') == FRAME), None)}


def finish(out, run, ph, refs):
    s_end = sha(b0.SET.read_bytes()), sha(TEST_SET.read_bytes())
    run['set_sha256_end'] = {'b0': s_end[0], 'test': s_end[1]}
    run['criteria'] = crit = criteria(ph, run, refs['test'], refs)
    ok = dict((c, v) for c, v, _ in crit)['K5'] == 'PASS' and s_end == (b0b.SET_SHA256, b2.TEST_SHA256)
    run['validity'] = 'VALID' if ok else 'INVALID'
    run['info'] = {p: {k: v for k, v in i.items() if k != 'test'} for p, (_, i) in ph.items()}
    (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
    text = report(run, ph, crit, refs)
    (out / 'report.txt').write_text(text)
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'artifacts.sha256')
    (out / 'artifacts.sha256').write_text(''.join(f'{sha(p.read_bytes())}  {p.relative_to(out)}\n' for p in files))
    return text


def main():
    s_bytes = b0.SET.read_bytes()
    if sha(s_bytes) != b0b.SET_SHA256:
        raise SystemExit(f'refused: {b0.SET} SHA-256 {sha(s_bytes)[:12]} != {b0b.SET_SHA256[:12]}')
    s = json.loads(s_bytes)
    refs = references()  # B2's TEST set: hash checked by load_set
    bad = [f'{p}: SHA-256 != pinned {h[:12]}' for p, h in PINS.items() if not Path(p).is_file() or cr.sha256(p) != h]
    bad += vis3d.frame_problems() + vis3d.env_problems()
    if bad:
        raise SystemExit('refused: ' + '; '.join(bad))
    if not b0.repro.port_free():
        raise SystemExit('refused: port 8080 in use (never stops a server it did not start)')
    utc = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = OUT_ROOT / utc
    out.mkdir(parents=True)  # a new folder; an existing one raises
    slot_dir = out / 'slots'
    slot_dir.mkdir()
    prov = provenance()
    (out / 'provenance.txt').write_text(prov)
    run = {'utc': utc, 'code_sha256': sha(Path(__file__).read_bytes()),
           'imports_sha256': {m.__name__: sha(Path(m.__file__).read_bytes()) for m in (b0, b0b, b2, b2.b1, mem1, vis3d)},
           'git_head': run_out('git', '-C', str(HERE), 'rev-parse', 'HEAD').strip(), 'pins': PINS,
           'set_sha256': {'b0': sha(s_bytes), 'test': b2.TEST_SHA256}, 'command': command(slot_dir),
           'provenance': prov, 'timings': 'NOT MEASUREMENTS', 'ram': {}}
    (out / 'run.json').write_text(json.dumps(run, indent=1) + '\n')

    def post(path, body):
        if body is None:  # GET
            with urllib.request.urlopen(f'http://127.0.0.1:{cr.PORT}{path}', timeout=30) as r:
                return json.loads(r.read())
        return cr.post(path, body, timeout=b0.REQUEST_TIMEOUT_S)

    ph = {}
    with open(out / 'requests.jsonl', 'x') as rq, open(out / 'server.log', 'x') as log:
        def emit(row):
            rq.write(json.dumps(row) + '\n')
            rq.flush()
        killed, proc, exc = False, None, None
        try:
            proc = subprocess.Popen(run['command'], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            t0 = time.monotonic()
            while not cr.healthy():
                if proc.poll() is not None or time.monotonic() - t0 > b0.LOAD_TIMEOUT_S:
                    raise RuntimeError(f'llama-server not healthy (exit {proc.poll()})')
                time.sleep(0.1)
            run['server_pid'], run['server_comm'] = proc.pid, Path(f'/proc/{proc.pid}/comm').read_text().strip()
            run['ram']['after_load'] = ram(proc.pid)
            ph['K1'] = k1(post, s, proc.pid, emit)
            ph['K2'] = k2(post, s, proc.pid, emit)
            ph['K3'] = k3(post, refs['test'], proc.pid, emit)
            ph['K4'] = k4(post, proc.pid, emit)
            run['ram']['end'] = ram(proc.pid)
        except BaseException as e:  # recorded; the report and manifest are still written, then a stop re-raises
            exc = e
            run['exception'] = f'{type(e).__name__}: {e}'
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=b0.TERM_WAIT_S)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    killed = True
            run.update(exit_code=proc.returncode if proc else None, killed=killed)
    if (slot_dir / b0b.SLOT_FILE).is_file():
        run['slot_file_bytes'] = (slot_dir / b0b.SLOT_FILE).stat().st_size
    print(finish(out, run, ph, refs) + f'\n{out}')
    if exc is not None and not isinstance(exc, Exception):
        raise exc


if __name__ == '__main__':
    main()
