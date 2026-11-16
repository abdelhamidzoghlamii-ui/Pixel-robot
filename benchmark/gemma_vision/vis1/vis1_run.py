"""VIS-1: owner-only 180 s request windows; motors off. --dry-run is NOT VALID TIMING.

Native --monitor bridges root telemetry to the Debian runner. No hardware/USB imports.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NATIVE = Path('/data/data/com.termux/files/home')
HOME = Path('/termux-home') if Path('/termux-home').is_dir() else NATIVE
ORDER = ('T1', 'T2', 'V1', 'V2', 'V3', 'V4')
IDS = 'A5 A6 B1 B2 B6 B7 C1 C6 C7 C8 C9 C10 C11'.split()
FRAMES = ('bench_photos__blocked_1', 'bench_photos__empty_2',
          'bench_photos__person_facing_150cm', 'bench_photos__person_sideon_300cm',
          'bench_photos__roomsig_couch_1', 'bench_photos__roomsig_refrigerator_1',
          'bench_photos__roomsig_toilet_1', 'bench_photos__roomsig_toilet_2',
          'test_photos__chiara_close_back', 'test_photos__abdel_close_right')
MODEL = NATIVE / 'models/gemma-4-E2B-it-Q4_0.gguf'
DRAFT = NATIVE / 'models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf'
DRAFT_ARGS = ['--model-draft', str(DRAFT), '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']
BF16 = NATIVE / 'models/vis0/mmproj-gemma-4-E2B-it-BF16.gguf'
Q8 = BF16.with_name('mmproj-gemma-4-E2B-it-Q8_0.gguf')
BINS = {b: NATIVE / f'llama.cpp-{b}/build/bin/llama-server' for b in ('b1609-dotprod', 'vis')}
EXPECTED_HASHES = {MODEL:'8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52',
                   DRAFT:'9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919',
                   BF16:'711e1e8f43fa0664adbac493129be1e6c25b81af4b4cdea97c7d798b25c0a3a4',
                   Q8:'9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292'}
DEBIAN_IMPORTS = ('requests', 'PIL', 'PIL.ImageDraw', 'PIL.ImageFont', 'PIL.ImageOps')
SMOKE_LABEL = 'SMOKE — NOT VALID TIMING'
POLICIES = ('policy0', 'policy4', 'policy6')
BAT = '/sys/class/power_supply/battery'
CPU = '/sys/devices/system/cpu/cpufreq'
BLOCK_S = 180
DRY_LABEL = 'NOT VALID TIMING (offline dry-run; no servers, gates or measurements)'


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')
        f.flush()


def rows(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text().splitlines()
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise
    return out


def blocks(only):
    selected = only.split(',') if only else list(ORDER)
    if len(set(selected)) != len(selected) or set(selected) - set(ORDER):
        raise ValueError('--only requires unique block names T1,T2,V1,V2,V3,V4')
    return [b for b in ORDER if b in selected]


def command(block, t2_draft=True):
    binary = BINS['b1609-dotprod' if block == 'T1' else 'vis']
    cmd = ['taskset', '-c', '4-7', 'timeout', '--signal=TERM', '--kill-after=3', '900', str(binary),
           '-m', str(MODEL), '--host', '127.0.0.1', '--port', '8091', '--ctx-size', '4096',
           '--threads', '4', '--threads-batch', '4', '--parallel', '1', '--swa-full', '--cache-ram', '0']
    if block == 'T1' or block == 'T2' and t2_draft:
        cmd += DRAFT_ARGS
    if block.startswith('V'):
        budget = 70 if block in ('V1', 'V2') else 140
        cmd += ['--mmproj', str(BF16 if block in ('V1', 'V4') else Q8),
                '--image-min-tokens', str(budget), '--image-max-tokens', str(budget),
                '--reasoning', 'off', '-lv', '4']
    return cmd


def text_payload(text):
    sys.path.insert(0, str(REPO))
    import server_manager
    assert Path(server_manager.__file__).resolve() == (REPO/'server_manager.py').resolve()
    import chat
    # chat injects the live checkout at import: remove it in isolated review copies.
    sys.path[:] = [str(REPO)] + [p for p in sys.path if p not in (str(REPO), str(NATIVE/'robot'))]
    assert Path(chat.__file__).resolve() == (REPO/'chat.py').resolve()
    return chat.chat_payload(chat.build_prompt('gemma', [{'role': 'user', 'content': text}]), 'gemma')


def vision_payload(jpeg, yolo):
    # Marked copy: VIS-0 build_vision_chat_payload + build_optional_yolo_vision_text.
    text = ('Describe in one or two short sentences what is in front of the robot. '
            'Mention only what you can see.\n\nObjects detected by YOLO:\n' + yolo)
    return dict(messages=[dict(role='user', content=[
        dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(jpeg).decode())),
        dict(type='text', text=text)])], temperature=0, top_k=1, seed=0, max_tokens=96,
        n_predict=96, cache_prompt=False, stream=False)


def inputs():
    bench = module('vis1_bench', 'benchmark/llm_objective_setting/bench.py')
    by_id = {p['id']: p for p in bench.PROMPTS_A + bench.PROMPTS_B + bench.PROMPTS_C}
    texts = [(i, text_payload(' '.join(t['user'] for t in by_id[i]['text'])
                             if by_id[i].get('multi') else by_id[i]['text'])) for i in IDS]
    root = REPO/'benchmark/relate_anything/quality1/runs/20261017T213604Z'
    source = json.loads((root/'results.json').read_text())
    photos = {Path(p['frame_file']).stem: p for p in source['photos']}
    images = []
    from PIL import Image
    for name in FRAMES:
        rec = photos[name]
        path = root/'frames'/f'{name}.jpg'
        with Image.open(path) as im:
            assert max(im.size) == 640
        # VIS-0 YOLO extraction; old stored Gemma-3 prompt is deliberately not reused after TPL-FIX.
        dets = rec.get('m2_detections') or rec.get('detections') or []
        yolo = '\n'.join(f"- {d['class_name']} [{i}], {d['position']}, {d['distance']}" for i,d in enumerate(dets))
        images.append((name, vision_payload(path.read_bytes(), yolo)))
    return texts, images


def own_stop(proc, server=None):
    """Only the private process group created by our Popen; timeout is a second hard bound."""
    if proc.poll() is None:
        # VIS1-FIX3: GNU timeout re-sends a TERM it receives to its whole group (kill(0, sig)), so
        # TERM to timeout reached llama-server twice ("second interrupt" -> abort). TERM the owned
        # server once (still our direct child in our group); timeout then exits with its status.
        # Server not known yet: fall back to timeout.
        owned = server is not None and (server, proc.pid, proc.pid) in set(proc_rows())
        try:
            os.kill(server if owned else proc.pid, signal.SIGTERM)
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except ProcessLookupError:
            proc.wait(timeout=5)
    return proc.returncode


OWNED = []
DEFERRED = dict(on=False, signum=None)


def interrupted(n, frame):
    if DEFERRED['on']:
        DEFERRED['signum'] = n
    else:
        raise SystemExit(f'interrupted {n}')


def spawn(cmd, log):
    DEFERRED.update(on=True, signum=None)
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        OWNED.append(proc)
    finally:
        DEFERRED['on'] = False
    if DEFERRED['signum'] is not None:
        own_stop(proc)
        raise SystemExit(f'interrupted {DEFERRED["signum"]}')
    return proc


AGENT_NAMES = {'claude', 'codex', 'agy'}


def agent_matches():
    """Processes whose own binary is claude/codex/agy, not merely a path or argument containing those
    letters (VIS1-FIX: pgrep -f alone matches substrings anywhere in argv, e.g. a --config path under
    ~/.codex/). pgrep -f still casts the wide net; comm/argv0 narrows it to real agent processes,
    including a lingering app-server-daemon left behind after the owner closes the interactive CLI."""
    p = subprocess.run(['pgrep', '-f', 'claude|codex|agy'], capture_output=True, text=True, timeout=10)
    if p.returncode not in (0, 1):
        raise RuntimeError('agent process check failed')
    found = []
    for token in p.stdout.split():
        pid = int(token)
        if pid == os.getpid():
            continue
        try:
            comm = Path(f'/proc/{pid}/comm').read_text().strip()
            parts = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            cmdline = ' '.join(c.decode(errors='replace') for c in parts if c)
        except (FileNotFoundError, ProcessLookupError):
            continue  # exited between pgrep and read
        argv0 = Path(cmdline.split(' ', 1)[0]).name if cmdline else ''
        if comm in AGENT_NAMES or argv0 in AGENT_NAMES:
            found.append(dict(pid=pid, comm=comm, cmdline=cmdline))
    return found


def log_agent_matches(matches):
    for m in matches:
        print(f"agent match: pid={m['pid']} comm={m['comm']} cmdline={m['cmdline']}", flush=True)


def agent_found(log=False):
    matches = agent_matches()
    if matches and log:
        log_agent_matches(matches)
    return bool(matches)


def free_ports():
    if os.environ.get('PREFIX') == '/data/data/com.termux/files/usr' and not Path('/termux-home').is_dir():
        listeners = root('/system/bin/ss -H -ltn','ports')
        for line in listeners.splitlines():
            fields = line.split()
            if len(fields) >= 4 and re.search(r':(?:8080|8090|8091)$',fields[3]):
                raise RuntimeError('existing listener on a VIS-1 reserved port')
    for port in (8080, 8090, 8091):
        # Check IPv4 and IPv6 loopback, including non-HTTP listeners.
        for family, host in ((socket.AF_INET, '127.0.0.1'), (socket.AF_INET6, '::1')):
            with socket.socket(family) as s:
                s.settimeout(.3)
                if s.connect_ex((host, port)) == 0:
                    raise RuntimeError(f'port {port} already has a listener')


def gate_ok(skin, idle):
    return skin is not None and math.isfinite(skin) and skin <= idle + 1.5


def telemetry(bridge, allow_agent=False):
    if (bridge/'abort').exists():
        raise SystemExit('owner aborted wrapper')
    d = json.loads((bridge/'latest.json').read_text())
    if time.monotonic() - d['t'] > 12:
        raise RuntimeError('native telemetry stale')
    if d.get('error'):
        raise RuntimeError('native telemetry failed: ' + d['error'])
    if not 50 <= d['capacity'] <= 80 or d['battery_status'] != 'Discharging':
        raise RuntimeError('battery outside 50-80% Discharging band')
    if d['agent'] and not allow_agent:
        log_agent_matches(d.get('agent_detail', []))
        raise RuntimeError('agent found')
    if d['skin'] is None or d['status'] is None:
        raise RuntimeError('missing VIRTUAL-SKIN or Android status')
    if d['status'] >= 4 or d['bat_c'] >= 45:
        raise RuntimeError('thermal stop (Android CRITICAL or battery >=45 C)')
    return d


def gate(bridge, idle, now=time.monotonic, sleep=time.sleep):
    start = now()
    while True:
        d = telemetry(bridge)
        if gate_ok(d['skin'], idle):
            return dict(skin=d['skin'], status=d['status'], waited_s=now()-start, missed=False)
        if now()-start >= 900:
            raise RuntimeError('gate missed: not cool after 15 minutes; no warm-start block')
        sleep(5)


def post(path, body=None, timeout=30):
    req = urllib.request.Request('http://127.0.0.1:8091'+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def proc_rows():
    """(pid, ppid, pgid) for every process, from /proc/<n>/stat (comm may contain spaces/parens)."""
    for d in Path('/proc').iterdir():
        if d.name.isdigit():
            try:
                rest = (d/'stat').read_text().rsplit(')', 1)[1].split()
                yield int(d.name), int(rest[1]), int(rest[2])
            except (OSError, IndexError, ValueError):
                pass  # exited mid-scan


def server_pid(proc):
    # VIS1-FIX2: /proc/<pid>/task/<pid>/children does not exist on this kernel, so every block
    # timed out here and killed a healthy server. Same rule as before, via ppid/pgid, never by name.
    children = [pid for pid, ppid, pgid in proc_rows() if ppid == proc.pid and pgid == proc.pid]
    if len(children) != 1:
        raise RuntimeError('cannot identify the owned timeout child')
    return children[0]


def group_alive(pgid):
    return any(g == pgid for _, _, g in proc_rows())


def check_cores(pid):
    # Also check the Android cpuset: affinity alone can hide a background restriction.
    status = Path(f'/proc/{pid}/status').read_text()
    m = re.search(r'^Cpus_allowed_list:\s*(.+)$', status, re.M)
    if not m or m.group(1).strip() != '4-7':
        raise RuntimeError('Termux/server moved off cores 4-7')
    # Each thread inherits the launch mask; inspect every server thread for later changes.
    for p in Path(f'/proc/{pid}/task').glob('*/status'):
        match = re.search(r'^Cpus_allowed_list:\s*(.+)$', p.read_text(), re.M)
        if not match or match.group(1).strip() != '4-7':
            raise RuntimeError('server thread moved off cores 4-7')


VIS0_PARSER = None

def request_record(name, data, start, end, log_delta, block):
    tm = data.get('timings', {})
    if not tm:
        raise RuntimeError('response missing server timings')
    reply = data.get('content')
    if block.startswith('V'):
        reply = data['choices'][0]['message']['content']
    # VIS-0 log parser has no server side effects. Rename its agent-era field explicitly.
    global VIS0_PARSER
    if block.startswith('V') and VIS0_PARSER is None:
        VIS0_PARSER = module('vis1_vis0', 'benchmark/gemma_vision/vis0/vis0.py').parse_server_log_delta
    parsed = VIS0_PARSER(log_delta) if block.startswith('V') else {}
    return dict(input=name, started=start, ended=end, wall_ms=(end-start)*1000, reply=reply,
                timings=tm, prompt_tokens=tm.get('prompt_n'), prompt_ms=tm.get('prompt_ms'),
                generated_tokens=tm.get('predicted_n'), predicted_per_second=tm.get('predicted_per_second'),
                draft_n=tm.get('draft_n'), draft_n_accepted=tm.get('draft_n_accepted'),
                image_encode_ms=parsed.get('image_encode_ms_not_a_measurement'),
                image_tokens=parsed.get('image_tokens_logged'), usage=data.get('usage'),
                budget=None if block.startswith('T') else (70 if block in ('V1','V2') else 140),
                mmproj=None if block.startswith('T') else ('BF16' if block in ('V1','V4') else 'Q8_0'))


def run_block(block, out, bridge, idle, payloads, t2_draft):
    record = dict(phase_start=time.monotonic(), block=block, label='NOT VALID', reason='interrupted', requests=[], samples=[],
                  gate=None, command=command(block,t2_draft), started_epoch=time.time())
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = out/f'{block}_{attempt}.json'
    logpath = out/f'{block}_{attempt}.server.log'
    proc = pid = None
    try:
        record['gate'] = gate(bridge,idle)
        free_ports()
        if agent_found(log=True):
            raise RuntimeError('agent found before launch')
        # Cold load handshake, served only by native owner wrapper.
        (bridge/'drop_done').unlink(missing_ok=True)
        (bridge/'drop_request').touch()
        began = time.monotonic()
        while not (bridge/'drop_done').exists():
            if time.monotonic()-began > 30:
                raise RuntimeError('cache-drop handshake timeout')
            time.sleep(.1)
        if (bridge/'drop_done').read_text().strip() != 'ok':
            raise RuntimeError('cache drop failed')
        (bridge/'drop_request').unlink()
        (bridge/'drop_done').unlink()
        # Cache drop can take time: the actual launch still must satisfy the gate.
        record['gate_after_drop'] = gate(bridge,idle)
        with logpath.open('w') as log:
            began = time.monotonic()
            proc = spawn(record['command'],log)
            pid = None
            while pid is None:
                if proc.poll() is not None or time.monotonic()-began > 10:
                    raise RuntimeError('server crashed or owned child not found')
                try:
                    pid = server_pid(proc)
                except (OSError,RuntimeError):
                    time.sleep(.05)
            write(bridge/'target.json',dict(pid=pid))
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('server crashed during load')
                try:
                    if post('/health',timeout=1).get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                telemetry(bridge)
                if time.monotonic()-began >= 180:
                    raise RuntimeError('server load timeout')
                time.sleep(.2)
            pid = server_pid(proc)
            write(bridge/'target.json',dict(pid=pid))
            record.update(server_pid=pid,load_s=time.monotonic()-began)
            check_cores(pid)
            endpoint = '/completion' if block.startswith('T') else '/v1/chat/completions'
            post(endpoint,payloads[0][1],timeout=180)  # discarded warm-up
            check_cores(pid)
            telemetry(bridge)
            record['measure_start'] = start = time.monotonic()
            deadline = start+BLOCK_S
            k = 0
            while time.monotonic() < deadline:
                telemetry(bridge)
                check_cores(pid)
                name, body = payloads[k % len(payloads)]
                offset = logpath.stat().st_size
                t = time.monotonic()
                try:
                    data = post(endpoint,body,timeout=max(.001,deadline-t))
                except (TimeoutError, OSError) as e:
                    if time.monotonic() >= deadline:
                        record['cutoff_request'] = dict(input=name,reason='180 s boundary; excluded',error=type(e).__name__)
                        break
                    raise
                end = time.monotonic()
                if end > deadline:
                    record['cutoff_request'] = dict(input=name,reason='reply after boundary; excluded')
                    break
                with logpath.open() as f:
                    f.seek(offset)
                    delta = f.read()
                row = request_record(name,data,t,end,delta,block)
                record['requests'].append(row)
                write(path,record)  # keep replies across interruption
                k += 1
            record['measure_end'] = deadline
            if proc.poll() is not None:
                raise RuntimeError('server crashed during measurement')
            check_cores(pid)
            telemetry(bridge)
            record.update(label='VALID',reason='completed 180 s request window')
    except (Exception, SystemExit) as e:
        record.update(label='NOT VALID',reason=str(e))
    finally:
        record['stop_start'] = time.monotonic()
        write(bridge/'target.json',dict(pid=None))
        if proc is not None:
            record['server_exit'] = own_stop(proc, pid)
        record['stopped'] = time.monotonic()
        record['samples'] = [s for s in rows(bridge/'samples.jsonl')
                             if s['t'] >= record['phase_start']-1
                             and s['t'] <= record['stopped']]
        record['lmk'] = lmk_request(bridge,record['started_epoch'])
        if not record['lmk'].get('ok'):
            record.update(label='NOT VALID',reason=record['reason']+'; LMK query failed')
        if not record['requests']:
            record.update(label='NOT VALID',reason=record['reason']+'; no completed requests')
        record['metrics'] = metrics(record)
        if any(s.get('agent') or s.get('battery_status') != 'Discharging' or not 50 <= s.get('capacity',-1) <= 80
               or s.get('cores_ok') is False for s in record['samples']):
            record.update(label='NOT VALID',reason=record['reason']+'; agent, battery or affinity violation in samples')
        errors = [e for e in rows(bridge/'errors.jsonl') if record['phase_start'] <= e['t'] <= record['stop_start']]
        record['sampling_errors'] = errors
        if errors:
            record.update(label='NOT VALID',reason=record['reason']+'; sampling failed')
        write(path,record)
    return record


def metrics(record):
    # Reuse the campaign's receipt-time trapezoids; no extrapolation across missing samples.
    sys.path.insert(0,str(REPO/'benchmark/campaign'))
    import power
    samples = record['samples']
    start,end = record.get('measure_start'),record.get('measure_end')
    measured = [s for s in samples if start is not None and start <= s['t'] <= (end or start)]
    result = dict(peak_vmhwm_kib=max((s['vmhwm_kib'] for s in samples if s.get('vmhwm_kib') is not None),default=None),
                  peak_pss_kib=max((s['pss_kib'] for s in samples if s.get('pss_kib') is not None),default=None),
                  min_mem_available_kib=min((s['mem_available_kib'] for s in samples),default=None))
    if start is None or end is None:
        return result
    result['power'] = power.weighted(samples,start,end)
    result['skin_start'] = measured[0]['skin'] if measured else None
    result['skin_end'] = measured[-1]['skin'] if measured else None
    result['status_max'] = max((s['status'] for s in measured if s.get('status') is not None),default=None)
    # Marked copy from power_map.capped_by_policy: later sample owns the previous gap.
    result['caps'] = {}
    for policy in POLICIES:
        previous,capped,known = start,0.,0.
        for s in measured:
            dt = s['t']-previous
            previous = s['t']
            value,top = s['max'].get(policy),s['cpuinfo_max_khz'].get(policy)
            if value is not None and top is not None:
                known += dt
                if value < top:
                    capped += dt
        result['caps'][policy] = dict(percent=100*capped/(end-start),unknown_s=(end-start)-known)
    return result


def lmk_request(bridge, epoch):
    write(bridge/'lmk_request.json',dict(epoch=epoch))
    start = time.monotonic()
    while not (bridge/'lmk_done.json').exists():
        if time.monotonic()-start > 40:
            return dict(ok=False,error='LMK bridge timeout')
        time.sleep(.1)
    d = json.loads((bridge/'lmk_done.json').read_text())
    (bridge/'lmk_request.json').unlink()
    (bridge/'lmk_done.json').unlink()
    return d


def root(script, tag):
    """#123 form, native only: Android-service output goes to a file before it is read."""
    dest = f'/data/local/tmp/vis1_{os.getpid()}_{tag}.txt'
    p = subprocess.run(['su','-c',f'({script}) </dev/null >{dest} 2>&1'],
                       capture_output=True,text=True,timeout=30)
    q = subprocess.run(['su','-c',f'cat {dest}'],capture_output=True,text=True,timeout=10)
    if p.returncode or q.returncode:
        raise RuntimeError(f'root {tag} failed ({p.returncode}/{q.returncode})')
    return q.stdout


def parse_dump(text):
    # Marked copy from coresidency.parse_dump: HAL current section only, never cached temperatures.
    status = re.search(r'^Thermal Status: (\d+)\s*$',text,re.M)
    section = re.search(r'^Current temperatures from HAL:\n((?:[ \t]+[^\n]*\n?)*)',text,re.M)
    skin = re.search(r'Temperature\{mValue=([^,]+), mType=-?\d+, mName=VIRTUAL-SKIN,',
                     section.group(1) if section else '')
    return dict(status=int(status.group(1)) if status else None,skin=float(skin.group(1)) if skin else None)


def monitor(bridge):
    if Path('/termux-home').is_dir() or os.environ.get('PREFIX') != '/data/data/com.termux/files/usr':
        raise SystemExit('monitor must run in native Termux')
    os.sched_setaffinity(0,{0,1,2,3})
    slow_state, ready = {}, threading.Event()
    def slow_samples():
        while not (bridge/'stop').exists():
            began = time.monotonic()
            try:
                matches = agent_matches()
                slow_state['latest'] = dict(parse_dump(root('dumpsys thermalservice','thermal')),
                                            thermal_at=time.monotonic(),agent=bool(matches),agent_detail=matches)
            except Exception as e:
                slow_state['latest'] = dict(error=str(e))
                append(bridge/'errors.jsonl',dict(t=time.monotonic(),error=str(e)))
            ready.set()
            # Slow HAL calls must not create holes in campaign-style power integration.
            for _ in range(max(0,int((5-(time.monotonic()-began))/.1))):
                if (bridge/'stop').exists():return
                time.sleep(.1)
    slow_thread = threading.Thread(target=slow_samples,daemon=True)
    slow_thread.start()
    if not ready.wait(35):
        raise RuntimeError('native thermal monitor did not initialize')
    with (bridge/'samples.jsonl').open('a') as log:
        while not (bridge/'stop').exists():
            began = time.monotonic()
            try:
                latest = dict(slow_state['latest'])
                if latest.get('error'):
                    raise RuntimeError(latest['error'])
                pid = json.loads((bridge/'target.json').read_text()).get('pid') if (bridge/'target.json').exists() else None
                script = f'cat {BAT}/current_now {BAT}/voltage_now {BAT}/status {BAT}/capacity {BAT}/temp; '
                script += 'echo CAPS; '
                script += '; '.join(f'cat {CPU}/{p}/scaling_max_freq {CPU}/{p}/cpuinfo_max_freq' for p in POLICIES)
                script += '; echo CPU; cat /sys/class/thermal/thermal_zone9/temp /sys/class/thermal/thermal_zone10/temp /sys/class/thermal/thermal_zone11/temp'
                if pid:
                    script += f'; echo MEMORY; cat /proc/{pid}/status; echo PSS; cat /proc/{pid}/smaps_rollup'
                text = root(script,'sample')
                head, text = text.split('CAPS\n',1)
                current,voltage,status,capacity,bat_c = head.split()
                caps,text = text.split('CPU\n',1)
                nums = list(map(int,caps.split()))
                if len(nums) != 6 or int(voltage) <= 0:
                    raise RuntimeError('invalid power/cap fields')
                cpu = text.split('MEMORY\n')[0].split()
                row = dict(latest,t_start=began,t=time.monotonic(),current_now_uA=int(current),voltage_now_uV=int(voltage),
                           battery_w=-int(current)*int(voltage)/1e12,battery_status=status,capacity=int(capacity),bat_c=int(bat_c)/10,
                           max=dict(zip(POLICIES,nums[::2])),cpuinfo_max_khz=dict(zip(POLICIES,nums[1::2])),
                           cpu_mc=list(map(int,cpu[:3])),pid=pid)
                mem = Path('/proc/meminfo').read_text()
                row['mem_available_kib'] = int(re.search(r'MemAvailable:\s*(\d+)',mem).group(1))
                if pid:
                    for key,pat in [('vmhwm_kib',r'^VmHWM:\s*(\d+)'),('pss_kib',r'^Pss:\s*(\d+)')]:
                        m = re.search(pat,text,re.M)
                        row[key] = int(m.group(1)) if m else None
                    masks = re.findall(r'^Cpus_allowed_list:\s*(.+)$',text,re.M)
                    row['cores_ok'] = masks == ['4-7']
                if time.monotonic()-row['thermal_at'] > 12:
                    raise RuntimeError('thermal sample stale')
                write(bridge/'latest.json',row)
                log.write(json.dumps(row)+'\n');log.flush()
                if (bridge/'lmk_request.json').exists() and not (bridge/'lmk_done.json').exists():
                    epoch = float(json.loads((bridge/'lmk_request.json').read_text())['epoch'])
                    # Marked copy: coresidency.lmk_lines patterns, with no raw logcat retained (privacy).
                    # Filter before root output hits disk: never save unrelated Android logs or identities.
                    awk = "BEGIN {n=0;k=0} {s=tolower($0)} s ~ /lowmemorykiller|lmkd|killinfo/ {n++; if (s ~ /(lowmemorykiller|lmkd).*kill|killinfo/) k++} END {print n,k}"
                    counts = root(f"set -o pipefail; logcat -d -b all -v epoch -T {epoch:.3f} | awk '{awk}'",'lmk')
                    n,k = map(int,counts.split())
                    write(bridge/'lmk_done.json',dict(ok=True,n_lines=n,n_kills=k))
            except Exception as e:
                write(bridge/'latest.json',dict(t=time.monotonic(),error=str(e)))
                append(bridge/'errors.jsonl',dict(t=time.monotonic(),error=str(e)))
            # Campaign de-phased receipt-time power sampling, same 0.37 s period, no catch-up bursts.
            time.sleep(max(0,.37-(time.monotonic()-began)))


def manifest(selected, t2_draft, texts, images):
    paths = ['chat.py','server_manager.py','benchmark/llm_objective_setting/bench.py',
             'benchmark/gemma_vision/vis0/vis0.py','benchmark/campaign/power.py',
             'benchmark/campaign/runtime.py','benchmark/gemma_vision/vis1/vis1_run.py',
             'benchmark/gemma_vision/vis1/oneshot.sh']
    return dict(blocks=selected,block_s=BLOCK_S,t2_draft=t2_draft,
                commands={b:command(b,t2_draft) for b in selected},
                payload_sha256={n:hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() for n,p in texts+images},
                code_sha256={p:sha(REPO/p) for p in paths})


def completed(out):
    return {d['block'] for p in out.glob('[TV][1-4]_*.json')
            if (d:=json.loads(p.read_text())).get('label') == 'VALID'}


def preflight():
    """Everything oneshot.sh needs true before it idles 5 minutes: ports, files/hashes, agent, plus
    whichever of (Debian imports) / (root, battery, cpuset) matches where this process is running."""
    free_ports()
    matches = agent_matches()
    if matches:
        log_agent_matches(matches)
        raise SystemExit('agent found; close Claude/AGY/Codex (see pid/comm/cmdline above)')
    missing = [str(p) for p in (*EXPECTED_HASHES, *BINS.values()) if not p.is_file()]
    if missing:
        raise SystemExit('missing file(s), fetch/build first: ' + ', '.join(missing))
    bad = [str(p) for p, h in EXPECTED_HASHES.items() if sha(p) != h]
    if bad:
        raise SystemExit('SHA-256 mismatch, re-fetch/rebuild: ' + ', '.join(bad))
    if Path('/termux-home').is_dir():
        missing_mods = [m for m in DEBIAN_IMPORTS if importlib.util.find_spec(m) is None]
        if missing_mods:
            raise SystemExit('Debian python3 is missing ' + ', '.join(missing_mods) +
                              ' — run: apt-get install python3-requests python3-pil')
    else:
        if subprocess.run(['su', '-c', 'id'], capture_output=True, timeout=10).returncode != 0:
            raise SystemExit('no root (su) access; telemetry needs root')
        cap, status = root('cat /sys/class/power_supply/battery/capacity'
                            ' /sys/class/power_supply/battery/status', 'preflight_battery').split()[:2]
        if not (cap.isdigit() and 50 <= int(cap) <= 80 and status == 'Discharging'):
            raise SystemExit('battery outside 50-80% Discharging band; unplug and bring it into range')
        before = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {4, 5, 6, 7})
        stuck = os.sched_getaffinity(0) == {4, 5, 6, 7}
        os.sched_setaffinity(0, before)
        if not stuck:
            raise SystemExit('Termux cpuset blocks cores 4-7; bring Termux to the foreground, screen on')


def smoke_block(block, out, bridge, t2_draft, allow_agent, payloads):
    """--smoke: one request per block, own server up and down, no 180 s loop or gate wait beyond
    the single telemetry reading already needed to launch. Always labelled SMOKE — NOT VALID TIMING."""
    record = dict(block=block, label=SMOKE_LABEL, command=command(block, t2_draft),
                  started_epoch=time.time(), server_started=False, reply_received=False,
                  exit_clean=None, image_tokens=None, reason=None, allow_agent=allow_agent)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logpath = out / f'{block}_{attempt}.server.log'
    proc = pid = None
    try:
        record['reading'] = telemetry(bridge, allow_agent=allow_agent)
        free_ports()
        if not allow_agent:
            matches = agent_matches()
            if matches:
                log_agent_matches(matches)
                raise RuntimeError('agent found before launch')
        with logpath.open('w') as log:
            began = time.monotonic()
            proc = spawn(record['command'], log)
            pid = None
            while pid is None:
                if proc.poll() is not None or time.monotonic() - began > 10:
                    raise RuntimeError('server crashed or owned child not found')
                try:
                    pid = server_pid(proc)
                except (OSError, RuntimeError):
                    time.sleep(.05)
            record['server_started'] = True
            write(bridge / 'target.json', dict(pid=pid))
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('server crashed during load')
                try:
                    if post('/health', timeout=1).get('status') == 'ok':
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() - began >= 180:
                    raise RuntimeError('server load timeout')
                time.sleep(.2)
            check_cores(pid)
            endpoint = '/completion' if block.startswith('T') else '/v1/chat/completions'
            name, body = payloads[0]
            offset = logpath.stat().st_size
            t0 = time.monotonic()
            data = post(endpoint, body, timeout=180)  # same bound as the real warm-up; V4 (BF16) took 55-72 s
            end = time.monotonic()
            with logpath.open() as f:
                f.seek(offset)
                delta = f.read()
            row = request_record(name, data, t0, end, delta, block)
            record['reply_received'] = True
            record['image_tokens'] = row.get('image_tokens')
            record['requests'] = [row]
    except Exception as e:
        record['reason'] = str(e)
    finally:
        write(bridge / 'target.json', dict(pid=None))
        if proc is not None:
            # One TERM to the server lets it exit 0; clean = exit 0 and nothing left in our group.
            record['server_exit'] = own_stop(proc, pid)
            record['exit_clean'] = record['server_exit'] == 0 and not group_alive(proc.pid)
        write(out / f'{block}_{attempt}.json', record)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only')
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--bridge',type=Path)
    ap.add_argument('--preflight',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--monitor',type=Path,help=argparse.SUPPRESS)
    ap.add_argument('--smoke',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--allow-agent',action='store_true',help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.allow_agent and not a.smoke:
        raise SystemExit('--allow-agent is only valid with --smoke; the real run never accepts it')
    if a.preflight:
        preflight()
        return
    if a.monitor:
        return monitor(a.monitor)
    selected = blocks(a.only)
    if a.dry_run:
        print(DRY_LABEL)
        print(json.dumps(dict(block_order=selected,commands={b:command(b) for b in selected},
                             text_example=text_payload('Hi'),vision_example=vision_payload(b'offline-fake-jpeg','- chair [0], center, near')),
                         indent=2))
        return
    if a.bridge is None or not Path('/termux-home').is_dir():
        raise SystemExit('launch only through native oneshot.sh into Debian')
    bridge = a.bridge.resolve()
    if not (bridge/'owner_launch').is_file():
        raise SystemExit('missing owner wrapper receipt')
    if not (a.smoke and a.allow_agent) and agent_found(log=True):
        raise SystemExit('agent found; no server started')
    free_ports()
    telemetry(bridge, allow_agent=a.smoke and a.allow_agent)
    os.sched_setaffinity(0,{4,5,6,7})
    check_cores(os.getpid())
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,interrupted)
    # Help is inspected only by the owner runner; no server/model is loaded by the preparer.
    help_result = subprocess.run(['timeout','10',str(BINS['vis']),'--help'],capture_output=True,text=True,timeout=15)
    if help_result.returncode:
        raise SystemExit('b11716 --help failed')
    help_text = help_result.stdout+help_result.stderr
    t2_draft = all(s in help_text for s in ('--model-draft','--spec-type','--spec-draft-n-max','draft-mtp'))
    texts,images = inputs()
    if a.smoke:
        out = HERE/'runs'/('smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        out.mkdir(parents=True)
        write(out/'smoke.json',dict(label=SMOKE_LABEL,allow_agent=a.allow_agent,t2_draft=t2_draft,
             base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
        print('RUN_FOLDER='+str(out),flush=True)
        for b in selected:
            result = smoke_block(b,out,bridge,t2_draft,a.allow_agent,texts if b.startswith('T') else images)
            print(b+': SMOKE server_started='+str(result['server_started'])+
                  ' reply_received='+str(result['reply_received'])+' image_tokens='+str(result['image_tokens'])+
                  ' exit_clean='+str(result['exit_clean'])+
                  ('' if not result['reason'] else ' reason='+result['reason']),flush=True)
        print('Smoke finished ('+SMOKE_LABEL+'). Send the RUN_FOLDER path and console log.',flush=True)
        return
    expected = EXPECTED_HASHES
    hashes = {str(p):sha(p) for p in expected}
    if any(hashes[str(p)] != h for p,h in expected.items()):
        raise SystemExit('model/projector/drafter SHA-256 mismatch')
    spec = manifest(selected,t2_draft,texts,images)
    if a.resume:
        out = a.resume.resolve()
        if not out.is_relative_to(HERE/'runs'):
            raise SystemExit('resume folder must be under vis1/runs')
        prior = json.loads((out/'run.json').read_text())
        if prior['manifest'] != spec:
            raise SystemExit('resume manifest mismatch (code, payloads, flags or block selection)')
        idle = prior['idle_skin']
    else:
        out = HERE/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out.mkdir(parents=True)
        idle = json.loads((bridge/'idle.json').read_text())['skin']
        write(out/'run.json',dict(manifest=spec,idle_skin=idle,weight_sha256=hashes,
             base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
             binaries={str(p):sha(p) for p in BINS.values()},
             note='180 s short blocks; sustained thermal equilibrium is not established (#93). T1 reference #137: 19.18 tok/s, 39.0% acceptance; pre TPL-FIX.'))
    print('RUN_FOLDER='+str(out),flush=True)
    print('T2 drafter: '+('enabled' if t2_draft else 'not supported by b11716 help; running without'),flush=True)
    for b in selected:
        if b in completed(out):
            print('resume skips '+b,flush=True)
            continue
        result = run_block(b,out,bridge,idle,texts if b.startswith('T') else images,t2_draft)
        print(b+': '+result['label']+' — '+result['reason'],flush=True)
        if result['label'] != 'VALID':
            raise SystemExit('stopped; inspect block record before --resume')
    print('Finished. Send the RUN_FOLDER path and console log; do not rerun for Phase 2.',flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        for proc in OWNED:
            own_stop(proc)
