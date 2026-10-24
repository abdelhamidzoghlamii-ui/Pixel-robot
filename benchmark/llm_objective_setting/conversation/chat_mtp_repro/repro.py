#!/data/data/com.termux/files/usr/bin/python
"""VIS-3C experiment 1: the Oct 14 chat_mtp workload (run chat_mtp_20261014T224541Z), one config per launch.
Research only: no motors, no camera, no main.py. Native Termux only; started by oneshot.sh in this folder.

The block is chat_mtp.run_block unchanged (5 s coresidency sampler, 175 s watchdog, TERM then KILL, cores 4-7 checks).
Only its command and payload are replaced, by the Oct 14 values frozen below: chat.py has changed since then (TPL-FIX
template, E4B entry), so neither is taken from it. Before starting, the frozen values are compared with the archived
run.json and the harness files with their pinned SHA-256.

Not as on Oct 14: one config per launch (no OFF -> ON sequence, no idle baseline, so no skin cooling gate: GATE_MAX_S
= 0); one block only, a run not finished in it is INCOMPLETE (no resume, no repeat). See README.md.

Usage: repro.py --config OFF|ON [--thermal-log PATH]
       repro.py --report OFF_DIR ON_DIR      (report only; reads the folders, starts nothing)
Writes ~/robot_runs/chat_mtp_repro/<UTC>_<config>/: run.json, prompts.jsonl, server_block00_<config>.log,
result.json, report.txt.
"""
import argparse
import json
import signal
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CHAT_MTP = HERE.parent / 'chat_mtp'
sys.path.insert(0, str(CHAT_MTP))
import chat_mtp as cm  # noqa: E402  block machinery; command and payload replaced below

cr, pm, bench = cm.cr, cm.pm, cm.bench
ARCHIVE = CHAT_MTP / 'runs' / 'chat_mtp_20261014T224541Z'
ARCHIVE_RUN_SHA256 = '8cc59ac3bb8ab009d4e69e24cf2783ff3fa9a50772dd985acbe2e6d7834cb012'  # chat_mtp/runs/ARTIFACTS.md
HOME = '/data/data/com.termux/files/home'
SERVER = f'{HOME}/llama.cpp-b1609-dotprod/build/bin/llama-server'
MODEL = f'{HOME}/models/gemma-4-E2B-it-Q4_0.gguf'
DRAFT = f'{HOME}/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf'
# chat.server_cmd and server_manager.MTP_ARGS at 2091f42 = the archived run.json 'commands'
OFF = ['taskset', '-c', '4-7', SERVER, '-m', MODEL, '--port', '8080', '--ctx-size', '4096', '--host', '127.0.0.1',
       '--threads', '4', '--threads-batch', '4', '--parallel', '1', '--swa-full', '--cache-ram', '0']
COMMANDS = {'OFF': OFF, 'ON': OFF + ['--model-draft', DRAFT, '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']}
LIBGGML_CPU = f'{HOME}/llama.cpp-b1609-dotprod/build/bin/libggml-cpu.so'
SHA256 = {MODEL: '8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52',  # Oct 14 run.json
          DRAFT: '9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919',
          LIBGGML_CPU: 'a4ef0d97e741f5902919af5e9cef95f808bd98581be76bca66fcee3f4ac8a0dd'}
# Imported harness files. coresidency, power_map and bench are byte-identical to Oct 14; chat_mtp.py gained the E4B
# option and the report-header comma after 2091f42, neither used by run_block for E2B.
CODE_SHA256 = {CHAT_MTP / 'chat_mtp.py': 'b2c370e23e5f7fa93996a28a9aaed2f3f5ee546f2ef70f94eaebe4b34a29cd24',
               Path(cr.__file__): 'a6b0d65454c0c060927f744194a8cbb9f2a419a59e39fc82b0a510b9f92ed177',
               Path(pm.__file__): '4756ce7dc7ba4e358578288c0d2509345fc4e2fe02d45f077f65355792e29539',
               Path(bench.__file__): '0792beeed813d94df2a1d9ce7bffe2f5e2a00a564de4a32aaecfa0182b8e6f8f'}
OUT_ROOT = cr.HOME / 'robot_runs' / 'chat_mtp_repro'
THERMAL_LOG = OUT_ROOT / 'thermal.log'  # oneshot.sh's logger
BATTERY_MAX_C = 45  # refused at or above (VIS-3B's stop value)
POLICIES = ('policy0', 'policy4', 'policy6')
SKIN_MATCH_C, BATTERY_MATCH_PCT = 1.0, 10  # --report: matched start conditions (VIS-3B's thresholds)


def command(config, model='e2b'):
    return list(COMMANDS[config])


def payload(text):
    """chat.chat_payload(chat.build_prompt('gemma', [one user message]), 'gemma') at 2091f42."""
    return {'prompt': f'<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n', 'n_predict': 512,
            'temperature': 0.7, 'stop': ['<end_of_turn>'], 'stream': False}


cm.command, cm.payload = command, payload  # run_block looks both up in chat_mtp at call time


def frozen_problems():
    """Differences between this file's frozen workload or pinned code and the Oct 14 archive."""
    why = []
    if cr.sha256(ARCHIVE / 'run.json') != ARCHIVE_RUN_SHA256:
        return [f'{ARCHIVE}/run.json is not the archived file']
    run = json.loads((ARCHIVE / 'run.json').read_text())
    texts = cm.prompt_texts()
    checks = {'ids': (cm.IDS, run['ids']), 'commands': (COMMANDS, run['commands']),
              'payload': (payload('<text>'), run['payload_example']), 'prompt texts': (texts, run['prompt_texts']),
              'prompt SHA-256': ({i: cm.sha256_text(t) for i, t in texts.items()}, run['prompt_text_sha256']),
              'model SHA-256': ({MODEL: SHA256[MODEL], DRAFT: SHA256[DRAFT]}, run['model_sha256']),
              'libggml-cpu SHA-256': (SHA256[LIBGGML_CPU], run['libggml_cpu_sha256']),
              'bench SHA-256': (bench.SOURCE_SHA256, run['bench_sha256']),
              'block constants': ((cm.BLOCK_S, cm.DEADLINE_S, cm.TERM_WAIT_S, cm.LAST_START_S),
                                  (run['block_s'], run['deadline_s'], run['term_wait_s'], run['last_start_s']))}
    why += [f'{k} differ from the Oct 14 run.json' for k, (mine, theirs) in checks.items() if mine != theirs]
    why += [f'{p} SHA-256 is not the pinned {h[:12]}' for p, h in CODE_SHA256.items() if cr.sha256(p) != h]
    return why


def check_hashes():
    got = {p: cr.sha256(p) if Path(p).is_file() else 'missing' for p in SHA256}
    bad = [f'{p}: {got[p]} != {want}' for p, want in SHA256.items() if got[p] != want]
    if bad:
        raise SystemExit('SHA-256 mismatch: ' + '; '.join(bad))
    return got


def build_sha256():
    """Every file in the build's bin folder (only libggml-cpu.so was hashed on Oct 14)."""
    return {p.name: cr.sha256(p) for p in sorted(Path(SERVER).parent.iterdir()) if p.is_file() and not p.is_symlink()}


def cpufreq():
    """Caps and time_in_state (kHz -> 10 ms units; kHz kept as a string, as JSON keys are) per policy; no su."""
    out = {}
    for p in POLICIES:
        d = Path('/sys/devices/system/cpu/cpufreq') / p
        try:
            out[p] = {'max': int((d / 'scaling_max_freq').read_text()), 'hw_max': int((d / 'cpuinfo_max_freq').read_text()),
                      'time_in_state': {f: int(t) for f, t in (l.split() for l in (d / 'stats/time_in_state').open())}}
        except (OSError, ValueError) as e:
            out[p] = {'error': str(e)}
    return out


def conditions(thermal_log):
    """Battery, Android thermal status, skin, CPU-zone log line and cpufreq; read outside the measurement."""
    c = {'utc': cr.utc(), 'cpufreq': cpufreq()}
    rc, out = cr.root('cat /sys/class/power_supply/battery/capacity /sys/class/power_supply/battery/status '
                      '/sys/class/power_supply/battery/temp', 'battery')
    try:
        level, status, temp = out.split()[:3]
        c.update(battery_pct=int(level), battery_status=status, battery_c=int(temp) / 10)
    except ValueError:
        c['battery_error'] = f'su rc {rc}: {out[:100]!r}'
    d = cr.read_dump()
    c.update(skin=d.get('skin'), status=d.get('status'))
    try:
        c.update(cr.read_thermal(thermal_log))
    except (SystemExit, OSError, ValueError, IndexError, KeyError) as e:
        c['thermal_error'] = str(e)
    return c


def thermal_problems(c):
    why = []
    if c.get('status') != 0:
        why.append(f'Android thermal status {c.get("status")}, not 0')
    if c.get('skin') is None:
        why.append('no VIRTUAL-SKIN reading')
    if c.get('battery_c') is None or c['battery_c'] >= BATTERY_MAX_C:
        why.append(f'battery {c.get("battery_c")} degC (refused at {BATTERY_MAX_C} or above, or unreadable)')
    if 'thermal_error' in c:
        why.append(c['thermal_error'])
    return why


def agents():
    return subprocess.run(['pgrep', '-fa', cm.AGENTS], capture_output=True, text=True).stdout.strip()


def port_free():
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex(('127.0.0.1', 8080)) != 0


# ---------------------------------------------------------------- report

def top_share(start, end, policy):
    """Fraction of the block's time_in_state delta spent at the policy's hardware maximum."""
    a, b = start['cpufreq'].get(policy, {}), end['cpufreq'].get(policy, {})
    if 'time_in_state' not in a or 'time_in_state' not in b:
        return None
    delta = {f: b['time_in_state'][f] - a['time_in_state'].get(f, 0) for f in b['time_in_state']}
    total = sum(delta.values())
    return delta.get(str(b['hw_max']), 0) / total if total else None


def med(xs):
    return statistics.median(xs) if xs else None


def summary(rows, block):
    """Server-timing figures of one config: Oct 14 aggregation plus pooled and per-pass views."""
    n = sum(r['predicted_n'] or 0 for r in rows)
    ms = sum(r['predicted_ms'] or 0 for r in rows)
    passes = sum((r['predicted_n'] or 0) - (r['draft_n_accepted'] or 0) for r in rows)
    return {'prompts': len(rows), 'tok_s': cm.tok_s(rows), 'pooled_tok_s': n / ms * 1000 if ms else None,
            'accept': cm.acceptance(rows), 'gen_tokens': n, 'prompt_tokens': sum(r['prompt_n'] or 0 for r in rows),
            'drafted': sum(r['draft_n'] or 0 for r in rows), 'decode_ms': ms, 'ms_per_pass': ms / passes if passes else None,
            'prompt_s': med([r['prompt_s'] for r in rows if r['prompt_s'] is not None]),
            'load_s': block['load_s'], 'requests_s': block['requests_s'], 'server_s': block['server_s'],
            'samples': len(block['samples']), 'root_s': med([s['root_s'] for s in block['samples'] if 'root_s' in s]),
            'mean_w': block['mean_w'], 'end_reason': block['end_reason']}


def load(folder):
    """(run, rows, result, problems) of a repro folder; problems make it not VALID."""
    folder = Path(folder)
    run = json.loads((folder / 'run.json').read_text())
    rows = cm.read_jsonl(folder / 'prompts.jsonl')
    if not (folder / 'result.json').exists():
        return run, rows, None, ['INCOMPLETE: no result.json (interrupted or failed)']
    res = json.loads((folder / 'result.json').read_text())
    b, why = res['block'], []
    if [r['prompt_id'] for r in rows] != cm.IDS:
        why.append(f'{len(rows)}/{len(cm.IDS)} prompts')
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
    if any(r['config'] != run['config'] for r in rows):
        why.append('prompt rows of another config')
    return run, rows, res, why


def report(folders):
    """Text report of repro folders (any order), plus the Oct 14 archive rows computed the same way."""
    f = lambda v, spec: format(v, spec) if v is not None else 'n/a'
    L = ['CHAT MTP REPRO REPORT  (Oct 14 workload: E2B Q4_0, ctx 4096, temperature 0.7, n_predict 512, cores 4-7)',
         'gen t/s = token-weighted over replies >= 10 tokens (Oct 14 rule); pooled = all tokens; accept = accepted /',
         'drafted; ms/pass = decode ms / (generated - accepted); prompt s = median server prompt_ms; times from spawn', '',
         f"{'run':<28}{'cfg':<4}{'n':>3}{'gen t/s':>8}{'pooled':>7}{'accept':>7}{'gen tok':>8}{'ms/pass':>8}"
         f"{'prompt s':>9}{'load s':>7}{'req s':>6}{'skin':>7}{'st':>3}{'bat%':>5}{'X1 cap':>7}{'X1 top':>7}"
         f"{'samples':>8}{'root s':>7}  label"]
    runs, incomplete = [], []
    arch = json.loads((ARCHIVE / 'run.json').read_text())
    arch_rows, arch_blocks = cm.read_jsonl(ARCHIVE / 'prompts.jsonl'), cm.read_jsonl(ARCHIVE / 'blocks.jsonl')
    for b in arch_blocks:
        rows = [r for r in arch_rows if r['config'] == b['config']]
        runs.append(('Oct14 ' + ARCHIVE.name[9:], b['config'], summary(rows, b), b['gate'], None, None, [],
                     arch['code_sha256'], 'ARCHIVE (VALID, #137)'))
    for folder in folders:
        run, rows, res, why = load(folder)
        if res is None:
            incomplete.append(f'{Path(folder).name:<28}{run["config"]:<4} {why[0]}')
            continue
        start, end = res['start'], res['end']
        runs.append((Path(folder).name, run['config'], summary(rows, res['block']), res['block']['gate'], start, end,
                     why, run, 'VALID' if not why else 'NOT VALID: ' + '; '.join(why)))
    for name, cfg, s, gate, start, end, why, _, label in runs:
        cap = start and start['cpufreq'].get('policy6', {}).get('max')
        L.append(f"{name:<28}{cfg:<4}{s['prompts']:>3}{f(s['tok_s'], '.2f'):>8}{f(s['pooled_tok_s'], '.2f'):>7}"
                 f"{f(s['accept'], '.1%'):>7}{s['gen_tokens']:>8}{f(s['ms_per_pass'], '.1f'):>8}{f(s['prompt_s'], '.2f'):>9}"
                 f"{s['load_s']:>7.1f}{s['requests_s']:>6.0f}{f(gate['skin'], '.2f'):>7}{f(gate['status'], 'd'):>3}"
                 f"{f(start and start.get('battery_pct'), 'd'):>5}{f(cap and cap // 1000, 'd'):>7}"
                 f"{f(start and top_share(start, end, 'policy6'), '.0%'):>7}{s['samples']:>8}{f(s['root_s'], '.2f'):>7}  {label}")
    L += incomplete + ['', 'server timings and end: prompt tokens, drafted, decode ms, server exit s (from spawn), W mean,'
                       ' skin/status after the stop, end reason']
    for name, cfg, s, gate, start, end, *_ in runs:
        e = (end or {}) if start else next(b['end'] for b in arch_blocks if b['config'] == cfg)
        L.append(f"  {name:<28}{cfg:<4}{s['prompt_tokens']:>5}{s['drafted']:>6}{s['decode_ms']:>10.0f}{s['server_s']:>6.0f}"
                 f"{f(s['mean_w'], '.2f'):>6}  {e.get('skin')}/{e.get('status')}  {s['end_reason']}")
    L += ['', 'skin/st = VIRTUAL-SKIN degC and Android status right before the spawn; X1 cap = policy6 scaling_max_freq MHz',
          'at start; X1 top = share of the block (start to end reading) at the X1 hardware maximum; samples = 5 s root',
          'samples in the block, root s = their median su duration. Oct 14 recorded no cpufreq or battery %.']
    mine = [r for r in runs if r[4] is not None]
    by = {r[1]: r for r in mine}
    if len(mine) == 2 and set(by) == {'OFF', 'ON'}:
        off, on = by['OFF'], by['ON']
        L += ['', f"ON/OFF gen t/s: {on[2]['tok_s'] / off[2]['tok_s']:.3f} (Oct 14: 19.18 / 13.37 = 1.435)"
              if on[2]['tok_s'] and off[2]['tok_s'] else 'ON/OFF gen t/s: n/a']
        L.append('configuration: ' + ('SAME' if not (d := config_diffs(off[7], on[7])) else 'DIFFERENT: ' + '; '.join(d)))
        L.append('conditions: ' + ('MATCHED' if not (d := condition_diffs(off, on)) else 'UNMATCHED: ' + '; '.join(d)))
    elif folders:
        L += ['', 'comparison: needs exactly one OFF and one ON folder']
    return '\n'.join(L)


def config_diffs(a, b):
    """run.json manifests may differ only in config, command (by the drafter flags) and run-specific fields."""
    why = []
    if b['command'] != a['command'] + COMMANDS['ON'][len(OFF):]:
        why.append('commands differ beyond the drafter flags')
    for k in ('payload_example', 'prompt_texts', 'model_sha256', 'build_sha256', 'code_sha256', 'ids'):
        if a[k] != b[k]:
            why.append(f'{k} differs')
    return why


def condition_diffs(off, on):
    why = []
    (s0, g0), (s1, g1) = (off[4], off[3]), (on[4], on[3])
    if g0['skin'] is None or g1['skin'] is None or abs(g0['skin'] - g1['skin']) > SKIN_MATCH_C:
        why.append(f'start skin {g0["skin"]} vs {g1["skin"]} (> {SKIN_MATCH_C} degC apart)')
    if (g0['status'], g1['status']) != (0, 0):
        why.append(f'start status {g0["status"]} / {g1["status"]}')
    if s0.get('battery_pct') is None or s1.get('battery_pct') is None or \
            abs(s0['battery_pct'] - s1['battery_pct']) > BATTERY_MATCH_PCT:
        why.append(f'battery {s0.get("battery_pct")} vs {s1.get("battery_pct")} % (> {BATTERY_MATCH_PCT} apart)')
    caps = lambda s: {p: s['cpufreq'][p].get('max') for p in POLICIES}
    if caps(s0) != caps(s1):
        why.append(f'CPU caps at start {caps(s0)} vs {caps(s1)}')
    return why


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', choices=('OFF', 'ON'))
    ap.add_argument('--report', nargs=2, type=Path, metavar=('OFF_DIR', 'ON_DIR'))
    ap.add_argument('--thermal-log', default=str(THERMAL_LOG))
    a = ap.parse_args(argv)
    if bool(a.config) == bool(a.report):
        ap.error('give exactly one of --config and --report')
    if a.report:
        print(report(a.report))
        return
    cr.require_native()  # before any root command or server
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cr.exit_on_signal)
    try:
        why = cm.refusals() + frozen_problems()
        if why:
            raise SystemExit('not started: ' + '; '.join(why))
        out = OUT_ROOT / f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{a.config}'
        out.mkdir(parents=True)
        print(f'checking model, drafter and build SHA-256 (about 30 s); run folder {out}', flush=True)
        hashes, build = check_hashes(), build_sha256()
        cr.wait_cores('start')
        start = conditions(a.thermal_log)
        if why := thermal_problems(start):
            raise SystemExit('not started: ' + '; '.join(why))
        texts = cm.prompt_texts()
        (out / 'run.json').write_text(json.dumps({
            'config': a.config, 'ids': cm.IDS, 'command': command(a.config), 'payload_example': payload('<text>'),
            'prompt_texts': texts, 'started': cr.utc(), 'model_sha256': {p: hashes[p] for p in (MODEL, DRAFT)},
            'build_sha256': build, 'archive': str(ARCHIVE), 'archive_run_sha256': ARCHIVE_RUN_SHA256,
            'code_sha256': {str(p): cr.sha256(p) for p in (Path(__file__), *CODE_SHA256)},
            'imported_unused_sha256': {str(p): cr.sha256(p) for p in (Path(cm.chat.__file__), Path(cm.sm.__file__))},
            'block_s': cm.BLOCK_S, 'deadline_s': cm.DEADLINE_S, 'gate_max_s': 0, 'sample_s': cr.SAMPLE_S},
            indent=1) + '\n')
        pm.GATE_MAX_S = 0  # no idle baseline in a single-config launch: one skin reading, no wait
        rows = []
        with open(out / 'prompts.jsonl', 'a') as plog:
            def emit(row):
                plog.write(json.dumps(row) + '\n')
                plog.flush()
                rows.append(row)
            ctx = {'out': out, 'thermal_log': a.thermal_log, 'idle': {'skin': start['skin']}, 'emit': emit}
            try:
                block = cm.run_block(0, a.config, list(cm.IDS), texts, ctx)
            except cr.CoresLost as e:
                raise SystemExit(f'not started: {e}; nothing measured, relaunch')
        result = {'block': block, 'start': start, 'end': conditions(a.thermal_log), 'agents_end': agents(),
                  'port_free_end': port_free()}
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
