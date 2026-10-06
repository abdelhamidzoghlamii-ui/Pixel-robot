#!/usr/bin/env python3
"""Offline test of coresidency.py: fake RobotCam frames, fake llama-server, fake root output, fake detector.
No camera, no root, no model, no motors. Needs a Python with Pillow, numpy and onnxruntime (native Termux's;
it also runs inside Debian/proot through /data/data/com.termux/files/usr/bin/python).

  python test_coresidency.py      full smoke run, then SIGTERM during B3 plus --resume
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 18080
POLICIES = {'policy0': 1803000, 'policy4': 2348000, 'policy6': 2850000}


def dump_text(status=0, skin=35.0):
    """`dumpsys thermalservice` as the phone prints it (thermal_char's test form, shortened)."""
    hal = ''.join(f'\tTemperature{{mValue={v}, mType={t}, mName={n}, mStatus=0}}\n'
                  for n, t, v in (('BIG', 0, 60.0), ('VIRTUAL-SKIN-CPU', -1, 36.0), ('VIRTUAL-SKIN', 3, skin)) if v is not None)
    return (f'IsStatusOverride: false\nThermal Status: {status}\nCached temperatures:\n'
            '\tTemperature{mValue=40.0, mType=3, mName=VIRTUAL-SKIN, mStatus=1}\n'
            f'HAL Ready: true\nCurrent temperatures from HAL:\n{hal}'
            'Current cooling devices from HAL:\n\tCoolingDevice{mValue=0, mType=2, mName=thermal-cpufreq-2}\n')


def make_sysfs(root_dir, cr):
    """Fake su on PATH and a fake sysfs tree for the persistent root shell (zone 3 is a decoy: type CPU3)."""
    b, th, cf, bat = (root_dir / n for n in ('bin', 'sys/thermal', 'sys/cpufreq', 'sys/battery'))
    b.mkdir(parents=True, exist_ok=True)
    (b / 'su').write_text('#!/bin/bash\nif [ "$1" = -c ]; then exec bash -c "$2"; fi\nexec bash\n')
    (b / 'su').chmod(0o755)
    os.environ['PATH'] = f'{b}:' + os.environ['PATH']
    for i, typ in (('3', 'CPU3'), ('9', 'BIG'), ('10', 'MID'), ('11', 'LITTLE'), ('12', 'BIG')):
        (th / f'thermal_zone{i}').mkdir(parents=True, exist_ok=True)
        (th / f'thermal_zone{i}/type').write_text(typ + '\n')
        (th / f'thermal_zone{i}/temp').write_text('60000\n')
    for pol, mx in POLICIES.items():
        (cf / pol).mkdir(parents=True, exist_ok=True)
        (cf / pol / 'cpuinfo_max_freq').write_text(f'{mx}\n')
        (cf / pol / 'scaling_max_freq').write_text(f'{mx}\n')
    bat.mkdir(parents=True, exist_ok=True)
    (bat / 'temp').write_text('300\n')
    cr.THERMAL, cr.CPUFREQ, cr.BATTERY = str(th), str(cf), str(bat)
    return th, cf, bat


def fake_server(port, pid_file, *extra):
    Path(pid_file).write_text(f'{os.getpid()} {" ".join(extra)}\n')
    ready = time.monotonic() + 0.3

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health' and time.monotonic() > ready:
                return self.reply({'status': 'ok'})
            self.send_response(503)
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                return self.reply({'tokens': ([2] if body.get('add_special') else []) + [ord(c) for c in body['content']]})
            time.sleep(0.2)
            if 'n_probs' in body:
                top = [{'id': ord(L), 'logprob': -0.5 * i} for i, L in enumerate('BACDEFGHIJKL')]
                return self.reply({'completion_probabilities': [{'top_logprobs': top}]})
            return self.reply({'content': 'fake reply', 'timings': {'predicted_n': 3, 'draft_n': 3, 'draft_n_accepted': 1}})

    ThreadingHTTPServer(('127.0.0.1', int(port)), H).serve_forever()


def child(root_dir, runner_args):
    """Runs coresidency.main with every phone dependency faked."""
    from PIL import Image
    sys.path.insert(0, str(HERE))
    import coresidency as cr
    root_dir = Path(root_dir)
    frames = root_dir / 'frames'
    frames.mkdir(exist_ok=True)
    (root_dir / 'downloads').mkdir(exist_ok=True)
    gguf = root_dir / 'fake.gguf'
    gguf.write_bytes(b'x' * 4096)
    cr.require_native = lambda: None
    cr.OUT_ROOT, cr.DOWNLOADS, cr.FRAME_DIR = root_dir, root_dir / 'downloads', str(frames)
    cr.MODEL = cr.DRAFT = str(gguf)
    cr.PORT = PORT
    cr.SMOKE_S, cr.SAMPLE_S, cr.SELECTOR_S, cr.DUMP_S = 8, 2, 3, 1
    th, cf, bat = make_sysfs(root_dir, cr)
    cr.allowed_cpus = lambda: set(range(8))
    cr.server_cmd = lambda extra=(): [sys.executable, __file__, '--fake-server', str(PORT),
                                      str(root_dir / 'server.pid'), *extra]
    log = lambda name, text: open(root_dir / name, 'a').write(f'{time.monotonic():.3f} {text}\n')

    class FakeDetector:
        def detect(self, frame, size=320):
            assert frame.size == (640, 480) and size in (320, 640)
            time.sleep(0.01 if size == 320 else 0.03)
            return [{'class_name': 'person'}]
    cr.Detector = FakeDetector

    cam = {'run': None, 'cached': False}

    def writer(stop, session):
        n = 0
        while not stop.is_set():
            n += 1
            boot_ms = int(time.clock_gettime(time.CLOCK_BOOTTIME) * 1000)
            comment = f'robotcam session={session} frame={n} capture_boot_ms={boot_ms} capture_wall_ms=0 clock=sensor'
            Image.new('RGB', (640, 480), (n % 255, 0, 0)).save(frames / 'tmp.jpg', quality=85, comment=comment.encode())
            os.replace(frames / 'tmp.jpg', frames / 'frame.jpg')
            stop.wait(0.5)

    def fake_am(*args):
        log('am.log', ' '.join(args))
        if os.environ.get('FAKE_STOP_FAIL') and args[0] == 'broadcast' and cam['run']:  # B1's STOP broadcast fails
            raise subprocess.CalledProcessError(1, ['am', *args])
        if cam['run']:
            cam['run'].set()
            cam['run'] = None  # the process stays cached after STOP, as on the phone, until the force-stop
        if args[0] == 'start':
            cam['run'], cam['cached'] = threading.Event(), True
            threading.Thread(target=writer, args=(cam['run'], f'{int(time.time() * 1000) % 0xffffff:x}'),
                             daemon=True).start()
    cr.am = fake_am

    def fake_root(cmd, tag, timeout=60):
        log('root.log', f'{tag}: {cmd[:80]}')
        if tag == 'sample':
            pids = re.findall(r'=== (\S+) (\d+)"', cmd)
            out = '-400000\n4100000\nDischarging\n'
            for name, pid in pids:
                out += f'=== {name} {pid}\n** MEMINFO in pid {pid} **\n        TOTAL PSS:   123,456  TOTAL RSS: 1\n'
            if cam['run']:
                out += '=== robotcam_app 4321\n                TOTAL    38,000    1\n'
            out += '=== camera_provider 777\n        TOTAL PSS:   270,000\n'
            return 0, out
        if tag == 'pidof':
            if 'pidof_rc' in cmd:  # camera_end_check keeps pidof's own status
                return 0, ('4321\npidof_rc=0\n' if cam['cached'] else 'pidof_rc=1\n')
            return (0, '4321\n') if cam['cached'] else (1, '')
        if tag == 'forcestop':
            assert cmd == 'am force-stop com.pixelrobot.robotcam', cmd
            if cam['run']:  # force-stop also ends a capture that STOP did not end
                cam['run'].set()
                cam['run'] = None
            cam['cached'] = False
            return 0, ''
        if tag == 'thermalservice':
            assert cmd == 'dumpsys thermalservice', cmd
            hot = hot_since['t'] is not None and time.monotonic() > hot_since['t'] + 4
            return 0, dump_text(status=4 if hot else 3, skin=46.6 if hot else 35.0)
        if tag == 'lmk':
            return 0, ('logcat_rc=0\n--------- beginning of main\n\n=== lmk lines\n1727650000.123  123  456 I lowmemorykiller: Kill \'com.example\' (999), uid 10123\n'
                       '1727650001.000  123  456 I lowmemorykiller: psi threshold reached\n'
                       "1727650002.000  123  456 I lmkd    : Kill 'com.other' (998), uid 10124\n")
        return 0, 'uid=0(root)\n'
    cr.root = fake_root

    # B4 reaches Android status CRITICAL (4) 4 s after it is entered, policy6 capped from 2 s in; status 3
    # (SEVERE) everywhere else is no stop; z9 stays at 90 degC (no longer a stop)
    hot_since = {'t': None}
    real_run_block = cr.run_block

    def run_block(name, *a):
        hot_since['t'] = time.monotonic() if name == 'B4_only640_gemma' else None
        try:
            return real_run_block(name, *a)
        finally:
            hot_since['t'] = None
    cr.run_block = run_block

    def thermal():
        while True:
            stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            hot = hot_since['t'] is not None and time.monotonic() > hot_since['t'] + 2
            (cf / 'policy6/tmp').write_text(f'{2400000 if hot else 2850000}\n')
            os.replace(cf / 'policy6/tmp', cf / 'policy6/scaling_max_freq')  # atomic: the sysfs value is never empty
            with open(root_dir / 'thermal.log', 'a') as f:
                f.write(f'{stamp} z9=90000 z10=35000 z11=34000\n')
            time.sleep(0.2)

    def watcher():
        while True:
            if (root_dir / '.drop_request').exists() and not (root_dir / '.drop_done').exists():
                (root_dir / '.drop_done').write_text('ok\n')
            time.sleep(0.2)
    threading.Thread(target=thermal, daemon=True).start()
    threading.Thread(target=watcher, daemon=True).start()
    time.sleep(1.2)
    cr.main(['--smoke', '--thermal-log', str(root_dir / 'thermal.log'), *runner_args])


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def spawn(root_dir, *args):
    return subprocess.Popen([sys.executable, __file__, '--child', str(root_dir), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def check_run(run):
    sys.path.insert(0, str(HERE))
    import coresidency as cr
    names = ['B1_mix_nollm', 'B2_only640_nollm', 'B3_mix_gemma', 'B4_only640_gemma']
    for n in names:
        b = json.loads((run / f'block_{n}.json').read_text())
        done = [r for r in b['reads'] if 'detect_ms' in r]
        heat = b['heat_stop']
        s = cr.block_summary(b)
        if n == 'B4_only640_gemma':  # the fake goes to Android status 4 4 s in: the block ends there, still valid
            assert heat['reached_limit'] and heat['limit'] == 'android_status' and 1 < heat['time_to_limit_s'] < 7.5, heat
            assert heat['reading']['status'] == 4 and heat['reading']['skin'] == 46.6, heat
            assert b['duration_s'] < b['planned_s'], (heat, b['duration_s'])
            assert all(r['t'] <= heat['time_to_limit_s'] for r in b['reads']), 'read after the limit stop'
            assert len(done) >= 3, (n, len(done))
            assert s['status_max'] == 4 and s['skin'][2] == 46.6 and s['capped'][0] >= 1, s
            assert s['low_max'] == [1803000, 2348000, 2400000], s['low_max']
        else:  # status 3 (SEVERE) and z9 90 degC throughout: no stop
            assert not heat['reached_limit'] and heat['limit'] is None and heat['time_to_limit_s'] is None, heat
            assert len(done) >= 10, (n, len(done))
            assert s['status_max'] == 3 and s['skin'] == (35.0, 35.0, 35.0) and s['capped'] == (0, 0.0), s
            assert s['low_max'] == list(POLICIES.values()), s['low_max']
        assert s['max_unread'] == (0, 0.0), s
        assert len(b['fast']) >= 5 and b['dumps'] and b['fast'][0]['t'] < 0, 'limit readings start before the camera'
        assert all(x['cpu_c'] == 60.0 and x['bat_c'] == 30.0 for x in b['fast']), b['fast'][:2]
        assert b['thermal_start']['skin'] == 35.0 and not b['thermal_start']['warm_start'], b['thermal_start']
        ce = b['camera_end']
        assert ce['capture_stopped'] and ce['check_s'] >= 3 and ce['force_stop_rc'] == 0, ce
        assert ce['pids_after_force_stop'] == [], ce
        frames = [r['frame'] for r in done]
        assert frames == sorted(set(frames)), f'{n}: frame numbers must advance'
        assert all('read_ms' in r and 'decode_ms' in r and 'age_s' in r for r in done), n
        sizes = {r['size'] for r in done}
        assert sizes == ({640} if 'only640' in n else {320, 640}), (n, sizes)
        assert b['samples'] and all('battery_w' in s and 'z9' in s for s in b['samples']), n
        assert b['samples'][0]['pss_kb']['runner'] == 123456 and b['samples'][0]['pss_kb']['camera_provider'] == 270000
        assert b['lmk']['ok'] and b['lmk']['n_lines'] == 3 and b['lmk']['n_kills'] == 2, b['lmk']
        assert not any('pss_error' in x for x in b['samples']), b['samples'][0]
        assert b['survived']['robotcam_new_frame_at_end'], (n, b['survived'])
        if 'gemma' in n:
            ok = [c for c in b['selector_calls'] if 'ms' in c]
            assert len(ok) >= 2 and all(c['letter'] == 'B' for c in ok), b['selector_calls']
            assert b['survived']['llama_server'] is True
            assert 'llama_server' in b['samples'][-1]['pss_kb']
        else:
            assert not b['selector_calls'] and b['survived']['llama_server'] is None
    ld = json.loads((run / 'loads.json').read_text())
    assert ld['cold_load_s'] > 0 and ld['warm_load_s'] > 0 and ld['cold_state']['mode'].startswith('weights-cold'), ld
    b5 = json.loads((run / 'block_B5_mtp_ram_snapshot.json').read_text())
    assert b5['cmd'][-6:] == ['--model-draft', b5['cmd'][-5], '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']
    assert b5['reply'] == 'fake reply' and b5['detected_640_on_frame'] > 0 and 'VmHWM' in b5['server_status_kb']
    assert b5['camera_end']['capture_stopped'] and b5['camera_end']['force_stop_rc'] == 0, b5['camera_end']
    text = (run / 'report.txt').read_text()
    for label in ('frames processed', 'detect320 ms median/P95', 'selector ms median/P95', 'min MemAvailable MiB',
                  'peak PSS camera provider MiB', 'LMK log lines', 'zone9 start/end/max', 'mean battery W',
                  'Gemma load', 'MTP snapshot', 'idle skin 35.0 degC'):
        assert label in text, label
    row = lambda label: next(l for l in text.splitlines() if l.startswith(label))[len(label):].split()
    assert row('time to limit s')[:3] == ['-', '-', '-'] and 1 < float(row('time to limit s')[3]) < 7.5, text
    assert row('stop limit') == ['-', '-', '-', 'android_status'], text
    assert row('VIRTUAL-SKIN start/end/max degC')[0] == '35.0/35.0/35.0' and \
        row('VIRTUAL-SKIN start/end/max degC')[3] == '35.0/46.6/46.6', text
    assert row('Android status max') == ['3', '3', '3', '4'], text
    assert row('policy capped s (% of block)')[:2] == ['0', '(0%)'] and row('policy capped s (% of block)')[6] != '0'
    assert row('lowest scaling_max MHz 0/4/6') == ['1803/2348/2850'] * 3 + ['1803/2348/2400'], text
    assert row('gate wait s') == ['0'] * 4, text
    with tempfile.TemporaryDirectory() as tmp:  # a gate not reached in 15 min: "warm start", not INCOMPLETE
        shutil.copytree(run, Path(tmp) / 'r')
        b1 = Path(tmp) / 'r/block_B1_mix_nollm.json'
        b = json.loads(b1.read_text())
        b['thermal_start'].update(waited_s=900, warm_start=True)
        b1.write_text(json.dumps(b))
        warm, bad = cr.report(Path(tmp) / 'r')
        assert not bad and next(l for l in warm.splitlines() if l.startswith('gate wait s')).split()[3:6] == \
            ['900', 'warm', 'start'], warm
    assert 'INCOMPLETE' not in text, text
    assert (run.parent / 'downloads' / f'coresidency_{run.name}_report.txt').exists()
    return text


def check_stopped(root_dir):
    am = (root_dir / 'am.log').read_text().splitlines()
    last_start = max(i for i, l in enumerate(am) if ' start ' in l)
    assert any('broadcast' in l and 'STOP' in l for l in am[last_start + 1:]), 'RobotCam not stopped after last start'
    start_t = float(am[last_start].split()[0])  # am.log and root.log share time.monotonic()
    assert any(l.split()[1] == 'forcestop:' and float(l.split()[0]) > start_t
               for l in (root_dir / 'root.log').read_text().splitlines()), 'app not force-stopped after the last start'
    pid = int((root_dir / 'server.pid').read_text().split()[0])
    for _ in range(50):
        if not alive(pid):
            break
        time.sleep(0.1)
    assert not alive(pid), f'llama-server stand-in {pid} still running'


def check_lmk_failure():
    sys.path.insert(0, str(HERE))
    import coresidency as cr
    cr.root = lambda cmd, tag, timeout=60: (0, 'logcat_rc=1\nlogcat: Invalid time\n\n=== lmk lines\n')
    r = cr.lmk_lines(time.time())
    assert not r['ok'] and r['logcat_rc'] == 1 and r['n_lines'] == 0, r
    print('failed logcat query reported as failed, not as 0 kills: PASS')


NO_HEAT = {'reached_limit': False, 'limit': None, 'reason': None, 'time_to_limit_s': None, 'reading': None}
END_OK = {'capture_stopped': True, 'frames_seen': [], 'check_s': 3.1, 'force_stop_rc': 0, 'force_stop_output': '',
          'pids_after_force_stop': [], 'pidof_root_rc': 0, 'pidof_rc': 1}
LIMIT_READINGS = {'fast': [], 'dumps': [], 'cpuinfo_max_khz': POLICIES}


def check_sampling_edges():
    sys.path.insert(0, str(HERE))
    import coresidency as cr
    out = ('-400000\n4100000\nDischarging\n=== runner 1\n TOTAL PSS: 100\n=== robotcam_app 2\n TOTAL PSS: 30\n'
           '=== camera_provider 3\n TOTAL PSS: 200\n=== camera_provider 4\nFailure calling service meminfo\n')
    cr.root = lambda cmd, tag, timeout=60: (0, out)
    s = cr.root_sample(None)
    assert s['pss_kb']['camera_provider'] == 200 and 'camera_provider 4' in s['pss_error'], s
    base = {'t': 0, 'mem_available_mib': 3000, 'swap_used_mib': 0, 'pss_kb': {}, 'z9': 40000, 'battery_w': 2.0}
    samples = [dict(base, t=0, t_end=2), dict(base, t=5, t_end=13), dict(base, t=15, t_end=17),
               dict(base, t=175, t_end=183, mem_available_mib=100)]
    b = {'reads': [], 'selector_calls': [], 'duration_s': 180.0, 'samples': samples, 'heat_stop': NO_HEAT,
         'thermal_start': {'z9': 40000}, 'thermal_end': {'z9': 41000}, **LIMIT_READINGS}
    s = cr.block_summary(b)
    assert s['late_samples'] == 1 and s['min_avail'] == 3000, s   # the sample finished after 180 s is dropped
    assert s['gaps'] == (2, 163.0), s['gaps']                      # 2 -> 13 and 17 -> 180
    print('per-pid PSS failure flagged; late sample dropped; gaps from completion times: PASS')


def check_load_interrupt_and_selector_failure():
    sys.path.insert(0, str(HERE))
    import coresidency as cr
    with tempfile.TemporaryDirectory() as tmp:
        pid_file = Path(tmp) / 'server.pid'
        cr.server_cmd = lambda extra=(): [sys.executable, __file__, '--fake-server', str(PORT), str(pid_file)]
        calls = []

        def healthy():  # free port at the pre-check, then a signal arrives during the health wait
            calls.append(1)
            if len(calls) == 1:
                return False
            while not pid_file.exists():
                time.sleep(0.05)
            raise SystemExit(128 + signal.SIGTERM)
        real_healthy, cr.healthy = cr.healthy, healthy
        try:
            cr.Server(Path(tmp) / 'log')
            raise AssertionError('Server() returned')
        except SystemExit:
            pass
        finally:
            cr.healthy = real_healthy
        pid = int(pid_file.read_text().split()[0])
        time.sleep(0.5)
        assert not alive(pid), 'llama-server stand-in left running after a signal during its load'
    with tempfile.TemporaryDirectory() as tmp:  # SIGTERM between Popen starting the server and self.proc
        pid_file = Path(tmp) / 'server.pid'
        cr.server_cmd = lambda extra=(): [sys.executable, __file__, '--fake-server', str(PORT + 1), str(pid_file)]
        real_popen = cr.subprocess.Popen

        def popen_then_signal(*a, **k):
            p = real_popen(*a, **k)
            while not pid_file.exists():
                time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGTERM)  # handled right here, before Popen returns
            return p
        cr.subprocess.Popen = popen_then_signal
        old_handler = signal.signal(signal.SIGTERM, cr.exit_on_signal)
        try:
            cr.Server(Path(tmp) / 'log')
            raise AssertionError('Server() returned')
        except SystemExit as e:
            assert e.code == 128 + signal.SIGTERM, e.code
        finally:
            cr.subprocess.Popen = real_popen
            signal.signal(signal.SIGTERM, old_handler)
        pid = int(pid_file.read_text().split()[0])
        time.sleep(0.5)
        assert not alive(pid), 'server left running after SIGTERM inside Popen'
        assert not cr.LIVE, cr.LIVE
        pid_file.unlink()
        cr.PORT = PORT + 1
        s = cr.Server(Path(tmp) / 'log')  # returned, but its owner never stored it (a signal right after return)
        assert s in cr.LIVE
        for server in list(cr.LIVE):  # what main's finally does
            server.stop()
        assert not cr.LIVE and not alive(int(pid_file.read_text().split()[0])), 'unowned server not stopped'
    ok_lmk = {'ok': True, 'logcat_rc': 0, 'raw_head': ''}
    alive_cam = {'robotcam_new_frame_at_end': True, 'robotcam_process': True, 'llama_server': True}
    slow = [{'t': 0.0, 'started_s': 0.0, 'ended_s': 170.0, 'ms': 1.7e5}, {'t': 20.0, 'started_s': 170.0, 'ended_s': 185.0, 'ms': 15000}]
    blocks = [{'block': 'B3_mix_gemma', 'mode': 'mix', 'gemma': True, 'lmk': ok_lmk, 'survived': alive_cam,
               'planned_s': 180, 'duration_s': 180.3, 'selector_calls': []},
              {'block': 'B1_mix_nollm', 'mode': 'mix', 'gemma': False,
               'survived': dict(alive_cam, llama_server=None, robotcam_process=False),
               'lmk': {'ok': False, 'logcat_rc': 1, 'raw_head': 'logcat_rc=1\nlogcat: Invalid time'}},
              {'block': 'B4_only640_gemma', 'mode': '640', 'gemma': True, 'lmk': ok_lmk,
               'survived': {'robotcam_new_frame_at_end': False, 'robotcam_process': True, 'llama_server': False},
               'selector_calls': [], 'duration_s': 180.0, 'planned_s': 180, 'heat_stop': NO_HEAT},
              {'block': 'B3_slow_selector', 'mode': 'mix', 'gemma': True, 'lmk': ok_lmk, 'survived': alive_cam,
               'selector_calls': slow, 'duration_s': 180.0, 'planned_s': 180, 'heat_stop': NO_HEAT}]
    for b in blocks[:2]:
        b['heat_stop'] = NO_HEAT
    for b in blocks:
        b['camera_end'] = END_OK
    good = {'max_unread': (0, 0.0), 'sel': (9, 0, 7, 1.0, 2.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'],
            'failed': {}, 'frames': 300, 'n320': 264, 'n640': 36, 'drift320': (70, 72), 'drift640': (260, 270),
            'drift_na': False}
    sums = [dict(good, sel=(0, 9, 0, None, None)),
            dict(good, sel=(0, 0, 0, None, None), sample_errors=3, bat_status=['Charging', 'Discharging']),
            dict(good, failed={'missing': 40}, frames=1, n320=0, n640=1, drift640=(9000, None), sel=(9, 0, 7, 1.0, 2.0)),
            dict(good, sel=(2, 0, 1, 1e5, 1.7e5))]
    bad = cr.problems(blocks, sums)
    assert bad == ['B3_mix_gemma: no successful selector call (9 failed)',
                   'B1_mix_nollm: 3 sample(s) with root/PSS/thermal errors, some usable samples',
                   'B1_mix_nollm: RobotCam: 300 frames, failed reads none, new frame at end True, process at end False',
                   "B1_mix_nollm: battery status ['Charging', 'Discharging'] (power needs Discharging throughout)",
                   bad[4],
                   "B4_only640_gemma: RobotCam: 1 frames, failed reads {'missing': 40}, new frame at end False, "
                   'process at end True',
                   'B4_only640_gemma: 1 detections at 640, drift first/last 30 s (9000, None) (a required measurement is missing)',
                   'B4_only640_gemma: llama-server did not survive the block (see LMK lines)',
                   'B3_slow_selector: selector cadence missed: 2/9 successful calls, started >5 s late at slots [20.0], '
                   'ended after the block at slots [20.0]'], bad
    assert bad[4].startswith('B1_mix_nollm: LMK logcat query failed (rc 1)'), bad
    print('signal during a server load or inside its Popen stops the server; all-failed selector block flagged: PASS')


def check_loads_cores_and_b5_errors():
    sys.path.insert(0, str(HERE))
    import coresidency as cr
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pid_file = tmp / 'server.pid'
        cr.PORT = PORT
        cr.server_cmd = lambda extra=(): [sys.executable, __file__, '--fake-server', str(PORT), str(pid_file)]
        cr.thermal_gate = lambda *a: {'waited_s': 0}
        cr.make_cold = lambda: {'mode': 'test'}
        checks = []
        cr.allowed_cpus = lambda: (checks.append(1), set(range(6)) if len(checks) == 2 else set(range(8)))[1]
        ctx = {'thermal_log': None, 'idle': None, 'smoke': True, 'server': None}
        try:
            cr.loads(ctx, tmp)
            raise AssertionError('loads() did not raise CoresLost')
        except cr.CoresLost:
            pass
        assert ctx['server'] is None and not alive(int(pid_file.read_text().split()[0])), 'server left after CoresLost'
        run = {'smoke': True, 'block_s': 20, 'idle': {'z9': 36000, 'skin': 35.0}, 'sha256': {'coresidency.py': '0' * 64}}
        (tmp / 'run.json').write_text(json.dumps(run))
        sample = {'mem_available_mib': 1, 'swap_used_mib': 0, 'pss_kb': {}, 'pss_error': 'su rc 1'}
        b5 = {'sample': sample, 'server_status_kb': {}, 'load_s': 1.0, 'cmd': ['x'], 'camera_end': END_OK}
        (tmp / 'block_B5_mtp_ram_snapshot.json').write_text(json.dumps(b5))
        text, bad = cr.report(tmp)
        assert bad and 'B5_mtp_ram_snapshot' in bad[0] and 'INCOMPLETE: B5_mtp_ram_snapshot' in text, (bad, text)
        sample.pop('pss_error')  # B5's own end check: a failed force-stop is INCOMPLETE before the report is final
        (tmp / 'block_B5_mtp_ram_snapshot.json').write_text(json.dumps(dict(b5, sample=sample)))
        assert cr.report(tmp)[1] == []
        (tmp / 'block_B5_mtp_ram_snapshot.json').write_text(json.dumps(
            dict(b5, sample=sample, camera_end=dict(END_OK, force_stop_rc=1, force_stop_output='Error'))))
        text, bad = cr.report(tmp)
        assert bad == ["B5_mtp_ram_snapshot: RobotCam end check: am force-stop failed (rc 1: 'Error')"], bad
        assert 'INCOMPLETE: B5_mtp_ram_snapshot: RobotCam end check' in text, text
        old = tmp / 'old_run'
        old.mkdir()
        (old / 'run.json').write_text(json.dumps(run))  # made by the frozen runner: no heat_stop
        cr.require_native = lambda: None
        try:
            cr.main(['--smoke', '--resume', str(old)])
            raise AssertionError('--resume of an old-format run was accepted')
        except SystemExit as e:
            assert 'older runner' in str(e), e
    print('cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE; '
          'old-format run refused by --resume: PASS')


def check_fix1_cadence_and_heat():
    sys.path.insert(0, str(HERE))
    import coresidency as cr
    cr.SELECTOR_S = 20
    ok_lmk = {'ok': True, 'logcat_rc': 0, 'raw_head': ''}
    alive_cam = {'robotcam_new_frame_at_end': True, 'robotcam_process': True, 'llama_server': True}
    good = {'max_unread': (0, 0.0), 'sel': (1, 0, 1, 1962.0, 1962.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'],
            'failed': {}, 'frames': 40, 'n320': 37, 'n640': 3, 'drift320': (101, 101), 'drift640': (493, 493),
            'drift_na': False}
    call = lambda t, end: {'t': t, 'started_s': t, 'ended_s': end, 'ms': 2000.0, 'correct': True}
    gemma = lambda **k: dict({'block': 'B3_mix_gemma', 'mode': 'mix', 'gemma': True, 'lmk': ok_lmk,
                              'survived': alive_cam, 'heat_stop': NO_HEAT, 'camera_end': END_OK, **LIMIT_READINGS}, **k)
    # the smoke evidence: 20 s block ran 20.25 s, one call at slot 0 (old count ceil(20.25/20) = 2)
    smoke = gemma(duration_s=20.25, planned_s=20, selector_calls=[call(0.0, 1.97)])
    assert cr.problems([smoke], [good]) == [], cr.problems([smoke], [good])
    # full run: slots 0..160 (9); the old count ceil(180.4/20) = 10 could never be met
    full = gemma(duration_s=180.4, planned_s=180, selector_calls=[call(20.0 * k, 20.0 * k + 2) for k in range(9)])
    assert cr.problems([full], [dict(good, sel=(9, 0, 9, 2000.0, 2000.0))]) == []
    assert cr.problems([full], [dict(good, sel=(8, 0, 8, 2000.0, 2000.0))]) == [
        'B3_mix_gemma: selector cadence missed: 8/9 successful calls, started >5 s late at slots [], '
        'ended after the block at slots []']
    print('selector slots = those that start inside the block (smoke 1, full 9): PASS')

    # limit stop at 42 s: 3 slots (0, 20, 40); the call in flight at the stop is not an overrun; drift n/a, valid
    reading = {'skin': 46.6, 'status': 4, 'bat_c': 38.0, 'cpu_c': 65.0, 'scaling_max': {'policy6': 1000000}}
    heat = dict(NO_HEAT, reached_limit=True, limit='android_status', reason='Android thermal status 4 >= 4 (CRITICAL)',
                time_to_limit_s=42.0, reading=reading)
    reads = [{'t': 0.5 * i, 'status': 'ok', 'detect_ms': 100.0, 'size': 640 if i % 10 == 0 else 320,
              'read_ms': 1, 'decode_ms': 5, 'age_s': 0.3, 'frame': i + 1} for i in range(84)]
    samples = [{'t': 5.0 * i, 't_end': 5.0 * i + 1, 'mem_available_mib': 3000, 'swap_used_mib': 0, 'pss_kb': {},
                'z9': 60000, 'battery_w': 5.0, 'battery_status': 'Discharging'} for i in range(9)]
    hot = gemma(duration_s=42.01, planned_s=180, heat_stop=heat, reads=reads, samples=samples,
                thermal_start={'z9': 36000, 'waited_s': 0}, thermal_end={'z9': 81000},
                selector_calls=[call(0.0, 2.0), call(20.0, 22.0), call(40.0, 43.5)])
    s = cr.block_summary(hot)
    assert s['drift_na'] and s['drift320'] == (None, None) and s['n640'] == 9, s
    assert cr.problems([hot], [s]) == [], cr.problems([hot], [s])
    assert cr.problems([dict(hot, selector_calls=hot['selector_calls'][:2])], [dict(s, sel=(2, 0, 2, 2000, 2000))])[0] \
        .startswith('B3_mix_gemma: selector cadence missed: 2/3 successful calls')
    # 70 s: both drift windows fit, so a missing one is still INCOMPLETE
    long = dict(hot, duration_s=70.0, heat_stop=dict(heat, time_to_limit_s=70.0))
    assert not cr.block_summary(long)['drift_na']
    assert any('drift' in p for p in cr.problems([long], [dict(s, drift_na=False, drift640=(100, None),
                                                               sel=(4, 0, 4, 2000, 2000))]))
    # mix stopped at 3 s: no 640 frame was due yet
    early = dict(hot, duration_s=3.0, heat_stop=dict(heat, time_to_limit_s=3.0), selector_calls=[call(0.0, 2.0)])
    assert cr.problems([early], [dict(s, n640=0, sel=(1, 0, 1, 2000, 2000))]) == []
    assert cr.problems([dict(early, heat_stop=NO_HEAT)], [dict(s, n640=0, drift_na=False)])  # no limit stop: missing
    # round-3 finding: a limit already met before the first frame is "not run: limit at start", INCOMPLETE, with the
    # readings, and nothing else (no RobotCam survival or selector lines)
    now = gemma(duration_s=0.02, planned_s=20, reads=[], selector_calls=[], thermal_start={'z9': 81000, 'waited_s': 0},
                thermal_end={'z9': 79000}, heat_stop=dict(heat, time_to_limit_s=0.01),
                samples=[dict(samples[0], t=0.0, t_end=1.3)])
    s = cr.block_summary(now)
    assert s['frames'] == 0 and s['min_avail'] is None, s
    assert cr.problems([now], [s]) == [
        "B3_mix_gemma: not run: limit at start (Android thermal status 4 >= 4 (CRITICAL); skin 46.6, status 4, "
        "battery 38.0, CPU max 65.0 degC, scaling_max {'policy6': 1000000})"], cr.problems([now], [s])
    # the same block without a limit stop, or with a failed read, stays INCOMPLETE on the usual rules
    assert len(cr.problems([dict(now, heat_stop=NO_HEAT)], [dict(s, drift_na=False)])) >= 4
    assert any('RobotCam' in p for p in cr.problems([now], [dict(s, failed={'missing': 1})]))
    # skin loss: the fail-closed stop is INCOMPLETE and its selector slots are counted to the planned end
    lost = dict(hot, heat_stop=dict(NO_HEAT, limit='fail_closed', reason='no VIRTUAL-SKIN and Android status reading '
                                     'for 60 s', time_to_limit_s=61.2, reading=reading))
    bad = cr.problems([lost], [dict(s, frames=120, n640=12, drift_na=False, sel=(4, 0, 4, 2000, 2000))])
    assert bad[0] == 'B3_mix_gemma: stopped fail-closed at 61.2 s: no VIRTUAL-SKIN and Android status reading for 60 s'
    print('limit stop: short stop valid, drift n/a, in-flight call allowed; limit at start and fail-closed stop '
          'INCOMPLETE: PASS')


def check_cleanup1():
    sys.path.insert(0, str(HERE))
    import coresidency as cr
    from PIL import Image
    # block limits, each at its edge (times: monotonic s; the monitor began at 100)
    fs = lambda t, cpu=60.0, bat=30.0: {'t': t, 'cpu_c': cpu, 'bat_c': bat, 'max': {}}
    dm = lambda t, status=0, skin=35.0: {'t': t, 'status': status, 'skin': skin}
    L = lambda now, fast, dumps: cr.block_limit(now, 100.0, fast, dumps)
    ok = [fs(100 + i) for i in range(5)]
    assert L(104.5, ok, [dm(104, status=3)]) is None, 'SEVERE (3) is no stop'
    assert L(104.5, ok, [dm(104, status=4)]) == ('android_status', 'Android thermal status 4 >= 4 (CRITICAL)')
    assert L(104.5, ok, [dm(104, status=4, skin=None)])[0] == 'android_status', 'status alone is enough'
    assert L(104.5, ok, [dm(104, skin=48.0)]) is None, 'skin itself is no stop (Android acts on it)'
    assert L(104.5, ok[:-1] + [fs(104, bat=44.9)], [dm(104)]) is None
    assert L(104.5, ok[:-1] + [fs(104, bat=45.0)], [dm(104)]) == ('battery', 'battery 45.0 >= 45.0 degC')
    assert L(104.5, ok[:-2] + [fs(103, bat=45.0), fs(104, bat=None)], [dm(104)])[0] == 'battery', 'latest readable'
    hot = lambda t: fs(t, cpu=110.0)
    assert L(104.5, [fs(100), fs(101), hot(102), hot(103), fs(104, cpu=109.9)], [dm(104)]) is None
    assert L(104.5, [fs(100), fs(101), fs(102), hot(103), hot(104)], [dm(104)]) is None, '2 hot samples'
    assert L(104.5, [fs(100), fs(101), hot(102), hot(103), hot(104)], [dm(104)])[0] == 'cpu_fault'
    assert L(104.5, [fs(100), hot(101), hot(102), fs(103, cpu=None), hot(104)], [dm(104)])[0] == 'cpu_fault', \
        'an unreadable sample neither breaks nor resets the run'
    # skin/status loss: fail closed after 60 s, counted from the monitor start; either one alone keeps it fresh
    assert L(159.9, [fs(159)], []) is None
    assert L(160.1, [fs(160)], []) == ('fail_closed', 'no VIRTUAL-SKIN and Android status reading for 60 s')
    assert L(170.0, [fs(170)], [dm(111)]) is None and L(171.5, [fs(171)], [dm(111)])[0] == 'fail_closed'
    assert L(171.5, [fs(171)], [dm(111), dm(150, skin=None)])[0] == 'fail_closed', 'skin lost at 111'
    assert L(171.5, [fs(171)], [dm(150, status=None), dm(151, skin=None)]) is None, 'from two dumps'
    assert L(105.0, [fs(100)], [dm(104)]) is None
    assert L(105.5, [fs(100)], [dm(105)]) == ('fail_closed', 'no CPU zone reading for 5 s')
    assert L(105.5, [fs(100)] + [fs(101 + i, bat=None) for i in range(5)], [dm(105)]) == \
        ('fail_closed', 'no battery temperature reading for 5 s')
    print('block limits: status 3 no stop / 4 stop, battery 44.9 / 45.0, CPU 110 in 3 consecutive (fault), '
          'skin+status loss 60 s and CPU/battery loss 5 s fail closed: PASS')

    # a limit met before the first frame: frame_loop returns before any read
    hit = {'limit': 'battery', 'reason': 'x', 'time_to_limit_s': 0.0}
    reads = []
    assert cr.frame_loop(None, 'mix', 's', time.monotonic(), 20, reads, lambda: hit) == (None, hit) and reads == []

    # cooldown gate: reached; not reached in GATE_MAX_S -> warm start (not INCOMPLETE); smoke does not wait
    real = cr.read_thermal, cr.read_dump, cr.GATE_MAX_S, cr.GATE_POLL_S
    idle = {'z9': 30000, 'skin': 32.0}
    try:
        cr.GATE_MAX_S, cr.GATE_POLL_S = 0.6, 0.1
        skins = iter([36.0, 34.0, 33.5])
        cr.read_thermal = lambda path: {'at': 'x', 'z9': 33000, 'z10': 0, 'z11': 0}
        cr.read_dump = lambda: {'skin': next(skins), 'status': 0}
        g = cr.thermal_gate('log', idle, 'B1', False)
        assert g['skin'] == 33.5 and not g['warm_start'], g   # 33.5 <= 32 + 1.5, z9 33 <= 30 + 4
        cr.read_dump = lambda: {'skin': 33.6, 'status': 1}
        g = cr.thermal_gate('log', idle, 'B1', False)
        assert g['warm_start'] and g['skin'] == 33.6, g
        cr.read_dump = lambda: {'skin': 33.0, 'status': 0}
        cr.read_thermal = lambda path: {'at': 'x', 'z9': 34001, 'z10': 0, 'z11': 0}
        assert cr.thermal_gate('log', idle, 'B1', False)['warm_start'], 'z9 still gates'
        cr.read_dump = lambda: {'skin': None, 'status': None}
        cr.read_thermal = lambda path: {'at': 'x', 'z9': 30000, 'z10': 0, 'z11': 0}
        assert cr.thermal_gate('log', idle, 'B1', False)['warm_start'], 'no skin reading is not cool'
        began = time.monotonic()
        assert not cr.thermal_gate('log', idle, 'B1', True)['warm_start'] and time.monotonic() - began < 0.1
    finally:
        cr.read_thermal, cr.read_dump, cr.GATE_MAX_S, cr.GATE_POLL_S = real
    print('cooldown gate: skin <= idle + 1.5 and z9 <= idle + 4 reached; else warm start after the limit: PASS')

    # RobotCam end check: capture stopped vs still advancing; force-stop failure
    with tempfile.TemporaryDirectory() as tmp:
        cr.FRAME_DIR = tmp
        stop = threading.Event()

        def writer(n_max):
            n = 0
            while not stop.is_set() and n < n_max:
                n += 1
                boot_ms = int(time.clock_gettime(time.CLOCK_BOOTTIME) * 1000)
                c = f'robotcam session=ab frame={n} capture_boot_ms={boot_ms} capture_wall_ms=0 clock=sensor'
                Image.new('RGB', (64, 48)).save(Path(tmp) / 't.jpg', comment=c.encode())
                os.replace(Path(tmp) / 't.jpg', Path(tmp) / 'frame.jpg')
                stop.wait(0.5)
        calls = []

        def fake_root(cmd, tag, timeout=60):
            calls.append((tag, cmd))
            return (0, '') if tag == 'forcestop' else (0, '4321\npidof_rc=0\n')
        th = threading.Thread(target=writer, args=(2,))  # two more frames after STOP, then capture ends
        th.start()
        r = cr.camera_end_check(fake_root)
        th.join()
        assert r['capture_stopped'] and [x['frame'] for x in r['frames_seen']] == [1, 2] and r['check_s'] >= 3.4, r
        assert calls == [('forcestop', 'am force-stop com.pixelrobot.robotcam'),
                         ('pidof', 'pidof com.pixelrobot.robotcam; echo "pidof_rc=$?"')]
        assert r['pids_after_force_stop'] == ['4321'] and 'still running after force-stop' in cr.camera_end_failed(r), \
            'a pid left after the force-stop fails the run'
        th = threading.Thread(target=writer, args=(1000,))  # still capturing
        th.start()
        r = cr.camera_end_check(fake_root, quiet_s=3, limit_s=5)
        stop.set()
        th.join()
        assert not r['capture_stopped'] and len(r['frames_seen']) >= 9, r
        assert cr.camera_end_failed(r).startswith('capture not shown stopped 5.'), r
        assert [c[0] for c in calls[-2:]] == ['forcestop', 'pidof'], 'force-stop also when capture did not stop'
        (Path(tmp) / 'frame.jpg').write_bytes(b'not a jpeg')  # unreadable reads are no evidence of a stop
        r = cr.camera_end_check(fake_root, quiet_s=1, limit_s=2)
        assert not r['capture_stopped'] and r['unreadable_reads'] >= 10 and not r['frames_seen'], r
        assert 'unreadable reads' in cr.camera_end_failed(r)
        os.unlink(Path(tmp) / 'frame.jpg')
        r = cr.camera_end_check(lambda cmd, tag, timeout=60: (1, 'Error: no permission\n'), quiet_s=0.3)
        assert r['capture_stopped'] and cr.camera_end_failed(r) == "am force-stop failed (rc 1: 'Error: no permission')"

        def boom(cmd, tag, timeout=60):
            raise subprocess.TimeoutExpired(cmd, timeout)
        r = cr.camera_end_check(boom, quiet_s=0.3)
        assert r['force_stop_rc'] is None and 'force-stop failed (rc None' in cr.camera_end_failed(r), r
        b = {'block': 'B1_mix_nollm', 'gemma': False, 'mode': '640', 'heat_stop': NO_HEAT, 'camera_end': r,
             'survived': {'robotcam_new_frame_at_end': True, 'robotcam_process': True, 'llama_server': None},
             'lmk': {'ok': True}}
        good = {'max_unread': (0, 0.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'], 'failed': {}, 'frames': 300,
                'n640': 300, 'drift640': (1, 1), 'drift_na': False}
        bad = cr.problems([b], [good])
        assert len(bad) == 1 and bad[0].startswith('B1_mix_nollm: RobotCam end check: am force-stop failed (rc None: ')\
            and 'TimeoutExpired' in bad[0], bad
        assert cr.problems([dict(b, camera_end=END_OK)], [good]) == []
    print('RobotCam end check: stopped after STOP / still advancing / unreadable reads; force-stop failure '
          'INCOMPLETE; cached pid recorded only: PASS')

    # capped time: in-block samples only, each counting the time since the previous one; unreadable flagged
    mx = lambda t, p6=2850000, p0=1803000: {'t': t, 'max': {'policy0': p0, 'policy4': 2348000, 'policy6': p6}}
    blk = {'reads': [], 'selector_calls': [], 'samples': [], 'heat_stop': NO_HEAT, 'dumps': [],
           'thermal_start': {'z9': 40000}, 'thermal_end': {'z9': 41000}, 'cpuinfo_max_khz': POLICIES,
           'duration_s': 10.0, 'fast': [mx(-1.0, p6=1000000), mx(1.0), mx(2.0, p6=2400000), mx(5.0, p6=2400000),
                                        mx(6.0), mx(7.0, p0=None), mx(10.0), mx(11.0, p6=500000)]}
    s = cr.block_summary(blk)
    assert s['capped'] == (4.0, 40.0), s['capped']                      # 1-2 and 2-5 (a skipped slot), not -1 or 11
    assert s['max_unread'] == (1, 1.0) and s['low_max'] == [1803000, 2348000, 2400000], s
    good = {'max_unread': (0, 0.0), 'sample_errors': 0, 'min_avail': 3000, 'bat_status': ['Discharging'], 'failed': {}, 'frames': 20,
            'n640': 20, 'drift640': (1, 1), 'drift_na': False}
    b = dict(blk, block='B2_only640_nollm', gemma=False, mode='640', camera_end=END_OK, lmk={'ok': True},
             survived={'robotcam_new_frame_at_end': True, 'robotcam_process': True, 'llama_server': None})
    assert cr.problems([b], [dict(good, max_unread=s['max_unread'])]) == [
        'B2_only640_nollm: 1 1 s sample(s) without every scaling_max_freq (1.0 s of the block not known capped or not)']
    assert cr.problems([b], [good]) == []
    print('capped time: pre-block and post-block samples excluded, time-weighted, % of block; unreadable '
          'scaling_max INCOMPLETE: PASS')

    # discovery from a fake sysfs through the real persistent root shell (lowest zone of each type)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.environ['PATH']
        make_sysfs(Path(tmp), cr)
        sh = cr.RootShell()
        try:
            lay = cr.discover(sh)
            assert lay == {'cpu_zones': {'BIG': '9', 'MID': '10', 'LITTLE': '11'}, 'policies': POLICIES}, lay
            x = cr.fast_sample(sh, cr.fast_keys(lay))
            assert x['cpu_c'] == 60.0 and x['bat_c'] == 30.0 and x['max'] == POLICIES and 'error' not in x, x
            os.unlink(Path(tmp) / 'sys/thermal/thermal_zone10/temp')
            assert cr.fast_sample(sh, cr.fast_keys(lay))['cpu_c'] is None
        finally:
            sh.close()
            os.environ['PATH'] = path
    d = cr.parse_dump(dump_text(status=4, skin=46.6))
    assert (d['status'], d['skin']) == (4, 46.6), d
    assert cr.parse_dump(dump_text(skin=None))['skin'] is None and cr.parse_dump('Failure calling service')['status'] is None
    print('layout discovery, 1 s sample and dump parsing: PASS')


def check_fix1_cold():
    sys.path.insert(0, str(HERE))
    import coresidency as cr
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        f = tmp / 'weights.gguf'
        with open(f, 'wb') as fh:
            fh.write(os.urandom(1 << 20))
            fh.flush()
            os.fsync(fh.fileno())
        f.read_bytes()
        assert cr.resident_pages([f]) > 0
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        assert cr.resident_pages([f]) == 0, cr.resident_pages([f])
        cr.OUT_ROOT, cr.MODEL, cr.HANDSHAKE_S = tmp, str(f), 5
        answer = {'text': 'ok'}

        def watcher(stop):
            while not stop.is_set():
                if (tmp / '.drop_request').exists() and not (tmp / '.drop_done').exists():
                    (tmp / '.drop_done').write_text(answer['text'] + '\n')
                time.sleep(0.05)
        stop = threading.Event()
        threading.Thread(target=watcher, args=(stop,), daemon=True).start()
        cached = iter([2413, 1223] * 4)  # the smoke run's Cached values: 50.7 %, the old check's fallback
        cr.meminfo_mib = lambda: {'cached_mib': next(cached)}
        cr.cold_files = lambda: [f]
        st = cr.make_cold()
        assert st['mode'].startswith('full-cold') and st['resident_pages_after'] == 0, st
        f.read_bytes()  # a load file still cached after an 'ok' answer: fallback
        st = cr.make_cold()
        assert st['mode'].startswith('weights-cold') and st['resident_pages_after'] > 0, st
        assert "handshake answer 'ok'" in st['fallback_reason'] and cr.resident_pages([f]) == 0, st
        answer['text'] = 'failed'
        st = cr.make_cold()
        assert st['mode'].startswith('weights-cold') and "answer 'failed'" in st['fallback_reason'], st
        cr.cold_files = lambda: [tmp / 'missing']
        answer['text'] = 'ok'
        st = cr.make_cold()
        assert st['mode'].startswith('weights-cold') and 'FileNotFoundError' in st['resident_pages_after'], st
        stop.set()
    print('cold load: full-cold iff the drop answered ok and no load-file page is cached (mincore): PASS')


def main():
    check_cleanup1()
    check_fix1_cadence_and_heat()
    check_fix1_cold()
    check_lmk_failure()
    check_sampling_edges()
    check_load_interrupt_and_selector_failure()
    check_loads_cores_and_b5_errors()
    with tempfile.TemporaryDirectory() as tmp:
        root_dir = Path(tmp) / 'a'
        root_dir.mkdir()
        p = spawn(root_dir)
        out, _ = p.communicate(timeout=300)
        assert p.returncode == 0, out[-3000:]
        run = next(root_dir.glob('run_*_smoke'))
        print(check_run(run))
        check_stopped(root_dir)
        print('full smoke run: PASS')

        root_dir = Path(tmp) / 'b'
        root_dir.mkdir()
        p = spawn(root_dir)
        deadline = time.monotonic() + 200
        am_log = root_dir / 'am.log'
        while not am_log.exists() or sum(' start ' in l for l in am_log.read_text().splitlines()) < 3:
            assert time.monotonic() < deadline and p.poll() is None, 'B3 never started'
            time.sleep(0.2)
        time.sleep(2)
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=60)
        assert p.returncode == 128 + signal.SIGTERM, (p.returncode, out[-2000:])
        check_stopped(root_dir)
        run = next(root_dir.glob('run_*_smoke'))
        assert not (run / 'block_B3_mix_gemma.json').exists()
        print(f'SIGTERM during B3: exit {p.returncode}, RobotCam stopped, server stopped: PASS')
        p = spawn(root_dir, '--resume', str(run))
        out, _ = p.communicate(timeout=300)
        assert p.returncode == 0, out[-3000:]
        assert '[B1_mix_nollm] already completed' in out and '[B2_only640_nollm] already completed' in out, out
        check_run(run)
        check_stopped(root_dir)
        print('--resume after SIGTERM: PASS')

        root_dir = Path(tmp) / 'c'  # the STOP broadcast fails at B1's end: the force-stop still follows
        root_dir.mkdir()
        p = subprocess.Popen([sys.executable, __file__, '--child', str(root_dir)], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=dict(os.environ, FAKE_STOP_FAIL='1'))
        out, _ = p.communicate(timeout=120)
        assert p.returncode != 0 and 'CalledProcessError' in out and 'AssertionError' not in out, (p.returncode, out[-2000:])
        assert not list(root_dir.glob('run_*_smoke/block_*.json')), 'a block with a failed STOP was kept'
        am = (root_dir / 'am.log').read_text().splitlines()
        assert sum(' start ' in l for l in am) == 1 and am[-1].split()[1] == 'broadcast', am
        stop_t = float(am[-1].split()[0])
        rl = [l.split() for l in (root_dir / 'root.log').read_text().splitlines()]
        assert any(l[1] == 'forcestop:' and float(l[0]) > stop_t for l in rl), 'no force-stop after the failed STOP'
        print('failed STOP broadcast: force-stop still runs: PASS')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--fake-server']:
        fake_server(*sys.argv[2:])
    elif sys.argv[1:2] == ['--child']:
        child(sys.argv[2], sys.argv[3:])
    else:
        main()
