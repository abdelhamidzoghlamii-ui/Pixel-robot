#!/usr/bin/env python3
"""Offline test of thermal_char.py: fake su/dumpsys on PATH, fake sysfs tree, fake RobotCam frames, fake streaming
llama-server, fake detector. No camera, no root, no model, no motors. Needs a Python with Pillow, numpy and
onnxruntime (native Termux's; it also runs inside Debian/proot):

  /data/data/com.termux/files/usr/bin/python test_thermal_char.py
"""
import faulthandler
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BIG, MID, LITTLE, QUIET, BAD_ZONE = '9', '10', '11', '17', '6'
TYPES = {'0': 'cp_on_chip_0', BAD_ZONE: 'cp_on_chip_6', BIG: 'BIG', MID: 'MID', LITTLE: 'LITTLE', QUIET: 'quiet_therm',
         '21': 'battery'}
POLICIES = {'policy0': ('0 1 2 3', 1803000), 'policy4': ('4 5', 2348000), 'policy6': ('6 7', 2850000)}


def dump_text(status=0, skin=35.0, with_skin=True):
    temps = [('battery', 2, 30.1), ('BIG', 0, 60.0), ('quiet_therm', -1, 31.0), ('VIRTUAL-SKIN-CPU', -1, 36.0)]
    if with_skin:
        temps.append(('VIRTUAL-SKIN', 3, skin))
    temps.append(('disp_therm', -1, float('nan')))
    hal = ''.join(f'\tTemperature{{mValue={v}, mType={t}, mName={n}, mStatus=0}}\n' for n, t, v in temps)
    return ('IsStatusOverride: false\nThermalEventListeners:\n\tcallbacks: 1\n'
            f'Thermal Status: {status}\nCached temperatures:\n'
            '\tTemperature{mValue=40.0, mType=3, mName=VIRTUAL-SKIN, mStatus=1}\n'
            'HAL Ready: true\nHAL connection:\n\tThermalHAL AIDL 1  connected: yes\n'
            f'Current temperatures from HAL:\n{hal}'
            'Current cooling devices from HAL:\n\tCoolingDevice{mValue=0, mType=2, mName=thermal-cpufreq-2}\n')


def make_env(T):
    """Fake bin (su, dumpsys) and sysfs tree under T."""
    b, sysd = T / 'bin', T / 'sys'
    b.mkdir(parents=True)
    (b / 'su').write_text('#!/bin/bash\nif [ "$1" = -c ]; then exec bash -c "$2"; fi\nexec bash\n')
    (b / 'dumpsys').write_text(f'#!/bin/bash\n[ "$1" = thermalservice ] || exit 1\ncat {T}/dump.txt\n')
    # as real pidof: the pids and exit 0, or nothing and exit 1
    (b / 'pidof').write_text(f'#!/bin/bash\np=$(cat {T}/robotcam_pids 2>/dev/null)\n'
                             '[ -n "${p//[[:space:]]/}" ] && { echo "$p"; exit 0; }\nexit 1\n')
    # only the force-stop goes through su; it ends the process STOP leaves cached
    (b / 'am').write_text(f'#!/bin/bash\n[ "$1" = force-stop ] || exit 1\necho "$*" >>{T}/am_root.log\n: >{T}/robotcam_pids\n')
    for x in b.iterdir():
        x.chmod(0o755)
    th, cf, bat = sysd / 'class/thermal', sysd / 'cpufreq', sysd / 'battery'
    for i, typ in TYPES.items():
        z = th / f'thermal_zone{i}'
        z.mkdir(parents=True)
        (z / 'type').write_text(typ + '\n')
        if i == BAD_ZONE:
            (z / 'temp').mkdir()  # unreadable, like the phone's zone6-8
        else:
            (z / 'temp').write_text('36000\n')
        if typ in ('BIG', 'MID', 'LITTLE'):
            for n, (tt, ty) in enumerate([(20000, 'active'), (80000, 'active'), (100000, 'passive'),
                                          (110000, 'active'), (120000, 'hot')]):
                (z / f'trip_point_{n}_temp').write_text(f'{tt}\n')
                (z / f'trip_point_{n}_type').write_text(f'{ty}\n')
    for n in range(3):
        c = th / f'cooling_device{n}'
        c.mkdir(parents=True)
        (c / 'type').write_text(f'thermal-cpufreq-{n}\n')
        (c / 'max_state').write_text('10\n')
        (c / 'cur_state').write_text('0\n')
    for p, (cpus, mx) in POLICIES.items():
        d = cf / p
        d.mkdir(parents=True)
        for k, v in (('related_cpus', cpus), ('cpuinfo_max_freq', mx), ('cpuinfo_min_freq', 300000),
                     ('scaling_max_freq', mx), ('scaling_cur_freq', mx), ('scaling_available_frequencies', f'300000 {mx}')):
            (d / k).write_text(f'{v}\n')
    bat.mkdir(parents=True)
    for k, v in (('temp', 300), ('current_now', -1000000), ('voltage_now', 4000000), ('status', 'Discharging')):
        (bat / k).write_text(f'{v}\n')
    (T / 'dump.txt').write_text(dump_text())
    (T / 'tmp').mkdir()
    (T / 'out').mkdir()
    (T / 'downloads').mkdir()
    return th, cf, bat


def put(path, v):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(f'{v}\n')
    os.replace(tmp, path)


def fake_server(port, pid_file, mode='normal'):
    Path(pid_file).write_text(f'{os.getpid()}\n')
    ready = time.monotonic() + (4 if mode == 'slow' else 0.3)  # slow: a llama-server load that takes 4 s

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            ok = self.path == '/health' and time.monotonic() > ready
            data = b'{"status": "ok"}' if ok else b'{}'
            self.send_response(200 if ok else 503)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body['n_predict'] == 256 and body['temperature'] == 0 and body['stream'], body
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            n = 20
            for i in range(n):
                self.wfile.write(b'data: {"content": "x", "stop": false}\n\n')
                self.wfile.flush()
                time.sleep(0.02)
                if mode == 'stall' and i == 4:
                    time.sleep(60)  # a request in progress when the load stops
            final = {'content': '', 'stop': True, 'timings': {'prompt_n': 30, 'prompt_ms': 50.0, 'predicted_n': n,
                                                              'predicted_ms': n * 20.0, 'predicted_per_second': 50.0}}
            self.wfile.write(f'data: {json.dumps(final)}\n\n'.encode())
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(('127.0.0.1', int(port)), H).serve_forever()


def child(T, scenario, port):
    """Runs thermal_char.main --smoke with every phone dependency faked; a driver thread changes the fake sensors."""
    from PIL import Image
    T = Path(T)
    faulthandler.dump_traceback_later(90, exit=True, file=open(T / 'watchdog.txt', 'w'))  # a hung child: stack, exit
    th, cf, bat = T / 'sys/class/thermal', T / 'sys/cpufreq', T / 'sys/battery'
    os.environ['PATH'] = f'{T}/bin:' + os.environ['PATH']
    import thermal_char as tc
    cr = tc.cr
    frames = T / 'frames'
    frames.mkdir()
    gguf = T / 'fake.gguf'
    gguf.write_bytes(b'x' * 4096)
    cr.require_native = lambda: None
    cr.FRAME_DIR, cr.MODEL, cr.PORT = str(frames), str(gguf), port
    cr.server_cmd = lambda extra=(): [sys.executable, __file__, '--fake-server', str(port), str(T / 'server.pid'),
                                      'slow' if scenario == 'startup' else 'normal']
    cpus = {'set': set(range(8))}
    cr.allowed_cpus = lambda: cpus['set']
    tc.THERMAL, tc.CPUFREQ, tc.BATTERY, tc.ROOT_TMP = str(th), str(cf), str(bat), str(T / 'tmp')
    tc.OUT_ROOT, tc.DOWNLOADS = T / 'out', T / 'downloads'
    tc.SMOKE_LOAD_S, tc.SMOKE_COOL_S, tc.BASELINE_S = 8, 4, 1.5
    tc.SAMPLE_S, tc.DUMP_S, tc.WINDOW_S, tc.TIMELINE_S, tc.MINUTE_S = 0.25, 0.5, 2, 3, 3
    tc.SKIN_WAIT_S, tc.SENSOR_WAIT_S, tc.FRAME_WAIT_S = 5, 6, 2  # 6: proot under CPU load can stall 1 s reads ~4 s

    class FakeDetector:
        def detect(self, frame, size=320):
            assert frame.size == (640, 480) and size == 640
            if cam.get('fail'):
                raise ValueError('fake detector failure')
            time.sleep(0.05)
            return []
    cr.Detector = FakeDetector
    cam = {'run': None, 'since': None}
    log = lambda text: open(T / 'am.log', 'a').write(text + '\n')

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
        log(' '.join(args))
        if cam['run']:
            cam['run'].set()
            cam['run'] = None  # the app process stays cached (pid kept) until the force-stop, as on the phone
        if args[0] == 'start':
            cam['run'], cam['since'] = threading.Event(), time.monotonic()
            (T / 'robotcam_pids').write_text('4321\n')
            threading.Thread(target=writer, args=(cam['run'], f'{int(time.time() * 1000) % 0xffffff:x}'),
                             daemon=True).start()
    cr.am = fake_am

    def driver():
        steps = set()
        once = lambda k: k not in steps and not steps.add(k)
        while True:
            time.sleep(0.05)
            if scenario == 'startup' and (T / 'server.pid').exists() and once('s'):  # while llama-server loads
                (T / 'dump.txt').write_text(dump_text(skin=48.5))
            if cam['since'] is None:
                continue
            t, on = time.monotonic() - cam['since'], cam['run'] is not None
            if scenario == 'duration':
                put(th / f'thermal_zone{BIG}/temp', 95000 if on else 37000)
                if t > 3 and once('a'):
                    put(cf / 'policy6/scaling_max_freq', 2400000)
                    put(th / 'cooling_device1/cur_state', 2)
                    (T / 'dump.txt').write_text(dump_text(status=1, skin=40.0))
                if t > 5 and once('b'):
                    put(cf / 'policy6/scaling_max_freq', 2000000)
                    put(cf / 'policy4/scaling_max_freq', 2000000)
            elif scenario == 'skin' and t > 3:
                (T / 'dump.txt').write_text(dump_text(skin=48.0))
            elif scenario == 'battery' and t > 3:
                put(bat / 'temp', 450)
            elif scenario == 'status' and t > 3:
                (T / 'dump.txt').write_text(dump_text(status=5))
            elif scenario == 'cpu' and t > 3:
                put(th / f'thermal_zone{MID}/temp', 110000)
            elif scenario == 'sensorfail' and t > 0.5 and once('a'):
                os.unlink(bat / 'temp')
            elif scenario == 'noframes' and t > 3 and cam['run']:
                cam['run'].set()
            elif scenario == 'serverdie' and t > 3 and once('a'):
                os.kill(int((T / 'server.pid').read_text()), signal.SIGKILL)
            elif scenario == 'detectfail' and t > 3:
                cam['fail'] = True
            elif scenario == 'skinlost' and once('a'):  # VIRTUAL-SKIN reported until the camera starts
                (T / 'dump.txt').write_text(dump_text(with_skin=False))
            elif scenario == 'cores' and t > 3:
                cpus['set'] = set(range(6))
    if scenario == 'noskin':  # the HAL never reports VIRTUAL-SKIN
        (T / 'dump.txt').write_text(dump_text(with_skin=False))
    threading.Thread(target=driver, daemon=True).start()
    try:
        tc.main(['--smoke', '--note', f'test {T.name} 22 degC'])
    finally:
        (T / 'modules.txt').write_text('\n'.join(sorted(sys.modules)))


# ---------------------------------------------------------------- unit checks (no subprocess)

def unit():
    import thermal_char as tc
    assert 'motors' not in sys.modules and not any('serial' in m or 'usb' in m for m in sys.modules), 'motors/serial import'
    d = tc.parse_dump(dump_text(status=2, skin=41.5))
    assert d['status'] == 2 and d['skin'] == 41.5 and 'disp_therm' not in d['hal'] and d['hal']['BIG'] == 60.0, d
    assert d['cached_names'] == ['VIRTUAL-SKIN'], d
    d = tc.parse_dump(dump_text(with_skin=False))
    assert d['skin'] is None and d['status'] == 0 and 'VIRTUAL-SKIN-CPU' in d['hal'], d
    d = tc.parse_dump('Failure calling service thermalservice: Failed transaction (2147483646)\n')
    assert d == {'status': None, 'skin': None, 'hal': {}, 'cached_names': []}, d

    layout = {'cpu_zones': {'BIG': '9', 'MID': '10', 'LITTLE': '11'}}
    smp = lambda t, big=50000, bat=30.0, cpu=True: {'t': t, 'cpu_ok': cpu, 'complete': cpu and bat is not None,
                                                     'bat_c': bat, 'zones': {'9': big if cpu else None, '10': 40000,
                                                                              '11': 40000}}
    dmp = lambda t, skin=35.0, status=0: {'t': t, 'skin': skin, 'status': status}
    S = lambda now, samples, dumps, load=100.0: tc.stop_reasons(now, load, samples, dumps, layout)
    ok_s = [smp(100 + i) for i in range(5)]
    assert S(104.5, ok_s, [dmp(100)]) == []
    assert S(104.5, ok_s, [dmp(100, skin=47.9)]) == []
    assert 'VIRTUAL-SKIN 48.0 >= 48.0' in S(104.5, ok_s, [dmp(100, skin=48.0)])[0]
    assert S(104.5, ok_s, [dmp(100, status=4)]) == []
    assert 'status 5 >= 5 (EMERGENCY)' in S(104.5, ok_s, [dmp(100, status=5)])[0]
    assert S(104.5, ok_s[:-1] + [smp(104, bat=44.9)], [dmp(100)]) == []
    assert 'battery 45.0 >= 45.0' in S(104.5, ok_s[:-1] + [smp(104, bat=45.0)], [dmp(100)])[0]
    hot = lambda t: smp(t, big=110000)
    assert S(104.5, [smp(100), smp(101), hot(102), hot(103), smp(104)], [dmp(100)]) == [], '2 hot samples'
    assert S(104.5, [smp(100), hot(101), hot(102), smp(103, big=109999), hot(104)], [dmp(100)]) == []
    assert 'CPU zone >= 110 degC in 3' in S(104.5, [smp(100), smp(101), hot(102), hot(103), hot(104)], [dmp(100)])[0]
    # each limit from its own readings: an unreadable CPU zone is skipped, a missing battery value does not matter
    assert 'CPU zone >= 110' in S(104.5, [smp(100), hot(101), hot(102), smp(103, cpu=False), hot(104)], [dmp(100)])[0]
    assert 'CPU zone >= 110' in S(104.5, [smp(100), hot(101), hot(102), smp(103, big=110000, bat=None)], [dmp(100)])[0]
    assert S(104.5, [smp(100), smp(101, bat=None), smp(102, bat=45.0), smp(103, bat=None)], [dmp(100)])[0] \
        .startswith('battery 45.0'), 'the latest battery reading counts even if later samples lack it'
    # no skin: counted from load start (60 s), a pre-load reading only lasts until load start + 60 s
    assert S(159.9, [smp(159)], []) == []
    assert S(160.1, [smp(160)], [])[0].startswith('fail closed: no VIRTUAL-SKIN')
    assert S(160.1, [smp(160)], [dmp(90)])[0].startswith('fail closed: no VIRTUAL-SKIN')
    assert S(170.0, [smp(170)], [dmp(111)]) == []
    assert S(172.0, [smp(172)], [dmp(111)])[0].startswith('fail closed: no VIRTUAL-SKIN')
    assert S(160.1, [smp(160)], [dmp(150, skin=None), dmp(151, status=None)]) == [], 'skin and status from two dumps'
    assert S(172.0, [smp(172)], [dmp(111), dmp(150, skin=None)])[0].startswith('fail closed: no VIRTUAL-SKIN')
    # no reading of a stop input for 5 s (each input on its own)
    nocpu = [smp(101 + i, cpu=False) for i in range(6)]
    assert S(105.0, [smp(100)] + nocpu[:4], [dmp(104)]) == []
    assert S(106.5, [smp(100)] + nocpu[:5], [dmp(106)]) == ['fail closed: no CPU zone reading for 5 s']
    nobat = [smp(101 + i, bat=None) for i in range(6)]
    assert S(106.5, [smp(100)] + nobat[:5], [dmp(106)]) == ['fail closed: no battery temperature reading for 5 s']
    alt = [smp(100 + i, cpu=bool(i % 2), bat=None if i % 2 else 30.0) for i in range(7)]  # never both in one sample
    assert S(106.5, alt, [dmp(106)]) == [], 'each input fresh: no fail-closed stop'
    assert S(104.0, [], [dmp(103)], load=100.0) == [] and len(S(105.5, [], [dmp(105)], load=100.0)) == 2
    print('ok: parse_dump and every stop condition at its edge')

    with tempfile.TemporaryDirectory() as T:
        T = Path(T)
        th, cf, bat = make_env(T)
        os.environ['PATH'] = f'{T}/bin:' + os.environ['PATH']
        tc.THERMAL, tc.CPUFREQ, tc.BATTERY = str(th), str(cf), str(bat)
        sh = tc.RootShell()
        try:
            layout = tc.discover(sh)
            assert layout['cpu_zones'] == {'BIG': BIG, 'MID': MID, 'LITTLE': LITTLE}, layout['cpu_zones']
            assert list(layout['policies']) == ['policy0', 'policy4', 'policy6'], layout['policies']
            assert layout['policies']['policy6']['cpuinfo_max_freq'] == 2850000
            assert layout['policies']['policy4']['related_cpus'] == '4 5'
            assert layout['zones'][BIG]['trips']['2'] == {'temp': 100000, 'type': 'passive'}, layout['zones'][BIG]
            assert list(layout['zones'][BIG]['trips']) == ['0', '1', '2', '3', '4']
            assert layout['cooling']['1'] == {'type': 'thermal-cpufreq-1', 'max_state': 10}
            keys = tc.sample_keys(layout)
            s = tc.take_sample(sh, keys, tc.sample_script(keys), layout, time.monotonic())
            assert s['complete'] and s['zones'][BAD_ZONE] is None and s['zones'][BIG] == 36000, s
            assert s['bat_c'] == 30.0 and s['bat_w'] == 4.0 and s['bat_status'] == 'Discharging', s
            assert s['max']['policy6'] == 2850000 and s['cd']['2'] == 0, s
            try:
                sh.run('sleep 2; echo late', timeout=0.5)
                raise AssertionError('no timeout')
            except RuntimeError as e:
                assert 'no answer' in str(e)
            assert sh.run('echo next') == ['next'], 'output of a timed-out command leaked into the next one'

            # a slow read: the sample, and so its event, is stamped when the read completed (start kept)
            class Slow:
                def run(self, script, timeout=5):
                    time.sleep(0.4)
                    return sh.run(script, timeout)
            script, t0 = tc.sample_script(keys), time.monotonic()
            rows = [tc.take_sample(Slow(), keys, script, layout, t0)]
            (cf / 'policy6/scaling_max_freq').write_text('2400000\n')
            rows.append(tc.take_sample(Slow(), keys, script, layout, t0))
            (cf / 'policy6/scaling_max_freq').write_text('2850000\n')
            assert all(r['complete'] and r['read_s'] >= 0.39 and abs(r['t'] - r['t_start'] - r['read_s']) <= 0.002
                       for r in rows), rows
            L0 = rows[1]['t_start'] + 0.2  # load starts during the second read: its change was seen after L0
            (T / 'rep').mkdir()
            (T / 'rep/run.json').write_text(json.dumps({'layout': layout, 'load_start': L0, 'smoke': False,
                                                        'sha256': {'thermal_char.py': 'x'}, 'load_s': 1, 'cool_s': 1}))
            (T / 'rep/samples_1s.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            rep = tc.report(T / 'rep')[0]
            ev = rep.split('EVENTS')[1].split('TIMELINE')[0]
            at = f'{rows[1]["t"] - L0:+.1f}'
            assert f'  {at} [load] policy6 scaling_max_freq 2850 -> 2400 MHz' in ev and f'(sample {at})' in ev, ev
            run = json.loads((T / 'rep/run.json').read_text())
            ok_cam = {'capture_stopped': True, 'frames_seen': [], 'check_s': 3.1, 'force_stop_rc': 0,
                      'force_stop_output': '', 'pids_after_force_stop': [], 'pidof_root_rc': 0, 'pidof_rc': 1}
            for cam, want in ((ok_cam, None),
                              (dict(ok_cam, pids_after_force_stop=['4321'], pidof_rc=0), "still running after force-stop (pids ['4321'])"),
                              (dict(ok_cam, capture_stopped=False, frames_seen=[{}] * 30, check_s=15.0),
                                               'capture not shown stopped 15.0 s after STOP (30 new frames, None unreadable reads)'),
                              (dict(ok_cam, force_stop_rc=1, force_stop_output='Error'), 'am force-stop failed (rc 1')):
                run['after_stop'] = {'robotcam': cam, 'llama_server_running': False, 'worker_threads_alive': []}
                (T / 'rep/run.json').write_text(json.dumps(run))
                bad = [b for b in tc.report(T / 'rep')[1] if 'after the load stop' in b]
                assert (bad == []) if want is None else (len(bad) == 1 and want in bad[0]), (want, bad)
            r0, r1 = rows[0]['read_s'], rows[1]['read_s']
            assert f'read duration (stamped at completion): median {(r0 + r1) / 2:.3f} s, max {max(r0, r1):.3f} s' \
                in rep, rep
            os.unlink(bat / 'temp')
            s = tc.take_sample(sh, keys, tc.sample_script(keys), layout, time.monotonic())
            assert not s['complete'] and s['bat_c'] is None, s
            (th / 'thermal_zone9/type').write_text('CPU9\n')
            try:
                tc.discover(sh)
                raise AssertionError('discover accepted a layout without BIG')
            except SystemExit as e:
                assert "missing ['BIG']" in str(e), e
        finally:
            sh.close()
        assert sh.p.poll() is not None, 'root shell left running'
    print('ok: persistent root shell, discovery, 1 s sample parsing, timeout recovery, slow read stamped at '
          'completion, missing CPU zone refused')

    # a request cut off by the load stop keeps its streamed tokens; one cut off without a stop is a load failure
    with tempfile.TemporaryDirectory() as T:
        for stop_first in (True, False):
            tc.cr.PORT = port = 18190 + stop_first
            srv = subprocess.Popen([sys.executable, __file__, '--fake-server', str(port), f'{T}/pid', 'stall'])
            try:
                while not tc.cr.healthy():
                    time.sleep(0.05)
                stop, st, path = threading.Event(), {'load_error': None}, Path(T) / f'gen{stop_first}.jsonl'
                with open(path, 'w') as fh:
                    th = threading.Thread(target=tc.gen_loop, args=(time.monotonic(), stop, st, fh))
                    th.start()
                    time.sleep(1.5)
                    if stop_first:
                        stop.set()
                    srv.kill()
                    th.join(timeout=10)
                    assert not th.is_alive(), 'gen_loop did not end when the server went away'
            finally:
                srv.kill()
                srv.wait()
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            assert len(rows) == 1 and rows[0]['streamed'] == 5 == len(rows[0]['arrivals']), rows
            assert rows[0]['interrupted'] and rows[0]['predicted_n'] is None, rows
            assert (st['load_error'] is None) == stop_first, st
    print('ok: interrupted request keeps its 5 streamed tokens; without a load stop it is a load failure')

    # a slow dumpsys: the reading is stamped when it was in hand, its start kept
    real_root, tc.DUMP_S = tc.root_file, 0.2
    tc.root_file = lambda cmd, tag, timeout=30: (time.sleep(0.4), (0, dump_text(status=3)))[1]
    try:
        t0, stop, dumps = time.monotonic(), threading.Event(), []
        with open(os.devnull, 'w') as fh:
            th = threading.Thread(target=tc.dump_loop, args=(t0, stop, dumps, fh, fh))
            th.start()
            time.sleep(1.5)
            stop.set()
            th.join()
    finally:
        tc.root_file = real_root
    assert len(dumps) >= 2 and all(d['t'] - d['t_start'] >= 0.39 and d['status'] == 3 for d in dumps), dumps
    assert all(b['t_start'] >= a['t'] for a, b in zip(dumps, dumps[1:])), 'dumps overlap'
    print('ok: a slow thermalservice dump is stamped at completion (start kept)')


# ---------------------------------------------------------------- end-to-end scenarios

SCENARIOS = {  # name: (expected stop reason fragment, exit code, report complete)
    'duration': ('planned load duration 8 s reached', 0, True),
    'skin': ('VIRTUAL-SKIN 48.0 >= 48.0 degC', 0, True),
    'battery': ('battery 45.0 >= 45.0 degC', 0, True),
    'status': ('Android thermal status 5 >= 5 (EMERGENCY)', 0, True),
    'cpu': ('CPU zone >= 110 degC in 3 consecutive', 0, True),
    'noskin': ('fail closed: no VIRTUAL-SKIN and Android status reading for 5 s', 1, False),
    'skinlost': ('fail closed: no VIRTUAL-SKIN and Android status reading for 5 s', 1, False),
    'sensorfail': ('fail closed: no battery temperature reading for 6 s', 1, False),
    'noframes': ('load failed: no new RobotCam frame for 2 s', 1, False),
    'serverdie': ('load failed:', 1, False),
    'cores': ('cores 4-7 lost', 1, False),
    'startup': ('during startup: VIRTUAL-SKIN 48.5 >= 48.0 degC', 1, False),
    'detectfail': ('load failed: frame/detect loop failed: ValueError: fake detector failure', 1, False),
    'sigterm': ('aborted: SystemExit: 143', 143, False),
}


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def check_after_stop(T, run):
    a = run['after_stop']
    rc = a.pop('robotcam')
    assert a == {'llama_server_running': False, 'worker_threads_alive': []}, a
    assert rc['capture_stopped'] and rc['check_s'] >= 3 and rc['force_stop_rc'] == 0, rc
    assert rc['pids_after_force_stop'] == [] and 'force-stop com.pixelrobot.robotcam' in (T / 'am_root.log').read_text()


def check(name, T, proc, output):
    frag, rc, complete = SCENARIOS[name]
    assert proc.returncode == rc, (name, proc.returncode, output[-1500:])
    runs = list((T / 'out').iterdir())
    assert len(runs) == 1 and runs[0].name.endswith('_smoke'), runs
    out = runs[0]
    run = json.loads((out / 'run.json').read_text())
    rep = (out / 'report.txt').read_text()
    assert any(frag in r for r in run['stop_reasons']), (name, run['stop_reasons'])
    assert run['stop_ok'] == complete and ('INCOMPLETE' not in rep) == complete, (name, rep[:800])
    assert run['note'] == f'test {name} 22 degC' and f'note: test {name} 22 degC' in rep
    assert (T / 'downloads' / f'thermal_char_{out.name}_report.txt').read_text() == rep
    if 'load_start' not in run:  # stopped while starting (startup; noskin when starting is slow): nothing left
        assert name in ('startup', 'noskin') and run['stop_reasons'][0].startswith('during startup: '), run
        assert 'WORK RATE' not in rep and 'EVENTS' in rep, rep[:600]
        if name == 'noskin':
            assert 'VIRTUAL-SKIN in HAL temperatures: NO' in rep
        assert not alive(int((T / 'server.pid').read_text().split()[0])), 'fake llama-server left running'
        am = (T / 'am.log').read_text().splitlines()
        assert am[-1].startswith('broadcast'), 'RobotCam not stopped last'  # a stop in camera_start comes after am start
        if name == 'startup':  # stopped while llama-server was loading: RobotCam never started
            assert not any(l.startswith('start ') for l in am), 'RobotCam started'
        check_after_stop(T, run)
        print(f'ok: {name}: stop "{run["stop_reasons"][0][:70]}", exit {proc.returncode}')
        return
    for sec in ('STOP REASON: ', 'LAYOUT', 'EVENTS', 'TIMELINE every 3 s', 'WORK RATE per 2 s', 'SLOWDOWN', 'COOLDOWN'):
        assert sec in rep, (name, sec)
    for n in ('samples_1s.jsonl', 'thermalservice_5s.jsonl', 'thermalservice_raw.jsonl', 'frames.jsonl', 'gen.jsonl'):
        assert (out / n).exists(), n
    assert run['layout']['cpu_zones'] == {'BIG': BIG, 'MID': MID, 'LITTLE': LITTLE}
    assert run['server_cmd'][2:4] == ['--fake-server', str(PORTS[name])]
    # cleanup: no llama-server, camera stopped last, root shell gone, no motors import
    assert not alive(int((T / 'server.pid').read_text())), f'{name}: fake llama-server left running'
    assert (T / 'am.log').read_text().splitlines()[-1].startswith('broadcast'), f'{name}: RobotCam not stopped last'
    assert not (T / 'robotcam_pids').read_text().strip()
    mods = (T / 'modules.txt').read_text().split() if name != 'sigterm' else []
    assert 'motors' not in mods and not any('serial' in m for m in mods), name
    check_after_stop(T, run)
    samples = [json.loads(l) for l in (out / 'samples_1s.jsonl').read_text().splitlines()]
    frames = [json.loads(l) for l in (out / 'frames.jsonl').read_text().splitlines()]
    gens = [json.loads(l) for l in (out / 'gen.jsonl').read_text().splitlines()]
    L0, L1 = run['load_start'], run['load_stop']
    assert all(r['t'] <= L1 + 1 for r in frames), f'{name}: frame detected after the load stop'
    full = [g for g in gens if not g['interrupted']]
    assert frames and full and all(g['streamed'] == g['predicted_n'] == 20 for g in full), name
    assert all(g['predicted_n'] is None and g['streamed'] <= 20 for g in gens if g['interrupted']), name
    assert sum(bool(g["interrupted"]) for g in gens) <= 1, name
    if name != 'sigterm':
        assert samples[-1]['t'] - L1 >= 4 - 0.5, f'{name}: cooldown not logged'
    load_s = L1 - L0
    if name == 'duration':
        assert 8 <= load_s < 9, load_s
        ev = rep.split('EVENTS')[1].split('TIMELINE')[0]
        assert 'scaling_max_freq start' not in ev, 'a policy at its cpuinfo_max at start is no event'
        for s in ('policy6 scaling_max_freq 2850 -> 2400 MHz (cpuinfo_max 2850) FIRST',
                  'policy6 scaling_max_freq 2400 -> 2000 MHz (cpuinfo_max 2850)\n',
                  'policy4 scaling_max_freq 2348 -> 2000 MHz (cpuinfo_max 2348) FIRST',
                  'cooling_device1 thermal-cpufreq-1 cur_state 0 -> 2 (max 10)',
                  'Android thermal status start -> 0 (NONE; dump ran ', 'Android thermal status 0 -> 1 (LIGHT; dump ran +',
                  '[load]', 'zones: cp_on_chip_0=36.0 cp_on_chip_6=n/a BIG=95.0', 'HAL: battery=30.1',
                  'policy6 ', '/2400 MHz'):
            assert s in ev, (s, ev[:2000])
        assert 'policy0 scaling_max_freq' not in ev, 'policy0 never changed'
        assert 'VIRTUAL-SKIN in HAL temperatures: yes' in rep and 'VIRTUAL-SKIN-CPU' in rep
        assert 'largest gap ' in rep, 'sample gap line'
        tl = rep.split('TIMELINE')[1].split('WORK RATE')[0].splitlines()
        assert any(' load ' in l and '95.0' in l and '40.0' in l and '2000' in l for l in tl), '\n'.join(tl)
        assert any('cooldown' in l and '37.0' in l for l in tl), '\n'.join(tl)
        wr = [l for l in rep.split('WORK RATE')[1].split('SLOWDOWN')[0].splitlines()[1:] if l.strip()]
        assert len(wr) == 4 and all('frames ' in l and '/  50.0' in l for l in wr), '\n'.join(wr)
        sd = rep.split('SLOWDOWN')[1].split('COOLDOWN')[0]
        assert 'first minute: frames ' in sd and 'last minute : frames ' in sd and 'tok/s timings 50.0' in sd, sd
        cd = rep.split('COOLDOWN')[1].split('STOP REASON')[0]
        assert 'skin: start 35.0; not reached' in cd and 'BIG: start 36.0; within 2 degC after' in cd, cd
    elif name in ('skin', 'battery', 'status'):  # changed 3 s after camera start (load start is up to ~1.5 s later)
        assert 1.5 < load_s < 6.5, (name, load_s)
    elif name == 'cpu':
        hot = [s['t'] for s in samples if s.get('zones', {}).get(MID) == 110000 and s['t'] <= L1]
        assert len(hot) >= 3 and 1.5 < load_s < 6.5, (hot, load_s)
    elif name == 'noskin':
        assert 'VIRTUAL-SKIN in HAL temperatures: NO' in rep and 'skin: no start value' in rep, load_s
    elif name == 'skinlost':  # last skin reading before load start: the 5 s count from load start
        assert 5 <= load_s < 6.5 and 'VIRTUAL-SKIN in HAL temperatures: yes' in rep, load_s
    if name == 'sigterm':  # no cooldown: no read started after the stop (one in flight completes after it)
        assert all(s['t_start'] <= L1 for s in samples), [s['t_start'] - L1 for s in samples[-3:]]
    elif name == 'sensorfail':
        assert 4.5 < load_s < 8, load_s  # removed 0.5 s after camera start, stop 6 s after the last reading
    raw = [json.loads(l) for l in (out / 'thermalservice_raw.jsonl').read_text().splitlines()]
    dumps = [json.loads(l) for l in (out / 'thermalservice_5s.jsonl').read_text().splitlines()]
    assert raw and len(raw) == sum(d['raw_kept'] for d in dumps) and len(raw) < len(dumps), (len(raw), len(dumps))
    print(f'ok: {name}: stop "{run["stop_reasons"][0][:70]}" after {load_s:.1f} s of load, exit {proc.returncode}')


PORTS = {n: 18200 + i for i, n in enumerate(SCENARIOS)}


def main():
    unit()
    fail = 0
    names = list(SCENARIOS)
    with tempfile.TemporaryDirectory() as root:
        for batch in (names[i:i + 4] for i in range(0, len(names), 4)):  # 11 at once starve the phone's CPU
            procs = {}
            for name in batch:
                T = Path(root) / name
                make_env(T)
                procs[name] = (T, subprocess.Popen(
                    [sys.executable, __file__, '--child', str(T), 'duration' if name == 'sigterm' else name,
                     str(PORTS[name])], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
            if 'sigterm' in procs:  # during the load: 2 s after RobotCam was started
                am = procs['sigterm'][0] / 'am.log'
                for _ in range(600):
                    if am.exists() and any(l.startswith('start ') for l in am.read_text().splitlines()):
                        break
                    time.sleep(0.1)
                time.sleep(2)
                procs['sigterm'][1].send_signal(signal.SIGTERM)
            for name, (T, p) in procs.items():
                try:
                    output, _ = p.communicate(timeout=120)
                    check(name, T, p, output)
                except (AssertionError, subprocess.TimeoutExpired) as e:
                    fail = 1
                    p.kill()
                    output = p.communicate()[0] or ''
                    wd = (T / 'watchdog.txt').read_text() if (T / 'watchdog.txt').exists() else ''
                    print(f'FAIL: {name}: {type(e).__name__}: {e}\n{output[-2500:]}\n{wd[-3000:]}')
                    traceback.print_exc(file=sys.stdout)
    print('ALL OK' if not fail else 'FAILED')
    sys.exit(fail)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--fake-server']:
        fake_server(*sys.argv[2:])
    elif sys.argv[1:2] == ['--child']:
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main()
