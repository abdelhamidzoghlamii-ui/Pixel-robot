#!/data/data/com.termux/files/usr/bin/python
"""CAMPOWER2, camera only. Human launches oneshot.sh in native Termux."""
import argparse
import json
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent.parent), str(HERE.parent / 'duty_cycle')]
import duty_cycle as dc
cr, pm = dc.cr, dc.pm
from PIL import Image
import numpy as np  # already used by the helper modules; no new dependency
import robotcam_reader as reader

OUT_ROOT = cr.HOME / 'camera_power'
DEFAULTS = dict(capture_template='still', processing='default', focus_diopters=None,
                frame_ms=0, camera_id='', dump_characteristics=False)
TYPES = dict(capture_template='--es', processing='--es', focus_diopters='--ef',
             frame_ms='--ei', camera_id='--es', dump_characteristics='--ez')


def blocks_for(smoke=False, camera_id=None, blocks=None):
    variants = dict(baseline_first={}, manual_200=dict(frame_ms=200),
                    manual_500=dict(frame_ms=500), manual_1000=dict(frame_ms=1000),
                    mode_A_manual_1000=dict(frame_ms=1000), mode_A_manual_500=dict(frame_ms=500),
                    mode_A={}, baseline_last={}, preview=dict(capture_template='preview'),
                    record=dict(capture_template='record'), fast=dict(processing='fast'),
                    off=dict(processing='off'), focus_1m=dict(focus_diopters=1.0),
                    dump=dict(dump_characteristics=True))
    default = list(variants)[:8]
    if camera_id is not None:
        if not camera_id or len(camera_id) > 128:
            raise ValueError('camera ID must be nonempty and <=128 characters')
        variants['other_camera'] = dict(camera_id=camera_id)
    names = default if blocks is None else blocks.split(',')
    if not names or len(names) != len(set(names)) or any(n not in variants for n in names):
        raise ValueError('--blocks needs distinct comma-separated names: '+','.join(variants))
    return [dict(name=n, extras=variants[n], mode='A' if n.startswith('mode_A') else 'B', rate=1,
                 duration=20 if smoke else 120) for n in names]


def extras_args(spec):
    args = []
    # Baselines deliberately pass NO extras, including the existing mode/rate.
    if spec['mode'] == 'A':
        args += ['--es', 'mode', 'A']
    for key, value in spec['extras'].items():
        args += [TYPES[key], key, str(value).lower() if isinstance(value, bool) else str(value)]
    return args


def check_build(meta, spec):
    if (not isinstance(meta, dict) or not isinstance(meta.get('variant'), str) or
            meta.get('power', {}).get('schema_version') != 2):
        raise RuntimeError('RobotCam is not the CAMPOWER2 test build: frame.json lacks CAMPOWER2 schema_version=2. Install the robotcam-camera-power APK.')
    expected = {**DEFAULTS, **spec['extras']}
    if meta.get('power', {}).get('options') != expected or meta.get('mode') != spec['mode'] or meta.get('rate') != 1:
        raise RuntimeError('RobotCam did not apply the requested variant options/mode/rate')


def diagnostic(r, spec):
    """Sidecar may lag JPEG; never use a mismatched pair as applied-value evidence."""
    try:
        meta = json.loads((Path(reader.FRAME_DIR) / 'frame.json').read_text())
    except (OSError, ValueError):
        return None
    # Refuse an old installed build even if its diagnostics are not paired yet.
    check_build(meta, spec)
    if (meta.get('session_id'), meta.get('frame')) != (r['session'], r['frame']):
        return None
    return meta


def image_metrics(image):
    """Upright RGB -> 320x320 bilinear gray; 4-neighbour Laplacian, interior only."""
    gray = np.asarray(image.convert('L').resize((320, 320), Image.Resampling.BILINEAR), dtype=np.float64)
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
           - 4 * gray[1:-1, 1:-1])
    return dict(luma=float(gray.mean()), sharpness=float(lap.var()))


def read_sample(spec, pinned, last, started_boot):
    # The reader reads JPEG once; save its validated upright decoded image.
    r = reader.read_frame(session=pinned, min_capture_boot_s=started_boot)
    if r['status'] != 'ok':
        return dict(status=r['status']), None, pinned, last
    if last is not None and r['frame'] <= last:
        return dict(status='repeat'), None, pinned, last
    meta = diagnostic(r, spec)
    row = dict(status='ok', session=r['session'], frame=r['frame'], age_s=r['age_s'],
               diagnostics=meta, **image_metrics(r['image']))
    # Save the validated upright image (re-encoded quality 95), not a second path read.
    return row, r['image'], r['session'], r['frame']


def poll_counts(reads):
    """Startup absence is latency, not a failed published frame. Keep raw statuses."""
    first = next((i for i, r in enumerate(reads) if r['status'] == 'ok'), len(reads))
    failed = lambda rows: sum(r['status'] not in ('ok', 'repeat') for r in rows)
    return dict(startup_wait_polls=failed(reads[:first]), frames_failed=failed(reads[first:]),
                raw_status_counts={status: sum(r['status'] == status for r in reads)
                                   for status in sorted({r['status'] for r in reads})})


def summary(b):
    good = [r for r in b['reads'] if r['status'] == 'ok']
    pair = [r['diagnostics'] for r in good if r.get('diagnostics')]
    stats = lambda key: dict(median=statistics.median([r[key] for r in good]),
                            min=min(r[key] for r in good)) if good else None
    cpu = dc.cpu_seconds(b['cpu_snapshots'], b['boot_start'])
    # Never display an absent group as zero CPU usage.
    for group in ('robotcam_app', 'camera_provider'):
        if not any(s['groups'][group] for s in b['cpu_snapshots']):
            cpu[group] = None
    counts = poll_counts(b['reads'])
    gaps = [y['t']-x['t'] for x, y in zip(good, good[1:])]
    return dict(mean_battery_w=dc.mean_power(b['power'], [(b['power'][0]['t'], b['duration_s'])]),
                cpu_seconds=cpu, start_to_first_fresh_s=b['first_frame_s'], stop=b['stop'],
                frames_ok=len(good), **counts,
                new_frame_gap_max_s=max(gaps) if gaps else None,
                observed_publish_hz=(len(good)-1)/(good[-1]['t']-good[0]['t']) if len(good)>1 else None,
                unpaired_diagnostics=len(good)-len(pair),
                repeat_polls=sum(r['status'] == 'repeat' for r in b['reads']),
                age_median_s=cr.med([r['age_s'] for r in good]), age_p95_s=cr.p95([r['age_s'] for r in good]),
                luma=stats('luma'), sharpness=stats('sharpness'), paired_diagnostics=len(pair),
                applied_observations=[dict(frame=m['frame'], variant=m['variant'], power=m['power']) for m in pair],
                skin_slope_c_min=pm.skin_slope(b['dumps'], b['duration_s']),
                android_status=pm.status_changes(b), gate=b['gate'], heat_stop=b['heat_stop'])


def problems(b, s):
    bad = []
    if b['heat_stop']:
        bad.append('block stopped at limit: '+str(b['heat_stop']))
    if b['duration_s'] < b['spec']['duration'] - .1:
        bad.append('short block')
    if s['mean_battery_w'] is None:
        bad.append('battery coverage missing or gap >1.5 s')
    if any(r.get('battery_status') != 'Discharging' or 'error' in r
           for r in b['power'] if 0 <= r['t'] <= b['duration_s']):
        bad.append('battery error/missing/not Discharging')
    if any(x['errors'] for x in b['cpu_snapshots']):
        bad.append('CPU snapshot error')
    for group in ('camera_provider', 'robotcam_app'):
        if s['cpu_seconds'][group] is None:
            bad.append(group+' CPU missing')
    if any(not x['groups']['camera_provider'] for x in b['cpu_snapshots']):
        bad.append('camera provider absent in a CPU snapshot')
    if b['first_frame_s'] is not None and any(not x['groups']['robotcam_app']
            for x in b['cpu_snapshots'] if x['t'] >= b['started_monotonic']+b['first_frame_s']):
        bad.append('RobotCam absent in a post-start CPU snapshot')
    post_start_bad = [r for r in b['reads'] if b['first_frame_s'] is not None and
                      r['t'] > b['first_frame_s'] and r['status'] not in ('ok', 'repeat')]
    if not s['frames_ok'] or post_start_bad or not s['paired_diagnostics']:
        bad.append('camera frames failed/missing or no paired diagnostics')
    gap_limit = 1/b['spec']['rate'] + min(b['spec']['extras'].get('frame_ms', 0)/1000, .5) + .35
    if s['new_frame_gap_max_s'] is not None and s['new_frame_gap_max_s'] > gap_limit:
        bad.append('publication gap exceeds rate/frame-duration tolerance')
    if len(b['saved_frames']) != 3:
        bad.append('three saved frames missing')
    if s['skin_slope_c_min'] is None or not b['dumps'] or any(d.get('status') is None for d in b['dumps']):
        bad.append('skin slope or Android status missing')
    if b['stop'].get('pids_after_force_stop') or cr.camera_end_failed(b['stop']):
        bad.append('camera cleanup failed')
    # A HAL that ignores the requested control is a valid negative experiment,
    # but never call it a successfully applied slower stream.
    if b['spec']['extras'].get('frame_ms') and not any(
            r.get('power', {}).get('requested', {}).get('ae_mode') == 0 and
            r.get('power', {}).get('result', {}).get('ae_mode') == 0
            for r in s['applied_observations']):
        bad.append('manual variant never produced an observed AE OFF result')
    return bad


def boundary_monitor(period, read, rows, stop, cpu=False):
    """One worker owns periodic AND final accounting; never delays camera teardown."""
    try:
        cr.monitor_loop(period, read, rows, stop)
        rows.append(read())
    except Exception as e:
        row = dict(t=time.monotonic(), error=f'{type(e).__name__}: {e}')
        if cpu:
            row.update(groups={n: [] for n in ('runner', 'llama_server', 'robotcam_app', 'camera_provider')},
                       errors=[row['error']])
        rows.append(row)


def run_block(spec, ctx):
    name = spec['name']
    gate = pm.thermal_gate(ctx['thermal_log'], ctx['idle'], name, ctx['smoke'])
    cr.check_cores('before '+name)
    fast, dumps, power, cpu, reads = [], [], [], [], []
    saved = {}
    stop = threading.Event()
    boot_start = time.clock_gettime(time.CLOCK_BOOTTIME)
    cpu.append(dc.cpu_snapshot())
    t0 = time.monotonic()
    power.append(dc.battery_sample(ctx['battery_shell'], t0))
    monitors = [threading.Thread(target=cr.monitor_loop, args=(period, read, rows, stop), daemon=True)
                for period, read, rows in (
                    (1, lambda: cr.fast_sample(ctx['shell'], cr.fast_keys(ctx['layout'])), fast),
                    (5, cr.read_dump, dumps))]
    monitors += [threading.Thread(target=boundary_monitor, args=(period, read, rows, stop, is_cpu), daemon=True)
                 for period, read, rows, is_cpu in (
                     (.5, lambda: dc.battery_sample(ctx['battery_shell'], t0), power, False),
                     (5, dc.cpu_snapshot, cpu, True))]
    for thread in monitors:
        thread.start()
    first, pinned, last, hit = None, None, None, None
    stopped = None
    try:
        cr.am('start', '-n', 'com.pixelrobot.robotcam/.StartActivity', *extras_args(spec))
        while time.monotonic() < t0+spec['duration']:
            cr.check_cores('during '+name)
            if (hit := cr.block_limit(time.monotonic(), t0, fast, dumps)):
                break
            row, image, pinned, last = read_sample(spec, pinned, last, boot_start)
            row['t'] = time.monotonic()-t0
            reads.append(row)
            if row['status'] == 'ok':
                first = row['t'] if first is None else first
                for label, threshold in [('start', 0), ('middle', spec['duration']/2), ('end', spec['duration']-2)]:
                    if label not in saved and row['t'] >= threshold:
                        path = ctx['out'] / f'{name}_{label}.jpg'
                        image.save(path, quality=95)
                        saved[label] = dict(path=path.name, session=pinned, frame=last, t=row['t'])
            elif first is not None and row['status'] == 'other_session':
                raise RuntimeError('RobotCam restarted during block; refusing mixed-session metrics')
            time.sleep(.1)
    finally:
        duration = time.monotonic()-t0
        stop.set()
        # STOP and end-check/force-stop start immediately, even when accounting
        # is stalled or fails. Boundary accounting runs independently in its owner.
        try:
            stopped = dc.stop_camera()
            if why := cr.camera_end_failed(stopped):
                raise RuntimeError(why)
        finally:
            for thread in monitors:
                thread.join(timeout=60)
            if any(thread.is_alive() for thread in monitors):
                raise RuntimeError('monitor still running; camera teardown already attempted')
    for rows in (fast, dumps):
        for row in rows:
            row['t'], row['t_start'] = row['t']-t0, row['t_start']-t0
    cr.check_cores('after '+name)
    return dict(block=name, spec=spec, duration_s=duration, boot_start=boot_start, started_monotonic=t0,
                first_frame_s=first, stop=stopped, gate=gate, heat_stop=hit, fast=fast,
                dumps=dumps, power=power, cpu_snapshots=cpu, reads=reads, saved_frames=saved)


def write_report(out, specs, smoke):
    lines = ['CAMPOWER2'+(' — SMOKE: not a measurement' if smoke else ''),
             'Camera-only, no detector/model/selector calls. Helper imports load modules only.',
             'Power includes camera startup; STOP/check/force-stop outside block. CPU boundary-query overhead uncorrected.',
             'Image metrics: upright image stretched to 320x320 bilinear grayscale; saved images re-encoded quality 95.',
             'frames_failed counts post-start non-ok/non-repeat polls; startup_wait_polls is pre-first-frame latency.',
             'missing means absent, age >2 s, or predates start; bad means decode/comment/clock failure. Repeats and unpaired sidecars are separate.',
             'Sharpness may rise with noise; inspect saved images. Latest result may describe preview/previous frame.',
             'Battery 0.5 s; fuel-gauge averaging unverified; gaps >1.5 s give n/a. Zone9/10/11 are fault sensors.']
    all_bad = []
    for spec in specs:
        path = out / f'block_{spec["name"]}.json'
        if not path.exists():
            all_bad.append(spec['name']+': missing block')
            continue
        b = json.loads(path.read_text())
        s = summary(b)
        bad = problems(b, s)
        all_bad.extend(spec['name']+': '+p for p in bad)
        lines += ['', spec['name']+(' INCOMPLETE' if bad else ' completed'), json.dumps(s, indent=2)]
    lines += ['', *['INCOMPLETE: '+p for p in all_bad]]
    (out / 'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    return all_bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--camera-id', help='enumerated BACK ID for --blocks other_camera; read dump first')
    ap.add_argument('--blocks', help='comma-separated block names in execution order; default: baseline_first,manual_200,manual_500,manual_1000,mode_A_manual_1000,mode_A_manual_500,mode_A,baseline_last')
    ap.add_argument('--thermal-log', default=str(OUT_ROOT / 'thermal.log'))
    a = ap.parse_args(argv)
    cr.require_native()  # before ANY Android/root action
    try:
        specs = blocks_for(a.smoke, a.camera_id, a.blocks)
    except ValueError as e:
        ap.error(str(e))
    block_s = sum(s['duration'] for s in specs)
    print(f'Blocks: {len(specs)}, {block_s/60:.1f} min. Wall estimate including idle: '
          f'{5+block_s/60:.1f}–{5+block_s/60+(0 if a.smoke else 8*len(specs)):.1f} min + startup/stop checks.', flush=True)
    print('SMOKE: not a measurement' if a.smoke else '120 s camera blocks; baselines first and last', flush=True)
    if a.resume:
        out = a.resume
        old = json.loads((out / 'run.json').read_text())
        if (old['smoke'], old['blocks']) != (a.smoke, specs):
            raise SystemExit('--resume: smoke/camera-id/--blocks list mismatch')
    else:
        out = OUT_ROOT / f'run_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}{"_smoke" if a.smoke else ""}'
        out.mkdir(parents=True)
    ctx = dict(out=out, smoke=a.smoke, thermal_log=a.thermal_log)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, cr.exit_on_signal)
    try:
        ctx['shell'], ctx['battery_shell'] = cr.RootShell(), cr.RootShell()
        ctx['layout'] = cr.discover(ctx['shell'])
        ctx['idle'] = old['idle'] if a.resume else {**cr.read_thermal(a.thermal_log), 'skin': cr.read_dump()['skin']}
        if ctx['idle']['skin'] is None:
            raise SystemExit('no idle VIRTUAL-SKIN')
        meta = dict(smoke=a.smoke, blocks=specs, idle=ctx['idle'], layout=ctx['layout'], started=cr.utc(),
                    sample_interval_s=.5, sha256={str(p): cr.sha256(p) for p in
                    (Path(__file__), Path(dc.__file__), Path(pm.__file__), Path(cr.__file__), Path(reader.__file__), HERE/'launcher.py')})
        if a.resume and meta['sha256'] != old['sha256']:
            raise SystemExit('--resume: code/helper hashes changed; start a new run')
        (out / (f'resume_{time.time_ns()}.json' if a.resume else 'run.json')).write_text(json.dumps(meta, indent=2))
        if why := cr.camera_end_failed(dc.stop_camera()):
            raise RuntimeError('initial camera cleanup: '+why)
        for spec in specs:
            if dc.completed(out, spec):
                saved = json.loads((out / f'block_{spec["name"]}.json').read_text())
                if saved.get('heat_stop') and saved['heat_stop'][0] == 'fail_closed':
                    raise RuntimeError('--resume: saved fail-closed sensor stop; repair monitoring and move that block JSON aside before retrying')
                print(spec['name']+': already completed, kept', flush=True)
                continue
            while True:
                cr.wait_cores(spec['name'])
                try:
                    b = run_block(spec, ctx)
                    break
                except cr.CoresLost as e:
                    print(f'{e}; discard and redo {spec["name"]}', flush=True)
                    if why := cr.camera_end_failed(dc.stop_camera()):
                        raise RuntimeError(why)
            dc.save_block(out, spec, b)
            if b['heat_stop'] and b['heat_stop'][0] == 'fail_closed':
                write_report(out, specs, a.smoke)
                raise RuntimeError('fail-closed sensor stop: block retained; remaining variants not started')
            if spec['extras'].get('dump_characteristics'):
                path = Path(reader.FRAME_DIR) / 'characteristics.json'
                try:
                    text = path.read_text()
                    json.loads(text)
                    (out / 'characteristics.json').write_text(text)
                except (OSError, ValueError) as e:
                    raise RuntimeError('characteristics dump missing/unreadable') from e
            # Refuse before spending another gate/block when no test-build frames appeared.
            if b['first_frame_s'] is None:
                raise RuntimeError('RobotCam produced no fresh frames; test build/variant unavailable. See camera notification.')
        if write_report(out, specs, a.smoke):
            raise SystemExit('RUN INCOMPLETE: see report.txt')
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            if why := cr.camera_end_failed(dc.stop_camera()):
                raise RuntimeError('final camera cleanup: '+why)
        finally:
            for key in ('shell', 'battery_shell'):
                if ctx.get(key):
                    ctx[key].close()


if __name__ == '__main__':
    main()
