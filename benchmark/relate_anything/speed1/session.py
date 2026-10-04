"""Native Termux owner session: one idle, gated ordered blocks; dry-run is NOT VALID."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

from variants import (HERE, VARIANTS, CLUSTERS, NAME, dc, sb, require_quality,
                      planned_blocks, make_head, build, sha)


def repin_monitor(monitor, cpus):
    shell, _, root_pid, layout = monitor
    allowed = os.sched_getaffinity(0) - cpus
    sb.require_monitor_mask(allowed, cpus)
    for tid in shell.monitor_tids:
        os.sched_setaffinity(tid, allowed)
        sb.require_monitor_mask(os.sched_getaffinity(tid), cpus)
    mask = format(sum(1 << c for c in allowed), 'x')
    shell.run(f'taskset -p {mask} {shell.p.pid}')
    shell.run(f'taskset -p {mask} {root_pid}')
    sb.require_monitor_mask(os.sched_getaffinity(shell.p.pid), cpus)
    sb.require_monitor_mask(os.sched_getaffinity(root_pid), cpus)
    return shell, allowed, root_pid, layout


def can_continue(record):
    # Only a model's missed cadence is local. Root/sensors, thermal stops,
    # charger, process/cores loss, inference and cleanup failures end the session.
    return (record.get('error', '').startswith('RuntimeError: cadence missed/overrun')
            or record.get('error', '').startswith('RuntimeError: call cadence overran block')) and not (
                record.get('cleanup_error') or record.get('monitor_errors'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--session', action='store_true', help='fixed owner-ordered session (default)')
    ap.add_argument('--variant', choices=VARIANTS)
    ap.add_argument('--cluster', choices=CLUSTERS, default='MID')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.session and args.variant:
        ap.error('--session and --variant are mutually exclusive')
    if sys.platform != 'android':
        ap.error('native Termux required')
    sb.load_speed_input()  # validate saved image/boxes before any session idle
    quality = json.loads((HERE / 'quality.json').read_text())
    blocks = [(args.variant, args.cluster)] if args.variant else planned_blocks(quality)
    for variant, cluster in blocks:
        require_quality(variant)
        cpus, _ = CLUSTERS[cluster]
        if not cpus <= os.sched_getaffinity(0):
            raise RuntimeError(f'{cluster}: required inference CPUs unavailable')
        if not args.dry_run:
            sb.require_monitor_mask(os.sched_getaffinity(0) - cpus, cpus)
    output = args.output
    if output.exists():
        ap.error('output exists; choose a new evidence filename')
    block_paths = [output.with_name(output.stem + f'_block_{i:02d}_{v}_{c}.json')
                   for i, (v, c) in enumerate(blocks, 1)]
    if any(p.exists() for p in block_paths):
        ap.error('a block output exists; choose a new evidence filename')
    minimum = 300 + 180 * len(blocks)
    maximum = minimum + sb.pm.GATE_MAX_S * len(blocks)
    plan = [{'block': i, 'variant': v, 'cluster': c, 'cpus': sorted(CLUSTERS[c][0]),
             'threads': CLUSTERS[c][1], 'output': str(p)}
            for i, ((v, c), p) in enumerate(zip(blocks, block_paths), 1)]
    print('Planned blocks:', json.dumps(plan), flush=True)
    print(f'Estimated owner session: {minimum / 60:.1f} min + loads/setup; up to {maximum / 60:.1f} min with all gates at 8 min.', flush=True)
    result = {'label': dc.LABEL + ' — SESSION DRY RUN' if args.dry_run else 'SESSION STARTING',
              'plan': plan, 'estimated_minimum_s': minimum, 'estimated_with_max_gates_s': maximum,
              'quality_sha256': sha(HERE / 'quality.json'), 'blocks': [],
              'continuation_rule': 'Continue only a cadence failure with successful cleanup and no monitor errors; all shared-state, thermal, charger, process, affinity, sensor, root or inference failures stop later blocks.'}
    lock_path = dc.HOME / '.cache/relate_anything/speed.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    screen, monitor = sb.Screen(), None
    context = {}  # supplied even for dry-run to keep one session lock
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            dc.write_json(output, result)  # refuse an unwritable output before idle
            if not args.dry_run:
                sb.cr.require_native()
                sb.processes_clear()
                sb.cr.check_cores('before session idle')
                sb.battery_sample()
                reading = sb.cr.read_dump()
                if reading['skin'] is None or reading['status'] is None or reading['status'] >= sb.cr.STATUS_STOP:
                    raise RuntimeError('pre-idle skin/status missing or thermal stop')
                # Verify every engine, thread count and affinity before spending idle time.
                for variant, cluster in dict.fromkeys(blocks):
                    head, caller, tids = build(variant, cluster)
                    try:
                        sb.check_pinning(tids, CLUSTERS[cluster][0])
                    finally:
                        caller.close()
                    head = caller = None
                monitor = sb.prepare_monitor(CLUSTERS[blocks[0][1]][0])
                for cluster in dict.fromkeys(c for _, c in blocks):
                    monitor = repin_monitor(monitor, CLUSTERS[cluster][0])
                monitor = repin_monitor(monitor, CLUSTERS[blocks[0][1]][0])
                screen.start()
                print('Idling 300 s once; unplugged, screen on, Termux foreground', flush=True)
                for _ in range(60):
                    time.sleep(5)
                    sb.processes_clear()
                    sb.battery_sample()
                    sb.cr.check_cores('during session idle')
                idle = sb.cr.read_dump()
                if idle['skin'] is None or idle['status'] is None:
                    raise RuntimeError('no idle skin/status')
                context.update(idle=idle, monitor=monitor)
                result['idle_absolute_monotonic'] = idle
            else:
                print(dc.LABEL + ' — two calls per block, no idle/gates/sensors', flush=True)
            for spec in plan:
                variant, cluster = spec['variant'], spec['cluster']
                try:
                    require_quality(variant)
                    cpus, threads = CLUSTERS[cluster]
                    if not args.dry_run:
                        monitor = repin_monitor(monitor, cpus)
                        context['monitor'] = monitor
                    block_args = SimpleNamespace(model=NAME, dry_run=args.dry_run, output=Path(spec['output']),
                                                 variant=variant, cluster=cluster)
                    record = sb.run_block(block_args, cpus=cpus, threads=threads,
                        head_factory=lambda name, threads: make_head(variant, threads), session=context)
                except BaseException as error:
                    if Path(spec['output']).exists():
                        record = json.loads(Path(spec['output']).read_text())
                    else:
                        record = {**spec, 'label': 'NOT VALID — REFUSED', 'validity': 'NOT VALID — REFUSED',
                                  'error': f'{type(error).__name__}: {error}', 'calls': [], 'power': [],
                                  'power_summary': sb.power_summary([], [], 0)}
                        dc.write_json(Path(spec['output']), record)
                    result['blocks'].append(record)
                    dc.write_json(output, result)
                    raise
                result['blocks'].append(record)
                dc.write_json(output, result)
                if record['label'] == 'NOT VALID — INCOMPLETE' and not can_continue(record):
                    raise RuntimeError('block invalidates later blocks')
            result['label'] = (dc.LABEL + ' — SESSION DRY RUN' if args.dry_run else
                               'SESSION COMPLETE; inspect every block validity and warm-start flags')
        except BaseException as error:
            result.update(label='NOT VALID — SESSION INCOMPLETE', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            try:
                if monitor: monitor[0].close()
            except BaseException as error:
                result.update(label='NOT VALID — SESSION INCOMPLETE', cleanup_error=str(error))
            try:
                screen.restore()
            except BaseException as error:
                result.update(label='NOT VALID — SESSION INCOMPLETE', cleanup_error=str(error))
            result['unrun_blocks'] = plan[len(result['blocks']):]
            dc.write_json(output, result)
            print('Session label:', result['label'], '\nEvidence:', output, flush=True)
    if result['label'] == 'NOT VALID — SESSION INCOMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, sb.cr.exit_on_signal)
    main()
