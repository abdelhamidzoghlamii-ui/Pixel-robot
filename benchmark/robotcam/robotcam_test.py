#!/usr/bin/env python3
"""Termux-side check of the RobotCam service (android/robotcam).

Reads frame.json + frame.jpg N times, one read per --interval seconds, and reports:
  read_ms    reading the sidecar and the JPEG from shared storage
  decode_ms  full PIL decode of the JPEG to RGB
  age_s      now - capture_wall_ms from the sidecar
  missing    sidecar or JPEG absent/unreadable, or the JPEG does not decode
  repeat     same frame counter as the previous read (expected now and then: the reader and
             the writer both run at ~1 Hz with independent phase)
  stale      age above --stale
  mismatch   the JPEG's embedded frame number differs from the sidecar's
  skipped    frame counter advanced by more than one between two reads

Timings are only valid with no agent resident (docs/WORKFLOW.md, "Timed benchmarks").

Usage: python robotcam_test.py [-n 60] [--interval 1.0] [--stale 2.0] [--dir DIR] [--out FILE]
"""
import argparse
import io
import json
import os
import re
import statistics
import time

from PIL import Image

DEFAULT_DIRS = [os.path.expanduser('~/storage/downloads/robotcam'), '/sdcard/Download/robotcam']


def read_once(d):
    """Return a result dict for one read of the sidecar and image."""
    t0 = time.perf_counter()
    try:
        with open(os.path.join(d, 'frame.json')) as f:
            meta = json.load(f)
        with open(os.path.join(d, 'frame.jpg'), 'rb') as f:
            data = f.read()
    except (OSError, ValueError) as e:
        return {'status': 'missing', 'error': f'{type(e).__name__}: {e}'}
    t1 = time.perf_counter()
    try:
        im = Image.open(io.BytesIO(data))
        comment = im.info.get('comment', b'')
        im = im.convert('RGB')  # forces a full decode
    except OSError as e:
        return {'status': 'missing', 'error': f'decode: {e}'}
    t2 = time.perf_counter()
    m = re.search(rb'frame=(\d+)', comment)
    return {
        'status': 'ok',
        'frame': meta['frame'],
        'session': meta['session_start_ms'],
        'jpeg_frame': int(m.group(1)) if m else None,
        'age_s': time.time() - meta['capture_wall_ms'] / 1000.0,
        'read_ms': (t1 - t0) * 1000,
        'decode_ms': (t2 - t1) * 1000,
        'size': f'{im.width}x{im.height}',
        'bytes': len(data),
    }


def summary(values):
    if not values:
        return 'n/a'
    v = sorted(values)
    p90 = v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))]
    return f'median {statistics.median(v):.3f}  p90 {p90:.3f}  max {v[-1]:.3f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-n', type=int, default=60, help='number of reads (default 60)')
    ap.add_argument('--interval', type=float, default=1.0, help='seconds between reads')
    ap.add_argument('--stale', type=float, default=2.0, help='age in seconds counted as stale')
    ap.add_argument('--dir', help='frame directory (default: first existing of %s)' % DEFAULT_DIRS)
    ap.add_argument('--out', help='write per-read results and summary as JSON here')
    args = ap.parse_args()

    d = args.dir or next((p for p in DEFAULT_DIRS if os.path.isdir(p)), DEFAULT_DIRS[0])
    print(f'dir {d}  n {args.n}  interval {args.interval}s  stale > {args.stale}s')

    results, prev = [], None
    counts = {'ok': 0, 'missing': 0, 'repeat': 0, 'stale': 0, 'mismatch': 0, 'skipped': 0}
    start = time.monotonic()
    for i in range(args.n):
        r = read_once(d)
        flags = []
        if r['status'] == 'missing':
            counts['missing'] += 1
            flags.append('MISSING ' + r['error'])
        else:
            counts['ok'] += 1
            same_session = prev is not None and prev['session'] == r['session']
            if same_session and r['frame'] == prev['frame']:
                counts['repeat'] += 1
                flags.append('REPEAT')
            if r['age_s'] > args.stale:
                counts['stale'] += 1
                flags.append('STALE')
            if same_session and r['frame'] > prev['frame'] + 1:
                counts['skipped'] += r['frame'] - prev['frame'] - 1
                flags.append(f'SKIPPED {r["frame"] - prev["frame"] - 1}')
            if r['jpeg_frame'] != r['frame']:
                counts['mismatch'] += 1
                flags.append(f'MISMATCH jpeg={r["jpeg_frame"]}')
            prev = r
            print(f'{i + 1:4d} frame {r["frame"]:6d} {r["size"]} {r["bytes"]:6d} B  '
                  f'read {r["read_ms"]:6.1f} ms  decode {r["decode_ms"]:6.1f} ms  '
                  f'age {r["age_s"]:6.3f} s  {" ".join(flags)}')
        if r['status'] == 'missing':
            print(f'{i + 1:4d} {" ".join(flags)}')
        r['flags'] = flags
        results.append(r)
        # Fixed-rate schedule, independent of how long the read took.
        time.sleep(max(0.0, start + (i + 1) * args.interval - time.monotonic()))

    ok = [r for r in results if r['status'] == 'ok']
    print('\nsummary')
    print('  counts    ' + '  '.join(f'{k} {v}' for k, v in counts.items()))
    print('  read_ms   ' + summary([r['read_ms'] for r in ok]))
    print('  decode_ms ' + summary([r['decode_ms'] for r in ok]))
    print('  age_s     ' + summary([r['age_s'] for r in ok]))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'dir': d, 'counts': counts, 'reads': results}, f, indent=1)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
