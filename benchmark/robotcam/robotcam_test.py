#!/usr/bin/env python3
"""Termux-side check of the RobotCam service (android/robotcam).

Reads frame.json + frame.jpg N times, one read per --interval seconds. Each read is
classified by read_frame(), which is also the reader rule a robot should follow:

  ok        sidecar and JPEG agree on session_id and frame, and age <= --max-age
  missing   sidecar or JPEG absent/unreadable, or the frame is older than --max-age
  bad       sidecar malformed or partial, or the JPEG does not decode
  mismatch  JPEG comment and sidecar differ in session_id or frame (read during an update)

Anything but ok means "no usable frame": the robot must treat it as stop, never as clear.

Also reported for ok reads:
  repeat    same session and frame as the previous ok read (expected now and then: the
            reader and the writer both run at ~1 Hz with independent phase)
  skipped   frame counter advanced by more than one between two ok reads
  read_ms / decode_ms / age_s   timing of the read, the full PIL decode, and frame age

Timings are only valid with no agent resident (docs/WORKFLOW.md, "Timed benchmarks").

Usage: python robotcam_test.py [-n 60] [--interval 1.0] [--max-age 2.0] [--dir DIR] [--out FILE]
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
COMMENT_RE = re.compile(rb'session=([0-9a-f]+) frame=(\d+)')


def read_frame(d, max_age):
    """Read and check one frame. Returns a dict with 'status' and, when decoded, 'image'."""
    t0 = time.perf_counter()
    try:
        with open(os.path.join(d, 'frame.json'), 'rb') as f:
            raw = f.read()
        with open(os.path.join(d, 'frame.jpg'), 'rb') as f:
            data = f.read()
    except OSError as e:
        return {'status': 'missing', 'error': f'{type(e).__name__}: {e}'}
    t1 = time.perf_counter()
    try:
        meta = json.loads(raw)
        session = str(meta['session_id'])
        frame = int(meta['frame'])
        capture_ms = float(meta['capture_wall_ms'])
    except (ValueError, TypeError, KeyError) as e:
        return {'status': 'bad', 'error': f'sidecar {type(e).__name__}: {e}'}
    try:
        im = Image.open(io.BytesIO(data))
        comment = im.info.get('comment', b'')
        im = im.convert('RGB')  # forces a full decode
    except (OSError, ValueError, SyntaxError) as e:
        return {'status': 'bad', 'error': f'decode {type(e).__name__}: {e}'}
    t2 = time.perf_counter()
    r = {
        'session': session,
        'frame': frame,
        'age_s': time.time() - capture_ms / 1000.0,
        'read_ms': (t1 - t0) * 1000,
        'decode_ms': (t2 - t1) * 1000,
        'size': f'{im.width}x{im.height}',
        'bytes': len(data),
        'mode': meta.get('mode'),
    }
    m = COMMENT_RE.search(comment)
    if not m or m.group(1).decode() != session or int(m.group(2)) != frame:
        r.update(status='mismatch', error=f'jpeg comment {comment[:80]!r}')
    elif r['age_s'] > max_age:
        r.update(status='missing', error=f'old: age {r["age_s"]:.3f} s > {max_age} s')
    else:
        r.update(status='ok', image=im)
    return r


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
    ap.add_argument('--max-age', type=float, default=2.0,
                    help='frames older than this many seconds count as missing (default 2.0)')
    ap.add_argument('--dir', help='frame directory (default: first existing of %s)' % DEFAULT_DIRS)
    ap.add_argument('--out', help='write per-read results and summary as JSON here')
    args = ap.parse_args()

    d = args.dir or next((p for p in DEFAULT_DIRS if os.path.isdir(p)), DEFAULT_DIRS[0])
    print(f'dir {d}  n {args.n}  interval {args.interval}s  max age {args.max_age}s')

    results, prev = [], None
    counts = {'ok': 0, 'missing': 0, 'bad': 0, 'mismatch': 0, 'repeat': 0, 'skipped': 0}
    start = time.monotonic()
    for i in range(args.n):
        r = read_frame(d, args.max_age)
        r.pop('image', None)
        counts[r['status']] += 1
        flags = [] if r['status'] == 'ok' else [r['status'].upper() + ' ' + r['error']]
        if r['status'] == 'ok':
            if prev is not None and prev['session'] == r['session']:
                if r['frame'] == prev['frame']:
                    counts['repeat'] += 1
                    flags.append('REPEAT')
                elif r['frame'] > prev['frame'] + 1:
                    counts['skipped'] += r['frame'] - prev['frame'] - 1
                    flags.append(f'SKIPPED {r["frame"] - prev["frame"] - 1}')
            prev = r
        if 'frame' in r:
            print(f'{i + 1:4d} frame {r["frame"]:6d} {r["size"]} {r["bytes"]:6d} B  '
                  f'read {r["read_ms"]:6.1f} ms  decode {r["decode_ms"]:6.1f} ms  '
                  f'age {r["age_s"]:6.3f} s  {" ".join(flags)}')
        else:
            print(f'{i + 1:4d} {" ".join(flags)}')
        r['flags'] = flags
        results.append(r)
        # Fixed-rate schedule, independent of how long the read took.
        time.sleep(max(0.0, start + (i + 1) * args.interval - time.monotonic()))

    timed = [r for r in results if 'frame' in r]
    print('\nsummary')
    print('  counts    ' + '  '.join(f'{k} {v}' for k, v in counts.items()))
    print('  read_ms   ' + summary([r['read_ms'] for r in timed]))
    print('  decode_ms ' + summary([r['decode_ms'] for r in timed]))
    print('  age_s     ' + summary([r['age_s'] for r in timed]))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'dir': d, 'counts': counts, 'reads': results}, f, indent=1)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
