#!/usr/bin/env python3
"""Termux-side check of the RobotCam service (android/robotcam).

Reads frame.jpg N times, one read per --interval seconds. Each read is classified by
read_frame(), which is also the reader rule a robot should follow. The reader contract is
the JPEG alone: session, frame counter and capture time come from its embedded comment
(`robotcam session=<id> frame=<n> capture_boot_ms=<b> capture_wall_ms=<w> clock=<c>`).
frame.jpg is replaced by a single rename, so one read always sees one whole frame.
frame.json is diagnostics only.

Age is computed on the boot clock: CLOCK_BOOTTIME now minus capture_boot_ms. The app takes
capture_boot_ms from the sensor timestamp (clock=sensor) or, if the camera's timestamps are
not on that base, from the frame's arrival time (clock=arrival). capture_wall_ms is for
humans only; wall-clock steps (NTP, manual changes) do not affect the age check.

  ok             JPEG decodes, comment parses, session is the expected one,
                 -0.2 s <= age <= --max-age
  missing        frame.jpg absent/unreadable, or the frame is older than --max-age
  bad            the JPEG does not decode, its comment is missing or malformed, or its age is
                 below -0.2 s (a capture time in the future: clock mismatch)
  other_session  the frame belongs to another session than the pinned one (the service was
                 restarted); the test pins the first session it sees

Anything but ok means "no usable frame": the robot must treat it as stop, never as clear.

Also reported for ok reads:
  repeat    same frame as the previous ok read (expected now and then: the reader and the
            writer run at independent phase)
  skipped   frame counter advanced by more than one between two ok reads
  read_ms / decode_ms / age_s   timing of the read, the full PIL decode, and frame age

--check-sidecar additionally reads frame.json after each ok read and flags SIDECAR_DIFF
when it describes another frame. This is expected now and then (the sidecar is renamed
after the JPEG) and never changes a read's status.

Timings are only valid with no agent resident (docs/WORKFLOW.md, "Timed benchmarks").

Usage: python robotcam_test.py [-n 60] [--interval 1.0] [--max-age 2.0] [--dir DIR]
                               [--check-sidecar] [--out FILE]
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
# Field lengths are bounded so a malformed comment can never produce a huge number.
COMMENT_RE = re.compile(rb'^robotcam session=([0-9a-f]{1,32}) frame=(\d{1,12}) '
                        rb'capture_boot_ms=(\d{1,15}) capture_wall_ms=(\d{1,15}) '
                        rb'clock=(sensor|arrival)$')
MIN_AGE = -0.2  # seconds; a younger (future) frame means the clocks disagree


def boot_now():
    """Seconds since boot including deep sleep: the base of SystemClock.elapsedRealtime()."""
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def read_frame(d, max_age, session=None):
    """Read and check frame.jpg. Returns a dict with 'status' and, when ok, 'image'.

    session: the expected session id; None accepts any session.
    """
    t0 = time.perf_counter()
    try:
        with open(os.path.join(d, 'frame.jpg'), 'rb') as f:
            data = f.read()
    except OSError as e:
        return {'status': 'missing', 'error': f'{type(e).__name__}: {e}'}
    t1 = time.perf_counter()
    try:
        im = Image.open(io.BytesIO(data))
        comment = im.info.get('comment', b'')
        im = im.convert('RGB')  # forces a full decode
    except (OSError, ValueError, SyntaxError) as e:
        return {'status': 'bad', 'error': f'decode {type(e).__name__}: {e}'}
    t2 = time.perf_counter()
    m = COMMENT_RE.match(comment) if isinstance(comment, bytes) else None
    if not m:
        return {'status': 'bad', 'error': f'jpeg comment {comment!r:.80}'}
    try:
        session_id = m.group(1).decode()
        frame = int(m.group(2))
        age_s = boot_now() - int(m.group(3)) / 1000.0
        clock = m.group(5).decode()
    except (ValueError, OverflowError) as e:
        return {'status': 'bad', 'error': f'jpeg comment {type(e).__name__}: {e}'}
    r = {
        'session': session_id,
        'frame': frame,
        'age_s': age_s,
        'clock': clock,
        'read_ms': (t1 - t0) * 1000,
        'decode_ms': (t2 - t1) * 1000,
        'size': f'{im.width}x{im.height}',
        'bytes': len(data),
    }
    if session is not None and r['session'] != session:
        r.update(status='other_session', error=f'session {r["session"]} != {session}')
    elif r['age_s'] < MIN_AGE:
        r.update(status='bad', error=f'age {r["age_s"]:.3f} s < {MIN_AGE} s (clock mismatch)')
    elif r['age_s'] > max_age:
        r.update(status='missing', error=f'old: age {r["age_s"]:.3f} s > {max_age} s')
    else:
        r.update(status='ok', image=im)
    return r


def sidecar_diff(d, r):
    """Diagnostic only: describe how frame.json differs from the JPEG read in r, or None."""
    try:
        with open(os.path.join(d, 'frame.json'), 'rb') as f:
            meta = json.loads(f.read())
        pair = (str(meta['session_id']), int(meta['frame']))
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as e:
        return f'sidecar unreadable ({type(e).__name__})'
    if pair != (r['session'], r['frame']):
        return f'sidecar {pair[0]}/{pair[1]}'
    return None


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
    ap.add_argument('--check-sidecar', action='store_true',
                    help='diagnostic: flag reads whose frame.json describes another frame')
    ap.add_argument('--out', help='write per-read results and summary as JSON here')
    args = ap.parse_args()

    d = args.dir or next((p for p in DEFAULT_DIRS if os.path.isdir(p)), DEFAULT_DIRS[0])
    print(f'dir {d}  n {args.n}  interval {args.interval}s  max age {args.max_age}s')

    results, prev, pinned = [], None, None
    counts = {'ok': 0, 'missing': 0, 'bad': 0, 'other_session': 0, 'repeat': 0, 'skipped': 0}
    if args.check_sidecar:
        counts['sidecar_diff'] = 0
    start = time.monotonic()
    for i in range(args.n):
        r = read_frame(d, args.max_age, pinned)
        r.pop('image', None)
        if r['status'] == 'ok' and pinned is None:
            pinned = r['session']
            print(f'     pinned session {pinned}')
        counts[r['status']] += 1
        flags = [] if r['status'] == 'ok' else [r['status'].upper() + ' ' + r['error']]
        if r['status'] == 'ok':
            if prev is not None:
                if r['frame'] == prev['frame']:
                    counts['repeat'] += 1
                    flags.append('REPEAT')
                elif r['frame'] > prev['frame'] + 1:
                    counts['skipped'] += r['frame'] - prev['frame'] - 1
                    flags.append(f'SKIPPED {r["frame"] - prev["frame"] - 1}')
            prev = r
            if args.check_sidecar:
                diff = sidecar_diff(d, r)
                if diff:
                    counts['sidecar_diff'] += 1
                    flags.append('SIDECAR_DIFF ' + diff)
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
            json.dump({'args': vars(args), 'dir': d, 'pinned_session': pinned,
                       'counts': counts, 'reads': results}, f, indent=1)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
