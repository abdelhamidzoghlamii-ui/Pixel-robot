#!/usr/bin/env python3
"""YOLO speed ladder: which YOLO family/size/input to retrain. Speed only. Research only: offline, no motors,
no main.py; the deployed model is only read.

Usage: yolo_speed.py --out DIR [--configs a,b] [--resume] [--thermal-log PATH] [--timed N] [--warmup N]
       yolo_speed.py --worker MODEL IMGSZ WARMUP TIMED      (one cold config; run with the robot's native python)

The parent (Debian python3) runs each config cold, in the ladder's style: wait for cores 4-7, thermal gate
(z9 <= idle + 4 degC), page-cache drop through the oneshot.sh handshake (ladder.make_cold), then a fresh
native-Termux python process -- the robot's interpreter and onnxruntime 1.25.1 -- that times WARMUP frames
untimed, then TIMED frames cycling the 20 frames in BENCH/frames (copies of bench_photos/, which capture_bench.py
took with main.take_photo's camera call; mapping in frames/SOURCES.txt). The worker does what detect_person.detect_scene does:
default InferenceSession (CPUExecutionProvider, ORT default threads), PIL open -> RGB -> exif_transpose ->
resize((imgsz, imgsz)) -> /255 CHW float32; post = per-class CONF filter + detect_person.nms, best box per class
(the 640-pixel position constants scaled to imgsz). Losing cores 4-7 during a config discards and redoes it.
Writes DIR/blocks.jsonl (one line per completed config, per-frame rows included) and DIR/report.txt.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BENCH = "/termux-home/yolo_bench"
FRAMES = [f"{BENCH}/frames/frame_{i:02d}.jpg" for i in range(1, 21)]
NATIVE_PY = "/data/data/com.termux/files/usr/bin/python"
NATIVE_ENV = {"LD_LIBRARY_PATH": "/data/data/com.termux/files/usr/lib"}
CONFIGS = {f"{fam}{s}_{sz}": (f"{BENCH}/models/{fam}{s}_{sz}.onnx", sz)
           for fam in ("yolo11", "yolo26") for s in "nsm" for sz in (320, 640)}
CONFIGS["deployed_yolo11m_640"] = ("/termux-home/robot/yolo11m.onnx", 640)
CONFIGS["deployed_yolo11m_640_int8dyn"] = (f"{BENCH}/models/yolo11m_deployed_int8dyn.onnx", 640)


def worker(model, imgsz, warmup, timed):
    """One cold config in this process. Prints one JSON line on stdout."""
    t0 = time.perf_counter()
    import numpy as np
    import onnxruntime as ort
    from PIL import Image, ImageOps
    sys.path.insert(0, "/termux-home/robot")
    from detect_person import CONF, IOU, nms
    import_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    session = ort.InferenceSession(model)  # as detect_person.get_session
    load_s = time.perf_counter() - t0
    name = session.get_inputs()[0].name
    third = imgsz / 3

    rows = []
    for i in range(warmup + timed):
        if not {4, 5, 6, 7} <= os.sched_getaffinity(0):
            print(json.dumps({"cores_lost": sorted(os.sched_getaffinity(0)), "frame_i": i}), flush=True)
            sys.exit(3)
        path = FRAMES[i % len(FRAMES)]
        t0 = time.perf_counter()
        img = Image.open(path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        img = img.resize((imgsz, imgsz))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis]
        t1 = time.perf_counter()
        out = session.run(None, {name: arr})[0][0].T
        t2 = time.perf_counter()
        by_class = {}
        for pred in out:
            scores = pred[4:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf > CONF:
                cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                by_class.setdefault(cls, []).append((box, conf, cx, cy, w, h))
        results = []
        for cls, items in by_class.items():
            keep = nms([it[0] for it in items], [it[1] for it in items], IOU)
            box, conf, cx, cy, w, h = items[keep[0]]
            area = (w * h) / (imgsz * imgsz)
            pos = 'left' if cx < third else 'right' if cx > 2 * third else 'center'
            dist = 'very close' if area > 0.3 else 'close' if area > 0.1 else 'medium' if area > 0.03 else 'far'
            results.append((cls, round(conf, 2), pos, dist))
        results.sort(key=lambda x: -x[1])
        t3 = time.perf_counter()
        rows.append({"i": i, "warmup": i < warmup, "frame": Path(path).name,
                     "pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3,
                     "total_ms": (t3 - t0) * 1e3, "n_det": len(results), "dets": results})
    hwm = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM:")) / 1024
    print(json.dumps({"import_s": import_s, "load_s": load_s, "peak_rss_mib": hwm, "ort": ort.__version__,
                      "providers": session.get_providers(), "rows": rows}), flush=True)


def run_config(name, a, thermal):
    import ladder
    model, imgsz = CONFIGS[name]
    therm = {}
    ladder.check_cores("before cold")
    if thermal:
        therm["start"] = ladder.thermal_gate(thermal[0], thermal[1], name)
        print(f"  [{name}] thermal start: {ladder.fmt_thermal(therm['start'])} (waited {therm['start']['waited_s']} s)",
              flush=True)
    load_state = ladder.make_cold(name, [model])
    ladder.check_cores("before starting worker")
    t0 = time.perf_counter()
    p = subprocess.run([NATIVE_PY, "-W", "ignore", __file__, "--worker", model, str(imgsz), str(a.warmup), str(a.timed)],
                       env={**os.environ, **NATIVE_ENV}, capture_output=True, text=True)
    wall_s = time.perf_counter() - t0
    (a.out / "logs" / f"{name}.stderr.txt").write_text(p.stderr)
    if p.returncode == 3:
        raise ladder.CoresLost(f"worker lost cores 4-7: {p.stdout.strip()}")
    if p.returncode != 0:
        raise SystemExit(f"[{name}] worker exit {p.returncode}; stderr in {a.out}/logs/{name}.stderr.txt")
    res = json.loads(p.stdout.strip().splitlines()[-1])
    if thermal:
        therm["end"] = ladder.read_thermal(thermal[0])
        print(f"  [{name}] thermal end:   {ladder.fmt_thermal(therm['end'])}", flush=True)
    ladder.check_cores("after worker")
    return {"config": name, "model": model, "imgsz": imgsz, "warmup": a.warmup, "timed": a.timed,
            "wall_s": wall_s, "load_state": load_state, "thermal": therm or None, **res}


def pct(xs, q):
    return statistics.median(xs) if q == 50 else statistics.quantiles(xs, n=100, method="inclusive")[q - 1]


def report(done, names):
    hdr = (f"{'config':<30}{'pre ms':>13}{'infer ms':>15}{'post ms':>11}{'total ms':>15}{'load s':>8}{'RSS MiB':>9}"
           f"{'dets/frame':>16}   thermal z9 start->end degC; load state")
    lines = ["median / P95 over the timed frames; dets/frame = mean [min-max]", hdr]
    for n in names:
        if n not in done:
            lines.append(f"{n:<30}not run")
            continue
        b = done[n]
        t = [r for r in b["rows"] if not r["warmup"]]
        mp = lambda k: f"{pct([r[k] for r in t], 50):.1f}/{pct([r[k] for r in t], 95):.1f}"
        d = [r["n_det"] for r in t]
        th = (f"{b['thermal']['start']['z9'] / 1000:.0f}->{b['thermal']['end']['z9'] / 1000:.0f}" if b["thermal"]
              else "n/a")
        lines.append(f"{n:<30}{mp('pre_ms'):>13}{mp('infer_ms'):>15}{mp('post_ms'):>11}{mp('total_ms'):>15}"
                     f"{b['load_s']:>8.2f}{b['peak_rss_mib']:>9.0f}"
                     f"{f'{statistics.mean(d):.2f} [{min(d)}-{max(d)}]':>16}   {th}; {b['load_state']['mode'].split(' (')[0]}")
    return "\n".join(lines)


def read_jsonl(p):
    """Rows of a JSONL file; an unterminated last line (a write cut off by a kill) is dropped, any other bad line
    raises."""
    if not p.exists():
        return []
    text = p.read_text()
    lines = text.split("\n")
    if lines[-1]:
        print(f"{p}: dropping torn last line", flush=True)
    return [json.loads(l) for l in lines[:-1]]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--thermal-log", default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=60)
    a = ap.parse_args()
    names = a.configs.split(",")
    if set(names) - set(CONFIGS):
        raise SystemExit(f"unknown configs {sorted(set(names) - set(CONFIGS))}")
    if a.resume and not a.out.is_dir():
        raise SystemExit(f"--resume: {a.out} does not exist")
    a.out.mkdir(parents=True, exist_ok=a.resume)
    (a.out / "logs").mkdir(exist_ok=True)
    sys.path.insert(0, "/termux-home/ladder")
    import ladder
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    common = lambda: {"yolo_speed_sha256": sha(__file__), "ladder_sha256": sha(ladder.__file__),
                      "detect_person_sha256": sha("/termux-home/robot/detect_person.py"),
                      "frames": {Path(f).name: sha(f) for f in FRAMES}, "warmup": a.warmup, "timed": a.timed}
    inputs = lambda n: {**common(), "model": [CONFIGS[n][0], sha(CONFIGS[n][0])]}  # re-hashed each call; stored in each block
    rows = read_jsonl(a.out / "blocks.jsonl") if a.resume else []
    for b in rows:  # every row checked before anything in DIR is rewritten
        n = b.get("config")
        if n not in CONFIGS or b.get("inputs") != inputs(n):
            raise SystemExit(f"--resume: completed {n} has no or different recorded inputs; not comparable")
        if (not {"load_s", "peak_rss_mib", "load_state", "thermal"} <= b.keys()
                or len([r for r in b.get("rows", []) if not r["warmup"]]) != a.timed):
            raise SystemExit(f"--resume: completed {n} is missing block fields or timed rows")
    done = {b["config"]: b for b in rows}
    if len(done) != len(rows):
        raise SystemExit("--resume: blocks.jsonl has duplicate configs")
    if a.resume:  # rewrite without a torn last line so appends start on a clean line
        (a.out / "blocks.jsonl.tmp").write_text("".join(json.dumps(b) + "\n" for b in done.values()))
        os.replace(a.out / "blocks.jsonl.tmp", a.out / "blocks.jsonl")
        print("resume: keeping " + (", ".join(done) or "nothing"), flush=True)
    (a.out / f"run_{stamp}.json").write_text(json.dumps(
        {**common(), "models": {n: inputs(n)["model"] for n in names}}, indent=1) + "\n")
    ladder.wait_cores("start")
    thermal = None
    if a.thermal_log:
        idle = ladder.read_thermal(a.thermal_log)
        print(f"thermal idle reading (gate z9 <= idle + {ladder.GATE_MC / 1000:.0f} degC): {ladder.fmt_thermal(idle)}",
              flush=True)
        thermal = (a.thermal_log, idle)
    with open(a.out / "blocks.jsonl", "a") as blog:
        for n in names:
            if n in done:
                print(f"[{n}] already completed, skipped (--resume)", flush=True)
                continue
            print(f"[{n}] cold: {a.warmup} warm-up + {a.timed} timed frames", flush=True)
            before = inputs(n)
            while True:
                ladder.wait_cores(n)
                try:
                    b = run_config(n, a, thermal)
                    break
                except ladder.CoresLost as e:
                    print(f"  [{n}] {e}: discarding this config's data and redoing it", flush=True)
            if inputs(n) != before:
                raise SystemExit(f"[{n}] inputs changed while it ran; its data is discarded")
            b["inputs"] = before
            blog.write(json.dumps(b) + "\n")
            blog.flush()
            done[n] = b
            print(report({n: b}, [n]).splitlines()[-1], flush=True)
    text = report(done, names)
    (a.out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
