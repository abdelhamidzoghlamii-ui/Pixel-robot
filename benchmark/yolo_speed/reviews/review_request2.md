ROUND 2 (fresh session). Round 1 verdict was CHANGES REQUIRED with two findings: (1) benchmark/* is gitignored so the scripts were untrackable; (2) --resume did not validate input hashes. Both are addressed below. Round 1 also reported the sandbox denied reading source files, so the full sources are INLINED here. Review the whole candidate again, not only the fixes.

BASE: main 825152b75fff241429acf69eccbbf01e131a7b8d, uncommitted candidate. The two untracked files under benchmark/llm_objective_setting/conversation/ predate this task and are NOT part of it.

## TASK
Goal: prepare a SHORT unattended YOLO speed benchmark on the phone to choose which YOLO family/size/input to
retrain. Speed only; accuracy is a later step on the owner's photos.
1. Find how the robot runs YOLO today (detect_person.py): runtime and version, execution provider, thread
   settings, preprocessing, NMS, how frames are captured. The benchmark must use the same runtime and settings.
2. Exports: stock yolo11n/s/m and yolo26n/s/m (Ultralytics weights) to ONNX at imgsz 320 and 640, same export
   options the deployed file appears to use. Also make an int8 dynamic-quantized copy of the deployed
   yolo11m.onnx. Do not touch the deployed file. Put all outputs in /termux-home/yolo_bench/models/ with a
   SHA-256 list. Export environment: use a NEW venv; never modify the von/von12 venvs.
3. Frames: a fixed set of 20 frames captured the way the robot captures them (camera, resolution), saved once to
   /termux-home/yolo_bench/frames/ with hashes. Also time 10 captures.
4. Write yolo_speed.py + run_yolo_speed.sh (ladder style: thermal gate before each config, cpuset guard cores
   4-7, --resume, cold start per config). Per config: 5 warm-up frames, then 60 timed frames cycling the 20
   frames. Report per config: preprocess, inference, NMS/post and total ms (median, P95), model load s, peak RSS
   MiB, thermal start/end, and detection count per frame. Configs: 12 stock (11/26 x n/s/m x 320/640) + deployed
   yolo11m@640 + its int8 copy.
5. Checks without timing (coder resident): each config loads and returns detections on 2 frames.
6. Freeze edits. Invoke the Reviewer.
7. STOP. No commit, no push, no timed run. Run via ~/ladder/oneshot.sh (SHA-256 db1f3156...8e8), run script
   copied into ~/ladder/.
RULES: no motors; do not edit main.py, detect_person.py or any robot code; do not touch the deployed model or
the independent test set; do not delete files.
HUMAN DECISION DURING THE TASK: the 20 live captures were all black (camera covered); the human chose "use
existing photos", so frames/ holds 20 copies of bench_photos/ (taken by capture_bench.py with main.take_photo's
exact camera call); the black set was moved (not deleted) to /termux-home/yolo_bench/frames_black_camera_covered/.

## CANDIDATE DIFF (complete, vs base)
```diff
diff --git a/.gitignore b/.gitignore
index 841d7dd..0c692fb 100644
--- a/.gitignore
+++ b/.gitignore
@@ -7,6 +7,7 @@
 bench_photos/
 benchmark/*
 !benchmark/strategic_selector/
+!benchmark/yolo_speed/
 
 # Run artifacts
 *.csv
diff --git a/benchmark/yolo_speed/yolo_speed.py b/benchmark/yolo_speed/yolo_speed.py
new file mode 100644
index 0000000..18916da
--- /dev/null
+++ b/benchmark/yolo_speed/yolo_speed.py
@@ -0,0 +1,238 @@
+#!/usr/bin/env python3
+"""YOLO speed ladder: which YOLO family/size/input to retrain. Speed only. Research only: offline, no motors,
+no main.py; the deployed model is only read.
+
+Usage: yolo_speed.py --out DIR [--configs a,b] [--resume] [--thermal-log PATH] [--timed N] [--warmup N]
+       yolo_speed.py --worker MODEL IMGSZ WARMUP TIMED      (one cold config; run with the robot's native python)
+
+The parent (Debian python3) runs each config cold, in the ladder's style: wait for cores 4-7, thermal gate
+(z9 <= idle + 4 degC), page-cache drop through the oneshot.sh handshake (ladder.make_cold), then a fresh
+native-Termux python process -- the robot's interpreter and onnxruntime 1.25.1 -- that times WARMUP frames
+untimed, then TIMED frames cycling the 20 frames in BENCH/frames (copies of bench_photos/, which capture_bench.py
+took with main.take_photo's camera call; mapping in frames/SOURCES.txt). The worker does what detect_person.detect_scene does:
+default InferenceSession (CPUExecutionProvider, ORT default threads), PIL open -> RGB -> exif_transpose ->
+resize((imgsz, imgsz)) -> /255 CHW float32; post = per-class CONF filter + detect_person.nms, best box per class
+(the 640-pixel position constants scaled to imgsz). Losing cores 4-7 during a config discards and redoes it.
+Writes DIR/blocks.jsonl (one line per completed config, per-frame rows included) and DIR/report.txt.
+"""
+import argparse
+import hashlib
+import json
+import os
+import statistics
+import subprocess
+import sys
+import time
+from datetime import datetime, timezone
+from pathlib import Path
+
+BENCH = "/termux-home/yolo_bench"
+FRAMES = [f"{BENCH}/frames/frame_{i:02d}.jpg" for i in range(1, 21)]
+NATIVE_PY = "/data/data/com.termux/files/usr/bin/python"
+NATIVE_ENV = {"LD_LIBRARY_PATH": "/data/data/com.termux/files/usr/lib"}
+CONFIGS = {f"{fam}{s}_{sz}": (f"{BENCH}/models/{fam}{s}_{sz}.onnx", sz)
+           for fam in ("yolo11", "yolo26") for s in "nsm" for sz in (320, 640)}
+CONFIGS["deployed_yolo11m_640"] = ("/termux-home/robot/yolo11m.onnx", 640)
+CONFIGS["deployed_yolo11m_640_int8dyn"] = (f"{BENCH}/models/yolo11m_deployed_int8dyn.onnx", 640)
+
+
+def worker(model, imgsz, warmup, timed):
+    """One cold config in this process. Prints one JSON line on stdout."""
+    t0 = time.perf_counter()
+    import numpy as np
+    import onnxruntime as ort
+    from PIL import Image, ImageOps
+    sys.path.insert(0, "/termux-home/robot")
+    from detect_person import CONF, IOU, nms
+    import_s = time.perf_counter() - t0
+    t0 = time.perf_counter()
+    session = ort.InferenceSession(model)  # as detect_person.get_session
+    load_s = time.perf_counter() - t0
+    name = session.get_inputs()[0].name
+    third = imgsz / 3
+
+    rows = []
+    for i in range(warmup + timed):
+        if not {4, 5, 6, 7} <= os.sched_getaffinity(0):
+            print(json.dumps({"cores_lost": sorted(os.sched_getaffinity(0)), "frame_i": i}), flush=True)
+            sys.exit(3)
+        path = FRAMES[i % len(FRAMES)]
+        t0 = time.perf_counter()
+        img = Image.open(path).convert('RGB')
+        img = ImageOps.exif_transpose(img)
+        img = img.resize((imgsz, imgsz))
+        arr = np.array(img).astype(np.float32) / 255.0
+        arr = arr.transpose(2, 0, 1)[np.newaxis]
+        t1 = time.perf_counter()
+        out = session.run(None, {name: arr})[0][0].T
+        t2 = time.perf_counter()
+        by_class = {}
+        for pred in out:
+            scores = pred[4:]
+            cls = int(np.argmax(scores))
+            conf = float(scores[cls])
+            if conf > CONF:
+                cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
+                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
+                by_class.setdefault(cls, []).append((box, conf, cx, cy, w, h))
+        results = []
+        for cls, items in by_class.items():
+            keep = nms([it[0] for it in items], [it[1] for it in items], IOU)
+            box, conf, cx, cy, w, h = items[keep[0]]
+            area = (w * h) / (imgsz * imgsz)
+            pos = 'left' if cx < third else 'right' if cx > 2 * third else 'center'
+            dist = 'very close' if area > 0.3 else 'close' if area > 0.1 else 'medium' if area > 0.03 else 'far'
+            results.append((cls, round(conf, 2), pos, dist))
+        results.sort(key=lambda x: -x[1])
+        t3 = time.perf_counter()
+        rows.append({"i": i, "warmup": i < warmup, "frame": Path(path).name,
+                     "pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3,
+                     "total_ms": (t3 - t0) * 1e3, "n_det": len(results), "dets": results})
+    hwm = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM:")) / 1024
+    print(json.dumps({"import_s": import_s, "load_s": load_s, "peak_rss_mib": hwm, "ort": ort.__version__,
+                      "providers": session.get_providers(), "rows": rows}), flush=True)
+
+
+def run_config(name, a, thermal):
+    import ladder
+    model, imgsz = CONFIGS[name]
+    therm = {}
+    ladder.check_cores("before cold")
+    if thermal:
+        therm["start"] = ladder.thermal_gate(thermal[0], thermal[1], name)
+        print(f"  [{name}] thermal start: {ladder.fmt_thermal(therm['start'])} (waited {therm['start']['waited_s']} s)",
+              flush=True)
+    load_state = ladder.make_cold(name, [model])
+    ladder.check_cores("before starting worker")
+    t0 = time.perf_counter()
+    p = subprocess.run([NATIVE_PY, "-W", "ignore", __file__, "--worker", model, str(imgsz), str(a.warmup), str(a.timed)],
+                       env={**os.environ, **NATIVE_ENV}, capture_output=True, text=True)
+    wall_s = time.perf_counter() - t0
+    (a.out / "logs" / f"{name}.stderr.txt").write_text(p.stderr)
+    if p.returncode == 3:
+        raise ladder.CoresLost(f"worker lost cores 4-7: {p.stdout.strip()}")
+    if p.returncode != 0:
+        raise SystemExit(f"[{name}] worker exit {p.returncode}; stderr in {a.out}/logs/{name}.stderr.txt")
+    res = json.loads(p.stdout.strip().splitlines()[-1])
+    if thermal:
+        therm["end"] = ladder.read_thermal(thermal[0])
+        print(f"  [{name}] thermal end:   {ladder.fmt_thermal(therm['end'])}", flush=True)
+    ladder.check_cores("after worker")
+    return {"config": name, "model": model, "imgsz": imgsz, "warmup": a.warmup, "timed": a.timed,
+            "wall_s": wall_s, "load_state": load_state, "thermal": therm or None, **res}
+
+
+def pct(xs, q):
+    return statistics.median(xs) if q == 50 else statistics.quantiles(xs, n=100, method="inclusive")[q - 1]
+
+
+def report(done, names):
+    hdr = (f"{'config':<30}{'pre ms':>13}{'infer ms':>15}{'post ms':>11}{'total ms':>15}{'load s':>8}{'RSS MiB':>9}"
+           f"{'dets/frame':>16}   thermal z9 start->end degC; load state")
+    lines = ["median / P95 over the timed frames; dets/frame = mean [min-max]", hdr]
+    for n in names:
+        if n not in done:
+            lines.append(f"{n:<30}not run")
+            continue
+        b = done[n]
+        t = [r for r in b["rows"] if not r["warmup"]]
+        mp = lambda k: f"{pct([r[k] for r in t], 50):.1f}/{pct([r[k] for r in t], 95):.1f}"
+        d = [r["n_det"] for r in t]
+        th = (f"{b['thermal']['start']['z9'] / 1000:.0f}->{b['thermal']['end']['z9'] / 1000:.0f}" if b["thermal"]
+              else "n/a")
+        lines.append(f"{n:<30}{mp('pre_ms'):>13}{mp('infer_ms'):>15}{mp('post_ms'):>11}{mp('total_ms'):>15}"
+                     f"{b['load_s']:>8.2f}{b['peak_rss_mib']:>9.0f}"
+                     f"{f'{statistics.mean(d):.2f} [{min(d)}-{max(d)}]':>16}   {th}; {b['load_state']['mode'].split(' (')[0]}")
+    return "\n".join(lines)
+
+
+def read_jsonl(p):
+    """Rows of a JSONL file; a torn last line (a write cut off by a kill) is dropped."""
+    if not p.exists():
+        return []
+    lines = p.read_text().splitlines()
+    rows = []
+    for i, l in enumerate(lines):
+        try:
+            rows.append(json.loads(l))
+        except json.JSONDecodeError:
+            if i != len(lines) - 1:
+                raise
+            print(f"{p}: dropping torn last line", flush=True)
+    return rows
+
+
+def main():
+    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
+        return worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
+    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    ap.add_argument("--out", type=Path, required=True)
+    ap.add_argument("--configs", default=",".join(CONFIGS))
+    ap.add_argument("--resume", action="store_true")
+    ap.add_argument("--thermal-log", default=None)
+    ap.add_argument("--warmup", type=int, default=5)
+    ap.add_argument("--timed", type=int, default=60)
+    a = ap.parse_args()
+    names = a.configs.split(",")
+    if set(names) - set(CONFIGS):
+        raise SystemExit(f"unknown configs {sorted(set(names) - set(CONFIGS))}")
+    if a.resume and not a.out.is_dir():
+        raise SystemExit(f"--resume: {a.out} does not exist")
+    a.out.mkdir(parents=True, exist_ok=a.resume)
+    (a.out / "logs").mkdir(exist_ok=True)
+    sys.path.insert(0, "/termux-home/ladder")
+    import ladder
+    done = {b["config"]: b for b in read_jsonl(a.out / "blocks.jsonl")} if a.resume else {}
+    for n, b in done.items():
+        if (b["warmup"], b["timed"]) != (a.warmup, a.timed):
+            raise SystemExit(f"--resume: {n} was run with warmup/timed {b['warmup']}/{b['timed']}")
+    if a.resume:  # rewrite without a torn last line so appends start on a clean line
+        (a.out / "blocks.jsonl.tmp").write_text("".join(json.dumps(b) + "\n" for b in done.values()))
+        os.replace(a.out / "blocks.jsonl.tmp", a.out / "blocks.jsonl")
+        print("resume: keeping " + (", ".join(done) or "nothing"), flush=True)
+    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
+    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
+    manifest = {
+        "yolo_speed_sha256": sha(__file__), "ladder_sha256": sha(ladder.__file__),
+        "detect_person_sha256": sha("/termux-home/robot/detect_person.py"),
+        "models": {n: [CONFIGS[n][0], sha(CONFIGS[n][0])] for n in names},
+        "frames": {Path(f).name: sha(f) for f in FRAMES}, "warmup": a.warmup, "timed": a.timed}
+    if a.resume:  # every earlier manifest must agree on the inputs, so completed and new configs are comparable
+        for old in sorted(a.out.glob("run_*.json")):
+            m = json.loads(old.read_text())
+            diff = [k for k in manifest if k != "models" and m.get(k) != manifest[k]]
+            diff += [n for n in set(m.get("models", {})) & set(manifest["models"]) if m["models"][n] != manifest["models"][n]]
+            if diff:
+                raise SystemExit(f"--resume: inputs differ from {old.name}: {diff}")
+    (a.out / f"run_{stamp}.json").write_text(json.dumps(manifest, indent=1) + "\n")
+    ladder.wait_cores("start")
+    thermal = None
+    if a.thermal_log:
+        idle = ladder.read_thermal(a.thermal_log)
+        print(f"thermal idle reading (gate z9 <= idle + {ladder.GATE_MC / 1000:.0f} degC): {ladder.fmt_thermal(idle)}",
+              flush=True)
+        thermal = (a.thermal_log, idle)
+    with open(a.out / "blocks.jsonl", "a") as blog:
+        for n in names:
+            if n in done:
+                print(f"[{n}] already completed, skipped (--resume)", flush=True)
+                continue
+            print(f"[{n}] cold: {a.warmup} warm-up + {a.timed} timed frames", flush=True)
+            while True:
+                ladder.wait_cores(n)
+                try:
+                    b = run_config(n, a, thermal)
+                    break
+                except ladder.CoresLost as e:
+                    print(f"  [{n}] {e}: discarding this config's data and redoing it", flush=True)
+            blog.write(json.dumps(b) + "\n")
+            blog.flush()
+            done[n] = b
+            print(report({n: b}, [n]).splitlines()[-1], flush=True)
+    text = report(done, names)
+    (a.out / "report.txt").write_text(text + "\n")
+    print(text)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/benchmark/yolo_speed/run_yolo_speed.sh b/benchmark/yolo_speed/run_yolo_speed.sh
new file mode 100755
index 0000000..8f03388
--- /dev/null
+++ b/benchmark/yolo_speed/run_yolo_speed.sh
@@ -0,0 +1,22 @@
+#!/bin/bash
+# YOLO speed ladder (yolo_speed.py): 14 configs, stock yolo11/yolo26 n/s/m at 320 and 640, the deployed
+# yolo11m.onnx at 640 and its int8 dynamic-quantized copy. Research only: offline, no motors, no main.py.
+# One cold config at a time: thermal gate z9 <= idle + 4 degC, page-cache drop through the handshake
+# (LADDER_COLD_HANDSHAKE=1), fresh native-Termux python worker, 5 warm-up + 60 timed frames; losing cores 4-7
+# redoes the config.
+#
+# Usage: run_yolo_speed.sh [OUT_DIR]          new run
+#        run_yolo_speed.sh --resume OUT_DIR   keep OUT_DIR's completed configs, run the rest
+# Unattended, from native Termux:  RUN_SCRIPT=run_yolo_speed.sh bash ~/ladder/oneshot.sh [--resume OUT_DIR]
+set -euo pipefail
+cd /termux-home/ladder
+RESUME=()
+if [ "${1:-}" = --resume ]; then
+  OUT=${2:?usage: run_yolo_speed.sh --resume OUT_DIR}
+  RESUME=(--resume)
+else
+  OUT=${1:-/termux-home/ladder/yolo_speed_$(date -u +%Y%m%dT%H%M%SZ)}
+fi
+python3 /termux-home/robot/benchmark/yolo_speed/yolo_speed.py --out "$OUT" "${RESUME[@]}" \
+  --thermal-log /termux-home/ladder/thermal.log \
+  > >(tee -a "$OUT.stdout.txt") 2> >(tee -a "$OUT.stderr.txt" >&2)
diff --git a/termux-home/yolo_bench/export_models.py b/termux-home/yolo_bench/export_models.py
new file mode 100644
index 0000000..a8fa56a
--- /dev/null
+++ b/termux-home/yolo_bench/export_models.py
@@ -0,0 +1,25 @@
+# Export stock Ultralytics yolo11/yolo26 n/s/m to ONNX at 320 and 640 with the deployed file's
+# metadata args (batch 1, half False, dynamic False, simplify True, nms False; opset 20 as in the
+# deployed graph), plus an int8 dynamic-quantized copy of the deployed yolo11m.onnx. Run in the
+# yolo_bench venv from /termux-home/yolo_bench/weights.
+import os, shutil
+from ultralytics import YOLO
+from onnxruntime.quantization import quantize_dynamic, QuantType
+
+OUT = '/termux-home/yolo_bench/models'
+for fam in ('yolo11', 'yolo26'):
+    for size in 'nsm':
+        for imgsz in (320, 640):
+            dst = f'{OUT}/{fam}{size}_{imgsz}.onnx'
+            if os.path.exists(dst):
+                continue
+            kw = dict(format='onnx', imgsz=imgsz, batch=1, half=False, dynamic=False,
+                      simplify=True, opset=20, nms=False, device='cpu')
+            if fam == 'yolo26':
+                kw['end2end'] = False  # deployed file is end2end False: same [1,84,N] output and post-processing
+            f = YOLO(f'{fam}{size}.pt').export(**kw)
+            shutil.move(f, dst)
+
+dst = f'{OUT}/yolo11m_deployed_int8dyn.onnx'
+if not os.path.exists(dst):
+    quantize_dynamic('/termux-home/robot/yolo11m.onnx', dst, weight_type=QuantType.QInt8)
diff --git a/termux-home/yolo_bench/capture_frames.py b/termux-home/yolo_bench/capture_frames.py
new file mode 100644
index 0000000..11e0549
--- /dev/null
+++ b/termux-home/yolo_bench/capture_frames.py
@@ -0,0 +1,15 @@
+# Capture 20 frames exactly as main.take_photo does (termux-camera-photo, default camera, then sleep 0.5),
+# timing each call. Frames 01-10 timings = the "10 captures". Run with native Termux python.
+import os, time, json
+D = '/data/data/com.termux/files/home/yolo_bench/frames'
+t = []
+for i in range(1, 21):
+    p = f'{D}/frame_{i:02d}.jpg'
+    t0 = time.perf_counter()
+    os.system(f'termux-camera-photo {p} 2>/dev/null')
+    t1 = time.perf_counter()
+    time.sleep(0.5)
+    ok = os.path.exists(p) and os.path.getsize(p) > 1000
+    t.append({'frame': i, 'call_s': round(t1 - t0, 3), 'take_photo_s': round(time.perf_counter() - t0, 3), 'ok': ok})
+    print(t[-1], flush=True)
+json.dump(t, open(f'{D}/../capture_timing.json', 'w'), indent=1)
```
Frozen SHA-256:
5fbd88589ba6000dabd52803a483bf9375189d2e627297bbda7a92be1f56d77c  benchmark/yolo_speed/run_yolo_speed.sh
3e59005106da81b0ee1a87ac94331d219bfbfa3460993b08fa8d36d4cb60499a  benchmark/yolo_speed/yolo_speed.py
41f6ce31722c2f2d8cbc24cd5817e95d4ad2b62b0a81a66ef5e81f845c37f19d  .gitignore
295b5b972296f7d574822522b96d58363656443d16caacb9f7ccbfb6c4960271  /termux-home/yolo_bench/export_models.py
295b5b972296f7d574822522b96d58363656443d16caacb9f7ccbfb6c4960271  /termux-home/yolo_bench/export_models.py
f4afd1ec1c35caac8bfc90c49bcc6017e964f290179e29bce6e7674768c5ecfe  /termux-home/yolo_bench/capture_frames.py

## INLINED CONTEXT
### detect_person.py (unchanged, sha 85be7d1b0f15b5665b4903018815f50e4aa7f383e343191882c0e10d123fcb7a)
```python
     1	import sys, os, time
     2	import numpy as np
     3	import onnxruntime as ort
     4	from PIL import Image, ImageOps
     5	
     6	CLASSES = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck',
     7	'boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird',
     8	'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
     9	'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    10	'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    11	'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    12	'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    13	'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    14	'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
    15	'clock','vase','scissors','teddy bear','hair dryer','toothbrush']
    16	
    17	MODEL = '/data/data/com.termux/files/home/robot/yolo11m.onnx'
    18	CONF  = 0.35
    19	IOU   = 0.45
    20	
    21	_session = None
    22	
    23	def get_session():
    24	    global _session
    25	    if _session is None:
    26	        _session = ort.InferenceSession(MODEL)
    27	    return _session
    28	
    29	def iou(a, b):
    30	    ax1,ay1,ax2,ay2 = a
    31	    bx1,by1,bx2,by2 = b
    32	    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    33	    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    34	    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    35	    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    36	    return inter/ua if ua>0 else 0
    37	
    38	def nms(boxes, confs, iou_thresh=0.45):
    39	    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    40	    keep = []
    41	    while order:
    42	        i = order.pop(0)
    43	        keep.append(i)
    44	        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
    45	    return keep
    46	
    47	def detect_scene(image_path):
    48	    """
    49	    Detect all objects in the scene.
    50	    Returns list of (label, confidence, position, distance)
    51	    """
    52	    session = get_session()
    53	    img = Image.open(image_path).convert('RGB')
    54	    img = ImageOps.exif_transpose(img)
    55	    img = img.resize((640, 640))
    56	    arr = np.array(img).astype(np.float32) / 255.0
    57	    arr = arr.transpose(2, 0, 1)[np.newaxis]
    58	
    59	    out = session.run(None, {session.get_inputs()[0].name: arr})[0][0].T
    60	
    61	    by_class = {}
    62	    for pred in out:
    63	        scores = pred[4:]
    64	        cls = int(np.argmax(scores))
    65	        conf = float(scores[cls])
    66	        if conf > CONF:
    67	            cx,cy,w,h = float(pred[0]),float(pred[1]),float(pred[2]),float(pred[3])
    68	            box = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
    69	            if cls not in by_class:
    70	                by_class[cls] = []
    71	            by_class[cls].append((box, conf, cx, cy, w, h))
    72	
    73	    results = []
    74	    for cls, items in by_class.items():
    75	        boxes = [i[0] for i in items]
    76	        confs = [i[1] for i in items]
    77	        keep = nms(boxes, confs, IOU)
    78	        best = items[keep[0]]
    79	        cx, cy, w, h = best[2], best[3], best[4], best[5]
    80	        conf = best[1]
    81	        area = (w*h)/(640*640)
    82	        pos = 'left' if cx < 213 else 'right' if cx > 427 else 'center'
    83	        dist = 'very close' if area>0.3 else 'close' if area>0.1 else 'medium' if area>0.03 else 'far'
    84	        results.append((CLASSES[cls], round(conf,2), pos, dist, round(cx), round(cy), round(w), round(h)))
    85	
    86	    results.sort(key=lambda x: -x[1])
    87	    return results
    88	
    89	def detect_person(image_path):
    90	    """
    91	    Legacy function — returns (found, confidence, position)
    92	    for backwards compatibility with existing code.
    93	    """
    94	    results = detect_scene(image_path)
    95	    for r in results:
    96	        label, conf, pos, dist = r[0], r[1], r[2], r[3]
    97	        if label == 'person':
    98	            return True, conf, pos
    99	    return False, 0.0, 'none'
   100	
   101	def scene_to_text(results, coords=False):
   102	    """
   103	    Convert detection results to a natural language sentence.
   104	    coords=True adds pixel coordinates for stereo vision.
   105	    """
   106	    if not results:
   107	        return "empty room, nothing detected"
   108	    if coords:
   109	        parts = [f"{r[0]} x={r[4]} y={r[5]} w={r[6]} h={r[7]} {r[2]} {r[3]}" for r in results]
   110	    else:
   111	        parts = [f"{r[0]} {r[2]}" for r in results]
   112	    return ', '.join(parts)
   113	
   114	def person_direction(results):
   115	    """
   116	    Return direction to move toward detected person.
   117	    Returns 'LEFT', 'RIGHT', 'FORWARD', or None
   118	    """
   119	    for r in results:
   120	        label, conf, pos, dist = r[0], r[1], r[2], r[3]
   121	        if label == 'person':
   122	            if pos == 'left':   return 'LEFT'
   123	            if pos == 'right':  return 'RIGHT'
   124	            if pos == 'center': return 'FORWARD'
   125	    return None
   126	
   127	if __name__ == '__main__':
   128	    path = sys.argv[1] if len(sys.argv) > 1 else 'test_photos/scene_test.jpg'
   129	    t0 = time.time()
   130	    results = detect_scene(path)
   131	    elapsed = round(time.time()-t0, 3)
   132	    print(f'Detected in {elapsed}s (640x640 frame):')
   133	    print(f'  {"object":<15} {"conf":<5} {"pos":<8} {"dist":<12} {"cx":>5} {"cy":>5} {"w":>5} {"h":>5}')
   134	    print('  ' + '-'*60)
   135	    for r in results:
   136	        print(f'  {r[0]:<15} {r[1]:<5} {r[2]:<8} {r[3]:<12} {r[4]:>5} {r[5]:>5} {r[6]:>5} {r[7]:>5}')
   137	    print(f'\nScene: "{scene_to_text(results)}"')
   138	    print(f'Coords: "{scene_to_text(results, coords=True)}"')
   139	    direction = person_direction(results)
   140	    if direction:
   141	        print(f"Person direction: {direction}")
   142	    else:
   143	        print("No person detected")```
### /termux-home/ladder/ladder.py lines 109-195 and 256-285 (sha c7e6d3c9...; allowed_cpus imported from jevlike = os.sched_getaffinity)
```python
class CoresLost(Exception):
    """Android moved Termux off the top-app cpuset mid-block; the block's data is discarded and redone."""


def check_cores(when):
    if not set(range(4, 8)) <= allowed_cpus():
        raise CoresLost(f"cores {CORES} not all allowed {when} (allowed {sorted(allowed_cpus())})")


def wait_cores(label):
    """Pauses until cores 4-7 are allowed again (Termux back in the foreground), rechecking every 10 s."""
    began = time.monotonic()
    while not set(range(4, 8)) <= allowed_cpus():
        print(f"  [{label}] paused: cores {CORES} not allowed (allowed {sorted(allowed_cpus())}); bring Termux to the "
              f"foreground. {time.monotonic() - began:.0f} s", flush=True)
        time.sleep(10)


DROP_CMD = "su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"


def cached_mib():
    return next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("Cached:"))


def evict(files):
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # read-only files: every page is clean
        finally:
            os.close(fd)


DROP_REQUEST, DROP_DONE = HERE / ".drop_request", HERE / ".drop_done"
HANDSHAKE_S = 60


def drop_handshake(name):
    """Unattended cache drop: create .drop_request; a native-Termux watcher drops the page cache as root and
    writes 'ok' or 'failed' to .drop_done. Returns that answer, or None after HANDSHAKE_S seconds."""
    DROP_DONE.unlink(missing_ok=True)
    DROP_REQUEST.touch()
    print(f"\n[{name}] cold load next: handshake, waiting up to {HANDSHAKE_S} s for {DROP_DONE}", flush=True)
    deadline, answer = time.monotonic() + HANDSHAKE_S, None
    try:
        while time.monotonic() < deadline:
            if DROP_DONE.exists() and (text := DROP_DONE.read_text().strip()):  # empty: still being written
                answer = text
                break
            time.sleep(0.5)
    finally:
        DROP_REQUEST.unlink(missing_ok=True)
        DROP_DONE.unlink(missing_ok=True)
    return answer


def make_cold(name, files):
    """Drop the page cache from native Termux (Magisk su does not work in proot) and check /proc/meminfo that
    it happened; otherwise evict only this model's weight files. LADDER_COLD_HANDSHAKE=1 asks through files
    (unattended runs); otherwise the human is prompted on the terminal."""
    before = cached_mib()
    if os.environ.get("LADDER_COLD_HANDSHAKE") == "1":
        answer = drop_handshake(name)
        after = cached_mib()
        if answer == "ok" and after < 0.5 * before:
            return {"mode": "full-cold (page cache dropped by the handshake watcher with drop_caches)",
                    "cached_mib_before": before, "cached_mib_after": after}
        why = (f"handshake timed out after {HANDSHAKE_S} s" if answer is None else
               f"handshake answered {answer!r}" if answer != "ok" else
               f"handshake answered 'ok' but page cache did not drop ({before} -> {after} MiB)")
    elif not sys.stdin.isatty():
        why = "no terminal to prompt on"
    else:
        print(f"\n[{name}] COLD LOAD NEXT. In native Termux (not proot) run:\n    {DROP_CMD}\n"
              "then press Enter here. Type s + Enter to skip (weights-cold fallback).", flush=True)
        answer = input("> ").strip().lower()
        after = cached_mib()
        if answer != "s" and after < 0.5 * before:
            return {"mode": "full-cold (page cache dropped by the human with drop_caches)",
                    "cached_mib_before": before, "cached_mib_after": after}
        why = "skipped by the human" if answer == "s" else f"Enter pressed but page cache did not drop ({before} -> {after} MiB)"
    evict(files)
    return {"mode": "weights-cold (posix_fadvise DONTNEED on the weight files; libraries stay cached)",
            "fallback_reason": why, "cached_mib_before": before}


THERMAL_MAX_AGE_S = 30  # the root logger writes every ~5 s; an older last line means it stopped
GATE_MC = 4000         # a block starts only when z9 <= idle reading + 4 degC (+2 cost ~6 min of waiting per block)


def read_thermal(path):
    """Last line of the root thermal log: '2026-09-25T21:51:27Z z9=36000 z10=36000 z11=37000' (millidegrees)."""
    last = Path(path).read_text().strip().splitlines()[-1].split()
    stamp = datetime.strptime(last[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if age > THERMAL_MAX_AGE_S:
        raise SystemExit(f"thermal log {path} last line is {age:.0f} s old: the logger is not running")
    z = dict(kv.split("=") for kv in last[1:])
    return {"at": last[0], **{k: int(z[k]) for k in ("z9", "z10", "z11")}}


def thermal_gate(path, idle, label):
    """Waits until z9 <= idle + GATE_MC (cooler than idle is fine); returns the reading the block starts at."""
    began = time.monotonic()
    while (t := read_thermal(path))["z9"] > idle["z9"] + GATE_MC:
        print(f"  [{label}] waiting: z9 {t['z9'] / 1000:.1f} degC, need <= {(idle['z9'] + GATE_MC) / 1000:.1f} "
              f"(idle {idle['z9'] / 1000:.1f} + {GATE_MC / 1000:.0f}), {time.monotonic() - began:.0f} s", flush=True)
        time.sleep(10)
    t["waited_s"] = round(time.monotonic() - began)
    return t


def fmt_thermal(t):
    return " ".join(f"{k} {t[k] / 1000:.1f}" for k in ("z9", "z10", "z11")) + f" degC at {t['at']}"


```
### /termux-home/ladder/oneshot.sh (sha db1f3156...8e8)
```bash
#!/data/data/com.termux/files/usr/bin/bash
# One-shot s1o speed run, v2. Start from NATIVE Termux (prompt "~ $"):
#   bash ~/ladder/oneshot.sh                  (fresh run)
#   bash ~/ladder/oneshot.sh --resume <dir>   (continue a run)
# Thermal logger, checks, screen kept on, 5 min idle, the run inside Debian,
# and root cache drops on request (file handshake with the runner).
H=/data/data/com.termux/files/home
L=$H/ladder
CONSOLE=$L/oneshot_console_$(date -u +%Y%m%dT%H%M%SZ).log
say() { echo "[oneshot] $*" | tee -a "$CONSOLE"; }
RUN_SCRIPT=${RUN_SCRIPT:-run_s1o_speed.sh}  # e.g. RUN_SCRIPT=run_conversation.sh bash ~/ladder/oneshot.sh

if [ -d /termux-home ]; then echo "Run this from native Termux (~ \$), not from Debian."; exit 1; fi
su -c id >/dev/null 2>&1 || { echo "No root: su failed. Check Magisk."; exit 1; }
[ -f "$L/$RUN_SCRIPT" ] || { echo "Missing $L/$RUN_SCRIPT"; exit 1; }
if pgrep -fa 'claude|agy|node' >/dev/null; then
  echo "An agent is still running:"; pgrep -fa 'claude|agy|node'; echo "Quit it, then rerun."; exit 1
fi

termux-wake-lock
# keep the screen on (Android moves Termux off the big cores when it is not in front)
OLD_TIMEOUT=$(su -c 'settings get system screen_off_timeout')
su -c 'settings put system screen_off_timeout 2147483647'
rm -f "$L/.stop_thermal" "$L/.drop_request" "$L/.drop_done" "$L/.drop_done.tmp"

su -c "while [ ! -f $L/.stop_thermal ]; do echo \"\$(date -u +%Y-%m-%dT%H:%M:%SZ) z9=\$(cat /sys/class/thermal/thermal_zone9/temp) z10=\$(cat /sys/class/thermal/thermal_zone10/temp) z11=\$(cat /sys/class/thermal/thermal_zone11/temp)\" >> $L/thermal.log; sleep 5; done" &

# cache-drop watcher: runner creates .drop_request, we drop and write .drop_done
( while [ ! -f "$L/.stop_thermal" ]; do
    if [ -f "$L/.drop_request" ] && [ ! -f "$L/.drop_done" ]; then
      if su -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; then r=ok; else r=failed; fi
      echo $r > "$L/.drop_done.tmp" && mv "$L/.drop_done.tmp" "$L/.drop_done"; say "cache drop: $r"
    fi
    sleep 1
  done ) &

cleanup() {
  touch "$L/.stop_thermal"
  su -c "settings put system screen_off_timeout ${OLD_TIMEOUT:-60000}"
  termux-wake-unlock
  say "stopped logger and watcher; screen timeout restored to ${OLD_TIMEOUT}"
}
trap cleanup EXIT

sleep 6
say "thermal: $(tail -1 "$L/thermal.log")"
say "idling 5 minutes; keep Termux in front"
sleep 300
say "thermal after idle: $(tail -1 "$L/thermal.log")"

proot-distro login debian --bind "$H:/termux-home" -- \
  env PYTHONUNBUFFERED=1 LADDER_COLD_HANDSHAKE=1 "/termux-home/ladder/$RUN_SCRIPT" "$@" 2>&1 | tee -a "$CONSOLE"
say "done; console log: $CONSOLE"
```
### main.py lines 29-32
```python
def take_photo(path):
    os.system(f'termux-camera-photo {path} 2>/dev/null')
    time.sleep(0.5)
    return os.path.exists(path) and os.path.getsize(path) > 1000
```
- Deployed yolo11m.onnx metadata: ultralytics 8.4.33, args {'batch':1,'half':False,'dynamic':False,'simplify':True,'opset':None,'nms':False}, end2end False, opset 20, producer pytorch 2.11.0. Exports: new venv /termux-home/yolo_bench/venv (ultralytics 8.4.33, torch 2.14.0, onnx 1.23.0, onnxruntime 1.30.0 for quantize_dynamic), opset=20 set explicitly; yolo26 exported with end2end=False.

## CHECK OUTPUT (actual)
Deployed model sha256 before and after exports: b6e24c02abc0f1a69e20b7fe6157b6a327d3d033c9d1a2441add9fe45d452649 (unchanged).
Export verification (all [1,3,S,S] -> [1,84,2100|8400], opset 20, end2end False, version 8.4.33). models/SHA256SUMS:
6bfff56c151b111c10c8a6be11bce780328d16b6ad614616825347112654436b  yolo11m_320.onnx
ca7ed8deba30e8fa5b9bffef31c6fcac39aa87e3086d40276be32f179329826b  yolo11m_640.onnx
229c50a78195fa0be4ab90e15503f06d43e6a05f9dd4ae0bf308a0efcd8a10db  yolo11m_deployed_int8dyn.onnx
df711c36d7cbf00f4222ccacd162144cf5f67d6a8fb3d3532f15d68bc31c4762  yolo11n_320.onnx
202f5b7eef8aca94d6be9d0ac539f3117301b61fe1f45f79a640ad42d9e6b1c0  yolo11n_640.onnx
9eab5ebc83d1254e0049abf2a939ef12db696b7e0ef74af7ec8b70cec993f863  yolo11s_320.onnx
dc9fe7c50e66ef272db1318f282d6e6dc4ba6adaa0103a2197a87300fbb9dcb5  yolo11s_640.onnx
7c5574028347a829c5f03175a69298270bedff7e9e91881e2c688e4ab84a8c26  yolo26m_320.onnx
45e372c673c03f288fb1c5e1e406b01b28bf507a69c062f255a3387e6f9f8210  yolo26m_640.onnx
ee3685c0df01129e9aafe2c12335d7676854a201dd709db972ecd7bdc61b3d6e  yolo26n_320.onnx
0f3bc42a3ea39e96dc0757b28df8a64e145321edd4f80be03df0c22b603e9187  yolo26n_640.onnx
e563945ae5396412c1e359811c7c787ea9ae59dce13f61a13f6697be0f29ed57  yolo26s_320.onnx
87347f226fd9bab775da9ed38ce8b748258bde39ccffd62dfca12caa94a9a68f  yolo26s_640.onnx
Stock vs deployed yolo11m: 245 vs 245 initializers, 20133752 params each, sorted per-tensor |w| sums max rel diff 0.0; raw outputs on frame_06/frame_14: max abs diff 0.00122 / 0.00061 (outputs up to ~640).
Live capture (capture_frames.py, native python, robot's call): 20 calls, call_s 2.15-3.39 s, 3/20 produced a 0-byte file; all 20 frames black -> moved aside; frames/ = bench_photos copies (4080x3072, EXIF orientation 6, same as live).

Step-5 check: `python3 yolo_speed.py --out <scratch>/check --warmup 5 --timed 2` (no thermal log, no handshake -> weights-cold fallback; agent resident, timings NOT valid). Cores guard fired 4 times (Termux backgrounded; allowed [0-5]); each time the config was discarded and redone. Final:
median / P95 over the timed frames; dets/frame = mean [min-max]
config                               pre ms       infer ms    post ms       total ms  load s  RSS MiB      dets/frame   thermal z9 start->end degC; load state
yolo11n_320                     190.8/191.9      29.9/33.1    6.9/7.0    227.6/231.9    0.08      187      2.00 [2-2]   n/a; weights-cold
yolo11n_640                     208.7/211.4    118.1/124.4  28.3/28.5    355.1/358.6    0.06      238      2.00 [2-2]   n/a; weights-cold
yolo11s_320                     187.8/189.5      79.0/79.7    7.1/7.2    273.9/276.2    0.11      228      2.00 [1-3]   n/a; weights-cold
yolo11s_640                     209.9/213.6    281.4/281.9  28.2/28.5    519.5/524.1    0.10      325      2.50 [2-3]   n/a; weights-cold
yolo11m_320                     188.2/192.1    204.7/211.0    7.1/7.2    400.0/410.2    0.19      299      1.50 [1-2]   n/a; weights-cold
yolo11m_640                     214.9/217.5    682.1/685.0  30.1/30.1    927.1/927.4    0.19      472      2.50 [2-3]   n/a; weights-cold
yolo26n_320                     201.2/203.7      31.4/36.0    7.0/7.0    239.6/241.6    0.07      188      1.50 [1-2]   n/a; weights-cold
yolo26n_640                     215.0/217.2      90.1/90.2  28.7/29.1    333.8/335.5    0.07      240      2.00 [1-3]   n/a; weights-cold
yolo26s_320                     200.0/201.2      68.3/75.2    7.0/7.0    275.3/281.0    0.11      230      1.50 [1-2]   n/a; weights-cold
yolo26s_640                     210.7/213.8    372.6/373.8  28.4/28.6    611.7/616.2    0.12      331      3.00 [3-3]   n/a; weights-cold
yolo26m_320                     190.8/190.9    200.4/201.7    7.2/7.4    398.5/399.6    0.18      300      2.50 [2-3]   n/a; weights-cold
yolo26m_640                     215.2/216.8    683.5/692.7  28.4/28.4    927.0/934.7    0.17      482      3.00 [2-4]   n/a; weights-cold
deployed_yolo11m_640            211.5/214.2    688.7/707.8  29.1/29.4    929.4/946.1    0.20      472      2.50 [2-3]   n/a; weights-cold
deployed_yolo11m_640_int8dyn    217.3/218.4  1801.3/1813.5  29.1/29.1  2047.7/2058.8    0.19      436      2.50 [2-3]   n/a; weights-cold

Timed frames = frame_06, frame_07 (class ids, per config):
yolo11n_320                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 2, [0, 62]), ('frame_07.jpg', 2, [0, 57])]
yolo11n_640                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 2, [0, 62]), ('frame_07.jpg', 2, [0, 59])]
yolo11s_320                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 1, [0]), ('frame_07.jpg', 3, [0, 59, 57])]
yolo11s_640                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 3, [0, 62, 58]), ('frame_07.jpg', 2, [0, 59])]
yolo11m_320                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 1, [0]), ('frame_07.jpg', 2, [59, 0])]
yolo11m_640                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 3, [0, 56, 62]), ('frame_07.jpg', 2, [59, 0])]
yolo26n_320                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 1, [0]), ('frame_07.jpg', 2, [0, 57])]
yolo26n_640                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 1, [0]), ('frame_07.jpg', 3, [0, 59, 57])]
yolo26s_320                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 1, [0]), ('frame_07.jpg', 2, [57, 0])]
yolo26s_640                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 3, [0, 45, 58]), ('frame_07.jpg', 3, [0, 59, 57])]
yolo26m_320                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 3, [0, 62, 58]), ('frame_07.jpg', 2, [0, 57])]
yolo26m_640                    1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 4, [0, 45, 58, 62]), ('frame_07.jpg', 2, [59, 0])]
deployed_yolo11m_640           1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 3, [0, 56, 62]), ('frame_07.jpg', 2, [59, 0])]
deployed_yolo11m_640_int8dyn   1.25.1 ['CPUExecutionProvider'] [('frame_06.jpg', 2, [0, 62]), ('frame_07.jpg', 3, [59, 0, 73])]
Worker deployed_yolo11m_640 dets == detect_person.detect_scene() on the same frames:
 frame_06: worker [[0,0.93,center,close],[56,0.42,right,far],[62,0.42,left,medium]]  detect_scene: person 0.93 center close, chair 0.42 right far, tv 0.42 left medium
 frame_07: worker [[59,0.85,center,very close],[0,0.57,center,close]]               detect_scene: bed 0.85 center very close, person 0.57 center close

## KNOWN LIMITATIONS the Coder states (evaluate, do not just accept)
- oneshot.sh runs the script inside proot Debian; the robot runs natively. The worker is the native Termux interpreter but runs under proot's ptrace, so absolute ms may differ from the robot; ranking is the purpose.
- Peak RSS = worker VmHWM (python+numpy+PIL+ORT+model), not model-only.
- Post for 320 configs scales the 640 position constants (213/427) to imgsz/3.

## REQUEST
Review correctness and fitness for the task: fidelity to the robot's runtime/preprocess/post; cold-start, thermal gate, cores guard, --resume and redo logic; the report's statistics and fields vs the task; export fidelity; any safety/rule violation (motors, robot code, deployed model, deletions). Cite file:line. End with a final verdict line: APPROVE, APPROVE WITH NOTES, or CHANGES REQUIRED, and list required changes.

## ROUND-2 CHECKS (actual)
.gitignore: `git check-ignore benchmark/yolo_speed/yolo_speed.py` -> not ignored; git status shows `?? benchmark/yolo_speed/` and ` M .gitignore`.
Resume validation (no configs executed; the one selected config was already complete):
 T1 resume onto round-1 output whose manifest has the older script hash -> "--resume: inputs differ from run_20260928T105804Z.json: ['yolo_speed_sha256']"
 T2 resume dir without any manifest -> passes, writes manifest, prints report row
 T3 resume again against its own manifest -> passes
 T4 add a manifest with a changed frame_03 hash -> "--resume: inputs differ from run_00000000T000000Z.json: ['frames']"
