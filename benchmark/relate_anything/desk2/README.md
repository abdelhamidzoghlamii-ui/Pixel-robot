# RELATE_FIX1 update — motors-off research candidate

M2 is the owner's benchmark choice, not deployed. See [owner run index](../RUN_INDEX.md)
for the immutable frozen-R1 results. Later code changes do not apply to those runs.
M2 guards now bind a true parity pass to the exact ONNX SHA-256 before load/lock.
Required CPU policies are checked before monitoring starts.

Power now uses battery `current_now` and `voltage_now` in the one-second fast
loop's persistent root shell. The runner does not average the sensor; the kernel
driver's filtering is unspecified. Samples and call windows share `time.monotonic`.
`power_summary` reports arithmetic sample means for the block, inside call windows
and outside, with counts and boundary-overlap counts. Dry runs return empty power
samples and null means, and remain NOT VALID TIMING.
Skin/Android status and battery status/memory remain on five-second monitors.
The fast sampler, root-shell pump and persistent root shell exclude inference
cores 4–5; masks are checked and power rows record sampler/root masks.
Live root affinity and the new sampling path still need an owner full run.

Annotations live once in `annotations/`, referenced by path/hash in
[ASSET_INDEX.json](../ASSET_INDEX.json). Parity components are referenced by
path/hash in `parity_desk3/feed_components.json`: one image/boxes archive per
photo, one W/alpha vocabulary archive per model outside the repository.
Existing parity JSONs retain their original full-feed hashes. Component extraction
verified exact NPY bytes. Runtime locks now live in the external home cache.
Desk reruns no longer copy images into Downloads. Our threshold mapping deliberately
uses float64; it does not claim upstream float32 bitwise equivalence.

The following DESK3 and DESK2 records describe their historical candidates.

---

# RELATE_DESK3 update — motors-off research candidate

Both models passed the owner-fixed real-pair parity criterion on horse and
blocked_1 using identical native PIL feeds on PyTorch and native ORT 1.25.1.
`parity_valid_<model>.json` and `parity_desk3/` contain the new evidence.
M2 now requires `parity_valid_relsgg-vits16.json` passed=true in both entry points;
its old external horse_parity.json is historical and remains unchanged.
M2 ran the same seven saved photos; annotations are copied to Downloads/relate_desk3.
Threshold mapping now uses float64. All desk and Coder dry-run times are
INFORMAL — agents resident, NOT VALID TIMING. No timed block was run.

The speed runner now refuses robot/LLM/other benchmark processes, manages and
restores screen timeout with the #123 root form, uses one monotonic clock and
saves sample times relative to block start. Process checks run before/after the
measured block, outside its sampling tail. MemAvailable summary excludes samples
outside the planned block. Fresh filenames preserve all earlier evidence.
See [RUN.md](RUN.md) for the four exact owner commands after review/owner decision.

The following is the historical RELATE_DESK2 record. Its STOP and guard details
refer to that frozen candidate, whose evidence files have been preserved.

---

# RELATE_DESK2 — motors-off research candidate

Base: `24b236aae9cc9d24924b09d240aebb7805389df4` on main. No robot integration.
All third-party source, weights, graphs and export venv remain outside this repo.

M1 (relsgg-vits16plus) passed horse sanity and ran the seven selected photos.
M2 (relsgg-vits16) exported, but FAILED the required horse parity check.
STOP M2: desk inference and speed dry-run were not run. The loaders refuse M2
while its external `horse_parity.json` does not report a pass. A future M2
repair/export requires an owner follow-up; do not treat the random-input export
check as sufficient. No valid timing exists for either model.

Native Termux desk commands (agents may remain resident):

```sh
python benchmark/relate_anything/desk2/desk_check.py --select
python benchmark/relate_anything/desk2/desk_check.py --model relsgg-vits16plus
python benchmark/relate_anything/desk2/self_check.py
```

Selection uses the deployed detector call at 640 on EXIF-oriented PIL RGB images;
box_xyxy and class_name are preserved. `speed_photo.jpg` has longest edge 640
(482×640 here); its detector boxes were computed on the decoded saved JPEG.
The two relation inputs use square PIL BILINEAR resize, RGB CHW float32 /255,
32 padded normalized cxcywh boxes and int64 true count. PIL bilinear can differ
numerically from OpenCV INTER_LINEAR on downsampling; no cv2 is imported.

Each head uses its bank's default 35 names in order and sidecar calibration.
One best predicate per valid ordered pair, top 10 by calibrated relation score,
without detector confidence weighting. Failed thresholds are retained and flagged.
Thresholds use runtime.py's sigmoid(a*logit(clipped bank thr)+b); missing thresholds
fall back to 0.40. The pre-existing GT-box/pair_weight=0 versus fusion w=1 regime
mismatch is unchanged. Sanity requires person→riding→horse in the top five,
regardless of its bank threshold flag.

`export_m2.py` and `parity_m2.py` are Debian-venv tools only. They invoke/import
external author code; no external source is copied here. Export uses EMA,
legacy exporter (dynamo off), opset 17, input vocab, 448, 32. `--vocab-npz`
encodes all 243 bank rows at export; dynamic vocabulary inputs are sliced to
35 at inference/parity. The published M2 sidecar is preserved externally as
`published_relateanything.json` before export overwrites relateanything.json.
The native horse input archive (including W/alpha) is also external.

## Later human timed run — DO NOT RUN with an agent

From native Termux only, unplugged, screen on, Termux in front, no Codex/Claude/
AGY/node processes. Full runs check those prerequisites and root battery reads,
idle five minutes, establish an idle VIRTUAL-SKIN baseline, then use power_map's
skin ≤ idle+1.5°C gate (eight-minute bound; timeout flagged WARM START). One model
per fresh invocation, order M1 then M2 only after M2 parity is repaired and reviewed:

```sh
python benchmark/relate_anything/desk2/speed_block.py --model relsgg-vits16plus
# M2 is currently blocked. Do not run this command yet:
# python benchmark/relate_anything/desk2/speed_block.py --model relsgg-vits16
```

One exclusive runner lock. Two ORT intra-op threads: one identified new worker
plus a dedicated caller, both pinned to cores 4–5 using power_map's method and
readback. Affinity is checked before/after calls and every second. Foreground
cpuset loss, agent arrival, charger connection, sensor errors, heat limits or
cadence overruns invalidate/stop the block. Reinvoke a fresh block after restoring
conditions; no partial block is resumed. The production path has not been run.

The saved JPEG and precomputed boxes are evaluated at slots 0,5,…,175 s in a
180 s block. Each call records ORT ms and preprocessing/inference/decoding caller
ms; the summary uses ORT median/P95. CPU caps/CPU zones/battery temperature and
affinity every 1 s; Android status/skin and battery watts/MemAvailable/VmHWM every
5 s. Monitoring, limits and cap accounting reuse the power_map/coresidency code.
Missing readings remain unknown. Output includes raw readings and caps by policy.
Load ms is warm/unspecified-cache, not a cold-load measurement.

Only M1 --dry-run was run: two calls, MID pinning checked, explicitly NOT VALID.
Dry-run intentionally skips agent/charger/idle/skin gates and sensor sampling so
it can verify inference from this agent-resident proot session using native Python.
Root is unavailable in proot; production refuses proot and requires native root.
M2's required STOP takes precedence over its requested dry-run.
