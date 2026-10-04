# RELATE_SPEED1 findings and evidence

Coder: Codex CLI, assigned gpt-6.1-sol, medium, Ponytail lite. Reviewer: Claude Code CLI headless, claude-opus-5-5, effort N/A, Ponytail off. No fallback used.

Initial and current HEAD: `b29debab424217bd145b77f12cc952d450fd285f`. Initial git status --short: empty. Motors off. No timed block or timed session, stage, commit or push. All times below are **INFORMAL — agents resident, NOT VALID TIMING**.

FIX1 F1: nonempty root mask excluding the measured cluster is accepted even if narrower than Termux; root/su/policies are checked before idle. Self-check covers safe/narrow/empty/overlapping masks and actual prepare_monitor with mocks. FIX1 F2: all four Send filenames now match the `_fix1` commands. F3: inspected ONNX initializers: reference and successful exports have no external-data tensors. No earlier committed evidence changed.

Shared logic: speed1 imports desk2/speed_block.py; extracted run_block/prepare_monitor/load_speed_input and added explicit cluster/thread/head parameters rather than copying the block. desk_check.Head accepts size/thread/provider/external-directory options while keeping its default M2 parity guard and decoder. No upstream code changes, graph hand edits, native package changes or venv additions.

## Build evidence

Debian venv versions: torch 2.13.0+cpu, onnx 1.23.1, onnxruntime 1.30.0, numpy 2.5.2, transformers 5.14.1. Python: 3.13.5 (main, Aug 10 2026, 12:06:59) [GCC 14.2.0].

Unchanged exporter SHA-256: `2f0c2511eb6a0d07f314e03ba18c96cdcc5bad70507911f7ec206dd7230f276e`; before/after identical: True. Exporter `--img-size` accepts an integer and passes it to RelateAnything.from_checkpoint and random tensor S; both actual exports with --check succeeded. Other settings match export_m2.py: ema, input vocabulary, opset17, max_boxes32, default legacy exporter, all243 bank rows dynamic; decoding default35.

Native ORT 1.25.1: `get_available_providers() = ['NnapiExecutionProvider', 'XnnpackExecutionProvider', 'CPUExecutionProvider']`. Current and all constructed session optimization levels: `GraphOptimizationLevel.ORT_ENABLE_ALL` (ORT_ENABLE_ALL is set by default; no override).

| Variant | Availability / quality | Graph bytes | SHA-256 |
|---|---|---:|---|
| fp32 reference | AVAILABLE; reproduced committed outputs | 179495602 | `24a2e2a2ef8ea7d1b7cffef9fe5e35ab9c8eeb413e73eb4dee5afa9b24dd98d5` |
| int8 | NOT AVAILABLE; quality FAIL / excluded | 53796701 | `d7d41edaec13b574179e69487cab690db6892367ffde97de712e487c4836ab16` |
| xnnpack | AVAILABLE; quality FAIL / excluded | same fp32 graph | `24a2e2a2ef8ea7d1b7cffef9fe5e35ab9c8eeb413e73eb4dee5afa9b24dd98d5` |
| img384 | AVAILABLE; quality FAIL / excluded | 179495602 | `a39133aea268810ccedbeee158933040bfe1ad2e35354917c32f92c3479542d8` |
| img336 | AVAILABLE; quality FAIL / excluded | 179495602 | `1d3912a3df4cfcf99078c635d4e4f6eb8c9853d776087efc0b4afde2b2d59b90` |

V2 creates no graph. XNNPACK initialized with one private thread (pinned ORT caller executes kernels) and CPU fallback, preserving two ORT threads on MID/BIG. Both horse and blocked_1 parity attempts failed at cross_attn/Reshape_7: input {1,384}, requested {32,1,384}. No parity PASS, no speed screening or timing.

V1: default dynamic QInt8 option failed datatype inference; DefaultTensorType=FLOAT produced a 53,796,701-byte graph but native loading failed MatMulInteger incompatible dimensions. That graph is retained externally as failed_default_tensor_type.onnx. Constant MatMul-only/MatMulConstBOnly retry failed the same datatype inference; no final runnable int8 graph. Retained failed graph size/hash above, not a valid candidate. Every failed option is reported; graphs were not edited.

Exact build invocations: `/termux-home/ext/venv-relate-export/bin/python benchmark/relate_anything/speed1/build_variants.py`, then `.../python benchmark/relate_anything/speed1/quantize_int8.py`, then `.../python benchmark/relate_anything/speed1/quantize_matmul.py`. API calls and exporter commands follow (the semicolon below separates invocation from the recorded API expression; it is not an owner shell command):


int8:
```text
/termux-home/ext/venv-relate-export/bin/python /data/data/com.termux/files/home/robot/benchmark/relate_anything/speed1/quantize_matmul.py; quantize_dynamic('/termux-home/models/relate_anything/relsgg-vits16/relateanything.onnx', '/termux-home/models/relate_anything/relsgg-vits16-int8/relateanything.onnx', weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'], extra_options={'MatMulConstBOnly': True})
```
```text
/termux-home/ext/venv-relate-export/bin/python /data/data/com.termux/files/home/robot/benchmark/relate_anything/speed1/build_variants.py: quantize_dynamic('/termux-home/models/relate_anything/relsgg-vits16/relateanything.onnx', '/termux-home/models/relate_anything/relsgg-vits16-int8/relateanything.onnx', weight_type=QuantType.QInt8)
```
RuntimeError("Unable to find data type for weight_name='/model/spatial_pool/scene_pe/proj/Gemm_output_0_MatMul'. shape_inference failed to return a type probably this node is from a different domain or using an input produced by such an operator. This may happen if you quantize a model already quantized. You may use extra_options `DefaultTensorType` to indicate the default weight type, usually `onnx.TensorProto.FLOAT`.")
```text
/termux-home/ext/venv-relate-export/bin/python /data/data/com.termux/files/home/robot/benchmark/relate_anything/speed1/quantize_int8.py; quantize_dynamic('/termux-home/models/relate_anything/relsgg-vits16/relateanything.onnx', '/termux-home/models/relate_anything/relsgg-vits16-int8/relateanything.onnx', weight_type=QuantType.QInt8, extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT})
```
MatMulInteger ShapeInferenceError: Incompatible dimensions for matrix multiplication (quality_initial.stderr)
RuntimeError("Unable to find data type for weight_name='/model/spatial_pool/scene_pe/proj/Gemm_output_0_MatMul'. shape_inference failed to return a type probably this node is from a different domain or using an input produced by such an operator. This may happen if you quantize a model already quantized. You may use extra_options `DefaultTensorType` to indicate the default weight type, usually `onnx.TensorProto.FLOAT`.")

img384:
```text
/termux-home/ext/venv-relate-export/bin/python /termux-home/ext/RelateAnything/deploy/export_onnx.py --checkpoint /termux-home/models/relate_anything/relsgg-vits16/model.pth --out /termux-home/models/relate_anything/relsgg-vits16-img384/relateanything.onnx --vocab-npz /termux-home/models/relate_anything/relsgg-vits16/predicate_bank.npz --vocab-mode input --opset 17 --img-size 384 --max-boxes 32 --weights ema --check
```

img336:
```text
/termux-home/ext/venv-relate-export/bin/python /termux-home/ext/RelateAnything/deploy/export_onnx.py --checkpoint /termux-home/models/relate_anything/relsgg-vits16/model.pth --out /termux-home/models/relate_anything/relsgg-vits16-img336/relateanything.onnx --vocab-npz /termux-home/models/relate_anything/relsgg-vits16/predicate_bank.npz --vocab-mode input --opset 17 --img-size 336 --max-boxes 32 --weights ema --check
```

## Fixed correctness / quality

Reference: committed desk3 M2 CPU EP, same seven photos/detections/boxes, max_per_pair=1, top10, calibrated scores, default35 bank. Exact subject/object box tuples identify triplets. Owner criterion remains literal: horse top1 person-riding-horse; each photo >=9 overlapping triplets; every common score delta<=0.05. Pass/fail threshold flips are diagnostics, not an added exclusion rule. V2 uses same real-pair set and <=1e-3 pred/pair logits delta on horse + blocked_1.

Both img384 and img336 pass horse top1 person-riding-horse. One reference image only has two triplets, so even exact reproduction is 2/10 and fails the unchanged changed-numbers rule. Baseline fp32 is the reference and is not subjected to that changed-numbers criterion; all its decoded triplets/scores exactly reproduce the committed outputs. Smaller images also fail other photos independently of this shortfall.

| Variant | Photo | Overlap | Max score delta | Threshold flips | Fixed criterion |
|---|---|---:|---:|---:|---|
| int8 | test_photos/PXL_20260405_213419918.MP.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| int8 | test_photos/PXL_20260405_213509269.MP.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| int8 | test_photos/PXL_20260405_213514191.MP.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| int8 | test_photos/PXL_20260405_213536608.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| int8 | test_photos/abdel_close_back.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| int8 | bench_photos/blocked_1.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| int8 | bench_photos/roomsig_couch_1.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| xnnpack | test_photos/PXL_20260405_213419918.MP.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| xnnpack | test_photos/PXL_20260405_213509269.MP.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| xnnpack | test_photos/PXL_20260405_213514191.MP.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| xnnpack | test_photos/PXL_20260405_213536608.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| xnnpack | test_photos/abdel_close_back.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| xnnpack | bench_photos/blocked_1.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| xnnpack | bench_photos/roomsig_couch_1.jpg | N/A | N/A | N/A | NOT EVALUABLE — build/load/inference failure; excluded |
| img384 | test_photos/PXL_20260405_213419918.MP.jpg | 9 | 0.082713634 | 1 | FAIL |
| img384 | test_photos/PXL_20260405_213509269.MP.jpg | 8 | 0.068978906 | 1 | FAIL |
| img384 | test_photos/PXL_20260405_213514191.MP.jpg | 7 | 0.066141099 | 0 | FAIL |
| img384 | test_photos/PXL_20260405_213536608.jpg | 1 | 0.023928046 | 0 | FAIL |
| img384 | test_photos/abdel_close_back.jpg | 10 | 0.040409625 | 1 | PASS |
| img384 | bench_photos/blocked_1.jpg | 10 | 0.102985620 | 0 | FAIL |
| img384 | bench_photos/roomsig_couch_1.jpg | 9 | 0.038132489 | 1 | PASS |
| img336 | test_photos/PXL_20260405_213419918.MP.jpg | 9 | 0.055845946 | 0 | FAIL |
| img336 | test_photos/PXL_20260405_213509269.MP.jpg | 8 | 0.061023355 | 1 | FAIL |
| img336 | test_photos/PXL_20260405_213514191.MP.jpg | 7 | 0.032528818 | 0 | FAIL |
| img336 | test_photos/PXL_20260405_213536608.jpg | 1 | 0.074331284 | 0 | FAIL |
| img336 | test_photos/abdel_close_back.jpg | 10 | 0.100605071 | 0 | FAIL |
| img336 | bench_photos/blocked_1.jpg | 10 | 0.055094123 | 0 | FAIL |
| img336 | bench_photos/roomsig_couch_1.jpg | 9 | 0.046584234 | 1 | PASS |

## Informal MID screening — NOT VALID TIMING

fp32: warmup then three fixed speed-photo calls; median 1116.809 ms, cpus4-5, two threads, readback before/after every call. **INFORMAL — agents resident, NOT VALID TIMING**. Failed variants received no speed blocks or fixed-photo screening; per-photo ORT durations attached to quality outputs are informal inference diagnostics only. No int8 speed benefit assumed.

Valid historical owner reference only: owner_speed_m2.json median/P951128/1142 ms, fp32 MID, two threads. Current informal numbers cannot validate or replace it.

## LITTLE / session design

power_map.py:124: `opts.intra_op_num_threads = 2 if setting == 'mid' else 4`; line131: `cpus = {4, 5} if setting == 'mid' else {0, 1, 2, 3}`; R3 enumerates default, mid, little at lines37-39. Hence LITTLE uses0-3/four threads. MID4-5/two, BIG6-7/two.

Fixed plan: REF fp32 MID; fp32 BIG; fp32 LITTLE; REF fp32 MID. No other passing variant or fastest variant different from fp32.17 min plus setup/load;49 min plus setup/load at maximum gates. Root/su/pump masks are repinned for every cluster; one-second sampler and root checked every power sample; nonempty exclusion masks mandatory. All preflight checks before one five-minute idle; existing skin gate before each180s block; full per-block JSON plus session records. Warm starts get NOT VALID — WARM START; dry gets NOT VALID — DRY RUN.

Continuation: only cadence failure with successful cleanup/no monitor error is local and may continue. Any root/sensors, thermal, charger, agent/process, cores/affinity, inference or cleanup failure stops later blocks. Refused blocks plus unrun plan are recorded. A warm-start block is recorded and next gate can cool independently. No live timed-path validation claimed. VmHWM is cumulative across the session.

## Checks

- self_check: exit0
- power_map_units: exit0
- coresidency_units: exit0
- detector_units: exit0
- mission_units: exit0
- cycle_units: exit0
- ast: exit0

Initial self-check failed because the pre-idle refusal retained the standalone runner lock; fixed by explicit close in finally and reran the changed self-check. Already-passing direct scripts were not repeated. Their robot/motor/camera messages use mocks, not hardware. Initial quality had the reference incorrectly labelled with the changed-numbers rule; final quality explicitly distinguishes baseline reproduction. Initial dry-run passed; after final input/mask/refusal checks, a second final-source dry-run passed. All dry runs are NOT VALID.

Final self-check output:
```text
PASS: RGB /255 CHW, square resize, cxcywh padding/count, invalid inputs, masks, score order, one predicate/pair, below-threshold retention
screen timeout restored/read back 60000 ms
PASS: explicit float64 formula (not runtime bitwise parity), missing/false/stale-graph M2 guards in both entry points, process refusal, screen restore after set failure
PASS: first-five shortfall checked before adding extras
PASS: required policies; one-second current_now fields, sign/units/monotonic window means, boundary counts, null dry means and non-MID sampler/root refusal
PASS: nonempty non-MID root masks accepted, empty/MID masks refused
PASS: all cluster/thread rules; literal quality thresholds and box identity; plan/order; continuation fail-closed; pre-idle root refusal; one session idle and cleanup (mocks, NOT VALID TIMING)
PASS: actual prepare_monitor accepts narrower safe root/su masks (mock), before idle
```

Final dry-run stdout (NOT VALID), includes every block and null power means/counts:
```text
Planned blocks: [{"block": 1, "variant": "fp32", "cluster": "MID", "cpus": [4, 5], "threads": 2, "output": "/termux-home/robot/benchmark/relate_anything/speed1/session_dry_final_block_01_fp32_MID.json"}, {"block": 2, "variant": "fp32", "cluster": "BIG", "cpus": [6, 7], "threads": 2, "output": "/termux-home/robot/benchmark/relate_anything/speed1/session_dry_final_block_02_fp32_BIG.json"}, {"block": 3, "variant": "fp32", "cluster": "LITTLE", "cpus": [0, 1, 2, 3], "threads": 4, "output": "/termux-home/robot/benchmark/relate_anything/speed1/session_dry_final_block_03_fp32_LITTLE.json"}, {"block": 4, "variant": "fp32", "cluster": "MID", "cpus": [4, 5], "threads": 2, "output": "/termux-home/robot/benchmark/relate_anything/speed1/session_dry_final_block_04_fp32_MID.json"}]
Estimated owner session: 17.0 min + loads/setup; up to 49.0 min with all gates at 8 min.
INFORMAL — agents resident, NOT VALID TIMING — two calls per block, no idle/gates/sensors
INFORMAL — agents resident, NOT VALID TIMING — dry-run; no charger/agent/idle/skin gates or sensor sampling
{
  "model": "relsgg-vits16",
  "label": "INFORMAL \u2014 agents resident, NOT VALID TIMING \u2014 DRY RUN",
  "photo": "/data/data/com.termux/files/home/robot/benchmark/relate_anything/desk2/speed_photo.jpg",
  "photo_sha256": "4da2d789dd9a60405658184de81e9f46b3c55e09024a6f2ce26a934d899d0fd8",
  "cadence_s": 5,
  "planned_s": 180,
  "intra_op_threads": 2,
  "cpus": [
    4,
    5
  ],
  "sampling": {
    "caps_cpu_battery_temp_s": 1,
    "battery_power_s": 1,
    "skin_android_status_s": 5,
    "battery_status_memory_s": 5
  },
  "power_source": {
    "current": "/sys/class/power_supply/battery/current_now",
    "voltage": "/sys/class/power_supply/battery/voltage_now",
    "kind": "current_now sysfs reading; no runner averaging; driver filtering unspecified"
  },
  "variant": "fp32",
  "cluster": "MID",
  "load_ms": 1936.8839529925026,
  "ort_tids": [
    17342,
    17343
  ],
  "providers": [
    "CPUExecutionProvider"
  ],
  "optimization_level": "GraphOptimizationLevel.ORT_ENABLE_ALL",
  "artifact_identity": {
    "relateanything.onnx": "24a2e2a2ef8ea7d1b7cffef9fe5e35ab9c8eeb413e73eb4dee5afa9b24dd98d5",
    "relateanything.json": "e70152faa7875bf9ae865d14b3539b7aac358f33bfb10db265365441db26bfaa",
    "predicate_bank.npz": "6043ff8fb765db34d89019617979d566b79967b7740c72bb4007786179839f48"
  },
  "block_start_s": 0.0,
  "time_base": "time.monotonic; all sample times relative to block start",
  "duration_s": 6.1674322949838825,
  "median_ort_ms": 1157.9255990218371,
  "p95_ort_ms": 1175.2026803209446,
  "VmHWM_KiB": 443616,
  "min_MemAvailable_MiB": null,
  "power_summary": {
    "mean_battery_w": null,
    "mean_inside_calls_w": null,
    "mean_outside_calls_w": null,
    "samples": 0,
    "inside_samples": 0,
    "outside_samples": 0,
    "classification": "sample receipt t in [started_s, ended_s); block 0 <= t <= duration",
    "read_overlaps_call_boundary": 0
  },
  "validity": "NOT VALID \u2014 DRY RUN"
}
Evidence: /termux-home/robot/benchmark/relate_anything/speed1/session_dry_final_block_01_fp32_MID.json
INFORMAL — agents resident, NOT VALID TIMING — dry-run; no charger/agent/idle/skin gates or sensor sampling
{
  "model": "relsgg-vits16",
  "label": "INFORMAL \u2014 agents resident, NOT VALID TIMING \u2014 DRY RUN",
  "photo": "/data/data/com.termux/files/home/robot/benchmark/relate_anything/desk2/speed_photo.jpg",
  "photo_sha256": "4da2d789dd9a60405658184de81e9f46b3c55e09024a6f2ce26a934d899d0fd8",
  "cadence_s": 5,
  "planned_s": 180,
  "intra_op_threads": 2,
  "cpus": [
    6,
    7
  ],
  "sampling": {
    "caps_cpu_battery_temp_s": 1,
    "battery_power_s": 1,
    "skin_android_status_s": 5,
    "battery_status_memory_s": 5
  },
  "power_source": {
    "current": "/sys/class/power_supply/battery/current_now",
    "voltage": "/sys/class/power_supply/battery/voltage_now",
    "kind": "current_now sysfs reading; no runner averaging; driver filtering unspecified"
  },
  "variant": "fp32",
  "cluster": "BIG",
  "load_ms": 1994.5445159683004,
  "ort_tids": [
    17347,
    17348
  ],
  "providers": [
    "CPUExecutionProvider"
  ],
  "optimization_level": "GraphOptimizationLevel.ORT_ENABLE_ALL",
  "artifact_identity": {
    "relateanything.onnx": "24a2e2a2ef8ea7d1b7cffef9fe5e35ab9c8eeb413e73eb4dee5afa9b24dd98d5",
    "relateanything.json": "e70152faa7875bf9ae865d14b3539b7aac358f33bfb10db265365441db26bfaa",
    "predicate_bank.npz": "6043ff8fb765db34d89019617979d566b79967b7740c72bb4007786179839f48"
  },
  "block_start_s": 0.0,
  "time_base": "time.monotonic; all sample times relative to block start",
  "duration_s": 5.650171756045893,
  "median_ort_ms": 623.0340169859119,
  "p95_ort_ms": 626.593990373658,
  "VmHWM_KiB": 447484,
  "min_MemAvailable_MiB": null,
  "power_summary": {
    "mean_battery_w": null,
    "mean_inside_calls_w": null,
    "mean_outside_calls_w": null,
    "samples": 0,
    "inside_samples": 0,
    "outside_samples": 0,
    "classification": "sample receipt t in [started_s, ended_s); block 0 <= t <= duration",
    "read_overlaps_call_boundary": 0
  },
  "validity": "NOT VALID \u2014 DRY RUN"
}
Evidence: /termux-home/robot/benchmark/relate_anything/speed1/session_dry_final_block_02_fp32_BIG.json
INFORMAL — agents resident, NOT VALID TIMING — dry-run; no charger/agent/idle/skin gates or sensor sampling
{
  "model": "relsgg-vits16",
  "label": "INFORMAL \u2014 agents resident, NOT VALID TIMING \u2014 DRY RUN",
  "photo": "/data/data/com.termux/files/home/robot/benchmark/relate_anything/desk2/speed_photo.jpg",
  "photo_sha256": "4da2d789dd9a60405658184de81e9f46b3c55e09024a6f2ce26a934d899d0fd8",
  "cadence_s": 5,
  "planned_s": 180,
  "intra_op_threads": 4,
  "cpus": [
    0,
    1,
    2,
    3
  ],
  "sampling": {
    "caps_cpu_battery_temp_s": 1,
    "battery_power_s": 1,
    "skin_android_status_s": 5,
    "battery_status_memory_s": 5
  },
  "power_source": {
    "current": "/sys/class/power_supply/battery/current_now",
    "voltage": "/sys/class/power_supply/battery/voltage_now",
    "kind": "current_now sysfs reading; no runner averaging; driver filtering unspecified"
  },
  "variant": "fp32",
  "cluster": "LITTLE",
  "load_ms": 2019.9072269606404,
  "ort_tids": [
    17349,
    17350,
    17351,
    17354
  ],
  "providers": [
    "CPUExecutionProvider"
  ],
  "optimization_level": "GraphOptimizationLevel.ORT_ENABLE_ALL",
  "artifact_identity": {
    "relateanything.onnx": "24a2e2a2ef8ea7d1b7cffef9fe5e35ab9c8eeb413e73eb4dee5afa9b24dd98d5",
    "relateanything.json": "e70152faa7875bf9ae865d14b3539b7aac358f33bfb10db265365441db26bfaa",
    "predicate_bank.npz": "6043ff8fb765db34d89019617979d566b79967b7740c72bb4007786179839f48"
  },
  "block_start_s": 0.0,
  "time_base": "time.monotonic; all sample times relative to block start",
  "duration_s": 8.346086552017368,
  "median_ort_ms": 3258.3548194961622,
  "p95_ort_ms": 3259.1168310435023,
  "VmHWM_KiB": 447616,
  "min_MemAvailable_MiB": null,
  "power_summary": {
    "mean_battery_w": null,
    "mean_inside_calls_w": null,
    "mean_outside_calls_w": null,
    "samples": 0,
    "inside_samples": 0,
    "outside_samples": 0,
    "classification": "sample receipt t in [started_s, ended_s); block 0 <= t <= duration",
    "read_overlaps_call_boundary": 0
  },
  "validity": "NOT VALID \u2014 DRY RUN"
}
Evidence: /termux-home/robot/benchmark/relate_anything/speed1/session_dry_final_block_03_fp32_LITTLE.json
INFORMAL — agents resident, NOT VALID TIMING — dry-run; no charger/agent/idle/skin gates or sensor sampling
{
  "model": "relsgg-vits16",
  "label": "INFORMAL \u2014 agents resident, NOT VALID TIMING \u2014 DRY RUN",
  "photo": "/data/data/com.termux/files/home/robot/benchmark/relate_anything/desk2/speed_photo.jpg",
  "photo_sha256": "4da2d789dd9a60405658184de81e9f46b3c55e09024a6f2ce26a934d899d0fd8",
  "cadence_s": 5,
  "planned_s": 180,
  "intra_op_threads": 2,
  "cpus": [
    4,
    5
  ],
  "sampling": {
    "caps_cpu_battery_temp_s": 1,
    "battery_power_s": 1,
    "skin_android_status_s": 5,
    "battery_status_memory_s": 5
  },
  "power_source": {
    "current": "/sys/class/power_supply/battery/current_now",
    "voltage": "/sys/class/power_supply/battery/voltage_now",
    "kind": "current_now sysfs reading; no runner averaging; driver filtering unspecified"
  },
  "variant": "fp32",
  "cluster": "MID",
  "load_ms": 2058.9617930236273,
  "ort_tids": [
    17369,
    17372
  ],
  "providers": [
    "CPUExecutionProvider"
  ],
  "optimization_level": "GraphOptimizationLevel.ORT_ENABLE_ALL",
  "artifact_identity": {
    "relateanything.onnx": "24a2e2a2ef8ea7d1b7cffef9fe5e35ab9c8eeb413e73eb4dee5afa9b24dd98d5",
    "relateanything.json": "e70152faa7875bf9ae865d14b3539b7aac358f33bfb10db265365441db26bfaa",
    "predicate_bank.npz": "6043ff8fb765db34d89019617979d566b79967b7740c72bb4007786179839f48"
  },
  "block_start_s": 0.0,
  "time_base": "time.monotonic; all sample times relative to block start",
  "duration_s": 6.165648115042131,
  "median_ort_ms": 1157.5906375364866,
  "p95_ort_ms": 1176.2384093919536,
  "VmHWM_KiB": 447616,
  "min_MemAvailable_MiB": null,
  "power_summary": {
    "mean_battery_w": null,
    "mean_inside_calls_w": null,
    "mean_outside_calls_w": null,
    "samples": 0,
    "inside_samples": 0,
    "outside_samples": 0,
    "classification": "sample receipt t in [started_s, ended_s); block 0 <= t <= duration",
    "read_overlaps_call_boundary": 0
  },
  "validity": "NOT VALID \u2014 DRY RUN"
}
Evidence: /termux-home/robot/benchmark/relate_anything/speed1/session_dry_final_block_04_fp32_MID.json
Session label: INFORMAL — agents resident, NOT VALID TIMING — SESSION DRY RUN 
Evidence: /termux-home/robot/benchmark/relate_anything/speed1/session_dry_final.json
```

Native ORT unsupported-Android warning remains. Full stderr/check artifacts are under speed1/. No torch in native Termux. No production sensor/root/screen/idle/gate/timed-transition execution during this task.

