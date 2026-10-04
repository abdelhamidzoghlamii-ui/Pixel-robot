# R1b findings — RELATE_FIX1 response

> 1. **The M2 guard is not tied to the M2 graph or checkpoint (low–medium).** `desk_check.py`, `require_parity`, only checks `passed`. The parity JSON records `graph_sha256` (24a2e2a2…) and `checkpoint_sha256`, but nothing compares them to the files on disk. If `relateanything.onnx` were re-exported or replaced later, the stale parity file would still unlock M2 desk runs and timing. Fix: compare the stored graph hash with `sha256(MODELS/name/'relateanything.onnx')` before loading.

**FIXED** — require_parity compares graph_sha256 with the actual external ONNX SHA-256 before both Head construction and speed runner locking/loading. The immutable two-photo parity pass remains unchanged. Missing/nonboolean/false pass and replaced graph are refused in self_check. Checkpoint hashes remain provenance: inference loads the graph, not the checkpoint.

> 2. **The production timed path has never run (validation limit; R1 F2 carries over).**
>    - The two final-source dry runs only cover load, two calls and MID pinning.
>    - Never executed for real: root screen handling, the idle/process loop, the skin gate, `RootShell`/`discover`, the four monitors, the relative-time conversion of `fast`/`dumps` rows, and `caps_by_policy`. Self-check covers screen and process handling with mocks only.
>    - `speed_block` also lacks power_map's `set(POLICIES) <= layout['policies']` check (`power_map.py:574`), so a missing policy only shows up as `unknown_s` in the caps.
>    - The owner dry runs in RUN.md steps 1–2 won't exercise this path either. The first real evidence will be step 3.

**FIXED (policy check); NOT FIXED (new live-path validation)** — Required policy0/policy4/policy6 subset check added immediately after discover, before monitors/block. Owner valid frozen-R1 M1/M2 runs now provide production-path evidence for R1. New FIX1 root affinity and power sampling are only tested with mocks; a timed block is explicitly forbidden in this task. No new production validation is claimed.

> 3. **The self-check's claim about runtime threshold precision is inaccurate (low, plausible).** `self_check.py` says `runtime.py:563-565` maps thresholds as scalar float64, including the clip endpoints. But:
>    - `head_thr` is float32 (`runtime.py:206, 249`), and under NumPy 2 rules `np.clip(np.float32, 1e-6, 1-1e-6)` returns float32.
>    - The mapped thresholds are then stored into a float32 vector (`postprocess.py:88-91`).
>    - So the attempt-1 expectation (float32 clip at the endpoints) probably matched runtime better, and the "fix" changed the test to agree with the new code.
>    - Practical effect: none. Only bank values ≥1-1e-6 or ≤1e-6 are affected, there are none, and the differences are about 1e-12. It also means my R1 "float32 vs float64" nit was itself imprecise.
>    - M1's historical results keep float32 thresholds while M2's are float64; cross-model threshold differences are about 1e-7.

**FIXED** — Corrected self_check comment/output: float64 mapping is our deliberate formula, not bitwise upstream runtime parity. Upstream clip/storage precision is float32; historical M1/M2 thresholds and evidence are left unchanged. The check tests our formula explicitly.

> 4. **A runtime lock file is in the candidate (low).** `benchmark/relate_anything/desk2/.speed.lock` is an empty file created when the runner runs. Ignore it or leave it out of any commit.

**FIXED** — Removed .speed.lock. Stable flock file moved to external HOME/.cache/relate_anything/speed.lock; it is kept there to avoid unlink/recreate lock races. Added scoped ignore rules; candidate has no runtime lock, cache or bytecode artifacts.

> 5. **Repository size (low, owner decision).**
>    - The M2 annotated JPEGs are byte-identical to M1's (same SHA256), because annotations only draw detections. That duplicates about 12 MB.
>    - `parity_desk3/*_feed.npz` stores the same image inside each per-model feed, on top of `*_image_boxes.npz`. That is roughly 15 MB of duplicated feed data.

**FIXED** — Seven byte-identical duplicate M2 annotations removed; a single shared annotations directory serves both models. Four full feeds removed; base image/boxes retained once per photo in repo, exact W/alpha NPY bytes retained once per model outside repo because these are vocabulary-bank tensors. ASSET_INDEX.json and feed_components.json record paths, hashes, original removed-file provenance. Removed 23,990,805 bytes (23.990805 MB), excluding external copies; base component bytes and vocabulary bytes were verified identical to original feeds.

> 6. **Nits.**
>    - `mapped_thresholds` evaluates `sigmoid` on NaN/inf before `np.where`, which causes the `logaddexp` RuntimeWarning in `self_check_desk3.stderr`. It is harmless.
>    - `desk_check.run` now copies into `~/storage/downloads/relate_desk3` on every run, with `import shutil` inside the loop. That is a side effect outside the repo on any rerun, including M1.
>    - Unlike `oneshot.sh:113`, nothing warns when the saved screen timeout is already 2147483647.
>    - The `node|main\.py` patterns are broad, but they are inherited from the existing checks.

**FIXED (three nits); NOT FIXED (inherited broad process patterns)** — Nonfinite thresholds are replaced with .5 before sigmoid evaluation, preserving .4 fallback without the NaN warning. Removed automatic Downloads image copies/import shutil; shared annotations stay in the candidate. Added a warning when the saved screen timeout already equals 2147483647. Broad node/main.py patterns remain intentionally conservative to refuse agents/robot code; narrowing them without validated process identities could allow a contaminated timed run.

