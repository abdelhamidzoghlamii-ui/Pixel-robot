# VIS-2 — Q8 projector residency vs Gemma text speed (MTP on/off), motors off

Preparation only; no measured results yet. Owner-run from native Termux. Reuses `vis1/vis1_run.py`
(imported unchanged) for telemetry, gates, cache drop, process ownership, LMK and power.

Same binary (`~/llama.cpp-vis`, b11716), Gemma E2B Q4_0, ctx 4096, 4 threads on cpus 4–7, parallel 1,
`--swa-full --cache-ram 0 --reasoning off`, port 8091. Configs differ only in these flags:

| Config | MTP drafter (n-max 3) | Q8 mmproj (70-token images) |
|---|---|---|
| A | on | off |
| B | on | on |
| C | off | off |
| D | off | on |
| M | = B, mixed text → image → text check | |

After load each block reads `/props` (`modalities.vision`) and `/slots` (`speculative`) and is
NOT VALID on any mismatch. Flags are never changed to make a run pass.

Per timed block (cold load after cache drop and skin gate): one discarded warm-up, then 6 rounds of
U (≈600-token log, `cache_prompt` false, 8 tokens), K (same log prefix + new question, `cache_prompt`
true) and G (one of 6 VIS-1 prompts, greedy, up to 128 tokens). `A,B,B,A` gives 12 matched samples per
metric per config. M: G, U, one Q8/70 image (VIS-1 frame `person_facing_150cm`), G, U.
Battery: at least 50 %, Discharging (no upper bound since VIS2-BATTERY-FIX); battery percentage and
voltage at the start of every block are saved as `battery_start`. Battery >= 45 °C, Android status
CRITICAL, abort, agent and affinity stops are unchanged.
Caps: 120 s per request (and never past the 600 s block cap), 900 s server lifetime. Battery,
thermal, abort and affinity stops are also checked every 0.5 s while a request runs. PSS is read from
fresh idle samples after load (before warm-up) and before/after the image.

`VALID` (or `SMOKE`) means execution conditions held, including clean server exit and no sample
crossing a stop threshold. It does not by itself make blocks comparable: `--summarize` prints per-block
checks (U uncached, K reused cache, replies non-empty) and whether generation lengths and U prompt
lengths are identical across timed blocks. Conclusions wait for that assessment.

```bash
bash ~/robot/benchmark/gemma_vision/vis2/oneshot.sh --preflight             # readiness, starts nothing
bash ~/robot/benchmark/gemma_vision/vis2/oneshot.sh --smoke --blocks A,M    # first test, NOT timing
bash ~/robot/benchmark/gemma_vision/vis2/oneshot.sh --blocks A,B,B,A,M      # timed (B works)
bash ~/robot/benchmark/gemma_vision/vis2/oneshot.sh --blocks C,D,D,C        # timed fallback (B fails)
```

Everything for one launch goes in one folder `vis2/runs/<UTC>/` (or `smoke_<UTC>/`): `run.json`
manifest (commands, code/payload/weight/binary+library hashes, base commit, git status), one JSON and
server log per block, `samples.jsonl`, `console.log`. A failed block stops the launch; earlier blocks
stay. There is no `--resume`: rerun only the blocks still needed, as a new folder, once per cause.

Summary of any folder (reads saved files only): `python3 -B vis2_run.py --summarize <folder>`.
Short blocks do not establish thermal equilibrium (#93); ABBA only cancels linear drift.
