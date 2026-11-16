# VIS-3B: b1609-dotprod (old) vs b11716 (new), one independent test per launch, motors off

Preparation only: no measured results yet. The owner runs it from native Termux with no agent resident. `vis3_run.py` imports
`vis2/vis2_run.py` unchanged (and through it `vis1/vis1_run.py`). It reuses their telemetry bridge, battery, thermal,
abort, agent and affinity stops, cache-drop handshake, owned-process stop, LMK query, PSS and power.

VIS-3B (owner, 2026-10-22) replaced the multi-phase O,N,N,O design. Each launch runs one test in one configuration,
saves it in its own folder and exits; nothing runs automatically after it. The 32 °C gate and the cooling wait are gone
(the timed run `runs/20261022T153641Z` missed that gate at idle skin 32.95 °C). The genuine stops remain:
battery ≥ 50 % Discharging, Android CRITICAL, battery ≥ 45 °C, owner abort, agent, cores 4–7, LMK, clean exit.
Earlier folders in `runs/` are preserved as they are.

## Why (diagnosis, read-only)

The builds are equivalent: same Clang 21.1.8, `-O3`, `-march=armv8.2-a+dotprod+fp16`, OpenMP, REPACK, LLAMAFILE,
shared libraries, no KleidiAI. Both use `sdot` (1045 vs 1117) and 456 FP16 `fmla .8h`. The effective runtime defaults
also match: mmap (OLD default `mmap`, NEW `auto` → mmap on CPU), flash attention `auto` → enabled, batch 2048 / ubatch
512, `--fit on`, `--swa-full` and `--cache-ram 0` honoured, and the MTP draft inherits the target's threads. The earlier
OLD/NEW figures were taken at start skins between 24 and 39 °C with capped CPUs, so they do not isolate the build. Full
evidence is in `reports/VIS3_CODER_REPORT.md`.

## Tests

| `--test` | Measures | Before the measurement |
|---|---|---|
| `cold_load` | spawn → `/health` ok, log-phase timestamps | page-cache drop (native handshake) |
| `warm_load` | spawn → `/health` ok | every file the server reads, read once (`prime`) |
| `first_response` | wall time of the first request (gen[0] prompt, 16 tokens) | prime, load, no warm-up |
| `prompt` | 4 × (U ≈1005-token uncached, K cached prefix), 8 tokens each | prime, load, 1 discarded request |
| `gen` | the 6 VIS-2 prompts, greedy, ≤ 128 tokens, no drafter | prime, load, 1 discarded request |
| `mtp` | as `gen`, with the drafter (`--spec-draft-n-max 3`); draft acceptance | as `gen` |
| `vision` | 4 VIS-1 frames through the Q8 projector, 70 image tokens (new build only) | prime, load, 1 discarded image |

`--build old|new` is required. `--mtp` adds the drafter to any test except `gen` (use `--test mtp`). `--q8` adds the
projector on the new build (implied by `vision`). The command is exactly `server_manager.start_setup('setup_q4')` (ctx
2048, 4/4 threads, parallel 1, `--swa-full --cache-ram 0`), plus `taskset -c 4-7`, `timeout 900` and port 8091. Old and
new differ only in the binary path (test-checked against `server_manager.py`).

Each test checks:
- `/props` and `/slots` (projector, drafter, n_ctx 2048);
- `/proc/<pid>/maps` (only the build's own libraries);
- hidden-setting environment variables (`LLAMA_*`, `GGML_*`, `OMP_*`, `KMP_*`, `LD_LIBRARY_PATH`, `LD_PRELOAD`), which
  are refused.

Each test records:
- operating conditions at the start and end of the measurement: skin, Android status, battery °C, %, V and W, CPU caps,
  MemAvailable, zone temperatures;
- mean CPU frequency and time at top frequency (policy4/6, from cpufreq `time_in_state`) over the measurement and per
  request;
- PSS after load and at the end, peak PSS, minimum MemAvailable;
- mean battery W, caps %, LMK, exit code and stop time.

A test that already has a VALID result with identical configuration, code, payloads, and the same hashes for the weights and
build it uses is refused by preflight. A rebuild of the other build does not reopen it (`--repeat` to measure it again).

## Owner commands (native Termux, no agent running, battery ≥ 50 % and unplugged, screen on, Termux in front)

```bash
O=~/robot/benchmark/gemma_vision/vis3
bash $O/oneshot.sh --preflight --test gen --build old       # readiness for that test, starts nothing
bash $O/oneshot.sh --test cold_load --build old
bash $O/oneshot.sh --test cold_load --build new
bash $O/oneshot.sh --test gen --build old
bash $O/oneshot.sh --test gen --build new
# likewise: warm_load, first_response, prompt, mtp (old/new), vision (new), --mtp / --q8 variants
python3 -B $O/vis3_run.py --report $O/runs/<old folder> $O/runs/<new folder>
```

Expected time per launch: about 10 s for the monitor to start, then load 5–10 s and priming a few seconds.
Measurement: load tests about 10 s, `first_response` about 3 s, `prompt` about 3 min, `gen`/`mtp` 1–2 min, `vision`
about 1 min. For comparable pairs, run old and new back to back and alternate their order when repeating.

Folders: `vis3/runs/<UTC>_<test>_<build>[_mtp][_q8][_repeat]/`, named from the validated configuration whatever the
argument spelling or order. Each holds `run.json` (configuration, command, payload and code hashes, hashes of the weights and the one
build this test uses, base commit, git status), `result.json`, `server.log`, `samples.jsonl`, `console.log`.

`--report A B` prints both results and B/A ratios, then two separate assessments. `configuration` (from both
`run.json` manifests) says whether the runs measured the same thing. Only the build may differ; it checks the same test and
`--mtp`/`--q8`, the same command apart from the binary, and the same payloads, weights, harness code and constants.
`conditions` says whether the two ran under matched operating conditions:
- same test, both VALID;
- start skin within 1.0 °C, both at Android status 0;
- battery within 10 %;
- equal policy4/6 caps at the start.

It also reports separately whether the measured mean frequencies are within 5 % (informational: they also reflect each
build's own load), and whether prompt token counts and generated lengths are identical. Unmatched pairs are labelled as
such, not discarded. Short tests do not establish thermal equilibrium (#93). These are Termux prototype results, not
the Android app.
