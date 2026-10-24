# VIS-3B MTP performance discrepancy — Coder investigation report (read-only)

Coder: Claude Code CLI, claude-opus-5-5, effort medium. (The session hook started Ponytail lite; this is a non-coding
read-only investigation, no code was written.) Base: HEAD = origin/main = 685a44be63aa88c30e09386ce8e5e196ebf11abc.
Pre-existing untracked: benchmark/relate_anything/quality1/reports/round{1,2}_stderr.txt. benchmark/gemma_vision/vis1..3
are git-ignored (local only). No file in the repository was modified; no server, benchmark or motor was run.

R = /data/data/com.termux/files/home/robot

## 1. Answer in one paragraph

The drafter is loaded and works the same as in the historical runs: it produces the same number of output tokens per
target pass (2.20 vs 2.19 on Oct 14) at the same acceptance (40.5 % vs 39.0 %). The gap is not specific to MTP. With
the drafter OFF, the same binary also decodes about 30 % slower in VIS-3B: 9.48 tok/s vs 13.39 (Oct 14) and 14.03
(Sep 28). Each MTP pass is slowed even more, by 1.75x (197.5 ms vs 111–114 ms per pass), than each single-token pass,
by 1.41x (105.5 vs 71–75 ms). That is why the MTP gain shrinks from +43–46 % to +17 %.

On Oct 19 an identical old-build+MTP command, prompt and code path ran at 16.4–18.0 tok/s when the VIS native
root-telemetry monitor was absent, and at 11.06–11.14 tok/s when it ran. The monitored runs started cooler and with the
X1 cores uncapped. The strongest supported explanation (LIKELY) is that the VIS-1/2/3 harness's continuous native
telemetry slows decode. That telemetry is two `su` spawns per sample, back to back every ~0.45 s, including
`cat /proc/<server>/smaps_rollup`. The exact mechanism is UNKNOWN. Binary, libraries, model, drafter, flags, threads and
affinity are identical to the historical runs, apart from ctx 4096 on Oct 14.

## 2. Evidence: drafter loaded and functioning (CONFIRMED)

- Command: `R/benchmark/gemma_vision/vis3/runs/20261023T063009Z_mtp_old/result.json` `command`, which includes
  `--model-draft …/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3`. Built by
  `vis3_run.py:76-77` (`cmd += v1.DRAFT_ARGS`), `vis1_run.py:36-37`.
- Server self-report: `result.json` `reported` = `{'vision': False, 'speculative': [True], 'n_ctx': [2048],
  'build_info': 'b1609-e1a1abb7'}` (checked by `vis3_run.py:166-174`).
- Drafter SHA-256 9eba8199…1919 = #137 and the Oct 14 run.json `model_sha256`. Main model 8e30dff3… is also the same.
- The server log (`…mtp_old/server.log`) has the same non-timing lines as Oct 14 (`R/benchmark/llm_objective_setting/
  conversation/chat_mtp/runs/chat_mtp_20261014T224541Z/server_block01_ON.log`): "common_speculative_init_result:
  loading draft model", KV sharing "layer 3: sharing with layer 34", "layer 0/1/2: sharing with layer 33". A `diff` of
  the two logs, excluding `slot`/`print_timing` lines, differs only in pointer addresses and the port (8091 vs 8080).
  Both logs also show the same "[spec] failed to measure draft model memory" warning.
- Per-request draft stats (server timings): drafted 748, accepted 303 (40.5 %); drafted per target pass 2.96 (n-max 3).
- Mapped libraries are only from `llama.cpp-b1609-dotprod/build/bin` (`result.json` `libs.ok` True).

## 3. Configuration comparison

| Item | Sep 28 conv_mtp | Oct 14 chat_mtp | Oct 23 VIS-3B mtp_old |
|---|---|---|---|
| Binary | b1609-dotprod llama-server | same path | same path |
| libggml-cpu SHA-256 | a4ef0d97… (run_…json) | a4ef0d97… | a4ef0d97… (run.json manifest) |
| Lib mtimes | — | — | all 2026-09-26 (unchanged since build) |
| Model / drafter | same files | 8e30dff3 / 9eba8199 | 8e30dff3 / 9eba8199 |
| Spec flags | draft-mtp, n-max 3 | same | same |
| Threads / batch threads / parallel | 4/4/1 | 4/4/1 | 4/4/1 |
| ctx | 2048 | 4096 | 2048 |
| --swa-full, --cache-ram 0 | yes | yes | yes |
| Affinity | taskset 4-7 | taskset 4-7 | taskset 4-7 |
| Server launched from | proot Debian (ladder oneshot.sh:57) | native Termux (run_chat_mtp.sh:5-7) | proot Debian (vis3 oneshot.sh:92) |
| Prompt format | bench.format_prompt + system msg (101–229 tok) | `<start_of_turn>` user only (18–46 tok) | chat.py TPL-FIX Gemma 4 turn tokens (14–29 tok) |
| Sampling | temp 0 (streamed) | temp 0.7, not streamed | temp 0, top_k 1, seed 0, not streamed |
| n_predict / stops | 512, all EOS (13–38 tok) | 512, EOS/word (9–261 tok) | 128; 4 of 6 hit the cap |
| cache_prompt | false | (chat payload) | false; cache_n 0 on every request |
| Telemetry during requests | thermal log only | root sample every 5.0 s (~0.7 s each) | root sample back to back every ~0.45 s, 2 `su` per sample + smaps_rollup |
| tok/s aggregation | token-weighted, turns ≥10 tok | token-weighted, turns ≥10 tok | median of per-request predicted_per_second |

Flash attention, batch/ubatch, mmap and `--fit` are not on any command line. They are binary defaults and are
therefore the same for the same binary (README "Why").

## 4. Tokens/sec audit and recomputation (server timings only)

All three harnesses use the server's `predicted_n` / `predicted_ms` (accepted output tokens over decode time; prompt time
excluded; draft tokens are not counted as output). VIS: `vis1_run.py:354` and `vis3_run.py:440` (median). Historical:
`conv_speed.py:158-165` (token-weighted, ≥10 tokens), chat_mtp report header (same rule).

Recomputed (pooled = Σpredicted_n / Σpredicted_ms; pass = predicted_n − draft_n_accepted):

| Run | tok | ms/tok | pooled tok/s | accept | tok/pass | ms/pass |
|---|---:|---:|---:|---:|---:|---:|
| Sep 28 OFF | 271 | 71.3 | 14.03 | – | 1.00 | 71.3 |
| Sep 28 ON | 271 | 48.7 | 20.54 | 0.445 | 2.34 | 113.7 |
| Oct 14 OFF | 675 | 74.7 | 13.39 | – | 1.00 | 74.7 |
| Oct 14 ON | 713 | 52.1 | 19.18 | 0.390 | 2.19 | 114.0 |
| Oct 14 OFF, VIS 6 ids only | 551 | 75.6 | 13.23 | – | 1.00 | 75.6 |
| Oct 14 ON, VIS 6 ids only | 498 | 51.3 | 19.50 | 0.387 | 2.17 | 111.0 |
| VIS-3B gen_old (OFF) | 556 | 105.5 | 9.48 | – | 1.00 | 105.5 |
| VIS-3B mtp_old (ON) | 556 | 89.9 | 11.13 | 0.405 | 2.20 | 197.5 |
| VIS-3B gen_new (OFF, b11716) | 556 | 118.6 | 8.43 | – | 1.00 | 118.6 |

- Reported 11.10 (median) vs pooled 11.13: the aggregation method does not matter. All VIS requests are ≥ 24 tokens,
  so the ≥ 10 filter changes nothing.
- The 6-prompt subset is not slower than the 13-prompt workload on Oct 14 (19.50 vs 19.18 ON, 13.23 vs 13.39 OFF).
- Long replies are not intrinsically slow: Oct 14 OFF A5 decoded 261 tokens at 13.08 tok/s (76.45 ms/token).
- The VIS warm-up (16 tokens, first request after load, skin ~25 °C) already ran at 87.85 ms/token (gen_old) and
  88.06 ms/token (mtp_old). Historical runs took 71–76 ms/token.

## 5. The Oct 19 natural experiment (strongest evidence)

`R/benchmark/gemma_vision/vis1/runs/smoke_*`. Every smoke uses the same `vis1_run.py smoke_block` code path
(`vis1_run.py:673-730`): one A5 request (29 prompt tokens), no polling during the request, and an identical command
(`taskset -c 4-7 … b1609-dotprod … --ctx-size 4096 … --model-draft … --spec-type draft-mtp --spec-draft-n-max 3`).

| Smoke (T1 = old build + MTP) | Monitor | Start skin / X1 max | tok/s | accept |
|---|---|---|---:|---:|
| 154839Z | none (SYNTHETIC bridge; Coder agent resident) | n/a | 17.43 | 191/513 |
| 155309Z | none (SYNTHETIC) | n/a | 18.01 | 167/429 |
| 155711Z | none (SYNTHETIC) | n/a | 16.41 | 170/447 |
| 183552Z | none (SYNTHETIC) | n/a | 17.62 | 171/381 |
| 180148Z | native monitor (owner, oneshot) | 28.7 °C / 2850 MHz | 11.14 | 197/423 |
| 193902Z | native monitor (owner, oneshot) | 25.1 °C / 2850 MHz | 11.06 | 136/354 |

The T2 rows (b11716 + MTP) follow the same pattern: 14.5–18.5 tok/s without the monitor, 9.3–9.5 with it. The
"SYNTHETIC" runs are labelled in `NOTE_CODER_SMOKE.txt` and in each block's `reading.source`. VIS-0
(`vis0/runs/20261018T010701Z/server.log`) used b11716, no MTP, no taskset, no monitor and had an agent resident. It
decoded at 74–89 ms/token (~13 tok/s); VIS-3B gen_new took 118.6 ms/token.

The monitor's load: `vis1_run.py:546-606, 526-534`. The monitor process is pinned to cores 0-3. Each sample calls
`root()`, which runs two `su -c` (script, then `cat`), and the script includes `cat /proc/<pid>/smaps_rollup` while the
server runs. In mtp_old `samples.jsonl` the sample interval had a median of 0.45 s, and each root call took 0.445 s with
the server alive vs 0.227 s without it. So the sampler is busy almost continuously during measurement. In Oct 14 chat_mtp
the samples were 5.0 s apart (root_s 0.61–0.79). Where the `su` children are scheduled (cores 0-3 or 4-7) was not
recorded.

## 6. Classified explanations

1. **The VIS native telemetry monitor slows decode by ~30–40 % overall, and MTP passes more than single-token passes —
   LIKELY.** Supported by §5 (same code and command, monitor on or off, the monitored runs cooler and uncapped) and by
   VIS-0. Not CONFIRMED: the smokes were not a controlled experiment (unmeasured thermal state in the synthetic runs,
   one request each, agent resident in the fast ones). The mechanism is UNKNOWN: `su`/process-spawn CPU time landing on
   cores 4-7, mmap_lock or page-table-walk interference from the smaps_rollup reads, or something else.
2. **MTP-specific misconfiguration or missing drafter — ruled out (CONFIRMED not the cause).** §2, plus identical
   tokens/pass and acceptance.
3. **Build, binary or model change — ruled out for the old build (CONFIRMED identical).** Same hashes and mtimes.
4. **Thermal throttling / CPU frequency as the primary cause — UNLIKELY (not established).** A78 policy4 was at 2348 MHz
   (its max) for 99.9 % of the measurement. X1 policy6 was capped for 100 % of the measurement: mean 2308 MHz, scaling_max
   2704 → 2188 MHz, Android status 0, skin 25.0 → 32.0 °C. Within the run, X1 per-request MHz (2549 → 2156) does not
   track tok/s: B7 at 2156 MHz is the fastest request, 12.99. On Oct 19 the monitored runs were slow with X1 uncapped at
   start. The historical runs recorded no cpufreq, so their X1 caps are UNKNOWN and a partial contribution cannot be
   excluded.
5. **Workload differences (6 vs 13 prompts, 128 cap, template, sampling, ctx 2048 vs 4096) — LIKELY minor.** The
   subset and long replies decode at historical speed on Oct 14 (§4), and acceptance and tokens/pass match. They still
   make the 11.10 and 19–20 figures not directly comparable as numbers.
6. **proot launch — UNLIKELY.** Sep 28 also launched the server inside proot and reached 14.03 / 20.54. The proot binary
   is dated 2026-08-23, before all of these runs.
7. **Harness tok/s arithmetic — ruled out.** Median and pooled agree within 0.03 (§4).

## 7. Memory figures

VIS-3B: mtp_old PSS after load 3480 MiB, peak 3519 MiB, VmHWM 3522 MiB. gen_old: 3604 / 3680 / 3687 MiB. With the
drafter the figure is lower, not +609 MiB (Sep 28 VmHWM 3975 → 4584) or +939 MiB (Oct 14 PSS). The methods differ:
VIS primes the files with read() (page cache warm, `vis3_run.py:200-207`) and samples smaps_rollup at ~2 Hz; Sep 28
dropped the page cache cold and read VmHWM at block end; Oct 14 used ctx 4096. The mtp_old run also logged 5 LMK kills
(gen_old 0), whose timing and victims were not recorded. Clean file-backed mmap pages are reclaimable, so resident
figures depend on memory state. Because §2 proves the drafter loaded, the PSS difference cannot mean a missing drafter.
Its cause is UNKNOWN (measurement-method/memory-state difference POSSIBLE). #137 already lists MTP memory as NOT
ESTABLISHED.

## 8. Is the VIS-3B metric trustworthy?

It is arithmetically correct and internally consistent: server timings, VALID label, verified configuration, and median
≈ pooled. It is not a valid estimate of unmonitored decode speed, and it must not be compared with #120 (20.54) or
#137 (19.18) or used to judge MTP's benefit, because the measurement harness itself LIKELY depressed decode. VIS-3B
old-vs-new comparisons share the same monitor, but the interference may not affect the two builds equally. Treat them
as harness-conditioned, not as build verdicts.

## 9. Minimum next action (proposal only, not executed; needs owner approval)

Experiment A — telemetry-load isolation (historical-workload reproduction, old build only): the same VIS-3B
`--test gen` and `--test mtp` (old) under identical start conditions, run twice each, in alternating order: (a) monitor
as now; (b) the monitor sampling interval raised to 5 s, or smaps_rollup removed (keep battery/thermal stops). Single
variable: monitor cadence/content. Success criterion: with (b), OFF ≈ 13–14 tok/s and ON ≈ 19–20 tok/s → confirms the
harness effect.
Optionally, add one native-Termux run of the archived chat_mtp runner (Oct 14 workload) as an unchanged-harness
reproduction.

Experiment B — old-vs-new build comparison: only after A, with the lighter monitor, gen/mtp old and new back to back,
alternating order. B must not be mixed with A.

## 10. Unknown

- Mechanism of the monitor interference (CPU placement of `su` children, smaps_rollup lock/page-walk cost).
- X1 cap state during the historical runs (no cpufreq recorded).
- Thermal state in the synthetic Oct 19 smokes.
- LMK victims and timing in mtp_old; the cause of the lower PSS with the drafter.
- Whether other device-state changes since Oct 14 contribute (none observed; VIS-0 on Oct 18 was fast).
