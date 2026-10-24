# VIS-3C experiment 1: Oct 14 MTP benchmark reproduction, one config per launch

Measured 2026-10-23 (owner-run): see [Results](#results-2026-10-23). Reproduces the workload of the VALID run
`../chat_mtp/runs/chat_mtp_20261014T224541Z` (#137: OFF 13.37, ON 19.18 tok/s, acceptance 39.0 %) on the same
b1609-dotprod build. The owner launches it from native Termux with no agent resident. No motors, no camera.

## What is frozen, and from where

| Item | Value | Source |
|---|---|---|
| Server command | `taskset -c 4-7 …/llama.cpp-b1609-dotprod/build/bin/llama-server -m …/gemma-4-E2B-it-Q4_0.gguf --port 8080 --ctx-size 4096 --host 127.0.0.1 --threads 4 --threads-batch 4 --parallel 1 --swa-full --cache-ram 0`; ON adds `--model-draft …/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3` | archived `run.json` `commands` (= `chat.server_cmd` + `server_manager.MTP_ARGS` at 2091f42) |
| Payload | `{"prompt": "<start_of_turn>user\n<text><end_of_turn>\n<start_of_turn>model\n", "n_predict": 512, "temperature": 0.7, "stop": ["<end_of_turn>"], "stream": false}` | archived `payload_example` (= `chat.chat_payload(chat.build_prompt('gemma', …))` at 2091f42) |
| Prompts | the 13 ids A5 A6 B1 B2 B6 B7 C1 C6–C11 in that order, one user message each, one server for all 13 | `chat_mtp.prompt_texts()` from `bench.py` (SHA-256 0792beee… = Oct 14) |
| Model / drafter / libggml-cpu.so | 8e30dff3… / 9eba8199… / a4ef0d97… | archived `run.json` |
| Block | `chat_mtp.run_block` unchanged: 180 s block, 175 s watchdog KILL, no prompt starts after 150 s, TERM then KILL after 3 s, cores 4-7 checked after every prompt | `chat_mtp.py` |
| Monitoring | `coresidency.sampler_loop`: one `su` sample every 5 s (battery W, PSS via dumpsys meminfo, MemAvailable); launcher's root CPU-zone logger every 5 s | `coresidency.py` (a6b0d654…, = Oct 14), `oneshot.sh` |
| tok/s | token-weighted over replies ≥ 10 tokens; acceptance = Σ accepted / Σ drafted | `chat_mtp.tok_s`, `chat_mtp.acceptance` |

`chat.py` is not used for the command or payload. It changed after Oct 14 (TPL-FIX `70f7508`: `<|turn>` template;
E4B-UP: menu entry), so a rerun of the archived runner would send a different prompt. `chat.py` and
`server_manager.py` are still imported indirectly through `chat_mtp.py`; their hashes are recorded as
`imported_unused_sha256`.

Before starting, `repro.py` refuses unless: the archived `run.json` has its ARTIFACTS SHA-256; the frozen commands,
payload, prompt texts and their SHA-256, ids, model, drafter and libggml-cpu hashes, `bench.py` hash and block
constants all equal the archived `run.json`; and `chat_mtp.py`, `coresidency.py`, `power_map.py`, `bench.py` have
their pinned SHA-256. Model, drafter and libggml-cpu.so are then hashed on disk. Every file in the build's `bin/` is
hashed into `run.json`.

Checked on the phone 2026-10-23 (read-only): model 8e30dff3…, drafter 9eba8199…, libggml-cpu.so a4ef0d97… equal
Oct 14. `llama-server` 77eb35a0… and the other libraries were not hashed on Oct 14; all their mtimes are 2026-09-26,
before Oct 14.

## Where it differs from Oct 14

1. **One config per launch.** Oct 14 ran OFF, then ON in the same process, each in one block. Each launch here runs
   one config in one block and exits. The other config is never started automatically.
2. **No 5 min idle and no skin cooling gate.** Oct 14's launcher idled 5 min, the runner took an idle skin baseline and
   gated each block at idle + 1.5 °C (ON waited 334 s; start skin OFF 24.15, ON 25.63 °C). Here there is no baseline,
   so `GATE_MAX_S = 0`: the gate takes one reading (recorded) and never waits. Comparability is judged afterwards by
   `--report` (start skin within 1.0 °C, both Android status 0, battery within 10 %, equal CPU caps at start).
3. **Thermal and battery preflight added.** Refused when Android thermal status is not 0, VIRTUAL-SKIN is missing,
   battery ≥ 45 °C or the CPU-zone log is stale. Oct 14 had no such refusal. The 175 s watchdog bounds the server.
4. **One block only.** A run whose 13 prompts do not all finish in the 180 s block is INCOMPLETE/NOT VALID. Oct 14
   would have continued in later blocks; it did not need to (60 s OFF, 49 s ON). No resume, no automatic repeat.
5. **Extra readings outside the measurement only:** cpufreq caps and `time_in_state` (plain sysfs reads, no `su`),
   battery %, status and °C, before the gate and after the server stop; agent processes and port 8080 after the stop.
6. **Launcher** `oneshot.sh` is `../chat_mtp/oneshot.sh` with: these paths, `--config OFF|ON` only, no 5 min idle, the
   runner called directly (no wrapper), and VIS runners (`vis[0-9]_run.py`, which run the dense monitor) added to the
   running-benchmark refusal. Logger, screen timeout, wake lock, battery and agent checks are unchanged.
7. `chat_mtp.py` is today's file (b2c370e2…), not 2091f42 (88759798…). The differences are the E4B option and the
   report-header comma. `run_block`'s E2B path is the same apart from passing `model='e2b'` to the (replaced) command.

Cannot be reproduced exactly: temperature 0.7 sampling without a seed (replies and lengths differ by design; Oct 14
also differed between OFF and ON); the device state (Android version, background load, X1 caps, which were not
recorded on Oct 14); the 5 min idle and the ON block's 334 s cooling wait before it.

## Owner commands (native Termux, no agent, battery ≥ 80 % and unplugged, screen on, Termux in front)

Before each launch: quit every agent and exit Debian; stop robot-chat/chat.py, any llama-server and any VIS run.

```bash
bash ~/robot/benchmark/llm_objective_setting/conversation/chat_mtp_repro/oneshot.sh --config OFF
bash ~/robot/benchmark/llm_objective_setting/conversation/chat_mtp_repro/oneshot.sh --config ON
python ~/robot/benchmark/llm_objective_setting/conversation/chat_mtp_repro/repro.py --report \
  ~/robot_runs/chat_mtp_repro/<UTC>_OFF ~/robot_runs/chat_mtp_repro/<UTC>_ON
```

Each launch takes about 2 min: 6 s logger start, about 30 s hashing, then load and 13 prompts (about 1 min). For a
matched pair, start the second launch at a skin within 1 °C of the first (the report shows both).

Folders: `~/robot_runs/chat_mtp_repro/<UTC>_<OFF|ON>/` with `run.json`, `prompts.jsonl`, `server_block00_<cfg>.log`,
`result.json` (block record with the 5 s samples, start and end conditions), `report.txt`. Launcher console and
`thermal.log`: `~/robot_runs/chat_mtp_repro/`. The archived Oct 14 folder is only read.

`--report` prints the Oct 14 rows computed from the archive by the same code (13.37 / 19.18 / 39.0 %), then each
folder's row with its label, the ON/OFF ratio, `configuration` (SAME: only the command's drafter flags differ) and
`conditions` (MATCHED or UNMATCHED with reasons). It starts nothing and needs no root.

## Files

| File | What |
|---|---|
| `repro.py` | runner and `--report` |
| `oneshot.sh` | native launcher |
| `test_repro.py` | offline tests (fakes and archived data only): `python3 -B -m unittest -v test_repro` with Termux python |

## Results (2026-10-23)

Owner-run from native Termux, no agent resident; one launch per config; source = the reviewed frozen candidate
(`repro.py` `29276a35…`, recorded in each `run.json`). Run index: [RUN_INDEX.md](RUN_INDEX.md); hashes:
[ARTIFACTS.md](ARTIFACTS.md); DECISIONS #142.

| Run | Config | Prompts | gen tok/s | accept | gen tok | prompt s | load s | start skin °C | bat % | Label |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `runs/20261023T124846Z_OFF` | OFF | 13/13 | 13.35 | n/a | 686 | 0.36 | 5.4 | 25.11 | 95 | VALID |
| `runs/20261023T132032Z_ON` | ON | 13/13 | 18.55 | 38.2 % | 674 | 0.34 | 7.3 | 25.51 | 92 | VALID |
| Oct 14 archive (#137) | OFF / ON | 13/13 | 13.37 / 19.18 | 39.0 % | 675 / 713 | 0.35 / 0.34 | 4.3 / 6.4 | 24.15 / 25.63 | n/a | VALID |

ON/OFF 1.390 (Oct 14: 1.435). Paired report: `configuration: SAME`, `conditions: MATCHED`. That paired report was
shown on the owner's console and not saved; [reports/REGENERATED_20261023T190340Z_combined_OFF_ON.txt](reports/REGENERATED_20261023T190340Z_combined_OFF_ON.txt)
is a report-only regeneration (`repro.py --report`, 2026-10-23T19:03:40Z, archival task, no inference) from the
unchanged run folders, and gives the same labels and ratio.

One earlier launch was refused before any server start (`not started: battery 78 % < 80 %`, console
`runs/oneshot_console_20261023T115731Z.log`): REFUSED, not a measurement.

Limits: one launch per config, one 180 s block each; temperature 0.7 without a seed (reply lengths differ by design);
`ms/pass` is a PROXY (decode ms / (generated − accepted)), not a count of verification steps. The Oct 14 rates are
reachable with this workload and 5 s monitoring; this does not show why the earlier VIS-3B runs were slower, and
root-monitoring overhead is not proven as their cause. J/token and memory were not evaluated here.

Review: AGY `gemini-3.8-flash-high`, Ponytail off, round 3 APPROVED WITH NOTES (records in `reports/`; validity of
each round in [RUN_INDEX.md](RUN_INDEX.md)).
