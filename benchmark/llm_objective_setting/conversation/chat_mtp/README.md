# robot-chat speed with the MTP drafter (task MTP-A)

Status: **NOT RUN.** No timed result exists yet; the owner runs it from native Termux ([RUN.md](RUN.md)).

## What

DECISIONS #120 measured the Gemma 4 E2B Q4_0 MTP drafter at temperature 0, ctx 2048: gen 14.03 → 20.54 tok/s,
acceptance 44.5 %, peak 3975 → 4584 MiB. robot-chat runs it differently: ctx 4096, temperature 0.7, n_predict
512, not streamed. `chat_mtp.py` measures MTP off vs on with robot-chat's exact settings:

| Config | Server command | Request |
|---|---|---|
| OFF | `taskset -c 4-7` + `chat.server_cmd(<E2B Q4_0>)` | `chat.chat_payload(chat.build_prompt('gemma', [user message]), 'gemma')` |
| ON | same + `--model-draft …/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3` | same |

The command and payload come from `chat.py` itself, so the runner launches what robot-chat launches.
Prompts: the 13 #120 ids (A5 A6 B1 B2 B6 B7 C1 C6–C11) from `../../bench.py`, the user text only, one
message, history cleared for each prompt. A5's three user turns are joined into one message (#120 sent them as
three turns inside its own wrapper).

Blocks last at most 180 s from the server spawn to its exit: skin gate (`power_map.thermal_gate`: VIRTUAL-SKIN ≤
idle + 1.5 °C, at most 8 min, else WARM START), server start (load s = spawn to `/health` ok), then the config's
next prompts. No prompt starts after 150 s. A watchdog timer KILLs the server 175 s after the spawn, which ends any
health wait or request in flight; a reply not in by 175 s is dropped and redone in a later block. (TERM alone does
not: llama-server then keeps a running request's connection open until the request ends.) The normal stop after
the last prompt sends TERM, then KILL after 3 s. `server_s` records spawn to server exit. The end skin/status reading is taken
after the stop. Block order OFF, ON, ON, OFF, OFF, ON, … until each config has run every prompt once. If Android
takes cores 4–7 away, the prompt in flight is dropped, the block ends, and the run waits for the cores.

Measured per prompt: `prompt_s` (the server's `prompt_ms`). TTFT is **not measurable** with chat's
non-streamed payload: `ttft_s` is recorded as null, and `prompt_s` excludes the first generated token and
client overhead. Also recorded: prompt and generated tokens, gen tok/s, `draft_n` / `draft_n_accepted` from the response timings
(as `conv_speed_mtp.py`), reply length, wall time. Per block: llama-server VmHWM and peak PSS, minimum
MemAvailable, mean battery power, energy per generated token, skin and Android thermal status at start and end,
the server's `Cpus_allowed_list`. Sampling runs from the spawn (load included) until the server stop begins.
Every 5 s sampling is `coresidency.sampler_loop` / `root_sample` (battery W =
−current_now × voltage_now / 10¹², camera_heat.py's method). Its `pss_error` field names `robotcam_app` (and
`camera_provider` if absent) because no camera runs here; only the `llama_server` PSS is used.

Report columns: gen tok/s token-weighted over replies ≥ 10 tokens (`conv_speed.summary`), acceptance = Σ
accepted / Σ drafted, W = mean of block means, J/tok = mean battery W during requests × request time /
generated tokens (whole phone, includes prompt processing), mean of block values.

Refuses to start: not native Termux, a claude/codex/agy process, battery < 80 %, battery not Discharging,
port 8080 in use, model or drafter SHA-256 not the `../conv_mtp/ARTIFACTS.md` values (hashed once at start).

## Files

| File | What |
|---|---|
| `chat_mtp.py` | the runner (native Termux python) |
| `run_chat_mtp.sh` | native-only wrapper, started by `oneshot.sh` |
| `oneshot.sh` | launcher: `benchmark/power_map/oneshot.sh` (the coresidency #123 form, no cache-drop watcher) with this runner; files in `~/robot_runs/chat_mtp/` |
| `test_chat_mtp_runner.py` | offline tests (fakes only) |
| `RUN.md` | owner commands |

## Outputs

`~/robot_runs/chat_mtp_<UTC>[_smoke]/`: `run.json` (commands, payload, prompt texts and SHA-256, `bench.py`
SHA-256, model/drafter SHA-256, code SHA-256 of the runner, chat.py, server_manager.py, coresidency.py,
power_map.py, `libggml-cpu.so`, idle reading), `prompts.jsonl` (one line per finished prompt, full reply and
timings), `blocks.jsonl` (one line per block, with the 5 s samples), `report.txt`,
`server_blockNN_<config>.log`. A resume adds `resume_<ns>.json` and keeps the old logs as
`*.before_resume_<ns>`. Launcher console and thermal log: `~/robot_runs/chat_mtp/`.

## Known limits

- Temperature 0.7 samples: replies differ between configs and runs; byte-identity is not checked.
- Untimed proot smoke (2026-10-14, agent resident, not a result): ON loaded in 8.25 s, B1 reply 104 tokens,
  `draft_n` 141, `draft_n_accepted` 56 in the non-streamed response; the server log line
  `[spec] failed to measure draft model memory` also appears in #120's archived log.
