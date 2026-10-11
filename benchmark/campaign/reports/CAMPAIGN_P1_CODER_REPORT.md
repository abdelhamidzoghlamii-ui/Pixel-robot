# CAMPAIGN_P1 Coder report

Assigned execution profile: Coder | Codex CLI | gpt-6.1-sol | medium | Ponytail lite.
Executor fallback (not used): AGY CLI | gemini-3.1-pro-high | effort N/A | Ponytail lite (explicit). Assigned reviewer: Claude Code CLI headless | claude-opus-5-5 |
effort N/A | Ponytail off via PONYTAIL_DEFAULT_MODE=off. Reviewer fallback: Codex CLI `codex exec -s read-only`, gpt-6.1-sol, medium, Ponytail off; owner approval required. No fallback authorized or used.

Base/HEAD: `90d50559d019cb002066d21c4949b8502f6f4d85`, confirmed at entry; initial
`git status --short` empty. Same repository is bind-mounted as `/termux-home/robot`
and `/data/data/com.termux/files/home/robot`. No commit, staging or push performed.
Required handoff/workflow/roles/prototype/status/decisions and benchmark reuse paths inspected.

## Archive

All ten Downloads owner SPEED1 files copied unchanged to speed1/runs; source and destination
SHA-256 verified, total 4,346,837 bytes. No owner evidence reinterpreted as live Coder timing.
New runs/RUN_INDEX.md holds fixed labels because existing relate_anything/RUN_INDEX.md is
outside the permitted modification scope. M2 fp32 benchmark candidate, not deployed.

# SPEED1 fixed owner archive

Bytes copied unchanged; SHA-256 checked against Downloads originals. M2 is a benchmark candidate, not deployed.

| File | Bytes | SHA-256 | Fixed label |
|---|---:|---|---|
| [owner_preflight_speed1_fix.json](owner_preflight_speed1_fix.json) | 4385 | `9918f724cd6906e9a047990b8afd60eddeb239d3ad3deed0a6db8b38c10cbf89` | PREFLIGHT ONLY — NO TIMING. Passed: first live root-mask readback; root/su masks excluded each measured cluster; engines pinned MID 4-5, BIG 6-7, LITTLE 0-3. |
| [owner_preflight_speed1_fix.stderr](owner_preflight_speed1_fix.stderr) | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` | PREFLIGHT ONLY — NO TIMING. Passed: first live root-mask readback; root/su masks excluded each measured cluster; engines pinned MID 4-5, BIG 6-7, LITTLE 0-3. |
| [owner_preflight_speed1_fix.stdout](owner_preflight_speed1_fix.stdout) | 1267 | `aac635ef9ea2b2bf6adc7fe705c600076a1793b2264cc2850a1d8a0d1c57037d` | PREFLIGHT ONLY — NO TIMING. Passed: first live root-mask readback; root/su masks excluded each measured cluster; engines pinned MID 4-5, BIG 6-7, LITTLE 0-3. |
| [owner_session_speed1_fix.json](owner_session_speed1_fix.json) | 2358903 | `104ff9bd771ad43f36226598ace98baed3a17165507ab5c836442eac4f8f6e2c` | SESSION COMPLETE; 3 VALID blocks, 1 warm start. |
| [owner_session_speed1_fix.stderr](owner_session_speed1_fix.stderr) | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` | SESSION COMPLETE; 3 VALID blocks, 1 warm start. |
| [owner_session_speed1_fix.stdout](owner_session_speed1_fix.stdout) | 13488 | `b7f583d4e584f01080e3af2a78a02a26fb0180e4da921b6d6cc057bb43c88764` | SESSION COMPLETE; 3 VALID blocks, 1 warm start. |
| [owner_session_speed1_fix_block_01_fp32_MID.json](owner_session_speed1_fix_block_01_fp32_MID.json) | 479134 | `ed887d23389f0213bae472ecc4ee5900de7ca03a054633f32037defb25a403aa` | VALID. 36/36 calls, 1116/1126 ms median/P95, no caps. Watts indicative only (1 s sampler phase-locked to the 5 s cadence; current_now lag suspected). |
| [owner_session_speed1_fix_block_02_fp32_BIG.json](owner_session_speed1_fix_block_02_fp32_BIG.json) | 477789 | `6e18797ec6e0acdaeafe3abf7a7e92ee3498979ddc0176fd2bf8f59d2c154a4f` | VALID timing, 36/36 calls, 646/671 ms, no caps. Watts NOT VALID (0 power samples inside calls; phase-locked sampler). |
| [owner_session_speed1_fix_block_03_fp32_LITTLE.json](owner_session_speed1_fix_block_03_fp32_LITTLE.json) | 533291 | `c0234bb04896f450b01b10c28bbb0323a760d61386eb46cf52ff5e562b6ce021` | VALID. 36/36 calls, 3130/3208 ms, no caps. Watts indicative only (same sampler limits). |
| [owner_session_speed1_fix_block_04_fp32_MID.json](owner_session_speed1_fix_block_04_fp32_MID.json) | 478102 | `0f6588b8e1d04dac2569a4dab20430caa55d3f0fa6895ac46fcc375a28aa59d9` | NOT VALID — WARM START (skin gate timed out after 482 s). 1118/1131 ms is consistent with block 1 but not valid. |

## Design, reuse, context, time estimate and exact owner commands

# CAMPAIGN_P1 owner steps

M2 fp32 is a benchmark candidate, not deployed. This screens speed, memory,
heat and power; **selector accuracy with relation context is NOT evaluated**.
Motors OFF, phone on a table, lights ON, charger unplugged, screen ON, native
Termux foreground. No motors, USB or serial API is imported or commanded.
Exit Codex, Claude, AGY, node and every robot/benchmark/model-server process
before steps 1, 2 and 4. The runner starts and owns one Gemma server itself.
Use fresh output/log stems on reruns; existing evidence is refused. Downloads
`campaign/` was created during preparation. Commands below are native Termux;
step 3 can also run inside Debian/proot using `/usr/bin/python3` and
`/termux-home/robot` / `/termux-home/storage/downloads` paths.

1. Lag probe (about 1 minute plus setup/cleanup):

```sh
python -u ~/robot/benchmark/campaign/lag_probe.py --output ~/storage/downloads/campaign/owner_lag_p1.json > ~/storage/downloads/campaign/owner_lag_p1.stdout 2> ~/storage/downloads/campaign/owner_lag_p1.stderr
```

Send `owner_lag_p1.json`, `owner_lag_p1.stdout`, `owner_lag_p1.stderr`.
Label: `POWER LAG PROBE — method check, not a benchmark`. After model load,
20 s quiet with camera/server off, then saved desk2 M2 input continuously on
BIG 6–7 / two threads for at least 10 s, then 20 s quiet. Last call may extend
the load by one inference; the first actual inference-call start and last end drive the step math.
Root sampler/shell/pump and safety monitors use available CPUs 0–3 only;
fastest sequential reads, no sampling sleep. Root/su/pump mask verification
occurs once per second in this probe (every sample in 0.37 s cadence mode),
plus final verification; achieved rate includes this verification overhead. Report achieved Hz, current_now
and optional current_avg, rise 50%/90% and recovery 50%/10% of observed step.
Observed baseline/plateau use the final 5 s of quiet/load. No positive step,
absent current_avg or unobserved crossing => null, never fabricated delay.

2. Phase-1 preflight (estimate 1–3 minutes, no idle/gates/timed blocks):

```sh
python -u ~/robot/benchmark/campaign/phase1.py --preflight --output ~/storage/downloads/campaign/owner_preflight_p1.json > ~/storage/downloads/campaign/owner_preflight_p1.stdout 2> ~/storage/downloads/campaign/owner_preflight_p1.stderr
```

Send `owner_preflight_p1.json`, `owner_preflight_p1.stdout`,
`owner_preflight_p1.stderr`, `owner_preflight_p1_llama-server.log`.
Success: `PREFLIGHT ONLY — NO TIMING`. Checks every layout's root/su/pump
safe mask, required policies, YOLO workers/caller, M2 workers/caller, charger,
agents/other processes, cores 4–7, skin/status and battery/CPU/policy reads,
fixed external graph/bank/sidecar/YOLO/Gemma/server hashes, installed RobotCam
version, camera startup on mode B at rate 1, server health and s1o warmup.
Screen timeout/wake lock and all owned resources are restored/closed.
Do not proceed to session unless preflight and cleanup succeed. Subsets are
accepted, but the default preflight checks L0 through L4.

3. Portable scheduling dry run (about 36 seconds; hardware fully mocked):

```sh
python -u ~/robot/benchmark/campaign/phase1.py --dry-run --output ~/storage/downloads/campaign/owner_dry_p1.json > ~/storage/downloads/campaign/owner_dry_p1.stdout 2> ~/storage/downloads/campaign/owner_dry_p1.stderr
```

Send `owner_dry_p1.json`, `owner_dry_p1.stdout`, `owner_dry_p1.stderr` and
`owner_dry_p1_block_01_L0.json`, `owner_dry_p1_block_02_L1.json`,
`owner_dry_p1_block_03_L2.json`, `owner_dry_p1_block_04_L3.json`,
`owner_dry_p1_block_05_L4.json`, `owner_dry_p1_block_06_L0.json`.
Label: `NOT VALID — DRY RUN (all hardware mocked)`. Six 6 s blocks, same
scheduling code, no models/camera/server/root/idle/gates/sensors. Null watts.
This checks wiring, not installed native execution. It deliberately shows
L3 skips using a fake 1.2 s relation call. Preflight/dry-run are exclusive.

4. Phase-1 session (23–71 minutes plus loads/setup/cleanup):

```sh
python -u ~/robot/benchmark/campaign/phase1.py --output ~/storage/downloads/campaign/owner_session_p1.json > ~/storage/downloads/campaign/owner_session_p1.stdout 2> ~/storage/downloads/campaign/owner_session_p1.stderr
```

Send `owner_session_p1.json`, `owner_session_p1.stdout`,
`owner_session_p1.stderr`, `owner_session_p1_llama-server.log`,
`owner_session_p1_block_01_L0.json`, `owner_session_p1_block_02_L1.json`,
`owner_session_p1_block_03_L2.json`, `owner_session_p1_block_04_L3.json`,
`owner_session_p1_block_05_L4.json`, `owner_session_p1_block_06_L0.json`.
All refusals/checks precede one 300 s idle with Gemma loaded/camera off.
All blocks are 180 s, order L0,L1,L2,L3,L4,L0 for drift. Each has the same
power-map skin condition (idle +1.5 °C) and 480 s maximum gate; timed-out
gates label the block `NOT VALID — WARM START`. Session completion does not
establish block validity. Full maximum-gate estimate exceeds 60 minutes;
the runner prints this warning before setup. To split, replace step 4 with
the following two separate owner invocations (14–38 minutes each plus setup):

```sh
python -u ~/robot/benchmark/campaign/phase1.py --blocks L0,L1,L2 --output ~/storage/downloads/campaign/owner_session_p1a.json > ~/storage/downloads/campaign/owner_session_p1a.stdout 2> ~/storage/downloads/campaign/owner_session_p1a.stderr
```

```sh
python -u ~/robot/benchmark/campaign/phase1.py --blocks L3,L4,L0 --output ~/storage/downloads/campaign/owner_session_p1b.json > ~/storage/downloads/campaign/owner_session_p1b.stdout 2> ~/storage/downloads/campaign/owner_session_p1b.stderr
```

Send `owner_session_p1a.{json,stdout,stderr}`, `owner_session_p1a_llama-server.log`,
`owner_session_p1a_block_01_L0.json`, `owner_session_p1a_block_02_L1.json`,
`owner_session_p1a_block_03_L2.json`, `owner_session_p1b.{json,stdout,stderr}`,
`owner_session_p1b_llama-server.log`, `owner_session_p1b_block_01_L3.json`,
`owner_session_p1b_block_02_L4.json`, `owner_session_p1b_block_03_L0.json`.
Each split starts its own idle baseline; cross-session comparisons need care.

## Reuse and layout details

Unmodified imported paths: `power_map.build_detector('mid')`,
`power_map.camera_start(1)`, `coresidency.read_frame`, `coresidency.Server`,
`coresidency.make_selector/select`, `variants.build('fp32', cluster)`,
`speed_block.prepare_monitor/root_mask/check_pinning/Screen`,
`coresidency.fast_sample/block_limit`, `power_map.capped_by_policy`.
No runner was fork-copied. The small campaign skin gate adapts power_map's
condition/bound because power_map also requires an externally maintained CPU
thermal log; this campaign reads CPU fault sensors directly instead.

Exact reused detector settings:

```python
opts.intra_op_num_threads = 2 if setting == 'mid' else 4
opts.inter_op_num_threads = 1
# MID workers/caller pinned to {4, 5}
```

Exact reused RobotCam command inside camera_start:

```python
cr.am('start', '-n', 'com.pixelrobot.robotcam/.StartActivity', '--es', 'mode', 'B', '--ei', 'rate', str(rate))
```

Frame path: `/storage/emulated/0/Download/robotcam/frame.jpg`. Same decoded
reader with JPEG session/frame/boottime validation and advancing frame numbers. Repeated frames retry for up to 1.35 s (one
publication period plus tolerance); a remaining repeat fails closed.
Installed versionCode/versionName are recorded by preflight; manual-exposure
options are not passed. Root shell/su/pump and sampler exclude the union of
YOLO MID and each layout's M2 cluster: L0/L3 safe 0–3,6–7; L1 safe 6–7;
L2/L4 safe 0–3. Nonempty masks and live readback required. Legacy transient root service shells
are adapted by a campaign-only wrapper: `taskset -p MASK $$` plus readback
before the original service command. Preflight exercises this for every layout.
The original `cr.root` function is restored after block/probe cleanup.

Exact reused server launch (`coresidency.server_cmd`, PORT=8080):

```python
return [server_manager.LLAMA_SERVER, '-m', MODEL, '--port', str(PORT), '--ctx-size', '2048',
        '--threads', server_manager.THREADS, '--threads-batch', server_manager.BATCH_THREADS,
        '--parallel', '1', '--swa-full', '--cache-ram', '0', '--host', '127.0.0.1', *extra]
```

MODEL is `~/models/gemma-4-E2B-it-Q4_0.gguf`, threads/batch threads both 4;
server binary is the existing `server_manager.LLAMA_SERVER`. No server CPU
pinning or MTP. `expected_hashes.json` fixes seven external artifacts.
Selector uses unchanged S1O prompt/scoring via make_selector/select every
20 s; only transport timeout is bounded to 15 s per HTTP request. Successful calls
finishing past the next selector slot count as misses as well as cadence failures. Prompt
is the existing situation/options/instruction, latest M2 text appended to
situation. No accuracy grading, no robot action is executed.

**Explicit detection instruction differs from historical SizePolicy**:
#128 selects one size per frame (640 replaces 320 at large slots). This task
explicitly requires 320 on every frame PLUS 640 every 5 s, so the campaign
runs both on the same frame at large slots (0,5,...,175). Detector sessions,
threads, pins, camera, selector and server are reused unchanged. This adds
36 small detections per full block compared with the historical mix.

L0 has no M2/context. L1: LITTLE 0–3 / four threads after each 640.
L2: BIG 6–7 / two threads immediately before each selector, using latest
completed 640 and its boxes; selection waits for M2. At later selector slots
this can be the 640 frame from five seconds earlier, while the current
YOLO-640 runs concurrently on MID. Context freshness/overlap are recorded,
and context accuracy is not evaluated. L3: MID 4–5 / two
threads after each 640; due YOLO slots while M2 runs are skipped, counted,
never queued. L4: BIG 6–7 / two threads after each 640. One M2 request at a
time; cadence overload is recorded. M2 requires 2–32 boxes: fewer than two
produces empty context and an insufficient-box slot count, not fabricated
inference; more than 32 fails closed rather than silently truncating boxes.

Context uses top five score-sorted rows with pass=true and score>=threshold:

```text
M2 relations:
- person [0] next to chair [1].
```

Both wall-clock M2 call stats and ORT `inference_ms` stats are recorded for
comparison with SPEED1. Skin/gate/end timestamps are relative to the common
block origin; caps and memory summary exclude post-180-s tails. Logcat read
failure invalidates the block. Existing server logs are refused rather than appended.

Underscores become spaces; indices distinguish same-class detections. Empty:
`M2 relations:\n- none above threshold.` L0 appends nothing. An initial
L1/L3/L4 selector may receive empty context before the first relation finishes.
All actual appended strings are retained per selector call.

Power uses 0.37 s nominal slots (overrun skips slots), root read-start and
receipt times, signed current_now and voltage_now, time-weighted mean over
receipt timestamps, full boundary brackets, inside/outside M2 call sample
counts and read-boundary crossings. Coverage gaps >1.5 s => null watts and
invalid block. De-phasing fixes cadence aliasing; gauge lag remains a method
limitation until the owner probe is inspected. CPU/battery/caps sampled every
1 s (start offset 0.11 s); the nominal five-second skin/status series uses 4.87 s (offset 0.37 s),
and memory/PSS uses 5.13 s (offset 1.31 s). These periods deliberately drift
relative to 1 s and 5 s inference slots; all actual receipt timestamps and
configured periods are recorded. Per-frame guard uses only in-process state/core checks;
process/HTTP health checks use a pinned monitor every 5.23 s (offset 0.71 s),
and before/after the block. The lag probe also uses pinned 5.23 s process checks,
not 10 Hz pgrep in its quiet baseline. All forked monitor clients inherit
the verified safe thread mask; Magisk-spawned command shells explicitly set
and read back their mask. Binder service work in existing Android processes
and Magisk daemon/control startup work remain outside campaign affinity
control. These instrumentation costs remain method limits shared by all layouts;
the non-divisor periods prevent a fixed phase at each 640 slot, but
do not eliminate measurement overhead. The LMK window is exactly the 180 s
block, excluding model builds/gates and cleanup; unparseable epoch timestamps
invalidate the logcat accounting. Per-policy cap accounting reuses the
existing policy0/4/6 method. Logcat LMK rows retained per block.

Continuation matches SPEED1: only a local cadence failure may continue with
successful cleanup and no monitor errors. Root/sensors, thermal stops,
charger/processes, cores/affinity, inference/camera/server and cleanup failures
stop later blocks. Warm starts remain invalid and the next block is gated.

## Verification limits

`self_check.py` exercises new scheduling/power/lag/refusal/lifecycle paths with
mocks. Existing direct-script test results are in `checks/results.json`.
Coder ran no real timed block, session or root preflight. Proot cannot reach
phone root; agents resident would invalidate timings. Live battery gauge,
root/su affinity/readback, Android services/policies/caps/thermal/logcat/PSS,
installed RobotCam startup/settings/frames, real M2/YOLO overlap and skips,
Gemma HTTP/health/timeouts/context, cooling gates, idle and native cleanup
remain owner validation paths. Portable dry-run covers scheduling with mocks.

Signals during a block are recorded as shared failures; cleanup runs and later
blocks stop. The outer session exits 1 rather than preserving 128+signal.

## Checks and limitations

{
  "self_check": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct script, executed from proot; no live root"
  },
  "speed1_self_check": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/relate_anything/speed1/self_check.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct script, executed from proot; no live root"
  },
  "desk2_self_check": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/relate_anything/desk2/self_check.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct script, executed from proot; no live root"
  },
  "power_map_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/power_map/test_power_map.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct script, executed from proot; no live root"
  },
  "coresidency_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/coresidency/test_coresidency.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct script, executed from proot; no live root"
  },
  "detector_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "test_detector_robotcam.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct script, executed from proot; no live root"
  },
  "mission_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "test_run_mission.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct script, executed from proot; no live root"
  },
  "cycle_units": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "run_cycle_safety_test.py"
    ],
    "exit_code": 0,
    "scope": "offline mocks/direct script, executed from proot; no live root"
  },
  "self_check_round2": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check.py"
    ],
    "exit_code": 0,
    "scope": "expanded mocks, no live root/timing"
  },
  "portable_dry_round2": {
    "command": [
      "/usr/bin/python3",
      "benchmark/campaign/phase1.py",
      "--dry-run",
      "--output",
      "/termux-home/storage/downloads/campaign/coder_dry_round2.json"
    ],
    "exit_code": 0,
    "scope": "all hardware mocked, six 6s blocks in proot"
  },
  "self_check_round3": {
    "command": [
      "/data/data/com.termux/files/usr/bin/python",
      "benchmark/campaign/self_check.py"
    ],
    "exit_code": 0,
    "scope": "expanded mocks incl round3 regressions; no live root/timing"
  },
  "portable_dry_round3": {
    "command": [
      "/usr/bin/python3",
      "benchmark/campaign/phase1.py",
      "--dry-run",
      "--output",
      "/termux-home/storage/downloads/campaign/coder_dry_round3.json"
    ],
    "exit_code": 0,
    "scope": "six complete hardware-mocked 6s blocks in proot"
  }
}

Latest self-check output:
```text
PASS sampler masks, de-phasing, timestamps, weighting/gaps/boundaries, battery refusals, lag crossings
PASS full-layout scheduling, L3 skips/no queue, L2 M2-before-selector, context, continuation/camera/inference refusals
PASS process/agent/foreign-server refusals, root/pump masks, gates/warm start, thermal/version refusals
PASS real preflight path all layouts, charger/agents/root/cores/hashes/policies/camera/server refusals; no idle/gates/blocks
PASS lag probe full quiet/load/quiet lifecycle, BIG/0-3 masks, current_avg absent, sampler/inference/cleanup failure (mocks)
PASS live block lifecycle/wiring, camera/inference/cleanup/power coverage failures, boundary accounting and file capture (all mocks)
PASS LiveOps frame retry/repeated/old, YOLO/M2 pins/box limits, actual context append/L0, bounded selector transport/timeout
PASS L1/L4 busy-M2 skip branch, selector timeout/OSError refusal, successful late-selector miss counts
PASS restrict repinning/empty masks, transient-root pin/readback/restore, periodic sampler cadence/final bracket, bare-runner refusals
PASS real session control flow: one mocked 300s idle, cadence continue, warm continue, shared stop/unrun capture, server cleanup
PASS pinned periodic worker startup offsets, health/process checks, read wiring, affinity failure and root-scope cleanup (mocks)
PASS de-phased telemetry periods, fastest-probe 1 Hz verification, LMK block window/error handling, busy recheck after YOLO
```

Portable final dry-run output:
```text
Plan: L0,L1,L2,L3,L4,L0 ; motors OFF; phone on table, lights ON.
Estimate 23.0–71.0 min plus setup/loads/cleanup.
WARNING over 60 minutes with maximum gates; split --blocks L0,L1,L2 and --blocks L3,L4,L0.
NOT VALID — DRY RUN (all hardware mocked) Evidence: /termux-home/storage/downloads/campaign/coder_dry_round3.json
```

`git diff --check` passed; five new Python modules parsed successfully. Expanded round-2 and round-3 mock coverage passed after review findings. Existing direct scripts
used native Termux Python called from proot, but hardware/model/root operations in those tests
are mocks. Portable final dry run used Debian `/usr/bin/python3`, full six-layout plan, six-second
mock blocks. No Coder timed block, real session, root preflight, motor command or model inference
was run. The first portable dry attempt failed before work on Downloads flock ENOSYS; fixed by
using a temporary dry-run lock. Final successful evidence is archived in campaign/checks.

Mock-only live paths: root/su/pump affinity and shell/service IPC; battery current/voltage/optional
current_avg, actual gauge lag and achieved de-phased sampling; real Android policy/cap/fault/
battery/skin/status/freshness checks; native process/charger/cpuset refusals; installed RobotCam
version, settings, startup, JPEG session/age/frames/gaps and cleanup; pinned YOLO/M2 worker
identification, real overlap, L3 skips, L2 order, box edge cases; Gemma launch/residency/health,
s1o HTTP/context/token counts/timeouts; real memory/PSS/logcat LMK; 300s idle, cooling gates,
warm starts, interruption/draining/resource cleanup and continuation. Owner must validate these.

## Independent review rounds (verbatim)


### Round 1

Full JSON stdout (verbatim):
```json
{"duration_api_ms":471950,"stop_reason":"end_turn","session_id":"248ecf0a-8344-45fb-8e1f-7d201f4b564e","total_cost_usd":2.8261005999999997,"usage":{"input_tokens":54,"cache_creation_input_tokens":156439,"cache_read_input_tokens":3332463,"output_tokens":45394,"output_tokens_details":{"thinking_tokens":37449},"server_tool_use":{"web_search_requests":0,"web_fetch_requests":0},"service_tier":"standard","cache_creation":{"ephemeral_1h_input_tokens":156439,"ephemeral_5m_input_tokens":0},"inference_geo":"not_available","iterations":[{"input_tokens":2,"output_tokens":3178,"cache_read_input_tokens":164187,"cache_creation_input_tokens":406,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":406},"type":"message"}],"speed":"standard","fallback_credit":null},"modelUsage":{"claude-opus-5-5":{"inputTokens":54,"outputTokens":45394,"cacheReadInputTokens":3332463,"cacheCreationInputTokens":156439,"webSearchRequests":0,"costUSD":2.8261005999999997,"contextWindow":1000000,"maxOutputTokens":128000,"thinkingTokens":37449,"canonicalModel":"claude-opus-5-5","provider":"firstParty","costBasis":"list"}},"permission_denials":[],"terminal_reason":"completed","fast_mode_state":"off","fast_mode_disabled_reason":"sdk_opt_in_required","subagent_stats":{"spawned":0,"requested":{"background":0,"foreground":0,"unset":0},"started_in_background":0,"max_depth":0,"spawned_by_subagents":0,"completed":0,"failed":0,"killed":{"parent":0,"user":0,"system":0},"refused":{"depth_limit":0,"concurrency_limit":0,"budget":0},"by_type":{}},"is_error":false,"num_turns":37,"subtype":"success","api_error_status":null,"result":"# CAMPAIGN_P1 independent review (Ponytail mode OFF)\n\nI only read files. I ran no commands. I could not check SHA-256 values without running commands, so the archive hashes are **not** independently verified. The archive is consistent with its labels: SPEED1 block JSONs show MID 1115.7/1126.3, BIG 645.9/670.7, LITTLE 3130.2/3207.7 with zero caps, and the final MID has a 481.85 s warm start at 1118.5/1130.5. File sizes match `RUN_INDEX.md`.\n\n## What checks out\n- **Hard rules:** new code is only in `benchmark/campaign/`. The archive is only in `speed1/runs/`. `INDEX.md` gets one row and `.gitignore` gets one negation. No models are added. `main.py`, `chat.py` and the detector are untouched. Nothing imports motors.\n- **Reused interfaces match their sources:**\n  - `cr.select` returns keys that don't collide with the `run_cycle` and `LiveOps` keys.\n  - `root_sample` provides `battery_status`, `root_rc` and `pss_error`.\n  - YOLO output (`class_name`, `box_xyxy`) matches the input `desk_check.Head.infer` expects. Its rows carry `pass`, `score` and `threshold`, which `context()` needs.\n  - `capped_by_policy` gets rows rebased to t=0.\n  - `S1O._post(path, body)` matches the bounded replacement.\n- **Layouts:**\n  - Monitor masks: L0/L3 use 0–3,6–7; L1 uses 6–7; L2/L4 use 0–3.\n  - L3 skips are counted, never queued. L2 runs M2 synchronously before each selector call.\n  - Context is the top 5 passing rows in the fixed format, with no context for L0.\n  - Each block's warm-start gate is independent.\n  - Continuation is allowed only after a cadence failure.\n- **Power math:** the trapezoid clipping, the gap > 1.5 s → null rule, the negative first bracket and the end bracket are all correct.\n- **Lag crossings:** the crossing math and its null handling are correct.\n\n## Findings\n\n1. **[Medium] The process refusal is weaker than SPEED1's.** The pattern at `benchmark/campaign/runtime.py:26` drops the explicit names `power_map\\.py`, `coresidency\\.py`, `thermal_char\\.py`, `duty_cycle\\.py`, `camera_power\\.py` and `speed_block\\.py` that `desk2/speed_block.py:30-32` checks. `benchmark/.*\\.py` only matches when the command line includes that path. A runner started from inside its own folder (for example `python power_map.py`) would pass the charger/agents/processes refusal.\n\n2. **[Medium] Required mock coverage is missing.**\n   - `check_live_block` replaces `run_cycle` (`self_check.py:281`), so `LiveOps` never runs, not even against mocks:\n     - the frame retry and the repeated-frame refusal\n     - the YOLO pinning check\n     - the M2 refusals for fewer than 2 or more than 32 boxes\n     - `select()` appending the M2 context to the situation, which the owner listed explicitly\n   - `check_preflight` replaces `live_block` and only runs `--preflight` (`self_check.py:186`). The session path never runs, so these are untested: the idle loop, continuing after a cadence failure, and stopping after a shared failure (`phase1.py:514`).\n   - Also untested:\n     - the selector timeout/`OSError` branch\n     - the L1/L4 \"M2 still busy at a 640 slot\" branch\n     - `rt.prepare(restrict=…)`, which re-pins the pump threads and root PIDs with taskset for the lag probe\n     - `bounded_selector`\n\n   Part D requires mocks for every new path, including context and continuation.\n\n3. **[Medium] Unpinned work runs during the timed window, and some of it is phase-locked.**\n   - `guard()` runs every frame slot (`phase1.py:125` → `298-301`). Each run spawns a `pgrep -fa` and makes an HTTP `/health` request from the unpinned main thread. These can land on measured cores while M2 runs asynchronously (L1/L3/L4). SPEED1 checked processes only before and after a block.\n   - The 5 s `read_dump` (`dumpsys thermalservice`) and the new `root_sample` (`dumpsys meminfo` on four or more processes) both go through `cr.root` (`phase1.py:356-357`). Those su processes are not affinity-controlled. Their monitor loops start at t0, so they line up with the 0, 5, 10 s… 640/M2 slots. That is the same phase-locking the 0.37 s power fix was meant to remove.\n   - `RUN.md:125` says \"Root shell/su/pump … exclude the union\", which is true only for the two persistent shells.\n   - **Fix:** offset or de-phase these monitors and run `guard()` less often, or document both as method limits.\n\n4. **[Low–Medium] The lag probe's quiet baseline and rise times are biased.**\n   - `load_start_s` is recorded at `lag_probe.py:69`, before `guard()` (pgrep plus core checks) and the pinning check run. The rise is measured from that point instead of `calls[0]['started_s']`, so it comes out too long by the guard time.\n   - The quiet phases call `guard()` at 10 Hz. That is an unpinned pgrep fork every 100 ms, which raises the baseline current.\n\n5. **[Low]** Selector `miss` is only set when a call raises. A selector that succeeds but finishes after its next 20 s slot only sets the block's `cadence_missed`, so `summary.selector.misses` undercounts late calls.\n\n6. **[Low]** A frame still repeated after 0.35 s (`phase1.py:150`) ends the whole session as a shared failure. `power_map.frame_loop` treats repeats as retries. This is a robustness risk at a camera rate of 1/s.\n\n7. **[Low] Smaller problems:**\n   - `power_origin_offset_s` (`phase1.py:400-401`) is never set, so that code does nothing.\n   - The `gate`, `skin_start` and `skin_end` `t` values are not rebased to t0.\n   - `lmk['ok']` being false does not affect validity (`phase1.py:390`).\n   - Caps use `duration_s`, which is over 180 s, so the tail after the block is included (`phase1.py:406`).\n   - When the situation is not a string, `json.dumps` without `indent=1` (`phase1.py:270`) differs from how S1O formats it. This is latent: all 60 cases are strings.\n   - The summary's M2 median/P95 use wall-clock `ms`, while the SPEED1 archive used `ort_ms`. Also report `inference_ms` stats so the two can be compared.\n   - An existing `_llama-server.log` is appended to instead of refused.\n\n## Every live path still only mock-tested\nNone of these has run on the phone; each still needs owner validation:\n- Native root, su and pump affinity/readback, including the lag probe's restrict re-pin and the root services.\n- The unpinned `cr.root` su processes (dumps, PSS, battery, LMK).\n- Battery `current_now`, `voltage_now` and `current_avg`, the gauge lag, and the 0.37 s achieved rate, coverage and gaps.\n- The real CPU exclusion of the samplers and monitors.\n- Live Android policies and caps, CPU fault, battery thermal, skin/status freshness and `block_limit`.\n- The charger, agents, process and cpuset/cores refusals under native conditions.\n- The installed RobotCam version, mode B at rate 1, real frame startup, session, age, gaps and repeats, and `camera_end_check` cleanup.\n- Real pinned YOLO and M2 thread identification, the 2–32 box limits on live detections, L3 skips, L2 ordering, and L1/L4 overlap.\n- The resident Gemma launch flags, `/health` under load, S1O HTTP with the 15 s bound, context appending, prompt tokens and timeouts.\n- MemAvailable, server PSS (`root_sample`) and logcat LMK.\n- The real 300 s idle, cooling, the skin gate and warm starts.\n- Native screen and wake-lock restore.\n- An interrupted native session: signal drains, resource cleanup, and continuation versus stop.\n- The lag probe end to end.\n- The SHA-256 values of the archived files and `expected_hashes.json` against the files on the phone.\n\nThe portable dry-run uses mocks for all hardware and checks scheduling only.\n\n## Final verdict\n**CHANGES REQUESTED.** Findings 1–3 must be fixed or documented. Finding 4 should be fixed. Approval would not give commit authority in any case.","ttft_ms":3953,"type":"result","duration_ms":479666,"uuid":"4bf083bd-4ce3-4c64-92ce-1401672521a0","ttft_stream_ms":2352,"time_to_request_ms":912,"first_content_frame_ms":2353,"queued_turn_count":0,"result_index":0}

```

Separate stderr (verbatim):
```text

```

### Round 2

Full JSON stdout (verbatim):
```json
{"duration_api_ms":348144,"stop_reason":"end_turn","session_id":"06c3f1bc-06a1-4b32-816f-dcd002ce5077","total_cost_usd":2.2741645999999998,"usage":{"input_tokens":36,"cache_creation_input_tokens":155480,"cache_read_input_tokens":2134103,"output_tokens":30168,"output_tokens_details":{"thinking_tokens":23145},"server_tool_use":{"web_search_requests":0,"web_fetch_requests":0},"service_tier":"standard","cache_creation":{"ephemeral_1h_input_tokens":155480,"ephemeral_5m_input_tokens":0},"inference_geo":"not_available","iterations":[{"input_tokens":2,"output_tokens":5870,"cache_read_input_tokens":152623,"cache_creation_input_tokens":12681,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":12681},"type":"message"}],"speed":"standard","fallback_credit":null},"modelUsage":{"claude-opus-5-5":{"inputTokens":36,"outputTokens":30168,"cacheReadInputTokens":2134103,"cacheCreationInputTokens":155480,"webSearchRequests":0,"costUSD":2.2741645999999998,"contextWindow":1000000,"maxOutputTokens":128000,"thinkingTokens":23145,"canonicalModel":"claude-opus-5-5","provider":"firstParty","costBasis":"list"}},"permission_denials":[],"terminal_reason":"completed","fast_mode_state":"off","fast_mode_disabled_reason":"sdk_opt_in_required","subagent_stats":{"spawned":0,"requested":{"background":0,"foreground":0,"unset":0},"started_in_background":0,"max_depth":0,"spawned_by_subagents":0,"completed":0,"failed":0,"killed":{"parent":0,"user":0,"system":0},"refused":{"depth_limit":0,"concurrency_limit":0,"budget":0},"by_type":{}},"is_error":false,"num_turns":28,"subtype":"success","api_error_status":null,"result":"# CAMPAIGN_P1 round-2 review (Ponytail mode OFF)\n\n**Verdict: CHANGES REQUESTED.** One Medium issue remains, and the fix is small.\n\nI only read files: I ran no commands and edited nothing. Without running commands I couldn't recompute any SHA-256 values. Those are the archived SPEED1 files, `expected_hashes.json` checked against the phone, and `frozen_manifest_campaign_p1.json`. The manifest's sizes and hashes do match the brief, and the `RUN_INDEX.md` labels match the fixed wording the owner gave, word for word.\n\n## Round-1 findings re-checked\n| Round 1 | Status |\n|---|---|\n| 1. Weaker process refusal | **Fixed.** The explicit runner names are back in the pattern (`runtime.py:26-28`), and tests cover them (`self_check.py:391-397`). |\n| 2. Missing mock coverage | **Fixed.** New mock checks cover `LiveOps`, `bounded_selector`, selector `TimeoutError`/`OSError`, the busy-M2 branch in L1/L4, `prepare(restrict=…)`, the root-scope wrapper, the periodic sampler, session idle/continue/stop, and the monitor workers. |\n| 3. Unpinned and phase-locked work during blocks | **Partly fixed** (see F1). The per-frame guard now only does in-process checks. Process and health checks moved to a pinned 5.23 s worker, and transient su shells are pinned and read back. |\n| 4. Lag-probe bias | **Fixed.** `load_start_s` is taken after `guard()` and the pin check (`lag_probe.py:66-71`). The quiet phases no longer call pgrep at 10 Hz. |\n| 5–7. Smaller items | **Fixed.** Successful-but-late selector calls now count as misses. Repeated frames retry for 1.35 s. Gate and skin times are rebased. An LMK `ok` that is not true invalidates the block. Caps and memory exclude the tail. S1O's `indent=1` formatting is kept. `inference_ms` stats were added. Existing server logs are refused. |\n\nI also re-checked the reused interfaces this round, and they all match their sources:\n- `cr.root`, `read_dump`, `root_sample` and `lmk_lines` resolve the wrapped `root` through module globals.\n- `read_frame` returns `age_s`, and `camera_start(1)` returns `session`.\n- S1O `decide` and `_post` work with the 15 s `_post` replacement.\n- `capped_by_policy` gets rows rebased to the block start.\n- The hash-path keys match `detect_person.MODELS` and the `server_manager` path.\n- The 480 s gate equals `power_map.GATE_MAX_S`.\n\n## Findings\n\n1. **[Medium] The 5 s monitors are still locked to the 5 s 640/M2 cadence, and `RUN.md` describes this wrongly.**\n   - **Where:** `phase1.py:374-375` runs `read_dump` (offset 0.37 s) and the PSS `root_sample` (offset 1.31 s) every exactly 5 s. `cr.monitor_loop` keeps that fixed offset.\n   - **Problem:** A fixed offset doesn't remove phase lock when the period is the same. Every 640 slot gets the same instrumentation at the same point in the cycle:\n     - `dumpsys thermalservice` lands at slot+0.37, inside the 640 YOLO, in every layout.\n     - The heavy multi-process `dumpsys meminfo` sample (which includes the runner and llama-server) starts at slot+1.31. That is probably inside the MID (L3) and LITTLE (L1) M2 calls, but after BIG calls (L2/L4) have finished. The exact overlap depends on live timings.\n   - **Why it matters:**\n     - The Binder/system_server work is outside affinity control (`RUN.md` says so), so this load can land on measured cores in a way that differs by layout. That biases exactly the cross-layout M2 latency comparison the campaign is screening.\n     - SPEED1 BIG watts were invalidated for this same kind of phase-locking.\n   - **Doc problem:** `RUN.md:196-197` says \"offsets reduce synchronization\". For these two monitors that is false.\n   - **Fix:** give these monitors periods that don't divide evenly into 1 s or 5 s, as the 5.23 s shared check already does (for example 4.87 s and 5.13 s). Alternatively, state plainly in `RUN.md` that they are phase-locked and what that does to M2/YOLO timing.\n\n2. **[Low] The lag probe isn't really \"fastest reads\".**\n   - **Where:** `power.py:76`. With `period=0`, `verify()` runs before every read: two `root_mask` shell round trips plus the pump-affinity syscalls.\n   - **Effect:** each sample costs roughly three times as much, so the rate is lower and the gauge-lag resolution is coarser. The achieved Hz is reported honestly.\n   - **Fix:** in period-0 mode, verify every N reads or every 1 s.\n\n3. **[Low] The LMK window starts too early.** `lmk_since` is set when `live_block` begins (`phase1.py:332`). LMK lines from model builds and up to 480 s of gate wait are therefore counted against the 180 s block.\n\n4. **[Low] L1/L4 can skip an M2 call unnecessarily.** `busy` is sampled before this frame's YOLO (`phase1.py:142-143`) and reused at the 640 launch (`phase1.py:159-161`). If the previous M2 finished during YOLO, this M2 is still skipped and the block becomes INCOMPLETE. It fails closed, and with the expected live timings it is unlikely.\n\n5. **[Info] L2 runs M2 on the 640 frame from 5 s earlier.** At selector slots after 0, M2 uses the previous slot's 640 while the current 640 is still running on MID. This matches `RUN.md`'s wording \"latest completed 640\", but BIG M2 and MID YOLO-640 then overlap at every selector slot. Say so in `RUN.md`.\n\n6. **[Info] Signals inside a block become a RuntimeError.** A SIGINT/SIGTERM during a block is caught by `live_block`'s `except BaseException` (`phase1.py:394`) and reported as a shared failure. Cleanup still runs and later blocks stop, but the process exits with 1 instead of 128+signum.\n\n## Live paths still only mock-tested\nNone of these has run on the phone; each still needs owner validation:\n- Native root/su/pump affinity and readback, including the lag probe's `restrict` re-pin, the transient `taskset -p … $$` wrapper and its output format, and the Magisk daemon spawn.\n- Root services: Binder/system_server work for dumpsys thermalservice/meminfo, logcat, `settings`, `am`, `pidof`.\n- Battery `current_now`, `voltage_now` and `current_avg` (present or absent), and the gauge lag.\n- The achieved 0.37 s rate and coverage, gaps over 1.5 s, the end bracket, and the real CPU exclusion of the sampler and every monitor (including forked pgrep/su children).\n- Live Android policies and caps, CPU fault, battery thermal, skin/status freshness, and `block_limit` stops.\n- The charger, agents, process and cpuset/cores 4–7 refusals under native conditions.\n- The installed RobotCam version, mode B at rate 1, real frames (startup, session, age, gaps, repeat retries), `camera_end_check` and force-stop cleanup.\n- Real pinned YOLO and M2 thread identification and the 2–32 box limits on live detections.\n- L3 skips, L2 ordering and overlap, and L1/L4 concurrency.\n- The resident Gemma launch flags, `/health`, S1O HTTP with the 15 s per-request bound, the appended context, prompt tokens and timeouts.\n- MemAvailable, server PSS from `root_sample`, and logcat LMK.\n- The real 300 s idle, cooling, the skin gate and warm starts.\n- Screen timeout and wake-lock restore.\n- An interrupted native session: drains, resource cleanup, continuation versus stop, and the `cr.root` restore.\n- The lag probe end to end.\n- SHA-256 of the archive, `expected_hashes.json` and the manifest.\n\nThe portable dry run mocks all hardware and checks scheduling only.\n\n## Final verdict\n**CHANGES REQUESTED.** Fix or honestly document F1. F2–F4 are optional. Approval would not give commit or push authority in any case.","ttft_ms":3900,"type":"result","duration_ms":353963,"uuid":"26d32233-5d1f-41f5-966a-221ebdc02627","ttft_stream_ms":2502,"time_to_request_ms":1078,"first_content_frame_ms":2503,"queued_turn_count":0,"result_index":0}

```

Separate stderr (verbatim):
```text

```

### Round 3

Full JSON stdout (verbatim):
```json
{"duration_api_ms":124212,"stop_reason":"end_turn","session_id":"5549d13f-d414-4d7e-9fc6-7ceae9b5f350","total_cost_usd":1.2341922,"usage":{"input_tokens":16,"cache_creation_input_tokens":108795,"cache_read_input_tokens":749041,"output_tokens":10698,"output_tokens_details":{"thinking_tokens":7252},"server_tool_use":{"web_search_requests":0,"web_fetch_requests":0},"service_tier":"standard","cache_creation":{"ephemeral_1h_input_tokens":108795,"ephemeral_5m_input_tokens":0},"inference_geo":"not_available","iterations":[{"input_tokens":2,"output_tokens":2558,"cache_read_input_tokens":116631,"cache_creation_input_tokens":1988,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":1988},"type":"message"}],"speed":"standard","fallback_credit":null},"modelUsage":{"claude-opus-5-5":{"inputTokens":16,"outputTokens":10698,"cacheReadInputTokens":749041,"cacheCreationInputTokens":108795,"webSearchRequests":0,"costUSD":1.2341922,"contextWindow":1000000,"maxOutputTokens":128000,"thinkingTokens":7252,"canonicalModel":"claude-opus-5-5","provider":"firstParty","costBasis":"list"}},"permission_denials":[],"terminal_reason":"completed","fast_mode_state":"off","fast_mode_disabled_reason":"sdk_opt_in_required","subagent_stats":{"spawned":0,"requested":{"background":0,"foreground":0,"unset":0},"started_in_background":0,"max_depth":0,"spawned_by_subagents":0,"completed":0,"failed":0,"killed":{"parent":0,"user":0,"system":0},"refused":{"depth_limit":0,"concurrency_limit":0,"budget":0},"by_type":{}},"is_error":false,"num_turns":9,"subtype":"success","api_error_status":null,"result":"**Final verdict: APPROVED.** Every round-2 finding is fixed or documented, and I found nothing new that blocks this. Approving is not permission to commit or push: the owner still has to say yes to the commit, then separately yes to the push.\n\nI only read files. I ran no commands and edited nothing, so I could not recompute any SHA-256 values: not the archived SPEED1 files, not `expected_hashes.json` against the phone, and not the manifest. The `RUN_INDEX.md` labels match the owner's fixed wording.\n\n## Round-2 findings re-checked\n\n| Round 2 | Status |\n|---|---|\n| F1 [Medium] 5 s monitors locked to the 640/M2 cadence | **Fixed.** `phase1.py:372-376` now uses periods of 4.87 s (skin/status), 5.13 s (memory/PSS) and 5.23 s (shared checks). None of these divides evenly into 1 s or 5 s, which `self_check.py:530` asserts. The reused `cr.monitor_loop` (`coresidency.py:692-696`) keeps a fixed period with no realignment, so these monitors now drift relative to the 640 slots. `RUN.md` now says plainly that the instrumentation cost itself remains. |\n| F2 Lag probe verified masks on every read | **Fixed.** `power.py:75-78` checks once per second when `period=0`, plus once at the end. The achieved rate still includes that cost and `RUN.md` says so. |\n| F3 LMK window started too early | **Fixed.** `lmk_since` is reset at the shared block start (`phase1.py:369`). `lmk_window` keeps only lines in [since, since+180) and marks the block invalid if any timestamp is unparseable or non-finite (`phase1.py:452-466`). Checked by `self_check.py:487-494`. |\n| F4 Stale busy check before M2 launch | **Fixed.** M2 busy state is re-read under the lock right after YOLO-640 (`phase1.py:165-166`). The regression test (`self_check.py:511-528`) would fail if the stale value were used. |\n| F5 L2 uses the previous 640 frame and overlaps the current YOLO-640 | **Documented** in `RUN.md` under \"Reuse and layout details\". |\n| F6 Signal inside a block becomes a shared failure and exits 1 | **Documented** at the end of `RUN.md`. |\n\n## Other things I re-checked this round\n- **Reused interfaces:**\n  - `PinnedDetector` and `build_detector('mid')` give two threads, inter-op 1, and workers plus caller pinned to {4,5}. No campaign threads exist yet when ORT workers are identified, so the thread-diff method is safe.\n  - `CLUSTERS` = MID {4,5}/2, BIG {6,7}/2, LITTLE {0–3}/4.\n  - `root_sample` and `read_dump` resolve `root` through module globals, so `root_mask_scope` pins them.\n- **Skin gate:** it matches `power_map.thermal_gate`'s skin-only condition and its 480 s bound. One difference: a missing skin reading stops the gate with an error instead of waiting, which is stricter.\n- **Selector rows:** the fields from `cr.select` don't collide with the scheduler's own keys, and `correct` is dropped.\n- **Output files:** the expected-hash keys match `v.directory`, `dp.MODELS`, `cr.MODEL` and the `LLAMA_SERVER` paths. `.gitignore` re-includes only `benchmark/campaign/`. `*.log` stays ignored, so server logs don't enter the repo. `INDEX.md` gains exactly one row.\n\n## Non-blocking notes (Info)\n1. **llama-server threads are not pinned (Info).** Its four threads are unpinned, which is the original launch command, as required. So in every layout the selector competes for the measured clusters. `RUN.md` says \"No server CPU pinning\", but the cross-layout comparison has to take this into account.\n2. **Gate poll interval (Info).** `runtime.py:skin_gate` polls every 10 s instead of using `cr.GATE_POLL_S`. This only changes how often it checks, not the condition or its bound.\n\n## Every live path still only mock-tested\n- Native root/su/pump affinity and readback. This includes:\n  - the lag probe's re-pin to 0–3,\n  - the `taskset -p MASK $$` wrapper and its output format,\n  - Magisk daemon/control spawns,\n  - restoring the original `cr.root`.\n- Root services: dumpsys thermalservice and meminfo (Binder/system_server work outside affinity control), logcat `-v epoch -T`, `settings`, `am`, `pidof`, and the `dumpsys package` version read.\n- Battery `current_now`, `voltage_now` and `current_avg` (present or absent), signs and units, and the real gauge lag.\n- The achieved 0.37 s sampling rate and coverage, gaps over 1.5 s, the end bracket, and real CPU exclusion of the sampler, every monitor and their forked pgrep/su children.\n- Live Android policies and caps, CPU fault, battery thermal, skin/status freshness, and real `block_limit` stops.\n- The charger, agent, process and cpuset/cores 4–7 refusals under native conditions.\n- Installed RobotCam:\n  - versionCode/versionName,\n  - mode B at rate 1 producing real frames (startup, session, age, gaps, repeat retries),\n  - `camera_end_check` and force-stop cleanup.\n- M2/YOLO threads and scheduling under live timing:\n  - real pinned YOLO and M2 thread identification,\n  - 2–32 box limits on live detections,\n  - L3 skips, L2 ordering and frame freshness, L1/L4 concurrency and overload.\n- Resident Gemma:\n  - launch flags,\n  - `/health`,\n  - S1O HTTP with the 15 s per-request bound,\n  - the appended context, prompt tokens, timeouts and misses.\n- MemAvailable, llama-server PSS from `root_sample`, and the logcat LMK window and kill accounting.\n- The real 300 s idle, cooling, the skin gate and warm-start labelling.\n- Screen timeout and wake-lock restore.\n- An interrupted native session: worker drains, resource cleanup, continuation versus stop, and the exit status.\n- The lag probe end to end on the device.\n- SHA-256 of the archived files, `expected_hashes.json` against the phone, and the frozen manifest. These are unverified because I was not allowed to run commands.\n\nThe portable dry run mocks all hardware and checks scheduling only.\n\n**APPROVED**","ttft_ms":7194,"type":"result","duration_ms":127185,"uuid":"f706244e-c846-47f1-9e3c-94cac3776224","ttft_stream_ms":4521,"time_to_request_ms":901,"first_content_frame_ms":4525,"queued_turn_count":0,"result_index":0}

```

Separate stderr (verbatim):
```text

```

## Proposed commit inventory (conditional on APPROVED and owner yes commit)

Message: `benchmark: campaign phase-1 runner, power method fix, SPEED1 session archive`

| Exact path | Bytes |
|---|---:|
| `.gitignore` | 1066 |
| `benchmark/INDEX.md` | 6533 |
| `benchmark/campaign/RUN.md` | 13594 |
| `benchmark/campaign/checks/coder_dry_final.json` | 28790 |
| `benchmark/campaign/checks/coder_dry_final.stderr` | 0 |
| `benchmark/campaign/checks/coder_dry_final.stdout` | 322 |
| `benchmark/campaign/checks/coder_dry_final_block_01_L0.json` | 3587 |
| `benchmark/campaign/checks/coder_dry_final_block_02_L1.json` | 4222 |
| `benchmark/campaign/checks/coder_dry_final_block_03_L2.json` | 3946 |
| `benchmark/campaign/checks/coder_dry_final_block_04_L3.json` | 3988 |
| `benchmark/campaign/checks/coder_dry_final_block_05_L4.json` | 4222 |
| `benchmark/campaign/checks/coder_dry_final_block_06_L0.json` | 3588 |
| `benchmark/campaign/checks/coder_dry_round2.json` | 29390 |
| `benchmark/campaign/checks/coder_dry_round2.stderr` | 0 |
| `benchmark/campaign/checks/coder_dry_round2.stdout` | 323 |
| `benchmark/campaign/checks/coder_dry_round2_block_01_L0.json` | 3677 |
| `benchmark/campaign/checks/coder_dry_round2_block_02_L1.json` | 4305 |
| `benchmark/campaign/checks/coder_dry_round2_block_03_L2.json` | 4027 |
| `benchmark/campaign/checks/coder_dry_round2_block_04_L3.json` | 4067 |
| `benchmark/campaign/checks/coder_dry_round2_block_05_L4.json` | 4312 |
| `benchmark/campaign/checks/coder_dry_round2_block_06_L0.json` | 3669 |
| `benchmark/campaign/checks/coder_dry_round3.json` | 29401 |
| `benchmark/campaign/checks/coder_dry_round3.stderr` | 0 |
| `benchmark/campaign/checks/coder_dry_round3.stdout` | 323 |
| `benchmark/campaign/checks/coder_dry_round3_block_01_L0.json` | 3675 |
| `benchmark/campaign/checks/coder_dry_round3_block_02_L1.json` | 4311 |
| `benchmark/campaign/checks/coder_dry_round3_block_03_L2.json` | 4025 |
| `benchmark/campaign/checks/coder_dry_round3_block_04_L3.json` | 4071 |
| `benchmark/campaign/checks/coder_dry_round3_block_05_L4.json` | 4311 |
| `benchmark/campaign/checks/coder_dry_round3_block_06_L0.json` | 3675 |
| `benchmark/campaign/checks/coresidency_units.stderr` | 239 |
| `benchmark/campaign/checks/coresidency_units.stdout` | 9349 |
| `benchmark/campaign/checks/cycle_units.stderr` | 1094 |
| `benchmark/campaign/checks/cycle_units.stdout` | 1406 |
| `benchmark/campaign/checks/desk2_self_check.stderr` | 239 |
| `benchmark/campaign/checks/desk2_self_check.stdout` | 631 |
| `benchmark/campaign/checks/detector_units.stderr` | 344 |
| `benchmark/campaign/checks/detector_units.stdout` | 0 |
| `benchmark/campaign/checks/mission_units.stderr` | 340 |
| `benchmark/campaign/checks/mission_units.stdout` | 427 |
| `benchmark/campaign/checks/power_map_units.stderr` | 239 |
| `benchmark/campaign/checks/power_map_units.stdout` | 682 |
| `benchmark/campaign/checks/results.json` | 3029 |
| `benchmark/campaign/checks/self_check.stderr` | 239 |
| `benchmark/campaign/checks/self_check.stdout` | 709 |
| `benchmark/campaign/checks/self_check_r2.stderr` | 239 |
| `benchmark/campaign/checks/self_check_r2.stdout` | 1330 |
| `benchmark/campaign/checks/self_check_r3.stderr` | 239 |
| `benchmark/campaign/checks/self_check_r3.stdout` | 1454 |
| `benchmark/campaign/checks/speed1_self_check.stderr` | 239 |
| `benchmark/campaign/checks/speed1_self_check.stdout` | 1460 |
| `benchmark/campaign/expected_hashes.json` | 1040 |
| `benchmark/campaign/lag_probe.py` | 5889 |
| `benchmark/campaign/phase1.py` | 29098 |
| `benchmark/campaign/power.py` | 5736 |
| `benchmark/campaign/runtime.py` | 6316 |
| `benchmark/campaign/self_check.py` | 35598 |
| `benchmark/relate_anything/speed1/runs/RUN_INDEX.md` | 2936 |
| `benchmark/relate_anything/speed1/runs/owner_preflight_speed1_fix.json` | 4385 |
| `benchmark/relate_anything/speed1/runs/owner_preflight_speed1_fix.stderr` | 239 |
| `benchmark/relate_anything/speed1/runs/owner_preflight_speed1_fix.stdout` | 1267 |
| `benchmark/relate_anything/speed1/runs/owner_session_speed1_fix.json` | 2358903 |
| `benchmark/relate_anything/speed1/runs/owner_session_speed1_fix.stderr` | 239 |
| `benchmark/relate_anything/speed1/runs/owner_session_speed1_fix.stdout` | 13488 |
| `benchmark/relate_anything/speed1/runs/owner_session_speed1_fix_block_01_fp32_MID.json` | 479134 |
| `benchmark/relate_anything/speed1/runs/owner_session_speed1_fix_block_02_fp32_BIG.json` | 477789 |
| `benchmark/relate_anything/speed1/runs/owner_session_speed1_fix_block_03_fp32_LITTLE.json` | 533291 |
| `benchmark/relate_anything/speed1/runs/owner_session_speed1_fix_block_04_fp32_MID.json` | 478102 |
| `benchmark/campaign/frozen_manifest_campaign_p1.json` | 12844 |

Total 69 files, 4,651,642 candidate bytes (working file sizes; existing-file sizes included, not diff size).

Current git status:
```text
 M .gitignore
 M benchmark/INDEX.md
?? benchmark/campaign/
?? benchmark/relate_anything/speed1/runs/RUN_INDEX.md
?? benchmark/relate_anything/speed1/runs/owner_preflight_speed1_fix.json
?? benchmark/relate_anything/speed1/runs/owner_preflight_speed1_fix.stderr
?? benchmark/relate_anything/speed1/runs/owner_preflight_speed1_fix.stdout
?? benchmark/relate_anything/speed1/runs/owner_session_speed1_fix.json
?? benchmark/relate_anything/speed1/runs/owner_session_speed1_fix.stderr
?? benchmark/relate_anything/speed1/runs/owner_session_speed1_fix.stdout
?? benchmark/relate_anything/speed1/runs/owner_session_speed1_fix_block_01_fp32_MID.json
?? benchmark/relate_anything/speed1/runs/owner_session_speed1_fix_block_02_fp32_BIG.json
?? benchmark/relate_anything/speed1/runs/owner_session_speed1_fix_block_03_fp32_LITTLE.json
?? benchmark/relate_anything/speed1/runs/owner_session_speed1_fix_block_04_fp32_MID.json
```

STOP at human gate. Approval does not authorize commit. Commit only after “yes commit”; push only after a separate “yes push”.

Final frozen/source-archive verification: PASS. HEAD unchanged, `git diff --check` passed and staged diff empty. All candidate sizes/SHA-256 and all ten source/archive hashes rechecked after APPROVED, with no candidate edits.

Manifest SHA-256: `8e998d61bf85fdb04b9ec708e7e3ae002b7a79be4d841ff7264a37530c6e0883`.


## Publication after owner authorization

Owner requested “commi and push”. Committed and pushed the approved 69-file candidate to origin/main.

Commit: c47fc41999d7190b4e8171b8e7806c6b77b5e7dd

Remote main verified equal to HEAD; working tree clean. Code and documentation passed staged whitespace checks. Two trailing spaces in archived owner stdout were preserved to satisfy byte-identical archival.
