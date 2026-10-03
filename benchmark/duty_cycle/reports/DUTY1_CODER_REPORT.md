# DUTY1 Coder report

Status: round 3 APPROVE WITH NOTES, no blocking findings. Termination-priority fix and all required offline reruns passed. All three independent verdicts and environment limitations retained below. No hardware measurements, staging, commit or push. Stopped for the human decision.

EXECUTION PROFILE
ROLE: Coder
PLATFORM: Codex CLI
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: lite
WHY: Bounded new benchmark built on the reviewed power_map/coresidency runners; no robot or motor code touched.
FALLBACK: Claude Code CLI | claude-opus-5-5 | effort medium | Ponytail lite

REVIEWER PROFILE (you invoke it; never review your own work)
ROLE: Reviewer
PLATFORM: Codex CLI (codex exec, fresh session every round)
MODEL: gpt-6.1-sol
EFFORT: medium
PONYTAIL: off
WHY: Independent correctness and safety review in a separate fresh session.
FALLBACK: AGY CLI | gemini-3.1-pro-high | effort N/A | Ponytail off


## Base state

Checked before acting: HEAD `7c77f1439544634ede8b8baf92aa03e3b35b24b6`.
Initial `git status --short`: ` M .gitignore`, `?? benchmark/power_map/`, exactly expected.
POWERMAP source SHA-256 `4756ce7dc7ba4e358578288c0d2509345fc4e2fe02d45f077f65355792e29539`;
launcher SHA-256 `874dfc850c47becfe5226dd0c184e5ffcb9b8a474b6848259ed193a27ebc49dd`.
Both rechecked unchanged after tests. No existing runner or robot source edited.
Read entry/role/prototype instructions, #121–#127, and all requested source/context.
The requested `benchmark/power_map/runs/` directory does not exist. File search found the unique
existing CONFIRM run at `/termux-home/power_map/run_20261002T090821Z_CONFIRM`; analysis prints its resolved native path.

## Changed/new files

`.gitignore` retains POWERMAP1's existing uncommitted exception and adds only `!benchmark/duty_cycle/`,
the same handling for the blanket `benchmark/*` rule. The human decides what to commit.


- `.gitignore` SHA-256 `2e82b2fc97e6c00ca7f0f014e34e413e57ab21afef169470719533021ec5f01b`

- `benchmark/duty_cycle/RUN.md` SHA-256 `ee174ae174e67ef6d5c9a435d713d4a18678bfe7b7b1e20888b8b281ea7d59a5`

- `benchmark/duty_cycle/confirm_view.py` SHA-256 `9e563fdea50b38c430b586d401c2bb5568df834e5b4d2cbf52d67c922308045f`

- `benchmark/duty_cycle/duty_cycle.py` SHA-256 `5a5d4bd01099572955057e7aef4881c30afe13b0e922696ad0ab7397767a6e0a`

- `benchmark/duty_cycle/oneshot.sh` SHA-256 `96c04273f78ba90ac3b8bba725165a87a9bfb94e45839eb689c0f9e613d6b880`

- `benchmark/duty_cycle/run_duty_cycle.sh` SHA-256 `33fa21f5b34dd919761c3ecb1c9aedac5cb65d8f983d44e90a7963ebfa30b4dc`

- `benchmark/duty_cycle/test_duty_cycle.py` SHA-256 `b26a1f61faa32890af691f3c212b0536d7f7150fb0ff893ae8d94c437bf5537b`

- `benchmark/duty_cycle/test_oneshot.sh` SHA-256 `eb2483d2a780e6a7bab8cd2f224d4c356020d0f610f4d75495481c1ee8637d7d`


## Reused and replaced

Reused by import: power_map/coresidency `thermal_gate` (skin only, bounded 8 min), `block_limit`, `latest`,
`camera_start(1)` with `min_capture_boot_s`, `camera_stop`, `camera_end_check` and force-stop,
`camera_end_failed`, `Server`/LIVE/signal ownership and deployed flags, `make_selector`, `load_cases`,
`select`, `WARMUP`, `resident_pages` (mincore), `RootShell`, `discover`, `fast_keys`, `fast_sample`,
`read_dump`, `monitor_loop` sampler cadence, `meminfo_mib`, `read_thermal`, cpuset guard, `lmk_lines`,
MID `build_detector`/`PinnedDetector`, `thread_sample`/`observed_workers`, `SizePolicy`,
`capped_by_policy`, `skin_slope`, `status_changes`, median/P95/format/SHA helpers.

Small DUTY1 replacements: `battery_sample` uses a separate persistent root shell at 0.5 s without
camera PSS queries (the existing 5 s sampler assumes camera-on); `selector_phase` and phase planner
count planned active time and drain calls before pause; `run_block` covers camera-off parts, camera
restarts and an atomic HEATCOOL; `load_once`/`supervised_load` record repeated loads and keep limit
checks live during the load/request; `cpu_snapshot`/`parse_stat`/`cpu_seconds` add process CPU accounting;
no-spinning detector construction adds/reads back ORT options while retaining MID verification;
energy integration and duty report handle derived baselines, whole cycles and COOL windows;
atomic block save/resume handles interruption. `confirm_view.py` is standalone stdlib file reading.

Launcher and native wrapper copied from POWERMAP1 with folder/name/set changes and duty_cycle added
to the other-runner refusal. Offline launcher test adapted from the existing coresidency mock test.
No dependencies added. Reusing these helpers is the simpler alternative to owning duplicate runners.

## Round 1 verification (historical)

All checks below are offline. Android/root/model operations are mocked. The existing harnesses use
fake `su` executables and local fake HTTP servers; no real `su`, `am`, RobotCam, launcher or
llama-server was executed. Existing tests' synthetic report numbers are fixtures, not measurements.
Five test commands exited zero. Bash syntax and `git diff --check` passed.


### `python -B benchmark/duty_cycle/test_duty_cycle.py` — exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
PARTS/CYCLE/smoke block lists and active-only 20 s selector clock: PASS
restart passes request CLOCK_BOOTTIME; actual reader rejects pre-restart capture: PASS
energy above baseline and per-call interpolation/missing coverage; proc stat parser: PASS
LOAD spawn/health/call CPU split, GGUF cold/warm, stop on interruption, worker joined: PASS
both ORT spinning options retained; rejection produces INCOMPLETE without fallback: PASS
synthetic report/invalid battery; atomic resume preserves finished blocks and redoes HEATCOOL; proot refusal: PASS
mocked main + real block loops: interrupt/resume preserves CONT, redoes HEATCOOL, no COOL gate, final cleanup: PASS
DUTY1 offline checks: PASS

```

### `bash benchmark/duty_cycle/test_oneshot.sh` — exit 0

```text
ok: old timeout read through the file
ok: runner gets the arguments
ok: timeout restored and read back
ok: setting is 60000 afterwards
ok: error text is not taken as a timeout
ok: not started
ok: no restore after a failed read
ok: setting untouched
ok: unchanged setting after put is caught
ok: not started
ok: failed restore is reported
ok: failed restore exits non-zero (1)
ok: refuses with a runner running
ok: refuses while another launcher holds the lock
ok: the other run's lock and logger untouched
ok: refused launcher ran no cleanup
ok: refuses with codex running
ok: refuses with llama-server running
ok: runner failure reported
ok: launcher exits with the runner status
ok: timeout restored after a runner failure
ok: SIGTERM during the idle stops the launcher at once (exit 143)
ok: timeout restored after SIGTERM during the idle
ok: logger and watcher told to stop after SIGTERM during the idle
ok: SIGHUP during the idle stops the launcher at once (exit 129)
ok: timeout restored after SIGHUP during the idle
ok: logger and watcher told to stop after SIGHUP during the idle
ok: SIGINT during the idle stops the launcher at once (exit 130)
ok: timeout restored after SIGINT during the idle
ok: logger and watcher told to stop after SIGINT during the idle
ok: SIGTERM to the launcher reaches the runner
ok: launcher waits for the runner after SIGTERM (exit 143)
ok: timeout restored after SIGTERM
ok: SIGHUP to the launcher reaches the runner
ok: launcher waits for the runner after SIGHUP (exit 143)
ok: timeout restored after SIGHUP
ok: charger plugged in during the idle is caught
ok: runner not started (exit 1)
ok: timeout restored after the late refusal
ok: refuses while charging
ok: no logger or watcher left running
ok: lock released after every run

```

### `python -B benchmark/power_map/test_power_map.py` — exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block lists and CLI setting validation: PASS
5/10/off size sequences and frame pacing at rates 1/2; limit before first read: PASS
separate capped policies, skipped gaps, unknown readings, excluded edges/tail; least-squares skin slope: PASS
stale pre-pin processor excluded for idle workers; active in/out-of-target CPUs verified: PASS
synthetic multi-column report, off cadence, INCOMPLETE cpufreq/missing block, ORT observations: PASS
native/proot refusal, /proc stat/status parser, detector-only caller lifecycle (mock affinity): PASS
fake integration: interruption/resume preserves completed block, resident Gemma, report copy, camera cleanup: PASS
POWERMAP offline checks: PASS

```

### `python -B benchmark/coresidency/test_coresidency.py` — exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block limits: status 3 no stop / 4 stop, battery 44.9 / 45.0, CPU 110 in 3 consecutive (fault), skin+status loss 60 s and CPU/battery loss 5 s fail closed: PASS
  [B1] waiting: skin 36.0 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 34.0 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 1 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 1 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 1 s
cooldown gate: skin <= idle + 1.5 and z9 <= idle + 4 reached; else warm start after the limit: PASS
RobotCam end check: stopped after STOP / still advancing / unreadable reads; force-stop failure INCOMPLETE; cached pid recorded only: PASS
capped time: pre-block and post-block samples excluded, time-weighted, % of block; unreadable scaling_max INCOMPLETE: PASS
layout discovery, 1 s sample and dump parsing: PASS
selector slots = those that start inside the block (smoke 1, full 9): PASS
limit stop: short stop valid, drift n/a, in-flight call allowed; limit at start and fail-closed stop INCOMPLETE: PASS
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpu07xf0ci/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpu07xf0ci/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpu07xf0ci/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpu07xf0ci/.drop_done
cold load: full-cold iff the drop answered ok and no load-file page is cached (mincore): PASS
failed logcat query reported as failed, not as 0 kills: PASS
per-pid PSS failure flagged; late sample dropped; gaps from completion times: PASS
signal during a server load or inside its Popen stops the server; all-failed selector block flagged: PASS
cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE; old-format run refused by --resume: PASS
Co-residency benchmark (DECISIONS #124), run run_20261002T130230Z_smoke  [SMOKE: not a measurement]
block length 8 s; idle z9 90.0 degC, idle skin 35.0 degC; runner sha256 85a6e4cb003b

                                        B1_mix_nollm    B2_only640_nollm        B3_mix_gemma    B4_only640_gemma
frames processed                                  16                  16                  16                   8
failed reads by status                          none                none                none                none
repeat reads (not failures)                        3                   5                   4                   1
detect320 ms median/P95                 10/11 (n 15)       n/a/n/a (n 0)        10/11 (n 15)       n/a/n/a (n 0)
detect640 ms median/P95                  30/30 (n 1)        30/31 (n 16)         30/30 (n 1)         30/31 (n 8)
drift320 first/last 30s                     10 -> 10          n/a -> n/a            10 -> 10          n/a -> n/a
drift640 first/last 30s                     30 -> 30            30 -> 30            30 -> 30          n/a -> n/a
read/decode ms median                       1.9/10.3            2.9/12.5             3.7/9.2            2.9/10.9
frame age s median                              0.08                0.09                0.06                0.09
selector ms median/P95                             -                   -             267/284             260/274
selector calls ok/err/correct                  0/0/0               0/0/0               3/0/1               2/0/1
min MemAvailable MiB                            3248                3273                3258                3265
max swap used MiB                               2238                2245                2241                2241
peak PSS runner MiB                              121                 121                 121                 121
peak PSS llama-server MiB                        n/a                 n/a                 121                 121
peak PSS RobotCam app MiB                         37                  37                  37                  37
peak PSS camera provider MiB                     264                 264                 264                 264
LMK log lines (kill lines)                     3 (2)               3 (2)               3 (2)               3 (2)
survived RobotCam / llama                    yes / -             yes / -           yes / yes           yes / yes
zone9 start/end/max degC              90.0/90.0/90.0      90.0/90.0/90.0      90.0/90.0/90.0      90.0/90.0/90.0
VIRTUAL-SKIN start/end/max degC       35.0/35.0/35.0      35.0/35.0/35.0      35.0/35.0/35.0      35.0/46.6/46.6
Android status max                                 3                   3                   3                   4
policy capped s (% of block)                  0 (0%)              0 (0%)              0 (0%)             2 (49%)
lowest scaling_max MHz 0/4/6          1803/2348/2850      1803/2348/2850      1803/2348/2850      1803/2348/2400
gate wait s                                        0                   0                   0                   0
stop limit                                         -                   -                   -      android_status
time to limit s                                    -                   -                   -                 4.1
mean battery W                                  1.64                1.64                1.64                1.64
battery status                           Discharging         Discharging         Discharging         Discharging
sample errors                                      0                   0                   0                   0
sample gaps >3s (max s)                      0 (2.0)             0 (2.0)             0 (2.0)             0 (2.0)
samples past block end (dropped)                   0                   0                   0                   0

Gemma load (spawn to /health ok): cold 0.90 s [weights-cold (posix_fadvise DONTNEED on the GGUF)], warm 0.68 s; gate wait 0 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmpw0njb5fo/a/server.pid

MTP snapshot (RobotCam + both detector sessions + Gemma with drafter, after one request): MemAvailable 3261 MiB, swap used 2238 MiB, PSS llama-server 121 MiB, VmHWM 24 MiB, load 0.76 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmpw0njb5fo/a/server.pid --model-draft /data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3

Notes: detect ms = Detector.detect (resize, inference, NMS); read/decode split read_frame at its BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Block limits, checked between frames: Android thermal status >= 4 (CRITICAL), battery >= 45.0 degC, CPU zone >= 110 degC in 3 consecutive 1 s samples (fault stop); time to limit = when the runner saw it (up to one detection plus the 1 s/5 s reading interval late); the block ends there and counts as run; drift is n/a if it ran under 60 s. Skin/status = `dumpsys thermalservice` every 1 s; capped = time (each in-block 1 s sample counts the time since the previous one) with any policy's scaling_max_freq below its cpuinfo_max_freq, % of the block length. Gate: skin <= idle + 1.5 and z9 <= idle + 4 degC; warm start = not reached in 15 min.

full smoke run: PASS
SIGTERM during B3: exit 143, RobotCam stopped, server stopped: PASS
--resume after SIGTERM: PASS
failed STOP broadcast: force-stop still runs: PASS

```

### `bash benchmark/coresidency/test_oneshot.sh` — exit 0

```text
ok: old timeout read through the file
ok: runner gets the arguments
ok: timeout restored and read back
ok: setting is 60000 afterwards
ok: error text is not taken as a timeout
ok: not started
ok: no restore after a failed read
ok: setting untouched
ok: unchanged setting after put is caught
ok: not started
ok: failed restore is reported
ok: failed restore exits non-zero (1)
ok: refuses with a runner running
ok: refuses while another launcher holds the lock
ok: the other run's lock and logger untouched
ok: refused launcher ran no cleanup
ok: refuses with codex running
ok: refuses with llama-server running
ok: runner failure reported
ok: launcher exits with the runner status
ok: timeout restored after a runner failure
ok: SIGTERM during the idle stops the launcher at once (exit 143)
ok: timeout restored after SIGTERM during the idle
ok: logger and watcher told to stop after SIGTERM during the idle
ok: SIGHUP during the idle stops the launcher at once (exit 129)
ok: timeout restored after SIGHUP during the idle
ok: logger and watcher told to stop after SIGHUP during the idle
ok: SIGINT during the idle stops the launcher at once (exit 130)
ok: timeout restored after SIGINT during the idle
ok: logger and watcher told to stop after SIGINT during the idle
ok: SIGTERM to the launcher reaches the runner
ok: launcher waits for the runner after SIGTERM (exit 143)
ok: timeout restored after SIGTERM
ok: SIGHUP to the launcher reaches the runner
ok: launcher waits for the runner after SIGHUP (exit 143)
ok: timeout restored after SIGHUP
ok: charger plugged in during the idle is caught
ok: runner not started (exit 1)
ok: timeout restored after the late refusal
ok: refuses while charging
ok: no logger or watcher left running
ok: lock released after every run

```

## Offline CONFIRM analysis

`python -B benchmark/duty_cycle/confirm_view.py` — exit 0. Expected answers are absent in recorded call JSON, so not inferred from today's cases. The short minute 21 is actual recorded block overrun.

```text
CONFIRM source: /data/data/com.termux/files/home/power_map/run_20261002T090821Z_CONFIRM
Capped s uses the POWERMAP later-reading convention; last-sample tail is uncredited.
minute | skin mean/max C | battery mean W | capped s policy0/4/6 | Android status max | selector calls/correct
1 | 27.356/28.251 | 2.328 | 56.283/0.0/0.0 | 0 | 3/3
2 | 28.789/29.213 | 2.277 | 60.0/0.0/0.0 | 0 | 3/3
3 | 29.752/30.144 | 1.984 | 60.0/0.0/0.0 | 0 | 3/3
4 | 30.512/30.836 | 2.658 | 60.0/0.0/0.0 | 0 | 3/3
5 | 31.263/31.615 | 2.708 | 60.0/0.0/0.0 | 0 | 3/2
6 | 31.937/32.214 | 2.297 | 60.0/0.0/0.0 | 0 | 3/2
7 | 32.505/32.748 | 3.472 | 60.0/0.0/0.0 | 0 | 3/2
8 | 33.009/33.244 | 3.587 | 60.0/0.0/0.0 | 0 | 3/2
9 | 33.558/33.801 | 3.559 | 60.0/0.997/0.997 | 0 | 3/1
10 | 34.015/34.211 | 3.144 | 60.0/0.0/0.0 | 0 | 3/2
11 | 34.502/34.809 | 2.546 | 60.0/1.006/1.006 | 0 | 3/3
12 | 34.936/35.099 | 2.061 | 60.0/0.0/0.0 | 0 | 3/2
13 | 35.29/35.455 | 2.752 | 60.0/0.0/0.0 | 0 | 3/3
14 | 35.644/35.906 | 2.103 | 60.0/0.0/0.0 | 0 | 3/2
15 | 36.08/36.221 | 2.979 | 60.0/0.0/0.0 | 0 | 3/2
16 | 36.398/36.626 | 1.984 | 60.0/1.002/1.002 | 0 | 3/2
17 | 36.955/37.174 | 3.272 | 60.0/0.0/7.287 | 0 | 3/3
18 | 37.2/37.357 | 3.995 | 60.0/9.006/26.713 | 0 | 3/3
19 | 37.451/37.652 | 3.65 | 60.0/14.004/31.295 | 0 | 3/1
20 | 37.745/37.912 | 3.366 | 60.0/14.0/26.697 | 0 | 3/2
21 | not recorded/not recorded | 3.24 | 0.705/0.0/0.0 | not recorded | 0/0
policy4 first recorded cap s: 520.705
policy6 first recorded cap s: 520.705
Selector calls not recorded correct: time s | case id | expected | returned | ms
280.0 | L2_03 | not recorded | go to the next unsearched room | 1872.4
300.0 | L2_04 | not recorded | search this room | 1922.2
360.0 | L2_07 | not recorded | go to the next unsearched room | 1841.5
440.0 | L2_11 | not recorded | go to the next unsearched room | 1846.1
480.0 | L3_01 | not recorded | go to the kitchen | 2316.9
500.0 | L3_02 | not recorded | go to the bedroom | 1860.3
580.0 | L3_06 | not recorded | go to the kitchen | 1779.6
700.0 | L3_12 | not recorded | go to the bedroom | 2035.8
820.0 | L4_06 | not recorded | go to the bedroom | 2459.1
860.0 | L4_08 | not recorded | go to the bedroom | 1973.2
940.0 | L4_12 | not recorded | go to the kitchen | 2203.4
1080.0 | L5_07 | not recorded | go to the living room | 3374.7
1120.0 | L5_09 | not recorded | search this room again | 3836.1
1140.0 | L5_10 | not recorded | go to the living room | 3624.5

```

## Independent review

Requested fresh Codex CLI session, `gpt-6.1-sol`, effort `medium`, read-only, Ponytail off.
Exact command per round: `codex exec -m gpt-6.1-sol -c model_reasoning_effort="medium" -s read-only -C /termux-home/robot - < /termux-home/duty1_review_request_<round>.md`.
Requests include complete task, verified base, tracked diff, full untracked new-file contents, source context and full actual check/analysis output.
Stdout review and stderr diagnostics are retained separately under `/termux-home/duty1_review_<round>.*.txt`.

### Round 1 — verbatim reviewer stdout

1. **[P1] Restore POWERMAP’s rate-1 frame pacing** — [duty_cycle.py:365](/termux-home/robot/benchmark/duty_cycle/duty_cycle.py:365). The loop sleeps only 50 ms after a new frame and 20 ms after repeats. Repeat detection happens after `read_frame()` reads and decodes the JPEG, so CAM and Active repeatedly decode unchanged frames between camera captures. POWERMAP instead paces successful reads to one second. This adds CPU and battery load while the report can still declare the measurement complete, undermining the requested CONFIRM workload and power comparison. Preserve POWERMAP’s pacing and add an offline cadence check.

2. **[P1] Finish selector cleanup before retrying on core loss** — [duty_cycle.py:387](/termux-home/robot/benchmark/duty_cycle/duty_cycle.py:387), [duty_cycle.py:698](/termux-home/robot/benchmark/duty_cycle/duty_cycle.py:698). During selector draining, `limit()` can raise `CoresLost`, escaping cleanup before the selector thread is joined. The retry handler stops the camera but retains the server and selector. A slow in-flight request can therefore overlap the replacement block, consume unrecorded CPU/power, and contaminate numbers subsequently reported as valid. Ensure exceptional cleanup cancels or drains the request and joins its thread before any retry; test core loss during a pending call.

HEAD and both supplied POWERMAP hashes matched. No edits or hardware execution occurred. Further inspection encountered a sandbox mount error; tests were not independently rerun.

**REQUEST CHANGES**


Nested round-1 `codex exec` exited 0 and returned REQUEST CHANGES. It reported a sandbox mount error blocking further reads and did not independently rerun tests. Coder initially stopped on that environment error. The human subsequently ruled that a verdict based on the complete supplied diff/context counts as a completed review for this task; record the error, but continue. Permission/authentication/quota failures yielding no verdict still require stopping. This rule supersedes the earlier incomplete-review classification. No fallback or permission bypass was used.

Reviewer stderr diagnostics retained separately: `/termux-home/duty1_review_1.stderr.txt`. Full request: `/termux-home/duty1_review_request_1.md`.


## Open findings and unverified items

Round-1 pacing finding is accepted in round 2. Core-loss draining now joins before retry and records drain evidence, but round 2 found that a later termination signal can be lost while another exception is retained. This finding was open at the round-2 stop. It was fixed under the later human authorization and accepted in round 3; no blocking findings remain.

Real camera restart behaviour and duration, Android ORT spinning-option
acceptance/effect, fuel-gauge averaging, GGUF warm/cold residency on the phone, real MID pinning and
performance, root-query sampling latency, heat/cap traces and process CPU readings are unverified.
Battery sysfs averaging is explicitly labelled unverified in RUN.md/report; 0.5 s is the requested
sample interval, not proof of a fuel-gauge update interval. Partial GGUF residency is labelled warm
with its page count; it is not a full-cold OS/library measurement.
CYCLE does not measure P0/P0U, so above-baseline derived energy is n/a there; direct required powers
are recorded. Derived energy is n/a when a boundary is uncovered or a gap exceeds 1.5 s.
Per-process snapshots include block boundaries and before process stops; short unobserved process
lifetimes and work after the last pre-stop snapshot can be missed. Hardware tests are human-only.

Stop for human decision. Nothing staged, committed or pushed.

## Final state and stop evidence

```text
 M .gitignore
?? benchmark/duty_cycle/
?? benchmark/power_map/
```

HEAD: `7c77f1439544634ede8b8baf92aa03e3b35b24b6`.
Final unchanged SHA-256 `benchmark/power_map/power_map.py`: `4756ce7dc7ba4e358578288c0d2509345fc4e2fe02d45f077f65355792e29539`.
Final unchanged SHA-256 `benchmark/power_map/oneshot.sh`: `874dfc850c47becfe5226dd0c184e5ffcb9b8a474b6848259ed193a27ebc49dd`.

Round-1 source hashes remain in its frozen request. Current candidate hashes above match the frozen round-2 request. No candidate edits during or after round 2.

## Round 2 fixes and authorization

Only `benchmark/duty_cycle/duty_cycle.py` and `test_duty_cycle.py` changed since round 1. Profiles remain unchanged.

1. Shared `camera_frame` now applies POWERMAP's exact successful-read pacing expression: `max(0.0, read_start + 1/rate - perf_counter())`. Read/decode/detection consume the one-second period. CAM, YOLO, YOLO_NOSPIN and Active all call this same iteration. Repeat/error retry intervals match POWERMAP. The offline regression runs the actual JPEG reader/decoder and counts exactly one read and decode per second for all four workloads, with simulated read/detect time included.
2. Shared `drain_selector` sets the stop event, waits for the pending request, catches exceptions from limit checking or joining, cancels via server shutdown on the existing 120 s timeout, and always joins before exception propagation/retry/exit. A stopped owned server is cleared before core-loss retry so it can reload. Drain duration is stored as `selector_drain_s` in phase data and printed; `selector_drains.jsonl` retains block/phase/error/drain time for interrupted/discarded blocks too. The regression injects CoresLost during a pending call's drain, proves `call finished` precedes `retry started`, and checks retained drain timing. Additional assertions cover generic errors, signals, an already-unwinding CoresLost and timeout cancellation.

Latest human instruction, verbatim:

```text
Local AI decision on DUTY1 round 1: the sandbox mount error is the same environmental issue seen in both POWERMAP1 reviews; the reviewer had the complete diff and context in its request, and its findings are valid. For this task, a reviewer that reports this mount error but returns a verdict based on the supplied request counts as a completed review, as long as you record the error in the report. Permission, authentication or quota failures that produce no verdict still mean stop.

Fix both round-1 findings, nothing else:
1. Restore POWERMAP's rate-1 pacing in every camera-on phase (CAM, YOLO, YOLO_NOSPIN, Active): reuse power_map's pacing so successful reads are paced to 1/rate seconds and unchanged frames are not decoded repeatedly. Add an offline check that counts reads/decodes per second at rate 1.
2. On CoresLost (or any exception) during a block, including during selector draining: drain or cancel the in-flight selector request and join its thread before any retry or exit; record the drain time. Add an offline test with core loss during a pending selector call that proves the call has finished before the retried block starts.

Rerun all tests and confirm_view.py, then run review round 2 in a fresh session (max rounds stays 3). Update DUTY1_CODER_REPORT.md in Downloads with the fixes, new SHA-256 values, test output and round 2 verbatim. Then stop.

```

## Round 2 verification

All five test commands below exited 0. New checks use virtual/mocked clocks, JPEG fixtures, mock Android/root/model calls, and a pending fake selector. Timings in these outputs are fixtures, not hardware measurements. No real su/am/launcher/camera/llama-server execution. Bash syntax and `git diff --check` passed. Existing POWERMAP source and launcher hashes remained unchanged.

### `python -B benchmark/duty_cycle/test_duty_cycle.py` — exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
PARTS/CYCLE/smoke block lists and active-only 20 s selector clock: PASS
restart passes request CLOCK_BOOTTIME; actual reader rejects pre-restart capture: PASS
energy above baseline and per-call interpolation/missing coverage; proc stat parser: PASS
CAM/YOLO/YOLO_NOSPIN/Active rate 1: exactly one read and JPEG decode per second, work included in period: PASS
LOAD spawn/health/call CPU split, GGUF cold/warm, stop on interruption, worker joined: PASS
[CONT] selector drain 0.050s; RuntimeError: failed limit query
[CONT] selector drain 0.052s; SystemExit: 143
[CONT] selector drain 0.052s; CoresLost: body core loss
[CONT] selector drain 122.000s; RuntimeError: selector drain exceeded 120 s; request cancelled
selector drain: arbitrary error, signal, unwinding CoresLost and timeout cancellation all join before propagation: PASS
both ORT spinning options retained; rejection produces INCOMPLETE without fallback: PASS
synthetic report/invalid battery; atomic resume preserves finished blocks and redoes HEATCOOL; proot refusal: PASS
core loss during pending selector drain: request finished before retry; discarded-block drain time recorded: PASS
mocked main + real block loops: interrupt/resume preserves CONT, redoes HEATCOOL, no COOL gate, final cleanup: PASS
DUTY1 offline checks: PASS

```
### `bash benchmark/duty_cycle/test_oneshot.sh` — exit 0

```text
ok: old timeout read through the file
ok: runner gets the arguments
ok: timeout restored and read back
ok: setting is 60000 afterwards
ok: error text is not taken as a timeout
ok: not started
ok: no restore after a failed read
ok: setting untouched
ok: unchanged setting after put is caught
ok: not started
ok: failed restore is reported
ok: failed restore exits non-zero (1)
ok: refuses with a runner running
ok: refuses while another launcher holds the lock
ok: the other run's lock and logger untouched
ok: refused launcher ran no cleanup
ok: refuses with codex running
ok: refuses with llama-server running
ok: runner failure reported
ok: launcher exits with the runner status
ok: timeout restored after a runner failure
ok: SIGTERM during the idle stops the launcher at once (exit 143)
ok: timeout restored after SIGTERM during the idle
ok: logger and watcher told to stop after SIGTERM during the idle
ok: SIGHUP during the idle stops the launcher at once (exit 129)
ok: timeout restored after SIGHUP during the idle
ok: logger and watcher told to stop after SIGHUP during the idle
ok: SIGINT during the idle stops the launcher at once (exit 130)
ok: timeout restored after SIGINT during the idle
ok: logger and watcher told to stop after SIGINT during the idle
ok: SIGTERM to the launcher reaches the runner
ok: launcher waits for the runner after SIGTERM (exit 143)
ok: timeout restored after SIGTERM
ok: SIGHUP to the launcher reaches the runner
ok: launcher waits for the runner after SIGHUP (exit 143)
ok: timeout restored after SIGHUP
ok: charger plugged in during the idle is caught
ok: runner not started (exit 1)
ok: timeout restored after the late refusal
ok: refuses while charging
ok: no logger or watcher left running
ok: lock released after every run

```
### `python -B benchmark/power_map/test_power_map.py` — exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block lists and CLI setting validation: PASS
5/10/off size sequences and frame pacing at rates 1/2; limit before first read: PASS
separate capped policies, skipped gaps, unknown readings, excluded edges/tail; least-squares skin slope: PASS
stale pre-pin processor excluded for idle workers; active in/out-of-target CPUs verified: PASS
synthetic multi-column report, off cadence, INCOMPLETE cpufreq/missing block, ORT observations: PASS
native/proot refusal, /proc stat/status parser, detector-only caller lifecycle (mock affinity): PASS
fake integration: interruption/resume preserves completed block, resident Gemma, report copy, camera cleanup: PASS
POWERMAP offline checks: PASS

```
### `python -B benchmark/coresidency/test_coresidency.py` — exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block limits: status 3 no stop / 4 stop, battery 44.9 / 45.0, CPU 110 in 3 consecutive (fault), skin+status loss 60 s and CPU/battery loss 5 s fail closed: PASS
  [B1] waiting: skin 36.0 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 34.0 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 1 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 1 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 1 s
cooldown gate: skin <= idle + 1.5 and z9 <= idle + 4 reached; else warm start after the limit: PASS
RobotCam end check: stopped after STOP / still advancing / unreadable reads; force-stop failure INCOMPLETE; cached pid recorded only: PASS
capped time: pre-block and post-block samples excluded, time-weighted, % of block; unreadable scaling_max INCOMPLETE: PASS
layout discovery, 1 s sample and dump parsing: PASS
selector slots = those that start inside the block (smoke 1, full 9): PASS
limit stop: short stop valid, drift n/a, in-flight call allowed; limit at start and fail-closed stop INCOMPLETE: PASS
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmph15nb_pi/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmph15nb_pi/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmph15nb_pi/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmph15nb_pi/.drop_done
cold load: full-cold iff the drop answered ok and no load-file page is cached (mincore): PASS
failed logcat query reported as failed, not as 0 kills: PASS
per-pid PSS failure flagged; late sample dropped; gaps from completion times: PASS
signal during a server load or inside its Popen stops the server; all-failed selector block flagged: PASS
cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE; old-format run refused by --resume: PASS
Co-residency benchmark (DECISIONS #124), run run_20261002T132419Z_smoke  [SMOKE: not a measurement]
block length 8 s; idle z9 90.0 degC, idle skin 35.0 degC; runner sha256 85a6e4cb003b

                                        B1_mix_nollm    B2_only640_nollm        B3_mix_gemma    B4_only640_gemma
frames processed                                  16                  16                  16                   8
failed reads by status                          none                none                none                none
repeat reads (not failures)                        5                   3                   4                   0
detect320 ms median/P95                 10/12 (n 15)       n/a/n/a (n 0)        10/12 (n 15)       n/a/n/a (n 0)
detect640 ms median/P95                  30/30 (n 1)        30/31 (n 16)         31/31 (n 1)         31/32 (n 8)
drift320 first/last 30s                     10 -> 10          n/a -> n/a            10 -> 10          n/a -> n/a
drift640 first/last 30s                     30 -> 30            30 -> 30            31 -> 31          n/a -> n/a
read/decode ms median                       3.5/16.5            3.5/12.6            4.0/13.5            4.0/14.6
frame age s median                              0.07                0.08                0.07                0.15
selector ms median/P95                             -                   -             257/287             275/275
selector calls ok/err/correct                  0/0/0               0/0/0               3/0/1               2/0/1
min MemAvailable MiB                            3397                3360                3380                3377
max swap used MiB                               2145                2145                2145                2145
peak PSS runner MiB                              121                 121                 121                 121
peak PSS llama-server MiB                        n/a                 n/a                 121                 121
peak PSS RobotCam app MiB                         37                  37                  37                  37
peak PSS camera provider MiB                     264                 264                 264                 264
LMK log lines (kill lines)                     3 (2)               3 (2)               3 (2)               3 (2)
survived RobotCam / llama                    yes / -             yes / -           yes / yes           yes / yes
zone9 start/end/max degC              90.0/90.0/90.0      90.0/90.0/90.0      90.0/90.0/90.0      90.0/90.0/90.0
VIRTUAL-SKIN start/end/max degC       35.0/35.0/35.0      35.0/35.0/35.0      35.0/35.0/35.0      35.0/46.6/46.6
Android status max                                 3                   3                   3                   4
policy capped s (% of block)                  0 (0%)              0 (0%)              0 (0%)             2 (49%)
lowest scaling_max MHz 0/4/6          1803/2348/2850      1803/2348/2850      1803/2348/2850      1803/2348/2400
gate wait s                                        0                   0                   0                   0
stop limit                                         -                   -                   -      android_status
time to limit s                                    -                   -                   -                 4.0
mean battery W                                  1.64                1.64                1.64                1.64
battery status                           Discharging         Discharging         Discharging         Discharging
sample errors                                      0                   0                   0                   0
sample gaps >3s (max s)                      0 (2.0)             0 (2.0)             0 (2.0)             0 (2.0)
samples past block end (dropped)                   0                   0                   0                   0

Gemma load (spawn to /health ok): cold 0.90 s [weights-cold (posix_fadvise DONTNEED on the GGUF)], warm 0.74 s; gate wait 0 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmpbebxymfn/a/server.pid

MTP snapshot (RobotCam + both detector sessions + Gemma with drafter, after one request): MemAvailable 3380 MiB, swap used 2145 MiB, PSS llama-server 121 MiB, VmHWM 24 MiB, load 0.78 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmpbebxymfn/a/server.pid --model-draft /data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3

Notes: detect ms = Detector.detect (resize, inference, NMS); read/decode split read_frame at its BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Block limits, checked between frames: Android thermal status >= 4 (CRITICAL), battery >= 45.0 degC, CPU zone >= 110 degC in 3 consecutive 1 s samples (fault stop); time to limit = when the runner saw it (up to one detection plus the 1 s/5 s reading interval late); the block ends there and counts as run; drift is n/a if it ran under 60 s. Skin/status = `dumpsys thermalservice` every 1 s; capped = time (each in-block 1 s sample counts the time since the previous one) with any policy's scaling_max_freq below its cpuinfo_max_freq, % of the block length. Gate: skin <= idle + 1.5 and z9 <= idle + 4 degC; warm start = not reached in 15 min.

full smoke run: PASS
SIGTERM during B3: exit 143, RobotCam stopped, server stopped: PASS
--resume after SIGTERM: PASS
failed STOP broadcast: force-stop still runs: PASS

```
### `bash benchmark/coresidency/test_oneshot.sh` — exit 0

```text
ok: old timeout read through the file
ok: runner gets the arguments
ok: timeout restored and read back
ok: setting is 60000 afterwards
ok: error text is not taken as a timeout
ok: not started
ok: no restore after a failed read
ok: setting untouched
ok: unchanged setting after put is caught
ok: not started
ok: failed restore is reported
ok: failed restore exits non-zero (1)
ok: refuses with a runner running
ok: refuses while another launcher holds the lock
ok: the other run's lock and logger untouched
ok: refused launcher ran no cleanup
ok: refuses with codex running
ok: refuses with llama-server running
ok: runner failure reported
ok: launcher exits with the runner status
ok: timeout restored after a runner failure
ok: SIGTERM during the idle stops the launcher at once (exit 143)
ok: timeout restored after SIGTERM during the idle
ok: logger and watcher told to stop after SIGTERM during the idle
ok: SIGHUP during the idle stops the launcher at once (exit 129)
ok: timeout restored after SIGHUP during the idle
ok: logger and watcher told to stop after SIGHUP during the idle
ok: SIGINT during the idle stops the launcher at once (exit 130)
ok: timeout restored after SIGINT during the idle
ok: logger and watcher told to stop after SIGINT during the idle
ok: SIGTERM to the launcher reaches the runner
ok: launcher waits for the runner after SIGTERM (exit 143)
ok: timeout restored after SIGTERM
ok: SIGHUP to the launcher reaches the runner
ok: launcher waits for the runner after SIGHUP (exit 143)
ok: timeout restored after SIGHUP
ok: charger plugged in during the idle is caught
ok: runner not started (exit 1)
ok: timeout restored after the late refusal
ok: refuses while charging
ok: no logger or watcher left running
ok: lock released after every run

```

### CONFIRM rerun — exit 0, output byte-identical to round 1

```text
CONFIRM source: /data/data/com.termux/files/home/power_map/run_20261002T090821Z_CONFIRM
Capped s uses the POWERMAP later-reading convention; last-sample tail is uncredited.
minute | skin mean/max C | battery mean W | capped s policy0/4/6 | Android status max | selector calls/correct
1 | 27.356/28.251 | 2.328 | 56.283/0.0/0.0 | 0 | 3/3
2 | 28.789/29.213 | 2.277 | 60.0/0.0/0.0 | 0 | 3/3
3 | 29.752/30.144 | 1.984 | 60.0/0.0/0.0 | 0 | 3/3
4 | 30.512/30.836 | 2.658 | 60.0/0.0/0.0 | 0 | 3/3
5 | 31.263/31.615 | 2.708 | 60.0/0.0/0.0 | 0 | 3/2
6 | 31.937/32.214 | 2.297 | 60.0/0.0/0.0 | 0 | 3/2
7 | 32.505/32.748 | 3.472 | 60.0/0.0/0.0 | 0 | 3/2
8 | 33.009/33.244 | 3.587 | 60.0/0.0/0.0 | 0 | 3/2
9 | 33.558/33.801 | 3.559 | 60.0/0.997/0.997 | 0 | 3/1
10 | 34.015/34.211 | 3.144 | 60.0/0.0/0.0 | 0 | 3/2
11 | 34.502/34.809 | 2.546 | 60.0/1.006/1.006 | 0 | 3/3
12 | 34.936/35.099 | 2.061 | 60.0/0.0/0.0 | 0 | 3/2
13 | 35.29/35.455 | 2.752 | 60.0/0.0/0.0 | 0 | 3/3
14 | 35.644/35.906 | 2.103 | 60.0/0.0/0.0 | 0 | 3/2
15 | 36.08/36.221 | 2.979 | 60.0/0.0/0.0 | 0 | 3/2
16 | 36.398/36.626 | 1.984 | 60.0/1.002/1.002 | 0 | 3/2
17 | 36.955/37.174 | 3.272 | 60.0/0.0/7.287 | 0 | 3/3
18 | 37.2/37.357 | 3.995 | 60.0/9.006/26.713 | 0 | 3/3
19 | 37.451/37.652 | 3.65 | 60.0/14.004/31.295 | 0 | 3/1
20 | 37.745/37.912 | 3.366 | 60.0/14.0/26.697 | 0 | 3/2
21 | not recorded/not recorded | 3.24 | 0.705/0.0/0.0 | not recorded | 0/0
policy4 first recorded cap s: 520.705
policy6 first recorded cap s: 520.705
Selector calls not recorded correct: time s | case id | expected | returned | ms
280.0 | L2_03 | not recorded | go to the next unsearched room | 1872.4
300.0 | L2_04 | not recorded | search this room | 1922.2
360.0 | L2_07 | not recorded | go to the next unsearched room | 1841.5
440.0 | L2_11 | not recorded | go to the next unsearched room | 1846.1
480.0 | L3_01 | not recorded | go to the kitchen | 2316.9
500.0 | L3_02 | not recorded | go to the bedroom | 1860.3
580.0 | L3_06 | not recorded | go to the kitchen | 1779.6
700.0 | L3_12 | not recorded | go to the bedroom | 2035.8
820.0 | L4_06 | not recorded | go to the bedroom | 2459.1
860.0 | L4_08 | not recorded | go to the bedroom | 1973.2
940.0 | L4_12 | not recorded | go to the kitchen | 2203.4
1080.0 | L5_07 | not recorded | go to the living room | 3374.7
1120.0 | L5_09 | not recorded | search this room again | 3836.1
1140.0 | L5_10 | not recorded | go to the living room | 3624.5

```

## Round 2 independent review

Fresh session completed using the designated model, effort and read-only invocation; nested command exited 0. Full request: `/termux-home/duty1_review_request_2.md`; stdout `/termux-home/duty1_review_2.stdout.txt`; separate diagnostics `/termux-home/duty1_review_2.stderr.txt`. The reviewer reported the known bubblewrap unsupported-host-mount error and completed its verdict from full supplied source/context/output. Tests were not independently rerun. Under the latest human decision this counts as a completed review, not approval. Candidate edits remained frozen.

### Round 2 — verbatim reviewer stdout

1. **[P1] Preserve termination signals during exceptional selector draining** — [duty_cycle.py:306](/termux-home/robot/benchmark/duty_cycle/duty_cycle.py:306). Once `error` contains `CoresLost`, subsequent `SystemExit(143)` from SIGTERM is discarded by `error = error or e`, including in the final join loop. The original `CoresLost` then reaches the retry handler, which restarts the block despite the human’s stop request. Preserve termination exceptions while still cancelling/draining and joining the selector before exit. Add an offline test combining core loss, a pending selector, and SIGTERM; assert the selector finishes and no retry starts.

The rate‑1 pacing fix matches POWERMAP. Selector cleanup now joins before retry on the tested core-loss path and retains drain evidence.

HEAD, status, both POWERMAP hashes, and both changed DUTY1 hashes matched. Further inspection encountered the documented bubblewrap unsupported-host-mount error. Review completed from the supplied full candidate, context, and test output; tests were not independently rerun. No edits or hardware execution occurred.

**REQUEST CHANGES**


## Historical open finding after round 2

**Blocking P1: termination priority during exceptional draining.** If `error` already contains `CoresLost`, the new cleanup's `error = error or e` can discard a later `SystemExit(143)` from SIGTERM, then propagate core loss to the retry handler. The pending selector is joined, but the runner may retry despite the human stop request. Needs termination exceptions preserved while joining, plus a combined core-loss/pending-call/SIGTERM regression proving no retry starts. Separate core-loss and signal checks passed; the combined case was not tested. Coder stopped after round 2 as instructed. A later human authorization below permits the termination-priority fix and third/final review.

Both review mount errors are recorded above. No no-verdict authentication/quota/permission failure occurred. All phone-only checks listed in Unverified items remain unverified. Candidate is **not approved for hardware measurement or commit**.

## Final state after round 2

```text
 M .gitignore
?? benchmark/duty_cycle/
?? benchmark/power_map/
```

At the round-2 stop, all DUTY1 file hashes matched its frozen request. HEAD remains `7c77f1439544634ede8b8baf92aa03e3b35b24b6`. POWERMAP source and launcher hashes still match the original task. Nothing staged, committed or pushed. Stop and wait for the human.

## Round 3 fix and authorization

Only `duty_cycle.py` and `test_duty_cycle.py` changed since round 2. Runner change: at all three exceptional-drain catches (bounded join, server cancellation, final join), `SystemExit` or `KeyboardInterrupt` replaces a retained ordinary error. Later ordinary errors preserve the retained termination exception. The existing drain/cancel/join and error propagation structure remains. Termination exits therefore reach `main` cleanup instead of the CoresLost retry handler.

The combined mocked main integration uses an actual SIGTERM sent to its own offline test process after a pending call encounters core loss. It verifies the selector call finished, exit 143 propagated, no retry started, no interrupted block was saved, drain evidence says `SystemExit: 143`, and mocked RobotCam/llama-server stopped. Final-join tests also retain TERM/HUP/INT SystemExit and KeyboardInterrupt over CoresLost. The new combined regression was run against the unmodified round-2 drain in an isolated Python process and rejected it as expected. All Android/root/camera/server dependencies stayed mocked; no real hardware run.

Latest human authorization, verbatim:

```text
Local AI: round 2 finding is valid. Fix it, nothing else:
During exceptional selector draining, a termination exception (SystemExit from SIGTERM/SIGHUP/SIGINT, KeyboardInterrupt) must always take priority over any retained error such as CoresLost, including in the final join loop. Still drain/cancel and join the selector before propagating. Add an offline test combining core loss, a pending selector call and SIGTERM: assert the selector call finishes, the termination exit propagates (exit 143), no retry starts, and RobotCam and llama-server are stopped.

Rerun all tests and confirm_view.py, then review round 3 in a fresh session (the last allowed round). Update DUTY1_CODER_REPORT.md in Downloads with the fix, new SHA-256 values, test output and round 3 verbatim. If round 3 still requests changes, do not fix further; report and stop. Then stop.

```

## Round 3 verification

All five required offline commands below exited 0. No real su/am/launcher/RobotCam/llama-server execution. The combined regression sends SIGTERM only to its own mocked offline runner/test process. Bash syntax and `git diff --check` passed; existing POWERMAP hashes unchanged. CONFIRM output remains byte-identical to rounds 1 and 2.

### `python -B benchmark/duty_cycle/test_duty_cycle.py` — exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
PARTS/CYCLE/smoke block lists and active-only 20 s selector clock: PASS
restart passes request CLOCK_BOOTTIME; actual reader rejects pre-restart capture: PASS
energy above baseline and per-call interpolation/missing coverage; proc stat parser: PASS
CAM/YOLO/YOLO_NOSPIN/Active rate 1: exactly one read and JPEG decode per second, work included in period: PASS
LOAD spawn/health/call CPU split, GGUF cold/warm, stop on interruption, worker joined: PASS
[CONT] selector drain 0.052s; RuntimeError: failed limit query
[CONT] selector drain 0.054s; SystemExit: 143
[CONT] selector drain 0.051s; CoresLost: body core loss
[CONT] selector drain 122.000s; RuntimeError: selector drain exceeded 120 s; request cancelled
[CONT] selector drain 122.000s; SystemExit: 143
[CONT] selector drain 122.000s; SystemExit: 129
[CONT] selector drain 122.000s; SystemExit: 130
[CONT] selector drain 122.000s; KeyboardInterrupt: 
SystemExit TERM/HUP/INT and KeyboardInterrupt override retained CoresLost in final join, after joining: PASS
selector drain: arbitrary error, signal, unwinding CoresLost and timeout cancellation all join before propagation: PASS
both ORT spinning options retained; rejection produces INCOMPLETE without fallback: PASS
synthetic report/invalid battery; atomic resume preserves finished blocks and redoes HEATCOOL; proot refusal: PASS
core loss during pending selector drain: request finished before retry; discarded-block drain time recorded: PASS
mocked main + real block loops: interrupt/resume preserves CONT, redoes HEATCOOL, no COOL gate, final cleanup: PASS
pending selector + core loss + actual SIGTERM: call finished, exit 143, no retry, RobotCam/server stopped: PASS
DUTY1 offline checks: PASS

```
### `bash benchmark/duty_cycle/test_oneshot.sh` — exit 0

```text
ok: old timeout read through the file
ok: runner gets the arguments
ok: timeout restored and read back
ok: setting is 60000 afterwards
ok: error text is not taken as a timeout
ok: not started
ok: no restore after a failed read
ok: setting untouched
ok: unchanged setting after put is caught
ok: not started
ok: failed restore is reported
ok: failed restore exits non-zero (1)
ok: refuses with a runner running
ok: refuses while another launcher holds the lock
ok: the other run's lock and logger untouched
ok: refused launcher ran no cleanup
ok: refuses with codex running
ok: refuses with llama-server running
ok: runner failure reported
ok: launcher exits with the runner status
ok: timeout restored after a runner failure
ok: SIGTERM during the idle stops the launcher at once (exit 143)
ok: timeout restored after SIGTERM during the idle
ok: logger and watcher told to stop after SIGTERM during the idle
ok: SIGHUP during the idle stops the launcher at once (exit 129)
ok: timeout restored after SIGHUP during the idle
ok: logger and watcher told to stop after SIGHUP during the idle
ok: SIGINT during the idle stops the launcher at once (exit 130)
ok: timeout restored after SIGINT during the idle
ok: logger and watcher told to stop after SIGINT during the idle
ok: SIGTERM to the launcher reaches the runner
ok: launcher waits for the runner after SIGTERM (exit 143)
ok: timeout restored after SIGTERM
ok: SIGHUP to the launcher reaches the runner
ok: launcher waits for the runner after SIGHUP (exit 143)
ok: timeout restored after SIGHUP
ok: charger plugged in during the idle is caught
ok: runner not started (exit 1)
ok: timeout restored after the late refusal
ok: refuses while charging
ok: no logger or watcher left running
ok: lock released after every run

```
### `python -B benchmark/power_map/test_power_map.py` — exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block lists and CLI setting validation: PASS
5/10/off size sequences and frame pacing at rates 1/2; limit before first read: PASS
separate capped policies, skipped gaps, unknown readings, excluded edges/tail; least-squares skin slope: PASS
stale pre-pin processor excluded for idle workers; active in/out-of-target CPUs verified: PASS
synthetic multi-column report, off cadence, INCOMPLETE cpufreq/missing block, ORT observations: PASS
native/proot refusal, /proc stat/status parser, detector-only caller lifecycle (mock affinity): PASS
fake integration: interruption/resume preserves completed block, resident Gemma, report copy, camera cleanup: PASS
POWERMAP offline checks: PASS

```
### `python -B benchmark/coresidency/test_coresidency.py` — exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
block limits: status 3 no stop / 4 stop, battery 44.9 / 45.0, CPU 110 in 3 consecutive (fault), skin+status loss 60 s and CPU/battery loss 5 s fail closed: PASS
  [B1] waiting: skin 36.0 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 34.0 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.6 degC, need <= 33.5; z9 33.0 degC, need <= 34.0; 1 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin 33.0 degC, need <= 33.5; z9 34.0 degC, need <= 34.0; 1 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 0 s
  [B1] waiting: skin None degC, need <= 33.5; z9 30.0 degC, need <= 34.0; 1 s
cooldown gate: skin <= idle + 1.5 and z9 <= idle + 4 reached; else warm start after the limit: PASS
RobotCam end check: stopped after STOP / still advancing / unreadable reads; force-stop failure INCOMPLETE; cached pid recorded only: PASS
capped time: pre-block and post-block samples excluded, time-weighted, % of block; unreadable scaling_max INCOMPLETE: PASS
layout discovery, 1 s sample and dump parsing: PASS
selector slots = those that start inside the block (smoke 1, full 9): PASS
limit stop: short stop valid, drift n/a, in-flight call allowed; limit at start and fail-closed stop INCOMPLETE: PASS
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpt6yyts2p/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpt6yyts2p/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpt6yyts2p/.drop_done
[loads] cold load next: handshake, waiting up to 5 s for /data/data/com.termux/files/usr/tmp/tmpt6yyts2p/.drop_done
cold load: full-cold iff the drop answered ok and no load-file page is cached (mincore): PASS
failed logcat query reported as failed, not as 0 kills: PASS
per-pid PSS failure flagged; late sample dropped; gaps from completion times: PASS
signal during a server load or inside its Popen stops the server; all-failed selector block flagged: PASS
cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE; old-format run refused by --resume: PASS
Co-residency benchmark (DECISIONS #124), run run_20261002T160828Z_smoke  [SMOKE: not a measurement]
block length 8 s; idle z9 90.0 degC, idle skin 35.0 degC; runner sha256 85a6e4cb003b

                                        B1_mix_nollm    B2_only640_nollm        B3_mix_gemma    B4_only640_gemma
frames processed                                  16                  16                  16                   8
failed reads by status                          none                none                none                none
repeat reads (not failures)                        4                   5                   4                   0
detect320 ms median/P95                 10/11 (n 15)       n/a/n/a (n 0)        10/11 (n 15)       n/a/n/a (n 0)
detect640 ms median/P95                  30/30 (n 1)        30/32 (n 16)         30/30 (n 1)         30/31 (n 8)
drift320 first/last 30s                     10 -> 10          n/a -> n/a            10 -> 10          n/a -> n/a
drift640 first/last 30s                     30 -> 30            30 -> 30            30 -> 30          n/a -> n/a
read/decode ms median                       3.8/18.8            2.7/13.1            3.8/15.7            5.1/17.8
frame age s median                              0.08                0.06                0.07                0.12
selector ms median/P95                             -                   -             265/270             275/294
selector calls ok/err/correct                  0/0/0               0/0/0               3/0/1               2/0/1
min MemAvailable MiB                            3278                3256                3256                3258
max swap used MiB                               2240                2234                2234                2234
peak PSS runner MiB                              121                 121                 121                 121
peak PSS llama-server MiB                        n/a                 n/a                 121                 121
peak PSS RobotCam app MiB                         37                  37                  37                  37
peak PSS camera provider MiB                     264                 264                 264                 264
LMK log lines (kill lines)                     3 (2)               3 (2)               3 (2)               3 (2)
survived RobotCam / llama                    yes / -             yes / -           yes / yes           yes / yes
zone9 start/end/max degC              90.0/90.0/90.0      90.0/90.0/90.0      90.0/90.0/90.0      90.0/90.0/90.0
VIRTUAL-SKIN start/end/max degC       35.0/35.0/35.0      35.0/35.0/35.0      35.0/35.0/35.0      35.0/46.6/46.6
Android status max                                 3                   3                   3                   4
policy capped s (% of block)                  0 (0%)              0 (0%)              0 (0%)             2 (50%)
lowest scaling_max MHz 0/4/6          1803/2348/2850      1803/2348/2850      1803/2348/2850      1803/2348/2400
gate wait s                                        0                   0                   0                   0
stop limit                                         -                   -                   -      android_status
time to limit s                                    -                   -                   -                 4.0
mean battery W                                  1.64                1.64                1.64                1.64
battery status                           Discharging         Discharging         Discharging         Discharging
sample errors                                      0                   0                   0                   0
sample gaps >3s (max s)                      0 (2.0)             0 (2.0)             0 (2.0)             0 (2.0)
samples past block end (dropped)                   0                   0                   0                   0

Gemma load (spawn to /health ok): cold 0.98 s [weights-cold (posix_fadvise DONTNEED on the GGUF)], warm 0.97 s; gate wait 0 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmpd5pe2ych/a/server.pid

MTP snapshot (RobotCam + both detector sessions + Gemma with drafter, after one request): MemAvailable 3265 MiB, swap used 2241 MiB, PSS llama-server 121 MiB, VmHWM 24 MiB, load 0.97 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmpd5pe2ych/a/server.pid --model-draft /data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3

Notes: detect ms = Detector.detect (resize, inference, NMS); read/decode split read_frame at its BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", or killinfo. Every sample runs one su call (dumpsys) in all blocks alike. Block limits, checked between frames: Android thermal status >= 4 (CRITICAL), battery >= 45.0 degC, CPU zone >= 110 degC in 3 consecutive 1 s samples (fault stop); time to limit = when the runner saw it (up to one detection plus the 1 s/5 s reading interval late); the block ends there and counts as run; drift is n/a if it ran under 60 s. Skin/status = `dumpsys thermalservice` every 1 s; capped = time (each in-block 1 s sample counts the time since the previous one) with any policy's scaling_max_freq below its cpuinfo_max_freq, % of the block length. Gate: skin <= idle + 1.5 and z9 <= idle + 4 degC; warm start = not reached in 15 min.

full smoke run: PASS
SIGTERM during B3: exit 143, RobotCam stopped, server stopped: PASS
--resume after SIGTERM: PASS
failed STOP broadcast: force-stop still runs: PASS

```
### `bash benchmark/coresidency/test_oneshot.sh` — exit 0

```text
ok: old timeout read through the file
ok: runner gets the arguments
ok: timeout restored and read back
ok: setting is 60000 afterwards
ok: error text is not taken as a timeout
ok: not started
ok: no restore after a failed read
ok: setting untouched
ok: unchanged setting after put is caught
ok: not started
ok: failed restore is reported
ok: failed restore exits non-zero (1)
ok: refuses with a runner running
ok: refuses while another launcher holds the lock
ok: the other run's lock and logger untouched
ok: refused launcher ran no cleanup
ok: refuses with codex running
ok: refuses with llama-server running
ok: runner failure reported
ok: launcher exits with the runner status
ok: timeout restored after a runner failure
ok: SIGTERM during the idle stops the launcher at once (exit 143)
ok: timeout restored after SIGTERM during the idle
ok: logger and watcher told to stop after SIGTERM during the idle
ok: SIGHUP during the idle stops the launcher at once (exit 129)
ok: timeout restored after SIGHUP during the idle
ok: logger and watcher told to stop after SIGHUP during the idle
ok: SIGINT during the idle stops the launcher at once (exit 130)
ok: timeout restored after SIGINT during the idle
ok: logger and watcher told to stop after SIGINT during the idle
ok: SIGTERM to the launcher reaches the runner
ok: launcher waits for the runner after SIGTERM (exit 143)
ok: timeout restored after SIGTERM
ok: SIGHUP to the launcher reaches the runner
ok: launcher waits for the runner after SIGHUP (exit 143)
ok: timeout restored after SIGHUP
ok: charger plugged in during the idle is caught
ok: runner not started (exit 1)
ok: timeout restored after the late refusal
ok: refuses while charging
ok: no logger or watcher left running
ok: lock released after every run

```

### Regression against unchanged round-2 drain — expected rejection, experiment exit 0

```text
/data/data/com.termux/files/usr/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_validation.py:66: UserWarning: Unsupported platform (android). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only.
  warnings.warn(
Combined core-loss/pending-call/SIGTERM regression rejects the unmodified round-2 drain: PASS (expected failure)

```

### CONFIRM analysis rerun — exit 0

```text
CONFIRM source: /data/data/com.termux/files/home/power_map/run_20261002T090821Z_CONFIRM
Capped s uses the POWERMAP later-reading convention; last-sample tail is uncredited.
minute | skin mean/max C | battery mean W | capped s policy0/4/6 | Android status max | selector calls/correct
1 | 27.356/28.251 | 2.328 | 56.283/0.0/0.0 | 0 | 3/3
2 | 28.789/29.213 | 2.277 | 60.0/0.0/0.0 | 0 | 3/3
3 | 29.752/30.144 | 1.984 | 60.0/0.0/0.0 | 0 | 3/3
4 | 30.512/30.836 | 2.658 | 60.0/0.0/0.0 | 0 | 3/3
5 | 31.263/31.615 | 2.708 | 60.0/0.0/0.0 | 0 | 3/2
6 | 31.937/32.214 | 2.297 | 60.0/0.0/0.0 | 0 | 3/2
7 | 32.505/32.748 | 3.472 | 60.0/0.0/0.0 | 0 | 3/2
8 | 33.009/33.244 | 3.587 | 60.0/0.0/0.0 | 0 | 3/2
9 | 33.558/33.801 | 3.559 | 60.0/0.997/0.997 | 0 | 3/1
10 | 34.015/34.211 | 3.144 | 60.0/0.0/0.0 | 0 | 3/2
11 | 34.502/34.809 | 2.546 | 60.0/1.006/1.006 | 0 | 3/3
12 | 34.936/35.099 | 2.061 | 60.0/0.0/0.0 | 0 | 3/2
13 | 35.29/35.455 | 2.752 | 60.0/0.0/0.0 | 0 | 3/3
14 | 35.644/35.906 | 2.103 | 60.0/0.0/0.0 | 0 | 3/2
15 | 36.08/36.221 | 2.979 | 60.0/0.0/0.0 | 0 | 3/2
16 | 36.398/36.626 | 1.984 | 60.0/1.002/1.002 | 0 | 3/2
17 | 36.955/37.174 | 3.272 | 60.0/0.0/7.287 | 0 | 3/3
18 | 37.2/37.357 | 3.995 | 60.0/9.006/26.713 | 0 | 3/3
19 | 37.451/37.652 | 3.65 | 60.0/14.004/31.295 | 0 | 3/1
20 | 37.745/37.912 | 3.366 | 60.0/14.0/26.697 | 0 | 3/2
21 | not recorded/not recorded | 3.24 | 0.705/0.0/0.0 | not recorded | 0/0
policy4 first recorded cap s: 520.705
policy6 first recorded cap s: 520.705
Selector calls not recorded correct: time s | case id | expected | returned | ms
280.0 | L2_03 | not recorded | go to the next unsearched room | 1872.4
300.0 | L2_04 | not recorded | search this room | 1922.2
360.0 | L2_07 | not recorded | go to the next unsearched room | 1841.5
440.0 | L2_11 | not recorded | go to the next unsearched room | 1846.1
480.0 | L3_01 | not recorded | go to the kitchen | 2316.9
500.0 | L3_02 | not recorded | go to the bedroom | 1860.3
580.0 | L3_06 | not recorded | go to the kitchen | 1779.6
700.0 | L3_12 | not recorded | go to the bedroom | 2035.8
820.0 | L4_06 | not recorded | go to the bedroom | 2459.1
860.0 | L4_08 | not recorded | go to the bedroom | 1973.2
940.0 | L4_12 | not recorded | go to the kitchen | 2203.4
1080.0 | L5_07 | not recorded | go to the living room | 3374.7
1120.0 | L5_09 | not recorded | search this room again | 3836.1
1140.0 | L5_10 | not recorded | go to the living room | 3624.5

```

## Round 3 independent review

Fresh session completed using designated model `gpt-6.1-sol`, effort `medium`, read-only, Ponytail off; nested `codex exec` exited 0. Request: `/termux-home/duty1_review_request_3.md`; stdout: `/termux-home/duty1_review_3.stdout.txt`; separate diagnostics: `/termux-home/duty1_review_3.stderr.txt`. The reviewer reported the same bubblewrap unsupported-host-mount error and completed its verdict from full supplied source/context/output under the human's rule. Tests were not independently rerun. No candidate changes during or after round 3.

### Round 3 — verbatim reviewer stdout

No blocking findings. The three assignments preserve termination exceptions over retained `CoresLost` while draining and joining the selector before propagation. The supplied regressions cover actual SIGTERM, exit 143, no retry, camera/server cleanup, and termination priority in the final join loop.

HEAD, status, both POWERMAP hashes, and both changed DUTY1 hashes matched the supplied candidate.

Further reads failed with: `error building bubblewrap command: app-server socket directory has an unsupported host mount` at `/tmp/codex-daemon-0/171b54ad83a225128b102d885c4b6b2fed0aa58118290e0d35546cd9d70163f5`. Review completed from the supplied full source, context, and verification output under the human’s stated rule. Tests were not independently rerun.

Real camera restarts, Android ORT spinning options, fuel-gauge averaging, and phone LOAD cache states remain unverified. No edits or hardware execution occurred.

**APPROVE WITH NOTES**


## Final findings and limitations after round 3

**No open blocking findings. APPROVE WITH NOTES.** Independent approval is not permission to stage, commit, push or run hardware. All historical REQUEST CHANGES findings above are retained verbatim with their fix evidence.

Known review environment limitation: all rounds encountered the recorded sandbox/bubblewrap mount error; the reviewers returned verdicts based on full supplied candidate/context/output. Under the human's explicit rule these are completed reviews. None independently reran tests. Coder reran every required offline test and analysis, with full output above.

Still unverified: real camera restart behaviour; spinning-option acceptance/effect on Android ORT; fuel-gauge averaging/update interval; GGUF LOAD warm/cold behaviour on the phone; real MID pinning/CPU traces, root-query timing, heat/cap behaviour and process CPU accounting. Smoke/hardware measurements remain human-only. No robot/motor files or prior benchmark source were edited.

## Final frozen state after round 3

```text
 M .gitignore
?? benchmark/duty_cycle/
?? benchmark/power_map/
```

Every DUTY1 source hash matches the frozen round-3 request and current hash list above. HEAD remains `7c77f1439544634ede8b8baf92aa03e3b35b24b6`.
Unchanged `benchmark/power_map/power_map.py` SHA-256 `4756ce7dc7ba4e358578288c0d2509345fc4e2fe02d45f077f65355792e29539`.
Unchanged `benchmark/power_map/oneshot.sh` SHA-256 `874dfc850c47becfe5226dd0c184e5ffcb9b8a474b6848259ed193a27ebc49dd`.

Nothing staged, committed or pushed. No hardware run. Stop and wait for the human.
