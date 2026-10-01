# CORESIDENCY — Coder report

**Role:** Coder. Claude Code CLI, `claude-opus-5-5`, medium effort, Ponytail lite.
**Reviewer:** Codex CLI 0.158.0, `gpt-6-sol`, medium effort, read-only sandbox, Ponytail off. A fresh `codex exec` session ran for each of 20 rounds.
**Base:** `main` at `3584b5eddd3639c00c37b24e6e31f4d86fce2f2b`. I verified HEAD at the start and at the end. The tracked tree is clean: no tracked file was modified, and nothing was staged, committed or pushed.
**Final verdict (round 20):** **APPROVE WITH NOTES** (one MINOR, left open; see below). A reviewer pass is not a commit decision: the work waits for the human.

## Summary

Six new files are in `benchmark/coresidency/`:
- `oneshot.sh`: the launcher.
- `run_coresidency.sh`: a thin run script.
- `coresidency.py`: the runner.
- `test_coresidency.py` and `test_oneshot.sh`: the offline tests.
- `SMOKE.md`: the native commands for the human's smoke run.

Nothing ran on the phone natively (see *Unverified*).

**The new folder is gitignored.** `.gitignore` line 8 (`benchmark/*`) ignores everything under `benchmark/` except `strategic_selector/` and `llm_objective_setting/`. `benchmark/coresidency/` therefore shows as ignored, like `benchmark/camera_heat/`. To commit it, the human or Doc Keeper needs a `.gitignore` exception or `git add -f`. I left `.gitignore` untouched, because it is a tracked file.

## Key design choices and assumptions (for Local AI)

- **Native Termux.** Both scripts and the runner refuse to run from Debian/proot: they require `PREFIX=/data/data/com.termux/files/usr` and no `/termux-home`. Nothing forced Debian.
- **Imports.** The runner imports `robotcam_reader`, `detector_size_policy.SizePolicy`, `detect_person.Detector` and `server_manager` (for the binary and thread flags only). It never imports `motors.py` and never opens USB/serial.
- **Code copied from `ladder.py`.** `ladder.py` cannot be imported natively, because its `jevlike` import path is `/termux-home/...`. The thermal gate (z9 ≤ idle + 4 °C), the cold-load handshake and the cpuset redo loop are therefore copied from it, with provenance comments. `allowed_cpus` is imported from `ladder/measure.py`.
- **Server flags.** llama-server uses `server_manager.start_server`'s exact `setup_q4` binary, model and flags, in the same order, and `run.json` and `loads.json` record the command. The ladder's `s1o_b1609dp_q40` run added `--flash-attn off --ctx-checkpoints 0 --batch-size 512 --ubatch-size 512`. These flags are not used here, so selector ms is **not directly comparable** to #119's 1408 ms. There is no `taskset`, because the robot does not pin the server.
- **Selector.** It uses `adapters.S1O` itself: `decide`, `prompt_ids` and `completion_probs` are unchanged. It is pointed at the resident server via `S1O.__new__`, because `S1O.__init__` would spawn its own server on port 8091; the tail of its `__init__` is repeated. Cases come from `ladder_cases_v1.jsonl`, with list options turned into `{o: ''}` as in `ladder.load_cases`. B3 and B4 each use cases 0–8, one call per 20 s slot, in its own thread. Each call records ms, the letter, the choice, correctness and the actual start and end times.
- **Gemma loads.** Load time is measured from spawn to the first `/health` returning `ok`. The cold load follows the handshake page-cache drop, checked in `/proc/meminfo` with a `posix_fadvise` fallback that is recorded. The warm load is an immediate restart; that server stays resident for B3 and B4, and each start gets one untimed warm-up call.
- **Read/decode split.** `robotcam_reader.read_frame` is used unchanged. To split its time into read and decode, the runner replaces `robotcam_reader.io` in-process with a shim whose `BytesIO` stamps the time. No file changes.
- **Frame pacing.** After a new frame, the next read is 0.5 s after that frame was first seen. A repeat retries after 20 ms. Every read attempt is logged with its status.
- **Every 5 s:** one `su` call in the #123 form, `su -c "{ …; } </dev/null >/data/local/tmp/coresidency_sample.txt 2>&1; …; cat …"`, reads the battery `current_now`, `voltage_now` and `status`. The same call runs `dumpsys meminfo </dev/null` for the runner pid, the llama-server pid, `pidof com.pixelrobot.robotcam` and each `android.hardware.camera.provider` pid.
  - **Power** = −(current_now µA × voltage_now µV) / 10¹², the same as `benchmark/camera_heat/camera_heat.py`.
  - **PSS parsing** is `camera_heat.py`'s regex.
  - **MemAvailable and swap** come from `/proc/meminfo`, and z9/10/11 from `oneshot`'s thermal log.
- **LMK.** At block end the runner runs `logcat -d -b all -v epoch -T <block start>` in the #123 form and records `logcat`'s own exit code. Lines matching `lowmemorykiller|lmkd|killinfo` are kept. Kills are lowmemorykiller/lmkd lines containing "kill", or killinfo lines.
- **B5.** RobotCam is running and both detector sessions are resident. Gemma starts with `--model-draft ~/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3`. The runner sends one `/completion` (n_predict 32), runs one 640 detection, takes one sample and records VmHWM, then stops.
- **Output.** Results go to `~/coresidency/run_<UTC>[_smoke]/`: `run.json` with hashes, `block_<name>.json`, `loads.json`, `report.txt` and the llama-server logs. The report is copied as `~/storage/downloads/coresidency_<run>_report.txt`, a unique name.
- **INCOMPLETE.** The runner exits non-zero and puts `INCOMPLETE:` lines in the report when:
  - a Gemma block misses its selector cadence;
  - a block has a sample error;
  - a block has a battery status other than `Discharging`;
  - a block has any failed RobotCam read, or ends without a new frame or without the RobotCam process;
  - a size is missing its detections or its drift windows;
  - llama-server did not survive a Gemma block;
  - the LMK query failed;
  - the B5 sample has errors.

  The block JSON is kept as evidence either way.
- **Launcher.** `oneshot.sh` is based on the live `~/ladder/oneshot.sh`.
  - It runs natively, with no proot.
  - The screen timeout is read, set and restored in the #123 form. The old value must be numeric, the set value is read back before starting, and the restored value is read back and compared.
  - It refuses to start when an agent (`claude|agy|node|codex`), llama-server, `chat.py` or another co-residency runner is running.
  - It holds an atomic `mkdir` lock. It checks the battery at start and again after the idle.
  - It keeps the thermal logger, the 5-minute idle, the cache-drop handshake and the cleanup trap.
  - It forwards TERM/HUP/INT to the runner and waits for its cleanup; it also handles signals during the idle and in the launch window.
  - It waits for the logger and watcher to exit before releasing the lock, and exits with the runner's status.
- **Runner signal handling.** On exit, SIGTERM or SIGHUP the runner stops RobotCam and every llama-server it started. A `LIVE` registry closes the windows between spawning a server and its owner holding it. `sys.dont_write_bytecode` stops it writing `.pyc` files into the ladder and v3 archive folders.

Ponytail lite, lazier alternative, stated once: most of the size of `coresidency.py` (~43 KB) and `oneshot.sh` comes from 19 rounds of review-driven failure handling. A version that trusts every step would be about a third of the size, but it would report incomplete or failed runs as valid measurements.

## Incidents during the session (no effect on the repo or phone state)

- The first offline test run wrote `benchmark/strategic_selector/ladder/__pycache__/measure.cpython-313.pyc`, a new gitignored file inside the ladder archive. I deleted it and added `sys.dont_write_bytecode`. A later check found no `.pyc` newer than the candidate outside my folder.
- A `pkill -f 'find / -xdev'` that I ran matched my own tool shell and killed it (COMMANDS §8). It killed only that shell.
- A heredoc quoting mistake ran a few stray lines in Debian. Two tried to write the real `/sys/class/power_supply/battery/status` and got "Permission denied"; the others called undefined functions. No file changed, and `git status` stayed clean.
- Some of the reviewer's own sandboxed shell commands failed with `exited 182` in 0 ms, in every round: 3–10 per round, alongside 2–13 that succeeded. Examples are `sed` on tracked files and `ls` or `sha256sum` of the ignored folder. The reviewer states each time that it relied on the complete numbered file contents and check output supplied in the request, which is what WORKFLOW requires the request to contain. It also verified HEAD, the clean tree and some hashes directly. There were no authentication, quota or model errors. **I report this as a partial sandbox limitation, not hidden. Whether it counts as a complete review is the human's call.**

## Files (final, frozen for round 20)

| File | Bytes | SHA-256 |
|---|---:|---|
| `benchmark/coresidency/SMOKE.md` | 2882 | `3e3cd4e21f476d32e5784ebf51b65b9076cb5cd1b6b233688033d3044cdbabae` |
| `benchmark/coresidency/coresidency.py` | 43267 | `ef622037a659a38e98f7372b5008f399bb22cc49bf8c5979a1f985b9dbd5784e` |
| `benchmark/coresidency/oneshot.sh` | 7351 | `efb3939d2a0a24d7e7a7abc2384f6e608b252ff12b2ca539f8bf8753146d41a2` |
| `benchmark/coresidency/run_coresidency.sh` | 426 | `dba86f4de4011a2bf6abd5745f3ac709633e717e5a72ff31a721dae0470962b0` |
| `benchmark/coresidency/test_coresidency.py` | 20114 | `e076915d58572cab6644fda9c0eba1c06dbd223f808b36f9a4aab8d3dad2c0d4` |
| `benchmark/coresidency/test_oneshot.sh` | 8056 | `bccd170dfe71bccdb316aaf86b555defed3bf528dfaef9b854d231c7289f4e19` |

The hashes on disk equal `review20/frozen.sha`, which is what round 20 reviewed: **True**.

`git rev-parse HEAD`, then `git status --short` (empty means no tracked changes):
```
3584b5eddd3639c00c37b24e6e31f4d86fce2f2b
(end)
```

## Live `~/ladder/oneshot.sh` vs repo `benchmark/strategic_selector/ladder/oneshot.sh`

The live copy (`db1f3156…`) is byte-identical to the archived `oneshot_executed_conversation_run.sh` and is **older** than the repo's final `oneshot.sh` (`f3d531fc…`, "hardened", never executed). The live copy reads the timeout with a bare `su -c 'settings get …'` (no `</dev/null`, no file). It sets it unconditionally, and on exit restores `${OLD_TIMEOUT:-60000}` without validation. That is the #123 failure mode: error text taken as the value. The repo copy strips non-digits (`tr -dc 0-9`), so error text containing "2147483646" would still pass as a number. It also moves the `RUN_SCRIPT` default and adds a `SCREEN_SET` guard. Neither copy uses the #123 file form. The new launcher starts from the live copy, as instructed.

```diff
f3d531fcbc306c8ef5f6e4d6504c91edcaeddc6e66313673dc839dd57e1b7d31  benchmark/strategic_selector/ladder/oneshot.sh
db1f315681892f0c1f89de92b86d7495a47cc171a3c03a170fc51d2f8dbdc8e8  /termux-home/ladder/oneshot.sh
5d4
< #   RUN_SCRIPT=run_conversation.sh bash ~/ladder/oneshot.sh   (other benchmark script in ~/ladder)
11a11
> RUN_SCRIPT=${RUN_SCRIPT:-run_s1o_speed.sh}  # e.g. RUN_SCRIPT=run_conversation.sh bash ~/ladder/oneshot.sh
15d14
< RUN_SCRIPT=${RUN_SCRIPT:-run_s1o_speed.sh}
23,29c22,23
< OLD_TIMEOUT=$(su -c 'settings get system screen_off_timeout' 2>/dev/null | tr -dc '0-9')
< if [ -n "$OLD_TIMEOUT" ] && su -c 'settings put system screen_off_timeout 2147483647' >/dev/null 2>&1; then
<   SCREEN_SET=1
< else
<   SCREEN_SET=0
<   echo "[oneshot] WARNING: could not change the screen timeout; keep the screen on yourself"
< fi
---
> OLD_TIMEOUT=$(su -c 'settings get system screen_off_timeout')
> su -c 'settings put system screen_off_timeout 2147483647'
45c39
<   if [ "$SCREEN_SET" = 1 ]; then su -c "settings put system screen_off_timeout $OLD_TIMEOUT" >/dev/null 2>&1; fi
---
>   su -c "settings put system screen_off_timeout ${OLD_TIMEOUT:-60000}"
47c41
<   say "stopped logger and watcher$( [ "$SCREEN_SET" = 1 ] && echo "; screen timeout restored to $OLD_TIMEOUT ms")"
---
>   say "stopped logger and watcher; screen timeout restored to ${OLD_TIMEOUT}"
diff exit=1
```

## Checks run (Debian/proot; output verbatim from `run_checks.sh`, final candidate)

Debian's `python3` lacks numpy/Pillow/onnxruntime. Native Termux's `/data/data/com.termux/files/usr/bin/python` (3.13.13, with `LD_LIBRARY_PATH` set to Termux's lib) runs inside proot. It was used for `py_compile` and the offline test. Fakes stand in for everything phone-side:
- RobotCam frames: real 640×480 JPEGs with the RobotCam comment, written at 2/s.
- llama-server: an HTTP server with `/health`, `/tokenize` and `/completion`.
- root output: canned battery, dumpsys, pidof and logcat text.
- detector, `am`, thermal log and cache-drop watcher.
The offline smoke uses 8 s blocks, 2 s samples and 3 s selector slots. onnxruntime's "Unsupported platform (android)" warnings are filtered.

```
$ git -C /termux-home/robot rev-parse HEAD; git status --short
3584b5eddd3639c00c37b24e6e31f4d86fce2f2b
(end git status)
$ sha256sum *
3e3cd4e21f476d32e5784ebf51b65b9076cb5cd1b6b233688033d3044cdbabae  SMOKE.md
ef622037a659a38e98f7372b5008f399bb22cc49bf8c5979a1f985b9dbd5784e  coresidency.py
efb3939d2a0a24d7e7a7abc2384f6e608b252ff12b2ca539f8bf8753146d41a2  oneshot.sh
dba86f4de4011a2bf6abd5745f3ac709633e717e5a72ff31a721dae0470962b0  run_coresidency.sh
e076915d58572cab6644fda9c0eba1c06dbd223f808b36f9a4aab8d3dad2c0d4  test_coresidency.py
bccd170dfe71bccdb316aaf86b555defed3bf528dfaef9b854d231c7289f4e19  test_oneshot.sh
$ bash -n oneshot.sh
exit 0
$ bash -n run_coresidency.sh
exit 0
$ bash -n test_oneshot.sh
exit 0
$ python3 -m py_compile coresidency.py test_coresidency.py   # Debian Python 3.13.5
exit 0
$ /data/data/com.termux/files/usr/bin/python -m py_compile coresidency.py test_coresidency.py   # native Termux python run inside proot
exit 0
$ /data/data/com.termux/files/usr/bin/python test_coresidency.py   # onnxruntime's 'Unsupported platform (android)' UserWarning lines filtered
failed logcat query reported as failed, not as 0 kills: PASS
per-pid PSS failure flagged; late sample dropped; gaps from completion times: PASS
signal during a server load or inside its Popen stops the server; all-failed selector block flagged: PASS
cores lost during the loads -> CoresLost, no server left; B5 sample error -> INCOMPLETE: PASS
Co-residency benchmark (DECISIONS #124), run run_20260930T135504Z_smoke  [SMOKE: not a measurement]
block length 8 s; idle z9 36.0 degC; runner sha256 ef622037a659

                                        B1_mix_nollm    B2_only640_nollm        B3_mix_gemma    B4_only640_gemma
frames processed                                  16                  16                  16                  16
failed reads by status                          none                none                none                none
repeat reads (not failures)                        2                   3                   3                   3
detect320 ms median/P95                 10/10 (n 15)       n/a/n/a (n 0)        10/11 (n 15)       n/a/n/a (n 0)
detect640 ms median/P95                  30/30 (n 1)        30/31 (n 16)         30/30 (n 1)        30/31 (n 16)
drift320 first/last 30s                     10 -> 10          n/a -> n/a            10 -> 10          n/a -> n/a
drift640 first/last 30s                     30 -> 30            30 -> 30            30 -> 30            30 -> 30
read/decode ms median                       1.0/11.5            1.2/13.0            1.2/13.5            1.1/12.7
frame age s median                              0.08                0.07                0.06                0.06
selector ms median/P95                             -                   -             248/250             249/259
selector calls ok/err/correct                  0/0/0               0/0/0               3/0/1               3/0/1
min MemAvailable MiB                            2929                2946                2933                2935
max swap used MiB                               2338                2338                2335                2335
peak PSS runner MiB                              121                 121                 121                 121
peak PSS llama-server MiB                        n/a                 n/a                 121                 121
peak PSS RobotCam app MiB                         37                  37                  37                  37
peak PSS camera provider MiB                     264                 264                 264                 264
LMK log lines (kill lines)                     3 (2)               3 (2)               3 (2)               3 (2)
survived RobotCam / llama                    yes / -             yes / -           yes / yes           yes / yes
zone9 start/end/max degC              36.0/36.0/36.0      36.0/36.0/36.0      36.0/36.0/36.0      36.0/36.0/36.0
gate wait s                                        0                   0                   0                   0
mean battery W                                  1.64                1.64                1.64                1.64
battery status                           Discharging         Discharging         Discharging         Discharging
sample errors                                      0                   0                   0                   0
sample gaps >3s (max s)                      0 (2.0)             0 (2.0)             0 (2.0)             0 (2.0)
samples past block end (dropped)                   0                   0                   0                   0

Gemma load (spawn to /health ok): cold 0.85 s [weights-cold (posix_fadvise DONTNEED on the GGUF)], warm 0.75 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmp2twc0wex/a/server.pid

MTP snapshot (RobotCam + both detector sessions + Gemma with drafter, after one request): MemAvailable 2936 MiB, swap used 2335 MiB, PSS llama-server 121 MiB, VmHWM 22 MiB, load 0.75 s
server command: /data/data/com.termux/files/usr/bin/python /data/data/com.termux/files/home/robot/benchmark/coresidency/test_coresidency.py --fake-server 18080 /data/data/com.termux/files/usr/tmp/tmp2twc0wex/a/server.pid --model-draft /data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3

Notes: detect ms = Detector.detect (resize, inference, NMS); read/decode split read_frame at its BytesIO call; drift = median of the first vs last 30 s; W = -(current_now*voltage_now) from battery sysfs as in camera_heat.py; PSS = root dumpsys meminfo every 5 s; LMK = logcat lines matching lowmemorykiller|lmkd|killinfo in the block window, kills = lowmemorykiller/lmkd lines containing "kill", or killinfo. Every sample runs one su call (dumpsys) in all blocks alike.

full smoke run: PASS
SIGTERM during B3: exit 143, RobotCam stopped, server stopped: PASS
--resume after SIGTERM: PASS
exit 0
$ bash test_oneshot.sh
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
exit 0
$ find /data/data/com.termux/files/home/robot -name '*.pyc' -newer oneshot.sh   # nothing written into archive folders
/data/data/com.termux/files/home/robot/benchmark/coresidency/__pycache__/coresidency.cpython-313.pyc
(end find)
```

## Unverified

- **Nothing ran natively on the phone.** Unverified: su, `am`/RobotCam, a real llama-server/Gemma load, real ONNX detection, `dumpsys meminfo` output on Android 17 (PSS for a non-app pid such as llama-server or the runner), `logcat -T <epoch>` acceptance and the lmkd kill line format, `pidof`/`ps -A -o PID,NAME` under su, `settings` via the #123 file form, the real cache drop, battery sysfs signs, Termux `pgrep` behaviour, and the timing overhead of one su+dumpsys call every 5 s. That overhead is present equally in all blocks, but not measured.
- `oneshot.sh` was tested only as a sed-patched copy: the native check was removed, paths moved to a temp dir, and fake `su`/`settings`/`pgrep`/`sleep` were used. The signal race between `&` and `RUNNER=$!` is handled but not tested.
- Whether 20 s smoke blocks meet every completeness rule on the phone is unverified. Examples: a 640 frame in both drift windows, and one selector call per slot.
- The selector uses `server_manager` flags (see above), so its ms differ in configuration from #119.
- The open MINOR from round 20: a final sample that finishes after the block end is dropped. That gap is shown in the report ("sample gaps", "samples past block end"), but it does not make the run INCOMPLETE. I did not change it, because any change needs a new review round. The human decides.

## Review

Exact command, run once per round in a fresh session (stdin = the request file; stdout and stderr kept separately):
```bash
codex exec -m gpt-6-sol -c model_reasoning_effort="medium" -s read-only -C /termux-home/robot - < request.md > stdout.txt 2> stderr.txt
```

`-s read-only` = sandbox; `-c model_reasoning_effort="medium"` = effort. No approval-bypass or dangerous flags. Codex's own header in each stderr confirms `model: gpt-6-sol`, `approval: never`, `sandbox: read-only`, `reasoning effort: medium`. Each request stated Ponytail off and read-only (no edits, staging, commits, pushes, motors or further reviewers). It contained the verbatim task, the base, every new file in full, the source context and the actual check output. From round 2 on, it also contained all prior verdicts verbatim and the Coder's changes. All artifacts (request, stdout, stderr, frozen.sha per round, plus the build scripts) are in `/termux-home/storage/downloads/coresidency_review_artifacts/`.

### Each round's verdict and findings (reviewer stdout verbatim)

<details><summary>Round 1: <b>REQUEST CHANGES</b></summary>

```
**MAJOR** — `benchmark/coresidency/coresidency.py:175-180`: LMK results can be wrong in both directions. `n_kill_lines` counts any line containing “kill,” including informational `lowmemorykiller` lines without a kill. If `logcat` fails or rejects `-T`, the code reports zero lines without reporting the command failure. A block could therefore claim no LMK kills when the query failed, or claim kills that never happened.

**MAJOR** — `benchmark/coresidency/coresidency.py:531-537`: B5 records a completed MTP snapshot even when `read_frame` returns a transient non-`ok` status. In that case the required 640 detection never runs, yet the report presents the snapshot as taken with the detector active.

**MAJOR** — `benchmark/coresidency/oneshot.sh:80-81`: The runner is piped to `tee` without `pipefail` or an explicit status check. If Python exits with an error, the launcher can still exit successfully and print “done,” making a failed phone run look successful to a caller.

**MINOR** — `benchmark/coresidency/coresidency.py:156-167,383-394,589`: Failed `dumpsys meminfo` calls or unparseable PSS output can leave PSS values absent while `sample_errors` remains zero. For example, an Android service transaction failure could produce `n/a` PSS throughout a block without being counted as a sampling error.

The supplied offline checks pass. Root, RobotCam, real ONNX inference, and llama-server behavior remain unverified in native Termux, as the Coder reported.

REQUEST CHANGES
```

</details>

<details><summary>Round 2: <b>REQUEST CHANGES</b></summary>

```
**MAJOR** — `benchmark/coresidency/coresidency.py:493-495`: A selector request can remain blocked for its 600 s HTTP timeout, but the block waits only 120 s for its thread and then proceeds. If B3’s request stalls, B4 can run while B3’s request is still using the same server and selector, invalidating both measurements. An unfinished selector call should fail or discard the block.

**MAJOR** — `benchmark/coresidency/oneshot.sh:42-45`: The launcher treats a zero exit status from `settings put` as proof that the screen timeout changed, without reading it back. If Android prints a service transaction failure but exits zero, the screen can time out during the idle period and move Termux off cores 4–7. Read back and verify the set value before continuing.

The four Round 1 fixes are present in the supplied candidate. I verified the base commit and clean tracked tree, but the read-only sandbox denied direct reads of the ignored candidate files; these findings refer to the complete file contents in the request. The supplied offline checks pass; native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 3: <b>REQUEST CHANGES</b></summary>

```
**MAJOR** — `benchmark/coresidency/coresidency.py:407-411,604`: If a `dumpsys meminfo` sample takes more than five seconds, the sampler skips the missed slot without recording a gap. An eight-second sample, for example, moves the next reading to ten seconds; the report can still show `sample errors 0` while missing a RAM or power peak during the gap.

**MAJOR** — `benchmark/coresidency/oneshot.sh:38-41,85-88`: Cleanup reads back the restored screen timeout but never checks that it equals the saved value. If `settings put` silently fails during restoration, the launcher can report success and leave the phone at `2147483647` instead of its original timeout.

The two Round 2 fixes are present in the supplied candidate. The sandbox denied direct content reads of the ignored files, so these findings use the complete file contents supplied in the request; I verified the stated hash of `coresidency.py`. The offline checks passed as supplied. Native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 4: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/coresidency.py:393-404,605-607`:** A sample is timestamped before its root commands finish. If a sample starts at 175 s and `dumpsys` takes eight seconds, its power and PSS values are collected after the 180 s block, yet the report assigns them to 175 s and shows no sample gap. This can distort the block’s RAM and power results. Record sample completion time and exclude or flag values collected beyond the block.

**MINOR — `benchmark/coresidency/coresidency.py:163-171`:** PSS validation checks process names, not each discovered PID. If two camera provider processes exist and one `dumpsys meminfo` fails, the other supplies `camera_provider` PSS, `pss_error` stays absent, and peak provider PSS is undercounted.

The Round 3 sample-gap and screen-timeout restore changes are present in the supplied candidate. The sandbox denied direct reads of the ignored candidate files, so these findings use their complete contents in the request. The supplied offline checks pass; native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 5: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/coresidency.py:70-71,183-193`:** The LMK query collects `lmkd` lines, but the kill counter recognizes only `lowmemorykiller: Kill` or `killinfo`. A log line such as `lmkd: Kill 'com.example'` would appear in the captured lines while the report counts zero kills. This can invalidate a required comparison metric.

**MINOR — `benchmark/coresidency/oneshot.sh:30-33,87-93`:** Battery status is checked before the five-minute idle and is not checked again when the benchmark starts. If charging begins during idle, the run proceeds and reports power figures that the launcher was meant to reject.

The base commit and clean tracked tree were verified. The sandbox denied direct reads of the ignored candidate folder, so this review uses the complete numbered candidate contents and check output supplied in the request. Native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 6: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/coresidency.py:308-315,525-533,545`:** SIGTERM or SIGHUP during a Gemma cold, warm, or MTP load can leave `llama-server` running. `Server.__init__` starts a detached process, but the caller receives the `Server` object only after the health wait finishes. A signal during that wait raises `SystemExit`; the outer cleanup has no reference to the process and cannot stop it.

**MAJOR — `benchmark/coresidency/coresidency.py:386-390,745-760`:** B3 or B4 can finish successfully when every selector call fails. The loop records each exception and continues; the run then writes a report and exits zero with no selector latency. For example, repeated completion API errors would leave the required Gemma selection workload unmeasured while presenting a completed benchmark.

**NIT — `benchmark/coresidency/coresidency.py:675-679`:** The report’s LMK note omits `lmkd` kill lines, which the updated counter now includes.

The Round 5 LMK regex and second battery check are present. The supplied offline checks pass; native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 7: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/coresidency.py:746-753,528-539`:** Gemma’s cold and warm load times are measured before the next cpuset check. If Termux loses cores 4–7 while the thermal gate waits, both loads can be timed on the wrong cores and saved to `loads.json`; `wait_cores()` then allows B3 to run without correcting those results. Check the cpuset around each measured load and redo invalid measurements.

**MINOR — `benchmark/coresidency/coresidency.py:556-559,681-688`:** B5 is saved as complete even when its RAM sample reports a root or PSS error. A failed `dumpsys meminfo` can leave llama-server PSS as `n/a` in the MTP snapshot while the run exits successfully. Surface the sample error and mark that snapshot incomplete.

The Round 6 fixes are present in the supplied candidate. The base commit and clean tracked tree were verified; the sandbox denied direct content reads of the ignored candidate files, so the review used their complete supplied contents and verified hashes. Native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 8: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/oneshot.sh:40-58,98-101`:** The launcher’s EXIT trap restores the screen timeout but does not signal the runner. If SIGTERM or SIGHUP targets the launcher PID while the runner is active, Python may keep running with RobotCam and llama-server. Forward those signals to the runner and wait for its cleanup; the Python signal handler only helps when Python receives the signal.

**MAJOR — `benchmark/coresidency/coresidency.py:667-668,624-630,786-795`:** A failed LMK query is printed as `QUERY FAILED`, but it does not make the run incomplete. If Android rejects `logcat -T`, the launcher can exit successfully with no LMK kill count for any block, leaving a required comparison metric unavailable.

The Round 7 cpuset checks and B5 sample-error handling are present in the supplied candidate. The sandbox denied direct reads of the ignored candidate files, so this review relies on their complete contents and the check output in the request. Native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 9: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/coresidency.py:616,624–632,788–797`:** B1–B4 can finish successfully with required sampling data missing. If `dumpsys meminfo` fails throughout B3 but the B5 sample succeeds, B3 reports `sample errors` and `n/a` PSS while the runner exits zero. Make block sampling failures render the run incomplete.

The Round 8 signal forwarding and LMK failure changes are present in the supplied candidate. The supplied offline checks pass; native phone behavior remains unverified. The sandbox denied direct content reads of the ignored files, so I reviewed the complete supplied contents and verified the stated hashes of `oneshot.sh` and `coresidency.py`.

REQUEST CHANGES
```

</details>

<details><summary>Round 10: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/oneshot.sh:58,91–102`:** The launcher installs its TERM/HUP/INT trap only after the five-minute idle. If the launcher receives one of those signals during idle, it cannot forward it or reliably complete cleanup; the screen timeout may remain at `2147483647`, and the thermal logger and cache-drop watcher may remain running. Install the signal trap before idle and use the same cleanup path.

The round 9 change flags in-block sampling failures as incomplete. The base commit and clean tracked tree were verified. The sandbox denied direct reads of the ignored candidate files, so this review uses the complete supplied contents and check output; native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 11: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/oneshot.sh:60-64,107-108`:** A signal can arrive after the runner starts in the background but before `RUNNER=$!`. The handler then sees an empty `RUNNER` and exits. The `EXIT` trap restores the screen timeout, but the runner can continue with RobotCam and llama-server active. The supplied signal tests send signals during idle or after “RUNNER STARTED”; they do not cover this launch window.

The round 10 idle-signal fix is present. The base commit and clean tracked tree are verified. This sandbox denied direct reads of the ignored candidate files, so the finding uses the complete supplied contents; native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 12: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/coresidency.py:308-310,801-805`:** A SIGTERM or SIGHUP can arrive after `Popen` starts the detached llama-server but before `self.proc` is assigned and the `try` block begins. `SystemExit` then reaches `main`’s cleanup while `ctx['server']` is still `None`, leaving llama-server running. This applies to cold, warm, and MTP loads. Put process creation inside a cleanup path that covers this window.

The round 11 launcher signal fix is present in the supplied candidate. I verified the base commit and clean tracked tree. The sandbox denied direct reads of the ignored candidate files, so this finding uses their complete contents in the request. The supplied offline checks pass; native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 13: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/coresidency.py:319-327`:** The deferred-signal flag is cleared before the cleanup `try` begins. If SIGTERM or SIGHUP arrives in that gap, `SystemExit` bypasses `self.stop()`. The caller has not received the `Server` object, so its final cleanup cannot stop the detached llama-server. The new test sends SIGTERM inside `Popen` and does not cover this gap. Keep signal deferral active until the server is covered by cleanup and handed to its caller.

The supplied offline checks pass. Native Termux and hardware behavior remain unverified; this sandbox denied direct content reads of the ignored candidate files, whose checked hashes match the supplied contents.

REQUEST CHANGES
```

</details>

<details><summary>Round 14: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — [coresidency.py](/termux-home/robot/benchmark/coresidency/coresidency.py:630):** A run can finish successfully with invalid power data if charging begins after the launcher’s final battery check. Samples record `Charging`, but `problems()` does not flag it; the report still presents mean watts for the affected block. Mark any block with a non-`Discharging` sample incomplete.

The Round 13 `LIVE` registration and final cleanup address the previously reported server ownership gap. The supplied offline checks pass; native phone behavior remains unverified. The sandbox denied direct reads of the ignored candidate files, so this review used their complete numbered contents in the request.

REQUEST CHANGES
```

</details>

<details><summary>Round 15: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/oneshot.sh:24-29,88-99`:** The launcher does not refuse a second co-residency run. During B1 or B2, no llama-server is running, so a second launch can pass the process checks. Both runs then share `.stop_thermal`, the cache-drop handshake files, RobotCam, and the root sample output path. When either launcher exits, its cleanup can stop the other run’s logger; overlapping work also invalidates both measurements. Refuse an existing co-residency runner or launcher before starting.

The Round 14 change flags `Charging` in an in-window block sample as incomplete. The supplied offline checks pass. Native Termux, root, camera, ONNX, and llama-server behavior remain unverified. This sandbox denied direct reads of the ignored candidate files, so the finding uses the complete file contents supplied in the request.

REQUEST CHANGES
```

</details>

<details><summary>Round 16: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/coresidency.py:641-654,773-800`:** A block can be saved as complete with no useful detection workload. For example, if RobotCam publishes the startup frame and then stops, the block can record few or zero detections and `robotcam_new_frame_at_end: false`, yet `problems()` does not mark it incomplete. `--resume` then skips that block, leaving missing YOLO latency and drift data in a run that can exit successfully. Keep the JSON as evidence, but flag the block as incomplete.

The Round 15 runner check and atomic launcher lock address the overlapping-run scenario. I verified the base commit and clean tracked tree. This read-only sandbox denied direct reads of the ignored candidate files, so the finding relies on the complete contents supplied in the request; the offline check results are Coder-supplied, and native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 17: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/oneshot.sh:40-58,101-112`:** Cleanup signals the thermal logger and cache-drop watcher to stop but does not wait for them before releasing `.lock`. If another run starts within the logger’s five-second sleep, it removes `.stop_thermal` before the old logger checks it. Both loggers can then run; the old watcher may also answer the new run’s cache-drop request. Stop and wait for both jobs before releasing the lock.

**MAJOR — `benchmark/coresidency/coresidency.py:475-481,614-615,650-652`:** The Round 16 check still accepts a block with one detection. If that detection takes nearly the full block, RobotCam can publish a new frame for the end check, so the block passes despite having no 640 detection in mix mode and no usable drift comparison. Keep the JSON as evidence, but mark the required missing measurements incomplete.

The Round 16 checks for zero frames, failed reads, and no new frame at the end are present in the supplied contents. I verified the base commit and clean tracked status; the sandbox denied direct reads of the ignored candidate files. The offline check results are Coder-supplied, and native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 18: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — `benchmark/coresidency/coresidency.py:514,526-529,650-652,702`:** The runner records whether the RobotCam process survived, but the report and completeness check use only whether a new frame appeared at the end. If RobotCam publishes that frame and then crashes before `pidof`, the JSON records `robotcam_process: false` while the report says `yes` and the run can succeed. Use the process result when reporting survival and assessing completeness.

The Round 17 changes check for missing size and drift measurements and keep the launcher lock until its logger and watcher exit. The supplied offline checks pass; native phone behavior remains unverified. The sandbox denied direct candidate reads, so this review used the complete supplied contents and verified the stated hashes of `oneshot.sh` and `coresidency.py`.

REQUEST CHANGES
```

</details>

<details><summary>Round 19: <b>REQUEST CHANGES</b></summary>

```
**MAJOR — [coresidency.py](/termux-home/robot/benchmark/coresidency/coresidency.py:400):** Selector calls can miss the required 20 s cadence while the run reports success. If the first B3 call takes 170 s and the second takes 15 s, only two calls occur, the second finishes after the block, and its recorded `t` is still 20 s. [The completeness check](/termux-home/robot/benchmark/coresidency/coresidency.py:645) requires only one successful call, so the block can be accepted despite missing most of the intended Gemma workload. Record actual call times and flag missed cadence or calls outside the block.

The Round 18 RobotCam process fix is present in the supplied candidate. The six file hashes match the request; the sandbox denied direct content reads, so this review used the complete supplied contents. Offline checks are Coder-supplied; native phone behavior remains unverified.

REQUEST CHANGES
```

</details>

<details><summary>Round 20: <b>APPROVE WITH NOTES</b></summary>

```
**MINOR — `benchmark/coresidency/coresidency.py:624-639,657-659`:** A sample that finishes after the block is dropped, but the resulting gap does not make the run incomplete. If the final `dumpsys` call starts at 175 s and finishes at 183 s, the block can exit successfully without RAM or power data for its final interval. The report shows the gap, so the missing coverage is visible.

Round 19’s selector cadence fix records actual start and end times and flags missed slots. The base commit matches and the tracked tree is clean. The sandbox denied direct reads of the ignored candidate files; this review used the complete numbered contents supplied in the request. Offline checks are Coder-supplied, and native phone behavior remains unverified.

APPROVE WITH NOTES
```

</details>

## Final review (round 20), verbatim

```
**MINOR — `benchmark/coresidency/coresidency.py:624-639,657-659`:** A sample that finishes after the block is dropped, but the resulting gap does not make the run incomplete. If the final `dumpsys` call starts at 175 s and finishes at 183 s, the block can exit successfully without RAM or power data for its final interval. The report shows the gap, so the missing coverage is visible.

Round 19’s selector cadence fix records actual start and end times and flags missed slots. The base commit matches and the tracked tree is clean. The sandbox denied direct reads of the ignored candidate files; this review used the complete numbered contents supplied in the request. Offline checks are Coder-supplied, and native phone behavior remains unverified.

APPROVE WITH NOTES
```

## Status

Edits are frozen. Nothing is staged, committed or pushed. **Waiting for the human's decision.** Suggested next steps for Local AI and the human:
1. Decide on the round-20 MINOR and the reviewer-sandbox note.
2. Decide how to commit an ignored folder.
3. Run `bash ~/robot/benchmark/coresidency/oneshot.sh --smoke` from native Termux per `SMOKE.md`.

DOC DIFF candidates for Doc Keeper after a real run: STATUS "Co-residency test (DECISIONS #124)" and "`oneshot.sh` screen-timeout restore is broken (#123)". The new launcher fixes the latter for this benchmark only; `~/ladder/oneshot.sh` is untouched.
