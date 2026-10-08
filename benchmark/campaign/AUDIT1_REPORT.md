# CAMPAIGN_AUDIT1: what can stop, corrupt or silently invalidate the 90-minute owner session

Read-only audit of `benchmark/campaign/` at the FIX3D commit. This report is the only file created. Nothing was
edited, staged, committed or pushed. No runner, camera, llama-server, motor, `su` or other process was started against the phone.

## 0. Profile and provenance

| Item | Value and how it was confirmed |
|---|---|
| Role | Auditor (read-only) |
| Platform | Claude Code CLI, interactive. Fresh session (started after `/clear`) |
| Model | `claude-opus-5-5`, as stated by the harness in this session's system context |
| Effort | `xhigh` was requested. **Not independently confirmed:** no tool or command in this session exposes the reasoning-effort setting, so it is reported as requested and unverified. |
| Ponytail | **Off.** The session-start hook announced "PONYTAIL MODE ACTIVE — level: lite". The execution profile and `docs/WORKFLOW.md` (independent review runs with Ponytail off) override it, so it was treated as off. It did not shape any finding. |
| `git rev-parse HEAD` | `56296995ea780141d757ba74ee768e07e27c2979`, "benchmark: campaign phase-1 thermal read retry, session archives". This is the FIX3D commit (`frozen_manifest_campaign_p1_fix3d.json`: task CAMPAIGN_P1_FIX3D, base b723002, review round 3). |
| `git status --short` | Empty. `main` is in sync with `origin/main`. |
| Tree vs frozen manifest | All 75 manifest files match their SHA-256 (checked with Python `hashlib`) |
| Native vs proot repo | `/data/data/com.termux/files/home/robot/...` and `/termux-home/robot/...` are the same files (`phase1.py` has the same inode in both) |

**Read in full:** AGENTS.md, docs/WORKFLOW.md, docs/REVIEWER.md, campaign RUN.md, RUN_INDEX.md, phase1.py,
runtime.py, diagnostics.py, power.py and lag_probe.py; coresidency/coresidency.py; relate_anything/desk2/speed_block.py;
speed1/variants.py.

**Read in part (the parts phase1 uses):** power_map.py (PinnedDetector, build_detector, camera_start,
capped_by_policy); desk_check.py (Head, preprocess); robotcam_reader.py; detect_person.py (`Detector.detect`); strategic_selector/v3/adapters.py
(S1O); ladder/measure.py (`allowed_cpus`); RobotCam CameraService.kt (publish and delete paths).

**Skimmed for coverage:** self_check_fix3d.py and the checks/p1_fix3d/ outputs.

**Owner evidence:** every `owner_*` .stdout, .stderr and .json in `~/storage/downloads/campaign/` (the same bytes as `runs/`), parsed with `python3 -I -c`.

## 1. How the real owner runs failed

| Run | Stop | Where |
|---|---|---|
| owner_session_p1a, p1a2 | Monitor mask refusal at setup | prepare_monitor (non-root mask readback) |
| owner_session_p1_fix2 | Camera-OFF PSS missing (app force-stopped) | rest() memory check, after the 180 s idle |
| owner_rehearsal_p1_fix3 | The guard matched our own `su -c … read_node` | Warm-up shared check |
| owner_session_p1_fix3c | Thermal worker blank read at 5.26 s. The owner had locked the screen; the bytes do not show this. | rest() worker, then 'pause sampler stopped' |
| owner_session_p1_fix3c2 | Main-thread blank read at about 54 s of the camera-ON idle | rest() loop, `phase1.py:571` |

The common shape is that one transient or state-dependent read, met with zero tolerance, ends the whole session. In fix3c the recorded
cause was also not the event that actually happened. The findings below look for the remaining instances of that shape.

## 2. Findings

**Severity:** HIGH means it can end or invalidate the session. MEDIUM means it degrades the evidence. LOW means it is cosmetic. Where the archive supports it, the likelihood is
stated separately. All line numbers refer to HEAD 5629699.

---

### H1. Rest-phase power bracket is taken before the second root shell is built; the first gap uses 60–80 % of the 1.5 s rejection budget in every pause
- **Class:** A (first-sample/startup edge), E (timing). **Severity: HIGH.** Likelihood: low per phase. Across 9 rest phases per session it is not negligible, and the risk is highest in the session's hot post-block pauses.
- **Where:**
  - `phase1.py:539–544`: bracket `power.sample`, then `began` is reset to t=0.
  - `phase1.py:557`: only then `fast_monitor=rt.prepare(sb,{4,5})`. This involves `setup_affinity`, the `su` launch, 2× `taskset`, `echo $$`, 2× `root_mask`, `cr.discover` (about 100 sysfs reads) and `verify_monitor`.
  - `phase1.py:560–567`: the sampler thread starts only after that.
  - `power.py:114,125–127,131–132`: any gap over 1.5 s inside [0, duration] sets `mean_battery_w=None`.
  - `phase1.py:603–604`: `RuntimeError('pause power coverage missing')`. The session then stops.
  - Contrast `live_block`: `fast_monitor` is built at `:379` before the bracket at `:394`, and threads start at `:403–412`. Its bracket gap is 0.05 s.
- **Scenario on the phone:** in any of the 3 idle sub-phases or 6 × 600 s pauses, the bracket sample lands at t≈0. The sampler's first sample arrives after the fast shell is ready, at 0.9–1.2 s. If the `su` launch and discover are about 0.35 s slower than in the archive, the gap exceeds 1.5 s. Causes include a busy Magisk daemon, the post-block heat with LITTLE capped at 1401 MHz, or the camera-ON idle. The whole pause is then rejected and the session ends as NOT VALID — SESSION INCOMPLETE, even though every other sample was fine.
- **Evidence:** in all 17 archived rest phases the largest power gap is the one starting at t≈0:
  - fix3c2 idle: 0.890, 1.168 and 1.145 s.
  - fix2 idle: 0.932 s.
  - fix3c session idle: 1.181 s.
  - fix3c rehearsal idle: 1.066, 1.050 and 1.186 s.
  - fix3c rehearsal pauses: 1.062, 1.001, 1.055, 0.960, 0.997 and 0.892 s.
  - fix3 rehearsal idle: 0.998, 0.984 and 1.112 s.

  The second-largest gap in every phase is ≤ 0.53 s. The first fast, thermal and memory rows start at 0.80–1.10 s for the same reason. The largest values come from camera-ON idles, the busiest state. The margin is 0.31 s.
- **Fix:** in `rest()`, move the `fast_monitor` preparation (`:557`) above the bracket sample (`:539`). Take the bracket, reset `began` and start the threads straight away, the same order `live_block` uses. No other change is needed.

### H2. Only the thermal read has a bounded retry; one transient blip among about 25,000 other reads ends the session
- **Class:** B. **Severity: HIGH.** Likelihood: unknown. The archive is too small to exclude it.
- **Where, with approximate per-session counts** (about 5,400 s of sampling):

| Read | Code | ≈ per session | On a single bad read |
|---|---|---|---|
| power current_now/voltage_now/status | `power.py:96–106` (`int('')`, `len!=3`), sampler `:146–169` | 15,000 | monitor error → block shared / pause raise |
| root/su/pump mask verify, every power and fast sample | `runtime.py:107–121`, `speed_block.py:182–200` | 45,000 shell round trips | 'missing/malformed root Cpus_allowed_list' |
| fast sample (3 CPU zones, battery temperature, 3 policies) | `coresidency.py:648–665`, `runtime.py:310–311` | 5,600 | any empty value → 'missing root CPU/battery/policy reading' |
| `sb.battery_sample` in the rest loop | `speed_block.py:79–88`, `phase1.py:570` | 800 | rc/fields → raise |
| root_sample PSS + battery | `coresidency.py:216–248`, `runtime.py:279–297` | 1,050 | pss_error → `phase1.py:605–606` / `:473–474` |
| pidof absence confirmation (camera OFF) | `runtime.py:290–296` | 750 | pss_error |
| `/health`, 2 s timeout | `coresidency.py:405–410`, `phase1.py:337–338`, `:572` | 1,000 | 'llama-server unhealthy' |
| am start/broadcast | `coresidency.py:130–136`, `power_map.py:168–181` | 40 | CalledProcessError or Timeout (camera_start retries only the no-frame case) |
| force-stop + one pidof | `coresidency.py:174–181,187–201` | 30 | 'still running/unconfirmed' → cleanup error |
| logcat LMK window | `coresidency.py:256–266`, `phase1.py:446–452` | 7 | block invalid + shared → stop |
| pgrep | `runtime.py:84–86` | 1,050 | rc ∉ {0,1} → raise |
| transient root-mask prefix | `runtime.py:327–337` | 4,600 | rc 97/98 → raise |
| diag battery/required nodes | `diagnostics.py:22–33,57–67` | 45 | raise |
| thermal (the only one with a retry) | `runtime.py:210–246` | 1,900 | rc 0 incomplete retried ≤ 3 times within 2 s |

- **Scenario on the phone:** examples include a fuel-gauge sysfs read that returns empty once, a Magisk `su` that exits non-zero once, a `/health` that takes 2.1 s while llama-server competes with ORT, or a pidof run immediately after force-stop that still sees a dying process. Any one of these ends the session, sometimes at minute 80.
- **Evidence:**
  - The campaign archive holds about 4,000 power, 1,450 fast, 300 memory, 300 thermal and 160 shared-check reads without a non-thermal transient. One full session performs about 4× that.
  - With 0 failures in about 4,000 power reads, the 95 % upper bound on the per-read rate is about 7.5e-4. The archive therefore cannot rule out several failures per session.
  - The blank thermal dumps in fix3c and fix3c2 show that single-read transients do occur on this phone.
- **Fix:** one shared helper performing at most one immediate re-read (≤ 0.3 s), only for unparsable or empty output, a `su` that fails to launch, or an HTTP timeout.
  - Record `retries` on the row and in a `read_retries` list, as `THERMAL_RETRIES` does.
  - Never retry a value that parsed correctly but is bad: Charging, status ≥ 4, an affinity mismatch, or a pid still present after force-stop. The last one should be re-checked once after 0.5 s and only then counted as a failure.
  - For `/health`, require 2 consecutive failures within one check period.
  - Cap retries per phase (for example, more than 3 makes the phase NOT VALID) so a degrading sensor cannot hide.

### H3. M2 on live camera boxes has never run on the phone; more than 32 live boxes ends the session; RUN.md lets the owner skip the fix3d rehearsal
- **Class:** G and A. **Severity: HIGH.** The first live M2 call happens about 45 min into the session (L1).
- **Where:**
  - `phase1.py:84–87`: fallback is used only when live boxes < 2.
  - `phase1.py:287–288`: more than 32 boxes raises.
  - `phase1.py:102–105`: the M2 error sets stop.
  - `:205–206`: run_cycle raises. `:428–430`: failure_kind 'shared'. `:771`: the session stops.
  - Detector threshold `CONF=0.35` (`detect_person.py:19`).
  - RUN.md:98–101: "you may start the session directly or run step 1 first".
- **Scenario on the phone:** the phone is upright facing the table. Background clutter (a shelf of books, bottles, a keyboard) yields more than 32 YOLO-640 boxes in any frame, and the L1 block stops the session. The live M2 path itself (live PIL frame plus detector boxes into `Head.infer`) has never executed on the phone. Neither has the live/fallback alternation and its context labelling.
- **Evidence:**
  - owner_rehearsal_p1_fix3c `rehearsal_coverage`: `m2_live_calls` was 0 in every layout (the phone lay on its back).
  - owner_session_p1a_fix: YOLO-640 `n_detections` was always 0–1.
  - The box format is compatible: `Detector.detect` returns clamped tuple boxes with x2 ≥ x1, which `desk_check.py:34–39` accepts. A crash from format is therefore unlikely. The risk is the scene.
- **Fix (no code):** make the fix3d rehearsal mandatory before the session. Its `rehearsal_pass` already requires at least one live M2 call per layout. Also check that the rehearsal blocks' YOLO-640 `n_detections` stay well under 32. RUN.md should say "≤ 10 objects in view, no shelves".
- **Optional code change:** add `max_live_boxes` per block to `coverage()` and stdout. Decide whether more than 32 should be a counted per-slot fallback rather than a session stop.

### H4. One late or absent camera frame is a session-ending shared failure
- **Class:** A and B. **Severity: HIGH.** Likelihood: low.
- **Where:**
  - `phase1.py:271–278`: `LiveOps.frame` returns any non-'ok' status at once. Only a repeat waits, for up to 1.35 s.
  - `phase1.py:167–168`: `RuntimeError('camera failed/repeated frame')`. It becomes shared at `:430` and stops the session at `:771`.
  - `robotcam_reader.py:38–39`: a frame older than 2.0 s reads as 'missing'.
  - `CameraService.kt:409–420`: published files are deleted on any camera error or restart.
- **Scenario on the phone:** at 1 frame/s the frame age normally cycles between 0.1 and 1.2 s. A single publish hiccup of about 0.8–1.5 s makes the slot's read 'missing' (or a repeat past 1.35 s), and all later blocks are lost. Possible causes are a JPEG encode stall under heat, a GC pause in RobotCam or a slow FUSE rename. The camera may have recovered one second later.
- **Evidence:**
  - About 1,100 archived block reads, 0 failures.
  - Frame age p99 was 0.66–0.74 s in the p1a_fix blocks, with a maximum of 1.18 s (fix3c rehearsal warm-up). That leaves about 0.8 s of margin to `max_age`.
  - RUN.md:146 classifies camera failures as stop-worthy. The problem is that a single late frame is indistinguishable from a dead camera.
- **Fix:** in `LiveOps.frame`, poll 'missing' as it already polls a repeat, until the same 1.35 s deadline. If there is still no newer 'ok' frame, count the slot as skipped (frames_skipped, camera_late, so a cadence miss) and continue. Stop the session only after N consecutive failed slots (for example 3), or immediately on 'other_session' or 'bad'.

### H5. Interruptions end the session and are never recorded as interruptions
Interruptions here means a call, an alarm or timer, a full-screen intent, a system dialog, or the screen being locked or switched off.
- **Class:** D. **Severity: HIGH.** These can happen without owner intent.
- **Where:**
  - `coresidency.py:306–308`: CoresLost, via `phase1.py:329,570`.
  - `speed_block.py:168–171`: check_pinning, via `phase1.py:280,283,290`.
  - Thread and pump affinity checks: `phase1.py:343–350,548–550`, `runtime.py:116–121`.
  - RUN.md:5–10 asks for screen ON and Termux in front, but not airplane mode, Do Not Disturb, alarms off or no updates.
  - No screen or foreground state is read anywhere. `diagnostics.py:71` records only the requested camera state.
- **Scenario on the phone:** a call or alarm at minute 60 moves Termux out of top-app. Cores 4–7 are lost or thread masks change, and the session stops correctly. However, the JSON and stderr show only 'pause monitor failed' or 'monitor affinity changed', with nothing saying the phone was interrupted.
- **Evidence:** owner_session_p1_fix3c. The owner locked the screen, but the bytes show only a thermal-read failure at 5.26 s. RUN_INDEX.md: "The bytes do not show the screen lock."
- **Fix (owner procedure):** add to RUN.md:
  - Airplane mode (the server is 127.0.0.1).
  - Do Not Disturb with alarms and timers off, and no alarm due in the next 2 h.
  - Automatic system/app updates paused.
  - Magisk superuser notifications set to none.
  - Do not touch the phone.
- **Fix (runner):** in `diag.snapshot` and once in each phase's failure path, read `dumpsys power | grep -m3 -E 'mWakefulness=|Display Power|mHoldingDisplay'` and the resumed activity (`dumpsys activity activities | grep -m1 -E 'topResumedActivity|mResumedActivity'`). A lock or a lost foreground then appears in the bytes.

---

### M1. The main JSON is rewritten in full, non-atomically, about 15 times while growing to roughly 8–10 MB
- **Class:** F. **Severity: MEDIUM.** It corrupts evidence; it does not stop the run.
- **Where:**
  - `phase1.py:499–500`: `path.write_text(...)` truncates the file and then writes it.
  - Called at `:732,746,751,758,770,787`.
  - The idle sub-phases and all 6 pauses exist only inside this file. Blocks and the warm-up also have their own files.
  - The label stays 'SESSION STARTING' (`:720`) until the final write.
- **Scenario on the phone:** a SIGKILL during a 1–3 s write leaves a truncated, unparseable JSON, and every idle and pause record is lost. Possible causes are LMK, the phantom-process killer, the owner force-closing Termux, the battery being cut, or storage filling up. A file that was killed before reaching the end says 'SESSION STARTING'.
- **Evidence:** fix3c2 was 555 KB after about 320 s of idle (about 1.6 KB per sampled second). The p1a_fix blocks are about 260 KB per 180 s. For the full schedule that extrapolates to about 8–10 MB.
- **Fix:** write to a temporary file in the same directory, flush and `os.fsync` it, then `os.replace`. Set the label to 'IN PROGRESS — NOT VALID' at the first write. Optionally write each pause to its own file, as blocks are written.

### M2. L1 is NOT VALID by construction
- **Class:** A and F. **Severity: MEDIUM.** About 14 minutes of the session produce a block that cannot be VALID. The run continues, because this is a cadence miss.
- **Where:** `phase1.py:181–186` (M2 busy at the next 640 frame sets cadence_missed), `:210–212` (36 M2 calls expected), `:426–427`.
- **Scenario:** M2 on LITTLE with 4 threads takes 4.4–5.2 s per call, which equals or exceeds the 5 s cadence. Every other 640 frame finds M2 still busy. L1 ends with about 18–20 of 36 calls and is labelled NOT VALID — INCOMPLETE.
- **Evidence:**
  - Rehearsal block_02_L1: M2 calls of 5120, 4787, 4422 and 5183 ms; the slot-5 frame was skipped as busy; failure_kind 'cadence'.
  - policy0 was capped at 1401 MHz for about 99 % of each p1a_fix block, and for 90 % of the camera-ON idle in fix3c2.
  - The mock dry run cannot show this, because MockOps sleeps 0.01 s for L1.
- **Fix:** decide before the session. Either accept and label L1 as a measured overload (record the achieved M2 rate and the skipped slots rather than INCOMPLETE), or change L1's trigger (for example every second 640 frame). Otherwise tell the owner that L1 will be invalid.

### M3. The redundant rest-loop reads are where fix3c2 died
- **Class:** B and A. **Severity: MEDIUM.**
- **Where:**
  - `phase1.py:570`: `sb.battery_sample()`, whose result is discarded.
  - `phase1.py:571`: `rt.dump_check(cr)`, whose result is also discarded.
  - Both run every ≤ 5 s alongside the thermal worker (`:561`), the power sampler (Discharging check in `power.py:102–103`) and the memory rows (battery_status).
  - `block_limit` at `:573` already enforces status ≥ 4 and 60 s skin freshness from the worker's rows.
- **Scenario:** these reads add about 1,600 `su` calls of failure exposure (see H2) and produce no evidence.
- **Evidence:** fix3c2's session-ending read was exactly `phase1.py:571` (stderr traceback).
- **Fix:** drop both from the rest loop (the worker and sampler already cover them), or append their rows to `record['dumps']` / `record['memory']` so they at least count as evidence.

### M4. Session-ending messages lose the cause
- **Class:** D and F. **Severity: MEDIUM.**
- **Where:**
  - `phase1.py:569` ('pause monitor failed'), `:575–576` ('pause sampler stopped'), `:598` ('pause cleanup failed'), `:601–602`, `:752`, `:771` ('… invalidates later blocks').
  - `runtime.py:310–311` replaces fast_sample's own error text (for example 'root shell: no answer in 5 s', `coresidency.py:656–657`) with a generic message.
  - stdout prints nothing between setup and the final label.
- **Scenario:** the owner and the next reviewer see only generic text in stderr and in `result['error']`. The real cause sits in nested `monitor_errors`, or nowhere for the fast shell.
- **Evidence:**
  - fix3c stderr: `RuntimeError: pause sampler stopped`. The cause appears only in `idle_phases[0].monitor_errors`.
  - fix3 rehearsal stderr: 'warm-up invalidates later blocks'.
- **Fix:**
  - Append the first one or two monitor or cleanup errors (truncated to 300 characters) to each raised message and to `result['error']`.
  - Keep fast_sample's `error` in the fast_check exception.
  - Print one flushed stdout line at each phase start and end (label and elapsed time).

### M5. No watchdog on main-thread progress; several waits are unbounded
- **Class:** E. **Severity: MEDIUM.** A hang rather than a stop.
- **Where:**
  - `phase1.py:166` → `robotcam_reader.py:18–19`: FUSE open and read with no timeout.
  - `phase1.py:174,289` → `power_map.py:101–106`: `answers.get()` with no timeout.
  - `runtime.py:84`: pgrep with no timeout.
  - `speed_block.py:54,76`: termux-wake-lock and termux-wake-unlock with no timeout.
  - `phase1.py:306` → `coresidency.py:476–479`: 13 `/tokenize` posts at every block start with S1O's 600 s timeout (`adapters.py:175`), before `bounded_selector` replaces `_post`.
  - `block_limit` and the guard are evaluated only on the main thread (`phase1.py:333,573`).
- **Scenario:** MediaProvider or FUSE stalls, or ORT hangs. The main thread blocks, the monitors keep sampling, and nobody evaluates the thermal limit or ends the run. The screen stays on at the INT_MAX timeout until the battery dies.
- **Fix:**
  - The shared-check worker (or the power sampler) compares a main-loop heartbeat timestamp; if it is older than about 15 s, record an error and `os.kill(os.getpid(), signal.SIGTERM)`. The existing handler raises in the main thread, and the normal cleanup runs.
  - Add `timeout=` to pgrep and termux-wake-*.
  - Set `_post` before make_selector's tokenize calls.

### M6. The fix3d rehearsal does not exercise much of what the session does
- **Class:** G. **Severity: MEDIUM.**
- **Not exercised by any rehearsal (fix3c or a fix3d one):**
  - 600 s pauses × 6 (the rehearsal uses 6 s). That is about 100× the exposure to H2-class reads, with long camera-OFF Gemma-idle stretches. The longest camera-OFF observed so far is 240 s.
  - 180 s blocks reaching steady heat.
  - L1 overload over 36 slots (M2).
  - RobotCam continuity over 180 frames (H4).
  - Live M2 (H3; fix3c had none).
  - Main-JSON growth (M1).
  - The battery-25 % stop path.
  - NOT COMPARABLE labelling (the rehearsal overrides validity).
- **New FIX3D code:** per-thread thermal files and the retry window have never run on the phone (RUN.md:236–241).
- **Fix:** run the fix3d rehearsal first (12 min) and treat `rehearsal_pass` as the go/no-go. If time allows, add a one-off "long rehearsal" with one 600 s pause and one 180 s M2 block (about 16 min) to cover the long-duration paths.

### M7. Fixed 600 s pauses against the ±1.5 °C T_ref band may make later blocks NOT COMPARABLE
- **Class:** A. **Severity: MEDIUM** (evidence). Likelihood: uncertain.
- **Where:** `phase1.py:503–514`, `:767`. RUN.md: "No skin gate".
- **Scenario:**
  - T_ref is the skin at the start of the first measured block, which follows the L0 warm-up and a 600 s pause.
  - Blocks after M2 layouts end hotter (rehearsal 5.1–5.8 W against L0's 4.5 W) and cool for the same 600 s.
  - Their start skin can sit more than 1.5 °C above T_ref, and the blocks are relabelled 'NOT COMPARABLE — START TEMP'.
- **Evidence:** p1a_fix cooling curves:
  - After L0 (end 27.9 °C), skin fell to 23.9 °C after 472 s.
  - After L1 (start of the L2 gate 29.9 °C), skin was 25.2 °C after 479 s, still falling at about 0.1–0.15 °C/min.

  This suggests that the post-pause skin tracks the previous block's end skin, with differences of more than 1 °C. That p1a_fix L1 had no M2 load.
- **Fix:** decide in advance whether a bounded post-pause skin gate is acceptable (for example ≤ +5 min, recorded). Otherwise accept that some blocks may be labelled NOT COMPARABLE.

### M8. A hard kill leaves the phone modified, and RUN.md has no recovery steps
- **Class:** D and F. **Severity: MEDIUM.**
- **Where:**
  - `speed_block.py:62` sets `screen_off_timeout 2147483647`, which is restored only in the `finally` at `phase1.py:780–783`.
  - RobotCam is a foreground camera service, stopped only by phase cleanup.
  - llama-server runs with `start_new_session=True` (`coresidency.py:425–426`).
- **Scenario:** after a SIGKILL of python (OOM/LMK, a force-close, a battery cut, or a second Ctrl-C (see L4)):
  - The screen never times out.
  - RobotCam may keep the camera open and keep writing frames.
  - llama-server may keep port 8080, so the next run refuses with 'something already answers'.
  - The wake lock stays held.
- **Fix:** a RUN.md recovery block:
  - `su -c 'settings put system screen_off_timeout <value printed as "screen timeout saved … ms">'`
  - `su -c 'am force-stop com.pixelrobot.robotcam'`
  - `pkill -f llama-server`
  - `termux-wake-unlock`

### M9. Owner-side actions and launch wrappers trip the process guard
- **Class:** C. **Severity: MEDIUM.** It can end the session, but only through a deliberate owner action.
- **Where:** pattern in `runtime.py:81–83`. Exclusions cover only descendants (`runtime.py:55–77,91`). WORKFLOW.md:240 says timed runs start through `oneshot.sh`.
- **Scenario:**
  - Mid-session refusals: opening `~/robot/benchmark/campaign/phase1.py` in `less` or `vim` in another Termux tab, a proot/Debian shell running any node tool, or a Termux:Boot or cron script named `run_*.sh`.
  - Refusals at preflight, which are cheap: launching through `oneshot.sh`, `script -c`, `timeout`, or `sh -c "…phase1.py…"`, because the ancestor matches and its ancestry ends at zygote, which is unreadable.
- **Evidence:** the guard pattern tested here matches `less …/benchmark/campaign/phase1.py` and `sh …/oneshot.sh`. It does not match `bash`, `-bash`, `bash -l` or `tail -f …/owner_session_p1_fix3d.stdout`.
- **Fix:** RUN.md should say: "start with exactly this command, no wrapper; open no other Termux session or editor during the run". Add a note in WORKFLOW.md that campaign runs do not use `oneshot.sh`.

---

### L1. The duration estimate is low
- **Class:** E. **Severity: LOW.**
- **Where:** RUN.md:102, `phase1.py:719`.
- **Evidence:** the fix3c rehearsal file times show each pause+block cycle took about 57 s of overhead beyond its schedule; rest phases add about 10 s (`duration_s` 190.5 for 180 s in fix3c2). Setup took about 1.7 min.
- **Consequence:** the full session takes about **95 min**, not 85–90. The owner might wrongly suspect a hang or stop it early.
- **Fix:** state 90–100 min.

### L2. Evidence is held in memory and re-serialised on the main thread
- **Class:** E. **Severity: LOW.**
- **Where:** `result` keeps every row of every phase.
- **Evidence:** in the fix3c rehearsal, runner PSS in pauses rose from 74 to 83 MiB.
- **Consequence:** the full session adds tens of MiB of runner PSS by the last L0, and each `json.dumps(indent=2)` of a file up to about 10 MB runs between the pause and the next block's setup. Both are small, but they grow over the session. This affects runner-PSS comparability between the first and last L0.
- **Fix:** M1's per-pause files would also fix this.

### L3. VALID ignores camera lateness and the fallback share
- **Class:** F. **Severity: LOW.**
- **Where:** `phase1.py:171` (camera_late is counted but not used), `:233–234`.
- **Consequence:** a block can be VALID with late camera frames, or with M2 running entirely on the fallback photo. The cost is the same because the graph is padded to 32 boxes, but the context differs.
- **Fix:** print `camera_late` and `live_calls`/`fallback_calls` next to each block's label.

### L4. A second signal during cleanup can skip the server stop and the screen restore
- **Class:** D. **Severity: LOW.**
- **Where:** `phase1.py:795–796`. The handler raises `SystemExit` anywhere, including inside the `finally` at `:779–783`. coresidency.main sets `SIG_IGN` first (`coresidency.py:1249–1250`); phase1 does not.
- **Fix:** ignore SIGINT, SIGTERM and SIGHUP at the top of main's `finally`.

### L5. A cleanup failure replaces a BatteryStop
- **Class:** F. **Severity: LOW.**
- **Where:** `phase1.py:598`. `raise RuntimeError('pause cleanup failed')` inside `finally` replaces a BatteryStop or CoresLost.
- **Consequence:** the label becomes 'SESSION INCOMPLETE' rather than 'BATTERY BELOW 25%'. stderr still chains the original.
- **Fix:** raise the cleanup error only if no exception is already propagating.

### L6. The FIX3C self-detection audit omits the guard's own pgrep
- **Class:** C (documentation). **Severity: LOW.**
- **Scenario:** the FIX3C audit says only diagnostics and the owned llama-server match the guard. However, `pgrep -fa <pattern>` matches itself, because the pattern text contains 'claude'. The `shared_check()` at `phase1.py:423` overlaps the shared-check worker, so one pgrep sees the other.
- **Assessment:** this is sound in practice, because the other pgrep is a direct child.
- **Fix:** add pgrep to the self_check_fix3c audit list.

### L7. Main-thread work can run on inference cores
- **Class:** E (measurement hygiene). **Severity: LOW.**
- **Where:** `setup_affinity` sets the main thread to cores 0–7 (`runtime.py:130–131`). Frame JPEG decode and EXIF handling (`robotcam_reader.py:23–25`) and the run_cycle bookkeeping therefore run on any core, including the M2 or YOLO cluster.
- **Fix:** after setup, pin the main thread to the monitor mask (cores 4–7 must remain *allowed* for `check_cores`, not used).

### L8. Small accumulations
- **Class:** E. **Severity: LOW.**
- **Details:**
  - Per-TID root files accumulate in `/data/local/tmp` (about 60 × 20 KB).
  - `rt.imports()` and `cr.make_selector()` prepend paths to `sys.path` on every phase or block.
  - `rest()` `duration_s` includes the start and end overhead (190.5 for 180 s).
- **Fix:** none needed. Optionally name the files by phase role rather than TID.

## 3. Ranked top list (do these before the session)

1. **H1:** move one line in `rest()`. This is the cheapest fix and the one with the most deterministic risk. Margin is 0.31 s in every pause.
2. **H5:** owner procedure (airplane mode, Do Not Disturb with alarms off, no updates, Magisk notifications off, hands off). Optionally add screen and foreground state to the snapshot.
3. **H3 + M6:** run the fix3d rehearsal first with the scene as it will be used, and go only on `rehearsal_pass: true`. Check that the YOLO-640 box counts stay well under 32.
4. **M3 + H2:** remove the redundant rest-loop reads. Add a single bounded, recorded re-read for empty or unparsable reads and for `/health`.
5. **H4:** treat one late or missing frame as a skipped slot (cadence), not a camera death.
6. **M2 / M7:** Local AI and owner decide in advance about L1 overload labelling and the start-temperature comparability expectations.
7. **M1:** atomic JSON writes.
8. **M4 + M5:** carry the cause in messages; add a main-thread watchdog.
9. **M8, M9, L1:** RUN.md wording (recovery steps, no wrappers or other sessions, about 95 min).

## 4. Checked and found sound

- **FIX3D per-thread thermal files.** I enumerated every `cr.root` tag and the thread that uses it in rest(), live_block() and preflight:
  - `'sample'` and `'campaign_pidof'`: memory worker only.
  - `'relate_battery'`, `'campaign_capacity'`, `'campaign_diagnostics'`, `'forcestop'`, `'pidof'`, `'lmk'`: main thread only.
  - Thermal: one file per TID.
  - Persistent shells: one owner each.

  No other tag is shared across threads at the same time. TIDs are unique among live threads.
- **read_dump retry.**
  - It fails closed on rc ≠ 0, a reader error, status ≥ 4, a late retry or exhausted attempts.
  - Raw output is kept, and there is no unbounded loop.
  - Retry rows are copied to `thermal_retries`.
  - `diag.snapshot` and lag_probe use the same reader.
- **Process guard ancestry.** The guard excludes our su clients, a concurrent own pgrep and the server pid, and refuses foreign or unreadable ancestry. Root-side Magisk shells are invisible to the non-root pgrep (hidepid). The fix3 evidence lists only the client pid 1312. This is an assumption the fix relies on.
- **Camera-OFF PSS.** The exact-string match plus a status-checked `pidof_rc=1` is consistent with the archive:
  - The provider was present in all 118 camera-OFF memory samples (fix2, fix3c, fix3c2, fix3 and fix3c rehearsals).
  - The app was present in all camera-ON samples.
- **Validity downgrade.**
  - VALID is set only after run_cycle, the guard, the shared check, `skin_end` and `diagnostics_end`.
  - The `finally` only downgrades: monitor and cleanup errors, thermal rows, LMK, power coverage and memory rows.
  - T_ref comparability is applied after the block file is written, and the file is then rewritten.
  - The warm-up keeps `result_validity`.
- **Bounds on joins and calls.** Joins are bounded (25, 120 and 60 s), and a worker still alive afterwards stops the session. Timeouts are set on:
  - `su` (30–60 s)
  - RootShell (5 s)
  - selector requests (15 s each)
  - `am` (10 s)
  - server start (180 s)
  - server stop (15 s, then SIGKILL)
- **Cadence scheduling.** All schedules use absolute time (`t0+slot`, `monitor_loop`), skipping slots rather than queuing them, so there is no drift over 90 min. Calls that cross the block boundary count as cadence misses.
- **Per-phase release.**
  - Runner PSS returns to 74–83 MiB in pauses after 240–650 MiB blocks.
  - Shells are closed, detector and M2 caller threads are joined, and ORT sessions are freed.
  - ORT worker identification is deterministic, because builds happen before the monitor threads start.
- **Camera publish** is atomic (temp file and rename, `CameraService.kt:484–488`), so torn JPEG reads cannot happen.
- **Live detector boxes** are format-compatible with M2's `preprocess`.
- **Evidence overwrite protection.** Existing stem files are refused at start, `set -C` protects stdout/stderr, and block indices are unique for the repeated L0.
- **Battery margin.** Using the rehearsal watts over the session schedule gives about 3.5–4 Wh (about 20 % of capacity), so starting at ≥ 80 % should not reach the 25 % stop.
- **LMK window parsing** handles `-v epoch` timestamps. Lines outside the window are separated.
- **JSON `allow_nan=False`:** no NaN source was found (logits are checked for finiteness, the HAL parser drops NaN, power values are integers).

## 5. Could not determine

- Magisk `su` failure rate and latency tail over about 4,600 calls in 90 min. Also Magisk's per-request toast and logging settings, which add I/O and latency per call.
- Fuel-gauge sysfs transient error rate (H2), and llama.cpp `/health` latency under full load (not logged).
- Whether the camera provider becomes lazy and exits during a 600 s camera-OFF pause. The longest one observed was 240 s, and it was always present.
- Whether this kernel resets per-thread affinity when Termux briefly leaves top-app. Pinning has survived every RobotCam StartActivity transition so far.
- The phantom-process-killer and child-process-restriction settings on the phone. Free space in `/data/local/tmp` and Downloads.
- The true cause of the fix3c/fix3c2 blank reads. The shared-file race is the likely cause, but its timing is unproven. The retry plus the per-thread files cover both hypotheses within 2 s, but a thermal HAL stall longer than 2 s would still end the session.
- Live box counts in the owner's scene; heat behaviour over 7 × 180 s blocks; whether later blocks fall outside ±1.5 °C of T_ref (M7).
- The selected reasoning-effort level of this audit session (section 0).
