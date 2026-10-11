# CAMPAIGN_P1_FIX4 — Coder report

## 0. Profile and repository state

| Item | Value |
|---|---|
| Role | Coder |
| Platform | Claude Code CLI, interactive (proot Debian on the phone) |
| Model | `claude-opus-5-5` (stated by the harness in this session's system context) |
| Effort | `high` was set by the owner via /model. **Not independently confirmed:** no tool or command in this session exposes the reasoning-effort setting. |
| Ponytail | `lite` — confirmed by the session-start hook message "PONYTAIL MODE ACTIVE — level: lite". |
| `git rev-parse HEAD` (start) | `56296995ea780141d757ba74ee768e07e27c2979` ("benchmark: campaign phase-1 thermal read retry, session archives") |
| `git status --short` (start) | empty |
| Reviewer | `codex --no-daemon exec`, `gpt-6.1-sol`, `model_reasoning_effort="high"`, `-s read-only`, `PONYTAIL_DEFAULT_MODE=off`, fresh isolated copy `/termux-home/review_campaign_p1_fix4/` |
| Fallbacks used | none |

Read in full before changing anything: AGENTS.md, docs/HANDOFF.md, docs/WORKFLOW.md, docs/CODER.md, docs/REVIEWER.md,
docs/CLAUDE.md, the audit report, every file in benchmark/campaign/ (code, RUN.md, RUN_INDEX.md, all six self-checks,
fix3d check outputs and manifest format), and the imported code the audit lists (coresidency.py root/am/camera/
root_sample/lmk/Server/make_selector/RootShell/discover/fast_sample/parse_dump/monitor_loop/block_limit; speed_block.py;
variants.py; power_map.py PinnedDetector/build_detector/camera_start/capped_by_policy; desk_check.py preprocess/Head.infer;
robotcam_reader.py; detect_person.py Detector.detect; adapters.py S1O._post; ladder/measure.py allowed_cpus).

Baseline at HEAD before any change (same proot, agents resident): self_check, fix2, fix3b, fix3c, fix3d exit 0;
**self_check_fix3 failed 3 of 4 runs** (`check_forced_fallback`, a pre-existing timing flake, see §4).

Nothing was committed or pushed. docs/ was not edited. No committed evidence was modified. No model files. Motors off.
No live, timed or root validation was run or is claimed.

## 1. Per finding: verified against base 5629699, fix (final candidate file:line), test

Base line numbers are the audit's; I re-read each site at 5629699. "Agree" means the code does what the audit says.
Candidate line numbers are those of the frozen round-3 candidate.

| Finding | Verified at base | Fix in candidate | Test (self_check_fix4 unless named) |
|---|---|---|---|
| H1 | Agree. phase1.py:539 bracket, :557 `fast_monitor=rt.prepare`, :560-567 threads; live_block builds it first (:379) and brackets at :394. Reproduced: the base `rest()` with a 1.6 s second-shell setup ends "pause power coverage missing". | phase1.py rest() :577; fast shell :620, pin :621, bracket :639, threads :640-652 at once (live_block: :414-417, :431, :441-450) | `check_h1` (base reproduction; fixed order; zero bracket→thread gap for 0.4/1.6/5 s setups) |
| H2 | Agree; only `read_dump` had a retry (runtime.py:210-246). Every listed read raised or failed closed on one blank. | runtime.py:26-93 (`Blip`, `ReadFailed`, `reread`, `retry`, `retries_since`, `counted`, `retry_check`, `CHARGER`, `causes`). Wrapped reads: pgrep :150-182; root mask :184-201; monitor setup :258-277; dumpsys package :392-401; am :403-412; end check + force-stop re-run + 0.5 s pid re-check :418-455; root_sample + pidof :458-494; battery :496-508; /health 2-of-2 :510-513; screen state :520-534; screen-timeout settings :537-554; fast sample :562-576; transient mask readback :583-614; power.py:19-49; diagnostics.py:22-38, :41-85; logcat phase1.py:532-533; lag probe lag_probe.py:27, :60, :104. Per row `retries` (power.py:100, :109; phase1.py counted rows); per phase `read_retries` + `read_retry_check` (blocks :521, pauses/idle :694, setup :880-889, lag probe :104), between-phase capacity checks counted in the phase they guard (`mark`); session list `read_retries` | `check_h2_helper`, `check_h2_reads`, `check_h2_cap`, `check_round1`, `check_round2`; fix3b pid re-check; fix3c audit |
| H3 | Agree. phase1.py:287-288 raises above 32; :84-87 fallback below 2. | phase1.py:86-96 top 32 by `conf`, detector order kept; `truncated_boxes` :101; `max_live_boxes`, `truncated_slots` :255-259; printed in block END lines (block_status :813); required by rehearsal_pass (:796) | `check_h3`, `check_main` |
| H4 | Agree. LiveOps.frame :271-278 returned any non-ok at once; :167-168 raised. | LiveOps.frame :296-305 polls missing/repeat to 1.35 s; run_cycle :176-190 skips the slot (camera_misses, camera_late, frames_skipped, cadence miss) with its own counter `camera_failed` (:149), stops at 3 in a row (:184) or at once on any other status; L2 first-frame wait 15 s (:120) | `check_h4`, `check_round1` (catch-up + miss) |
| H5 | Agree; nothing read screen/foreground state. | runtime.py:516-534 `screen_state` (needs `mWakefulness=`; one re-read; never raises); diagnostics.py:77/:84; failures: block phase1.py:524, pause :609-612/:672, session :940/:943/:959 | `check_h5`, `check_round1`, `check_round2` |
| M1 | Agree. `write` truncated then wrote (:499-500); label 'SESSION STARTING'. | phase1.py write :550-557 (same-dir temp, flush, fsync, os.replace; lag_probe uses it); label `IN PROGRESS — NOT VALID` (:852); `_idle_N`/`_pause_NN_<layout>` files (:840-841) written also on failure (pause_phase :863); refused if present (:843) | `check_atomic_write`, `check_main` |
| M2 | Agree (L1 M2 4.4-5.2 s ≥ 5 s cadence). Owner D1. | §2 | `check_d1` |
| M3 | Agree. :570-571 results discarded; fix3c2 died at :571. Removed, not kept as evidence: the power sampler checks Discharging every 0.37 s, memory rows check battery status, the thermal worker feeds `block_limit`. | phase1.py:653-664 loop without them (comment :653-654) | rest harness fails on any rest-loop `battery_sample`/`dump_check`; fix3d `check_rest` |
| M4 | Agree. | `rt.causes` runtime.py:91-93; pause/block/session messages (phase1.py:362, :656, :662, :697-710, :910, :935); block downgrades keep their cause (:494, :509, :520); cleanup failures keep the first two (:950-957); fast_check keeps fast_sample's error (runtime.py:571-572); `say()` START/END lines (:860-866, :885/:889, :908, :933) | `check_m3_m4_l5_l7`, `check_main`, `check_round1/2` |
| M5 | Agree. A main thread blocked in an uninterruptible FUSE read cannot run the SIGTERM handler; the flushed stdout line still records the event. | heartbeat phase1.py:356-371, armed :440, disarmed :461/:472; pgrep timeout runtime.py:155; termux-wake-lock/unlock timeout 30 s (speed_block.py:54, :76); `make_selector(post)` (coresidency.py:467-478) set before the tokenize calls (phase1.py:331-337) | `check_m5` |
| M6 | Agree (no code). | RUN.md: rehearsal MANDATORY; rehearsal_pass also needs `max_live_boxes`, no phase over the re-read cap and no cleanup error (phase1.py:775-801) | `check_main`, `check_round2` |
| M7 | Agree. Owner D2. | §2 | `check_d2`, `check_main`, `check_round1` |
| M8 | Agree. | RUN.md RECOVERY block | `check_run_md` |
| M9 | Agree (guard pattern matches `oneshot.sh`, `less …/phase1.py`). | RUN.md "How to start"; proposed WORKFLOW.md text §3 | `check_run_md` |
| L1 | Agree. | RUN.md and print (phase1.py:851): about 95–100 min plus up to 25 min of D2 cooling (5 × 300 s; the first pause has no T_ref; the owner's "up to 30 min" bounds it; the reviewer accepted 25) | `check_main`, `check_run_md` |
| L2 | Agree; not requested. Per-phase files (M1) are a partial mitigation. | — | — |
| L3 | Agree. | block END line (block_status phase1.py:813-820) | `check_main` |
| L4 | Agree. | SIG_IGN at the top of main's finally (:947), previous handlers restored (:965-966) | `check_main` |
| L5 | Agree (rest :598; main's finally also overwrote the battery label; preflight raised over an unwinding error). | rest: no cleanup raise while an error is recorded (:697); main: `unwinding` (:948) keeps the label and the propagating exception, preflight raise only without one (:967) | `check_m3_m4_l5_l7`, `check_main`, `check_round2` |
| L6 | Agree; pgrep's argv contains its own pattern. | self_check_fix3c `check_audit`: pgrep (plus the screen-state and pid re-check sources); a foreign pgrep with identical text is refused | fix3c PASS |
| L7 | **Disagree with the literal fix.** `cr.check_cores` (coresidency.py:306-308) reads `measure.allowed_cpus()` (ladder/measure.py:157-165) = `/proc/self/status` = the main thread's mask; pinning the main thread would make every cores check raise CoresLost. | A parked daemon "sentinel" thread kept at all CPUs (refreshed with the main thread in setup_affinity, runtime.py:219, :234-241); in-phase checks `rt.check_cores` on it (runtime.py:244-248; phase1.py:360, :657); `rt.pin_main` with readback (runtime.py:251-255) after setup (phase1.py:417, :621). Preflight keeps `cr.check_cores` (main unpinned). llama-server starts in preflight before any pin. Side effect: `setup_affinity['before']` shows the previous phase's pin. The reviewer judged the design acceptable (live cpuset behaviour remains an assumption). | `check_m3_m4_l5_l7`, `check_h1` |
| L8 | Agree; not requested. | — | — |

H2 classification (what is re-read once, what is not):

| Read | Re-read once on | Never re-read (parsed but bad, also next to a missing field) |
|---|---|---|
| power (persistent shell) | missing lines, empty status, unparsable current/voltage/average | status other than Discharging anywhere, voltage ≤ 0, shell timeout/exit |
| root mask (verify, setup) | missing/malformed Cpus_allowed_list | mask mismatch, inference cores, shell timeout |
| monitor setup | malformed mask line, unparsable `echo $$`, discover 'layout:' line missing, required policies missing (fresh shell) | inference-core mask, shell timeout |
| transient pinned root shell | rc 98 with no mask read back (the command did not run): a Blip for the caller's re-read | a mask read back that differs (rc 98), a failed pin (rc 97) |
| fast sample | wrong line count, empty value | battery ≥ 45 °C (judged first), shell timeout/exit (error kept) |
| pgrep | rc ∉ {0,1}, output disagreeing with rc, malformed line, launch OSError | a listed foreign process (judged first), timeout (10 s) |
| battery (su) | rc ≠ 0 / field count, unparsable, launch OSError | Charging/Full/Not charging/Unknown anywhere in the answer |
| capacity | rc ≠ 0, unparsable | outside 0..100, below 25 % / 80 % (judged before rc) |
| diagnostics nodes | rc ≠ 0, required node unreadable | (thermal part: FIX3D rule) |
| dumpsys package | rc ≠ 0 or no version | — |
| root_sample | battery fields, missing PSS section, rc ≠ 0 | a parsed or embedded non-Discharging status; camera-OFF app absence goes to pidof |
| pidof (camera OFF) | rc ≠ 0 or anything but a clean `pidof_rc=1` | a listed RobotCam pid (judged first); a non-read error propagates (only `ReadFailed` is kept in the row) |
| end check / force-stop / pidof after it | end check on an unreadable mask readback; force-stop rc ≠ 0 (not a timeout): run again with its pidof; unparsable pidof at once; **pid present: re-check after 0.5 s** | force-stop timeout |
| logcat | not ok, launch OSError | — |
| am start/broadcast | non-zero exit, launch OSError | 10 s timeout; no frame in 3 attempts |
| /health | first failure (needs 2 consecutive) | — |
| screen-timeout settings | su failure, empty/non-numeric `settings get` | a readback that differs (checked by Screen) |
| screen state (evidence) | no `mWakefulness=` line or no separator; after the re-read the error is recorded, never raised | — |
| thermal | FIX3D rule kept: an incomplete rc-0 read up to 3 attempts in 2 s | rc ≠ 0, error, status ≥ 4 (a su launch or transient readback failure on a thermal read still stops) |

## 2. Owner decisions D1 and D2

**D1 (audit M2).** `M2_EVERY = dict(L1=10, L3=5, L4=5)` (phase1.py:28). After a 640 frame, M2 starts only when
`slot % M2_EVERY[name] == 0` (:203): L1 runs at 0, 10, …, 170 s = 18 calls per 180 s block; L3/L4 every 5 s
(unchanged, 36); L2 unchanged (before each 20 s selector, 9). The expected-call count that marks a cadence miss uses the
same table (:232). The 640 frames at 5, 15, … s in L1 no longer find M2 busy, so the busy-skip no longer fires there.
The rehearsal's forced fallback slot for L1 moves from 15 s (no longer an M2 slot) to 10 s, the M2 slot whose context the
20 s selector reads (:454); without this, L1 would never take the fallback path and rehearsal_pass would always fail.
Test: `check_d1` — L1 M2 slots exactly 0..170 step 10 with no cadence miss; L2 9, L4 36; on every run the M2 slots equal
the 640 slots on the layout's grid.

**D2 (audit M7).** `rest(..., t_ref=None, cool_max=0)` (phase1.py:571-599). `main()` passes `t_ref=reference` and
`cool_max=times['cool']` to every pause (:893); `reference` is set by `compare_start` on the first measured block, as before,
so the first pause (before it) never extends. `cool` is 300 s in the session, 6 s in the rehearsal, 0.1 s in the mock
(:757-759). The pause loop (:646) continues after the fixed 600 s while the newest thermal-worker skin of the pause
(≤ 4.87 s old) is above `T_ref + 1.5` (`START_BAND_C`, :554, also used by `compare_start`), for at most `cool_max` s more,
checking every ≤ 5 s with exactly the same checks (monitor errors, processes, cores, /health, block limits) while all
samplers keep running. `record['cooling']` (:655-660) holds `t_ref_c`, `target_skin_c`, `max_extra_s`, `extra_s`,
`end_skin_c`, `reached`, `below_band`. If the pause ends still above the target, the block runs and `compare_start`, using
the block's own `skin_start`, labels it **NOT COMPARABLE — START TEMP** as before. A skin more than 1.5 °C below T_ref never
waits (`below_band: true`), and `compare_start` labels the block NOT COMPARABLE. The fixed-600 s `power_summary` and
`caps_by_policy` keep their [0, 600] window; the extension's rows are still recorded. The pause files and END lines show
the extra seconds.
Test: `check_d2` — no T_ref (no extension), cooled after +50..55 s, still above after +300 s (not reached), below the band
(no wait), exactly at the band edge 31.5 °C (no wait), a sensor stop during the extension still stops the pause, and the
three block labels; `check_main` — first pause `t_ref None`, later pauses `t_ref 30.0, cool_max 300` (rehearsal 6).

Difference from the literal text: the D2 decision is made on the pause's newest skin at its end, not on the block's
`skin_start`, which is measured only after about 30–60 s of block setup with the camera on. The block's `skin_start`
still decides comparability.

## 3. Proposed WORKFLOW.md text (M9) — not applied; docs/ untouched

Proposed addition to WORKFLOW.md, section "Timed benchmarks on the phone", after the `oneshot.sh` sentence:

> Exception: the campaign runner (`benchmark/campaign/phase1.py`: preflight, rehearsal, session, lag probe) is **not**
> started through `oneshot.sh` or any other wrapper (`script`, `timeout`, `sh -c`, a `run_*.sh` file). It performs its
> own agent/process guard, idle, thermal and cores checks, and its guard refuses a wrapper whose command line matches
> it. The owner pastes exactly the command in `benchmark/campaign/RUN.md` into one native Termux session, opens no
> other Termux session or editor on the campaign files during the run, and follows that file's owner checklist and
> recovery block. A campaign session starts only after a passing owner rehearsal (`rehearsal_pass: true`).

## 4. Checks (offline mocks / direct scripts; agents resident; proot; NOT VALID for timing; no live validation)

Final run on the frozen round-3 candidate, with `/data/data/com.termux/files/usr/bin/python <script>` from the repo root.
stdout/stderr and `results.json` are in `benchmark/campaign/checks/p1_fix4/`. Each set of checks was rerun after each round of changes.

| Check | Script | Exit | Seconds |
|---|---|---|---|
| self_check | `benchmark/campaign/self_check.py` | 0 | 15.6 |
| self_check_fix2 | `benchmark/campaign/self_check_fix2.py` | 0 | 8.7 |
| self_check_fix3 | `benchmark/campaign/self_check_fix3.py` | 0 | 40.6 |
| self_check_fix3b | `benchmark/campaign/self_check_fix3b.py` | 0 | 11.1 |
| self_check_fix3c | `benchmark/campaign/self_check_fix3c.py` | 0 | 8.2 |
| self_check_fix3d | `benchmark/campaign/self_check_fix3d.py` | 0 | 8.5 |
| self_check_fix4 | `benchmark/campaign/self_check_fix4.py` | 0 | 16.7 |
| dry_run_mock | `benchmark/campaign/phase1.py --dry-run --mock --output /termux-home/robot/benchmark/campaign/checks/p1_fix4/coder_dry_mock_p1_fix4.json` | 0 | - |
| coresidency_units | `benchmark/coresidency/test_coresidency.py` | 0 | 192.4 |
| power_map_units | `benchmark/power_map/test_power_map.py` | 0 | 96.3 |
| duty_cycle_units | `benchmark/duty_cycle/test_duty_cycle.py` | 0 | 7.5 |
| camera_power_units | `benchmark/camera_power/test_camera_power.py` | 0 | 28.2 |
| thermal_char_units | `benchmark/thermal_char/test_thermal_char.py` | 0 | 153.9 |
| speed1_self_check | `benchmark/relate_anything/speed1/self_check.py` | 0 | 5.4 |
| desk2_self_check | `benchmark/relate_anything/desk2/self_check.py` | 0 | 4.3 |
| detector_units | `test_detector_robotcam.py` | 0 | 5.6 |
| mission_units | `test_run_mission.py` | 0 | 4.5 |
| cycle_units | `run_cycle_safety_test.py` | 0 | 2.2 |

`benchmark/campaign/checks/p1_fix4/self_check_fix4.stdout` (FIX4 round-3 outputs): 16 PASS lines, one per check function, and the final PASS line; 4,938 bytes, SHA-256 0558fa229b3393a98890c0c45dd2cb1f5259df0a5f7ec045ee994424d820edba.

Shared modules changed and their importers' checks: `coresidency.make_selector(post=None)` (backward compatible; callers:
campaign only) and `speed_block.Screen` wake-lock timeouts. Every runner importing them was checked: coresidency,
power_map, duty_cycle, camera_power, thermal_char units, speed1 and desk2 self-checks, plus detector, mission and cycle units:
all exit 0. Their outputs equal the FIX3D archive except the coresidency mock smoke run's timings and memory figures and runner hash
(coresidency.py changed), and thermal_char timings.

Existing self-checks were adapted only where FIX4 intentionally changes behaviour: D1 counts, M3 (fix3d `check_rest`), M4 message prefixes, the H2 pid re-check and stricter pgrep, L7/H5 stubs (`phase_stubs`), mock signatures, and the diagnostics capture tag. The pre-existing `self_check_fix3.check_forced_fallback` flake was also fixed: it had failed 3 of 4 runs at base, and now uses a 100 ms virtual second, a 3-virtual-second live M2, and precondition-gated assertions; it then passed 5 of 5. The per-file details are in the review requests and FIX4B_FULL_DIFF.patch.

Mock dry run (`--dry-run --mock`, NOT VALID): exit 0, 0.6 min scheduled, START/END lines for every phase, per-phase files
written; coverage `rehearsal_pass: false` as expected for mocks (no setup, no fallback calls).

## 5. RUN.md (FIX4) — checklist, start rule, rehearsal gate, session, recovery (verbatim excerpt)

The full file is `benchmark/campaign/RUN.md` (also the Fix4 change notes, session plan, load and verification limits).

````markdown
## OWNER CHECKLIST (before the rehearsal and again before the session)

- [ ] Airplane mode ON (the server is 127.0.0.1; no calls, no network updates).
- [ ] Do Not Disturb ON, **alarms and timers OFF**, and no alarm due in the next 2.5 h.
- [ ] Automatic system and app updates paused (Play Store, system update).
- [ ] Magisk superuser notifications (toasts) set to none.
- [ ] 3–10 objects in view of the camera, no shelves, lights ON, nothing moving.
- [ ] Phone upright in its mount, facing the table.
- [ ] Battery ≥ 80 %, charger unplugged.
- [ ] Exit Codex, Claude, AGY, node, editors and every robot/benchmark/model-server process.
- [ ] Screen ON, native Termux in the foreground; then **hands off** until the final label prints.

## How to start (M9)

Open **one** native Termux session and paste **exactly** the command below — **no wrapper**: not
`oneshot.sh`, not `script`, not `timeout`, not `sh -c "…"`, not a `run_*.sh` file. Do not open another
Termux session, `less`, `vim` or any editor on the campaign files while it runs: the process guard refuses
any matching process outside the runner's own tree (a wrapper or an open `phase1.py` matches), and a
refusal mid-session ends it. Use a fresh stem for every invocation, changing it in all filenames.
Commands use shell noclobber (`set -C`) to protect stdout/stderr; the runner refuses existing
JSON/block/pause/idle/warm-up/server evidence. Never truncate, remove or reuse evidence.

1. Owner rehearsal — **MANDATORY before the session**, **NOT VALID**, about **12 minutes** (estimate;
up to 30 s more of D2 cooling). Same checklist and start rule as the session:

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --dry-run --output ~/storage/downloads/campaign/owner_rehearsal_p1_fix4.json > ~/storage/downloads/campaign/owner_rehearsal_p1_fix4.stdout 2> ~/storage/downloads/campaign/owner_rehearsal_p1_fix4.stderr)
```

It runs every live path of the session with real root, battery, sensors,
camera, models and server: full preflight setup; idle 6 s camera OFF, 6 s
camera OFF, 6 s camera ON (no inference); 20 s L0 warm-up; six 6 s pauses
(camera OFF, Gemma loaded, D2 extension at most 6 s); blocks L0,L1,L2,L3,L4,L0 of 25 s each (camera ON);
selector; live M2 slots and the real fallback path, forced on the M2 slot the
20 s selector reads (L2 at 0 s, L1 at 10 s, L3/L4 at 15 s) so every M2 layout also sends
a fallback-scene context to the selector; diagnostics snapshots, power/caps/memory/PSS/LMK;
cleanup and screen restore. Everything is labelled **NOT VALID — REHEARSAL**.
Short phases make cadence misses likely in L2/L3; that is expected and does
not stop it. The stdout ends with a `Coverage:` line (also
`rehearsal_coverage` in the JSON).

**Go / no-go: start the session only if** the final label is exactly
`NOT VALID — OWNER REHEARSAL (live hardware, shortened phases) COMPLETE; inspect rehearsal_coverage`
**and** `"rehearsal_pass": true`. That flag requires setup, 3 idle sub-phases, the warm-up, 6 pauses
and all six blocks without errors; for each of L1–L4 at least one **live** M2 call, one fallback M2
call and one selector call with a fallback-scene context; `max_live_boxes` reported for every block;
and no phase over the re-read cap (`read_retry_over_cap` empty). Live M2 calls 0 in a layout means the
camera saw fewer than two objects; a `max_live_boxes` near or above 32 means too much clutter: fix the
scene (3–10 objects, no shelves) and rehearse again. Any other failure: send the files and do not
start the session.

(File list to send: see RUN.md.)

2. Full session, only after a passing rehearsal. Recharge to **≥ 80 %** if needed, unplug the charger,
run the checklist again. One command, about **95–100 minutes** (86 minutes scheduled plus setup, model
builds, camera transitions and cleanup), **plus up to 25 min of D2 cooling** (at most 300 s after each
of the five pauses that follow the first measured block):

```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --output ~/storage/downloads/campaign/owner_session_p1_fix4.json > ~/storage/downloads/campaign/owner_session_p1_fix4.stdout 2> ~/storage/downloads/campaign/owner_session_p1_fix4.stderr)
```

Progress: `tail -n 3 ~/storage/downloads/campaign/owner_session_p1_fix4.stdout` is safe **after** the
run; during the run do not open a second session (see "How to start"). Each phase prints START/END lines.

(File list to send: see RUN.md.)

## RECOVERY (M8) — only if the runner was killed or hung and never printed its final label

A hard kill (out-of-memory kill, force-closing Termux, a battery cut) skips the runner's cleanup. From
**native Termux**, after any runner process is gone (`pgrep -fa phase1.py` prints nothing):

```sh
su -c 'settings put system screen_off_timeout SAVED_MS'   # SAVED_MS = the number printed in the .stdout as "screen timeout saved SAVED_MS ms"
su -c 'am force-stop com.pixelrobot.robotcam'
pkill -f llama-server
termux-wake-unlock
```

If the saved value printed was already 2147483647, use your normal timeout (for example 60000).
Keep the files as they are and send them; do not rerun under the same stem.
````

## 6. Independent review

Command (each round on a fresh isolated copy: `git archive` of the base plus the frozen candidate files, SHA-256 verified):

```sh
cd /termux-home/review_campaign_p1_fix4[_rN] && PONYTAIL_DEFAULT_MODE=off \
  codex --no-daemon exec --skip-git-repo-check -m gpt-6.1-sol -c model_reasoning_effort="high" \
    -s read-only - < /termux-home/storage/downloads/campaign/review_request_campaign_p1_fix4[_rN].txt \
    > review_campaign_p1_fix4_rN.txt 2> review_campaign_p1_fix4_rN.stderr
```

codex-cli 0.160.0. Stderr (the full tool transcript) is kept beside each review file and not reproduced. Each run printed a sandbox warning that bubblewrap was not on PATH, so codex used its bundled copy.

### Round 1 — request `review_request_campaign_p1_fix4.txt` (95,906 bytes, SHA-256 c26792cad22361d53bd39acb529daa5769ed853519779ef0c0bc3ccb35a23630), copy `/termux-home/review_campaign_p1_fix4/`, codex exit 0

Review text verbatim (`/termux-home/review_campaign_p1_fix4/review_campaign_p1_fix4_r1.txt`, 6,827 bytes):

````text
CHANGES REQUESTED

I reviewed the actual source, audit report, tracked diff, new FIX4 self-check, RUN.md and archived check results. The production files, audit, patch and FIX4 self-check hashes I checked match the supplied fingerprints.

Findings, ordered by severity:

1. **HIGH — H4’s consecutive-camera-failure counter is overwritten by cadence catch-up.** [phase1.py:155](/termux-home/review_campaign_p1_fix4/benchmark/campaign/phase1.py:155) uses `for missed in skipped`, overwriting the counter initialized at line 149. For example, slow detection causes slot 21 to be skipped; the first missing frame at slot 22 increments `missed` from 21 to 22 and stops the session immediately. Use separate variables. The regression at [self_check_fix4.py:347](/termux-home/review_campaign_p1_fix4/benchmark/campaign/self_check_fix4.py:347) returns missing frames immediately and does not test this catch-up interaction.

2. **HIGH — H2 can erase an already parsed battery thermal stop.** At [runtime.py:531](/termux-home/review_campaign_p1_fix4/benchmark/campaign/runtime.py:531), missing CPU/policy fields cause a `Blip` before the battery-stop check at line 538. A first sample containing `bat_c=45` and a missing policy value can be replaced by a complete, cooler reread. The stop disappears, and the block can remain VALID. Check independently parsed stop values before deciding whether incomplete fields justify a reread. Add a regression combining a stop value with missing fields.

3. **HIGH — empty successful process enumeration passes the guard.** [runtime.py:147](/termux-home/review_campaign_p1_fix4/benchmark/campaign/runtime.py:147) accepts `pgrep` rc 0 with empty stdout; the following loop finds no foreign processes. That is an inconsistent successful enumeration, and a transient blank answer can conceal resident agents or runners. Malformed PID lines also bypass the shared reread handling. Validate the output inside the wrapped read, reread once, and fail closed if it remains inconsistent.

4. **MEDIUM — D2 extensions escape power-coverage validation.** [phase1.py:680](/termux-home/review_campaign_p1_fix4/benchmark/campaign/phase1.py:680) summarizes power and caps using the original `duration`, even after 300 seconds of additional cooling. A power gap exceeding 1.5 seconds entirely within the extension does not reject the pause. Record the actual sampled pause interval and validate its full coverage; separate fixed-pause and extension summaries are also reasonable. The constant-power D2 test does not expose this.

5. **MEDIUM — H2 coverage and phase accounting are incomplete.** Concrete omissions include:

   - [runtime.py:509](/termux-home/review_campaign_p1_fix4/benchmark/campaign/runtime.py:509): screen-state dumpsys reads have no reread.
   - [speed_block.py:48](/termux-home/review_campaign_p1_fix4/benchmark/relate_anything/desk2/speed_block.py:48): screen-timeout root reads remain unwrapped.
   - [lag_probe.py:28](/termux-home/review_campaign_p1_fix4/benchmark/campaign/lag_probe.py:28) and line 59: battery and fast reads bypass the campaign wrappers.
   - [phase1.py:866](/termux-home/review_campaign_p1_fix4/benchmark/campaign/phase1.py:866): setup has no phase retry list or cap assessment. Rehearsal coverage excludes setup from its over-cap checks.

   Being evidence-only does not exempt a dumpsys read from the explicit “EVERY … read” requirement. Screen evidence can remain nonfatal after exhausting its reread.

6. **MEDIUM — M4 still loses underlying memory failures.** [phase1.py:513](/termux-home/review_campaign_p1_fix4/benchmark/campaign/phase1.py:513) downgrades a block for failed memory rows without setting `record['error']` or copying their causes. If persistent PSS failures were returned as rows, the main exception at line 910 becomes `block invalidates later blocks: ` with no cause. Copy the first underlying row errors into the block error and resulting session error.

7. **MEDIUM — H5 and failure progress reporting have uncovered paths.** The BatteryStop handler at [phase1.py:913](/termux-home/review_campaign_p1_fix4/benchmark/campaign/phase1.py:913) and session cleanup failures at line 925 omit failure screen state. Setup failure also skips its END line because that line follows successful preflight. These contradict the requested coverage of every failure path and phase start/end.

The remaining requested changes are substantially implemented: H1, H3, M1, M3, the principal M5 changes, M8, RUN.md’s M9 launch restrictions, L3–L7, D1, the checklist and mandatory rehearsal procedure. H2, H4, H5 and M4 remain partial as above. D2’s waiting logic is implemented, but extension coverage needs correction.

L1 differs from the requested wording: 25 minutes is the correct maximum for the default six-block plan because only five pauses have `T_ref`. That correction is acceptable. I found no concrete proposed WORKFLOW.md replacement note in the candidate report material; preserve the audit unchanged and supply that proposal separately.

On the coder decisions:

- **L7 sentinel:** acceptable design for preserving the core-availability check while pinning the main thread. Android cpuset behavior remains a live-validation assumption.
- **Unchanged thermal reader:** acceptable; it preserves the explicitly requested existing thermal rule.
- **Screen state without reread:** not acceptable under H2’s literal scope.
- **Pause skin controlling D2:** reasonable operational interpretation, with the block’s actual start reading still deciding comparability. However, “≤4.87 seconds old” is not guaranteed when reads overrun; document the distinction and evidence age.
- **Heartbeat limited to `run_cycle`:** acceptable for the requested block main-loop watchdog. It does not provide complete hang coverage; camera-start frame reads remain potentially blocking.
- **Shared-module edits:** justified and backward compatible by inspection.
- **Test adaptations and fallback-flake change:** reasonable, but the new regressions miss the combined failure scenarios above.

Retries generally preserve persistent dead-server, affinity, camera-cleanup and thermal-reader failures. Findings 2 and 3 are concrete exceptions that can conceal failures and permit VALID blocks. Finding 1 causes premature session stops.

Session and rehearsal are affected by these findings; preflight has the retry-accounting gaps, and the lag probe receives only partial H2 handling. I found no additional regression in the other shared-module consumers by inspection.

I reviewed the archived passing checks but did not independently rerun them: interpreter probes exited with code 182 without diagnostics. No files were changed, and no hardware, root, models or live runner were invoked. This review grants no commit or push permission.
````

Coder response to round 1: all seven findings were verified in the code and accepted. Finding 1 was a real bug introduced by
my H4 change: the existing catch-up loop's variable `missed` overwrote the consecutive-miss counter. Its regression fails on
the round-1 code ("no new frame in 6 consecutive slots" after a single miss) and passes after the fix. While fixing
finding 2 I also found that a charger word inside `repr()` output follows a literal `\n`, so the regex has no leading `\b`.
Changes (all re-tested; every check rerun):
1. phase1.py `camera_failed` counter (H4).
2. runtime.py fast_check raises a parsed battery stop before any re-read; `CHARGER` keeps a parsed charger status next to a
   missing field from being re-read (power.sample, battery_sample, root_sample).
3. runtime.py clear_processes: pgrep output must agree with its exit status and be well formed; else one re-read, then fail
   closed (self_check.check_runtime updated: rc 0 with empty output now fails).
4. rest(): `power_summary_with_cooling` over [0, 600 + extra] must be complete as well; `cooling.end_skin_age_s` recorded.
5. screen_state re-read once (still never raises); `rt.screen(sb)` routes the screen-timeout settings reads through the
   helper (preflight, lag probe); lag probe battery/fast reads through `rt.battery_sample` / `rt.fast_check`; setup gets
   `result['setup']` (read_retries, read_retry_check), counted by rehearsal_pass.
6. live_block keeps the cause of memory-row, power-coverage and LMK downgrades in `record['error']`.
7. screen state on the BatteryStop path and on a session cleanup failure; setup END line also on failure.
RUN.md Fix4 notes updated (H2 scope, D2 evidence age and full-interval coverage).

### Round 2 — request `review_request_campaign_p1_fix4_r2.txt` (87,057 bytes, SHA-256 59cf730e9e05e4534c4468e9999130a081dd4937d7ff265a26e1db38e9ca7307), copy `/termux-home/review_campaign_p1_fix4_r2/`, codex exit 0

Review text verbatim (`/termux-home/review_campaign_p1_fix4_r2/review_campaign_p1_fix4_r2.txt`, 6,057 bytes):

````text
CHANGES REQUESTED

I reviewed the actual candidate, full tracked diff, new self-check, audit, RUN.md and archived checks. All 71 frozen files and the three supplied review-material fingerprints match.

1. **HIGH — H2 can still reread away an invalid voltage.** [power.py:36](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/power.py:36) parses current and voltage, then parses `current_avg`, before checking voltage at line 43. With `average=True`, `['-1000000', '0', 'Discharging', 'garbage']` produces a `Blip` despite the already-parsed zero voltage. A healthy reread replaces it. My isolated check returned 4 W after two reads. The ordinary sampling path similarly accepts a healthy reread after `['', '0', 'Discharging']`. Validate independently readable bad values before treating other fields as retryable. Add combined-field regressions.

2. **MEDIUM — H2 still misses several empty or unparsable reads.**
   - [runtime.py:534](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/runtime.py:534): the settings wrapper retries exceptions, but accepts empty successful output. `Screen.start()` validates that output afterward and immediately raises `invalid screen timeout: ''`. My check made one read and recorded zero retries. Validate `settings get` output inside the helper; successful `settings put` replies may legitimately be empty.
   - [runtime.py:517](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/runtime.py:517): the screen-state parser accepts the separator alone. `(1, '===\n')` returned empty power/activity fields after one read, with no retry or recorded error. This also leaves H5’s evidence incomplete.
   - [runtime.py:588](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/runtime.py:588): an empty transient `taskset` readback becomes an ordinary `RuntimeError`, bypassing reread. Distinguish unreadable output from a parsed affinity mismatch; the latter must remain immediately fatal.
   - [runtime.py:260](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/runtime.py:260): `missing required CPU frequency policies; not started` from monitor discovery also bypasses reread. The current tests cover discovery’s `SystemExit` variant, but miss this variant.

   I reproduced each bypass with isolated mocks: one attempt, zero retries.

3. **MEDIUM — lag-probe retry accounting and the cap remain incomplete.** [lag_probe.py:59](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/lag_probe.py:59) now uses `fast_check`, but does not wrap its rows with `counted`. Four recovered fast-read failures produced four global retry entries and no row-level `retries`. The probe never calls `retry_check`; [lag_probe.py:84](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/lag_probe.py:84) can consequently set `complete=True` without assessing the phase’s retry cap. Add row accounting and a phase retry assessment. Its existing method-check label already prevents benchmark validity claims.

4. **MEDIUM — rehearsal coverage ignores session cleanup failures, and cleanup reporting can lose causes.** [phase1.py:790](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/phase1.py:790) includes `result['error']` in coverage errors but excludes `cleanup_error`. An otherwise passing fixture with `cleanup_error='stop failed'` returned `rehearsal_pass=True` and `errors=[]`. The final label correctly becomes incomplete, so RUN.md’s requirement to check both label and flag protects the owner; the flag itself is wrong. Also, [phase1.py:945](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/phase1.py:945) overwrites the first cleanup error if both server stop and screen restore fail, and does not populate `result['error']` when cleanup is the only failure. Include cleanup in coverage and preserve the first underlying causes.

5. **MEDIUM — L5 remains broken for preflight.** [phase1.py:957](/termux-home/review_campaign_p1_fix4_r2/benchmark/campaign/phase1.py:957) raises a cleanup error unconditionally when `--preflight` is selected. If preflight is already unwinding a `CoresLost` exception and screen restoration fails, the propagated exception becomes `RuntimeError: preflight cleanup failed: …`. An isolated check of this exact branch confirmed that replacement. Preserve the propagating exception; raise the cleanup error only when no earlier exception is unwinding.

The seven round-1 findings are resolved as follows:

| Round-1 finding | Round-2 assessment |
|---|---|
| 1 — camera counter overwritten by catch-up | Resolved |
| 2 — battery thermal stop reread away | Resolved for that scenario; confirmed one read |
| 3 — inconsistent pgrep enumeration accepted | Resolved; confirmed persistent failures stop after two reads |
| 4 — D2 extension lacks power validation | Resolved |
| 5 — H2 coverage/accounting gaps | Partially resolved; findings 2–3 remain |
| 6 — memory-row cause lost | Resolved |
| 7 — failure screen hooks/setup END gaps | Resolved; screen-content validation still needs finding 2 |

H2, H5’s screen-content handling, M4’s session-cleanup reporting, L5’s preflight path and rehearsal cleanup coverage remain partial. The other requested changes are implemented by inspection: H1, H3, H4, M1, M3, M5, M8, M9, L3, L4, L6, L7, D1 and D2. The checklist and mandatory rehearsal procedure are present. The documented 25-minute D2 maximum remains the acceptable correction to 30 minutes: only five default pauses have `T_ref`.

The voltage finding is a concrete retry that can conceal a bad reading. The isolated checks confirmed persistent dead-server checks fail after two reads and parsed affinity mismatches fail without retry. I found no additional regression in the two shared-module changes or their other consumers by inspection.

I inspected the archived passing checks and ran only read-only, in-memory probes. I did not rerun suites that create evidence, execute the phone runner, invoke hardware/root/models, or change files. This review grants no commit or push permission.
````

Coder response to round 2: all five findings were verified and accepted. Because round 3 is the last, I also swept every
wrapped read for the same two patterns and fixed three more instances:
- a listed foreign process next to a malformed pgrep line was re-read away;
- a listed RobotCam pid next to rc ≠ 0 in the camera-OFF confirmation was re-read away;
- a capacity below the minimum next to rc ≠ 0 was re-read away.

Changes:
1. power.py: each field is parsed on its own; a charger status anywhere and a voltage ≤ 0 in its known position raise
   before any re-read.
2. runtime.py:
   - an empty `settings get` answer is a Blip;
   - screen state needs an `mWakefulness=` line;
   - an unreadable transient mask readback (rc 98 with no mask; the command did not run) is a Blip for the caller's single
     re-read, while a read-back mismatch or rc 97 stays fatal;
   - the end check is re-run after such a Blip;
   - 'missing required CPU frequency policies' is a Blip.
3. lag_probe.py: per-row `retries` on fast rows, `retry_check` from the first read, and > 3 re-reads make the probe incomplete.
4. phase1.py main: every cleanup failure is kept (first two, with source), a cleanup-only failure sets `result['error']`,
   and coverage errors include cleanup.
5. phase1.py main: the preflight cleanup error is raised only when no cause is already recorded (L5).
6. The three sweep fixes above. In addition, the helper's exhausted re-read raises `ReadFailed(RuntimeError)`; only that is
   absorbed into a memory row, so a non-read error propagates as it did before FIX4.
7. The between-phase capacity checks count in the phase they guard (`mark` passed to rest()/live_block()).
RUN.md H2 note updated. Every check rerun: all 17 plus the mock dry run exit 0.

### Round 3 (final allowed) — request `review_request_campaign_p1_fix4_r3.txt` (88,069 bytes, SHA-256 d8f61f660cb2f08af685155dcaada52414c29fc36861e9aa27f95b6ecfafa9b9), copy `/termux-home/review_campaign_p1_fix4_r3/`, codex exit 0

Review text verbatim (`/termux-home/review_campaign_p1_fix4_r3/review_campaign_p1_fix4_r3.txt`, 3957 bytes, SHA-256 c3100122e7cdcd4fa90818a1bd688af216b66cf3fa230610e252e61086bada98):

````text
CHANGES REQUESTED

I reviewed the actual candidate, prior reviews, audit, self-check changes, RUN.md and archived check results. All 71 frozen files match the manifest; the patch, manifest and audit hashes match the supplied fingerprints.

1. **HIGH — a partial fast read can still erase a battery thermal stop.** [coresidency.py:656](/termux-home/review_campaign_p1_fix4_r3/benchmark/coresidency/coresidency.py:656) discards every field when the line count differs. Consequently, [runtime.py:574](/termux-home/review_campaign_p1_fix4_r3/benchmark/campaign/runtime.py:574) retries without seeing an independently readable battery value. My isolated probe supplied the normal seven-field layout with battery `450` and only the final policy line missing. A healthy second response was accepted as **30 °C after two reads**. Preserve readable stop values before discarding malformed output. The round-1 regression mocks an already-parsed row and misses this parser interaction.

2. **HIGH — zero voltage is still re-read away when the response is short.** [power.py:39](/termux-home/review_campaign_p1_fix4_r3/benchmark/campaign/power.py:39) checks voltage only when `len(values) == n`. Both `['-1000000', '0']` in ordinary mode and `['-1000000', '0', 'Discharging']` in average mode bypass that check. My probes accepted a healthy second response as **4 W**, recording one retry. Check the independently readable voltage before the field-count failure. Add shortened-response regressions alongside the existing combined-field tests.

3. **HIGH — retrying the entire camera end check can erase a failed capture-stop result.** [runtime.py:438](/termux-home/review_campaign_p1_fix4_r3/benchmark/campaign/runtime.py:438) repeats the composite check after any `Blip`. In [coresidency.py:172](/termux-home/review_campaign_p1_fix4_r3/benchmark/coresidency/coresidency.py:172), `capture_stopped=False` can already be established before force-stop and pidof run. If force-stop succeeds but pidof’s mask readback is unreadable, the retry observes the now-force-stopped camera and replaces the earlier failure. Using the actual functions with mocked transport and time, my probe recorded **`capture_stopped` results `[False, True]`, two force-stop calls, and accepted cleanup**. Retry individual root reads while preserving the original capture result. Add a regression combining failed capture cessation with a later pidof readback blip.

These cases can conceal failures with fewer than four retries, so the retry cap does not prevent an otherwise healthy block from remaining VALID.

The earlier findings are addressed as follows:

| Earlier finding | Round-3 assessment |
|---|---|
| R1-1, R1-3, R1-4, R1-6, R1-7 | Resolved |
| R1-2 | Original mocked case resolved; actual partial-parser case remains, finding 1 |
| R1-5 | Previously identified omissions addressed; H2 remains incomplete |
| R2-1 | Both reported cases now fail after one read; shortened-response case remains, finding 2 |
| R2-2 through R2-5 | Reported cases resolved; the camera end-check retry adds finding 3 |

By inspection, H1, H3–H5, M1, M3–M5, M8, M9 and L3–L7 are implemented. D1, D2, the owner checklist and mandatory rehearsal procedure are present; coverage requires live M2 in every M2 layout, reported `max_live_boxes`, and no phase over the retry cap. The accepted D2 interpretation and 25-minute default maximum remain documented. The WORKFLOW proposal is supplied separately.

Findings 1–3 affect session/rehearsal behavior; power sampling also affects the lag probe, and camera cleanup affects preflight. I found no additional regression in the optional selector transport or wake-command timeout changes for other consumers.

I inspected the archived passing checks and ran only read-only, in-memory mock probes. I did not rerun evidence-writing suites, run phase1.py, invoke hardware/root/models, or change files. This review grants no commit or push permission.
````

## 7. Outcome: review gate NOT passed after the maximum 3 rounds — stopped

All three round-3 findings are **confirmed** in the frozen candidate. The task allows at most 3 rounds and any code change
after a review needs a new review, so the three were **not fixed**. The working tree is exactly the frozen round-3
candidate: all 71 files match `frozen_manifest_campaign_p1_fix4.json`, review_round 3. Nothing is staged, committed or pushed.
**The candidate must not be used for the owner rehearsal or session as it is.** Each finding can make a block that should
fail look healthy, with fewer than 4 re-reads.

| R3 finding | Confirmed at | Mechanism | Proposed fix (not applied) | Regression to add |
|---|---|---|---|---|
| 1 HIGH: a short fast answer hides a battery stop | `coresidency.fast_sample` coresidency.py:655-657 blanks every value when the line count is wrong; runtime.py:562-576 `fast_check` then re-reads (Blip) | A truncated 1 s answer containing battery 450 (45.0 °C) is replaced by a cooler complete re-read | In `rt.fast_check`, run the same script through the shell itself (or parse the `lines[:5]` the error carries) and, when the answer is short, judge the battery value at its known position (index 3 in `fast_keys` order) before the re-read: ≥ `BATTERY_STOP_C` raises 'battery thermal stop'. A short answer without a readable stop value still gets the single re-read. | the reviewer's probe: seven-field layout, battery `450`, last policy line missing → 'battery thermal stop', 1 read |
| 2 HIGH: voltage 0 in a short power answer re-read away | power.py:39 checks voltage only when `len(values) == n` | `['-1000000', '0']` (or the average form missing its last line) re-read to a healthy 4 W | Check `num(1) <= 0` whenever `len(values) >= 2` (the voltage line comes before status and average, so a short answer keeps its position) | `['-1000000','0']` normal, `['-1000000','0','Discharging']` average → 'invalid battery voltage', 1 read |
| 3 HIGH: re-running the whole end check erases a failed capture stop | runtime.py:438 `reread('camera end check', cr.camera_end_check)` (my round-2 change); `capture_stopped` is computed at coresidency.py:172, before force-stop and pidof | capture not stopped → force-stop OK → pidof hits an unreadable mask readback (Blip) → the re-run sees a stopped camera and passes | Do not re-run the composite. Call `cr.camera_end_check(rootf=…)` with a rootf that turns a readback Blip into `OSError`. camera_end_check then keeps its `capture_stopped` result and records `force_stop_rc=None`, and the existing force-stop/pidof re-run in `stop_camera` repeats only the root reads. The original `capture_stopped=False` still fails cleanup. | capture_stopped False + pidof Blip → cleanup fails ('capture not shown stopped'), 1 end check |

All three are the same class the earlier rounds found: a parsed bad value that a re-read can replace. Two sit in shared parsing
(`fast_sample`, `camera_end_check`) that the campaign wraps from outside, and one is a residual of R2-1. A fresh round on a
fresh copy is needed after the fixes. Everything else was judged implemented by inspection in round 3: H1, H3–H5, M1, M3–M5,
M8, M9, L3–L7, D1, D2, the checklist and the mandatory rehearsal gate.

**Owner decision needed:** authorize a fourth review round (or a follow-up task, e.g. CAMPAIGN_P1_FIX4B) for the three fixes
above, with the same reviewer profile on a fresh isolated copy. Until then: no rehearsal, no session, no commit.

## 8. Commit preparation — NOT proposed

The task proposes a commit only after APPROVED; no round approved, so no commit is proposed. For reference only, a future
commit after an approved review would contain the 71 files in `benchmark/campaign/frozen_manifest_campaign_p1_fix4.json`
plus the manifest itself:
- **Code:** benchmark/campaign/{phase1,runtime,power,diagnostics,lag_probe}.py; benchmark/coresidency/coresidency.py;
  benchmark/relate_anything/desk2/speed_block.py.
- **Self-checks:** benchmark/campaign/self_check{,_fix2,_fix3,_fix3b,_fix3c,_fix3d,_fix4}.py.
- **Docs and archive:** benchmark/campaign/{RUN.md, RUN_INDEX.md, AUDIT1_REPORT.md}.
- **Outputs:** benchmark/campaign/checks/p1_fix4/* (54 files: check outputs, results.json and the mock dry-run evidence).

Proposed message (unchanged from the task): "benchmark: campaign phase-1 audit fixes (bounded re-reads, camera slot skip,
atomic evidence, watchdog), L1 10 s cadence, cooling extension". No `.log` files are involved. Commit only after
"yes commit"; push only after a separate "yes push".

## 9. Files written outside the repository

| Path | Purpose |
|---|---|
| /termux-home/review_campaign_p1_fix4/, _r2/, _r3/ | fresh isolated review copies (base archive + frozen files + review material), with review_campaign_p1_fix4_rN.txt/.stderr |
| /termux-home/storage/downloads/campaign/review_request_campaign_p1_fix4.txt, _r2.txt, _r3.txt | the three review requests (95,906 / 87,057 / 88,069 bytes) |
| /termux-home/storage/downloads/CAMPAIGN_P1_FIX4_CODER_REPORT.md | this report |


---

# FIX4B — re-read safety (owner-approved follow-up, fresh review budget of 3 rounds)

## B0. Scope and state

The owner approved CAMPAIGN_P1_FIX4B on the same uncommitted FIX4 candidate, with a fresh review budget of at most 3
new rounds and the same reviewer profile (codex `--no-daemon exec`, gpt-6.1-sol, `model_reasoning_effort="high"`,
read-only, `PONYTAIL_DEFAULT_MODE=off`, a fresh isolated copy each round). At the start of FIX4B the working tree was exactly
the FIX4 round-3 candidate (all 71 files matched `frozen_manifest_campaign_p1_fix4.json`), and HEAD was still
`56296995ea780141d757ba74ee768e07e27c2979`.
Profile: Claude Code CLI, `claude-opus-5-5`; effort high was set by the owner via /model (the CLI printed "Set model to Opus 5.5 …
with high effort"), and no tool lets me confirm it independently. Ponytail lite (session hook).

## B1. Changes

**The three fixes proposed in §7:**

| Item | Change (final candidate) |
|---|---|
| R3-1 | `rt.Answer` is a shell proxy that keeps the raw answer lines. `fast_check` judges every value it can read in its slot before any re-read, also when `fast_sample` blanked a short answer: a battery stop value (0..1000 tenths °C), and CPU fault inputs (≥ 110 °C; 0..200000 m°C) when the answer is incomplete. |
| R3-2 | `rt.battery_bad`: a charger status anywhere, or a voltage ≤ 0 in its slot, whenever that line is present. Used by power.sample, the su battery read and root_sample rows. A complete answer must say exactly `Discharging`. |
| R3-3 | `stop_camera` calls `camera_end_check` once. A `rootf` turns an unreadable pinned readback into the OSError that `camera_end_check` records itself, so `capture_stopped` is kept. A force-stop is re-run (with its pidof) only when it did not run: a launch failure, or rc ≠ 0 with no output. |

**Sweep fixes (same class), found by the sweep and by FIX4B review rounds 1–2:**
- **Root mask:** `judged_mask` judges every range component of a malformed mask answer, in the monitor checks and in setup (an invalid range, measured cores, an empty or differing complete mask).
- **Capacity:** the first integer of a partial answer is judged for range and minimum before the re-read.
- **Su battery read:** done by the campaign itself (same command and tag as `speed_block.battery_sample`) so its raw fields are judged.
- **PSS battery:** `coresidency.root_sample` exposes its raw battery tokens as a `battery_head` row key (shared change; the raw answer exists only there). A fault there is marked `battery_error` and rejected by the new `phase1.memory_row_bad` predicate in blocks and pauses.
- **Settings:** a screen-timeout readback that differs from the value just set fails at once, also in a failed or partial answer.
- **Logcat:** a failed logcat answer holding any LMK line is never re-read.

Every regression was checked to fail on the code it fixes: the FIX4 round-3 code, or the previous FIX4B round's code.

## B2. Sweep — every reader wrapped by the H2 helper -> test (`self_check_fix4b.py`, 26 tests)

Each first answer is short, partial or malformed and carries a parsed bad value, with a healthy answer queued for the
re-read. Each must fail on the first read with no re-read recorded, or, for memory rows, keep the bad row, which the phase
then rejects.

| Reader (re-read site) | Bad values judged | Tests |
|---|---|---|
| fast sample (`rt.fast_check`) | battery stop, CPU fault input | test_r3_1_fast_short_answer_battery_stop, test_fast_complete_count_battery_stop_with_missing_policy, test_fast_cpu_fault_in_incomplete_answer |
| power (`power.sample`) | charger, voltage ≤ 0, exact status | test_r3_2_power_voltage_zero_short_answer, test_power_charger_short_answer, test_complete_answer_numeric_status |
| su battery (`rt.battery_sample`) | charger, voltage ≤ 0, exact status | test_battery_charger_in_malformed_answer, test_complete_answer_numeric_status |
| root_sample (`rt.memory_sample`) | charger, voltage ≤ 0 | test_root_sample_charger_in_malformed_answer, test_root_sample_voltage_zero_actual_parser, test_memory_battery_fault_rejected_without_retry |
| pidof, camera OFF (`rt.memory_sample`) | listed pid | test_pidof_pid_in_failed_answer |
| pgrep (`rt.clear_processes`) | listed foreign process | test_pgrep_resident_in_malformed_answer |
| root mask (`rt.verify_monitor`) | inference cores, differing, invalid range, bad component | test_root_mask_bad_in_malformed_answer, test_mask_bad_component_in_malformed_line |
| monitor setup (`rt.prepare_monitor`) | inference cores, invalid range, bad component | test_monitor_setup_bad_mask_in_malformed_answer, test_monitor_setup_invalid_range_in_malformed_answer, test_mask_bad_component_in_malformed_line |
| capacity (`diagnostics.battery`) | below minimum, impossible | test_capacity_low_with_rc_failure, test_capacity_low_in_partial_answer, test_capacity_impossible_in_partial_answer |
| screen-timeout settings (`rt.screen`) | readback differing from the value set | test_settings_mismatch_in_failed_or_partial_answer |
| logcat (`phase1.lmk_window`) | any LMK line in a failed answer | test_logcat_kill_in_failed_answer, test_logcat_unparsed_line_in_failed_answer |
| transient pinned readback (`root_mask_scope`) | mask read back that differs | test_transient_mask_mismatch_readback |
| camera end check / force-stop (`rt.stop_camera`) | capture not stopped, force-stop failed with output | test_r3_3_end_check_capture_not_stopped_kept, test_force_stop_error_output_not_rerun |
| thermal (not wrapped; FIX3D rule) | partial stop status | test_thermal_partial_stop_status |

Readers whose answers carry no stop or fault value, each listed by the self-check with its reason:
- dumpsys package: a version string only;
- am start / am broadcast: exit status only;
- /health: owner rule, 2 consecutive failures;
- screen state: evidence only;
- diagnostics nodes: evidence only;
- pidof after a force-stop that ran: owner rule, re-checked once after 0.5 s.

## B3. Checks (final candidate; offline mocks; agents resident; proot; NOT VALID for timing)

All exit 0: self_check, self_check_fix2, self_check_fix3, self_check_fix3b, self_check_fix3c, self_check_fix3d, self_check_fix4, self_check_fix4b, dry_run_mock, coresidency_units, power_map_units, duty_cycle_units, camera_power_units, thermal_char_units, speed1_self_check, desk2_self_check, detector_units, mission_units, cycle_units.

Outputs: `benchmark/campaign/checks/p1_fix4b/`. `checks/p1_fix4/` keeps the FIX4 round-3 outputs, which
`frozen_manifest_campaign_p1_fix4.json` pins. Freeze: `frozen_manifest_campaign_p1_fix4b.json` (129 files).

`checks/p1_fix4b/self_check_fix4b.stdout`: 26 PASS lines, 6 N/A lines with reasons, and the final PASS line (SHA-256 e11f84696987555c0d5e1a247d8671d5f9964e29ebfdca22793fb0bbc6ba4a79).

## B4. FIX4B independent reviews (command as in §6; a fresh isolated copy each round)

### FIX4B round 1: request `review_request_campaign_p1_fix4b_r1.txt` (95,457 bytes, SHA-256 dc312452229ab6e88c3e1ce28e86307932ba7fae1a70925040609c73a35c5d00), copy `/termux-home/review_campaign_p1_fix4b_r1/`, codex exit 0

Review verbatim (`/termux-home/review_campaign_p1_fix4b_r1/review_campaign_p1_fix4b_r1.txt`, 4,491 bytes, SHA-256 937a2245e5f03333ddc9e5a042110662a7080d6a800caeb5f68de53aea1e02f6):

````text
CHANGES REQUESTED

All 129 frozen files match the manifest. The manifest, patches and audit hashes match the supplied fingerprints. All 15 explicitly selected FIX4B in-memory test functions pass, but additional probes expose five gaps.

1. **HIGH — a retry can erase the third CPU fault reading.** [runtime.py:619](/data/data/com.termux/files/home/review_campaign_p1_fix4b_r1/benchmark/campaign/runtime.py:619) preserves battery stops but not readable CPU fault inputs. After two 110 °C rows, I supplied BIG=`110000` with one blank policy, followed by a healthy answer. The reader accepted **52 °C after two reads**, and the actual `block_limit()` returned **None**. A shortened response produced the same result. Preserve the original CPU fault input when another field fails; add a regression with two preceding fault rows.

2. **HIGH — invalid voltage can still be replaced in both other battery readers.** [runtime.py:509](/data/data/com.termux/files/home/review_campaign_p1_fix4b_r1/benchmark/campaign/runtime.py:509) and its battery wrapper at line 545 protect charger statuses without judging voltage. Using the actual shared parsers:
   - `battery_sample`: first answer `(0, '-1000000\n0\n')`, then healthy → **4 W, two reads**.
   - `memory_sample`: first answer with voltage `0` and missing camera-provider PSS, then healthy → **4 W, no remaining error, two reads**.

   The second row passes the phase’s memory predicate. Judge independently readable voltage before retry classification, and add actual-parser regressions for both readers.

3. **HIGH — partial capacity answers can erase a parsed sensor fault.** [diagnostics.py:30](/data/data/com.termux/files/home/review_campaign_p1_fix4b_r1/benchmark/campaign/diagnostics.py:30) preserves only digit tokens below the minimum. Both `101\ngarbage` and `-1\ngarbage`, followed by `90`, were accepted as **90% after two reads**. Those first values fail the existing range check when supplied alone. Parse the first integer and judge its range and minimum before allowing a retry; cover partial impossible values.

4. **MEDIUM — monitor setup can retry away an invalid CPU range.** [runtime.py:117](/data/data/com.termux/files/home/review_campaign_p1_fix4b_r1/benchmark/campaign/runtime.py:117) expands ranges without the shared parser’s range validation. Setup calls it with `expected=None`. A malformed answer containing `Cpus_allowed_list: 3-0` plus another malformed mask line becomes an empty set and a `Blip`; CPU `8` likewise escapes validation on an eight-core device. My probe through the actual `sb.prepare_monitor()` accepted a healthy second shell in both cases, recording one recovered setup retry. Validate ranges and empty masks before retrying.

5. **MEDIUM — settings verification can replace a readable mismatch.** [runtime.py:585](/data/data/com.termux/files/home/review_campaign_p1_fix4b_r1/benchmark/campaign/runtime.py:585) retries setting failures before `Screen` judges the readback. After requesting timeout `2147483647`, either `(1, '60000')` or `(0, '60000\ngarbage')`, followed by the requested value, made the actual `Screen.start()` succeed. Judge readable mismatches during set/restore verification before retrying. The sweep’s “settings carry no bad value” exclusion is incorrect.

**R3-1, R3-2 and R3-3 are resolved as reported.** Their targeted regressions pass against the actual functions. In particular, the camera composite check runs once and retains `capture_stopped=False` through the later root-read failure.

**The sweep is incomplete.** It covers reader names but misses CPU fault inputs, voltage in the battery/PSS readers, invalid capacity and mask ranges, and settings verification mismatches. Each reproduced gap uses only one retry, so the cap cannot prevent acceptance.

H2 therefore remains unresolved. By inspection, the other requested audit fixes, D1, the accepted D2 interpretation, owner checklist and mandatory rehearsal gate remain implemented. I found no additional regression in the shared selector transport or wake-command timeout changes. The remaining gaps affect session/rehearsal checks, preflight, and relevant lag-probe paths.

I ran no hardware, root, models or `phase1.py`. An import-path collision unexpectedly executed speed1’s import-time offline checks, creating and removing temporary files; that exceeded your in-memory-only restriction. All frozen candidate hashes still match. This review grants no commit or push permission.
````

Response: all five findings were reproduced with the actual parsers and fixed. Each reader now judges every value it can read in
its slot. The su battery read moved into the campaign. A `battery_head` key was added to `coresidency.root_sample`. Mask
ranges, settings readbacks and logcat lines are judged. Six regressions were added, each failing on the round-1 code.

### FIX4B round 2: request `review_request_campaign_p1_fix4b_r2.txt` (93,221 bytes, SHA-256 74513228663f88e75a865c97bb66e2a2f82cbe795d71f55e7b45a277e80b874d), copy `/termux-home/review_campaign_p1_fix4b_r2/`, codex exit 0

Review verbatim (`/termux-home/review_campaign_p1_fix4b_r2/review_campaign_p1_fix4b_r2.txt`, 4,644 bytes, SHA-256 fe3972ebb8ec35c56257db3ed30234f7e872950eae190d33994e7016ba5973cc):

````text
CHANGES REQUESTED

All 129 frozen files match the manifest. The manifest, both patches, prior review and audit fingerprints match. The following findings are based on actual source inspection; Python launches returned exit code 182, so I could not execute independent mock probes.

1. **HIGH — retaining invalid memory voltage does not ensure rejection.** [runtime.py:525](/termux-home/review_campaign_p1_fix4b_r2/benchmark/campaign/runtime.py:525) recognizes `battery_bad()` only to suppress retries; it adds no battery failure to the row. The actual `root_sample()` parser accepts voltage `0` with status `Discharging`. With complete PSS, that row passes both phase predicates at [phase1.py:518](/termux-home/review_campaign_p1_fix4b_r2/benchmark/campaign/phase1.py:518) and line 709. With camera OFF and only app PSS missing, successful absence confirmation deletes the sole `pss_error`, producing the same acceptance. Healthy independent power samples leave the block/pause otherwise valid. Explicitly reject or permanently mark the battery fault. Add actual-parser regressions with complete PSS and with the camera-OFF app-only omission; the existing regression depends on missing provider PSS to invalidate the row.

2. **HIGH — logcat retry can erase a readable in-window LMK kill.** [phase1.py:534](/termux-home/review_campaign_p1_fix4b_r2/benchmark/campaign/phase1.py:534) protects only lines with unparsed timestamps. A first result containing `ok=False`, `logcat_rc=1` and `lines=['0.5 lmkd: Kill com.example']`, followed by `ok=True, lines=[]`, is replaced by the second result. For window `[0,1)`, the returned evidence becomes `ok=True, n_kills=0`; the block’s failed-logcat predicate also disappears. Preserve readable adverse events before classifying the query for retry. Add a finite-timestamp kill regression alongside the `nan` regression.

3. **MEDIUM — complete battery answers with numeric status are now accepted.** [runtime.py:135](/termux-home/review_campaign_p1_fix4b_r2/benchmark/campaign/runtime.py:135) exempts numeric third tokens, while [power.py:40](/termux-home/review_campaign_p1_fix4b_r2/benchmark/campaign/power.py:40) checks only that status is nonempty. Consequently, `['-1000000','4000000','123']` returns a 4 W power row with status `123`. The campaign’s su battery reader likewise accepts the corresponding three-token answer. Both previously required `Discharging`. Power integration performs no later status validation, so healthy independent memory reads cannot catch this sample. Restore exact status validation for complete answers and cover ordinary/average power plus su battery reads.

4. **MEDIUM — malformed mask components still bypass the bad-value judge.** [runtime.py:116](/termux-home/review_campaign_p1_fix4b_r2/benchmark/campaign/runtime.py:116) judges a line only when the entire mask matches `MASK_LINE`. A partial line such as `Cpus_allowed_list: 0-5,` contains readable inference cores, but the trailing comma makes it a `Blip`; a healthy second mask replaces it. Likewise, `Cpus_allowed_list: 3-0,` bypasses the new invalid-range check. Inspect independently readable range components before retrying. The sweep currently tests malformed answers containing a separate, fully valid mask line, rather than malformed mask lines containing bad components.

The prior findings assess as follows:

| Finding | Assessment |
|---|---|
| FIX4 R3-1, R3-2, R3-3 | Resolved by inspection; archived targeted regressions pass |
| FIX4B R1-1 CPU fault | Reported cases resolved |
| FIX4B R1-2 voltage | Reported retry cases resolved; phase rejection remains incomplete, finding 1 |
| FIX4B R1-3 capacity | Reported cases resolved |
| FIX4B R1-4 range | Reported cases resolved; partial-line gap remains, finding 4 |
| FIX4B R1-5 settings | Reported cases resolved |

**H2 and the sweep remain incomplete.** Findings 2 and 4 permit replacement after one retry; findings 1 and 3 permit acceptance without any retry. The retry cap therefore cannot prevent these failures.

By inspection, the other requested audit changes, D1, accepted D2 interpretation, owner checklist and mandatory rehearsal procedure remain implemented. I found no additional compatibility regression from the additive `battery_head` key, optional selector transport or wake-command timeouts. The findings affect session/rehearsal validation; battery and mask paths also affect preflight and the lag probe.

I inspected the archived passing checks. I did not run evidence-writing suites, `phase1.py`, hardware, root or models, launch another agent, or change files. This review grants no commit or push permission.
````

Response: all four findings were reproduced and fixed. Finding 3 was a regression I introduced in round 1.
- Memory rows with a battery fault are now marked `battery_error` and rejected by `phase1.memory_row_bad`.
- A failed logcat answer holding any LMK line is not re-read.
- A complete battery answer must say exactly `Discharging`.
- Mask components in a malformed line are judged.

My own sweep also found that a force-stop which failed with output was being re-run; it no longer is. Five regressions were
added, each failing on the round-2 code.

### FIX4B round 3: request `review_request_campaign_p1_fix4b_r3.txt` (93,587 bytes, SHA-256 b0cbe9304a5f07716a9baf007c8b20ccd68965cd9e1dd733f45dbcda46106cba), copy `/termux-home/review_campaign_p1_fix4b_r3/`, codex exit 0

Review verbatim (`/termux-home/review_campaign_p1_fix4b_r3/review_campaign_p1_fix4b_r3.txt`, 6,076 bytes, SHA-256 a38f6f3f3e2cc974cf97f158f548bc1796168b878efce731d5dfeef9bf2a5f06):

````text
CHANGES REQUESTED

All 129 frozen files match the manifest. The manifest, both patches and audit hashes match the supplied fingerprints. All 97 tracked patch hunks match the actual candidate. Six selected existing regression functions pass in isolated memory, but additional probes reproduce six gaps.

1. **HIGH — numeric fault checks use a narrower parser than the readers.** [runtime.py:135](/termux-home/review_campaign_p1_fix4b_r3/benchmark/campaign/runtime.py:135) recognizes voltage only with `-?\d+`, while [power.py:32](/termux-home/review_campaign_p1_fix4b_r3/benchmark/campaign/power.py:32) uses `int()`. A complete answer `['-1000000', '+0', 'Discharging']` therefore returns **0 V, 0 W, no retry**. The reconstructed audited-base reader rejects it as invalid voltage. The actual memory parser likewise accepts `+0`; no `battery_error` is added, and `memory_row_bad()` returns false. Similarly, [diagnostics.py:30](/termux-home/review_campaign_p1_fix4b_r3/benchmark/campaign/diagnostics.py:30) retries partial `+20\ngarbage` and `+101\ngarbage` into **90% after two reads**. Judge numeric tokens using the same conversion as the underlying reader, then enforce the fault conditions.

2. **HIGH — a later pidof failure can erase an already-failed force-stop.** In [coresidency.py:183](/termux-home/review_campaign_p1_fix4b_r3/benchmark/coresidency/coresidency.py:183), the shared exception handler overwrites `force_stop_rc` and its output even when force-stop already returned a readable failure. My actual-function probe supplied `(1, 'Error: unknown package')`, then a pidof readback `Blip`. [runtime.py:503](/termux-home/review_campaign_p1_fix4b_r3/benchmark/campaign/runtime.py:503) classified the overwritten result as a launch failure, repeated force-stop, and **accepted cleanup after four root calls and one retry**. Preserve the original force-stop result and retry only the failed read. The new regression mocks the completed end-check dictionary and misses this parser interaction.

3. **HIGH — a readable LMK kill in a partial logcat header can still be replaced.** [coresidency.py:262](/termux-home/review_campaign_p1_fix4b_r3/benchmark/coresidency/coresidency.py:262) extracts LMK lines only after the separator. A first output containing `logcat_rc=1\n0.5 lmkd: Kill com.example\n`, with the separator missing, retains the kill in `raw_head` but produces `lines=[]`. [phase1.py:545](/termux-home/review_campaign_p1_fix4b_r3/benchmark/campaign/phase1.py:545) then retries it. My probe returned **`ok=True`, `n_kills=0` after two reads**. Preserve readable adverse evidence from partial raw output; add an actual-parser regression alongside the mocked `lines` cases.

4. **HIGH — upper-range filters erase thermal stop/fault readings in shortened answers.** [runtime.py:656](/termux-home/review_campaign_p1_fix4b_r3/benchmark/campaign/runtime.py:656) discards raw battery values above `1000` and CPU values above `200000`. With only the final policy line missing, battery `1001` and CPU `200001` were each replaced by **30 °C battery / 52 °C CPU after two reads**. The complete shared parser interprets those same values as **100.1 °C** and **200.001 °C**, respectively. These exceed the existing stop/fault thresholds. Preserve them or reject them as sensor faults; do not turn them into retryable missing values.

5. **MEDIUM — malformed masks can retry away cores outside the expected mask.** [runtime.py:124](/termux-home/review_campaign_p1_fix4b_r3/benchmark/campaign/runtime.py:124) checks expected-mask differences only when the whole line matches. Using the lag probe’s actual configuration—expected `{0,1,2,3}`, measured `{6,7}`—my first answer `Cpus_allowed_list: 0-4,` was replaced by `0-3`, and verification succeeded with one retry. Core 4 is independently readable and outside the permitted mask. Reject readable components outside `expected`, even when the line is incomplete.

6. **MEDIUM — numeric non-Discharging status remains retryable in incomplete answers.** [power.py:39](/termux-home/review_campaign_p1_fix4b_r3/benchmark/campaign/power.py:39) enforces exact status only at the complete field count, while `battery_bad()` exempts numeric third tokens. In average mode, `['-1000000','4000000','123']`, followed by a healthy four-field answer, was accepted as **4 W after two reads**. The su battery wrapper similarly replaced a four-token answer containing status `123`. Preserve a readable invalid status when another field is missing or extra; the complete-answer regression does not cover these cases.

The earlier findings assess as follows:

| Findings | Assessment |
|---|---|
| FIX4 R3-1, R3-2, R3-3 | Their exact reported cases are resolved. Findings 1, 2 and 4 expose remaining related gaps. |
| FIX4B R1-1 through R1-5 | Reported cases are resolved; broader numeric, thermal and mask protection remains incomplete. |
| FIX4B R2-1 through R2-4 | Reported cases are resolved; findings 1, 3, 5 and 6 expose additional parser/partial-answer cases. |

**H2 and the FIX4B sweep remain incomplete.** The sweep lists the relevant readers, but misses the cases above. They require zero or one retry, so the cap cannot prevent an otherwise healthy block or pause from remaining VALID. The claim that accepted answers are the audited base’s answers or a strict subset is disproved by the complete `+0` power sample.

By inspection, the other requested audit changes, D1, the accepted D2 interpretation, owner checklist and mandatory rehearsal procedure remain implemented. I found no additional compatibility regression in the optional selector transport, additive `battery_head` field or wake-command timeouts. These findings affect session/rehearsal validation; applicable battery, thermal, mask and camera paths also affect preflight and the lag probe.

I inspected the archived passing checks and ran only read-only, in-memory probes. I did not run `phase1.py`, evidence-writing suites, hardware, root or models, launch another agent, or change files. This review grants no commit or push permission.
````

## B5. Outcome: FIX4B NOT APPROVED after its 3 new rounds — stopped

All six round-3 findings were **confirmed** against the frozen candidate with the actual functions:

| # | Confirmed result |
|---|---|
| 1 | `power.sample(['-1000000','+0','Discharging'])` returns voltage 0, so the answer is **accepted**. `d.battery` turns `+20\ngarbage` into 90 %. |
| 2 | Force-stop fails with output `Error: unknown package`, then the pidof readback fails. The re-run is accepted after 4 root calls. |
| 3 | A partial logcat answer (`logcat_rc=1`, a kill line, no separator) is re-read to `ok=True, n_kills=0`. |
| 4 | Battery `1001` in a short fast answer is re-read to 30 °C; CPU `200001` the same. |
| 5 | Mask `0-4,` against expected `{0,1,2,3}` is re-read away. |
| 6 | Status `123` in an average-mode answer missing a line is re-read to 4 W. |

**Finding 1 is a regression I introduced.** The audited base rejected voltage ≤ 0 after `int()`. My move of `power.sample`
to the shared judge in FIX4B round 1 dropped that post-parse check, and the judge's `-?\d+` regex misses `+0`. The working
tree is exactly the frozen FIX4B round-3 candidate (`frozen_manifest_campaign_p1_fix4b.json`, review_round 3). Nothing is
staged, committed or pushed. **Do not use this candidate for the owner rehearsal or session.**

**What six review rounds show (coder assessment).** The FIX4B approach keeps hitting the same pattern. The campaign tries to
re-judge values inside answers that the shared parsers reject, with its own regexes, range filters and slot positions. Each
round, a reviewer found another place where that second parser disagrees with the real one: `+0`, an over-range value, a
partial header, a component outside the expected mask. More patches of the same kind are unlikely to converge.

**Recommended next step (owner decision; not implemented).** Narrow H2's re-read to answers that carry nothing at all:
- an empty answer;
- a su, am or pgrep launch failure (OSError, or rc ≠ 0 with no output);
- an unreadable transient mask readback;
- an HTTP timeout, plus /health's 2-of-2 rule and the 0.5 s pid re-check, both owner rules.

Every partial, malformed or unparsable answer would then fail closed exactly as in the audited base. That is the class of
blank read the audit observed (fix3c and fix3c2 were blank thermal reads; thermal keeps its FIX3D rule). With a non-empty
answer never re-read, no parsed bad value can be replaced. The per-slot judging code (`Answer`, `battery_bad`, the fast
slots, `judged_mask`'s components, the capacity token, the settings and logcat special cases) could then be deleted.
Readers keep their base parsers and post-parse checks, which also restores power.py's voltage check.

The cost: a partial-but-non-empty transient (e.g. one blank sysfs line among seven) stops the phase, as it did before FIX4.
The H2 wording "empty/unparsable output" would be narrowed to "empty output". A new task with a fresh review budget would need
to say whether that narrowing is acceptable.

## B6. Commit — NOT proposed

The owner allowed a commit proposal only after APPROVED, and no FIX4B round approved. No commit is proposed. Commit and
push remain gated on "yes commit" and a separate "yes push".

## B7. Files written outside the repository (FIX4B)

| Path | Purpose |
|---|---|
| /termux-home/review_campaign_p1_fix4b_r1/, _r2/, _r3/ | fresh isolated review copies (base archive + 129 frozen files + review material), with review_campaign_p1_fix4b_rN.txt/.stderr |
| /termux-home/storage/downloads/campaign/review_request_campaign_p1_fix4b_r1.txt, _r2.txt, _r3.txt | the three FIX4B review requests |
