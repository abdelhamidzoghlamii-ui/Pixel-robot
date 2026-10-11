# CAMPAIGN_P1_FIX4 — Coder report (FIX4 → FIX4D)

## 0. Profile and state

| Item | Value |
|---|---|
| Role | Coder (Claude Code CLI, proot Debian on the phone) |
| Model | `claude-opus-5-5` (harness context); effort `high` set by the owner via /model, not independently confirmable |
| Ponytail | `lite` (session-start hook message) |
| Base | `56296995ea780141d757ba74ee768e07e27c2979` (FIX3D; the audited commit); nothing committed or pushed |
| Validation | offline mocks and direct scripts only; agents resident; proot cannot reach root: NOT VALID for timing, no live claim |

docs/ was not edited; no committed evidence was modified; no model files; motors off.

## 1. FIX4 and FIX4B — summary (full detail moved)

The detailed FIX4 and FIX4B record is in `/termux-home/storage/downloads/CAMPAIGN_P1_FIX4B_CODER_REPORT_full.md` (79,414
bytes, SHA-256 86d2650ce987303964a4e57a3ff902932c4ffab7ecb4a13103d32716b63936a1; unchanged since FIX4B). It holds:
- per-finding verification at the base with file:line;
- D1/D2 detail, checks and RUN.md excerpts;
- all six review rounds, with verbatim reviews, and the coder responses.

| Finding | State in the candidate |
|---|---|
| H1 | both root shells are built before the power bracket in idle phases and pauses; the bracket, time origin and threads start together |
| H2 | bounded re-read, narrowed in FIX4C/FIX4D (§C, §D): one re-read only for an answer that carried nothing; per-row and per-phase records; > 3 per phase = NOT VALID — READ RETRIES |
| H3 | over 32 live boxes: the top 32 by confidence (detector order kept); `truncated_boxes`, `max_live_boxes` |
| H4 | a missing or repeated frame skips the slot (polled to 1.35 s); 3 failed slots in a row, or bad/other_session, stop |
| H5 | `screen_state` in every diagnostics snapshot and on every failure path |
| M1 | atomic writes (temp file, fsync, rename); `IN PROGRESS — NOT VALID` label; per-phase idle/pause files |
| M2, M7 | owner decisions D1, D2 (below) |
| M3 | the rest loop's own battery and thermal reads are removed; the sampler, memory rows and thermal worker cover them |
| M4 | the first two causes go into the pause, block and session errors; cleanup causes are kept |
| M5 | main-loop heartbeat watchdog (self-SIGTERM); pgrep 10 s and termux-wake 30 s timeouts; selector bound from its first tokenize |
| M6, M8, M9 | RUN.md: mandatory rehearsal gate (`rehearsal_pass`), owner checklist, recovery block, no-wrapper start rule |
| L1 | about 95–100 min, plus up to 25 min of D2 cooling (5 × 300 s; the first pause has no T_ref) |
| L3, L4, L5 | block END line; signals ignored in the final cleanup; a cleanup error never replaces a battery-stop or cores-lost cause |
| L6, L7 | the guard's own pgrep is audited (fix3c); main pinned to the monitor cores, cores checked on an unpinned sentinel thread |
| L2, L8 | not requested |

- **D1:** L1 runs M2 after every second 640 frame (18 calls per 180 s).
- **D2:** after a fixed pause, a start skin more than 1.5 °C above T_ref extends the pause, up to 300 s. Below
  T_ref − 1.5 °C the block is NOT COMPARABLE, with no wait.

**Reviews.** Both earlier tasks stopped at their limit of 3 rounds:
- FIX4: CHANGES REQUESTED in all three rounds (7, 5 and 3 findings).
- FIX4B: CHANGES REQUESTED in all three rounds (5, 4 and 6 findings). Its B5 recommendation to narrow H2 became FIX4C.

Proposed WORKFLOW.md text (M9). It is not applied, and docs/ is untouched. It would go in "Timed benchmarks on the phone",
after the `oneshot.sh` sentence:

> Exception: the campaign runner (`benchmark/campaign/phase1.py`: preflight, rehearsal, session, lag probe) is **not**
> started through `oneshot.sh` or any other wrapper (`script`, `timeout`, `sh -c`, a `run_*.sh` file). It performs its
> own agent/process guard, idle, thermal and cores checks, and its guard refuses a wrapper whose command line matches
> it. The owner pastes exactly the command in `benchmark/campaign/RUN.md` into one native Termux session, opens no
> other Termux session or editor on the campaign files during the run, and follows that file's owner checklist and
> recovery block. A campaign session starts only after a passing owner rehearsal (`rehearsal_pass: true`).


# FIX4C — narrowed re-read (owner-approved, fresh review budget of 3 rounds)

## C0. Scope and state

Same uncommitted candidate on main 5629699. The owner narrowed H2 (my B5 recommendation): a re-read happens only when the
answer carried nothing (empty output; su/am/pgrep launch failure — `OSError`, or rc != 0 with empty output; an empty
transient mask readback; an HTTP timeout). `/health` 2 of 2, the 0.5 s pid re-check and the FIX3D thermal rule are kept.
Every non-empty answer is judged once by the reader's 5629699 parser and post-parse checks.
The full report as it stood after FIX4B is kept as `CAMPAIGN_P1_FIX4B_CODER_REPORT_full.md`. To stay under 80 KB, the six
verbatim review texts in §6 and B4 were replaced by their file paths and SHA-256; the text above is otherwise unchanged.

## C1. Changes

- `runtime.py`: removed `Answer`, `battery_bad`, `CHARGER`, `MASK_LINE`, `judged_mask`, `am()`, `pidof_problem`, the
  fast-slot judging, and every other judging of non-empty answers. Added three emptiness helpers that judge no value:
  - `Heard`: a persistent-shell proxy that remembers only whether the last answer was blank; `last` is None when the run
    raised, so a shell timeout never counts as empty.
  - `blank`.
  - `blank_repr`: base error messages ending with the repr of the whole answer.
  Each reader is its base parser plus one emptiness guard: `clear_processes`, `root_mask`/`verify_monitor`,
  `prepare_monitor`/`prepare` (setup), `camera_version`, `camera_start`, `stop_camera`, `memory_sample` (root_sample,
  pidof), `battery_sample`, `screen`, `screen_state`, `fast_check`, `root_mask_scope`.
- `power.py`, `diagnostics.py` (`battery`, `snapshot`), `phase1.py` (`lmk_window`, `memory_row_bad` = base predicate):
  base parser plus the guard. `power.sample`'s parser is byte-identical to 5629699, including the voltage check after
  `int()`.
- `coresidency.py`: `battery_head` removed. Remaining changes: the M5 `make_selector(post=None)`, plus one
  additive key `answer_blank` in `root_sample` and `lmk_lines` (round-1 fix). Only these functions see the complete su
  answer, and emptiness must be judged from the complete answer. `speed_block.py`: unchanged since FIX4 (wake
  timeouts).
- Kept from FIX4B: R3-3 (the end check runs once) and the preserved force-stop result. Round 2 also made each `stop_camera`
  root read stand on its own answer: a force-stop re-run never re-reads the pidof.
- New: `screen_state` records an error for a separator-only answer. This is its own post-parse check (H5 evidence), not a
  re-read.
- Tests:
  - `self_check_fix4c.py` (new, §C2).
  - `self_check_fix4.py`: the H2 read tests that encoded re-reads of non-empty answers moved to fix4c. Its lag-probe mock
    feeds empty shell answers, and its logcat stub has a `raw_head`.
  - `self_check_fix4b.py` is superseded and not in the candidate.
- `RUN.md`: H2 text rewritten; FIX4C bullet; run stems `owner_rehearsal_p1_fix4c` and `owner_session_p1_fix4c`.

## C2. Invariant — every wrapped reader (`self_check_fix4c.py`; actual parsers, only the transport mocked)

For each reader:
- its good answer and every listed non-empty answer (partial, malformed, unparsable, bad) is read exactly once with no
  re-read;
- each empty answer is re-read once and recovers on a good re-read;
- it fails closed (`ReadFailed`, or the reader's own failure or bad row) on an empty or bad re-read.

`check_all_wrapped` parses the campaign sources for every `reread`/`retry` call name (19) and asserts each one is swept.

- power (power.sample): 8 non-empty answers read once; 2 empty answers re-read once (recovered / failed closed when empty or bad again)
- fast sample (rt.fast_check over cr.fast_sample): 8 non-empty answers read once; 2 empty answers re-read once (recovered / failed closed when empty or bad again)
- root mask (verify_monitor over sb.root_mask): 9 non-empty answers read once; 2 empty answers re-read once (recovered / failed closed when empty or bad again)
- monitor setup (rt.prepare over sb.prepare_monitor): 9 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
- pgrep (clear_processes): 8 non-empty answers read once; 4 empty answers re-read once (recovered / failed closed when empty or bad again)
- battery (rt.battery_sample over sb.battery_sample): 6 non-empty answers read once; 4 empty answers re-read once (recovered / failed closed when empty or bad again)
- root_sample (rt.memory_sample over cr.root_sample): 6 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
- pidof (camera-OFF app absence in rt.memory_sample): 5 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
- battery capacity (diagnostics.battery): 9 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
- diagnostics nodes (diagnostics.snapshot): 4 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
- dumpsys package (camera_version): 4 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
- logcat (phase1.lmk_window over cr.lmk_lines): 8 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
- settings (rt.screen over sb.Screen.setting): 4 non-empty answers read once; 4 empty answers re-read once (recovered / failed closed when empty or bad again)
- dumpsys power/activity (screen_state): 5 non-empty answers read once; 3 empty answers re-read once (recovered / failed closed when empty or bad again)
- am start RobotCam (camera_start over pm.camera_start): 3 non-empty answers read once; 1 empty answers re-read once (recovered / failed closed when empty or bad again)
- am broadcast STOP (stop_camera): 3 non-empty answers read once; 1 empty answers re-read once (recovered / failed closed when empty or bad again)
- transient mask readback (root_mask_scope): 5 non-empty answers read once; 2 empty answers re-read once (recovered / failed closed when empty or bad again)
- am force-stop / pidof (stop_camera over the real camera_end_check): non-empty answers read once (rc 1 with output, malformed pidof, timeout); a force-stop that did not run is re-run once with its pidof; an empty pidof is read once more; a remaining pid is re-checked once after 0.5 s; capture_stopped=False is kept
- /health: one failure read once more (2 of 2 rule), two fail; thermal reader identical to 5629699 (AST), a partial status-4 answer fails on the first read
- every wrapped reader swept (19 re-read call names): /health, RobotCam pidof after force-stop, am broadcast STOP, am force-stop, am start RobotCam, battery, battery capacity, diagnostics, dumpsys package, dumpsys power/activity, fast sample, logcat, monitor setup, pgrep, pidof, power, root mask, root_sample, settings
- every review-round probe: each non-empty answer fails closed (or is judged as in 5629699) on its first read; each answer that carried nothing is re-read once and fails closed when empty again

## C3. Probes of every review round (FIX4 R1–R3, FIX4B R1–R3, FIX4C R1–R2), re-run on the final candidate

Each first answer has a healthy answer queued behind it.

Table: superseded by the FIX4D table (§D3), which re-runs every one of these probes on the final code.

Two probes (FIX4B R2-1, R3-1: `root_sample` with voltage 0/+0 and complete PSS) are read once and **accepted, as in
5629699**: that base parser has no voltage check. See §C4.

## C4. Per-reader diff against 5629699

Remaining behaviour differences of each reader against 5629699 (function-level diff: `FIX4C_READER_DIFF_VS_5629699.txt`).
Common to every wrapped reader: an answer that carried nothing is read once more after 0.3 s, recorded in `read_retries`
(per row and per phase; > 3 per phase = NOT VALID — READ RETRIES); if the re-read carries nothing again the read fails
(`ReadFailed`, or the reader's own failure on the empty value). Nothing else is re-read.

| # | Reader | Difference from 5629699 | Reason |
|---|---|---|---|
| 1 | `power.sample` | blank answer (all lines empty or none) re-read; parser and checks byte-identical (incl. voltage after `int()`) | H2 |
| 2 | `rt.fast_check` (`cr.fast_sample`) | shell answered no non-blank line -> re-read (a shell timeout/dead shell is not an answer: fails as before); error text now carries `fast_sample`'s error | H2; M4 |
| 3 | `verify_monitor` / `sb.root_mask` | mask read that answered nothing -> re-read | H2 |
| 4 | `rt.prepare` (`sb.prepare_monitor`) | a setup failing right after an answer with no output (`echo $$`, mask, discover) -> one fresh setup (the failed one closes its shell, as before) | H2 |
| 5 | `clear_processes` (pgrep) | 10 s timeout (`TimeoutExpired` fails, not re-read); rc 0 or rc >= 2 with no output at all (stdout and stderr blank) -> re-read, then fails closed (5629699 accepted rc 0 + empty output as "nothing resident"); rc 1 = no match as before; rc 0 with no process line but stderr text is judged as in 5629699 (accepted) | M5; H2 / FIX4 R1-3 |
| 6 | `rt.battery_sample` (`sb.battery_sample`) | `root battery read failed` on a blank answer, or a su launch `OSError` -> re-read | H2 |
| 7 | `rt.memory_sample` (`cr.root_sample`) | whole su answer blank (`answer_blank`, set by `cr.root_sample` from the complete answer, never from the 200-char excerpt) or a launch `OSError` -> re-read; persistent empty -> the row stays bad (phase NOT VALID, as before) | H2 |
| 8 | camera-OFF pidof in `memory_sample` | blank answer or launch `OSError` -> re-read; if it fails again it is written into the row (`app absence unconfirmed: ...`, row bad) instead of raising (5629699: a blank answer was already "unconfirmed"; an `OSError` propagated) | H2: both fail the phase |
| 9 | `diagnostics.battery` (capacity), `diagnostics.snapshot`, `camera_version` | blank answer -> re-read; snapshot also records `screen` | H2; H5 |
| 10 | `phase1.lmk_window` (`cr.lmk_lines`) | whole su answer blank (`answer_blank`, from the complete answer) -> re-read | H2 |
| 11 | `rt.screen` (`sb.Screen.setting`) | `screen setting failed: ` with no output, or an empty `settings get` -> re-read | H2 / FIX4 R1-5 |
| 12 | `rt.screen_state` | new evidence-only reader (H5): blank -> re-read; no separator or nothing around it -> `error` recorded; never raises | H5 |
| 13 | `rt.camera_start`, STOP broadcast | an `am` launch `OSError` -> the call once more (for `camera_start` that is `pm.camera_start` again, with its own 3-attempt frame loop, also when the `OSError` came from its in-loop STOP broadcast); a non-zero `am` exit (`CalledProcessError`) or timeout fails as before (`am` output is discarded, so it cannot be shown empty) | H2 |
| 14 | `rt.stop_camera` | `camera_end_check` runs once (R3-3: `capture_stopped=False` stays); a force-stop that ran keeps its own rc/output even if the later pidof fails (5629699: the shared `except` overwrote it with rc None, which also failed); a force-stop that did not run (launch failure, empty pinned readback, rc != 0 with no output) is run once more ALONE (the pidof's own answer is never replaced, FIX4C R2-1); a pidof that never ran (the force-stop raised first) is read, and one that did not run or answered nothing is read once more; a pid listed by a successful pidof (root rc 0, pidof rc 0) is re-checked once after 0.5 s (owner rule; a failed pidof stands). 5629699 failed all four at once | H2, owner rules, FIX4B R3-3 / R3-2 |
| 15 | `root_mask_scope` | rc 98 whose whole answer is the marker line with an empty `actual=` (no command ran) -> Blip for the caller's re-read; any non-empty readback or rc 97 fails as before | H2 |
| 16 | `rt.server_ok` | `/health` fails only on 2 consecutive failures | owner rule |
| 17 | `read_dump`, `dump_once`, `thermal_failure`, `dump_check` | identical (AST-checked in `self_check_fix4c.py`) | FIX3D rule kept |
| 18 | `coresidency.root_sample`, `cr.lmk_lines` (shared) | one additive key `answer_blank` (= not answer.strip()); no other change; their own consumers ignore it | FIX4C R1-1/R1-2: emptiness must come from the complete answer, which only these functions see |
| 19 | `lag_probe.py` | fast rows through `rt.fast_check` (5629699 stored `cr.fast_sample` rows unjudged: a missing reading now fails the probe), battery/screen through the campaign wrappers, per-row `retries`, probe cap | FIX4 R1-5 / R2-3 |

Known base behaviour kept on purpose (not an H2 path, not an audit finding): `cr.root_sample` and `sb.battery_sample`
accept a complete answer with voltage 0 or `+0` (their 5629699 parsers have no voltage check; `power.sample` rejects it).
Under FIX4C these answers are read once and judged as in 5629699 (probes FIX4B R2-1 and R3-1 in the table). Adding a
voltage check would be a new parser rule; I left it for an owner decision.

## C5. Checks (final candidate; offline mocks; agents resident; proot; NOT VALID for timing)

Outputs: `benchmark/campaign/checks/p1_fix4c/` (`results.json`, one `.stdout`/`.stderr` per check). Manifest
`frozen_manifest_campaign_p1_fix4c.json` (74 files). It excludes the superseded uncommitted intermediates:
`checks/p1_fix4/`, `checks/p1_fix4b/`, the fix4/fix4b manifests and `self_check_fix4b.py`. They remain untracked in the
working tree, and I will delete them on your word.

| Check | Command | Exit |
|---|---|---|
| self_check | `benchmark/campaign/self_check.py` | 0 |
| self_check_fix2 | `benchmark/campaign/self_check_fix2.py` | 0 |
| self_check_fix3 | `benchmark/campaign/self_check_fix3.py` | 0 |
| self_check_fix3b | `benchmark/campaign/self_check_fix3b.py` | 0 |
| self_check_fix3c | `benchmark/campaign/self_check_fix3c.py` | 0 |
| self_check_fix3d | `benchmark/campaign/self_check_fix3d.py` | 0 |
| self_check_fix4 | `benchmark/campaign/self_check_fix4.py` | 0 |
| self_check_fix4c | `benchmark/campaign/self_check_fix4c.py` | 0 |
| coresidency_units | `benchmark/coresidency/test_coresidency.py` | 0 |
| power_map_units | `benchmark/power_map/test_power_map.py` | 0 |
| duty_cycle_units | `benchmark/duty_cycle/test_duty_cycle.py` | 0 |
| camera_power_units | `benchmark/camera_power/test_camera_power.py` | 0 |
| thermal_char_units | `benchmark/thermal_char/test_thermal_char.py` | 0 |
| speed1_self_check | `benchmark/relate_anything/speed1/self_check.py` | 0 |
| desk2_self_check | `benchmark/relate_anything/desk2/self_check.py` | 0 |
| detector_units | `test_detector_robotcam.py` | 0 |
| mission_units | `test_run_mission.py` | 0 |
| cycle_units | `run_cycle_safety_test.py` | 0 |
| dry_run_mock | `benchmark/campaign/phase1.py --dry-run --mock --output /termux-home/robot/benchmark/campaign/checks/p1_fix4c/coder_dry_mock_p1_fix4c.json` | 0 |

## C6. FIX4C independent reviews

Each round used the §6 command on a fresh isolated copy, `/termux-home/review_campaign_p1_fix4c_rN/`: `git archive` of the
base plus the 74 frozen files, every SHA-256 verified. Each request carried "Ponytail mode: off", the owner decision
verbatim and the FIX4B round-3 review verbatim, and asked two questions: can any re-read replace a non-empty answer, and is
any audit finding unresolved? From round 2 on, it also carried the earlier FIX4C reviews verbatim and the delta since the
last round. Review texts are kept beside each copy; stderr holds the full tool transcript.

| Round | Request (bytes, SHA-256) | Manifest SHA-256 | Review file (bytes, SHA-256) | Verdict |
|---|---|---|---|---|
| 1 | 30,228, `7b69fdd3c31e87d6…` | `9855553727e78496…` | `/termux-home/review_campaign_p1_fix4c_r1/review_campaign_p1_fix4c_r1.txt` (4,443, `f6ee90ea1c604411…`) | CHANGES REQUESTED (5 findings) |
| 2 | 38,692, `84091842847b5641…` | `bb8371b230c7a3d7…` | `/termux-home/review_campaign_p1_fix4c_r2/review_campaign_p1_fix4c_r2.txt` (2,247, `22517c42784b0342…`) | CHANGES REQUESTED (1 finding) |
| 3 | 42,987, `77d4aea62207f26c…` | `1f755031997a3395…` | `/termux-home/review_campaign_p1_fix4c_r3/review_campaign_p1_fix4c_r3.txt` (912, `d33d018762d8807b…`) | **no review**: sandbox blocker (see below) |

**Round 1, five findings, all reproduced and fixed.** The common root: emptiness was decided from a partial view of the
answer instead of the whole raw answer. Each fix has a regression and a probe row (FIX4C R1-1…R1-5 in §C3).
1. HIGH — `memory_sample` judged emptiness from the 200-char `pss_error` excerpt. Fix: `root_sample.answer_blank`.
2. HIGH — `lmk_window` judged emptiness from `raw_head` and the filtered lines. Fix: `lmk_lines.answer_blank`.
3. HIGH — the 0.5 s pid re-check also ran after a failed pidof that happened to contain a number. Fix: it now runs only
   after a successful pidof (root rc 0, pidof rc 0).
4. MEDIUM — an empty marker line could hide a non-empty transient readback. Fix: the whole answer must be the marker line.
5. MEDIUM — pgrep ignored stderr. Fix: "no output" now means stdout and stderr are both blank.

**Round 2, one finding, reproduced and fixed.** HIGH — re-running a force-stop that did not run also re-read pidof, which
replaced a non-empty pidof failure. Fix: the force-stop is re-run alone, and pidof is judged on its own answer (probe
FIX4C R2-1). The reviewer confirmed the rest:
- the round-1 fixes and the FIX4B round-3 probes are resolved;
- no unlisted change to non-empty judgments;
- thermal is AST-identical;
- empty answers are re-read once and fail closed;
- the other audit work, D1/D2, the checklist and the rehearsal gate are in place.

**Round 3, no review.** The reviewer reported this blocker: every command on candidate files ("runtime.py", the delta
patch, `results.json`) and every Python probe returned exit code 182 with no output. Its stderr shows 21 such failures,
each "exited 182 in 0ms". The copy itself is intact: the files are present, readable (0644) and match the manifest. The
blocker is codex's sandbox, which warns on every run that bubblewrap is not on PATH and falls back to its bundled copy.
Round 2's transcript shows the same sporadic exit-182 failures, but that round still finished its probes. The reviewer
called this "an execution blocker, not a demonstrated candidate-code defect" and gave no approval.

## C7. Outcome: FIX4C NOT APPROVED after its 3 new rounds — stopped

Under the owner rule ("No verdict means incomplete, never a pass"; stop after 3 new rounds), I stopped. No fourth
review was run. The candidate is the round-3 frozen state. The round-2 fix is in it, but no reviewer has verified it.
- **Offline checks:** all 18 checks and the mock dry run pass on the final candidate (§C5).
- **Invariant tests:** the `self_check_fix4c.py` sweep and the probe table (§C2, §C3) cover every wrapped reader and
  every probe from all eight review rounds that returned findings.
- **Live validation:** none. Agents were resident, and root is unreachable from proot.

**Recommendation.** Grant one more fresh read-only review of this exact round-3 candidate, unchanged, with the same request.
Round 3 failed for an environment reason, not on the code. The run could first check that its sandbox can read the copy;
installing `bubblewrap` (or running where codex's sandbox works) should remove the exit-182 failures. If that review
approves, the commit proposal below applies as is.

**Commit — NOT proposed now (not APPROVED).** For reference, after an APPROVED review the single commit would be:
- message: "benchmark: campaign phase-1 audit fixes (bounded re-reads, camera slot skip, atomic evidence, watchdog),
  L1 10 s cadence, cooling extension, narrowed re-read" plus the session trailers;
- files: the 74 paths in `frozen_manifest_campaign_p1_fix4c.json` plus that manifest itself.

Not part of the commit (superseded, untracked; I will delete them on your word): `checks/p1_fix4/`, `checks/p1_fix4b/`,
`frozen_manifest_campaign_p1_fix4.json`, `frozen_manifest_campaign_p1_fix4b.json` and `self_check_fix4b.py`.

(Base behaviour left for an owner decision: §C9.)

## C8. Files written outside the repository (FIX4C)

`CAMPAIGN_P1_FIX4B_CODER_REPORT_full.md` (pre-FIX4C report), `campaign/review_request_campaign_p1_fix4c_r{1,2,3}.txt`, and
`/termux-home/review_campaign_p1_fix4c_r{1..4}/` (copies, review material, reviews, `.stderr`).

## C9. Round 4 (owner-approved, exact round-3 candidate): NOT RUN, sandbox blocked

Manifest verified (74 files, SHA-256 1f755031…, = round 3); fresh copy `review_campaign_p1_fix4c_r4/` verified.
- **Before:** no `bwrap` (`which` rc 1, `--version` rc 127); codex warned it would use its bundled bubblewrap.
- **System bwrap:** `apt-get install -y bubblewrap` → rc 0, bubblewrap 0.12.0-1~deb13u1. Probe 1 (`head` of runtime.py,
  `sha256sum`, `python3 -I -c`) gave exit 1, 0 and 1: "bwrap: Can't bind mount /dev/null on null: readlink(/proc/self/fd/N)
  … Invalid argument", because proot cannot provide that bind.
- **Bundled bwrap:** I ran `apt-get remove -y bubblewrap` (rc 0) to revert my install; its dependencies, such as procps,
  stay installed. Probe 2 gave exit 182, 0 and 182 (0 ms, no output).
- **Result:** both probes failed, so the review was not run (owner step 3). Outputs: `sandbox_probe*` in the copy.
  Codex's read-only sandbox cannot run commands in this proot; another environment, sandbox mode
  or reviewer is your decision.
- **Base behaviour noted, no change** (owner step 5): a complete battery answer with voltage 0 is accepted, and so is
  pgrep rc 0 with stderr but no process line. Both are left for a later owner decision.


# FIX4D — su stderr kept (owner option (b), fresh review budget of 3 rounds)

## D0. FIX4C round 4 (owner option A: codex, `-s danger-full-access`, only inside the isolated copy)

The run used the exact round-3 candidate (manifest SHA-256 1f755031…980e), on the fresh copy
`/termux-home/review_campaign_p1_fix4c_r4/`. The request (45,419 bytes) added a hard-rule block and the round-2 and round-3
reviews verbatim. Its first question was the round-2 fix. The command:

```sh
cd /termux-home/review_campaign_p1_fix4c_r4 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 codex --no-daemon exec \
  --skip-git-repo-check -m gpt-6.1-sol -c model_reasoning_effort="high" -s danger-full-access - \
  < /termux-home/storage/downloads/campaign/review_request_campaign_p1_fix4c_r4.txt \
  > /termux-home/review_campaign_p1_fix4c_r4_out/review_campaign_p1_fix4c_r4.txt 2> …/review_campaign_p1_fix4c_r4.stderr
```

Integrity:
- before = after, byte-identical (snapshot SHA-256 `1a9576bf5f06219d…`, both files);
- every file of the copy hashed (2,634 entries);
- the live tree matched the manifest and `git status --short` was unchanged (25 lines);
- the reviewer ran 25 commands, none outside the copy and none writing.

Verdict: CHANGES REQUESTED, one HIGH finding (`/termux-home/review_campaign_p1_fix4c_r4_out/review_campaign_p1_fix4c_r4.txt`, SHA-256
`595e638e5ff6de3e…`). `cr.root` dropped su's stderr, so a stderr-only su failure
("su: permission denied", rc 1, empty stdout) looked empty. It was re-read, and a healthy answer replaced it in the
force-stop, pidof, battery and package-version reads. I reproduced it: battery 4.0 W after 2 su calls and 1 re-read. The
round-2 fix itself was verified to work.

## D1. Changes

- `coresidency.py`, shared:
  - `root()` still returns (rc, stdout). su's stderr of the last call is kept per thread: `root_stderr()`, reset at every
    call. It has to be thread-local because root() runs in several worker threads, and each reader checks it right after
    its own call.
  - `RootShell`: su's stderr was /dev/null. A second pump thread now drains it. Each `run()` also echoes its end marker to
    stderr and keeps that command's stderr in `last_stderr`. `run()` still returns stdout only. Late output from a
    timed-out command is discarded on both streams.
  - `root_sample` and `lmk_lines`: `answer_blank` requires stdout AND stderr to be blank.
- `runtime.py`, `power.py`, `diagnostics.py`: every "carried nothing" decision uses stdout plus stderr:
  - `rt.nothing(cr, text)` and `rt.su_stderr(cr)`;
  - `rt.shell_nothing(shell, lines)` and `Heard`;
  - `stop_camera`'s recorded (rc, stdout, stderr) answers;
  - battery and settings, which add su stderr;
  - the root_mask_scope readback.
  `cr.am` is unchanged: only its launch failure is re-read.
- Round-1 fix: a launch failure is only an `OSError` raised inside `subprocess.Popen` (`rt.launch_failure`). Any other
  `OSError` stands as the reader's own. `camera_start` re-reads only a failure to launch its first `am start` (§D5).
- Importers of coresidency.py:
  - `power_map.py`, which leads on to `duty_cycle.py` and `speed_block.py`, then `speed1/variants.py`, then the campaign;
  - `thermal_char.py`;
  - `test_coresidency.py`.
  Their checks all pass (§D4).
- Unchanged, and on no campaign path: thermal_char's own root_file/RootShell copies and camera_heat.root.
- Tests: `self_check_fix4d.py` (new). RUN.md: H2 text, a FIX4D note, run stems `owner_*_p1_fix4d`.
- The report was restructured: the FIX4/FIX4B detail moved to the full file (§1), and the FIX4C C3 table to §D3.

## D2. Regressions through the REAL cr.root and the REAL RootShell (`self_check_fix4d.py`)

- `cr.root` is driven with subprocess stdout and stderr mocked.
- `RootShell` is driven over a fake `su` (a plain sh) on PATH.
- A stderr-only failure is read once and fails closed.
- A blank stdout+stderr answer is re-read once, recovering on a good answer and failing closed on an empty or bad one.

- battery (rt.battery_sample): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
- memory/PSS (rt.memory_sample, cr.root_sample): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
- camera-OFF pidof (rt.memory_sample): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
- package version (camera_version): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
- logcat (phase1.lmk_window, cr.lmk_lines): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
- capacity (diagnostics.battery): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
- diagnostics nodes (diagnostics.snapshot): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
- screen state (screen_state): stderr-only su failure read once and fails closed; blank stdout+stderr re-read once (recovered / failed closed when empty or bad again)
- settings (rt.screen, sb.Screen.setting): stderr-only su failure read once; blank re-read once
- force-stop / pidof (rt.stop_camera, real camera_end_check): stderr-only failures read once and fail; blank ones re-read once
- transient readback (root_mask_scope over the real cr.root): with su stderr read once; blank re-read once
- power (power.sample over the real RootShell): stderr-only answer read once and fails; blank re-read once
- mask (verify_monitor, sb.root_mask over the real RootShell): stderr-only answer read once and fails
- launch failures only (FIX4D R1): a local OSError after a complete battery answer and a cleanup PermissionError after an unsafe setup mask stand (1 read); su/RootShell/am that could not be started are re-read once; camera_start re-reads only a launch failure of its first `am start` (a later one fails closed)
- every earlier probe (FIX4, FIX4B, FIX4C R1-R2 from self_check_fix4c, FIX4C R4) on this code; FIX4C R3 had no findings (sandbox blocker)

## D3. Every earlier probe, re-run on the FIX4D candidate

The table covers FIX4 R1–R3, FIX4B R1–R3 and FIX4C R1–R2 (via `self_check_fix4c.probes()`), plus FIX4C R4. FIX4C R3 had
no findings. Each first answer has a healthy answer queued behind it.

| Finding | Probe (healthy re-read queued) | FIX4D result |
|---|---|---|
| FIX4 R1-2 | fast: battery 45.0 C next to a missing policy, healthy re-read queued | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading |
| FIX4 R1-3 | pgrep rc 0 with empty stdout (twice) | 2 reads, 1 re-read(s): ReadFailed: pgrep: Blip: process enumeration failed with no output: rc 0, '' (re-read once; first: Blip: process enumera |
| FIX4 R1-3 | pgrep rc 1 with a listed line | 1 read(s), no re-read: RuntimeError: agents/robot/other runners resident: 2001 codex |
| FIX4 R1-3 | pgrep rc 0 with a malformed line | 1 read(s), no re-read: ValueError: invalid literal for int() with base 10: 'garbage' |
| FIX4 R1-5 | screen state: su answer (1, empty), then nothing again | 2 reads, 1 re-read(s): returned {'error': 'ReadFailed: dumpsys power/activity: Blip: empty screen state answer: rc 1 (re-read once; first: Blip |
| FIX4 R1-5 | screen-timeout settings get answering nothing, twice | 2 reads, 1 re-read(s): ReadFailed: settings: Blip: empty answer to 'settings get system screen_off_timeout' (re-read once; first: Blip: empty a |
| FIX4 R2-1 | power average ['-1000000','0','Discharging','garbage'] | 1 read(s), no re-read: RuntimeError: invalid battery voltage |
| FIX4 R2-1 | power ['', '0', 'Discharging'] | 1 read(s), no re-read: ValueError: invalid literal for int() with base 10: '' |
| FIX4 R2-2 | settings get answering '' (then '') | 2 reads, 1 re-read(s): ReadFailed: settings: Blip: empty answer to 'settings get system screen_off_timeout' (re-read once; first: Blip: empty a |
| FIX4 R2-2 | screen state (1, '===\n') | 1 read(s), no re-read: returned {'rc': 1, 'error': "unparsable: '===\\n'"} |
| FIX4 R2-2 | transient readback empty actual= (twice) | 2 reads, 1 re-read(s): ReadFailed: dumpsys package: Blip: transient root shell pin/readback failed: actual='CAMPAIGN_ROOT_MASK pid=77 actual=\n |
| FIX4 R2-2 | transient readback mismatch | 1 read(s), no re-read: RuntimeError: transient root shell pin/readback failed: actual="CAMPAIGN_ROOT_MASK pid=77 actual=pid 77's current affini |
| FIX4 R2-2 | setup discover answer without the required policies | 1 read(s), no re-read: RuntimeError: missing required CPU frequency policies; not started |
| FIX4 R3-1 | fast: 7-field layout, battery 450, final policy line missing | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading: expected 7 values, got 6: ['50000', '51000', '52000', '450', '180 |
| FIX4 R3-2 | power ['-1000000','0'] | 1 read(s), no re-read: RuntimeError: power read missing/charger connected: ['-1000000', '0'] |
| FIX4 R3-2 | power average ['-1000000','0','Discharging'] | 1 read(s), no re-read: RuntimeError: power read missing/charger connected: ['-1000000', '0', 'Discharging'] |
| FIX4 R3-3 | capture not stopped, force-stop ok, pidof readback blip | 3 reads, 1 re-read(s): RuntimeError: camera cleanup: capture not shown stopped 15.1 s after STOP (151 new frames, 0 unreadable reads) |
| FIX4B R1-1 | fast: BIG 110000 with one blank policy | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading |
| FIX4B R1-1 | fast: BIG 110000, shortened answer | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading: expected 7 values, got 6: ['110000', '51000', '52000', '300', '18 |
| FIX4B R1-2 | su battery (0, '-1000000\n0\n') | 1 read(s), no re-read: RuntimeError: root battery read failed: rc=0, '-1000000\n0\n' |
| FIX4B R1-2 | root_sample voltage 0 with provider PSS missing | 1 read(s), no re-read: returned {'t': 13190.79835609, 'root_rc': 0, 'root_s': 0.0, 'pss_kb': {'runner': 100, 'llama_server': 100, 'robotcam_app |
| FIX4B R1-3 | capacity '101\ngarbage' | 1 read(s), no re-read: RuntimeError: battery capacity unreadable |
| FIX4B R1-3 | capacity '-1\ngarbage' | 1 read(s), no re-read: RuntimeError: battery capacity unreadable |
| FIX4B R1-4 | setup mask '3-0' plus a malformed line | 1 read(s), no re-read: RuntimeError: missing/malformed root Cpus_allowed_list |
| FIX4B R1-4 | setup mask CPU 8 on 8 cores | 1 read(s), no re-read: RuntimeError: invalid root CPU range |
| FIX4B R1-5 | after put: readback (1, '60000') | 1 read(s), no re-read: RuntimeError: screen setting failed: 60000 |
| FIX4B R1-5 | after put: readback '60000\ngarbage' | 1 read(s), no re-read: RuntimeError: screen timeout set/readback failed |
| FIX4B R2-1 | root_sample voltage 0, complete PSS (base parser has no voltage check) | 1 read(s), no re-read: returned {'t': 13190.803349091, 'root_rc': 0, 'root_s': 0.0, 'pss_kb': {'runner': 100, 'llama_server': 100, 'robotcam_ap |
| FIX4B R2-2 | logcat ok=False, rc 1, in-window kill line | 1 read(s), no re-read: returned {'logcat_rc': 1, 'raw_head': 'logcat_rc=1\nerr\n\n', 'answer_blank': False, 'lines': ['0.5 lmkd: Kill com.examp |
| FIX4B R2-3 | power ['-1000000','4000000','123'] | 1 read(s), no re-read: RuntimeError: power read missing/charger connected: ['-1000000', '4000000', '123'] |
| FIX4B R2-3 | su battery status '123' | 1 read(s), no re-read: RuntimeError: charger/battery status: 123 |
| FIX4B R2-4 | mask '0-5,' | 1 read(s), no re-read: RuntimeError: missing/malformed root Cpus_allowed_list |
| FIX4B R2-4 | mask '3-0,' | 1 read(s), no re-read: RuntimeError: missing/malformed root Cpus_allowed_list |
| FIX4B R3-1 | power ['-1000000','+0','Discharging'] | 1 read(s), no re-read: RuntimeError: invalid battery voltage |
| FIX4B R3-1 | capacity '+20\ngarbage' | 1 read(s), no re-read: RuntimeError: battery capacity unreadable |
| FIX4B R3-1 | capacity '+101\ngarbage' | 1 read(s), no re-read: RuntimeError: battery capacity unreadable |
| FIX4B R3-1 | root_sample voltage +0, complete PSS (base parser has no voltage check) | 1 read(s), no re-read: returned {'t': 13190.806373994, 'root_rc': 0, 'root_s': 0.0, 'pss_kb': {'runner': 100, 'llama_server': 100, 'robotcam_ap |
| FIX4B R3-2 | force-stop (1, 'Error: unknown package'), then pidof readback blip | 2 read(s), no re-read: RuntimeError: camera cleanup: am force-stop failed (rc 1: 'Error: unknown package') |
| FIX4B R3-3 | logcat partial header with a kill, separator missing | 1 read(s), no re-read: returned {'logcat_rc': 1, 'raw_head': 'logcat_rc=1\n0.5 lmkd: Kill com.example\n', 'answer_blank': False, 'lines': [], ' |
| FIX4B R3-4 | fast: battery 1001, shortened answer | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading: expected 7 values, got 6: ['50000', '51000', '52000', '1001', '18 |
| FIX4B R3-4 | fast: CPU 200001, shortened answer | 1 read(s), no re-read: RuntimeError: missing root CPU/battery/policy reading: expected 7 values, got 6: ['200001', '51000', '52000', '300', '18 |
| FIX4B R3-5 | lag mask config: '0-4,' then '0-3' | 1 read(s), no re-read: RuntimeError: missing/malformed root Cpus_allowed_list |
| FIX4B R3-6 | power average ['-1000000','4000000','123'] | 1 read(s), no re-read: RuntimeError: power read missing/charger connected: ['-1000000', '4000000', '123'] |
| FIX4B R3-6 | su battery four tokens with status '123' | 1 read(s), no re-read: RuntimeError: root battery read failed: rc=0, '-1\n4000000\n123\nx\n' |
| FIX4C R1-1 | root_sample: 200 newlines, then Charging, PSS missing (blank 200-char excerpt) | 1 read(s), no re-read: returned {'t': 13190.811049815, 'root_rc': 0, 'root_s': 0.0, 'pss_kb': {}, 'pss_pids': {}, 'answer_blank': False, 'batte |
| FIX4C R1-2 | logcat: 400 newlines, then logcat_rc=1 and a kill, no separator | 1 read(s), no re-read: returned {'logcat_rc': 1, 'raw_head': '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\ |
| FIX4C R1-2 | logcat '=== lmk lines\ngarbage\n' | 1 read(s), no re-read: returned {'logcat_rc': None, 'raw_head': '', 'answer_blank': False, 'lines': [], 'n_lines': 0, 'n_kills': 0, 'outside_bl |
| FIX4C R1-3 | force-stop ok, pidof (1, 'su: error 123\npidof_rc=2\n') | 2 read(s), no re-read: RuntimeError: camera cleanup: RobotCam still running after force-stop (pids ['123']) |
| FIX4C R1-4 | readback rc 98: empty marker line, then a different mask line | 1 read(s), no re-read: RuntimeError: transient root shell pin/readback failed: actual="CAMPAIGN_ROOT_MASK pid=77 actual=\npid 77's current affi |
| FIX4C R2-1 | force-stop (1, ''), pidof (0, 'pidof_rc=2\n'), re-run force-stop (0, ''), healthy pidof queued | 3 reads, 1 re-read(s): RuntimeError: camera cleanup: RobotCam absence after force-stop unconfirmed (root rc 0, pidof rc 2) |
| FIX4C R1-5 | pgrep rc 2, stdout empty, stderr 'pgrep: permission denied' | 1 read(s), no re-read: RuntimeError: process enumeration failed: pgrep: permission denied |
| FIX4C R4-1 | force-stop rc 1, stdout '', stderr 'su: permission denied'; pidof ok; healthy force-stop queued | 2 read(s), no re-read: RuntimeError: camera cleanup: am force-stop failed (rc 1: '') |
| FIX4C R4-1 | pidof rc 1, stdout '', stderr 'su: permission denied'; healthy pidof queued | 2 read(s), no re-read: RuntimeError: camera cleanup: RobotCam absence after force-stop unconfirmed (root rc 1, pidof rc None) |
| FIX4C R4-1 | battery: the same stderr-only failure; healthy answer queued | 1 read(s), no re-read: RuntimeError: root battery read failed: rc=1, '' |
| FIX4C R4-1 | package version: the same stderr-only failure; healthy answer queued | 1 read(s), no re-read: RuntimeError: installed RobotCam version unreadable |
| FIX4D R1-2 | setup: mask 4-7 (+ su stderr) with measured {4,5}, then the cleanup kill raises PermissionError; healthy setup queued | 1 read(s), no re-read: PermissionError: kill |
| FIX4D R1-1 | battery: complete 2 W answer + su stderr, then meminfo raises OSError; healthy 4 W queued | 1 read(s), no re-read: OSError: meminfo unreadable |

## D4. Checks (FIX4D candidate; offline; agents resident; proot; NOT VALID for timing)

Outputs: `checks/p1_fix4d/`. Manifest `frozen_manifest_campaign_p1_fix4d.json` (77 files). It excludes the superseded
untracked intermediates:
- `checks/p1_fix4/`, `checks/p1_fix4b/`, `checks/p1_fix4c/`;
- the fix4/fix4b/fix4c manifests;
- `self_check_fix4b.py`.

| Check | Command | Exit |
|---|---|---|
| self_check | `benchmark/campaign/self_check.py` | 0 |
| self_check_fix2 | `benchmark/campaign/self_check_fix2.py` | 0 |
| self_check_fix3 | `benchmark/campaign/self_check_fix3.py` | 0 |
| self_check_fix3b | `benchmark/campaign/self_check_fix3b.py` | 0 |
| self_check_fix3c | `benchmark/campaign/self_check_fix3c.py` | 0 |
| self_check_fix3d | `benchmark/campaign/self_check_fix3d.py` | 0 |
| self_check_fix4 | `benchmark/campaign/self_check_fix4.py` | 0 |
| self_check_fix4c | `benchmark/campaign/self_check_fix4c.py` | 0 |
| self_check_fix4d | `benchmark/campaign/self_check_fix4d.py` | 0 |
| coresidency_units | `benchmark/coresidency/test_coresidency.py` | 0 |
| power_map_units | `benchmark/power_map/test_power_map.py` | 0 |
| duty_cycle_units | `benchmark/duty_cycle/test_duty_cycle.py` | 0 |
| camera_power_units | `benchmark/camera_power/test_camera_power.py` | 0 |
| thermal_char_units | `benchmark/thermal_char/test_thermal_char.py` | 0 |
| speed1_self_check | `benchmark/relate_anything/speed1/self_check.py` | 0 |
| desk2_self_check | `benchmark/relate_anything/desk2/self_check.py` | 0 |
| detector_units | `test_detector_robotcam.py` | 0 |
| mission_units | `test_run_mission.py` | 0 |
| cycle_units | `run_cycle_safety_test.py` | 0 |
| dry_run_mock | `benchmark/campaign/phase1.py --dry-run --mock --output /termux-home/robot/benchmark/campaign/checks/p1_fix4d/coder_dry_mock_p1_fix4d.json` | 0 |

## D5. FIX4D independent reviews (option A, as FIX4C round 4)

Every round ran the following command in a fresh copy `/termux-home/review_campaign_p1_fix4d_rN/`. The copy is a
`git archive` of the base plus the frozen files, each SHA-256 verified. Outputs go to `…_rN_out/`, outside the copy.
The request carried "Ponytail mode: off", the hard-rule block and the FIX4C round-4 review verbatim, and asked whether
any re-read can replace any non-empty answer (stdout or stderr).

```sh
cd /termux-home/review_campaign_p1_fix4d_rN && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 codex --no-daemon exec \
  --skip-git-repo-check -m gpt-6.1-sol -c model_reasoning_effort="high" -s danger-full-access - \
  < /termux-home/storage/downloads/campaign/review_request_campaign_p1_fix4d_rN.txt \
  > /termux-home/review_campaign_p1_fix4d_rN_out/review_campaign_p1_fix4d_rN.txt 2> …/review_campaign_p1_fix4d_rN.stderr
```

| Round | Request bytes, SHA-256 | Integrity: before = after (snapshot SHA-256) | Review (bytes, SHA-256) | Verdict |
|---|---|---|---|---|
| 1 | 29,663, `465f2106a5ae2131…` | byte-identical (`69e7c51d084ec889…`) | 3,016, `045354a3d6bd6dbc…` | CHANGES REQUESTED |
| 2 | 35,710, `83bc285beccd5d94…` | byte-identical (`570c3f249b6c568e…`) | 1,779, `c26c28da78b468df…` | APPROVED |

**Round 1: CHANGES REQUESTED, one HIGH finding, reproduced and fixed.** `reread()` treated every `OSError` raised
anywhere in a reader as a launch failure, including errors after the transport had answered:
- a `/proc/meminfo` failure after a complete battery answer;
- a cleanup `PermissionError`, raised by `kill`, after an unsafe setup mask.

Fix: `rt.launch_failure(e)`. Only an `OSError` whose traceback is inside `subprocess.Popen` (`__init__` or
`_execute_child`: pipes, fork, exec) counts, because that is before any answer. Any other `OSError` stands as the reader's
own, also on the re-read. `camera_start` re-reads only a launch failure of its first `am start`; a later one fails closed
as a RuntimeError. `stop_camera` uses the same rule.

Regressions in `self_check_fix4d.check_launch_only`, plus probes FIX4D R1-1 and R1-2. The self-checks now build real
launch failures with `launch_error()`, which captures Popen's OSError for a missing executable.

The reviewer confirmed the rest:
- the FIX4C round-4 finding is resolved;
- the 55 historical probes pass;
- thread-local isolation and late-output separation pass;
- the thermal functions are AST-identical to the base;
- the 19 checks pass.

It could not see the restructured reports, which live outside the repo. From round 2 on, both report files are in the
copy root as review material.

**Round 2: APPROVED, no blocking findings.** The reviewer confirmed:
- no re-read can replace a non-empty stdout or su-stderr answer;
- the shared-module change is backward compatible, with thread-local stderr isolation;
- empty answers are re-read once and fail closed;
- no audit finding is unresolved, and D1/D2, the checklist, the rehearsal gate and the report split are in place
  (main report under 80 KB, history hash matching).

It verified the 77 frozen files, 105 diff hunks and 57 historical probes. Integrity check: before = after,
byte-identical. None of the reviewer's commands touched the live tree or wrote anything. The approval grants no commit
or push permission.

## D6. After APPROVED (owner step 6)

- Deleted, after checking each was untracked; copies remain in `/termux-home/review_campaign_p1_fix4b_r3/`:
  `checks/p1_fix4/`, `checks/p1_fix4b/`, `frozen_manifest_campaign_p1_fix4.json`, `frozen_manifest_campaign_p1_fix4b.json`,
  `self_check_fix4b.py`.
- `git status --short`: the 77 manifest files plus `frozen_manifest_campaign_p1_fix4d.json`, all hashes matching. Two
  more superseded FIX4C intermediates are left untracked: `checks/p1_fix4c/` and `frozen_manifest_campaign_p1_fix4c.json`.
  They were not on the owner's list, so they wait for the owner's word, and the commit names its files explicitly.
- Proposed single commit (not made):
  - message: `benchmark: campaign phase-1 audit fixes (bounded re-reads, camera slot skip, atomic evidence, watchdog),
    L1 10 s cadence, cooling extension, narrowed re-read, su stderr kept` plus the session trailers;
  - files: the 77 paths in `frozen_manifest_campaign_p1_fix4d.json` (19 code/doc/self-check files and
    `checks/p1_fix4d/`, 58 files) plus that manifest.
- Base behaviour still left for a later owner decision:
  - a complete battery answer with voltage 0 is accepted (`root_sample`, `sb.battery_sample`);
  - pgrep rc 0 with stderr but no process line is accepted.

**Committed after the owner's "yes commit":** `074d5b92ce32330d6fb3c04d9d0ee7db0556ef3d` on main (78 files: the 77 manifest paths plus the manifest; staged blobs checked against the manifest hashes). Before the commit, `checks/p1_fix4c/` and `frozen_manifest_campaign_p1_fix4c.json` were deleted on the owner's word after checking both were untracked (copies remain in `/termux-home/review_campaign_p1_fix4c_r4/`). Not pushed: that needs a separate "yes push".

**Pushed after the owner's separate "yes push":** `origin/main` 5629699..074d5b9 (fast-forward); `main` and `origin/main` both at `074d5b92ce32330d6fb3c04d9d0ee7db0556ef3d`.
