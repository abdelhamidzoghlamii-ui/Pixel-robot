# CAMPAIGN_P1_FIX3D — Coder report

Coder: Claude Code CLI, claude-opus-5-5, Ponytail lite. Reviewer: Codex CLI 0.160.0, gpt-6.1-sol, effort medium, `-s read-only`,
Ponytail off. No fallback used. **Final review: VERDICT: APPROVED (round 3 of 3).** Nothing committed or pushed.

## 1. HEAD and status
- Start: HEAD `b723002614c37450ad86025b80e1c831665a5bee`; `git status --short` empty (reported at task start).
- Now: HEAD unchanged; working tree = frozen candidate (round 3), 75 files + manifest, all SHA-256 re-verified after the review.

## 2. Facts verified from the bytes (owner Downloads)
- **owner_rehearsal_p1_fix3c**: label `NOT VALID — OWNER REHEARSAL ... COMPLETE`; `rehearsal_coverage`: setup, 3 idle, warm-up,
  6 pauses, 6 blocks, `errors []`, `rehearsal_pass false`; live M2 calls 0 in every block. **Discrepancy:** fallback M2 calls
  are **16** (L1 4, L2 2, L3 5, L4 5; coverage, `summary.m2.fallback_calls` and m2 rows agree), not 21; 5 selector calls had a
  fallback-scene context (16 + 5 = 21). Owner chose 16. Idle policy0 caps: 0 % camera OFF, 86.1 % (1401 MHz) camera ON.
- **owner_session_p1_fix3c**: `RuntimeError: pause sampler stopped` (phase1.py:576 in that code), in the **first idle sub-phase
  `IDLE` (camera OFF), 5.26 s**. Recorded cause: `monitor_errors ['thermal read failed: skin None, status None']` from the
  thermal worker, 0 thermal rows. The screen lock is not visible in the bytes; same signature as fix3c2. Owner chose: keep the
  fixed screen-lock label and add the byte facts.
- **owner_session_p1_fix3c2**: traceback confirmed exactly: phase1.py:744 → :571 `rt.dump_check(cr)` → runtime.py:201 → :192
  `RuntimeError('thermal read failed: skin None, status None')` (rc 0: the rc branch did not fire). Sub-phase `IDLE CAMERA ON —
  NO INFERENCE`, duration 58.56 s, last fast row 54.12 s. Thermal rows 39 + 14 + 12 = **65, all rc 0, status 0**; skin
  26.9–27.3, 27.4–27.7, 28.1–29.4 °C. caps_by_policy: policy0 0 % / 0 % camera OFF, **90.2 %** (54.12 s, lowest 1401 MHz)
  camera ON; policy4/policy6 0 %. No warm-up, no block.

## 3. Root cause (likely) and fix
**Likely cause:** in `rest()` and `live_block()` the 4.87 s thermal worker and the main thread's `rt.dump_check` / `diag.snapshot`
both ran `dumpsys thermalservice` via `cr.root` with the same tag, i.e. the same `/data/local/tmp/coresidency_thermalservice.txt`
(`su -c "{ cmd; } >FILE 2>&1; rc=$?; cat FILE; exit $rc"`). A concurrent truncation gives rc 0 with empty or NUL-holed text, so
skin None and status None. In fix3c2 the worker read ran 54.63–54.81 s and the failing main read ended between 54.12 s and the stop:
a very likely overlap, not proven. Reproduced deterministically with the real `cr.root` form and fake su/dumpsys
(`self_check_fix3d.py:159`). Owner chose: retry **and** per-thread tag.

Fix (`benchmark/campaign/`):
- `runtime.py:194` `dump_once(cr, timeout=30)`: cr.read_dump's read/parse plus raw text; **per-thread tag**
  `campaign_thermal_<native tid>` (`runtime.py:203`).
- `runtime.py:210` `read_dump`: an incomplete read (rc 0, no error, skin/status missing/unparsable) is retried, at most 3 attempts
  (`:189` DUMP_ATTEMPTS 3, DUMP_RETRY_S 2.0, DUMP_PAUSE_S 0.5, RAW_KEEP 1024). The deadline is the first read's end + 2 s. The pause is
  min(0.5, time left) (`:243`), there is no attempt after the deadline (`:244`), a retry's su timeout is the time left (`:224`), and a
  complete retry that ends after the deadline counts as incomplete (`:228`). rc ≠ 0, an error, or a **partial read with status ≥
  STATUS_STOP** raise at once (`:238`). A complete row gets `attempts`; when retried it also gets `incomplete_attempts` (attempt,
  start_offset_s, elapsed_s, why, raw ≤ 1 KB), and a copy goes to `THERMAL_RETRIES` (`:191`, `:233`). Failure raises with every
  attempt's raw in the message (`:246`).
- Callers: `phase1.py:405,561` thermal workers, `:389/:424` skin_start/end, `:571` rest loop, `:623,:634,:664` preflight;
  `diagnostics.py:68` snapshot (+ `thermal_attempts`, `thermal_incomplete_attempts`, `:74`); `lag_probe.py:30` preflight and
  **`lag_probe.py:60` worker now `rt.read_dump`** (was unchecked `cr.read_dump`); result JSON `thermal_retries`: `phase1.py:724`,
  `lag_probe.py:34`.
- Other dumpsys readers, deliberately **not** retried (the reviewer agreed: no shared-file collision, failures stay fatal):
  `runtime.py:252` camera_version (`dumpsys package`, preflight main thread only, refuses setup);
  `benchmark/coresidency/coresidency.py:221–227` root_sample (`dumpsys meminfo`, tag `sample`, via `runtime.py:277`
  memory_sample, memory worker only; a missing PSS has a real cause handled by fix3).
- Side effect: one small `campaign_thermal_<tid>.txt` per reading thread stays in `/data/local/tmp` (about 17 per session).

## 4. Worst-case timing (as approved)
- Retries start and are accepted only within **2.0 s after the first read ends**. At the measured 0.12–0.20 s per read, three
  attempts end at **1.6 s**. Retries add at most 2.0 s to an accepted read. A failing read can exceed that by the launch/cleanup
  time of the last retry and one OS oversleep; there is no hard wall-clock bound.
- The first read keeps its existing 30 s su timeout, so the idealised total is up to about 32 s (reviewer's figure).
- Cadence: `cr.monitor_loop` keeps the k·4.87 s grid and skips slots on overrun (never bunches). A 2.2 s read skips no slot; any read
  longer than the period skips one (a 4.87 s gap), the existing behaviour for slow reads.
- Persistent failure can't be hidden: every sweep case with no complete read raises. A transient dropout that clears within 2 s is
  accepted but visible (`attempts`, raw, `thermal_retries`); it cannot be told apart from truncated output.

## 5. Regressions (`self_check_fix3d.py`, plus adapted older checks)
- `check_reader` (:50): empty → complete passes, attempts 2, raw '' kept in row and THERMAL_RETRIES, per-thread tag, retry
  timeout ≤ window. Partial/oversized raw is truncated. **Three empty reads fail closed** with 3 raws. **rc 1 / rc 1 empty / TimeoutExpired /
  OSError → 1 attempt, no sleep.** Thermal stop is kept, and a partial status-4 read fails at once (reader and dump_check).
- `check_timing` (:103): first read 0.01–29 s, retries 0.01–29.9 s, launch 0/3 s outside the timeout, oversleep 0/0.3/3 s.
  A persistent failure always raises, an accepted retry ended ≤ deadline, and a late complete retry fails closed; this case **fails on
  the round-2 code** (ends [0.01, 3.52] accepted). The cadence grid is checked too.
- `check_shared_file_race` (:159): with a shared tag, reader A gets rc 0, skin None, status None; with per-thread tags both are complete.
- `check_rest` (:221), **camera OFF and ON**, real dump_check/dump_once: the last in-phase check is empty once, then the phase
  passes with the retry in THERMAL_RETRIES. Three empty fails the phase, rc 1 fails with 1 attempt, and a partial status-4 read in the main
  check or in the worker fails the phase.
- `check_live_block` (:257), session and rehearsal (live_block always runs with the camera ON): skin_end partial once → VALID,
  attempts 2, raw in the block row. Three empty or rc 1 at skin_end → NOT VALID / shared.
- `self_check.py` lag lifecycle: modes `thermal` (persistent incomplete fails) and `thermal_retry` (completes, attempts 2,
  `thermal_retries` saved).
- Old mocks of `cr.read_dump` on runner paths were retargeted to `rt.dump_once` (`self_check.py`, `_fix2`, `_fix3`, `_fix3b`).
  `rest_harness` and `live_record` gained `reads=` (real dump_check). The fix3c audit gained the `rt.read_dump` spawn source
  (14 sources; only diagnostics `node` and llama-server match).

## 6. Archive (STEP 3)
19 owner files copied byte-for-byte; SHA-256 verified against source and copy. Inventory:
`benchmark/campaign/runs/archive_sha256_p1_fix3d.json`. Committed evidence was not modified. RUN_INDEX rows:
- owner_rehearsal_p1_fix3c: **REHEARSAL COMPLETE — NOT A PASS. Every phase, pause and all 6 blocks ran with no errors; 0 live M2
  calls because the phone lay on its back (camera saw nothing); 16 fallback M2 calls. NOT VALID as results.** (16 per owner
  decision; the evidence column records the 21-vs-16 note, the per-layout counts, the caps and links to the warm-up/blocks.)
- owner_session_p1_fix3c: **NOT VALID — SESSION INCOMPLETE. The owner locked the screen (emergency); 'pause sampler stopped' in a
  rest phase; no block ran.** Evidence column: phase `IDLE` (camera OFF), 5.26 s, the recorded thermal-worker error, 0 thermal
  rows, screen-lock causality not shown in the bytes.
- owner_session_p1_fix3c2: **the fixed label verbatim**; evidence column: traceback, 65 valid rows, skin ranges, caps, likely overlap,
  fixed in FIX3D.

## 7. Checks (STEP 4)
All with `/data/data/com.termux/files/usr/bin/python` from the repo root, in proot with agents resident: **NOT VALID for timing; no
root, no live validation.** Outputs and `results.json` are in `benchmark/campaign/checks/p1_fix3d/`. **All 17 exit 0:** self_check, _fix2,
_fix3, _fix3b, _fix3c, _fix3d, coresidency, power_map, duty_cycle, camera_power and thermal_char units, speed1 and desk2
self-checks, detector, mission and cycle units, and `phase1.py --dry-run --mock` →
`NOT VALID — DRY RUN (all hardware mocked)`, `thermal_retries []`. The reviewer could not rerun Python in its sandbox (exit
182) and claims no rerun.

## 8. RUN.md
Heading CAMPAIGN_P1_FIX3D. The Fix3d paragraph covers the cause, the per-thread file, retry rules, honest timing bounds, the stop-status rule, the
evidence fields and lag_probe. Step 1 rehearsal is **optional** (stem `owner_rehearsal_p1_fix3d`). Step 2 is the full session:
```sh
(set -C; python -u ~/robot/benchmark/campaign/phase1.py --output ~/storage/downloads/campaign/owner_session_p1_fix3d.json > ~/storage/downloads/campaign/owner_session_p1_fix3d.stdout 2> ~/storage/downloads/campaign/owner_session_p1_fix3d.stderr)
```
Files to send (only those created): `owner_session_p1_fix3d.json`, `.stdout`, `.stderr`, `_llama-server.log`, `_warmup_L0.json`,
`_block_01_L0.json`, `_block_02_L1.json`, `_block_03_L2.json`, `_block_04_L3.json`, `_block_05_L4.json`, `_block_06_L0.json`.
The phone must be upright facing the table, otherwise live M2 is 0 again. The verification-limits section is updated, and live
validation is explicitly pending.

## 9. Review
Freeze: `benchmark/campaign/frozen_manifest_campaign_p1_fix3d.json` (base b723002…, review_round 3, 75 files). Every round
used a fresh isolated copy (git archive of base + frozen files, hashes verified), because code changed after rounds 1 and 2:
`/termux-home/review_campaign_p1_fix3d/` (r1), `_r2/`, `_r3/`. Requests (all < 60 KB, "Ponytail mode: off"):
`/termux-home/storage/downloads/campaign/review_request_campaign_p1_fix3d.txt` (58.7 KB), `..._r2.txt` (56.9 KB),
`..._r3.txt` (58.0 KB); docs diffs `review_campaign_p1_fix3d_docs{,_r2,_r3}.diff` alongside. Diagnostics: the `.stderr` files next to
each review.

Command (round N = 2, 3; for round 1 drop the `_rN` suffix from the directory and request names only):
```sh
cd /termux-home/review_campaign_p1_fix3d_rN && PONYTAIL_DEFAULT_MODE=off \
codex --no-daemon exec --skip-git-repo-check -m gpt-6.1-sol -c model_reasoning_effort="medium" \
  -s read-only - < /termux-home/storage/downloads/campaign/review_request_campaign_p1_fix3d_rN.txt \
  > review_campaign_p1_fix3d_rN.txt 2> review_campaign_p1_fix3d_rN.stderr
```

### Round 1 review (verbatim)
The candidate needs changes for thermal-stop handling, retry timing, and the remaining unchecked thermal worker.

Persistent incomplete reads fail closed; intermittent incomplete reads can recover, with raw evidence retained in the phase1 runner. However, a partial thermal-stop reading can be superseded by a normal retry.

The per-thread tag is correct for the identified collision. I found no other concurrent same-tag `cr.root` callers in phase1: package/version reads are main-thread-only (`runtime.py:252`), meminfo reads belong to the memory worker (`coresidency.py:227`, called by `runtime.py:277`), and diagnostics have their own tag (`diagnostics.py:57`). The remaining thermalservice reader is `lag_probe.py:59`; its unchecked behavior is addressed below.

The regressions meaningfully exercise the production parser, checked reader, rest checks with camera OFF/ON, and live-block end checks. The empty-then-success and shared-file-race scenarios distinguish the intended fix from base behavior, although the new suite requires adaptation to run against base’s API. I reran the reader, timing, rest, and live-block checks successfully. They miss the counterexamples below.

All 74 frozen files match the manifest. All 19 newly archived owner files match their Downloads sources byte for byte. `RUN_INDEX.md:18–20` faithfully records the owner-adjusted labels, 16 fallback calls, five fallback-context selectors, screen-lock caveat, thermal-row counts, and caps. `RUN.md:60,97,102–111` uses the correct fresh stems and matches the runner’s output filenames. Its timing claims need correction.

VERDICT: CHANGES REQUESTED

1. **High — retries can hide a thermal-stop reading.** [runtime.py:224](/data/data/com.termux/files/home/review_campaign_p1_fix3d/benchmark/campaign/runtime.py:224) treats `rc=0, status=4, skin=None` as retryable. If the next read contains normal status and valid skin, `dump_check()` succeeds. I reproduced this with read-only mocks. The stop remains only in incomplete-attempt evidence; it never triggers the stop path. Preserve a parsed stop status even when another field is incomplete, and add partial-stop-then-normal regressions for the reader and worker paths. The existing test at `self_check_fix3d.py:87` covers only a complete stop reading.

2. **Medium — the advertised retry window is not enforced, and total delay can greatly exceed two seconds.** [runtime.py:233](/data/data/com.termux/files/home/review_campaign_p1_fix3d/benchmark/campaign/runtime.py:233) checks the deadline before sleeping, then starts another attempt without rechecking. A mocked oversleep produced attempt starts at **0.0 and 3.2 seconds**. Each attempt also retains the **30-second** timeout (`runtime.py:203`): a 0.2-second incomplete read followed by a 29.9-second successful retry returned after **30.6 seconds**, which I reproduced. The 1.6-second calculation is correct for three 0.2-second reads, but it is not the worst case. `monitor_loop` preserves the grid by skipping slots after overruns; it does not guarantee unchanged sampling cadence. Enforce the remaining retry budget, test oversleep and slow reads, and correct `RUN.md:41–45` and the timing assertions in `self_check_fix3d.py:94`.

3. **Medium — the lag-probe thermal worker bypasses the required checked reader and evidence capture.** [lag_probe.py:59](/data/data/com.termux/files/home/review_campaign_p1_fix3d/benchmark/campaign/lag_probe.py:59) still calls `cr.read_dump`, so its rows lack attempts and raw incomplete-output evidence, and read failures are deferred to `block_limit` rather than failing through the checked reader. Being the sole caller of its tag removes the collision but does not remove the rc-zero/incomplete-output failure mode. Its preflight now uses `rt.dump_check` (`lag_probe.py:30`), yet the result dictionary (`:33`) does not serialize `THERMAL_RETRIES`, so a recovered preflight read also loses raw evidence from the saved JSON. Route the worker through the checked reader, retain retry evidence, and add regressions for this runner.

### Round 2 review (verbatim)
Round-1 F1 and F3 are resolved. F2 is improved, but the claimed worst-case retry bound is still stronger than the implementation guarantees.

Persistent incomplete reads fail closed after three attempts or exhaustion of the retry budget. Intermittent missing readings can recover, as explicitly requested, with attempt counts and raw evidence retained; this cannot distinguish a temporary sensor dropout from truncated command output. A parsed status ≥ `STATUS_STOP` with missing skin now fails immediately (`runtime.py:235`). Complete stop readings remain subject to `dump_check()` and worker safety checks.

The per-thread tag is correct for the identified collision (`runtime.py:203`). Linux native thread IDs distinguish concurrent readers, and phase workers are joined before subsequent phases. I found no other concurrent same-tag `cr.root` users in the campaign runner. Other dumpsys readers are package metadata (`runtime.py:258`) and PSS (`coresidency.py:221–225`, through `runtime.py:283`); their tags have single owners and their failures remain fatal. Thermal workers in `phase1.py:405,561` and `lag_probe.py:60`, plus diagnostics (`diagnostics.py:69`), use the checked reader.

The regressions meaningfully cover recovery, persistent failure, immediate rc failure, partial-stop handling, camera-OFF/ON rest checks, and live-block end checks. Recovery and tag-isolation behavior distinguish the candidate from base, although the new suite needs adaptation to base’s API. The partial-stop tests catch round-1 F1; the lag lifecycle tests exercise F3’s worker and saved retry evidence. The remaining timing-test limitation is described below.

All **75 frozen SHA-256 entries match**. All **19 owner archive files match their Downloads-source SHA-256 values**. `RUN_INDEX.md:18–20` preserves the owner-adjusted labels and byte facts: 16 fallback calls, five fallback-context selectors, the screen-lock attribution distinguished from recorded errors, and the 90.2% camera-ON cap figure. `RUN.md:66,103,108–117` uses the correct fresh stems and matches the runner’s output filenames.

Archived verification records 16 checks and the mock dry run with exit code zero. I inspected the candidate and recorded outputs; independent Python execution attempts returned exit code 182, so I do not claim a successful rerun. No files were changed and no live validation was performed.

VERDICT: CHANGES REQUESTED

1. **Medium — F2 remains incomplete: a successful retry can finish outside the deadline and still pass.** At `benchmark/campaign/runtime.py:223`, the remaining budget becomes a relative `subprocess.run()` timeout. Python starts that timeout after `Popen` construction; process creation, wrapper processing and parsing can consume additional time. The native Python implementation confirms this ordering at `subprocess.py:554–556,1217`. `runtime.py:226–231` then accepts complete output without checking the completion time against `deadline`.

   For example, with 1.5 seconds remaining, a retry can spend three seconds in process creation and then return complete output promptly; the reader accepts it outside the advertised window. `self_check_fix3d.py:113–121` assumes the timeout bounds the entire root call, so its sweep misses this case.

   Check the deadline again before launching a retry and before accepting its result, and add a regression for delay outside the subprocess communication timeout. Correct `RUN.md:41–45,223–225` to distinguish the two-second retry budget from unavoidable launch, scheduling and cleanup overhead. Under the idealized timeout model, total reader latency can approach **32 seconds**, because the first attempt retains 30 seconds. The worker grid is preserved, but crossing a slot boundary can add a **4.87-second skipped-slot gap**; the demonstrated 2.2-second read is not a worst-case cadence test.

### Round 3 review (verbatim) — final
All round-1 findings are resolved, including round-2’s late-success counterexample. I found no new blocking regression.

I read both full reviews, the docs diff, candidate source and archived checks. All **75 frozen SHA-256 entries match**; all **19 owner archive files match their Downloads sources**. Python execution returned exit code 182, so I do not claim an independent test rerun. No files were changed or hardware run. Approval is a review finding only.

VERDICT: APPROVED

1. **Sensor failures — R1 F1 resolved.** `benchmark/campaign/runtime.py:238` immediately rejects rc failures, reader errors and partial readings with parsed status ≥ `STATUS_STOP`. Complete stop readings reach `dump_check()` or worker safety checks. Persistent incomplete reads fail after three attempts or window exhaustion. Intermittent missing readings can recover intentionally; retained raw evidence exposes the dropout, but cannot distinguish a temporary sensor failure from truncated output. A sensor returning plausible, incorrect values remains outside this fix’s detection capability.

2. **Thread tags and worker coverage — R1 F3 resolved.** `runtime.py:203` uses native thread IDs, which distinguish concurrent readers. Phase workers are joined before subsequent phases. Thermal workers at `phase1.py:405,561`, diagnostics at `diagnostics.py:68`, and the lag worker at `lag_probe.py:60` use the checked reader; `lag_probe.py:34` retains retry evidence. I found no other concurrent same-tag root users. Other dumpsys readers are package metadata (`runtime.py:261`, main thread only) and PSS (`benchmark/coresidency/coresidency.py:221–227`, memory worker only). They do not share the identified collision mechanism, and incomplete results remain fatal.

3. **Timing — R1 F2 and R2 F2 resolved, with explicit limits.** `runtime.py:224` passes the remaining budget to retries; `:244` checks after sleeping; `:227` rejects late complete results. The accepted-retry allowance is two seconds after the first read, not a two-second total reader guarantee. The first attempt retains 30 seconds, so idealized total latency can approach **32 seconds**; launch, scheduling and cleanup overhead prevent a hard wall-clock bound. `coresidency.py:702–706` preserves the grid by skipping slots. Crossing a boundary can add a **4.87-second** gap, and sufficiently long reads skip multiple slots. `RUN.md:41–48` acknowledges these qualifications; the 1.6-second example is typical-case arithmetic.

4. **Regressions are meaningful.** `self_check_fix3d.py:49,103,159,221,253` exercises recovery, persistent failure, immediate rc failure, partial-stop preservation, delayed launch, tag isolation, camera OFF/ON rest checks and live-block end checks. Recovery and tag isolation distinguish the candidate from base behavior, although the new suite requires adaptation to base’s API. The late-launch assertions catch the R2 implementation. `self_check.py:221` covers the lag worker and serialized retry evidence. Archived results record all 16 checks and the mock dry run exiting zero.

5. **Evidence and owner commands are faithful.** `RUN_INDEX.md:18–20` preserves 16 fallback calls, five fallback-context selectors, the owner’s screen-lock attribution separately from recorded byte facts, and the thermal/cap evidence. The race remains labelled likely rather than proven. `RUN.md:69,106` uses fresh fix3d stems; its file lists match `phase1.py:710–712,650`. Live validation remains explicitly pending at `RUN.md:236–241`.

## 10. Proposed commit (NOT made; waiting for "yes commit", then a separate "yes push")
Message: `benchmark: campaign phase-1 thermal read retry, session archives` (+ Co-Authored-By trailer).
The 3 `.log` files are Git-ignored (`*.log`) and need `git add -f`. Exact files (76):
```
benchmark/campaign/RUN.md
benchmark/campaign/RUN_INDEX.md
benchmark/campaign/checks/p1_fix3d/camera_power_units.stderr
benchmark/campaign/checks/p1_fix3d/camera_power_units.stdout
benchmark/campaign/checks/p1_fix3d/coder_dry_mock_p1_fix3d.json
benchmark/campaign/checks/p1_fix3d/coder_dry_mock_p1_fix3d_block_01_L0.json
benchmark/campaign/checks/p1_fix3d/coder_dry_mock_p1_fix3d_block_02_L1.json
benchmark/campaign/checks/p1_fix3d/coder_dry_mock_p1_fix3d_block_03_L2.json
benchmark/campaign/checks/p1_fix3d/coder_dry_mock_p1_fix3d_block_04_L3.json
benchmark/campaign/checks/p1_fix3d/coder_dry_mock_p1_fix3d_block_05_L4.json
benchmark/campaign/checks/p1_fix3d/coder_dry_mock_p1_fix3d_block_06_L0.json
benchmark/campaign/checks/p1_fix3d/coder_dry_mock_p1_fix3d_warmup_L0.json
benchmark/campaign/checks/p1_fix3d/coresidency_units.stderr
benchmark/campaign/checks/p1_fix3d/coresidency_units.stdout
benchmark/campaign/checks/p1_fix3d/cycle_units.stderr
benchmark/campaign/checks/p1_fix3d/cycle_units.stdout
benchmark/campaign/checks/p1_fix3d/desk2_self_check.stderr
benchmark/campaign/checks/p1_fix3d/desk2_self_check.stdout
benchmark/campaign/checks/p1_fix3d/detector_units.stderr
benchmark/campaign/checks/p1_fix3d/detector_units.stdout
benchmark/campaign/checks/p1_fix3d/dry_run_mock.stderr
benchmark/campaign/checks/p1_fix3d/dry_run_mock.stdout
benchmark/campaign/checks/p1_fix3d/duty_cycle_units.stderr
benchmark/campaign/checks/p1_fix3d/duty_cycle_units.stdout
benchmark/campaign/checks/p1_fix3d/mission_units.stderr
benchmark/campaign/checks/p1_fix3d/mission_units.stdout
benchmark/campaign/checks/p1_fix3d/power_map_units.stderr
benchmark/campaign/checks/p1_fix3d/power_map_units.stdout
benchmark/campaign/checks/p1_fix3d/results.json
benchmark/campaign/checks/p1_fix3d/self_check.stderr
benchmark/campaign/checks/p1_fix3d/self_check.stdout
benchmark/campaign/checks/p1_fix3d/self_check_fix2.stderr
benchmark/campaign/checks/p1_fix3d/self_check_fix2.stdout
benchmark/campaign/checks/p1_fix3d/self_check_fix3.stderr
benchmark/campaign/checks/p1_fix3d/self_check_fix3.stdout
benchmark/campaign/checks/p1_fix3d/self_check_fix3b.stderr
benchmark/campaign/checks/p1_fix3d/self_check_fix3b.stdout
benchmark/campaign/checks/p1_fix3d/self_check_fix3c.stderr
benchmark/campaign/checks/p1_fix3d/self_check_fix3c.stdout
benchmark/campaign/checks/p1_fix3d/self_check_fix3d.stderr
benchmark/campaign/checks/p1_fix3d/self_check_fix3d.stdout
benchmark/campaign/checks/p1_fix3d/speed1_self_check.stderr
benchmark/campaign/checks/p1_fix3d/speed1_self_check.stdout
benchmark/campaign/checks/p1_fix3d/thermal_char_units.stderr
benchmark/campaign/checks/p1_fix3d/thermal_char_units.stdout
benchmark/campaign/diagnostics.py
benchmark/campaign/lag_probe.py
benchmark/campaign/phase1.py
benchmark/campaign/runs/archive_sha256_p1_fix3d.json
benchmark/campaign/runs/owner_rehearsal_p1_fix3c.json
benchmark/campaign/runs/owner_rehearsal_p1_fix3c.stderr
benchmark/campaign/runs/owner_rehearsal_p1_fix3c.stdout
benchmark/campaign/runs/owner_rehearsal_p1_fix3c_block_01_L0.json
benchmark/campaign/runs/owner_rehearsal_p1_fix3c_block_02_L1.json
benchmark/campaign/runs/owner_rehearsal_p1_fix3c_block_03_L2.json
benchmark/campaign/runs/owner_rehearsal_p1_fix3c_block_04_L3.json
benchmark/campaign/runs/owner_rehearsal_p1_fix3c_block_05_L4.json
benchmark/campaign/runs/owner_rehearsal_p1_fix3c_block_06_L0.json
benchmark/campaign/runs/owner_rehearsal_p1_fix3c_llama-server.log
benchmark/campaign/runs/owner_rehearsal_p1_fix3c_warmup_L0.json
benchmark/campaign/runs/owner_session_p1_fix3c.json
benchmark/campaign/runs/owner_session_p1_fix3c.stderr
benchmark/campaign/runs/owner_session_p1_fix3c.stdout
benchmark/campaign/runs/owner_session_p1_fix3c2.json
benchmark/campaign/runs/owner_session_p1_fix3c2.stderr
benchmark/campaign/runs/owner_session_p1_fix3c2.stdout
benchmark/campaign/runs/owner_session_p1_fix3c2_llama-server.log
benchmark/campaign/runs/owner_session_p1_fix3c_llama-server.log
benchmark/campaign/runtime.py
benchmark/campaign/self_check.py
benchmark/campaign/self_check_fix2.py
benchmark/campaign/self_check_fix3.py
benchmark/campaign/self_check_fix3b.py
benchmark/campaign/self_check_fix3c.py
benchmark/campaign/self_check_fix3d.py
benchmark/campaign/frozen_manifest_campaign_p1_fix3d.json
```
DOC DIFF for Doc Keeper (not written by me): campaign phase-1 thermal reads now retry incomplete rc-0 reads (≤3 attempts, 2 s
window, raw evidence) and use per-thread root files; fix3c rehearsal and two incomplete sessions archived.
