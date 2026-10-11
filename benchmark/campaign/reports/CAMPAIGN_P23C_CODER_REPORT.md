# CAMPAIGN_P23C Coder report

Status: committed and pushed `c0ba141dc1f6abce558f578cbae361a8d9cc16d1` to origin/main after separate owner approvals. Owner accepted all48 archive files and F3. Independent round2 APPROVED;21/21 offline checks pass; working tree clean.

EXECUTION PROFILE
- Coder: Codex CLI, gpt-6.1-sol, reasoning high, Ponytail lite; proot Debian.
- Reviewer: Claude Code headless, claude-opus-5-5, effort high, Ponytail off; same restricted read-only configuration as P23/P23B; isolated copy; byte identity of copy and live tree before/after every round. Maximum3 rounds; no verdict=incomplete.
- Offline only; no root, camera, model loads, motors, live/timed runs.

Base: d5d1e87c66db86848cc93495046cde8f6e3d75f6. Initial worktree clean. Actual repo path /termux-home/robot (same phone tree as /data/data/com.termux/files/home/robot).

## Evidence and root cause

All facts below are from unchanged owner_rehearsal_p23_01 JSON/checkpoints/stdout/stderr/server log and base source. The task's approximate sample description conflates memory start timestamps with receipt-time power samples. The interval94752.17–94754.64 spans the initial synchronous memory/PSS read and the transition to monitoring. There is no evidence of a running periodic power sampler stalling in that interval: its thread is started only AFTER the synchronous initial reads.

| Event | Monotonic seconds | Evidence |
|---|---:|---|
| Initial power read start / receipt | 94751.849840332 / 94751.921060669 | samples.power[0], duration0.071220337s |
| Initial fast read start / receipt | 94751.989222982 / 94751.995454915 | samples.fast[0] |
| Initial thermal read start / receipt | 94751.995786092 / 94752.170081299 | samples.dumps[0], attempts1 |
| Initial memory read START | 94752.170332275 | samples.memory[0].t is stamped BEFORE meminfo/root_sample (runtime.memory_sample); root_s2.21, rc0, retries0 |
| Worker thermal read START | 94754.638967205 | samples.dumps[1].t_start; proves workers have begun by then |
| Worker memory read START | 94754.639805909 | samples.memory[1].t, root_s0.62 |
| First raw checkpoint END | 94754.644365276 | endurance_samples_0001.end, only initial power receipt present |
| First idle origin | 94754.681470501 | endurance.idle.origin |
| First periodic power read start / receipt | 94754.707303549 / 94754.795496217 | samples.power[1] |
| First shared-check receipt | 94754.924562868 | shared_checks[0].ok=true |
| Idle initial skin read start / receipt | 94760.114481775 / 94760.272985071 | endurance.idle.skin_start |
| Idle observed end | 94790.274645079 | idle observed_end/end |

LiveSession.start ordering: prepare monitor shells/masks, load YOLO and M2, selector/cases/fallback, pin main, camera(False), initial power/fast/dump/memory reads, arm watchdog, start fast/dump/memory/shared workers, then start periodic power worker. `memory_sample` calls meminfo plus root_sample (battery, dumpsys meminfo for runner/llama-server/camera-provider; pidof/ps discover processes), and while camera OFF confirms RobotCam absence with another pidof. The initial root/PSS transaction accounts for the recorded2.21s; the memory path and remaining return/thread-launch work occupy the rest of the approximate2.469s interval. Exact root begin/end, absence-confirmation, thread launch and scheduling overhead are not independently timestamped: no per-subcommand attribution is claimed.

Both read_retries and thermal_retries are empty; no FIX4D or FIX3D retry caused this delay. No monitor/cleanup failure is recorded. The initial synchronous memory read is valid, not malformed or retried. Explicit gc.collect appears in runtime.release during preflight and final cleanup, not in LiveSession.start's initial-read/thread-launch sequence. Automatic GC is unlogged, so its contribution cannot be established or excluded. Preflight's model/server load, warmup and camera check precede these samples by source order. Server load_s5.084; server log says loaded at elapsed4.994575s and warmup complete6.0166s, with the next selector processing at elapsed71.366493s. Its clock is server-relative (no direct launch-origin mapping saved), so no fabricated monotonic server timestamp is assigned. No camera start or selector/M2/YOLO work occurs between these initial samples in campaign source. llama-server remains resident; its internal background activity is not separately traced.

The actual initial power receipt gap is2.874435548s (94751.921060669→94754.795496217). Idle starts0.114025716s before the second receipt. aggregate retains samples only from origin−2s: the first receipt is2.760409832s before origin and is excluded. Thus weighted sees no point at/before its left boundary and reports35.479148862s covered of35.593174578s, mean=null, gaps0. The gap count is0 because no bad pair remains in the filtered input; the missing head alone fails the unchanged full-coverage rule. This is first-origin exposure to synchronous startup, not evidence of a defective periodic sample reader. No sampler-start redesign is implemented or proposed beyond placing the origin after the requested settle gate.

Idle duration35.593174578s is correct under existing source: origin precedes camera(False) confirmation plus the initial checked skin read, consuming5.591514570s through skin_start receipt; the subsequent configured30s wait ends30.001660008s later. Camera(False) rechecks/force-stops even though already OFF and takes the memory-state lock. Exact portions spent waiting for that lock versus camera commands are unlogged. The initial thermal read itself is0.158503296s; the preceding5.433011274s includes camera/OFF work and any lock delay, without finer attribution. Preserve this phase definition and timings.

## Smallest complete change and scope

- phase23.py: setup-only settle helper, using unchanged receipt-time power rows and existing guarded wait. Check the latest10 consecutive receipts: every interval strictly positive and <=1.5s, last <=1.5s old at the accepted idle origin. Poll<=0.05s, deadline60s. Record setup.power_settle.settle_s/sample_count/total_samples/max_interval_s/settled/origin/last_sample_age_s. sample_count/max_interval describe the accepted10-sample tail; total_samples is all power receipts already available. No extrapolation, reader change or new transport.
- The initial raw checkpoint is written BEFORE settle so checkpoint I/O cannot make the accepted freshness stale. Accepted timestamp becomes first idle origin directly; no blocking setup read intervenes. setup_complete staysfalse through start/settle and becomes true only after settle. Existing setup retry accounting includes settling and preserves the invocation's exact ledger mark, passed from main as setup_mark; a regression injects retries after the earlier snapshot and during settle and verifies the four-retry failure cap. Timeout propagates through normal cleanup, failure_kind=setup and NOT VALID — SETUP FAILURE; no idle/active/pause measured window is created. Mocks retain the mandated mocked NOT VALID label.
- Rehearsal uses one shared setup and sampler; only its first idle needs this gate. Each measured session is a fresh invocation and therefore repeats it. Preflight-only returns before a sampler exists and before any idle origin, so settle is inapplicable; RUN.md says so. Phase1 and lag probe are untouched.
- self_check_p23c.py: independent scheduled receipts; startup/steady/stale/empty/short/slow/boundary cases, actual weighted first-idle integration, mocked main timeout/native failure classification and cleanup, differential all3 modes FULL/SHORT against d5d1e87 after a common origin. Existing setup-complete controller unit tests retain their established scope and outputs.
- RUN.md: one settle paragraph; every rehearsal stem changed to owner_rehearsal_p23_02 including command/file list/3 session arguments. RUN_INDEX and raw archive as requested.
- No changes to power.py/weighted/max_gap1.5, runtime readers, phase1 readers/preflight, thresholds, FULL/SHORT, FIX3D/FIX4D or P23B. No unrequested sampler redesign, dependencies or controls.

## Archive and count discrepancy

The task says50 files. The source glob AND recursive Downloads search contain48 files, and all48 have been archived unchanged, including the ignored llama-server log. Their breakdown is4 top-level main/stdout/stderr/server log +3 mode +3 idle +3 gate +1 endurance active +6 fixed active/pause +8 adaptive active/pause +20 checkpoint files =48. No missing referenced phase/checkpoint file was inferred from this inventory. `runs/archive_sha256_p23_01.json` records source, destination, byte count and SHA-256 for every actual file. No original file or historical label was changed. The owner was asked whether50 is a count mistake or two other files exist; absent clarification this remains an explicit task-count discrepancy, not fabricated completion of50.

Label NOT VALID — OWNER REHEARSAL; rehearsal_pass=false, only error power coverage. Passing:3 modes; live M2 endurance/fixed/adaptive7/2/11; fallback M2 and fallback selector contexts each1/mode; fixed pauses3; adaptive 4 high switches and 3 low restarts and camera restarts3; all camera restarts5, failed attempts0; read and thermal retries0. These remain diagnostic rehearsal evidence, NOT VALID as hardware results.

## Verification

P23C dedicated check passed: steady cadence gate3.35s (tenth receipt3.33s);2.47s startup gap gate5.85s; actual2.874435548s gap gate6.25s. All first-idle windows fully covered, watts1.0, gaps0. Ten stale, only nine, absent, and continuing2s samplers timeout at60s. Exactly1.5s intervals accepted. Warm-ready history adds0s gate delay. Polling overhead at most50ms in deterministic virtual-clock tests; OS scheduling is not a hardware timing guarantee. Main timeout creates no idle file or measured phase and cleans up. FULL/SHORT controller differential checks all3 mode objects and ALL saved phase/checkpoint JSON bytes at an identical accepted origin against d5d1e87.

Full suite: 21/21 commands exited0. stdout/stderr/results retained in checks/p23c.

- `benchmark/campaign/self_check.py`: exit0
- `benchmark/campaign/self_check_fix2.py`: exit0
- `benchmark/campaign/self_check_fix3.py`: exit0
- `benchmark/campaign/self_check_fix3b.py`: exit0
- `benchmark/campaign/self_check_fix3c.py`: exit0
- `benchmark/campaign/self_check_fix3d.py`: exit0
- `benchmark/campaign/self_check_fix4.py`: exit0
- `benchmark/campaign/self_check_fix4c.py`: exit0
- `benchmark/campaign/self_check_fix4d.py`: exit0
- `benchmark/campaign/self_check_p23.py`: exit0
- `benchmark/campaign/self_check_p23c.py`: exit0
- `benchmark/relate_anything/speed1/self_check.py`: exit0
- `benchmark/relate_anything/desk2/self_check.py`: exit0
- `benchmark/coresidency/test_coresidency.py`: exit0
- `benchmark/power_map/test_power_map.py`: exit0
- `benchmark/thermal_char/test_thermal_char.py`: exit0
- `benchmark/duty_cycle/test_duty_cycle.py`: exit0
- `benchmark/camera_power/test_camera_power.py`: exit0
- `test_detector_robotcam.py`: exit0
- `test_run_mission.py`: exit0
- `run_cycle_safety_test.py`: exit0

Frozen manifest: 29494 bytes, SHA-256 `422e346a59b2dca0de0fb522879402c329db76bef46a6a5b199dce078c309eab`. Own bytes excluded; all candidate source/check/archive paths included.

Standalone coder_mock_p23c: NOT VALID — DRY RUN (all hardware mocked); rehearsal_pass=False; setup settle={"settle_s": 0.3499999999999801, "sample_count": 10, "total_samples": 10, "max_interval_s": 0.37000000000000455, "settled": true, "origin": 101.34999999999998, "last_sample_age_s": 0.0}.

## Independent review round 1 — complete final response verbatim

REVIEW_BEGIN
# P23C Round 1 review: CHANGES REQUESTED

The settle fix itself is correct and matches the task. Two small things need fixing before commit: a wrong count in `RUN_INDEX.md` and a fragile way of recovering the setup re-read counter. Everything else checks out. The 48-vs-50 archive file count is still open.

## How I reviewed

- **No writes.** I used only Read/Grep and in-memory `python3 -I -B -c`. I ran no harness, wrote nothing, and touched no root, camera or models.
- **Patch.** It is 10,394,896 bytes with the expected sha256 `542fcb71…cd093`. It covers 146 files: the 145 manifest entries plus the manifest itself.
- **Base.** I read `d5d1e87`'s tree read-only from the live `.git`. Its `benchmark/campaign/phase23.py` is blob `4461b7f`, which matches the external base copy. That copy also matches the `phase23.py` sha in the P23B manifest. The candidate blob is `0be91ae`, matching the diff header. My own diff of base against candidate matches the source diff you supplied.
- **Manifest.** All 145 entries in `frozen_manifest_campaign_p23c.json` match by sha256. Compared with the P23B manifest, only `phase23.py` and `RUN.md` changed. `runtime.py`, `power.py`, `diagnostics.py` and `lag_probe.py` match the FIX4D manifest. `phase1.py` matches the P23 manifest. `self_check_p23.py` matches P23B.
- **Archive.** All 48 inventory entries match bytes and SHA-256 in both the copy and the original Downloads source paths. The 48 hashes are all different from each other. The source folder, the copy and the inventory hold exactly the same 48 names. Nothing named `owner_rehearsal_p23_02*` exists yet.

## Root-cause facts (checked against the archived JSON)

- **Power receipts.** The initial power sample arrived at 94751.921061. The next one, from the sampler thread, arrived at 94754.795496. That is a 2.874436 s gap and the only interval over 1.5 s among 2,605 receipts.
- **What 94752.170 is.** It is `memory[0].t`, and `runtime.py:583` stamps that time *before* the read runs. That read took `root_s` 2.21 s.
- **Source order.** In `LiveSession.start`, the power, fast, dump and memory reads all run before any worker or sampler thread starts (`phase23.py:177-189`). Workers started at about 94754.639. The first checkpoint ended at 94754.644 holding only that one power receipt.
- **The failure.** Idle origin was 94754.681471. The first receipt after origin came 0.114026 s later. `aggregate` keeps only samples from origin−2 s onward, so the 94751.921 receipt was dropped. The window reports covered 35.479149 of 35.593175 s, gaps 0, mean null. This fails the unchanged full-coverage rule.
- **Retries and GC.** Re-read count is 0 and thermal retry count is 0. The only explicit `gc.collect` is in `runtime.release` (`runtime.py:671`), outside this sequence.
- **Idle length.** The skin-read start was origin+5.433 s and its result arrived at origin+5.5915 s; the 30 s wait then took 30.00166 s. This is unchanged base behaviour.

The coder's evidence-discrepancy statements are accurate. The sampler did not stall; the first idle origin simply came before the sampler thread had produced its first receipt.

## Settle and how it fits in

**`settle_power` (`phase23.py:364-378`)**
- It checks the last 10 receipts: 9 intervals, each strictly positive and ≤1.5 s.
- Freshness is checked against the same `now` that becomes the idle origin. `guard()` runs before the samples and `now` are read, so checkpoint writes cannot make an accepted sample stale.
- It polls every ≤50 ms through the existing guarded `wait`, so the heartbeat, cores, monitor errors, emergency limit and battery checks all stay active.
- It gives up after 60 s from when settle starts. The `<=60` acceptance check plus the `>=60` raise is correct.
- It records `settle_s`, sample count, total samples, max interval, `settled`, origin and last-sample age.
- Concurrent reads of `data['power']` are safe because the sampler appends to the same list object.

**Where it runs (`phase23.py:404-410`)**
- Only before the first idle of an invocation. In a rehearsal, later modes skip it, which matches the task.
- Preflight-only returns before any sampler exists (`phase23.py:661-664`), which is correct.

**Timeout behaviour**
- The `RuntimeError` is raised before `row['idle']` exists, so no idle, gate or phase file is written.
- `main` labels it `NOT VALID — SETUP FAILURE` with `failure_kind='setup'`, and the normal cleanup in `finally` runs.
- A `BatteryStop` during settle keeps the battery label.
- Rehearsal coverage fails because `modes != MODES` and an error is set.

**No relaxation.** `power.weighted`/max_gap, the readers, FIX3D/FIX4D, thresholds and FULL/SHORT are untouched. The P23B controller after the origin is unchanged. The differential compares runs and all JSON bytes for all three modes in FULL and SHORT. That differential, plus `check_settle`'s independent receipt schedules (0.37 s, 2.47 s, and the owner's exact 2.874436 s gap), is an adequate test.

## Findings

**F1 (Low, code) — `phase23.py:406`: the setup re-read counter start point is reconstructed instead of carried over.**
- The line `mark=len(rt.READ_RETRIES)-len(result['setup'].get('read_retries',[]))` assumes nothing was added to `READ_RETRIES` between `main`'s check at `:668` and this line.
- But the monitor and sampler threads are already running then, and the gap includes `checkpoint(True)`'s fsync'd write.
- If k background re-reads land in that gap, `setup.read_retries` drops the earliest k setup entries, and those k themselves are counted nowhere.
- The cap is never weaker than base: the count is the base setup count plus re-reads during settle.
- Still, the record can be wrong, and the report's claim that it "preserves the invocation's ledger mark" is not true when this happens.
- **Fix:** pass the exact mark from `main` (for example as a `run_mode` argument or a session attribute). Add a mock that appends a re-read in the gap and one during settle.

**F2 (Low, evidence accuracy) — `RUN_INDEX.md:64`: "adaptive high and low x3" is wrong.**
- The archived adaptive run has 4 high switches (at 60.2, 192.7, 319.3 and 446.3 s) and 3 low restarts.
- The coder report repeats this as "adaptive high+low3 each".
- **Fix:** change it to "4 high switches, 3 low restarts". The other counts on that line are correct.

**F3 (Info, scope).** `phase23.py:665` and `:679-681` relabel *every* failure before settle completes, including failures in `p.preflight`, in `session.start`, and in `--preflight` runs, from "SESSION INCOMPLETE" to "SETUP FAILURE" with `failure_kind='setup'`. They are still NOT VALID, and nothing in phase23 reads `failure_kind`. The owner should know about this.

**F4 (Info, archive).** The task says 50 files, but the source holds 48, and all 48 are archived unchanged. The archive is complete for what exists. The explicit 50-file requirement is not met and cannot be waived until the owner clarifies. This is separate from whether the code is correct.

**F5 (Info, commit).** `.gitignore` matches `*.log`, so `runs/owner_rehearsal_p23_01_llama-server.log` needs `git add -f`, or the committed archive will have 47 files.

**F6 (Info, tests).** The native path in `self_check_p23c.py` takes a lock on the real `~/.cache/campaign_p1.lock` and needs `git show` from a git checkout. No test covers a guard exception during settle; that is acceptable because it uses the existing guard.

## Docs

The `RUN.md` settle paragraph is accurate. The rehearsal stem is `_02` everywhere: the command, the file list and all three `--rehearsal` arguments. No `_01` rehearsal stem remains. `code_hashes()` also means the `_01` rehearsal can no longer authorize a session.

## Verdict

Required for round 2: F1 (pass the exact setup mark, plus a test) and F2 (correct the RUN_INDEX count, and the matching line in the report). F4 still needs the owner's clarification. This review does not authorize a commit.

CHANGES REQUESTED
REVIEW_END

## Identity proof round 1

```json
{
  "exit": 0,
  "copy_byte_identity": true,
  "live_byte_identity": true,
  "model": "claude-opus-5-5",
  "effort": "high",
  "ponytail": "off",
  "copy_files": 2730,
  "live_files": 4474,
  "command": "cd /termux-home/review_campaign_p23c_r1 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 claude -p 'Your complete review instructions are on standard input. Ponytail off. End with APPROVED or CHANGES REQUESTED.' --model claude-opus-5-5 --effort high --permission-mode dontAsk --setting-sources '' --strict-mcp-config --mcp-config '{\"mcpServers\":{}}' --tools Bash,Read,Grep,Glob --allowedTools 'Bash(python3 -I -B -c *)' 'Bash(cat *)' 'Bash(grep *)' 'Bash(ls *)' 'Bash(sha256sum *)' Read Grep Glob --disallowedTools Edit Write NotebookEdit WebFetch WebSearch --output-format json < /termux-home/storage/downloads/campaign/review_request_campaign_p23c_r1.txt\n"
}
```

- `/termux-home/storage/downloads/campaign/review_identity_campaign_p23c_r1_before.json`: 1417928 bytes, SHA-256 `7afcd859b7d6104b422f9e93df7b346157c9efce15fabba167079e91bb45807f`.
  copy: 2730 files; canonical inventory SHA-256 `ce768fbe887fb580e4a055e308764fb6acb6815ee7f32d17431e667302d96aaa`.
  live: 4474 files; canonical inventory SHA-256 `54799b9066209a9689f80730b4837fabc28477ad7d889be6584c5146a2ef40a0`.
- `/termux-home/storage/downloads/campaign/review_identity_campaign_p23c_r1_after.json`: 1417928 bytes, SHA-256 `7afcd859b7d6104b422f9e93df7b346157c9efce15fabba167079e91bb45807f`.
  copy: 2730 files; canonical inventory SHA-256 `ce768fbe887fb580e4a055e308764fb6acb6815ee7f32d17431e667302d96aaa`.
  live: 4474 files; canonical inventory SHA-256 `54799b9066209a9689f80730b4837fabc28477ad7d889be6584c5146a2ef40a0`.

Reviewer command/request/full patch/JSON/stderr/identity lists remain in Downloads/campaign.

## Independent review round 2 — complete final response verbatim

REVIEW_BEGIN
# P23C Round 2 review: APPROVED

Both round-1 findings are fixed. F1 is fixed in `main`, which now passes its exact setup mark to `run_mode` and has a regression test. F2 is fixed in `RUN_INDEX.md`, which now says 4 high switches and 3 low restarts. I found nothing new that blocks the change. The 48-vs-50 file count still needs your clarification. It is a question about the archive, not about whether the code is correct.

## How I reviewed

- **No writes.** I used only Read/Grep and in-memory `python3 -I -B -c` that hashes and parses files. I ran no harness, wrote nothing, and touched no root, camera or models. Motors stayed off.
- **One denied call.** One attempted in-memory run of `run_mode` (base against candidate, with `p.write` captured in memory) was denied by the permission layer. I did not retry it in another form. So my evidence for the after-origin identity is my own static diff plus the coder's logged differential, not a run of my own.
- **Patch.** It is 10,398,409 bytes with sha256 `1eba5416…0417f`, as expected. It covers 147 files: the 146 manifest entries plus the manifest itself. That includes the new, git-ignored `runs/owner_rehearsal_p23_01_llama-server.log`.
- **Manifest.** All 146 entries in `frozen_manifest_campaign_p23c.json` match the copy by sha256, with 0 mismatches.
- **Unchanged modules.** These match their last frozen manifests:
  - `runtime.py`, `power.py`, `diagnostics.py`, `lag_probe.py` and `self_check*.py` (FIX4D)
  - `phase1.py` and `self_check_fix4.py` (P23)
  - `self_check_p23.py` (P23B)
  - the other suite tests (FIX3B/FIX4D)
- **Base.** The external base file has git blob `4461b7f` and its sha256 equals the P23B manifest's `phase23.py`. The candidate's blob is `1709aab`. Both match the patch's index line. My own diff of base against candidate is exactly the source diff you supplied.
- **Archive.** The inventory has 48 entries. Every one matches bytes and SHA-256 in both the copy and its original Downloads source path, and all 48 hashes are unique. The copy, the source folder and the inventory hold the same 48 names. No `owner_rehearsal_p23_02*` files exist yet.
- **RUN_INDEX facts.** These match the archived JSON:
  - live M2 calls 7/2/11, fallback calls and contexts 1 per mode, 3 fixed pauses
  - adaptive: 4 high switches, 3 low restarts, 3 restarts
  - 5 camera restarts in total, 0 failed attempts
  - 0 read re-reads and 0 thermal retries; setup check `OK`
  - errors `[['power coverage']]`, `rehearsal_pass=false`

## Settle and how it fits in (`phase23.py`)

**`settle_power` (`:364-378`)**
- It runs `guard()` first, then takes the last 10 receipts and `now`.
- It accepts only if all of these hold:
  - exactly 10 samples, so 9 intervals, each strictly positive and ≤1.5 s
  - the last sample is between 0 and 1.5 s old at the same `now` that becomes the idle origin
  - `now-began` is ≤60
- Otherwise it raises at ≥60 s, polling every ≤50 ms through the guarded `wait`. Heartbeat, cores, monitor errors, emergency limit and BatteryStop checks all stay active while it waits.
- The 1.5 s limit matches `power.weighted(max_gap=1.5)`. Receipt `t` comes from `time.monotonic()` with `t0=0` (`power.py:28`), the same clock as `LiveSession.now`.
- It records `settle_s`, `sample_count`, `total_samples`, `max_interval_s`, `settled`, `origin` and `last_sample_age_s`.

**Where it runs (`:404-410`)**
- It runs once per invocation, gated by `setup_complete`, which `main` sets False (`:664`) after the preflight-only return (`:660-663`).
- Then `rt.retry_check(result['setup'], setup_mark)` uses the exact `mark` from `:657`, passed in at `:672`. **This resolves R1 F1.**
- The setup ledger is now everything from the mark through the end of settle. That is a superset of the base setup ledger, so the cap can only get stricter. A re-read logged by another thread between settle's `now` and `retry_check` could count in both setup and idle, which errs strict. Preflight-only has no sampler, so it correctly does not settle.

**Timeout behaviour**
- The `RuntimeError` is raised before `row['idle']` exists and before any `try` that writes phase files. No idle, gate, active or pause file is written.
- `main` labels it `NOT VALID — SETUP FAILURE` with `failure_kind='setup'` (`:678-680`), re-raises, and runs the normal `finally` cleanup (session close, server, screen, LMK).
- `rehearsal_pass` fails because the modes list is incomplete, `error` is set and `setup_complete` is false.
- The only files written are the samples checkpoint(s), which hold setup data. Base already wrote `checkpoint(True)` before idle too.

**Unchanged.**
- Everything after the origin is textually identical to base, and with `setup_complete=True` the origin expression is too.
- The readers, FIX3D/FIX4D, `weighted`/max_gap, idle coverage, thresholds, FULL/SHORT and the P23B controller are untouched.

## Tests (`self_check_p23c.py`) are adequate

- **`check_settle`.** It uses independent receipt schedules: startup gaps of 0.37 s, 2.47 s and the owner's exact 2.874436 s. Origin is delayed by exactly the gap plus 9 intervals, plus at most 50 ms. The real `run_mode` idle is then fully covered with 0 gaps.
- **Timeouts.** Stale, short, absent and 2 s-cadence streams each time out at exactly 60 s. An interval of exactly 1.5 s is accepted. A stale history waits for 10 new receipts.
- **`check_main_failure`.** It covers both the mock path and the native classification path.
- **Differential.** All modes × FULL/SHORT, comparing run objects and every JSON file's bytes against `git show d5d1e87` (blob confirmed to be `4461b7f`).
- **`check_setup_retries`.** It exercises the R1 F1 scenario exactly:
  - it excludes a previous invocation's re-read
  - it keeps re-reads after the snapshot and during settle
  - it confirms the cap fires at 4>3 with the `setup read retry cap` issue

## Remaining findings (none block the change)

- **F3 (Info, carried over).** `:678-680` labels any failure before settle completes as `SETUP FAILURE` / `failure_kind='setup'`. That covers `p.preflight`, `session.start` and `--preflight` failures, which base labelled `SESSION INCOMPLETE`. They are still NOT VALID, and nothing reads `failure_kind`.
- **F4 (Info, archive, unresolved).** The task says 50 files; only 48 exist, and all 48 are archived unchanged and inventoried. This needs your clarification and must not be waived or made up.
- **F5 (Info, commit).** `.gitignore` matches `*.log`, so `runs/owner_rehearsal_p23_01_llama-server.log` needs `git add -f` or the committed archive will have 47 files.
- **F6 (Info, tests).**
  - The native-path test takes the real `~/.cache/campaign_p1.lock`.
  - The differential needs a git checkout.
  - The 2.47 s case checks the settle function on its own, then a `run_mode` whose own settle finds the samples already settled (`settle_s==0`). No test has `run_mode` itself wait through a stall before idle. That is acceptable because the origin is just settle's return value.
  - No test shows the pre-fix base failing coverage on the stall.
- **F7 (Info).** On a settle timeout, the mode row stays in `result['runs']` with `IN PROGRESS — NOT VALID` and no idle. The top-level label is still `SETUP FAILURE`. Base behaves the same way for any failure before the try at `:445`.

## Docs

The `RUN.md` settle paragraph is accurate, including that preflight-only does not settle. The `_02` stem is used everywhere: the command, the file list and all three `--rehearsal` arguments. No `owner_rehearsal_p23_01` remains in `RUN.md`. The report draft says "adaptive4 high switches and3 low restarts" and includes the round-1 review verbatim.

This review is not permission to commit. The commit and push gates stay with you.

APPROVED
REVIEW_END

## Identity proof round 2

```json
{
  "exit": 0,
  "copy_byte_identity": true,
  "live_byte_identity": true,
  "model": "claude-opus-5-5",
  "effort": "high",
  "ponytail": "off",
  "copy_files": 2731,
  "live_files": 4475,
  "command": "cd /termux-home/review_campaign_p23c_r2 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 claude -p 'Your complete review instructions are on standard input. Ponytail off. End with APPROVED or CHANGES REQUESTED.' --model claude-opus-5-5 --effort high --permission-mode dontAsk --setting-sources '' --strict-mcp-config --mcp-config '{\"mcpServers\":{}}' --tools Bash,Read,Grep,Glob --allowedTools 'Bash(python3 -I -B -c *)' 'Bash(cat *)' 'Bash(grep *)' 'Bash(ls *)' 'Bash(sha256sum *)' Read Grep Glob --disallowedTools Edit Write NotebookEdit WebFetch WebSearch --output-format json < /termux-home/storage/downloads/campaign/review_request_campaign_p23c_r2.txt\n"
}
```

- `/termux-home/storage/downloads/campaign/review_identity_campaign_p23c_r2_before.json`: 1418312 bytes, SHA-256 `3f52f3534851b5a8633b1cfe84a2a22b8e8ccb1f8f71ec5faa057ecf0f606f83`.
  copy: 2731 files; canonical inventory SHA-256 `e70bc3049a327898705f92e8fd2be393461aee140f88e0091c2c8d3b44c0d22f`.
  live: 4475 files; canonical inventory SHA-256 `f26a5fe1826642a00cbe27b7da301dea38e5dbca2ea6fa9dcb0b781e6bbc2591`.
- `/termux-home/storage/downloads/campaign/review_identity_campaign_p23c_r2_after.json`: 1418312 bytes, SHA-256 `3f52f3534851b5a8633b1cfe84a2a22b8e8ccb1f8f71ec5faa057ecf0f606f83`.
  copy: 2731 files; canonical inventory SHA-256 `e70bc3049a327898705f92e8fd2be393461aee140f88e0091c2c8d3b44c0d22f`.
  live: 4475 files; canonical inventory SHA-256 `f26a5fe1826642a00cbe27b7da301dea38e5dbca2ea6fa9dcb0b781e6bbc2591`.

Reviewer command/request/full patch/JSON/stderr/identity lists remain in Downloads/campaign.

## Disposition

No stage/commit/push. Round1 requested changes to carry the exact retry mark and correct the adaptive switch count. Both are fixed and round2 is APPROVED. Both rounds passed copy/live byte identity. The final suite has21 passing commands; after the R1 edits only the two checks importing phase23 were rerun (both passed); the other19 test/code paths were unchanged. R1 checks/manifest are retained externally in campaign/p23c_round1_checks and campaign/frozen_manifest_campaign_p23c_r1.json. Setup failures before the first idle (including preflight/start failures) now have failure_kind=setup and the NOT VALID setup label; this is a label-only expansion noted as reviewer F3. The ignored archived llama-server.log must be force-added only after owner authorization. Stop for the owner. Independent review is complete and approved. Archive disposition:48 source files found and48 archived unchanged; no claim of completing a nonexistent50-file inventory, and no waiver of the task count. The owner must resolve that discrepancy. The reviewer did not rerun the differential: its attempted in-memory call was denied by the existing restricted tool configuration and not retried; it accepted the static diff plus coder test evidence. No post-review candidate edits were made. Commit requires “yes commit”, then push requires separate “yes push”.

Skipped: hardware/root/camera/model loads/live/timed runs; all unrequested sampler or controller redesign. Risk: no new hardware rehearsal success is established; historical rehearsal remains NOT VALID; file-count discrepancy is pending.

## Owner acceptance and commit authorization — 2026-10-10

Owner confirmed the planner miscounted:48 is the complete archive and is accepted in full. F3 is accepted. Owner explicitly instructed force-add of runs/owner_rehearsal_p23_01_llama-server.log, verification with git ls-files before committing, and “yes commit”. This resolves the earlier pending count/disposition statements, which are retained as historical review context. No candidate code or frozen artifact was changed after review. No push authorization was given.

Pre-commit verification: all146 manifest entries match; manifest SHA-256 matches the reviewed final verification;21/21 recorded checks pass; exactly147 approved files staged; git ls-files lists all48 archive files including the force-added log; every staged archive blob matches the original bytes/SHA-256 inventory. Source/docs/metadata whitespace checks pass. The full staged check reports one trailing space in the verbatim duty-cycle test stdout (line13); it is preserved because removing it would change frozen evidence. Full precheck evidence: campaign/commit_precheck_campaign_p23c.json.

## Local commit result — 2026-10-10

Commit `c0ba141dc1f6abce558f578cbae361a8d9cc16d1`: `campaign: settle power sampler before first idle (P23C)`. Parent `d5d1e87c66db86848cc93495046cde8f6e3d75f6`. Exactly147 approved paths committed, all146 manifest entries verified against committed blobs. git ls-files lists all48 owner-rehearsal archive files including the force-added llama-server.log; all committed archive bytes match original sources. Final git status is clean. No push, DOC DIFF, hardware/live run or additional test execution occurred. Existing tests were not rerun because the committed bytes match the reviewed/verified candidate. Commit evidence: campaign/commit_campaign_p23c.json.

## Push result — 2026-10-10

Owner separately authorized “yes push”. `git push origin main` exited0:

```text
To https://github.com/abdelhamidzoghlamii-ui/Pixel-robot.git
   d5d1e87..c0ba141  main -> main
```

`git ls-remote origin refs/heads/main` independently returned c0ba141dc1f6abce558f578cbae361a8d9cc16d1. Local HEAD and origin/main match it. Final git status: `## main...origin/main`, with a clean working tree. No additional tests or hardware/live runs. Push evidence: campaign/push_campaign_p23c.json.
