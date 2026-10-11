# CAMPAIGN_P23B Coder report

Status: INCOMPLETE / CHANGES REQUESTED after independent review round 1. Requested all-six feasibility requirement fails on two archived curves. No commit or push; awaiting a revised owner task.

EXECUTION PROFILE: Coder Codex CLI, gpt-6.1-sol, reasoning high, Ponytail lite; proot Debian. Reviewer Claude Code headless, claude-opus-5-5, high, Ponytail off; exact P23 read-only configuration, isolated copy. Offline only, no root/camera/live/timed runs, motors OFF. No DOC DIFF.

Base: `49bc13bc6a3f1eb1fca242d344a4cdb1c9c77541`; initial working tree clean.

## Diff summary

- phase23.py: only SHORT adaptive 240 → 480; FULL, thresholds, hysteresis, minimums, 61 s reserve, coverage and readers unchanged.
- RUN.md: adaptive 480 s, about 16–19 min, one sentence with the specified Phase-1 heating/cooling numbers.
- self_check_p23.py: shifted-curve replay through actual run_mode/switch and actual restart-reserve logic, using a curve-backed MockSession; 240 reported, all-six 480 assertion retained and honestly failing. Step-held raw dumps, no extrapolation; G=30 C is arbitrary because only temperature offsets matter. First dump defines curve t=0; mock transitions are 3 s start and 2 s stop. OFF curve begins at confirmed OFF and is shifted to the high-switch skin.
- checks/p23b: same 20-command suite as P23, mock coder_mock_p23b, full logs/results/inventory; frozen_manifest_campaign_p23b.json records hashes/bytes.

## Feasibility

Adaptive feasibility ESTIMATE from shifted curves; NOT VALID as a hardware result.
Step-held dumps, first dump at t=0; G=30 C; mock 3 s start / 2 s stop; OFF curve begins at confirmed OFF, shifted to high-switch skin; no extrapolation.
| Pause | Window s | High-switch s | Low-restart s | Remaining s | Result |
|---|---:|---:|---:|---:|---|
| pause_01_L0 | 240 | 60.0 | — | — | FAIL |
| pause_01_L0 | 480 | 60.0 | — | — | FAIL |
| pause_02_L1 | 240 | 60.0 | — | — | FAIL |
| pause_02_L1 | 480 | 60.0 | — | — | FAIL |
| pause_03_L2 | 240 | 60.0 | — | — | FAIL |
| pause_03_L2 | 480 | 60.0 | 316.1 | 163.9 | PASS |
| pause_04_L3 | 240 | 60.0 | — | — | FAIL |
| pause_04_L3 | 480 | 60.0 | 403.7 | 76.3 | PASS |
| pause_05_L4 | 240 | 60.0 | — | — | FAIL |
| pause_05_L4 | 480 | 60.0 | 311.3 | 168.7 | PASS |
| pause_06_L0 | 240 | 60.0 | — | — | FAIL |
| pause_06_L0 | 480 | 60.0 | 389.1 | 90.9 | PASS |

The high-switch skin is G + 2.987700 C, so restart at G − 1 C requires 3.987700 C cooling. Across their entire archives, pause01 cools at most 3.590299 C and pause02 at most 3.869300 C; under the requested first-dump alignment, increasing the window alone cannot make these two raw shifted curves cross the low threshold within the recorded archives. This conclusion is specific to that alignment; the unrecorded early camera-OFF cooling is not credited, and later real cooling is not extrapolated. No success was fabricated, thresholds adjusted, or curves scaled. The assertion is deliberately retained as requested, so the candidate is not ready for commit. A task revision is required to resolve the inconsistency.

## Prior final P23 reviewer verdict (verbatim)

**Verdict: APPROVED.** All round-1 and round-2 findings are resolved and the owner's M3 decision is implemented as stated. I found four new Low items; none blocks approval.

Final standalone verdict line, verbatim:

> APPROVED

## Proposed commit message (pending successful verification and review)

`campaign: extend rehearsal adaptive window to 480 seconds (P23B)`

No commit authorization requested while checks fail. Commit requires owner “yes commit”; push requires a separate “yes push”.

## Self-check results

19 of 20 commands exited 0. Only the new feasibility assertion fails; all preceding P23 regression checks passed in that command. Full stdout/stderr retained in checks/p23b.

| Command | Exit |
|---|---:|
| `benchmark/campaign/self_check.py` | 0 |
| `benchmark/campaign/self_check_fix2.py` | 0 |
| `benchmark/campaign/self_check_fix3.py` | 0 |
| `benchmark/campaign/self_check_fix3b.py` | 0 |
| `benchmark/campaign/self_check_fix3c.py` | 0 |
| `benchmark/campaign/self_check_fix3d.py` | 0 |
| `benchmark/campaign/self_check_fix4.py` | 0 |
| `benchmark/campaign/self_check_fix4c.py` | 0 |
| `benchmark/campaign/self_check_fix4d.py` | 0 |
| `benchmark/campaign/self_check_p23.py` | 1 |
| `benchmark/relate_anything/speed1/self_check.py` | 0 |
| `benchmark/relate_anything/desk2/self_check.py` | 0 |
| `benchmark/coresidency/test_coresidency.py` | 0 |
| `benchmark/power_map/test_power_map.py` | 0 |
| `benchmark/thermal_char/test_thermal_char.py` | 0 |
| `benchmark/duty_cycle/test_duty_cycle.py` | 0 |
| `benchmark/camera_power/test_camera_power.py` | 0 |
| `test_detector_robotcam.py` | 0 |
| `test_run_mission.py` | 0 |
| `run_cycle_safety_test.py` | 0 |

Standalone `coder_mock_p23b`: exit 0, `NOT VALID — DRY RUN (all hardware mocked)`; `rehearsal_pass=false` as required for mock evidence.

Frozen manifest: `benchmark/campaign/frozen_manifest_campaign_p23b.json` (92 source/evidence entries; excludes itself).

## Alignment and review limitations

These are estimates from shifted curves, NOT VALID as hardware results. The archived L2 curve begins with inference and the camera already on, rather than at the actual camera-start instant. The pause archives omit an early cooling interval; crediting it would change the estimate, and was not requested. The reviewer calculates that even crediting that gap still leaves pause01/pause02 failing 480 s (approximately 633/594 s windows instead). Cool-start curves shifted upward can be pessimistic proxies; a real low restart below G may also approach equilibrium and remain unreachable. The user-specified 2.6 C lower bound at 300 s differs from the reviewer's first-dump calculation of 2.75 C. Readers, thresholds, curve bytes and acceptance criteria have not been changed to force success.

The full review patch retains the original absolute prefixes on new artifacts (F4); it is review evidence, not an apply-ready patch. The actual candidate files in the isolated copy match the manifest. No additional review round was started because the blocker requires an owner task revision, and no source changes were made after review.

## Review round 1 — complete final response, verbatim

REVIEW_BEGIN
# CAMPAIGN P23B — Reviewer round 1 final review

**Reviewer:** Claude Code headless, claude-opus-5-5, effort high, Ponytail off. Offline and read-only. I made no edits, staging, commits, pushes, hardware/root/camera access, model loads, timed or live runs, and launched no other reviewer. I did not run the repo's own self-check harness. Everything I computed was done in-process with `python3 -I -B -c` reading JSON, with no writes. The before/after byte-identity record belongs to the Coder.

## What I verified myself

- **Patch:** `review_full_diff_campaign_p23b_r1.patch` is 6213615 bytes, SHA-256 `2af7eb40…58d9ab`, matching the request.
- **Frozen manifest:** all 92 entries in `frozen_manifest_campaign_p23b.json` match the copy, with 0 mismatches.
- **Raw archive:** all 20 `owner_session_p1_fix4d_*` files match both `runs/archive_sha256_p1_fix4d.json` and the base `frozen_manifest_campaign_p23.json`. The patch does not touch `runs/`.
- **"Nothing else moves":** I reverse-applied the patch's hunks for `RUN.md`, `phase23.py` and `self_check_p23.py` in memory. All three come out byte-identical to the base P23 manifest hashes. Base `49bc13b` is the P23 commit, per `commit_campaign_p23.json`. So the source change is exactly:
  - `phase23.py:30` `adaptive=480`
  - the `RUN.md` hunk at lines 383–386
  - the new `check_adaptive_feasibility` and its call in `self_check_p23.py`
- **Unchanged rules:** FULL, `switch()` (`phase23.py:34-40`), the rehearsal T_hi = G+1 (`:411`), the 61 s reserve (`:426`, `:468`), coverage (`:547-568`) and all readers are unchanged.
- **No stray 240 values:** the only remaining `240` in campaign `.py`/`.md` files is the deliberate report-only window (`self_check_p23.py:525`). `RUN.md`'s duration change from 12–15 to 16–19 min equals the extra 240 s.
- **Mock run:** `coder_mock_p23b_adaptive.json` has planned 480 s, TIME LIMIT, no issues, 3 camera restarts, and correct high/low switches. `rehearsal_pass=false` because it is a mock.
- **Check logs:** `checks/p23b/results.json` shows only `self_check_p23.py` with exit 1. The stderr is the feasibility `AssertionError` for pause_01 and pause_02. All earlier P23 PASS lines are present in that log.

## The main question: replay bug or contradictory task?

**It is a contradictory task expectation, not a replay bug.**

**The harness uses the real controller.** `check_adaptive_feasibility` (`self_check_p23.py:485-541`) drives the real `q.run_mode`, with the real `switch()`, the real reserve and `dry=True`. That last part means the real `high = gate skin + 1` rule sets T_hi = 31 from G = 30. The curve subclass behaves correctly:

- It starts the heating curve when the camera-start call begins, at skin G.
- Heating continues through the 2 s mock stop.
- Cooling starts at confirmed OFF, offset to the skin at the switch.
- Dumps are step-held. Nothing is extrapolated, and the guard against running past the archive is never hit because the archives are about 604 s long.

**My own arithmetic, independent of the harness, reproduces every row:**

- **Heating:** with `block_03_L2` step-held, the skin at the earliest allowed switch (60 s) is G + **2.9877**. So reaching G − 1 takes **3.9877 C** of cooling.
- **Cooling available in each archive, from its first dump:**

| Pause | Start C | Drop at 120 s | Drop at 300 s | Max drop, whole archive | Time to 3.9877 C drop |
|---|---:|---:|---:|---:|---|
| pause_01 | 32.66 | 1.97 | 2.75 | **3.590** | never |
| pause_02 | 33.86 | 2.13 | 3.04 | **3.869** | never |
| pause_03 | 36.31 | 2.97 | 4.29 | 5.521 | 253.2 s |
| pause_04 | 35.56 | 2.32 | 3.77 | 5.067 | 341.0 s |
| pause_05 | 36.40 | 2.89 | 4.27 | 5.543 | 248.4 s |
| pause_06 | 36.53 | 2.41 | 3.85 | 5.530 | 326.3 s |

- **Agreement with the Coder's table:** adding about 62.4 s (60 s switch, 2 s stop, 0.37 s bracket) gives restart times of about 315.6, 403.4, 310.8 and 388.7 s. The Coder's table shows 316.1, 403.7, 311.3 and 389.1 s, a difference of about 0.3–0.5 s, which is the mock's 0.2 s polling. The 240 s window leaves at most about 117 s of cooling, and the largest 120 s drop is 2.97 C, so all six fail there too.

**Why the task is contradictory:** it says "needed ~2.5–3.5 C drop". But under its own premises the switch can't happen before 60 s (minimum active time), and L2 rises about 3 C in that first 60 s. So the switch happens near G+3, and restarting at G−1 needs about 4 C. Its own data says the 300 s drop is 2.6–4.3 C. For the two coolest-starting curves, no window length is enough under the specified alignment.

## Findings

**F1 — Blocks commit (needs an owner decision, not a code fix).** The required all-six 480 s assertion is false on the archived data. `self_check_p23.py:539` fails and the self-check exits 1. The code is correct and the failure is honest. Under the task's own rules this candidate can't be committed until the owner revises the task. I am not waiving the assertion or proposing threshold changes.

**F2 — Low: one claim in the Coder's report is too strong.** The report says "increasing the window alone cannot make these two raw shifted curves cross". That is only true under the literal alignment, where cooling starts at the archive's first dump:

- Each Phase-1 pause file starts about 18 s after the previous block's `diagnostics_end`.
- In that gap the skin had already fallen 0.58 C (warmup_L0 to pause_01) and 0.52 C (block_01 to pause_02). That is the steepest part of the cooling.
- If that early drop is credited, pause_01 and pause_02 would cross after about 492 s and 453 s of archive time. That means windows of about **633 s and 594 s**.
- 480 s still fails for both in this case, so F1 and the 480 s verdict stand. The report should say the conclusion depends on the alignment.

**F3 — Low (hardware risk the window change doesn't address):**

- **Structural limit:** restarting at T_hi − 2 = G − 1 means cooling *below the gate skin*. pause_01 and pause_02 bottom out at 29.1 and 30.0 C after starting at 32.7 and 33.9 C, close to the phone's resting floor. If a rehearsal's G is that close to equilibrium, the low restart is unreachable at any window.
- **Pessimistic proxy:** shifting cool-start curves up to a hot switch skin probably underestimates cooling, since cooling scales with how far the skin is above ambient. So pause_01 and pause_02 are likely pessimistic stand-ins.
- **Rehearsal order helps but doesn't guarantee:** adaptive runs right after endurance and fixed, with only 30 s idle and at most 30 s gate, so a warm G is likely but not certain.
- **Heating curve offset:** in `block_03_L2`, t=0 is the start of inference with the camera already on. The first read at t=0.03 is frame 20. So "rise after camera start" is approximate.

**F4 — Low (review artifact hygiene).** All 90 new-file entries in the review patch use `a/…`/`b/termux-home/robot/…` path prefixes. Running `git apply` at the repo root would put them in the wrong place. The files themselves are correct in the copy and verified against the manifest.

**F5 — Info.** `RUN.md`'s sentence uses the owner's figures: "2.0–3.0 C in 120 s, 2.6–4.3 C in 300 s". Measuring step-held from each archive's first dump, I get 1.97–2.97 and **2.75**–4.29. The 2.6 lower bound doesn't reproduce with this method. It isn't blocking because the text was dictated by the owner.

**F6 — Info.** The Coder report is still incomplete: review verdicts and the byte-identity proof are pending, which is expected at round 1. Its quote of the final P23 verdict ("APPROVED") is present.

## Limits

- I did not run any repo harness; my replay check is independent arithmetic on the same archives.
- I checked against the base through the committed P23 manifest, because the isolated copy has no git.
- I did not record byte identity of the copy or the live tree.

**Verdict:** the code and the replay are correct and match the task. The only blocker is F1: the task's own all-six 480 s requirement can't be met with the archived data. The owner needs to revise the task, for example on the window, the curve alignment, or the acceptance rule. The Coder should not change thresholds or curves on their own.

CHANGES REQUESTED
REVIEW_END

## Before/after byte-identity proof

The reviewer ran with the exact P23 restricted configuration. Exit 0; actual logged model `claude-opus-5-5`; effort high; Ponytail off. Request arrived on stdin; no settings sources or MCP servers. Full command and status:

```json
{
  "exit": 0,
  "copy_byte_identity": true,
  "live_byte_identity": true,
  "model": "claude-opus-5-5",
  "effort": "high",
  "ponytail": "off",
  "copy_files": 2587,
  "live_files": 4289,
  "command": "cd /termux-home/review_campaign_p23b_r1 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 claude -p 'Your complete review instructions are on standard input. Ponytail off. End with APPROVED or CHANGES REQUESTED.' --model claude-opus-5-5 --effort high --permission-mode dontAsk --setting-sources '' --strict-mcp-config --mcp-config '{\"mcpServers\":{}}' --tools Bash,Read,Grep,Glob --allowedTools 'Bash(python3 -I -B -c *)' 'Bash(cat *)' 'Bash(grep *)' 'Bash(ls *)' 'Bash(sha256sum *)' Read Grep Glob --disallowedTools Edit Write NotebookEdit WebFetch WebSearch --output-format json < /termux-home/storage/downloads/campaign/review_request_campaign_p23b_r1.txt\n"
}
```

- `campaign/review_identity_campaign_p23b_r1_before.json`: 1352645 bytes, SHA-256 `6dbc26519d4592dffec3421dfd159d031a66c1532075440e21fbbf9d2aca3d04`.
- `campaign/review_identity_campaign_p23b_r1_after.json`: 1352645 bytes, SHA-256 `6dbc26519d4592dffec3421dfd159d031a66c1532075440e21fbbf9d2aca3d04`.

copy: 2587 files; before/after full relative-path/byte-count/SHA-256 lists equal; canonical inventory SHA-256 `9cc1e8c5c397e5c1b8a1ef3df4086fdd6c8dbcd4cfde7964d3b1f18057d7d5e4` (sorted JSON keys, compact separators).

live: 4289 files; before/after full relative-path/byte-count/SHA-256 lists equal; canonical inventory SHA-256 `3bf88c2cbaed70388c6d3e48363271aac851699d029c440ad90159034ef1c67a` (sorted JSON keys, compact separators).

Frozen manifest remains valid after review: all 92 entries match. Manifest SHA-256 `3a9354cd251dace30be3aea50ed2d9e2475cf476b986d8a3d951ddcb54a73394`.

## Final disposition

CHANGES REQUESTED; task incomplete because the required feasibility assertion fails. Source and evidence remain frozen and uncommitted. Do not commit this failing candidate. Await an owner task revision; any source change requires fresh independent review. No DOC DIFF, root, camera, hardware, motors, model loads or live/timed runs. No commit or push.

Skipped: hardware validation and any unrequested thermal/controller changes. Risk: 480 s does not satisfy the specified all-six archive estimate, and no hardware rehearsal success is established.
