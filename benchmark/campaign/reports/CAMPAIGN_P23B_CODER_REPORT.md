# CAMPAIGN_P23B Coder report — owner revision P23B-R2

Status: APPROVED in review round 2; ready for owner commit decision. No commit or push. Base `49bc13bc6a3f1eb1fca242d344a4cdb1c9c77541`; initial P23B working tree was clean. P23B round 1 failed honestly; the owner revised the task, and its original report/checks/manifest remain preserved externally under campaign/.

EXECUTION PROFILE: Coder Codex CLI, gpt-6.1-sol, reasoning high, Ponytail lite; proot Debian. Reviewer Claude Code headless, claude-opus-5-5, high, Ponytail off; same P23 restricted read-only configuration, isolated copy. Round 2 of max 3. Offline only, motors OFF. No root, camera, hardware, model loads or live/timed runs. No DOC DIFF. Commit requires owner “yes commit”; push requires separate “yes push”.

## Diff summary

- phase23.py: SHORT adaptive 240 → 480; explicit low argument to switch(). Measured controller uses high−2. Rehearsal adaptive pauses use the checked high-switch trigger−2, and record switch_trigger_skin_c and low_threshold_c. The triggering skin value is captured once for the rehearsal decision and stored unchanged. Measured pause/switch output schema and skin/status reads are preserved. FULL, minimum active 60 s / confirmed-OFF 30 s, 61 s reserve, high-switch coverage and hardware readers remain unchanged. Low-restart coverage uses each pause's own recorded low threshold, failing closed if absent.
- RUN.md: about 16–19 min, adaptive 480 s, new rehearsal low rule, measured high−2 retained, corrected step-held cooling ranges 1.97–2.97 C at 120 s and 2.75–4.29 C at 300 s. The rationale is specific to archived L2 overshoot rather than a universal guarantee for every heating proxy.
- self_check_p23.py: measured fixed/adaptive FULL run_mode outputs and every emitted JSON byte compared with the dynamically loaded source at 49bc13b; boundary decision/threshold tests; overshoot and changing-reading trigger tests; per-pause coverage test. Curve adapter and first-dump alignment preserved. L2 × six new-rule curves at 480 s asserted; complete 144-row matrix reported only.
- checks/p23b/: full refreshed 20-command suite, standalone coder_mock_p23b, result/inventory logs. frozen_manifest_campaign_p23b.json freezes all candidate source/evidence. Agent orchestration scripts and reviewer copies remain outside repo.

## Measured-session preservation

The differential mock runs use dry=False and the unchanged FULL spec. For fixed and adaptive, the candidate run results and every phase/checkpoint JSON file are byte-identical to 49bc13b. Boundary switch decisions also match; switch rows retain high=37 C and low=35 C. The overshoot test proves rehearsal trigger40→low38 versus measured high37→low35; another mock advances the skin between reads and proves the rehearsal records the actual checked trigger, not a subsequent reading.

This proves deterministic controller/output parity for the tested mocks, supported by the guarded source diff; it is not a hardware measurement. Top-level provenance code hashes necessarily change because the source changed, and timing/environment fields are inherently variable. No claim of identical live-run bytes is made.

## Required L2 feasibility table

ESTIMATE FROM SHIFTED CURVES — NOT VALID as hardware results. Same first-dump time/temperature alignment and step-held samples; G=30 C, 3 s mock start / 2 s stop, OFF cooling starts at confirmed OFF, shifted to switch skin; no curve extrapolation. The L2 archive starts with inference and camera already on.

| Heating | Pause | Rule | Window s | High-switch s | Low-restart s | Remaining s | Result |
|---|---|---|---:|---:|---:|---:|---|
| block_03_L2 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_03_L2 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_03_L2 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_03_L2 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_03_L2 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_03_L2 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |

All six L2 curves restart within 480 s with >=61 s left. The broader report-only matrix gives new-rule 36/36 at 480 s, worst restart 184.74 s; 30/36 at 240 s. Old G−1 gives 13/36 at 480 s (rather than the planner's approximate 14/36), and 0/36 at 240 s. No values were tuned to match expectations.

## Complete feasibility matrix (report-only except required six rows above)

Adaptive feasibility ESTIMATE from shifted curves; NOT VALID as a hardware result.
Step-held dumps, first dump at t=0; G=30 C; mock 3 s start / 2 s stop; OFF curve begins at confirmed OFF, shifted to high-switch skin; no extrapolation. Archived heating starts with camera already on.
ASSERT only L2 heating x six pauses, new rule, 480 s. All other matrix rows REPORT ONLY.
| Heating | Pause | Rule | Window s | High-switch s | Low-restart s | Remaining s | Result |
|---|---|---|---:|---:|---:|---:|---|
| block_01_L0 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_01_L0 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_01_L0 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_01_L0 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_01_L0 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_01_L0 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_03_L2 | old G-1 | 480 | 60.0 | 247.9 | 232.1 | PASS |
| block_01_L0 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_01_L0 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_01_L0 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_04_L3 | old G-1 | 480 | 60.0 | 330.7 | 149.3 | PASS |
| block_01_L0 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_01_L0 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_01_L0 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_05_L4 | old G-1 | 480 | 60.0 | 257.7 | 222.3 | PASS |
| block_01_L0 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_01_L0 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_01_L0 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_01_L0 | pause_06_L0 | old G-1 | 480 | 60.0 | 316.1 | 163.9 | PASS |
| block_02_L1 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_02_L1 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_02_L1 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_02_L1 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_02_L1 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_02_L1 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_03_L2 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_02_L1 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_02_L1 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_04_L3 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_02_L1 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_02_L1 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_05_L4 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_02_L1 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_02_L1 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_02_L1 | pause_06_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_03_L2 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_03_L2 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_03_L2 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_03_L2 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_03_L2 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_03_L2 | old G-1 | 480 | 60.0 | 316.1 | 163.9 | PASS |
| block_03_L2 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_03_L2 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_03_L2 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_04_L3 | old G-1 | 480 | 60.0 | 403.7 | 76.3 | PASS |
| block_03_L2 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_03_L2 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_03_L2 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_05_L4 | old G-1 | 480 | 60.0 | 311.3 | 168.7 | PASS |
| block_03_L2 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_03_L2 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_03_L2 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_03_L2 | pause_06_L0 | old G-1 | 480 | 60.0 | 389.1 | 90.9 | PASS |
| block_04_L3 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_04_L3 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_04_L3 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_04_L3 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_04_L3 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_04_L3 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_03_L2 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_04_L3 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_04_L3 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_04_L3 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_04_L3 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_04_L3 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_05_L4 | old G-1 | 480 | 60.0 | 418.5 | 61.5 | PASS |
| block_04_L3 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_04_L3 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_04_L3 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_04_L3 | pause_06_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_05_L4 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_05_L4 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_05_L4 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_05_L4 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_05_L4 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_03_L2 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_05_L4 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_05_L4 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_04_L3 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_05_L4 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_05_L4 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_05_L4 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_05_L4 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_05_L4 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_05_L4 | pause_06_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_01_L0 | trigger-2 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_01_L0 | trigger-2 | 480 | 60.0 | 184.7 | 295.3 | PASS |
| block_06_L0 | pause_01_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_01_L0 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_02_L1 | trigger-2 | 240 | 60.0 | 165.1 | 74.9 | PASS |
| block_06_L0 | pause_02_L1 | trigger-2 | 480 | 60.0 | 165.1 | 314.9 | PASS |
| block_06_L0 | pause_02_L1 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_02_L1 | old G-1 | 480 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_03_L2 | trigger-2 | 240 | 60.0 | 111.5 | 128.5 | PASS |
| block_06_L0 | pause_03_L2 | trigger-2 | 480 | 60.0 | 111.5 | 368.5 | PASS |
| block_06_L0 | pause_03_L2 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_03_L2 | old G-1 | 480 | 60.0 | 242.9 | 237.1 | PASS |
| block_06_L0 | pause_04_L3 | trigger-2 | 240 | 60.0 | 150.5 | 89.5 | PASS |
| block_06_L0 | pause_04_L3 | trigger-2 | 480 | 60.0 | 150.5 | 329.5 | PASS |
| block_06_L0 | pause_04_L3 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_04_L3 | old G-1 | 480 | 60.0 | 316.1 | 163.9 | PASS |
| block_06_L0 | pause_05_L4 | trigger-2 | 240 | 60.0 | 121.3 | 118.7 | PASS |
| block_06_L0 | pause_05_L4 | trigger-2 | 480 | 60.0 | 121.3 | 358.7 | PASS |
| block_06_L0 | pause_05_L4 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_05_L4 | old G-1 | 480 | 60.0 | 247.9 | 232.1 | PASS |
| block_06_L0 | pause_06_L0 | trigger-2 | 240 | 60.0 | 145.7 | 94.3 | PASS |
| block_06_L0 | pause_06_L0 | trigger-2 | 480 | 60.0 | 145.7 | 334.3 | PASS |
| block_06_L0 | pause_06_L0 | old G-1 | 240 | 60.0 | — | — | FAIL |
| block_06_L0 | pause_06_L0 | old G-1 | 480 | 60.0 | 306.3 | 173.7 | PASS |
MATRIX REPORT ONLY: trigger-2, 240 s: 30/36 restart; worst successful restart 165.1399999999942 s
MATRIX REPORT ONLY: trigger-2, 480 s: 36/36 restart; worst successful restart 184.73999999999307 s
MATRIX REPORT ONLY: old G-1, 240 s: 0/36 restart; worst successful restart None s
MATRIX REPORT ONLY: old G-1, 480 s: 13/36 restart; worst successful restart 418.5399999999819 s
PASS shifted-curve feasibility: REAL run_mode/switch and restart reserve; all six L2 480 s estimates restart with >=61 s left; remaining matrix report only

## Alignment limits and round-1 resolutions

F1 is resolved by the explicit owner-authorized rehearsal rule change, not by suppressing the failed check. F2 is softened: the earlier impossibility conclusion depends on the literal first-dump alignment; unrecorded early cooling was not credited, and later cooling was not extrapolated. Shifted cool-start curves can be pessimistic proxies. F3 remains a hardware limitation: real feasibility is unvalidated, while measured sessions intentionally retain high−2. F4 is corrected in the round-2 review patch: every a/ and b/ path is repo-relative; the original round-1 patch is preserved as history. F5's cooling ranges are corrected. F6 is addressed by this completed report and recorded review/identity artifacts below.

## Self-check results

20/20 commands exit 0. Complete stdout/stderr are retained in benchmark/campaign/checks/p23b. Warnings and intentional argparse refusal diagnostics remain in raw stderr.

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
| `benchmark/campaign/self_check_p23.py` | 0 |
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

Standalone `coder_mock_p23b`: exit 0; `NOT VALID — DRY RUN (all hardware mocked)`; mock rehearsal_pass=false. It exercises all three modes and the new adaptive thresholds without claiming hardware validity.

## Final P23 reviewer verdict — verbatim

**Verdict: APPROVED.** All round-1 and round-2 findings are resolved and the owner's M3 decision is implemented as stated. I found four new Low items; none blocks approval.

Final standalone verdict line, verbatim:

> APPROVED

## P23B review round 1 — final response verbatim, untruncated

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

### Byte-identity proof and reviewer command

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

copy: 2587 files; before/after full relative-path/byte-count/SHA-256 lists equal. Canonical inventory SHA-256 `9cc1e8c5c397e5c1b8a1ef3df4086fdd6c8dbcd4cfde7964d3b1f18057d7d5e4` (sorted JSON keys, compact separators).

live: 4289 files; before/after full relative-path/byte-count/SHA-256 lists equal. Canonical inventory SHA-256 `3bf88c2cbaed70388c6d3e48363271aac851699d029c440ad90159034ef1c67a` (sorted JSON keys, compact separators).

`campaign/review_identity_campaign_p23b_r1_before.json`: 1352645 bytes; SHA-256 `6dbc26519d4592dffec3421dfd159d031a66c1532075440e21fbbf9d2aca3d04`.

`campaign/review_identity_campaign_p23b_r1_after.json`: 1352645 bytes; SHA-256 `6dbc26519d4592dffec3421dfd159d031a66c1532075440e21fbbf9d2aca3d04`.

## P23B review round 2 — final response verbatim, untruncated

REVIEW_BEGIN
# CAMPAIGN P23B: Reviewer round 2 final review

**Reviewer:** Claude Code headless, claude-opus-5-5, effort high, Ponytail off. The review was offline and read-only. I made no edits, staging, commits or pushes. I did not touch hardware, root or the camera, load models, or do timed or live runs, and I launched no other reviewer. I did not run the repo's self-check harness or `run_mode`. Everything I computed ran in-process through `python3 -I -B -c` or Read/Grep/Glob, and none of it writes files. Recording before/after byte identity of the copy and the live tree is the Coder's job.

## Direct answer: is measured-session behaviour provably unchanged from 49bc13b?

**Yes, for the controller logic.** Two independent lines of evidence support this.

### 1. Static proof from the source (my own reading of `phase23.py`)

With `dry=False`, the only places the candidate differs from 49bc13b are these:

- **`:442` `low=high-2.`** This is the same float expression the base passed inline (`skin <= high-2.`). `switch()` (`:34-40`) now compares `skin <= low`, so it gets the identical value for every input.
- **`:458` `trigger_skin=... if dry and mode=='adaptive' else None`.** When `dry` is False, `session.skin()` is not called here. Then:
  - `:459` evaluates `session.now()` and then `session.skin()` in the same order as the base.
  - `:460` reads `skin_c` with a second `session.skin()`, exactly like the base. The number and order of reads in measured sessions are unchanged.
- **`:443-446`** only runs when `dry` is True. Measured pause records get no new keys.
- **`:491` `threshold_c=high if active else low`** gives the same value as the base's `high-2.`.
- **No other callers.** Repo-wide grep finds no other callers of `switch()` outside `self_check_p23.py`, and `low` shadows no existing name in `run_mode`. `FULL`, the `--t-hi` default and range, the 61 s reserve (`:426`, `:474`), readers, FIX4D and FIX3D are untouched.
- **Coverage** changed only in `coverage()` (`:564-565`), which runs only when `--dry-run` is set (`:692`). `require_rehearsal` (`:587-591`) reads the stored `rehearsal_pass` and does not recompute it.

### 2. Differential test (`self_check_p23.py:485-538`)

I checked this test independently:

- **Real base code.** It loads the base controller with `git show 49bc13b:benchmark/campaign/phase23.py` (the real committed base, not a copy).
- **Boundary grid.** It compares `switch()` across a 2×2×6×5 grid of boundary inputs.
- **Full mock sessions.** It runs base and candidate `MockSession`/`run_mode` for fixed and adaptive under `FULL` with `dry=False`. It then asserts that the result dicts **and every JSON file written** are equal, and that every switch threshold is 37 or 35.
- **The adaptive comparison is not vacuous.** The mock leaves idle at about 18 C and heats or cools at 0.04 C/s. A 2160 s run therefore goes through several 37 C high switches and 35 C restarts.
- **Overshoot test.** The mock gives a trigger of 40 with T_hi 37:
  - rehearsal restart threshold 38;
  - session restart threshold 35;
  - session pauses carry neither new key.
- **Coverage mutation.** Changing `low_threshold_c` flips `adaptive_low_restart`, which proves coverage now uses the per-pause value.
- **`ChangingTrigger` test.** It proves the rehearsal records the reading that actually triggered the switch (40, not the later 41). The base code's second `skin()` read would have recorded 41, so this test tells the two versions apart.
- **Log.** The `PASS P23B-R2` line is present in `checks/p23b/campaign_self_check_p23.stdout:109`.

**Limits of the proof:** it covers the controller logic and mock output, not live bytes. Code hashes in the top-level CLI output change by design, and live timing fields vary by nature. The Coder's report says so correctly.

## What I verified myself

- **Patch:** `review_full_diff_campaign_p23b_r2.patch` is 6222965 bytes, SHA-256 `281b9696…d5`, matching the request.
- **F4 fix:** all 93 `diff --git` headers use repo-relative `a/benchmark/…` and `b/benchmark/…` paths, with no `/termux-home` prefixes.
- **Frozen manifest:** all 92 entries in `frozen_manifest_campaign_p23b.json` match the copy (0 bad). The manifest's SHA-256 is `3e401195…334b`, matching the report. The only source entries are `phase23.py`, `self_check_p23.py` and `RUN.md`. The manifest has no `runs/` entries.
- **Raw archive:** all 40 entries in `runs/archive_sha256_p1_fix4d.json` match. The 17 `owner_session_p1_fix4d_*` entries in the base P23 manifest also match.
- **Nothing else moves:** I reverse-applied the patch's hunks in memory:

  | File | Hunks | Result |
  |---|---:|---|
  | `RUN.md` | 1 | byte-identical to the P23 manifest hash |
  | `phase23.py` | 6 | byte-identical to the P23 manifest hash |
  | `self_check_p23.py` | 4 | byte-identical to the P23 manifest hash |

  Every hunk's new side matched the copy exactly. `commit_campaign_p23.json` records 49bc13b as the pushed P23 commit, which included that manifest.
- **Check suite:** `checks/p23b/results.json` shows 20 of 20 commands with exit 0. The argparse "evidence exists" and "--mock requires --dry-run" lines in stderr are the deliberate refusal tests.
- **Mock run (`coder_mock_p23b_adaptive.json`):**
  - planned 480 s, `TIME LIMIT`, no issues, 3 camera restarts;
  - every pause has `switch_trigger_skin_c` equal to its active phase's `stop_decision.skin_c`, and `low_threshold_c` equal to trigger − 2;
  - every recorded restart skin is at or below its pause's low;
  - the last pause correctly stays OFF to the deadline because of the 61 s reserve;
  - `rehearsal_pass=false` because it is a mock.
- **Possible KeyError at `:444` (unreachable).** I checked whether `:444` (`stop_decision`) could raise a KeyError in the rehearsal. It can't:
  - Both the live `run_cycle` (`phase1.py:176-282`) and the mock cycle either raise on worker errors or run to `duration`.
  - `ops.origin` is set after `remaining` is computed, so a cycle without a controlled end finishes at or after the deadline. The loop then exits at `:489` before a pause starts.
  - Even if it were reached, it would fail closed, so the rehearsal would fail rather than pass.

## Six asserted L2 estimates and the 144-row matrix (checked independently, no tuning)

I wrote my own step-held replay. It uses the same first-dump alignment: switch at about 60.0 s, cooling starting at about 62.37 s, the 30 s minimum pause, and the 61 s reserve. It does not use the harness.

- **Asserted rows (L2 × six pauses, new rule, 480 s):** my restart times are 184.2, 164.7, 111.1, 150.0, 120.9 and 145.2 s. The Coder's are 184.7, 165.1, 111.5, 150.5, 121.3 and 145.7 s. The 0.4–0.5 s difference comes from the mock's 0.2 s polling. All six restart with at least 295 s left, so all pass.
- **Matrix counts, mine against the Coder's:**

  | Rule | 240 s | 480 s | Worst successful restart |
  |---|---|---|---|
  | trigger − 2 | 30/36 (both) | 36/36 (both) | ≈184 s |
  | old G − 1 | 0/36 (both) | 13/36 (both) | ≈418 s |

  The Coder's table has 144 unique rows, the asserted L2 rows contain no failures, and the counts match mine exactly.
- **Old rule, L2:** restart times of about 315.6, 403.3, 310.8 and 388.7 s reproduce R1's figures, and pauses 01 and 02 never cross.
- **Planner's 14/36 against 13/36:** reporting 13 rather than tuning to 14 is correct. Several old-rule cases cross before 480 s but leave less than 61 s, for example block_04_L3 × pause_03 at about 432.5 s. If the planner did not apply the reserve, that probably explains the 14th.

## R1 findings against the authorized revision

- **F1 — Resolved.** The owner authorized a rule change. The required assertion is now L2 × six at 480 s under trigger − 2, and it passes on the real controller. The old rule is reported only and fails honestly.
- **F2 — Resolved.** The report now says the impossibility conclusion depends on the literal first-dump alignment.
- **F3 — Mitigated for the rehearsal; still an acknowledged hardware limit.** The rehearsal restart no longer needs cooling below G when trigger ≥ G + 2, which is the case for every archived curve at the 60 s switch (rise of 2.50–4.13 C). Measured sessions keep 35 C by design. The proxy and offset caveats in the report remain correct.
- **F4 — Resolved.** The headers are repo-relative, as verified above.
- **F5 — Resolved.** `RUN.md:387` now says "1.97–2.97 C in 120 s and 2.75–4.29 C in 300 s".
- **F6 — Resolved for round 2, pending as expected.** The report now holds the round-1 review verbatim, the round-1 identity proof, and the P23 verdict quote. The round-2 verdict and the round-2 before/after identity have to be appended after this review.

## New findings (none block approval)

- **N1 — Info: `RUN.md` wording differs from the owner's dictated text, and is more accurate.** The owner wrote "never depends on cooling below the gate reading". `RUN.md:386` says "With the archived L2 overshoot, restart no longer depends…". The qualifier is correct: low = trigger − 2 ≥ G − 1, and it is below G whenever G + 1 ≤ trigger < G + 2. The owner should know the sentence is deliberately qualified.
- **N2 — Info: rehearsal thresholds ratchet upward.** Each rehearsal restart happens at trigger − 2. The next active phase still runs at least 60 s, so each trigger tends to be higher than the last. The mock shows lows of 36.88, 37.26, 37.64 and 38.03 C. On hardware, L2's rise of about 3 C per 60 s makes this roughly +1 C per cycle over about 2–3 cycles in 480 s. Emergency/SEVERE stops still apply, and endurance (180 s continuous L2) heats more. So the rehearsal proves the switching and restart mechanism, not the measured sessions' fixed 35 C band. This follows from the owner's rule; it is not a defect.
- **N3 — Info: the asserted estimate barely uses the heating curve.** In the step-held replay, the skin at the switch and the skin at camera stop fall in the same dump. The needed drop is therefore exactly 2.0 C, and the rows are identical across all six heating blocks. On hardware, skin keeps rising through the roughly 2 s stop and the 4.87 s dump lag, adding perhaps 0.1–0.3 C to the drop needed. The worst estimated restart (about 185 s at 480 s) leaves wide margin.
- **N4 — Info: the new test depends on `git` and the working directory.** `self_check_p23.py:487` calls `git show` without `cwd`, like the existing `:424/:427` in the same file. Run from outside the repo, it fails closed rather than passing silently. This is consistent with the file's existing practice.

## Limits

- I did not run the repo harness or `run_mode`. My replay is independent arithmetic on the archived dumps.
- The isolated copy has no git, so I checked against the base through the committed P23 manifest. The differential test's own `git show` was run by the Coder.
- I did not record byte identity of the copy or the live tree.

**Verdict:** under the owner's P23B-R2 revision, measured-session controller behaviour is unchanged from 49bc13b. The static diff shows that, and a byte-level differential test against the real base source confirms it. The rehearsal rule, its per-pause recording and the per-pause coverage are implemented as specified. The six asserted L2 estimates and the 144-row report-only matrix reproduce independently, with no sign of tuning. Every R1 finding is resolved or properly acknowledged. Approval is a review finding only, not permission to commit. The round-2 identity proof and this verdict still have to be added to the report before the owner's "yes commit".

APPROVED
REVIEW_END

### Byte-identity proof and reviewer command

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
  "command": "cd /termux-home/review_campaign_p23b_r2 && PONYTAIL_DEFAULT_MODE=off PYTHONDONTWRITEBYTECODE=1 claude -p 'Your complete review instructions are on standard input. Ponytail off. End with APPROVED or CHANGES REQUESTED.' --model claude-opus-5-5 --effort high --permission-mode dontAsk --setting-sources '' --strict-mcp-config --mcp-config '{\"mcpServers\":{}}' --tools Bash,Read,Grep,Glob --allowedTools 'Bash(python3 -I -B -c *)' 'Bash(cat *)' 'Bash(grep *)' 'Bash(ls *)' 'Bash(sha256sum *)' Read Grep Glob --disallowedTools Edit Write NotebookEdit WebFetch WebSearch --output-format json < /termux-home/storage/downloads/campaign/review_request_campaign_p23b_r2.txt\n"
}
```

copy: 2587 files; before/after full relative-path/byte-count/SHA-256 lists equal. Canonical inventory SHA-256 `425f20c3ed73cdc66f7e749d3e3fcb27487d9b4c8e86e81c2f7ffafe35702f2b` (sorted JSON keys, compact separators).

live: 4289 files; before/after full relative-path/byte-count/SHA-256 lists equal. Canonical inventory SHA-256 `517b4e4e55d7f62ffee878400086f50ba60435871bab20607be37292136909af` (sorted JSON keys, compact separators).

`campaign/review_identity_campaign_p23b_r2_before.json`: 1352643 bytes; SHA-256 `15eae9815e809d75c243049b551518bbf0d10092cbb3b8d62cdb80d7a526df32`.

`campaign/review_identity_campaign_p23b_r2_after.json`: 1352643 bytes; SHA-256 `15eae9815e809d75c243049b551518bbf0d10092cbb3b8d62cdb80d7a526df32`.

## Frozen manifest

All 92 entries verified after checks/review. Manifest excludes itself; report/reviewer evidence remain external. Manifest SHA-256 `3e401195f3f06372f3a293bdd0cb1648a8cab355286f1a81d20dee09adcf334b`.

## Round-2 informational findings acknowledged

N1: the RUN.md sentence is deliberately qualified to archived L2 overshoot. Trigger−2 can be below G if the trigger is between G+1 and G+2, so a universal promise would be false.

N2: rehearsal lows can ratchet upward over repeated cycles (roughly +1 C per cycle for the archived L2 rise). This authorized rule exercises switching/restart mechanics and does not validate the measured sessions' fixed 35 C band. All existing stop rules remain unchanged.

N3: on hardware, heating during stop and dump lag can add cooling demand beyond the step-held estimate; these estimates remain NOT VALID as hardware results. N4: the git-based differential test must run from inside the repository, consistent with existing checks.

## Proposed commit message

`campaign: make rehearsal adaptive restart follow switch skin (P23B)`

STOPPED for owner's “yes commit”. No commit, staging, push or DOC DIFF performed. Any source changes after approval require new review.

Skipped: hardware/root/live/timed validation and unrequested controller changes. Risk: rehearsal thresholds can rise across cycles; shifted-curve estimates and mocks do not establish a successful owner hardware rehearsal.


## Commit receipt

Owner authorized “yes commit”. Created `d5d1e87c66db86848cc93495046cde8f6e3d75f6` with the proposed message and exact 93 reviewed files. Parent `49bc13bc6a3f1eb1fca242d344a4cdb1c9c77541`. All committed bytes match the round-2 isolated candidate and frozen manifest; all 92 manifest entries verified. Working tree clean. No push; separate “yes push” remains required. No code/evidence/doc changes after review. Full receipt: `campaign/commit_campaign_p23b.json`.

Staged whitespace check flagged only raw check artifacts; their reviewed bytes were preserved. Diagnostics:

```text
benchmark/campaign/checks/p23b/duty_cycle_test_duty_cycle.stdout:13: trailing whitespace.
+[CONT] selector drain 122.000s; KeyboardInterrupt: 
```


## Push receipt

Owner authorized “yes push”. Pushed main to origin; git ls-remote verifies refs/heads/main at `d5d1e87c66db86848cc93495046cde8f6e3d75f6`. Working tree clean. Earlier no-push statements describe the pre-push record. No live validation or DOC DIFF. Full receipt: `campaign/push_campaign_p23b.json`.
