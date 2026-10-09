# Campaign owner evidence

Archived unchanged from owner Downloads, SHA-256 verified against each source. Includes Git-ignored `_llama-server.log` files, explicitly listed in the frozen manifest and commit proposal (force-add only after the human gate). Original inventory: [archive_sha256.json](runs/archive_sha256.json); session A/fix2 inventory: [archive_sha256_p1_fix2.json](runs/archive_sha256_p1_fix2.json); fix3 inventory: [archive_sha256_p1_fix3.json](runs/archive_sha256_p1_fix3.json); fix3c inventory: [archive_sha256_p1_fix3c.json](runs/archive_sha256_p1_fix3c.json); fix3d inventory: [archive_sha256_p1_fix3d.json](runs/archive_sha256_p1_fix3d.json).

Read-only audit of the FIX3D commit 5629699 (findings H1–H5, M1–M9, L1–L8), archived unchanged from owner Downloads as [AUDIT1_REPORT.md](AUDIT1_REPORT.md), SHA-256 `d30d2bcb82d0cd469a1e4656b0e1399284c323c0a1f8c3654e0e88bfa4e8c7be` (35,215 bytes); fixed in CAMPAIGN_P1_FIX4 ([RUN.md](RUN.md)). No new owner run is archived by FIX4.

| Stem | Fixed label | Evidence |
|---|---|---|
| [owner_lag_p1](runs/owner_lag_p1.json) | POWER LAG PROBE — method check, not a benchmark | Complete, 11.5 Hz. current_now rise 50 % 0.25 s, 90 % 0.55 s, fall 50 % 0.29 s; current_avg slow (rise 50 % 3.0 s, 90 % 6.5 s): current_now is the valid power source. |
| [owner_preflight_p1](runs/owner_preflight_p1.json) | PREFLIGHT ONLY — NO TIMING | Passed; all layouts. |
| [owner_preflight_p1b](runs/owner_preflight_p1b.json) | PREFLIGHT ONLY — NO TIMING | Passed; all layouts. |
| [owner_preflight_p1c](runs/owner_preflight_p1c.json) | PREFLIGHT ONLY — NO TIMING | Passed; --blocks L0,L1,L2. |
| [owner_session_p1a](runs/owner_session_p1a.json) | NOT VALID — SESSION INCOMPLETE | Setup refusal "root shell/monitor mask empty or includes inference cores" after L0 check; only L0 recorded. No idle, no block. .stdout/.stderr were overwritten by an immediate rerun that refused with "evidence exists". |
| [owner_session_p1a2](runs/owner_session_p1a2.json) | NOT VALID — SESSION INCOMPLETE | Same refusal reproduced with full traceback; only L0 recorded. No idle, no block. |
| [owner_preflight_p1_fix](runs/owner_preflight_p1_fix.json) | PREFLIGHT ONLY — NO TIMING. Passed. | Setup complete and cleanup successful. Owner resolved F6 on 2026-10-05: keep the evidence-backed passed label; the charger attempt was the second, blocked invocation under the same stem, with no other stem. Supplied files preserve the passed preflight; no refusal reason is present in stderr. |
| [owner_preflight_p1_fix2](runs/owner_preflight_p1_fix2.json) | PREFLIGHT ONLY — NO TIMING. Passed. | Setup complete and cleanup successful; all layouts. |
| [owner_preflight_p1_fix3](runs/owner_preflight_p1_fix3.json) | PREFLIGHT ONLY — NO TIMING. Passed. | Setup complete and cleanup successful; all layouts; battery 90 %. Server log/stdout/stderr preserved. |
| [owner_session_p1_fix2](runs/owner_session_p1_fix2.json) | NOT VALID — SESSION INCOMPLETE. Stopped after the first 180 s idle sub-phase (camera OFF): "pause memory/battery read failed", caused by PSS missing for the RobotCam app while the camera was off (app not running); no warm-up, no block. | Diagnostic only: during that camera-OFF idle with llama-server loaded, policy0/policy4/policy6 caps were 0 %. All 37 memory samples had root_rc 0, Discharging and MemAvailable; only `robotcam_app` was missing. Fixed in CAMPAIGN_P1_FIX3. Server log/stdout/stderr preserved. |
| [owner_rehearsal_p1_fix3](runs/owner_rehearsal_p1_fix3.json) | NOT VALID — REHEARSAL INCOMPLETE. Setup, 3 idle sub-phases and the warm-up ran; the warm-up process guard flagged the runner's own root diagnostics shell as "agents/robot/other runners resident" (failure_kind shared), so no pause or block ran. | Diagnostic only: warm-up YOLO 320 20/20, 640 4/4, camera 20/20, selector 1 call. The flagged pid 1312 was the runner's own `su -c` diagnostics client; its text `read_node` matched the guard alternative `node`. No block files were produced. Warm-up: [owner_rehearsal_p1_fix3_warmup_L0](runs/owner_rehearsal_p1_fix3_warmup_L0.json). Fixed in CAMPAIGN_P1_FIX3C. Server log/stdout/stderr preserved. |
| [owner_rehearsal_p1_fix3c](runs/owner_rehearsal_p1_fix3c.json) | REHEARSAL COMPLETE — NOT A PASS. Every phase, pause and all 6 blocks ran with no errors; 0 live M2 calls because the phone lay on its back (camera saw nothing); 16 fallback M2 calls. NOT VALID as results. | `rehearsal_coverage`: setup, 3 idle, warm-up, 6 pauses, 6 blocks, errors []; `rehearsal_pass` false only because live M2 calls were 0 in L1–L4. Fallback M2 calls L1 4, L2 2, L3 5, L4 5 (16; the task text said 21, the bytes say 16, owner chose the bytes); 5 selector calls carried a fallback-scene context. Idle policy0 caps 0 % camera OFF, 86.1 % (lowest 1401 MHz) camera ON. Warm-up: [owner_rehearsal_p1_fix3c_warmup_L0](runs/owner_rehearsal_p1_fix3c_warmup_L0.json); blocks: [01_L0](runs/owner_rehearsal_p1_fix3c_block_01_L0.json), [02_L1](runs/owner_rehearsal_p1_fix3c_block_02_L1.json), [03_L2](runs/owner_rehearsal_p1_fix3c_block_03_L2.json), [04_L3](runs/owner_rehearsal_p1_fix3c_block_04_L3.json), [05_L4](runs/owner_rehearsal_p1_fix3c_block_05_L4.json), [06_L0](runs/owner_rehearsal_p1_fix3c_block_06_L0.json). Server log/stdout/stderr preserved. |
| [owner_session_p1_fix3c](runs/owner_session_p1_fix3c.json) | NOT VALID — SESSION INCOMPLETE. The owner locked the screen (emergency); 'pause sampler stopped' in a rest phase; no block ran. | Phase from the bytes: first idle sub-phase `IDLE` (camera OFF), stopped after 5.26 s. Recorded cause: `monitor_errors` ['thermal read failed: skin None, status None'] from the thermal worker, 0 thermal rows, then `RuntimeError: pause sampler stopped` (phase1.py:576). The bytes do not show the screen lock; the signature is the same as fix3c2. No warm-up, no pause, no block. Server log/stdout/stderr preserved. |
| [owner_session_p1_fix3c2](runs/owner_session_p1_fix3c2.json) | NOT VALID — SESSION INCOMPLETE. Stopped at the end of idle sub-phase "camera ON, no inference" (about 54 s): one checked thermal read returned no skin and no status (rc 0); all 65 recorded thermal rows were valid; no warm-up, no block. Diagnostic only: policy0 (LITTLE) caps 0 % in both camera-OFF sub-phases and 90 % at 1401 MHz in the camera-ON sub-phase, skin about 29 C; policy4/policy6 0 %. | phase1.py:744 rest → :571 `rt.dump_check` → runtime.py:192 `RuntimeError('thermal read failed: skin None, status None')`. Thermal rows 39 + 14 + 12, all rc 0, status 0; skin 26.9–27.3, 27.4–27.7, 28.1–29.4 C. Camera-ON policy0 capped 54.1 s (90.2 %), lowest 1401 MHz. The worker read ran 54.63–54.81 s, and the failing main-thread read ended between the last 1 s fast row (54.12 s) and the stop, so it very likely overlapped the worker read on the shared `coresidency_thermalservice.txt` file (the timing is not proven). Fixed in CAMPAIGN_P1_FIX3D (retry plus per-thread file). Server log/stdout/stderr preserved. |
| [owner_session_p1a_fix](runs/owner_session_p1a_fix.json) | SESSION COMPLETE; L0 VALID; L1 and L2 NOT VALID. | Full session completed; L1/L2 invalid for warm starts and absent M2 inference. Server log/stdout/stderr preserved. |
| [block_01_L0](runs/owner_session_p1a_fix_block_01_L0.json) | VALID. | Full cycle, no M2. YOLO 320 180/180, 150/197 ms; 640 36/36, 556/814 ms; camera 180/180, 0 late; selector 9 calls, 1771/2060 ms, 0 misses; power 3.88 W time-weighted (0.37 s sampler; not comparable with #128 method); skin 21.5 → 27.9 C; MemAvailable min 2522 MiB; policy0 capped 99.4% at 1401 MHz (max 1803 MHz; unexplained), policy4/policy6 0%. |
| [block_02_L1](runs/owner_session_p1a_fix_block_02_L1.json) | NOT VALID — WARM START (gate 482 s) and NO M2 LOAD (36/36 slots < 2 boxes). | M2 calls zero, selector median 1935 ms. |
| [block_03_L2](runs/owner_session_p1a_fix_block_03_L2.json) | NOT VALID — WARM START (gate 490 s) and NO M2 LOAD (9/9 slots < 2 boxes). | M2 calls zero, selector median 1817 ms. |

All three blocks show policy0 capped about 99.4% at 1401 MHz, policy4/policy6
0%. Camera causality is unknown. PSS starts around 3.89 GB in L0, declines
through L0/L1 and stays around 2.93 GB late in L1 and throughout L2; observation
only. Source-derived details: [owner_findings.json](checks/p1_fix2/owner_findings.json).
Live 640 box counts were below two for every L1/L2 M2 slot; camera orientation
cannot be established from these JSON files alone.

## FIX4D owner archive (CAMPAIGN_P23)

All 40 files copied unchanged and SHA-256 verified: [inventory](runs/archive_sha256_p1_fix4d.json).

| Stem | Fixed label | Verified evidence |
|---|---|---|
| [owner_rehearsal_p1_fix4d](runs/owner_rehearsal_p1_fix4d.json) | REHEARSAL PASS — NOT VALID as results. | All phases, 6 pauses and 6 blocks ran; live M2 in L1-L4; 0 read re-reads, 0 thermal retries. |
| [owner_session_p1_fix4d](runs/owner_session_p1_fix4d.json) | SESSION COMPLETE | About 96 min (last block END +5767.3 s), 0 errors, 0 re-reads, Android status 0, 0 LMK kills. Warm-up NOT A RESULT. L0 VALID, L1 VALID, L3 VALID; L2, L4 and final L0 NOT COMPARABLE — START TEMP (+1.92, +2.02, +1.86 C over T_ref 29.54 C). D2 checked at pause end, before block setup; comparability read followed setup. Owner chose L2 for Phases 2-3. |

180 s blocks; power is time-weighted current_now, 0.37 s sampling; M2 is inference median/P95:

| Block | Fixed runner label | YOLO-320 done/due | M2 ms | Selector median ms | W |
|---|---|---|---|---|---|
| First L0 | VALID | 180/180 | — | 1733 | 4.10 |
| L1 | VALID | 180/180 | 4919/6451 | 2819 | 4.99 |
| L2 | NOT COMPARABLE — START TEMP | 180/180 | 1176 | 2129 | 4.72 |
| L3 | VALID | 109/180 (71 skipped by design) | 1313 | 2956 | 5.10 |
| L4 | NOT COMPARABLE — START TEMP | 180/180 | 871/2001 | 3686 | 5.41 |
| Final L0 | NOT COMPARABLE — START TEMP | 180/180 | — | 1775 | 4.04 |

L4 policy6 capped 43.9 %. Camera-ON idle policy0 capped 98.5 % vs 0 % camera OFF.
Original JSONs, phase JSONs, stdout, stderr and llama-server logs remain byte-identical to owner files.
