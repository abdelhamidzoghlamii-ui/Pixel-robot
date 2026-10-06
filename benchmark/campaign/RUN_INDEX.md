# Campaign owner evidence

Archived unchanged from owner Downloads, SHA-256 verified against each source. Includes Git-ignored `_llama-server.log` files, explicitly listed in the frozen manifest and commit proposal (force-add only after the human gate). Original inventory: [archive_sha256.json](runs/archive_sha256.json); session A/fix2 inventory: [archive_sha256_p1_fix2.json](runs/archive_sha256_p1_fix2.json); fix3 inventory: [archive_sha256_p1_fix3.json](runs/archive_sha256_p1_fix3.json); fix3c inventory: [archive_sha256_p1_fix3c.json](runs/archive_sha256_p1_fix3c.json).

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
