# SPEED1 fixed owner archive

Bytes copied unchanged; SHA-256 checked against Downloads originals. M2 is a benchmark candidate, not deployed.

| File | Bytes | SHA-256 | Fixed label |
|---|---:|---|---|
| [owner_preflight_speed1_fix.json](owner_preflight_speed1_fix.json) | 4385 | `9918f724cd6906e9a047990b8afd60eddeb239d3ad3deed0a6db8b38c10cbf89` | PREFLIGHT ONLY — NO TIMING. Passed: first live root-mask readback; root/su masks excluded each measured cluster; engines pinned MID 4-5, BIG 6-7, LITTLE 0-3. |
| [owner_preflight_speed1_fix.stderr](owner_preflight_speed1_fix.stderr) | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` | PREFLIGHT ONLY — NO TIMING. Passed: first live root-mask readback; root/su masks excluded each measured cluster; engines pinned MID 4-5, BIG 6-7, LITTLE 0-3. |
| [owner_preflight_speed1_fix.stdout](owner_preflight_speed1_fix.stdout) | 1267 | `aac635ef9ea2b2bf6adc7fe705c600076a1793b2264cc2850a1d8a0d1c57037d` | PREFLIGHT ONLY — NO TIMING. Passed: first live root-mask readback; root/su masks excluded each measured cluster; engines pinned MID 4-5, BIG 6-7, LITTLE 0-3. |
| [owner_session_speed1_fix.json](owner_session_speed1_fix.json) | 2358903 | `104ff9bd771ad43f36226598ace98baed3a17165507ab5c836442eac4f8f6e2c` | SESSION COMPLETE; 3 VALID blocks, 1 warm start. |
| [owner_session_speed1_fix.stderr](owner_session_speed1_fix.stderr) | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` | SESSION COMPLETE; 3 VALID blocks, 1 warm start. |
| [owner_session_speed1_fix.stdout](owner_session_speed1_fix.stdout) | 13488 | `b7f583d4e584f01080e3af2a78a02a26fb0180e4da921b6d6cc057bb43c88764` | SESSION COMPLETE; 3 VALID blocks, 1 warm start. |
| [owner_session_speed1_fix_block_01_fp32_MID.json](owner_session_speed1_fix_block_01_fp32_MID.json) | 479134 | `ed887d23389f0213bae472ecc4ee5900de7ca03a054633f32037defb25a403aa` | VALID. 36/36 calls, 1116/1126 ms median/P95, no caps. Watts indicative only (1 s sampler phase-locked to the 5 s cadence; current_now lag suspected). |
| [owner_session_speed1_fix_block_02_fp32_BIG.json](owner_session_speed1_fix_block_02_fp32_BIG.json) | 477789 | `6e18797ec6e0acdaeafe3abf7a7e92ee3498979ddc0176fd2bf8f59d2c154a4f` | VALID timing, 36/36 calls, 646/671 ms, no caps. Watts NOT VALID (0 power samples inside calls; phase-locked sampler). |
| [owner_session_speed1_fix_block_03_fp32_LITTLE.json](owner_session_speed1_fix_block_03_fp32_LITTLE.json) | 533291 | `c0234bb04896f450b01b10c28bbb0323a760d61386eb46cf52ff5e562b6ce021` | VALID. 36/36 calls, 3130/3208 ms, no caps. Watts indicative only (same sampler limits). |
| [owner_session_speed1_fix_block_04_fp32_MID.json](owner_session_speed1_fix_block_04_fp32_MID.json) | 478102 | `0f6588b8e1d04dac2569a4dab20430caa55d3f0fa6895ac46fcc375a28aa59d9` | NOT VALID — WARM START (skin gate timed out after 482 s). 1118/1131 ms is consistent with block 1 but not valid. |
