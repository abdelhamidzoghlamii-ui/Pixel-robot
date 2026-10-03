# power_map — run index

One row per run/console attempt. UTC console launch time and run.json start time are separate; failed starts retain metadata without measurement data.

| Folder / console | Date UTC | Set / blocks | Label |
|---|---|---|---|
| `runs/run_20261001T213425Z_R1_smoke/`; `runs/oneshot_console_20261001T212918Z.log` | 2026-10-01 21:29:18Z (console); 2026-10-01T21:34:26+00:00 (run.json) | R1_r2_c5, R1_r2_c10, R1_r2_off | run_20261001T213425Z_R1_smoke, run_20261001T215543Z_R3_smoke: SMOKE, not a measurement; pinning verified. |
| `runs/run_20261001T215543Z_R3_smoke/`; `runs/oneshot_console_20261001T215036Z.log` | 2026-10-01 21:50:36Z (console); 2026-10-01T21:55:44+00:00 (run.json) | R3_default, R3_mid, R3_little | run_20261001T213425Z_R1_smoke, run_20261001T215543Z_R3_smoke: SMOKE, not a measurement; pinning verified. |
| `runs/run_20261001T222645Z_R1/`; `runs/oneshot_console_20261001T222137Z.log` | 2026-10-01 22:21:37Z (console); 2026-10-01T22:26:46+00:00 (run.json) | R1_r2_c5, R1_r2_c10, R1_r2_off | run_20261001T222645Z_R1: VALID. Blocks 2-3 started warm (heat comparisons approximate); 640 cadence not a lever. |
| `runs/run_20261001T231621Z_R2/`; `runs/oneshot_console_20261001T231114Z.log` | 2026-10-01 23:11:14Z (console); 2026-10-01T23:16:22+00:00 (run.json) | R2_r1_c5, R2_r1_c10, R2_r1_off | run_20261001T231621Z_R2: VALID. 1 frame/s about halves power vs 2 frames/s. |
| `runs/run_20261002T083606Z_R3/`; `runs/oneshot_console_20261002T083058Z.log` | 2026-10-02 08:30:58Z (console); 2026-10-02T08:36:07+00:00 (run.json) | R3_default, R3_mid, R3_little | run_20261002T083606Z_R3: VALID. MID chosen; LITTLE out (149/180 frames). Default block 2.83 W vs 2.10 W in R2: run-to-run noise ~0.7 W, first-block bias suspected. |
| `runs/run_20261002T090821Z_CONFIRM/`; `runs/oneshot_console_20261002T090314Z.log` | 2026-10-02 09:03:14Z (console); 2026-10-02T09:08:22+00:00 (run.json) | CONFIRM | run_20261002T090821Z_CONFIRM: VALID. 20 min; policy4/6 capping starts ~min 17 at skin ~37 C; skin still rising. |

## Fixed labels (copied exactly)

- run_20261001T213425Z_R1_smoke, run_20261001T215543Z_R3_smoke: SMOKE, not a measurement; pinning verified.
- run_20261001T222645Z_R1: VALID. Blocks 2-3 started warm (heat comparisons approximate); 640 cadence not a lever.
- run_20261001T231621Z_R2: VALID. 1 frame/s about halves power vs 2 frames/s.
- run_20261002T083606Z_R3: VALID. MID chosen; LITTLE out (149/180 frames). Default block 2.83 W vs 2.10 W in R2: run-to-run noise ~0.7 W, first-block bias suspected.
- run_20261002T090821Z_CONFIRM: VALID. 20 min; policy4/6 capping starts ~min 17 at skin ~37 C; skin still rising.
- Open across power_map: absolute watts contradict #122 (unexplained).

## Console matches

- oneshot_console_20261001T212918Z.log: explicitly names run_20261001T213425Z_R1_smoke; run timestamp follows console launch and 5 min idle.
- oneshot_console_20261001T215036Z.log: explicitly names run_20261001T215543Z_R3_smoke; run timestamp follows console launch and 5 min idle.
- oneshot_console_20261001T222137Z.log: explicitly names run_20261001T222645Z_R1; run timestamp follows console launch and 5 min idle.
- oneshot_console_20261001T231114Z.log: explicitly names run_20261001T231621Z_R2; run timestamp follows console launch and 5 min idle.
- oneshot_console_20261002T083058Z.log: explicitly names run_20261002T083606Z_R3; run timestamp follows console launch and 5 min idle.
- oneshot_console_20261002T090314Z.log: explicitly names run_20261002T090821Z_CONFIRM; run timestamp follows console launch and 5 min idle.

No ambiguous match was filled by guessing. `thermal.log` spans sessions and is not a per-run console.
