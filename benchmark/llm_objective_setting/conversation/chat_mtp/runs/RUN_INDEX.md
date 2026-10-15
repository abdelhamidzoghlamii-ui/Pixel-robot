# chat_mtp — run index

One row per run with its launcher console. Run start is `started` in the run's `run.json`; console start is the UTC
time in the console file name. Labels: fixed by Local AI (2026-10-15), copied verbatim. `thermal.log` (launcher
CPU-zone logger, 2026-10-14 22:06:56Z – 22:53:04Z) spans both sessions and belongs to neither row alone. Hashes:
[ARTIFACTS.md](ARTIFACTS.md).

| Folder / console | Date UTC | Set / blocks | Label |
|---|---|---|---|
| `chat_mtp_20261014T221203Z_smoke/`; `oneshot_console_20261014T220655Z.log` | console 2026-10-14 22:06:55Z; run 22:12:08Z | `--smoke`: prompt A5 only; blocks 0 OFF, 1 ON | chat_mtp_20261014T221203Z_smoke: SMOKE, not a measurement. The runner works end to end: 1/1 prompt per config, ON draft acceptance 50.2 %, server on cores 4-7 and stopped after each block. Its speed, memory and power figures are NOT VALID as results; block 1 (ON) was a WARM START. |
| `chat_mtp_20261014T224541Z/`; `oneshot_console_20261014T224033Z.log` | console 2026-10-14 22:40:33Z; run 22:45:45Z | full: 13 prompts per config; blocks 0 OFF, 1 ON | chat_mtp_20261014T224541Z: VALID, single run at robot-chat settings (ctx 4096, temperature 0.7, n_predict 512, cores 4-7, dotprod b1609 build). 13/13 prompts per config, one block each (OFF, then ON; the planned OFF/ON alternation did not occur because each config finished in its first block; ON started 1.5 °C warmer, 25.63 vs 24.15 °C); no WARM START; Android thermal status 0. Generation 13.37 -> 19.18 tok/s (+43 %), draft acceptance 39.0 %, prompt 0.35 / 0.34 s, load 4.3 -> 6.4 s. Energy per generated token 0.73 -> 0.58 J: INDICATIVE ONLY (5 s power sampling, one block per config). Memory: median llama-server PSS from 10 s on was 3,427,284 kB OFF vs 4,389,009 kB ON (+939 MiB) in this run, but OFF VmHWM was 4,177,884 kB in the smoke run vs 3,422,148 kB here while ON matched (4,394,768 vs 4,398,784 kB): the drafter's memory cost is NOT ESTABLISHED. MemAvailable is not a usable comparison (swap use changed within blocks). Reply quality NOT EVALUATED (temperature 0.7: replies differ by design). |
