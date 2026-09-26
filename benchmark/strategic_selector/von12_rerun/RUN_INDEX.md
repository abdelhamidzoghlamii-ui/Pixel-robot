# Von 1.2 rerun — run index

Rooted Pixel 7, Termux + Debian proot, cores 4–7, `von-sdk` 1.2.0 (reports `von-1.2.0`), torch
2.14.0+cpu, separate venv and HF cache. Thermal from a native-Termux root logger (`raw/thermal.log`).

| Run | When (UTC) | Output | Status | Result |
|---|---|---|---|---|
| probe | 2026-09-24 ~18:05 | `raw/probe.*`, `raw/von12-probe.json` | complete | loads `von-1.2.0` offline, no `UNEXPECTED` head-key warning; first decision 7387 ms |
| determinism check | 2026-09-24 | `raw/determinism_*`, `raw/negative_control/` | complete | 132 archived 1.1 rows regenerated, 824 fields, 0 mismatches; control detected |
| Block A | 2026-09-24T18:41Z | `raw/runs/2026-09-24T184107Z-von-68e0c438/` | complete | selected `filtered_json`; held-out P 3/11, A 4/11, median 3067 ms, flips 0/11; peak RSS 3097 MiB; z9 peak 106 °C |
| Block B | 2026-09-24 ~18:55 | `raw/von12-heldout-extra-results.json` | complete | `filtered_text` P 2/11 A 5/11, `two_stage_text` P 2/11 A 2/11, flips 0/11 each; z9 peak 104 °C |
