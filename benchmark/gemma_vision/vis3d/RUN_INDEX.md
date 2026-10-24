# VIS-3D — run index

One row per launch with its console. Run start is `started` in the run's `run.json`; console start is the UTC time in
the console file name. Labels: archival task VIS-3C/3D-ARCHIVE (2026-10-23), from the runner's own per-run labels and
DECISIONS #143; no measurement was rewritten. `runs/thermal.log` (launcher CPU-zone logger, appended by every
launch, 2026-10-23 15:07:31Z – 16:35:30Z) spans all six consoles and belongs to no single row; its zone readings are
CPU-core readings, not a heat state. Hashes: [ARTIFACTS.md](ARTIFACTS.md).

| Folder / console | Date UTC | Test | Label |
|---|---|---|---|
| — ; `runs/oneshot_console_20261023T150730Z.log` | console 15:07:30Z | not recorded | REFUSED: `not started: battery 79 % < 80 %` (exit 1). No server, no run folder. NOT A MEASUREMENT. |
| `runs/20261023T151754Z_A/`; `runs/oneshot_console_20261023T151746Z.log` | console 15:17:46Z; run 15:17:59Z | A | VALID (runner label). 13/13; 13.22 tok/s; start skin 25.30 °C, status 0, battery 82 %. |
| `runs/20261023T152217Z_B/` (B1); `runs/oneshot_console_20261023T152209Z.log` | console 15:22:09Z; run 15:22:21Z | B | VALID (runner label). 13/13; 13.53 tok/s; start skin 28.62 °C, status 0, battery 81 %. Kept; not used for the official comparison. |
| `runs/20261023T161842Z_B/` (B2); `runs/oneshot_console_20261023T161834Z.log` | console 16:18:34Z; run 16:18:47Z | B | VALID (runner label). 13/13; 13.46 tok/s; start skin 24.23 °C, status 0, battery 99 %. **Official B for the A/B/C comparison.** |
| `runs/20261023T162349Z_C/`; `runs/oneshot_console_20261023T162342Z.log` | console 16:23:42Z; run 16:23:54Z | C | VALID (runner label). 13/13; 18.11 tok/s, acceptance 35.2 %, 341 server-counted verification steps; start skin 27.95 °C, status 0, battery 98 %. |
| `runs/20261023T163445Z_D/`; `runs/oneshot_console_20261023T163437Z.log` | console 16:34:37Z; run 16:34:50Z | D | VALID (runner label). 8/8 images answered and complete; images 2–8 median wall 4.42 s; reply quality NOT SCORED (Coder-drafted references, no rubric). |

Comparisons (each ratio one observation, not a cause):

| Comparison | Ratio | Conditions | Original status of the label |
|---|---:|---|---|
| B2/A | 1.018 | UNMATCHED: start skin 25.30 vs 24.23 °C; battery 82 vs 99 % | combined console report, owner-reported, not saved; REGENERATED |
| C/B2 | 1.346 | UNMATCHED: start skin 24.23 vs 27.95 °C | as above |
| A/OFF (VIS-3C) | 0.990 | UNMATCHED: battery 95 vs 82 % | saved in `runs/20261023T151754Z_A/report.txt` |
| C/ON (VIS-3C) | 0.976 | UNMATCHED: start skin 25.51 vs 27.95 °C | saved in `runs/20261023T162349Z_C/report.txt` |
| B1/A, C/B1 (supplementary) | 1.023, 1.339 | UNMATCHED: start skin 25.30 vs 28.62 °C (B1/A) | REGENERATED only |

REGENERATED reports: `reports/REGENERATED_20261023T190341Z_combined_A_B2_C.txt` (official) and
`reports/REGENERATED_20261023T190341Z_combined_A_B1_C.txt`, generated 2026-10-23T19:03:41Z by `vis3d.py --report` on
the unchanged phone folders (report-only, no inference, no root). They are not original artifacts of the runs. A
D-only report regenerated in the same way was byte-identical to `runs/20261023T163445Z_D/report.txt` and is not kept.

## Review records (`reports/`)

Candidate: the five source files, frozen SHA-256 in `reports/frozen_candidate.sha256` (= the `run.json` code hashes).
AGY reviewer `gemini-3.8-flash-high`, Ponytail off; requests `review_request.txt` (round 1) and
`review_request_r2.txt` (round 2); settings overlay and restore hashes in `settings_*.sha256`, `round2_*`.

| Round | AGY log label | Identity pre = post | Verdict | Validity |
|---|---|---|---|---|
| 1 | "Gemini 3.8 Flash (High)" (`cli-20261023_140749.log`) | DIFFER | APPROVED WITH NOTES | INVALID: pre/post identity snapshots differ (WORKFLOW step 5); `check_citations.py` finds 8 bad citations, e.g. `references.json` line 326 in a 71-line file |
| 2 | "Gemini 3.8 Flash (High)" (`cli-20261023_141730.log`) | identical | APPROVED WITH NOTES (no P1/P2, three P3) | review of record (#143); `check_citations.py`: 9 ok, 0 bad |

Smoke calls (`smoke/smoke.stdout`, `diag/smoke.stdout`) returned exactly `REVIEW_SMOKE_PASS\n`. No standalone Coder
report file exists for VIS-3D; the Coder's offline check output is `reports/checks.txt` (28 new + 12 + 35 tests),
build and identity evidence `reports/identity.txt`, b11716 source excerpts `reports/b11716_source_excerpts.txt`.
