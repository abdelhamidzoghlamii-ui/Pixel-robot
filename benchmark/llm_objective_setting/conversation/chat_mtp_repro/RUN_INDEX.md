# chat_mtp_repro (VIS-3C) — run index

One row per launch with its console. Run start is `started` in the run's `run.json`; console start is the UTC time in
the console file name. Labels: archival task VIS-3C/3D-ARCHIVE (2026-10-23), from the runner's own per-run labels and
DECISIONS #142; no measurement was rewritten. `runs/thermal.log` (launcher CPU-zone logger, appended by every
launch, 2026-10-23 11:57:33Z – 13:21:21Z) spans all three consoles and belongs to no single row; its zone readings are
CPU-core readings, not a heat state. Hashes: [ARTIFACTS.md](ARTIFACTS.md).

| Folder / console | Date UTC | Config | Label |
|---|---|---|---|
| — ; `runs/oneshot_console_20261023T115731Z.log` | console 11:57:31Z | not recorded | REFUSED: `not started: battery 78 % < 80 %` (exit 1). No server, no run folder. NOT A MEASUREMENT. |
| `runs/20261023T124846Z_OFF/`; `runs/oneshot_console_20261023T124838Z.log` | console 12:48:38Z; run 12:48:51Z | OFF | VALID (runner label). 13/13 prompts in one block, `all prompts done`; 13.35 tok/s, 686 gen tokens, prompt 0.36 s, load 5.4 s; start skin 25.11 °C, status 0, battery 95 %, X1 cap 2850 MHz. Single run. |
| `runs/20261023T132032Z_ON/`; `runs/oneshot_console_20261023T132024Z.log` | console 13:20:24Z; run 13:20:35Z | ON | VALID (runner label). 13/13, `all prompts done`; 18.55 tok/s, 674 gen tokens, acceptance 38.2 %, prompt 0.34 s, load 7.3 s; start skin 25.51 °C, status 0, battery 92 %, X1 cap 2850 MHz. Single run. |

Pair OFF/ON: ON/OFF 1.390; `configuration: SAME`, `conditions: MATCHED`. Original status: owner-reported runner
console output, not saved. `reports/REGENERATED_20261023T190340Z_combined_OFF_ON.txt` is a REGENERATED report
(generated 2026-10-23T19:03:40Z by `repro.py --report` on the unchanged phone folders, report-only, no inference);
it shows the same ratio and labels. It is not an original artifact of the run.

## Review records (`reports/`)

Candidate: the four source files, frozen SHA-256 in `reports/frozen_candidate.sha256` (= the `run.json` code hash of
`repro.py`). AGY reviewer, Ponytail off. Round validity below is from the archived records and AGY logs, checked
in this archival task.

| Round | Model (TASK / AGY log label) | Identity pre = post | Verdict | Validity |
|---|---|---|---|---|
| 1 | `gemini-3.8-flash-medium` / "Gemini 3.8 Flash (Medium)" (`cli-20261023_090340.log`) | identical | APPROVED | INVALID: cites `repro.py` lines 513, 574 and 689 in a 364-line file, and line 280 quotes the code at line 44 (WORKFLOW AGY procedure step 7) |
| 2 | `gemini-3.8-flash-medium` / "Gemini 3.8 Flash (Medium)" (`cli-20261023_091155.log`) | identical | APPROVED WITH NOTES | SUPERSEDED: the owner changed the reviewer model to `gemini-3.8-flash-high` before round 3 (`round2_TASK.md` vs `TASK.md`) |
| 3 | `gemini-3.8-flash-high` / "Gemini 3.8 Flash (High)" (`cli-20261023_091954.log`) | identical | APPROVED WITH NOTES | review of record (#142). The archival check matched 3 of its citations; the rest were not machine-checked here. |

Smoke calls: `smoke_stdout.txt` and `smoke3_stdout.txt` returned exactly `REVIEW_SMOKE_PASS\n`; `smoke2_stdout.txt`
and `smoke2b_stdout.txt` returned `REVIEW_SMOKE_PASS.\n` (trailing period, not the exact WORKFLOW reply) before round 2.
`round1_TASK.md`/`round2_TASK.md` are the TASK copies given to those rounds; `TASK.md` is round 3's.
`review_context/` holds the two VIS-3B background files that were in round 3's review context. No standalone Coder
report file exists for VIS-3C; the Coder's offline check output is `reports/checks.txt` (12 new + 35 chat_mtp tests).
