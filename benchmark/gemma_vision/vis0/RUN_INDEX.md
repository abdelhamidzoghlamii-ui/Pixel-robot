# VIS-0 — run index

One row per harness run. Run start/end are `start_utc`/`end_utc` in the run's `manifest.json`. Labels are copied from
the runner (`manifest.json` `labels`), [the Coder report](reports/VIS0_CODER_REPORT.md) and
[the Local AI notes](grading/VIS0_LOCALAI_NOTES.md); archival task VIS-0/M2-Q1 catalog completion (2026-10-24) adds
no measurement, grade or label. Both runs: AGY Coder session resident, base `33c2c78`, research build `~/llama.cpp-vis`
(b11716, `797de63b…`), E2B Q4_0 `8e30dff3…`, **BF16** projector `711e1e8f…`, port 8090, motors off. No numbered
decision records VIS-0 yet. Hashes: [ARTIFACTS.md](ARTIFACTS.md).

| Run folder | UTC start – end | `vis0.py` SHA-256 | Frames | Label |
|---|---|---|---|---|
| `runs/20261018T004817Z/` | 00:48:17Z – 00:57:27Z | `edfc3997…` = review round 2 candidate (VERDICT: REVISE), not the approved code | 2 (`--limit 2`) | SMOKE. Step A verdict CONFIRMED; 8/8 vision calls ok. Not used for any result. Timings and memory NOT measurements (agent resident). |
| `runs/20261018T010701Z/` | 01:07:01Z – 01:52:26Z | `ea3d1e38…` = round 3 APPROVED candidate = tracked file | 10 | FULL RUN. See the three separate labels below. |

Full run labels, kept separate:

| Claim | Label | Source |
|---|---|---|
| Template check (Step A): Gemma 3 markers are plain text for this GGUF, `<\|turn>`/`<turn\|>` are single special tokens; OLD markers 9/10 replies broken, NEW (GGUF template) 10/10 clean | VALID (deterministic) — CONFIRMED | `results.json` `template_check`; Coder report §1; Local AI notes |
| Vision replies at budgets 70/140/280 and 140 + YOLO: 40/40 calls, 0 crashes; 63/154/300 image tokens | VALID as outputs (temperature 0) | `run.log`, `results.json`, Coder report §1 |
| Vision reply quality (best: photo + YOLO at 140; larger budget not better; YOLO label errors copied) | INDICATIVE — Local AI, 10 frames, not blind, from `vis0_sheet.pdf` | Local AI notes |
| Wall times (~21 s / 54 s / 109 s per image) and `peak_vmhwm_kib_not_a_measurement` | NOT measurements (agent resident). Never a timing result; the follow-up owner-timed test was VIS-1 (not in this archive) | runner `timing_label` |

Archival observation (2026-10-24, not a reclassification): `results.json` `servers[].exit_code` records how each
llama-server exited when the harness stopped it after its pass: smoke 1/1/0/1/0, full run 1/0/1/−6/−6 (−6 = SIGABRT,
budget 280 and 140 + YOLO). All calls of those passes had already returned ok; the shutdown exits were not
investigated in the original records.

Frames are the tracked M2-Q1 full-run frames (`benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/`),
not copied here; all 2 + 10 `frame_sha256` in the two manifests match those files (rechecked 2026-10-24).
`vis0_sheet.pdf` of each run equals its owner copy in `/storage/emulated/0/Download/vis0/` (`vis0_sheet.pdf` and
`vis0_sheet_20261018T004817Z.pdf`; `cmp` 2026-10-24).

## Review records (`reports/`)

Reviewer: Codex CLI `codex exec`, `gpt-6.1-sol`, effort medium, Ponytail off, `--sandbox read-only`, isolated copies under
`/termux-home/review/vis0/` (Coder report §5). Each `roundN_stdout.txt` is the final review; each `roundN_stderr.txt` is
the full Codex transcript of the same call. Its last Codex message is identical to `roundN_stdout.txt`.

| Round | UTC (audit) | Candidate `vis0.py` | Verdict | Validity as recorded |
|---|---|---|---|---|
| 1 | 00:41:33Z – 00:43:10Z | `d7ed145d…` (1,381 lines) | REVISE (4 × P2) | exit 0; phone audit: Git, live candidates and isolated copy unchanged |
| 2 | 01:00:18Z – 01:01:44Z | `edfc3997…` (1,419 lines) | REVISE (1 × P2) | as round 1 |
| 3 | 01:04:54Z – 01:06:31Z | `ea3d1e38…` (1,437 lines) | APPROVED — review of record | as round 1 |

Limitations, unchanged from the original records: the reviewer's interpreter commands were refused by the sandbox
(`exited 182`), so it did not rerun tests; Codex warned that bubblewrap was missing on PATH and used its bundled copy;
the audit JSONs, review scripts and isolated copies remain on the phone only (hashes in [ARTIFACTS.md](ARTIFACTS.md)).
The smoke ran before round 3 on code that round 2 rejected.
