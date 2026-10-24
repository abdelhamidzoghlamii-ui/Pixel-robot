# M2-Q1 — run index

One row per harness run. Run start/end are `start_utc`/`end_utc` in the run's `manifest.json`. Labels are copied from
the runner (`manifest.json` `accuracy_label`, `timing_label`), [the Coder report](reports/M2Q1_CODER_REPORT.md) and
[the Local AI blind grades](grading/M2Q1_BLIND_GRADES.md); archival task VIS-0/M2-Q1 catalog completion (2026-10-24)
adds no measurement, grade or label. Both runs: AGY Coder session resident, base `33c2c78`, `quality1.py`
`bda996b6…` (round 2 APPROVED candidate = tracked file), robot server `setup_q4` (b1609, E2B Q4_0 `8e30dff3…`) on
port 8080, RelateAnything M2 `relsgg-vits16` fp32, yolo11s 640, motors off. No numbered decision records M2-Q1 yet.
Hashes: [ARTIFACTS.md](ARTIFACTS.md).

| Run folder | UTC start – end | Photos | Label |
|---|---|---|---|
| `runs/20261017T212311Z/` | 21:23:11Z – 21:24:50Z | 3 of 70 (`--limit 3`), all ≥ 2 boxes | SMOKE (review evidence for round 2). 3/3 M2 and Gemma identical twice, 0 failed. Not graded. Timings NOT measurements. |
| `runs/20261017T213604Z/` | 21:36:04Z – 22:11:44Z | 70 (60 with ≥ 2 boxes, 10 with < 2), 0 failed | FULL RUN. See the separate labels below. |

Full run labels, kept separate:

| Claim | Label | Source |
|---|---|---|
| Harness outputs: M2 rows identical twice 60/60; Gemma X and Y replies identical twice 60/60 | Accuracy VALID (temperature 0), deterministic | `manifest.json` `counts`, Coder report §6 |
| M2 shown-line grading: 152 lines, R 96 (63 %), V 13, B 10, W 33; pass bar ≥ 70 % strictly right NOT MET; "M2 as built FAILS" | Local AI grade, blind to `key.json`, 50 unique scenes (duplicates skipped) | blind grades §A, §B2 |
| Gemma A/B (Y-ONLY vs Y+M2): unblinded M2 better 34, Y-ONLY 6, tie 10; false statements 10 vs 1 | INDICATIVE, not decisive: many replies broken in both variants by the Gemma 3 prompt template (later CONFIRMED by VIS-0); blindness weak | blind grades §B, §B2 |
| Post-hoc predicate filter (~71 % strictly right) | INDICATIVE only, chosen after seeing the data; needs new frames and a fixed prompt | blind grades §B2 |
| Coder report §6 list of the 10 photos with < 2 boxes | NOT VALID (contradicts the PDF summary page); `results.json`/PDFs are the evidence | blind grades §C |
| `infer_ms_not_a_measurement`, `elapsed_s_not_a_measurement` and other timings | NOT measurements (agent resident) | runner `timing_label` |

All 3 + 70 `frames/` files match `processed_photos[].frame_sha256` and `frame_bytes` in their `manifest.json`
(rechecked 2026-10-24). The full-run `sheet_m2.pdf`, `sheet_ab.pdf` and `key.json` equal the owner copies in
`/storage/emulated/0/Download/m2q1/` (`cmp` 2026-10-24; the `key.json` copy is dated 22:50, after the PDFs).
The 70 frames are reused by VIS-0 and VIS-3D.

## Review records (`reports/`)

Reviewer: Codex CLI `codex exec`, `gpt-6.1-sol`, effort medium, Ponytail off, `--sandbox read-only`, isolated copies
`/termux-home/review/m2q1/round{1,2}_copy` (Coder report §1, §7). Each `roundN_stdout.txt` is the final review; each
`roundN_stderr.txt` is the full Codex transcript of the same call. Its last Codex message is identical to
`roundN_stdout.txt`.

| Round | Candidate `quality1.py` | Verdict | Validity as recorded |
|---|---|---|---|
| 1 | `b8284585…` (668 lines; hash from the phone-only `round1_pre.json`) | CHANGES REQUESTED (import provenance; PDF blindness test) | exit 0; phone audit: HEAD, Git status, copy and live hashes unchanged |
| 2 | `bda996b6…` (695 lines) | APPROVED — review of record | as round 1 |

Limitations, unchanged from the original records: the reviewer's interpreter commands were refused (`exited 182`), so
tests were not rerun by the reviewer; Codex warned that bubblewrap was missing on PATH and used its bundled copy; in
round 1 the reviewer also read Ponytail skill files under `/root/.codex/` outside the isolated copy (read-only). The
audit JSONs, pre/post hash snapshots, review scripts and isolated copies remain on the phone only (hashes in
[ARTIFACTS.md](ARTIFACTS.md)). The smoke ran before round 2 on the code that round 2 then approved.
