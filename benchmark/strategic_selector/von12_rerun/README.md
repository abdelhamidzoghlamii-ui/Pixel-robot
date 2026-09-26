# Von 1.2 rerun of strategic-selector v2 (2026-09-24)

Status: **complete** (research only; offline synthetic text cases, no motors, no robot integration).
Relates to DECISIONS #109 and #112. It reruns the archived v2 procedure
([../README.md](../README.md), [../RUN_INDEX.md](../RUN_INDEX.md)) with `von-sdk` 1.2.0 instead of
1.1.1. The v2 benchmark source `../robot_selector_benchmark.py` (SHA-256 `3aa9d399…8c14e2`) was run
unmodified through the repo's own `../archive_run.py`.

## Result

- Reversed-order flips were **0/11** for every Von 1.2 frame tested (`filtered_json`, `filtered_text`,
  `two_stage_text`); 1.1 had 9/11.
- Held-out accuracy (v2's 11 held-out cases, not the sealed v3 split) **fell**: `two_stage_text`
  acceptable 8/11 → 2/11, `filtered_text` 8/11 → 5/11; the development-selected frame changed to
  `filtered_json` (acceptable 4/11). 1.2 chose `ask_gemma` in 58/110 development rows (1.1: 12).
- Peak RSS 3097 MiB (external `wait4`), about 35 MiB above 1.1's self-reported 3062 MiB.
- Weights: `wfzyx/von` snapshot `5df8185a4f2327ad0a7cd117cc4f701ac557b9ae`, `option_marker.pt`
  SHA-256 `3faf27f8…d4139ed`; the loader pins no revision.

Von 1.2 was dropped (DECISIONS #112). Its venv and weights were deleted from the phone on 2026-09-26
(human decision); none are in Git.

## Contents

| Path | What |
|---|---|
| `raw/runs/2026-09-24T184107Z-von-68e0c438/` | Block A: full v2 procedure via `archive_run.py` (manifest, results, stdout, stderr) |
| `raw/von12-heldout-extra-results.json`, `raw/von12_heldout_extra.py` | Block B: `filtered_text` and `two_stage_text` on the held-out cases, both orders, and its executed script |
| `raw/run_block.py`, `raw/block*.{block.json,stdout.txt,stderr.txt}`, `raw/probe.*`, `raw/von12-probe.json` | the wrapper (external peak RSS, battery, thermal start/end) and its per-block records |
| `raw/determinism_check.py` + outputs, `raw/negative_control/` | case-identity check against the archived 1.1 rows (824 fields, 0 mismatches) and its negative control |
| `raw/thermal.log`, `raw/cooldown_gate.txt` | root thermal log (zones 9/10/11, 5 s) and the cooldown gate readings |
| `raw/archived_pins.txt`, `raw/venv_freeze.txt`, `raw/archived_freeze.txt`, `raw/pip_install_*`, `raw/download_*` | environment: 1.1 pins, 1.2 venv freeze, install and download logs |
| `review/` | the Reviewer requests, stdout/stderr of both attempts, frozen-file hashes |
| `reports/CODER_REPORT_von12_rerun.md` | the Coder's reports, verbatim from the session transcript |

[RUN_INDEX.md](RUN_INDEX.md) lists the runs; [ARTIFACTS.md](ARTIFACTS.md) has the SHA-256 of every file.

## Review status

Reviewer AGY `gemini-3.1-pro-high`, `--sandbox`. Attempt 1 was **incomplete** (a denied shell command,
empty response). Attempt 2: **PASS WITH CAVEATS** — the determinism script omitted the instruction
fields (checked separately afterwards: identical), the 1.1/1.2 latency comparison is thermally
confounded, and the weights are unpinned upstream.

## Evidence gaps

- The 1.1 reference has no thermal or phone-state record; 1.1 vs 1.2 latency is not controlled.
- The 1.1 `filtered_text` held-out follow-up saved only stdout, so only its labels were compared.
- The two warm-up calls are excluded from the aggregates, as in 1.1 (source unchanged).
- The `src122/` Von 1.2 source excerpts inspected during the run are third-party code and not archived.
- The ladder cases elsewhere in `benchmark/strategic_selector/ladder/` are a development bench, not
  the final exam; the same holds for every generator-derived case set here.
