# VIS-2 run index

**Archived 2026-11-16, never indexed before.** Until then this folder was git-ignored (`.gitignore` `benchmark/gemma_vision/*`)
and existed only on the phone. Files are unchanged from the phone; hashes in `ARTIFACTS.md`. `README.md` still reads
"Preparation only" — it was written before the runs and was not updated.

What: Q8 projector residency vs Gemma text speed, MTP on/off, configs A/B/C/D/M, motors off (see `README.md`).
Dates: 2026-10-22 (UTC, from folder names).

Code used:
- `vis2_run.py`, `oneshot.sh` (this folder), `vis1/vis1_run.py` imported unchanged, repo `chat.py`, `server_manager.py`,
  `bench.py`, `vis0.py`, `power.py`, `runtime.py` at base commit `685a44be63aa`. In all three `run.json` files every
  `code_sha256` equals the archived file (checked 2026-11-16).
- Binary: `~/llama.cpp-vis` (b11716) `llama-server` `0674f5c2…` and its libraries, listed in `run.json`
  `manifest.binary_sha256`.

Labels as `vis3d/README.md` cites this folder:
- line 86: "`llama-server` and the seven b11716 libraries (pinned in `BUILD_SHA256`; equal to the VIS-2 manifest)."

| Folder | Kind |
|---|---|
| `runs/smoke_20261022T105154Z` | smoke |
| `runs/20261022T114634Z` | timed run |
| `runs/20261022T122913Z` | timed run |
