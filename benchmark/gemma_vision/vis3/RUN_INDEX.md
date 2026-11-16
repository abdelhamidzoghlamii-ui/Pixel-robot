# VIS-3 / VIS-3B run index

**Archived 2026-11-16, never indexed before.** Until then this folder was git-ignored (`.gitignore` `benchmark/gemma_vision/*`)
and existed only on the phone. Files are unchanged from the phone; hashes in `ARTIFACTS.md`. `README.md` still reads
"Preparation only" — it was written before the runs and was not updated.

What: b1609-dotprod (old) vs b11716 (new) llama-server, motors off. First the multi-phase O,N,N,O design (VIS-3), then
VIS-3B (owner, 2026-10-22): one test per launch (see `README.md`). Diagnosis and review records in `reports/`.
Dates: 2026-10-22 to 2026-10-23 (UTC, from folder names).

Code used:
- `vis3_run.py`, `oneshot.sh` (this folder), `vis2/vis2_run.py` and `vis1/vis1_run.py` imported unchanged, repo
  `chat.py`, `server_manager.py`, `bench.py`, `vis0.py`, `power.py`, `runtime.py` at base commit `685a44be63aa`.
- The 11 VIS-3B runs (`*_old`, `*_new`): every `code_sha256` in `run.json` equals the archived file (checked 2026-11-16).
- `runs/smoke_20261022T153003Z` and `runs/20261022T153641Z` used an earlier `vis3_run.py` and
  `oneshot.sh` (pre-VIS-3B) whose hashes are in their `run.json`; those versions are not in this folder.
- Binaries: `run.json` `binary_sha256` per run (old = `~/llama.cpp-b1609-dotprod`, new = `~/llama.cpp-vis` b11716).

Labels as `vis3d/README.md` cites this folder:
- line 164: "VIS-3 found a small EOG tokenizer difference."

| Folder | Kind |
|---|---|
| `runs/smoke_20261022T153003Z` | smoke, pre-VIS-3B code |
| `runs/20261022T153641Z` | timed run, O,N,N,O design, pre-VIS-3B code (missed the 32 °C gate; `README.md`) |
| `runs/20261022T173244Z_cold_load_old`, `…174952Z_cold_load_new` | VIS-3B `cold_load` |
| `runs/20261022T193415Z_warm_load_old`, `…193617Z_warm_load_new` | VIS-3B `warm_load` |
| `runs/20261022T194033Z_first_response_old`, `…194651Z_first_response_new` | VIS-3B `first_response` |
| `runs/20261022T195150Z_gen_old`, `…200953Z_gen_new` | VIS-3B `gen` |
| `runs/20261023T055204Z_prompt_old`, `…061251Z_prompt_new` | VIS-3B `prompt` |
| `runs/20261023T063009Z_mtp_old` | VIS-3B `mtp`, old build only |
