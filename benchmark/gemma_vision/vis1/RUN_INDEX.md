# VIS-1 run index

**Archived 2026-11-16, never indexed before.** Until then this folder was git-ignored (`.gitignore` `benchmark/gemma_vision/*`)
and existed only on the phone. Files are unchanged from the phone; hashes in `ARTIFACTS.md`. `README.md` still reads
"Preparation only" — it was written before the runs and was not updated.

What: owner-timed Gemma text and vision blocks T1, T2, V1–V4, motors off (see `README.md`).
Dates: 2026-10-19 to 2026-10-21 (UTC, from folder names).

Code used:
- `vis1_run.py`, `oneshot.sh` (this folder) with repo `chat.py`, `server_manager.py`, `bench.py`, `vis0.py`, `power.py`,
  `runtime.py` at base commit `685a44be63aa`. For `runs/20261019T222505Z` every `code_sha256` in its `run.json`
  equals the archived file (checked 2026-11-16).
- Binaries (`run.json`): `~/llama.cpp-b1609-dotprod/build/bin/llama-server` `77eb35a0…`,
  `~/llama.cpp-vis/build/bin/llama-server` `0674f5c2…`. Q8 projector: `mmproj.json`.
- `bridge_*` and `smoke_*` folders have no `run.json`; their code version is not recorded.

Labels as `vis3d/README.md` cites this folder:
- line 49: "VIS-0 and VIS-1 logged 63 tokens per 482×640 frame."
- line 50: "VIS-1 V2 (Q8, 70): encode about 2.1–2.7 s, wall 8.7–13.6 s with a YOLO list."
- line 51: "VIS-1 V3 (Q8, 140): 154 tokens, encode about 7 s."
- line 76: "(VIS-1 V2 log)"
- line 85: "Q8 projector `9406f99c…` (`vis1/mmproj.json`; ggml-org rev `b4243c15`)"

| Folder | Kind |
|---|---|
| `runs/20261019T222505Z` | timed run (T1…V4 JSON + server logs, `run.json`) |
| `runs/bridge_20261019T124412Z` … `runs/bridge_20261021T085219Z` (10) | native telemetry bridge per launch (`samples.jsonl`, `monitor.log`) |
| `runs/smoke_20261019T144248Z` … `runs/smoke_20261019T193902Z` (7) | smokes; `NOTE_CODER_SMOKE.txt` marks coder smokes |
