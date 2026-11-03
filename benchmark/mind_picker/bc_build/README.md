# MIND-BC — can the vision build (b11716 + projector) replace b1609 for the picker?

Research only. On `~/llama.cpp-vis` (b11716, source `797de63b…`) with the Q8_0 projector loaded, do prefix caching,
slot save/restore and the B2 picker (V3) behave as on the robot build `~/llama.cpp-b1609-dotprod` (upstream tag
**b10194**, `e1a1abb7…`, #119)? Counters and accuracy are deterministic (temperature 0; agent resident allowed);
timings NOT MEASUREMENTS; RAM INDICATIVE.

- Imports B0 (`../b0_cache`), B0b (`../b0b_slots`), B2 (`../b2_rules`) and VIS-3D (`../../gemma_vision/vis3d`)
  unchanged; only the server command differs: setup_q4's flags (`mem1.command(2048)`) with the b11716
  `llama-server`, plus `--slot-save-path <run>/slots --mmproj ~/models/vis0/mmproj-gemma-4-E2B-it-Q8_0.gguf
  --image-min-tokens 70 --image-max-tokens 70`. Same model file and SHA-256 as b1609; b11716 server, CPU library
  and projector pinned (VIS-3D's hashes). No MTP.
- `provenance.txt` (read-only, both trees): commit and tag, CMakeCache CPU flags, SHA-256 of `llama-server` and
  `libggml-cpu`, `--help` presence of the B0/B0b/VIS flags.
- One launch: K1 B0 sequence (C1–C5 + row diff vs the B0 run of record), K2 B0b L1 save/restore (B0b's K1 + slot
  file size), K3 B2 V3 on B2's TEST set (`../b2_rules/set/b2_test.json`, SHA-256 `d5faed98…`; custody ended), FWD
  then REV with 60 s pauses (B2's PASS rule + items whose choice differs from the B2 run of record), K4 one image
  (VIS-3D frame 1, its prompt and request settings; `enable_thinking` false per request stands in for VIS-3D D's
  `--reasoning off`), K5 one PID / all rows / 0 errors. The server's exit code after SIGTERM is reported; nonzero is
  a finding (#145). Details: docstring of `bc.py`.
- Run: `python bc.py` (Termux python, also from proot Debian). Writes `~/robot_runs/mind_bc/<UTC>/` (run.json,
  requests.jsonl, provenance.txt, server.log, slots/, report.txt, artifacts.sha256).
- Offline tests: `python -B test_bc.py` (B0b's and B2's fake servers; no phone).
- Known limitation carried over from B0: an interrupt or a server-stop error can skip report.txt and
  artifacts.sha256 (completed rows stay in requests.jsonl).
