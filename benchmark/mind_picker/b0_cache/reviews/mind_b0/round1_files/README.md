# MIND-B0 — prefix-cache mechanics of the mind/picker layout

Research only (#146). How many prompt tokens the deployed one-slot server re-processes when the picker is called again
on a growing event transcript, when a chat turn shares that transcript, and when a chat turn uses a separate context;
and whether cached scoring gives the same action probabilities as fresh scoring. Counters and probabilities only;
timings are recorded but are NOT MEASUREMENTS (agent resident allowed per WORKFLOW "Timed benchmarks").

- Runtime: MEM-1's `mem1.command(2048)` (b1609 dotprod, Gemma 4 E2B Q4_0, `--parallel 1 --swa-full --cache-ram 0`)
  with MEM-1's SHA-256 pins; no MTP, no projector; temperature 0, seed 0.
- Frozen set `b0_set.json`, SHA-256 `dddb288ace1642b1e93a6e268cb68ff1a028121b6853dfa837ac7849cfaa187b`
  (recorded 2026-11-02T10:06:44Z, before any inference; sizing used tokenize-only server calls).
- Layout, sequence R1..R8 + FRESH repeats, and criteria C1–C5: see the docstring of `b0.py`.
- Run: `python b0.py` with Termux python (works from proot Debian); writes
  `~/robot_runs/mind_b0/<UTC>/` (run.json, requests.jsonl, server.log, report.txt, artifacts.sha256).
- Offline tests: `python -B test_b0.py` (fake server; no phone).
