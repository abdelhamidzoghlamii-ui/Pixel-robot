# MIND-BV — Gemma vision vs YOLO: room, person, distance

Research only. On the robot's photos (M2-Q1 frames, `relate_anything/quality1/runs/20261017T213604Z/frames/`, ids
1–60 and 66), how well does Gemma 4 E2B vision name the room, see a person and judge the person's distance, compared
with the robot's YOLO detections plus simple rules? Accuracy is deterministic (temperature 0; agent resident
allowed); timings NOT MEASUREMENTS.

- Methods: **Y** (results.json detections + room/person/distance rules, no new YOLO run), **V70** / **V140** (Gemma
  vision, 70 / 140 image tokens), **VY70** (V70 plus a line listing Y's detections). Rules, prompt, parsing,
  metrics and the PASS rule are in the docstring of `bv.py` and were fixed before the first Gemma call.
- Owner decisions 2026-11-03: the key has 40 person photos (not 44); photos with one detection keep no box in
  results.json, so Y's distance there is UNAVAILABLE (scored wrong) and VY70 lists them as
  `<class> (position unknown, distance unknown)`.
- Server: BC's b11716 command (`mem1.command(2048)`, b11716 `llama-server`, BC's pins) + `--mmproj` Q8_0 +
  `--image-min-tokens`/`--image-max-tokens` 70 (launch A: V70, VY70) or 140 (launch B: V140); no slot path, no MTP.
- Key custody: `bv_key.json` stays outside the repo (refused inside it); SHA-256 pinned in `bv.py`
  (`456f633b…`). Every frame's SHA-256 is checked against results.json before any call and again at the end.
- Run: `python bv.py --key /termux-home/mind_bv_key/<UTC>/bv_key.json` (Termux python, also from proot Debian).
  Writes `~/robot_runs/mind_bv/<UTC>/` (run.json, requests.jsonl, server_A.log, server_B.log, report.txt,
  artifacts.sha256).
- Offline tests: `python -B test_bv.py` (fake server; no phone).
- Known limitation carried over from B0: an interrupt or a server-stop error can skip report.txt and
  artifacts.sha256 (completed rows stay in requests.jsonl).
