# MIND-B1 — picker decision quality: old layout vs new layout

Research only (#146, after #139/#140). Python has already removed forbidden actions; does Gemma 4 E2B pick the
action the owner would pick from mission wording, speech, events and memory? Old layout (A, `picker.Gemma` exactly)
vs new layout (B, MIND-B0's chat-template layout), each in catalogue order (FWD) and reversed (REV). Accuracy is
deterministic (temperature 0; agent resident allowed); timings are recorded but are NOT MEASUREMENTS.

- Set: 40 owner-labelled items (4 missions, strict label plus "also ok" labels), kept OUTSIDE the repo per WORKFLOW
  test-set custody: `/termux-home/mind_b1_set/b1_set.json`, SHA-256
  `b1abba5daa3cb87c30a225af40d3e19e6f0feaf82ba95af48f8b48c4a23461eb`, 40 rows, recorded 2026-11-02T13:10:33Z before
  any Gemma call. `b1.py` pins this hash and refuses a set path inside the repo. The set enters the repo only in the
  archive step after scoring, on the owner's decision.
- Imports B0 (`../b0_cache/b0.py`, `b0_set.json` SHA-256 checked) and `picker.py` unchanged. Arms, blocks, metrics,
  the fixed PASS rule and the validity conditions: see the docstring of `b1.py`.
- Choices fixed before the first Gemma call: the current activity is shown as a gerund (exploring, visiting, greeting,
  looking, waiting); the state's "Allowed actions" line stays in catalogue order in both orders; in B the user turn
  is the state lines, a blank line, then B0's tail with the allowed names; picker's letter list is extended to five
  letters (A-E) because it builds only four (its own MODES).
- Run: `python b1.py` with Termux python (works from proot Debian); writes `~/robot_runs/mind_b1/<UTC>/`
  (run.json, requests.jsonl, server.log, report.txt, artifacts.sha256).
- Offline tests: `python -B test_b1.py` (fake server, synthetic three-item set; no phone).
- Known limitation carried over from B0: an interrupt or a server-stop error can skip report.txt and
  artifacts.sha256 (completed rows stay in requests.jsonl).
