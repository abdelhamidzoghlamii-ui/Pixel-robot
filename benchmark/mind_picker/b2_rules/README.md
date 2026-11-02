# MIND-B2 — picker quality: rules, worked examples, one reason sentence

Research only (#146, after MIND-B1). With the policy written as explicit rules, does Gemma 4 E2B pick the owner's
action on new situations? V1 rules only, V2 rules + 13 worked examples, V3 rules + examples + one reason sentence
before the action; each in catalogue order (FWD) and reversed names (REV). Accuracy is deterministic (temperature 0;
agent resident allowed); timings are recorded but are NOT MEASUREMENTS.

- Actions: explore, approach (B1's visit renamed), greet, look, wait.
- DEV: `../b1_quality/set/b1_set.json` (SHA-256 `b1abba5d…`, unchanged; "visit" read as "approach").
- TEST: 40 owner-labelled items kept OUTSIDE the repo per WORKFLOW test-set custody:
  `/termux-home/mind_b2_test/b2_test.json`, SHA-256
  `d5faed98fb424ce8ca959ae58fe6f0ca1cd16d85bb7ed30f7cff8f74cd9f7b1f`, 40 rows, recorded 2026-11-02T14:29:08Z before
  any Gemma call. `b2.py` pins this hash, refuses a TEST path inside the repo (symlinks resolved, inode check kept
  for bind mounts; B1 review P1) and opens TEST only after `selection.json` is written and its SHA-256 logged.
- Selection (fixed before the first Gemma call): among variants with DEV flips ≤ 4, the highest mean acceptable over
  both orders; tie → V1, V2, V3; none → the highest anyway, flagged FLIPS. PASS on TEST: acceptable ≥ 0.80 and
  strict ≥ 0.65 in both orders, flips ≤ 4/40 (B1's rule).
- Imports B1 (`metrics`, `Server`) and B0 (`render`, `tokenize`, `probs`) unchanged. Rendering, V3's two steps and
  the validity conditions: see the docstring of `b2.py`.
- Choices fixed before the first Gemma call: the system text's wrapped lines are joined into single lines (paragraphs
  and list lines kept); examples follow the rules after a blank line and `Examples:`; for an item with restricted
  allowed actions (DEV item 4) the tail names only the allowed ones (as B1); V3 scores the names after a space
  (`Action: wait` tokenizes as `Action`, `:`, ` wait`).
- Run: `python b2.py` with Termux python (works from proot Debian); writes `~/robot_runs/mind_b2/<UTC>/`
  (run.json, requests.jsonl, selection.json, server.log, report.txt, artifacts.sha256).
- Offline tests: `python -B test_b2.py` (fake server, synthetic sets; no phone).
- Known limitation carried over from B0: an interrupt or a server-stop error can skip report.txt and
  artifacts.sha256 (completed rows stay in requests.jsonl).
