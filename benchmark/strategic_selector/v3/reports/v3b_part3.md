
**My note on the review:** I disagree with "no remaining threats". These are still open:

- laya_micro's tokenizer matched stock on 0 of 132 calls;
- s1o is zero-shot base Gemma, not the trained system-one-open model;
- the timings are provisional;
- v2's `hall_hint` label defect (v3 skips variations 3–5);
- only one frame (`filtered_text`) has been run.

## 4. Held-out

Recorded in `/termux-home/v3-runs/HELDOUT_SEAL_NOTE.md`. It stays procedurally sealed with no code change: the threat is the team tuning on it, not an outsider. 66 cases, seed 7919, SHA-256 `96d41308eb0d2fa0c3bf30c1370eb93d1b4b7afbfd13cd433370cfd5cc3b2601`, never loaded, printed or scored. `run.py` refuses it without `--unseal-heldout`, and `test_v3.py` checks that.

Nothing is committed or pushed, and `docs/` is untouched.
