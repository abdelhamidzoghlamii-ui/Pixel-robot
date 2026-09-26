# Harness v3: table fix, possible_person dump, re-review

Pixel Robot, Coder session, 2026-09-25. Research only: no motors, no `main.py`, no `docs/`, no commit, no push.

All four items are done, and the re-review passed. It confirms that the previous review's "inverted flips" finding was wrong.

## 1. Per-family table fixed

`report.py` now prints each cell as `P x/n · A y/n`. That was the only code change; its new SHA-256 is `f10d94b1193513d00fa5db7eb1c93aeef558e4b0265be57c66a108fe2ec4a009`. The results weren't re-run, `test_v3.py` passes, and the new report is `/termux-home/v3-runs/dev-full.report.v2.md`.

Development set, 66 cases, frame `filtered_text`, canonical order:

