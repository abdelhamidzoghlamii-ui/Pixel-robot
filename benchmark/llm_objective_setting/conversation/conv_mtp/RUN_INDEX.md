# Conversation MTP speed — run index

| Step | When (UTC) | Output | Status |
|---|---|---|---|
| review rounds 1–5 of `conv_speed_mtp.py` / `run_conv_mtp.sh` | 2026-09-28 before 08:01 | `reviews/review1…5/` | round 5 APPROVE |
| timed run, 5 configs × 13 prompts (oneshot 08:13:59Z, 5 min idle) | 2026-09-28 08:19:06–08:36Z | `conv_mtp_20260928T081906Z/`, `conv_mtp_20260928T081906Z.{stdout,stderr}.txt`, `oneshot_console_20260928T081359Z.log` | complete, no resume, stderr empty |
| C scoring (`../score_c.py`, report only) | 2026-09-28 08:36Z | `conv_mtp_20260928T081906Z/c_scores.txt` | complete |

The smoke logs and three toy runs made during preparation (`/termux-home/ladder/conv_mtp_prep/`) are not
archived; the review requests quote their output.
