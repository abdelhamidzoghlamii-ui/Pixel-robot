# Campaign owner evidence

Archived unchanged from owner Downloads, SHA-256 verified against each source. Includes the three Git-ignored preflight `_llama-server.log` files, explicitly listed in the frozen manifest and commit proposal (force-add only after the human gate). Full file inventory: [archive_sha256.json](runs/archive_sha256.json). No timed benchmark completed.

| Stem | Fixed label | Evidence |
|---|---|---|
| [owner_lag_p1](runs/owner_lag_p1.json) | POWER LAG PROBE — method check, not a benchmark | Complete, 11.5 Hz. current_now rise 50 % 0.25 s, 90 % 0.55 s, fall 50 % 0.29 s; current_avg slow (rise 50 % 3.0 s, 90 % 6.5 s): current_now is the valid power source. |
| [owner_preflight_p1](runs/owner_preflight_p1.json) | PREFLIGHT ONLY — NO TIMING | Passed; all layouts. |
| [owner_preflight_p1b](runs/owner_preflight_p1b.json) | PREFLIGHT ONLY — NO TIMING | Passed; all layouts. |
| [owner_preflight_p1c](runs/owner_preflight_p1c.json) | PREFLIGHT ONLY — NO TIMING | Passed; --blocks L0,L1,L2. |
| [owner_session_p1a](runs/owner_session_p1a.json) | NOT VALID — SESSION INCOMPLETE | Setup refusal "root shell/monitor mask empty or includes inference cores" after L0 check; only L0 recorded. No idle, no block. .stdout/.stderr were overwritten by an immediate rerun that refused with "evidence exists". |
| [owner_session_p1a2](runs/owner_session_p1a2.json) | NOT VALID — SESSION INCOMPLETE | Same refusal reproduced with full traceback; only L0 recorded. No idle, no block. |
