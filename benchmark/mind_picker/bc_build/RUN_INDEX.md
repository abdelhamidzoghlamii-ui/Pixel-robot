# MIND-BC run index

Archived 2026-11-03 by `cp -p` from `~/robot_runs/mind_bc/20261102T213956Z/` and
`/termux-home/mind_bc_stdout_20261102T213949Z.txt` (runs/) and `/termux-home/review/mind_bc_20261102T215512Z/`
(reviews/); every file `cmp` equal (`diff -r`); the run's `artifacts.sha256` passes inside the archive. Before
archiving, `bc.py` (`e39abfdc…`, = run.json `code_sha256`), `test_bc.py` (`4c87ea90…`) and `README.md` (`665bcba0…`)
matched the review's `frozen_candidate.sha256`. SHA-256 of every archived file: [ARTIFACTS.md](ARTIFACTS.md). Base
`391bd55`. Run from proot Debian with Termux python, Claude Code Coder session resident, nothing else started.

## Build provenance (read-only; `runs/20261102T213956Z/provenance.txt`)

| | robot build `~/llama.cpp-b1609-dotprod` | `~/llama.cpp-vis` |
|---|---|---|
| Upstream tag / commit | **b10194** / `e1a1abb7…` (#119) | b11716 / `797de63b…` |
| GGML_CPU_ARM_ARCH, GGML_NATIVE, HAVE_DOTPROD | `armv8.2-a+dotprod+fp16`, OFF, 1 | same |
| i8mm, extra C/CXX flags | none, none | none, none |
| llama-server / libggml-cpu SHA-256 | `77eb35a0…` / `a4ef0d97…` (.so.0.18.0) | `0674f5c2…` / `9e4e11cf…` (.so.0.27.0) |
| Flags used (`--swa-full`, `--cache-ram`, `--parallel`, `--slot-save-path`, `--kv-unified`, `--mmproj`, `--image-min/max-tokens`) | all listed | all listed |

b11716 adds `--kv-unified-per-slot` (unset = unchanged); its `--cache-ram` help names "0 for decision models" (the
run passes 0 explicitly); its log says "--cache-idle-slots requires --cache-ram, disabling".

## Run

| Run | Content | Result | Labels |
|---|---|---|---|
| `runs/20261102T213956Z` | one launch, b11716 + Q8_0 projector, setup_q4 flags + `--slot-save-path` + projector flags; K1 B0, K2 B0b L1, K3 B2 V3 TEST, K4 one image | all K1–K5 **PASS**; exit 0 after SIGTERM | counters and accuracy VALID (107/107 rows, one PID 30907, 0 errors, set hashes unchanged); timings NOT A MEASUREMENT; RAM INDICATIVE |

Console log of the run: `runs/mind_bc_stdout_20261102T213949Z.txt`.

| Check | b11716 + projector | b1609 record |
|---|---|---|
| K1 B0 sequence | C1, C2, C4, C5 PASS; 0/13 rows differ (tokens, ids hash, prompt_n, cache_n, top) | B0 `20261102T110006Z` |
| K1 C4 max \|dp\| cached vs fresh | 0.016179 | 0.016179 |
| K2 R7 after restore | cache_n 393, prompt_n 41 (PASS) | B0b L1: 393, 41 |
| K2 slot file (422 tokens) | 7,790,556 bytes | 7,790,524 bytes |
| K3 TEST V3 strict / acceptable (FWD/REV), flips | 0.900/0.900, 0.900/0.900, 3 [6, 36, 39] → PASS | identical (B2 `20261102T143251Z`) |
| K3 items with a different choice | 0/80 | — |
| K4 image (frame `bench_photos__blocked_1`) | non-empty reply (PASS) | — |
| RAM after load: Pss / VmHWM (kB) | 3,544,201 / 3,732,228 | B0b L0: 3,601,416 / 3,884,860 |
| RAM end: Pss / MemAvailable (kB) | 3,374,881 / 2,024,824 | B0b L0: 3,535,400 / 2,351,596 |

K4 reply (verbatim): "In front of the robot, there is a wooden table with a laptop, a blue object, and some other
items like a glass and scissors." Owner-approved deviation from VIS-3D: thinking is disabled per request
(`chat_template_kwargs.enable_thinking` false) instead of the server flag `--reasoning off`; K4 is a function check
only, not comparable to VIS-3D replies.

Post-run check outside `bc.py` (`reviews/mind_bc_20261102T215512Z/prob_compare.py` → `prob_compare.txt`, written by
the Coder after the run, before the review): K1 vs B0 and K3 vs B2 max |dp| 0.0, generated ids equal, 80/80 reasons
word-for-word; K2 vs B0b L1 differs only at R1/R2 (K2 starts on a slot warm from K1: R1 cache_n 363 vs 0; |dp| up to
0.027; same tops).

## Review (`reviews/mind_bc_20261102T215512Z/`)

Codex CLI `gpt-6.1-sol`, medium, Ponytail off, fresh session (thread `01a19e43…`, model and effort confirmed in the
session log), source-only static review (no tools; #144 R3), 2026-11-02 21:56:46–21:57:36Z, exit 0, empty stderr,
0 command executions, `snap_pre.txt` = `snap_post.txt`: **APPROVED WITH NOTES**, one P3 (K1's report line shows the
R6 reply as `None` although the K1 R6 row has it, `bc.py:236`); citation verified exact (365 lines). Review of
record: `final_review.md`, `review_identity.txt`. Owner decision 2026-11-02: accepted; the P3 is recorded, not fixed;
no code change after review. b11716 + Q8 projector becomes the test build for the next tests (not a deployment
change). `empty_cwd/` (the reviewer's empty working directory) is not tracked by Git; `server.log` and
`slots/picker.bin` are added with `git add -f`.
