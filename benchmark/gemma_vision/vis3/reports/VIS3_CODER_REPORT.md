# TASK VIS-3 CODER REPORT: Gemma Vision Runtime Performance Parity (b1609-dotprod vs b11716)

**Execution profile:**
- **Role:** Coder
- **Platform:** Claude Code CLI (proot Debian, `/data/data/com.termux/files/home/robot`)
- **Model:** `claude-opus-5-5`, effort medium
- **Ponytail:** lite
- **Reviewer:** Codex CLI 0.163.0 (`codex exec`), `gpt-6.1-sol`, effort `medium`, Ponytail `off`, sandbox `read-only`. Each
  round ran in a fresh session in an isolated copy under `/data/data/com.termux/files/home/review/vis3/roundN_copy/`.

Status: the harness is prepared and reviewed. **No timed benchmark was run.** Nothing is committed or pushed. Owner
authorization is needed before the commit and before the timed run.

---

## 1. Executive summary

1. **The builds are equivalent (Scope A).** b1609-dotprod (`e1a1abb7`, upstream b10194) and b11716 (`797de63`) match on:
   - compiler: Clang 21.1.8, `-O3 -DNDEBUG`;
   - CPU flags: `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, OpenMP (statically linked libomp), REPACK, LLAMAFILE, no
     KleidiAI, shared libraries;
   - instructions in `libggml-cpu.so`: `sdot` 1045 vs 1117; FP16 `fmla .8h` 456 vs 456; `smmla` 0 (Tensor G2 has no i8mm).

   The CMake caches differ in `LLAMA_BUILD_IS_DEV` (version string) and `LLAMA_BUILD_UI`. NEW also adds some inactive
   entries (reviewer note).
2. **The effective runtime defaults match:**
   - mmap loading: the OLD default is `mmap`; the NEW default `auto` resolves to mmap on CPU (logged `load_mode = mmap`);
   - flash attention `auto` → enabled; n_batch 2048 / n_ubatch 512; `--fit on`;
   - the MTP draft gets the target's threads (the thread-inheritance branch of `common_base_params_to_speculative` is
     the same in both; NEW adds other behaviour to that function);
   - `--swa-full` and `--cache-ram 0` honoured.

   Differences found: NEW removed `--no-mmap`/`--mlock` (use `--load-mode`); NEW compiles out per-request speculative
   overrides (`#if 0` in `server-schema.cpp`); NEW creates an explicit `common_threadpools` (under OpenMP plausibly
   neutral, unverified). Tokenizer EOG handling differs slightly: OLD logs "special_eog_ids contains '<|tool_response>',
   removing '</s>'" and NEW logs "'</s>' is a normal token here". The harness checks tokenization parity per prompt
   (`prompt_n`).
3. **The earlier "b11716 is slower" readings are thermally confounded.** The VIS gate is idle + 1.5 °C, and session idles
   ranged from 27 to 36.7 °C.

   | Run | Build + MTP | Start skin | Gen tok/s |
   |---|---|---|---|
   | VIS-1 T1 | b1609 | 39.1 °C, X1 `scaling_max` 1.58 GHz | 7.07 |
   | VIS-1 T2 | b11716 | 32.5 °C | 9.22 |
   | VIS-2 A | b11716 | 34.2 °C | 7.74 |
   | #137 chat_mtp | b1609 | ~24 °C | 19.18 |

   Prompts and sampling also differed (temperature 0.7 vs greedy, ctx 4096). None of these runs isolates the build.
   Load-to-ready times in the MTP logs were similar: "model loaded" at 9.56 s (OLD) vs 9.74 s / 8.69 s (NEW).
4. **VIS-3 harness (Scope B)** prepared in `benchmark/gemma_vision/vis3/`. Offline tests pass 17/17. Review round 2
   returned APPROVED WITH NOTES.
5. **Dynamic loading (Scope C):**
   - b11716 cannot free the projector or the MTP drafter independently.
   - Only whole-server unloading exists: `--sleep-idle-seconds`, or router mode, which is disabled on this build.
   - Upstream PR #25963 closed unmerged.
   - Independent freeing therefore needs a custom patch or a future upstream feature.

## 2. Artifact and git provenance

- Base: `685a44be63aa88c30e09386ce8e5e196ebf11abc` = `origin/main` = `git ls-remote` (checked before and after both
  review rounds).
- Pre-existing local state: two untracked `benchmark/relate_anything/quality1/reports/round{1,2}_stderr.txt` files, not
  touched.
- `benchmark/*` is git-ignored (`.gitignore:8`). VIS-1, VIS-2 and VIS-3 are local only, so committing needs `git add -f`.
- Candidate (four new files; SHA-256 identical to what review round 2 saw):

| File | Lines | SHA-256 |
|---|---:|---|
| `vis3/vis3_run.py` | 566 | `6c18f9f40d2879a847a2b88c2037f643f08f0d595717374780d3abaac51e0034` |
| `vis3/test_vis3.py` | 336 | `32257111211ffeaa36227f0a90c65e679fbfe8699649d5c920f3165b05548b76` |
| `vis3/oneshot.sh` | 110 | `399fcdac14684e3977c00e983c05ffca4a2e2d5c6e4a4417b8b5014b57ee7ebe` |
| `vis3/README.md` | 81 | `22c219e6ba06c3811c11ecbea7391d20b32e4604b3e26559d9ae7ab4d155cefb` |

- This report and the archived review files in `vis3/reports/` were added after review. They are documentation, not
  harness code.
- Full diff: `/data/data/com.termux/files/home/review/vis3/round2_copy/review_inputs/candidate.diff`. Round-1 → round-2
  changes: `reports/round1_to_round2.diff`.
- Unchanged, as required: `server_manager.py`, `chat.py`, robot navigation, deployed binaries, model files, the VIS-1 and
  VIS-2 runners and their results.
- Verified hashes:
  - `gemma-4-E2B-it-Q4_0.gguf` `8e30dff3…`
  - `mtp-gemma-4-E2B-it-Q8_0.gguf` `9eba8199…`
  - `mmproj-gemma-4-E2B-it-Q8_0.gguf` `9406f99c…`
  - all b11716 binaries and libraries match the VIS-2 manifest.
  - `$PREFIX/lib` holds no `libggml`/`libllama`/`libmtmd`; both launchers list it before their own build dir in RUNPATH.

## 3. Harness design

| Block | Binary | MTP |
|---|---|---|
| O | `~/llama.cpp-b1609-dotprod` | off |
| N | `~/llama.cpp-vis` | off |
| OM / NM (optional) | as O / N | `--model-draft … --spec-type draft-mtp --spec-draft-n-max 3` |

- **Command:** exactly `server_manager.start_setup('setup_q4' | 'setup_q4_mtp')` (ctx 2048, threads 4/4, parallel 1,
  `--swa-full --cache-ram 0`), plus `taskset -c 4-7`, `timeout 900` and port 8091. A test captures the real
  `server_manager` command with `os.system` stubbed. OLD and NEW differ only in the binary path. No projector, no
  `--reasoning` flag (`/completion` with the raw `chat.py` prompt bypasses the template).
- **Gate:** VIS-2 telemetry stops reused unchanged (battery ≥ 50 % Discharging, Android CRITICAL, battery ≥ 45 °C, abort,
  agent, affinity). Added on top: skin ≤ `SKIN_START_MAX` 32.0 °C, waiting up to 1800 s.
- **Cold phase:** gate, cache drop, gate, launch, `/health`, verify (`/props` vision false, `/slots` speculative ==
  intended, n_ctx 2048), mapped-library check (`/proc/<pid>/maps` must stay inside the block's own build dir), PSS,
  first response (gen[0] prompt, 16 tokens), stop after the sampler detaches.
- **Warm phase:** gate (no drop), relaunch, the same checks and first response, 4 rounds of U (1005-token uncached log,
  8 tokens), K (cached prefix, 8 tokens) and G (greedy, ≤ 128 tokens), stop.
- **Recorded:**
  - load time (spawn to `/health`) and log-phase timestamps;
  - first-response wall time, prompt and generation tok/s, cache reuse, draft acceptance;
  - mean CPU frequency per request from `time_in_state` (policy4/6, no root needed);
  - PSS, peak PSS, minimum MemAvailable, battery W, skin, Android status, caps;
  - LMK lines, exit code, stop time.
- **Environment:** the runner and the Debian preflight refuse `LLAMA_*`, `GGML_*`, `OMP_*`, `KMP_*`, `LD_LIBRARY_PATH` and
  `LD_PRELOAD`.
- **Partial runs:** each block can run on its own. `--summarize` pools folders only when their code, payloads, weights,
  binaries, commands and constants match. It reports per-config medians, NEW/OLD ratios and comparability (prompt_n and
  generated tokens identical, cold and warm start-skin spread, all checks pass).

## 4. Owner commands and expected runtime

```bash
bash ~/robot/benchmark/gemma_vision/vis3/oneshot.sh --preflight
bash ~/robot/benchmark/gemma_vision/vis3/oneshot.sh --smoke --blocks O,N        # ~2 min, NOT timing
bash ~/robot/benchmark/gemma_vision/vis3/oneshot.sh --blocks O,N,N,O            # essential
bash ~/robot/benchmark/gemma_vision/vis3/oneshot.sh --blocks OM,NM,NM,OM        # optional
python3 -B ~/robot/benchmark/gemma_vision/vis3/vis3_run.py --summarize <folder> [<folder> ...]
```

`O,N,N,O` should take about 45–100 min: a 5-min idle, about 4 active minutes per block, plus cooling to the gate before
each block and each warm phase. If the phone idles above 32 °C, cool the room or remove the case. Never raise the
ceiling to make a run pass.

## 5. Checks run by the Coder

- `test_vis3.py`: 17/17 ok (log-marker test uses the real archived b1609 and b11716 logs).
- `test_vis2.py`: ok.
- `test_vis1.py`: 31 tests, **1 pre-existing error**. `test_wrapper_contract` expects
  `[ "$PREFLIGHT" = 0 ] || exit 0`, which is no longer in `vis1/oneshot.sh`. Not touched.
- `--dry-run`: ok.
- Debian `--preflight`:
  - refused "agent found" with the agent resident, as expected;
  - with only the agent check stubbed: `PREFLIGHT OK`;
  - it refuses a set `LD_LIBRARY_PATH`.
- No `__pycache__` or other stray files left.

## 6. Scope C: dynamic loading (b11716 source, research only)

- **Projector:** `clip.cpp` reads it into allocated buffers (`fin.read`), not mmap, so it stays resident until the
  process exits. Text requests skip inference on it but free no RAM.
- **MTP drafter:** no per-request disable; the speculative fields are under `#if 0` in `server-schema.cpp`.
- **Freeing RAM:**
  - `server_context_impl::destroy()` frees the target, draft and mmproj together. It is used by
    `--sleep-idle-seconds`, which reloads all three.
  - Router `/models/load` and `/models/unload` restart whole child processes. On this build router mode is unavailable:
    `LLAMA_SUBPROCESS` defaults OFF on Android ("subprocess is not enabled on this build").
- **Upstream:**
  - PR #25963 (`reload_context` / `reload_mmproj` / `reload_speculative`, no fit on reload) closed unmerged on
    2026-08-11.
  - Discussion #25674 is still open.
  - Draft PR #28346 (`--mmproj-evict-draft`) targets GPU VRAM swapping.
  - #30776 (persistent draft threadpool) is a draft and matters only for non-OpenMP builds.
- **Conclusion:** no supported mechanism exists today. A newer runtime might provide one, but nothing is merged.
  Independent freeing needs a custom patch. Restarting the server with the files in the page cache (warm load) is the
  supported route.

## 7. Optimizations

- **Present on both builds:** dotprod, FP16, REPACK, LLAMAFILE, OpenMP, flash attention, mmap, `--swa-full`,
  `--cache-ram 0`, 4 threads on cores 4–7.
- **Plausible:** `--fit off` might save load time. The b11716 fit step took about 2.4 s vs 1.1 s on b1609 in the MTP logs.
  The explicit threadpool is plausibly neutral under OpenMP.
- **Unverified:** actual speed parity, until the owner run.

## 8. Review history

| Round | Verdict | Integrity |
|---|---|---|
| 1 | CHANGES REQUESTED: 5 × P2 (environment overrides; warm phase not gated; library failure not fatal; sampler race at the cold stop; preflight missing `requests`) | VALID: copy, live tree, git, Codex config unchanged; every citation exact |
| 2 | **APPROVED WITH NOTES**: 1 × P3 (sampler detach is best-effort; on a > 5 s monitor stall the block may fail closed; the README states the guarantee too strongly) | VALID, same checks |

All five P2 findings were fixed between rounds. The P3 is open: fixing it is a code change and needs a new review.
Verbatim reviews are in `reports/round{1,2}_stdout.txt`, with diagnostics in `round{1,2}_stderr.txt` and requests in
`round{1,2}_request.md`. In both rounds the reviewer could not rerun the tests (exit 182 in the sandbox) and inspected
the supplied outputs instead.

## 9. Discrepancies with pushed documentation (for Doc Keeper)

1. VIS-1, VIS-2, VIS-3 and the `~/llama.cpp-vis` (b11716) build are absent from STATUS and DECISIONS.
2. STATUS.md line 36 calls `~/llama.cpp` (b1609) the documented deployed server. `server_manager.py` points to
   `~/llama.cpp-b1609-dotprod` (#114).
3. `~/llama.cpp-vis` is a shallow one-commit clone: `/props` reports build 1, while `git describe` says b11716.
4. The VIS-1 `test_wrapper_contract` failure (section 5).

## 10. Owner decisions pending

1. Commit the four candidate files plus `vis3/reports/` with `git add -f` (push separately).
2. Fix the P3 first (needs review round 3), or accept it as noted.
3. Authorize the timed run: preflight, then smoke, then `O,N,N,O`, then optionally `OM,NM,NM,OM`.
