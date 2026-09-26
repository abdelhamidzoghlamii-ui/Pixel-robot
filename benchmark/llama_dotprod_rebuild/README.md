# llama.cpp b1609 rebuilt with ARM dotprod (2026-09-26)

Status: **complete**. The robot's llama.cpp (build 1609, commit `e1a1abb78746c025f5e9039f590e37ccdb758ae7`)
was rebuilt at the same commit with dotprod enabled, in a new directory
`/data/data/com.termux/files/home/llama.cpp-b1609-dotprod`; the original `~/llama.cpp` build is
untouched. `server_manager.py` points at the new build from commit `341bde6`.

## Build

```
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/data/data/com.termux/files/usr/bin/cc -DCMAKE_CXX_COMPILER=/data/data/com.termux/files/usr/bin/c++ \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
make -C build -j6 llama-server
```

Termux clang 21.1.8 for both builds. `build/cmake_cache_diff.txt`: the only functional CMake cache
difference from the original build is `GGML_CPU_ARM_ARCH` (empty → `armv8.2-a+dotprod+fp16`); the
compiler entries differ only in their cache type (FILEPATH vs STRING, same path). The web-UI bundle for
b1609 could not be downloaded, so the build used the latest one; the robot only calls `/completion`.
`build/llama-dotprod-*.log` are the configure/build logs; `build/llama-dotprod-nofp16-*.log` are the
diagnostic dotprod-only build (`GGML_CPU_ARM_ARCH` without `+fp16`), whose directory was removed earlier;
the phone copies of these two logs were deleted on 2026-09-26 after archiving (human decision).

## Dotprod evidence

`build/sdot_counts.txt` (recomputed at archive time with `llvm-objdump -d`): `libggml-cpu.so` has
**0 → 1045** `sdot` and **0 → 456** fp16 `fmla v.8h` instructions; `llama-server` (a 6 KB launcher),
`libllama.so` and `libggml-base.so` have 0 in both builds. New `libggml-cpu.so` SHA-256
`a4ef0d97…a8a0dd`. `checks/sysinfo.log`: `system_info` gains `FP16_VA = 1 | DOTPROD = 1`.

## The three behaviour checks

| Check | Files | Result |
|---|---|---|
| a. s1o full ladder, both orders, 4 threads (`../strategic_selector/ladder/` cases) | `checks/ladder_dp.py`, `checks/ladder_new/`, `checks/ladder_new_L3/`, `checks/ladder_dponly*`, `checks/compare_3a.py` | 118/120 decisions identical to the old b1609 main block; the 2 changes (L3_01 reversed, L3_12 written) both went from wrong to correct; max probability difference 0.40; new build deterministic (L3 rerun 24/24, diff 0.0); the dotprod-only build flips the same two, so dotprod, not fp16, is the cause. Median decision 6650 → 2478 ms |
| b. voice parsing with `main.py`'s own `parse_command` and `PARSE_SYS`, server via `server_manager.start_setup('setup_q4')` (then Q4_K_M) | `checks/voice_gen.py`, `checks/voice_gen.json`, `checks/voice_warm.py`, `checks/voice_old1.py`, `checks/smoke.*` | 15/15 identical parsed actions and identical raw text; cold first parse 13.4–18.9 s against the 20 s timeout (the old build exceeded it) |
| c. generation, 200 tokens × 6 rounds per build, alternated old/new | `checks/voice_gen.py` (same script as b), `checks/voice_gen.stdout.txt` (last two lines) | median 6.46 → 8.52 tok/s (+32%); prompt eval 10.0 → 31.6 tok/s |

The ladder cases used in check a are a development bench, not the final exam.

## Review status

AGY `gemini-3.1-pro-high`, `--sandbox`, Ponytail off: **APPROVE WITH NOTES** (`checks/review.stdout.json`).
It flagged the 20 s `parse_command` timeout as too close; the 2026-09-26 Coder final task raised it to 40 s.
Coder report: `reports/CODER_REPORT_llama_dotprod_rebuild.md`.

## Evidence gaps

- "Speed only, not behaviour" holds for voice parsing but not strictly for s1o (2/120 changed).
- Check c had no thermal gate (proot cannot read thermal zones).
- The build trees and binaries are not archived (only logs, hashes and instruction counts).
