# llama.cpp dotprod rebuild — run index

| Step | When (UTC) | Output | Status |
|---|---|---|---|
| configure + build, dotprod+fp16 | 2026-09-26 09:25–09:33 | `build/llama-dotprod-{configure,build}.log` | complete |
| configure + build, dotprod only (diagnostic) | 2026-09-26 09:58–10:07 | `build/llama-dotprod-nofp16-*.log` | complete; build directory removed afterwards |
| check a: s1o ladder, new build and dotprod-only | 2026-09-26 | `checks/ladder_new*`, `checks/ladder_dponly*` | complete |
| check b: voice parsing | 2026-09-26 | `checks/voice_*`, `checks/smoke.*` | complete |
| check c: generation speed | 2026-09-26 | `checks/voice_gen.*` | complete, no thermal gate |
| instruction counts, CMake diff | 2026-09-26T22:12Z (archive time) | `build/sdot_counts.txt`, `build/cmake_cache_diff.txt` | recomputed |
