# Artifacts

Every archived file below with its SHA-256. The six source files were moved unchanged from the repository root on 2026-09-27; their SHA-256 values equal those recorded before the move. `README.md` and `RUN_INDEX.md` were written for this archive. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 2837 | `f25ea963226865e1c68861bfa0068573c2d97d2ddcee592a58674fefabf921ab` |
| `RUN_INDEX.md` | 1062 | `3caee5f6238ece30b30eea2b0f00854d8b39c5274012b7efaa5169cfa732e218` |
| `qwen35_vision_probe.py` | 13008 | `a52655a15ac5dbf370249d103b8dd53a90b627eb9d17816db29c0a8477cabc27` |
| `qwen_vision_bench.py` | 19417 | `d9ba7661fc25d8d9a50b83ea84f20c0c0e2817d89439a4ff1689d38742732f28` |
| `qwen_vision_bench.sha256` | 367 | `14bb50dd614612cb8dd60d07f55c9fd4d99c10e0eb33f3e83f2d5839ce1667f2` |
| `qwen_vision_bench_report.md` | 10689 | `cf5075d737d0cd9325b4820cb7d6ec1f8e6fa0c92f897ac02d5b10bcc6d1eb9a` |
| `qwen_vision_bench_review.md` | 3557 | `ff44b37f655cffcf65ea5a38072e668da1a1cd9c0129cf0e2e6cef21fb8833b5` |
| `qwen_vision_bench_test.py` | 5827 | `07608f23d888320bba4837a073b7756c5c4181e71253a7d4579a7b255883beed` |

## sha256sum format

```sha256sums
f25ea963226865e1c68861bfa0068573c2d97d2ddcee592a58674fefabf921ab  README.md
3caee5f6238ece30b30eea2b0f00854d8b39c5274012b7efaa5169cfa732e218  RUN_INDEX.md
a52655a15ac5dbf370249d103b8dd53a90b627eb9d17816db29c0a8477cabc27  qwen35_vision_probe.py
d9ba7661fc25d8d9a50b83ea84f20c0c0e2817d89439a4ff1689d38742732f28  qwen_vision_bench.py
14bb50dd614612cb8dd60d07f55c9fd4d99c10e0eb33f3e83f2d5839ce1667f2  qwen_vision_bench.sha256
cf5075d737d0cd9325b4820cb7d6ec1f8e6fa0c92f897ac02d5b10bcc6d1eb9a  qwen_vision_bench_report.md
ff44b37f655cffcf65ea5a38072e668da1a1cd9c0129cf0e2e6cef21fb8833b5  qwen_vision_bench_review.md
07608f23d888320bba4837a073b7756c5c4181e71253a7d4579a7b255883beed  qwen_vision_bench_test.py
```
