# Artifacts — chat_mtp runs

Every archived file is listed below. Each was copied byte for byte from the phone and compared with `cmp` at archive
time (task MTP-ARCHIVE, 2026-10-15); phone SHA-256 = archive SHA-256. `ARTIFACTS.md` and `RUN_INDEX.md` are
documentation written for this archive and are not listed. `/termux-home` is the native Termux-home bind
(`/data/data/com.termux/files/home`). Recheck from this folder:
`sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

`thermal.log` is the launcher's CPU-zone logger file; it spans both sessions (2026-10-14 22:06:56Z – 22:53:04Z) and
is not assigned to one run.

## Archived files (15)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `chat_mtp_20261014T221203Z_smoke/blocks.jsonl` | `/termux-home/robot_runs/chat_mtp_20261014T221203Z_smoke/blocks.jsonl` | 6665 | `a4be461514a657691a5a93e8777accc50d277b06edf4d468e58a21d8581c7be7` |
| `chat_mtp_20261014T221203Z_smoke/prompts.jsonl` | `/termux-home/robot_runs/chat_mtp_20261014T221203Z_smoke/prompts.jsonl` | 2499 | `58815886b10b446b3ee65dc7cd2368ce765e70c543a0ac39f9f5ca581c490707` |
| `chat_mtp_20261014T221203Z_smoke/report.txt` | `/termux-home/robot_runs/chat_mtp_20261014T221203Z_smoke/report.txt` | 1092 | `764c4cfb5401bcd8bd9e7453df4b403da5d14aa7f3f5e575373c15b5b6440467` |
| `chat_mtp_20261014T221203Z_smoke/run.json` | `/termux-home/robot_runs/chat_mtp_20261014T221203Z_smoke/run.json` | 3046 | `4c94f58cdbced02497b0af73442917b7ab27f5304488dd322f8a8a2626a75a00` |
| `chat_mtp_20261014T221203Z_smoke/server_block00_OFF.log` | `/termux-home/robot_runs/chat_mtp_20261014T221203Z_smoke/server_block00_OFF.log` | 2483 | `c1454324e79aed33f2da41f7b8d040aa63e02fd4c6155e1bb1256022718a2bb0` |
| `chat_mtp_20261014T221203Z_smoke/server_block01_ON.log` | `/termux-home/robot_runs/chat_mtp_20261014T221203Z_smoke/server_block01_ON.log` | 3928 | `8cff226d9b2307ea8e97852960e1f18e40c2f62d8f417ecb08ab19b132496279` |
| `chat_mtp_20261014T224541Z/blocks.jsonl` | `/termux-home/robot_runs/chat_mtp_20261014T224541Z/blocks.jsonl` | 18324 | `32d9c7c4b749d94e016a8c9dc24175c7edfe136a37e024bb3784093fe4d34e22` |
| `chat_mtp_20261014T224541Z/prompts.jsonl` | `/termux-home/robot_runs/chat_mtp_20261014T224541Z/prompts.jsonl` | 22198 | `6ba22a057aeb55a706e93fb1f30b84b20057dc19501401da58697277d46badf3` |
| `chat_mtp_20261014T224541Z/report.txt` | `/termux-home/robot_runs/chat_mtp_20261014T224541Z/report.txt` | 1157 | `72404b30d47192ba1543179db917d9a2627badbeca008e1684d96f6df6507ada` |
| `chat_mtp_20261014T224541Z/run.json` | `/termux-home/robot_runs/chat_mtp_20261014T224541Z/run.json` | 4462 | `8cc59ac3bb8ab009d4e69e24cf2783ff3fa9a50772dd985acbe2e6d7834cb012` |
| `chat_mtp_20261014T224541Z/server_block00_OFF.log` | `/termux-home/robot_runs/chat_mtp_20261014T224541Z/server_block00_OFF.log` | 12719 | `e87ac112855594679659696ababa8e0be5ff22b2cad6dc42407de0c5e8bf1da3` |
| `chat_mtp_20261014T224541Z/server_block01_ON.log` | `/termux-home/robot_runs/chat_mtp_20261014T224541Z/server_block01_ON.log` | 15423 | `521696614b37adca3ddc7ef605f0e7e5fbc9e03246b538d83bc4083dbe3a1a0b` |
| `oneshot_console_20261014T220655Z.log` | `/termux-home/robot_runs/chat_mtp/oneshot_console_20261014T220655Z.log` | 2414 | `f45ffcfe99782930a311b580b71143e92c48ee25ae232fa4bdd818ed9eb37bd2` |
| `oneshot_console_20261014T224033Z.log` | `/termux-home/robot_runs/chat_mtp/oneshot_console_20261014T224033Z.log` | 4307 | `96cad4cf05226984ec97bc9f8040d4bd3d19e27363526d66df1fe897ea0bb901` |
| `thermal.log` | `/termux-home/robot_runs/chat_mtp/thermal.log` | 19150 | `b95db25aec1f8330e760130bce073aef9e730c4eab38d30c5578d65f436b50ad` |

## Not archived: model files

Paths and SHA-256 from `model_sha256` in both runs' `run.json` (identical in both; the runner refuses to start on any
other hash); sizes from [`../../conv_mtp/ARTIFACTS.md`](../../conv_mtp/ARTIFACTS.md).

| File (phone path) | Bytes | SHA-256 |
|---|---:|---|
| `/data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf` | 2841481184 | `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52` |
| `/data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf` | 97817664 | `9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919` |

## sha256sum format

```sha256sums
a4be461514a657691a5a93e8777accc50d277b06edf4d468e58a21d8581c7be7  chat_mtp_20261014T221203Z_smoke/blocks.jsonl
58815886b10b446b3ee65dc7cd2368ce765e70c543a0ac39f9f5ca581c490707  chat_mtp_20261014T221203Z_smoke/prompts.jsonl
764c4cfb5401bcd8bd9e7453df4b403da5d14aa7f3f5e575373c15b5b6440467  chat_mtp_20261014T221203Z_smoke/report.txt
4c94f58cdbced02497b0af73442917b7ab27f5304488dd322f8a8a2626a75a00  chat_mtp_20261014T221203Z_smoke/run.json
c1454324e79aed33f2da41f7b8d040aa63e02fd4c6155e1bb1256022718a2bb0  chat_mtp_20261014T221203Z_smoke/server_block00_OFF.log
8cff226d9b2307ea8e97852960e1f18e40c2f62d8f417ecb08ab19b132496279  chat_mtp_20261014T221203Z_smoke/server_block01_ON.log
32d9c7c4b749d94e016a8c9dc24175c7edfe136a37e024bb3784093fe4d34e22  chat_mtp_20261014T224541Z/blocks.jsonl
6ba22a057aeb55a706e93fb1f30b84b20057dc19501401da58697277d46badf3  chat_mtp_20261014T224541Z/prompts.jsonl
72404b30d47192ba1543179db917d9a2627badbeca008e1684d96f6df6507ada  chat_mtp_20261014T224541Z/report.txt
8cc59ac3bb8ab009d4e69e24cf2783ff3fa9a50772dd985acbe2e6d7834cb012  chat_mtp_20261014T224541Z/run.json
e87ac112855594679659696ababa8e0be5ff22b2cad6dc42407de0c5e8bf1da3  chat_mtp_20261014T224541Z/server_block00_OFF.log
521696614b37adca3ddc7ef605f0e7e5fbc9e03246b538d83bc4083dbe3a1a0b  chat_mtp_20261014T224541Z/server_block01_ON.log
f45ffcfe99782930a311b580b71143e92c48ee25ae232fa4bdd818ed9eb37bd2  oneshot_console_20261014T220655Z.log
96cad4cf05226984ec97bc9f8040d4bd3d19e27363526d66df1fe897ea0bb901  oneshot_console_20261014T224033Z.log
b95db25aec1f8330e760130bce073aef9e730c4eab38d30c5578d65f436b50ad  thermal.log
```
