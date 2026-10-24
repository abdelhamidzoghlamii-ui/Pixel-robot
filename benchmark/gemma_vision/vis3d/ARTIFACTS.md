# Artifacts — VIS-3D

Every archived original is listed below. Each was copied byte for byte from the phone (`cp -p`) and compared with
`cmp` and SHA-256 at archive time (task VIS-3C/3D-ARCHIVE, 2026-10-23); phone SHA-256 = archive SHA-256 for all 70.
`README.md`, `RUN_INDEX.md` and `ARTIFACTS.md` are documentation; `README.md` is listed under source files but left
out of the check block so later wording edits do not break it. `/termux-home` is the native Termux-home bind
(`/data/data/com.termux/files/home`). Recheck from this folder:
`sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

`runs/thermal.log` is the launcher's CPU-zone logger file; it spans all six consoles and is not assigned to one run.
The eight test-D frames are not copied: they are already tracked at
`benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/` and their SHA-256 are in `references.json` and in
`runs/20261023T163445Z_D/run.json` (`frames_sha256`); all eight matched at archive time.

## Archived originals (70)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `reports/TASK.md` | `/termux-home/review/vis3d/TASK.md` | 4997 | `c39ab130862e6679685b9dd3614670ef4f5caabb7195d91202a0ccf4ee33e42c` |
| `reports/b11716_source_excerpts.txt` | `/termux-home/review/vis3d/b11716_source_excerpts.txt` | 11605 | `89626eddfdf448ef77fde9d767d96d50152491b0b2f13f4523ab28c7e6ed8c46` |
| `reports/candidate.diff` | `/termux-home/review/vis3d/candidate.diff` | 82506 | `fc814784711ddc9c6c87242acde1805625be0eaa1a7eace0f73093d99ffbb2d7` |
| `reports/check_citations.py` | `/termux-home/review/vis3d/check_citations.py` | 1440 | `dd60e1d28f298326ea0984fd95f69ba3d831488041e8ca17bdb71b63a1a61998` |
| `reports/checks.txt` | `/termux-home/review/vis3d/checks.txt` | 8250 | `f60a868c580cb696157d249a617d03cd26543108310bfe766fae2ac01273d9e8` |
| `reports/diag/after_smoke.json` | `/termux-home/review/vis3d/diag/after_smoke.json` | 851 | `fa0d921b021e2f079963c63e0dcb51722884280c1192117d9bd4b27b7f50c96e` |
| `reports/diag/overlay.json` | `/termux-home/review/vis3d/diag/overlay.json` | 840 | `66234ce53bd42e4d6bfbf2bb5df4ebfa53d434638f56944b34c445d2b957272f` |
| `reports/diag/smoke.stderr` | `/termux-home/review/vis3d/diag/smoke.stderr` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/diag/smoke.stdout` | `/termux-home/review/vis3d/diag/smoke.stdout` | 268 | `f964b32325dc51d249cf9dca9590c94757a5622504e2cbfc9e7a78f1a47c1189` |
| `reports/frozen_candidate.sha256` | `/termux-home/review/vis3d/frozen_candidate.sha256` | 535 | `5e70b3b958bfdafeac6516b4601a17462eae63765c8adad66821f1b297819b3d` |
| `reports/identity.txt` | `/termux-home/review/vis3d/identity.txt` | 8371 | `e37d50b74582761bd4c49bfe8f7da24ec2afdc5c410a844feeba29b106304e08` |
| `reports/review_request.txt` | `/termux-home/review/vis3d/review_request.txt` | 4614 | `77f15aa48a9356e13360b31e67f37f6cf908d22ddf7531635084f6918fad11b7` |
| `reports/review_request_r2.txt` | `/termux-home/review/vis3d/review_request_r2.txt` | 4933 | `81edda3562f530a35fb02429b4042d4d7edb5f8870b0fcbded90b213a5ffebd2` |
| `reports/review_tool.py` | `/termux-home/review/vis3d/review_tool.py` | 3999 | `b07aec608c4d5dff1ddcfeb84fb092825b8fb3ee44520b1eb6ac888b102c3cf0` |
| `reports/round1_agy_end.txt` | `/termux-home/review/vis3d/round1_agy_end.txt` | 21 | `893410efc22f7547c650509671ab6295c369c13885444d637617bedcc3788146` |
| `reports/round1_agy_exit.txt` | `/termux-home/review/vis3d/round1_agy_exit.txt` | 11 | `f46aeb5b366a585aa2e4299e760064b68f2698d5226e8536a73858c4d3e7020f` |
| `reports/round1_agy_start.txt` | `/termux-home/review/vis3d/round1_agy_start.txt` | 21 | `b7e4e3e211d9be927cb8f84deae4ef6c3a681b751364d5b1c018f6d44d7a4de7` |
| `reports/round1_logname.txt` | `/termux-home/review/vis3d/round1_logname.txt` | 24 | `7b52b01339d7e31919b35ece5a3edf1fa9c5882ea9afcda756d19dd7d820224a` |
| `reports/round1_review.txt` | `/termux-home/review/vis3d/round1_review.txt` | 15038 | `fd08b63ed86a42724f2512c1325c90fdab4ead2e7cee223c37f8045e21ea1aa9` |
| `reports/round1_stderr.txt` | `/termux-home/review/vis3d/round1_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/round1_stdout.txt` | `/termux-home/review/vis3d/round1_stdout.txt` | 15789 | `2d90480a359a3946aa786c8511ff97a6225af25b993397b5c8b07cf9cb2ca416` |
| `reports/round2_agy_end.txt` | `/termux-home/review/vis3d/round2_agy_end.txt` | 21 | `33c3fb94a61b34f190468f36a4b46e519c8ebd12c465a04fd31293b14cc383df` |
| `reports/round2_agy_exit.txt` | `/termux-home/review/vis3d/round2_agy_exit.txt` | 11 | `f46aeb5b366a585aa2e4299e760064b68f2698d5226e8536a73858c4d3e7020f` |
| `reports/round2_agy_start.txt` | `/termux-home/review/vis3d/round2_agy_start.txt` | 21 | `17c32a0a753ab7556e8cd4249c4585a21dae21bd0cc0f4a07f7caacd1e9a96f5` |
| `reports/round2_logname.txt` | `/termux-home/review/vis3d/round2_logname.txt` | 24 | `23f49963973558d10e4adf558a5f097c01f61e0472afa8a177d75955bdfa2772` |
| `reports/round2_overlay.sha256` | `/termux-home/review/vis3d/round2_overlay.sha256` | 110 | `04d46710335e8fab88d87da34a74303c66f05f8b46bf822df12edb8c0f1ddf40` |
| `reports/round2_review.txt` | `/termux-home/review/vis3d/round2_review.txt` | 12988 | `48adf595b684ea53e7bab918b98ddeadcbc118af19b5ed161a6e03502b3b55e7` |
| `reports/round2_settings_after_call.json` | `/termux-home/review/vis3d/round2_settings_after_call.json` | 851 | `fa0d921b021e2f079963c63e0dcb51722884280c1192117d9bd4b27b7f50c96e` |
| `reports/round2_settings_restored.sha256` | `/termux-home/review/vis3d/round2_settings_restored.sha256` | 110 | `ec694a05e0dfd309740b09fb481ea3b22d819b277cf127a286bea962a185b1c7` |
| `reports/round2_stderr.txt` | `/termux-home/review/vis3d/round2_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/round2_stdout.txt` | `/termux-home/review/vis3d/round2_stdout.txt` | 13536 | `37ad432faaf3a44d351ac6f91a3f5d139d4c844e602812214fe1655599294db0` |
| `reports/settings.overlay.sha256` | `/termux-home/review/vis3d/settings.overlay.sha256` | 110 | `168789ac7102a4d7263ae1f74a1769d9fc8d3d1d033c84152b2a18032cba3118` |
| `reports/settings_after.sha256` | `/termux-home/review/vis3d/settings_after.sha256` | 110 | `ec694a05e0dfd309740b09fb481ea3b22d819b277cf127a286bea962a185b1c7` |
| `reports/settings_before.sha256` | `/termux-home/review/vis3d/settings_before.sha256` | 223 | `f7af3c54d5628dd80e1c1dd96cb86e937bcbd445816a7dab82d1b704c6f9effc` |
| `reports/settings_initial.sha256` | `/termux-home/review/vis3d/settings_initial.sha256` | 110 | `ec694a05e0dfd309740b09fb481ea3b22d819b277cf127a286bea962a185b1c7` |
| `reports/smoke/log_excerpt.txt` | `/termux-home/review/vis3d/smoke/log_excerpt.txt` | 1244 | `375d943327b83cfa83dc5e03e7cdc069d90ab067256b5b1bb38ceb2951655859` |
| `reports/smoke/smoke.stderr` | `/termux-home/review/vis3d/smoke/smoke.stderr` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/smoke/smoke.stdout` | `/termux-home/review/vis3d/smoke/smoke.stdout` | 267 | `b24432ca49a86e78a90557ae0862451fc912c4b5ca34b62c323d650d13da453e` |
| `runs/20261023T151754Z_A/prompts.jsonl` | `/termux-home/robot_runs/vis3d/20261023T151754Z_A/prompts.jsonl` | 42783 | `3f9c3451ca6febeb1c8c899c696e764c650b78aa0970255d3c3f0f778d837049` |
| `runs/20261023T151754Z_A/report.txt` | `/termux-home/robot_runs/vis3d/20261023T151754Z_A/report.txt` | 2072 | `033852ef0895e234da58c1ce70fe0755d553e51b387f290bc408837492c5e15f` |
| `runs/20261023T151754Z_A/result.json` | `/termux-home/robot_runs/vis3d/20261023T151754Z_A/result.json` | 12534 | `94a716b99135f105bc399f83d0614e783aa1eee8847d8f5aa5419fd3e13d342a` |
| `runs/20261023T151754Z_A/run.json` | `/termux-home/robot_runs/vis3d/20261023T151754Z_A/run.json` | 4519 | `70b4aac75636a905ccdbefb307fe199eff6df8001da8a277465c02aed3b133f6` |
| `runs/20261023T151754Z_A/server_block00_A.log` | `/termux-home/robot_runs/vis3d/20261023T151754Z_A/server_block00_A.log` | 13846 | `66228634eb91cc91d12fa6247b4f40c2d0d175c32a085fb9b610d7455216872c` |
| `runs/20261023T152217Z_B/prompts.jsonl` | `/termux-home/robot_runs/vis3d/20261023T152217Z_B/prompts.jsonl` | 42275 | `a9e06a2d7a4b9f1ac0b9279625251f989687564cdfd92bbe70cf3b0be309505d` |
| `runs/20261023T152217Z_B/report.txt` | `/termux-home/robot_runs/vis3d/20261023T152217Z_B/report.txt` | 1794 | `ce98d1e9972dde38d993150f7b7dc30adad013ef376a96d9c24294263a362fa4` |
| `runs/20261023T152217Z_B/result.json` | `/termux-home/robot_runs/vis3d/20261023T152217Z_B/result.json` | 10984 | `4241ccc38109969a2953c1d43b140593766b10504889393bbc8360139e7306f4` |
| `runs/20261023T152217Z_B/run.json` | `/termux-home/robot_runs/vis3d/20261023T152217Z_B/run.json` | 4679 | `45d45bbc5ba7e7b7d5fa9ae3f09bfd558e1b6cbcdb055d229e169386deac8cb7` |
| `runs/20261023T152217Z_B/server_block00_B.log` | `/termux-home/robot_runs/vis3d/20261023T152217Z_B/server_block00_B.log` | 13923 | `de6906a9d64e50d5744d9aa41c4a03d6224debfc5c4afcff4fd662bc8128557f` |
| `runs/20261023T161842Z_B/prompts.jsonl` | `/termux-home/robot_runs/vis3d/20261023T161842Z_B/prompts.jsonl` | 42320 | `a1ee25c1e0ca02fb0817a4841cf5d0c803c87986e2058f187f81bae6cbcc2500` |
| `runs/20261023T161842Z_B/report.txt` | `/termux-home/robot_runs/vis3d/20261023T161842Z_B/report.txt` | 1794 | `47fb36f4307c923dca3a46a909e336d7f41b5ada53dd66a2473ba7a160fce7b2` |
| `runs/20261023T161842Z_B/result.json` | `/termux-home/robot_runs/vis3d/20261023T161842Z_B/result.json` | 11762 | `829f303bd99a245cd031f58126c0ffaa4a0b8ade8ccfe70e2a9f5ad71b6bc3dd` |
| `runs/20261023T161842Z_B/run.json` | `/termux-home/robot_runs/vis3d/20261023T161842Z_B/run.json` | 4679 | `b218313c45af13faf72510d2d41b23c39e75f810aa9fda558becbc667e02ba3b` |
| `runs/20261023T161842Z_B/server_block00_B.log` | `/termux-home/robot_runs/vis3d/20261023T161842Z_B/server_block00_B.log` | 14136 | `ab6c443ab8d2bb0cdd07d9f431076ef83b06f413490ccccc46b36aeafbb39904` |
| `runs/20261023T162349Z_C/prompts.jsonl` | `/termux-home/robot_runs/vis3d/20261023T162349Z_C/prompts.jsonl` | 43643 | `0ee79a512646777139295309586fcd58e25f5e6fc96c57f1ee06f191e91040b4` |
| `runs/20261023T162349Z_C/report.txt` | `/termux-home/robot_runs/vis3d/20261023T162349Z_C/report.txt` | 2141 | `6b7e604b7125a9e2f180057aba371dc0545bd7aec3e0bd60c80e2b5b44b821ce` |
| `runs/20261023T162349Z_C/result.json` | `/termux-home/robot_runs/vis3d/20261023T162349Z_C/result.json` | 11763 | `4f1704f7afbda79c374a48f8fe503aa31df8a477f64eab4c2d1ed4ad69801d3f` |
| `runs/20261023T162349Z_C/run.json` | `/termux-home/robot_runs/vis3d/20261023T162349Z_C/run.json` | 4839 | `c481438675569302a98f8677750b558e565be056895f7d7b7e1f245319b7843b` |
| `runs/20261023T162349Z_C/server_block00_C.log` | `/termux-home/robot_runs/vis3d/20261023T162349Z_C/server_block00_C.log` | 17576 | `2198ed44a19454db32e0c54765fd765113ec2496a43bd798f7bb1f0a39e16ee4` |
| `runs/20261023T163445Z_D/prompts.jsonl` | `/termux-home/robot_runs/vis3d/20261023T163445Z_D/prompts.jsonl` | 34853 | `ecf6b8d634c9d4028b6e815acc77f0e82f4335f6a570d5af0d60598ca9d8dd93` |
| `runs/20261023T163445Z_D/report.txt` | `/termux-home/robot_runs/vis3d/20261023T163445Z_D/report.txt` | 9689 | `aa8fdb96d8592388be14f5be43bfe9bb3e01db1e92d4fc4a03893c712f00c275` |
| `runs/20261023T163445Z_D/result.json` | `/termux-home/robot_runs/vis3d/20261023T163445Z_D/result.json` | 10428 | `c6862135af352f6449286124bd1325bef9325cc20b8eb3b873ba8f4e853afc58` |
| `runs/20261023T163445Z_D/run.json` | `/termux-home/robot_runs/vis3d/20261023T163445Z_D/run.json` | 6638 | `bd18bc821078ea8ed20342122f70870abc687db57b6a0e4a3c1870fcf7d64f72` |
| `runs/20261023T163445Z_D/server_block00_D.log` | `/termux-home/robot_runs/vis3d/20261023T163445Z_D/server_block00_D.log` | 55070 | `37be1ce294633bf6203bf25bda7a20b9533cff6f4110f97df63d6f5b45f42b58` |
| `runs/oneshot_console_20261023T150730Z.log` | `/termux-home/robot_runs/vis3d/oneshot_console_20261023T150730Z.log` | 715 | `5f2d052facdca4f5d3366dac17b813c8ff14ef6dadec994bf700665e76d4239b` |
| `runs/oneshot_console_20261023T151746Z.log` | `/termux-home/robot_runs/vis3d/oneshot_console_20261023T151746Z.log` | 2937 | `800f481f6bff8723b299dadc8631f161dc19cd033998a3cc2f9206c37c0c40cc` |
| `runs/oneshot_console_20261023T152209Z.log` | `/termux-home/robot_runs/vis3d/oneshot_console_20261023T152209Z.log` | 2659 | `d1285f2cb2490027149ee4dc7a2408d80564a950a0df021f0ac6cc18702e3f94` |
| `runs/oneshot_console_20261023T161834Z.log` | `/termux-home/robot_runs/vis3d/oneshot_console_20261023T161834Z.log` | 2657 | `13a5dc251d0b8c9320513a16aa4608b57faf26ab68e22fee4882fc7de90ec0cc` |
| `runs/oneshot_console_20261023T162342Z.log` | `/termux-home/robot_runs/vis3d/oneshot_console_20261023T162342Z.log` | 3006 | `367f4aebe9379ba1735e25649886f06caec1e832af0f742209ddb001325cde04` |
| `runs/oneshot_console_20261023T163437Z.log` | `/termux-home/robot_runs/vis3d/oneshot_console_20261023T163437Z.log` | 10554 | `ebade12f4fa23e881d2a4ec0526b557ccdf2d4d2c342f35d96596f75286bbc2e` |
| `runs/thermal.log` | `/termux-home/robot_runs/vis3d/thermal.log` | 5774 | `f6a48010cb197e55246b20db434d80db98859e45a8e99cedeef9c0f67773a3d0` |

## Regenerated reports (2; written by this archival task, no phone original)

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `reports/REGENERATED_20261023T190341Z_combined_A_B1_C.txt` | 3302 | `b67229822446c06bba7496c6fd2d2f207211dfbb72f9acfa8422ca38597db7ea` |
| `reports/REGENERATED_20261023T190341Z_combined_A_B2_C.txt` | 3352 | `c435427e819694d9c9d55534d741ac6192a1cb17bc9a3652b0785dc6ac13288d` |

## Source files (5)

| File | Bytes | SHA-256 now | Reviewed frozen SHA-256 |
|---|---:|---|---|
| `README.md` | 14662 | `539e1ed3f17c0f7551ecdb6689fbd74a167e3b183a0670854e0198c0a1759e47` | `35b84822ff208a7b2d724cc7d5c6456ef9c887ef4a0bcd182102c0a299d2d963` (only the results section and first line were changed in the archival task) |
| `oneshot.sh` | 6834 | `5783ab3fb285b69b7b443af32380c1f1172a60dd926567d4432769af79ffebc5` | same |
| `references.json` | 7073 | `7d3d5b775706b28f76431fef613e7d2e6beb2d862821f74a391397346a8e09c4` | same |
| `test_vis3d.py` | 25609 | `50dd8f0c3c8f534d4a5bd12d81133d2e4ab19a8360c99f6e1724d01a6cc0437f` | same |
| `vis3d.py` | 29164 | `cd4995b21d28ffb91da43806557871fe9689333a93ef43768131d4dfc2bbc690` | same |

## Not archived

Weights and build (hashes from the runs' `run.json`, identical in all five; sizes by `stat` on the phone):

| File (phone path) | Bytes | SHA-256 |
|---|---:|---|
| `/data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf` | 2841481184 | `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52` |
| `/data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf` | 97817664 | `9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919` |
| `/data/data/com.termux/files/home/models/vis0/mmproj-gemma-4-E2B-it-Q8_0.gguf` | 557368064 | `9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292` |
| `~/llama.cpp-vis/build/bin/llama-server` (source commit `797de63b2fa7396dcc8caca76834c699ac84e1c3`) | – | `0674f5c28aaf23068220fa4290c7a4199aa590127724d2b766bdbb961b768fdf` |

The seven b11716 libraries and the CMake flags are in `build_sha256` / `build_info` of each `run.json`.

Review evidence left on the phone (`/termux-home/review/vis3d/`):

| Item | Why not archived | SHA-256 / size |
|---|---|---|
| `round{1,2}_copy/` | full isolated repository copies given to the reviewer (about 173 MB each); reconstructible from base `685a44b` plus the frozen candidate | – |
| `round1_pre.json` / `round1_post.json` | 1,037,528 B identity snapshots; DIFFER (round 1 INVALID) | `8099bf6ad5e4fc054076ae97046761d7e47c6ee714e1fcf09af46db9f11ea4ee` / `94ea4f28872997f3f0671527bc6bae6b118d89316d9bc701841d971d3e954869` |
| `round2_pre.json` = `round2_post.json` | as above; identical | `94ea4f28872997f3f0671527bc6bae6b118d89316d9bc701841d971d3e954869` |
| `settings.backup.json` | shared AGY settings backup (agent configuration, not benchmark evidence) | `6e5273bf3864677e135b104758ba5724c8c45f8a17d3e55d9af19e66ee4b7421` |
| `/root/.gemini/antigravity-cli/log/cli-20261023_140749.log` (r1), `_141730.log` (r2) | AGY session logs (model labels quoted in RUN_INDEX) | `548249a2…`, `f43f9118…` |
| `/termux-home/robot_runs/vis3d/.stop_thermal` | 0-byte launcher control file | `e3b0c442…` (empty) |

## sha256sum format

```sha256sums
c39ab130862e6679685b9dd3614670ef4f5caabb7195d91202a0ccf4ee33e42c  reports/TASK.md
89626eddfdf448ef77fde9d767d96d50152491b0b2f13f4523ab28c7e6ed8c46  reports/b11716_source_excerpts.txt
fc814784711ddc9c6c87242acde1805625be0eaa1a7eace0f73093d99ffbb2d7  reports/candidate.diff
dd60e1d28f298326ea0984fd95f69ba3d831488041e8ca17bdb71b63a1a61998  reports/check_citations.py
f60a868c580cb696157d249a617d03cd26543108310bfe766fae2ac01273d9e8  reports/checks.txt
fa0d921b021e2f079963c63e0dcb51722884280c1192117d9bd4b27b7f50c96e  reports/diag/after_smoke.json
66234ce53bd42e4d6bfbf2bb5df4ebfa53d434638f56944b34c445d2b957272f  reports/diag/overlay.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/diag/smoke.stderr
f964b32325dc51d249cf9dca9590c94757a5622504e2cbfc9e7a78f1a47c1189  reports/diag/smoke.stdout
5e70b3b958bfdafeac6516b4601a17462eae63765c8adad66821f1b297819b3d  reports/frozen_candidate.sha256
e37d50b74582761bd4c49bfe8f7da24ec2afdc5c410a844feeba29b106304e08  reports/identity.txt
77f15aa48a9356e13360b31e67f37f6cf908d22ddf7531635084f6918fad11b7  reports/review_request.txt
81edda3562f530a35fb02429b4042d4d7edb5f8870b0fcbded90b213a5ffebd2  reports/review_request_r2.txt
b07aec608c4d5dff1ddcfeb84fb092825b8fb3ee44520b1eb6ac888b102c3cf0  reports/review_tool.py
893410efc22f7547c650509671ab6295c369c13885444d637617bedcc3788146  reports/round1_agy_end.txt
f46aeb5b366a585aa2e4299e760064b68f2698d5226e8536a73858c4d3e7020f  reports/round1_agy_exit.txt
b7e4e3e211d9be927cb8f84deae4ef6c3a681b751364d5b1c018f6d44d7a4de7  reports/round1_agy_start.txt
7b52b01339d7e31919b35ece5a3edf1fa9c5882ea9afcda756d19dd7d820224a  reports/round1_logname.txt
fd08b63ed86a42724f2512c1325c90fdab4ead2e7cee223c37f8045e21ea1aa9  reports/round1_review.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/round1_stderr.txt
2d90480a359a3946aa786c8511ff97a6225af25b993397b5c8b07cf9cb2ca416  reports/round1_stdout.txt
33c3fb94a61b34f190468f36a4b46e519c8ebd12c465a04fd31293b14cc383df  reports/round2_agy_end.txt
f46aeb5b366a585aa2e4299e760064b68f2698d5226e8536a73858c4d3e7020f  reports/round2_agy_exit.txt
17c32a0a753ab7556e8cd4249c4585a21dae21bd0cc0f4a07f7caacd1e9a96f5  reports/round2_agy_start.txt
23f49963973558d10e4adf558a5f097c01f61e0472afa8a177d75955bdfa2772  reports/round2_logname.txt
04d46710335e8fab88d87da34a74303c66f05f8b46bf822df12edb8c0f1ddf40  reports/round2_overlay.sha256
48adf595b684ea53e7bab918b98ddeadcbc118af19b5ed161a6e03502b3b55e7  reports/round2_review.txt
fa0d921b021e2f079963c63e0dcb51722884280c1192117d9bd4b27b7f50c96e  reports/round2_settings_after_call.json
ec694a05e0dfd309740b09fb481ea3b22d819b277cf127a286bea962a185b1c7  reports/round2_settings_restored.sha256
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/round2_stderr.txt
37ad432faaf3a44d351ac6f91a3f5d139d4c844e602812214fe1655599294db0  reports/round2_stdout.txt
168789ac7102a4d7263ae1f74a1769d9fc8d3d1d033c84152b2a18032cba3118  reports/settings.overlay.sha256
ec694a05e0dfd309740b09fb481ea3b22d819b277cf127a286bea962a185b1c7  reports/settings_after.sha256
f7af3c54d5628dd80e1c1dd96cb86e937bcbd445816a7dab82d1b704c6f9effc  reports/settings_before.sha256
ec694a05e0dfd309740b09fb481ea3b22d819b277cf127a286bea962a185b1c7  reports/settings_initial.sha256
375d943327b83cfa83dc5e03e7cdc069d90ab067256b5b1bb38ceb2951655859  reports/smoke/log_excerpt.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/smoke/smoke.stderr
b24432ca49a86e78a90557ae0862451fc912c4b5ca34b62c323d650d13da453e  reports/smoke/smoke.stdout
3f9c3451ca6febeb1c8c899c696e764c650b78aa0970255d3c3f0f778d837049  runs/20261023T151754Z_A/prompts.jsonl
033852ef0895e234da58c1ce70fe0755d553e51b387f290bc408837492c5e15f  runs/20261023T151754Z_A/report.txt
94a716b99135f105bc399f83d0614e783aa1eee8847d8f5aa5419fd3e13d342a  runs/20261023T151754Z_A/result.json
70b4aac75636a905ccdbefb307fe199eff6df8001da8a277465c02aed3b133f6  runs/20261023T151754Z_A/run.json
66228634eb91cc91d12fa6247b4f40c2d0d175c32a085fb9b610d7455216872c  runs/20261023T151754Z_A/server_block00_A.log
a9e06a2d7a4b9f1ac0b9279625251f989687564cdfd92bbe70cf3b0be309505d  runs/20261023T152217Z_B/prompts.jsonl
ce98d1e9972dde38d993150f7b7dc30adad013ef376a96d9c24294263a362fa4  runs/20261023T152217Z_B/report.txt
4241ccc38109969a2953c1d43b140593766b10504889393bbc8360139e7306f4  runs/20261023T152217Z_B/result.json
45d45bbc5ba7e7b7d5fa9ae3f09bfd558e1b6cbcdb055d229e169386deac8cb7  runs/20261023T152217Z_B/run.json
de6906a9d64e50d5744d9aa41c4a03d6224debfc5c4afcff4fd662bc8128557f  runs/20261023T152217Z_B/server_block00_B.log
a1ee25c1e0ca02fb0817a4841cf5d0c803c87986e2058f187f81bae6cbcc2500  runs/20261023T161842Z_B/prompts.jsonl
47fb36f4307c923dca3a46a909e336d7f41b5ada53dd66a2473ba7a160fce7b2  runs/20261023T161842Z_B/report.txt
829f303bd99a245cd031f58126c0ffaa4a0b8ade8ccfe70e2a9f5ad71b6bc3dd  runs/20261023T161842Z_B/result.json
b218313c45af13faf72510d2d41b23c39e75f810aa9fda558becbc667e02ba3b  runs/20261023T161842Z_B/run.json
ab6c443ab8d2bb0cdd07d9f431076ef83b06f413490ccccc46b36aeafbb39904  runs/20261023T161842Z_B/server_block00_B.log
0ee79a512646777139295309586fcd58e25f5e6fc96c57f1ee06f191e91040b4  runs/20261023T162349Z_C/prompts.jsonl
6b7e604b7125a9e2f180057aba371dc0545bd7aec3e0bd60c80e2b5b44b821ce  runs/20261023T162349Z_C/report.txt
4f1704f7afbda79c374a48f8fe503aa31df8a477f64eab4c2d1ed4ad69801d3f  runs/20261023T162349Z_C/result.json
c481438675569302a98f8677750b558e565be056895f7d7b7e1f245319b7843b  runs/20261023T162349Z_C/run.json
2198ed44a19454db32e0c54765fd765113ec2496a43bd798f7bb1f0a39e16ee4  runs/20261023T162349Z_C/server_block00_C.log
ecf6b8d634c9d4028b6e815acc77f0e82f4335f6a570d5af0d60598ca9d8dd93  runs/20261023T163445Z_D/prompts.jsonl
aa8fdb96d8592388be14f5be43bfe9bb3e01db1e92d4fc4a03893c712f00c275  runs/20261023T163445Z_D/report.txt
c6862135af352f6449286124bd1325bef9325cc20b8eb3b873ba8f4e853afc58  runs/20261023T163445Z_D/result.json
bd18bc821078ea8ed20342122f70870abc687db57b6a0e4a3c1870fcf7d64f72  runs/20261023T163445Z_D/run.json
37be1ce294633bf6203bf25bda7a20b9533cff6f4110f97df63d6f5b45f42b58  runs/20261023T163445Z_D/server_block00_D.log
5f2d052facdca4f5d3366dac17b813c8ff14ef6dadec994bf700665e76d4239b  runs/oneshot_console_20261023T150730Z.log
800f481f6bff8723b299dadc8631f161dc19cd033998a3cc2f9206c37c0c40cc  runs/oneshot_console_20261023T151746Z.log
d1285f2cb2490027149ee4dc7a2408d80564a950a0df021f0ac6cc18702e3f94  runs/oneshot_console_20261023T152209Z.log
13a5dc251d0b8c9320513a16aa4608b57faf26ab68e22fee4882fc7de90ec0cc  runs/oneshot_console_20261023T161834Z.log
367f4aebe9379ba1735e25649886f06caec1e832af0f742209ddb001325cde04  runs/oneshot_console_20261023T162342Z.log
ebade12f4fa23e881d2a4ec0526b557ccdf2d4d2c342f35d96596f75286bbc2e  runs/oneshot_console_20261023T163437Z.log
f6a48010cb197e55246b20db434d80db98859e45a8e99cedeef9c0f67773a3d0  runs/thermal.log
b67229822446c06bba7496c6fd2d2f207211dfbb72f9acfa8422ca38597db7ea  reports/REGENERATED_20261023T190341Z_combined_A_B1_C.txt
c435427e819694d9c9d55534d741ac6192a1cb17bc9a3652b0785dc6ac13288d  reports/REGENERATED_20261023T190341Z_combined_A_B2_C.txt
5783ab3fb285b69b7b443af32380c1f1172a60dd926567d4432769af79ffebc5  oneshot.sh
7d3d5b775706b28f76431fef613e7d2e6beb2d862821f74a391397346a8e09c4  references.json
50dd8f0c3c8f534d4a5bd12d81133d2e4ab19a8360c99f6e1724d01a6cc0437f  test_vis3d.py
cd4995b21d28ffb91da43806557871fe9689333a93ef43768131d4dfc2bbc690  vis3d.py
```
