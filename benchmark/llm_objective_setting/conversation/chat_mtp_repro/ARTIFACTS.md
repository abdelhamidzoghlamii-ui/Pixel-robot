# Artifacts — chat_mtp_repro (VIS-3C)

Every archived original is listed below. Each was copied byte for byte from the phone (`cp -p`) and compared with
`cmp` and SHA-256 at archive time (task VIS-3C/3D-ARCHIVE, 2026-10-23); phone SHA-256 = archive SHA-256 for all 42.
`README.md`, `RUN_INDEX.md` and `ARTIFACTS.md` are documentation; `README.md` is listed under source files but left
out of the check block so later wording edits do not break it. `/termux-home` is the native Termux-home bind
(`/data/data/com.termux/files/home`). Recheck from this folder:
`sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

`runs/thermal.log` is the launcher's CPU-zone logger file; it spans all three consoles and is not assigned to one run.
`reports/round1_TASK.md` and `reports/round2_TASK.md` come from `_review_context/TASK.md` inside the round 1 and
round 2 review copies; `reports/review_context/*` from round 3's `_review_context/`.

## Archived originals (42)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `reports/TASK.md` | `/termux-home/review/vis3c/TASK.md` | 6422 | `6efc55bb315b18186f74e1f54ef735c50cfacd5bd2e952cac2619cbde975f538` |
| `reports/candidate.diff` | `/termux-home/review/vis3c/candidate.diff` | 46324 | `747abe328da51d4df63dc2cd8376a33948ef0c98a1780c786b687008df15259f` |
| `reports/checks.txt` | `/termux-home/review/vis3c/checks.txt` | 1488 | `9ad9981e6d12b0d2ef4b0fcd394d6ec67bab5bb6803604c7f035f66c0a95bf72` |
| `reports/current_hashes.txt` | `/termux-home/review/vis3c/current_hashes.txt` | 1125 | `5a6665e3aea564fd25a7877be8bb1fcd5ff0419fd3bf0a0ec713f9e29e906b91` |
| `reports/frozen_candidate.sha256` | `/termux-home/review/vis3c/frozen_candidate.sha256` | 548 | `36f7c84fb997c4e027c017d153226e5f210fb9fc007ca0fe8a7f1ed768c9cc64` |
| `reports/review_context/VIS3B_MTP_INVESTIGATION.md` | `/termux-home/review/vis3c/round3_copy/_review_context/VIS3B_MTP_INVESTIGATION.md` | 13451 | `70ad2f87f5cfbf7058a296f781b680cc203a6fffbf52bd881cf9b10ef000f5a0` |
| `reports/review_context/VIS3B_MTP_REVIEW_ROUND2.json` | `/termux-home/review/vis3c/round3_copy/_review_context/VIS3B_MTP_REVIEW_ROUND2.json` | 4727 | `782adf9c93f46d09d47474238c11543de60337c5f8c04398f3673d65b7a6db79` |
| `reports/review_tool.py` | `/termux-home/review/vis3c/review_tool.py` | 3960 | `ab2fe5b1ae793dc0d6513ef15efc5f70f3be44856dbecbf430c9f4a8185e2f00` |
| `reports/round1_TASK.md` | `/termux-home/review/vis3c/round1_copy/_review_context/TASK.md` | 6054 | `9d497750c6bf2d5ce11aec2fa83be4a39dd8d9012152925804e94f4dad3dc319` |
| `reports/round1_review.txt` | `/termux-home/review/vis3c/round1_review.txt` | 5994 | `b6e8f64a9aea62e786d3f262eb138cbde4d4bddba86e76290eb5191ba9666016` |
| `reports/round1_stderr.txt` | `/termux-home/review/vis3c/round1_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/round1_stdout.txt` | `/termux-home/review/vis3c/round1_stdout.txt` | 6404 | `be510f75a7baf80ab7c27944eacad04ec03a61c7ec6ceeeebf06d92f2fc3f7c8` |
| `reports/round2_TASK.md` | `/termux-home/review/vis3c/round2_copy/_review_context/TASK.md` | 6364 | `d1944dea216138b1ab977ead28d7aabb7f454a8492f3a7dd01d57e3a27f99894` |
| `reports/round2_review.txt` | `/termux-home/review/vis3c/round2_review.txt` | 11478 | `caa9ecc9ad5d1ea5f4c3fe6499afbbe10b501e86b48458134883aa26d08aa18e` |
| `reports/round2_stderr.txt` | `/termux-home/review/vis3c/round2_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/round2_stdout.txt` | `/termux-home/review/vis3c/round2_stdout.txt` | 12244 | `b1c75769da90618d962c859e8c81d186620810c5c140d0628d02647f2e9efa58` |
| `reports/round3_review.txt` | `/termux-home/review/vis3c/round3_review.txt` | 16610 | `b09fc94599e86c20124b12ef974800a90f5460c9f4a4af529c32b394de4ccb88` |
| `reports/round3_stderr.txt` | `/termux-home/review/vis3c/round3_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/round3_stdout.txt` | `/termux-home/review/vis3c/round3_stdout.txt` | 17250 | `2abf87eb3618e80644d6a40583755fdd4a7ab75398d0f0808eb7d12fb3e49fbb` |
| `reports/settings.overlay.sha256` | `/termux-home/review/vis3c/settings.overlay.sha256` | 110 | `8436055899a15ec8cc42f0f3678df46b8eebaddb2b94bc8c2fe56af9eeb5ac3e` |
| `reports/smoke2_stderr.txt` | `/termux-home/review/vis3c/smoke2_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/smoke2_stdout.txt` | `/termux-home/review/vis3c/smoke2_stdout.txt` | 269 | `0525c26052207ea2f1895106b99dd08baa1fb2ff277dedab02f01d4a3ef840dd` |
| `reports/smoke2b_stderr.txt` | `/termux-home/review/vis3c/smoke2b_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/smoke2b_stdout.txt` | `/termux-home/review/vis3c/smoke2b_stdout.txt` | 269 | `2f08018e3504c6951a21df257cb8f602bab5e92c33cfd5caba623e9d62257e29` |
| `reports/smoke3_stderr.txt` | `/termux-home/review/vis3c/smoke3_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/smoke3_stdout.txt` | `/termux-home/review/vis3c/smoke3_stdout.txt` | 275 | `14261d0fb1d1d9ae8408cf26eebfc3b7c310fdacfe56a40151f82358cb3b8730` |
| `reports/smoke_stderr.txt` | `/termux-home/review/vis3c/smoke_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/smoke_stdout.txt` | `/termux-home/review/vis3c/smoke_stdout.txt` | 268 | `c3846dbb437e4849088f47a3decad923dfb5e82b32e7ed4971f6e2d578642e4c` |
| `runs/20261023T124846Z_OFF/prompts.jsonl` | `/termux-home/robot_runs/chat_mtp_repro/20261023T124846Z_OFF/prompts.jsonl` | 10988 | `21b624765995dfb696d046fc05369bfeed5537ffa069df527c567652542dc728` |
| `runs/20261023T124846Z_OFF/report.txt` | `/termux-home/robot_runs/chat_mtp_repro/20261023T124846Z_OFF/report.txt` | 1737 | `e9f8f2f94088bd2894cc43aaa7a3feaa42d3744eac5309a022d6fefbaceed196` |
| `runs/20261023T124846Z_OFF/result.json` | `/termux-home/robot_runs/chat_mtp_repro/20261023T124846Z_OFF/result.json` | 13286 | `dfc882a70b70b2b0c8b52891312f227f2014d26e943a53e66c561a56b48fdbd5` |
| `runs/20261023T124846Z_OFF/run.json` | `/termux-home/robot_runs/chat_mtp_repro/20261023T124846Z_OFF/run.json` | 3780 | `48e355ca57dfffb737e26b450e8b24323d61b9c381d98dc77c08eb98d8c6e2d0` |
| `runs/20261023T124846Z_OFF/server_block00_OFF.log` | `/termux-home/robot_runs/chat_mtp_repro/20261023T124846Z_OFF/server_block00_OFF.log` | 12504 | `a20207f3a992f36c1282784c8e940718c1d4a43917076deb4627625671f5361e` |
| `runs/20261023T132032Z_ON/prompts.jsonl` | `/termux-home/robot_runs/chat_mtp_repro/20261023T132032Z_ON/prompts.jsonl` | 11330 | `3a0f20a590a41c10ac694b22db99b00f075efb53a909f46d5f8b946359577eff` |
| `runs/20261023T132032Z_ON/report.txt` | `/termux-home/robot_runs/chat_mtp_repro/20261023T132032Z_ON/report.txt` | 1739 | `a1c9199294c5847d62ed0b1d9163a175062c756a89037748e118485b778ecb58` |
| `runs/20261023T132032Z_ON/result.json` | `/termux-home/robot_runs/chat_mtp_repro/20261023T132032Z_ON/result.json` | 10965 | `8f6b8f601fa152d2b002668f5f5efcf2887b4b00c35a32f07fbe4ec8d54e2d57` |
| `runs/20261023T132032Z_ON/run.json` | `/termux-home/robot_runs/chat_mtp_repro/20261023T132032Z_ON/run.json` | 3939 | `364ed69da482b659c2e07608730c18ba19f35096043715abcf17b4369e6d825d` |
| `runs/20261023T132032Z_ON/server_block00_ON.log` | `/termux-home/robot_runs/chat_mtp_repro/20261023T132032Z_ON/server_block00_ON.log` | 15544 | `864591deb3849b624c1d1d5253eb37985508c15c3dbee7224a99e112bad01935` |
| `runs/oneshot_console_20261023T115731Z.log` | `/termux-home/robot_runs/chat_mtp_repro/oneshot_console_20261023T115731Z.log` | 724 | `47714e712bc18d2732384162dfcf0380c95477bf5cf378ee0d25e16354309de7` |
| `runs/oneshot_console_20261023T124838Z.log` | `/termux-home/robot_runs/chat_mtp_repro/oneshot_console_20261023T124838Z.log` | 2613 | `57e237e807eff4f6fe2facc35a763b1b42bab78abd651fc4124277d0fbe257c3` |
| `runs/oneshot_console_20261023T132024Z.log` | `/termux-home/robot_runs/chat_mtp_repro/oneshot_console_20261023T132024Z.log` | 2613 | `59518a8ced6d0c33689702ea716b93d0514aba7046afec87b93a93e03155967b` |
| `runs/thermal.log` | `/termux-home/robot_runs/chat_mtp_repro/thermal.log` | 2534 | `5a8277f84bcb8316999665479a51e8f9920cbc18fb39825e0167f19c8869c379` |

## Regenerated reports (1; written by this archival task, no phone original)

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `reports/REGENERATED_20261023T190340Z_combined_OFF_ON.txt` | 2025 | `1c7f31ea8b8be9198fc538778c95f46f41b892f928b87436b1286aef84de44ea` |

## Source files (4)

| File | Bytes | SHA-256 now | Reviewed frozen SHA-256 |
|---|---:|---|---|
| `README.md` | 9108 | `66224fdf52b677230edb1ab8443157c551b6312f1736e845654eeeb1713e1e6b` | `b094dd811615cc1f024cef77751ab9993fb9d7cb4ab4332fc9ca13f271e79114` (only the results section and first line were changed in the archival task) |
| `oneshot.sh` | 6988 | `98cbf178eac35b39d149dfca37ee5aed0eecba674746ee7c582d031bcd1a0be3` | same |
| `repro.py` | 20069 | `29276a35618e66833a214fba77c8ce0770bbe37d4fa9e2b4b9de105c13a00a46` | same |
| `test_repro.py` | 10201 | `00fb0831150e07e0728e38089a49e7ec3e84a3473ce5a7330340672166b46b0b` | same |

## Not archived

Weights and build (hashes from both runs' `run.json`, identical in both; sizes by `stat` on the phone):

| File (phone path) | Bytes | SHA-256 |
|---|---:|---|
| `/data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf` | 2841481184 | `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52` |
| `/data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf` | 97817664 | `9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919` |
| `~/llama.cpp-b1609-dotprod/build/bin/llama-server` | – | `77eb35a0e716dff307e0d33b0b45ce0ed91e216a20db9411a7b9dc94e6624053` |
| `~/llama.cpp-b1609-dotprod/build/bin/libggml-cpu.so.0.18.0` | – | `a4ef0d97e741f5902919af5e9cef95f808bd98581be76bca66fcee3f4ac8a0dd` |

The other six b1609 libraries are hashed in `build_sha256` of each `run.json`.

Review evidence left on the phone (`/termux-home/review/vis3c/`):

| Item | Why not archived | SHA-256 / size |
|---|---|---|
| `round{1,2,3}_copy/` | full isolated repository copies given to the reviewer (about 173 MB each); reconstructible from base `685a44b` plus the frozen candidate | – |
| `round1_pre.json` = `round1_post.json` | 1,035,273 B identity snapshot each; identical | `a9d1e05e0460079947b596ecfce754a2c75b618cf93d3c8330a2f06d33c8e574` |
| `round2_pre.json` = `round2_post.json` | as above; identical | `ee7d57ec2d08bd05e310f2957ffab31a6ce694cf1d568fdd5dc108affb93043f` |
| `round3_pre.json` = `round3_post.json` | as above; identical | `95c6b985f776654268024d346acbf4fef3a4cf60dad45993944f1a196707aef1` |
| `settings.backup{,2,3}.json` | shared AGY settings backups (agent configuration, not benchmark evidence); all three identical | `6e5273bf3864677e135b104758ba5724c8c45f8a17d3e55d9af19e66ee4b7421` |
| `round*_copy/_review_context/{chat,chat_mtp,server_manager}_2091f42.py` | equal to `git show 2091f42:<path>` | `8ec90263…`, `88759798…`, `47bf2dad…` |
| `/root/.gemini/antigravity-cli/log/cli-20261023_090340.log` (r1), `_091155.log` (r2), `_091954.log` (r3) | AGY session logs (model labels quoted in RUN_INDEX) | `50c1258b…`, `d262d9d1…`, `000b3c82…` |
| `/termux-home/robot_runs/chat_mtp_repro/.stop_thermal` | 0-byte launcher control file | `e3b0c442…` (empty) |

## sha256sum format

```sha256sums
6efc55bb315b18186f74e1f54ef735c50cfacd5bd2e952cac2619cbde975f538  reports/TASK.md
747abe328da51d4df63dc2cd8376a33948ef0c98a1780c786b687008df15259f  reports/candidate.diff
9ad9981e6d12b0d2ef4b0fcd394d6ec67bab5bb6803604c7f035f66c0a95bf72  reports/checks.txt
5a6665e3aea564fd25a7877be8bb1fcd5ff0419fd3bf0a0ec713f9e29e906b91  reports/current_hashes.txt
36f7c84fb997c4e027c017d153226e5f210fb9fc007ca0fe8a7f1ed768c9cc64  reports/frozen_candidate.sha256
70ad2f87f5cfbf7058a296f781b680cc203a6fffbf52bd881cf9b10ef000f5a0  reports/review_context/VIS3B_MTP_INVESTIGATION.md
782adf9c93f46d09d47474238c11543de60337c5f8c04398f3673d65b7a6db79  reports/review_context/VIS3B_MTP_REVIEW_ROUND2.json
ab2fe5b1ae793dc0d6513ef15efc5f70f3be44856dbecbf430c9f4a8185e2f00  reports/review_tool.py
9d497750c6bf2d5ce11aec2fa83be4a39dd8d9012152925804e94f4dad3dc319  reports/round1_TASK.md
b6e8f64a9aea62e786d3f262eb138cbde4d4bddba86e76290eb5191ba9666016  reports/round1_review.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/round1_stderr.txt
be510f75a7baf80ab7c27944eacad04ec03a61c7ec6ceeeebf06d92f2fc3f7c8  reports/round1_stdout.txt
d1944dea216138b1ab977ead28d7aabb7f454a8492f3a7dd01d57e3a27f99894  reports/round2_TASK.md
caa9ecc9ad5d1ea5f4c3fe6499afbbe10b501e86b48458134883aa26d08aa18e  reports/round2_review.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/round2_stderr.txt
b1c75769da90618d962c859e8c81d186620810c5c140d0628d02647f2e9efa58  reports/round2_stdout.txt
b09fc94599e86c20124b12ef974800a90f5460c9f4a4af529c32b394de4ccb88  reports/round3_review.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/round3_stderr.txt
2abf87eb3618e80644d6a40583755fdd4a7ab75398d0f0808eb7d12fb3e49fbb  reports/round3_stdout.txt
8436055899a15ec8cc42f0f3678df46b8eebaddb2b94bc8c2fe56af9eeb5ac3e  reports/settings.overlay.sha256
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/smoke2_stderr.txt
0525c26052207ea2f1895106b99dd08baa1fb2ff277dedab02f01d4a3ef840dd  reports/smoke2_stdout.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/smoke2b_stderr.txt
2f08018e3504c6951a21df257cb8f602bab5e92c33cfd5caba623e9d62257e29  reports/smoke2b_stdout.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/smoke3_stderr.txt
14261d0fb1d1d9ae8408cf26eebfc3b7c310fdacfe56a40151f82358cb3b8730  reports/smoke3_stdout.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reports/smoke_stderr.txt
c3846dbb437e4849088f47a3decad923dfb5e82b32e7ed4971f6e2d578642e4c  reports/smoke_stdout.txt
21b624765995dfb696d046fc05369bfeed5537ffa069df527c567652542dc728  runs/20261023T124846Z_OFF/prompts.jsonl
e9f8f2f94088bd2894cc43aaa7a3feaa42d3744eac5309a022d6fefbaceed196  runs/20261023T124846Z_OFF/report.txt
dfc882a70b70b2b0c8b52891312f227f2014d26e943a53e66c561a56b48fdbd5  runs/20261023T124846Z_OFF/result.json
48e355ca57dfffb737e26b450e8b24323d61b9c381d98dc77c08eb98d8c6e2d0  runs/20261023T124846Z_OFF/run.json
a20207f3a992f36c1282784c8e940718c1d4a43917076deb4627625671f5361e  runs/20261023T124846Z_OFF/server_block00_OFF.log
3a0f20a590a41c10ac694b22db99b00f075efb53a909f46d5f8b946359577eff  runs/20261023T132032Z_ON/prompts.jsonl
a1c9199294c5847d62ed0b1d9163a175062c756a89037748e118485b778ecb58  runs/20261023T132032Z_ON/report.txt
8f6b8f601fa152d2b002668f5f5efcf2887b4b00c35a32f07fbe4ec8d54e2d57  runs/20261023T132032Z_ON/result.json
364ed69da482b659c2e07608730c18ba19f35096043715abcf17b4369e6d825d  runs/20261023T132032Z_ON/run.json
864591deb3849b624c1d1d5253eb37985508c15c3dbee7224a99e112bad01935  runs/20261023T132032Z_ON/server_block00_ON.log
47714e712bc18d2732384162dfcf0380c95477bf5cf378ee0d25e16354309de7  runs/oneshot_console_20261023T115731Z.log
57e237e807eff4f6fe2facc35a763b1b42bab78abd651fc4124277d0fbe257c3  runs/oneshot_console_20261023T124838Z.log
59518a8ced6d0c33689702ea716b93d0514aba7046afec87b93a93e03155967b  runs/oneshot_console_20261023T132024Z.log
5a8277f84bcb8316999665479a51e8f9920cbc18fb39825e0167f19c8869c379  runs/thermal.log
1c7f31ea8b8be9198fc538778c95f46f41b892f928b87436b1286aef84de44ea  reports/REGENERATED_20261023T190340Z_combined_OFF_ON.txt
98cbf178eac35b39d149dfca37ee5aed0eecba674746ee7c582d031bcd1a0be3  oneshot.sh
29276a35618e66833a214fba77c8ce0770bbe37d4fa9e2b4b9de105c13a00a46  repro.py
00fb0831150e07e0728e38089a49e7ec3e84a3473ce5a7330340672166b46b0b  test_repro.py
```
