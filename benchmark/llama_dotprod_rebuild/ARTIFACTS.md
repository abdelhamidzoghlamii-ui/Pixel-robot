# Artifacts

Every archived file below with its SHA-256. Copied files were compared byte for byte (`cmp`) with the phone original at archive time, and the phone and archive SHA-256 values match. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Copied from the phone (52 files)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `build/llama-dotprod-build.log` | `/termux-home/llama-dotprod-build.log` | 24130 | `60a8567a2804153bdb75b85a16c94dfaf2ca3a14b54424e48568b72aa78d3809` |
| `build/llama-dotprod-configure.log` | `/termux-home/llama-dotprod-configure.log` | 2827 | `744d7e1b1a0dfabb6d8c77e56ff2725acda4899830c007d42d1cd9e17cb29e08` |
| `build/llama-dotprod-nofp16-build.log` | `/termux-home/llama-dotprod-nofp16-build.log (deleted from the phone 2026-09-26 after archiving)` | 24165 | `9f8c676caecaf86c7422b6930b31544e59697a512f2981b13e7cec1f98ce6df8` |
| `build/llama-dotprod-nofp16-configure.log` | `/termux-home/llama-dotprod-nofp16-configure.log (deleted from the phone 2026-09-26 after archiving)` | 2823 | `1e035f43474f67a21e797547728a24168b872d798e7a1ad2a8fea321fd25ac20` |
| `checks/compare_3a.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/compare_3a.py` | 937 | `59ae41f7cffc42344f1ed7544c4cdb5bfa7fac525636ed38f2d4f9dc86f5e600` |
| `checks/ladder_dp.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dp.py` | 641 | `4c9926425ccd722fe4084e228ee5712b4b33ab672357f7cc4fd9ae61f2ac0a84` |
| `checks/ladder_dponly.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly.py` | 660 | `ddfe118f02ea4e479fb390a3cf4197a89874e2e949a0440caaaed9da0e09df03` |
| `checks/ladder_dponly.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `checks/ladder_dponly.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly.stdout.txt` | 3134 | `b0862fb4c13b4b37b416ce433e59d5edb74bf3c086b6f49237231904de1ecdc4` |
| `checks/ladder_dponly/blocks.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly/blocks.jsonl` | 774 | `0f6ce96457f27970d5d18a8ab5784822b53348fe7801c735fa73890a9d617517` |
| `checks/ladder_dponly/decisions.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly/decisions.jsonl` | 116727 | `befa9d9fb1d9384827bf2cf2d577779cf2af9bc0f5dded801f9d848615833245` |
| `checks/ladder_dponly/logs/s1o_b1609dponly.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly/logs/s1o_b1609dponly.main.server.log` | 98433 | `b6846a73c6ac1150d233948b14fca263bb2dbd6d11f8f9b36cc4e8a2b4a84321` |
| `checks/ladder_dponly/logs/s1o_b1609dponly.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly/logs/s1o_b1609dponly.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `checks/ladder_dponly/report.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly/report.txt` | 3064 | `112d6a11e147c828e4e9818d5d2456f3f3493433d6294292e24abb5e931c7e87` |
| `checks/ladder_dponly/results.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly/results.json` | 4987 | `6d0fc99d20d977199087c3ffc8c2db22b375f8cbd9dbf2a818d572895ba9daed` |
| `checks/ladder_dponly/run.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_dponly/run.json` | 197 | `0be6a9a034960a47cc8bbb33a64dda4eea7e00e3c7145090c3aac61c0ed4d2d5` |
| `checks/ladder_new.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `checks/ladder_new.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new.stdout.txt` | 3087 | `3ef87918e5e7f3464367f5c56d1703b2233e142cfb93ff895d8d3e0ec56649f6` |
| `checks/ladder_new/blocks.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new/blocks.jsonl` | 769 | `5252db938a6adf9e0a48649cc903b8972865105fa57e18627cc1a0e5cc7b5c11` |
| `checks/ladder_new/decisions.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new/decisions.jsonl` | 116293 | `6d87f2ae3b508e3973fb0cbef779384a5438043819104c799ecbef00deee61d0` |
| `checks/ladder_new/logs/s1o_b1609dp.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new/logs/s1o_b1609dp.main.server.log` | 98433 | `cd120544f39ccf32be0ad4c85d47d317c7e8acf0c9ddfd7e8365682c8285f299` |
| `checks/ladder_new/logs/s1o_b1609dp.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new/logs/s1o_b1609dp.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `checks/ladder_new/report.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new/report.txt` | 3021 | `f44a63a67f862b8aa592e448c1ba7eb0b6b00ad248fc62172297bbd5c985cd0a` |
| `checks/ladder_new/results.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new/results.json` | 4981 | `85e4670afdea7ef683e0371d6f6ec1dbab6fee8e703246302087831b593c3fcf` |
| `checks/ladder_new/run.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new/run.json` | 197 | `0be6a9a034960a47cc8bbb33a64dda4eea7e00e3c7145090c3aac61c0ed4d2d5` |
| `checks/ladder_new_L3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new_L3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `checks/ladder_new_L3/blocks.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new_L3/blocks.jsonl` | 768 | `496e020cb804ae2d0f79e04fb41f62330429282bcc440e56b087a12e444d6687` |
| `checks/ladder_new_L3/decisions.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new_L3/decisions.jsonl` | 21791 | `51d4a3ff261a33c15b100b9472d4539bafa057f9ece48d6581c82ddae078e86d` |
| `checks/ladder_new_L3/logs/s1o_b1609dp.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new_L3/logs/s1o_b1609dp.main.server.log` | 21283 | `b8ead68793444bf56c104e7fee5b1722682ef9bb1a1fd41846120c72a20042c0` |
| `checks/ladder_new_L3/logs/s1o_b1609dp.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new_L3/logs/s1o_b1609dp.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `checks/ladder_new_L3/report.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new_L3/report.txt` | 2981 | `856e2fd7def43ea7a926df1f49a2b9df718aa3d91537daed00a93de6bb2b074e` |
| `checks/ladder_new_L3/results.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new_L3/results.json` | 4501 | `db795dddf9fc51766f089218d351dfd467b5a7cf6795b842eadd46601ec9baab` |
| `checks/ladder_new_L3/run.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/ladder_new_L3/run.json` | 185 | `44a5f97c8748f36df6b73d558adcc06332cdda656eb6cffcd667c9a10a5f3675` |
| `checks/review.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/review.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `checks/review.stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/review.stdout.json` | 3968 | `dc323b60cd9ccc2bd89bf6d2be267c4561524423d06e4eac9361e66776e04a60` |
| `checks/review/EVIDENCE.md` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/review/EVIDENCE.md` | 8843 | `53081d105ac1c30cca11097f94c57d781b0b2f3c1633388c6ed4e3d71b2c151a` |
| `checks/review/REQUEST.md` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/review/REQUEST.md` | 1752 | `b544ea1684d377bce4b5d55629aa62657b8bd2a7ffc31c9e3c8099de74d9ce94` |
| `checks/review/frozen.sha256` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/review/frozen.sha256` | 385 | `a49e7d3306d430f7bda0f6991565821754ecf4b4c041a88e3b625a3fdea79b07` |
| `checks/review/prompt.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/review/prompt.txt` | 33960 | `68a99f7920c75a818d044328eede1af49bf2dc01f1670bc30867e634b1682769` |
| `checks/review/server_manager.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/review/server_manager.diff` | 470 | `b130e32213ea355e28a3517955d0dc0f9559d8cba7271cc31463f9770c142577` |
| `checks/smoke.out` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/smoke.out` | 792 | `695971b8afcb702395efe1b93bc6402a191b4713cf80eb1f9b361458db3b2106` |
| `checks/smoke.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/smoke.py` | 701 | `aba76895b708d1760888f2ab41bf80f4c3234cb7920fad04144fdc15aaab075c` |
| `checks/smoke.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/smoke.sh` | 334 | `407d9428e05b53f7171278823c0f28171f4fa513e085e89b8f1e02e58f81fc7b` |
| `checks/sysinfo.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/sysinfo.log` | 168342 | `0422d66bf14e1e02f8c738ae1d5add8b13e1f860669a4b1e25d259db0d7192b9` |
| `checks/voice_gen.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/voice_gen.json` | 8856 | `b9923757ad5fb39a35e39d5c5993d0957a1a8c218cf8680175ee9908541974c0` |
| `checks/voice_gen.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/voice_gen.py` | 3876 | `d944f24b0a17bb027453b2a233c925580cb76e8e20d0ab7e33526507becf7629` |
| `checks/voice_gen.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/voice_gen.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `checks/voice_gen.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/voice_gen.stdout.txt` | 4589 | `9624245070d61adbc54badf0fda47824b91840382c73b77746bb70aea5c75e1a` |
| `checks/voice_old1.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/voice_old1.py` | 579 | `274e7bb259c72bae6c26df83528ccc200a11b6d0b78d68329434c67038a00045` |
| `checks/voice_warm.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/voice_warm.py` | 1379 | `cf0b8df14d0e15104f988c19aa104f91fd35ca4f476c3efef831731a8a8f4644` |
| `checks/voice_warm.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/dp/voice_warm.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reports/CODER_REPORT_llama_dotprod_rebuild.md` | `/termux-home/archive-staging/reports/CODER_REPORT_llama_dotprod_rebuild.md` | 10546 | `36eacedffac6f75e6fad7cfc59e874bbdbbf4baaa7b3314c960451ae7c73814f` |

## Written for this archive or already in place in the repository (4 files)

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 3723 | `615a1922db7ea73f279915e74bbf500ad918ffd42b2172bac73b3eb00f075cd7` |
| `RUN_INDEX.md` | 826 | `2b9f76fab8e9c2f4067278c9edd239990ba8ee699aeb36a1ee5fb4837ca8dca7` |
| `build/cmake_cache_diff.txt` | 603 | `5a30b11f8d6ac8f449c7772b4fb0b4bfdbac0dad5246460ed4936f52101f2fc5` |
| `build/sdot_counts.txt` | 1372 | `3d99eae5062f1a36a35cc8e87d472c688c0490e0e54fe892ae3018787a49fd60` |

## sha256sum format

```sha256sums
615a1922db7ea73f279915e74bbf500ad918ffd42b2172bac73b3eb00f075cd7  README.md
2b9f76fab8e9c2f4067278c9edd239990ba8ee699aeb36a1ee5fb4837ca8dca7  RUN_INDEX.md
5a30b11f8d6ac8f449c7772b4fb0b4bfdbac0dad5246460ed4936f52101f2fc5  build/cmake_cache_diff.txt
60a8567a2804153bdb75b85a16c94dfaf2ca3a14b54424e48568b72aa78d3809  build/llama-dotprod-build.log
744d7e1b1a0dfabb6d8c77e56ff2725acda4899830c007d42d1cd9e17cb29e08  build/llama-dotprod-configure.log
9f8c676caecaf86c7422b6930b31544e59697a512f2981b13e7cec1f98ce6df8  build/llama-dotprod-nofp16-build.log
1e035f43474f67a21e797547728a24168b872d798e7a1ad2a8fea321fd25ac20  build/llama-dotprod-nofp16-configure.log
3d99eae5062f1a36a35cc8e87d472c688c0490e0e54fe892ae3018787a49fd60  build/sdot_counts.txt
59ae41f7cffc42344f1ed7544c4cdb5bfa7fac525636ed38f2d4f9dc86f5e600  checks/compare_3a.py
4c9926425ccd722fe4084e228ee5712b4b33ab672357f7cc4fd9ae61f2ac0a84  checks/ladder_dp.py
ddfe118f02ea4e479fb390a3cf4197a89874e2e949a0440caaaed9da0e09df03  checks/ladder_dponly.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checks/ladder_dponly.stderr.txt
b0862fb4c13b4b37b416ce433e59d5edb74bf3c086b6f49237231904de1ecdc4  checks/ladder_dponly.stdout.txt
0f6ce96457f27970d5d18a8ab5784822b53348fe7801c735fa73890a9d617517  checks/ladder_dponly/blocks.jsonl
befa9d9fb1d9384827bf2cf2d577779cf2af9bc0f5dded801f9d848615833245  checks/ladder_dponly/decisions.jsonl
b6846a73c6ac1150d233948b14fca263bb2dbd6d11f8f9b36cc4e8a2b4a84321  checks/ladder_dponly/logs/s1o_b1609dponly.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checks/ladder_dponly/logs/s1o_b1609dponly.main.stderr.txt
112d6a11e147c828e4e9818d5d2456f3f3493433d6294292e24abb5e931c7e87  checks/ladder_dponly/report.txt
6d0fc99d20d977199087c3ffc8c2db22b375f8cbd9dbf2a818d572895ba9daed  checks/ladder_dponly/results.json
0be6a9a034960a47cc8bbb33a64dda4eea7e00e3c7145090c3aac61c0ed4d2d5  checks/ladder_dponly/run.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checks/ladder_new.stderr.txt
3ef87918e5e7f3464367f5c56d1703b2233e142cfb93ff895d8d3e0ec56649f6  checks/ladder_new.stdout.txt
5252db938a6adf9e0a48649cc903b8972865105fa57e18627cc1a0e5cc7b5c11  checks/ladder_new/blocks.jsonl
6d87f2ae3b508e3973fb0cbef779384a5438043819104c799ecbef00deee61d0  checks/ladder_new/decisions.jsonl
cd120544f39ccf32be0ad4c85d47d317c7e8acf0c9ddfd7e8365682c8285f299  checks/ladder_new/logs/s1o_b1609dp.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checks/ladder_new/logs/s1o_b1609dp.main.stderr.txt
f44a63a67f862b8aa592e448c1ba7eb0b6b00ad248fc62172297bbd5c985cd0a  checks/ladder_new/report.txt
85e4670afdea7ef683e0371d6f6ec1dbab6fee8e703246302087831b593c3fcf  checks/ladder_new/results.json
0be6a9a034960a47cc8bbb33a64dda4eea7e00e3c7145090c3aac61c0ed4d2d5  checks/ladder_new/run.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checks/ladder_new_L3.stderr.txt
496e020cb804ae2d0f79e04fb41f62330429282bcc440e56b087a12e444d6687  checks/ladder_new_L3/blocks.jsonl
51d4a3ff261a33c15b100b9472d4539bafa057f9ece48d6581c82ddae078e86d  checks/ladder_new_L3/decisions.jsonl
b8ead68793444bf56c104e7fee5b1722682ef9bb1a1fd41846120c72a20042c0  checks/ladder_new_L3/logs/s1o_b1609dp.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checks/ladder_new_L3/logs/s1o_b1609dp.main.stderr.txt
856e2fd7def43ea7a926df1f49a2b9df718aa3d91537daed00a93de6bb2b074e  checks/ladder_new_L3/report.txt
db795dddf9fc51766f089218d351dfd467b5a7cf6795b842eadd46601ec9baab  checks/ladder_new_L3/results.json
44a5f97c8748f36df6b73d558adcc06332cdda656eb6cffcd667c9a10a5f3675  checks/ladder_new_L3/run.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checks/review.stderr.txt
dc323b60cd9ccc2bd89bf6d2be267c4561524423d06e4eac9361e66776e04a60  checks/review.stdout.json
53081d105ac1c30cca11097f94c57d781b0b2f3c1633388c6ed4e3d71b2c151a  checks/review/EVIDENCE.md
b544ea1684d377bce4b5d55629aa62657b8bd2a7ffc31c9e3c8099de74d9ce94  checks/review/REQUEST.md
a49e7d3306d430f7bda0f6991565821754ecf4b4c041a88e3b625a3fdea79b07  checks/review/frozen.sha256
68a99f7920c75a818d044328eede1af49bf2dc01f1670bc30867e634b1682769  checks/review/prompt.txt
b130e32213ea355e28a3517955d0dc0f9559d8cba7271cc31463f9770c142577  checks/review/server_manager.diff
695971b8afcb702395efe1b93bc6402a191b4713cf80eb1f9b361458db3b2106  checks/smoke.out
aba76895b708d1760888f2ab41bf80f4c3234cb7920fad04144fdc15aaab075c  checks/smoke.py
407d9428e05b53f7171278823c0f28171f4fa513e085e89b8f1e02e58f81fc7b  checks/smoke.sh
0422d66bf14e1e02f8c738ae1d5add8b13e1f860669a4b1e25d259db0d7192b9  checks/sysinfo.log
b9923757ad5fb39a35e39d5c5993d0957a1a8c218cf8680175ee9908541974c0  checks/voice_gen.json
d944f24b0a17bb027453b2a233c925580cb76e8e20d0ab7e33526507becf7629  checks/voice_gen.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checks/voice_gen.stderr.txt
9624245070d61adbc54badf0fda47824b91840382c73b77746bb70aea5c75e1a  checks/voice_gen.stdout.txt
274e7bb259c72bae6c26df83528ccc200a11b6d0b78d68329434c67038a00045  checks/voice_old1.py
cf0b8df14d0e15104f988c19aa104f91fd35ca4f476c3efef831731a8a8f4644  checks/voice_warm.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checks/voice_warm.stderr.txt
36eacedffac6f75e6fad7cfc59e874bbdbbf4baaa7b3314c960451ae7c73814f  reports/CODER_REPORT_llama_dotprod_rebuild.md
```
