# Artifacts

Every archived file below with its SHA-256. Copied files were compared byte for byte (`cmp`) with the phone original at archive time, and the phone and archive SHA-256 values match. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS_2026-09-26.md | grep -v '^```' | sha256sum -c`.

## Copied from the phone (176 files)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `conversation/conv_quality.py` | `/termux-home/ladder/conv_quality.py` | 2471 | `4b578dc988a24575da053b2df4a756c99f7ab9053c4f9a3fd4812ac9c4f4b4c6` |
| `conversation/conv_speed.py` | `/termux-home/ladder/conv_speed.py` | 12998 | `9b721d8678f53a122f022f8bf66ee318707aa1700b5c684ca144fa06f7b21e7a` |
| `conversation/conversation_20260926T185957Z.stderr.txt` | `/termux-home/ladder/conversation_20260926T185957Z.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/conversation_20260926T185957Z.stdout.txt` | `/termux-home/ladder/conversation_20260926T185957Z.stdout.txt` | 6929 | `da0c071404d7297425e16e3438e5c9521156e52bc6e80380109be52b8430022c` |
| `conversation/conversation_20260926T185957Z/blocks.jsonl` | `/termux-home/ladder/conversation_20260926T185957Z/blocks.jsonl` | 3789 | `441b0730b932a72f9c271a2c22c93e69e18722cc89454bafcf2f24d53a3ad5b9` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e2b_q40.cached.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/gemma_e2b_q40.cached.server.log` | 22623 | `c150ec7322bfb0c30961d41156aead34ca0f41906e98663f9493a26304e8c2fa` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e2b_q40.cold.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/gemma_e2b_q40.cold.server.log` | 22623 | `7210a92f32a2edc0c442829a1597e91aba9909916b3385e1a665bd25f78d49da` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e2b_q4km.cached.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/gemma_e2b_q4km.cached.server.log` | 22631 | `a39b643e578f5ebef95658d778b5053935249878da6c3e0224822228e4533c67` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e2b_q4km.cold.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/gemma_e2b_q4km.cold.server.log` | 22631 | `7a5c8078950272ecd16588c76dc10e11e2023dc6dae2b030f7112ecbfe280ade` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e4b_q4km.cached.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/gemma_e4b_q4km.cached.server.log` | 22889 | `8b7d8c05b62aa98ca36f0db34896acf5365a40ff10c269bc6e3193057b9c50f5` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e4b_q4km.cold.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/gemma_e4b_q4km.cold.server.log` | 22883 | `4572ef516f60f01a295d89ae2a275f9f31c122a9c6f25bee65b6beafb6e456ba` |
| `conversation/conversation_20260926T185957Z/logs/qwen35_2b.cached.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/qwen35_2b.cached.server.log` | 23684 | `130c296031e5e2940ace4a0d0077f45321d6bea4b4a8f23fde9f2058cdad332c` |
| `conversation/conversation_20260926T185957Z/logs/qwen35_2b.cold.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/qwen35_2b.cold.server.log` | 23684 | `dc09e176671262a3474f60919055d5c9c52693d9132a7cec0204a0c74a3769ad` |
| `conversation/conversation_20260926T185957Z/logs/qwen35_4b.cached.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/qwen35_4b.cached.server.log` | 25737 | `9d8d7b45a26f5ebabf5b48663fd95e454fed3bfe0e8c2611cc773e4e71dfd368` |
| `conversation/conversation_20260926T185957Z/logs/qwen35_4b.cold.server.log` | `/termux-home/ladder/conversation_20260926T185957Z/logs/qwen35_4b.cold.server.log` | 25737 | `ee0a95acfe219f0150964bddf4badf464cf3085dfde350d53169394b5087235b` |
| `conversation/conversation_20260926T185957Z/report.txt` | `/termux-home/ladder/conversation_20260926T185957Z/report.txt` | 1766 | `3a917faeb494ac9b24c8e3aad3c6ea35da788117f64b4a97206decd9214796ef` |
| `conversation/conversation_20260926T185957Z/run_20260926T185957Z.json` | `/termux-home/ladder/conversation_20260926T185957Z/run_20260926T185957Z.json` | 1381 | `f894d112211ce542aa55feddccfc92112cc8371580cc0953b5ef6af9b181fa45` |
| `conversation/conversation_20260926T185957Z/turns.jsonl` | `/termux-home/ladder/conversation_20260926T185957Z/turns.jsonl` | 97940 | `6b3e7da5101ddf45e4a8698d21c545a7e925981dd16ac8ae82e7455732a14b35` |
| `conversation/conversation_blind_key.tsv` | `/sdcard/.trash-storage/Download/.trashed-1793045488-conversation_blind_key.tsv` | 1266 | `b8657ac39d1eaf054e334d5827bf6ae6c0fe7205f46529406d3352ffb36cb77f` |
| `conversation/conversation_blind_sheet.md` | `/sdcard/.trash-storage/Download/.trashed-1793045488-conversation_blind_sheet.md` | 12812 | `0c1678c7d0af08dcbdd660ebbf5cdafb973f0c4ee3b6abce46723b4e71fb684e` |
| `conversation/conversation_quality.stderr.txt` | `/termux-home/ladder/conversation_quality.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/conversation_quality.stdout.txt` | `/termux-home/ladder/conversation_quality.stdout.txt` | 5662 | `2921ef45759d2c34e42137872a24663e187c1008c7bc85d0ef9ba8b104882f5b` |
| `conversation/conversation_quality/bench_blind_key.txt` | `/termux-home/ladder/conversation_quality/bench_blind_key.txt` | 84 | `c7d19e8185e535453de0d32e0068bc37f5b5af11cdd35e7a05d304f73b2e2c3e` |
| `conversation/conversation_quality/bench_blind_transcripts.txt` | `/termux-home/ladder/conversation_quality/bench_blind_transcripts.txt` | 6447 | `c3203ad6c26a438e9fe36d3c3614af89eaae0a5fd4603761b5f2a7df08564f96` |
| `conversation/conversation_quality/bench_c_summary.txt` | `/termux-home/ladder/conversation_quality/bench_c_summary.txt` | 3547 | `b63dc5b8ef89e2f62c450d7547884451f2d21109e34cb38f27efbdb5af4e6dc9` |
| `conversation/conversation_quality/bench_gemma_e2b_q40_20260926_163419.log` | `/termux-home/ladder/conversation_quality/bench_gemma_e2b_q40_20260926_163419.log` | 69777 | `4d55c8c591bce02d28ce38fb42d7257887e6883932a8825cfaefc9304743567e` |
| `conversation/conversation_quality/bench_gemma_e2b_q4km_20260926_163419.log` | `/termux-home/ladder/conversation_quality/bench_gemma_e2b_q4km_20260926_163419.log` | 69970 | `4f8fcf85b59f7a4d305b96e2d11d6fba14f7f914754b10d81f0c8d34af0306f1` |
| `conversation/conversation_quality/bench_gemma_e4b_q4km_20260926_163419.log` | `/termux-home/ladder/conversation_quality/bench_gemma_e4b_q4km_20260926_163419.log` | 70327 | `3848f700485b8c85c4e35bc6abcd39ce96fc95afd2c52d24e60aec0a94eb9372` |
| `conversation/conversation_quality/bench_qwen35_2b_20260926_163419.log` | `/termux-home/ladder/conversation_quality/bench_qwen35_2b_20260926_163419.log` | 75036 | `d30497d11b584058aae2965b4b2d1adb61d0c3f997366f85adf6fada13c9fe2e` |
| `conversation/conversation_quality/bench_qwen35_4b_20260926_163419.log` | `/termux-home/ladder/conversation_quality/bench_qwen35_4b_20260926_163419.log` | 23914 | `e09b3605ce7232f8817359145c99a228715223cf0b1da0562a2a2ba71ab5c5e3` |
| `conversation/conversation_quality/bench_results_20260926_163419.json` | `/termux-home/ladder/conversation_quality/bench_results_20260926_163419.json` | 542095 | `398e36f554860da02220dd183296d8b3d9f6c56bd24e480c8693af73a9465e72` |
| `conversation/conversation_quality/bench_speed_summary.txt` | `/termux-home/ladder/conversation_quality/bench_speed_summary.txt` | 1037 | `452ba4adc348e01de858c43b322145cab603af188501620c45e4ec71a8588e45` |
| `conversation/conversation_quality_combined.json` | `/termux-home/ladder/conversation_quality_combined.json` | 568881 | `a655686de7f4e6bea222994cc2b4a5c5114b286ee97b5265d0a154072e70d6cd` |
| `conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734.stderr.txt` | `/termux-home/ladder/conversation_quality_qwen35_4b.killed_by_lmk_1734.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734.stdout.txt` | `/termux-home/ladder/conversation_quality_qwen35_4b.killed_by_lmk_1734.stdout.txt` | 176 | `469a7584dc4d8e95ca8ea50fd8a9b4530fa5388c7ca8f678c10d276de3709e22` |
| `conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734/bench_blind_key.txt` | `/termux-home/ladder/conversation_quality_qwen35_4b.killed_by_lmk_1734/bench_blind_key.txt` | 14 | `952a7cb1ca044aba075bf1e1406edf1d0adab1ba8a90f9b18352daca7d5c0f64` |
| `conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734/bench_qwen35_4b_20260926_173003.log` | `/termux-home/ladder/conversation_quality_qwen35_4b.killed_by_lmk_1734/bench_qwen35_4b_20260926_173003.log` | 23768 | `825416d1b4fe097b0a95f41ea6e7d5ff8848b5970f517c4175546b67b73ac83b` |
| `conversation/conversation_quality_qwen35_4b.stderr.txt` | `/termux-home/ladder/conversation_quality_qwen35_4b.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/conversation_quality_qwen35_4b.stdout.txt` | `/termux-home/ladder/conversation_quality_qwen35_4b.stdout.txt` | 208 | `0dc52c7bd51cc38d26854530a7bc6b572efdf13f4cc5ead25a3c7a8600d99248` |
| `conversation/conversation_quality_qwen35_4b/bench_blind_key.txt` | `/termux-home/ladder/conversation_quality_qwen35_4b/bench_blind_key.txt` | 14 | `952a7cb1ca044aba075bf1e1406edf1d0adab1ba8a90f9b18352daca7d5c0f64` |
| `conversation/conversation_quality_qwen35_4b/bench_blind_transcripts.txt` | `/termux-home/ladder/conversation_quality_qwen35_4b/bench_blind_transcripts.txt` | 1781 | `09cb67e2559bb953dfc331771cf8c130e6906195f83b3d520311d6cb6d6e3414` |
| `conversation/conversation_quality_qwen35_4b/bench_c_summary.txt` | `/termux-home/ladder/conversation_quality_qwen35_4b/bench_c_summary.txt` | 1175 | `44f2cac264f738f8ac3a97a516e07736402756f5b356a545b7a9d1025dce7f8e` |
| `conversation/conversation_quality_qwen35_4b/bench_qwen35_4b_20260926_182310.log` | `/termux-home/ladder/conversation_quality_qwen35_4b/bench_qwen35_4b_20260926_182310.log` | 82006 | `3b1aeb8b0a5e9275fd7ba790d63c1538cbae4b29ed8c22dc46c12086b16ea106` |
| `conversation/conversation_quality_qwen35_4b/bench_results_20260926_182310.json` | `/termux-home/ladder/conversation_quality_qwen35_4b/bench_results_20260926_182310.json` | 121663 | `1458ea65de10333e8d2142a8eece3ae04e18ec91c5dca5ec5c91bfa245bcb6c1` |
| `conversation/conversation_quality_qwen35_4b/bench_speed_summary.txt` | `/termux-home/ladder/conversation_quality_qwen35_4b/bench_speed_summary.txt` | 510 | `b2ec62f069f96fc976807ab0655e1bce3f056b17748d09419672cbb8c0795843` |
| `conversation/conversation_quality_scores.txt` | `/termux-home/ladder/conversation_quality_scores.txt` | 3809 | `8002085be7a0327a27fd56a10fb9d3e1f816a1bfdc1480214654c0169418ac58` |
| `conversation/local_ai_blind_scores.tsv` | `/sdcard/Download/local_ai_blind_scores.tsv` | 1909 | `0da2f2b8c6af6445b9b06749ead6f5444a57fbc5e96fb4ebd56b61daf859bd59` |
| `conversation/make_blind_sheet.py` | `/termux-home/ladder/make_blind_sheet.py` | 3427 | `c698988cf418d37bf1388816cfefa8624a0e8cfde377094c92837ce682bbde49` |
| `conversation/prep/full_quality.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/full_quality.sh` | 244 | `2c2bc7f88c40b6c0642fd1e6706af8261661329127b94d3ef2cbb56db9ea1900` |
| `conversation/prep/mem_diag.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/mem_diag.py` | 1729 | `74132400e1afba33690ca47bd8b7c94ed9efe79905b3a55f58a183d7174dae31` |
| `conversation/prep/mem_diag.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/mem_diag.server.log` | 11953 | `22ee65f38df9cb8a02f806e0f980702df057d8dd9a38a8255ca65be0a4fb5d6b` |
| `conversation/prep/mem_diag_cr0.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/mem_diag_cr0.server.log` | 18443 | `7aa5138626d3db8b13cb1069f957373d58f1aa1765a80765393c2e4e83968db7` |
| `conversation/prep/mem_diag_cr0.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/mem_diag_cr0.txt` | 1530 | `ff4449216f382be39f938ef289309cce8f59474fe32103c5214d805789776900` |
| `conversation/prep/oneshot.sh.orig` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/oneshot.sh.orig` | 2533 | `0edcae559ea922a7be9ceefc9b4e20e5191eb8526494929046eb44103232a456` |
| `conversation/prep/quick.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick.sh` | 1172 | `22da3105d56434889fd3a0f8c3a9481f67dfc32ebef2647acb5c8babcb86ce86` |
| `conversation/prep/quick_key.tsv` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_key.tsv` | 1266 | `d9100bdf12b3516463943d3782d581f9f14f76de46ddd694c0ed86ada744aca2` |
| `conversation/prep/quick_quality.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality.py` | 388 | `c0e405e6b024cc19c9e6b20f4369cbf8fd2d2d085af11af0f8b376553803eb83` |
| `conversation/prep/quick_quality.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/prep/quick_quality.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality.stdout.txt` | 350 | `6a91b99d5f486013928d9efae66cd6a65f11a2855a46d515942c5bddafcc2ec0` |
| `conversation/prep/quick_quality/bench_blind_key.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_blind_key.txt` | 84 | `e342699140a128ada2b3aebad36eb0e0e5d925f8944f734ac31582b68c35159a` |
| `conversation/prep/quick_quality/bench_blind_transcripts.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_blind_transcripts.txt` | 366 | `f4b101ec3e2b7da9b1c2463376203501743d82cd6ddbabdcc1fd54b3205228b5` |
| `conversation/prep/quick_quality/bench_c_summary.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_c_summary.txt` | 1029 | `388b6e90fe58a595282e1498e0c5742a10c6dac79be95394a97f9d151291315a` |
| `conversation/prep/quick_quality/bench_gemma_e2b_q40_20260926_162121.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_gemma_e2b_q40_20260926_162121.log` | 2946 | `0a61cf78ae265008375056dc79f468f0afe5bc36c64e90e64eab2c62ab905fd8` |
| `conversation/prep/quick_quality/bench_gemma_e2b_q4km_20260926_162121.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_gemma_e2b_q4km_20260926_162121.log` | 2948 | `8777aad501da238ce6f1331d4b2b0d21c40935973f56d88ebd932b243214d548` |
| `conversation/prep/quick_quality/bench_gemma_e4b_q4km_20260926_162121.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_gemma_e4b_q4km_20260926_162121.log` | 3206 | `9ccd6e607f0a524301ee1be3faf3bce824ec3c4b3182aa1c1bc57df9c97a0009` |
| `conversation/prep/quick_quality/bench_qwen35_2b_20260926_162121.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_qwen35_2b_20260926_162121.log` | 3407 | `e9cddea81b90a0c87618d2a5f319bccbcd4905638bbb80e479c50c3257e17bb6` |
| `conversation/prep/quick_quality/bench_qwen35_4b_20260926_162121.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_qwen35_4b_20260926_162121.log` | 3407 | `dd75299f97a6d15d93fb5794dca7843596dba9623044239e7759ec856e5a258e` |
| `conversation/prep/quick_quality/bench_results_20260926_162121.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_results_20260926_162121.json` | 14126 | `d523e21edd6be12286918327120f1920c83a57de0bf1452edd6b9b6d6cda15c7` |
| `conversation/prep/quick_quality/bench_speed_summary.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_quality/bench_speed_summary.txt` | 1035 | `3bbcd02d370f48510ecad6e0729477d1cc9695081f6ddc8b121a61e3a1c164dd` |
| `conversation/prep/quick_sheet.md` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_sheet.md` | 10603 | `65ffabf6b6e51d211ec94e2f227513f4cad2dc747fac2d2e8f43efbe7bec57d9` |
| `conversation/prep/quick_speed.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed.py` | 441 | `8ce982e5ae37a79b60c41f9880d7df7402e514fed18352e8fae9eb641e35e522` |
| `conversation/prep/quick_speed.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/prep/quick_speed.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed.stdout.txt` | 5542 | `7c754c14c62acf5ef0a8f9e71ce713e53e1a9eeb831bb20549ac2aff688aa27c` |
| `conversation/prep/quick_speed/blocks.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/blocks.jsonl` | 3942 | `2f1d449b9701eb7f59cb59d5fbe534fbe8b649f87640c8bf63d1c15e4f361211` |
| `conversation/prep/quick_speed/logs/gemma_e2b_q40.cached.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/gemma_e2b_q40.cached.server.log` | 2946 | `c2de57f3b03c9fcfdb58396cff79e44badbd81677b161985fffd29756701bd81` |
| `conversation/prep/quick_speed/logs/gemma_e2b_q40.cold.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/gemma_e2b_q40.cold.server.log` | 2946 | `731667e68b1299dd5a9024fe079c60c985602fd81e39652dad008a4697686bf2` |
| `conversation/prep/quick_speed/logs/gemma_e2b_q4km.cached.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/gemma_e2b_q4km.cached.server.log` | 2948 | `13b2f94115ce90839d7f568dcc7da4c3dcb182d5daaedb902bd82f5a82dbc901` |
| `conversation/prep/quick_speed/logs/gemma_e2b_q4km.cold.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/gemma_e2b_q4km.cold.server.log` | 2948 | `710498599567aadc391c24239f8b87b6b00ff320ff90bf670ee0ffa691d16a2e` |
| `conversation/prep/quick_speed/logs/gemma_e4b_q4km.cached.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/gemma_e4b_q4km.cached.server.log` | 3206 | `7af4f9bbcc905f6218ecc415da21ff9eb2ffd49c991193ef8b97e136bb0f0e9c` |
| `conversation/prep/quick_speed/logs/gemma_e4b_q4km.cold.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/gemma_e4b_q4km.cold.server.log` | 3206 | `51f21c0a2f3c446d5d2b7062a2e36a1b4b0719331b5136d4f7af79b3bfbb6355` |
| `conversation/prep/quick_speed/logs/qwen35_2b.cached.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/qwen35_2b.cached.server.log` | 2500 | `1e7211b934ec37bc35a93be3d48cb0b01f5299fc895878b83e833ab39575b1ba` |
| `conversation/prep/quick_speed/logs/qwen35_2b.cold.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/qwen35_2b.cold.server.log` | 2500 | `c730e022bba8d6b9d5aba24b6fedf8e5ca0b032c1d643ccfb4b1d19da7ad20a0` |
| `conversation/prep/quick_speed/logs/qwen35_4b.cached.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/qwen35_4b.cached.server.log` | 2646 | `2c8753e8ba22f002d0f6b2359e2e5a0d3d2c732eeb4de0e62c8c601bc6907e47` |
| `conversation/prep/quick_speed/logs/qwen35_4b.cold.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/logs/qwen35_4b.cold.server.log` | 2646 | `daf4bde5d65cd76cd63dc953e0bc2a965814ca95974be20a8a743f73c29c973a` |
| `conversation/prep/quick_speed/report.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/report.txt` | 1784 | `ac89115e42a55a0516659f5b08a768fcb3f935eee35f71f2e67c2924c2c953f4` |
| `conversation/prep/quick_speed/run_20260926T162800Z.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/run_20260926T162800Z.json` | 1357 | `3e65d3d38cead9a893f57c19ec4dc8d3d6e29248df7cc8328ca7ac8009624652` |
| `conversation/prep/quick_speed/turns.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/quick_speed/turns.jsonl` | 6350 | `c0ee547e120f14e071cad0b11ab4bf5ec7439fd2c51b46345def18e674f77479` |
| `conversation/prep/qwen4b_a7.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/qwen4b_a7.py` | 1115 | `be53afcba1ceb663d0792e270124f3326714b6985d3cd8bc3c5a2921fdc042f8` |
| `conversation/prep/qwen4b_a7.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/qwen4b_a7.server.log` | 4462 | `5a2a47c8575a279d7f235a2ffb90aad37275dd27b9f22d18b8c48e5a7d8d7300` |
| `conversation/prep/rerun_qwen4b.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/rerun_qwen4b.sh` | 586 | `1ca302eeb20a186d77b5b16db852b30a8b3f4529f53913475f0e9cd63fec7eea` |
| `conversation/prep/review.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/prep/review.stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review.stdout.json` | 3400 | `f87fddcf0919bd7cacbe35e12d678b4afb00258897af8f5cdf0e29e517a976e8` |
| `conversation/prep/review/REQUEST.md` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review/REQUEST.md` | 3719 | `6a76038f342c5a5752b4dae242375fbbed39b82b73d01b9cbea5b16f914b6890` |
| `conversation/prep/review/bench_rubric.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review/bench_rubric.diff` | 2827 | `fd8fbf41b0b2921f5db5341f058d9e9676c44bdc8aa0e8f9e8f1ee7e2e20eed2` |
| `conversation/prep/review/frozen.sha256` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review/frozen.sha256` | 1092 | `9f282ada5dd308dd3b20f6c6ab9b041919addcd18ba1b9a8daae3658cf4bde8d` |
| `conversation/prep/review/ladder_used.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review/ladder_used.py` | 5379 | `33ab31ddb49efef68cdb618cbb025b978ae43a0f3abe23f2f934e1d9d71cb8dc` |
| `conversation/prep/review/oneshot.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review/oneshot.diff` | 1376 | `cc47e5f11e3a72be229f5a2fb956a2ac749f270aad73694064d72034a03414c3` |
| `conversation/prep/review/prompt.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review/prompt.txt` | 78234 | `e26a5ee90359692a299fc279780623b19014dca0e6ccf6b33da0c11aa8c9c4e1` |
| `conversation/prep/review/tests.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review/tests.txt` | 9 | `1e04dbfa7dc1165fddb246f5da4f18c8a7bbf521b5814d96d1729bbaebd57f74` |
| `conversation/prep/review2.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2.stderr.txt` | 228 | `207996a09c6f7284a435e2ccfcfbf672f17540940f953f267979ff59418efbbb` |
| `conversation/prep/review2.stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2.stdout.json` | 426 | `dc6520546a8d670989b5170f6bf37a3e9c838ae8e3a980fe568c8bf216820001` |
| `conversation/prep/review2/REQUEST.md` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2/REQUEST.md` | 2087 | `4c6f218ab6bcef4107295a089f2765d68db9adeafafc1a0a53ef3e0de3fdb6b4` |
| `conversation/prep/review2/conv_speed.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2/conv_speed.diff` | 1119 | `88a2c4572b3202c90aec7e1390de8b85fb51a5de8e41189ada9505765ed1b70f` |
| `conversation/prep/review2/conv_speed.py.reviewed` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2/conv_speed.py.reviewed` | 12609 | `206f27512dac5ea6b25bbba6728abde3aeb5edd9ec7d1f7f8bc9f085c1bfb538` |
| `conversation/prep/review2/frozen.sha256` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2/frozen.sha256` | 758 | `2bb264743420cd5758ff3a72325c58b852b90cbc641264a3d149ce8c811fd766` |
| `conversation/prep/review2/prompt.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2/prompt.txt` | 25570 | `b79d41622963e40e2dd7724fbb5b2a31f0e1b54d2db25f506c18be1d7b453682` |
| `conversation/prep/review2/run_conversation.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2/run_conversation.diff` | 715 | `081d074b903d255af95ae46c3c60df40be78f638531fd556edc09892f5e16754` |
| `conversation/prep/review2/run_conversation.sh.reviewed` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2/run_conversation.sh.reviewed` | 1318 | `b150b223adbeffa759aba247e93387ee086277d3afa2f708bc7a76a810760a00` |
| `conversation/prep/review2b.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2b.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/prep/review2b.stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/review2b.stdout.json` | 2527 | `d6511e32d7b748e9d564022816cf909c1567c3699a2b03b252ab0f4c764045a6` |
| `conversation/run_conversation.sh` | `/termux-home/ladder/run_conversation.sh` | 1411 | `a335b7633c9c0bce6d86744258e101c4b076982e618f6c434e34cc82ef6faee8` |
| `conversation/score_c.py` | `/termux-home/ladder/score_c.py` | 2702 | `49c4069c69f958d41f57fe5458276a3f33ce3abe3aa086c8d241e992ec6e3d84` |
| `memory_check/conv_cram0.stderr.txt` | `/termux-home/memcheck/conv_cram0.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/conv_cram0.stdout.txt` | `/termux-home/memcheck/conv_cram0.stdout.txt` | 28396 | `e1a5529cdf3bfe0db313aaf81b55f336ff650acaeb3c2b76c36904f5ff31ab52` |
| `memory_check/conv_cram0/meta.json` | `/termux-home/memcheck/conv_cram0/meta.json` | 703 | `d3576413a10ed73787fc6177fae8ba88802b1ea84173a53157d75b94a70ce77e` |
| `memory_check/conv_cram0/server.log` | `/termux-home/memcheck/conv_cram0/server.log` | 34011 | `1e589a78a3c13134ba01526a8d4bdfeaa7d279c9bbcad6de79b10bc30af63644` |
| `memory_check/conv_cram0/turns.jsonl` | `/termux-home/memcheck/conv_cram0/turns.jsonl` | 28396 | `e1a5529cdf3bfe0db313aaf81b55f336ff650acaeb3c2b76c36904f5ff31ab52` |
| `memory_check/conv_default.stderr.txt` | `/termux-home/memcheck/conv_default.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/conv_default.stdout.txt` | `/termux-home/memcheck/conv_default.stdout.txt` | 28777 | `c0444db2653af95cd77569a2fd3445573ab0a7368f53ea5bd8f8cf7e1647f7c2` |
| `memory_check/conv_default/meta.json` | `/termux-home/memcheck/conv_default/meta.json` | 676 | `3432a59f3a5aac226b95a30a861b2951b4c32dfc433d7a303f09e8579968bbeb` |
| `memory_check/conv_default/server.log` | `/termux-home/memcheck/conv_default/server.log` | 34071 | `2bf8e443932605119e8cbae64c47a3abba2384baa38d3c8e5b0a569757f9a671` |
| `memory_check/conv_default/turns.jsonl` | `/termux-home/memcheck/conv_default/turns.jsonl` | 28777 | `c0444db2653af95cd77569a2fd3445573ab0a7368f53ea5bd8f8cf7e1647f7c2` |
| `memory_check/memcheck.py` | `/termux-home/memcheck/memcheck.py` | 6732 | `04ee1f627f18a269cbab0736ad80b1cd57a9265edb2b9f87e34e800a7b74dc19` |
| `memory_check/memcheck_pilot.py` | `/termux-home/memcheck/memcheck_pilot.py` | 5784 | `d1d791e128196bdf76e9a7ea74c4140132dfe7c76c38db759328a742cafb8376` |
| `memory_check/pilot_robot_default.stderr.txt` | `/termux-home/memcheck/pilot_robot_default.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/pilot_robot_default.stdout.txt` | `/termux-home/memcheck/pilot_robot_default.stdout.txt` | 18276 | `b0cf63a3ea9fe5f525748285afddb65b7bffb1b86756a8be8aa4cf8493ef8121` |
| `memory_check/pilot_robot_default/meta.json` | `/termux-home/memcheck/pilot_robot_default/meta.json` | 657 | `58c13653247289542e6dd8ec163376345be1132fdd87e94d1af0e82a2f83b569` |
| `memory_check/pilot_robot_default/server.log` | `/termux-home/memcheck/pilot_robot_default/server.log` | 33711 | `c9f5836a1ef7f2b617d608592cb90455b2358b907cf39e0eb4ea3772b198e26d` |
| `memory_check/pilot_robot_default/turns.jsonl` | `/termux-home/memcheck/pilot_robot_default/turns.jsonl` | 18276 | `b0cf63a3ea9fe5f525748285afddb65b7bffb1b86756a8be8aa4cf8493ef8121` |
| `memory_check/robot_cram0.stderr.txt` | `/termux-home/memcheck/robot_cram0.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_cram0.stdout.txt` | `/termux-home/memcheck/robot_cram0.stdout.txt` | 18236 | `e8241b02f2e18e55ef55d8a3c7b12d8a300b01f793227b6bb726fecaef56aae2` |
| `memory_check/robot_cram0/meta.json` | `/termux-home/memcheck/robot_cram0/meta.json` | 704 | `f0d52feafd55f7b5b36f1ba793e2a07b59d223d8750c4cecf7556e3db0bebd84` |
| `memory_check/robot_cram0/server.log` | `/termux-home/memcheck/robot_cram0/server.log` | 33796 | `0df93cb88eae4b64f33d69de9b2fb2cec195fce9d575955f7d410d46e9b02112` |
| `memory_check/robot_cram0/turns.jsonl` | `/termux-home/memcheck/robot_cram0/turns.jsonl` | 18236 | `e8241b02f2e18e55ef55d8a3c7b12d8a300b01f793227b6bb726fecaef56aae2` |
| `memory_check/robot_cram0_pin47_a.stderr.txt` | `/termux-home/memcheck/robot_cram0_pin47_a.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_cram0_pin47_a.stdout.txt` | `/termux-home/memcheck/robot_cram0_pin47_a.stdout.txt` | 18284 | `ee50e45c50a7c6be91c41b7602087e967b89fe27333af10e5a623316dbff05ab` |
| `memory_check/robot_cram0_pin47_a/meta.json` | `/termux-home/memcheck/robot_cram0_pin47_a/meta.json` | 704 | `6b9fb808a6b4357a11c202fd0f738ebaf6ac4a2b8425e882e7497c1acfb434ea` |
| `memory_check/robot_cram0_pin47_a/server.log` | `/termux-home/memcheck/robot_cram0_pin47_a/server.log` | 33796 | `972e27e634738b985c47833daa9669d48044aa600ba083a69c01c6c24ce846df` |
| `memory_check/robot_cram0_pin47_a/turns.jsonl` | `/termux-home/memcheck/robot_cram0_pin47_a/turns.jsonl` | 18284 | `ee50e45c50a7c6be91c41b7602087e967b89fe27333af10e5a623316dbff05ab` |
| `memory_check/robot_cram0_pin47_b.stderr.txt` | `/termux-home/memcheck/robot_cram0_pin47_b.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_cram0_pin47_b.stdout.txt` | `/termux-home/memcheck/robot_cram0_pin47_b.stdout.txt` | 18253 | `edc882dfa7615c31c508a9a17bffb6851c334e1551ad0559b82cd15bacc7e30e` |
| `memory_check/robot_cram0_pin47_b/meta.json` | `/termux-home/memcheck/robot_cram0_pin47_b/meta.json` | 704 | `f3417474a78531c7cca6b2624eb029f9b7a22a0397ad23c0524d7acc48257ee7` |
| `memory_check/robot_cram0_pin47_b/server.log` | `/termux-home/memcheck/robot_cram0_pin47_b/server.log` | 33796 | `ffe65316c55ab55d9cb9025e7285e15a03f816d333afb78bce828dffeddc22ba` |
| `memory_check/robot_cram0_pin47_b/turns.jsonl` | `/termux-home/memcheck/robot_cram0_pin47_b/turns.jsonl` | 18253 | `edc882dfa7615c31c508a9a17bffb6851c334e1551ad0559b82cd15bacc7e30e` |
| `memory_check/robot_default.stderr.txt` | `/termux-home/memcheck/robot_default.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_default.stdout.txt` | `/termux-home/memcheck/robot_default.stdout.txt` | 18240 | `70ff989d83837d8c94f6c8442341238a0a16409f2b97d15ae5786e6c7e79d0bd` |
| `memory_check/robot_default/meta.json` | `/termux-home/memcheck/robot_default/meta.json` | 677 | `f375727004750ab3f1449755923de7ca1fa717b9a0d2963765e25cdac5804364` |
| `memory_check/robot_default/server.log` | `/termux-home/memcheck/robot_default/server.log` | 33711 | `66dec6a7a354e52d30ce7eae8c31387b7a3baab78d3a5b6dfe6703cd99b5f8d1` |
| `memory_check/robot_default/turns.jsonl` | `/termux-home/memcheck/robot_default/turns.jsonl` | 18240 | `70ff989d83837d8c94f6c8442341238a0a16409f2b97d15ae5786e6c7e79d0bd` |
| `memory_check/robot_default_pin47_a.stderr.txt` | `/termux-home/memcheck/robot_default_pin47_a.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_default_pin47_a.stdout.txt` | `/termux-home/memcheck/robot_default_pin47_a.stdout.txt` | 18255 | `dd81e2d52f58180c6c868323724952b1feadba6e28e49077cd6de05753ff667d` |
| `memory_check/robot_default_pin47_a/meta.json` | `/termux-home/memcheck/robot_default_pin47_a/meta.json` | 677 | `5f8b36a22c34f98446860729eb5b05f07cb0c0aa1db8179da79f3c45d370aaf9` |
| `memory_check/robot_default_pin47_a/server.log` | `/termux-home/memcheck/robot_default_pin47_a/server.log` | 33711 | `f1339029ec40dcf844462621a00d321991331a0473c24a8b0e2dfd63bdc05705` |
| `memory_check/robot_default_pin47_a/turns.jsonl` | `/termux-home/memcheck/robot_default_pin47_a/turns.jsonl` | 18255 | `dd81e2d52f58180c6c868323724952b1feadba6e28e49077cd6de05753ff667d` |
| `memory_check/robot_default_pin47_b.stderr.txt` | `/termux-home/memcheck/robot_default_pin47_b.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_default_pin47_b.stdout.txt` | `/termux-home/memcheck/robot_default_pin47_b.stdout.txt` | 18253 | `ad60c3da0efbce2ac6f0fad65a7d0fcf846600c7ee7e9b14db7e6d5fa1155c12` |
| `memory_check/robot_default_pin47_b/meta.json` | `/termux-home/memcheck/robot_default_pin47_b/meta.json` | 677 | `2cb9d2a1772d3e8c48c6eb97d030f41bfddb435cae89a682e25e5f2eabf4e82c` |
| `memory_check/robot_default_pin47_b/server.log` | `/termux-home/memcheck/robot_default_pin47_b/server.log` | 33711 | `b4bc0fc54c6397c487dee3b1f84777bb2400e47701addbc09c05e7c6d0caabc8` |
| `memory_check/robot_default_pin47_b/turns.jsonl` | `/termux-home/memcheck/robot_default_pin47_b/turns.jsonl` | 18253 | `ad60c3da0efbce2ac6f0fad65a7d0fcf846600c7ee7e9b14db7e6d5fa1155c12` |
| `memory_check/run_all.log` | `/termux-home/memcheck/run_all.log` | 1367 | `0463f6f0a08d3974f54e0bca36b674c345aa27cff81606f174e9afec98afb788` |
| `memory_check/run_all.sh` | `/termux-home/memcheck/run_all.sh` | 755 | `5d5f56ea30bec7381a632ece1099a9a78eb7293046db693289a7548ed2b34a4c` |
| `memory_check/summarize.py` | `/termux-home/memcheck/summarize.py` | 1566 | `4edfa9cffe941c96b58f5ab0a31701281f33f2f8d0995629f1e44951aaea5859` |
| `memory_check/summary.md` | `/termux-home/memcheck/summary.md` | 1483 | `5c7b2f7d34b9a751ab5cc430bf6709a89352635db54df9d4d44692d464e36937` |
| `memory_check/timeout_check.py` | `/termux-home/memcheck/timeout_check.py` | 1734 | `691e4cd58baa775e3c6a29ca4527825c925f48e6643c1224d016b709ff4fd8a9` |
| `memory_check/timeout_check.stderr.txt` | `/termux-home/memcheck/timeout_check.stderr.txt` | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` |
| `memory_check/timeout_check.stdout.txt` | `/termux-home/memcheck/timeout_check.stdout.txt` | 2645 | `2d2ebc8cc13b1fc651932b092be857a97807f5a6dff1efc7062b089308d7a630` |
| `memory_check/timeout_check.thermal.txt` | `/termux-home/memcheck/timeout_check.thermal.txt` | 271 | `e19078c7d960790b4948250c251bbcb3e0ba0b9b8735ddfeeb3ce603839fd41d` |
| `memory_check/timeout_check_warmup.py` | `/termux-home/memcheck/timeout_check_warmup.py` | 1932 | `350f8d7a0fb023443ff211c8c2bd200016ec65f271e4a76f535e58ff97cacf94` |
| `memory_check/timeout_check_warmup.stderr.txt` | `/termux-home/memcheck/timeout_check_warmup.stderr.txt` | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` |
| `memory_check/timeout_check_warmup.stdout.txt` | `/termux-home/memcheck/timeout_check_warmup.stdout.txt` | 2707 | `8847dd55dd356e8f68b880a0dc6665cb8171ee6caead009ea6066aa82b6beb98` |
| `memory_check/timeout_check_warmup.thermal.txt` | `/termux-home/memcheck/timeout_check_warmup.thermal.txt` | 131 | `21c5bc61fd8ff21851e252bc41ed120b13d82d4ad1be61d5ead0e90b169e4b34` |
| `memory_check/timeout_check_warmup2.stderr.txt` | `/termux-home/memcheck/timeout_check_warmup2.stderr.txt` | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` |
| `memory_check/timeout_check_warmup2.stdout.txt` | `/termux-home/memcheck/timeout_check_warmup2.stdout.txt` | 2709 | `0f43664dd893fbabb407722026fb54656bb85584a340938c5bb3759129df30fb` |
| `memory_check/timeout_check_warmup2.thermal.txt` | `/termux-home/memcheck/timeout_check_warmup2.thermal.txt` | 131 | `cf6ea07f7455331749c96e60ee4212a009f79d7f2fcc205c902ab264d611fac0` |
| `memory_check/warmup_no_server.txt` | `/termux-home/memcheck/warmup_no_server.txt` | 91 | `cba2093f73f43e4925274d060363d013ba0e4247a1536d0031e762cb9bec3048` |
| `reports/CODER_REPORT_conversation_prep.md` | `/termux-home/archive-staging/reports/CODER_REPORT_conversation_prep.md` | 12223 | `984bd324294b78dd979af320ad240f5ec34c35f2f3912c1e0408e190fdabd822` |

## Written for this archive or already in place in the repository (5 files)

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `ARTIFACTS.md` | 3585 | `d829ff5c7f9df9af5d9ce3b1b836813197dae6b28bc406d53fb6ed963942e688` |
| `README.md` | 15250 | `f4e7f0c2d002b12c2301fdd182c305731681aec60930391301a4b5ea397f6121` |
| `RUN_INDEX.md` | 4030 | `cd39ed7ff5f48febea1ddc702ca2e913afdb26d8bb2a909c618174ebf06c672c` |
| `bench.py` | 27801 | `0792beeed813d94df2a1d9ce7bffe2f5e2a00a564de4a32aaecfa0182b8e6f8f` |
| `test_bench.py` | 1374 | `123484c38cea33f7a00fdbbcfecbedfb07129ca116a9a8db73a40a2de37543ae` |

README.md re-hashed 2026-10-01 after the heat banner (03ab5b5); previous SHA-256 e2861e168a574fd82a7014b6fb589a40fbc06853b4a9a09047d8552469bf2cb7.

## sha256sum format

```sha256sums
d829ff5c7f9df9af5d9ce3b1b836813197dae6b28bc406d53fb6ed963942e688  ARTIFACTS.md
f4e7f0c2d002b12c2301fdd182c305731681aec60930391301a4b5ea397f6121  README.md
cd39ed7ff5f48febea1ddc702ca2e913afdb26d8bb2a909c618174ebf06c672c  RUN_INDEX.md
0792beeed813d94df2a1d9ce7bffe2f5e2a00a564de4a32aaecfa0182b8e6f8f  bench.py
4b578dc988a24575da053b2df4a756c99f7ab9053c4f9a3fd4812ac9c4f4b4c6  conversation/conv_quality.py
9b721d8678f53a122f022f8bf66ee318707aa1700b5c684ca144fa06f7b21e7a  conversation/conv_speed.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  conversation/conversation_20260926T185957Z.stderr.txt
da0c071404d7297425e16e3438e5c9521156e52bc6e80380109be52b8430022c  conversation/conversation_20260926T185957Z.stdout.txt
441b0730b932a72f9c271a2c22c93e69e18722cc89454bafcf2f24d53a3ad5b9  conversation/conversation_20260926T185957Z/blocks.jsonl
c150ec7322bfb0c30961d41156aead34ca0f41906e98663f9493a26304e8c2fa  conversation/conversation_20260926T185957Z/logs/gemma_e2b_q40.cached.server.log
7210a92f32a2edc0c442829a1597e91aba9909916b3385e1a665bd25f78d49da  conversation/conversation_20260926T185957Z/logs/gemma_e2b_q40.cold.server.log
a39b643e578f5ebef95658d778b5053935249878da6c3e0224822228e4533c67  conversation/conversation_20260926T185957Z/logs/gemma_e2b_q4km.cached.server.log
7a5c8078950272ecd16588c76dc10e11e2023dc6dae2b030f7112ecbfe280ade  conversation/conversation_20260926T185957Z/logs/gemma_e2b_q4km.cold.server.log
8b7d8c05b62aa98ca36f0db34896acf5365a40ff10c269bc6e3193057b9c50f5  conversation/conversation_20260926T185957Z/logs/gemma_e4b_q4km.cached.server.log
4572ef516f60f01a295d89ae2a275f9f31c122a9c6f25bee65b6beafb6e456ba  conversation/conversation_20260926T185957Z/logs/gemma_e4b_q4km.cold.server.log
130c296031e5e2940ace4a0d0077f45321d6bea4b4a8f23fde9f2058cdad332c  conversation/conversation_20260926T185957Z/logs/qwen35_2b.cached.server.log
dc09e176671262a3474f60919055d5c9c52693d9132a7cec0204a0c74a3769ad  conversation/conversation_20260926T185957Z/logs/qwen35_2b.cold.server.log
9d8d7b45a26f5ebabf5b48663fd95e454fed3bfe0e8c2611cc773e4e71dfd368  conversation/conversation_20260926T185957Z/logs/qwen35_4b.cached.server.log
ee0a95acfe219f0150964bddf4badf464cf3085dfde350d53169394b5087235b  conversation/conversation_20260926T185957Z/logs/qwen35_4b.cold.server.log
3a917faeb494ac9b24c8e3aad3c6ea35da788117f64b4a97206decd9214796ef  conversation/conversation_20260926T185957Z/report.txt
f894d112211ce542aa55feddccfc92112cc8371580cc0953b5ef6af9b181fa45  conversation/conversation_20260926T185957Z/run_20260926T185957Z.json
6b3e7da5101ddf45e4a8698d21c545a7e925981dd16ac8ae82e7455732a14b35  conversation/conversation_20260926T185957Z/turns.jsonl
b8657ac39d1eaf054e334d5827bf6ae6c0fe7205f46529406d3352ffb36cb77f  conversation/conversation_blind_key.tsv
0c1678c7d0af08dcbdd660ebbf5cdafb973f0c4ee3b6abce46723b4e71fb684e  conversation/conversation_blind_sheet.md
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  conversation/conversation_quality.stderr.txt
2921ef45759d2c34e42137872a24663e187c1008c7bc85d0ef9ba8b104882f5b  conversation/conversation_quality.stdout.txt
c7d19e8185e535453de0d32e0068bc37f5b5af11cdd35e7a05d304f73b2e2c3e  conversation/conversation_quality/bench_blind_key.txt
c3203ad6c26a438e9fe36d3c3614af89eaae0a5fd4603761b5f2a7df08564f96  conversation/conversation_quality/bench_blind_transcripts.txt
b63dc5b8ef89e2f62c450d7547884451f2d21109e34cb38f27efbdb5af4e6dc9  conversation/conversation_quality/bench_c_summary.txt
4d55c8c591bce02d28ce38fb42d7257887e6883932a8825cfaefc9304743567e  conversation/conversation_quality/bench_gemma_e2b_q40_20260926_163419.log
4f8fcf85b59f7a4d305b96e2d11d6fba14f7f914754b10d81f0c8d34af0306f1  conversation/conversation_quality/bench_gemma_e2b_q4km_20260926_163419.log
3848f700485b8c85c4e35bc6abcd39ce96fc95afd2c52d24e60aec0a94eb9372  conversation/conversation_quality/bench_gemma_e4b_q4km_20260926_163419.log
d30497d11b584058aae2965b4b2d1adb61d0c3f997366f85adf6fada13c9fe2e  conversation/conversation_quality/bench_qwen35_2b_20260926_163419.log
e09b3605ce7232f8817359145c99a228715223cf0b1da0562a2a2ba71ab5c5e3  conversation/conversation_quality/bench_qwen35_4b_20260926_163419.log
398e36f554860da02220dd183296d8b3d9f6c56bd24e480c8693af73a9465e72  conversation/conversation_quality/bench_results_20260926_163419.json
452ba4adc348e01de858c43b322145cab603af188501620c45e4ec71a8588e45  conversation/conversation_quality/bench_speed_summary.txt
a655686de7f4e6bea222994cc2b4a5c5114b286ee97b5265d0a154072e70d6cd  conversation/conversation_quality_combined.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734.stderr.txt
469a7584dc4d8e95ca8ea50fd8a9b4530fa5388c7ca8f678c10d276de3709e22  conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734.stdout.txt
952a7cb1ca044aba075bf1e1406edf1d0adab1ba8a90f9b18352daca7d5c0f64  conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734/bench_blind_key.txt
825416d1b4fe097b0a95f41ea6e7d5ff8848b5970f517c4175546b67b73ac83b  conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734/bench_qwen35_4b_20260926_173003.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  conversation/conversation_quality_qwen35_4b.stderr.txt
0dc52c7bd51cc38d26854530a7bc6b572efdf13f4cc5ead25a3c7a8600d99248  conversation/conversation_quality_qwen35_4b.stdout.txt
952a7cb1ca044aba075bf1e1406edf1d0adab1ba8a90f9b18352daca7d5c0f64  conversation/conversation_quality_qwen35_4b/bench_blind_key.txt
09cb67e2559bb953dfc331771cf8c130e6906195f83b3d520311d6cb6d6e3414  conversation/conversation_quality_qwen35_4b/bench_blind_transcripts.txt
44f2cac264f738f8ac3a97a516e07736402756f5b356a545b7a9d1025dce7f8e  conversation/conversation_quality_qwen35_4b/bench_c_summary.txt
3b1aeb8b0a5e9275fd7ba790d63c1538cbae4b29ed8c22dc46c12086b16ea106  conversation/conversation_quality_qwen35_4b/bench_qwen35_4b_20260926_182310.log
1458ea65de10333e8d2142a8eece3ae04e18ec91c5dca5ec5c91bfa245bcb6c1  conversation/conversation_quality_qwen35_4b/bench_results_20260926_182310.json
b2ec62f069f96fc976807ab0655e1bce3f056b17748d09419672cbb8c0795843  conversation/conversation_quality_qwen35_4b/bench_speed_summary.txt
8002085be7a0327a27fd56a10fb9d3e1f816a1bfdc1480214654c0169418ac58  conversation/conversation_quality_scores.txt
0da2f2b8c6af6445b9b06749ead6f5444a57fbc5e96fb4ebd56b61daf859bd59  conversation/local_ai_blind_scores.tsv
c698988cf418d37bf1388816cfefa8624a0e8cfde377094c92837ce682bbde49  conversation/make_blind_sheet.py
2c2bc7f88c40b6c0642fd1e6706af8261661329127b94d3ef2cbb56db9ea1900  conversation/prep/full_quality.sh
74132400e1afba33690ca47bd8b7c94ed9efe79905b3a55f58a183d7174dae31  conversation/prep/mem_diag.py
22ee65f38df9cb8a02f806e0f980702df057d8dd9a38a8255ca65be0a4fb5d6b  conversation/prep/mem_diag.server.log
7aa5138626d3db8b13cb1069f957373d58f1aa1765a80765393c2e4e83968db7  conversation/prep/mem_diag_cr0.server.log
ff4449216f382be39f938ef289309cce8f59474fe32103c5214d805789776900  conversation/prep/mem_diag_cr0.txt
0edcae559ea922a7be9ceefc9b4e20e5191eb8526494929046eb44103232a456  conversation/prep/oneshot.sh.orig
22da3105d56434889fd3a0f8c3a9481f67dfc32ebef2647acb5c8babcb86ce86  conversation/prep/quick.sh
d9100bdf12b3516463943d3782d581f9f14f76de46ddd694c0ed86ada744aca2  conversation/prep/quick_key.tsv
c0e405e6b024cc19c9e6b20f4369cbf8fd2d2d085af11af0f8b376553803eb83  conversation/prep/quick_quality.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  conversation/prep/quick_quality.stderr.txt
6a91b99d5f486013928d9efae66cd6a65f11a2855a46d515942c5bddafcc2ec0  conversation/prep/quick_quality.stdout.txt
e342699140a128ada2b3aebad36eb0e0e5d925f8944f734ac31582b68c35159a  conversation/prep/quick_quality/bench_blind_key.txt
f4b101ec3e2b7da9b1c2463376203501743d82cd6ddbabdcc1fd54b3205228b5  conversation/prep/quick_quality/bench_blind_transcripts.txt
388b6e90fe58a595282e1498e0c5742a10c6dac79be95394a97f9d151291315a  conversation/prep/quick_quality/bench_c_summary.txt
0a61cf78ae265008375056dc79f468f0afe5bc36c64e90e64eab2c62ab905fd8  conversation/prep/quick_quality/bench_gemma_e2b_q40_20260926_162121.log
8777aad501da238ce6f1331d4b2b0d21c40935973f56d88ebd932b243214d548  conversation/prep/quick_quality/bench_gemma_e2b_q4km_20260926_162121.log
9ccd6e607f0a524301ee1be3faf3bce824ec3c4b3182aa1c1bc57df9c97a0009  conversation/prep/quick_quality/bench_gemma_e4b_q4km_20260926_162121.log
e9cddea81b90a0c87618d2a5f319bccbcd4905638bbb80e479c50c3257e17bb6  conversation/prep/quick_quality/bench_qwen35_2b_20260926_162121.log
dd75299f97a6d15d93fb5794dca7843596dba9623044239e7759ec856e5a258e  conversation/prep/quick_quality/bench_qwen35_4b_20260926_162121.log
d523e21edd6be12286918327120f1920c83a57de0bf1452edd6b9b6d6cda15c7  conversation/prep/quick_quality/bench_results_20260926_162121.json
3bbcd02d370f48510ecad6e0729477d1cc9695081f6ddc8b121a61e3a1c164dd  conversation/prep/quick_quality/bench_speed_summary.txt
65ffabf6b6e51d211ec94e2f227513f4cad2dc747fac2d2e8f43efbe7bec57d9  conversation/prep/quick_sheet.md
8ce982e5ae37a79b60c41f9880d7df7402e514fed18352e8fae9eb641e35e522  conversation/prep/quick_speed.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  conversation/prep/quick_speed.stderr.txt
7c754c14c62acf5ef0a8f9e71ce713e53e1a9eeb831bb20549ac2aff688aa27c  conversation/prep/quick_speed.stdout.txt
2f1d449b9701eb7f59cb59d5fbe534fbe8b649f87640c8bf63d1c15e4f361211  conversation/prep/quick_speed/blocks.jsonl
c2de57f3b03c9fcfdb58396cff79e44badbd81677b161985fffd29756701bd81  conversation/prep/quick_speed/logs/gemma_e2b_q40.cached.server.log
731667e68b1299dd5a9024fe079c60c985602fd81e39652dad008a4697686bf2  conversation/prep/quick_speed/logs/gemma_e2b_q40.cold.server.log
13b2f94115ce90839d7f568dcc7da4c3dcb182d5daaedb902bd82f5a82dbc901  conversation/prep/quick_speed/logs/gemma_e2b_q4km.cached.server.log
710498599567aadc391c24239f8b87b6b00ff320ff90bf670ee0ffa691d16a2e  conversation/prep/quick_speed/logs/gemma_e2b_q4km.cold.server.log
7af4f9bbcc905f6218ecc415da21ff9eb2ffd49c991193ef8b97e136bb0f0e9c  conversation/prep/quick_speed/logs/gemma_e4b_q4km.cached.server.log
51f21c0a2f3c446d5d2b7062a2e36a1b4b0719331b5136d4f7af79b3bfbb6355  conversation/prep/quick_speed/logs/gemma_e4b_q4km.cold.server.log
1e7211b934ec37bc35a93be3d48cb0b01f5299fc895878b83e833ab39575b1ba  conversation/prep/quick_speed/logs/qwen35_2b.cached.server.log
c730e022bba8d6b9d5aba24b6fedf8e5ca0b032c1d643ccfb4b1d19da7ad20a0  conversation/prep/quick_speed/logs/qwen35_2b.cold.server.log
2c8753e8ba22f002d0f6b2359e2e5a0d3d2c732eeb4de0e62c8c601bc6907e47  conversation/prep/quick_speed/logs/qwen35_4b.cached.server.log
daf4bde5d65cd76cd63dc953e0bc2a965814ca95974be20a8a743f73c29c973a  conversation/prep/quick_speed/logs/qwen35_4b.cold.server.log
ac89115e42a55a0516659f5b08a768fcb3f935eee35f71f2e67c2924c2c953f4  conversation/prep/quick_speed/report.txt
3e65d3d38cead9a893f57c19ec4dc8d3d6e29248df7cc8328ca7ac8009624652  conversation/prep/quick_speed/run_20260926T162800Z.json
c0ee547e120f14e071cad0b11ab4bf5ec7439fd2c51b46345def18e674f77479  conversation/prep/quick_speed/turns.jsonl
be53afcba1ceb663d0792e270124f3326714b6985d3cd8bc3c5a2921fdc042f8  conversation/prep/qwen4b_a7.py
5a2a47c8575a279d7f235a2ffb90aad37275dd27b9f22d18b8c48e5a7d8d7300  conversation/prep/qwen4b_a7.server.log
1ca302eeb20a186d77b5b16db852b30a8b3f4529f53913475f0e9cd63fec7eea  conversation/prep/rerun_qwen4b.sh
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  conversation/prep/review.stderr.txt
f87fddcf0919bd7cacbe35e12d678b4afb00258897af8f5cdf0e29e517a976e8  conversation/prep/review.stdout.json
6a76038f342c5a5752b4dae242375fbbed39b82b73d01b9cbea5b16f914b6890  conversation/prep/review/REQUEST.md
fd8fbf41b0b2921f5db5341f058d9e9676c44bdc8aa0e8f9e8f1ee7e2e20eed2  conversation/prep/review/bench_rubric.diff
9f282ada5dd308dd3b20f6c6ab9b041919addcd18ba1b9a8daae3658cf4bde8d  conversation/prep/review/frozen.sha256
33ab31ddb49efef68cdb618cbb025b978ae43a0f3abe23f2f934e1d9d71cb8dc  conversation/prep/review/ladder_used.py
cc47e5f11e3a72be229f5a2fb956a2ac749f270aad73694064d72034a03414c3  conversation/prep/review/oneshot.diff
e26a5ee90359692a299fc279780623b19014dca0e6ccf6b33da0c11aa8c9c4e1  conversation/prep/review/prompt.txt
1e04dbfa7dc1165fddb246f5da4f18c8a7bbf521b5814d96d1729bbaebd57f74  conversation/prep/review/tests.txt
207996a09c6f7284a435e2ccfcfbf672f17540940f953f267979ff59418efbbb  conversation/prep/review2.stderr.txt
dc6520546a8d670989b5170f6bf37a3e9c838ae8e3a980fe568c8bf216820001  conversation/prep/review2.stdout.json
4c6f218ab6bcef4107295a089f2765d68db9adeafafc1a0a53ef3e0de3fdb6b4  conversation/prep/review2/REQUEST.md
88a2c4572b3202c90aec7e1390de8b85fb51a5de8e41189ada9505765ed1b70f  conversation/prep/review2/conv_speed.diff
206f27512dac5ea6b25bbba6728abde3aeb5edd9ec7d1f7f8bc9f085c1bfb538  conversation/prep/review2/conv_speed.py.reviewed
2bb264743420cd5758ff3a72325c58b852b90cbc641264a3d149ce8c811fd766  conversation/prep/review2/frozen.sha256
b79d41622963e40e2dd7724fbb5b2a31f0e1b54d2db25f506c18be1d7b453682  conversation/prep/review2/prompt.txt
081d074b903d255af95ae46c3c60df40be78f638531fd556edc09892f5e16754  conversation/prep/review2/run_conversation.diff
b150b223adbeffa759aba247e93387ee086277d3afa2f708bc7a76a810760a00  conversation/prep/review2/run_conversation.sh.reviewed
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  conversation/prep/review2b.stderr.txt
d6511e32d7b748e9d564022816cf909c1567c3699a2b03b252ab0f4c764045a6  conversation/prep/review2b.stdout.json
a335b7633c9c0bce6d86744258e101c4b076982e618f6c434e34cc82ef6faee8  conversation/run_conversation.sh
49c4069c69f958d41f57fe5458276a3f33ce3abe3aa086c8d241e992ec6e3d84  conversation/score_c.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  memory_check/conv_cram0.stderr.txt
e1a5529cdf3bfe0db313aaf81b55f336ff650acaeb3c2b76c36904f5ff31ab52  memory_check/conv_cram0.stdout.txt
d3576413a10ed73787fc6177fae8ba88802b1ea84173a53157d75b94a70ce77e  memory_check/conv_cram0/meta.json
1e589a78a3c13134ba01526a8d4bdfeaa7d279c9bbcad6de79b10bc30af63644  memory_check/conv_cram0/server.log
e1a5529cdf3bfe0db313aaf81b55f336ff650acaeb3c2b76c36904f5ff31ab52  memory_check/conv_cram0/turns.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  memory_check/conv_default.stderr.txt
c0444db2653af95cd77569a2fd3445573ab0a7368f53ea5bd8f8cf7e1647f7c2  memory_check/conv_default.stdout.txt
3432a59f3a5aac226b95a30a861b2951b4c32dfc433d7a303f09e8579968bbeb  memory_check/conv_default/meta.json
2bf8e443932605119e8cbae64c47a3abba2384baa38d3c8e5b0a569757f9a671  memory_check/conv_default/server.log
c0444db2653af95cd77569a2fd3445573ab0a7368f53ea5bd8f8cf7e1647f7c2  memory_check/conv_default/turns.jsonl
04ee1f627f18a269cbab0736ad80b1cd57a9265edb2b9f87e34e800a7b74dc19  memory_check/memcheck.py
d1d791e128196bdf76e9a7ea74c4140132dfe7c76c38db759328a742cafb8376  memory_check/memcheck_pilot.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  memory_check/pilot_robot_default.stderr.txt
b0cf63a3ea9fe5f525748285afddb65b7bffb1b86756a8be8aa4cf8493ef8121  memory_check/pilot_robot_default.stdout.txt
58c13653247289542e6dd8ec163376345be1132fdd87e94d1af0e82a2f83b569  memory_check/pilot_robot_default/meta.json
c9f5836a1ef7f2b617d608592cb90455b2358b907cf39e0eb4ea3772b198e26d  memory_check/pilot_robot_default/server.log
b0cf63a3ea9fe5f525748285afddb65b7bffb1b86756a8be8aa4cf8493ef8121  memory_check/pilot_robot_default/turns.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  memory_check/robot_cram0.stderr.txt
e8241b02f2e18e55ef55d8a3c7b12d8a300b01f793227b6bb726fecaef56aae2  memory_check/robot_cram0.stdout.txt
f0d52feafd55f7b5b36f1ba793e2a07b59d223d8750c4cecf7556e3db0bebd84  memory_check/robot_cram0/meta.json
0df93cb88eae4b64f33d69de9b2fb2cec195fce9d575955f7d410d46e9b02112  memory_check/robot_cram0/server.log
e8241b02f2e18e55ef55d8a3c7b12d8a300b01f793227b6bb726fecaef56aae2  memory_check/robot_cram0/turns.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  memory_check/robot_cram0_pin47_a.stderr.txt
ee50e45c50a7c6be91c41b7602087e967b89fe27333af10e5a623316dbff05ab  memory_check/robot_cram0_pin47_a.stdout.txt
6b9fb808a6b4357a11c202fd0f738ebaf6ac4a2b8425e882e7497c1acfb434ea  memory_check/robot_cram0_pin47_a/meta.json
972e27e634738b985c47833daa9669d48044aa600ba083a69c01c6c24ce846df  memory_check/robot_cram0_pin47_a/server.log
ee50e45c50a7c6be91c41b7602087e967b89fe27333af10e5a623316dbff05ab  memory_check/robot_cram0_pin47_a/turns.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  memory_check/robot_cram0_pin47_b.stderr.txt
edc882dfa7615c31c508a9a17bffb6851c334e1551ad0559b82cd15bacc7e30e  memory_check/robot_cram0_pin47_b.stdout.txt
f3417474a78531c7cca6b2624eb029f9b7a22a0397ad23c0524d7acc48257ee7  memory_check/robot_cram0_pin47_b/meta.json
ffe65316c55ab55d9cb9025e7285e15a03f816d333afb78bce828dffeddc22ba  memory_check/robot_cram0_pin47_b/server.log
edc882dfa7615c31c508a9a17bffb6851c334e1551ad0559b82cd15bacc7e30e  memory_check/robot_cram0_pin47_b/turns.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  memory_check/robot_default.stderr.txt
70ff989d83837d8c94f6c8442341238a0a16409f2b97d15ae5786e6c7e79d0bd  memory_check/robot_default.stdout.txt
f375727004750ab3f1449755923de7ca1fa717b9a0d2963765e25cdac5804364  memory_check/robot_default/meta.json
66dec6a7a354e52d30ce7eae8c31387b7a3baab78d3a5b6dfe6703cd99b5f8d1  memory_check/robot_default/server.log
70ff989d83837d8c94f6c8442341238a0a16409f2b97d15ae5786e6c7e79d0bd  memory_check/robot_default/turns.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  memory_check/robot_default_pin47_a.stderr.txt
dd81e2d52f58180c6c868323724952b1feadba6e28e49077cd6de05753ff667d  memory_check/robot_default_pin47_a.stdout.txt
5f8b36a22c34f98446860729eb5b05f07cb0c0aa1db8179da79f3c45d370aaf9  memory_check/robot_default_pin47_a/meta.json
f1339029ec40dcf844462621a00d321991331a0473c24a8b0e2dfd63bdc05705  memory_check/robot_default_pin47_a/server.log
dd81e2d52f58180c6c868323724952b1feadba6e28e49077cd6de05753ff667d  memory_check/robot_default_pin47_a/turns.jsonl
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  memory_check/robot_default_pin47_b.stderr.txt
ad60c3da0efbce2ac6f0fad65a7d0fcf846600c7ee7e9b14db7e6d5fa1155c12  memory_check/robot_default_pin47_b.stdout.txt
2cb9d2a1772d3e8c48c6eb97d030f41bfddb435cae89a682e25e5f2eabf4e82c  memory_check/robot_default_pin47_b/meta.json
b4bc0fc54c6397c487dee3b1f84777bb2400e47701addbc09c05e7c6d0caabc8  memory_check/robot_default_pin47_b/server.log
ad60c3da0efbce2ac6f0fad65a7d0fcf846600c7ee7e9b14db7e6d5fa1155c12  memory_check/robot_default_pin47_b/turns.jsonl
0463f6f0a08d3974f54e0bca36b674c345aa27cff81606f174e9afec98afb788  memory_check/run_all.log
5d5f56ea30bec7381a632ece1099a9a78eb7293046db693289a7548ed2b34a4c  memory_check/run_all.sh
4edfa9cffe941c96b58f5ab0a31701281f33f2f8d0995629f1e44951aaea5859  memory_check/summarize.py
5c7b2f7d34b9a751ab5cc430bf6709a89352635db54df9d4d44692d464e36937  memory_check/summary.md
691e4cd58baa775e3c6a29ca4527825c925f48e6643c1224d016b709ff4fd8a9  memory_check/timeout_check.py
1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca  memory_check/timeout_check.stderr.txt
2d2ebc8cc13b1fc651932b092be857a97807f5a6dff1efc7062b089308d7a630  memory_check/timeout_check.stdout.txt
e19078c7d960790b4948250c251bbcb3e0ba0b9b8735ddfeeb3ce603839fd41d  memory_check/timeout_check.thermal.txt
350f8d7a0fb023443ff211c8c2bd200016ec65f271e4a76f535e58ff97cacf94  memory_check/timeout_check_warmup.py
1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca  memory_check/timeout_check_warmup.stderr.txt
8847dd55dd356e8f68b880a0dc6665cb8171ee6caead009ea6066aa82b6beb98  memory_check/timeout_check_warmup.stdout.txt
21c5bc61fd8ff21851e252bc41ed120b13d82d4ad1be61d5ead0e90b169e4b34  memory_check/timeout_check_warmup.thermal.txt
1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca  memory_check/timeout_check_warmup2.stderr.txt
0f43664dd893fbabb407722026fb54656bb85584a340938c5bb3759129df30fb  memory_check/timeout_check_warmup2.stdout.txt
cf6ea07f7455331749c96e60ee4212a009f79d7f2fcc205c902ab264d611fac0  memory_check/timeout_check_warmup2.thermal.txt
cba2093f73f43e4925274d060363d013ba0e4247a1536d0031e762cb9bec3048  memory_check/warmup_no_server.txt
984bd324294b78dd979af320ad240f5ec34c35f2f3912c1e0408e190fdabd822  reports/CODER_REPORT_conversation_prep.md
123484c38cea33f7a00fdbbcfecbedfb07129ca116a9a8db73a40a2de37543ae  test_bench.py
```
