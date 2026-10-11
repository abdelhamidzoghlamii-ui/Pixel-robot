# Archived phone artifacts

This archive contains the final available harness and every `~/bench_results_*.json` found in the phone home directory on 2026-09-23. It deliberately departs from the strategic selector's hashes-only phone-history precedent: DECISIONS #110 must be independently checkable from raw rows, and a successor thread needs the replies to reproduce the strict, lenient and corrected-refusal regrades. Only `bench.py` and JSON results were copied. No GGUF weights, caches, logs or `bench.py.bak` are in Git.

Each archive file below matched the phone original byte-for-byte after copying. `cmp` succeeded, sizes agree, and phone/archive SHA-256 values match.

| Archived artifact | Bytes | Phone SHA-256 | Archive SHA-256 | Match |
|---|---:|---|---|---|
| [`runs/bench.py`](runs/bench.py) | 26982 | `1e1b506c3276222b470125fa551d9afa692f98b9e5701ca351ce975f64c06080` | `1e1b506c3276222b470125fa551d9afa692f98b9e5701ca351ce975f64c06080` | YES |
| [`runs/bench_results_20260919_083347.json`](runs/bench_results_20260919_083347.json) — gemma-e2b, 81 rows | 109679 | `8a46ce308a54dd2eeac2340a89a04bb14ba886f5399092fef3c3b2558d3fda11` | `8a46ce308a54dd2eeac2340a89a04bb14ba886f5399092fef3c3b2558d3fda11` | YES |
| [`runs/bench_results_20260919_115350.json`](runs/bench_results_20260919_115350.json) — gemma-e4b, 81 rows | 111778 | `e228bf0e2b222b481a8c18ffc655f1abef9a321dc512d82cd8566aa2c5098a47` | `e228bf0e2b222b481a8c18ffc655f1abef9a321dc512d82cd8566aa2c5098a47` | YES |
| [`runs/bench_results_20260921_054508.json`](runs/bench_results_20260921_054508.json) — qwen3.5-2B-think-off, 27 rows | 36800 | `4c85fc58ead899b41c84f3327c7a46f90b140f4b564b9a71c0a2e36100df5358` | `4c85fc58ead899b41c84f3327c7a46f90b140f4b564b9a71c0a2e36100df5358` | YES |
| [`runs/bench_results_20260921_164938.json`](runs/bench_results_20260921_164938.json) — qwen3.5-2B-think-on, 81 rows | 235645 | `804cade9a772b9f2cb415c5843621edc70d8ecff37487f3a7e966081ea7ae6d6` | `804cade9a772b9f2cb415c5843621edc70d8ecff37487f3a7e966081ea7ae6d6` | YES |

No other `~/bench_results_*.json` files were present. Qwen2.5-3B and both Qwen3.5-4B modes therefore have no raw result artifact.

## Provenance boundary

The pre-patch `~/bench.py.bak` hash was verified on the phone as `dc53c4d15bdb9be69178d831f045d74d3dc6cf98a53d7c0b0ca7c2675996f442`; it is not archived. The copied final harness compiles and has the reported post-patch hash. However, none of the result JSONs records `source_sha256`. The September 19 Gemma results predate the final harness mtime; the September 21 Qwen results follow it. File times and successful handling of the former crash case support, but do not prove, grader-version association. Exact per-run source identity is **unverified**.

The original blind transcript/key and server logs were not specified as durable source artifacts. Their exact run mapping and hashes remain unverified. If they are archived later, preserve original bytes and add hashes without altering these raw JSONs.

## Recheck

From `benchmark/llm_objective_setting`:

```bash
stat -c '%s %n' runs/bench.py runs/bench_results_*.json
sha256sum runs/bench.py runs/bench_results_*.json
python -m py_compile runs/bench.py aggregate.py
python aggregate.py
```

`aggregate.py` reads the JSONs and writes only derived, ignored files in this directory. It does not run inference, contact a network, alter raw evidence or touch hardware.

Additions of 2026-09-26 (rubric-fixed rerun, conversation grading and speed, server memory check) are hashed in [ARTIFACTS_2026-09-26.md](ARTIFACTS_2026-09-26.md).

## P23D current inventory

Historical sections above retain their exact original archive-time records; some describe older code or documentation bytes. The table below is the current file inventory, including required ignored artifacts explicitly force-added for review. It excludes this manifest, Python caches and the new frozen manifest to avoid self-reference.

| Archive path | Source | Bytes | SHA-256 |
|---|---|---:|---|
| `ARTIFACTS_2026-09-26.md` | `existing repository / P23D generated documentation or offline evidence` | 60453 | `73db90bec20e395b44c95e3e3d27862dacaa444a8ede5bcff531e16f23f5d1b6` |
| `README.md` | `existing repository / P23D generated documentation or offline evidence` | 15544 | `3266fde30c7e5345779e86f00fda06024ee3280e1a98194d31513e48ba643bd7` |
| `RUN_INDEX.md` | `existing repository / P23D generated documentation or offline evidence` | 4030 | `cd39ed7ff5f48febea1ddc702ca2e913afdb26d8bb2a909c618174ebf06c672c` |
| `aggregate.py` | `existing repository / P23D generated documentation or offline evidence` | 13041 | `dc24ce9efc880f236de5b72f70fa522146b1d8f36d1463c2d34dc22998a47164` |
| `bench.py` | `existing repository / P23D generated documentation or offline evidence` | 27801 | `0792beeed813d94df2a1d9ce7bffe2f5e2a00a564de4a32aaecfa0182b8e6f8f` |
| `conversation/conv_mtp/ARTIFACTS.md` | `existing repository / P23D generated documentation or offline evidence` | 26556 | `4e865a6dc96caa4fa4e1d39030883a313d409ae62e0d93e6f154334a64dfb121` |
| `conversation/conv_mtp/README.md` | `existing repository / P23D generated documentation or offline evidence` | 4340 | `829168bf0b43a79573ea2c97e38b6c00cd72bf94e624cc2c84ef76cd63fdd476` |
| `conversation/conv_mtp/RUN_INDEX.md` | `existing repository / P23D generated documentation or offline evidence` | 767 | `2d1022e42669b9ccf84b79ae5d680e51a7c972f5bf20db5e94b5d50ea6022b4c` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/blocks.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 2210 | `7e209077bf430bbf71c46b6b92cb9ca0b01fb785f396984db2efa5b1cee873ac` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/c_results.json` | `existing repository / P23D generated documentation or offline evidence` | 16510 | `bad315804110075da55a6c9f53fecba60a3b319bb825e0096f6b380bba809b07` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/c_scores.txt` | `existing repository / P23D generated documentation or offline evidence` | 2203 | `edc66e01d5089a148a94dd3a377ba6354ba520bde3d014587f4027af1ec2d0d6` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/logs/gemma_e2b_q40.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 11673 | `7e520cbf107993f444551464331707ec5964e82b4a4432bdb1664f8407b10df7` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/logs/gemma_e2b_q40_mtp.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 14782 | `755d6387f23edda93bc2b366ae2a708b2ba436c94749491f1b76247c73d5a4c7` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/logs/qwen35_4b_q40mtp.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 12068 | `8df64a7273a42abd46b356cfc1f537e4b54a784d511ffe96d9b3057b7a6754a7` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/logs/qwen35_4b_q40mtp_on.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 14060 | `ce48edc9298dccc71db5c3dc117ef3d8d7351638d4dc1af62e57f6fbe5787616` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/logs/qwen35_4b_q4km.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 12207 | `94e455c50c9e9bf29a26c019cb7b57c1a8d5758275010b3d66b8b9187cd87b54` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/report.txt` | `existing repository / P23D generated documentation or offline evidence` | 1221 | `cc1dc6d1a4f43ccfc386763001f98991307e4e588f0e4b07f87f82ccdc5390d5` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/run_20260928T081907Z.json` | `existing repository / P23D generated documentation or offline evidence` | 2933 | `ed64eb593bdb7a871bb296703d980ff3036522abc1b699e600f3debc2fca69dd` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 44275 | `c3db9e3bf19a926e3f95e785245af92f95a906ac0725d58f6ebdf9946ff0cc93` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/conv_mtp/conv_mtp_20260928T081906Z.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 8363 | `3893faccf271f667a90e352443c3cc68fa119e0a361dc8dcc1eafd34ca40a4b0` |
| `conversation/conv_mtp/oneshot_console_20260928T081359Z.log` | `existing repository / P23D generated documentation or offline evidence` | 11131 | `ab3b716bdf73a9babfe598ba1801eb9bc5dd07e589a73c00e617b77b5ecb93c6` |
| `conversation/conv_mtp/reviews/review1/candidate.diff` | `existing repository / P23D generated documentation or offline evidence` | 13287 | `a38f7193d49271692e96860fdd7e6742e1216b5106afec780e4f31391634553b` |
| `conversation/conv_mtp/reviews/review1/final.md` | `existing repository / P23D generated documentation or offline evidence` | 1561 | `0c68aef5017377210475b15d1d9c3c331a500ec774bd9816d2feb95d64308a45` |
| `conversation/conv_mtp/reviews/review1/frozen.sha` | `existing repository / P23D generated documentation or offline evidence` | 256 | `5b3e8e997eeee5502eec023ccb7c5aba09ec5768d710867553fdb50ee202ad5c` |
| `conversation/conv_mtp/reviews/review1/request.md` | `existing repository / P23D generated documentation or offline evidence` | 20333 | `25ecc190ffbc068c28b2db8609a7ba9f40eb03ff478f05786c43a247c34f5628` |
| `conversation/conv_mtp/reviews/review1/stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 140241 | `fb94d2a229b4a8198aa66018661a82a987b69874ea91add8e33b2b78c3ad7a24` |
| `conversation/conv_mtp/reviews/review1/stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 1562 | `d98392a23282a17e4889ee851009b1676cf581dc0191fd8c1c333649aa4ada4b` |
| `conversation/conv_mtp/reviews/review2/candidate.diff` | `existing repository / P23D generated documentation or offline evidence` | 14147 | `3c7afe03a756f9ead3bc59a1d906a639d5fdbc0ef7ed8578ca0c2fd73055990c` |
| `conversation/conv_mtp/reviews/review2/delta.diff` | `existing repository / P23D generated documentation or offline evidence` | 1459 | `75fff6ecb27cc31699c38d482e7824047b4693f3bb19d796e381cad805679a4c` |
| `conversation/conv_mtp/reviews/review2/final.md` | `existing repository / P23D generated documentation or offline evidence` | 926 | `201fad1e4549acff22db2d2706567f071e5cc643be97472764870a6a60455018` |
| `conversation/conv_mtp/reviews/review2/frozen.sha` | `existing repository / P23D generated documentation or offline evidence` | 256 | `38bb7c693df9a3d3e81b0b58f38222b5feb105018729ea8b626045beedad1fa3` |
| `conversation/conv_mtp/reviews/review2/request.md` | `existing repository / P23D generated documentation or offline evidence` | 25405 | `63a5a82eb3b198135bb08c8dcf85dadf562925a65ed9f772fc0342c2912f64c0` |
| `conversation/conv_mtp/reviews/review2/stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 190061 | `fa74f8006ff5fb41e29fe73840c0a2522224aac82169391253e13a0e9c526854` |
| `conversation/conv_mtp/reviews/review2/stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 927 | `17daa6c272c22b135f43ffa44445b72ec2e1991b6c6132ba324675e2044acb95` |
| `conversation/conv_mtp/reviews/review3/candidate.diff` | `existing repository / P23D generated documentation or offline evidence` | 15363 | `7697223a7537787ad0df82b491143387d60578b41bb68c701ba9cf8f5ffd0e64` |
| `conversation/conv_mtp/reviews/review3/checks.txt` | `existing repository / P23D generated documentation or offline evidence` | 1438 | `c83d4f4f330bfa2717d3b7d48e354e35998efc686f664cee72cfe4d26c11f2bc` |
| `conversation/conv_mtp/reviews/review3/delta.diff` | `existing repository / P23D generated documentation or offline evidence` | 2450 | `358c03d838cc1807b794470524761d03e32b7004bb44e2292d7f00e4d8abb625` |
| `conversation/conv_mtp/reviews/review3/final.md` | `existing repository / P23D generated documentation or offline evidence` | 810 | `b9d8b796e51f7eed8bac181758ff843fcb486fe607599c51d3b96062cbc02908` |
| `conversation/conv_mtp/reviews/review3/frozen.sha` | `existing repository / P23D generated documentation or offline evidence` | 256 | `1d79e99b53b74066a3d86a129909b4b5e2d0da8ff71cc3da9f57223c8a9c70fe` |
| `conversation/conv_mtp/reviews/review3/request.md` | `existing repository / P23D generated documentation or offline evidence` | 29241 | `738fe90717ff74751b8f6f1b01ea0600f3c7443d954673729c003a92d6679c76` |
| `conversation/conv_mtp/reviews/review3/stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 149013 | `2042f681282046b16981562c5ebbc115bf410467ca4eb0c88986db1838e5ebdb` |
| `conversation/conv_mtp/reviews/review3/stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 811 | `fab6d18ae52e36b0a033f0c4615114c2afdcb8c707d5daf06c21e9978d8b8862` |
| `conversation/conv_mtp/reviews/review4/candidate.diff` | `existing repository / P23D generated documentation or offline evidence` | 15873 | `069ceb922a159e718d1249c9ef040eec7572bee71fd801cf87d8693678d98f08` |
| `conversation/conv_mtp/reviews/review4/delta.diff` | `existing repository / P23D generated documentation or offline evidence` | 1788 | `95c07f9365dd0f0167c1ed8f3bf03aeb10b63d1e8b5ab5467f3d7389aedbeebd` |
| `conversation/conv_mtp/reviews/review4/final.md` | `existing repository / P23D generated documentation or offline evidence` | 868 | `5f34cee0618fa6df1ce1be79f5fd2d24ff76d0dd79338d5d69bf414127a5c5a4` |
| `conversation/conv_mtp/reviews/review4/frozen.sha` | `existing repository / P23D generated documentation or offline evidence` | 256 | `f159756abdd47b7fa24dcda9a4eec4bbb83f40b8e14f223d67373163b32abb1c` |
| `conversation/conv_mtp/reviews/review4/request.md` | `existing repository / P23D generated documentation or offline evidence` | 30653 | `9fba1328033b3398a3077af67be3272ba88350d2c96ef0a3cc72ba6f0f2b0f1a` |
| `conversation/conv_mtp/reviews/review4/stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 192215 | `a5bd9dd60ddc610afee02e590b07d7d9499fd3944ddf65bf86d5f02c11762954` |
| `conversation/conv_mtp/reviews/review4/stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 869 | `2b8c024920b48794b82cd479ab8428e253561f95b1f18713174a6889e1d2ac66` |
| `conversation/conv_mtp/reviews/review5/candidate.diff` | `existing repository / P23D generated documentation or offline evidence` | 16160 | `d35883a1538bf8e11af67d687fdf9a9290eab0c0018fbe246f95f79a0bc432e4` |
| `conversation/conv_mtp/reviews/review5/checks.txt` | `existing repository / P23D generated documentation or offline evidence` | 923 | `5411594ea597961785138c205411feff384a2899c720f5d434070b91bda2b4d7` |
| `conversation/conv_mtp/reviews/review5/delta.diff` | `existing repository / P23D generated documentation or offline evidence` | 1931 | `bc0fce8c3997fcda99062b07f1a550320d53606eb5bf9a3206b19612cf5b0724` |
| `conversation/conv_mtp/reviews/review5/final.md` | `existing repository / P23D generated documentation or offline evidence` | 341 | `432b12c6f8e16b93e477ba6e48e404799c72c02a0fd2757c5dbbb8dab15e2f66` |
| `conversation/conv_mtp/reviews/review5/frozen.sha` | `existing repository / P23D generated documentation or offline evidence` | 256 | `644cb6d95757a5ed1baddfe7bf23b87edf2d814ae27e29b991b88be60cf3f0b0` |
| `conversation/conv_mtp/reviews/review5/request.md` | `existing repository / P23D generated documentation or offline evidence` | 32339 | `b8191c4b2427d9927d359fd3d26653b8eba4714284f759bb585362de88e4bce3` |
| `conversation/conv_mtp/reviews/review5/stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 164909 | `9fb8f663740eb52cf7645dfe5bc029c7b577f7e47aa81a8e3a17e4ef0914a44e` |
| `conversation/conv_mtp/reviews/review5/stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 342 | `8fb42edaa9b07ccf19f6272b16755219d3b74f40cf8da028cbb00b2e9a36507b` |
| `conversation/conv_quality.py` | `existing repository / P23D generated documentation or offline evidence` | 2471 | `4b578dc988a24575da053b2df4a756c99f7ab9053c4f9a3fd4812ac9c4f4b4c6` |
| `conversation/conv_speed.py` | `existing repository / P23D generated documentation or offline evidence` | 12998 | `9b721d8678f53a122f022f8bf66ee318707aa1700b5c684ca144fa06f7b21e7a` |
| `conversation/conv_speed_mtp.py` | `existing repository / P23D generated documentation or offline evidence` | 13647 | `e8fbbdb6b33f0e3190a05699397263a7632abf14a0cf1be30ff7f0628a5f2630` |
| `conversation/conversation_20260926T185957Z/blocks.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 3789 | `441b0730b932a72f9c271a2c22c93e69e18722cc89454bafcf2f24d53a3ad5b9` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e2b_q40.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 22623 | `c150ec7322bfb0c30961d41156aead34ca0f41906e98663f9493a26304e8c2fa` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e2b_q40.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 22623 | `7210a92f32a2edc0c442829a1597e91aba9909916b3385e1a665bd25f78d49da` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e2b_q4km.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 22631 | `a39b643e578f5ebef95658d778b5053935249878da6c3e0224822228e4533c67` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e2b_q4km.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 22631 | `7a5c8078950272ecd16588c76dc10e11e2023dc6dae2b030f7112ecbfe280ade` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e4b_q4km.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 22889 | `8b7d8c05b62aa98ca36f0db34896acf5365a40ff10c269bc6e3193057b9c50f5` |
| `conversation/conversation_20260926T185957Z/logs/gemma_e4b_q4km.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 22883 | `4572ef516f60f01a295d89ae2a275f9f31c122a9c6f25bee65b6beafb6e456ba` |
| `conversation/conversation_20260926T185957Z/logs/qwen35_2b.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 23684 | `130c296031e5e2940ace4a0d0077f45321d6bea4b4a8f23fde9f2058cdad332c` |
| `conversation/conversation_20260926T185957Z/logs/qwen35_2b.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 23684 | `dc09e176671262a3474f60919055d5c9c52693d9132a7cec0204a0c74a3769ad` |
| `conversation/conversation_20260926T185957Z/logs/qwen35_4b.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 25737 | `9d8d7b45a26f5ebabf5b48663fd95e454fed3bfe0e8c2611cc773e4e71dfd368` |
| `conversation/conversation_20260926T185957Z/logs/qwen35_4b.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 25737 | `ee0a95acfe219f0150964bddf4badf464cf3085dfde350d53169394b5087235b` |
| `conversation/conversation_20260926T185957Z/report.txt` | `existing repository / P23D generated documentation or offline evidence` | 1766 | `3a917faeb494ac9b24c8e3aad3c6ea35da788117f64b4a97206decd9214796ef` |
| `conversation/conversation_20260926T185957Z/run_20260926T185957Z.json` | `existing repository / P23D generated documentation or offline evidence` | 1381 | `f894d112211ce542aa55feddccfc92112cc8371580cc0953b5ef6af9b181fa45` |
| `conversation/conversation_20260926T185957Z/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 97940 | `6b3e7da5101ddf45e4a8698d21c545a7e925981dd16ac8ae82e7455732a14b35` |
| `conversation/conversation_20260926T185957Z.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/conversation_20260926T185957Z.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 6929 | `da0c071404d7297425e16e3438e5c9521156e52bc6e80380109be52b8430022c` |
| `conversation/conversation_blind_key.tsv` | `existing repository / P23D generated documentation or offline evidence` | 1266 | `b8657ac39d1eaf054e334d5827bf6ae6c0fe7205f46529406d3352ffb36cb77f` |
| `conversation/conversation_blind_sheet.md` | `existing repository / P23D generated documentation or offline evidence` | 12812 | `0c1678c7d0af08dcbdd660ebbf5cdafb973f0c4ee3b6abce46723b4e71fb684e` |
| `conversation/conversation_quality/bench_blind_key.txt` | `existing repository / P23D generated documentation or offline evidence` | 84 | `c7d19e8185e535453de0d32e0068bc37f5b5af11cdd35e7a05d304f73b2e2c3e` |
| `conversation/conversation_quality/bench_blind_transcripts.txt` | `existing repository / P23D generated documentation or offline evidence` | 6447 | `c3203ad6c26a438e9fe36d3c3614af89eaae0a5fd4603761b5f2a7df08564f96` |
| `conversation/conversation_quality/bench_c_summary.txt` | `existing repository / P23D generated documentation or offline evidence` | 3547 | `b63dc5b8ef89e2f62c450d7547884451f2d21109e34cb38f27efbdb5af4e6dc9` |
| `conversation/conversation_quality/bench_gemma_e2b_q40_20260926_163419.log` | `existing repository / P23D generated documentation or offline evidence` | 69777 | `4d55c8c591bce02d28ce38fb42d7257887e6883932a8825cfaefc9304743567e` |
| `conversation/conversation_quality/bench_gemma_e2b_q4km_20260926_163419.log` | `existing repository / P23D generated documentation or offline evidence` | 69970 | `4f8fcf85b59f7a4d305b96e2d11d6fba14f7f914754b10d81f0c8d34af0306f1` |
| `conversation/conversation_quality/bench_gemma_e4b_q4km_20260926_163419.log` | `existing repository / P23D generated documentation or offline evidence` | 70327 | `3848f700485b8c85c4e35bc6abcd39ce96fc95afd2c52d24e60aec0a94eb9372` |
| `conversation/conversation_quality/bench_qwen35_2b_20260926_163419.log` | `existing repository / P23D generated documentation or offline evidence` | 75036 | `d30497d11b584058aae2965b4b2d1adb61d0c3f997366f85adf6fada13c9fe2e` |
| `conversation/conversation_quality/bench_qwen35_4b_20260926_163419.log` | `existing repository / P23D generated documentation or offline evidence` | 23914 | `e09b3605ce7232f8817359145c99a228715223cf0b1da0562a2a2ba71ab5c5e3` |
| `conversation/conversation_quality/bench_results_20260926_163419.json` | `existing repository / P23D generated documentation or offline evidence` | 542095 | `398e36f554860da02220dd183296d8b3d9f6c56bd24e480c8693af73a9465e72` |
| `conversation/conversation_quality/bench_speed_summary.txt` | `existing repository / P23D generated documentation or offline evidence` | 1037 | `452ba4adc348e01de858c43b322145cab603af188501620c45e4ec71a8588e45` |
| `conversation/conversation_quality.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/conversation_quality.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 5662 | `2921ef45759d2c34e42137872a24663e187c1008c7bc85d0ef9ba8b104882f5b` |
| `conversation/conversation_quality_combined.json` | `existing repository / P23D generated documentation or offline evidence` | 568881 | `a655686de7f4e6bea222994cc2b4a5c5114b286ee97b5265d0a154072e70d6cd` |
| `conversation/conversation_quality_qwen35_4b/bench_blind_key.txt` | `existing repository / P23D generated documentation or offline evidence` | 14 | `952a7cb1ca044aba075bf1e1406edf1d0adab1ba8a90f9b18352daca7d5c0f64` |
| `conversation/conversation_quality_qwen35_4b/bench_blind_transcripts.txt` | `existing repository / P23D generated documentation or offline evidence` | 1781 | `09cb67e2559bb953dfc331771cf8c130e6906195f83b3d520311d6cb6d6e3414` |
| `conversation/conversation_quality_qwen35_4b/bench_c_summary.txt` | `existing repository / P23D generated documentation or offline evidence` | 1175 | `44f2cac264f738f8ac3a97a516e07736402756f5b356a545b7a9d1025dce7f8e` |
| `conversation/conversation_quality_qwen35_4b/bench_qwen35_4b_20260926_182310.log` | `existing repository / P23D generated documentation or offline evidence` | 82006 | `3b1aeb8b0a5e9275fd7ba790d63c1538cbae4b29ed8c22dc46c12086b16ea106` |
| `conversation/conversation_quality_qwen35_4b/bench_results_20260926_182310.json` | `existing repository / P23D generated documentation or offline evidence` | 121663 | `1458ea65de10333e8d2142a8eece3ae04e18ec91c5dca5ec5c91bfa245bcb6c1` |
| `conversation/conversation_quality_qwen35_4b/bench_speed_summary.txt` | `existing repository / P23D generated documentation or offline evidence` | 510 | `b2ec62f069f96fc976807ab0655e1bce3f056b17748d09419672cbb8c0795843` |
| `conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734/bench_blind_key.txt` | `existing repository / P23D generated documentation or offline evidence` | 14 | `952a7cb1ca044aba075bf1e1406edf1d0adab1ba8a90f9b18352daca7d5c0f64` |
| `conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734/bench_qwen35_4b_20260926_173003.log` | `existing repository / P23D generated documentation or offline evidence` | 23768 | `825416d1b4fe097b0a95f41ea6e7d5ff8848b5970f517c4175546b67b73ac83b` |
| `conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/conversation_quality_qwen35_4b.killed_by_lmk_1734.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 176 | `469a7584dc4d8e95ca8ea50fd8a9b4530fa5388c7ca8f678c10d276de3709e22` |
| `conversation/conversation_quality_qwen35_4b.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/conversation_quality_qwen35_4b.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 208 | `0dc52c7bd51cc38d26854530a7bc6b572efdf13f4cc5ead25a3c7a8600d99248` |
| `conversation/conversation_quality_scores.txt` | `existing repository / P23D generated documentation or offline evidence` | 3809 | `8002085be7a0327a27fd56a10fb9d3e1f816a1bfdc1480214654c0169418ac58` |
| `conversation/local_ai_blind_scores.tsv` | `existing repository / P23D generated documentation or offline evidence` | 1909 | `0da2f2b8c6af6445b9b06749ead6f5444a57fbc5e96fb4ebd56b61daf859bd59` |
| `conversation/make_blind_sheet.py` | `existing repository / P23D generated documentation or offline evidence` | 3427 | `c698988cf418d37bf1388816cfefa8624a0e8cfde377094c92837ce682bbde49` |
| `conversation/prep/full_quality.sh` | `existing repository / P23D generated documentation or offline evidence` | 244 | `2c2bc7f88c40b6c0642fd1e6706af8261661329127b94d3ef2cbb56db9ea1900` |
| `conversation/prep/mem_diag.py` | `existing repository / P23D generated documentation or offline evidence` | 1729 | `74132400e1afba33690ca47bd8b7c94ed9efe79905b3a55f58a183d7174dae31` |
| `conversation/prep/mem_diag.server.log` | `existing repository / P23D generated documentation or offline evidence` | 11953 | `22ee65f38df9cb8a02f806e0f980702df057d8dd9a38a8255ca65be0a4fb5d6b` |
| `conversation/prep/mem_diag_cr0.server.log` | `existing repository / P23D generated documentation or offline evidence` | 18443 | `7aa5138626d3db8b13cb1069f957373d58f1aa1765a80765393c2e4e83968db7` |
| `conversation/prep/mem_diag_cr0.txt` | `existing repository / P23D generated documentation or offline evidence` | 1530 | `ff4449216f382be39f938ef289309cce8f59474fe32103c5214d805789776900` |
| `conversation/prep/oneshot.sh.orig` | `existing repository / P23D generated documentation or offline evidence` | 2533 | `0edcae559ea922a7be9ceefc9b4e20e5191eb8526494929046eb44103232a456` |
| `conversation/prep/quick.sh` | `existing repository / P23D generated documentation or offline evidence` | 1172 | `22da3105d56434889fd3a0f8c3a9481f67dfc32ebef2647acb5c8babcb86ce86` |
| `conversation/prep/quick_key.tsv` | `existing repository / P23D generated documentation or offline evidence` | 1266 | `d9100bdf12b3516463943d3782d581f9f14f76de46ddd694c0ed86ada744aca2` |
| `conversation/prep/quick_quality/bench_blind_key.txt` | `existing repository / P23D generated documentation or offline evidence` | 84 | `e342699140a128ada2b3aebad36eb0e0e5d925f8944f734ac31582b68c35159a` |
| `conversation/prep/quick_quality/bench_blind_transcripts.txt` | `existing repository / P23D generated documentation or offline evidence` | 366 | `f4b101ec3e2b7da9b1c2463376203501743d82cd6ddbabdcc1fd54b3205228b5` |
| `conversation/prep/quick_quality/bench_c_summary.txt` | `existing repository / P23D generated documentation or offline evidence` | 1029 | `388b6e90fe58a595282e1498e0c5742a10c6dac79be95394a97f9d151291315a` |
| `conversation/prep/quick_quality/bench_gemma_e2b_q40_20260926_162121.log` | `existing repository / P23D generated documentation or offline evidence` | 2946 | `0a61cf78ae265008375056dc79f468f0afe5bc36c64e90e64eab2c62ab905fd8` |
| `conversation/prep/quick_quality/bench_gemma_e2b_q4km_20260926_162121.log` | `existing repository / P23D generated documentation or offline evidence` | 2948 | `8777aad501da238ce6f1331d4b2b0d21c40935973f56d88ebd932b243214d548` |
| `conversation/prep/quick_quality/bench_gemma_e4b_q4km_20260926_162121.log` | `existing repository / P23D generated documentation or offline evidence` | 3206 | `9ccd6e607f0a524301ee1be3faf3bce824ec3c4b3182aa1c1bc57df9c97a0009` |
| `conversation/prep/quick_quality/bench_qwen35_2b_20260926_162121.log` | `existing repository / P23D generated documentation or offline evidence` | 3407 | `e9cddea81b90a0c87618d2a5f319bccbcd4905638bbb80e479c50c3257e17bb6` |
| `conversation/prep/quick_quality/bench_qwen35_4b_20260926_162121.log` | `existing repository / P23D generated documentation or offline evidence` | 3407 | `dd75299f97a6d15d93fb5794dca7843596dba9623044239e7759ec856e5a258e` |
| `conversation/prep/quick_quality/bench_results_20260926_162121.json` | `existing repository / P23D generated documentation or offline evidence` | 14126 | `d523e21edd6be12286918327120f1920c83a57de0bf1452edd6b9b6d6cda15c7` |
| `conversation/prep/quick_quality/bench_speed_summary.txt` | `existing repository / P23D generated documentation or offline evidence` | 1035 | `3bbcd02d370f48510ecad6e0729477d1cc9695081f6ddc8b121a61e3a1c164dd` |
| `conversation/prep/quick_quality.py` | `existing repository / P23D generated documentation or offline evidence` | 388 | `c0e405e6b024cc19c9e6b20f4369cbf8fd2d2d085af11af0f8b376553803eb83` |
| `conversation/prep/quick_quality.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/prep/quick_quality.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 350 | `6a91b99d5f486013928d9efae66cd6a65f11a2855a46d515942c5bddafcc2ec0` |
| `conversation/prep/quick_sheet.md` | `existing repository / P23D generated documentation or offline evidence` | 10603 | `65ffabf6b6e51d211ec94e2f227513f4cad2dc747fac2d2e8f43efbe7bec57d9` |
| `conversation/prep/quick_speed/blocks.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 3942 | `2f1d449b9701eb7f59cb59d5fbe534fbe8b649f87640c8bf63d1c15e4f361211` |
| `conversation/prep/quick_speed/logs/gemma_e2b_q40.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 2946 | `c2de57f3b03c9fcfdb58396cff79e44badbd81677b161985fffd29756701bd81` |
| `conversation/prep/quick_speed/logs/gemma_e2b_q40.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 2946 | `731667e68b1299dd5a9024fe079c60c985602fd81e39652dad008a4697686bf2` |
| `conversation/prep/quick_speed/logs/gemma_e2b_q4km.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 2948 | `13b2f94115ce90839d7f568dcc7da4c3dcb182d5daaedb902bd82f5a82dbc901` |
| `conversation/prep/quick_speed/logs/gemma_e2b_q4km.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 2948 | `710498599567aadc391c24239f8b87b6b00ff320ff90bf670ee0ffa691d16a2e` |
| `conversation/prep/quick_speed/logs/gemma_e4b_q4km.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 3206 | `7af4f9bbcc905f6218ecc415da21ff9eb2ffd49c991193ef8b97e136bb0f0e9c` |
| `conversation/prep/quick_speed/logs/gemma_e4b_q4km.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 3206 | `51f21c0a2f3c446d5d2b7062a2e36a1b4b0719331b5136d4f7af79b3bfbb6355` |
| `conversation/prep/quick_speed/logs/qwen35_2b.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 2500 | `1e7211b934ec37bc35a93be3d48cb0b01f5299fc895878b83e833ab39575b1ba` |
| `conversation/prep/quick_speed/logs/qwen35_2b.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 2500 | `c730e022bba8d6b9d5aba24b6fedf8e5ca0b032c1d643ccfb4b1d19da7ad20a0` |
| `conversation/prep/quick_speed/logs/qwen35_4b.cached.server.log` | `existing repository / P23D generated documentation or offline evidence` | 2646 | `2c8753e8ba22f002d0f6b2359e2e5a0d3d2c732eeb4de0e62c8c601bc6907e47` |
| `conversation/prep/quick_speed/logs/qwen35_4b.cold.server.log` | `existing repository / P23D generated documentation or offline evidence` | 2646 | `daf4bde5d65cd76cd63dc953e0bc2a965814ca95974be20a8a743f73c29c973a` |
| `conversation/prep/quick_speed/report.txt` | `existing repository / P23D generated documentation or offline evidence` | 1784 | `ac89115e42a55a0516659f5b08a768fcb3f935eee35f71f2e67c2924c2c953f4` |
| `conversation/prep/quick_speed/run_20260926T162800Z.json` | `existing repository / P23D generated documentation or offline evidence` | 1357 | `3e65d3d38cead9a893f57c19ec4dc8d3d6e29248df7cc8328ca7ac8009624652` |
| `conversation/prep/quick_speed/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 6350 | `c0ee547e120f14e071cad0b11ab4bf5ec7439fd2c51b46345def18e674f77479` |
| `conversation/prep/quick_speed.py` | `existing repository / P23D generated documentation or offline evidence` | 441 | `8ce982e5ae37a79b60c41f9880d7df7402e514fed18352e8fae9eb641e35e522` |
| `conversation/prep/quick_speed.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/prep/quick_speed.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 5542 | `7c754c14c62acf5ef0a8f9e71ce713e53e1a9eeb831bb20549ac2aff688aa27c` |
| `conversation/prep/qwen4b_a7.py` | `existing repository / P23D generated documentation or offline evidence` | 1115 | `be53afcba1ceb663d0792e270124f3326714b6985d3cd8bc3c5a2921fdc042f8` |
| `conversation/prep/qwen4b_a7.server.log` | `existing repository / P23D generated documentation or offline evidence` | 4462 | `5a2a47c8575a279d7f235a2ffb90aad37275dd27b9f22d18b8c48e5a7d8d7300` |
| `conversation/prep/rerun_qwen4b.sh` | `existing repository / P23D generated documentation or offline evidence` | 586 | `1ca302eeb20a186d77b5b16db852b30a8b3f4529f53913475f0e9cd63fec7eea` |
| `conversation/prep/review/REQUEST.md` | `existing repository / P23D generated documentation or offline evidence` | 3719 | `6a76038f342c5a5752b4dae242375fbbed39b82b73d01b9cbea5b16f914b6890` |
| `conversation/prep/review/bench_rubric.diff` | `existing repository / P23D generated documentation or offline evidence` | 2827 | `fd8fbf41b0b2921f5db5341f058d9e9676c44bdc8aa0e8f9e8f1ee7e2e20eed2` |
| `conversation/prep/review/frozen.sha256` | `existing repository / P23D generated documentation or offline evidence` | 1092 | `9f282ada5dd308dd3b20f6c6ab9b041919addcd18ba1b9a8daae3658cf4bde8d` |
| `conversation/prep/review/ladder_used.py` | `existing repository / P23D generated documentation or offline evidence` | 5379 | `33ab31ddb49efef68cdb618cbb025b978ae43a0f3abe23f2f934e1d9d71cb8dc` |
| `conversation/prep/review/oneshot.diff` | `existing repository / P23D generated documentation or offline evidence` | 1376 | `cc47e5f11e3a72be229f5a2fb956a2ac749f270aad73694064d72034a03414c3` |
| `conversation/prep/review/prompt.txt` | `existing repository / P23D generated documentation or offline evidence` | 78234 | `e26a5ee90359692a299fc279780623b19014dca0e6ccf6b33da0c11aa8c9c4e1` |
| `conversation/prep/review/tests.txt` | `existing repository / P23D generated documentation or offline evidence` | 9 | `1e04dbfa7dc1165fddb246f5da4f18c8a7bbf521b5814d96d1729bbaebd57f74` |
| `conversation/prep/review.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/prep/review.stdout.json` | `existing repository / P23D generated documentation or offline evidence` | 3400 | `f87fddcf0919bd7cacbe35e12d678b4afb00258897af8f5cdf0e29e517a976e8` |
| `conversation/prep/review2/REQUEST.md` | `existing repository / P23D generated documentation or offline evidence` | 2087 | `4c6f218ab6bcef4107295a089f2765d68db9adeafafc1a0a53ef3e0de3fdb6b4` |
| `conversation/prep/review2/conv_speed.diff` | `existing repository / P23D generated documentation or offline evidence` | 1119 | `88a2c4572b3202c90aec7e1390de8b85fb51a5de8e41189ada9505765ed1b70f` |
| `conversation/prep/review2/conv_speed.py.reviewed` | `existing repository / P23D generated documentation or offline evidence` | 12609 | `206f27512dac5ea6b25bbba6728abde3aeb5edd9ec7d1f7f8bc9f085c1bfb538` |
| `conversation/prep/review2/frozen.sha256` | `existing repository / P23D generated documentation or offline evidence` | 758 | `2bb264743420cd5758ff3a72325c58b852b90cbc641264a3d149ce8c811fd766` |
| `conversation/prep/review2/prompt.txt` | `existing repository / P23D generated documentation or offline evidence` | 25570 | `b79d41622963e40e2dd7724fbb5b2a31f0e1b54d2db25f506c18be1d7b453682` |
| `conversation/prep/review2/run_conversation.diff` | `existing repository / P23D generated documentation or offline evidence` | 715 | `081d074b903d255af95ae46c3c60df40be78f638531fd556edc09892f5e16754` |
| `conversation/prep/review2/run_conversation.sh.reviewed` | `existing repository / P23D generated documentation or offline evidence` | 1318 | `b150b223adbeffa759aba247e93387ee086277d3afa2f708bc7a76a810760a00` |
| `conversation/prep/review2.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 228 | `207996a09c6f7284a435e2ccfcfbf672f17540940f953f267979ff59418efbbb` |
| `conversation/prep/review2.stdout.json` | `existing repository / P23D generated documentation or offline evidence` | 426 | `dc6520546a8d670989b5170f6bf37a3e9c838ae8e3a980fe568c8bf216820001` |
| `conversation/prep/review2b.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conversation/prep/review2b.stdout.json` | `existing repository / P23D generated documentation or offline evidence` | 2527 | `d6511e32d7b748e9d564022816cf909c1567c3699a2b03b252ab0f4c764045a6` |
| `conversation/run_conv_mtp.sh` | `existing repository / P23D generated documentation or offline evidence` | 1669 | `709a132de76bd0932d1212d61edeefeba1cbb512cb26a48e86acb23fe762a52d` |
| `conversation/run_conversation.sh` | `existing repository / P23D generated documentation or offline evidence` | 1411 | `a335b7633c9c0bce6d86744258e101c4b076982e618f6c434e34cc82ef6faee8` |
| `conversation/score_c.py` | `existing repository / P23D generated documentation or offline evidence` | 2702 | `49c4069c69f958d41f57fe5458276a3f33ce3abe3aa086c8d241e992ec6e3d84` |
| `memory_check/conv_cram0/meta.json` | `existing repository / P23D generated documentation or offline evidence` | 703 | `d3576413a10ed73787fc6177fae8ba88802b1ea84173a53157d75b94a70ce77e` |
| `memory_check/conv_cram0/server.log` | `existing repository / P23D generated documentation or offline evidence` | 34011 | `1e589a78a3c13134ba01526a8d4bdfeaa7d279c9bbcad6de79b10bc30af63644` |
| `memory_check/conv_cram0/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 28396 | `e1a5529cdf3bfe0db313aaf81b55f336ff650acaeb3c2b76c36904f5ff31ab52` |
| `memory_check/conv_cram0.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/conv_cram0.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 28396 | `e1a5529cdf3bfe0db313aaf81b55f336ff650acaeb3c2b76c36904f5ff31ab52` |
| `memory_check/conv_default/meta.json` | `existing repository / P23D generated documentation or offline evidence` | 676 | `3432a59f3a5aac226b95a30a861b2951b4c32dfc433d7a303f09e8579968bbeb` |
| `memory_check/conv_default/server.log` | `existing repository / P23D generated documentation or offline evidence` | 34071 | `2bf8e443932605119e8cbae64c47a3abba2384baa38d3c8e5b0a569757f9a671` |
| `memory_check/conv_default/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 28777 | `c0444db2653af95cd77569a2fd3445573ab0a7368f53ea5bd8f8cf7e1647f7c2` |
| `memory_check/conv_default.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/conv_default.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 28777 | `c0444db2653af95cd77569a2fd3445573ab0a7368f53ea5bd8f8cf7e1647f7c2` |
| `memory_check/memcheck.py` | `existing repository / P23D generated documentation or offline evidence` | 6732 | `04ee1f627f18a269cbab0736ad80b1cd57a9265edb2b9f87e34e800a7b74dc19` |
| `memory_check/memcheck_pilot.py` | `existing repository / P23D generated documentation or offline evidence` | 5784 | `d1d791e128196bdf76e9a7ea74c4140132dfe7c76c38db759328a742cafb8376` |
| `memory_check/pilot_robot_default/meta.json` | `existing repository / P23D generated documentation or offline evidence` | 657 | `58c13653247289542e6dd8ec163376345be1132fdd87e94d1af0e82a2f83b569` |
| `memory_check/pilot_robot_default/server.log` | `existing repository / P23D generated documentation or offline evidence` | 33711 | `c9f5836a1ef7f2b617d608592cb90455b2358b907cf39e0eb4ea3772b198e26d` |
| `memory_check/pilot_robot_default/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 18276 | `b0cf63a3ea9fe5f525748285afddb65b7bffb1b86756a8be8aa4cf8493ef8121` |
| `memory_check/pilot_robot_default.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/pilot_robot_default.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 18276 | `b0cf63a3ea9fe5f525748285afddb65b7bffb1b86756a8be8aa4cf8493ef8121` |
| `memory_check/robot_cram0/meta.json` | `existing repository / P23D generated documentation or offline evidence` | 704 | `f0d52feafd55f7b5b36f1ba793e2a07b59d223d8750c4cecf7556e3db0bebd84` |
| `memory_check/robot_cram0/server.log` | `existing repository / P23D generated documentation or offline evidence` | 33796 | `0df93cb88eae4b64f33d69de9b2fb2cec195fce9d575955f7d410d46e9b02112` |
| `memory_check/robot_cram0/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 18236 | `e8241b02f2e18e55ef55d8a3c7b12d8a300b01f793227b6bb726fecaef56aae2` |
| `memory_check/robot_cram0.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_cram0.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 18236 | `e8241b02f2e18e55ef55d8a3c7b12d8a300b01f793227b6bb726fecaef56aae2` |
| `memory_check/robot_cram0_pin47_a/meta.json` | `existing repository / P23D generated documentation or offline evidence` | 704 | `6b9fb808a6b4357a11c202fd0f738ebaf6ac4a2b8425e882e7497c1acfb434ea` |
| `memory_check/robot_cram0_pin47_a/server.log` | `existing repository / P23D generated documentation or offline evidence` | 33796 | `972e27e634738b985c47833daa9669d48044aa600ba083a69c01c6c24ce846df` |
| `memory_check/robot_cram0_pin47_a/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 18284 | `ee50e45c50a7c6be91c41b7602087e967b89fe27333af10e5a623316dbff05ab` |
| `memory_check/robot_cram0_pin47_a.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_cram0_pin47_a.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 18284 | `ee50e45c50a7c6be91c41b7602087e967b89fe27333af10e5a623316dbff05ab` |
| `memory_check/robot_cram0_pin47_b/meta.json` | `existing repository / P23D generated documentation or offline evidence` | 704 | `f3417474a78531c7cca6b2624eb029f9b7a22a0397ad23c0524d7acc48257ee7` |
| `memory_check/robot_cram0_pin47_b/server.log` | `existing repository / P23D generated documentation or offline evidence` | 33796 | `ffe65316c55ab55d9cb9025e7285e15a03f816d333afb78bce828dffeddc22ba` |
| `memory_check/robot_cram0_pin47_b/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 18253 | `edc882dfa7615c31c508a9a17bffb6851c334e1551ad0559b82cd15bacc7e30e` |
| `memory_check/robot_cram0_pin47_b.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_cram0_pin47_b.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 18253 | `edc882dfa7615c31c508a9a17bffb6851c334e1551ad0559b82cd15bacc7e30e` |
| `memory_check/robot_default/meta.json` | `existing repository / P23D generated documentation or offline evidence` | 677 | `f375727004750ab3f1449755923de7ca1fa717b9a0d2963765e25cdac5804364` |
| `memory_check/robot_default/server.log` | `existing repository / P23D generated documentation or offline evidence` | 33711 | `66dec6a7a354e52d30ce7eae8c31387b7a3baab78d3a5b6dfe6703cd99b5f8d1` |
| `memory_check/robot_default/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 18240 | `70ff989d83837d8c94f6c8442341238a0a16409f2b97d15ae5786e6c7e79d0bd` |
| `memory_check/robot_default.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_default.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 18240 | `70ff989d83837d8c94f6c8442341238a0a16409f2b97d15ae5786e6c7e79d0bd` |
| `memory_check/robot_default_pin47_a/meta.json` | `existing repository / P23D generated documentation or offline evidence` | 677 | `5f8b36a22c34f98446860729eb5b05f07cb0c0aa1db8179da79f3c45d370aaf9` |
| `memory_check/robot_default_pin47_a/server.log` | `existing repository / P23D generated documentation or offline evidence` | 33711 | `f1339029ec40dcf844462621a00d321991331a0473c24a8b0e2dfd63bdc05705` |
| `memory_check/robot_default_pin47_a/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 18255 | `dd81e2d52f58180c6c868323724952b1feadba6e28e49077cd6de05753ff667d` |
| `memory_check/robot_default_pin47_a.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_default_pin47_a.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 18255 | `dd81e2d52f58180c6c868323724952b1feadba6e28e49077cd6de05753ff667d` |
| `memory_check/robot_default_pin47_b/meta.json` | `existing repository / P23D generated documentation or offline evidence` | 677 | `2cb9d2a1772d3e8c48c6eb97d030f41bfddb435cae89a682e25e5f2eabf4e82c` |
| `memory_check/robot_default_pin47_b/server.log` | `existing repository / P23D generated documentation or offline evidence` | 33711 | `b4bc0fc54c6397c487dee3b1f84777bb2400e47701addbc09c05e7c6d0caabc8` |
| `memory_check/robot_default_pin47_b/turns.jsonl` | `existing repository / P23D generated documentation or offline evidence` | 18253 | `ad60c3da0efbce2ac6f0fad65a7d0fcf846600c7ee7e9b14db7e6d5fa1155c12` |
| `memory_check/robot_default_pin47_b.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `memory_check/robot_default_pin47_b.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 18253 | `ad60c3da0efbce2ac6f0fad65a7d0fcf846600c7ee7e9b14db7e6d5fa1155c12` |
| `memory_check/run_all.log` | `existing repository / P23D generated documentation or offline evidence` | 1367 | `0463f6f0a08d3974f54e0bca36b674c345aa27cff81606f174e9afec98afb788` |
| `memory_check/run_all.sh` | `existing repository / P23D generated documentation or offline evidence` | 755 | `5d5f56ea30bec7381a632ece1099a9a78eb7293046db693289a7548ed2b34a4c` |
| `memory_check/summarize.py` | `existing repository / P23D generated documentation or offline evidence` | 1566 | `4edfa9cffe941c96b58f5ab0a31701281f33f2f8d0995629f1e44951aaea5859` |
| `memory_check/summary.md` | `existing repository / P23D generated documentation or offline evidence` | 1483 | `5c7b2f7d34b9a751ab5cc430bf6709a89352635db54df9d4d44692d464e36937` |
| `memory_check/timeout_check.py` | `existing repository / P23D generated documentation or offline evidence` | 1734 | `691e4cd58baa775e3c6a29ca4527825c925f48e6643c1224d016b709ff4fd8a9` |
| `memory_check/timeout_check.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` |
| `memory_check/timeout_check.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 2645 | `2d2ebc8cc13b1fc651932b092be857a97807f5a6dff1efc7062b089308d7a630` |
| `memory_check/timeout_check.thermal.txt` | `existing repository / P23D generated documentation or offline evidence` | 271 | `e19078c7d960790b4948250c251bbcb3e0ba0b9b8735ddfeeb3ce603839fd41d` |
| `memory_check/timeout_check_warmup.py` | `existing repository / P23D generated documentation or offline evidence` | 1932 | `350f8d7a0fb023443ff211c8c2bd200016ec65f271e4a76f535e58ff97cacf94` |
| `memory_check/timeout_check_warmup.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` |
| `memory_check/timeout_check_warmup.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 2707 | `8847dd55dd356e8f68b880a0dc6665cb8171ee6caead009ea6066aa82b6beb98` |
| `memory_check/timeout_check_warmup.thermal.txt` | `existing repository / P23D generated documentation or offline evidence` | 131 | `21c5bc61fd8ff21851e252bc41ed120b13d82d4ad1be61d5ead0e90b169e4b34` |
| `memory_check/timeout_check_warmup2.stderr.txt` | `existing repository / P23D generated documentation or offline evidence` | 239 | `1a09333f9395face7055c3658bf3821b19a8c27012057c3690e1d958407265ca` |
| `memory_check/timeout_check_warmup2.stdout.txt` | `existing repository / P23D generated documentation or offline evidence` | 2709 | `0f43664dd893fbabb407722026fb54656bb85584a340938c5bb3759129df30fb` |
| `memory_check/timeout_check_warmup2.thermal.txt` | `existing repository / P23D generated documentation or offline evidence` | 131 | `cf6ea07f7455331749c96e60ee4212a009f79d7f2fcc205c902ab264d611fac0` |
| `memory_check/warmup_no_server.txt` | `existing repository / P23D generated documentation or offline evidence` | 91 | `cba2093f73f43e4925274d060363d013ba0e4247a1536d0031e762cb9bec3048` |
| `reports/CODER_REPORT_conversation_prep.md` | `existing repository / P23D generated documentation or offline evidence` | 12223 | `984bd324294b78dd979af320ad240f5ec34c35f2f3912c1e0408e190fdabd822` |
| `runs/bench.py` | `existing repository / P23D generated documentation or offline evidence` | 26982 | `1e1b506c3276222b470125fa551d9afa692f98b9e5701ca351ce975f64c06080` |
| `runs/bench_results_20260919_083347.json` | `existing repository / P23D generated documentation or offline evidence` | 109679 | `8a46ce308a54dd2eeac2340a89a04bb14ba886f5399092fef3c3b2558d3fda11` |
| `runs/bench_results_20260919_115350.json` | `existing repository / P23D generated documentation or offline evidence` | 111778 | `e228bf0e2b222b481a8c18ffc655f1abef9a321dc512d82cd8566aa2c5098a47` |
| `runs/bench_results_20260921_054508.json` | `existing repository / P23D generated documentation or offline evidence` | 36800 | `4c85fc58ead899b41c84f3327c7a46f90b140f4b564b9a71c0a2e36100df5358` |
| `runs/bench_results_20260921_164938.json` | `existing repository / P23D generated documentation or offline evidence` | 235645 | `804cade9a772b9f2cb415c5843621edc70d8ecff37487f3a7e966081ea7ae6d6` |
| `test_aggregate.py` | `existing repository / P23D generated documentation or offline evidence` | 2065 | `51bd46a566b37c9bf0728a7f43594502eb052ae0a518e425543f284070a6d16c` |
| `test_bench.py` | `existing repository / P23D generated documentation or offline evidence` | 1374 | `123484c38cea33f7a00fdbbcfecbedfb07129ca116a9a8db73a40a2de37543ae` |
