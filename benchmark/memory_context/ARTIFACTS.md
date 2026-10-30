# Artifacts — MEM-1

**Run artifacts archived (2026-10-30): 105 files under `runs/`, copied from the phone's `~/robot_runs/mem1/`.**
The owner's 2026-10-29 check `MEM1_ALL_HASHES_OK` was reported, not seen here; this archive was checked on its own
(below). When runs are archived, each original is copied byte for byte from
`~/robot_runs/mem1/` (`cp -p`), compared with `cmp`, and listed below with its SHA-256. Every run folder except the
crashed `20261027T091311Z_primary_ctx2048_smoke/` (see [RUN_INDEX.md](RUN_INDEX.md)) carries the runner's own `artifacts.sha256` (all files of the folder at the end of the run); recheck it in place with
`sha256sum -c artifacts.sha256`. `README.md`, `RUN_INDEX.md` and `ARTIFACTS.md` are documentation and stay out of
the check block. Recheck from this folder:
`sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Source as prepared at `1ab6809` (superseded; historical, not part of the recheck)

`mem1.py`, `oneshot.sh` and `test_mem1.py` changed in `d269eaf`, `6271476` and `6f60a7e`; this block is kept as
recorded and is no longer a check block. The source that ran the primary run is in the next section.

```text
ff54bb646836f186f90b5d032413eb21328d778d9b794c2aacc3a78fc0a4b634  build_set.py
2c636eb6a2588e4307c83e6d5552b3651c12e44d5f2ebfd96e0ab5d2d1d78c92  mem1.py
8feb74f97429417e8e504be1abce979b958950e1b80692b332b7d88cde8f8df4  memory_set.json
84e57b8878d350b339ff92e2f0148c096d317d8fafa0c72cc8a6050abf28b4ce  oneshot.sh
e338e03c7342a6976842889c1fc2a28227f3bfa57fb9e272735474b960df3f2f  test_mem1.py
```

## Source at `6f60a7e` (ran the primary run)

`run.json` `code_sha256` of `20261028T161326Z_primary_ctx2048` records `mem1.py` `806d6531…` and `oneshot.sh`
`0da6a521…`, equal to these files at `6f60a7e`. Earlier launches ran earlier commits; see [RUN_INDEX.md](RUN_INDEX.md).

```sha256sums
ff54bb646836f186f90b5d032413eb21328d778d9b794c2aacc3a78fc0a4b634  build_set.py
806d6531e24727c4248825fdaaf619f1938a121b1d30487581b929944ccdd620  mem1.py
8feb74f97429417e8e504be1abce979b958950e1b80692b332b7d88cde8f8df4  memory_set.json
0da6a521b505346e736d88b620925786af596cfa2175896c2f5a09903915ac46  oneshot.sh
d325812817e93c6f07214c31953a1b1e269342ef8993c56d86449417f3dfe96d  test_mem1.py
```

## Archived originals (105)

Every file is a `cp -p` copy of `~/robot_runs/mem1/<same path below runs/>` (2026-10-30), `cmp` equal to its original.
The empty sentinel `~/robot_runs/mem1/.stop_thermal` (a thermal-logger stop marker) is not copied. `thermal.log` is the
complete logger file as of 2026-10-29 08:06Z; the logger was stopped and is not running.

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `runs/20261027T091311Z_primary_ctx2048_smoke/blocks.jsonl` | 24672 | `42bc699d0826c160f057e575338662bde4644c8f2341d00088ebf5207fe398f0` |
| `runs/20261027T091311Z_primary_ctx2048_smoke/requests.jsonl` | 11485 | `59f0853a83c5ed73aaa76f324d70a56b6d94106dbf7e40848cc0a7f7086faee1` |
| `runs/20261027T091311Z_primary_ctx2048_smoke/run.json` | 6373 | `2e2c9da5a8fc7401f5e74d4df014514bafb50da7e2bbcc25ee6915e454958323` |
| `runs/20261027T091311Z_primary_ctx2048_smoke/server_block00_L0_FRESH.log` | 4630 | `4eeacb64b94310d41646906c5c6e1cbf8faaa0a43facd52619776f5c51a7b85c` |
| `runs/20261027T091311Z_primary_ctx2048_smoke/server_block01_L1024_CACHED.log` | 4633 | `244572fb76f189c8d152ad35c95c409ad59128fd48f57357ea635188df9d2f38` |
| `runs/20261027T091311Z_primary_ctx2048_smoke/server_block02_L1024_FRESH.log` | 4633 | `dd4d01bb9b31abad1114b98e6896ff042ab3c4960162f2f417db5a95d53fbd51` |
| `runs/20261027T091311Z_primary_ctx2048_smoke/server_block03_L0_CACHED.log` | 4630 | `1847fa05ebfdddfcef8bb59f3cabf57dfa244c880f5408aa8e9842e53d817d90` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/artifacts.sha256` | 857 | `2aeb1366fa1ef6f715e5405316f9a6603e729ef3d06c1b85b49fd3a065aa5446` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/blocks.jsonl` | 24852 | `f105bcee21084604b8568c37cd4588dab4aab98816c13c653eae9885451dd6a5` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/comparisons.json` | 5782 | `c22e0fd948184535056200806bc04b169f64dfaa9cda401d0a6be82ab86cd1a5` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/report.txt` | 7470 | `9087670bf0394854b50baa7d1a462a3c77afc115d5d63e432e53b1fb56077b24` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/requests.jsonl` | 11450 | `eb74135eae0b3ac12ad6cf1c82f7c80c2d6821d8f84216472c174dc515ce9349` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/result.json` | 252 | `2aa446a90048e439d68a650f8d695b66b5aa99ebe24f422c89890fff048e6529` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/run.json` | 6484 | `bedae207484de7d04889f153946e9df16a63dd774b50ce6cf0cc9bfbaed28e7b` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/server_block00_L0_FRESH.log` | 4630 | `6c5214503857f61a2a4b42c2480ec05fc1d8e2d2f02456eacc607fafb3d432d7` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/server_block01_L1024_CACHED.log` | 4633 | `90706f1fb86ed62206bcbcb92bd5d4ae4a074d9795f8618c1b42f7719eaed537` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/server_block02_L1024_FRESH.log` | 4633 | `aff0c5621f7fa5545addbe8825ff9687e24e38b7db8290444637a2042bd86231` |
| `runs/20261028T011149Z_primary_ctx2048_smoke/server_block03_L0_CACHED.log` | 4630 | `8c57abca04d63af4e39acad3cfbcf106cbb0167edbc63ed9e25b92d27ff43906` |
| `runs/20261028T080156Z_capacity_ctx2048/artifacts.sha256` | 863 | `e98843918ace4f8b763668b6bde69da71e5ec66fd567f6e6c5fc262b7876d4a3` |
| `runs/20261028T080156Z_capacity_ctx2048/blocks.jsonl` | 35386 | `a5ffe667c92bc555c028041f805b77bafe8c5c6ced670b5d4371aa36028c31f3` |
| `runs/20261028T080156Z_capacity_ctx2048/comparisons.json` | 2071 | `a27bdbbad7b15e6866f7f2cb4a5e2fe45039c0698b989f3adab1556340f3c6bd` |
| `runs/20261028T080156Z_capacity_ctx2048/report.txt` | 5947 | `903e0c67e3c43e1db3addea83d6b293a1233c27e9a8919cf8e924dcbe920445c` |
| `runs/20261028T080156Z_capacity_ctx2048/requests.jsonl` | 22395 | `6265287ae786c5d04c3d0c74a59a243016f6b92632999ec04b9430a0c3fada31` |
| `runs/20261028T080156Z_capacity_ctx2048/result.json` | 312 | `93dd9f2c19bb1d7beb7626965f1d8f8a25e1e64f615d136eca7041ee0051edf0` |
| `runs/20261028T080156Z_capacity_ctx2048/run.json` | 7003 | `9f439cd07ec74fb39a1b3edbf206d15d0c437eb5e0cc9ae16c56d6cf5de438cd` |
| `runs/20261028T080156Z_capacity_ctx2048/server_block00_L1024_FRESH.log` | 7849 | `280b6329f8b7319248c7e589b546e2fd7071e2419de066c40cf69cc99978dc77` |
| `runs/20261028T080156Z_capacity_ctx2048/server_block01_L1024_CACHED.log` | 7849 | `69324e6048f84a87140d51ce1a3ac4069f9788a3f767a8ec7986e9bd749b22c5` |
| `runs/20261028T080156Z_capacity_ctx2048/server_block02_L1024_CACHED.log` | 7849 | `8c22c912e4cca717fe5ca97691e19d254ed3a9b65f11520529a06ed93dcb5e7a` |
| `runs/20261028T080156Z_capacity_ctx2048/server_block03_L1024_FRESH.log` | 3829 | `75a90684ab5382435dd17ebd24190903e22aa724da03247769c1dc6d12fd2d83` |
| `runs/20261028T161326Z_primary_ctx2048/artifacts.sha256` | 4842 | `5d95d8c50810df112e8fa62170c9b35157040b471660d924c95d2a548ad840dd` |
| `runs/20261028T161326Z_primary_ctx2048/blocks.jsonl` | 155320 | `98ab40934bc09a3969fcca85dde90b19bf393f23c5e92187d218121dbbc58ae5` |
| `runs/20261028T161326Z_primary_ctx2048/comparisons.json` | 27892 | `666d9a8912b092a83ae5dd35f298dda429c3624cab7fc88f9471e1252bb87403` |
| `runs/20261028T161326Z_primary_ctx2048/report.txt` | 20788 | `ada10770503205b1d088f4ea656a64b47f67a625400bb21b0e6707f4d2d7c95d` |
| `runs/20261028T161326Z_primary_ctx2048/requests.jsonl` | 135457 | `75c6ca0df147e08f6e258b96a751f24f152f3476559a7e15c7383bc197a13224` |
| `runs/20261028T161326Z_primary_ctx2048/result.json` | 28470 | `13be8000ad0143884c1ae6be2a2ef6c6fe70df827ca015a0680a07cece15e7c6` |
| `runs/20261028T161326Z_primary_ctx2048/run.json` | 10741 | `49fc095db43f6332df697100b481c47093b951488c64084347a8e0bcfc4e2b85` |
| `runs/20261028T161326Z_primary_ctx2048/server_block00_L0_FRESH.log` | 7843 | `eb39293bc2c3c6ba41d1770b1ef5d421e2c67a5679fe726bde1222e47f0ec0c5` |
| `runs/20261028T161326Z_primary_ctx2048/server_block01_L1024_CACHED.log` | 7850 | `2c8eb64df88bec96183e5c5695c9b98d9f882341509589f870df354d21edc799` |
| `runs/20261028T161326Z_primary_ctx2048/server_block02_L128_FRESH.log` | 7843 | `9dc2aabf53ec227da95cd6a9415f658c5ec70830bffd3881b6929f4ea8c090a3` |
| `runs/20261028T161326Z_primary_ctx2048/server_block03_L512_CACHED.log` | 7843 | `f2b6009c4d1a25ccf98daffc779ba18747b14ca8ff6de5696a9d7f40d3652b85` |
| `runs/20261028T161326Z_primary_ctx2048/server_block04_L256_FRESH.log` | 7843 | `ffa1aef36228c5c71ab3de2157e92072550556ce82a651fd282e7868c315c526` |
| `runs/20261028T161326Z_primary_ctx2048/server_block05_L256_CACHED.log` | 7843 | `a151df39b60b412673eb6b9436923302ce62a8dbe556c0e2c6d5d71b6328e2d5` |
| `runs/20261028T161326Z_primary_ctx2048/server_block06_L512_FRESH.log` | 7843 | `b31d73c94bf8cdc2de61eae51360141a69a233c7c6531fc8caccb14f00116115` |
| `runs/20261028T161326Z_primary_ctx2048/server_block07_L128_CACHED.log` | 7843 | `601d2898777ce6473c9ed6b1e6a6943f9f3c6e7eb51330ead8834407927b7d77` |
| `runs/20261028T161326Z_primary_ctx2048/server_block08_L1024_FRESH.log` | 7850 | `5771ba09d978ace5ea7e18fe79a7bfb575c57e6d43e7e97c06f7f2137556c2cf` |
| `runs/20261028T161326Z_primary_ctx2048/server_block09_L0_CACHED.log` | 7843 | `653b6128704b3f25967ec26ee2a5380f7e1f06e2e10faed43e4145d089f150c0` |
| `runs/20261028T161326Z_primary_ctx2048/server_block10_L0_CACHED.log` | 7843 | `dbfc267d29c76b81c016f7d71d2a629f1efd87831152a1aeca90c9ab1a907163` |
| `runs/20261028T161326Z_primary_ctx2048/server_block11_L1024_FRESH.log` | 7850 | `a41c76f2ba748182e4a6c007f3766410b055fe93227346b2a8cc42b342f6c5ae` |
| `runs/20261028T161326Z_primary_ctx2048/server_block12_L128_CACHED.log` | 7843 | `98086ca99cacc1d62f126a3c2b1fbb805f055753c2a02c3e5c9dc393324d70d1` |
| `runs/20261028T161326Z_primary_ctx2048/server_block13_L512_FRESH.log` | 7843 | `d147efc6d2536b8a2660da1aa83149d88db79dbd383b4679427183105374adcc` |
| `runs/20261028T161326Z_primary_ctx2048/server_block14_L256_CACHED.log` | 7843 | `cb371ae41697f969dd1f89231143029f5b1d54ee590ca94f2bf14c8c699337ad` |
| `runs/20261028T161326Z_primary_ctx2048/server_block15_L256_FRESH.log` | 7843 | `b77fa07e27d5088d9feeae2e4e3c11a8082f4103bc7120cd74e93a2d23656251` |
| `runs/20261028T161326Z_primary_ctx2048/server_block16_L512_CACHED.log` | 7843 | `25bd0003f9090c33b852c22ce68fef4d7fc9cc5f974dd057617c92ea33419087` |
| `runs/20261028T161326Z_primary_ctx2048/server_block17_L128_FRESH.log` | 7843 | `0e81f64445177125a9a4b232fd299ce984bfc426875ecb782a83497bfc71b981` |
| `runs/20261028T161326Z_primary_ctx2048/server_block18_L1024_CACHED.log` | 7850 | `79e382b2edda34caf0b4dff804f1af2f0df935a6b5330b99ca446b2b52c28773` |
| `runs/20261028T161326Z_primary_ctx2048/server_block19_L0_FRESH.log` | 7843 | `357df7323fb83c84310014a69c6459acb85744669c0468d7af5847668ebe171c` |
| `runs/20261028T161326Z_primary_ctx2048/session01.json` | 2767 | `d3b8f53e92dafeef16bec21cd9ebe3007ebf8a9759bbeffdac99a1549bfef40a` |
| `runs/20261028T161326Z_primary_ctx2048/session01_end.json` | 4206 | `e28c30d2baa3513d102fe6d02d1366dbcbb08a891b7349bf6880723ec0bc3a1d` |
| `runs/20261028T161326Z_primary_ctx2048/session01_report.txt` | 19854 | `5ecfe993db1719531873c331336140ab4ef91c73a2d214e3d8f33e2d5e08099e` |
| `runs/20261028T161326Z_primary_ctx2048/session02.json` | 2767 | `565188d24ef3843f329977bef4b77be095bed996f3719bd447b51155025393e9` |
| `runs/20261028T161326Z_primary_ctx2048/session02_end.json` | 4262 | `b180c0e2d75e849960da35134df2cb46ea0db8a5cd073ebf4c7da73358f7448e` |
| `runs/20261028T161326Z_primary_ctx2048/session02_report.txt` | 20815 | `44abfcf33fc854087da3922ba03a7915c9c737f0febcb45a903e8fe110583329` |
| `runs/20261028T161326Z_primary_ctx2048/session03.json` | 2767 | `6175ceaa1282406947f98a163da8b1784bf7386f900d98e0bc6935d3467b6969` |
| `runs/20261028T161326Z_primary_ctx2048/session03_end.json` | 2284 | `5e0269da4886139fd9580437c8a0fb204a45b1693177741be0dda3ab98b8017b` |
| `runs/20261028T161326Z_primary_ctx2048/session03_report.txt` | 21111 | `e4fe772f7147db4431413cdf57d3bd57add5ac98738c65fce0925ca868ce64d9` |
| `runs/20261028T161326Z_primary_ctx2048/session04.json` | 2767 | `c1fac6071d62de2d63a4898aadefacbe0749c87620cc3a9063d8415894a6ebb7` |
| `runs/20261028T161326Z_primary_ctx2048/session04_end.json` | 2284 | `61b466b3465b2505412df358309d0830c5216b975a81f772168816f0cb980026` |
| `runs/20261028T161326Z_primary_ctx2048/session04_report.txt` | 21407 | `f8926872da2c151366c2c6b57884eec70fe9de9c8e64c76e3ae769cdc538fc97` |
| `runs/20261028T161326Z_primary_ctx2048/session05.json` | 2767 | `4306dc6667eadebcbb510eec4c703f0afa479b42c68e871e2a5d6bf2786b03e7` |
| `runs/20261028T161326Z_primary_ctx2048/session05_end.json` | 4214 | `422dc8b5052990354c99ca8e06eeed42fc41e879cc580404c1cbbd16f6c488e3` |
| `runs/20261028T161326Z_primary_ctx2048/session05_report.txt` | 24062 | `543c35d94b55b88daac53dde91671d771b7a747839e969f7e1c193916042d358` |
| `runs/20261028T161326Z_primary_ctx2048/session06.json` | 2768 | `5c83ab77e95044ff072bb67ff2853ceb246a56a66668e2b0e83f12d00912c1ff` |
| `runs/20261028T161326Z_primary_ctx2048/session06_end.json` | 2285 | `7f400353cb508085239c25b76af8aa5260d3908ea4c278c31baa710d57b3eaac` |
| `runs/20261028T161326Z_primary_ctx2048/session06_report.txt` | 24359 | `955725e863129ad058b7f706a8731789e2bb28003cab4b4ad3e40c59de474a45` |
| `runs/20261028T161326Z_primary_ctx2048/session07.json` | 2768 | `313e17524d3e451b6b704aae5758ccefdb3a8db483818d691c86663432f6ab38` |
| `runs/20261028T161326Z_primary_ctx2048/session07_end.json` | 4282 | `eab9d180ac43bab0a41618ee1abf4d6e1757304626c679c378a72007d1afad5a` |
| `runs/20261028T161326Z_primary_ctx2048/session07_report.txt` | 24595 | `6d0ce12f979804dd349cdba820a090fff01de48d9bdd0bf135875211b5b47829` |
| `runs/20261028T161326Z_primary_ctx2048/session08.json` | 2769 | `1236bf2b7ef753b8aca001bf2d1ee00952b947952c0a4039990cd0ebcf40a16a` |
| `runs/20261028T161326Z_primary_ctx2048/session08_end.json` | 2281 | `e3a5d8889a1df28b87a8214307eb124eba7c630ca9f4d2c3bdf3158fb781e170` |
| `runs/20261028T161326Z_primary_ctx2048/session08_report.txt` | 24893 | `4c5c9cb58e809680915ff5ed95aae8d17c34458723aa6b538b2a26cb79d60b61` |
| `runs/20261028T161326Z_primary_ctx2048/session09.json` | 2769 | `79c9c7fa52579dd0f501648cdeec9f0aae93db31076bf86a4a70243ed9b1e870` |
| `runs/20261028T161326Z_primary_ctx2048/session09_end.json` | 4221 | `5201b69a41fd64eb53713db5e2afdd08bd2038568ce88adca418b4e163d6320d` |
| `runs/20261028T161326Z_primary_ctx2048/session09_report.txt` | 24839 | `56768bb93c5eed4f3d659620a0cf6f2185f73c6c175e13fc5c824b4773580e55` |
| `runs/20261028T161326Z_primary_ctx2048/session10.json` | 2770 | `858a77011b8fd1983451c682993c60f789e96ce121e9044cc4fce86d3ecd2ded` |
| `runs/20261028T161326Z_primary_ctx2048/session10_end.json` | 4193 | `e0c655ff153c70b173e8a11f87d4b12bb95002581e5c862fa2e41e5845bb7ba8` |
| `runs/cpu_cooling_20261028T011136Z.txt` | 1638 | `2ac986dfa22a2368470a91d857eb01c6968e60ac42c7d1f05c9b3ae4392a4de6` |
| `runs/cpu_cooling_diagnostic_20261027T233716Z.txt` | 85386 | `3cfbcf2e00bbf2565295c99dd014d6d3d07ab2c0b6e24794de8ea254b7de5790` |
| `runs/oneshot_console_20261027T010326Z.log` | 131 | `d01817bbf6ccb4ce9b1fcb76b92f5969c45dab89df78d42f4cc8cd3a02dbe089` |
| `runs/oneshot_console_20261027T091259Z.log` | 2811 | `9d33df5be89542291cc5a63129ab7d50c60741d42a420f61c5013a523be5b24e` |
| `runs/oneshot_console_20261027T233716Z.log` | 714 | `ad6f325265429a6646c1ff012c8e57f64d80524aca461b01d80774cecb64d6a5` |
| `runs/oneshot_console_20261028T011136Z.log` | 9681 | `2f11cc5fe87a53419084db1f81bdda5844f7025b3a5bcb01ef03ba8ab0cfe991` |
| `runs/oneshot_console_20261028T080144Z.log` | 15364 | `3ce68bdc2ac8a93539377b16c8afe0764850ce95fa62f1c1dcbf2c4d02fb3ea2` |
| `runs/oneshot_console_20261028T161314Z.log` | 26064 | `c6a2b02bfc85b8af952e0d1872a3d6c25f1708d15ca6e714b5e74ac60635cce9` |
| `runs/oneshot_console_20261028T164120Z.log` | 26200 | `839f60fc055cc568fc698a24f5a4b93d33bd118483b0a61508f1e06bdb5153cb` |
| `runs/oneshot_console_20261028T165941Z.log` | 24752 | `7ca4720120334458181c27e5f15d85ffbc2b5a38a1bc93b56b389b9898963b64` |
| `runs/oneshot_console_20261028T172155Z.log` | 25065 | `accf56c1a124ced25bfaf9025f16d0873ba0f88eaa2e2b48d31041c0ac6199e8` |
| `runs/oneshot_console_20261028T180713Z.log` | 26654 | `0453fc27cdc6bc62cd3add31de3855d7b6aa995be456d54cf6fc4a46af533742` |
| `runs/oneshot_console_20261029T001751Z.log` | 28015 | `1a61d7ac0a908b3cdce631f14d96c639989278b5bd5988c4e97ac42417cea77b` |
| `runs/oneshot_console_20261029T022710Z.log` | 28908 | `f11cc566dabf2c205ce46f90ba38defcbaaf0bcb18cd54ce6aef4ef34e7efd9a` |
| `runs/oneshot_console_20261029T061320Z.log` | 28581 | `689e14ea7115efebfdcc1a7cc7d3faf0a8e764d5dd2f3e71c628c4c27b11acea` |
| `runs/oneshot_console_20261029T071533Z.log` | 26864 | `cae8e9170ed05426e44a89f8b4ad899d1fe64d0f82118386d2e0bcde6f387376` |
| `runs/oneshot_console_20261029T074532Z.log` | 714 | `21ebcb92e764e2f504400f31b58af798d39de5114a1cb534687ebb5b50957385` |
| `runs/oneshot_console_20261029T075858Z.log` | 23880 | `8c6d9f4d5808b417de7f8e41bcc9934cb92b5eddfc97f99cb57124130e8bfb56` |
| `runs/thermal.log` | 44753 | `f6b46ad853e252f3f6b9b968714a9812f20d0951bbca7ee67f0a1c454982bedd` |
| `runs/thermal_logger_sched.txt` | 31 | `1c6d032ce1f03e37a59eeead25daa8ba2a459b64140ad2b80fd67a0537465ce3` |

## Phone-only copies (not archived)

`~/robot_runs/mem1_preserved/` is an owner copy made 2026-10-28 10:15. `20261028T080156Z_capacity_ctx2048/` there is
`diff -r` identical to `runs/20261028T080156Z_capacity_ctx2048/`. In `20261028T080156Z_capacity_ctx2048-external/`,
`oneshot_console_20261028T080144Z.log` and `thermal_logger_sched.txt` are `cmp` equal to the archived files of the same name,
and `thermal.log` (11499 bytes) is byte-identical to the first 11499 bytes of the archived `runs/thermal.log`.

```sha256sums
42bc699d0826c160f057e575338662bde4644c8f2341d00088ebf5207fe398f0  runs/20261027T091311Z_primary_ctx2048_smoke/blocks.jsonl
59f0853a83c5ed73aaa76f324d70a56b6d94106dbf7e40848cc0a7f7086faee1  runs/20261027T091311Z_primary_ctx2048_smoke/requests.jsonl
2e2c9da5a8fc7401f5e74d4df014514bafb50da7e2bbcc25ee6915e454958323  runs/20261027T091311Z_primary_ctx2048_smoke/run.json
4eeacb64b94310d41646906c5c6e1cbf8faaa0a43facd52619776f5c51a7b85c  runs/20261027T091311Z_primary_ctx2048_smoke/server_block00_L0_FRESH.log
244572fb76f189c8d152ad35c95c409ad59128fd48f57357ea635188df9d2f38  runs/20261027T091311Z_primary_ctx2048_smoke/server_block01_L1024_CACHED.log
dd4d01bb9b31abad1114b98e6896ff042ab3c4960162f2f417db5a95d53fbd51  runs/20261027T091311Z_primary_ctx2048_smoke/server_block02_L1024_FRESH.log
1847fa05ebfdddfcef8bb59f3cabf57dfa244c880f5408aa8e9842e53d817d90  runs/20261027T091311Z_primary_ctx2048_smoke/server_block03_L0_CACHED.log
2aeb1366fa1ef6f715e5405316f9a6603e729ef3d06c1b85b49fd3a065aa5446  runs/20261028T011149Z_primary_ctx2048_smoke/artifacts.sha256
f105bcee21084604b8568c37cd4588dab4aab98816c13c653eae9885451dd6a5  runs/20261028T011149Z_primary_ctx2048_smoke/blocks.jsonl
c22e0fd948184535056200806bc04b169f64dfaa9cda401d0a6be82ab86cd1a5  runs/20261028T011149Z_primary_ctx2048_smoke/comparisons.json
9087670bf0394854b50baa7d1a462a3c77afc115d5d63e432e53b1fb56077b24  runs/20261028T011149Z_primary_ctx2048_smoke/report.txt
eb74135eae0b3ac12ad6cf1c82f7c80c2d6821d8f84216472c174dc515ce9349  runs/20261028T011149Z_primary_ctx2048_smoke/requests.jsonl
2aa446a90048e439d68a650f8d695b66b5aa99ebe24f422c89890fff048e6529  runs/20261028T011149Z_primary_ctx2048_smoke/result.json
bedae207484de7d04889f153946e9df16a63dd774b50ce6cf0cc9bfbaed28e7b  runs/20261028T011149Z_primary_ctx2048_smoke/run.json
6c5214503857f61a2a4b42c2480ec05fc1d8e2d2f02456eacc607fafb3d432d7  runs/20261028T011149Z_primary_ctx2048_smoke/server_block00_L0_FRESH.log
90706f1fb86ed62206bcbcb92bd5d4ae4a074d9795f8618c1b42f7719eaed537  runs/20261028T011149Z_primary_ctx2048_smoke/server_block01_L1024_CACHED.log
aff0c5621f7fa5545addbe8825ff9687e24e38b7db8290444637a2042bd86231  runs/20261028T011149Z_primary_ctx2048_smoke/server_block02_L1024_FRESH.log
8c57abca04d63af4e39acad3cfbcf106cbb0167edbc63ed9e25b92d27ff43906  runs/20261028T011149Z_primary_ctx2048_smoke/server_block03_L0_CACHED.log
e98843918ace4f8b763668b6bde69da71e5ec66fd567f6e6c5fc262b7876d4a3  runs/20261028T080156Z_capacity_ctx2048/artifacts.sha256
a5ffe667c92bc555c028041f805b77bafe8c5c6ced670b5d4371aa36028c31f3  runs/20261028T080156Z_capacity_ctx2048/blocks.jsonl
a27bdbbad7b15e6866f7f2cb4a5e2fe45039c0698b989f3adab1556340f3c6bd  runs/20261028T080156Z_capacity_ctx2048/comparisons.json
903e0c67e3c43e1db3addea83d6b293a1233c27e9a8919cf8e924dcbe920445c  runs/20261028T080156Z_capacity_ctx2048/report.txt
6265287ae786c5d04c3d0c74a59a243016f6b92632999ec04b9430a0c3fada31  runs/20261028T080156Z_capacity_ctx2048/requests.jsonl
93dd9f2c19bb1d7beb7626965f1d8f8a25e1e64f615d136eca7041ee0051edf0  runs/20261028T080156Z_capacity_ctx2048/result.json
9f439cd07ec74fb39a1b3edbf206d15d0c437eb5e0cc9ae16c56d6cf5de438cd  runs/20261028T080156Z_capacity_ctx2048/run.json
280b6329f8b7319248c7e589b546e2fd7071e2419de066c40cf69cc99978dc77  runs/20261028T080156Z_capacity_ctx2048/server_block00_L1024_FRESH.log
69324e6048f84a87140d51ce1a3ac4069f9788a3f767a8ec7986e9bd749b22c5  runs/20261028T080156Z_capacity_ctx2048/server_block01_L1024_CACHED.log
8c22c912e4cca717fe5ca97691e19d254ed3a9b65f11520529a06ed93dcb5e7a  runs/20261028T080156Z_capacity_ctx2048/server_block02_L1024_CACHED.log
75a90684ab5382435dd17ebd24190903e22aa724da03247769c1dc6d12fd2d83  runs/20261028T080156Z_capacity_ctx2048/server_block03_L1024_FRESH.log
5d95d8c50810df112e8fa62170c9b35157040b471660d924c95d2a548ad840dd  runs/20261028T161326Z_primary_ctx2048/artifacts.sha256
98ab40934bc09a3969fcca85dde90b19bf393f23c5e92187d218121dbbc58ae5  runs/20261028T161326Z_primary_ctx2048/blocks.jsonl
666d9a8912b092a83ae5dd35f298dda429c3624cab7fc88f9471e1252bb87403  runs/20261028T161326Z_primary_ctx2048/comparisons.json
ada10770503205b1d088f4ea656a64b47f67a625400bb21b0e6707f4d2d7c95d  runs/20261028T161326Z_primary_ctx2048/report.txt
75c6ca0df147e08f6e258b96a751f24f152f3476559a7e15c7383bc197a13224  runs/20261028T161326Z_primary_ctx2048/requests.jsonl
13be8000ad0143884c1ae6be2a2ef6c6fe70df827ca015a0680a07cece15e7c6  runs/20261028T161326Z_primary_ctx2048/result.json
49fc095db43f6332df697100b481c47093b951488c64084347a8e0bcfc4e2b85  runs/20261028T161326Z_primary_ctx2048/run.json
eb39293bc2c3c6ba41d1770b1ef5d421e2c67a5679fe726bde1222e47f0ec0c5  runs/20261028T161326Z_primary_ctx2048/server_block00_L0_FRESH.log
2c8eb64df88bec96183e5c5695c9b98d9f882341509589f870df354d21edc799  runs/20261028T161326Z_primary_ctx2048/server_block01_L1024_CACHED.log
9dc2aabf53ec227da95cd6a9415f658c5ec70830bffd3881b6929f4ea8c090a3  runs/20261028T161326Z_primary_ctx2048/server_block02_L128_FRESH.log
f2b6009c4d1a25ccf98daffc779ba18747b14ca8ff6de5696a9d7f40d3652b85  runs/20261028T161326Z_primary_ctx2048/server_block03_L512_CACHED.log
ffa1aef36228c5c71ab3de2157e92072550556ce82a651fd282e7868c315c526  runs/20261028T161326Z_primary_ctx2048/server_block04_L256_FRESH.log
a151df39b60b412673eb6b9436923302ce62a8dbe556c0e2c6d5d71b6328e2d5  runs/20261028T161326Z_primary_ctx2048/server_block05_L256_CACHED.log
b31d73c94bf8cdc2de61eae51360141a69a233c7c6531fc8caccb14f00116115  runs/20261028T161326Z_primary_ctx2048/server_block06_L512_FRESH.log
601d2898777ce6473c9ed6b1e6a6943f9f3c6e7eb51330ead8834407927b7d77  runs/20261028T161326Z_primary_ctx2048/server_block07_L128_CACHED.log
5771ba09d978ace5ea7e18fe79a7bfb575c57e6d43e7e97c06f7f2137556c2cf  runs/20261028T161326Z_primary_ctx2048/server_block08_L1024_FRESH.log
653b6128704b3f25967ec26ee2a5380f7e1f06e2e10faed43e4145d089f150c0  runs/20261028T161326Z_primary_ctx2048/server_block09_L0_CACHED.log
dbfc267d29c76b81c016f7d71d2a629f1efd87831152a1aeca90c9ab1a907163  runs/20261028T161326Z_primary_ctx2048/server_block10_L0_CACHED.log
a41c76f2ba748182e4a6c007f3766410b055fe93227346b2a8cc42b342f6c5ae  runs/20261028T161326Z_primary_ctx2048/server_block11_L1024_FRESH.log
98086ca99cacc1d62f126a3c2b1fbb805f055753c2a02c3e5c9dc393324d70d1  runs/20261028T161326Z_primary_ctx2048/server_block12_L128_CACHED.log
d147efc6d2536b8a2660da1aa83149d88db79dbd383b4679427183105374adcc  runs/20261028T161326Z_primary_ctx2048/server_block13_L512_FRESH.log
cb371ae41697f969dd1f89231143029f5b1d54ee590ca94f2bf14c8c699337ad  runs/20261028T161326Z_primary_ctx2048/server_block14_L256_CACHED.log
b77fa07e27d5088d9feeae2e4e3c11a8082f4103bc7120cd74e93a2d23656251  runs/20261028T161326Z_primary_ctx2048/server_block15_L256_FRESH.log
25bd0003f9090c33b852c22ce68fef4d7fc9cc5f974dd057617c92ea33419087  runs/20261028T161326Z_primary_ctx2048/server_block16_L512_CACHED.log
0e81f64445177125a9a4b232fd299ce984bfc426875ecb782a83497bfc71b981  runs/20261028T161326Z_primary_ctx2048/server_block17_L128_FRESH.log
79e382b2edda34caf0b4dff804f1af2f0df935a6b5330b99ca446b2b52c28773  runs/20261028T161326Z_primary_ctx2048/server_block18_L1024_CACHED.log
357df7323fb83c84310014a69c6459acb85744669c0468d7af5847668ebe171c  runs/20261028T161326Z_primary_ctx2048/server_block19_L0_FRESH.log
d3b8f53e92dafeef16bec21cd9ebe3007ebf8a9759bbeffdac99a1549bfef40a  runs/20261028T161326Z_primary_ctx2048/session01.json
e28c30d2baa3513d102fe6d02d1366dbcbb08a891b7349bf6880723ec0bc3a1d  runs/20261028T161326Z_primary_ctx2048/session01_end.json
5ecfe993db1719531873c331336140ab4ef91c73a2d214e3d8f33e2d5e08099e  runs/20261028T161326Z_primary_ctx2048/session01_report.txt
565188d24ef3843f329977bef4b77be095bed996f3719bd447b51155025393e9  runs/20261028T161326Z_primary_ctx2048/session02.json
b180c0e2d75e849960da35134df2cb46ea0db8a5cd073ebf4c7da73358f7448e  runs/20261028T161326Z_primary_ctx2048/session02_end.json
44abfcf33fc854087da3922ba03a7915c9c737f0febcb45a903e8fe110583329  runs/20261028T161326Z_primary_ctx2048/session02_report.txt
6175ceaa1282406947f98a163da8b1784bf7386f900d98e0bc6935d3467b6969  runs/20261028T161326Z_primary_ctx2048/session03.json
5e0269da4886139fd9580437c8a0fb204a45b1693177741be0dda3ab98b8017b  runs/20261028T161326Z_primary_ctx2048/session03_end.json
e4fe772f7147db4431413cdf57d3bd57add5ac98738c65fce0925ca868ce64d9  runs/20261028T161326Z_primary_ctx2048/session03_report.txt
c1fac6071d62de2d63a4898aadefacbe0749c87620cc3a9063d8415894a6ebb7  runs/20261028T161326Z_primary_ctx2048/session04.json
61b466b3465b2505412df358309d0830c5216b975a81f772168816f0cb980026  runs/20261028T161326Z_primary_ctx2048/session04_end.json
f8926872da2c151366c2c6b57884eec70fe9de9c8e64c76e3ae769cdc538fc97  runs/20261028T161326Z_primary_ctx2048/session04_report.txt
4306dc6667eadebcbb510eec4c703f0afa479b42c68e871e2a5d6bf2786b03e7  runs/20261028T161326Z_primary_ctx2048/session05.json
422dc8b5052990354c99ca8e06eeed42fc41e879cc580404c1cbbd16f6c488e3  runs/20261028T161326Z_primary_ctx2048/session05_end.json
543c35d94b55b88daac53dde91671d771b7a747839e969f7e1c193916042d358  runs/20261028T161326Z_primary_ctx2048/session05_report.txt
5c83ab77e95044ff072bb67ff2853ceb246a56a66668e2b0e83f12d00912c1ff  runs/20261028T161326Z_primary_ctx2048/session06.json
7f400353cb508085239c25b76af8aa5260d3908ea4c278c31baa710d57b3eaac  runs/20261028T161326Z_primary_ctx2048/session06_end.json
955725e863129ad058b7f706a8731789e2bb28003cab4b4ad3e40c59de474a45  runs/20261028T161326Z_primary_ctx2048/session06_report.txt
313e17524d3e451b6b704aae5758ccefdb3a8db483818d691c86663432f6ab38  runs/20261028T161326Z_primary_ctx2048/session07.json
eab9d180ac43bab0a41618ee1abf4d6e1757304626c679c378a72007d1afad5a  runs/20261028T161326Z_primary_ctx2048/session07_end.json
6d0ce12f979804dd349cdba820a090fff01de48d9bdd0bf135875211b5b47829  runs/20261028T161326Z_primary_ctx2048/session07_report.txt
1236bf2b7ef753b8aca001bf2d1ee00952b947952c0a4039990cd0ebcf40a16a  runs/20261028T161326Z_primary_ctx2048/session08.json
e3a5d8889a1df28b87a8214307eb124eba7c630ca9f4d2c3bdf3158fb781e170  runs/20261028T161326Z_primary_ctx2048/session08_end.json
4c5c9cb58e809680915ff5ed95aae8d17c34458723aa6b538b2a26cb79d60b61  runs/20261028T161326Z_primary_ctx2048/session08_report.txt
79c9c7fa52579dd0f501648cdeec9f0aae93db31076bf86a4a70243ed9b1e870  runs/20261028T161326Z_primary_ctx2048/session09.json
5201b69a41fd64eb53713db5e2afdd08bd2038568ce88adca418b4e163d6320d  runs/20261028T161326Z_primary_ctx2048/session09_end.json
56768bb93c5eed4f3d659620a0cf6f2185f73c6c175e13fc5c824b4773580e55  runs/20261028T161326Z_primary_ctx2048/session09_report.txt
858a77011b8fd1983451c682993c60f789e96ce121e9044cc4fce86d3ecd2ded  runs/20261028T161326Z_primary_ctx2048/session10.json
e0c655ff153c70b173e8a11f87d4b12bb95002581e5c862fa2e41e5845bb7ba8  runs/20261028T161326Z_primary_ctx2048/session10_end.json
2ac986dfa22a2368470a91d857eb01c6968e60ac42c7d1f05c9b3ae4392a4de6  runs/cpu_cooling_20261028T011136Z.txt
3cfbcf2e00bbf2565295c99dd014d6d3d07ab2c0b6e24794de8ea254b7de5790  runs/cpu_cooling_diagnostic_20261027T233716Z.txt
d01817bbf6ccb4ce9b1fcb76b92f5969c45dab89df78d42f4cc8cd3a02dbe089  runs/oneshot_console_20261027T010326Z.log
9d33df5be89542291cc5a63129ab7d50c60741d42a420f61c5013a523be5b24e  runs/oneshot_console_20261027T091259Z.log
ad6f325265429a6646c1ff012c8e57f64d80524aca461b01d80774cecb64d6a5  runs/oneshot_console_20261027T233716Z.log
2f11cc5fe87a53419084db1f81bdda5844f7025b3a5bcb01ef03ba8ab0cfe991  runs/oneshot_console_20261028T011136Z.log
3ce68bdc2ac8a93539377b16c8afe0764850ce95fa62f1c1dcbf2c4d02fb3ea2  runs/oneshot_console_20261028T080144Z.log
c6a2b02bfc85b8af952e0d1872a3d6c25f1708d15ca6e714b5e74ac60635cce9  runs/oneshot_console_20261028T161314Z.log
839f60fc055cc568fc698a24f5a4b93d33bd118483b0a61508f1e06bdb5153cb  runs/oneshot_console_20261028T164120Z.log
7ca4720120334458181c27e5f15d85ffbc2b5a38a1bc93b56b389b9898963b64  runs/oneshot_console_20261028T165941Z.log
accf56c1a124ced25bfaf9025f16d0873ba0f88eaa2e2b48d31041c0ac6199e8  runs/oneshot_console_20261028T172155Z.log
0453fc27cdc6bc62cd3add31de3855d7b6aa995be456d54cf6fc4a46af533742  runs/oneshot_console_20261028T180713Z.log
1a61d7ac0a908b3cdce631f14d96c639989278b5bd5988c4e97ac42417cea77b  runs/oneshot_console_20261029T001751Z.log
f11cc566dabf2c205ce46f90ba38defcbaaf0bcb18cd54ce6aef4ef34e7efd9a  runs/oneshot_console_20261029T022710Z.log
689e14ea7115efebfdcc1a7cc7d3faf0a8e764d5dd2f3e71c628c4c27b11acea  runs/oneshot_console_20261029T061320Z.log
cae8e9170ed05426e44a89f8b4ad899d1fe64d0f82118386d2e0bcde6f387376  runs/oneshot_console_20261029T071533Z.log
21ebcb92e764e2f504400f31b58af798d39de5114a1cb534687ebb5b50957385  runs/oneshot_console_20261029T074532Z.log
8c6d9f4d5808b417de7f8e41bcc9934cb92b5eddfc97f99cb57124130e8bfb56  runs/oneshot_console_20261029T075858Z.log
f6b46ad853e252f3f6b9b968714a9812f20d0951bbca7ee67f0a1c454982bedd  runs/thermal.log
1c6d032ce1f03e37a59eeead25daa8ba2a459b64140ad2b80fd67a0537465ce3  runs/thermal_logger_sched.txt
```
