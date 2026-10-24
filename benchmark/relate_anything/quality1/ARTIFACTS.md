# Artifacts — M2-Q1

Every file in this folder except this one is listed with its bytes and SHA-256 (archival task VIS-0/M2-Q1
catalog completion, 2026-10-24; 95 files). Files marked `tracked (685a44b)` were already in Git and were not
recopied; their working-tree bytes equal `HEAD`. Review copies were compared with their phone originals by `cmp`.
`/termux-home` is the native Termux home (`/data/data/com.termux/files/home`). ARTIFACTS.md excludes its own
size and hash. Recheck from this folder:
`sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Archived files (95)

| Archive path | Provenance | Bytes | SHA-256 |
|---|---|---:|---|
| `README.md` | Coder-authored source/doc in place; tracked (685a44b) | 2622 | `8682306b929e863a462710ce3dc3ed7ba9bf953f420d54f33751d3d0a5eae69d` |
| `RUN_INDEX.md` | archive documentation (this task) | 4120 | `522b7c3d057a3d0b45692839a33c20820a454a9267788c2ad3491a4277f7cf78` |
| `grading/M2Q1_BLIND_GRADES.md` | Local AI grading notes; no phone original outside the repo; tracked (685a44b) | 7145 | `ae9339bbcd31f82ecfa0c6a61fae8b9afa5d6ee4103acf3ac8831b671fb8c31e` |
| `quality1.py` | Coder-authored source/doc in place; tracked (685a44b) | 27765 | `bda996b683706bbaffefdfb6ddbf19f51c60a732aa62aecd5e8f6cc8afc6eb7d` |
| `reports/M2Q1_CODER_REPORT.md` | `/termux-home/review/m2q1/M2Q1_CODER_REPORT.md` (cmp equal); tracked (685a44b) | 26140 | `a1e1731936b56e7acda4fc2c56f440483d178a354d72ff579dda1563a4f13ff9` |
| `reports/round1_stderr.txt` | `/termux-home/review/m2q1/round1_stderr.txt` (cmp equal); untracked until this task (local copy dated 2026-10-19 07:23Z) | 383438 | `8fdbe7c3cbc74ccb0ec390a6c480092a73106c77d3284e22958fc7307968e7bf` |
| `reports/round1_stdout.txt` | `/termux-home/review/m2q1/round1_stdout.txt` (cmp equal); tracked (685a44b) | 4449 | `7504da3cf1043765d74456cd53942594d7cb78e32edec1d45a117858182bafbb` |
| `reports/round2_stderr.txt` | `/termux-home/review/m2q1/round2_stderr.txt` (cmp equal); untracked until this task (local copy dated 2026-10-19 07:23Z) | 293844 | `ba464c3172b8bdda2efec8e2e8ad69448e7e24d876d4dee9bf109e7771e8151a` |
| `reports/round2_stdout.txt` | `/termux-home/review/m2q1/round2_stdout.txt` (cmp equal); tracked (685a44b) | 3777 | `385e47182363401e5dd108146594cb9e846265dd96daabd11379e95dd3516367` |
| `runs/20261017T212311Z/frames/bench_photos__blocked_1.jpg` | written in place by the harness run `20261017T212311Z`; tracked (685a44b) | 58598 | `b7b275eff91067841d7da78999619fa73b42c0078300ec6c8ef157026b1fc222` |
| `runs/20261017T212311Z/frames/bench_photos__blocked_2.jpg` | written in place by the harness run `20261017T212311Z`; tracked (685a44b) | 46332 | `fd6514bf80042f4a3a764a3bdac14038728dda3ac6b2a8a3f41c8f9832aa3a30` |
| `runs/20261017T212311Z/frames/bench_photos__blocked_3.jpg` | written in place by the harness run `20261017T212311Z`; tracked (685a44b) | 49541 | `a9253e00dc255556974a0e1da9bad127f42805befe71961b7ea51931d524fda1` |
| `runs/20261017T212311Z/key.json` | written in place by the harness run `20261017T212311Z`; tracked (685a44b) | 225 | `e3b5d52902912df873718749aafe24f0f8329c41a2774db8b7a2b52257d7bd4f` |
| `runs/20261017T212311Z/manifest.json` | written in place by the harness run `20261017T212311Z`; tracked (685a44b) | 16188 | `77fc619a4c6a0e84d2f828b099c92907db74aff9674e76366995a2cd12fb0bd9` |
| `runs/20261017T212311Z/results.json` | written in place by the harness run `20261017T212311Z`; tracked (685a44b) | 43186 | `90cdcfac4e954df932588c123ccc728d2db86f6170e7e54455316199dce873bb` |
| `runs/20261017T212311Z/run.log` | written in place by the harness run `20261017T212311Z`; tracked (685a44b) | 1140 | `358133507fa19048e6cfe5be33f71f3107bf5c310e3d4db73f994cac0f2df9c1` |
| `runs/20261017T212311Z/sheet_ab.pdf` | written in place by the harness run `20261017T212311Z`; tracked (685a44b) | 238754 | `17d934ff7dd777cc53645d412699de9e475571806bc649f4e878e79d77a31bb7` |
| `runs/20261017T212311Z/sheet_m2.pdf` | written in place by the harness run `20261017T212311Z`; tracked (685a44b) | 389710 | `d930207de9e75cb0a8e8187ebc2197b3e115f88413e564b9a42bfa1d797fc7e3` |
| `runs/20261017T213604Z/frames/bench_photos__blocked_1.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 58598 | `b7b275eff91067841d7da78999619fa73b42c0078300ec6c8ef157026b1fc222` |
| `runs/20261017T213604Z/frames/bench_photos__blocked_2.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 46332 | `fd6514bf80042f4a3a764a3bdac14038728dda3ac6b2a8a3f41c8f9832aa3a30` |
| `runs/20261017T213604Z/frames/bench_photos__blocked_3.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 49541 | `a9253e00dc255556974a0e1da9bad127f42805befe71961b7ea51931d524fda1` |
| `runs/20261017T213604Z/frames/bench_photos__blocked_4.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 52116 | `68d720b218da6f20496bddafc5808a3072026838f40c5bd8d76fa092eccb8ced` |
| `runs/20261017T213604Z/frames/bench_photos__empty_1.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 39384 | `8e7e981f39fe67c064b7c8023e6ba301f3e79685f38a0e885a8bfade4b31b52b` |
| `runs/20261017T213604Z/frames/bench_photos__empty_2.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 50329 | `971ace92f19973a035de33a4dc2148550623d187a7b701eaee9131ac878de40c` |
| `runs/20261017T213604Z/frames/bench_photos__empty_3.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 42470 | `e645598dcee920be28908d11074f2b5714071eac574cbb3b0c68446b6af86bed` |
| `runs/20261017T213604Z/frames/bench_photos__empty_4.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 53048 | `cb592f06112ecf9a6c37f639586dcabed111e0ae9150cb41da832cfc7aea6d67` |
| `runs/20261017T213604Z/frames/bench_photos__empty_5.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 42490 | `ea682e52a196973eaca7a16877b1c47caf8105f4bd9245457084d890b249ce43` |
| `runs/20261017T213604Z/frames/bench_photos__person_facing_040cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 60316 | `d7a0685f0bebf8ed29b45f198f11f92f5727e5c507c75caeeb64f23234fa8d61` |
| `runs/20261017T213604Z/frames/bench_photos__person_facing_060cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 33372 | `9a423a82109488243c11efef7501b452e2a7fe6bb6d448328caee3d4ff3fa5fd` |
| `runs/20261017T213604Z/frames/bench_photos__person_facing_080cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 53502 | `7794fb0877d01971c67d4198909c687c940610bab120c03d454a6f931aa3b33d` |
| `runs/20261017T213604Z/frames/bench_photos__person_facing_100cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 61918 | `95e3d935350df833b13eeaa68aaf1df809827208f62790114641ce48a3035890` |
| `runs/20261017T213604Z/frames/bench_photos__person_facing_150cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 50208 | `03b7d0e05c361ba4ef08f1d42d6ddb18c71ae8e2a53a9c06b035e8a0e0241f11` |
| `runs/20261017T213604Z/frames/bench_photos__person_facing_200cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 48085 | `01bfb23a9876ec87b40b93f8f15a1be33f49de698418eacbbc0ec0b179022990` |
| `runs/20261017T213604Z/frames/bench_photos__person_facing_300cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 41826 | `31b9ebeb029e609656743a196e7505a8afa07b614ae0ede587bd261a2dd7a8b9` |
| `runs/20261017T213604Z/frames/bench_photos__person_sideon_040cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 56018 | `9bd1acc9f59a44eca54bd70350738a99c3af9915dd46a6bb58745fc7d693b490` |
| `runs/20261017T213604Z/frames/bench_photos__person_sideon_060cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 18864 | `43cf18aebf904d6290e8d75ae75c41f006ae041366c67bb6553e41a2a09c2619` |
| `runs/20261017T213604Z/frames/bench_photos__person_sideon_080cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 52460 | `8be4c659c69c7e30f807d209bc5f859f98c03434fe26c54da2379dcabf73f1c1` |
| `runs/20261017T213604Z/frames/bench_photos__person_sideon_100cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 34847 | `838e42867943d8e643e436c06e929ea927190fbc417fd4211edcf8695f156226` |
| `runs/20261017T213604Z/frames/bench_photos__person_sideon_150cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 60385 | `285ccd3f9d856114440611942daea112c3ee3a04946fe4591042339729c75112` |
| `runs/20261017T213604Z/frames/bench_photos__person_sideon_200cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 60598 | `e2901848ceeae4567b1012928922c89261553997c8b8aa07a87fedd11a1e679c` |
| `runs/20261017T213604Z/frames/bench_photos__person_sideon_300cm.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 32807 | `fc43eb330ebd9b5e9e55a2f0369a68c5483d16d31015bfb13e0c192ceaba3afb` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_bed_1.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 23157 | `95a21243c7ea95a9f1ddd2bc76cad9450730cb7bb2d0e33ffddd22b659a796c3` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_bed_2.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 25121 | `8a54dceb0a3c87f154d3e54643f61a44bae8f765760a9cba32d703ddeff1af7e` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_bed_3.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 25749 | `789a3cdf9845e3b656cbedc0620cb1738e8470f0bbaecc40c74509d92488e4dc` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_couch_1.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 28586 | `17fccf29f86fdec349ee5e232250ab239289d9f5328b1a57ee47d99430d7f8cd` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_couch_2.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 38498 | `a7a018281b414ed1837d729613faa8a7cab2d5b37f9087a628947352e2c19fc8` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_couch_3.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 46930 | `48a5a48c00e1fb854b512786514222e36463a3e0e8bd20952adba88d9657bec3` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_1.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 45339 | `c41ba3db98e6dafdc9812a5d691c07e0fbc6db0ee35bb83e683d1fe997e09766` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_2.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 39734 | `2b0fc31c599c265eeb32215ca9cd3e9267f6c0b99ff6c1d63d84e46372dbdc7e` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_3.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 37505 | `97374a703dbf56af09a2db4c5c1cbc1f26f4e4e973af8393c5ab1436f9764441` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_1.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 28730 | `ee823cc95bf2728659aebe7e779a64c178ca7f3ae7f0dab1cc87a7acbbb01500` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_2.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 30576 | `d4bda73e012d730c9ad1e773c19f723288261818effebfa936f60c2c8bba7f4c` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_3.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 23586 | `a372e7e62a45788faba8741ef97ad19109999518cf9b893b76484c41e991e38c` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_tv_1.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 39142 | `e414b4e43ae935e2d779d9b97832dfb5da6e42d5954dfeb5e6ff0ba953911e30` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_tv_2.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 35442 | `2161dc5fca33bd1f9223c6e979a97b947c791a8245e0d4ea533339958b9456af` |
| `runs/20261017T213604Z/frames/bench_photos__roomsig_tv_3.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 38425 | `1b58d305906ac1002b34a89ba9abfa12c3a74e8b7745494cbde83b402bc5a975` |
| `runs/20261017T213604Z/frames/test_photos__PXL_20260405_213419918.MP.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 54884 | `842308e061992023fe67b50186d151adaba0376dd9706a12192cdd378d5150b4` |
| `runs/20261017T213604Z/frames/test_photos__PXL_20260405_213509269.MP.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 58403 | `a90de00455492fea95efdf9cc21265c83cfb9a73629576c360fe79a71a80a48c` |
| `runs/20261017T213604Z/frames/test_photos__PXL_20260405_213514191.MP.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 58389 | `8ff8b9105eddd41407b441de675736b3fe9711741aadd63381e84bb593746cba` |
| `runs/20261017T213604Z/frames/test_photos__PXL_20260405_213532927.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 43886 | `6afbc8fbc5c193ad77d3678eb81f395bb23d37ba37a88d7f0774ea47a01ed7f0` |
| `runs/20261017T213604Z/frames/test_photos__PXL_20260405_213536608.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 37416 | `400fd604a5d12d13f74dd32d8453e7622999f910bd763d492cfedeefa622c1a8` |
| `runs/20261017T213604Z/frames/test_photos__PXL_20260405_213539353.MP.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 46777 | `6c92d90543d773490e1f3cdc0b001d12d7a0c022eb94e3ad555cdf23b0bbc200` |
| `runs/20261017T213604Z/frames/test_photos__abdel_close_back.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 32793 | `2a6bb011065adbc44044c3eb3c1c59b36934be2662fe3c3be51188450e7c4d80` |
| `runs/20261017T213604Z/frames/test_photos__abdel_close_front.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 38858 | `b9bc07af50d1fa3271ab263e2579990c576fcb18fce12b20989b5ebe93eea561` |
| `runs/20261017T213604Z/frames/test_photos__abdel_close_left.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 31059 | `5a15e9c6790fbf474f8dd8afa53fbcfa16eb748e2b9322901dcdd58557120c3c` |
| `runs/20261017T213604Z/frames/test_photos__abdel_close_right.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 37001 | `e0f53687d9c885acbfa1e9ed1855ebd58fb4cd92364d80002da1a54715369e61` |
| `runs/20261017T213604Z/frames/test_photos__abdel_far_back.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 35963 | `a60cbabf527f55aec0318492b81046d9d9b3e13fa650912947bc472f518e2a58` |
| `runs/20261017T213604Z/frames/test_photos__abdel_far_front.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 36264 | `1bd5e236d104bca8842f1250d25daa60010274a499bb5025a41d0025c104314e` |
| `runs/20261017T213604Z/frames/test_photos__abdel_far_left.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 32807 | `fc43eb330ebd9b5e9e55a2f0369a68c5483d16d31015bfb13e0c192ceaba3afb` |
| `runs/20261017T213604Z/frames/test_photos__abdel_far_right.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 36532 | `2f4b74a35e9cbaa06b28df2326b802b3769ce21d926ead486a15ce73d982b844` |
| `runs/20261017T213604Z/frames/test_photos__chiara_close_back.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 60608 | `493b74e1384ead070164efeb7a56bc93876883e2effef5c7b8ad9d6ac95ac00f` |
| `runs/20261017T213604Z/frames/test_photos__chiara_close_front.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 44032 | `fe4a739795497937cd8de8da0e1831cafb9d153706ed72e8f7fdb8cf3392ba98` |
| `runs/20261017T213604Z/frames/test_photos__chiara_close_left.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 40251 | `69725b246e57af9295515891d97e63e8f29ec7e8122a7639eae9e6407158ea84` |
| `runs/20261017T213604Z/frames/test_photos__chiara_close_right.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 60677 | `87ab6a712c10adb2c4ac80b428c3ba1660600608b1ef55f28126914075c2f586` |
| `runs/20261017T213604Z/frames/test_photos__chiara_far_back.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 54820 | `d2343858a3f5f8b775e6f840142b771ba63652726d97c04e86f0424df899607e` |
| `runs/20261017T213604Z/frames/test_photos__chiara_far_front.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 41826 | `31b9ebeb029e609656743a196e7505a8afa07b614ae0ede587bd261a2dd7a8b9` |
| `runs/20261017T213604Z/frames/test_photos__chiara_far_left.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 41326 | `4269540bad790ba4a1fe6f8f6d2d5dbfa6a508e06f10c909967c5b49cf7778a5` |
| `runs/20261017T213604Z/frames/test_photos__chiara_far_right.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 49532 | `c9724e17ba1a94a428fb72f5496fca83a19b0b5710c2f70c3ee3e3fc5edc2183` |
| `runs/20261017T213604Z/frames/test_photos__empty_1.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 39384 | `8e7e981f39fe67c064b7c8023e6ba301f3e79685f38a0e885a8bfade4b31b52b` |
| `runs/20261017T213604Z/frames/test_photos__empty_2.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 50329 | `971ace92f19973a035de33a4dc2148550623d187a7b701eaee9131ac878de40c` |
| `runs/20261017T213604Z/frames/test_photos__empty_3.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 42470 | `e645598dcee920be28908d11074f2b5714071eac574cbb3b0c68446b6af86bed` |
| `runs/20261017T213604Z/frames/test_photos__empty_4.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 53048 | `cb592f06112ecf9a6c37f639586dcabed111e0ae9150cb41da832cfc7aea6d67` |
| `runs/20261017T213604Z/frames/test_photos__empty_5.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 42490 | `ea682e52a196973eaca7a16877b1c47caf8105f4bd9245457084d890b249ce43` |
| `runs/20261017T213604Z/frames/test_photos__scene_test.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 47368 | `d7284fc275d482f4393a8dca35f6b6b7b2704902012b1b2620b042c134a47b14` |
| `runs/20261017T213604Z/frames/test_photos__stereo_a.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 46332 | `fd6514bf80042f4a3a764a3bdac14038728dda3ac6b2a8a3f41c8f9832aa3a30` |
| `runs/20261017T213604Z/frames/test_photos__stereo_b.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 49541 | `a9253e00dc255556974a0e1da9bad127f42805befe71961b7ea51931d524fda1` |
| `runs/20261017T213604Z/frames/test_photos__table_stereo_a.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 52116 | `68d720b218da6f20496bddafc5808a3072026838f40c5bd8d76fa092eccb8ced` |
| `runs/20261017T213604Z/frames/test_photos__table_top.jpg` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 58598 | `b7b275eff91067841d7da78999619fa73b42c0078300ec6c8ef157026b1fc222` |
| `runs/20261017T213604Z/key.json` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 4777 | `4ee960baccd0eedc86e32a73412c2064fcfb04be89af4bb091617b0421dc8eea` |
| `runs/20261017T213604Z/manifest.json` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 44521 | `235e19a4a213c1db0d238c6f3029c4ff5a676774f279c7707166dd16258e2346` |
| `runs/20261017T213604Z/results.json` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 558579 | `924fcc10ad7c6249c2baac1c8fd4f63bd5749bbf081dba30274be0101808be9f` |
| `runs/20261017T213604Z/run.log` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 10509 | `5558f8eac0d6d80cf02014996d348c370ef189d1b8007c20ff15a290b0e0a19a` |
| `runs/20261017T213604Z/sheet_ab.pdf` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 4339568 | `bfb0fcede7f2c508f06191754da2310730dc4fb7bd6886e7fe31288da91399ed` |
| `runs/20261017T213604Z/sheet_m2.pdf` | written in place by the harness run `20261017T213604Z`; tracked (685a44b) | 6267115 | `3445940770c59a7cf899727326b592b201fcc69eeb157658f193f42cc8229585` |
| `test_quality1.py` | Coder-authored source/doc in place; tracked (685a44b) | 13009 | `43ea5c7d7c4665ff3e3e415dae007f7261623a40d123d8fca2097d67c40addfa` |

## Phone-only review evidence (not archived)

Kept in `/termux-home/review/m2q1/` and not copied, so these hashes pin them without making them archive files.
The isolated review copies `round1_copy/` and `round2_copy/` (160 MB each) are not hashed here. Each round's pre and
post snapshot are byte-identical.

| Phone path | Bytes | SHA-256 |
|---|---:|---|
| `/termux-home/review/m2q1/round1_audit.json` | 142 | `accfffb93a2c3a8f74ba9efe2894bb80b5d9810fe2ca0524fc5754db6180fb4d` |
| `/termux-home/review/m2q1/round2_audit.json` | 142 | `accfffb93a2c3a8f74ba9efe2894bb80b5d9810fe2ca0524fc5754db6180fb4d` |
| `/termux-home/review/m2q1/round1_pre.json` | 946467 | `188266a8010482c2553960fc256b35b60e0b4d05771bc3a70410fda13f8dacc2` |
| `/termux-home/review/m2q1/round1_post.json` | 946467 | `188266a8010482c2553960fc256b35b60e0b4d05771bc3a70410fda13f8dacc2` |
| `/termux-home/review/m2q1/round2_pre.json` | 946573 | `eb7c99f300ad59105d686140ab0e3c064cdf443f10ce099c1fcd479ad8610de4` |
| `/termux-home/review/m2q1/round2_post.json` | 946573 | `eb7c99f300ad59105d686140ab0e3c064cdf443f10ce099c1fcd479ad8610de4` |
| `/termux-home/review/m2q1/run_review_round1.py` | 5902 | `3be40734bd3c220bdf68d024acdb555ee0f5ded4c8539b13f0b44b7f5b262592` |
| `/termux-home/review/m2q1/run_review_round2.py` | 9630 | `e91a04e4942727b3d831262e4942355646f8e640e944a1656ec75417f7f68104` |

Owner Download copies (`cmp` equal to the archived full-run files on 2026-10-24):
`/storage/emulated/0/Download/m2q1/sheet_m2.pdf`, `sheet_ab.pdf` and `key.json` = the same names in
`runs/20261017T213604Z/`. Source photos in `bench_photos/` and `test_photos/` stay outside Git; their SHA-256 and bytes
are `processed_photos[].source_sha256`/`source_bytes` in each run's `manifest.json`. Model files stay outside Git;
their hashes are in the manifests and the Coder report §2.

```sha256sums
8682306b929e863a462710ce3dc3ed7ba9bf953f420d54f33751d3d0a5eae69d  README.md
522b7c3d057a3d0b45692839a33c20820a454a9267788c2ad3491a4277f7cf78  RUN_INDEX.md
ae9339bbcd31f82ecfa0c6a61fae8b9afa5d6ee4103acf3ac8831b671fb8c31e  grading/M2Q1_BLIND_GRADES.md
bda996b683706bbaffefdfb6ddbf19f51c60a732aa62aecd5e8f6cc8afc6eb7d  quality1.py
a1e1731936b56e7acda4fc2c56f440483d178a354d72ff579dda1563a4f13ff9  reports/M2Q1_CODER_REPORT.md
8fdbe7c3cbc74ccb0ec390a6c480092a73106c77d3284e22958fc7307968e7bf  reports/round1_stderr.txt
7504da3cf1043765d74456cd53942594d7cb78e32edec1d45a117858182bafbb  reports/round1_stdout.txt
ba464c3172b8bdda2efec8e2e8ad69448e7e24d876d4dee9bf109e7771e8151a  reports/round2_stderr.txt
385e47182363401e5dd108146594cb9e846265dd96daabd11379e95dd3516367  reports/round2_stdout.txt
b7b275eff91067841d7da78999619fa73b42c0078300ec6c8ef157026b1fc222  runs/20261017T212311Z/frames/bench_photos__blocked_1.jpg
fd6514bf80042f4a3a764a3bdac14038728dda3ac6b2a8a3f41c8f9832aa3a30  runs/20261017T212311Z/frames/bench_photos__blocked_2.jpg
a9253e00dc255556974a0e1da9bad127f42805befe71961b7ea51931d524fda1  runs/20261017T212311Z/frames/bench_photos__blocked_3.jpg
e3b5d52902912df873718749aafe24f0f8329c41a2774db8b7a2b52257d7bd4f  runs/20261017T212311Z/key.json
77fc619a4c6a0e84d2f828b099c92907db74aff9674e76366995a2cd12fb0bd9  runs/20261017T212311Z/manifest.json
90cdcfac4e954df932588c123ccc728d2db86f6170e7e54455316199dce873bb  runs/20261017T212311Z/results.json
358133507fa19048e6cfe5be33f71f3107bf5c310e3d4db73f994cac0f2df9c1  runs/20261017T212311Z/run.log
17d934ff7dd777cc53645d412699de9e475571806bc649f4e878e79d77a31bb7  runs/20261017T212311Z/sheet_ab.pdf
d930207de9e75cb0a8e8187ebc2197b3e115f88413e564b9a42bfa1d797fc7e3  runs/20261017T212311Z/sheet_m2.pdf
b7b275eff91067841d7da78999619fa73b42c0078300ec6c8ef157026b1fc222  runs/20261017T213604Z/frames/bench_photos__blocked_1.jpg
fd6514bf80042f4a3a764a3bdac14038728dda3ac6b2a8a3f41c8f9832aa3a30  runs/20261017T213604Z/frames/bench_photos__blocked_2.jpg
a9253e00dc255556974a0e1da9bad127f42805befe71961b7ea51931d524fda1  runs/20261017T213604Z/frames/bench_photos__blocked_3.jpg
68d720b218da6f20496bddafc5808a3072026838f40c5bd8d76fa092eccb8ced  runs/20261017T213604Z/frames/bench_photos__blocked_4.jpg
8e7e981f39fe67c064b7c8023e6ba301f3e79685f38a0e885a8bfade4b31b52b  runs/20261017T213604Z/frames/bench_photos__empty_1.jpg
971ace92f19973a035de33a4dc2148550623d187a7b701eaee9131ac878de40c  runs/20261017T213604Z/frames/bench_photos__empty_2.jpg
e645598dcee920be28908d11074f2b5714071eac574cbb3b0c68446b6af86bed  runs/20261017T213604Z/frames/bench_photos__empty_3.jpg
cb592f06112ecf9a6c37f639586dcabed111e0ae9150cb41da832cfc7aea6d67  runs/20261017T213604Z/frames/bench_photos__empty_4.jpg
ea682e52a196973eaca7a16877b1c47caf8105f4bd9245457084d890b249ce43  runs/20261017T213604Z/frames/bench_photos__empty_5.jpg
d7a0685f0bebf8ed29b45f198f11f92f5727e5c507c75caeeb64f23234fa8d61  runs/20261017T213604Z/frames/bench_photos__person_facing_040cm.jpg
9a423a82109488243c11efef7501b452e2a7fe6bb6d448328caee3d4ff3fa5fd  runs/20261017T213604Z/frames/bench_photos__person_facing_060cm.jpg
7794fb0877d01971c67d4198909c687c940610bab120c03d454a6f931aa3b33d  runs/20261017T213604Z/frames/bench_photos__person_facing_080cm.jpg
95e3d935350df833b13eeaa68aaf1df809827208f62790114641ce48a3035890  runs/20261017T213604Z/frames/bench_photos__person_facing_100cm.jpg
03b7d0e05c361ba4ef08f1d42d6ddb18c71ae8e2a53a9c06b035e8a0e0241f11  runs/20261017T213604Z/frames/bench_photos__person_facing_150cm.jpg
01bfb23a9876ec87b40b93f8f15a1be33f49de698418eacbbc0ec0b179022990  runs/20261017T213604Z/frames/bench_photos__person_facing_200cm.jpg
31b9ebeb029e609656743a196e7505a8afa07b614ae0ede587bd261a2dd7a8b9  runs/20261017T213604Z/frames/bench_photos__person_facing_300cm.jpg
9bd1acc9f59a44eca54bd70350738a99c3af9915dd46a6bb58745fc7d693b490  runs/20261017T213604Z/frames/bench_photos__person_sideon_040cm.jpg
43cf18aebf904d6290e8d75ae75c41f006ae041366c67bb6553e41a2a09c2619  runs/20261017T213604Z/frames/bench_photos__person_sideon_060cm.jpg
8be4c659c69c7e30f807d209bc5f859f98c03434fe26c54da2379dcabf73f1c1  runs/20261017T213604Z/frames/bench_photos__person_sideon_080cm.jpg
838e42867943d8e643e436c06e929ea927190fbc417fd4211edcf8695f156226  runs/20261017T213604Z/frames/bench_photos__person_sideon_100cm.jpg
285ccd3f9d856114440611942daea112c3ee3a04946fe4591042339729c75112  runs/20261017T213604Z/frames/bench_photos__person_sideon_150cm.jpg
e2901848ceeae4567b1012928922c89261553997c8b8aa07a87fedd11a1e679c  runs/20261017T213604Z/frames/bench_photos__person_sideon_200cm.jpg
fc43eb330ebd9b5e9e55a2f0369a68c5483d16d31015bfb13e0c192ceaba3afb  runs/20261017T213604Z/frames/bench_photos__person_sideon_300cm.jpg
95a21243c7ea95a9f1ddd2bc76cad9450730cb7bb2d0e33ffddd22b659a796c3  runs/20261017T213604Z/frames/bench_photos__roomsig_bed_1.jpg
8a54dceb0a3c87f154d3e54643f61a44bae8f765760a9cba32d703ddeff1af7e  runs/20261017T213604Z/frames/bench_photos__roomsig_bed_2.jpg
789a3cdf9845e3b656cbedc0620cb1738e8470f0bbaecc40c74509d92488e4dc  runs/20261017T213604Z/frames/bench_photos__roomsig_bed_3.jpg
17fccf29f86fdec349ee5e232250ab239289d9f5328b1a57ee47d99430d7f8cd  runs/20261017T213604Z/frames/bench_photos__roomsig_couch_1.jpg
a7a018281b414ed1837d729613faa8a7cab2d5b37f9087a628947352e2c19fc8  runs/20261017T213604Z/frames/bench_photos__roomsig_couch_2.jpg
48a5a48c00e1fb854b512786514222e36463a3e0e8bd20952adba88d9657bec3  runs/20261017T213604Z/frames/bench_photos__roomsig_couch_3.jpg
c41ba3db98e6dafdc9812a5d691c07e0fbc6db0ee35bb83e683d1fe997e09766  runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_1.jpg
2b0fc31c599c265eeb32215ca9cd3e9267f6c0b99ff6c1d63d84e46372dbdc7e  runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_2.jpg
97374a703dbf56af09a2db4c5c1cbc1f26f4e4e973af8393c5ab1436f9764441  runs/20261017T213604Z/frames/bench_photos__roomsig_refrigerator_3.jpg
ee823cc95bf2728659aebe7e779a64c178ca7f3ae7f0dab1cc87a7acbbb01500  runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_1.jpg
d4bda73e012d730c9ad1e773c19f723288261818effebfa936f60c2c8bba7f4c  runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_2.jpg
a372e7e62a45788faba8741ef97ad19109999518cf9b893b76484c41e991e38c  runs/20261017T213604Z/frames/bench_photos__roomsig_toilet_3.jpg
e414b4e43ae935e2d779d9b97832dfb5da6e42d5954dfeb5e6ff0ba953911e30  runs/20261017T213604Z/frames/bench_photos__roomsig_tv_1.jpg
2161dc5fca33bd1f9223c6e979a97b947c791a8245e0d4ea533339958b9456af  runs/20261017T213604Z/frames/bench_photos__roomsig_tv_2.jpg
1b58d305906ac1002b34a89ba9abfa12c3a74e8b7745494cbde83b402bc5a975  runs/20261017T213604Z/frames/bench_photos__roomsig_tv_3.jpg
842308e061992023fe67b50186d151adaba0376dd9706a12192cdd378d5150b4  runs/20261017T213604Z/frames/test_photos__PXL_20260405_213419918.MP.jpg
a90de00455492fea95efdf9cc21265c83cfb9a73629576c360fe79a71a80a48c  runs/20261017T213604Z/frames/test_photos__PXL_20260405_213509269.MP.jpg
8ff8b9105eddd41407b441de675736b3fe9711741aadd63381e84bb593746cba  runs/20261017T213604Z/frames/test_photos__PXL_20260405_213514191.MP.jpg
6afbc8fbc5c193ad77d3678eb81f395bb23d37ba37a88d7f0774ea47a01ed7f0  runs/20261017T213604Z/frames/test_photos__PXL_20260405_213532927.jpg
400fd604a5d12d13f74dd32d8453e7622999f910bd763d492cfedeefa622c1a8  runs/20261017T213604Z/frames/test_photos__PXL_20260405_213536608.jpg
6c92d90543d773490e1f3cdc0b001d12d7a0c022eb94e3ad555cdf23b0bbc200  runs/20261017T213604Z/frames/test_photos__PXL_20260405_213539353.MP.jpg
2a6bb011065adbc44044c3eb3c1c59b36934be2662fe3c3be51188450e7c4d80  runs/20261017T213604Z/frames/test_photos__abdel_close_back.jpg
b9bc07af50d1fa3271ab263e2579990c576fcb18fce12b20989b5ebe93eea561  runs/20261017T213604Z/frames/test_photos__abdel_close_front.jpg
5a15e9c6790fbf474f8dd8afa53fbcfa16eb748e2b9322901dcdd58557120c3c  runs/20261017T213604Z/frames/test_photos__abdel_close_left.jpg
e0f53687d9c885acbfa1e9ed1855ebd58fb4cd92364d80002da1a54715369e61  runs/20261017T213604Z/frames/test_photos__abdel_close_right.jpg
a60cbabf527f55aec0318492b81046d9d9b3e13fa650912947bc472f518e2a58  runs/20261017T213604Z/frames/test_photos__abdel_far_back.jpg
1bd5e236d104bca8842f1250d25daa60010274a499bb5025a41d0025c104314e  runs/20261017T213604Z/frames/test_photos__abdel_far_front.jpg
fc43eb330ebd9b5e9e55a2f0369a68c5483d16d31015bfb13e0c192ceaba3afb  runs/20261017T213604Z/frames/test_photos__abdel_far_left.jpg
2f4b74a35e9cbaa06b28df2326b802b3769ce21d926ead486a15ce73d982b844  runs/20261017T213604Z/frames/test_photos__abdel_far_right.jpg
493b74e1384ead070164efeb7a56bc93876883e2effef5c7b8ad9d6ac95ac00f  runs/20261017T213604Z/frames/test_photos__chiara_close_back.jpg
fe4a739795497937cd8de8da0e1831cafb9d153706ed72e8f7fdb8cf3392ba98  runs/20261017T213604Z/frames/test_photos__chiara_close_front.jpg
69725b246e57af9295515891d97e63e8f29ec7e8122a7639eae9e6407158ea84  runs/20261017T213604Z/frames/test_photos__chiara_close_left.jpg
87ab6a712c10adb2c4ac80b428c3ba1660600608b1ef55f28126914075c2f586  runs/20261017T213604Z/frames/test_photos__chiara_close_right.jpg
d2343858a3f5f8b775e6f840142b771ba63652726d97c04e86f0424df899607e  runs/20261017T213604Z/frames/test_photos__chiara_far_back.jpg
31b9ebeb029e609656743a196e7505a8afa07b614ae0ede587bd261a2dd7a8b9  runs/20261017T213604Z/frames/test_photos__chiara_far_front.jpg
4269540bad790ba4a1fe6f8f6d2d5dbfa6a508e06f10c909967c5b49cf7778a5  runs/20261017T213604Z/frames/test_photos__chiara_far_left.jpg
c9724e17ba1a94a428fb72f5496fca83a19b0b5710c2f70c3ee3e3fc5edc2183  runs/20261017T213604Z/frames/test_photos__chiara_far_right.jpg
8e7e981f39fe67c064b7c8023e6ba301f3e79685f38a0e885a8bfade4b31b52b  runs/20261017T213604Z/frames/test_photos__empty_1.jpg
971ace92f19973a035de33a4dc2148550623d187a7b701eaee9131ac878de40c  runs/20261017T213604Z/frames/test_photos__empty_2.jpg
e645598dcee920be28908d11074f2b5714071eac574cbb3b0c68446b6af86bed  runs/20261017T213604Z/frames/test_photos__empty_3.jpg
cb592f06112ecf9a6c37f639586dcabed111e0ae9150cb41da832cfc7aea6d67  runs/20261017T213604Z/frames/test_photos__empty_4.jpg
ea682e52a196973eaca7a16877b1c47caf8105f4bd9245457084d890b249ce43  runs/20261017T213604Z/frames/test_photos__empty_5.jpg
d7284fc275d482f4393a8dca35f6b6b7b2704902012b1b2620b042c134a47b14  runs/20261017T213604Z/frames/test_photos__scene_test.jpg
fd6514bf80042f4a3a764a3bdac14038728dda3ac6b2a8a3f41c8f9832aa3a30  runs/20261017T213604Z/frames/test_photos__stereo_a.jpg
a9253e00dc255556974a0e1da9bad127f42805befe71961b7ea51931d524fda1  runs/20261017T213604Z/frames/test_photos__stereo_b.jpg
68d720b218da6f20496bddafc5808a3072026838f40c5bd8d76fa092eccb8ced  runs/20261017T213604Z/frames/test_photos__table_stereo_a.jpg
b7b275eff91067841d7da78999619fa73b42c0078300ec6c8ef157026b1fc222  runs/20261017T213604Z/frames/test_photos__table_top.jpg
4ee960baccd0eedc86e32a73412c2064fcfb04be89af4bb091617b0421dc8eea  runs/20261017T213604Z/key.json
235e19a4a213c1db0d238c6f3029c4ff5a676774f279c7707166dd16258e2346  runs/20261017T213604Z/manifest.json
924fcc10ad7c6249c2baac1c8fd4f63bd5749bbf081dba30274be0101808be9f  runs/20261017T213604Z/results.json
5558f8eac0d6d80cf02014996d348c370ef189d1b8007c20ff15a290b0e0a19a  runs/20261017T213604Z/run.log
bfb0fcede7f2c508f06191754da2310730dc4fb7bd6886e7fe31288da91399ed  runs/20261017T213604Z/sheet_ab.pdf
3445940770c59a7cf899727326b592b201fcc69eeb157658f193f42cc8229585  runs/20261017T213604Z/sheet_m2.pdf
43ea5c7d7c4665ff3e3e415dae007f7261623a40d123d8fca2097d67c40addfa  test_quality1.py
```
