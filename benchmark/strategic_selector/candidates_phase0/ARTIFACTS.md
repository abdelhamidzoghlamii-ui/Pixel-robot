# Artifacts

Every archived file below with its SHA-256. Copied files were compared byte for byte (`cmp`) with the phone original at archive time, and the phone and archive SHA-256 values match. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Copied from the phone (55 files)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `raw/base_pins.txt` | `/termux-home/sel-candidates/base_pins.txt` | 731 | `e766bd5369e4b633adc0de49d28d201f818d229a1dbd71edec0b5714b09872d3` |
| `raw/laya-micro/build1_prune.block.json` | `/termux-home/sel-candidates/laya-micro/build1_prune.block.json` | 512 | `bfe4bcb63977897fa5bc03844a300ea326d85a80278955ef5fb2d6ea6c67f27f` |
| `raw/laya-micro/build1_prune.stderr.txt` | `/termux-home/sel-candidates/laya-micro/build1_prune.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `raw/laya-micro/build1_prune.stdout.txt` | `/termux-home/sel-candidates/laya-micro/build1_prune.stdout.txt` | 186 | `8a8faba9e9079a0170392b11f4a3701fd43911bb3cea9103a7c954ec3d59cf58` |
| `raw/laya-micro/build1_prune_8192.block.json` | `/termux-home/sel-candidates/laya-micro/build1_prune_8192.block.json` | 522 | `4fe891e769bafd2411a2ce559266a74236b09a8fd8666f9a7abb4c6b0d2482e9` |
| `raw/laya-micro/build1_prune_8192.stderr.txt` | `/termux-home/sel-candidates/laya-micro/build1_prune_8192.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `raw/laya-micro/build1_prune_8192.stdout.txt` | `/termux-home/sel-candidates/laya-micro/build1_prune_8192.stdout.txt` | 193 | `838056791b4b1c62036c4bd03f2bfd43037f1bad97db0c1e205ec1144f19359a` |
| `raw/laya-micro/build2_export.block.json` | `/termux-home/sel-candidates/laya-micro/build2_export.block.json` | 456 | `6bc62362a36645497ac385ec17ed29cf8939d36c40ad7d118481a06306a5021f` |
| `raw/laya-micro/build2_export.stderr.txt` | `/termux-home/sel-candidates/laya-micro/build2_export.stderr.txt` | 1850 | `1367762b98a6ecbcc3bcf5c451c487215d99fb5340e246b61088573d2ce89a27` |
| `raw/laya-micro/build2_export.stdout.txt` | `/termux-home/sel-candidates/laya-micro/build2_export.stdout.txt` | 1056 | `0c7e1e71b40b9af290f9df04d04894e68ab6ed6aa952031a9d3b971f55df2388` |
| `raw/laya-micro/build2_export_8192.block.json` | `/termux-home/sel-candidates/laya-micro/build2_export_8192.block.json` | 471 | `41000ba514e42a66f22211cb68f376a3bc8f6d2242ae08f4ec47502fb4e950bf` |
| `raw/laya-micro/build2_export_8192.stderr.txt` | `/termux-home/sel-candidates/laya-micro/build2_export_8192.stderr.txt` | 1850 | `b63a9b4703b40d7c2cd3d1279bd2c24cbe8db4e85b793061d978b0158ff837cd` |
| `raw/laya-micro/build2_export_8192.stdout.txt` | `/termux-home/sel-candidates/laya-micro/build2_export_8192.stdout.txt` | 1076 | `dd946839a0ad34f10a733c5a9ed8c53060b633d984edfd9105a5fdf7622379f2` |
| `raw/laya-micro/build3_quantize.block.json` | `/termux-home/sel-candidates/laya-micro/build3_quantize.block.json` | 476 | `8a81c0f033d430f48beb8cf3dfcb65b0624abfc3ab05d17e32c58ff0d1825fa4` |
| `raw/laya-micro/build3_quantize.stderr.txt` | `/termux-home/sel-candidates/laya-micro/build3_quantize.stderr.txt` | 201944 | `735a1cf497b00c89e9a4dfbd7ba15938bc3c5dac4a23c82bb06d016e1a299d38` |
| `raw/laya-micro/build3_quantize.stdout.txt` | `/termux-home/sel-candidates/laya-micro/build3_quantize.stdout.txt` | 351 | `d6e6a30a60f3479a6dcbe1ed7a104612e513ede1dae5cceb4d173f64d0bf6c06` |
| `raw/laya-micro/build3_quantize_8192.block.json` | `/termux-home/sel-candidates/laya-micro/build3_quantize_8192.block.json` | 491 | `fe1a50bb9225686c9d409733ec181dd552c9e7d549ab6c9bd90b021c02ea7cc7` |
| `raw/laya-micro/build3_quantize_8192.stderr.txt` | `/termux-home/sel-candidates/laya-micro/build3_quantize_8192.stderr.txt` | 201944 | `f694d01d4764e47c506cdf0387611962c7defba9cfb206bffce0f2ebaa3f3688` |
| `raw/laya-micro/build3_quantize_8192.stdout.txt` | `/termux-home/sel-candidates/laya-micro/build3_quantize_8192.stdout.txt` | 351 | `254a023f3116320fb1f386d32840e6a7a968a82520bd93ef6a95258785fbf409` |
| `raw/laya-micro/build_8192_sha256.txt` | `/termux-home/sel-candidates/laya-micro/build_8192_sha256.txt` | 416 | `a2563885daf0301b5434b167534aa65b22724f688a07c25d41fd3ca17ba4855d` |
| `raw/laya-micro/build_sha256.txt` | `/termux-home/sel-candidates/laya-micro/build_sha256.txt` | 396 | `5636bfb3464f406267d89d77f6c468c2d515ad52a3b02ac02fe65d65cf742f7e` |
| `raw/laya-micro/download_stderr.txt` | `/termux-home/sel-candidates/laya-micro/download_stderr.txt` | 511 | `2926fd98985888f8a7b59fe03182d0fabe1b6ab171ba2d87af9d21b891a041e9` |
| `raw/laya-micro/download_stdout.txt` | `/termux-home/sel-candidates/laya-micro/download_stdout.txt` | 144 | `91034285c2c3674c84bf795e89b3f59515242f379a0e18ced2a368371a10e5ea` |
| `raw/laya-micro/laya-micro-8192.meta.json` | `/termux-home/sel-candidates/laya-micro/laya-micro-8192.meta.json (deleted from the phone 2026-09-26 after archiving)` | 254 | `7dcede60cb05aa98a542dbfe37849c849b9848013eef186f9733badec82c0b7d` |
| `raw/laya-micro/laya-micro.meta.json` | `/termux-home/sel-candidates/laya-micro/laya-micro.meta.json (deleted from the phone 2026-09-26 after archiving)` | 254 | `7dcede60cb05aa98a542dbfe37849c849b9848013eef186f9733badec82c0b7d` |
| `raw/laya-micro/pins.txt` | `/termux-home/sel-candidates/laya-micro/pins.txt` | 744 | `a6d50b8c7439cf6262eb2b7f43e8302e3570721e8256b32025aec5d7dfdf7718` |
| `raw/laya-micro/pip_stderr.txt` | `/termux-home/sel-candidates/laya-micro/pip_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `raw/laya-micro/pip_stdout.txt` | `/termux-home/sel-candidates/laya-micro/pip_stdout.txt` | 12990 | `69af850f97760e0d51f37e712ac31164e5d6cb80a9e176c5971d0158b003909a` |
| `raw/laya-micro/sha256.txt` | `/termux-home/sel-candidates/laya-micro/sha256.txt` | 538 | `8cdd445165a2452a579df115e557d0091238f8bcff18637883902093c921709c` |
| `raw/laya-micro/shape_timing.py` | `/termux-home/sel-candidates/laya-micro/shape_timing.py` | 1584 | `b5ad60c2d6c663bc3cba895dff7f3d83362227cbd4eed0a6d8509daeb85e3927` |
| `raw/laya-micro/shape_timing.stderr.txt` | `/termux-home/sel-candidates/laya-micro/shape_timing.stderr.txt` | 313 | `507d48b527c08a37a444ee424d6c31ebf30abde895eb05997278b8908a538d69` |
| `raw/laya-micro/shape_timing.stdout.txt` | `/termux-home/sel-candidates/laya-micro/shape_timing.stdout.txt` | 241 | `36587bc8e34bf5958e315326e988320abab7ce6de93f9e650a3f4919e32cc57d` |
| `raw/laya-micro/smoke.block.json` | `/termux-home/sel-candidates/laya-micro/smoke.block.json` | 342 | `db79143b475616c0e020f3a539cd0bbc1dd8b7bea4f7c7dc3a152e190749dd75` |
| `raw/laya-micro/smoke.stderr.txt` | `/termux-home/sel-candidates/laya-micro/smoke.stderr.txt` | 313 | `e0932157a6161dcd60392569a0085394c452ed1830058157fe844ea54dfdcd1d` |
| `raw/laya-micro/smoke.stdout.txt` | `/termux-home/sel-candidates/laya-micro/smoke.stdout.txt` | 262 | `5bfecba3ceab715344ea0b770a19d4484ce4c28af0515ee0a7db9d9389fddd97` |
| `raw/laya-micro/smoke_micro.py` | `/termux-home/sel-candidates/laya-micro/smoke_micro.py` | 1557 | `3ad0ed60817d9fef0d8cde7ecc0decab3b8fc4f37c80510ceceac1eff1a4557b` |
| `raw/laya-micro/venv_freeze.txt` | `/termux-home/sel-candidates/laya-micro/venv_freeze.txt` | 866 | `e222a117e64e07200cd872aaf43c81348abab16cf944f6895a8b255502282d17` |
| `raw/laya0320/download_stderr.txt` | `/termux-home/sel-candidates/laya0320/download_stderr.txt` | 591 | `c85aed50529d23eecb95d5ec3467cd1fda1597f8d287135791c0981dfa0a0845` |
| `raw/laya0320/download_stdout.txt` | `/termux-home/sel-candidates/laya0320/download_stdout.txt` | 142 | `63dbb1eda16ca855876fb6b44c3524f32bd6d671240f4be4986e3fd814a9d251` |
| `raw/laya0320/pins.txt` | `/termux-home/sel-candidates/laya0320/pins.txt` | 744 | `a6d50b8c7439cf6262eb2b7f43e8302e3570721e8256b32025aec5d7dfdf7718` |
| `raw/laya0320/pip_stderr.txt` | `/termux-home/sel-candidates/laya0320/pip_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `raw/laya0320/pip_stdout.txt` | `/termux-home/sel-candidates/laya0320/pip_stdout.txt` | 10323 | `922c76817bf2c467afd132c642aebf000d8bf4e709969f5be16516519a0fb6be` |
| `raw/laya0320/sha256.txt` | `/termux-home/sel-candidates/laya0320/sha256.txt` | 472 | `223dbb114a0fd4d6214ac029ce5f5d50a6c42394f2f76ff3a4c123fddf2b9c88` |
| `raw/laya0320/smoke.block.json` | `/termux-home/sel-candidates/laya0320/smoke.block.json` | 340 | `7832c4e4fb71a524800be63bbee2c33fdebd8cf3b2867a81a1f95b1c5bc255cc` |
| `raw/laya0320/smoke.stderr.txt` | `/termux-home/sel-candidates/laya0320/smoke.stderr.txt` | 401 | `a4a387646f6b1ccb55c2d596394cac1e2e53cf9be5cdbc57e326cc7a86b220b3` |
| `raw/laya0320/smoke.stdout.txt` | `/termux-home/sel-candidates/laya0320/smoke.stdout.txt` | 240 | `44367d7dd58ef6425096d836c38409691c4b0b8a5aec46581816059c26b83b31` |
| `raw/laya0320/smoke_laya.py` | `/termux-home/sel-candidates/laya0320/smoke_laya.py` | 1207 | `934ebef297e70fd53805223e4438a45de4f75a3582597e9cd6e1e424fdb0ab16` |
| `raw/run_smoke.py` | `/termux-home/sel-candidates/run_smoke.py` | 1671 | `d5ff14a386270159889693e141b0f97267445e5690f0a83127f6ce7047eb3fae` |
| `raw/s1o/server.log` | `/termux-home/sel-candidates/s1o/server.log` | 2168 | `3c8e84b7f91741088e3a702d7577619741453f31cf2e5bb521fae3c664bcc3f9` |
| `raw/s1o/smoke.block.json` | `/termux-home/sel-candidates/s1o/smoke.block.json` | 298 | `d32c174a10114493da4e9f52e58ad7dc61466e23cd4bb3e2e740a027aff5bf47` |
| `raw/s1o/smoke.stderr.txt` | `/termux-home/sel-candidates/s1o/smoke.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `raw/s1o/smoke.stdout.txt` | `/termux-home/sel-candidates/s1o/smoke.stdout.txt` | 412 | `e14e8a6667ce50b0b2798a40c3cb959239049edb3bfcea7299ecc99f3962d3b2` |
| `raw/s1o/smoke_s1o.py` | `/termux-home/sel-candidates/s1o/smoke_s1o.py` | 4332 | `e8fcd234bcfdb4b1b68d7846ff6957ed8ac636ee51b5c0b25ae0bc8c7f18578d` |
| `raw/toy.json` | `/termux-home/sel-candidates/toy.json` | 586 | `ac533a0ce6befdfaeb22c777d551b16cc7cea33199a03da116160d48ac653314` |
| `reports/CODER_REPORT_candidates_phase0.md` | `/termux-home/archive-staging/reports/CODER_REPORT_candidates_phase0.md` | 9126 | `1633f8b554b2146bfbb6df1abe1aa55b895d0904061375b97077ed0f24ed56f1` |

## Written for this archive or already in place in the repository (3 files)

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 2868 | `5554b4f43285b27dc54b067d1bd5352e89a180e4275d37e9159ab08948fcef9e` |
| `RUN_INDEX.md` | 652 | `d8cabe8b2f23ac7b1c8625b076cc2c92c88c6a8779e539c375507060f5edc797` |
| `THIRD_PARTY_SHA256.txt` | 4864 | `10cd68f861153c4591368b3d811cd73c8a520bf2e58b1eae04a8b397c8402864` |

## sha256sum format

```sha256sums
5554b4f43285b27dc54b067d1bd5352e89a180e4275d37e9159ab08948fcef9e  README.md
d8cabe8b2f23ac7b1c8625b076cc2c92c88c6a8779e539c375507060f5edc797  RUN_INDEX.md
10cd68f861153c4591368b3d811cd73c8a520bf2e58b1eae04a8b397c8402864  THIRD_PARTY_SHA256.txt
e766bd5369e4b633adc0de49d28d201f818d229a1dbd71edec0b5714b09872d3  raw/base_pins.txt
bfe4bcb63977897fa5bc03844a300ea326d85a80278955ef5fb2d6ea6c67f27f  raw/laya-micro/build1_prune.block.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  raw/laya-micro/build1_prune.stderr.txt
8a8faba9e9079a0170392b11f4a3701fd43911bb3cea9103a7c954ec3d59cf58  raw/laya-micro/build1_prune.stdout.txt
4fe891e769bafd2411a2ce559266a74236b09a8fd8666f9a7abb4c6b0d2482e9  raw/laya-micro/build1_prune_8192.block.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  raw/laya-micro/build1_prune_8192.stderr.txt
838056791b4b1c62036c4bd03f2bfd43037f1bad97db0c1e205ec1144f19359a  raw/laya-micro/build1_prune_8192.stdout.txt
6bc62362a36645497ac385ec17ed29cf8939d36c40ad7d118481a06306a5021f  raw/laya-micro/build2_export.block.json
1367762b98a6ecbcc3bcf5c451c487215d99fb5340e246b61088573d2ce89a27  raw/laya-micro/build2_export.stderr.txt
0c7e1e71b40b9af290f9df04d04894e68ab6ed6aa952031a9d3b971f55df2388  raw/laya-micro/build2_export.stdout.txt
41000ba514e42a66f22211cb68f376a3bc8f6d2242ae08f4ec47502fb4e950bf  raw/laya-micro/build2_export_8192.block.json
b63a9b4703b40d7c2cd3d1279bd2c24cbe8db4e85b793061d978b0158ff837cd  raw/laya-micro/build2_export_8192.stderr.txt
dd946839a0ad34f10a733c5a9ed8c53060b633d984edfd9105a5fdf7622379f2  raw/laya-micro/build2_export_8192.stdout.txt
8a81c0f033d430f48beb8cf3dfcb65b0624abfc3ab05d17e32c58ff0d1825fa4  raw/laya-micro/build3_quantize.block.json
735a1cf497b00c89e9a4dfbd7ba15938bc3c5dac4a23c82bb06d016e1a299d38  raw/laya-micro/build3_quantize.stderr.txt
d6e6a30a60f3479a6dcbe1ed7a104612e513ede1dae5cceb4d173f64d0bf6c06  raw/laya-micro/build3_quantize.stdout.txt
fe1a50bb9225686c9d409733ec181dd552c9e7d549ab6c9bd90b021c02ea7cc7  raw/laya-micro/build3_quantize_8192.block.json
f694d01d4764e47c506cdf0387611962c7defba9cfb206bffce0f2ebaa3f3688  raw/laya-micro/build3_quantize_8192.stderr.txt
254a023f3116320fb1f386d32840e6a7a968a82520bd93ef6a95258785fbf409  raw/laya-micro/build3_quantize_8192.stdout.txt
a2563885daf0301b5434b167534aa65b22724f688a07c25d41fd3ca17ba4855d  raw/laya-micro/build_8192_sha256.txt
5636bfb3464f406267d89d77f6c468c2d515ad52a3b02ac02fe65d65cf742f7e  raw/laya-micro/build_sha256.txt
2926fd98985888f8a7b59fe03182d0fabe1b6ab171ba2d87af9d21b891a041e9  raw/laya-micro/download_stderr.txt
91034285c2c3674c84bf795e89b3f59515242f379a0e18ced2a368371a10e5ea  raw/laya-micro/download_stdout.txt
7dcede60cb05aa98a542dbfe37849c849b9848013eef186f9733badec82c0b7d  raw/laya-micro/laya-micro-8192.meta.json
7dcede60cb05aa98a542dbfe37849c849b9848013eef186f9733badec82c0b7d  raw/laya-micro/laya-micro.meta.json
a6d50b8c7439cf6262eb2b7f43e8302e3570721e8256b32025aec5d7dfdf7718  raw/laya-micro/pins.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  raw/laya-micro/pip_stderr.txt
69af850f97760e0d51f37e712ac31164e5d6cb80a9e176c5971d0158b003909a  raw/laya-micro/pip_stdout.txt
8cdd445165a2452a579df115e557d0091238f8bcff18637883902093c921709c  raw/laya-micro/sha256.txt
b5ad60c2d6c663bc3cba895dff7f3d83362227cbd4eed0a6d8509daeb85e3927  raw/laya-micro/shape_timing.py
507d48b527c08a37a444ee424d6c31ebf30abde895eb05997278b8908a538d69  raw/laya-micro/shape_timing.stderr.txt
36587bc8e34bf5958e315326e988320abab7ce6de93f9e650a3f4919e32cc57d  raw/laya-micro/shape_timing.stdout.txt
db79143b475616c0e020f3a539cd0bbc1dd8b7bea4f7c7dc3a152e190749dd75  raw/laya-micro/smoke.block.json
e0932157a6161dcd60392569a0085394c452ed1830058157fe844ea54dfdcd1d  raw/laya-micro/smoke.stderr.txt
5bfecba3ceab715344ea0b770a19d4484ce4c28af0515ee0a7db9d9389fddd97  raw/laya-micro/smoke.stdout.txt
3ad0ed60817d9fef0d8cde7ecc0decab3b8fc4f37c80510ceceac1eff1a4557b  raw/laya-micro/smoke_micro.py
e222a117e64e07200cd872aaf43c81348abab16cf944f6895a8b255502282d17  raw/laya-micro/venv_freeze.txt
c85aed50529d23eecb95d5ec3467cd1fda1597f8d287135791c0981dfa0a0845  raw/laya0320/download_stderr.txt
63dbb1eda16ca855876fb6b44c3524f32bd6d671240f4be4986e3fd814a9d251  raw/laya0320/download_stdout.txt
a6d50b8c7439cf6262eb2b7f43e8302e3570721e8256b32025aec5d7dfdf7718  raw/laya0320/pins.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  raw/laya0320/pip_stderr.txt
922c76817bf2c467afd132c642aebf000d8bf4e709969f5be16516519a0fb6be  raw/laya0320/pip_stdout.txt
223dbb114a0fd4d6214ac029ce5f5d50a6c42394f2f76ff3a4c123fddf2b9c88  raw/laya0320/sha256.txt
7832c4e4fb71a524800be63bbee2c33fdebd8cf3b2867a81a1f95b1c5bc255cc  raw/laya0320/smoke.block.json
a4a387646f6b1ccb55c2d596394cac1e2e53cf9be5cdbc57e326cc7a86b220b3  raw/laya0320/smoke.stderr.txt
44367d7dd58ef6425096d836c38409691c4b0b8a5aec46581816059c26b83b31  raw/laya0320/smoke.stdout.txt
934ebef297e70fd53805223e4438a45de4f75a3582597e9cd6e1e424fdb0ab16  raw/laya0320/smoke_laya.py
d5ff14a386270159889693e141b0f97267445e5690f0a83127f6ce7047eb3fae  raw/run_smoke.py
3c8e84b7f91741088e3a702d7577619741453f31cf2e5bb521fae3c664bcc3f9  raw/s1o/server.log
d32c174a10114493da4e9f52e58ad7dc61466e23cd4bb3e2e740a027aff5bf47  raw/s1o/smoke.block.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  raw/s1o/smoke.stderr.txt
e14e8a6667ce50b0b2798a40c3cb959239049edb3bfcea7299ecc99f3962d3b2  raw/s1o/smoke.stdout.txt
e8fcd234bcfdb4b1b68d7846ff6957ed8ac636ee51b5c0b25ae0bc8c7f18578d  raw/s1o/smoke_s1o.py
ac533a0ce6befdfaeb22c777d551b16cc7cea33199a03da116160d48ac653314  raw/toy.json
1633f8b554b2146bfbb6df1abe1aa55b895d0904061375b97077ed0f24ed56f1  reports/CODER_REPORT_candidates_phase0.md
```
