# Artifacts

Every archived file below with its SHA-256. Copied files were compared byte for byte (`cmp`) with the phone original at archive time, and the phone and archive SHA-256 values match. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Copied from the phone (371 files)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `cases/ladder_cases_v1.jsonl` | `/termux-home/ladder/cases/ladder_cases_v1.jsonl` | 25538 | `41eeafd28e2ce499437f75091e2069bdd89d4e9e0d789f223ff4dd1c9bebb473` |
| `ladder.py` | `/termux-home/ladder/ladder.py` | 44447 | `c1a4f65df8dc78e0400a1a27d98e6e1766635a5d1185a31a8b8940e3ae39a0f7` |
| `ladder_worker.py` | `/termux-home/ladder/ladder_worker.py` | 9000 | `f795b354d6b615a681b25066d79b91e58ad8c1f68ae16b516a364bff8473b07e` |
| `logs/oneshot_console_20260926T045141Z.log` | `/termux-home/ladder/oneshot_console_20260926T045141Z.log` | 12455 | `00413cc2960ef50225ddd095c1144a7ca8f7326750f8ec29d2276f122d3e45d9` |
| `logs/oneshot_console_20260926T065115Z.log` | `/termux-home/ladder/oneshot_console_20260926T065115Z.log` | 58739 | `cdd485ea0ed5818a378c3d49b87b0187f557da38930bba97add088ad40207d90` |
| `logs/oneshot_console_20260926T185449Z.log` | `/termux-home/ladder/oneshot_console_20260926T185449Z.log` | 7519 | `31a632e94a00ad2eed33fd03e83a04f5eb108d18366ba05ac3a10a8ce99bd6cb` |
| `logs/s1o_speed_20260926T044457Z.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T044457Z.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `logs/s1o_speed_20260926T044457Z.stdout.txt` | `/termux-home/ladder/s1o_speed_20260926T044457Z.stdout.txt` | 462 | `1ed8db43757dfc4be901a0db0af31ae684a214046080bae3c6707c50eb97a184` |
| `logs/s1o_speed_20260926T045648Z.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z.stderr.txt` | 123 | `09de8ccfd04dbf3bc2cbc871ad2d14c998349c50a5890d476a195f58ff38d02f` |
| `logs/s1o_speed_20260926T045648Z.stdout.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z.stdout.txt` | 70307 | `c7c20061cbed3e9db1f55c8344ef40287b1fec57b2568738bd316fffdd0ac53c` |
| `logs/smoke_run.stderr.txt` | `/termux-home/ladder/smoke_run.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `logs/smoke_run.stdout.txt` | `/termux-home/ladder/smoke_run.stdout.txt` | 16612 | `8c5fa4aa212f599760d496625ce5c8a0b5ea181d5558fc40f7695d12dcfed963` |
| `logs/thermal.log` | `/termux-home/archive-staging/thermal_ladder_snapshot_20260926T220753Z.log` | 737896 | `7e94472e87bb73cdac94c321d1edef880e252785c4f07e848d0596c4eb246fce` |
| `logs/toy_run.pty.txt` | `/termux-home/ladder/toy_run.pty.txt` | 7743 | `33e4c366929d655b01aa11ab0c713cd4f7faf8c1f4953497b9a0a23db7e81ea4` |
| `measure.py` | `/termux-home/robot/benchmark/strategic_selector/v3/measure.py` | 4115 | `7040be22ec0bf2ef8dadb5b46a52e10402ab84d1ffd38eab0020430519705242` |
| `oneshot.sh` | `/sdcard/Download/oneshot.sh` | 2949 | `f3d531fcbc306c8ef5f6e4d6504c91edcaeddc6e66313673dc839dd57e1b7d31` |
| `oneshot_executed_conversation_run.sh` | `/termux-home/ladder/oneshot.sh` | 2627 | `db1f315681892f0c1f89de92b86d7495a47cc171a3c03a170fc51d2f8dbdc8e8` |
| `oneshot_executed_s1o_resume.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/conv/oneshot.sh.orig` | 2533 | `0edcae559ea922a7be9ceefc9b4e20e5191eb8526494929046eb44103232a456` |
| `reports/CODER_REPORT_ladder_build.md` | `/termux-home/archive-staging/reports/CODER_REPORT_ladder_build.md` | 27412 | `cdcdcfcc74c15b9a4c07b0c20b6a127e1315008e0933c7172d0e65834804e0e7` |
| `reports/CODER_REPORT_ladder_s1o_speed_prep.md` | `/termux-home/archive-staging/reports/CODER_REPORT_ladder_s1o_speed_prep.md` | 21052 | `55c1e5ec4985c48e44f8320c063addb22069e7b4d0751993d1eee6642910322c` |
| `reports/s1o_speed_prep_report.md` | `/sdcard/Download/s1o_speed_prep_report.md` | 11804 | `954b9844198bbdc35d1711aea2c2ef44f27d284688f3314d7e5365f3dd89e2d3` |
| `reviews/ladder-review-20260925T204237Z/FROZEN.sha256` | `/termux-home/ladder-review-20260925T204237Z/FROZEN.sha256` | 322 | `d4169b20e577e22515bb51d92d9db412b4815fc22d712722195e5495b09eab0d` |
| `reviews/ladder-review-20260925T204237Z/REVIEW_REQUEST.txt` | `/termux-home/ladder-review-20260925T204237Z/REVIEW_REQUEST.txt` | 57779 | `4ad29ec9ef0cf56aca2cc35a5dcd37a35ac28e359c63573547913e08de8072aa` |
| `reviews/ladder-review-20260925T204237Z/candidate/ladder.py` | `/termux-home/ladder-review-20260925T204237Z/candidate/ladder.py` | 28320 | `aef3639a6460def336de74ea0fe37c19b472ba638b8f79654451d7735e2ae6cd` |
| `reviews/ladder-review-20260925T204237Z/candidate/ladder_worker.py` | `/termux-home/ladder-review-20260925T204237Z/candidate/ladder_worker.py` | 7960 | `1174a0f3632b43884bd39b05e7a39f405ae9e1f118d9d4c5a23c587711c92e6d` |
| `reviews/ladder-review-20260925T204237Z/candidate/test_ladder.py` | `/termux-home/ladder-review-20260925T204237Z/candidate/test_ladder.py` | 2469 | `ab02053e120887def6f275a22c07e0e628e03bbaf1bee547bd2702d37e986d32` |
| `reviews/ladder-review-20260925T204237Z/candidate/toy_cases.jsonl` | `/termux-home/ladder-review-20260925T204237Z/candidate/toy_cases.jsonl` | 1736 | `2ebe1f9fd4eac5d9329d6fe8af2b2bb7dbccf170239c687191ed6ae9e5ebb7bc` |
| `reviews/ladder-review-20260925T204237Z/check_test_ladder.txt` | `/termux-home/ladder-review-20260925T204237Z/check_test_ladder.txt` | 3 | `dc51b8c96c2d745df3bd5590d990230a482fd247123599548e0632fdbf97fc22` |
| `reviews/ladder-review-20260925T204237Z/context/adapters.py` | `/termux-home/ladder-review-20260925T204237Z/context/adapters.py` | 17240 | `23e90a3d51daea89f3689aaadb3f4a56a43f23f26dd76785257f15c653e099bc` |
| `reviews/ladder-review-20260925T204237Z/context/jevlike.py` | `/termux-home/ladder-review-20260925T204237Z/context/jevlike.py` | 22077 | `fcce2b24dabd475088765296792da408cf23c24b0f5bb4daf3720893b757d6d4` |
| `reviews/ladder-review-20260925T204237Z/context/jevlike_worker.py` | `/termux-home/ladder-review-20260925T204237Z/context/jevlike_worker.py` | 4908 | `06eecc622378efd91e6cb0839a39f84fefa8eef797a8512845f6117df61661f9` |
| `reviews/ladder-review-20260925T204237Z/context/laya_micro_runtime.py` | `/termux-home/ladder-review-20260925T204237Z/context/laya_micro_runtime.py` | 9120 | `9a1f6b416fe65623a43645a5adb68d769af05e7ee105e5de4d3055babab0c029` |
| `reviews/ladder-review-20260925T204237Z/context/measure.py` | `/termux-home/ladder-review-20260925T204237Z/context/measure.py` | 4115 | `7040be22ec0bf2ef8dadb5b46a52e10402ab84d1ffd38eab0020430519705242` |
| `reviews/ladder-review-20260925T204237Z/review.stderr.txt` | `/termux-home/ladder-review-20260925T204237Z/review.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reviews/ladder-review-20260925T204237Z/review.stdout.json` | `/termux-home/ladder-review-20260925T204237Z/review.stdout.json` | 7094 | `c8ba2851c3328d0e810185b98f98184db2f4d7e7a58ce6cbcc1a2bcdc68e329f` |
| `reviews/ladder-review-20260925T204237Z/toy_run/decisions.jsonl` | `/termux-home/ladder-review-20260925T204237Z/toy_run/decisions.jsonl` | 19838 | `4b75d9fee3ad438d94632adc54aaaa98d57f79e17804d9411f8528c822c39cad` |
| `reviews/ladder-review-20260925T204237Z/toy_run/report.txt` | `/termux-home/ladder-review-20260925T204237Z/toy_run/report.txt` | 6253 | `6b3544e2b087df7a5460b97c1c46dad59f9924297c6c38773b9f700e72a9d9df` |
| `reviews/ladder-review-20260925T204237Z/toy_run/results.json` | `/termux-home/ladder-review-20260925T204237Z/toy_run/results.json` | 13492 | `ba25a9c1f9e5ce0ef4424483d0070054ffe9228cd98fb5e12f52eee2b08a64a7` |
| `reviews/ladder-review-20260925T204237Z/toy_run/toy_run.stderr.txt` | `/termux-home/ladder-review-20260925T204237Z/toy_run/toy_run.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reviews/ladder-review-20260925T204237Z/toy_run/toy_run.stdout.txt` | `/termux-home/ladder-review-20260925T204237Z/toy_run/toy_run.stdout.txt` | 6669 | `ec1bdee7dcf981b50e34657dce982720baebd7cfea7409d035d3097095eda5ab` |
| `reviews/ladder-review-20260925T211413Z/FROZEN.sha256` | `/termux-home/ladder-review-20260925T211413Z/FROZEN.sha256` | 322 | `9f0f7a7c62e37e72f5eb7d00e54b3c6892ac04872034a6e14d60fe9d621aa21b` |
| `reviews/ladder-review-20260925T211413Z/REVIEW_REQUEST.txt` | `/termux-home/ladder-review-20260925T211413Z/REVIEW_REQUEST.txt` | 91296 | `87cedc59d323af0f5592ab7b43d2bb0a2c65cea5396052601d7bb8a9ea65bc1d` |
| `reviews/ladder-review-20260925T211413Z/candidate.diff` | `/termux-home/ladder-review-20260925T211413Z/candidate.diff` | 25601 | `e99c5b073bc4f38a0c97861ddc37b88966e657a57e4c6de544f14531d5bcf4e8` |
| `reviews/ladder-review-20260925T211413Z/candidate/ladder.py` | `/termux-home/ladder-review-20260925T211413Z/candidate/ladder.py` | 32023 | `00fbe2fab81021cb884f54550042b21be9f3b9e61943efea6d63f79139bb5f8b` |
| `reviews/ladder-review-20260925T211413Z/candidate/ladder_worker.py` | `/termux-home/ladder-review-20260925T211413Z/candidate/ladder_worker.py` | 9000 | `f795b354d6b615a681b25066d79b91e58ad8c1f68ae16b516a364bff8473b07e` |
| `reviews/ladder-review-20260925T211413Z/candidate/test_ladder.py` | `/termux-home/ladder-review-20260925T211413Z/candidate/test_ladder.py` | 4785 | `daf7402ab0365725e66d3dd7da13f2c6925ebca3b422f806e8e7026c78101e8a` |
| `reviews/ladder-review-20260925T211413Z/candidate/toy_cases.jsonl` | `/termux-home/ladder-review-20260925T211413Z/candidate/toy_cases.jsonl` | 1736 | `2ebe1f9fd4eac5d9329d6fe8af2b2bb7dbccf170239c687191ed6ae9e5ebb7bc` |
| `reviews/ladder-review-20260925T211413Z/check_test_ladder.txt` | `/termux-home/ladder-review-20260925T211413Z/check_test_ladder.txt` | 3 | `dc51b8c96c2d745df3bd5590d990230a482fd247123599548e0632fdbf97fc22` |
| `reviews/ladder-review-20260925T211413Z/context/adapters.py` | `/termux-home/ladder-review-20260925T211413Z/context/adapters.py` | 17240 | `23e90a3d51daea89f3689aaadb3f4a56a43f23f26dd76785257f15c653e099bc` |
| `reviews/ladder-review-20260925T211413Z/context/jevlike.py` | `/termux-home/ladder-review-20260925T211413Z/context/jevlike.py` | 22077 | `fcce2b24dabd475088765296792da408cf23c24b0f5bb4daf3720893b757d6d4` |
| `reviews/ladder-review-20260925T211413Z/context/jevlike_worker.py` | `/termux-home/ladder-review-20260925T211413Z/context/jevlike_worker.py` | 4908 | `06eecc622378efd91e6cb0839a39f84fefa8eef797a8512845f6117df61661f9` |
| `reviews/ladder-review-20260925T211413Z/context/laya_micro_runtime.py` | `/termux-home/ladder-review-20260925T211413Z/context/laya_micro_runtime.py` | 9120 | `9a1f6b416fe65623a43645a5adb68d769af05e7ee105e5de4d3055babab0c029` |
| `reviews/ladder-review-20260925T211413Z/context/measure.py` | `/termux-home/ladder-review-20260925T211413Z/context/measure.py` | 4115 | `7040be22ec0bf2ef8dadb5b46a52e10402ab84d1ffd38eab0020430519705242` |
| `reviews/ladder-review-20260925T211413Z/previous_review/FROZEN.sha256` | `/termux-home/ladder-review-20260925T211413Z/previous_review/FROZEN.sha256` | 322 | `d4169b20e577e22515bb51d92d9db412b4815fc22d712722195e5495b09eab0d` |
| `reviews/ladder-review-20260925T211413Z/previous_review/candidate/ladder.py` | `/termux-home/ladder-review-20260925T211413Z/previous_review/candidate/ladder.py` | 28320 | `aef3639a6460def336de74ea0fe37c19b472ba638b8f79654451d7735e2ae6cd` |
| `reviews/ladder-review-20260925T211413Z/previous_review/candidate/ladder_worker.py` | `/termux-home/ladder-review-20260925T211413Z/previous_review/candidate/ladder_worker.py` | 7960 | `1174a0f3632b43884bd39b05e7a39f405ae9e1f118d9d4c5a23c587711c92e6d` |
| `reviews/ladder-review-20260925T211413Z/previous_review/candidate/test_ladder.py` | `/termux-home/ladder-review-20260925T211413Z/previous_review/candidate/test_ladder.py` | 2469 | `ab02053e120887def6f275a22c07e0e628e03bbaf1bee547bd2702d37e986d32` |
| `reviews/ladder-review-20260925T211413Z/previous_review/candidate/toy_cases.jsonl` | `/termux-home/ladder-review-20260925T211413Z/previous_review/candidate/toy_cases.jsonl` | 1736 | `2ebe1f9fd4eac5d9329d6fe8af2b2bb7dbccf170239c687191ed6ae9e5ebb7bc` |
| `reviews/ladder-review-20260925T211413Z/previous_review/review.stdout.json` | `/termux-home/ladder-review-20260925T211413Z/previous_review/review.stdout.json` | 7094 | `c8ba2851c3328d0e810185b98f98184db2f4d7e7a58ce6cbcc1a2bcdc68e329f` |
| `reviews/ladder-review-20260925T211413Z/review.stderr.txt` | `/termux-home/ladder-review-20260925T211413Z/review.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `reviews/ladder-review-20260925T211413Z/review.stdout.json` | `/termux-home/ladder-review-20260925T211413Z/review.stdout.json` | 5024 | `4de3aa8bf3ae21ab83af5919d5c9ffe96e74e5e56ef41c7b0690fb8ecf5f21b5` |
| `reviews/ladder-review-20260925T211413Z/toy_run/decisions.jsonl` | `/termux-home/ladder-review-20260925T211413Z/toy_run/decisions.jsonl` | 19832 | `19c13ce213dc5d9f24963368292964674b485a820c189dd470ebd2b574dc07f8` |
| `reviews/ladder-review-20260925T211413Z/toy_run/report.txt` | `/termux-home/ladder-review-20260925T211413Z/toy_run/report.txt` | 6620 | `ab9c8f5c344c5b6de8e78d56d5d2a820784d20c789f5a9f10633d97f19d45d16` |
| `reviews/ladder-review-20260925T211413Z/toy_run/results.json` | `/termux-home/ladder-review-20260925T211413Z/toy_run/results.json` | 14916 | `b0576b5b1de17924ffbad96a0cadbbc264a9bf19c58c3edcb815d67600cc17cb` |
| `reviews/ladder-review-20260925T211413Z/toy_run/toy_run.pty.txt` | `/termux-home/ladder-review-20260925T211413Z/toy_run/toy_run.pty.txt` | 7743 | `33e4c366929d655b01aa11ab0c713cd4f7faf8c1f4953497b9a0a23db7e81ea4` |
| `run_conversation.sh` | `/termux-home/ladder/run_conversation.sh` | 1411 | `a335b7633c9c0bce6d86744258e101c4b076982e618f6c434e34cc82ef6faee8` |
| `run_s1o_speed.sh` | `/termux-home/ladder/run_s1o_speed.sh` | 1852 | `308a19c5c38fae6c2cbabe71fb0af17d1eda94bb841f3746b06ec06fc1df8efa` |
| `runs/old_runs/toy_run.stderr.txt` | `/termux-home/ladder/old_runs/toy_run.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/old_runs/toy_run.stdout.txt` | `/termux-home/ladder/old_runs/toy_run.stdout.txt` | 6669 | `ec1bdee7dcf981b50e34657dce982720baebd7cfea7409d035d3097095eda5ab` |
| `runs/old_runs/toy_run/decisions.jsonl` | `/termux-home/ladder/old_runs/toy_run/decisions.jsonl` | 19838 | `4b75d9fee3ad438d94632adc54aaaa98d57f79e17804d9411f8528c822c39cad` |
| `runs/old_runs/toy_run/logs/laya_en.main.stderr.txt` | `/termux-home/ladder/old_runs/toy_run/logs/laya_en.main.stderr.txt` | 313 | `7b7dd217a7b63ea9637b220c9f10a99fbc6c7f2b7464921d76fca05b32213293` |
| `runs/old_runs/toy_run/logs/laya_en.t2.stderr.txt` | `/termux-home/ladder/old_runs/toy_run/logs/laya_en.t2.stderr.txt` | 313 | `6cdb0e96d30a5ef5f6655e1a037c15fd714fe862c21f5ffeeb62ed096e514a09` |
| `runs/old_runs/toy_run/logs/laya_en.t3.stderr.txt` | `/termux-home/ladder/old_runs/toy_run/logs/laya_en.t3.stderr.txt` | 313 | `ab046b0553ca029309eaeb16194bba2a2a5ba9aeb4c4e4ff420f2971769b7ebe` |
| `runs/old_runs/toy_run/logs/laya_en.t4.stderr.txt` | `/termux-home/ladder/old_runs/toy_run/logs/laya_en.t4.stderr.txt` | 313 | `475be39578519967c3f8d7cb17ec267517a6b7c2b06a0216b9f1ddf0cde29cb2` |
| `runs/old_runs/toy_run/logs/von11.main.stderr.txt` | `/termux-home/ladder/old_runs/toy_run/logs/von11.main.stderr.txt` | 1114 | `d6af1f3179b319dd91b4fd0d06a5a75ec40aacff535a83647c792be5d85104c8` |
| `runs/old_runs/toy_run/logs/von11.t2.stderr.txt` | `/termux-home/ladder/old_runs/toy_run/logs/von11.t2.stderr.txt` | 883 | `3d77d35894c18287f42944a20e59859b3dd270ec21e2772091d7c980571efdba` |
| `runs/old_runs/toy_run/logs/von11.t3.stderr.txt` | `/termux-home/ladder/old_runs/toy_run/logs/von11.t3.stderr.txt` | 881 | `88c03054ae8f2e959c648c6b3dcf1ba7644afe18e89bc6ee5b71b3b200f404bf` |
| `runs/old_runs/toy_run/logs/von11.t4.stderr.txt` | `/termux-home/ladder/old_runs/toy_run/logs/von11.t4.stderr.txt` | 883 | `e0b69733024bba12f36c8f5a7ee016f80a53517fa957b135d474cad7dd7562f2` |
| `runs/old_runs/toy_run/report.txt` | `/termux-home/ladder/old_runs/toy_run/report.txt` | 6253 | `6b3544e2b087df7a5460b97c1c46dad59f9924297c6c38773b9f700e72a9d9df` |
| `runs/old_runs/toy_run/results.json` | `/termux-home/ladder/old_runs/toy_run/results.json` | 13492 | `ba25a9c1f9e5ce0ef4424483d0070054ffe9228cd98fb5e12f52eee2b08a64a7` |
| `runs/old_runs/toy_run_1.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_1.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/old_runs/toy_run_1.stdout.txt` | `/termux-home/ladder/old_runs/toy_run_1.stdout.txt` | 6449 | `98e8bd49e3e2748c52627c1e028051e59c163fc85178baedc3a82a680f81e574` |
| `runs/old_runs/toy_run_1/decisions.jsonl` | `/termux-home/ladder/old_runs/toy_run_1/decisions.jsonl` | 19839 | `61f9fb2310ca0cf2fb7d5e817db9e0b4fc29b8b8ecc7948f57bafbf040fa9968` |
| `runs/old_runs/toy_run_1/logs/laya_en.main.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_1/logs/laya_en.main.stderr.txt` | 313 | `8c7b82b6d0347a5ef45bf9084101428dcba33b9eb95c53ba2f9f6002818be92f` |
| `runs/old_runs/toy_run_1/logs/laya_en.t2.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_1/logs/laya_en.t2.stderr.txt` | 313 | `478825aad4a2933adedcca3dbce0a6a2f5992f9eed36f6a88e552f78d795ae47` |
| `runs/old_runs/toy_run_1/logs/laya_en.t3.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_1/logs/laya_en.t3.stderr.txt` | 313 | `677dde8f45f0298927ee56293bb2c95f545d0038454bda9da767089d0edf9f0e` |
| `runs/old_runs/toy_run_1/logs/laya_en.t4.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_1/logs/laya_en.t4.stderr.txt` | 313 | `0b6e53274cde8653d453722a57f3e5eda88bf167cac7f4bc7a065af1f5a943ff` |
| `runs/old_runs/toy_run_1/logs/von11.main.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_1/logs/von11.main.stderr.txt` | 1181 | `ae3b76705a98a6d93714f13ef78fd907f5769f3469febdb0e1775dfb40802476` |
| `runs/old_runs/toy_run_1/logs/von11.t2.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_1/logs/von11.t2.stderr.txt` | 883 | `a13ff651299efb5fa1455b71ce6b8ed6e8d750e39155ea45bdb0eb101cbc7f45` |
| `runs/old_runs/toy_run_1/logs/von11.t3.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_1/logs/von11.t3.stderr.txt` | 881 | `e98959054266fda33be4b943364e50a37901c3913f7f9cf3112a83afea8bbced` |
| `runs/old_runs/toy_run_1/logs/von11.t4.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_1/logs/von11.t4.stderr.txt` | 883 | `e2a352b65a9880a2b989dd4440aff82a309d3f0854113d76242e997c072b13e3` |
| `runs/old_runs/toy_run_1/report.txt` | `/termux-home/ladder/old_runs/toy_run_1/report.txt` | 6033 | `6c30e5696c5e3c984fdb1956acb2fdb09dff8a6feb4ecd38c1a9804b8c65d537` |
| `runs/old_runs/toy_run_1/results.json` | `/termux-home/ladder/old_runs/toy_run_1/results.json` | 13093 | `5c0d4a0b23e9b56cd1bbc1be7184e0019a4e2c7ad31c76ec25894079bcef4c9c` |
| `runs/old_runs/toy_run_2.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_2.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/old_runs/toy_run_2.stdout.txt` | `/termux-home/ladder/old_runs/toy_run_2.stdout.txt` | 6579 | `9fe86d1b9faf07f88c55dfddb8c456d00f90b7d4cb46449feb8df0387e0d7b99` |
| `runs/old_runs/toy_run_2/decisions.jsonl` | `/termux-home/ladder/old_runs/toy_run_2/decisions.jsonl` | 19834 | `98c0f6219367d56f484261ba9ceb572115ec06d172ed7d8d844355eac2860b4e` |
| `runs/old_runs/toy_run_2/logs/laya_en.main.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_2/logs/laya_en.main.stderr.txt` | 313 | `6f6f9da57b9f5bf47b34284d18996b05f1b6826f1ea6affbdca7e29c0997b98e` |
| `runs/old_runs/toy_run_2/logs/laya_en.t2.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_2/logs/laya_en.t2.stderr.txt` | 313 | `301f3166de69153faddc11ea133de0b08ae84885d8c3d074f6c743ff1653531f` |
| `runs/old_runs/toy_run_2/logs/laya_en.t3.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_2/logs/laya_en.t3.stderr.txt` | 313 | `03ba78710b8767f7100faf60ee61768fdbbf78d8e573eedd46080724b14c7403` |
| `runs/old_runs/toy_run_2/logs/laya_en.t4.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_2/logs/laya_en.t4.stderr.txt` | 313 | `6cc9e6e601eaf08e1494c50fabe8b0a745f5717a3c90f6023964576bb763088c` |
| `runs/old_runs/toy_run_2/logs/von11.main.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_2/logs/von11.main.stderr.txt` | 1686 | `37059b28c16ba775af28dccad0d97c43c88f75c4dc2c252c0fb7c19f13d3551a` |
| `runs/old_runs/toy_run_2/logs/von11.t2.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_2/logs/von11.t2.stderr.txt` | 881 | `7aae466d9d8b8e65fd172ce8a23ed332550d671eac563625fe64991d39595d60` |
| `runs/old_runs/toy_run_2/logs/von11.t3.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_2/logs/von11.t3.stderr.txt` | 880 | `d164c0570635579141e30c948dff180b7c3eff640bd57cdf85fc49b779e26a0c` |
| `runs/old_runs/toy_run_2/logs/von11.t4.stderr.txt` | `/termux-home/ladder/old_runs/toy_run_2/logs/von11.t4.stderr.txt` | 883 | `114430b025fd5e4535283717472161a70703fd11abc1a284fd4f05d79b57bc7a` |
| `runs/old_runs/toy_run_2/report.txt` | `/termux-home/ladder/old_runs/toy_run_2/report.txt` | 6163 | `0147b349a0f011fa42dcd741f483d3c1738cf9529bc68e97512371443aac6833` |
| `runs/old_runs/toy_run_2/results.json` | `/termux-home/ladder/old_runs/toy_run_2/results.json` | 13396 | `9c1d5dc6c81870b3a5febc2b37779afa900ae0f74953e6e59564cf1dedbe6465` |
| `runs/real_run/decisions.jsonl` | `/termux-home/ladder/real_run/decisions.jsonl` | 1440832 | `684908fb5b1233b35ca1973393ab4c8a640886a4942118ee7e0217802e313230` |
| `runs/real_run/logs/laya_en.main.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_en.main.stderr.txt` | 313 | `e4acfc9dcd8063041f81b49d3f08df2ddb35f4e8921837536076fe906236cc52` |
| `runs/real_run/logs/laya_en.t2.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_en.t2.stderr.txt` | 313 | `05fe3e419ca0e316b3e6e908ecae0713c77d2a0a0c54f427c24a32d0193f197f` |
| `runs/real_run/logs/laya_en.t3.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_en.t3.stderr.txt` | 313 | `3d4050692a5db5afe2c5f4001ba5f9ec3ecec1e97cb3a3fa87ceedfe6f541fc6` |
| `runs/real_run/logs/laya_en.t4.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_en.t4.stderr.txt` | 313 | `2f3b916ecfce3ca44d56406419bc3e4a5895fdc88d2b19b29f43200f8119c4ce` |
| `runs/real_run/logs/laya_micro.main.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_micro.main.stderr.txt` | 313 | `3599623d702d2d39ea232921b11eb9feca773980c106dc8aab42bfabd24b9599` |
| `runs/real_run/logs/laya_micro.t2.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_micro.t2.stderr.txt` | 313 | `f3ce0e23444eba1803639e5d8c3f9b02c3779a65b3772ec4622d84a13e4f7474` |
| `runs/real_run/logs/laya_micro.t3.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_micro.t3.stderr.txt` | 313 | `6318742ae7f104ad50e5d47dd5523eb2376a19c6f6640db84bca61097eb9acdc` |
| `runs/real_run/logs/laya_micro.t4.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_micro.t4.stderr.txt` | 313 | `6f244de151c118cf7388e6273deb5f9223dcf28f39507ae730b7ccdcbff921b2` |
| `runs/real_run/logs/laya_multi.main.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_multi.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/real_run/logs/laya_multi.t2.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_multi.t2.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/real_run/logs/laya_multi.t3.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_multi.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/real_run/logs/laya_multi.t4.stderr.txt` | `/termux-home/ladder/real_run/logs/laya_multi.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/real_run/logs/s1o.main.server.log` | `/termux-home/ladder/real_run/logs/s1o.main.server.log` | 98682 | `fa83f933efb70f01e1fe3ecd155d9d6485b9a2eeefa672c6411d70fd41104aa7` |
| `runs/real_run/logs/s1o.main.stderr.txt` | `/termux-home/ladder/real_run/logs/s1o.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/real_run/logs/s1o.t2.server.log` | `/termux-home/ladder/real_run/logs/s1o.t2.server.log` | 49866 | `2591e2d00422780e35ad4d65ed7cc2c7af0000f93dafc50289b17a760b64c0fb` |
| `runs/real_run/logs/s1o.t2.stderr.txt` | `/termux-home/ladder/real_run/logs/s1o.t2.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/real_run/logs/s1o.t3.server.log` | `/termux-home/ladder/real_run/logs/s1o.t3.server.log` | 49810 | `8f156523a3130cf644890c77f2ff30a8baed614de0b82b65b6071d9a66198ba6` |
| `runs/real_run/logs/s1o.t3.stderr.txt` | `/termux-home/ladder/real_run/logs/s1o.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/real_run/logs/s1o.t4.server.log` | `/termux-home/ladder/real_run/logs/s1o.t4.server.log` | 49797 | `e82751b6b064c56734acc18b42d0929425b1a6401782ca7b74b996575b91c88f` |
| `runs/real_run/logs/s1o.t4.stderr.txt` | `/termux-home/ladder/real_run/logs/s1o.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/real_run/logs/von10_nli.main.stderr.txt` | `/termux-home/ladder/real_run/logs/von10_nli.main.stderr.txt` | 467 | `46a5151afa9c00d8ad9ac24544a30a62cdcce1e1705e2c179c881a358f6e6348` |
| `runs/real_run/logs/von10_nli.t2.stderr.txt` | `/termux-home/ladder/real_run/logs/von10_nli.t2.stderr.txt` | 466 | `695a846656788afcfd05d12ccda42c8f71d15f54ab5c203527ca0bbd5ef53312` |
| `runs/real_run/logs/von10_nli.t3.stderr.txt` | `/termux-home/ladder/real_run/logs/von10_nli.t3.stderr.txt` | 467 | `4a852531fe88f7389e48f75fe9d1dcd5ca239ee7ec5fb3acc8bdb4f7b1c40554` |
| `runs/real_run/logs/von10_nli.t4.stderr.txt` | `/termux-home/ladder/real_run/logs/von10_nli.t4.stderr.txt` | 466 | `01e986a7e09d6391b7b7c2c3546ad49bcdae4306e5cf5c0fe0131e557fe6dd41` |
| `runs/real_run/logs/von11.main.stderr.txt` | `/termux-home/ladder/real_run/logs/von11.main.stderr.txt` | 881 | `59366d8113a6b6b54e26a5830b81f28d76c9565b9aff631af5bbaf6984cc096f` |
| `runs/real_run/logs/von11.t2.stderr.txt` | `/termux-home/ladder/real_run/logs/von11.t2.stderr.txt` | 1028 | `0060f686698f5120f42b9e4db34157e51aba50d4a2d12921013fce2f4b0ac49b` |
| `runs/real_run/logs/von11.t3.stderr.txt` | `/termux-home/ladder/real_run/logs/von11.t3.stderr.txt` | 881 | `16c52106fb54c6eb6ecaa45f5c46b6e1b34573be1985b2d2ec08f9b32bf3028d` |
| `runs/real_run/logs/von11.t4.stderr.txt` | `/termux-home/ladder/real_run/logs/von11.t4.stderr.txt` | 880 | `4fe511a93d565282710299746a3bc39eb140e7df4a1a3c4e975a7202b5798049` |
| `runs/real_run/logs/von12.main.stderr.txt` | `/termux-home/ladder/real_run/logs/von12.main.stderr.txt` | 708 | `9e3eb02f65fffc81e2dde4630a3c83579bf0592bc4988dc03dc7c1c20382bcbf` |
| `runs/real_run/logs/von12.t2.stderr.txt` | `/termux-home/ladder/real_run/logs/von12.t2.stderr.txt` | 473 | `869e1d2e804dad2f2663f7c5324d2bcf5748d7f19f3d57538b3061b0192e74e4` |
| `runs/real_run/logs/von12.t3.stderr.txt` | `/termux-home/ladder/real_run/logs/von12.t3.stderr.txt` | 473 | `25cb2db2d3ebc9df7bf87e387ef2ad665df4976af520bb726b8bb423e043ae67` |
| `runs/real_run/logs/von12.t4.stderr.txt` | `/termux-home/ladder/real_run/logs/von12.t4.stderr.txt` | 472 | `d985b9d86b7857dac723e2f6ec5714a16af3977ba98f88bc11762b694ebf0f3b` |
| `runs/real_run/report.txt` | `/termux-home/ladder/real_run/report.txt` | 21002 | `5e9f1e25519f81a541bafbfe434fcb2a6c78a7bafecae3082fb7d63d5a30dd29` |
| `runs/real_run/results.json` | `/termux-home/ladder/real_run/results.json` | 52157 | `47676f3ed2b04f62cc40979566246c1e24b50580bfd93fb29c2a0d6e4f507124` |
| `runs/s1o_speed_20260926T044457Z/decisions.jsonl` | `/termux-home/ladder/s1o_speed_20260926T044457Z/decisions.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/blocks.jsonl` | `/termux-home/ladder/s1o_speed_20260926T045648Z/blocks.jsonl` | 9933 | `f0edf0b0f23c90aac5df87c90fb067d5d10b4e64014b2263cbded8c5fb75d89a` |
| `runs/s1o_speed_20260926T045648Z/decisions.jsonl` | `/termux-home/ladder/s1o_speed_20260926T045648Z/decisions.jsonl` | 1034972 | `7fea145d3cf106d468c064a75abf747333689e851f003dde3964304bc5aa8b04` |
| `runs/s1o_speed_20260926T045648Z/decisions.jsonl.before_resume_20260926T065623Z` | `/termux-home/ladder/s1o_speed_20260926T045648Z/decisions.jsonl.before_resume_20260926T065623Z` | 403980 | `f76cfe04516b6e04f049093d2a747370758bee645d8f16cb02cd302e2ae27ca2` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.main.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b1609.main.server.log` | 98577 | `e9f01ec33d956d0c9f2a54a37cbb69670006b1e0256d81855875ec7ceb39ebc7` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.main.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b1609.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.t3.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b1609.t3.server.log` | 49797 | `53f93157b0a22cf0143c15578216de0ad3a397204ae40d39e4d8eee02b6ca197` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.t3.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b1609.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.t4.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b1609.t4.server.log` | 49797 | `ae13ae53a42fb82059cdf26efa530ee87458867369c4b3e674c15a670aee471f` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.t4.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b1609.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.main.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351.main.server.log` | 98591 | `9dc796547c5883872519b5b9b6ba5d12deb537ac5d7abc8ef81a550c47ac2836` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.main.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.t3.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351.t3.server.log` | 50107 | `3755d5b7e78f575c0e189af55d1605d47126e491daf7bf86e017ce236647cce4` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.t3.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.t4.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351.t4.server.log` | 50107 | `3d514937afab49492e3c7a1c4e47b513e00f9517dedc3f0d2b7306abc0d66c74` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.t4.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.main.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.main.server.log` | 98605 | `441fc07062e6895fe3b3d5bcc315ff681735ececcd999c6bccffb1c919704023` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.main.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t3.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t3.server.log` | 50105 | `9a50fbd15dbfd359fce7b0aa79e1de6faa745b549189a6a38fde0b3e7225057a` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t3.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t4.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t4.server.log` | 50105 | `35257f535ef1fed69d6e8e67b75d807f35fcc602d4f126f86ce63c9f7653533c` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t4.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.main.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.main.server.log` | 97725 | `0f6f629000e405ac117c4ee4b23e2c5d27c6aa4ec0729dd894484120d4302e6b` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.main.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t3.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t3.server.log` | 49231 | `f12fb3fe0ff4994ad6531fcdc60094160c3595a0e41924313e9fc9e1fe99b07a` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t3.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t4.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t4.server.log` | 49231 | `c5d3cd1b2e70ca8836fc316d54d4499f6e71f24e981372fdfbc1a525fc4af873` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t4.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.main.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.main.server.log` | 97723 | `ff797f13d0dae8a55901db0f4d4ddd7b4cd83ab6e3a496ab982cb654b7a0c628` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.main.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t3.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t3.server.log` | 49229 | `698d14a70924959b606a1f3908147c8fa943eedb2b214e8a4a3806e9c4990073` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t3.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t4.server.log` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t4.server.log` | 49229 | `b8b7c58d4896bb5458c9f54402b4af70d34a889baba340d9d65f54198ae6afe4` |
| `runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t4.stderr.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/s1o_speed_20260926T045648Z/report.txt` | `/termux-home/ladder/s1o_speed_20260926T045648Z/report.txt` | 14479 | `e026d33acbcb65d10f306953041d3db5aa1e6958a44072bb3b6b83aa23ed4bd6` |
| `runs/s1o_speed_20260926T045648Z/results.json` | `/termux-home/ladder/s1o_speed_20260926T045648Z/results.json` | 33825 | `6dd8c5934bfa8d3ea5949932f4bf98824118e2ada1e62cad744b460ffb388135` |
| `runs/s1o_speed_20260926T045648Z/run.json` | `/termux-home/ladder/s1o_speed_20260926T045648Z/run.json` | 197 | `0be6a9a034960a47cc8bbb33a64dda4eea7e00e3c7145090c3aac61c0ed4d2d5` |
| `runs/smoke_run/decisions.jsonl` | `/termux-home/ladder/smoke_run/decisions.jsonl` | 54548 | `25d0ca7b2beee7bbf565d8a31d12c6566380aad05a4c4e7c2f25176f3456b751` |
| `runs/smoke_run/logs/laya_micro.main.stderr.txt` | `/termux-home/ladder/smoke_run/logs/laya_micro.main.stderr.txt` | 313 | `9e1c93b77a6e25adbc5864da95fa28859c5006ba8b8009b0962ac4b935dcd1fb` |
| `runs/smoke_run/logs/laya_micro.t2.stderr.txt` | `/termux-home/ladder/smoke_run/logs/laya_micro.t2.stderr.txt` | 313 | `aec77d7adcdbdf9e225056296ce80e5d5d9799cb554fbf6c89373e84d15fc342` |
| `runs/smoke_run/logs/laya_micro.t3.stderr.txt` | `/termux-home/ladder/smoke_run/logs/laya_micro.t3.stderr.txt` | 313 | `03fce3b9f7b0b417e5260c0f50a4a17459ee863cd60696226042f8d5e1b86955` |
| `runs/smoke_run/logs/laya_micro.t4.stderr.txt` | `/termux-home/ladder/smoke_run/logs/laya_micro.t4.stderr.txt` | 313 | `20cf68e4ab451957f6884242cd8dd0788d358b548b2aac32768ce7ff125344cb` |
| `runs/smoke_run/logs/laya_multi.main.stderr.txt` | `/termux-home/ladder/smoke_run/logs/laya_multi.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/smoke_run/logs/laya_multi.t2.stderr.txt` | `/termux-home/ladder/smoke_run/logs/laya_multi.t2.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/smoke_run/logs/laya_multi.t3.stderr.txt` | `/termux-home/ladder/smoke_run/logs/laya_multi.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/smoke_run/logs/laya_multi.t4.stderr.txt` | `/termux-home/ladder/smoke_run/logs/laya_multi.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/smoke_run/logs/s1o.main.server.log` | `/termux-home/ladder/smoke_run/logs/s1o.main.server.log` | 6851 | `39773be00e03b7121ebb30e606594e228625ec5b4b86bf6cf9dc2b08aab732b0` |
| `runs/smoke_run/logs/s1o.main.stderr.txt` | `/termux-home/ladder/smoke_run/logs/s1o.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/smoke_run/logs/s1o.t2.server.log` | `/termux-home/ladder/smoke_run/logs/s1o.t2.server.log` | 4443 | `657551792dd27111bb0e4af0326dc0277ce6ae0bda62f37faee54500411e200f` |
| `runs/smoke_run/logs/s1o.t2.stderr.txt` | `/termux-home/ladder/smoke_run/logs/s1o.t2.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/smoke_run/logs/s1o.t3.server.log` | `/termux-home/ladder/smoke_run/logs/s1o.t3.server.log` | 4443 | `3d11a98601b40d5b2aae258be352b34838d0955697b2ccc164404701ad75b989` |
| `runs/smoke_run/logs/s1o.t3.stderr.txt` | `/termux-home/ladder/smoke_run/logs/s1o.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/smoke_run/logs/s1o.t4.server.log` | `/termux-home/ladder/smoke_run/logs/s1o.t4.server.log` | 4443 | `25eff2bf88a9036eeea0aed9b9b4e87ba5c38c9498c8282dbc87e099274efeab` |
| `runs/smoke_run/logs/s1o.t4.stderr.txt` | `/termux-home/ladder/smoke_run/logs/s1o.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `runs/smoke_run/logs/von10_nli.main.stderr.txt` | `/termux-home/ladder/smoke_run/logs/von10_nli.main.stderr.txt` | 467 | `fab7395d4890a74fb76ea2ebe155eab9bbf47868d0545bde3c0550349aad2eae` |
| `runs/smoke_run/logs/von10_nli.t2.stderr.txt` | `/termux-home/ladder/smoke_run/logs/von10_nli.t2.stderr.txt` | 466 | `3f92ad7856e1a1670a0ac062faad67745c3d374799388fa1c63ba1200b98838c` |
| `runs/smoke_run/logs/von10_nli.t3.stderr.txt` | `/termux-home/ladder/smoke_run/logs/von10_nli.t3.stderr.txt` | 469 | `5edbcafe3fd3570fa45ad28196cb01ce59b7fbf760dd6057277d218d48f1f276` |
| `runs/smoke_run/logs/von10_nli.t4.stderr.txt` | `/termux-home/ladder/smoke_run/logs/von10_nli.t4.stderr.txt` | 467 | `9236833e135aaacc0d3a98df282f1805461f7e25046a55e174a87d85db24074f` |
| `runs/smoke_run/logs/von12.main.stderr.txt` | `/termux-home/ladder/smoke_run/logs/von12.main.stderr.txt` | 473 | `b71bd7d142d5d37ad7ea7e6933c61532198b9b90f4fa456cfad729e119e473ad` |
| `runs/smoke_run/logs/von12.t2.stderr.txt` | `/termux-home/ladder/smoke_run/logs/von12.t2.stderr.txt` | 473 | `0b1590950e815ec520389cf2b2d1626f3a8c8dc16562d4697ff684f0bb8ac9af` |
| `runs/smoke_run/logs/von12.t3.stderr.txt` | `/termux-home/ladder/smoke_run/logs/von12.t3.stderr.txt` | 475 | `dd1ea51d611fc28b08729f6ee2578c836cf34f9022f9bbc3150be6c7fbef1e30` |
| `runs/smoke_run/logs/von12.t4.stderr.txt` | `/termux-home/ladder/smoke_run/logs/von12.t4.stderr.txt` | 470 | `dc71b11dcc34d67194aa4b81bf87c9b1920c6ea868c939ca686da922020bdc85` |
| `runs/smoke_run/report.txt` | `/termux-home/ladder/smoke_run/report.txt` | 15544 | `fb2299d7dc181c9e77c6e4ef7ffb25df508fb1c66c263febd0f155cb610ad4ce` |
| `runs/smoke_run/results.json` | `/termux-home/ladder/smoke_run/results.json` | 36498 | `4ee0e3195c8d21a84c02bedd0e85e9cef085d57674814cdb0d0a8188b131e071` |
| `runs/toy_run/decisions.jsonl` | `/termux-home/ladder/toy_run/decisions.jsonl` | 19832 | `19c13ce213dc5d9f24963368292964674b485a820c189dd470ebd2b574dc07f8` |
| `runs/toy_run/logs/laya_en.main.stderr.txt` | `/termux-home/ladder/toy_run/logs/laya_en.main.stderr.txt` | 313 | `67e8826ac4995b4c6a1b527e6ef055f55eb22c0690ffd345d0af769249b09245` |
| `runs/toy_run/logs/laya_en.t2.stderr.txt` | `/termux-home/ladder/toy_run/logs/laya_en.t2.stderr.txt` | 313 | `0d0d26f1dd5e010a7fda5467ae4ca33d1a1e82ca908b713518aba24261c8eb50` |
| `runs/toy_run/logs/laya_en.t3.stderr.txt` | `/termux-home/ladder/toy_run/logs/laya_en.t3.stderr.txt` | 313 | `b1ef6b8e67248a93eb0fd3e27410710c07c418f51d80f9c1d75087974ee697cd` |
| `runs/toy_run/logs/laya_en.t4.stderr.txt` | `/termux-home/ladder/toy_run/logs/laya_en.t4.stderr.txt` | 313 | `e0fc0c96966ad696235b1d53ffb8009d649deaca7f09feb932af73ed3eac615d` |
| `runs/toy_run/logs/von11.main.stderr.txt` | `/termux-home/ladder/toy_run/logs/von11.main.stderr.txt` | 1116 | `f3869b10dd3aa5bd19a113d5e521fa79f5f30ac4f9aab7715b2141d30b6ddf86` |
| `runs/toy_run/logs/von11.t2.stderr.txt` | `/termux-home/ladder/toy_run/logs/von11.t2.stderr.txt` | 883 | `ec4147bc747ae4446eb0b15f4e03c803e1e801c840bb9975b416468b704ad237` |
| `runs/toy_run/logs/von11.t3.stderr.txt` | `/termux-home/ladder/toy_run/logs/von11.t3.stderr.txt` | 881 | `6897a4529630900b05fcb74cc6a44a123020fe3482d141d1617ec3fb77b3f898` |
| `runs/toy_run/logs/von11.t4.stderr.txt` | `/termux-home/ladder/toy_run/logs/von11.t4.stderr.txt` | 883 | `6d8b53c93c744c55ea7f8d433ee2c9a2a39d7f9cabf88411e0d9dcb4d59118b5` |
| `runs/toy_run/report.txt` | `/termux-home/ladder/toy_run/report.txt` | 6620 | `ab9c8f5c344c5b6de8e78d56d5d2a820784d20c789f5a9f10633d97f19d45d16` |
| `runs/toy_run/results.json` | `/termux-home/ladder/toy_run/results.json` | 14916 | `b0576b5b1de17924ffbad96a0cadbbc264a9bf19c58c3edcb815d67600cc17cb` |
| `s1o_speed_prep/adapters.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/adapters.diff` | 698 | `4c51c25ace9bc67ec66e7b57749894476b3bc78bfdcd41ea40fcc304968f9d58` |
| `s1o_speed_prep/base2/ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/base2/ladder.py` | 36016 | `1f2a5185d43c6f02a149c7b42ccd076088300b3cbdc8873cf73c770e2e2e7dd2` |
| `s1o_speed_prep/base2/test_ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/base2/test_ladder.py` | 5861 | `9aa8cc2462197a5e99b79ba520dc8df447619624ddae00f3909f50d4ce456bf4` |
| `s1o_speed_prep/base3/ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/base3/ladder.py` | 36738 | `7ab64a4e8cfccf0b87edbb685c8c0bb8d670d8278c10ee2ad5d7945b97aaf161` |
| `s1o_speed_prep/base3/run_s1o_speed.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/base3/run_s1o_speed.sh` | 1241 | `07e6ee25dad15e09a12cc5e6e1fe3de06892bd1954de85eb45485ccd98b000b6` |
| `s1o_speed_prep/base3/test_ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/base3/test_ladder.py` | 6413 | `7f030b30a7748091a7b06560a8243f82392811654603303b5984234c18555493` |
| `s1o_speed_prep/e2e.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e.sh` | 1342 | `00ba0bc6cc975547fe1a528a23be9dfad8622f012a9e602a28bf9c07a0e7aaea` |
| `s1o_speed_prep/e2e_1.err` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_1.err` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/e2e_1.out` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_1.out` | 1630 | `ca0b0c17f230d35001360d8cd0e567dac668943a8fdde202e0920749052d8c9f` |
| `s1o_speed_prep/e2e_2.err` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_2.err` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/e2e_2.out` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_2.out` | 6691 | `72c787061e507876dbcef7fd5942ddfe7d1d0eae23f02c939ea2d4e2fcf91b9b` |
| `s1o_speed_prep/e2e_run/blocks.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/blocks.jsonl` | 3500 | `3c02f67255b34e6fbdd7cda72a876c2f41fdb8086145e04df8d26f0dd3eef007` |
| `s1o_speed_prep/e2e_run/blocks.jsonl.before_resume_20260926T062950Z` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/blocks.jsonl.before_resume_20260926T062950Z` | 1902 | `8e73239fda00c7ba0537adc7e5190a5e820f5e4f02a397866d364da348fcf719` |
| `s1o_speed_prep/e2e_run/decisions.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/decisions.jsonl` | 28944 | `0276a83f28c376ef39428c9a017a3ddae9aa2e1e6d6f98246bcaae84b2ea777f` |
| `s1o_speed_prep/e2e_run/decisions.jsonl.before_resume_20260926T062950Z` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/decisions.jsonl.before_resume_20260926T062950Z` | 19316 | `3c00df3ca3c69e77f9998708c14d4c54c91ffe0704fe5bf3b1d46090b2212de2` |
| `s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen08b.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/logs/s1o_b2351_qwen08b.main.server.log` | 9791 | `7a9f2a6ab89d22a0201f1ce4fe8a91f5bb4fe529f129d78e9a763c2ba707407d` |
| `s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen08b.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/logs/s1o_b2351_qwen08b.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/logs/s1o_b2351_qwen2b.main.server.log` | 9789 | `e44a626603fe9803f6ee0e88bd00a9f17318b53e38905cce22a128022cd951b5` |
| `s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/logs/s1o_b2351_qwen2b.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.t3.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/logs/s1o_b2351_qwen2b.t3.server.log` | 5778 | `bbec9aa5b598def606824cfd6de70c475d90b9f13609cb8226c664c51150eb2a` |
| `s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.t3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/logs/s1o_b2351_qwen2b.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.t4.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/logs/s1o_b2351_qwen2b.t4.server.log` | 5778 | `6adee5a11fedbeaf3f2e85242d31f94da2eedd39abc5f1c50ac0b10b5da9e17c` |
| `s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.t4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/logs/s1o_b2351_qwen2b.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/e2e_run/report.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/report.txt` | 5836 | `1058843760960bc415dd0b2480944058575651b0fe0664a1fd9c8206df8b5fa0` |
| `s1o_speed_prep/e2e_run/results.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/results.json` | 12537 | `acb39ea15b72c66e4188ca5b03b634fdfcbbe3a3640c365028813d49468aa576` |
| `s1o_speed_prep/e2e_run/run.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/e2e_run/run.json` | 243 | `ce7b267279bc1081dd6fae829f7f44b0656faca1ba6292f0b548021e085626a2` |
| `s1o_speed_prep/fake_thermal.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/fake_thermal.log` | 960 | `907fd538df73539f0313f3260f4f742af3f579d9511cd661e8867a452ecb6216` |
| `s1o_speed_prep/fake_thermal.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/fake_thermal.sh` | 186 | `5617e962ba2d29f2de69512d8f0f56ceae5d4c238670b25d0942e78f6e1fb484` |
| `s1o_speed_prep/gguf.sha256` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/gguf.sha256` | 2111 | `b173d26e938da8ad4b8db50e7aa9281109942c9e7c9ac16fcf82d6985efcd2be` |
| `s1o_speed_prep/l1.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/l1.log` | 3176 | `6968da002fc928636b37c7225bd4b9991363969de3ba36d3bd02fef0a795fbf8` |
| `s1o_speed_prep/l2.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/l2.jsonl` | 333 | `2a20be506dca492fa8c70bfa3a06122c1dc4f727e2e8da9abc59c6af29a76d82` |
| `s1o_speed_prep/l2.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/l2.log` | 6482 | `c9bedd1ebcb8fa3e7e9cadd2f04b35d4c00752b4f520ba7022a950b9888d0ef4` |
| `s1o_speed_prep/l3.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/l3.log` | 5062 | `f5eef28c034c481e4d8e757db3589287593f78cf8f7b358f0e467fe7164138c4` |
| `s1o_speed_prep/ladder.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/ladder.diff` | 8255 | `44139e57a71eef95e6c985a55d646ab36863e03198120c1efe5e92403868c52d` |
| `s1o_speed_prep/lens.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/lens.sh` | 873 | `541b56e8407ac463029db7c85cd120dbcaa5e2c45fa0e09025d80eefa7e3d28e` |
| `s1o_speed_prep/live.sha` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/live.sha` | 260 | `e1fdc796a9b0bfa042f9d13fd97879f28fdcb0489b3e1d3250e05cbe749fe847` |
| `s1o_speed_prep/orig/adapters.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/orig/adapters.py` | 17240 | `23e90a3d51daea89f3689aaadb3f4a56a43f23f26dd76785257f15c653e099bc` |
| `s1o_speed_prep/orig/ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/orig/ladder.py` | 32023 | `00fbe2fab81021cb884f54550042b21be9f3b9e61943efea6d63f79139bb5f8b` |
| `s1o_speed_prep/orig/test_ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/orig/test_ladder.py` | 4785 | `daf7402ab0365725e66d3dd7da13f2c6925ebca3b422f806e8e7026c78101e8a` |
| `s1o_speed_prep/partial.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/partial.py` | 1169 | `871a7d0755baa0c2615e77d92dd7f14f096c1c917f793f0f760fbd7451d63b40` |
| `s1o_speed_prep/partial_report.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/partial_report.txt` | 5038 | `e5736d7fe1e40d853b59cbc4ee050c560dd1332593324db3025068bfd43055b1` |
| `s1o_speed_prep/probe.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/probe.py` | 854 | `3964e1eb8d9fe204e91f5d1a7681c20fe3cf2eefa4e886bf125ab487610a1ebb` |
| `s1o_speed_prep/quick.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick.py` | 263 | `71247c36f50e1b1ac5df7db78362de6ad1faa2a365a678257d195d1ad9ddecb8` |
| `s1o_speed_prep/quick2.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick2.py` | 357 | `7ffe6a56eaa9632e0135b45be512c8cbc21662924d108a923142b51679a1f4cd` |
| `s1o_speed_prep/quick5.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick5.jsonl` | 2032 | `0ea78770cfb86dc2defb8a763acb8078667a4b52c181ec86e42f4561c8876658` |
| `s1o_speed_prep/quick_run1.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run1.stderr.txt` | 960 | `5a466d305f4b04f15eb2f5ef81608c266cbd06cba8bb97d7faa2c29cdae4b209` |
| `s1o_speed_prep/quick_run1.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run1.stdout.txt` | 265 | `8d53c40bb233c05a64bafc6001c31565a354a6ce9e66a00de9b76808f57a6951` |
| `s1o_speed_prep/quick_run1/decisions.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run1/decisions.jsonl` | 1642 | `ae7e982f31307c166d403d1f3768fe0d1cea90ba0336282fe2d29486a8f3d2f7` |
| `s1o_speed_prep/quick_run1/logs/s1o_b2351.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run1/logs/s1o_b2351.main.server.log` | 4015 | `e2e5d4dcbab9385bd1f1a20c52458baa66a1b0e031447b8defe6381e16320464` |
| `s1o_speed_prep/quick_run1/logs/s1o_b2351.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run1/logs/s1o_b2351.main.stderr.txt` | 466 | `8dd33f68ee16b34c397ed176cbbd05b611d55f84c8d482443f659d74d83356b0` |
| `s1o_speed_prep/quick_run2.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2.stdout.txt` | 15054 | `2ad28d84ea4a921726ecd497fd978249eff5d9cc2801bb9a276129f5b6e7059a` |
| `s1o_speed_prep/quick_run2/decisions.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/decisions.jsonl` | 77441 | `58c5dd20206b2c6a70bfc0ee5bbb0ab88feba557a14e77b31d2036a19bf72448` |
| `s1o_speed_prep/quick_run2/logs/s1o_b1609.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b1609.main.server.log` | 10101 | `13b730f5f91dddd2bb0aec6234fda7328483162b347566863c552d28233fd26f` |
| `s1o_speed_prep/quick_run2/logs/s1o_b1609.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b1609.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b1609.t3.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b1609.t3.server.log` | 6080 | `aee08e26d56417098d8ba3b41ef001ddde969e163bb350e62cf88d247ade6732` |
| `s1o_speed_prep/quick_run2/logs/s1o_b1609.t3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b1609.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b1609.t4.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b1609.t4.server.log` | 6080 | `3af59280cc48ec206e1ffc41230f32885479fd11758fee0aa915c83520872a73` |
| `s1o_speed_prep/quick_run2/logs/s1o_b1609.t4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b1609.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351.main.server.log` | 10495 | `72b1989fd08567b9768eb1dc5fc7678f09c7cf573903c255aad95f736a867e98` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351.t3.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351.t3.server.log` | 6484 | `07ba4dd0422ef313565a0f07817cc3972a8619cbf82de8bfbe496008dbaadbd3` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351.t3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351.t4.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351.t4.server.log` | 6484 | `86f6a78208649120d22b037d50e46ab0ff72f6f4cbb36671e6eacd46fa8aa5ba` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351.t4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen08b.main.server.log` | 9789 | `eebebf4131f4a3d73a438ee8c37c8d5e8ceb985363870eebf6465653e12f80dd` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen08b.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.t3.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen08b.t3.server.log` | 5778 | `d86b5fc49a04be8d97110508bcb14fffb5811275777cc89d1234b57e22d3e66e` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.t3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen08b.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.t4.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen08b.t4.server.log` | 5778 | `fafc2f8ef12d4a4c7763a227acba95b5633a08f642a8b48d3706de63f93b9c7f` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.t4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen08b.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen2b.main.server.log` | 9787 | `100aa21cc97a157b49dd5ee7244bce78ee15c6296c6d0b5bb0c7a57adb7c0f22` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen2b.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.t3.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen2b.t3.server.log` | 5776 | `50a25b185271e855343281ea1c521acc206a069113b81617a2ab03f3bb2cd67f` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.t3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen2b.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.t4.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen2b.t4.server.log` | 5776 | `c217f9fa8f9d3974a7280386a6fe8afde6f6caafdb35889b65d237655677267d` |
| `s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.t4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/logs/s1o_b2351_qwen2b.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run2/report.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/report.txt` | 11887 | `207c7f4e14a57640f38a8c00ade83651952f75b5b02136cc49448e4aba43766f` |
| `s1o_speed_prep/quick_run2/results.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run2/results.json` | 29460 | `baab3271999ba0a7ca47bcf3893a295c65d5675247ad5f1ce4155e41b4893026` |
| `s1o_speed_prep/quick_run3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run3.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3.stdout.txt` | 4470 | `f9ce4d021afdd54a23334cd615c5629be55448257603c910d84f0ee3308ae906` |
| `s1o_speed_prep/quick_run3/decisions.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3/decisions.jsonl` | 19346 | `e406522cd093bbfc30da359e1c8de37c914fb8b8da54bbb4359c6b3abf07c5ce` |
| `s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3/logs/s1o_b2351_q40.main.server.log` | 10493 | `d307878b45546ee6313d29416ed8df9b9199af1dc17bfbd100c62e3832c51f4b` |
| `s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3/logs/s1o_b2351_q40.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.t3.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3/logs/s1o_b2351_q40.t3.server.log` | 6482 | `1875c132063e8fdf651b4b556e0607da0d3c850c36a1a84967fc33cc8d9db7b6` |
| `s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.t3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3/logs/s1o_b2351_q40.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.t4.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3/logs/s1o_b2351_q40.t4.server.log` | 6482 | `1813d7e700174773987a44b6c9479c8927a432ae91a4fe43807bb4b570a101a5` |
| `s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.t4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3/logs/s1o_b2351_q40.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run3/report.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3/report.txt` | 3604 | `90e0cb1ac7fe493a211cb7aead7cf308cea66900a78e387fc4de165567701803` |
| `s1o_speed_prep/quick_run3/results.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run3/results.json` | 7877 | `90ba16a809b69b51aed0920c3142178189998175d77b10dfed6b08e49378f5ef` |
| `s1o_speed_prep/quick_run4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run4.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4.stdout.txt` | 4426 | `f66fa73fbf27a6fb5769137a15491aa82aa20bc178684afead62068af60d7141` |
| `s1o_speed_prep/quick_run4/decisions.jsonl` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4/decisions.jsonl` | 19283 | `24c662f9e1d33532c93c2c3d4886e93e032329e0ed1702d9db6814f37a3af98a` |
| `s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.main.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4/logs/s1o_b2351_qwen08b.main.server.log` | 9789 | `2f57829de36a5e248634e24212366ba4afb61f3ceb7bdaa7b2e8dc56d1d2b4f2` |
| `s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.main.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4/logs/s1o_b2351_qwen08b.main.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.t3.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4/logs/s1o_b2351_qwen08b.t3.server.log` | 5778 | `2017266f51812592cdec9835b6d4ae5041b55fe77f600918ffbc37be98c8c056` |
| `s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.t3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4/logs/s1o_b2351_qwen08b.t3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.t4.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4/logs/s1o_b2351_qwen08b.t4.server.log` | 5778 | `e8072567ba0dff8f948a172d958b26e03aa3d32214f6b39286c9035119f42a10` |
| `s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.t4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4/logs/s1o_b2351_qwen08b.t4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/quick_run4/report.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4/report.txt` | 3516 | `3e4ff043ffb28b9d0db697f010210d48c379d451ca578298da67e3fbbfeb4070` |
| `s1o_speed_prep/quick_run4/results.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/quick_run4/results.json` | 7773 | `4e4960ec14344cbbc3ddd32423e1c26bee75228e4d9cb5688873061be453fd7e` |
| `s1o_speed_prep/r.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/r.log` | 1584 | `41124fcea6995d74f2b9070bc86a4c16761a61111f0345af9c301e017a9da27b` |
| `s1o_speed_prep/repro.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/repro.py` | 926 | `9ef49c0d90dc4bdfdcd863e0a8996158784f5fc451bd38053e0637f8606df1c6` |
| `s1o_speed_prep/repro.server.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/repro.server.log` | 3261 | `0fe5f489aba3d197cf7fe6d8bf049927ba7a2ee41e3e1695c26f5c43b42d5b30` |
| `s1o_speed_prep/review.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review.stderr.txt` | 303 | `6f9b232ca1f3b43aa72e2ba63698c09b0796b612d5a17cf05efee31e6c79b713` |
| `s1o_speed_prep/review.stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review.stdout.json` | 322 | `5e58a86726cb1725579889b2787fc69c77bb5fd18dae7f2cfa708e65dde27048` |
| `s1o_speed_prep/review/INVENTORY.md` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/INVENTORY.md` | 4185 | `58abf0528dbb92ff0adac1874fcac567ae38441c86a9f3c776a7a16adfcbd976` |
| `s1o_speed_prep/review/REVIEW_REQUEST.md` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/REVIEW_REQUEST.md` | 3155 | `6769aa1142694bd216cd9171e64ca913d3551f45bf8db7ec5497735d97e44143` |
| `s1o_speed_prep/review/adapters.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/adapters.diff` | 698 | `4c51c25ace9bc67ec66e7b57749894476b3bc78bfdcd41ea40fcc304968f9d58` |
| `s1o_speed_prep/review/candidate.sha256` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/candidate.sha256` | 991 | `0b296d3d68cf172436816154ad2aa5f7efb4af1c9f01f72f13f84ec4b9a3ae79` |
| `s1o_speed_prep/review/candidate/adapters.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/candidate/adapters.py` | 17401 | `b452ab09d3a9772b43771e1d687c0bc013056a09c45aa4a464c07ac2dd0e3fec` |
| `s1o_speed_prep/review/candidate/ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/candidate/ladder.py` | 36016 | `1f2a5185d43c6f02a149c7b42ccd076088300b3cbdc8873cf73c770e2e2e7dd2` |
| `s1o_speed_prep/review/candidate/ladder_worker.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/candidate/ladder_worker.py` | 9000 | `f795b354d6b615a681b25066d79b91e58ad8c1f68ae16b516a364bff8473b07e` |
| `s1o_speed_prep/review/candidate/run_s1o_speed.sh` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/candidate/run_s1o_speed.sh` | 1249 | `65505ae626f17b3f03628345f15d80fd870c427b316be9696b1a3d726a2d6c41` |
| `s1o_speed_prep/review/candidate/test_ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/candidate/test_ladder.py` | 5861 | `9aa8cc2462197a5e99b79ba520dc8df447619624ddae00f3909f50d4ce456bf4` |
| `s1o_speed_prep/review/ladder.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/ladder.diff` | 8255 | `44139e57a71eef95e6c985a55d646ab36863e03198120c1efe5e92403868c52d` |
| `s1o_speed_prep/review/orig/adapters.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/orig/adapters.py` | 17240 | `23e90a3d51daea89f3689aaadb3f4a56a43f23f26dd76785257f15c653e099bc` |
| `s1o_speed_prep/review/orig/ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/orig/ladder.py` | 32023 | `00fbe2fab81021cb884f54550042b21be9f3b9e61943efea6d63f79139bb5f8b` |
| `s1o_speed_prep/review/orig/test_ladder.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/orig/test_ladder.py` | 4785 | `daf7402ab0365725e66d3dd7da13f2c6925ebca3b422f806e8e7026c78101e8a` |
| `s1o_speed_prep/review/quick.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/quick.py` | 263 | `71247c36f50e1b1ac5df7db78362de6ad1faa2a365a678257d195d1ad9ddecb8` |
| `s1o_speed_prep/review/quick_run2.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/quick_run2.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/review/quick_run2.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/quick_run2.stdout.txt` | 11624 | `81143c86a01ab1ffa6b8cf17f1b20e439207bcceb2242e27db71b7fe4c0c6806` |
| `s1o_speed_prep/review/quick_run3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/quick_run3.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/review/quick_run3.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/quick_run3.stdout.txt` | 3486 | `a18e6bcb3cbaf34e217dbdecb53a646e4aea3381fa41c0521786f6ca53e880f7` |
| `s1o_speed_prep/review/test_ladder.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/test_ladder.diff` | 1367 | `2a3d448248d7a8e6732812f3e4514a492a69424049547e876d2f70de3f8943b9` |
| `s1o_speed_prep/review/test_output.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review/test_output.txt` | 56 | `49651752fc7ad70385aaf532ff5c93f1eb23bb26778effb860e7bb5c9b0ec7c2` |
| `s1o_speed_prep/review2.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review2.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/review2.stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review2.stdout.json` | 4128 | `3f1f41090b9d2a4f0311541ea37b934b10b4d5d9828884876a6cbb72f7e58ee3` |
| `s1o_speed_prep/review3.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3.stderr.txt` | 480 | `4df579619f582ea4ce93d7304548d3049c66bbfb4b1d1a2b547075d9efaef08c` |
| `s1o_speed_prep/review3.stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3.stdout.json` | 421 | `b2389dd937ca95711b8f41f21de23614735c202dae9503214053267c685a1c2f` |
| `s1o_speed_prep/review3/REQUEST.md` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3/REQUEST.md` | 2123 | `30d0191d4008e722f6b73f6d337c8c25e2d1ebec832bee02ad73c5363846929b` |
| `s1o_speed_prep/review3/candidate.sha256` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3/candidate.sha256` | 300 | `945fedf0edc621630b734b72880a354b2c0800e80daa45355136c8786d6052ef` |
| `s1o_speed_prep/review3/ladder.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3/ladder.diff` | 4981 | `2e1d6dc3641ba6c535d61bf1b6f300d4696b5875ba8ac9dd4edc8d2b4b7da13b` |
| `s1o_speed_prep/review3/prompt.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3/prompt.txt` | 58005 | `edf4c75f6a0879953bf022c3263553854215ae6bc02f9e3feec60df171c8a703` |
| `s1o_speed_prep/review3/quick_run4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3/quick_run4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/review3/quick_run4.stdout.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3/quick_run4.stdout.txt` | 3573 | `6501a0f4f825e8c376921eece4132c1da2340a20e8b7b6a7db1639aa6a0952ba` |
| `s1o_speed_prep/review3/run_s1o_speed.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3/run_s1o_speed.diff` | 915 | `6d3535f4ae4d59a760054011c4ac673307b3628dd4c7f37d81bdb91227af39dd` |
| `s1o_speed_prep/review3/test_ladder.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3/test_ladder.diff` | 1531 | `158bf15c9ac90f558e356acf6e91ba02266a568f20ae20586a2e8fe0d57e487b` |
| `s1o_speed_prep/review3/test_output.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3/test_output.txt` | 139 | `f2e9b6001f9293bb5225b64a7d0a43ace8970255b4e95c7818934103553301fb` |
| `s1o_speed_prep/review3b.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3b.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/review3b.stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review3b.stdout.json` | 2182 | `209f9e756b88243af9e1bf81cecd29d8420a9586ee4cd0795529f38737494aef` |
| `s1o_speed_prep/review4.stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review4.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o_speed_prep/review4.stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review4.stdout.json` | 3863 | `1620b8f0161c7c7187c6ed22a731ba70cb739cb5da6f0364f7f55b41d80f4ae7` |
| `s1o_speed_prep/review4/REQUEST.md` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review4/REQUEST.md` | 3623 | `b3e55e83a2e470864e890985f09686e00f09fff556a7d0cd5aa32584e1a6677c` |
| `s1o_speed_prep/review4/candidate.sha256` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review4/candidate.sha256` | 300 | `c6c996df213b0217faef9061456e8fd320d3acb02681aedcd59f51daba1327cc` |
| `s1o_speed_prep/review4/ladder.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review4/ladder.diff` | 21008 | `04df6a0a026ccc3e0ab673f25cb3f0819615f1ebc49a53b89d84cab67f43d310` |
| `s1o_speed_prep/review4/prompt.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review4/prompt.txt` | 106476 | `c3041089eb2c2213a84f93ef41441866d8bb24d231b911c1b41df358c3280297` |
| `s1o_speed_prep/review4/run_s1o_speed.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review4/run_s1o_speed.diff` | 2590 | `84327c6eaee8b9c9e7ae8a34c427c6fc26f4d10f67bbd1691d68ffbc270461c6` |
| `s1o_speed_prep/review4/test_ladder.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review4/test_ladder.diff` | 5029 | `15d3f445ba7e255838fa22d3a25e7732b2af3a9b0a68d0b32ad5ad25984ac01f` |
| `s1o_speed_prep/review4/test_output.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review4/test_output.txt` | 383 | `20c1270984aad8b618a909bbdb6d056ac2d49450586cc4ade4910d2b9fcb48f9` |
| `s1o_speed_prep/review_prompt.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/review_prompt.txt` | 97405 | `227f2a00ae6f9bc50ce8f098274e8a1d3dac3aa9834c4273a5fa8fdc7d39221b` |
| `s1o_speed_prep/test_ladder.diff` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/test_ladder.diff` | 1367 | `2a3d448248d7a8e6732812f3e4514a492a69424049547e876d2f70de3f8943b9` |
| `s1o_speed_prep/v.log` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/b32f7c1a-ea17-4af1-98e0-189c71dbb218/scratchpad/v.log` | 246207 | `2fe34c55b0f2c10e44313df0aee7591ea51ebfd87c4c0fecfc692ad5db5628dc` |
| `test_ladder.py` | `/termux-home/ladder/test_ladder.py` | 9728 | `19da97c0fb8726d06d454f1d1044e9577c0fc10ef30bd774c80d60c5f208170f` |
| `toy_cases.jsonl` | `/termux-home/ladder/toy_cases.jsonl` | 1736 | `2ebe1f9fd4eac5d9329d6fe8af2b2bb7dbccf170239c687191ed6ae9e5ebb7bc` |

## Written for this archive or already in place in the repository (2 files)

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 5353 | `f5c743841c6315fca32b0c77359fa46109d26a977834f28972562e29bf61d5a0` |
| `RUN_INDEX.md` | 1066 | `a86c05477a2b803b72d272ccfb203bb59c3c57913377562f0c502facafb6418f` |

## sha256sum format

```sha256sums
f5c743841c6315fca32b0c77359fa46109d26a977834f28972562e29bf61d5a0  README.md
a86c05477a2b803b72d272ccfb203bb59c3c57913377562f0c502facafb6418f  RUN_INDEX.md
41eeafd28e2ce499437f75091e2069bdd89d4e9e0d789f223ff4dd1c9bebb473  cases/ladder_cases_v1.jsonl
c1a4f65df8dc78e0400a1a27d98e6e1766635a5d1185a31a8b8940e3ae39a0f7  ladder.py
f795b354d6b615a681b25066d79b91e58ad8c1f68ae16b516a364bff8473b07e  ladder_worker.py
00413cc2960ef50225ddd095c1144a7ca8f7326750f8ec29d2276f122d3e45d9  logs/oneshot_console_20260926T045141Z.log
cdd485ea0ed5818a378c3d49b87b0187f557da38930bba97add088ad40207d90  logs/oneshot_console_20260926T065115Z.log
31a632e94a00ad2eed33fd03e83a04f5eb108d18366ba05ac3a10a8ce99bd6cb  logs/oneshot_console_20260926T185449Z.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/s1o_speed_20260926T044457Z.stderr.txt
1ed8db43757dfc4be901a0db0af31ae684a214046080bae3c6707c50eb97a184  logs/s1o_speed_20260926T044457Z.stdout.txt
09de8ccfd04dbf3bc2cbc871ad2d14c998349c50a5890d476a195f58ff38d02f  logs/s1o_speed_20260926T045648Z.stderr.txt
c7c20061cbed3e9db1f55c8344ef40287b1fec57b2568738bd316fffdd0ac53c  logs/s1o_speed_20260926T045648Z.stdout.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/smoke_run.stderr.txt
8c5fa4aa212f599760d496625ce5c8a0b5ea181d5558fc40f7695d12dcfed963  logs/smoke_run.stdout.txt
7e94472e87bb73cdac94c321d1edef880e252785c4f07e848d0596c4eb246fce  logs/thermal.log
33e4c366929d655b01aa11ab0c713cd4f7faf8c1f4953497b9a0a23db7e81ea4  logs/toy_run.pty.txt
7040be22ec0bf2ef8dadb5b46a52e10402ab84d1ffd38eab0020430519705242  measure.py
f3d531fcbc306c8ef5f6e4d6504c91edcaeddc6e66313673dc839dd57e1b7d31  oneshot.sh
db1f315681892f0c1f89de92b86d7495a47cc171a3c03a170fc51d2f8dbdc8e8  oneshot_executed_conversation_run.sh
0edcae559ea922a7be9ceefc9b4e20e5191eb8526494929046eb44103232a456  oneshot_executed_s1o_resume.sh
cdcdcfcc74c15b9a4c07b0c20b6a127e1315008e0933c7172d0e65834804e0e7  reports/CODER_REPORT_ladder_build.md
55c1e5ec4985c48e44f8320c063addb22069e7b4d0751993d1eee6642910322c  reports/CODER_REPORT_ladder_s1o_speed_prep.md
954b9844198bbdc35d1711aea2c2ef44f27d284688f3314d7e5365f3dd89e2d3  reports/s1o_speed_prep_report.md
d4169b20e577e22515bb51d92d9db412b4815fc22d712722195e5495b09eab0d  reviews/ladder-review-20260925T204237Z/FROZEN.sha256
4ad29ec9ef0cf56aca2cc35a5dcd37a35ac28e359c63573547913e08de8072aa  reviews/ladder-review-20260925T204237Z/REVIEW_REQUEST.txt
aef3639a6460def336de74ea0fe37c19b472ba638b8f79654451d7735e2ae6cd  reviews/ladder-review-20260925T204237Z/candidate/ladder.py
1174a0f3632b43884bd39b05e7a39f405ae9e1f118d9d4c5a23c587711c92e6d  reviews/ladder-review-20260925T204237Z/candidate/ladder_worker.py
ab02053e120887def6f275a22c07e0e628e03bbaf1bee547bd2702d37e986d32  reviews/ladder-review-20260925T204237Z/candidate/test_ladder.py
2ebe1f9fd4eac5d9329d6fe8af2b2bb7dbccf170239c687191ed6ae9e5ebb7bc  reviews/ladder-review-20260925T204237Z/candidate/toy_cases.jsonl
dc51b8c96c2d745df3bd5590d990230a482fd247123599548e0632fdbf97fc22  reviews/ladder-review-20260925T204237Z/check_test_ladder.txt
23e90a3d51daea89f3689aaadb3f4a56a43f23f26dd76785257f15c653e099bc  reviews/ladder-review-20260925T204237Z/context/adapters.py
fcce2b24dabd475088765296792da408cf23c24b0f5bb4daf3720893b757d6d4  reviews/ladder-review-20260925T204237Z/context/jevlike.py
06eecc622378efd91e6cb0839a39f84fefa8eef797a8512845f6117df61661f9  reviews/ladder-review-20260925T204237Z/context/jevlike_worker.py
9a1f6b416fe65623a43645a5adb68d769af05e7ee105e5de4d3055babab0c029  reviews/ladder-review-20260925T204237Z/context/laya_micro_runtime.py
7040be22ec0bf2ef8dadb5b46a52e10402ab84d1ffd38eab0020430519705242  reviews/ladder-review-20260925T204237Z/context/measure.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reviews/ladder-review-20260925T204237Z/review.stderr.txt
c8ba2851c3328d0e810185b98f98184db2f4d7e7a58ce6cbcc1a2bcdc68e329f  reviews/ladder-review-20260925T204237Z/review.stdout.json
4b75d9fee3ad438d94632adc54aaaa98d57f79e17804d9411f8528c822c39cad  reviews/ladder-review-20260925T204237Z/toy_run/decisions.jsonl
6b3544e2b087df7a5460b97c1c46dad59f9924297c6c38773b9f700e72a9d9df  reviews/ladder-review-20260925T204237Z/toy_run/report.txt
ba25a9c1f9e5ce0ef4424483d0070054ffe9228cd98fb5e12f52eee2b08a64a7  reviews/ladder-review-20260925T204237Z/toy_run/results.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reviews/ladder-review-20260925T204237Z/toy_run/toy_run.stderr.txt
ec1bdee7dcf981b50e34657dce982720baebd7cfea7409d035d3097095eda5ab  reviews/ladder-review-20260925T204237Z/toy_run/toy_run.stdout.txt
9f0f7a7c62e37e72f5eb7d00e54b3c6892ac04872034a6e14d60fe9d621aa21b  reviews/ladder-review-20260925T211413Z/FROZEN.sha256
87cedc59d323af0f5592ab7b43d2bb0a2c65cea5396052601d7bb8a9ea65bc1d  reviews/ladder-review-20260925T211413Z/REVIEW_REQUEST.txt
e99c5b073bc4f38a0c97861ddc37b88966e657a57e4c6de544f14531d5bcf4e8  reviews/ladder-review-20260925T211413Z/candidate.diff
00fbe2fab81021cb884f54550042b21be9f3b9e61943efea6d63f79139bb5f8b  reviews/ladder-review-20260925T211413Z/candidate/ladder.py
f795b354d6b615a681b25066d79b91e58ad8c1f68ae16b516a364bff8473b07e  reviews/ladder-review-20260925T211413Z/candidate/ladder_worker.py
daf7402ab0365725e66d3dd7da13f2c6925ebca3b422f806e8e7026c78101e8a  reviews/ladder-review-20260925T211413Z/candidate/test_ladder.py
2ebe1f9fd4eac5d9329d6fe8af2b2bb7dbccf170239c687191ed6ae9e5ebb7bc  reviews/ladder-review-20260925T211413Z/candidate/toy_cases.jsonl
dc51b8c96c2d745df3bd5590d990230a482fd247123599548e0632fdbf97fc22  reviews/ladder-review-20260925T211413Z/check_test_ladder.txt
23e90a3d51daea89f3689aaadb3f4a56a43f23f26dd76785257f15c653e099bc  reviews/ladder-review-20260925T211413Z/context/adapters.py
fcce2b24dabd475088765296792da408cf23c24b0f5bb4daf3720893b757d6d4  reviews/ladder-review-20260925T211413Z/context/jevlike.py
06eecc622378efd91e6cb0839a39f84fefa8eef797a8512845f6117df61661f9  reviews/ladder-review-20260925T211413Z/context/jevlike_worker.py
9a1f6b416fe65623a43645a5adb68d769af05e7ee105e5de4d3055babab0c029  reviews/ladder-review-20260925T211413Z/context/laya_micro_runtime.py
7040be22ec0bf2ef8dadb5b46a52e10402ab84d1ffd38eab0020430519705242  reviews/ladder-review-20260925T211413Z/context/measure.py
d4169b20e577e22515bb51d92d9db412b4815fc22d712722195e5495b09eab0d  reviews/ladder-review-20260925T211413Z/previous_review/FROZEN.sha256
aef3639a6460def336de74ea0fe37c19b472ba638b8f79654451d7735e2ae6cd  reviews/ladder-review-20260925T211413Z/previous_review/candidate/ladder.py
1174a0f3632b43884bd39b05e7a39f405ae9e1f118d9d4c5a23c587711c92e6d  reviews/ladder-review-20260925T211413Z/previous_review/candidate/ladder_worker.py
ab02053e120887def6f275a22c07e0e628e03bbaf1bee547bd2702d37e986d32  reviews/ladder-review-20260925T211413Z/previous_review/candidate/test_ladder.py
2ebe1f9fd4eac5d9329d6fe8af2b2bb7dbccf170239c687191ed6ae9e5ebb7bc  reviews/ladder-review-20260925T211413Z/previous_review/candidate/toy_cases.jsonl
c8ba2851c3328d0e810185b98f98184db2f4d7e7a58ce6cbcc1a2bcdc68e329f  reviews/ladder-review-20260925T211413Z/previous_review/review.stdout.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  reviews/ladder-review-20260925T211413Z/review.stderr.txt
4de3aa8bf3ae21ab83af5919d5c9ffe96e74e5e56ef41c7b0690fb8ecf5f21b5  reviews/ladder-review-20260925T211413Z/review.stdout.json
19c13ce213dc5d9f24963368292964674b485a820c189dd470ebd2b574dc07f8  reviews/ladder-review-20260925T211413Z/toy_run/decisions.jsonl
ab9c8f5c344c5b6de8e78d56d5d2a820784d20c789f5a9f10633d97f19d45d16  reviews/ladder-review-20260925T211413Z/toy_run/report.txt
b0576b5b1de17924ffbad96a0cadbbc264a9bf19c58c3edcb815d67600cc17cb  reviews/ladder-review-20260925T211413Z/toy_run/results.json
33e4c366929d655b01aa11ab0c713cd4f7faf8c1f4953497b9a0a23db7e81ea4  reviews/ladder-review-20260925T211413Z/toy_run/toy_run.pty.txt
a335b7633c9c0bce6d86744258e101c4b076982e618f6c434e34cc82ef6faee8  run_conversation.sh
308a19c5c38fae6c2cbabe71fb0af17d1eda94bb841f3746b06ec06fc1df8efa  run_s1o_speed.sh
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/old_runs/toy_run.stderr.txt
ec1bdee7dcf981b50e34657dce982720baebd7cfea7409d035d3097095eda5ab  runs/old_runs/toy_run.stdout.txt
4b75d9fee3ad438d94632adc54aaaa98d57f79e17804d9411f8528c822c39cad  runs/old_runs/toy_run/decisions.jsonl
7b7dd217a7b63ea9637b220c9f10a99fbc6c7f2b7464921d76fca05b32213293  runs/old_runs/toy_run/logs/laya_en.main.stderr.txt
6cdb0e96d30a5ef5f6655e1a037c15fd714fe862c21f5ffeeb62ed096e514a09  runs/old_runs/toy_run/logs/laya_en.t2.stderr.txt
ab046b0553ca029309eaeb16194bba2a2a5ba9aeb4c4e4ff420f2971769b7ebe  runs/old_runs/toy_run/logs/laya_en.t3.stderr.txt
475be39578519967c3f8d7cb17ec267517a6b7c2b06a0216b9f1ddf0cde29cb2  runs/old_runs/toy_run/logs/laya_en.t4.stderr.txt
d6af1f3179b319dd91b4fd0d06a5a75ec40aacff535a83647c792be5d85104c8  runs/old_runs/toy_run/logs/von11.main.stderr.txt
3d77d35894c18287f42944a20e59859b3dd270ec21e2772091d7c980571efdba  runs/old_runs/toy_run/logs/von11.t2.stderr.txt
88c03054ae8f2e959c648c6b3dcf1ba7644afe18e89bc6ee5b71b3b200f404bf  runs/old_runs/toy_run/logs/von11.t3.stderr.txt
e0b69733024bba12f36c8f5a7ee016f80a53517fa957b135d474cad7dd7562f2  runs/old_runs/toy_run/logs/von11.t4.stderr.txt
6b3544e2b087df7a5460b97c1c46dad59f9924297c6c38773b9f700e72a9d9df  runs/old_runs/toy_run/report.txt
ba25a9c1f9e5ce0ef4424483d0070054ffe9228cd98fb5e12f52eee2b08a64a7  runs/old_runs/toy_run/results.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/old_runs/toy_run_1.stderr.txt
98e8bd49e3e2748c52627c1e028051e59c163fc85178baedc3a82a680f81e574  runs/old_runs/toy_run_1.stdout.txt
61f9fb2310ca0cf2fb7d5e817db9e0b4fc29b8b8ecc7948f57bafbf040fa9968  runs/old_runs/toy_run_1/decisions.jsonl
8c7b82b6d0347a5ef45bf9084101428dcba33b9eb95c53ba2f9f6002818be92f  runs/old_runs/toy_run_1/logs/laya_en.main.stderr.txt
478825aad4a2933adedcca3dbce0a6a2f5992f9eed36f6a88e552f78d795ae47  runs/old_runs/toy_run_1/logs/laya_en.t2.stderr.txt
677dde8f45f0298927ee56293bb2c95f545d0038454bda9da767089d0edf9f0e  runs/old_runs/toy_run_1/logs/laya_en.t3.stderr.txt
0b6e53274cde8653d453722a57f3e5eda88bf167cac7f4bc7a065af1f5a943ff  runs/old_runs/toy_run_1/logs/laya_en.t4.stderr.txt
ae3b76705a98a6d93714f13ef78fd907f5769f3469febdb0e1775dfb40802476  runs/old_runs/toy_run_1/logs/von11.main.stderr.txt
a13ff651299efb5fa1455b71ce6b8ed6e8d750e39155ea45bdb0eb101cbc7f45  runs/old_runs/toy_run_1/logs/von11.t2.stderr.txt
e98959054266fda33be4b943364e50a37901c3913f7f9cf3112a83afea8bbced  runs/old_runs/toy_run_1/logs/von11.t3.stderr.txt
e2a352b65a9880a2b989dd4440aff82a309d3f0854113d76242e997c072b13e3  runs/old_runs/toy_run_1/logs/von11.t4.stderr.txt
6c30e5696c5e3c984fdb1956acb2fdb09dff8a6feb4ecd38c1a9804b8c65d537  runs/old_runs/toy_run_1/report.txt
5c0d4a0b23e9b56cd1bbc1be7184e0019a4e2c7ad31c76ec25894079bcef4c9c  runs/old_runs/toy_run_1/results.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/old_runs/toy_run_2.stderr.txt
9fe86d1b9faf07f88c55dfddb8c456d00f90b7d4cb46449feb8df0387e0d7b99  runs/old_runs/toy_run_2.stdout.txt
98c0f6219367d56f484261ba9ceb572115ec06d172ed7d8d844355eac2860b4e  runs/old_runs/toy_run_2/decisions.jsonl
6f6f9da57b9f5bf47b34284d18996b05f1b6826f1ea6affbdca7e29c0997b98e  runs/old_runs/toy_run_2/logs/laya_en.main.stderr.txt
301f3166de69153faddc11ea133de0b08ae84885d8c3d074f6c743ff1653531f  runs/old_runs/toy_run_2/logs/laya_en.t2.stderr.txt
03ba78710b8767f7100faf60ee61768fdbbf78d8e573eedd46080724b14c7403  runs/old_runs/toy_run_2/logs/laya_en.t3.stderr.txt
6cc9e6e601eaf08e1494c50fabe8b0a745f5717a3c90f6023964576bb763088c  runs/old_runs/toy_run_2/logs/laya_en.t4.stderr.txt
37059b28c16ba775af28dccad0d97c43c88f75c4dc2c252c0fb7c19f13d3551a  runs/old_runs/toy_run_2/logs/von11.main.stderr.txt
7aae466d9d8b8e65fd172ce8a23ed332550d671eac563625fe64991d39595d60  runs/old_runs/toy_run_2/logs/von11.t2.stderr.txt
d164c0570635579141e30c948dff180b7c3eff640bd57cdf85fc49b779e26a0c  runs/old_runs/toy_run_2/logs/von11.t3.stderr.txt
114430b025fd5e4535283717472161a70703fd11abc1a284fd4f05d79b57bc7a  runs/old_runs/toy_run_2/logs/von11.t4.stderr.txt
0147b349a0f011fa42dcd741f483d3c1738cf9529bc68e97512371443aac6833  runs/old_runs/toy_run_2/report.txt
9c1d5dc6c81870b3a5febc2b37779afa900ae0f74953e6e59564cf1dedbe6465  runs/old_runs/toy_run_2/results.json
684908fb5b1233b35ca1973393ab4c8a640886a4942118ee7e0217802e313230  runs/real_run/decisions.jsonl
e4acfc9dcd8063041f81b49d3f08df2ddb35f4e8921837536076fe906236cc52  runs/real_run/logs/laya_en.main.stderr.txt
05fe3e419ca0e316b3e6e908ecae0713c77d2a0a0c54f427c24a32d0193f197f  runs/real_run/logs/laya_en.t2.stderr.txt
3d4050692a5db5afe2c5f4001ba5f9ec3ecec1e97cb3a3fa87ceedfe6f541fc6  runs/real_run/logs/laya_en.t3.stderr.txt
2f3b916ecfce3ca44d56406419bc3e4a5895fdc88d2b19b29f43200f8119c4ce  runs/real_run/logs/laya_en.t4.stderr.txt
3599623d702d2d39ea232921b11eb9feca773980c106dc8aab42bfabd24b9599  runs/real_run/logs/laya_micro.main.stderr.txt
f3ce0e23444eba1803639e5d8c3f9b02c3779a65b3772ec4622d84a13e4f7474  runs/real_run/logs/laya_micro.t2.stderr.txt
6318742ae7f104ad50e5d47dd5523eb2376a19c6f6640db84bca61097eb9acdc  runs/real_run/logs/laya_micro.t3.stderr.txt
6f244de151c118cf7388e6273deb5f9223dcf28f39507ae730b7ccdcbff921b2  runs/real_run/logs/laya_micro.t4.stderr.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/real_run/logs/laya_multi.main.stderr.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/real_run/logs/laya_multi.t2.stderr.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/real_run/logs/laya_multi.t3.stderr.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/real_run/logs/laya_multi.t4.stderr.txt
fa83f933efb70f01e1fe3ecd155d9d6485b9a2eeefa672c6411d70fd41104aa7  runs/real_run/logs/s1o.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/real_run/logs/s1o.main.stderr.txt
2591e2d00422780e35ad4d65ed7cc2c7af0000f93dafc50289b17a760b64c0fb  runs/real_run/logs/s1o.t2.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/real_run/logs/s1o.t2.stderr.txt
8f156523a3130cf644890c77f2ff30a8baed614de0b82b65b6071d9a66198ba6  runs/real_run/logs/s1o.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/real_run/logs/s1o.t3.stderr.txt
e82751b6b064c56734acc18b42d0929425b1a6401782ca7b74b996575b91c88f  runs/real_run/logs/s1o.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/real_run/logs/s1o.t4.stderr.txt
46a5151afa9c00d8ad9ac24544a30a62cdcce1e1705e2c179c881a358f6e6348  runs/real_run/logs/von10_nli.main.stderr.txt
695a846656788afcfd05d12ccda42c8f71d15f54ab5c203527ca0bbd5ef53312  runs/real_run/logs/von10_nli.t2.stderr.txt
4a852531fe88f7389e48f75fe9d1dcd5ca239ee7ec5fb3acc8bdb4f7b1c40554  runs/real_run/logs/von10_nli.t3.stderr.txt
01e986a7e09d6391b7b7c2c3546ad49bcdae4306e5cf5c0fe0131e557fe6dd41  runs/real_run/logs/von10_nli.t4.stderr.txt
59366d8113a6b6b54e26a5830b81f28d76c9565b9aff631af5bbaf6984cc096f  runs/real_run/logs/von11.main.stderr.txt
0060f686698f5120f42b9e4db34157e51aba50d4a2d12921013fce2f4b0ac49b  runs/real_run/logs/von11.t2.stderr.txt
16c52106fb54c6eb6ecaa45f5c46b6e1b34573be1985b2d2ec08f9b32bf3028d  runs/real_run/logs/von11.t3.stderr.txt
4fe511a93d565282710299746a3bc39eb140e7df4a1a3c4e975a7202b5798049  runs/real_run/logs/von11.t4.stderr.txt
9e3eb02f65fffc81e2dde4630a3c83579bf0592bc4988dc03dc7c1c20382bcbf  runs/real_run/logs/von12.main.stderr.txt
869e1d2e804dad2f2663f7c5324d2bcf5748d7f19f3d57538b3061b0192e74e4  runs/real_run/logs/von12.t2.stderr.txt
25cb2db2d3ebc9df7bf87e387ef2ad665df4976af520bb726b8bb423e043ae67  runs/real_run/logs/von12.t3.stderr.txt
d985b9d86b7857dac723e2f6ec5714a16af3977ba98f88bc11762b694ebf0f3b  runs/real_run/logs/von12.t4.stderr.txt
5e9f1e25519f81a541bafbfe434fcb2a6c78a7bafecae3082fb7d63d5a30dd29  runs/real_run/report.txt
47676f3ed2b04f62cc40979566246c1e24b50580bfd93fb29c2a0d6e4f507124  runs/real_run/results.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T044457Z/decisions.jsonl
f0edf0b0f23c90aac5df87c90fb067d5d10b4e64014b2263cbded8c5fb75d89a  runs/s1o_speed_20260926T045648Z/blocks.jsonl
7fea145d3cf106d468c064a75abf747333689e851f003dde3964304bc5aa8b04  runs/s1o_speed_20260926T045648Z/decisions.jsonl
f76cfe04516b6e04f049093d2a747370758bee645d8f16cb02cd302e2ae27ca2  runs/s1o_speed_20260926T045648Z/decisions.jsonl.before_resume_20260926T065623Z
e9f01ec33d956d0c9f2a54a37cbb69670006b1e0256d81855875ec7ceb39ebc7  runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.main.stderr.txt
53f93157b0a22cf0143c15578216de0ad3a397204ae40d39e4d8eee02b6ca197  runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.t3.stderr.txt
ae13ae53a42fb82059cdf26efa530ee87458867369c4b3e674c15a670aee471f  runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b1609.t4.stderr.txt
9dc796547c5883872519b5b9b6ba5d12deb537ac5d7abc8ef81a550c47ac2836  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.main.stderr.txt
3755d5b7e78f575c0e189af55d1605d47126e491daf7bf86e017ce236647cce4  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.t3.stderr.txt
3d514937afab49492e3c7a1c4e47b513e00f9517dedc3f0d2b7306abc0d66c74  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351.t4.stderr.txt
441fc07062e6895fe3b3d5bcc315ff681735ececcd999c6bccffb1c919704023  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.main.stderr.txt
9a50fbd15dbfd359fce7b0aa79e1de6faa745b549189a6a38fde0b3e7225057a  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t3.stderr.txt
35257f535ef1fed69d6e8e67b75d807f35fcc602d4f126f86ce63c9f7653533c  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_q40.t4.stderr.txt
0f6f629000e405ac117c4ee4b23e2c5d27c6aa4ec0729dd894484120d4302e6b  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.main.stderr.txt
f12fb3fe0ff4994ad6531fcdc60094160c3595a0e41924313e9fc9e1fe99b07a  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t3.stderr.txt
c5d3cd1b2e70ca8836fc316d54d4499f6e71f24e981372fdfbc1a525fc4af873  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen08b.t4.stderr.txt
ff797f13d0dae8a55901db0f4d4ddd7b4cd83ab6e3a496ab982cb654b7a0c628  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.main.stderr.txt
698d14a70924959b606a1f3908147c8fa943eedb2b214e8a4a3806e9c4990073  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t3.stderr.txt
b8b7c58d4896bb5458c9f54402b4af70d34a889baba340d9d65f54198ae6afe4  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/s1o_speed_20260926T045648Z/logs/s1o_b2351_qwen2b.t4.stderr.txt
e026d33acbcb65d10f306953041d3db5aa1e6958a44072bb3b6b83aa23ed4bd6  runs/s1o_speed_20260926T045648Z/report.txt
6dd8c5934bfa8d3ea5949932f4bf98824118e2ada1e62cad744b460ffb388135  runs/s1o_speed_20260926T045648Z/results.json
0be6a9a034960a47cc8bbb33a64dda4eea7e00e3c7145090c3aac61c0ed4d2d5  runs/s1o_speed_20260926T045648Z/run.json
25d0ca7b2beee7bbf565d8a31d12c6566380aad05a4c4e7c2f25176f3456b751  runs/smoke_run/decisions.jsonl
9e1c93b77a6e25adbc5864da95fa28859c5006ba8b8009b0962ac4b935dcd1fb  runs/smoke_run/logs/laya_micro.main.stderr.txt
aec77d7adcdbdf9e225056296ce80e5d5d9799cb554fbf6c89373e84d15fc342  runs/smoke_run/logs/laya_micro.t2.stderr.txt
03fce3b9f7b0b417e5260c0f50a4a17459ee863cd60696226042f8d5e1b86955  runs/smoke_run/logs/laya_micro.t3.stderr.txt
20cf68e4ab451957f6884242cd8dd0788d358b548b2aac32768ce7ff125344cb  runs/smoke_run/logs/laya_micro.t4.stderr.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/smoke_run/logs/laya_multi.main.stderr.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/smoke_run/logs/laya_multi.t2.stderr.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/smoke_run/logs/laya_multi.t3.stderr.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/smoke_run/logs/laya_multi.t4.stderr.txt
39773be00e03b7121ebb30e606594e228625ec5b4b86bf6cf9dc2b08aab732b0  runs/smoke_run/logs/s1o.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/smoke_run/logs/s1o.main.stderr.txt
657551792dd27111bb0e4af0326dc0277ce6ae0bda62f37faee54500411e200f  runs/smoke_run/logs/s1o.t2.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/smoke_run/logs/s1o.t2.stderr.txt
3d11a98601b40d5b2aae258be352b34838d0955697b2ccc164404701ad75b989  runs/smoke_run/logs/s1o.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/smoke_run/logs/s1o.t3.stderr.txt
25eff2bf88a9036eeea0aed9b9b4e87ba5c38c9498c8282dbc87e099274efeab  runs/smoke_run/logs/s1o.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  runs/smoke_run/logs/s1o.t4.stderr.txt
fab7395d4890a74fb76ea2ebe155eab9bbf47868d0545bde3c0550349aad2eae  runs/smoke_run/logs/von10_nli.main.stderr.txt
3f92ad7856e1a1670a0ac062faad67745c3d374799388fa1c63ba1200b98838c  runs/smoke_run/logs/von10_nli.t2.stderr.txt
5edbcafe3fd3570fa45ad28196cb01ce59b7fbf760dd6057277d218d48f1f276  runs/smoke_run/logs/von10_nli.t3.stderr.txt
9236833e135aaacc0d3a98df282f1805461f7e25046a55e174a87d85db24074f  runs/smoke_run/logs/von10_nli.t4.stderr.txt
b71bd7d142d5d37ad7ea7e6933c61532198b9b90f4fa456cfad729e119e473ad  runs/smoke_run/logs/von12.main.stderr.txt
0b1590950e815ec520389cf2b2d1626f3a8c8dc16562d4697ff684f0bb8ac9af  runs/smoke_run/logs/von12.t2.stderr.txt
dd1ea51d611fc28b08729f6ee2578c836cf34f9022f9bbc3150be6c7fbef1e30  runs/smoke_run/logs/von12.t3.stderr.txt
dc71b11dcc34d67194aa4b81bf87c9b1920c6ea868c939ca686da922020bdc85  runs/smoke_run/logs/von12.t4.stderr.txt
fb2299d7dc181c9e77c6e4ef7ffb25df508fb1c66c263febd0f155cb610ad4ce  runs/smoke_run/report.txt
4ee0e3195c8d21a84c02bedd0e85e9cef085d57674814cdb0d0a8188b131e071  runs/smoke_run/results.json
19c13ce213dc5d9f24963368292964674b485a820c189dd470ebd2b574dc07f8  runs/toy_run/decisions.jsonl
67e8826ac4995b4c6a1b527e6ef055f55eb22c0690ffd345d0af769249b09245  runs/toy_run/logs/laya_en.main.stderr.txt
0d0d26f1dd5e010a7fda5467ae4ca33d1a1e82ca908b713518aba24261c8eb50  runs/toy_run/logs/laya_en.t2.stderr.txt
b1ef6b8e67248a93eb0fd3e27410710c07c418f51d80f9c1d75087974ee697cd  runs/toy_run/logs/laya_en.t3.stderr.txt
e0fc0c96966ad696235b1d53ffb8009d649deaca7f09feb932af73ed3eac615d  runs/toy_run/logs/laya_en.t4.stderr.txt
f3869b10dd3aa5bd19a113d5e521fa79f5f30ac4f9aab7715b2141d30b6ddf86  runs/toy_run/logs/von11.main.stderr.txt
ec4147bc747ae4446eb0b15f4e03c803e1e801c840bb9975b416468b704ad237  runs/toy_run/logs/von11.t2.stderr.txt
6897a4529630900b05fcb74cc6a44a123020fe3482d141d1617ec3fb77b3f898  runs/toy_run/logs/von11.t3.stderr.txt
6d8b53c93c744c55ea7f8d433ee2c9a2a39d7f9cabf88411e0d9dcb4d59118b5  runs/toy_run/logs/von11.t4.stderr.txt
ab9c8f5c344c5b6de8e78d56d5d2a820784d20c789f5a9f10633d97f19d45d16  runs/toy_run/report.txt
b0576b5b1de17924ffbad96a0cadbbc264a9bf19c58c3edcb815d67600cc17cb  runs/toy_run/results.json
4c51c25ace9bc67ec66e7b57749894476b3bc78bfdcd41ea40fcc304968f9d58  s1o_speed_prep/adapters.diff
1f2a5185d43c6f02a149c7b42ccd076088300b3cbdc8873cf73c770e2e2e7dd2  s1o_speed_prep/base2/ladder.py
9aa8cc2462197a5e99b79ba520dc8df447619624ddae00f3909f50d4ce456bf4  s1o_speed_prep/base2/test_ladder.py
7ab64a4e8cfccf0b87edbb685c8c0bb8d670d8278c10ee2ad5d7945b97aaf161  s1o_speed_prep/base3/ladder.py
07e6ee25dad15e09a12cc5e6e1fe3de06892bd1954de85eb45485ccd98b000b6  s1o_speed_prep/base3/run_s1o_speed.sh
7f030b30a7748091a7b06560a8243f82392811654603303b5984234c18555493  s1o_speed_prep/base3/test_ladder.py
00ba0bc6cc975547fe1a528a23be9dfad8622f012a9e602a28bf9c07a0e7aaea  s1o_speed_prep/e2e.sh
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/e2e_1.err
ca0b0c17f230d35001360d8cd0e567dac668943a8fdde202e0920749052d8c9f  s1o_speed_prep/e2e_1.out
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/e2e_2.err
72c787061e507876dbcef7fd5942ddfe7d1d0eae23f02c939ea2d4e2fcf91b9b  s1o_speed_prep/e2e_2.out
3c02f67255b34e6fbdd7cda72a876c2f41fdb8086145e04df8d26f0dd3eef007  s1o_speed_prep/e2e_run/blocks.jsonl
8e73239fda00c7ba0537adc7e5190a5e820f5e4f02a397866d364da348fcf719  s1o_speed_prep/e2e_run/blocks.jsonl.before_resume_20260926T062950Z
0276a83f28c376ef39428c9a017a3ddae9aa2e1e6d6f98246bcaae84b2ea777f  s1o_speed_prep/e2e_run/decisions.jsonl
3c00df3ca3c69e77f9998708c14d4c54c91ffe0704fe5bf3b1d46090b2212de2  s1o_speed_prep/e2e_run/decisions.jsonl.before_resume_20260926T062950Z
7a9f2a6ab89d22a0201f1ce4fe8a91f5bb4fe529f129d78e9a763c2ba707407d  s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen08b.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen08b.main.stderr.txt
e44a626603fe9803f6ee0e88bd00a9f17318b53e38905cce22a128022cd951b5  s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.main.stderr.txt
bbec9aa5b598def606824cfd6de70c475d90b9f13609cb8226c664c51150eb2a  s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.t3.stderr.txt
6adee5a11fedbeaf3f2e85242d31f94da2eedd39abc5f1c50ac0b10b5da9e17c  s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/e2e_run/logs/s1o_b2351_qwen2b.t4.stderr.txt
1058843760960bc415dd0b2480944058575651b0fe0664a1fd9c8206df8b5fa0  s1o_speed_prep/e2e_run/report.txt
acb39ea15b72c66e4188ca5b03b634fdfcbbe3a3640c365028813d49468aa576  s1o_speed_prep/e2e_run/results.json
ce7b267279bc1081dd6fae829f7f44b0656faca1ba6292f0b548021e085626a2  s1o_speed_prep/e2e_run/run.json
907fd538df73539f0313f3260f4f742af3f579d9511cd661e8867a452ecb6216  s1o_speed_prep/fake_thermal.log
5617e962ba2d29f2de69512d8f0f56ceae5d4c238670b25d0942e78f6e1fb484  s1o_speed_prep/fake_thermal.sh
b173d26e938da8ad4b8db50e7aa9281109942c9e7c9ac16fcf82d6985efcd2be  s1o_speed_prep/gguf.sha256
6968da002fc928636b37c7225bd4b9991363969de3ba36d3bd02fef0a795fbf8  s1o_speed_prep/l1.log
2a20be506dca492fa8c70bfa3a06122c1dc4f727e2e8da9abc59c6af29a76d82  s1o_speed_prep/l2.jsonl
c9bedd1ebcb8fa3e7e9cadd2f04b35d4c00752b4f520ba7022a950b9888d0ef4  s1o_speed_prep/l2.log
f5eef28c034c481e4d8e757db3589287593f78cf8f7b358f0e467fe7164138c4  s1o_speed_prep/l3.log
44139e57a71eef95e6c985a55d646ab36863e03198120c1efe5e92403868c52d  s1o_speed_prep/ladder.diff
541b56e8407ac463029db7c85cd120dbcaa5e2c45fa0e09025d80eefa7e3d28e  s1o_speed_prep/lens.sh
e1fdc796a9b0bfa042f9d13fd97879f28fdcb0489b3e1d3250e05cbe749fe847  s1o_speed_prep/live.sha
23e90a3d51daea89f3689aaadb3f4a56a43f23f26dd76785257f15c653e099bc  s1o_speed_prep/orig/adapters.py
00fbe2fab81021cb884f54550042b21be9f3b9e61943efea6d63f79139bb5f8b  s1o_speed_prep/orig/ladder.py
daf7402ab0365725e66d3dd7da13f2c6925ebca3b422f806e8e7026c78101e8a  s1o_speed_prep/orig/test_ladder.py
871a7d0755baa0c2615e77d92dd7f14f096c1c917f793f0f760fbd7451d63b40  s1o_speed_prep/partial.py
e5736d7fe1e40d853b59cbc4ee050c560dd1332593324db3025068bfd43055b1  s1o_speed_prep/partial_report.txt
3964e1eb8d9fe204e91f5d1a7681c20fe3cf2eefa4e886bf125ab487610a1ebb  s1o_speed_prep/probe.py
71247c36f50e1b1ac5df7db78362de6ad1faa2a365a678257d195d1ad9ddecb8  s1o_speed_prep/quick.py
7ffe6a56eaa9632e0135b45be512c8cbc21662924d108a923142b51679a1f4cd  s1o_speed_prep/quick2.py
0ea78770cfb86dc2defb8a763acb8078667a4b52c181ec86e42f4561c8876658  s1o_speed_prep/quick5.jsonl
5a466d305f4b04f15eb2f5ef81608c266cbd06cba8bb97d7faa2c29cdae4b209  s1o_speed_prep/quick_run1.stderr.txt
8d53c40bb233c05a64bafc6001c31565a354a6ce9e66a00de9b76808f57a6951  s1o_speed_prep/quick_run1.stdout.txt
ae7e982f31307c166d403d1f3768fe0d1cea90ba0336282fe2d29486a8f3d2f7  s1o_speed_prep/quick_run1/decisions.jsonl
e2e5d4dcbab9385bd1f1a20c52458baa66a1b0e031447b8defe6381e16320464  s1o_speed_prep/quick_run1/logs/s1o_b2351.main.server.log
8dd33f68ee16b34c397ed176cbbd05b611d55f84c8d482443f659d74d83356b0  s1o_speed_prep/quick_run1/logs/s1o_b2351.main.stderr.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2.stderr.txt
2ad28d84ea4a921726ecd497fd978249eff5d9cc2801bb9a276129f5b6e7059a  s1o_speed_prep/quick_run2.stdout.txt
58c5dd20206b2c6a70bfc0ee5bbb0ab88feba557a14e77b31d2036a19bf72448  s1o_speed_prep/quick_run2/decisions.jsonl
13b730f5f91dddd2bb0aec6234fda7328483162b347566863c552d28233fd26f  s1o_speed_prep/quick_run2/logs/s1o_b1609.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b1609.main.stderr.txt
aee08e26d56417098d8ba3b41ef001ddde969e163bb350e62cf88d247ade6732  s1o_speed_prep/quick_run2/logs/s1o_b1609.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b1609.t3.stderr.txt
3af59280cc48ec206e1ffc41230f32885479fd11758fee0aa915c83520872a73  s1o_speed_prep/quick_run2/logs/s1o_b1609.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b1609.t4.stderr.txt
72b1989fd08567b9768eb1dc5fc7678f09c7cf573903c255aad95f736a867e98  s1o_speed_prep/quick_run2/logs/s1o_b2351.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b2351.main.stderr.txt
07ba4dd0422ef313565a0f07817cc3972a8619cbf82de8bfbe496008dbaadbd3  s1o_speed_prep/quick_run2/logs/s1o_b2351.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b2351.t3.stderr.txt
86f6a78208649120d22b037d50e46ab0ff72f6f4cbb36671e6eacd46fa8aa5ba  s1o_speed_prep/quick_run2/logs/s1o_b2351.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b2351.t4.stderr.txt
eebebf4131f4a3d73a438ee8c37c8d5e8ceb985363870eebf6465653e12f80dd  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.main.stderr.txt
d86b5fc49a04be8d97110508bcb14fffb5811275777cc89d1234b57e22d3e66e  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.t3.stderr.txt
fafc2f8ef12d4a4c7763a227acba95b5633a08f642a8b48d3706de63f93b9c7f  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen08b.t4.stderr.txt
100aa21cc97a157b49dd5ee7244bce78ee15c6296c6d0b5bb0c7a57adb7c0f22  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.main.stderr.txt
50a25b185271e855343281ea1c521acc206a069113b81617a2ab03f3bb2cd67f  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.t3.stderr.txt
c217f9fa8f9d3974a7280386a6fe8afde6f6caafdb35889b65d237655677267d  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run2/logs/s1o_b2351_qwen2b.t4.stderr.txt
207c7f4e14a57640f38a8c00ade83651952f75b5b02136cc49448e4aba43766f  s1o_speed_prep/quick_run2/report.txt
baab3271999ba0a7ca47bcf3893a295c65d5675247ad5f1ce4155e41b4893026  s1o_speed_prep/quick_run2/results.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run3.stderr.txt
f9ce4d021afdd54a23334cd615c5629be55448257603c910d84f0ee3308ae906  s1o_speed_prep/quick_run3.stdout.txt
e406522cd093bbfc30da359e1c8de37c914fb8b8da54bbb4359c6b3abf07c5ce  s1o_speed_prep/quick_run3/decisions.jsonl
d307878b45546ee6313d29416ed8df9b9199af1dc17bfbd100c62e3832c51f4b  s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.main.stderr.txt
1875c132063e8fdf651b4b556e0607da0d3c850c36a1a84967fc33cc8d9db7b6  s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.t3.stderr.txt
1813d7e700174773987a44b6c9479c8927a432ae91a4fe43807bb4b570a101a5  s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run3/logs/s1o_b2351_q40.t4.stderr.txt
90e0cb1ac7fe493a211cb7aead7cf308cea66900a78e387fc4de165567701803  s1o_speed_prep/quick_run3/report.txt
90ba16a809b69b51aed0920c3142178189998175d77b10dfed6b08e49378f5ef  s1o_speed_prep/quick_run3/results.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run4.stderr.txt
f66fa73fbf27a6fb5769137a15491aa82aa20bc178684afead62068af60d7141  s1o_speed_prep/quick_run4.stdout.txt
24c662f9e1d33532c93c2c3d4886e93e032329e0ed1702d9db6814f37a3af98a  s1o_speed_prep/quick_run4/decisions.jsonl
2f57829de36a5e248634e24212366ba4afb61f3ceb7bdaa7b2e8dc56d1d2b4f2  s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.main.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.main.stderr.txt
2017266f51812592cdec9835b6d4ae5041b55fe77f600918ffbc37be98c8c056  s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.t3.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.t3.stderr.txt
e8072567ba0dff8f948a172d958b26e03aa3d32214f6b39286c9035119f42a10  s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.t4.server.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/quick_run4/logs/s1o_b2351_qwen08b.t4.stderr.txt
3e4ff043ffb28b9d0db697f010210d48c379d451ca578298da67e3fbbfeb4070  s1o_speed_prep/quick_run4/report.txt
4e4960ec14344cbbc3ddd32423e1c26bee75228e4d9cb5688873061be453fd7e  s1o_speed_prep/quick_run4/results.json
41124fcea6995d74f2b9070bc86a4c16761a61111f0345af9c301e017a9da27b  s1o_speed_prep/r.log
9ef49c0d90dc4bdfdcd863e0a8996158784f5fc451bd38053e0637f8606df1c6  s1o_speed_prep/repro.py
0fe5f489aba3d197cf7fe6d8bf049927ba7a2ee41e3e1695c26f5c43b42d5b30  s1o_speed_prep/repro.server.log
6f9b232ca1f3b43aa72e2ba63698c09b0796b612d5a17cf05efee31e6c79b713  s1o_speed_prep/review.stderr.txt
5e58a86726cb1725579889b2787fc69c77bb5fd18dae7f2cfa708e65dde27048  s1o_speed_prep/review.stdout.json
58abf0528dbb92ff0adac1874fcac567ae38441c86a9f3c776a7a16adfcbd976  s1o_speed_prep/review/INVENTORY.md
6769aa1142694bd216cd9171e64ca913d3551f45bf8db7ec5497735d97e44143  s1o_speed_prep/review/REVIEW_REQUEST.md
4c51c25ace9bc67ec66e7b57749894476b3bc78bfdcd41ea40fcc304968f9d58  s1o_speed_prep/review/adapters.diff
0b296d3d68cf172436816154ad2aa5f7efb4af1c9f01f72f13f84ec4b9a3ae79  s1o_speed_prep/review/candidate.sha256
b452ab09d3a9772b43771e1d687c0bc013056a09c45aa4a464c07ac2dd0e3fec  s1o_speed_prep/review/candidate/adapters.py
1f2a5185d43c6f02a149c7b42ccd076088300b3cbdc8873cf73c770e2e2e7dd2  s1o_speed_prep/review/candidate/ladder.py
f795b354d6b615a681b25066d79b91e58ad8c1f68ae16b516a364bff8473b07e  s1o_speed_prep/review/candidate/ladder_worker.py
65505ae626f17b3f03628345f15d80fd870c427b316be9696b1a3d726a2d6c41  s1o_speed_prep/review/candidate/run_s1o_speed.sh
9aa8cc2462197a5e99b79ba520dc8df447619624ddae00f3909f50d4ce456bf4  s1o_speed_prep/review/candidate/test_ladder.py
44139e57a71eef95e6c985a55d646ab36863e03198120c1efe5e92403868c52d  s1o_speed_prep/review/ladder.diff
23e90a3d51daea89f3689aaadb3f4a56a43f23f26dd76785257f15c653e099bc  s1o_speed_prep/review/orig/adapters.py
00fbe2fab81021cb884f54550042b21be9f3b9e61943efea6d63f79139bb5f8b  s1o_speed_prep/review/orig/ladder.py
daf7402ab0365725e66d3dd7da13f2c6925ebca3b422f806e8e7026c78101e8a  s1o_speed_prep/review/orig/test_ladder.py
71247c36f50e1b1ac5df7db78362de6ad1faa2a365a678257d195d1ad9ddecb8  s1o_speed_prep/review/quick.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/review/quick_run2.stderr.txt
81143c86a01ab1ffa6b8cf17f1b20e439207bcceb2242e27db71b7fe4c0c6806  s1o_speed_prep/review/quick_run2.stdout.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/review/quick_run3.stderr.txt
a18e6bcb3cbaf34e217dbdecb53a646e4aea3381fa41c0521786f6ca53e880f7  s1o_speed_prep/review/quick_run3.stdout.txt
2a3d448248d7a8e6732812f3e4514a492a69424049547e876d2f70de3f8943b9  s1o_speed_prep/review/test_ladder.diff
49651752fc7ad70385aaf532ff5c93f1eb23bb26778effb860e7bb5c9b0ec7c2  s1o_speed_prep/review/test_output.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/review2.stderr.txt
3f1f41090b9d2a4f0311541ea37b934b10b4d5d9828884876a6cbb72f7e58ee3  s1o_speed_prep/review2.stdout.json
4df579619f582ea4ce93d7304548d3049c66bbfb4b1d1a2b547075d9efaef08c  s1o_speed_prep/review3.stderr.txt
b2389dd937ca95711b8f41f21de23614735c202dae9503214053267c685a1c2f  s1o_speed_prep/review3.stdout.json
30d0191d4008e722f6b73f6d337c8c25e2d1ebec832bee02ad73c5363846929b  s1o_speed_prep/review3/REQUEST.md
945fedf0edc621630b734b72880a354b2c0800e80daa45355136c8786d6052ef  s1o_speed_prep/review3/candidate.sha256
2e1d6dc3641ba6c535d61bf1b6f300d4696b5875ba8ac9dd4edc8d2b4b7da13b  s1o_speed_prep/review3/ladder.diff
edf4c75f6a0879953bf022c3263553854215ae6bc02f9e3feec60df171c8a703  s1o_speed_prep/review3/prompt.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/review3/quick_run4.stderr.txt
6501a0f4f825e8c376921eece4132c1da2340a20e8b7b6a7db1639aa6a0952ba  s1o_speed_prep/review3/quick_run4.stdout.txt
6d3535f4ae4d59a760054011c4ac673307b3628dd4c7f37d81bdb91227af39dd  s1o_speed_prep/review3/run_s1o_speed.diff
158bf15c9ac90f558e356acf6e91ba02266a568f20ae20586a2e8fe0d57e487b  s1o_speed_prep/review3/test_ladder.diff
f2e9b6001f9293bb5225b64a7d0a43ace8970255b4e95c7818934103553301fb  s1o_speed_prep/review3/test_output.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/review3b.stderr.txt
209f9e756b88243af9e1bf81cecd29d8420a9586ee4cd0795529f38737494aef  s1o_speed_prep/review3b.stdout.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o_speed_prep/review4.stderr.txt
1620b8f0161c7c7187c6ed22a731ba70cb739cb5da6f0364f7f55b41d80f4ae7  s1o_speed_prep/review4.stdout.json
b3e55e83a2e470864e890985f09686e00f09fff556a7d0cd5aa32584e1a6677c  s1o_speed_prep/review4/REQUEST.md
c6c996df213b0217faef9061456e8fd320d3acb02681aedcd59f51daba1327cc  s1o_speed_prep/review4/candidate.sha256
04df6a0a026ccc3e0ab673f25cb3f0819615f1ebc49a53b89d84cab67f43d310  s1o_speed_prep/review4/ladder.diff
c3041089eb2c2213a84f93ef41441866d8bb24d231b911c1b41df358c3280297  s1o_speed_prep/review4/prompt.txt
84327c6eaee8b9c9e7ae8a34c427c6fc26f4d10f67bbd1691d68ffbc270461c6  s1o_speed_prep/review4/run_s1o_speed.diff
15d3f445ba7e255838fa22d3a25e7732b2af3a9b0a68d0b32ad5ad25984ac01f  s1o_speed_prep/review4/test_ladder.diff
20c1270984aad8b618a909bbdb6d056ac2d49450586cc4ade4910d2b9fcb48f9  s1o_speed_prep/review4/test_output.txt
227f2a00ae6f9bc50ce8f098274e8a1d3dac3aa9834c4273a5fa8fdc7d39221b  s1o_speed_prep/review_prompt.txt
2a3d448248d7a8e6732812f3e4514a492a69424049547e876d2f70de3f8943b9  s1o_speed_prep/test_ladder.diff
2fe34c55b0f2c10e44313df0aee7591ea51ebfd87c4c0fecfc692ad5db5628dc  s1o_speed_prep/v.log
19da97c0fb8726d06d454f1d1044e9577c0fc10ef30bd774c80d60c5f208170f  test_ladder.py
2ebe1f9fd4eac5d9329d6fe8af2b2bb7dbccf170239c687191ed6ae9e5ebb7bc  toy_cases.jsonl
```
