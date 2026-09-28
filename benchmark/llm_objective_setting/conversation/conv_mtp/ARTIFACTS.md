# Artifacts

Every archived file below with its SHA-256. Copied files were compared byte for byte (`cmp`) with the phone original at archive time, and the phone and archive SHA-256 values match. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Copied from the phone (50 files)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `conv_mtp_20260928T081906Z.stderr.txt` | `/termux-home/ladder/conv_mtp_20260928T081906Z.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `conv_mtp_20260928T081906Z.stdout.txt` | `/termux-home/ladder/conv_mtp_20260928T081906Z.stdout.txt` | 8363 | `3893faccf271f667a90e352443c3cc68fa119e0a361dc8dcc1eafd34ca40a4b0` |
| `conv_mtp_20260928T081906Z/blocks.jsonl` | `/termux-home/ladder/conv_mtp_20260928T081906Z/blocks.jsonl` | 2210 | `7e209077bf430bbf71c46b6b92cb9ca0b01fb785f396984db2efa5b1cee873ac` |
| `conv_mtp_20260928T081906Z/c_results.json` | `/termux-home/ladder/conv_mtp_20260928T081906Z/c_results.json` | 16510 | `bad315804110075da55a6c9f53fecba60a3b319bb825e0096f6b380bba809b07` |
| `conv_mtp_20260928T081906Z/c_scores.txt` | `/termux-home/ladder/conv_mtp_20260928T081906Z/c_scores.txt` | 2203 | `edc66e01d5089a148a94dd3a377ba6354ba520bde3d014587f4027af1ec2d0d6` |
| `conv_mtp_20260928T081906Z/logs/gemma_e2b_q40.cold.server.log` | `/termux-home/ladder/conv_mtp_20260928T081906Z/logs/gemma_e2b_q40.cold.server.log` | 11673 | `7e520cbf107993f444551464331707ec5964e82b4a4432bdb1664f8407b10df7` |
| `conv_mtp_20260928T081906Z/logs/gemma_e2b_q40_mtp.cold.server.log` | `/termux-home/ladder/conv_mtp_20260928T081906Z/logs/gemma_e2b_q40_mtp.cold.server.log` | 14782 | `755d6387f23edda93bc2b366ae2a708b2ba436c94749491f1b76247c73d5a4c7` |
| `conv_mtp_20260928T081906Z/logs/qwen35_4b_q40mtp.cold.server.log` | `/termux-home/ladder/conv_mtp_20260928T081906Z/logs/qwen35_4b_q40mtp.cold.server.log` | 12068 | `8df64a7273a42abd46b356cfc1f537e4b54a784d511ffe96d9b3057b7a6754a7` |
| `conv_mtp_20260928T081906Z/logs/qwen35_4b_q40mtp_on.cold.server.log` | `/termux-home/ladder/conv_mtp_20260928T081906Z/logs/qwen35_4b_q40mtp_on.cold.server.log` | 14060 | `ce48edc9298dccc71db5c3dc117ef3d8d7351638d4dc1af62e57f6fbe5787616` |
| `conv_mtp_20260928T081906Z/logs/qwen35_4b_q4km.cold.server.log` | `/termux-home/ladder/conv_mtp_20260928T081906Z/logs/qwen35_4b_q4km.cold.server.log` | 12207 | `94e455c50c9e9bf29a26c019cb7b57c1a8d5758275010b3d66b8b9187cd87b54` |
| `conv_mtp_20260928T081906Z/report.txt` | `/termux-home/ladder/conv_mtp_20260928T081906Z/report.txt` | 1221 | `cc1dc6d1a4f43ccfc386763001f98991307e4e588f0e4b07f87f82ccdc5390d5` |
| `conv_mtp_20260928T081906Z/run_20260928T081907Z.json` | `/termux-home/ladder/conv_mtp_20260928T081906Z/run_20260928T081907Z.json` | 2933 | `ed64eb593bdb7a871bb296703d980ff3036522abc1b699e600f3debc2fca69dd` |
| `conv_mtp_20260928T081906Z/turns.jsonl` | `/termux-home/ladder/conv_mtp_20260928T081906Z/turns.jsonl` | 44275 | `c3db9e3bf19a926e3f95e785245af92f95a906ac0725d58f6ebdf9946ff0cc93` |
| `oneshot_console_20260928T081359Z.log` | `/termux-home/ladder/oneshot_console_20260928T081359Z.log` | 11131 | `ab3b716bdf73a9babfe598ba1801eb9bc5dd07e589a73c00e617b77b5ecb93c6` |
| `reviews/review1/candidate.diff` | `/termux-home/ladder/conv_mtp_prep/reviews/review1/candidate.diff` | 13287 | `a38f7193d49271692e96860fdd7e6742e1216b5106afec780e4f31391634553b` |
| `reviews/review1/final.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review1/final.md` | 1561 | `0c68aef5017377210475b15d1d9c3c331a500ec774bd9816d2feb95d64308a45` |
| `reviews/review1/frozen.sha` | `/termux-home/ladder/conv_mtp_prep/reviews/review1/frozen.sha` | 256 | `5b3e8e997eeee5502eec023ccb7c5aba09ec5768d710867553fdb50ee202ad5c` |
| `reviews/review1/request.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review1/request.md` | 20333 | `25ecc190ffbc068c28b2db8609a7ba9f40eb03ff478f05786c43a247c34f5628` |
| `reviews/review1/stderr.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review1/stderr.txt` | 140241 | `fb94d2a229b4a8198aa66018661a82a987b69874ea91add8e33b2b78c3ad7a24` |
| `reviews/review1/stdout.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review1/stdout.txt` | 1562 | `d98392a23282a17e4889ee851009b1676cf581dc0191fd8c1c333649aa4ada4b` |
| `reviews/review2/candidate.diff` | `/termux-home/ladder/conv_mtp_prep/reviews/review2/candidate.diff` | 14147 | `3c7afe03a756f9ead3bc59a1d906a639d5fdbc0ef7ed8578ca0c2fd73055990c` |
| `reviews/review2/delta.diff` | `/termux-home/ladder/conv_mtp_prep/reviews/review2/delta.diff` | 1459 | `75fff6ecb27cc31699c38d482e7824047b4693f3bb19d796e381cad805679a4c` |
| `reviews/review2/final.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review2/final.md` | 926 | `201fad1e4549acff22db2d2706567f071e5cc643be97472764870a6a60455018` |
| `reviews/review2/frozen.sha` | `/termux-home/ladder/conv_mtp_prep/reviews/review2/frozen.sha` | 256 | `38bb7c693df9a3d3e81b0b58f38222b5feb105018729ea8b626045beedad1fa3` |
| `reviews/review2/request.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review2/request.md` | 25405 | `63a5a82eb3b198135bb08c8dcf85dadf562925a65ed9f772fc0342c2912f64c0` |
| `reviews/review2/stderr.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review2/stderr.txt` | 190061 | `fa74f8006ff5fb41e29fe73840c0a2522224aac82169391253e13a0e9c526854` |
| `reviews/review2/stdout.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review2/stdout.txt` | 927 | `17daa6c272c22b135f43ffa44445b72ec2e1991b6c6132ba324675e2044acb95` |
| `reviews/review3/candidate.diff` | `/termux-home/ladder/conv_mtp_prep/reviews/review3/candidate.diff` | 15363 | `7697223a7537787ad0df82b491143387d60578b41bb68c701ba9cf8f5ffd0e64` |
| `reviews/review3/checks.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review3/checks.txt` | 1438 | `c83d4f4f330bfa2717d3b7d48e354e35998efc686f664cee72cfe4d26c11f2bc` |
| `reviews/review3/delta.diff` | `/termux-home/ladder/conv_mtp_prep/reviews/review3/delta.diff` | 2450 | `358c03d838cc1807b794470524761d03e32b7004bb44e2292d7f00e4d8abb625` |
| `reviews/review3/final.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review3/final.md` | 810 | `b9d8b796e51f7eed8bac181758ff843fcb486fe607599c51d3b96062cbc02908` |
| `reviews/review3/frozen.sha` | `/termux-home/ladder/conv_mtp_prep/reviews/review3/frozen.sha` | 256 | `1d79e99b53b74066a3d86a129909b4b5e2d0da8ff71cc3da9f57223c8a9c70fe` |
| `reviews/review3/request.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review3/request.md` | 29241 | `738fe90717ff74751b8f6f1b01ea0600f3c7443d954673729c003a92d6679c76` |
| `reviews/review3/stderr.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review3/stderr.txt` | 149013 | `2042f681282046b16981562c5ebbc115bf410467ca4eb0c88986db1838e5ebdb` |
| `reviews/review3/stdout.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review3/stdout.txt` | 811 | `fab6d18ae52e36b0a033f0c4615114c2afdcb8c707d5daf06c21e9978d8b8862` |
| `reviews/review4/candidate.diff` | `/termux-home/ladder/conv_mtp_prep/reviews/review4/candidate.diff` | 15873 | `069ceb922a159e718d1249c9ef040eec7572bee71fd801cf87d8693678d98f08` |
| `reviews/review4/delta.diff` | `/termux-home/ladder/conv_mtp_prep/reviews/review4/delta.diff` | 1788 | `95c07f9365dd0f0167c1ed8f3bf03aeb10b63d1e8b5ab5467f3d7389aedbeebd` |
| `reviews/review4/final.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review4/final.md` | 868 | `5f34cee0618fa6df1ce1be79f5fd2d24ff76d0dd79338d5d69bf414127a5c5a4` |
| `reviews/review4/frozen.sha` | `/termux-home/ladder/conv_mtp_prep/reviews/review4/frozen.sha` | 256 | `f159756abdd47b7fa24dcda9a4eec4bbb83f40b8e14f223d67373163b32abb1c` |
| `reviews/review4/request.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review4/request.md` | 30653 | `9fba1328033b3398a3077af67be3272ba88350d2c96ef0a3cc72ba6f0f2b0f1a` |
| `reviews/review4/stderr.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review4/stderr.txt` | 192215 | `a5bd9dd60ddc610afee02e590b07d7d9499fd3944ddf65bf86d5f02c11762954` |
| `reviews/review4/stdout.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review4/stdout.txt` | 869 | `2b8c024920b48794b82cd479ab8428e253561f95b1f18713174a6889e1d2ac66` |
| `reviews/review5/candidate.diff` | `/termux-home/ladder/conv_mtp_prep/reviews/review5/candidate.diff` | 16160 | `d35883a1538bf8e11af67d687fdf9a9290eab0c0018fbe246f95f79a0bc432e4` |
| `reviews/review5/checks.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review5/checks.txt` | 923 | `5411594ea597961785138c205411feff384a2899c720f5d434070b91bda2b4d7` |
| `reviews/review5/delta.diff` | `/termux-home/ladder/conv_mtp_prep/reviews/review5/delta.diff` | 1931 | `bc0fce8c3997fcda99062b07f1a550320d53606eb5bf9a3206b19612cf5b0724` |
| `reviews/review5/final.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review5/final.md` | 341 | `432b12c6f8e16b93e477ba6e48e404799c72c02a0fd2757c5dbbb8dab15e2f66` |
| `reviews/review5/frozen.sha` | `/termux-home/ladder/conv_mtp_prep/reviews/review5/frozen.sha` | 256 | `644cb6d95757a5ed1baddfe7bf23b87edf2d814ae27e29b991b88be60cf3f0b0` |
| `reviews/review5/request.md` | `/termux-home/ladder/conv_mtp_prep/reviews/review5/request.md` | 32339 | `b8191c4b2427d9927d359fd3d26653b8eba4714284f759bb585362de88e4bce3` |
| `reviews/review5/stderr.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review5/stderr.txt` | 164909 | `9fb8f663740eb52cf7645dfe5bc029c7b577f7e47aa81a8e3a17e4ef0914a44e` |
| `reviews/review5/stdout.txt` | `/termux-home/ladder/conv_mtp_prep/reviews/review5/stdout.txt` | 342 | `8fb42edaa9b07ccf19f6272b16755219d3b74f40cf8da028cbb00b2e9a36507b` |

## Written for this archive (2 files)

`ARTIFACTS.md` itself is not listed.

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 3815 | `ad14e65db0cca0e22d58f0f821007d8ff5c39599654128bd5322121955cd411a` |
| `RUN_INDEX.md` | 767 | `2d1022e42669b9ccf84b79ae5d680e51a7c972f5bf20db5e94b5d50ea6022b4c` |

## Executed scripts, archived one level up (same hashes as `reviews/review5/frozen.sha`)

| File (phone path) | Bytes | SHA-256 |
|---|---:|---|
| `../conv_speed_mtp.py` | 13647 | `e8fbbdb6b33f0e3190a05699397263a7632abf14a0cf1be30ff7f0628a5f2630` |
| `../run_conv_mtp.sh` | 1669 | `709a132de76bd0932d1212d61edeefeba1cbb512cb26a48e86acb23fe762a52d` |

## Not archived: model files (hashed at archive time)

| File (phone path) | Bytes | SHA-256 |
|---|---:|---|
| `/data/data/com.termux/files/home/models/gemma-4-E2B-it-Q4_0.gguf` | 2841481184 | `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52` |
| `/data/data/com.termux/files/home/models/mtp/mtp-gemma-4-E2B-it-Q8_0.gguf` | 97817664 | `9eba819938efccfd6044f8af84e3bbfddc639a2bcf32ebc36420e6a649191919` |
| `/data/data/com.termux/files/home/models/qwen35/Qwen3.5-4B-Q4_K_M.gguf` | 2740937888 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| `/data/data/com.termux/files/home/models/qwen35/mtp/Qwen3.5-4B-Q4_0-MTP.gguf` | 2669209920 | `14e6ef39302330c63c2c1a1ab548c7f6f1b7e36b3150ca8b42cab7193b0c3669` |

## sha256sum format

```sha256sums
ad14e65db0cca0e22d58f0f821007d8ff5c39599654128bd5322121955cd411a  README.md
2d1022e42669b9ccf84b79ae5d680e51a7c972f5bf20db5e94b5d50ea6022b4c  RUN_INDEX.md
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  conv_mtp_20260928T081906Z.stderr.txt
3893faccf271f667a90e352443c3cc68fa119e0a361dc8dcc1eafd34ca40a4b0  conv_mtp_20260928T081906Z.stdout.txt
7e209077bf430bbf71c46b6b92cb9ca0b01fb785f396984db2efa5b1cee873ac  conv_mtp_20260928T081906Z/blocks.jsonl
bad315804110075da55a6c9f53fecba60a3b319bb825e0096f6b380bba809b07  conv_mtp_20260928T081906Z/c_results.json
edc66e01d5089a148a94dd3a377ba6354ba520bde3d014587f4027af1ec2d0d6  conv_mtp_20260928T081906Z/c_scores.txt
7e520cbf107993f444551464331707ec5964e82b4a4432bdb1664f8407b10df7  conv_mtp_20260928T081906Z/logs/gemma_e2b_q40.cold.server.log
755d6387f23edda93bc2b366ae2a708b2ba436c94749491f1b76247c73d5a4c7  conv_mtp_20260928T081906Z/logs/gemma_e2b_q40_mtp.cold.server.log
8df64a7273a42abd46b356cfc1f537e4b54a784d511ffe96d9b3057b7a6754a7  conv_mtp_20260928T081906Z/logs/qwen35_4b_q40mtp.cold.server.log
ce48edc9298dccc71db5c3dc117ef3d8d7351638d4dc1af62e57f6fbe5787616  conv_mtp_20260928T081906Z/logs/qwen35_4b_q40mtp_on.cold.server.log
94e455c50c9e9bf29a26c019cb7b57c1a8d5758275010b3d66b8b9187cd87b54  conv_mtp_20260928T081906Z/logs/qwen35_4b_q4km.cold.server.log
cc1dc6d1a4f43ccfc386763001f98991307e4e588f0e4b07f87f82ccdc5390d5  conv_mtp_20260928T081906Z/report.txt
ed64eb593bdb7a871bb296703d980ff3036522abc1b699e600f3debc2fca69dd  conv_mtp_20260928T081906Z/run_20260928T081907Z.json
c3db9e3bf19a926e3f95e785245af92f95a906ac0725d58f6ebdf9946ff0cc93  conv_mtp_20260928T081906Z/turns.jsonl
ab3b716bdf73a9babfe598ba1801eb9bc5dd07e589a73c00e617b77b5ecb93c6  oneshot_console_20260928T081359Z.log
a38f7193d49271692e96860fdd7e6742e1216b5106afec780e4f31391634553b  reviews/review1/candidate.diff
0c68aef5017377210475b15d1d9c3c331a500ec774bd9816d2feb95d64308a45  reviews/review1/final.md
5b3e8e997eeee5502eec023ccb7c5aba09ec5768d710867553fdb50ee202ad5c  reviews/review1/frozen.sha
25ecc190ffbc068c28b2db8609a7ba9f40eb03ff478f05786c43a247c34f5628  reviews/review1/request.md
fb94d2a229b4a8198aa66018661a82a987b69874ea91add8e33b2b78c3ad7a24  reviews/review1/stderr.txt
d98392a23282a17e4889ee851009b1676cf581dc0191fd8c1c333649aa4ada4b  reviews/review1/stdout.txt
3c7afe03a756f9ead3bc59a1d906a639d5fdbc0ef7ed8578ca0c2fd73055990c  reviews/review2/candidate.diff
75fff6ecb27cc31699c38d482e7824047b4693f3bb19d796e381cad805679a4c  reviews/review2/delta.diff
201fad1e4549acff22db2d2706567f071e5cc643be97472764870a6a60455018  reviews/review2/final.md
38bb7c693df9a3d3e81b0b58f38222b5feb105018729ea8b626045beedad1fa3  reviews/review2/frozen.sha
63a5a82eb3b198135bb08c8dcf85dadf562925a65ed9f772fc0342c2912f64c0  reviews/review2/request.md
fa74f8006ff5fb41e29fe73840c0a2522224aac82169391253e13a0e9c526854  reviews/review2/stderr.txt
17daa6c272c22b135f43ffa44445b72ec2e1991b6c6132ba324675e2044acb95  reviews/review2/stdout.txt
7697223a7537787ad0df82b491143387d60578b41bb68c701ba9cf8f5ffd0e64  reviews/review3/candidate.diff
c83d4f4f330bfa2717d3b7d48e354e35998efc686f664cee72cfe4d26c11f2bc  reviews/review3/checks.txt
358c03d838cc1807b794470524761d03e32b7004bb44e2292d7f00e4d8abb625  reviews/review3/delta.diff
b9d8b796e51f7eed8bac181758ff843fcb486fe607599c51d3b96062cbc02908  reviews/review3/final.md
1d79e99b53b74066a3d86a129909b4b5e2d0da8ff71cc3da9f57223c8a9c70fe  reviews/review3/frozen.sha
738fe90717ff74751b8f6f1b01ea0600f3c7443d954673729c003a92d6679c76  reviews/review3/request.md
2042f681282046b16981562c5ebbc115bf410467ca4eb0c88986db1838e5ebdb  reviews/review3/stderr.txt
fab6d18ae52e36b0a033f0c4615114c2afdcb8c707d5daf06c21e9978d8b8862  reviews/review3/stdout.txt
069ceb922a159e718d1249c9ef040eec7572bee71fd801cf87d8693678d98f08  reviews/review4/candidate.diff
95c07f9365dd0f0167c1ed8f3bf03aeb10b63d1e8b5ab5467f3d7389aedbeebd  reviews/review4/delta.diff
5f34cee0618fa6df1ce1be79f5fd2d24ff76d0dd79338d5d69bf414127a5c5a4  reviews/review4/final.md
f159756abdd47b7fa24dcda9a4eec4bbb83f40b8e14f223d67373163b32abb1c  reviews/review4/frozen.sha
9fba1328033b3398a3077af67be3272ba88350d2c96ef0a3cc72ba6f0f2b0f1a  reviews/review4/request.md
a5bd9dd60ddc610afee02e590b07d7d9499fd3944ddf65bf86d5f02c11762954  reviews/review4/stderr.txt
2b8c024920b48794b82cd479ab8428e253561f95b1f18713174a6889e1d2ac66  reviews/review4/stdout.txt
d35883a1538bf8e11af67d687fdf9a9290eab0c0018fbe246f95f79a0bc432e4  reviews/review5/candidate.diff
5411594ea597961785138c205411feff384a2899c720f5d434070b91bda2b4d7  reviews/review5/checks.txt
bc0fce8c3997fcda99062b07f1a550320d53606eb5bf9a3206b19612cf5b0724  reviews/review5/delta.diff
432b12c6f8e16b93e477ba6e48e404799c72c02a0fd2757c5dbbb8dab15e2f66  reviews/review5/final.md
644cb6d95757a5ed1baddfe7bf23b87edf2d814ae27e29b991b88be60cf3f0b0  reviews/review5/frozen.sha
b8191c4b2427d9927d359fd3d26653b8eba4714284f759bb585362de88e4bce3  reviews/review5/request.md
9fb8f663740eb52cf7645dfe5bc029c7b577f7e47aa81a8e3a17e4ef0914a44e  reviews/review5/stderr.txt
8fb42edaa9b07ccf19f6272b16755219d3b74f40cf8da028cbb00b2e9a36507b  reviews/review5/stdout.txt
```
