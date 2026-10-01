# Artifacts

Every file in this folder with its size and SHA-256 (`ARTIFACTS.md` itself and `__pycache__/` excluded). Copied files were compared byte for byte (`cmp`) with their source at archive time; the source and archive SHA-256 match except for the redacted files listed below. `/termux-home/` is the Debian bind of the native `/data/data/com.termux/files/home/`; `/termux-home/storage/downloads/` is `/sdcard/Download/`. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

No raw file was left out for size: the whole folder is under the 15 MB budget.

## Copied (27 files)

| Archive path | Source | Bytes | SHA-256 |
|---|---|---:|---|
| `coresidency_executed_65c572e01826.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/22fa5409-6f42-426e-899a-74805d4e9c0c/scratchpad/frozen/coresidency/coresidency.py` | 49674 | `65c572e01826525e5f26881e9d5167145beb430a07e41f6ff15b37c58d1cf60d` |
| `phone_config/thermal_info_config.json` | `/termux-home/storage/downloads/thermal_info_config.json` | 19897 | `a4ba034e30c5cccebe18ad4b924e276d72e4ad1e2a98d8d101f24f8d955f843a` |
| `phone_config/thermal_info_config_charge.json` | `/termux-home/storage/downloads/thermal_info_config_charge.json` | 6291 | `884742776b013e36c518352e595df1cbe4f953c990b6bfdb978450c7b2278495` |
| `phone_config/thermal_info_config_proto.json` | `/termux-home/storage/downloads/thermal_info_config_proto.json` | 2594 | `5ac637c1157136bdd7a78bc3e2fb3a8723379702fc7bab01c4ef3c2368b807e5` |
| `phone_config/thermal_zones.txt` | `/termux-home/storage/downloads/thermal_zones.txt` | 2400 | `2c60633b27ef25e02eb4984edbebf71d21be8f28294af5bdd32bae39dd9185e4` |
| `reports/THERMAL_CHAR_CODER_REPORT.md` | `/termux-home/storage/downloads/THERMAL_CHAR_CODER_REPORT.md` | 25374 | `9bf80b3acc8abc3e1473b0c401dfc47df156e8dca7e2cc3b24e799a6163c79c1` |
| `reports/THERMAL_CHAR_FIX1_REPORT.md` | `/termux-home/storage/downloads/THERMAL_CHAR_FIX1_REPORT.md` | 21507 | `2538d111e1cfdc8a64fb43da3c2c49f7682420f6719d8e5a4f6321dc98ca7210` |
| `runs/oneshot_console_20260930T230609Z.log` | `/termux-home/thermal_char/oneshot_console_20260930T230609Z.log` | 214519 | `afddc513863c68e83da1af5dfe75f0aa02e19affbf57bd1d52f6abe151c1c968` |
| `runs/oneshot_console_20260930T232700Z.log` | `/termux-home/thermal_char/oneshot_console_20260930T232700Z.log` | 1045023 | `ab061b35054b9faa2c01699bbf910364b2a54510e846e7b265c748eab0e0c085` |
| `runs/run_20260930T231117Z_smoke/frames.jsonl` | `/termux-home/thermal_char/run_20260930T231117Z_smoke/frames.jsonl` | 3296 | `aeb8e78e87164683660ae6c0d4146198db32c2914f93cd4fe9d194cfb055b2f4` |
| `runs/run_20260930T231117Z_smoke/gen.jsonl` | `/termux-home/thermal_char/run_20260930T231117Z_smoke/gen.jsonl` | 2861 | `45272a456dc3913792710ac63e384a4ec0503c5647c7848e9a7ed9721dc7f517` |
| `runs/run_20260930T231117Z_smoke/llama-server.log` | `/termux-home/thermal_char/run_20260930T231117Z_smoke/llama-server.log` | 3665 | `f54eddfab1364330eef8b4c7b433773f55dcbd6c8f384a58917f5f9ba559bdbf` |
| `runs/run_20260930T231117Z_smoke/report.txt` | `/termux-home/thermal_char/run_20260930T231117Z_smoke/report.txt` | 213310 | `47fc97f28ae5fef40de89ddbf72cd09ccb1093eac874ff382defa9036ff7cd84` |
| `runs/run_20260930T231117Z_smoke/run.json` | `/termux-home/thermal_char/run_20260930T231117Z_smoke/run.json` | 15013 | `e9fddd7276147ab0de4daa032972a46b45b80adc3394198fa080a4acf8ab41f3` |
| `runs/run_20260930T231117Z_smoke/samples_1s.jsonl` | `/termux-home/thermal_char/run_20260930T231117Z_smoke/samples_1s.jsonl` | 126928 | `6ddb5986fd5a5f6b4595886a447c0b4bfede1281cea8aefc9b3b029699561115` |
| `runs/run_20260930T231117Z_smoke/thermalservice_5s.jsonl` | `/termux-home/thermal_char/run_20260930T231117Z_smoke/thermalservice_5s.jsonl` | 24445 | `d946b94d42fed2ba7bb702a8bb65dce35266a3ee0db899bfe4b06741218e3d7d` |
| `runs/run_20260930T231117Z_smoke/thermalservice_raw.jsonl` | `/termux-home/thermal_char/run_20260930T231117Z_smoke/thermalservice_raw.jsonl` | 192188 | `81058f8093e22fc324e73cdd2d982393137e06d0d6b5d235246f78194459672f` |
| `runs/run_20260930T233208Z/frames.jsonl` | `/termux-home/thermal_char/run_20260930T233208Z/frames.jsonl` | 44493 | `357674677da19ab3bfdd09eea574f37743184deb25b96c093d3fa9dda4d0ff31` |
| `runs/run_20260930T233208Z/gen.jsonl` | `/termux-home/thermal_char/run_20260930T233208Z/gen.jsonl` | 39245 | `b718583062f8f2c2becc73b7d6767c72541a768cd8de3e8d5388f56fd6c6db35` |
| `runs/run_20260930T233208Z/llama-server.log` | `/termux-home/thermal_char/run_20260930T233208Z/llama-server.log` | 39455 | `8503c84b7554e939a5f3a9f803c1601d79dc3a16570e5b581dca66539fb435ba` |
| `runs/run_20260930T233208Z/report.txt` | `/termux-home/thermal_char/run_20260930T233208Z/report.txt` | 1043715 | `2c3624a7e398fc9c187739a610b092452e121e10e1a14eac16f02ba1153ebb62` |
| `runs/run_20260930T233208Z/run.json` | `/termux-home/thermal_char/run_20260930T233208Z/run.json` | 15007 | `0723a305cbd310da6c6281c35910c96e84cb59505ac1c128e0aa752edc846efa` |
| `runs/run_20260930T233208Z/samples_1s.jsonl` | `/termux-home/thermal_char/run_20260930T233208Z/samples_1s.jsonl` | 1524560 | `e2d0ad5380c592fd4b6f9d9b165dc7df30b8d12aa774e2b90e190197462aac2a` |
| `runs/run_20260930T233208Z/thermalservice_5s.jsonl` | `/termux-home/thermal_char/run_20260930T233208Z/thermalservice_5s.jsonl` | 288820 | `3294b00c2e56d1bf00f899c698f522deb5c855c86d43fa3e3509ad20c9998b37` |
| `runs/run_20260930T233208Z/thermalservice_raw.jsonl` | `/termux-home/thermal_char/run_20260930T233208Z/thermalservice_raw.jsonl` | 2256969 | `d417c16f9b7aecd36c8d5f9a3ed37d0ff0b1172ac43f50025693960f0828477a` |
| `runs/thermal.log` | `/termux-home/thermal_char/thermal.log` | 22353 | `f4e45567037a9b0de95c3a195c8988dbb0dd38179d7b94310981532eb0fdc168` |
| `thermal_char_executed_cbf73d12b5bc.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/22fa5409-6f42-426e-899a-74805d4e9c0c/scratchpad/frozen/thermal_char/thermal_char.py` | 43763 | `cbf73d12b5bcebb80db1d3c09f5b1e72f0231df915c490669c925f1bd9365d1d` |

## Tools (6 files)

On disk before this task (unstaged), unchanged; hashes equal CLEANUP1_REPORT.md "New SHA-256".

| Path | Bytes | SHA-256 |
|---|---:|---|
| `thermal_char.py` | 43958 | `b5b0c812c535cb69987ffb726bd1a37d2ba4d341c7fedd20d7b13d23e8f89b84` |
| `oneshot.sh` | 7687 | `16787b0679c24b711eb9e370cdcc9a9942cf689718764874b8ec3e418f84d77b` |
| `run_thermal_char.sh` | 424 | `c4df2fa2799f019fd67ac68580c9296001e1e8ea63d51f75ab906184864d682e` |
| `test_thermal_char.py` | 33902 | `e54d5a9338fa01837f316e58241007f58d5ba9e695787c0316d9ee14dc609b43` |
| `test_oneshot.sh` | 8818 | `e640e6f33e0c44585322f37394b68936a59b22adc32bde824b649d3d7a6ee2f6` |
| `RUN.md` | 3741 | `46c49c727fa02fbc71b4fd5a97c1919a53652bf802a335cfe4a3dea9c03a336d` |

## Written for this archive (3 files)

| Path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 7669 | `ed7dcf5c0099b5febcf9e8967b04d35bc31473d3c4eb29f01a24d7251e397762` |
| `RUN_INDEX.md` | 1041 | `bdfa1a8739f9eef62b42707d83bc50864825d8823437886fd54e0e062df58dc5` |
| `HEAT_EVIDENCE.md` | 15728 | `5e749f5c51cfa841412f048cdc226f20f75bd4f32845860e3d7470a1f6255ab0` |

## Redactions

None. The privacy scan (emails, phone numbers, IMEI/serial, Wi-Fi names, locations, tokens/keys, account names, third-party app names in logcat lines) found nothing to redact in this folder.

## sha256sum format

```sha256sums
5e749f5c51cfa841412f048cdc226f20f75bd4f32845860e3d7470a1f6255ab0  HEAT_EVIDENCE.md
ed7dcf5c0099b5febcf9e8967b04d35bc31473d3c4eb29f01a24d7251e397762  README.md
46c49c727fa02fbc71b4fd5a97c1919a53652bf802a335cfe4a3dea9c03a336d  RUN.md
bdfa1a8739f9eef62b42707d83bc50864825d8823437886fd54e0e062df58dc5  RUN_INDEX.md
65c572e01826525e5f26881e9d5167145beb430a07e41f6ff15b37c58d1cf60d  coresidency_executed_65c572e01826.py
16787b0679c24b711eb9e370cdcc9a9942cf689718764874b8ec3e418f84d77b  oneshot.sh
a4ba034e30c5cccebe18ad4b924e276d72e4ad1e2a98d8d101f24f8d955f843a  phone_config/thermal_info_config.json
884742776b013e36c518352e595df1cbe4f953c990b6bfdb978450c7b2278495  phone_config/thermal_info_config_charge.json
5ac637c1157136bdd7a78bc3e2fb3a8723379702fc7bab01c4ef3c2368b807e5  phone_config/thermal_info_config_proto.json
2c60633b27ef25e02eb4984edbebf71d21be8f28294af5bdd32bae39dd9185e4  phone_config/thermal_zones.txt
9bf80b3acc8abc3e1473b0c401dfc47df156e8dca7e2cc3b24e799a6163c79c1  reports/THERMAL_CHAR_CODER_REPORT.md
2538d111e1cfdc8a64fb43da3c2c49f7682420f6719d8e5a4f6321dc98ca7210  reports/THERMAL_CHAR_FIX1_REPORT.md
c4df2fa2799f019fd67ac68580c9296001e1e8ea63d51f75ab906184864d682e  run_thermal_char.sh
afddc513863c68e83da1af5dfe75f0aa02e19affbf57bd1d52f6abe151c1c968  runs/oneshot_console_20260930T230609Z.log
ab061b35054b9faa2c01699bbf910364b2a54510e846e7b265c748eab0e0c085  runs/oneshot_console_20260930T232700Z.log
aeb8e78e87164683660ae6c0d4146198db32c2914f93cd4fe9d194cfb055b2f4  runs/run_20260930T231117Z_smoke/frames.jsonl
45272a456dc3913792710ac63e384a4ec0503c5647c7848e9a7ed9721dc7f517  runs/run_20260930T231117Z_smoke/gen.jsonl
f54eddfab1364330eef8b4c7b433773f55dcbd6c8f384a58917f5f9ba559bdbf  runs/run_20260930T231117Z_smoke/llama-server.log
47fc97f28ae5fef40de89ddbf72cd09ccb1093eac874ff382defa9036ff7cd84  runs/run_20260930T231117Z_smoke/report.txt
e9fddd7276147ab0de4daa032972a46b45b80adc3394198fa080a4acf8ab41f3  runs/run_20260930T231117Z_smoke/run.json
6ddb5986fd5a5f6b4595886a447c0b4bfede1281cea8aefc9b3b029699561115  runs/run_20260930T231117Z_smoke/samples_1s.jsonl
d946b94d42fed2ba7bb702a8bb65dce35266a3ee0db899bfe4b06741218e3d7d  runs/run_20260930T231117Z_smoke/thermalservice_5s.jsonl
81058f8093e22fc324e73cdd2d982393137e06d0d6b5d235246f78194459672f  runs/run_20260930T231117Z_smoke/thermalservice_raw.jsonl
357674677da19ab3bfdd09eea574f37743184deb25b96c093d3fa9dda4d0ff31  runs/run_20260930T233208Z/frames.jsonl
b718583062f8f2c2becc73b7d6767c72541a768cd8de3e8d5388f56fd6c6db35  runs/run_20260930T233208Z/gen.jsonl
8503c84b7554e939a5f3a9f803c1601d79dc3a16570e5b581dca66539fb435ba  runs/run_20260930T233208Z/llama-server.log
2c3624a7e398fc9c187739a610b092452e121e10e1a14eac16f02ba1153ebb62  runs/run_20260930T233208Z/report.txt
0723a305cbd310da6c6281c35910c96e84cb59505ac1c128e0aa752edc846efa  runs/run_20260930T233208Z/run.json
e2d0ad5380c592fd4b6f9d9b165dc7df30b8d12aa774e2b90e190197462aac2a  runs/run_20260930T233208Z/samples_1s.jsonl
3294b00c2e56d1bf00f899c698f522deb5c855c86d43fa3e3509ad20c9998b37  runs/run_20260930T233208Z/thermalservice_5s.jsonl
d417c16f9b7aecd36c8d5f9a3ed37d0ff0b1172ac43f50025693960f0828477a  runs/run_20260930T233208Z/thermalservice_raw.jsonl
f4e45567037a9b0de95c3a195c8988dbb0dd38179d7b94310981532eb0fdc168  runs/thermal.log
e640e6f33e0c44585322f37394b68936a59b22adc32bde824b649d3d7a6ee2f6  test_oneshot.sh
e54d5a9338fa01837f316e58241007f58d5ba9e695787c0316d9ee14dc609b43  test_thermal_char.py
b5b0c812c535cb69987ffb726bd1a37d2ba4d341c7fedd20d7b13d23e8f89b84  thermal_char.py
cbf73d12b5bcebb80db1d3c09f5b1e72f0231df915c490669c925f1bd9365d1d  thermal_char_executed_cbf73d12b5bc.py
```
