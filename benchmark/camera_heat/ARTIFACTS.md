# Artifacts

Every file in this folder with its size and SHA-256 (`ARTIFACTS.md` itself and `__pycache__/` excluded). Copied files were compared byte for byte (`cmp`) with their source at archive time; the source and archive SHA-256 match except for the redacted files listed below. `/termux-home/` is the Debian bind of the native `/data/data/com.termux/files/home/`; `/termux-home/storage/downloads/` is `/sdcard/Download/`. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

No raw file was left out for size: the whole folder is under the 15 MB budget.

## Copied (38 files)

| Archive path | Source | Bytes | SHA-256 |
|---|---|---:|---|
| `reports/CODER_REPORT_camera_heat.md` | `/termux-home/storage/downloads/CODER_REPORT_camera_heat.md` | 8438 | `50ee90250721658564b9a77c4fbdd5b3decc3f2f18ececfff75cf5624fb46f55` |
| `runs/camera_heat_20260929T022511Z_16984/request_0.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_0.json` | 18 | `2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd` |
| `runs/camera_heat_20260929T022511Z_16984/request_1.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_1.json` | 30 | `96b5ac94b95f55e202cc443ff6347d46679f1814a0b3ef6fbfb1ff8f2bf4a71c` |
| `runs/camera_heat_20260929T022511Z_16984/request_10.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_10.json` | 18 | `2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd` |
| `runs/camera_heat_20260929T022511Z_16984/request_11.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_11.json` | 40 | `b8aec640591491fa01a0d507eb087be88e7deb3940742055c3521583141e3f34` |
| `runs/camera_heat_20260929T022511Z_16984/request_12.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_12.json` | 21 | `ea02414e3003528ffc259ca509b1b1ff6550c5f8b1f3e2e56c4726cf38d28ba5` |
| `runs/camera_heat_20260929T022511Z_16984/request_13.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_13.json` | 22 | `b45d6ab904612d8fca2024bed9fbdb24316e020854a62ac69064612914f82746` |
| `runs/camera_heat_20260929T022511Z_16984/request_14.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_14.json` | 18 | `2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd` |
| `runs/camera_heat_20260929T022511Z_16984/request_15.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_15.json` | 20 | `8059753f3ff5ed2ff069ebc93150579fb5c79c40d35cb53653515ac110a3a97d` |
| `runs/camera_heat_20260929T022511Z_16984/request_2.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_2.json` | 66 | `163c69c16ad71067bf184e9449e754f615bdf5eec8250a41b8102d6ffb45b597` |
| `runs/camera_heat_20260929T022511Z_16984/request_3.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_3.json` | 21 | `ea02414e3003528ffc259ca509b1b1ff6550c5f8b1f3e2e56c4726cf38d28ba5` |
| `runs/camera_heat_20260929T022511Z_16984/request_4.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_4.json` | 48 | `c9ed5b463ac62f093f5f8f40831b5bd45adaae237c9d6aebdccda881c497cc96` |
| `runs/camera_heat_20260929T022511Z_16984/request_5.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_5.json` | 18 | `2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd` |
| `runs/camera_heat_20260929T022511Z_16984/request_6.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_6.json` | 30 | `c379fcf17c3f11cfb84d61ceee4e0a96dc829581a182eed1d2bf7be057d5eb60` |
| `runs/camera_heat_20260929T022511Z_16984/request_7.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_7.json` | 66 | `3d00fd414f13c693e0df1f0438c4466f043f235bb013d28a9fbdc6f41e9d0937` |
| `runs/camera_heat_20260929T022511Z_16984/request_8.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_8.json` | 21 | `ea02414e3003528ffc259ca509b1b1ff6550c5f8b1f3e2e56c4726cf38d28ba5` |
| `runs/camera_heat_20260929T022511Z_16984/request_9.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/request_9.json` | 48 | `8ac9b4a4cc464f438f5d3b37580c6a1fc3d9c877a3f7a0e3181c9fa50c6e1050` |
| `runs/camera_heat_20260929T022511Z_16984/response_0.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_0.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_1.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_1.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_10.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_10.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_11.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_11.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_12.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_12.json` | 238 | `5e9798c1d7c9d6a2dcdc731f43e023248e8d19094e82b2ad20d56218282060ad` |
| `runs/camera_heat_20260929T022511Z_16984/response_13.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_13.json` | 25 | `a59ffa48206df48b656a28855b5b701bd9f31b3e0d58d1b74141917b1fa6099c` |
| `runs/camera_heat_20260929T022511Z_16984/response_14.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_14.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_15.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_15.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_2.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_2.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_3.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_3.json` | 238 | `b9415c69eea368854679ef1f5cc35e1909360c511cbe6e79fbc8a5f0354adafc` |
| `runs/camera_heat_20260929T022511Z_16984/response_4.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_4.json` | 59 | `db2b80df7dcd2c06a0397edaf5268cb807003ec6bed0576a09e5d043171df31a` |
| `runs/camera_heat_20260929T022511Z_16984/response_5.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_5.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_6.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_6.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_7.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_7.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `runs/camera_heat_20260929T022511Z_16984/response_8.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_8.json` | 238 | `dfc9b1e453cc770ec431cffb53b82cd5d9673d6dc358ccfc201d30d9046944b9` |
| `runs/camera_heat_20260929T022511Z_16984/response_9.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/response_9.json` | 59 | `0a0b9cb31205d53a44cea7eaac0412b0fd8606ffea9cdee420ad50bc1073829c` |
| `runs/camera_heat_20260929T022511Z_16984/results.json` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/results.json` | 1943 | `5228da0e1578fdc07e51fb7167464c1669711714f9648a169177e29fa6849bcd` |
| `runs/camera_heat_20260929T022511Z_16984/robotcam_1.reader.txt` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/robotcam_1.reader.txt` | 15636 | `83fecea28dc3325e5cd68d6de2414dc88ad3381681ff93114c8d70bd4e83f3a4` |
| `runs/camera_heat_20260929T022511Z_16984/robotcam_2.reader.txt` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/robotcam_2.reader.txt` | 31046 | `58069446bd3d7673aaea0226241676db5a4516464861a5a7a4a246f40b696a39` |
| `runs/camera_heat_20260929T022511Z_16984/sensors.jsonl` | `/termux-home/ladder/camera_heat_20260929T022511Z_16984/sensors.jsonl` | 37874 | `7b66c640c4bb2a0bc01cedd0c902aa27f587e58648994f0bc3aadade7c66ba32` |
| `runs/oneshot_console_20260929T022511Z.log` | `/termux-home/ladder/oneshot_console_20260929T022511Z.log` | 2620 | `72d5f2d481c2bc70371cb8b3cfc33214a2b9a2163f9a2fa69e0ff0c89bf7b6b8` |

## Tools (2 files)

On disk before this task (unstaged, since 2026-09-29), unchanged; hashes equal CODER_REPORT_camera_heat.md.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `camera_heat.py` | 13655 | `63275fa0f967e0905c662d1a2daad21bc1b0fed9e39270d85c6ce6906eb95059` |
| `run_camera_heat.sh` | 556 | `4058b71094f408b2fd901dff6865f7d7e87f9ab7309ab8ce8c86e361881e4dae` |

## Written for this archive (2 files)

| Path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 4425 | `b75a41d6930121b8b23b066291a14136eccae276e577a19bc2b82df138926dbc` |
| `RUN_INDEX.md` | 1469 | `4f3c62a2baccb6980c548e7d7074dea5888db0cd86b93291e9a6e06808891576` |

## Redactions

None. The privacy scan (emails, phone numbers, IMEI/serial, Wi-Fi names, locations, tokens/keys, account names, third-party app names in logcat lines) found nothing to redact in this folder.

## Not archived (left on the phone; see RUN_INDEX.md)

| Phone path | Bytes | SHA-256 |
|---|---:|---|
| `/termux-home/ladder/camera_heat_20260929T010749Z_10076/sensors.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_0.json` | 18 | `2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_1.json` | 30 | `96b5ac94b95f55e202cc443ff6347d46679f1814a0b3ef6fbfb1ff8f2bf4a71c` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_10.json` | 18 | `2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_11.json` | 39 | `31aa4f5701071c55e953af87d9ecf62d730d08c3daf98e1e404b62c49084f66a` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_12.json` | 21 | `ea02414e3003528ffc259ca509b1b1ff6550c5f8b1f3e2e56c4726cf38d28ba5` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_13.json` | 22 | `b45d6ab904612d8fca2024bed9fbdb24316e020854a62ac69064612914f82746` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_14.json` | 18 | `2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_15.json` | 20 | `8059753f3ff5ed2ff069ebc93150579fb5c79c40d35cb53653515ac110a3a97d` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_2.json` | 66 | `163c69c16ad71067bf184e9449e754f615bdf5eec8250a41b8102d6ffb45b597` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_3.json` | 21 | `ea02414e3003528ffc259ca509b1b1ff6550c5f8b1f3e2e56c4726cf38d28ba5` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_4.json` | 48 | `c9ed5b463ac62f093f5f8f40831b5bd45adaae237c9d6aebdccda881c497cc96` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_5.json` | 18 | `2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_6.json` | 30 | `c379fcf17c3f11cfb84d61ceee4e0a96dc829581a182eed1d2bf7be057d5eb60` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_7.json` | 66 | `3d00fd414f13c693e0df1f0438c4466f043f235bb013d28a9fbdc6f41e9d0937` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_8.json` | 21 | `ea02414e3003528ffc259ca509b1b1ff6550c5f8b1f3e2e56c4726cf38d28ba5` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/request_9.json` | 48 | `8ac9b4a4cc464f438f5d3b37580c6a1fc3d9c877a3f7a0e3181c9fa50c6e1050` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_0.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_1.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_10.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_11.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_12.json` | 238 | `8baba5c5dd4d0aa0c4b856757bb062f62225fd08ebc52fc26eb457637d423eae` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_13.json` | 23 | `97e69ec5dd043d45d58a3e433e028c8cd4083f90b49b24b7e0bb21a7e5761a33` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_14.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_15.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_2.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_3.json` | 238 | `a97e406cdbf67a271bd5b48a641e6c09681e121217e3995becd2b6c921d05488` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_4.json` | 57 | `6bfd146a59acface7fcadc7d21f9f65352fb36950e76ec7d520628e56c6fb51a` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_5.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_6.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_7.json` | 12 | `6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_8.json` | 238 | `5464d1bd2c912832d642b0398b97f0234a6fe6964166851d9b98f8159fd18584` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/response_9.json` | 57 | `39a803eb6a976bc141ff86b39ab46c06588ab7453acbdfde6442e531000300bc` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/results.json` | 1925 | `224787b18d43b0ffac0dff8ce89dd4a3a1fec36fde5ee30772a3d263664d1143` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/robotcam_1.reader.txt` | 1951 | `f70ddd6123ac5ca5c0c8b11c8b52c1cd89b391c7a2b85f683826dfe467ee302e` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/robotcam_2.reader.txt` | 3651 | `68dacddd82bce69edb4c61b18e2fb06da09b3d710bbbb4f801f52fb4f9aace93` |
| `/termux-home/ladder/camera_heat_20260929T021119Z_12612/sensors.jsonl` | 14329 | `1d638d95cf0d72a4604e6308dc5daa8ab5f088cf6f3d4864eefab05f5d60a412` |
| `/termux-home/ladder/oneshot_console_20260929T021119Z.log` | 2497 | `adeccd3af866f70013d0cd836b9d21531b66848b15b7c695950aaf0d89a6d71e` |

## sha256sum format

```sha256sums
b75a41d6930121b8b23b066291a14136eccae276e577a19bc2b82df138926dbc  README.md
4f3c62a2baccb6980c548e7d7074dea5888db0cd86b93291e9a6e06808891576  RUN_INDEX.md
63275fa0f967e0905c662d1a2daad21bc1b0fed9e39270d85c6ce6906eb95059  camera_heat.py
50ee90250721658564b9a77c4fbdd5b3decc3f2f18ececfff75cf5624fb46f55  reports/CODER_REPORT_camera_heat.md
4058b71094f408b2fd901dff6865f7d7e87f9ab7309ab8ce8c86e361881e4dae  run_camera_heat.sh
2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd  runs/camera_heat_20260929T022511Z_16984/request_0.json
96b5ac94b95f55e202cc443ff6347d46679f1814a0b3ef6fbfb1ff8f2bf4a71c  runs/camera_heat_20260929T022511Z_16984/request_1.json
2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd  runs/camera_heat_20260929T022511Z_16984/request_10.json
b8aec640591491fa01a0d507eb087be88e7deb3940742055c3521583141e3f34  runs/camera_heat_20260929T022511Z_16984/request_11.json
ea02414e3003528ffc259ca509b1b1ff6550c5f8b1f3e2e56c4726cf38d28ba5  runs/camera_heat_20260929T022511Z_16984/request_12.json
b45d6ab904612d8fca2024bed9fbdb24316e020854a62ac69064612914f82746  runs/camera_heat_20260929T022511Z_16984/request_13.json
2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd  runs/camera_heat_20260929T022511Z_16984/request_14.json
8059753f3ff5ed2ff069ebc93150579fb5c79c40d35cb53653515ac110a3a97d  runs/camera_heat_20260929T022511Z_16984/request_15.json
163c69c16ad71067bf184e9449e754f615bdf5eec8250a41b8102d6ffb45b597  runs/camera_heat_20260929T022511Z_16984/request_2.json
ea02414e3003528ffc259ca509b1b1ff6550c5f8b1f3e2e56c4726cf38d28ba5  runs/camera_heat_20260929T022511Z_16984/request_3.json
c9ed5b463ac62f093f5f8f40831b5bd45adaae237c9d6aebdccda881c497cc96  runs/camera_heat_20260929T022511Z_16984/request_4.json
2bcc337a33841b5a2c00408b09d907a8ee77f6a65fc8dfde007d2d6bccc839dd  runs/camera_heat_20260929T022511Z_16984/request_5.json
c379fcf17c3f11cfb84d61ceee4e0a96dc829581a182eed1d2bf7be057d5eb60  runs/camera_heat_20260929T022511Z_16984/request_6.json
3d00fd414f13c693e0df1f0438c4466f043f235bb013d28a9fbdc6f41e9d0937  runs/camera_heat_20260929T022511Z_16984/request_7.json
ea02414e3003528ffc259ca509b1b1ff6550c5f8b1f3e2e56c4726cf38d28ba5  runs/camera_heat_20260929T022511Z_16984/request_8.json
8ac9b4a4cc464f438f5d3b37580c6a1fc3d9c877a3f7a0e3181c9fa50c6e1050  runs/camera_heat_20260929T022511Z_16984/request_9.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_0.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_1.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_10.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_11.json
5e9798c1d7c9d6a2dcdc731f43e023248e8d19094e82b2ad20d56218282060ad  runs/camera_heat_20260929T022511Z_16984/response_12.json
a59ffa48206df48b656a28855b5b701bd9f31b3e0d58d1b74141917b1fa6099c  runs/camera_heat_20260929T022511Z_16984/response_13.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_14.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_15.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_2.json
b9415c69eea368854679ef1f5cc35e1909360c511cbe6e79fbc8a5f0354adafc  runs/camera_heat_20260929T022511Z_16984/response_3.json
db2b80df7dcd2c06a0397edaf5268cb807003ec6bed0576a09e5d043171df31a  runs/camera_heat_20260929T022511Z_16984/response_4.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_5.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_6.json
6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3  runs/camera_heat_20260929T022511Z_16984/response_7.json
dfc9b1e453cc770ec431cffb53b82cd5d9673d6dc358ccfc201d30d9046944b9  runs/camera_heat_20260929T022511Z_16984/response_8.json
0a0b9cb31205d53a44cea7eaac0412b0fd8606ffea9cdee420ad50bc1073829c  runs/camera_heat_20260929T022511Z_16984/response_9.json
5228da0e1578fdc07e51fb7167464c1669711714f9648a169177e29fa6849bcd  runs/camera_heat_20260929T022511Z_16984/results.json
83fecea28dc3325e5cd68d6de2414dc88ad3381681ff93114c8d70bd4e83f3a4  runs/camera_heat_20260929T022511Z_16984/robotcam_1.reader.txt
58069446bd3d7673aaea0226241676db5a4516464861a5a7a4a246f40b696a39  runs/camera_heat_20260929T022511Z_16984/robotcam_2.reader.txt
7b66c640c4bb2a0bc01cedd0c902aa27f587e58648994f0bc3aadade7c66ba32  runs/camera_heat_20260929T022511Z_16984/sensors.jsonl
72d5f2d481c2bc70371cb8b3cfc33214a2b9a2163f9a2fa69e0ff0c89bf7b6b8  runs/oneshot_console_20260929T022511Z.log
```
