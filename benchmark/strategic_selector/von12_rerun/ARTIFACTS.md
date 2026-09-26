# Artifacts

Every archived file below with its SHA-256. Copied files were compared byte for byte (`cmp`) with the phone original at archive time, and the phone and archive SHA-256 values match. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Copied from the phone (39 files)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `raw/archived_freeze.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/archived_freeze.txt` | 758 | `de8a9f235a71d55bcc5af6daa26c6f1105331911e9c91be071228cea0819b1b7` |
| `raw/archived_pins.txt` | `/termux-home/von12-test/archived_pins.txt` | 758 | `0b21b0a73de0bde4eab128940f3f360ce095c217b3a4e674c33580cca0c2d929` |
| `raw/blockA.block.json` | `/termux-home/von12-test/blockA.block.json` | 1575 | `293cad0972db75255e34dcb77ed9b97e6bd0cf597fb77267ff952b519975ae9f` |
| `raw/blockA.stderr.txt` | `/termux-home/von12-test/blockA.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `raw/blockA.stdout.txt` | `/termux-home/von12-test/blockA.stdout.txt` | 554 | `9bf278a39cccb1ef102abaceb510a9e92b31d50a00f824332e1fb2c4883bfff5` |
| `raw/blockB.block.json` | `/termux-home/von12-test/blockB.block.json` | 1484 | `db3a1dba8323e4a4ea16e485147081d98d7eee64d2ed8a2de8bcf52b8af28445` |
| `raw/blockB.stderr.txt` | `/termux-home/von12-test/blockB.stderr.txt` | 581 | `1bf4ef920f13db3388e40dcfd268524be578bf925b332ef2397fe3d010e2906f` |
| `raw/blockB.stdout.txt` | `/termux-home/von12-test/blockB.stdout.txt` | 4044 | `6eaf779502c6aca9d151c29cd9d702a4255d11c5f8f2cf679e8be9c6a5445869` |
| `raw/cooldown_gate.txt` | `/termux-home/von12-test/cooldown_gate.txt` | 264 | `af49774c564209597e74431b70812d21ca759950523c94f8049dd09b014e8caa` |
| `raw/determinism_check.py` | `/termux-home/von12-test/determinism_check.py` | 3586 | `4f11659df7aff0f14ab1b3158ac1a60c25c1634eab0400bb8471af9eed272417` |
| `raw/determinism_stderr.txt` | `/termux-home/von12-test/determinism_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `raw/determinism_stdout.txt` | `/termux-home/von12-test/determinism_stdout.txt` | 600 | `d090e93e0dc2a3b38c6e6f1db0b7a7caf0be0237fc53e4c5e76d1f49e22a79d8` |
| `raw/download_stderr.txt` | `/termux-home/von12-test/download_stderr.txt` | 663 | `465447dc8f90e7aaabd33cce39f5d45b4fb060edd6451c65ab61abfa50fc1c2b` |
| `raw/download_stdout.txt` | `/termux-home/von12-test/download_stdout.txt` | 116 | `0b9df21ee5593cca9c4fc62dfda55f5747937ea90d811f2b56eba7683a05f1e8` |
| `raw/negative_control/negctl.py` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/negctl.py` | 3655 | `29a2d23a8fa8833dbc2f02b32f2df31601218fadda2e8599b95339cdda201bac` |
| `raw/negative_control/perturbed.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/perturbed.json` | 263095 | `b6620de324c4972dc48b6ff9dfc1fc44a2faca6c77d6bdd6924594f324fa9fd9` |
| `raw/pip_install_stderr.txt` | `/termux-home/von12-test/pip_install_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `raw/pip_install_stdout.txt` | `/termux-home/von12-test/pip_install_stdout.txt` | 13062 | `afb1498eee7ed37b498aada438fd7a0e4a42cfe7ddc104b59dbe74b34e2b57a5` |
| `raw/probe.block.json` | `/termux-home/von12-test/probe.block.json` | 1460 | `950ed408a45f5ed77b4e7317c7eb39a8212c1e84e29c7b7cb374b2fe4f100016` |
| `raw/probe.stderr.txt` | `/termux-home/von12-test/probe.stderr.txt` | 457 | `67edfda21a5e503d3d781a8ff9d2aeed8b9c0c4e3f02a48b396e46997c534c45` |
| `raw/probe.stdout.txt` | `/termux-home/von12-test/probe.stdout.txt` | 527 | `bc0a80fd044aa6d8ef28530d31802c742af3d1766876f36ed762a6ed57c4eec5` |
| `raw/run_block.py` | `/termux-home/von12-test/run_block.py` | 2736 | `79b404782c3e2056ab0e9e43da70c0facecb1a689eebc38e2569347fc0299a25` |
| `raw/runs/2026-09-24T184107Z-von-68e0c438/manifest.json` | `/termux-home/von12-test/runs/2026-09-24T184107Z-von-68e0c438/manifest.json` | 17102 | `330545a00697adf7568de221686ea4fd38f8b3e67c36e712b485fcdfcbd639bc` |
| `raw/runs/2026-09-24T184107Z-von-68e0c438/results.json` | `/termux-home/von12-test/runs/2026-09-24T184107Z-von-68e0c438/results.json` | 316643 | `8a50c966270068bb4c32d183ada1bb02995045a51771a653a52120ee77e0b9c9` |
| `raw/runs/2026-09-24T184107Z-von-68e0c438/stderr.txt` | `/termux-home/von12-test/runs/2026-09-24T184107Z-von-68e0c438/stderr.txt` | 514 | `cd249624b58b07af5aefa604a192f9dc8235fd7406a484ead9c7b6a9e3f83d0e` |
| `raw/runs/2026-09-24T184107Z-von-68e0c438/stdout.txt` | `/termux-home/von12-test/runs/2026-09-24T184107Z-von-68e0c438/stdout.txt` | 13963 | `192d01dbdd9493925cc385328ce5abe47871dd35fe7655b356b8a744739b2b31` |
| `raw/thermal.log` | `/termux-home/von12-test/thermal.log` | 603799 | `fad1549b9b0977a7ddd6323aa08eba3934434c7184c11dbb663add9e4226b0c6` |
| `raw/venv_freeze.txt` | `/termux-home/von12-test/venv_freeze.txt` | 758 | `0b21b0a73de0bde4eab128940f3f360ce095c217b3a4e674c33580cca0c2d929` |
| `raw/von12-heldout-extra-results.json` | `/termux-home/von12-test/von12-heldout-extra-results.json` | 94296 | `c3167677e65c74b36ff1e0b93b60642e3015916f82d547fa7b5308eb049af4af` |
| `raw/von12-probe.json` | `/termux-home/von12-test/von12-probe.json` | 131 | `5f4ed975bc8f7911afb97357a9ff554d23259e528eabf3a3b9066c3f45359e2c` |
| `raw/von12_heldout_extra.py` | `/termux-home/von12-test/von12_heldout_extra.py` | 3690 | `d889cd5a8ed6749b33305240968b0269c0319c66caeceec805c2389755c15a8b` |
| `reports/CODER_REPORT_von12_rerun.md` | `/termux-home/archive-staging/reports/CODER_REPORT_von12_rerun.md` | 16464 | `ba664c7c107c31df0d138fc20fc4fd06dcfe8cb92b6858ae192a97dff9fe6666` |
| `review/FROZEN_SHA256SUMS.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/review_ws/FROZEN_SHA256SUMS.txt` | 4004 | `14a64d8a3cf0f47381416034ec0f64805df9ea78493a35a04f6080ef0e9d0274` |
| `review/review_attempt1_stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/review_attempt1_stderr.txt` | 303 | `6f9b232ca1f3b43aa72e2ba63698c09b0796b612d5a17cf05efee31e6c79b713` |
| `review/review_attempt1_stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/review_attempt1_stdout.json` | 321 | `400c91019470fb3d0d9ff3fc64797f9141065492d0340cac87ca0db457b7f4d3` |
| `review/review_attempt2_stderr.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/review_attempt2_stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `review/review_attempt2_stdout.json` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/review_attempt2_stdout.json` | 4704 | `0454b184dc1a9d6511e2a10c0717a3929b05b98c262a357449d848361377d78a` |
| `review/review_request.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/review_request.txt` | 34998 | `eee515ad10f7a1e08b954b79717f748b03ec5c66ba13f829e680d10bd4f1dbbc` |
| `review/review_request_attempt2.txt` | `/tmp/claude-0/-data-data-com-termux-files-home-robot/861a0a6f-957a-4f4c-89f0-a59cee9d75e3/scratchpad/review_request_attempt2.txt` | 35320 | `e7225317d8c110fedb42cb0b29f2e55c3f20212a2d61bc0f7a4856236556bfa1` |

## Written for this archive or already in place in the repository (2 files)

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 3586 | `edf7d044a86a3a0bb1bc73ed5a433bc1ac219f751aeb4d1c13b33f31124483b7` |
| `RUN_INDEX.md` | 1028 | `1d0c4cd548f59e888fc64a94302934cb42e52ec3a235ae4c31b375778029ab09` |

## sha256sum format

```sha256sums
edf7d044a86a3a0bb1bc73ed5a433bc1ac219f751aeb4d1c13b33f31124483b7  README.md
1d0c4cd548f59e888fc64a94302934cb42e52ec3a235ae4c31b375778029ab09  RUN_INDEX.md
de8a9f235a71d55bcc5af6daa26c6f1105331911e9c91be071228cea0819b1b7  raw/archived_freeze.txt
0b21b0a73de0bde4eab128940f3f360ce095c217b3a4e674c33580cca0c2d929  raw/archived_pins.txt
293cad0972db75255e34dcb77ed9b97e6bd0cf597fb77267ff952b519975ae9f  raw/blockA.block.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  raw/blockA.stderr.txt
9bf278a39cccb1ef102abaceb510a9e92b31d50a00f824332e1fb2c4883bfff5  raw/blockA.stdout.txt
db3a1dba8323e4a4ea16e485147081d98d7eee64d2ed8a2de8bcf52b8af28445  raw/blockB.block.json
1bf4ef920f13db3388e40dcfd268524be578bf925b332ef2397fe3d010e2906f  raw/blockB.stderr.txt
6eaf779502c6aca9d151c29cd9d702a4255d11c5f8f2cf679e8be9c6a5445869  raw/blockB.stdout.txt
af49774c564209597e74431b70812d21ca759950523c94f8049dd09b014e8caa  raw/cooldown_gate.txt
4f11659df7aff0f14ab1b3158ac1a60c25c1634eab0400bb8471af9eed272417  raw/determinism_check.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  raw/determinism_stderr.txt
d090e93e0dc2a3b38c6e6f1db0b7a7caf0be0237fc53e4c5e76d1f49e22a79d8  raw/determinism_stdout.txt
465447dc8f90e7aaabd33cce39f5d45b4fb060edd6451c65ab61abfa50fc1c2b  raw/download_stderr.txt
0b9df21ee5593cca9c4fc62dfda55f5747937ea90d811f2b56eba7683a05f1e8  raw/download_stdout.txt
29a2d23a8fa8833dbc2f02b32f2df31601218fadda2e8599b95339cdda201bac  raw/negative_control/negctl.py
b6620de324c4972dc48b6ff9dfc1fc44a2faca6c77d6bdd6924594f324fa9fd9  raw/negative_control/perturbed.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  raw/pip_install_stderr.txt
afb1498eee7ed37b498aada438fd7a0e4a42cfe7ddc104b59dbe74b34e2b57a5  raw/pip_install_stdout.txt
950ed408a45f5ed77b4e7317c7eb39a8212c1e84e29c7b7cb374b2fe4f100016  raw/probe.block.json
67edfda21a5e503d3d781a8ff9d2aeed8b9c0c4e3f02a48b396e46997c534c45  raw/probe.stderr.txt
bc0a80fd044aa6d8ef28530d31802c742af3d1766876f36ed762a6ed57c4eec5  raw/probe.stdout.txt
79b404782c3e2056ab0e9e43da70c0facecb1a689eebc38e2569347fc0299a25  raw/run_block.py
330545a00697adf7568de221686ea4fd38f8b3e67c36e712b485fcdfcbd639bc  raw/runs/2026-09-24T184107Z-von-68e0c438/manifest.json
8a50c966270068bb4c32d183ada1bb02995045a51771a653a52120ee77e0b9c9  raw/runs/2026-09-24T184107Z-von-68e0c438/results.json
cd249624b58b07af5aefa604a192f9dc8235fd7406a484ead9c7b6a9e3f83d0e  raw/runs/2026-09-24T184107Z-von-68e0c438/stderr.txt
192d01dbdd9493925cc385328ce5abe47871dd35fe7655b356b8a744739b2b31  raw/runs/2026-09-24T184107Z-von-68e0c438/stdout.txt
fad1549b9b0977a7ddd6323aa08eba3934434c7184c11dbb663add9e4226b0c6  raw/thermal.log
0b21b0a73de0bde4eab128940f3f360ce095c217b3a4e674c33580cca0c2d929  raw/venv_freeze.txt
c3167677e65c74b36ff1e0b93b60642e3015916f82d547fa7b5308eb049af4af  raw/von12-heldout-extra-results.json
5f4ed975bc8f7911afb97357a9ff554d23259e528eabf3a3b9066c3f45359e2c  raw/von12-probe.json
d889cd5a8ed6749b33305240968b0269c0319c66caeceec805c2389755c15a8b  raw/von12_heldout_extra.py
ba664c7c107c31df0d138fc20fc4fd06dcfe8cb92b6858ae192a97dff9fe6666  reports/CODER_REPORT_von12_rerun.md
14a64d8a3cf0f47381416034ec0f64805df9ea78493a35a04f6080ef0e9d0274  review/FROZEN_SHA256SUMS.txt
6f9b232ca1f3b43aa72e2ba63698c09b0796b612d5a17cf05efee31e6c79b713  review/review_attempt1_stderr.txt
400c91019470fb3d0d9ff3fc64797f9141065492d0340cac87ca0db457b7f4d3  review/review_attempt1_stdout.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  review/review_attempt2_stderr.txt
0454b184dc1a9d6511e2a10c0717a3929b05b98c262a357449d848361377d78a  review/review_attempt2_stdout.json
eee515ad10f7a1e08b954b79717f748b03ec5c66ba13f829e680d10bd4f1dbbc  review/review_request.txt
e7225317d8c110fedb42cb0b29f2e55c3f20212a2d61bc0f7a4856236556bfa1  review/review_request_attempt2.txt
```
