# Artifacts

Every file in this folder with its size and SHA-256 (`ARTIFACTS.md` itself and `__pycache__/` excluded). Copied files were compared byte for byte (`cmp`) with their source at archive time; the source and archive SHA-256 match except for the redacted files listed below. `/termux-home/` is the Debian bind of the native `/data/data/com.termux/files/home/`; `/termux-home/storage/downloads/` is `/sdcard/Download/`. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

No raw file was left out for size: the whole folder is under the 15 MB budget.

## Copied (26 files)

| Archive path | Source | Bytes | SHA-256 |
|---|---|---:|---|
| `reports/CLEANUP1_REPORT.md` | `/termux-home/storage/downloads/CLEANUP1_REPORT.md` | 105688 | `4a132c50512ccde40423694779f132f437520f01d6174d523cb8af91c80d3b6d` |
| `reports/CORESIDENCY_CODER_REPORT.md` | `/termux-home/storage/downloads/CORESIDENCY_CODER_REPORT.md` | 45258 | `85b45b1d385889eabd8b129fb8e938cc6b978e620a90d36f59548374cf3498ce` |
| `reports/CORESIDENCY_FIX1_REPORT.md` | `/termux-home/storage/downloads/CORESIDENCY_FIX1_REPORT.md` | 51103 | `87932e0cf7a941849b642dfe497f27545b9479e66d092297a4940aa64503ad61` |
| `runs/oneshot_console_20261001T040014Z.log` | `/termux-home/coresidency/oneshot_console_20261001T040014Z.log` | 7846 | `e0ad1c5be49cc56d8b52cc5d1e5d9c76cafff63154188086a1eaf17820b1df0f` |
| `runs/oneshot_console_20261001T041235Z.log` | `/termux-home/coresidency/oneshot_console_20261001T041235Z.log` | 18677 | `312df09b6f278c2243ad4031d32442b0d52d9e51d13598c12d025b59e22a8101` |
| `runs/run_20261001T040522Z_smoke/block_B1_mix_nollm.json` | `/termux-home/coresidency/run_20261001T040522Z_smoke/block_B1_mix_nollm.json` | 15206 | `2c8a151c2de1abdac909d394ca4587ed34ebc7ca7783856582f52166f12c202e` |
| `runs/run_20261001T040522Z_smoke/block_B2_only640_nollm.json` | `/termux-home/coresidency/run_20261001T040522Z_smoke/block_B2_only640_nollm.json` | 15158 | `5f34ce5a753aeee5c3d6e85ad8276847c7b11e1aa93e503e68cb794e8987fb47` |
| `runs/run_20261001T040522Z_smoke/block_B3_mix_gemma.json` | `/termux-home/coresidency/run_20261001T040522Z_smoke/block_B3_mix_gemma.json` | 15666 | `d007bdcd8c7de3714b8a585aa3562155c2c48ac4f422a7de6daeb93c1a26e8fd` |
| `runs/run_20261001T040522Z_smoke/block_B4_only640_gemma.json` | `/termux-home/coresidency/run_20261001T040522Z_smoke/block_B4_only640_gemma.json` | 15182 | `40bd663355926315bd61d8905d2a0ecaf189c0d03a31f98d6668cbf1ec79623c` |
| `runs/run_20261001T040522Z_smoke/block_B5_mtp_ram_snapshot.json` | `/termux-home/coresidency/run_20261001T040522Z_smoke/block_B5_mtp_ram_snapshot.json` | 1872 | `cf3329030656bf2a99b2c5853c0dda729aa91a98eb40c15f94e5e4f7da63bce0` |
| `runs/run_20261001T040522Z_smoke/llama-server-mtp.log` | `/termux-home/coresidency/run_20261001T040522Z_smoke/llama-server-mtp.log` | 3691 | `fd6ec1f4b83c78e34d644a69f58eb98bfacde425bf820650bca61ceb3cefa5d9` |
| `runs/run_20261001T040522Z_smoke/llama-server.log` | `/termux-home/coresidency/run_20261001T040522Z_smoke/llama-server.log` | 5348 | `81d9196c1c92b80456b6bbf4af36d7081e6b67f7861c7383ebf715954d6bd993` |
| `runs/run_20261001T040522Z_smoke/loads.json` | `/termux-home/coresidency/run_20261001T040522Z_smoke/loads.json` | 835 | `b9aed73721a8a8e9033c7d7ce8ac8d6d0181b06aa11bb083095baa1b891de2cc` |
| `runs/run_20261001T040522Z_smoke/report.txt` | `/termux-home/coresidency/run_20261001T040522Z_smoke/report.txt` | 6035 | `72523d49dfdc520feda5a4a0f353eab62e01c3a8b68ca5393aba3a028cf55b70` |
| `runs/run_20261001T040522Z_smoke/run.json` | `/termux-home/coresidency/run_20261001T040522Z_smoke/run.json` | 2847 | `45a3f00c5364d5c7b84d9fb94c791702075d250e34bccd6122df648143ba6ed9` |
| `runs/run_20261001T041743Z/block_B1_mix_nollm.json` | `/termux-home/coresidency/run_20261001T041743Z/block_B1_mix_nollm.json` (redacted, see below) | 110788 | `ee2185bc2aaf031dd48972868551d75d95d5b214ebe716852f676d18cb45e5a9` |
| `runs/run_20261001T041743Z/block_B2_only640_nollm.json` | `/termux-home/coresidency/run_20261001T041743Z/block_B2_only640_nollm.json` (redacted, see below) | 108495 | `4d1f9204969753c39612159d63c63bc303afc8c06c8b621187d0178e38a63e50` |
| `runs/run_20261001T041743Z/block_B3_mix_gemma.json` | `/termux-home/coresidency/run_20261001T041743Z/block_B3_mix_gemma.json` (redacted, see below) | 113853 | `74f79b4d02d15aa677336bc23445e2985e4778b173e19df29650fd6873d41732` |
| `runs/run_20261001T041743Z/block_B4_only640_gemma.json` | `/termux-home/coresidency/run_20261001T041743Z/block_B4_only640_gemma.json` | 105044 | `3a6e15c6b2ebdffc217a2f62bbfa89d447987e6cd63db5ca56f08a1d9d922a82` |
| `runs/run_20261001T041743Z/block_B5_mtp_ram_snapshot.json` | `/termux-home/coresidency/run_20261001T041743Z/block_B5_mtp_ram_snapshot.json` | 1884 | `5b6754978736b3dea8cfe1fa2fcc084229da177b2089a08465dea2c715b5c203` |
| `runs/run_20261001T041743Z/llama-server-mtp.log` | `/termux-home/coresidency/run_20261001T041743Z/llama-server-mtp.log` | 3691 | `ef6d27353b0bd158ef54dc805238014dd4753ef82a0f5a47eb3191813c653eac` |
| `runs/run_20261001T041743Z/llama-server.log` | `/termux-home/coresidency/run_20261001T041743Z/llama-server.log` | 18264 | `74cb5186aeb3131ffbdc33b70b5d0c40944763abc8d8cc5676066245f0b5094c` |
| `runs/run_20261001T041743Z/loads.json` | `/termux-home/coresidency/run_20261001T041743Z/loads.json` | 835 | `c86c4085cd23d337a95a50c1661d1e20bd26f46065287175435d43ae810edf74` |
| `runs/run_20261001T041743Z/report.txt` | `/termux-home/coresidency/run_20261001T041743Z/report.txt` | 6005 | `e710e77c93150c7030274dd0d956cf0fc0a36edd5ca17d281dfe29d59f1a3c15` |
| `runs/run_20261001T041743Z/run.json` | `/termux-home/coresidency/run_20261001T041743Z/run.json` | 2849 | `e7e91fef766d64b3c6dd30b466512ab025baa11fddf0e50e651b3864cebcc4f0` |
| `runs/thermal.log` | `/termux-home/coresidency/thermal.log` | 38562 | `b567ef9c0070ac32a7f6236e1316b3e191be2b65a501f036b8bf48f7bca80a67` |

## Tools (6 files)

On disk before this task (unstaged). Hashes equal CLEANUP1_REPORT.md "New SHA-256", except `SMOKE.md`, whose prose this task corrected (was `3e3cd4e21f476d32e5784ebf51b65b9076cb5cd1b6b233688033d3044cdbabae`, 2882 bytes).

| Path | Bytes | SHA-256 |
|---|---:|---|
| `coresidency.py` | 67834 | `85a6e4cb003bd4f56d501c186a8cbac1b528fb1aa4369380c34be25ad2743626` |
| `oneshot.sh` | 7351 | `efb3939d2a0a24d7e7a7abc2384f6e608b252ff12b2ca539f8bf8753146d41a2` |
| `run_coresidency.sh` | 426 | `dba86f4de4011a2bf6abd5745f3ac709633e717e5a72ff31a721dae0470962b0` |
| `test_coresidency.py` | 49281 | `fca2eacd4880ddc98fe4846cc67719826e5952539e2dffee59b742ed7a48ec45` |
| `test_oneshot.sh` | 8056 | `bccd170dfe71bccdb316aaf86b555defed3bf528dfaef9b854d231c7289f4e19` |
| `SMOKE.md` | 3566 | `6526b06fa99347aa2218f3bca9cdf27822aff95f59bff269e26d63327de2b468` |

## Written for this archive (2 files)

| Path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 7424 | `fbdbfa46d29f3bc7629c1a5cbe34ceff196cfcc53aa4ff083ac2b1a14d1e50a0` |
| `RUN_INDEX.md` | 1858 | `223d18865f7a100050a5068f79050f535c5d4c70846d9aca9ddfa0e65b0e968c` |

## Redactions

Privacy scan (emails, phone numbers, IMEI/serial, Wi-Fi names, locations, tokens/keys, account names, third-party app names in logcat lines): the only finding was the third-party package name of the root manager app in logcat lines (`lmk.raw_head`) of three block JSONs. Each occurrence was replaced by `[redacted-third-party-app]` (JSON still valid; all measured values unchanged). The phone originals keep it.

| Archive path | Occurrences | Phone bytes | Phone SHA-256 | Archive bytes | Archive SHA-256 |
|---|---:|---:|---|---:|---|
| `runs/run_20261001T041743Z/block_B1_mix_nollm.json` | 1 | 110782 | `474c6aeb161ca904c70d896e0adb690bacedb0740dbadf11e2632329189ae047` | 110788 | `ee2185bc2aaf031dd48972868551d75d95d5b214ebe716852f676d18cb45e5a9` |
| `runs/run_20261001T041743Z/block_B2_only640_nollm.json` | 2 | 108483 | `cd24bf8622c2e207a4b64efffc8beee333045c724d764794cb65672dd4f8d119` | 108495 | `4d1f9204969753c39612159d63c63bc303afc8c06c8b621187d0178e38a63e50` |
| `runs/run_20261001T041743Z/block_B3_mix_gemma.json` | 2 | 113841 | `d586a14116879cae3672787603971046212f2c214fdb75a90e22eab54e592930` | 113853 | `74f79b4d02d15aa677336bc23445e2985e4778b173e19df29650fd6873d41732` |

## Not archived (left on the phone; see RUN_INDEX.md)

| Phone path | Bytes | SHA-256 |
|---|---:|---|
| `/termux-home/coresidency/run_20260930T150019Z_smoke/block_B1_mix_nollm.json` | 9561 | `bfcf822b13faefb2f7c64ed669ecd4f0dd5c93e6a08570f4234042cd7dd481f9` |
| `/termux-home/coresidency/run_20260930T150019Z_smoke/block_B2_only640_nollm.json` | 9334 | `c8a7226301276fe9b627cd1a5020c646a4c73d070fab553ada93cf7472fe0377` |
| `/termux-home/coresidency/run_20260930T150019Z_smoke/block_B3_mix_gemma.json` | 9708 | `d2deefc58c0dd870bfa2fa4f6bff08d98cd1d203aa98ba3c0d7dccea7cd67ba1` |
| `/termux-home/coresidency/run_20260930T150019Z_smoke/block_B4_only640_gemma.json` | 9228 | `0b4164035ba11efa3e7c9dab43ecf433a672bbc1799aa552051b64c2d6e70a3f` |
| `/termux-home/coresidency/run_20260930T150019Z_smoke/block_B5_mtp_ram_snapshot.json` | 1606 | `de5c33c16f5fc3fcdbc519c7867204daaf690f62133db5869c3d26a8138d7ed8` |
| `/termux-home/coresidency/run_20260930T150019Z_smoke/llama-server-mtp.log` | 3691 | `cdce6b28d357322ce410da7ad63114abee04f03c753bb7420c24d0d974438413` |
| `/termux-home/coresidency/run_20260930T150019Z_smoke/llama-server.log` | 5347 | `45781c65c43c0fdf513150b97ba48f7c5a5a56b1de100d59f43b704e32216608` |
| `/termux-home/coresidency/run_20260930T150019Z_smoke/loads.json` | 812 | `493b197001a34d0e490ad0768512a431313f48d93b4e536620703c5df9444aa3` |
| `/termux-home/coresidency/run_20260930T150019Z_smoke/report.txt` | 4953 | `cddaae96b6875dd353321541779995aea21aa0bdde51b7fbd607a3189eb7d9ef` |
| `/termux-home/coresidency/run_20260930T150019Z_smoke/run.json` | 2357 | `66299a47862a322f13d1d2e42bd80c93d97901e1728858296e27d89731dd586b` |
| `/termux-home/coresidency/oneshot_console_20260930T145512Z.log` | 7238 | `bf4e0b2e1daa2937e0c6350c8742d8929d53e727929db7a3630198255d55ca2e` |
| `/termux-home/coresidency/run_20260930T182618Z/block_B1_mix_nollm.json` | 63414 | `c317c672e8f247b9408c5d1e6356a75610660e8d12eedffa81d91213e803eb16` |
| `/termux-home/coresidency/run_20260930T182618Z/block_B2_only640_nollm.json` | 2770 | `9478ed26296ff799c2ff9e2a49586c146e000489431056d8d035109b6af2e3e8` |
| `/termux-home/coresidency/run_20260930T182618Z/block_B3_mix_gemma.json` | 2538 | `6781b6fe8826b3eb8ce758ea2e0f8fc54eaba83fbb3874deb671cf09ab16c234` |
| `/termux-home/coresidency/run_20260930T182618Z/block_B4_only640_gemma.json` | 2686 | `7f8b521d2bbc455fe77d7d3b1302e3cfe7f2c3ecc044363be574e42827e34ba3` |
| `/termux-home/coresidency/run_20260930T182618Z/block_B5_mtp_ram_snapshot.json` | 1613 | `b7b1371652bb9f4c9d559a4a9b1c0f6bd41c94094cb4ce031db05af2d0bcfa7e` |
| `/termux-home/coresidency/run_20260930T182618Z/llama-server-mtp.log` | 3691 | `e893523f2c586c21f68aa8cec113c01464c896e8a1f818a2aed13c52c737c614` |
| `/termux-home/coresidency/run_20260930T182618Z/llama-server.log` | 5347 | `5b6f497de0514ba63d904cab3c34810bc8398407c018f6e53f729dc719cfbd77` |
| `/termux-home/coresidency/run_20260930T182618Z/loads.json` | 777 | `347933efe72618853a64e6a4504946d3453fbed22447307d8d0d7da516ead413` |
| `/termux-home/coresidency/run_20260930T182618Z/report.txt` | 5033 | `e3ec8bb3018f1d0c74b67b93a1e99f0262b74adc8ff680854fabdd8bfc1ec152` |
| `/termux-home/coresidency/run_20260930T182618Z/run.json` | 2492 | `6f3a6358b72d8019ac9a2c40e0d94c1c75a94bbd2cf2ac747079a1bec10724e9` |
| `/termux-home/coresidency/oneshot_console_20260930T182110Z.log` | 8152 | `0a0b48cb2ab944f8fa96ff125fc4b3884f96077d44214bf0c9418e663d86ea69` |

Review artifacts (requests, stdout, stderr, frozen.sha per round) are not archived (about 12.5 MB) and stay on the phone: `/termux-home/storage/downloads/coresidency_review_artifacts/` (9.2 MB), `/termux-home/storage/downloads/coresidency_fix1_review_artifacts/` (1.3 MB) and `/termux-home/storage/downloads/cleanup1_review_artifacts/` (2.0 MB). The verdicts are quoted in `reports/`.

## sha256sum format

```sha256sums
fbdbfa46d29f3bc7629c1a5cbe34ceff196cfcc53aa4ff083ac2b1a14d1e50a0  README.md
223d18865f7a100050a5068f79050f535c5d4c70846d9aca9ddfa0e65b0e968c  RUN_INDEX.md
6526b06fa99347aa2218f3bca9cdf27822aff95f59bff269e26d63327de2b468  SMOKE.md
85a6e4cb003bd4f56d501c186a8cbac1b528fb1aa4369380c34be25ad2743626  coresidency.py
efb3939d2a0a24d7e7a7abc2384f6e608b252ff12b2ca539f8bf8753146d41a2  oneshot.sh
4a132c50512ccde40423694779f132f437520f01d6174d523cb8af91c80d3b6d  reports/CLEANUP1_REPORT.md
85b45b1d385889eabd8b129fb8e938cc6b978e620a90d36f59548374cf3498ce  reports/CORESIDENCY_CODER_REPORT.md
87932e0cf7a941849b642dfe497f27545b9479e66d092297a4940aa64503ad61  reports/CORESIDENCY_FIX1_REPORT.md
dba86f4de4011a2bf6abd5745f3ac709633e717e5a72ff31a721dae0470962b0  run_coresidency.sh
e0ad1c5be49cc56d8b52cc5d1e5d9c76cafff63154188086a1eaf17820b1df0f  runs/oneshot_console_20261001T040014Z.log
312df09b6f278c2243ad4031d32442b0d52d9e51d13598c12d025b59e22a8101  runs/oneshot_console_20261001T041235Z.log
2c8a151c2de1abdac909d394ca4587ed34ebc7ca7783856582f52166f12c202e  runs/run_20261001T040522Z_smoke/block_B1_mix_nollm.json
5f34ce5a753aeee5c3d6e85ad8276847c7b11e1aa93e503e68cb794e8987fb47  runs/run_20261001T040522Z_smoke/block_B2_only640_nollm.json
d007bdcd8c7de3714b8a585aa3562155c2c48ac4f422a7de6daeb93c1a26e8fd  runs/run_20261001T040522Z_smoke/block_B3_mix_gemma.json
40bd663355926315bd61d8905d2a0ecaf189c0d03a31f98d6668cbf1ec79623c  runs/run_20261001T040522Z_smoke/block_B4_only640_gemma.json
cf3329030656bf2a99b2c5853c0dda729aa91a98eb40c15f94e5e4f7da63bce0  runs/run_20261001T040522Z_smoke/block_B5_mtp_ram_snapshot.json
fd6ec1f4b83c78e34d644a69f58eb98bfacde425bf820650bca61ceb3cefa5d9  runs/run_20261001T040522Z_smoke/llama-server-mtp.log
81d9196c1c92b80456b6bbf4af36d7081e6b67f7861c7383ebf715954d6bd993  runs/run_20261001T040522Z_smoke/llama-server.log
b9aed73721a8a8e9033c7d7ce8ac8d6d0181b06aa11bb083095baa1b891de2cc  runs/run_20261001T040522Z_smoke/loads.json
72523d49dfdc520feda5a4a0f353eab62e01c3a8b68ca5393aba3a028cf55b70  runs/run_20261001T040522Z_smoke/report.txt
45a3f00c5364d5c7b84d9fb94c791702075d250e34bccd6122df648143ba6ed9  runs/run_20261001T040522Z_smoke/run.json
ee2185bc2aaf031dd48972868551d75d95d5b214ebe716852f676d18cb45e5a9  runs/run_20261001T041743Z/block_B1_mix_nollm.json
4d1f9204969753c39612159d63c63bc303afc8c06c8b621187d0178e38a63e50  runs/run_20261001T041743Z/block_B2_only640_nollm.json
74f79b4d02d15aa677336bc23445e2985e4778b173e19df29650fd6873d41732  runs/run_20261001T041743Z/block_B3_mix_gemma.json
3a6e15c6b2ebdffc217a2f62bbfa89d447987e6cd63db5ca56f08a1d9d922a82  runs/run_20261001T041743Z/block_B4_only640_gemma.json
5b6754978736b3dea8cfe1fa2fcc084229da177b2089a08465dea2c715b5c203  runs/run_20261001T041743Z/block_B5_mtp_ram_snapshot.json
ef6d27353b0bd158ef54dc805238014dd4753ef82a0f5a47eb3191813c653eac  runs/run_20261001T041743Z/llama-server-mtp.log
74cb5186aeb3131ffbdc33b70b5d0c40944763abc8d8cc5676066245f0b5094c  runs/run_20261001T041743Z/llama-server.log
c86c4085cd23d337a95a50c1661d1e20bd26f46065287175435d43ae810edf74  runs/run_20261001T041743Z/loads.json
e710e77c93150c7030274dd0d956cf0fc0a36edd5ca17d281dfe29d59f1a3c15  runs/run_20261001T041743Z/report.txt
e7e91fef766d64b3c6dd30b466512ab025baa11fddf0e50e651b3864cebcc4f0  runs/run_20261001T041743Z/run.json
b567ef9c0070ac32a7f6236e1316b3e191be2b65a501f036b8bf48f7bca80a67  runs/thermal.log
fca2eacd4880ddc98fe4846cc67719826e5952539e2dffee59b742ed7a48ec45  test_coresidency.py
bccd170dfe71bccdb316aaf86b555defed3bf528dfaef9b854d231c7289f4e19  test_oneshot.sh
```
