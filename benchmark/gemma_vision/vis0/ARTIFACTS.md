# Artifacts — VIS-0

Every file in this folder except this one is listed with its bytes and SHA-256 (archival task VIS-0/M2-Q1
catalog completion, 2026-10-24; 22 files). Files marked `tracked (685a44b)` were already in Git and were not
recopied; their working-tree bytes equal `HEAD`. Review copies were compared with their phone originals by `cmp`.
`/termux-home` is the native Termux home (`/data/data/com.termux/files/home`). ARTIFACTS.md excludes its own
size and hash. Recheck from this folder:
`sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Archived files (22)

| Archive path | Provenance | Bytes | SHA-256 |
|---|---|---:|---|
| `README.md` | Coder-authored source/doc in place; tracked (685a44b) | 4398 | `da2334b84d212bb11c15e78d0fbf84e869c06772a979713dcd94f96df0078fdf` |
| `RUN_INDEX.md` | archive documentation (this task) | 4214 | `85a1ecf644308fe480e5f7b7323e51934900a21391c4800330a91a7a5c7333a0` |
| `grading/VIS0_LOCALAI_NOTES.md` | Local AI grading notes; no phone original outside the repo; tracked (685a44b) | 1706 | `4fa80f5a765b0e742539f975414b30a5cb43636cdfbed687ee97b3b9cb1c88cf` |
| `reports/VIS0_CODER_REPORT.md` | `/termux-home/review/vis0/VIS0_CODER_REPORT.md` (cmp equal); tracked (685a44b) | 7318 | `b8c96d51f6dd32e9bab60187f08e86f0f1a938e7dc9285b1d3009bf6fbd4125c` |
| `reports/round1_stderr.txt` | `/termux-home/review/vis0/round1_stderr.txt` (cmp equal); untracked until this task (local copy dated 2026-10-19 07:23Z) | 262194 | `77ece86c543be60c6345dd89fb566794163f914489add72b3d21e8352238e2df` |
| `reports/round1_stdout.txt` | `/termux-home/review/vis0/round1_stdout.txt` (cmp equal); tracked (685a44b) | 5969 | `20ef7a6a30816792daf9d601f0cc8b8f7895e822fe1542cdb8f785af85cd0985` |
| `reports/round2_stderr.txt` | `/termux-home/review/vis0/round2_stderr.txt` (cmp equal); untracked until this task (local copy dated 2026-10-19 07:23Z) | 265003 | `e2f6f69a14f993e09d7790b51b535869b48cf4f657882a76aa27c43c621b6ef2` |
| `reports/round2_stdout.txt` | `/termux-home/review/vis0/round2_stdout.txt` (cmp equal); tracked (685a44b) | 1740 | `05536bb1e301864a01cb715e2963edb0fbbcec3f8d309ffce06149f40dfa4bc8` |
| `reports/round3_stderr.txt` | `/termux-home/review/vis0/round3_stderr.txt` (cmp equal); untracked until this task (local copy dated 2026-10-19 07:23Z) | 853579 | `056164027677ceda11326f7147c2428e0abb7a9c09913d20c4ff33aec9c4e258` |
| `reports/round3_stdout.txt` | `/termux-home/review/vis0/round3_stdout.txt` (cmp equal); tracked (685a44b) | 2110 | `9381ce790bd9a3ae3b55852bf92a90024090be0b92cf54eed08077621c1cfaa7` |
| `runs/20261018T004817Z/manifest.json` | written in place by the harness run `20261018T004817Z`; tracked (685a44b) | 7762 | `6a602bb81f8533196bd41f3b4793b560acd8e1c1c795e37a16e1fa36a7bec79b` |
| `runs/20261018T004817Z/results.json` | written in place by the harness run `20261018T004817Z`; tracked (685a44b) | 50145 | `c87fe2c54063ddbc65cfc12f21499d78eb193539360dbef53a8977d2d6da7bdc` |
| `runs/20261018T004817Z/run.log` | written in place by the harness run `20261018T004817Z`; tracked (685a44b) | 2296 | `8a0620cc1ecc178cc67d6a700e7b46cde70fbb911185f4c9d111e72795f4768b` |
| `runs/20261018T004817Z/server.log` | written in place by the harness run `20261018T004817Z`; tracked (685a44b) | 206742 | `8a09927818c11cf65d0724a865facb6683a73519bd277e745af25a9881ac5bf2` |
| `runs/20261018T004817Z/vis0_sheet.pdf` | written in place by the harness run `20261018T004817Z`; tracked (685a44b) | 338852 | `81ba2ac06737fb8e777d2acdbb1f9f35ce75406533c4f991787fc55b23c61aaa` |
| `runs/20261018T010701Z/manifest.json` | written in place by the harness run `20261018T010701Z`; tracked (685a44b) | 12241 | `984969bfd37e783af2d23909aa552996d28c21f144875aeb5d56aed339e26ec6` |
| `runs/20261018T010701Z/results.json` | written in place by the harness run `20261018T010701Z`; tracked (685a44b) | 104809 | `aeabeaa1957fc9602de60e2e73fb003113cd283cd368c31d2f1587f218dfab44` |
| `runs/20261018T010701Z/run.log` | written in place by the harness run `20261018T010701Z`; tracked (685a44b) | 7522 | `4155b446f84f4fe6ded65196cbaac24b1b160ba204237d764876a15a2ae8cd0b` |
| `runs/20261018T010701Z/server.log` | written in place by the harness run `20261018T010701Z`; tracked (685a44b) | 318516 | `5f7ea0f5a23fea6be2750f2f5da8d4fcfec7fdbc9ac30be84ef02eb275a1950d` |
| `runs/20261018T010701Z/vis0_sheet.pdf` | written in place by the harness run `20261018T010701Z`; tracked (685a44b) | 1184772 | `5aa8cd5bbde16daba5f2e195dca10289dc6a4331453ec8c33fe5ef33636f990f` |
| `test_vis0.py` | Coder-authored source/doc in place; tracked (685a44b) | 20188 | `0bc938d604278317633175110dd2e7119fcfe98887fe59d547ba9886a987ad21` |
| `vis0.py` | Coder-authored source/doc in place; tracked (685a44b) | 56889 | `ea3d1e38477b0ff2f85a04ed828da3c60ce7264d805b409f7c0d84e010aec50a` |

## Phone-only review evidence (not archived)

Kept in `/termux-home/review/vis0/` and not copied, so these hashes pin them without making them archive files.
The isolated review copies `repo/` (round 1), `round2_copy/` and `round3_copy/` (164–165 MB each) are not hashed here.

| Phone path | Bytes | SHA-256 |
|---|---:|---|
| `/termux-home/review/vis0/round1_audit.json` | 1828 | `2615c0bd54a76d3cb4a305cf34089f2656c144d1251107d5be0ed43d190a19a4` |
| `/termux-home/review/vis0/round2_audit.json` | 1835 | `85d59c04abe9a3db8f7d4d9ff11c1e2d8ef8bb019b662a5e64645c2866067f35` |
| `/termux-home/review/vis0/round3_audit.json` | 1835 | `92e7a4b754c34309563037caa47b610c6aa783a1faadb6d976c558134dfe3c1c` |
| `/termux-home/review/vis0/run_review_round1.py` | 8840 | `1b979e57d00f9fba07df4bcbc6831588d9136775bc94a84fcd94c7234b1f6323` |
| `/termux-home/review/vis0/run_review_round2.py` | 11183 | `e542df14c0cc0dca8f82cee8af84826f2bac81910211d244a6308f953734869a` |
| `/termux-home/review/vis0/run_review_round3.py` | 10786 | `c38eb2019579403ff2e3219feb89cd1e24ff28ddd33900d2f781939b10082697` |

Owner Download copies (`cmp` equal to the archived run files on 2026-10-24): `/storage/emulated/0/Download/vis0/vis0_sheet.pdf`
= `runs/20261018T010701Z/vis0_sheet.pdf`; `/storage/emulated/0/Download/vis0/vis0_sheet_20261018T004817Z.pdf` =
`runs/20261018T004817Z/vis0_sheet.pdf`. The input frames are tracked under
`benchmark/relate_anything/quality1/runs/20261017T213604Z/frames/` and catalogued in that folder's ARTIFACTS.md.
Model, projector and build binaries stay outside Git; their identities are in `README.md` and the run manifests.

```sha256sums
da2334b84d212bb11c15e78d0fbf84e869c06772a979713dcd94f96df0078fdf  README.md
85a1ecf644308fe480e5f7b7323e51934900a21391c4800330a91a7a5c7333a0  RUN_INDEX.md
4fa80f5a765b0e742539f975414b30a5cb43636cdfbed687ee97b3b9cb1c88cf  grading/VIS0_LOCALAI_NOTES.md
b8c96d51f6dd32e9bab60187f08e86f0f1a938e7dc9285b1d3009bf6fbd4125c  reports/VIS0_CODER_REPORT.md
77ece86c543be60c6345dd89fb566794163f914489add72b3d21e8352238e2df  reports/round1_stderr.txt
20ef7a6a30816792daf9d601f0cc8b8f7895e822fe1542cdb8f785af85cd0985  reports/round1_stdout.txt
e2f6f69a14f993e09d7790b51b535869b48cf4f657882a76aa27c43c621b6ef2  reports/round2_stderr.txt
05536bb1e301864a01cb715e2963edb0fbbcec3f8d309ffce06149f40dfa4bc8  reports/round2_stdout.txt
056164027677ceda11326f7147c2428e0abb7a9c09913d20c4ff33aec9c4e258  reports/round3_stderr.txt
9381ce790bd9a3ae3b55852bf92a90024090be0b92cf54eed08077621c1cfaa7  reports/round3_stdout.txt
6a602bb81f8533196bd41f3b4793b560acd8e1c1c795e37a16e1fa36a7bec79b  runs/20261018T004817Z/manifest.json
c87fe2c54063ddbc65cfc12f21499d78eb193539360dbef53a8977d2d6da7bdc  runs/20261018T004817Z/results.json
8a0620cc1ecc178cc67d6a700e7b46cde70fbb911185f4c9d111e72795f4768b  runs/20261018T004817Z/run.log
8a09927818c11cf65d0724a865facb6683a73519bd277e745af25a9881ac5bf2  runs/20261018T004817Z/server.log
81ba2ac06737fb8e777d2acdbb1f9f35ce75406533c4f991787fc55b23c61aaa  runs/20261018T004817Z/vis0_sheet.pdf
984969bfd37e783af2d23909aa552996d28c21f144875aeb5d56aed339e26ec6  runs/20261018T010701Z/manifest.json
aeabeaa1957fc9602de60e2e73fb003113cd283cd368c31d2f1587f218dfab44  runs/20261018T010701Z/results.json
4155b446f84f4fe6ded65196cbaac24b1b160ba204237d764876a15a2ae8cd0b  runs/20261018T010701Z/run.log
5f7ea0f5a23fea6be2750f2f5da8d4fcfec7fdbc9ac30be84ef02eb275a1950d  runs/20261018T010701Z/server.log
5aa8cd5bbde16daba5f2e195dca10289dc6a4331453ec8c33fe5ef33636f990f  runs/20261018T010701Z/vis0_sheet.pdf
0bc938d604278317633175110dd2e7119fcfe98887fe59d547ba9886a987ad21  test_vis0.py
ea3d1e38477b0ff2f85a04ed828da3c60ce7264d805b409f7c0d84e010aec50a  vis0.py
```
