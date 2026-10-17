# Storage breakdown — 2026-10-16 UTC

Data filesystem: 118.02 GB total; 111.74 GB used; 6.14 GB free (decimal GB). Shared storage and Termux use the same filesystem.

| Location | GB |
|---|---:|
| Termux models | 43.98 |
| Other Termux home files | 29.53 |
| Debian /root (agent files, tools, caches) | 8.10 |
| Debian and native Termux installed software/var | 3.14 |
| Debian /tmp | 2.13 |
| Proot /.l2s files with no hard links | 2.04 |
| Readable shared phone storage | 1.12 |
| Not attributable from this accessible inventory | approximately 21.71 |

Other Termux home includes: yolo_bench 7.12 GB, sel-candidates 5.55 GB, laya-test 3.46 GB, von-test 3.17 GB, laya-onnx-test 2.00 GB, clean1 1.35 GB, ext 1.23 GB, laya-q8-test 1.20 GB, robot repo 1.06 GB, laya-web-browser 0.82 GB, plus builds and smaller files.

Debian /root includes .codex 3.85 GB, .npm 1.81 GB, .local 1.00 GB, .cache 0.64 GB and .claude 0.55 GB.

/.l2s occupies 2.57 GB in total, but 0.53 GB has multiple hard links and may overlap other categories; only the 2.04 GB with single links is counted separately above. Counts are rounded and scans were sequential, not an atomic whole-device snapshot. Private Android app data and Android/data are inaccessible to this session; the residual is not an independently measured Android-app total. It may also include unscanned files and filesystem overhead. No root used. No additional deletions performed during this investigation.

The inventory cannot establish when 50 GB was free: no historical disk-usage snapshot was supplied. Current usage is explained largely by model files and accumulated experiments, rather than shared photos/downloads (the readable shared-storage total is only 1.12 GB).
