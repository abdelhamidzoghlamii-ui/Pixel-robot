# Artifacts

Every archived file below with its SHA-256. Copied files were compared byte for byte (`cmp`) with the phone original at archive time, and the phone and archive SHA-256 values match. Recheck from this folder: `sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Copied from the phone (9 files)

| Archive path | Phone source | Bytes | SHA-256 (phone = archive) |
|---|---|---:|---|
| `laya_en.log` | `/termux-home/jevlike/logs/laya_en.log` | 313 | `d4e3475eb87508087cba7b47bb2062226ad8e331881f54bd985a8f8d627d1131` |
| `laya_micro.log` | `/termux-home/jevlike/logs/laya_micro.log` | 313 | `31acbba40ec92c2ac3967b6e571b3d0554dbde3b3b62e7683eed041964706f03` |
| `laya_multi.log` | `/termux-home/jevlike/logs/laya_multi.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `request.txt` | `/termux-home/jevlike/request.txt` | 1415 | `aadd627cf7e3f4ca471783a31397f1031e46b4be6f371483c641af542c5d2abf` |
| `s1o.log` | `/termux-home/jevlike/logs/s1o.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `s1o.server.log` | `/termux-home/jevlike/logs/s1o.server.log` | 2925 | `a5bbb13d3c27ad56e5d04bae5e15511a9f0136e1124b098ea0db10d68fa75d49` |
| `von10_nli.log` | `/termux-home/jevlike/logs/von10_nli.log` | 1139 | `4ae06dad002f046dcfb8d281bb005e0e3f5757efb223052ecd39090c74fa2f6e` |
| `von11.log` | `/termux-home/jevlike/logs/von11.log` | 957 | `2252c13986ee5dce62b87c05b2bb104b596bb45b57b1521ca6d0ec3a323e329c` |
| `von12.log` | `/termux-home/jevlike/logs/von12.log` | 944 | `1476afdd2642e72c78756cd12ef5d840d17595955698460b6cd3c4fb5488a35a` |

## Written for this archive or already in place in the repository (1 files)

| Archive path | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 759 | `cb1fe50c32e4039f55c42038086084299102542129bdcab8b851cc57d7807188` |

## sha256sum format

```sha256sums
cb1fe50c32e4039f55c42038086084299102542129bdcab8b851cc57d7807188  README.md
d4e3475eb87508087cba7b47bb2062226ad8e331881f54bd985a8f8d627d1131  laya_en.log
31acbba40ec92c2ac3967b6e571b3d0554dbde3b3b62e7683eed041964706f03  laya_micro.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  laya_multi.log
aadd627cf7e3f4ca471783a31397f1031e46b4be6f371483c641af542c5d2abf  request.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  s1o.log
a5bbb13d3c27ad56e5d04bae5e15511a9f0136e1124b098ea0db10d68fa75d49  s1o.server.log
4ae06dad002f046dcfb8d281bb005e0e3f5757efb223052ecd39090c74fa2f6e  von10_nli.log
2252c13986ee5dce62b87c05b2bb104b596bb45b57b1521ca6d0ec3a323e329c  von11.log
1476afdd2642e72c78756cd12ef5d840d17595955698460b6cd3c4fb5488a35a  von12.log
```
