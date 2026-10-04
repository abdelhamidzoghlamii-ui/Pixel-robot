# RelateAnything benchmark

Motors-off desk checks, export/parity tooling and owner speed-block evidence.
M2 is the owner's benchmark candidate; it is not deployed in the robot.

Licence placement rule: upstream AGPL source, restricted model weights, exported
graphs, vocabulary banks and the export venv stay outside this repository.
Only our harness and evidence belong here. This is not APK integration approval.

Run the checks from native Termux at the repository root:

```sh
python benchmark/relate_anything/desk2/self_check.py
python benchmark/relate_anything/desk2/speed_block.py --model relsgg-vits16plus --dry-run
python benchmark/relate_anything/desk2/speed_block.py --model relsgg-vits16 --dry-run
```

Dry runs are NOT VALID TIMING and do not sample power. Read
[desk2/RUN.md](desk2/RUN.md) before any owner-only full run after exiting agents.
External model/source paths are declared in `desk2/desk_check.py`.
See [RUN_INDEX.md](RUN_INDEX.md) for owner measurements and their validity labels,
and [ASSET_INDEX.json](ASSET_INDEX.json) for shared image paths and SHA-256s.
