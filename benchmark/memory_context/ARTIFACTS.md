# Artifacts — MEM-1

**No run artifacts archived in GitHub yet.** Owner reports a completed
phone-local MEM-1 primary run with a manifest check `MEM1_ALL_HASHES_OK`;
original artifact bytes and checksums have not been transferred or verified
here. Do not add hash rows without the originals. When runs are archived, each original is copied byte for byte from
`~/robot_runs/mem1/` (`cp -p`), compared with `cmp`, and listed below with its SHA-256. Every run folder already
carries the runner's own `artifacts.sha256` (all files of the folder at the end of the run); recheck it in place with
`sha256sum -c artifacts.sha256`. `README.md`, `RUN_INDEX.md` and `ARTIFACTS.md` are documentation and stay out of
the check block. Recheck from this folder:
`sed -n '/^```sha256sums/,/^```$/p' ARTIFACTS.md | grep -v '^```' | sha256sum -c`.

## Source (frozen candidate)

```sha256sums
ff54bb646836f186f90b5d032413eb21328d778d9b794c2aacc3a78fc0a4b634  build_set.py
2c636eb6a2588e4307c83e6d5552b3651c12e44d5f2ebfd96e0ab5d2d1d78c92  mem1.py
8feb74f97429417e8e504be1abce979b958950e1b80692b332b7d88cde8f8df4  memory_set.json
84e57b8878d350b339ff92e2f0148c096d317d8fafa0c72cc8a6050abf28b4ce  oneshot.sh
e338e03c7342a6976842889c1fc2a28227f3bfa57fb9e272735474b960df3f2f  test_mem1.py
```

## Archived originals (0)

None.
