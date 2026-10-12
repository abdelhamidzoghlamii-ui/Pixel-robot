# Campaign archive and reusable results

Offline archive of Phase 1 layout screening and Phase 2/3 L2 endurance, fixed and adaptive sessions. This is benchmark evidence, not deployment approval. [RUN.md](RUN.md) gives the executed owner workflow; [RUN_INDEX.md](RUN_INDEX.md) preserves original labels.

The valid measured P23 modes used 4.399329 / 3.303409 / 3.541867 W, inference duty 0.996793 / 0.628934 / 0.629003, and max skin 40.757202 / 38.022205 / 37.434700 C (endurance/fixed/adaptive). Phase 1 retained L0/L1/L3 VALID; L2/L4/final L0 were NOT COMPARABLE — START TEMP. See [generated results and caveats](CAMPAIGN_RESULTS.md) and [exact JSON](CAMPAIGN_RESULTS.json). Picker accuracy is NOT EVALUATED. Start skins differ and fixed had a short rest; boot continuity and ambient are UNVERIFIED in the recorded metadata.

[Run cards](runs/cards/) hold recorded metadata, missing-value reasons and evidence hashes. P23 source hashes match the named historical revision; Phase 1 invocation JSONs have no code hashes, so their matching revision is unknown. [Queryable CSVs and example](data/README.md) and [column definitions](data/schema.md) support offline reuse. [ARTIFACTS.md](ARTIFACTS.md) lists file hashes, and [reports/INDEX.md](reports/INDEX.md) indexes unchanged public-safe historical reports. One historical coder report contains an uncommitted email and is inventory-only; see [sweep inventory](../UNARCHIVED_INVENTORY.md) and [all decisions](checks/p23d/sweep_decisions.json).

## Rebuild and verify

```sh
python3 -B benchmark/campaign/campaign_data.py
python3 -B benchmark/campaign/summarize_campaign.py
python3 -B benchmark/campaign/self_check_p23d.py
```

Generation uses only archived bytes, the existing power.weighted function and local Git history through the P23D base c0ba141. It does not load models or start hardware. The two generator entry points intentionally produce the same cards, datasets and summary. Stdlib only. CSV loaders return strings and empty fields for null.

## Rerun command and raw-data location

Offline rerun command: `python3 -B benchmark/campaign/campaign_data.py` ([campaign_data.py](campaign_data.py)). Hardware reruns remain owner-only and use the exact commands in [RUN.md](RUN.md); none was executed for P23D. Raw bytes: [runs/](runs/), with source paths, byte sizes and hashes in [session inventory](runs/archive_sha256_p23_sessions.json) and earlier inventories listed by [RUN_INDEX.md](RUN_INDEX.md).

P23E archived the three P23D external files; `self_check_p23d.py` is historical (passes at `a34087c`); use `self_check_p23e.py` for the current tree.
