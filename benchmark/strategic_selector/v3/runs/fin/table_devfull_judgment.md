Families: room_finished, hall_hint, repeat_search, heard_from_room

| Run | Candidate | Cores/threads | Cases | Acceptable | Preferred | Acc. rev | Pref. rev | Flips | Load ms | First-call ms | Warm p50 ms | Warm p95 ms | Peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev-full | laya_en | 4-7/4 | 24 | 24 | 10 | 7 | 2 | 18/24 | 28487.7 | 5265.7 | 3647.0 | 5012.8 | 2810.9 |
| dev-full | laya_micro | 4-7/4 | 24 | 4 | 0 | 13 | 7 | 15/24 | 2028.3 | 1965.5 | 1425.3 | 1568.1 | 540.0 |
| dev-full | laya_multi | 4-7/4 | 24 | 8 | 3 | 14 | 7 | 6/24 | 33385.6 | 1104.7 | 1221.5 | 1297.8 | 2585.4 |
| dev-full | s1o | 4-7/4 | 24 | 24 | 12 | 20 | 9 | 11/24 | 7717.0 | 29138.9 | 28296.9 | 34173.2 | 3665.9 |
| dev-full | von11 | 4-7/4 | 24 | 24 | 8 | 24 | 6 | 24/24 | 23329.3 | 7973.9 | 2921.3 | 5267.8 | 3362.8 |

Per family, canonical order. Each cell: acceptable y/n · preferred x/n.

| Family | dev-full/laya_en | dev-full/laya_micro | dev-full/laya_multi | dev-full/s1o | dev-full/von11 |
|---|---:|---:|---:|---:|---:|
| room_finished | A 6/6 · P 3/6 | A 0/6 · P 0/6 | A 1/6 · P 1/6 | A 6/6 · P 3/6 | A 6/6 · P 3/6 |
| hall_hint | A 6/6 · P 2/6 | A 0/6 · P 0/6 | A 3/6 · P 1/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 |
| repeat_search | A 6/6 · P 1/6 | A 4/6 · P 0/6 | A 4/6 · P 1/6 | A 6/6 · P 1/6 | A 6/6 · P 1/6 |
| heard_from_room | A 6/6 · P 4/6 | A 0/6 · P 0/6 | A 0/6 · P 0/6 | A 6/6 · P 6/6 | A 6/6 · P 2/6 |

- dev-full/laya_en: exit 0; argmax == native choice 48/48 calls

- dev-full/laya_micro: exit 0; tokenizer parity vs stock 0/48 calls; argmax == native choice 48/48 calls

- dev-full/laya_multi: exit 0; argmax == native choice 48/48 calls

- dev-full/s1o: exit 0; letter mass in full vocab min 0.9997 median 1.0000; prompt tokens median 329 (min 323, max 340); server {'server_exit': 0, 'server_peak_rss_mib_wait4': 3665.9}

- dev-full/von11: exit 0; argmax == native choice 48/48 calls
