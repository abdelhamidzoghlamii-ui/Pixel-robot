| Candidate | Cases | Preferred | Acceptable | Pref. rev | Acc. rev | Invalid | Order flips | Ties | Load ms | First-call ms | Warm p50 ms | Warm p95 ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| laya_en | 66 | 34 | 54 | 25 | 36 | 0 | 19/66 | 0 | 28487.7 | 5265.7 | 3497.7 | 4699.6 | 2810.9 |
| laya_micro | 66 | 22 | 32 | 31 | 43 | 0 | 29/66 | 0 | 2028.3 | 1965.5 | 1306.3 | 1535.9 | 540.0 |
| laya_multi | 66 | 26 | 37 | 21 | 34 | 0 | 17/66 | 0 | 33385.6 | 1104.7 | 1082.5 | 1326.5 | 2585.4 |
| s1o | 66 | 46 | 64 | 38 | 55 | 0 | 16/66 | 0 | 7717.0 | 29138.9 | 27173.5 | 30903.3 | 3665.9 |
| von11 | 66 | 33 | 49 | 28 | 46 | 0 | 55/66 | 0 | 23329.3 | 7973.9 | 2806.2 | 5212.9 | 3362.8 |

Per family, canonical order. Each cell: preferred x/n · acceptable y/n.

| Family | laya_en | laya_micro | laya_multi | s1o | von11 |
|---|---:|---:|---:|---:|---:|
| new_room | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 5/6 · A 5/6 | P 6/6 · A 6/6 | P 0/6 · A 0/6 |
| room_finished | P 3/6 · A 6/6 | P 0/6 · A 0/6 | P 1/6 · A 1/6 | P 3/6 · A 6/6 | P 3/6 · A 6/6 |
| hall_hint | P 2/6 · A 6/6 | P 0/6 · A 0/6 | P 1/6 · A 3/6 | P 2/6 · A 6/6 | P 2/6 · A 6/6 |
| target_confirmed | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 |
| possible_person | P 0/6 · A 6/6 | P 0/6 · A 6/6 | P 0/6 · A 6/6 | P 0/6 · A 6/6 | P 0/6 · A 0/6 |
| all_rooms_first | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 2/6 · A 2/6 |
| all_rooms_called | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 4/6 · A 4/6 | P 6/6 · A 6/6 |
| localization_lost | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 6/6 · A 6/6 | P 5/6 · A 5/6 |
| route_blocked | P 6/6 · A 6/6 | P 4/6 · A 4/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 | P 6/6 · A 6/6 |
| repeat_search | P 1/6 · A 6/6 | P 0/6 · A 4/6 | P 1/6 · A 4/6 | P 1/6 · A 6/6 | P 1/6 · A 6/6 |
| heard_from_room | P 4/6 · A 6/6 | P 0/6 · A 0/6 | P 0/6 · A 0/6 | P 6/6 · A 6/6 | P 2/6 · A 6/6 |

- laya_en: exit 0; argmax == native choice 132/132 calls

- laya_micro: exit 0; tokenizer parity vs stock 0/132 calls; argmax == native choice 132/132 calls

- laya_multi: exit 0; argmax == native choice 132/132 calls

- s1o: exit 0; A-letter mass in full vocab min 0.6302 median 0.9999; server {'server_exit': 0, 'server_peak_rss_mib_wait4': 3665.9}

- von11: exit 0; argmax == native choice 132/132 calls
