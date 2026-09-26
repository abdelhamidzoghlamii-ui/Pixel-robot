Families: room_finished, hall_hint, repeat_search, heard_from_room

| Run | Candidate | Cores/threads | Cases | Acceptable | Preferred | Acc. rev | Pref. rev | Flips | Load ms | First-call ms | Warm p50 ms | Warm p95 ms | Peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| von11_c47_t2 | von11 | 4-7/2 | 24 | 24 | 8 | 24 | 6 | 24/24 | 16718.7 | 8084.3 | 3651.3 | 4472.2 | 3316.4 |
| von11_c47_t3 | von11 | 4-7/3 | 24 | 24 | 8 | 24 | 6 | 24/24 | 19405.4 | 9200.9 | 3157.0 | 3331.1 | 3078.2 |
| von11_c47_t4 | von11 | 4-7/4 | 24 | 24 | 8 | 24 | 6 | 24/24 | 21212.4 | 7665.7 | 2753.3 | 2945.3 | 3360.1 |
| von11_c67_t2 | von11 | 6-7/2 | 24 | 24 | 8 | 24 | 6 | 24/24 | 20193.6 | 13103.2 | 4029.5 | 5056.4 | 3062.6 |
| von11_perm | von11_perm | 4-7/4 | 24 | 24 | 7 | 24 | 7 | 24/24 | 35082.7 | 22553.7 | 13771.7 | 28048.8 | 3370.9 |
| von10_nli | von10_nli | 4-7/4 | 24 | 24 | 6 | 24 | 6 | 0/24 | 33273.9 | 7196.8 | 8812.9 | 13653.7 | 1845.4 |

Per family, canonical order. Each cell: acceptable y/n · preferred x/n.

| Family | von11_c47_t2/von11 | von11_c47_t3/von11 | von11_c47_t4/von11 | von11_c67_t2/von11 | von11_perm/von11_perm | von10_nli/von10_nli |
|---|---:|---:|---:|---:|---:|---:|
| room_finished | A 6/6 · P 3/6 | A 6/6 · P 3/6 | A 6/6 · P 3/6 | A 6/6 · P 3/6 | A 6/6 · P 3/6 | A 6/6 · P 0/6 |
| hall_hint | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 3/6 |
| repeat_search | A 6/6 · P 1/6 | A 6/6 · P 1/6 | A 6/6 · P 1/6 | A 6/6 · P 1/6 | A 6/6 · P 0/6 | A 6/6 · P 3/6 |
| heard_from_room | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 0/6 |

- von11_c47_t2/von11: exit 0; argmax == native choice 48/48 calls

- von11_c47_t3/von11: exit 0; argmax == native choice 48/48 calls

- von11_c47_t4/von11: exit 0; argmax == native choice 48/48 calls

- von11_c67_t2/von11: exit 0; argmax == native choice 48/48 calls

- von11_perm/von11_perm: exit 0; rotations per decision 5-5; batched==sequential check on 8 decisions, max |diff| 4.995436668397968e-05

- von10_nli/von10_nli: exit 0; argmax == native choice 48/48 calls
