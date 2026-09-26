Families: room_finished, hall_hint, repeat_search, heard_from_room

| Run | Candidate | Cores/threads | Cases | Acceptable | Preferred | Acc. rev | Pref. rev | Flips | Load ms | First-call ms | Warm p50 ms | Warm p95 ms | Peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| onnx_c47_t2 | laya_en_onnx | 4-7/2 | 24 | 24 | 10 | 7 | 2 | 18/24 | 2659.7 | 2377.6 | 3246.8 | 3895.0 | 1645.1 |
| onnx_c47_t3 | laya_en_onnx | 4-7/3 | 24 | 24 | 10 | 7 | 2 | 18/24 | 3221.7 | 2775.9 | 3090.8 | 3265.1 | 1645.1 |
| onnx_c47_t4 | laya_en_onnx | 4-7/4 | 24 | 24 | 10 | 7 | 2 | 18/24 | 2637.6 | 2041.8 | 2092.3 | 2721.7 | 1646.4 |
| onnx_c67_t2 | laya_en_onnx | 6-7/2 | 24 | 24 | 10 | 7 | 2 | 18/24 | 3777.0 | 3062.1 | 2979.2 | 3765.5 | 1645.1 |
| onnx_c67_t3 | laya_en_onnx | 6-7/3 | 24 | 24 | 10 | 7 | 2 | 18/24 | 3692.0 | 4357.9 | 3781.0 | 4516.2 | 1641.6 |
| onnx_c67_t4 | laya_en_onnx | 6-7/4 | 24 | 24 | 10 | 7 | 2 | 18/24 | 4242.7 | 5727.7 | 4303.1 | 5591.8 | 1639.2 |

Per family, canonical order. Each cell: acceptable y/n · preferred x/n.

| Family | onnx_c47_t2/laya_en_onnx | onnx_c47_t3/laya_en_onnx | onnx_c47_t4/laya_en_onnx | onnx_c67_t2/laya_en_onnx | onnx_c67_t3/laya_en_onnx | onnx_c67_t4/laya_en_onnx |
|---|---:|---:|---:|---:|---:|---:|
| room_finished | A 6/6 · P 3/6 | A 6/6 · P 3/6 | A 6/6 · P 3/6 | A 6/6 · P 3/6 | A 6/6 · P 3/6 | A 6/6 · P 3/6 |
| hall_hint | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 | A 6/6 · P 2/6 |
| repeat_search | A 6/6 · P 1/6 | A 6/6 · P 1/6 | A 6/6 · P 1/6 | A 6/6 · P 1/6 | A 6/6 · P 1/6 | A 6/6 · P 1/6 |
| heard_from_room | A 6/6 · P 4/6 | A 6/6 · P 4/6 | A 6/6 · P 4/6 | A 6/6 · P 4/6 | A 6/6 · P 4/6 | A 6/6 · P 4/6 |

- onnx_c47_t2/laya_en_onnx: exit 0; argmax == native choice 48/48 calls

- onnx_c47_t3/laya_en_onnx: exit 0; argmax == native choice 48/48 calls

- onnx_c47_t4/laya_en_onnx: exit 0; argmax == native choice 48/48 calls

- onnx_c67_t2/laya_en_onnx: exit 0; argmax == native choice 48/48 calls

- onnx_c67_t3/laya_en_onnx: exit 0; argmax == native choice 48/48 calls

- onnx_c67_t4/laya_en_onnx: exit 0; argmax == native choice 48/48 calls
