# repeat_search development rows (frame filtered_text), finalists

## dev_repeat_search_0 — normal

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: searched_no_target; toilet: unsearched. Reachable rooms: bedroom (8.8 m), kitchen (7.8 m), toilet (1.7 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 193 cm; ultrasonic front: 195 cm; sensor age 0.35 s. The room search already repeated without finding Chiara.

Offered (in order): `call_chiara`, `ask_gemma`, `travel:bedroom`, `travel:kitchen`, `travel:toilet`

Label: preferred `travel:toilet`; acceptable ['travel:bedroom', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:bedroom` | False | True | call_chiara 0.332, ask_gemma 0.056, travel:bedroom 0.422, travel:kitchen 0.088, travel:toilet 0.102 |
| s1o | `travel:bedroom` | False | True | call_chiara 0.012, ask_gemma 0.003, travel:bedroom 0.977, travel:kitchen 0.008, travel:toilet 0.000 |
| von11 | `travel:bedroom` | False | True | call_chiara 0.065, ask_gemma 0.130, travel:bedroom 0.479, travel:kitchen 0.175, travel:toilet 0.150 |

## dev_repeat_search_0 — reverse

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: searched_no_target; toilet: unsearched. Reachable rooms: bedroom (8.8 m), kitchen (7.8 m), toilet (1.7 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 193 cm; ultrasonic front: 195 cm; sensor age 0.35 s. The room search already repeated without finding Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:bedroom`, `ask_gemma`, `call_chiara`

Label: preferred `travel:toilet`; acceptable ['travel:bedroom', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:bedroom` | False | True | travel:toilet 0.136, travel:kitchen 0.158, travel:bedroom 0.340, ask_gemma 0.090, call_chiara 0.276 |
| s1o | `ask_gemma` | False | False | travel:toilet 0.053, travel:kitchen 0.014, travel:bedroom 0.013, ask_gemma 0.541, call_chiara 0.379 |
| von11 | `travel:toilet` | True | True | travel:toilet 0.267, travel:kitchen 0.253, travel:bedroom 0.212, ask_gemma 0.151, call_chiara 0.117 |

## dev_repeat_search_1 — normal

State: Mission: find Chiara. Current room: kitchen. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: searched_no_target; living_room: unsearched; toilet: unsearched. Reachable rooms: bedroom (1.5 m), living_room (5.0 m), toilet (7.0 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 218 cm; ultrasonic front: 222 cm; sensor age 0.27 s. The room search already repeated without finding Chiara.

Offered (in order): `call_chiara`, `ask_gemma`, `travel:bedroom`, `travel:living_room`, `travel:toilet`

Label: preferred `travel:bedroom`; acceptable ['travel:bedroom', 'travel:living_room', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:bedroom` | True | True | call_chiara 0.295, ask_gemma 0.061, travel:bedroom 0.399, travel:living_room 0.127, travel:toilet 0.118 |
| s1o | `travel:bedroom` | True | True | call_chiara 0.006, ask_gemma 0.001, travel:bedroom 0.959, travel:living_room 0.033, travel:toilet 0.000 |
| von11 | `travel:bedroom` | True | True | call_chiara 0.079, ask_gemma 0.154, travel:bedroom 0.399, travel:living_room 0.202, travel:toilet 0.166 |

## dev_repeat_search_1 — reverse

State: Mission: find Chiara. Current room: kitchen. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: searched_no_target; living_room: unsearched; toilet: unsearched. Reachable rooms: bedroom (1.5 m), living_room (5.0 m), toilet (7.0 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 218 cm; ultrasonic front: 222 cm; sensor age 0.27 s. The room search already repeated without finding Chiara.

Offered (in order): `travel:toilet`, `travel:living_room`, `travel:bedroom`, `ask_gemma`, `call_chiara`

Label: preferred `travel:bedroom`; acceptable ['travel:bedroom', 'travel:living_room', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:living_room` | False | True | travel:toilet 0.109, travel:living_room 0.347, travel:bedroom 0.169, ask_gemma 0.090, call_chiara 0.286 |
| s1o | `call_chiara` | False | False | travel:toilet 0.099, travel:living_room 0.335, travel:bedroom 0.059, ask_gemma 0.130, call_chiara 0.377 |
| von11 | `travel:toilet` | False | True | travel:toilet 0.415, travel:living_room 0.200, travel:bedroom 0.124, ask_gemma 0.144, call_chiara 0.117 |

## dev_repeat_search_2 — normal

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: searched_no_target; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (5.0 m), kitchen (2.1 m), toilet (8.9 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 67 cm; ultrasonic front: 67 cm; sensor age 0.45 s. The room search already repeated without finding Chiara.

Offered (in order): `call_chiara`, `ask_gemma`, `travel:living_room`, `travel:kitchen`, `travel:toilet`

Label: preferred `travel:kitchen`; acceptable ['travel:living_room', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:living_room` | False | True | call_chiara 0.284, ask_gemma 0.069, travel:living_room 0.457, travel:kitchen 0.069, travel:toilet 0.122 |
| s1o | `travel:living_room` | False | True | call_chiara 0.001, ask_gemma 0.000, travel:living_room 0.998, travel:kitchen 0.000, travel:toilet 0.000 |
| von11 | `travel:living_room` | False | True | call_chiara 0.067, ask_gemma 0.131, travel:living_room 0.479, travel:kitchen 0.174, travel:toilet 0.148 |

## dev_repeat_search_2 — reverse

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: searched_no_target; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (5.0 m), kitchen (2.1 m), toilet (8.9 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 67 cm; ultrasonic front: 67 cm; sensor age 0.45 s. The room search already repeated without finding Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:living_room`, `ask_gemma`, `call_chiara`

Label: preferred `travel:kitchen`; acceptable ['travel:living_room', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `call_chiara` | False | False | travel:toilet 0.150, travel:kitchen 0.143, travel:living_room 0.234, ask_gemma 0.123, call_chiara 0.351 |
| s1o | `travel:living_room` | False | True | travel:toilet 0.008, travel:kitchen 0.003, travel:living_room 0.971, ask_gemma 0.009, call_chiara 0.009 |
| von11 | `travel:toilet` | False | True | travel:toilet 0.425, travel:kitchen 0.194, travel:living_room 0.180, ask_gemma 0.111, call_chiara 0.090 |

## dev_repeat_search_3 — normal

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: searched_no_target; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (3.7 m), kitchen (1.7 m), toilet (6.6 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 49 cm; ultrasonic front: 50 cm; sensor age 0.13 s. The room search already repeated without finding Chiara.

Offered (in order): `call_chiara`, `ask_gemma`, `travel:living_room`, `travel:kitchen`, `travel:toilet`

Label: preferred `travel:kitchen`; acceptable ['travel:living_room', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:living_room` | False | True | call_chiara 0.297, ask_gemma 0.073, travel:living_room 0.459, travel:kitchen 0.074, travel:toilet 0.097 |
| s1o | `travel:living_room` | False | True | call_chiara 0.001, ask_gemma 0.000, travel:living_room 0.998, travel:kitchen 0.000, travel:toilet 0.000 |
| von11 | `travel:living_room` | False | True | call_chiara 0.068, ask_gemma 0.134, travel:living_room 0.486, travel:kitchen 0.167, travel:toilet 0.146 |

## dev_repeat_search_3 — reverse

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: searched_no_target; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (3.7 m), kitchen (1.7 m), toilet (6.6 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 49 cm; ultrasonic front: 50 cm; sensor age 0.13 s. The room search already repeated without finding Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:living_room`, `ask_gemma`, `call_chiara`

Label: preferred `travel:kitchen`; acceptable ['travel:living_room', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `call_chiara` | False | False | travel:toilet 0.146, travel:kitchen 0.147, travel:living_room 0.200, ask_gemma 0.127, call_chiara 0.381 |
| s1o | `travel:living_room` | False | True | travel:toilet 0.023, travel:kitchen 0.008, travel:living_room 0.807, ask_gemma 0.040, call_chiara 0.122 |
| von11 | `travel:toilet` | False | True | travel:toilet 0.412, travel:kitchen 0.207, travel:living_room 0.191, ask_gemma 0.106, call_chiara 0.084 |

## dev_repeat_search_4 — normal

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: searched_no_target; toilet: unsearched. Reachable rooms: bedroom (8.1 m), kitchen (6.9 m), toilet (2.6 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 152 cm; ultrasonic front: 156 cm; sensor age 0.20 s. The room search already repeated without finding Chiara.

Offered (in order): `call_chiara`, `ask_gemma`, `travel:bedroom`, `travel:kitchen`, `travel:toilet`

Label: preferred `travel:toilet`; acceptable ['travel:bedroom', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:bedroom` | False | True | call_chiara 0.344, ask_gemma 0.061, travel:bedroom 0.406, travel:kitchen 0.086, travel:toilet 0.103 |
| s1o | `travel:bedroom` | False | True | call_chiara 0.034, ask_gemma 0.004, travel:bedroom 0.953, travel:kitchen 0.009, travel:toilet 0.000 |
| von11 | `travel:bedroom` | False | True | call_chiara 0.065, ask_gemma 0.131, travel:bedroom 0.481, travel:kitchen 0.172, travel:toilet 0.151 |

## dev_repeat_search_4 — reverse

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: searched_no_target; toilet: unsearched. Reachable rooms: bedroom (8.1 m), kitchen (6.9 m), toilet (2.6 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 152 cm; ultrasonic front: 156 cm; sensor age 0.20 s. The room search already repeated without finding Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:bedroom`, `ask_gemma`, `call_chiara`

Label: preferred `travel:toilet`; acceptable ['travel:bedroom', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:bedroom` | False | True | travel:toilet 0.147, travel:kitchen 0.150, travel:bedroom 0.312, ask_gemma 0.096, call_chiara 0.294 |
| s1o | `call_chiara` | False | False | travel:toilet 0.057, travel:kitchen 0.011, travel:bedroom 0.010, ask_gemma 0.181, call_chiara 0.742 |
| von11 | `travel:toilet` | True | True | travel:toilet 0.296, travel:kitchen 0.251, travel:bedroom 0.179, ask_gemma 0.152, call_chiara 0.121 |

## dev_repeat_search_5 — normal

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: searched_no_target; toilet: unsearched. Reachable rooms: bedroom (5.2 m), kitchen (8.0 m), toilet (1.4 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 41 cm; ultrasonic front: 43 cm; sensor age 0.10 s. The room search already repeated without finding Chiara.

Offered (in order): `call_chiara`, `ask_gemma`, `travel:bedroom`, `travel:kitchen`, `travel:toilet`

Label: preferred `travel:toilet`; acceptable ['travel:bedroom', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:bedroom` | False | True | call_chiara 0.339, ask_gemma 0.060, travel:bedroom 0.398, travel:kitchen 0.098, travel:toilet 0.105 |
| s1o | `travel:bedroom` | False | True | call_chiara 0.015, ask_gemma 0.004, travel:bedroom 0.976, travel:kitchen 0.005, travel:toilet 0.000 |
| von11 | `travel:bedroom` | False | True | call_chiara 0.069, ask_gemma 0.130, travel:bedroom 0.477, travel:kitchen 0.174, travel:toilet 0.150 |

## dev_repeat_search_5 — reverse

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: searched_no_target; toilet: unsearched. Reachable rooms: bedroom (5.2 m), kitchen (8.0 m), toilet (1.4 m). Target evidence: not_seen. Voice hint: none. Calls without answer: 0. Previous script: search_here; outcome: room_coverage_complete_twice. LiDAR front: 41 cm; ultrasonic front: 43 cm; sensor age 0.10 s. The room search already repeated without finding Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:bedroom`, `ask_gemma`, `call_chiara`

Label: preferred `travel:toilet`; acceptable ['travel:bedroom', 'travel:kitchen', 'travel:toilet']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| laya_en | `travel:bedroom` | False | True | travel:toilet 0.149, travel:kitchen 0.151, travel:bedroom 0.311, ask_gemma 0.102, call_chiara 0.287 |
| s1o | `call_chiara` | False | False | travel:toilet 0.039, travel:kitchen 0.017, travel:bedroom 0.014, ask_gemma 0.161, call_chiara 0.769 |
| von11 | `travel:kitchen` | False | True | travel:toilet 0.242, travel:kitchen 0.269, travel:bedroom 0.212, ask_gemma 0.157, call_chiara 0.120 |

