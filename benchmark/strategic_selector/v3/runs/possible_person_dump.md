# possible_person development rows (frame filtered_text)

## dev_possible_person_0 — normal

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: bedroom (9.0 m), kitchen (9.0 m), toilet (2.9 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 176 cm; ultrasonic front: 173 cm; sensor age 0.17 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `search_here`, `call_chiara`, `ask_gemma`, `travel:bedroom`, `travel:kitchen`, `travel:toilet`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:bedroom` | False | False | search_here 0.207, call_chiara 0.059, ask_gemma 0.088, travel:bedroom 0.386, travel:kitchen 0.145, travel:toilet 0.114 |
| laya_en | `search_here` | False | True | search_here 0.821, call_chiara 0.067, ask_gemma 0.007, travel:bedroom 0.069, travel:kitchen 0.024, travel:toilet 0.012 |
| laya_multi | `search_here` | False | True | search_here 0.993, call_chiara 0.003, ask_gemma 0.001, travel:bedroom 0.002, travel:kitchen 0.001, travel:toilet 0.001 |
| laya_micro | `search_here` | False | True | search_here 0.920, call_chiara 0.058, ask_gemma 0.005, travel:bedroom 0.006, travel:kitchen 0.005, travel:toilet 0.005 |
| s1o | `search_here` | False | True | search_here 0.999, call_chiara 0.001, ask_gemma 0.000, travel:bedroom 0.000, travel:kitchen 0.000, travel:toilet 0.000 |

## dev_possible_person_0 — reverse

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: bedroom (9.0 m), kitchen (9.0 m), toilet (2.9 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 176 cm; ultrasonic front: 173 cm; sensor age 0.17 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:bedroom`, `ask_gemma`, `call_chiara`, `search_here`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:kitchen` | False | False | travel:toilet 0.186, travel:kitchen 0.256, travel:bedroom 0.176, ask_gemma 0.102, call_chiara 0.085, search_here 0.196 |
| laya_en | `search_here` | False | True | travel:toilet 0.005, travel:kitchen 0.006, travel:bedroom 0.030, ask_gemma 0.001, call_chiara 0.008, search_here 0.949 |
| laya_multi | `search_here` | False | True | travel:toilet 0.003, travel:kitchen 0.003, travel:bedroom 0.024, ask_gemma 0.003, call_chiara 0.012, search_here 0.955 |
| laya_micro | `search_here` | False | True | travel:toilet 0.046, travel:kitchen 0.046, travel:bedroom 0.045, ask_gemma 0.046, call_chiara 0.053, search_here 0.764 |
| s1o | `search_here` | False | True | travel:toilet 0.000, travel:kitchen 0.000, travel:bedroom 0.000, ask_gemma 0.000, call_chiara 0.000, search_here 1.000 |

## dev_possible_person_1 — normal

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (7.8 m), kitchen (1.5 m), toilet (7.6 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 202 cm; ultrasonic front: 202 cm; sensor age 0.30 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `search_here`, `call_chiara`, `ask_gemma`, `travel:living_room`, `travel:kitchen`, `travel:toilet`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:living_room` | False | False | search_here 0.171, call_chiara 0.067, ask_gemma 0.088, travel:living_room 0.411, travel:kitchen 0.137, travel:toilet 0.125 |
| laya_en | `search_here` | False | True | search_here 0.802, call_chiara 0.077, ask_gemma 0.009, travel:living_room 0.077, travel:kitchen 0.012, travel:toilet 0.022 |
| laya_multi | `search_here` | False | True | search_here 0.991, call_chiara 0.003, ask_gemma 0.001, travel:living_room 0.003, travel:kitchen 0.001, travel:toilet 0.001 |
| laya_micro | `search_here` | False | True | search_here 0.992, call_chiara 0.005, ask_gemma 0.001, travel:living_room 0.001, travel:kitchen 0.001, travel:toilet 0.001 |
| s1o | `search_here` | False | True | search_here 0.672, call_chiara 0.002, ask_gemma 0.001, travel:living_room 0.325, travel:kitchen 0.000, travel:toilet 0.000 |

## dev_possible_person_1 — reverse

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (7.8 m), kitchen (1.5 m), toilet (7.6 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 202 cm; ultrasonic front: 202 cm; sensor age 0.30 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:living_room`, `ask_gemma`, `call_chiara`, `search_here`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:toilet` | False | False | travel:toilet 0.250, travel:kitchen 0.247, travel:living_room 0.217, ask_gemma 0.075, call_chiara 0.062, search_here 0.149 |
| laya_en | `search_here` | False | True | travel:toilet 0.009, travel:kitchen 0.006, travel:living_room 0.012, ask_gemma 0.001, call_chiara 0.011, search_here 0.960 |
| laya_multi | `search_here` | False | True | travel:toilet 0.012, travel:kitchen 0.013, travel:living_room 0.016, ask_gemma 0.012, call_chiara 0.039, search_here 0.908 |
| laya_micro | `search_here` | False | True | travel:toilet 0.059, travel:kitchen 0.061, travel:living_room 0.070, ask_gemma 0.059, call_chiara 0.065, search_here 0.687 |
| s1o | `search_here` | False | True | travel:toilet 0.001, travel:kitchen 0.000, travel:living_room 0.000, ask_gemma 0.000, call_chiara 0.000, search_here 0.998 |

## dev_possible_person_2 — normal

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (6.7 m), kitchen (2.5 m), toilet (9.5 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 129 cm; ultrasonic front: 132 cm; sensor age 0.11 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `search_here`, `call_chiara`, `ask_gemma`, `travel:living_room`, `travel:kitchen`, `travel:toilet`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:living_room` | False | False | search_here 0.160, call_chiara 0.070, ask_gemma 0.094, travel:living_room 0.420, travel:kitchen 0.141, travel:toilet 0.115 |
| laya_en | `search_here` | False | True | search_here 0.816, call_chiara 0.068, ask_gemma 0.009, travel:living_room 0.068, travel:kitchen 0.015, travel:toilet 0.024 |
| laya_multi | `search_here` | False | True | search_here 0.991, call_chiara 0.003, ask_gemma 0.001, travel:living_room 0.003, travel:kitchen 0.001, travel:toilet 0.001 |
| laya_micro | `search_here` | False | True | search_here 0.986, call_chiara 0.009, ask_gemma 0.001, travel:living_room 0.002, travel:kitchen 0.001, travel:toilet 0.001 |
| s1o | `search_here` | False | True | search_here 0.499, call_chiara 0.003, ask_gemma 0.001, travel:living_room 0.496, travel:kitchen 0.000, travel:toilet 0.001 |

## dev_possible_person_2 — reverse

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (6.7 m), kitchen (2.5 m), toilet (9.5 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 129 cm; ultrasonic front: 132 cm; sensor age 0.11 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:living_room`, `ask_gemma`, `call_chiara`, `search_here`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:toilet` | False | False | travel:toilet 0.251, travel:kitchen 0.248, travel:living_room 0.225, ask_gemma 0.070, call_chiara 0.058, search_here 0.148 |
| laya_en | `search_here` | False | True | travel:toilet 0.008, travel:kitchen 0.006, travel:living_room 0.012, ask_gemma 0.001, call_chiara 0.010, search_here 0.962 |
| laya_multi | `search_here` | False | True | travel:toilet 0.011, travel:kitchen 0.011, travel:living_room 0.013, ask_gemma 0.011, call_chiara 0.037, search_here 0.917 |
| laya_micro | `search_here` | False | True | travel:toilet 0.034, travel:kitchen 0.034, travel:living_room 0.043, ask_gemma 0.033, call_chiara 0.036, search_here 0.819 |
| s1o | `search_here` | False | True | travel:toilet 0.001, travel:kitchen 0.000, travel:living_room 0.000, ask_gemma 0.000, call_chiara 0.001, search_here 0.997 |

## dev_possible_person_3 — normal

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: bedroom (7.3 m), kitchen (9.5 m), toilet (2.4 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 151 cm; ultrasonic front: 151 cm; sensor age 0.11 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `search_here`, `call_chiara`, `ask_gemma`, `travel:bedroom`, `travel:kitchen`, `travel:toilet`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:bedroom` | False | False | search_here 0.175, call_chiara 0.062, ask_gemma 0.088, travel:bedroom 0.427, travel:kitchen 0.134, travel:toilet 0.115 |
| laya_en | `search_here` | False | True | search_here 0.815, call_chiara 0.075, ask_gemma 0.009, travel:bedroom 0.064, travel:kitchen 0.024, travel:toilet 0.014 |
| laya_multi | `search_here` | False | True | search_here 0.990, call_chiara 0.003, ask_gemma 0.001, travel:bedroom 0.005, travel:kitchen 0.001, travel:toilet 0.001 |
| laya_micro | `search_here` | False | True | search_here 0.926, call_chiara 0.050, ask_gemma 0.004, travel:bedroom 0.010, travel:kitchen 0.005, travel:toilet 0.004 |
| s1o | `search_here` | False | True | search_here 0.999, call_chiara 0.001, ask_gemma 0.000, travel:bedroom 0.000, travel:kitchen 0.000, travel:toilet 0.000 |

## dev_possible_person_3 — reverse

State: Mission: find Chiara. Current room: living_room. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: bedroom (7.3 m), kitchen (9.5 m), toilet (2.4 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 151 cm; ultrasonic front: 151 cm; sensor age 0.11 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:bedroom`, `ask_gemma`, `call_chiara`, `search_here`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:kitchen` | False | False | travel:toilet 0.181, travel:kitchen 0.258, travel:bedroom 0.194, ask_gemma 0.096, call_chiara 0.078, search_here 0.193 |
| laya_en | `search_here` | False | True | travel:toilet 0.006, travel:kitchen 0.005, travel:bedroom 0.014, ask_gemma 0.001, call_chiara 0.007, search_here 0.966 |
| laya_multi | `search_here` | False | True | travel:toilet 0.006, travel:kitchen 0.006, travel:bedroom 0.112, ask_gemma 0.006, call_chiara 0.023, search_here 0.847 |
| laya_micro | `search_here` | False | True | travel:toilet 0.084, travel:kitchen 0.085, travel:bedroom 0.083, ask_gemma 0.084, call_chiara 0.090, search_here 0.576 |
| s1o | `search_here` | False | True | travel:toilet 0.000, travel:kitchen 0.000, travel:bedroom 0.000, ask_gemma 0.000, call_chiara 0.000, search_here 1.000 |

## dev_possible_person_4 — normal

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (7.7 m), kitchen (2.3 m), toilet (4.9 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 187 cm; ultrasonic front: 191 cm; sensor age 0.52 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `search_here`, `call_chiara`, `ask_gemma`, `travel:living_room`, `travel:kitchen`, `travel:toilet`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:living_room` | False | False | search_here 0.185, call_chiara 0.067, ask_gemma 0.088, travel:living_room 0.411, travel:kitchen 0.137, travel:toilet 0.112 |
| laya_en | `search_here` | False | True | search_here 0.807, call_chiara 0.075, ask_gemma 0.009, travel:living_room 0.076, travel:kitchen 0.013, travel:toilet 0.020 |
| laya_multi | `search_here` | False | True | search_here 0.992, call_chiara 0.003, ask_gemma 0.001, travel:living_room 0.003, travel:kitchen 0.001, travel:toilet 0.001 |
| laya_micro | `search_here` | False | True | search_here 0.983, call_chiara 0.012, ask_gemma 0.001, travel:living_room 0.003, travel:kitchen 0.001, travel:toilet 0.001 |
| s1o | `search_here` | False | True | search_here 0.763, call_chiara 0.002, ask_gemma 0.001, travel:living_room 0.234, travel:kitchen 0.000, travel:toilet 0.000 |

## dev_possible_person_4 — reverse

State: Mission: find Chiara. Current room: bedroom. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: living_room (7.7 m), kitchen (2.3 m), toilet (4.9 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 187 cm; ultrasonic front: 191 cm; sensor age 0.52 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `travel:toilet`, `travel:kitchen`, `travel:living_room`, `ask_gemma`, `call_chiara`, `search_here`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:kitchen` | False | False | travel:toilet 0.194, travel:kitchen 0.271, travel:living_room 0.221, ask_gemma 0.085, call_chiara 0.070, search_here 0.159 |
| laya_en | `search_here` | False | True | travel:toilet 0.007, travel:kitchen 0.006, travel:living_room 0.012, ask_gemma 0.002, call_chiara 0.011, search_here 0.963 |
| laya_multi | `search_here` | False | True | travel:toilet 0.010, travel:kitchen 0.010, travel:living_room 0.012, ask_gemma 0.009, call_chiara 0.040, search_here 0.920 |
| laya_micro | `search_here` | False | True | travel:toilet 0.038, travel:kitchen 0.040, travel:living_room 0.050, ask_gemma 0.038, call_chiara 0.041, search_here 0.793 |
| s1o | `search_here` | False | True | travel:toilet 0.000, travel:kitchen 0.000, travel:living_room 0.000, ask_gemma 0.000, call_chiara 0.000, search_here 0.999 |

## dev_possible_person_5 — normal

State: Mission: find Chiara. Current room: kitchen. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: bedroom (1.7 m), living_room (7.3 m), toilet (4.0 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 204 cm; ultrasonic front: 204 cm; sensor age 0.22 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `search_here`, `call_chiara`, `ask_gemma`, `travel:bedroom`, `travel:living_room`, `travel:toilet`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:bedroom` | False | False | search_here 0.150, call_chiara 0.081, ask_gemma 0.093, travel:bedroom 0.365, travel:living_room 0.175, travel:toilet 0.135 |
| laya_en | `search_here` | False | True | search_here 0.727, call_chiara 0.057, ask_gemma 0.008, travel:bedroom 0.150, travel:living_room 0.035, travel:toilet 0.023 |
| laya_multi | `search_here` | False | True | search_here 0.984, call_chiara 0.004, ask_gemma 0.001, travel:bedroom 0.010, travel:living_room 0.001, travel:toilet 0.001 |
| laya_micro | `search_here` | False | True | search_here 0.988, call_chiara 0.008, ask_gemma 0.001, travel:bedroom 0.002, travel:living_room 0.002, travel:toilet 0.001 |
| s1o | `search_here` | False | True | search_here 0.997, call_chiara 0.001, ask_gemma 0.000, travel:bedroom 0.002, travel:living_room 0.000, travel:toilet 0.000 |

## dev_possible_person_5 — reverse

State: Mission: find Chiara. Current room: kitchen. Localization: confirmed. Room history: bedroom: unsearched; hall: transit; kitchen: unsearched; living_room: unsearched; toilet: unsearched. Reachable rooms: bedroom (1.7 m), living_room (7.3 m), toilet (4.0 m). Target evidence: person_seen_60_cm_identity_unknown. Voice hint: none. Calls without answer: 0. Previous script: entered_room; outcome: arrived. LiDAR front: 204 cm; ultrasonic front: 204 cm; sensor age 0.22 s. Camera detected a person; it has not identified Chiara.

Offered (in order): `travel:toilet`, `travel:living_room`, `travel:bedroom`, `ask_gemma`, `call_chiara`, `search_here`

Label: preferred `call_chiara`; acceptable ['call_chiara', 'search_here']

| Candidate | Choice | Preferred? | Acceptable? | Distribution |
|---|---|---|---|---|
| von11 | `travel:living_room` | False | False | travel:toilet 0.186, travel:living_room 0.256, travel:bedroom 0.184, ask_gemma 0.093, call_chiara 0.081, search_here 0.200 |
| laya_en | `search_here` | False | True | travel:toilet 0.006, travel:living_room 0.016, travel:bedroom 0.022, ask_gemma 0.002, call_chiara 0.011, search_here 0.942 |
| laya_multi | `search_here` | False | True | travel:toilet 0.011, travel:living_room 0.012, travel:bedroom 0.228, ask_gemma 0.011, call_chiara 0.027, search_here 0.711 |
| laya_micro | `search_here` | False | True | travel:toilet 0.035, travel:living_room 0.076, travel:bedroom 0.093, ask_gemma 0.035, call_chiara 0.036, search_here 0.724 |
| s1o | `search_here` | False | True | travel:toilet 0.000, travel:living_room 0.000, travel:bedroom 0.000, ask_gemma 0.000, call_chiara 0.000, search_here 0.999 |

