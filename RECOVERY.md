# ROBOT PROJECT RECOVERY GUIDE
# Read this first in any new conversation

## What this project is
Offline AI robot on rooted Pixel 7.
Gemma E2B Q4_K_M + YOLO11m + Whisper + ESP32.
Fully offline, no cloud, no internet needed.

## Current status (April 17, 2026)
- Software: complete and working
- Hardware: partially wired, ESP32 in hand
- Vision: real via mmproj (940MB file on phone)
- Next step: solder MX1508, wire ESP32, first drive

## Key files to read in order
1. PROJECT_STATE.md  — full architecture
2. main.py           — robot brain
3. server_manager.py — model loading
4. detect_person.py  — vision pipeline
5. motors.py         — hardware control

## Critical facts not obvious from code
- CYCLE_MOVE_TIME = 1.3s (benchmarked optimal)
- Model: gemma-4-e2b-it-q4_k_m.gguf (NOT q8_0)
- mmproj required for real vision (940MB separate file)
- ESP32 replaces Arduino Due as motor controller
- MX1508 needs soldering (bare holes, not screw terminals)
- L298N has screw terminals (usable without soldering)
- Batteries: 2x Samsung 30Q, measured 8.2V
- Phone powers ESP32 via USB OTG temporarily
- Buck converter (arriving AliExpress) replaces USB power

## Open questions
See bottom of PROJECT_STATE.md — 42 questions, 11 categories
Currently at: Category 1 — wire hardware

## GitHub
https://github.com/abdelhamidzoghlamii-ui/pixel-robot
Username: abdelhamidzoghlamii-ui

## Models on phone (not in repo, too large)
~/models/gemma-4-e2b-it-q4_k_m.gguf   3.3GB
~/models/gemma-4-e2b-it-q8_0.gguf     4.7GB
~/models/mmproj-gemma4-e2b.gguf        940MB
~/robot/yolo11m.onnx                   77MB
~/whisper.cpp/models/ggml-base.bin
