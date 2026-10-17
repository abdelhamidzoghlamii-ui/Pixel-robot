# Decision index — current orientation

This is a short reading guide, not another decision log. The numbered entries in
[DECISIONS.md](DECISIONS.md) remain the authority and retain their original text.
Read [STATUS.md](STATUS.md) or [APP_STATUS.md](APP_STATUS.md) for what is currently
implemented. Before appending a decision, read the last number in DECISIONS.md
itself. An entry labelled *decided* here may still be unimplemented.

| Area | Current orientation | Evidence |
|---|---|---|
| Prototype navigation | Python rules and sensors own low-level movement; Gemma was removed from the navigation loop. No hardware validation yet. | #104, #111, #88; STATUS Architecture |
| Strategic selector | **Wired as the mission picker (#138) but shelved as a per-cycle chooser:** always EXPLORE in the motors-off run (#139); no prompt or model qualifies in desk tests (#140). Owner direction: Gemma as the robot's event-driven "mind", Python keeps moves (#140). Fine-tuning deferred to a later robot version. One resident Gemma for talk + selection decided (option 3, #125). | #109, #112, #113, #116, #125, #138–#140; [index](../benchmark/INDEX.md) |
| Motor safety | Firmware watchdog is 1000 ms; the `nav_stuck` safety-move override was fixed in Python; LLM failure paths now select STOP. Hardware validation remains separate. | #47, #88, #108; STATUS Pending |
| Firmware and wiring | Corrected `mode2_auto.ino` and corner map are committed; what is flashed on the physical board is unconfirmed. | #43, #89; STATUS Architecture |
| Person stop | Area bucket implemented with `close`/`very close` → HOLD, mission continues (#138); held 168 forward moves in the motors-off run (#139); tracks distance poorly; not run motors-live. | #78–#80, #138, #139; STATUS Pending |
| Thermal control | Android's skin-based governor sets the sustained limit; zone9 is not a heat signal. Power map and duty-cycle measurements show continuous use needs heat pauses. Camera-off pauses with sensor-only moves and resident Gemma are implemented with the adaptive 37/35 °C schedule; #127 emergency stops retain precedence. Validated in three motors-off dry runs (0 transition-rule violations); with the picker on, ACTIVE share fell to 0.48 (#139). | #74–#75, #90–#93, #126–#129, #135, #139; STATUS Thermal/Pending |
| Model vision | Deployed image requests were effectively text-only; later projector experiments did not yield usable on-device LLM vision. YOLO remains perception. | #96, #105; STATUS Vision |
| Android app | Root-free USB and sustained inference spikes passed; the native app itself has no code yet. Prototype measurements do not transfer automatically. | #67–#69; APP_STATUS |
| Licensing | [LICENSES.md](LICENSES.md) owns the app allowlist. Its derived copy in APP_CLAUDE.md is intentional. | #66 |
| Collaboration | Independent Coder/Reviewer/human commit gate and per-task execution profiles are defined in [WORKFLOW.md](WORKFLOW.md). Push needs separate authorization. | #86–#87 |
| Strategic selector harness v3 | s1o dropped on speed; laya_en and von11 proceed; hardware timings provisional. | #112 |
| Data Engineer role | Selector moves to a fine-tuning track; new Data Engineer role created to generate the dataset. | #113 |
| LLM runtime | Robot runs b1609 rebuilt with ARM dotprod (~3x prompt speed); default model Gemma E2B Q4_0 with `--cache-ram 0` and a voice warm-up. MTP drafter on in robot-chat, opt-in `setup_q4_mtp` for the robot (#137). E4B Q4_0 + drafter in robot-chat, opt-in `setup_e4b_q4(_mtp)` (#141). | #114, #115, #137, #141 |
| Selector direction | Gemma letter-scoring leads (1.55 s, 77%); one resident Gemma for talk + selection decided (option 3, #125). | #116, #125 |
| Language | English-only speech input and replies are acceptable; English-only components are eligible. | #117 |
| Detection and co-residency | yolo11s 320/640 mix kept over 640-only (less power, fewer caps, steadier detect ms); RobotCam + detector + resident Gemma fit in RAM with no LMK kills. | #121, #125 |
| Power and camera candidates | Power-map choice: 1 frame/s, 640 every 5 s, MID cores (4–5); deployed in `e208aaf` (#136); motors-off dry run 5.26 h (#139). Manual-exposure 2 fps mode A remains an unadopted candidate on `robotcam-camera-power` (`90ef439`); torch work and mission validation remain pending. | #128, #129, #136, #139; STATUS Pending |
| RelateAnything and heat schedules | M2 (our relsgg-vits16 export) passes parity; standalone BIG median 646 ms; owner chose L2 (BIG, before each picker call); continuous L2 shows substantial capping and picker slowdown after ~15 min; fixed 120/60 s and adaptive 37/35 °C peak at 38.02/37.43 °C with overall inference duty 0.63; owner chose adaptive (#135). | #130–#133, #135 |

## Corrections worth knowing before citing an older entry

| Earlier record | Later correction or change |
|---|---|
| #33 front/rear motor wording, then #40 left/right mapping | #43 corrects the left-side corner assignment; #89 identifies the corrected committed firmware. |
| #52's avoidance escalation | #59 changes the counter logic; #62 limits Gemma consultation to once per blockage episode. The planned LLM-free nav direction is #104 and remains unimplemented. |
| #81 open `nav_stuck` safety defect | #88 records the code fix; #108 records the later STOP fallback for LLM failures. Do not treat #81 as an open code defect. |
| #23 keeps `nav_sim.py` | #118 deleted it (commit `78b1cc5`). |
| #84 says seven orphaned scripts were deleted | #85 verifies they remain tracked and explicitly defers deletion. |
| #77 records a built b2233 binary | #107 says its binary was replaced by b2351; the old source commit is still recoverable. |
| #95's negative `strings` check for Qwen support | #107 / STATUS report a successful Qwen3.5 text load on b2351; #105 separately records unsuccessful vision tests. |
| #99's proposed model-in-loop work order | #100–#102 tested output and rule defects; #104 changes the intended architecture. #109 is separate strategic research. |
| #97 says `yolo11m.onnx` is fine-tuned | #121 corrects it to stock COCO: no fine-tuned YOLO has existed; detector is now yolo11s (commit `9b49ce0`). |
| #12 chooses yolo11m | #121 switches to stock yolo11s with a 320/640 policy; yolo11m remains for rollback. |
| #116 says the 1.55 s Q4_0 ladder result is b2351-only | #119 measures 1408 ms on the robot's dotprod build. |
| Local build names b1609 / b2351 | #119 identifies upstream llama.cpp tags b10194 / b10936. |
| #29, #74, #110, #119–#122, #124 zone9-based heat conclusions | #126 marks them not trusted; speed, RAM and power results stand. Register: `benchmark/thermal_char/HEAT_EVIDENCE.md`. |
| #122 says camera-idle mode is not needed now | #129 decides camera OFF during planned preventive heat pauses; #127 emergency motor and battery stops retain precedence. |
| #133 lists the P23D review/receipt as external and archive cleanups as open | #134 archives the review, receipt and coder report; closes the listed link, Qwen wording and rerun-documentation cleanups; P23E review records and coder report remain external. |
| #127/#129 heat rules and pauses "not implemented"; #132 schedule pending | #135 implements them with adaptive 37/35 °C (`6f81f2a`) and records the motors-off dry run (`92492ec`). |
| #128 power-map choice "benchmark choice, not deployed config" | #136 deploys 1 frame/s and MID 4–5 pinning (`e208aaf`); not yet run in a mission. |
| #120 MTP drafter "not deployed" | #137 turns it on in robot-chat and adds opt-in `setup_q4_mtp`; robot default unchanged; memory cost at chat settings not established. |
| #79 person-stop "not implemented"; #109 selector "not wired"; #130 M2 "not deployed" | #138 implements person-stop and wires the picker with M2; #139 motors-off run; #140 shelves the per-cycle picker. |
| #136 CAM1-MID "not yet run in a mission" | #139 motors-off dry run 5.26 h; motors-live still pending. |
| #115 E4B "kept, not recommended" | #141 E4B Q4_0 + drafter in robot-chat, opt-in robot setups; robot default unchanged. |

For any claim outside these selected topics, search the numbered log and inspect
the later entries that cite it. This index makes no claim to classify all decisions.
