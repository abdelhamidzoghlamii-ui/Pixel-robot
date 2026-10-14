# robot-chat MTP on/off — owner runs from native Termux

No motors, no camera. Do not run inside Debian/proot. Before **each** run: quit every agent and exit Debian;
stop robot-chat/chat.py and any llama-server. In native Termux run:

```bash
pkill -f security_reminder_hook
```

Unplug the charger (Discharging required) and start with the battery at 80 % or more. Keep the screen on and
Termux in front throughout. Agent residency invalidates timings.

1. Smoke (one prompt per config, gate at most 20 s; about 7 min including the 5 min launcher idle):

```bash
bash ~/robot/benchmark/llm_objective_setting/conversation/chat_mtp/oneshot.sh --smoke
```

Check that `report.txt` printed at the end shows `1/1` for OFF and ON and an `accept` value for ON.

2. Full run (13 prompts per config; roughly 25–70 min: 5 min idle, about 30 s hashing, then 180 s blocks with up
to 8 min of skin gate before each):

```bash
bash ~/robot/benchmark/llm_objective_setting/conversation/chat_mtp/oneshot.sh
```

Resume an interrupted run (finished prompts are kept, the original idle baseline is reused; use `--smoke` again
only for a smoke folder):

```bash
bash ~/robot/benchmark/llm_objective_setting/conversation/chat_mtp/oneshot.sh --resume ~/robot_runs/chat_mtp_<UTC>
```

Results: `~/robot_runs/chat_mtp_<UTC>/report.txt`; console log in `~/robot_runs/chat_mtp/`.
