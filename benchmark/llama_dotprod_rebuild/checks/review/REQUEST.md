# Review request: rebuild of the robot's llama.cpp (b1609) with dotprod, and server_manager.py path change
ROLE: Reviewer. PONYTAIL: off. Independent correctness/safety review. Do NOT call any tool or run any command, do
not edit, create, stage, commit or push anything, do not run motors, and do not launch another reviewer. All
material is inlined below; answer from this text only.

## Task (human decision, Pixel Robot)
Fix the missing dotprod in the robot's llama.cpp build; must change speed only, not behaviour. (1) Build the SAME
commit as the robot's b1609 (e1a1abb7) with ARMv8.2 dotprod (+fp16 if supported) into a NEW directory without
touching the current binary; record flags. (2) Prove dotprod: system_info + objdump sdot counts old vs new.
(3) Behaviour: a) s1o full ladder both orders vs old b1609 results (agreement, differing cases, max prob diff,
accuracy, median/P95); b) main.py parse_command() with PARSE_SYS on 15 fixed commands, old vs new side by side;
c) tokens/s generating 200 tokens old vs new. (4) server_manager.py: change the binary path only.

## Base
Robot repo /termux-home/robot at 6322dc1; the only tracked change is server_manager.py (diff below). Builds and
check scripts are outside the repo. frozen.sha256 pins server_manager.py and the new binaries.

## Please report
Whether the build is the same commit with only the arch flag changed; whether the dotprod evidence is sufficient;
whether the behaviour checks support "speed only, not behaviour" (note the 2/120 s1o flips and the old-binary
parse timeouts); any risk in the path change for the robot (runtime from native Termux, RUNPATH, libs, the 20 s
parse timeout); defects in the check scripts. End with APPROVE / APPROVE WITH NOTES / CHANGES REQUESTED.
