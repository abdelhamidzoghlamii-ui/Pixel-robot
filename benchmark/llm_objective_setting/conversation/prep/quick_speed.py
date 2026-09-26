"""Quick check only: conv_speed with 2 prompts per block (A1, C8); core guard faked (timings do not count)."""
import sys
sys.path.insert(0, "/termux-home/ladder")
import conv_speed as cs
cs.ladder.allowed_cpus = lambda: set(range(8))  # Termux is on cores 0-5 now; check only
full = cs.prompts
cs.prompts = lambda family, think: [p for p in full(family, think) if p[0] in ("A1", "C8")]
sys.argv = ["conv_speed.py", *sys.argv[1:]]
cs.main()
