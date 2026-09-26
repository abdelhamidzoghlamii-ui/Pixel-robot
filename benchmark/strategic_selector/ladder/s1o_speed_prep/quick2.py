"""Check only: ladder.main with cores simulated (lost while the flag file exists) because Termux is on cores 0-5."""
import os, sys
sys.path.insert(0, "/termux-home/ladder")
import ladder
FLAG = sys.argv.pop(1)
ladder.allowed_cpus = lambda: {0, 1, 2, 3, 4, 5} if os.path.exists(FLAG) else set(range(8))
sys.argv = ["ladder.py", *sys.argv[1:]]
ladder.main()
