"""Quick check only: ladder.main with the top-app core check disabled (Termux is on cores 0-5 now)."""
import sys
sys.path.insert(0, "/termux-home/ladder")
import ladder
ladder.check_cores = lambda when: None
sys.argv = ["ladder.py", *sys.argv[1:]]
ladder.main()
