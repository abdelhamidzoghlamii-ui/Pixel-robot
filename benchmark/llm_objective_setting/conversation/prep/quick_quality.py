"""Quick check only: conv_quality with 2 prompts per model (A1, C8), 1 run."""
import sys
sys.path.insert(0, "/termux-home/ladder")
import conv_quality as q
q.bench.PROMPTS_A = [p for p in q.bench.PROMPTS_A if p["id"] == "A1"]
q.bench.PROMPTS_B = []
q.bench.PROMPTS_C = [p for p in q.bench.PROMPTS_C if p["id"] == "C8"]
sys.argv = ["conv_quality.py", sys.argv[1], "--runs", "1"]
q.main()
