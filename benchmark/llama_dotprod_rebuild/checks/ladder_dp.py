"""3a: the ladder runner, unchanged, with one extra variant added at runtime: b1609 rebuilt with dotprod+fp16.
Same S1O adapter, same GGUF, same server flags as s1o_b1609 (no extra args)."""
import sys
from pathlib import Path
sys.path.insert(0, "/termux-home/ladder")
import ladder
NEW = "/data/data/com.termux/files/home/llama.cpp-b1609-dotprod/build/bin"
ladder.VARIANT_ENV["s1o_b1609dp"] = {"S1O_LLAMA_BIN": NEW, "S1O_GGUF": ladder.GGUFS["q4km"], "S1O_SERVER_ARGS": ""}
ladder.MODELS["s1o_b1609dp"] = ladder.MODELS["s1o"]
ladder.WEIGHTS["s1o_b1609dp"] = [Path(ladder.GGUFS["q4km"])]
sys.argv = ["ladder.py", *sys.argv[1:]]
ladder.main()
