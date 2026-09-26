#!/usr/bin/env python3
"""Model-free checks: deterministic cases, v2 labels kept, held-out sealed, mock run end to end.

Run from any Python 3.12+: python3 test_v3.py. Held-out is only hashed, never parsed.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import encode, generate, load_v2


def main():
    v2 = load_v2()
    manifest = json.loads((HERE / "CASES.json").read_text())
    for split in ("dev", "heldout"):
        blob = encode(generate(v2, split))
        assert hashlib.sha256(blob).hexdigest() == manifest["splits"][split]["sha256"], split
        assert (HERE / f"{split}.jsonl").read_bytes() == blob, split
    dev = [json.loads(l) for l in (HERE / "dev.jsonl").read_text().splitlines()]
    assert len(dev) >= 60 and len({c["id"] for c in dev}) == len(dev)
    for c in dev:
        ref = v2.build_case("laya", c["family"], c["v2_variation"])
        assert (c["preferred"], c["acceptable"]) == (ref["preferred"], ref["acceptable"]), c["id"]
        assert c["preferred"] in v2.options_for(c, "filtered_text", "flat", None), c["id"]

    run = [sys.executable, str(HERE / "run.py"), "--candidate", "mock", "--families", "all"]
    sealed = subprocess.run(run + ["--split", "heldout", "--out", "/nonexistent"], capture_output=True, text=True)
    assert sealed.returncode != 0 and "sealed" in sealed.stderr, sealed.stderr

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mock.json"
        subprocess.run(run + ["--dev-subset", "11", "--out", str(out)], check=True, capture_output=True)
        d = json.loads(out.read_text())
        assert len(d["rows"]) == 22 and {r["family"] for r in d["rows"]} == set(v2.FAMILIES)
        assert d["summary"]["order_flips"] == 0  # mock scores are order-invariant
        for r in d["rows"]:
            call = r["calls"][0]
            assert call["options"] == r["offered_first"] and call["argmax"] == r["first_choice"]
            assert abs(sum(call["distribution"].values()) - 1) < 1e-9
        rev = [r for r in d["rows"] if r["order"] == "reverse"]
        assert all(r["offered_first"] == list(reversed(n["offered_first"]))
                   for n, r in zip([r for r in d["rows"] if r["order"] == "normal"], rev))
    with tempfile.TemporaryDirectory() as tmp:  # default = the four judgment families
        out = Path(tmp) / "mock.json"
        subprocess.run(run[:-2] + ["--out", str(out)], check=True, capture_output=True)
        fams = {r["family"] for r in json.loads(out.read_text())["rows"]}
        assert fams == {"room_finished", "hall_hint", "repeat_search", "heard_from_room"}, fams
    print("test_v3: all checks passed")


if __name__ == "__main__":
    main()
