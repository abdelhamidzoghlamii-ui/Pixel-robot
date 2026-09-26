"""Choice agreement: new main block vs the old b1609 main block (same cases, both orders)."""
import json, sys
old = {(r["case_id"], r["order"]): r for r in map(json.loads, open(sys.argv[1])) if r["model"] == "s1o_b1609" and r["block"] == "main"}
new = {(r["case_id"], r["order"]): r for r in map(json.loads, open(sys.argv[2])) if r["block"] == "main"}
assert old.keys() == new.keys(), (len(old), len(new))
diff = [k for k in old if old[k]["choice"] != new[k]["choice"]]
pd = max((abs(old[k]["distribution"][o] - new[k]["distribution"][o]), k, o) for k in old for o in old[k]["distribution"])
print(f"decisions {len(old)}; same choice {len(old) - len(diff)}/{len(old)}")
for k in diff:
    print(f"  DIFFERENT {k}: old {old[k]['choice']!r} p={max(old[k]['distribution'].values()):.3f}  new {new[k]['choice']!r} p={max(new[k]['distribution'].values()):.3f}")
print(f"max |probability difference| {pd[0]:.4f} at {pd[1]} option {pd[2]!r}")
