"""Exercises navigate_rules' safety branches directly. No motors, no server."""
import sys
sys.path.insert(0, "/data/data/com.termux/files/home/robot")
import main as R

def fresh():
    r = R.Robot()          # no motors
    r.last_moves = []
    return r

print(f"OBSTACLE_DIST = {R.OBSTACLE_DIST}\n")

print("=== single-shot: distance -> move (no history) ===")
for d in (5, 14, 15, 20, 24, 25, 26, 100, 400, 999):
    r = fresh()
    m = r.navigate_rules([], d)
    print(f"  dist {d:4}  ->  {m:<13} stuck={getattr(r,'nav_stuck',False)}")

print("\n=== persistent blockage: 16 cycles at 20cm ===")
print("   (feeding each returned move back into last_moves, as move() does)")
r = fresh()
for i in range(1, 17):
    m = r.navigate_rules([], 20)
    r.last_moves.append(m)
    if len(r.last_moves) > 10:
        r.last_moves.pop(0)
    print(f"  cycle {i:2}  ->  {m:<13} side={r.avoid_side:<5} stuck={r.nav_stuck}")

print("\n=== blocked then cleared ===")
r = fresh()
seq = [20, 20, 20, 20, 20, 300, 300]
for i, d in enumerate(seq, 1):
    m = r.navigate_rules([], d)
    r.last_moves.append(m)
    print(f"  cycle {i}  dist {d:3}  ->  {m:<13} stuck={r.nav_stuck}")
