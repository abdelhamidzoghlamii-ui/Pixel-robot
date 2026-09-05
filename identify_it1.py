#!/usr/bin/env python3
"""IT1 identification for forward motion.

  G(s) = K_I / (s*(T1*s + 1)) * e^(-Tt*s)
  x(t) = K*[(t-Tt) - T1*(1 - exp(-(t-Tt)/T1))]   for t >= Tt

  python3 identify_it1.py run130.csv 130
CSV: time_s,travel_cm  (no header)
"""
import sys
import numpy as np


def step_response(t, K, T1, Tt):
    te = np.maximum(t - Tt, 0.0)
    return K * (te - T1 * (1.0 - np.exp(-te / max(T1, 1e-9))))


def fit(t, x, T1_grid, Tt_grid):
    best = None
    for T1 in T1_grid:
        for Tt in Tt_grid:
            basis = step_response(t, 1.0, T1, Tt)
            denom = float(basis @ basis)
            if denom < 1e-12:
                continue
            K = float(basis @ x) / denom
            rms = float(np.sqrt(np.mean((x - K * basis) ** 2)))
            if best is None or rms < best[3]:
                best = (K, T1, Tt, rms)
    return best


def duration_for(target_cm, K, T1, Tt, hi=30.0):
    if step_response(np.array([hi]), K, T1, Tt)[0] < target_cm:
        return None
    lo = Tt
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if step_response(np.array([mid]), K, T1, Tt)[0] < target_cm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    duty = float(sys.argv[2]) if len(sys.argv) > 2 else None

    data = np.loadtxt(sys.argv[1], delimiter=",")
    if data.ndim != 2 or data.shape[1] < 2:
        raise SystemExit("need two columns: time_s,travel_cm")
    t, x = data[:, 0].astype(float), data[:, 1].astype(float)
    t = t - t.min()
    x = x - x[0]
    if x[-1] < 0:
        x = -x
        print("(position decreasing - sign flipped)")

    print(f"{len(t)} samples, {t[-1]:.3f}s, travel {x[-1]:.1f}cm\n")

    K, T1, Tt, rms = fit(t, x, np.linspace(0.01, 1.5, 150),
                         np.linspace(0.0, min(1.0, t[-1] * 0.5), 100))
    K, T1, Tt, rms = fit(t, x,
                         np.linspace(max(T1 - 0.05, 1e-3), T1 + 0.05, 120),
                         np.linspace(max(Tt - 0.05, 0.0), Tt + 0.05, 120))

    print("=== IT1 fit ===")
    print(f"  Zeitkonstante  T1 = {T1:.4f} s")
    print(f"  Totzeit        Tt = {Tt:.4f} s")
    print(f"  Endgeschwindigkeit = {K:.2f} cm/s")
    if duty:
        print(f"  K_I = {K/duty:.5f} (cm/s)/duty  at duty {duty:.0f}")

    resid = x - step_response(t, K, T1, Tt)
    print(f"\n  residual {resid.min():+.2f} .. {resid.max():+.2f} cm"
          f"   RMS {rms:.3f} cm")

    print("\n=== commanded duration per target distance ===")
    for d in (10, 25, 50, 100, 150, 200):
        dur = duration_for(d, K, T1, Tt)
        if dur is None:
            print(f"  {d:4} cm  - beyond fitted range")
        else:
            naive = d / K
            print(f"  {d:4} cm  ->  {dur:5.2f} s   "
                  f"(naive {naive:.2f}s, err {100*(naive-dur)/dur:+.0f}%)")

    print("\n=== fit vs measured ===")
    for i in range(0, len(t), max(1, len(t) // 12)):
        m = step_response(np.array([t[i]]), K, T1, Tt)[0]
        print(f"  t={t[i]:5.2f}s  measured {x[i]:6.1f}  model {m:6.1f}"
              f"  diff {x[i]-m:+5.1f}")


if __name__ == "__main__":
    main()
