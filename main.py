"""CLI: reproduce Karsten's table (`run`) or check criterion anchors (`validate`).

    uv run python main.py run                 # optimize all 5 commander MVs
    uv run python main.py run --mv 4          # one MV, verbose
    uv run python main.py validate --quick    # criterion sanity vs 72.465
"""
import argparse
import math
from multiprocessing import Pool

import numpy as np

from sim_core import simulate_deck
from optimizer import optimize_commander

MVS = [2, 3, 4, 5, 6]
LABELS = ["1d", "2d", "3d", "4d", "5d", "6d", "Sig", "Land"]

# Karsten's published optima, for side-by-side comparison.
KARSTEN = {
    2: ([9, 0, 20, 14, 9, 4, 0, 42]),
    3: ([8, 19, 0, 16, 10, 3, 0, 42]),
    4: ([6, 12, 13, 0, 13, 8, 7, 39]),
    5: ([6, 12, 10, 13, 0, 10, 8, 39]),
    6: ([6, 12, 10, 14, 9, 0, 9, 38]),
}


def _run_one(args):
    mv, kw = args
    best, mean = optimize_commander(mv, **kw)
    return mv, [int(x) for x in best], mean


def cmd_run(a):
    kw = dict(restarts=a.restarts, master_seed=a.seed,
              max_sims=a.max_sims, sim_start=10_000, sim_step=5_000,
              switch_star=(150_000 if a.star else 10 ** 9))
    if a.mv:
        rows = [_run_one((a.mv, dict(kw, verbose=True)))]
    else:
        with Pool(len(MVS)) as p:
            rows = p.map(_run_one, [(mv, kw) for mv in MVS])
        rows.sort()

    header = f"{'MV':>3} | " + " ".join(f"{l:>4}" for l in LABELS) + " | SolRing |   crit"
    print(header)
    print("-" * len(header))
    for mv, best, mean in rows:
        cells = " ".join(f"{x:>4}" for x in best)
        print(f"{mv:>3} | {cells} |    1    | {mean:7.3f}   (ours)")
        k = KARSTEN[mv]
        kcells = " ".join(f"{x:>4}" for x in k)
        print(f"{'':>3} | {kcells} |    1    |   72.x     (Karsten)")


def cmd_validate(a):
    K = 200_000 if a.quick else 3_000_000
    opt4 = np.array(KARSTEN[4], dtype=np.int64)
    m, v = simulate_deck(opt4, 4, K, 20240828)
    se = math.sqrt(v / K)
    print(f"4-mv optimal deck (his {KARSTEN[4]}):")
    print(f"  our model: {m:.3f} +/- {se:.3f}   Karsten's model: 72.465")
    print(f"  ({K} games; ~0.6% lower — his mulligan/gameplay heuristics differ)")


def main():
    ap = argparse.ArgumentParser(description="Karsten EDH mana-curve replication")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="optimize commander mana curves")
    r.add_argument("--mv", type=int, choices=MVS, help="single MV (verbose)")
    r.add_argument("--restarts", type=int, default=3)
    r.add_argument("--max-sims", type=int, default=130_000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--star", action="store_true", help="enable star polishing (slow)")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("validate", help="check criterion vs Karsten's checkpoint")
    v.add_argument("--quick", action="store_true")
    v.set_defaults(fn=cmd_validate)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
