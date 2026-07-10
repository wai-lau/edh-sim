"""Turn-count sweep + Commander-bracket weighted curves, with live progress.

Optimizes the mana curve for every (turn-horizon, commander-MV) pair, then folds
the per-horizon optima into per-bracket decks via a normal weighting centred on
each bracket's characteristic game length.

    uv run python sweep.py                        # defaults: T=5..12, best-of-3, 300k
    uv run python sweep.py --restarts 1 --max-sims 80000   # fast/rough
    uv run python sweep.py --turns-min 6 --turns-max 10 --sigma 1.5

Progress streams a line per completed unit (imap_unordered) with a running ETA.
"""
import argparse
import json
import math
import time
from multiprocessing import Pool

from optimizer import optimize_commander

MVS = [2, 3, 4, 5, 6]
LABELS = "[1d 2d 3d 4d 5d 6d | Sig | Land] +1 Sol Ring"
# bracket -> characteristic game length (mu). Centers pulled inward (5/7/9) so the
# normal tails stay inside the sim range. Grouped: {4,5}->5, {3}->7, {1,2}->9.
BRACKET_MU = {5: 5, 4: 5, 3: 7, 2: 9, 1: 9}
BRACKET_NAME = {1: "Exhibition", 2: "Core", 3: "Upgraded",
                4: "Optimized", 5: "cEDH"}


def worker(task):
    t, mv, r, cfg = task
    best, mean = optimize_commander(
        mv, restarts=1, master_seed=1000 * r + 1,
        max_sims=cfg["max_sims"], sim_start=cfg["sim_start"],
        sim_step=cfg["sim_step"], switch_star=10 ** 9, n_turns=t)
    return (t, mv), [int(x) for x in best], float(mean), (t, mv, r)


def _fmt(mv, c, crit=None):
    row = (f"{mv:>3} | {c[0]:>3}{c[1]:>4}{c[2]:>4}{c[3]:>4}{c[4]:>4}{c[5]:>4} |"
           f" {c[6]:>3} | {c[7]:>4}")
    return row + (f" | {crit:8.3f}" if crit is not None else "")


def run(cfg):
    turns = list(range(cfg["turns_min"], cfg["turns_max"] + 1))
    tasks = [(t, mv, r, cfg) for t in turns for mv in MVS
             for r in range(cfg["restarts"])]
    N = len(tasks)
    print(f"sweep: T={turns[0]}..{turns[-1]}  MV={MVS}  restarts={cfg['restarts']}"
          f"  max_sims={cfg['max_sims']}  workers={cfg['workers']}  units={N}",
          flush=True)

    decks, crits = {}, {}
    t0 = time.time()
    with Pool(cfg["workers"]) as p:
        for i, (key, deck, mean, meta) in enumerate(
                p.imap_unordered(worker, tasks), start=1):
            if key not in crits or mean > crits[key]:
                crits[key] = mean
                decks[key] = deck
            el = time.time() - t0
            eta = el / i * (N - i)
            print(f"[{i:>4}/{N}] {el:6.0f}s  ETA {eta/60:4.1f}m  | "
                  f"T={meta[0]:>2} MV={meta[1]} r={meta[2]}  crit={mean:8.3f}",
                  flush=True)

    # raw tables
    print("\n" + "=" * 62 + f"\nRAW per-turn optima   {LABELS}\n" + "=" * 62)
    for t in turns:
        print(f"\n### Turns = {t}")
        hdr = f"{'1d':>3}{'2d':>4}{'3d':>4}{'4d':>4}{'5d':>4}{'6d':>4}"
        print(f"{'MV':>3} | {hdr} | Sig | Land | crit")
        for mv in MVS:
            print(_fmt(mv, decks[(t, mv)], crits[(t, mv)]))

    # bracket weighted averages
    print("\n" + "=" * 62)
    print(f"BRACKET weighted-average decks  (normal, sigma={cfg['sigma']})\n" + "=" * 62)
    brackets = {}
    for b in (1, 2, 3, 4, 5):
        mu = BRACKET_MU[b]
        w = {t: math.exp(-((t - mu) ** 2) / (2 * cfg["sigma"] ** 2)) for t in turns}
        z = sum(w.values())
        w = {t: w[t] / z for t in turns}
        print(f"\n### Bracket {b} — {BRACKET_NAME[b]}  (mu={mu}, sigma={cfg['sigma']})")
        print(f"{'MV':>3} | {'1d':>3}{'2d':>4}{'3d':>4}{'4d':>4}{'5d':>4}{'6d':>4} | Sig | Land")
        brackets[b] = {}
        for mv in MVS:
            acc = [0.0] * 8
            for t in turns:
                d = decks[(t, mv)]
                for k in range(8):
                    acc[k] += w[t] * d[k]
            rd = [int(round(x)) for x in acc]
            rd[7] += 98 - sum(rd)
            brackets[b][mv] = rd
            print(_fmt(mv, rd))

    if cfg["out"]:
        with open(cfg["out"], "w") as f:
            json.dump({"raw": {f"{t},{mv}": {"deck": decks[(t, mv)],
                                             "crit": crits[(t, mv)]}
                               for t in turns for mv in MVS},
                       "brackets": {str(b): {str(mv): brackets[b][mv] for mv in MVS}
                                    for b in brackets},
                       "config": cfg}, f, indent=1)
        print(f"\nsaved -> {cfg['out']}")
    print(f"\n[total {time.time() - t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-min", type=int, default=5)
    ap.add_argument("--turns-max", type=int, default=12)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--max-sims", type=int, default=300_000)
    ap.add_argument("--sim-start", type=int, default=40_000)
    ap.add_argument("--sim-step", type=int, default=25_000)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    run(vars(a))


if __name__ == "__main__":
    main()
