"""Overnight experiment: definitive high-sample per-horizon sweep + bracket curves.

Self-pacing and crash-safe:
- Re-optimizes every (T, MV) cell at heavy sims (max_sims=500k), cross neighborhood.
- Runs in restart-ROUNDS; each round is one more independent restart per cell, and
  we keep the best criterion seen for each cell across all rounds.
- After every round: save best-so-far raw optima + reweighted bracket curves
  (sigma = 1.5 / 2 / 2.5) to JSON, and print a round summary. So a crash or a
  morning kill loses at most the in-flight round.
- Stops once the wall-clock budget (default 7h) is exceeded, checked between
  rounds. More rounds => more restarts => less residual noise.

    uv run python overnight.py            # 7h default
    uv run python overnight.py --hours 8 --max-sims 600000
"""
import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np

from optimizer import optimize_commander, local_search

MVS = [2, 3, 4, 5, 6]
TURNS = list(range(2, 16))                 # 2..15: covers B5 (1-6) up to B1 (12+)
# bracket centers from r/EDH game-length data (midpoints). B5 (1-6) and B1 (12+)
# are ignored per request -- the extremes the model fits worst. Middle three only.
CENTERS = {7: "B4 Optimized", 9: "B3 Upgraded", 11: "B2 Core"}
SIGMAS = [1.5, 2.0, 2.5]

OUT_JSON = ("/tmp/claude-1000/-home-wai-src-edh-sim/"
            "60ad851b-b04c-47bd-8207-d12d4aaf370d/scratchpad/overnight.json")

CFG = {"max_sims": 500_000, "sim_start": 50_000, "sim_step": 30_000}
SEED_DECKS = None          # {(t, mv): deck} warm-start decks, or None (cold start)


def worker(task):
    t, mv, seed = task
    try:
        if SEED_DECKS is not None and (t, mv) in SEED_DECKS:   # warm start
            start = np.asarray(SEED_DECKS[(t, mv)], dtype=np.int64)
            best, mean = local_search(
                mv, start, base_seed=seed,
                max_sims=CFG["max_sims"], sim_start=CFG["sim_start"],
                sim_step=CFG["sim_step"], switch_star=10 ** 9, n_turns=t,
                adaptive=True)
        else:
            best, mean = optimize_commander(
                mv, restarts=1, master_seed=seed,
                max_sims=CFG["max_sims"], sim_start=CFG["sim_start"],
                sim_step=CFG["sim_step"], switch_star=10 ** 9, n_turns=t,
                adaptive=True)             # use the DP-derived adaptive mulligan
        return (t, mv), [int(x) for x in best], float(mean), None
    except Exception as e:                 # never let one cell kill the round
        return (t, mv), None, -1e9, repr(e)


def reweight(best_deck, turns, mu, sigma):
    w = {t: math.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) for t in turns}
    z = sum(w.values())
    out = {}
    for mv in MVS:
        n = len(best_deck[(TURNS[0], mv)])          # 8 (no draw) or 9 (with draw)
        acc = [0.0] * n
        for t in turns:
            for k in range(n):
                acc[k] += (w[t] / z) * best_deck[(t, mv)][k]
        r = [int(round(x)) for x in acc]
        r[7] += 98 - sum(r)                          # land absorbs rounding residual
        out[mv] = r
    return out


def save(best_deck, best_crit, rounds_done, elapsed):
    turns = sorted({t for (t, _) in best_deck})
    brackets = {}
    if all((t, mv) in best_deck for t in TURNS for mv in MVS):
        for sigma in SIGMAS:
            brackets[str(sigma)] = {
                str(mu): {str(mv): reweight(best_deck, TURNS, mu, sigma)[mv] for mv in MVS}
                for mu in CENTERS}
    with open(OUT_JSON, "w") as f:
        json.dump({
            "config": CFG, "rounds_done": rounds_done, "elapsed_s": round(elapsed),
            "raw": {f"{t},{mv}": {"deck": best_deck[(t, mv)], "crit": best_crit[(t, mv)]}
                    for (t, mv) in best_deck},
            "brackets": brackets,
        }, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--max-sims", type=int, default=500_000)
    ap.add_argument("--sim-start", type=int, default=50_000)
    ap.add_argument("--sim-step", type=int, default=30_000)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--max-rounds", type=int, default=40)
    ap.add_argument("--seed-json", type=str, default="",
                    help="warm-start each cell from the 'raw' optima in this JSON")
    a = ap.parse_args()
    CFG["max_sims"] = a.max_sims
    CFG["sim_start"] = a.sim_start
    CFG["sim_step"] = a.sim_step
    budget = a.hours * 3600

    if a.seed_json:
        global SEED_DECKS
        raw = json.load(open(a.seed_json))["raw"]
        SEED_DECKS = {tuple(int(x) for x in k.split(",")): v["deck"]
                      for k, v in raw.items()}
        print(f"warm-starting from {a.seed_json} ({len(SEED_DECKS)} cells)", flush=True)

    cells = [(t, mv) for t in TURNS for mv in MVS]
    best_deck, best_crit = {}, {}
    t0 = time.time()
    print(f"overnight: {len(cells)} cells, max_sims={CFG['max_sims']}, "
          f"workers={a.workers}, budget={a.hours}h", flush=True)

    rnd = 0
    with Pool(a.workers) as p:
        while rnd < a.max_rounds and (time.time() - t0) < budget:
            rnd += 1
            tasks = [(t, mv, rnd * 100003 + i) for i, (t, mv) in enumerate(cells)]
            improved = 0
            errs = 0
            for key, deck, mean, err in p.imap_unordered(worker, tasks):
                if err is not None:
                    errs += 1
                    continue
                if key not in best_crit or mean > best_crit[key]:
                    best_crit[key] = mean
                    best_deck[key] = deck
                    improved += 1
            el = time.time() - t0
            save(best_deck, best_crit, rnd, el)
            print(f"[round {rnd:>2}] {el/3600:4.2f}h elapsed | improved {improved:>2}/"
                  f"{len(cells)} | errs {errs} | saved", flush=True)

    el = time.time() - t0
    save(best_deck, best_crit, rnd, el)
    print(f"\nDONE: {rnd} rounds in {el/3600:.2f}h -> {OUT_JSON}", flush=True)

    # final human-readable dump
    print("\nBEST per-horizon optima  [1d 2d 3d 4d 5d 6d | Sig | Land | Draw | crit]")
    for t in TURNS:
        print(f"### T={t}")
        for mv in MVS:
            c = best_deck.get((t, mv))
            if c:
                dr = c[8] if len(c) > 8 else 0
                print(f" MV{mv} | {c[0]:>2} {c[1]:>2} {c[2]:>2} {c[3]:>2} {c[4]:>2} "
                      f"{c[5]:>2} | {c[6]:>2} | {c[7]:>2} | {dr:>2} | {best_crit[(t, mv)]:.3f}")
    if all((t, mv) in best_deck for t in TURNS for mv in MVS):
        print("\nBRACKET curves (sigma=2.0)  [.. | Sig | Land | Draw]:")
        for mu, label in CENTERS.items():
            bd = reweight(best_deck, TURNS, mu, 2.0)
            print(f"### mu={mu} {label}")
            for mv in MVS:
                c = bd[mv]
                dr = c[8] if len(c) > 8 else 0
                print(f" MV{mv} | {c[0]:>2} {c[1]:>2} {c[2]:>2} {c[3]:>2} {c[4]:>2} "
                      f"{c[5]:>2} | {c[6]:>2} | {c[7]:>2} | {dr:>2}")


if __name__ == "__main__":
    main()
