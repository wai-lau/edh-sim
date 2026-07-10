"""Final converge run: optimize every (horizon, MV) cell with optimize_precise
(explore-cheap -> select-precise), full model = adaptive mulligan + board wipes +
score cap 10 + draw rules. Warm-started, per-cell crash-safe save, bracket curves.

    uv run python run_final.py --seed-json <wipe_pass3.json>
"""
import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np

from optimizer import optimize_precise, _fix_sum, START_DECKS

MVS = [2, 3, 4, 5, 6]
TURNS = list(range(2, 16))                     # 2..15
CENTERS = {7: "B4 Optimized", 9: "B3 Upgraded", 11: "B2 Core"}
SIGMAS = [1.5, 2.0, 2.5]
CAP = 12.0
SC = "/tmp/claude-1000/-home-wai-src-edh-sim/60ad851b-b04c-47bd-8207-d12d4aaf370d/scratchpad"
OUT = f"{SC}/final.json"
SEED_DECKS = None
CFG = {"n_restarts": 14, "cheap_sims": 35_000, "final_sims": 900_000,
       "sim_start": 12_000, "sim_step": 12_000}


def worker(task):
    t, mv = task
    try:
        if SEED_DECKS is not None and (t, mv) in SEED_DECKS:
            start = np.asarray(SEED_DECKS[(t, mv)], dtype=np.int64)
        else:
            start = _fix_sum(START_DECKS[mv])
        best, crit = optimize_precise(
            mv, t, start, master_seed=1,
            n_restarts=CFG["n_restarts"], cheap_sims=CFG["cheap_sims"],
            sim_start=CFG["sim_start"], sim_step=CFG["sim_step"],
            final_sims=CFG["final_sims"], adaptive=True, wipes=True, cap=CAP)
        return (t, mv), [int(x) for x in best], float(crit), None
    except Exception as e:
        return (t, mv), None, -1e9, repr(e)


def reweight(decks, mu, sigma):
    w = {t: math.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) for t in TURNS}
    z = sum(w.values())
    out = {}
    for mv in MVS:
        acc = [0.0] * 9
        for t in TURNS:
            for k in range(9):
                acc[k] += (w[t] / z) * decks[(t, mv)][k]
        r = [int(round(x)) for x in acc]
        r[7] += 98 - sum(r)
        out[mv] = r
    return out


def save(decks, crits, done, total, elapsed):
    brackets = {}
    if all((t, mv) in decks for t in TURNS for mv in MVS):
        for s in SIGMAS:
            brackets[str(s)] = {str(mu): {str(mv): reweight(decks, mu, s)[mv] for mv in MVS}
                                for mu in CENTERS}
    with open(OUT, "w") as f:
        json.dump({"cfg": CFG, "cap": CAP, "done": done, "total": total,
                   "elapsed_s": round(elapsed),
                   "raw": {f"{t},{mv}": {"deck": decks[(t, mv)], "crit": crits[(t, mv)]}
                           for (t, mv) in decks},
                   "brackets": brackets}, f, indent=1)


def main():
    global OUT, SEED_DECKS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-json", type=str, default="")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--restarts", type=int, default=CFG["n_restarts"])
    ap.add_argument("--cheap-sims", type=int, default=CFG["cheap_sims"])
    ap.add_argument("--final-sims", type=int, default=CFG["final_sims"])
    ap.add_argument("--out", type=str, default=OUT)
    a = ap.parse_args()
    CFG["n_restarts"] = a.restarts
    CFG["cheap_sims"] = a.cheap_sims
    CFG["final_sims"] = a.final_sims
    OUT = a.out
    if a.seed_json:
        raw = json.load(open(a.seed_json))["raw"]
        SEED_DECKS = {tuple(int(x) for x in k.split(",")): v["deck"] for k, v in raw.items()}
        print(f"warm-start: {a.seed_json} ({len(SEED_DECKS)} cells)", flush=True)

    cells = [(t, mv) for t in TURNS for mv in MVS]
    decks, crits = {}, {}
    t0 = time.time()
    print(f"run_final: {len(cells)} cells, cap={CAP}, {CFG}", flush=True)
    with Pool(a.workers) as p:
        for i, (key, deck, crit, err) in enumerate(p.imap_unordered(worker, cells), 1):
            if err is None:
                decks[key] = deck
                crits[key] = crit
            save(decks, crits, i, len(cells), time.time() - t0)
            m = "" if err is None else f" ERR {err}"
            print(f"[{i:>3}/{len(cells)}] T={key[0]:>2} MV={key[1]} "
                  f"crit={crit:6.1f} D={deck[8] if deck else '-'}{m}", flush=True)
    print(f"DONE {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
