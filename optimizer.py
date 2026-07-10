"""Local-search optimizer over the compounded-mana criterion.

Pure-Python orchestration; the hot path stays in sim_core. Faithful to
Karsten's cross/star local search plus common-random-numbers (CRN) variance
reduction: within an iteration every candidate is scored on the SAME game seeds,
so Sol Ring's luck cancels in the pairwise comparison. Seeds are refreshed each
iteration to avoid seed overfit; multiple restarts guard local maxima.

See docs/superpowers/specs/2026-07-09-edh-mana-curve-sim-design.md §4.2.
"""
import itertools
import numpy as np

from sim_core import simulate_deck


# --------------------------------------------------------------------------- #
# Neighborhoods (balanced 99-card moves; Sol Ring fixed at 1, not a category). #
# --------------------------------------------------------------------------- #
def neighbors_cross(deck):
    """Single swaps: cut one card from A, add one to B. sum stays 98."""
    out = []
    for a in range(8):
        if deck[a] == 0:
            continue
        for b in range(8):
            if a == b:
                continue
            d = deck.copy()
            d[a] -= 1
            d[b] += 1
            out.append(d)
    return out


def neighbors_star(deck):
    """All decks whose every category count differs by <=1 from deck, sum 98."""
    out = []
    for delta in itertools.product((-1, 0, 1), repeat=8):
        dv = np.array(delta, dtype=np.int64)
        if dv.sum() != 0:
            continue
        if np.abs(dv).sum() == 0:
            continue
        cand = deck + dv
        if (cand >= 0).all():
            out.append(cand)
    return out


# --------------------------------------------------------------------------- #
# Local search.                                                                #
# --------------------------------------------------------------------------- #
def _key(deck):
    return tuple(int(x) for x in deck)


def local_search(commander_mv, start_deck, base_seed,
                 max_sims=200_000, sim_step=1000, sim_start=10_000,
                 switch_star=150_000, verbose=False, n_turns=7, adaptive=False):
    cache = {}   # key -> (n_games, pooled_mean)

    def evaluate(deck, n_games, seed):
        k = _key(deck)
        m, _ = simulate_deck(np.asarray(deck, dtype=np.int64),
                             commander_mv, n_games, seed, n_turns, adaptive)
        prev = cache.get(k)
        if prev is None:
            cache[k] = (n_games, m)
        else:
            pn, pm = prev
            tot = pn + n_games
            cache[k] = (tot, (pm * pn + m * n_games) / tot)
        return cache[k][1]

    best = np.asarray(start_deck, dtype=np.int64).copy()
    sims = sim_start
    evaluate(best, sims, base_seed)
    best_mean = cache[_key(best)][1]
    it = 0

    while True:
        it += 1
        seed = base_seed + it * 7919                     # fresh seed batch
        best_sims = cache[_key(best)][0]
        hood = neighbors_star(best) if best_sims >= switch_star else neighbors_cross(best)

        # CRN: best + every candidate scored on this iteration's seed batch.
        best_cand = best
        best_cand_mean = evaluate(best, sims, seed)
        for cand in hood:
            m = evaluate(cand, sims, seed)
            if m > best_cand_mean:
                best_cand_mean = m
                best_cand = cand

        moved = not np.array_equal(best_cand, best)
        if moved:
            best = best_cand.copy()
            best_mean = best_cand_mean
        if verbose:
            print(f"  it{it:3d} sims={sims:>7} {'*' if best_sims>=switch_star else 'x'} "
                  f"best={_key(best)} crit={best_mean:.3f} {'MOVE' if moved else 'stay'}")
        sims += sim_step

        if not moved and cache[_key(best)][0] > max_sims:
            break
        if sims > max_sims * 3:                          # safety cap
            break
    return best, best_mean


# --------------------------------------------------------------------------- #
# Multi-restart wrapper.                                                       #
# --------------------------------------------------------------------------- #
# Start decks per commander MV (sum 98). Deliberately NOT his answers, so a
# match is the optimizer's own doing. [c1,c2,c3,c4,c5,c6, signet, land].
START_DECKS = {
    2: np.array([10, 10, 12, 10, 6, 2, 0, 48], dtype=np.int64),
    3: np.array([8, 12, 10, 10, 6, 3, 4, 45], dtype=np.int64),
    4: np.array([6, 12, 10, 10, 8, 4, 6, 42], dtype=np.int64),
    5: np.array([6, 12, 10, 10, 8, 5, 8, 39], dtype=np.int64),
    6: np.array([6, 12, 10, 12, 8, 3, 9, 38], dtype=np.int64),
}


def _fix_sum(deck):
    deck = deck.copy()
    while deck.sum() > 98:
        deck[int(np.argmax(deck))] -= 1
    while deck.sum() < 98:
        deck[7] += 1
    return deck


def optimize_commander(commander_mv, restarts=3, master_seed=0, verbose=False, **kw):
    base = _fix_sum(START_DECKS[commander_mv])
    results = []
    for r in range(restarts):
        start = base.copy()
        if r > 0:
            start[(r * 3) % 8] += 1
            start = _fix_sum(start)
        best, mean = local_search(commander_mv, start,
                                  base_seed=master_seed + r * 104729,
                                  verbose=verbose, **kw)
        results.append((mean, best))
    results.sort(key=lambda t: t[0], reverse=True)
    return results[0][1], results[0][0]
