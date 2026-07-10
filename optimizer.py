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

from sim_core import simulate_deck, simulate_deck_cum


# --------------------------------------------------------------------------- #
# Neighborhoods (balanced 99-card moves; Sol Ring fixed at 1, not a category). #
# --------------------------------------------------------------------------- #
def neighbors_cross(deck):
    """Single swaps: cut one card from A, add one to B. sum stays 98."""
    out = []
    n = deck.shape[0]
    for a in range(n):
        if deck[a] == 0:
            continue
        for b in range(n):
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
    for delta in itertools.product((-1, 0, 1), repeat=deck.shape[0]):
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
                 switch_star=150_000, verbose=False, n_turns=7, adaptive=False,
                 wipes=False, cap=1.0e9):
    cache = {}   # key -> (n_games, pooled_mean)

    def evaluate(deck, n_games, seed):
        k = _key(deck)
        m, _ = simulate_deck(np.asarray(deck, dtype=np.int64),
                             commander_mv, n_games, seed, n_turns, adaptive, wipes, cap)
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
    2: np.array([10, 10, 12, 10, 6, 2, 0, 48, 0], dtype=np.int64),
    3: np.array([8, 12, 10, 10, 6, 3, 4, 45, 0], dtype=np.int64),
    4: np.array([6, 12, 10, 10, 8, 4, 6, 42, 0], dtype=np.int64),
    5: np.array([6, 12, 10, 10, 8, 5, 8, 39, 0], dtype=np.int64),
    6: np.array([6, 12, 10, 12, 8, 3, 9, 38, 0], dtype=np.int64),
}


def sweep_mv_joint(mv, horizons, start_decks, base_seed, max_sims=200_000,
                   sim_start=15_000, sim_step=15_000, adaptive=True, wipes=True,
                   verbose=False, cap=1.0e9):
    """Optimize the deck for EVERY horizon at once (one commander MV), sharing
    evaluations across horizons. Each iteration: take the union of the cross
    neighborhoods of all current per-horizon bests, evaluate every unique
    candidate ONCE with a cumulative game batch (all horizons from one pass), then
    update each horizon's best. Since adjacent-horizon optima overlap, the union
    is far smaller than 14 separate neighborhoods -> big speedup vs per-cell.

    Returns {T: (best_deck, criterion)}.
    """
    max_turn = max(horizons)
    best = {T: np.asarray(start_decks[T], dtype=np.int64).copy() for T in horizons}
    sims = sim_start
    it = 0
    while True:
        it += 1
        seed = base_seed + it * 7919                     # CRN within iter, fresh per iter
        cands = {}                                        # union of neighborhoods + bests
        for T in horizons:
            cands[tuple(int(x) for x in best[T])] = best[T]
            for nb in neighbors_cross(best[T]):
                cands[tuple(int(x) for x in nb)] = nb
        crit = {k: simulate_deck_cum(np.asarray(d, dtype=np.int64), mv, sims, seed,
                                     max_turn, adaptive, wipes, cap)
                for k, d in cands.items()}
        moved = False
        for T in horizons:
            bd, bv = best[T], -1e18
            for k, d in cands.items():
                v = crit[k][T - 1]
                if v > bv:
                    bv, bd = v, d
            if not np.array_equal(bd, best[T]):
                best[T] = np.asarray(bd, dtype=np.int64).copy()
                moved = True
        if verbose:
            print(f"  mv{mv} it{it} sims={sims} cands={len(cands)} "
                  f"{'MOVE' if moved else 'stay'}", flush=True)
        sims += sim_step
        if not moved and sims > max_sims:
            break
        if sims > max_sims * 3:
            break
    seed = base_seed + 987659
    out = {}
    for T in horizons:
        m = simulate_deck_cum(best[T], mv, max_sims, seed, max_turn, adaptive, wipes, cap)
        out[T] = (best[T], float(m[T - 1]))
    return out


def deck_with_draw(base8, Y):
    """Add Y draw cards to an 8-slot deck [1d..6d, sig, land] (sums 98, +Sol Ring),
    renormalizing the other categories by 99/(99+Y), rounding, then round-robin
    over CMC 1..6 to land exactly on 98 (+ Sol Ring = 99). Returns a 9-slot vector
    [1d..6d, sig, land, draw]."""
    f = 99.0 / (99.0 + Y)
    v = [int(round(base8[i] * f)) for i in range(8)] + [int(Y)]
    s = sum(v)
    j = 0
    while s != 98:
        idx = j % 6                       # cycle CMC 1..6 (indices 0..5)
        if s < 98:
            v[idx] += 1
            s += 1
        elif v[idx] > 0:
            v[idx] -= 1
            s -= 1
        j += 1
    return np.array(v, dtype=np.int64)


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


def optimize_precise(mv, n_turns, start_deck, master_seed=0, n_restarts=10,
                     cheap_sims=35_000, sim_start=12_000, sim_step=12_000,
                     final_sims=600_000, adaptive=True, wipes=True, cap=1.0e9):
    """Explore cheap, select precise. Run n_restarts CHEAP local searches from
    jittered starts (few sims each -> diverse but noisy), collect the unique
    finalist decks, then RE-EVALUATE them all at final_sims on one shared CRN seed
    and return the true best. This avoids the max-of-noisy-estimates bias of naive
    best-of-N: exploration is cheap, but the final pick is high-precision."""
    base = np.asarray(start_deck, dtype=np.int64)
    n = base.shape[0]
    finalists = {}
    for r in range(n_restarts):
        s = base.copy()
        if r > 0:
            s[(r * 3) % n] += 1
            s = _fix_sum(s)
        best, _ = local_search(mv, s, base_seed=master_seed + r * 104729,
                               max_sims=cheap_sims, sim_start=sim_start,
                               sim_step=sim_step, switch_star=10 ** 9,
                               n_turns=n_turns, adaptive=adaptive, wipes=wipes, cap=cap)
        finalists[tuple(int(x) for x in best)] = best
    seed = master_seed + 999983                       # shared CRN seed for the showdown
    best_deck, best_crit = base, -1e18
    for f in finalists.values():
        m, _ = simulate_deck(np.asarray(f, dtype=np.int64), mv, final_sims, seed,
                             n_turns, adaptive, wipes, cap)
        if m > best_crit:
            best_crit, best_deck = m, np.asarray(f, dtype=np.int64)
    return best_deck, best_crit
