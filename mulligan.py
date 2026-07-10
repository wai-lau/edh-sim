"""Optimal mulligan via value-function DP (keep + bottoming, per horizon).

For a fixed deck / commander MV / horizon H:
  V_keep(h)          = E[criterion | keep opening hand h]         (Monte Carlo)
  V_keepbottom(h,b)  = max over which b cards to drop of V_keep(h - dropped)
  V(5) = E_h[V_keepbottom(h,3)]                         # forced keep floor (mull to 4)
  V(m) = E_h[max(V_keepbottom(h,b_m), V(m+1))]          # m = 4..1
Optimal policy at attempt m: keep h iff V_keepbottom(h, b_m) >= V(m+1);
bottom the cards achieving that max.

Bottom counts per attempt (London + free first mulligan): [0, 0, 1, 2, 3].
All V_keep share one CRN seed so comparisons are fair.
"""
import numpy as np
from itertools import combinations_with_replacement

from sim_core import vkeep, build_library, shuffle, new_rng, simulate_deck

CODE_NAMES = ["L", "1", "2", "3", "4", "5", "6", "S", "R"]  # by code 0..8
BOTTOM = [0, 0, 1, 2, 3]          # attempt 1..5
CRN = 20260709                    # shared seed for all V_keep (common random numbers)
Z = np.zeros(9, dtype=np.int64)


def sample_hands(deck, n_hands, seed):
    rng = new_rng(seed)
    hands = []
    for _ in range(n_hands):
        lib = build_library(deck)
        shuffle(lib, rng)
        h = np.zeros(9, dtype=np.int64)
        for i in range(7):
            h[lib[i]] += 1
        hands.append(h)
    return hands


class VKeepCache:
    def __init__(self, deck, mv, H, n_games):
        self.deck, self.mv, self.H, self.n = deck, mv, H, n_games
        self.cache = {}

    def get(self, kept, removed):
        key = (tuple(int(x) for x in kept), tuple(int(x) for x in removed))
        v = self.cache.get(key)
        if v is None:
            v = vkeep(self.deck, kept, removed, self.mv, self.n, CRN, self.H)
            self.cache[key] = v
        return v


def best_bottom(h, b, cache):
    """Max V_keep over choices of b cards to bottom. Returns (value, removal_vec)."""
    if b == 0:
        return cache.get(h, Z), Z.copy()
    codes = [c for c in range(9) if h[c] > 0]
    best_v, best_r = -1e18, None
    for combo in combinations_with_replacement(codes, b):
        r = np.zeros(9, dtype=np.int64)
        for c in combo:
            r[c] += 1
        if np.any(r > h):
            continue
        v = cache.get(h - r, r)
        if v > best_v:
            best_v, best_r = v, r
    return best_v, best_r


def solve(deck, mv, H, n_hands=1500, n_games=30000, seed=1):
    cache = VKeepCache(deck, mv, H, n_games)
    hands = sample_hands(deck, n_hands, seed)

    V = [0.0] * 6
    V[5] = np.mean([best_bottom(h, 3, cache)[0] for h in hands])
    for m in (4, 3, 2, 1):
        b = BOTTOM[m - 1]
        V[m] = np.mean([max(best_bottom(h, b, cache)[0], V[m + 1]) for h in hands])
    return {"V": V, "cache": cache, "hands": hands, "deck": deck, "mv": mv, "H": H}


def keep_rate_by_lands(res, attempt):
    """Fraction of hands optimal-policy keeps at `attempt`, bucketed by land count."""
    V, hands, cache = res["V"], res["hands"], res["cache"]
    b = BOTTOM[attempt - 1]
    cont = V[attempt + 1] if attempt < 5 else -1e18   # value of mulliganing again
    buckets = {}
    for h in hands:
        L = int(h[0])
        keep = best_bottom(h, b, cache)[0] >= cont
        d = buckets.setdefault(L, [0, 0])
        d[0] += 1
        d[1] += 1 if keep else 0
    return {L: (n, k) for L, (n, k) in sorted(buckets.items())}


def gain_vs_karsten(res, n_games=2_000_000):
    """Optimal-policy value V[1] vs Karsten's heuristic mulligan (simulate_deck)."""
    kar, _ = simulate_deck(np.asarray(res["deck"], dtype=np.int64), res["mv"],
                           n_games, CRN, res["H"])
    return res["V"][1], kar
