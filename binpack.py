"""Value-maximizing mana spend (bin-pack) for the per-turn play policy.

Replaces the greedy highest-first dump: after ramp (land / Sol Ring / signets),
choose the subset of hand creature-drops (+ the commander) that MAXIMIZES
min(sum of values, space) with total cost <= open mana -- breaking ties by FEWEST
cards played (hold the rest for post-wipe rebuild). value == cmc for a k-drop; the
commander is one item of cost == value == its MV. A card may OVERSHOOT the cap to
reach it (scoring stops at the cap).

The bounded-knapsack DP is small (<=~8x25 cells, and only runs while board < cap),
so it is recomputed per call rather than memoized -- numba forbids a global
typed.Dict (compile-time-constant limitation). The plan is packed into an int64:
q1..q6 in 5-bit fields (bits 0..29), commander in bit 30.
"""
import numpy as np
from numba import int64, njit


@njit(cache=True)
def solve_pack(mana, space, c1, c2, c3, c4, c5, c6, cmdr_cmc):
    """Bounded knapsack. Items: k-drops (cost=value=k, qty=c_k) + commander
    (cost=value=cmdr_cmc, qty 1 if cmdr_cmc>0). Maximize min(sum, space) with
    sum-of-costs <= mana; tie-break fewest cards. Returns the packed int64 plan."""
    n = 6 + (1 if cmdr_cmc > 0 else 0)
    costs = np.empty(n, dtype=np.int64)
    qtys = np.empty(n, dtype=np.int64)
    for k in range(6):
        costs[k] = k + 1
    qtys[0] = c1
    qtys[1] = c2
    qtys[2] = c3
    qtys[3] = c4
    qtys[4] = c5
    qtys[5] = c6
    if cmdr_cmc > 0:
        costs[6] = cmdr_cmc
        qtys[6] = 1
    total = 0
    for i in range(n):
        total += costs[i] * qtys[i]
    C = mana if mana < total else total
    C = max(C, 0)
    INF = 1 << 30
    dp = np.full((n + 1, C + 1), INF, dtype=np.int64)   # dp[i,s]=min cards for sum s
    dp[0, 0] = 0
    for i in range(1, n + 1):
        cost = costs[i - 1]
        q = qtys[i - 1]
        for s in range(C + 1):
            bestc = INF
            t = 0
            while t <= q and t * cost <= s:
                cand = dp[i - 1, s - t * cost] + t
                bestc = min(bestc, cand)
                t += 1
            dp[i, s] = bestc
    # pick the sum s maximizing scored value min(s, space); tie-break fewest cards
    best_s = 0
    best_scored = -1.0e18
    best_cards = INF
    for s in range(C + 1):
        if dp[n, s] >= INF:
            continue
        sc = s if s < space else space
        if sc > best_scored or (sc == best_scored and dp[n, s] < best_cards):
            best_scored = sc
            best_cards = dp[n, s]
            best_s = s
    # reconstruct chosen counts (smallest matching t per item) -> packed int64
    plan = int64(0)
    s = best_s
    for i in range(n, 0, -1):
        cost = costs[i - 1]
        q = qtys[i - 1]
        chosen = 0
        t = 0
        while t <= q and t * cost <= s:
            if dp[i - 1, s - t * cost] + t == dp[i, s]:
                chosen = t
                break
            t += 1
        s -= chosen * cost
        if i <= 6:
            plan |= int64(chosen) << int64((i - 1) * 5)
        elif chosen > 0:
            plan |= int64(1) << int64(30)
    return plan
