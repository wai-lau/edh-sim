import numpy as np

from sim_core import simulate_deck, simulate_deck_cum

DECK = np.array([6, 12, 10, 10, 8, 4, 6, 42, 0], dtype=np.int64)


def test_cum_matches_per_horizon_for_T7plus():
    # one cumulative pass must equal separate per-horizon sims (same seed) for
    # T>=7, where the mulligan is horizon-independent.
    K, seed, mv = 20000, 123, 4
    means = simulate_deck_cum(DECK, mv, K, seed, 12, True, True)  # adaptive + wipes
    for T in range(7, 13):
        m, _ = simulate_deck(DECK, mv, K, seed, T, True, True)
        assert abs(m - means[T - 1]) < 1e-9, (T, m, means[T - 1])


def test_cum_is_increasing_in_horizon():
    means = simulate_deck_cum(DECK, 4, 20000, 5, 12, True, True)
    for t in range(1, 12):
        assert means[t] >= means[t - 1]      # cumulative -> non-decreasing
