import numpy as np
from sim_core import simulate_game, simulate_deck

DECK4 = np.array([6, 12, 13, 0, 13, 8, 7, 39], dtype=np.int64)


def test_game_runs_and_is_positive():
    assert simulate_game(DECK4, 4, 12345) > 0


def test_game_deterministic():
    assert simulate_game(DECK4, 4, 7) == simulate_game(DECK4, 4, 7)


def test_seed_changes_outcome():
    vals = {simulate_game(DECK4, 4, s) for s in range(50)}
    assert len(vals) > 1


def test_mc_mean_deterministic():
    m1, _ = simulate_deck(DECK4, 4, 5000, 1)
    m2, _ = simulate_deck(DECK4, 4, 5000, 1)
    assert m1 == m2


def test_mc_reasonable_range():
    m, _ = simulate_deck(DECK4, 4, 20000, 2)
    assert 60 < m < 90
