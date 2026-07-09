import numpy as np
from sim_core import build_library, draw_opening_hand, new_rng

DECK = np.array([6, 12, 13, 0, 13, 8, 7, 39], dtype=np.int64)  # 4-mv row, sums 98


def test_opening_hand_size_and_ptr():
    lib = build_library(DECK)
    hand, ptr = draw_opening_hand(lib, new_rng(99))
    assert 4 <= int(hand.sum()) <= 7
    assert ptr == 7


def test_opening_deterministic():
    h1, _ = draw_opening_hand(build_library(DECK), new_rng(3))
    h2, _ = draw_opening_hand(build_library(DECK), new_rng(3))
    assert np.array_equal(h1, h2)


def test_kept_hand_size_valid():
    for s in range(200):
        h, _ = draw_opening_hand(build_library(DECK), new_rng(s))
        assert int(h.sum()) >= 4
