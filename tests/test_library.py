import numpy as np
from sim_core import build_library, shuffle, new_rng

DECK = np.array([9, 0, 20, 14, 9, 4, 0, 42], dtype=np.int64)  # 2-mv row, sums 98


def test_library_composition():
    lib = build_library(DECK)
    assert lib.shape[0] == 99
    counts = np.bincount(lib, minlength=9)
    assert counts[0] == 42
    assert list(counts[1:7]) == [9, 0, 20, 14, 9, 4]
    assert counts[7] == 0
    assert counts[8] == 1


def test_shuffle_is_permutation():
    lib = build_library(DECK)
    ref = lib.copy()
    shuffle(lib, new_rng(123))
    assert sorted(lib.tolist()) == sorted(ref.tolist())


def test_shuffle_deterministic():
    a = build_library(DECK)
    b = build_library(DECK)
    shuffle(a, new_rng(5))
    shuffle(b, new_rng(5))
    assert np.array_equal(a, b)
