import numpy as np
from sim_core import mulligan_keep_adaptive


def hand(land=0, sig=0, sr=0, drops=None):
    h = np.zeros(9, dtype=np.int64)
    h[0] = land
    h[7] = sig
    h[8] = sr
    for c, n in (drops or {}).items():
        h[c] = n
    return h


def test_keeps_3_4_lands_attempt1():
    assert mulligan_keep_adaptive(hand(land=3, drops={2: 2, 3: 2}), 1, 7)
    assert mulligan_keep_adaptive(hand(land=4, drops={2: 2, 3: 1}), 1, 7)


def test_mulls_five_lands_early():
    # the DP win: 5-land hands are a mulligan on the free early keeps
    assert not mulligan_keep_adaptive(hand(land=5, drops={2: 2}), 1, 7)
    assert not mulligan_keep_adaptive(hand(land=5, drops={2: 2}), 2, 7)


def test_two_lands_need_a_rock():
    assert not mulligan_keep_adaptive(hand(land=2, drops={2: 5}), 1, 7)
    assert mulligan_keep_adaptive(hand(land=2, sig=1, drops={2: 4}), 1, 7)
    assert mulligan_keep_adaptive(hand(land=2, sr=1, drops={2: 4}), 1, 9)


def test_one_land_only_behind_solring():
    assert not mulligan_keep_adaptive(hand(land=1, drops={2: 6}), 1, 7)
    assert mulligan_keep_adaptive(hand(land=1, sr=1, drops={2: 5}), 1, 7)


def test_floor_keeps_anything():
    assert mulligan_keep_adaptive(hand(land=0, drops={6: 7}), 5, 7)


def test_late_attempts_loosen_to_five_lands():
    # a 5-land hand becomes keepable once we're already bottoming
    assert mulligan_keep_adaptive(hand(land=5, drops={2: 1}), 3, 7)
