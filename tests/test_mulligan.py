import numpy as np
from sim_core import mulligan_keep, bottom_cards


def hand(land=0, sig=0, sr=0, drops=None):
    h = np.zeros(9, dtype=np.int64)
    h[0] = land
    h[7] = sig
    h[8] = sr
    for code, n in (drops or {}).items():
        h[code] = n
    return h


def test_hand1_keeps_3_5_lands():
    assert mulligan_keep(hand(land=4, drops={2: 3}), 1)
    assert not mulligan_keep(hand(land=2, drops={2: 5}), 1)
    assert not mulligan_keep(hand(land=6, drops={2: 1}), 1)


def test_hand1_land_signet_cap():
    assert not mulligan_keep(hand(land=3, sig=3, drops={2: 1}), 1)


def test_hand1_solring_branch():
    assert mulligan_keep(hand(land=1, sr=1, drops={2: 5}), 1)
    assert not mulligan_keep(hand(land=0, sr=1, drops={2: 6}), 1)


def test_hand2_keeps_two_land():
    assert not mulligan_keep(hand(land=2, drops={2: 5}), 1)
    assert mulligan_keep(hand(land=2, drops={2: 5}), 2)


def test_hand3_and_4_thresholds():
    for a in (3, 4):
        assert mulligan_keep(hand(land=2, drops={2: 5}), a)
        assert mulligan_keep(hand(land=4, drops={2: 3}), a)
        assert not mulligan_keep(hand(land=5, drops={2: 2}), a)
        assert not mulligan_keep(hand(land=1, drops={2: 6}), a)
        assert mulligan_keep(hand(land=1, sr=1, drops={2: 5}), a)


def test_hand5_always_keep():
    assert mulligan_keep(hand(land=0, drops={6: 7}), 5)


def test_bottom_superfluous_signets_first():
    h = hand(land=3, sig=3, drops={2: 1})
    bottom_cards(h, 2)
    assert h[7] == 1
    assert h[0] == 3 and h[2] == 1


def test_bottom_lands_toward_three():
    h = hand(land=6, drops={2: 1})
    bottom_cards(h, 3)
    assert h[0] == 3


def test_solring_counts_as_land_never_bottomed():
    h = hand(land=3, sr=1, drops={2: 2})
    bottom_cards(h, 1)
    assert h[8] == 1
    assert h[0] == 2


def test_bottom_spells_most_expensive_first():
    h = hand(land=3, drops={2: 1, 5: 1, 6: 1})
    bottom_cards(h, 2)
    assert h[6] == 0 and h[5] == 0 and h[2] == 1
    assert h[0] == 3
