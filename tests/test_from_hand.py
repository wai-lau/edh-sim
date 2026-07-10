import numpy as np
from sim_core import simulate_from_hand, vkeep

DECK4 = np.array([7, 19, 16, 0, 10, 5, 0, 41], dtype=np.int64)  # 4-mv, T7 optimum
Z = np.zeros(10, dtype=np.int64)


def h9(d):  # hand by code {code: n}
    a = np.zeros(10, dtype=np.int64)
    for c, n in d.items():
        a[c] = n
    return a


BALANCED = h9({0: 4, 2: 1, 3: 1, 4: 1})   # 4 lands + 3 spells
FLOODED = h9({0: 7})                        # 7 lands
SCREWED = h9({2: 2, 3: 2, 4: 2, 5: 1})      # 0 lands, 7 spells


def test_from_hand_deterministic():
    assert simulate_from_hand(DECK4, BALANCED, Z, 4, 123, 7) == \
           simulate_from_hand(DECK4, BALANCED, Z, 4, 123, 7)


def test_from_hand_positive():
    assert simulate_from_hand(DECK4, BALANCED, Z, 4, 7, 7) > 0


def test_vkeep_reasonable_range():
    v = vkeep(DECK4, BALANCED, Z, 4, 20000, 1, 7)
    assert 50 < v < 90


def test_balanced_beats_flooded():
    vb = vkeep(DECK4, BALANCED, Z, 4, 30000, 1, 7)
    vf = vkeep(DECK4, FLOODED, Z, 4, 30000, 1, 7)
    assert vb > vf


def test_balanced_beats_screwed():
    vb = vkeep(DECK4, BALANCED, Z, 4, 30000, 1, 7)
    vs = vkeep(DECK4, SCREWED, Z, 4, 30000, 1, 7)
    assert vb > vs


def test_removed_cards_leave_draw_pile():
    # keep 4-land hand, bottom 3 spells -> remaining pile has 3 fewer spells.
    kept = h9({0: 4})
    removed = h9({2: 3})
    v = vkeep(DECK4, kept, removed, 4, 10000, 1, 7)
    assert v > 0
