import numpy as np

from optimizer import neighbors_cross, neighbors_star

DECK = np.array([6, 12, 13, 0, 13, 8, 7, 39], dtype=np.int64)


def _valid(ds):
    for d in ds:
        assert d.sum() == 98 and (d >= 0).all() and d.shape[0] == 8


def test_cross_are_single_swaps():
    ns = neighbors_cross(DECK)
    _valid(ns)
    for d in ns:
        diff = d - DECK
        assert np.abs(diff).sum() == 2 and diff.max() == 1 and diff.min() == -1


def test_star_within_one_each_axis():
    ns = neighbors_star(DECK)
    _valid(ns)
    for d in ns:
        assert np.abs(d - DECK).max() <= 1
    assert len(ns) > 100
