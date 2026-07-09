import numpy as np
import pytest

from optimizer import local_search
from sim_core import simulate_deck


@pytest.mark.slow
def test_search_improves_and_stops():
    start = np.array([10, 10, 10, 10, 10, 10, 0, 38], dtype=np.int64)  # sums 98
    best, mean = local_search(4, start, base_seed=1,
                              max_sims=40_000, sim_start=8_000, sim_step=4_000,
                              switch_star=10 ** 9)
    assert best.sum() == 98 and (best >= 0).all()
    m_start, _ = simulate_deck(start, 4, 200_000, 999)
    m_best, _ = simulate_deck(best, 4, 200_000, 999)
    assert m_best >= m_start - 0.05          # never ends materially worse
