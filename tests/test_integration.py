"""Integration: the optimizer independently reproduces Karsten's key insights.

We do NOT assert his exact table (his own runs varied, and our ramp valuation
differs). We assert the qualitative insights the optimizer must land on:
  - Insight #2: ~zero N-drops at the commander's own mana value.
  - Insight #4: a high land count, ~37-43.
"""
import pytest

from optimizer import optimize_commander


@pytest.mark.slow
@pytest.mark.parametrize("mv", [4, 6])
def test_reproduces_insights(mv):
    best, mean = optimize_commander(mv, restarts=2, master_seed=1,
                                    max_sims=45_000, sim_start=10_000,
                                    sim_step=5_000, switch_star=10 ** 9)
    assert best.sum() == 98 and (best >= 0).all()
    assert best[mv - 1] <= 2                 # Insight #2: ~zero N-drops
    assert 37 <= int(best[7]) <= 43          # Insight #4: high land count
