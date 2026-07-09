"""Criterion regression anchors.

Karsten reports 72.465 for his 4-mv optimal deck and 72.434 for a named
perturbation (a +0.031 ordering). Our from-prose reimplementation scores that
same deck ~72.0 in absolute terms (~0.6% lower) because his hand-crafted
mulligan/gameplay heuristics have unspecified degrees of freedom. The robust,
model-independent anchor is the *ordering* (his perturbation is worse), which we
reproduce with the same sign. The absolute band guards our own model against
regressions.
"""
import math
import numpy as np
import pytest

from sim_core import simulate_deck

OPT4 = np.array([6, 12, 13, 0, 13, 8, 7, 39], dtype=np.int64)   # Karsten 72.465
# -1 one/two/three-drop, +1 five/six-drop, +1 signet  (Karsten 72.434)
PERT = np.array([5, 11, 12, 0, 14, 9, 8, 39], dtype=np.int64)


def test_deck_sums():
    assert OPT4.sum() == 98 and PERT.sum() == 98


@pytest.mark.slow
def test_checkpoint_absolute_our_model():
    K = 2_000_000
    m, v = simulate_deck(OPT4, 4, K, 20240828)
    se = math.sqrt(v / K)
    # our model: ~72.0 (Karsten's model: 72.465). Band covers seed/K jitter.
    assert 71.8 <= m <= 72.2, f"{m:.4f} +/- {se:.4f}"


@pytest.mark.slow
def test_checkpoint_ordering_matches_sign():
    K = 4_000_000
    m_opt, _ = simulate_deck(OPT4, 4, K, 111)
    m_pert, _ = simulate_deck(PERT, 4, K, 111)   # CRN: same base seed
    assert m_opt > m_pert                          # same sign as Karsten (+)
