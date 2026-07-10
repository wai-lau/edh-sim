import numpy as np
from sim_core import score_board


def board(d=None):
    b = np.zeros(9, dtype=np.int64)
    for code, n in (d or {}).items():
        b[code] = n
    return b


def test_worked_example_sums_to_23():
    cmv = 6
    per_turn = [
        score_board(board({1: 0}), 0, cmv),   # T1: 0
        score_board(board({1: 0}), 0, cmv),   # T2: 0
        score_board(board({1: 1}), 0, cmv),   # T3: 1
        score_board(board({1: 1}), 0, cmv),   # T4: 1
        score_board(board({1: 1}), 1, cmv),   # T5: 1+6
        score_board(board({1: 1}), 1, cmv),   # T6: 7
        score_board(board({1: 1}), 1, cmv),   # T7: 7
    ]
    assert sum(per_turn) == 23


def test_six_drop_worth_6point2():
    assert score_board(board({6: 1}), 0, 4) == 6.2


def test_rocks_and_lands_excluded():
    assert score_board(board({0: 5, 7: 3, 8: 1}), 0, 4) == 0.0


def test_commander_raw_mv_not_6point2():
    assert score_board(board(), 1, 6) == 6.0


def test_score_board_caps_at_value():
    assert score_board(board({5: 5}), 0, 4, 10.0) == 10.0    # 25 value -> min(.,10)
    assert score_board(board({5: 5}), 0, 4) == 25.0          # default: no cap
