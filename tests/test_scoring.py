import numpy as np

from sim_core import score_board


def board(d=None):
    b = np.zeros(9, dtype=np.int64)
    for code, n in (d or {}).items():
        b[code] = n
    return b


def test_worked_example():
    # Karsten's example, but our MV-6 commander scores 6.2 (a 6-drop), not 6, so
    # T5-7 are 1 + 6.2 = 7.2 -> 2*1 + 3*7.2 = 23.6 (Karsten's raw-6 gave 23).
    cmv = 6
    per_turn = [
        score_board(board({1: 0}), 0, cmv),   # T1: 0
        score_board(board({1: 0}), 0, cmv),   # T2: 0
        score_board(board({1: 1}), 0, cmv),   # T3: 1
        score_board(board({1: 1}), 0, cmv),   # T4: 1
        score_board(board({1: 1}), 1, cmv),   # T5: 1+6.2
        score_board(board({1: 1}), 1, cmv),   # T6: 7.2
        score_board(board({1: 1}), 1, cmv),   # T7: 7.2
    ]
    assert abs(sum(per_turn) - 23.6) < 1e-9


def test_six_drop_worth_6point2():
    assert score_board(board({6: 1}), 0, 4) == 6.2


def test_rocks_and_lands_excluded():
    assert score_board(board({0: 5, 7: 3, 8: 1}), 0, 4) == 0.0


def test_commander_mv6_worth_6point2():
    assert score_board(board(), 1, 6) == 6.2      # MV-6 commander = a 6-drop
    assert score_board(board(), 1, 4) == 4.0      # MV 1-5 commander = raw MV


def test_score_board_caps_at_value():
    assert score_board(board({5: 5}), 0, 4, 10.0) == 10.0    # 25 value -> min(.,10)
    assert score_board(board({5: 5}), 0, 4) == 25.0          # default: no cap
