import numpy as np
from sim_core import play_turn, new_rng


def blank():
    return np.zeros(9, dtype=np.int64)


def cst(mv, cast=0):
    return np.array([mv, cast], dtype=np.int64)


def lib_of(*cards, pad=20):
    a = list(cards) + [0] * pad
    return np.array(a, dtype=np.int8)


def test_turn1_solring_stops_turn():
    hand = blank()
    hand[0] = 1
    hand[8] = 1
    hand[2] = 1
    board = blank()
    play_turn(hand, board, cst(4), 1, lib_of(0), 0, new_rng(1))
    assert board[0] == 1 and board[8] == 1
    assert hand[2] == 1                       # 2-drop NOT cast


def test_turn3_rock_plus_nminus1():
    hand = blank()
    hand[0] = 1
    hand[7] = 1
    hand[2] = 1
    board = blank()
    board[0] = 2
    play_turn(hand, board, cst(5), 3, lib_of(0), 0, new_rng(1))
    assert board[7] == 1 and board[2] == 1    # signet + (N-1)=2-drop


def test_gapfill_two_plus_nminus2():
    hand = blank()
    hand[2] = 1
    hand[3] = 1
    board = blank()
    board[0] = 5
    play_turn(hand, board, cst(4, cast=1), 5, lib_of(0), 0, new_rng(1))
    assert board[2] == 1 and board[3] == 1


def test_greedy_highest_first():
    hand = blank()
    hand[6] = 1
    hand[2] = 1
    board = blank()
    board[0] = 6
    play_turn(hand, board, cst(3, cast=1), 6, lib_of(0), 0, new_rng(1))
    assert board[6] == 1


def test_commander_cast_on_curve():
    hand = blank()
    board = blank()
    board[0] = 4
    cs = cst(4)
    play_turn(hand, board, cs, 4, lib_of(0), 0, new_rng(1))
    assert cs[1] == 1
