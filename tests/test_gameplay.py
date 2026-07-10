import numpy as np
from sim_core import play_turn, new_rng


def blank():
    return np.zeros(10, dtype=np.int64)


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


def test_draw_card_sink_pays_x_draws_x():
    # only a draw card in hand, 4 lands in play, library of 1-drops.
    hand = blank(); hand[9] = 1                 # DRAW=9
    board = blank(); board[0] = 4               # N=4
    cs = cst(4, cast=1)                          # commander already cast
    lib = np.array([1] * 12, dtype=np.int8)      # all 1-drops
    ptr = play_turn(hand, board, cs, 5, lib, 0, new_rng(1))
    assert hand[9] == 0                          # draw card was cast
    # turn-draw (1) casts a 1-drop (mana 4->3); draw sink pays X=3 -> draws 3 more
    assert board[1] == 1                         # the drawn 1-drop got played
    assert ptr == 4                              # 1 turn-draw + 3 from pay-3-draw-3
    assert hand[1] == 3                          # three drawn 1-drops sit in hand


def test_value_cap_plays_crossing_then_stops():
    hand = blank(); hand[6] = 5                 # five 6-drops (6.2 each)
    board = blank(); board[0] = 20              # plenty of mana
    cs = cst(1, cast=0)                          # MV-1 cmdr, uncast (capped out anyway)
    play_turn(hand, board, cs, 8, lib_of(0), 0, new_rng(1), 10.0)  # cap=10
    # board value 0 -> 6.2 (cast) -> 12.4 (cast the crossing card), now >=10 -> stop.
    assert board[6] == 2                         # crossing card played, then stopped
    assert hand[6] == 3


def test_draw_held_when_hand_full():
    hand = blank(); hand[9] = 1; hand[1] = 7    # draw + 7 one-drops (castable by mana)
    board = blank(); board[0] = 5; board[5] = 2  # board value 10 -> at cap
    cs = cst(4, cast=1)
    play_turn(hand, board, cs, 8, lib_of(0), 0, new_rng(1), 10.0)  # cap=10
    assert hand[9] == 1                          # hand>=7 + a castable (cap-blocked) card -> held


def test_draw_digs_when_hand_full_but_stuck():
    hand = blank(); hand[9] = 1; hand[6] = 8    # draw + 8 uncastable 6-drops (only ~2 mana)
    board = blank(); board[0] = 1
    cs = cst(4, cast=1)
    play_turn(hand, board, cs, 8, lib_of(0), 0, new_rng(1))
    assert hand[9] == 0                          # hand>=7 but nothing castable -> dig


def test_no_cap_dumps_everything():
    hand = blank(); hand[6] = 3
    board = blank(); board[0] = 20
    cs = cst(1, cast=0)
    play_turn(hand, board, cs, 8, lib_of(0), 0, new_rng(1))       # default: no cap
    assert board[6] == 3                         # 20 mana -> all three 6-drops dumped


def test_wipe_clears_drops_keeps_rocks_lands():
    from sim_core import maybe_wipe
    board = blank()
    board[1] = 2; board[3] = 1                # creatures
    board[0] = 5; board[7] = 2; board[8] = 1  # lands + signets + sol ring
    cs = cst(4, cast=1)
    wstate = np.array([13], dtype=np.int64)   # 0.10*1.2^13 > 1 -> guaranteed wipe
    maybe_wipe(board, cs, wstate, new_rng(1), 5)
    assert board[1] == 0 and board[3] == 0    # drops wiped
    assert board[0] == 5 and board[7] == 2 and board[8] == 1  # lands + rocks survive
    assert cs[1] == 0                          # commander to command zone
    assert wstate[0] == 0                      # counter reset


def test_no_wipe_before_turn5():
    from sim_core import maybe_wipe
    board = blank(); board[1] = 2
    wstate = np.array([13], dtype=np.int64)
    maybe_wipe(board, cst(4, 1), wstate, new_rng(1), 4)   # turn 4 -> nothing
    assert board[1] == 2 and wstate[0] == 13


def test_commander_cast_on_curve():
    hand = blank()
    board = blank()
    board[0] = 4
    cs = cst(4)
    play_turn(hand, board, cs, 4, lib_of(0), 0, new_rng(1))
    assert cs[1] == 1
