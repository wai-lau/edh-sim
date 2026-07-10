"""Numba hot path for the Karsten Commander mana-curve replication.

All per-game work runs here on fixed-size int arrays: build library, seeded
shuffle, mulligan, bottoming, 7-turn play, scoring. See
docs/superpowers/specs/2026-07-09-edh-mana-curve-sim-design.md.

Card codes (int8): 0=land, 1..6=k-drop, 7=signet, 8=Sol Ring.
Hand/board = int64[9] indexed by card code. Deck = int64[8] [c1..c6, sig, land],
sum 98; one implicit Sol Ring -> 99-card library.
"""
import numpy as np
from numba import njit, int64

LAND = 0
SIGNET = 7
SOLRING = 8


# --------------------------------------------------------------------------- #
# PRNG: seedable xorshift64 with splitmix64 seeding (Numba-friendly).          #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def new_rng(seed):
    s = np.empty(1, dtype=np.uint64)
    z = np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    if z == np.uint64(0):
        z = np.uint64(0x9E3779B97F4A7C15)
    s[0] = z
    return s


@njit(cache=True)
def next_u64(rng):
    x = rng[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    rng[0] = x
    return x


@njit(cache=True)
def rand_below(rng, n):
    return int64(next_u64(rng) % np.uint64(n))


# --------------------------------------------------------------------------- #
# Library.                                                                     #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def build_library(deck_counts):
    lib = np.empty(99, dtype=np.int8)
    idx = 0
    for k in range(1, 7):                 # k-drops -> codes 1..6
        for _ in range(deck_counts[k - 1]):
            lib[idx] = k
            idx += 1
    for _ in range(deck_counts[6]):       # signets
        lib[idx] = SIGNET
        idx += 1
    for _ in range(deck_counts[7]):       # lands
        lib[idx] = LAND
        idx += 1
    lib[idx] = SOLRING                    # the one Sol Ring
    idx += 1
    return lib


@njit(cache=True)
def shuffle(lib, rng):
    n = lib.shape[0]
    for i in range(n - 1, 0, -1):
        j = rand_below(rng, i + 1)
        tmp = lib[i]
        lib[i] = lib[j]
        lib[j] = tmp


# --------------------------------------------------------------------------- #
# Criterion.                                                                   #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def score_board(board, commander_on, commander_mv):
    total = 0.0
    for k in range(1, 6):                 # 1..5 drops worth k
        total += board[k] * k
    total += board[6] * 6.2               # six-drops worth 6.2
    if commander_on:
        total += commander_mv             # commander worth raw MV
    return total


# --------------------------------------------------------------------------- #
# Mulligan + bottoming.                                                        #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def mulligan_keep(hand, attempt):
    L = hand[LAND]
    S = hand[SIGNET]
    SR = hand[SOLRING] > 0
    if attempt >= 5:
        return True                       # to four: always keep
    if attempt <= 2:
        base = (3 <= L <= 5) and (L + S <= 5)
        base = base or (SR and (1 <= L <= 5))
        if attempt == 2:
            base = base or (L == 2)
        return base
    return (2 <= L <= 4) or (L == 1 and SR)   # attempt 3 or 4


@njit(cache=True)
def bottom_cards(hand, n_bottom):
    left = n_bottom
    # 1. superfluous signets (keep at most one)
    extra_sig = hand[SIGNET] - 1
    if extra_sig > 0:
        take = extra_sig if extra_sig < left else left
        hand[SIGNET] -= take
        left -= take
    # 2. lands toward 3 (Sol Ring counts as a land, never bottomed)
    if left > 0:
        removable = (hand[LAND] + hand[SOLRING]) - 3
        if removable > 0:
            take = removable if removable < left else left
            if take > hand[LAND]:
                take = hand[LAND]
            hand[LAND] -= take
            left -= take
    # 3. spells, most expensive first (6 -> 1)
    k = 6
    while left > 0 and k >= 1:
        if hand[k] > 0:
            take = hand[k] if hand[k] < left else left
            hand[k] -= take
            left -= take
        else:
            k -= 1
    # 4. last resort: the kept signet
    if left > 0 and hand[SIGNET] > 0:
        take = hand[SIGNET] if hand[SIGNET] < left else left
        hand[SIGNET] -= take
        left -= take


@njit(cache=True)
def mulligan_keep_adaptive(hand, attempt, n_turns):
    """Horizon-aware keep policy distilled from the value-function DP.

    Universal: never keep 5+ lands or 0-1 land on the free first two hands.
    Fast (H<=6): 3 lands + a turn-1-2 play (4 lands needs a stronger curve).
    Mid  (7-8): 3-4 lands + two cheap (1-3 drop) plays.
    Slow (H>=9): 3-4 lands, or 2 lands + a mana rock (rocks cover a land).
    Attempts 3+ (already bottoming) loosen toward keeping any 2-5 land hand.
    """
    L = hand[LAND]
    SR = hand[SOLRING] > 0
    rocks = hand[SIGNET] + (1 if SR else 0)

    if attempt >= 5:
        return True
    if attempt >= 3:                       # loosened (already bottoming)
        if L >= 6:
            return False
        if L == 0:
            return attempt >= 4 and rocks >= 1
        if L == 1:
            return rocks >= 1 or attempt >= 4
        return L <= 5                       # keep 2-5 lands
    # attempts 1-2: keep 3-4 lands, MULL 5+ (the robust DP win). A hard curve
    # requirement (even at the free attempt 1) over-mulligans and loses value, so
    # keep any 3-4 land hand; the DP's curve-sensitivity is too soft to encode.
    if 3 <= L <= 4:
        # fast decks (B4/5): need at least one non-rock play (a real spell), not
        # an all-lands-and-rocks hand. Rarely fires (fast decks run ~no rocks).
        if n_turns <= 6:
            nonrock = hand[1] + hand[2] + hand[3] + hand[4] + hand[5] + hand[6]
            return nonrock >= 1
        return True
    if L == 2:
        return rocks >= 1                  # 2 lands ok with a rock (helps slow decks)
    if L == 1:
        return SR                          # 1 land only behind a Sol Ring
    return False                           # 0 or 5+ lands: mull


@njit(cache=True)
def _hand_from(lib):
    h = np.zeros(9, dtype=np.int64)
    for i in range(7):
        h[lib[i]] += 1
    return h


@njit(cache=True)
def draw_opening_hand(lib, rng):
    attempt = 1
    while True:
        shuffle(lib, rng)
        hand = _hand_from(lib)
        if mulligan_keep(hand, attempt):
            n_bottom = 0 if attempt <= 2 else (attempt - 2)
            if n_bottom > 0:
                bottom_cards(hand, n_bottom)
            return hand, 7
        attempt += 1


@njit(cache=True)
def draw_opening_hand_adaptive(lib, rng, n_turns):
    attempt = 1
    while True:
        shuffle(lib, rng)
        hand = _hand_from(lib)
        if mulligan_keep_adaptive(hand, attempt, n_turns):
            n_bottom = 0 if attempt <= 2 else (attempt - 2)
            if n_bottom > 0:
                bottom_cards(hand, n_bottom)
            return hand, 7
        attempt += 1


# --------------------------------------------------------------------------- #
# Gameplay.                                                                    #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def mana_available(board):
    return board[LAND] + 2 * board[SOLRING] + board[SIGNET]


@njit(cache=True)
def _has_drop(hand, cstate, mv):
    if 1 <= mv <= 6 and hand[mv] > 0:
        return True
    if mv == cstate[0] and cstate[1] == 0:
        return True
    return False


@njit(cache=True)
def _cast_mv(hand, board, cstate, mv):
    if 1 <= mv <= 6 and hand[mv] > 0:
        hand[mv] -= 1
        board[mv] += 1
        return True
    if mv == cstate[0] and cstate[1] == 0:
        cstate[1] = 1
        return True
    return False


@njit(cache=True)
def play_turn(hand, board, cstate, turn, lib, draw_ptr, rng):
    # draw for the turn (always on the draw, every turn incl. T1)
    hand[lib[draw_ptr]] += 1
    draw_ptr += 1

    mana = mana_available(board)          # prior sources untap

    # 1. play a land
    if hand[LAND] > 0:
        hand[LAND] -= 1
        board[LAND] += 1
        mana += 1
    # 2. Sol Ring
    played_solring = False
    if hand[SOLRING] > 0 and mana >= 1:
        hand[SOLRING] -= 1
        board[SOLRING] += 1
        mana -= 1
        mana += 2
        played_solring = True
    # 3. turn-1 Sol Ring stop
    if turn == 1 and played_solring:
        return draw_ptr
    # 4. T1/T2 signet
    if turn <= 2 and hand[SIGNET] > 0 and mana >= 2:
        hand[SIGNET] -= 1
        board[SIGNET] += 1
        mana -= 2
        mana += 1

    # 5. turns 3 & 4: rock + (N-1)-drop (only if both are possible)
    if turn == 3 or turn == 4:
        N = mana
        if hand[SIGNET] > 0 and N >= 2 and (N - 1) >= 1 and _has_drop(hand, cstate, N - 1):
            hand[SIGNET] -= 1
            board[SIGNET] += 1
            mana -= 2
            mana += 1                     # mana now N-1
            _cast_mv(hand, board, cstate, N - 1)
            mana -= (N - 1)               # mana now 0

    # 6. gap-fill: no N-drop but 2-drop + distinct (N-2)-drop
    while mana >= 3 and not _has_drop(hand, cstate, mana):
        n2 = mana - 2
        if n2 == 2:
            ok = hand[2] >= 2
        else:
            ok = hand[2] >= 1 and _has_drop(hand, cstate, n2)
        if not ok:
            break
        _cast_mv(hand, board, cstate, 2)
        mana -= 2
        _cast_mv(hand, board, cstate, n2)
        mana -= n2

    # 7. greedy: highest-MV castable, down from 6
    progress = True
    while progress and mana >= 1:
        progress = False
        mv = 6
        while mv >= 1:
            if mv <= mana and _has_drop(hand, cstate, mv):
                _cast_mv(hand, board, cstate, mv)
                mana -= mv
                progress = True
                break
            mv -= 1

    # 8. retroactive rock if mana left over
    if mana >= 2 and hand[SIGNET] > 0:
        hand[SIGNET] -= 1
        board[SIGNET] += 1

    return draw_ptr


# --------------------------------------------------------------------------- #
# Whole-game + Monte Carlo.                                                    #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def simulate_game(deck_counts, commander_mv, seed, n_turns=7, adaptive=False):
    rng = new_rng(seed)
    lib = build_library(deck_counts)
    if adaptive:
        hand, ptr = draw_opening_hand_adaptive(lib, rng, n_turns)
    else:
        hand, ptr = draw_opening_hand(lib, rng)
    board = np.zeros(9, dtype=np.int64)
    cstate = np.array([commander_mv, 0], dtype=np.int64)
    total = 0.0
    for turn in range(1, n_turns + 1):
        ptr = play_turn(hand, board, cstate, turn, lib, ptr, rng)
        total += score_board(board, cstate[1], cstate[0])
    return total


@njit(cache=True)
def _deck9(deck_counts):
    """Convert deck [c1..c6, sig, land] (len 8) to a length-9 code vector."""
    d = np.zeros(9, dtype=np.int64)
    d[LAND] = deck_counts[7]
    for k in range(1, 7):
        d[k] = deck_counts[k - 1]
    d[SIGNET] = deck_counts[6]
    d[SOLRING] = 1
    return d


@njit(cache=True)
def build_library_codes(counts9):
    n = 0
    for i in range(9):
        n += counts9[i]
    lib = np.empty(n, dtype=np.int8)
    idx = 0
    for code in range(9):
        for _ in range(counts9[code]):
            lib[idx] = code
            idx += 1
    return lib


@njit(cache=True)
def simulate_from_hand(deck_counts, kept_hand, removed, commander_mv, seed, n_turns):
    """One game from a FIXED opening hand. Draw pile = deck minus the original 7
    (kept_hand + removed, the bottomed cards). Returns the criterion."""
    rng = new_rng(seed)
    remaining = _deck9(deck_counts) - kept_hand - removed
    lib = build_library_codes(remaining)
    shuffle(lib, rng)
    hand = kept_hand.copy()
    board = np.zeros(9, dtype=np.int64)
    cstate = np.array([commander_mv, 0], dtype=np.int64)
    total = 0.0
    ptr = 0
    for turn in range(1, n_turns + 1):
        ptr = play_turn(hand, board, cstate, turn, lib, ptr, rng)
        total += score_board(board, cstate[1], cstate[0])
    return total


@njit(cache=True)
def vkeep(deck_counts, kept_hand, removed, commander_mv, n_games, base_seed, n_turns):
    """Monte Carlo mean of simulate_from_hand over n_games -> V_keep(hand)."""
    mean = 0.0
    for i in range(n_games):
        s = _game_seed(base_seed, i)
        x = simulate_from_hand(deck_counts, kept_hand, removed, commander_mv, s, n_turns)
        mean += (x - mean) / (i + 1)
    return mean


@njit(cache=True)
def _game_seed(base_seed, i):
    z = np.uint64(base_seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(i) + np.uint64(1)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(31))
    return z


@njit(cache=True)
def simulate_deck(deck_counts, commander_mv, n_games, base_seed, n_turns=7, adaptive=False):
    mean = 0.0
    m2 = 0.0
    for i in range(n_games):
        s = _game_seed(base_seed, i)
        x = simulate_game(deck_counts, commander_mv, s, n_turns, adaptive)
        d = x - mean
        mean += d / (i + 1)
        m2 += d * (x - mean)
    var = m2 / (n_games - 1) if n_games > 1 else 0.0
    return mean, var
