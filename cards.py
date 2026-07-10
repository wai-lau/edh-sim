"""Card codes, a seedable Numba-friendly PRNG, and library construction.

Card codes (int8): 0=land, 1..6=k-drop, 7=signet, 8=Sol Ring, 9=draw.
Hands/boards are length-NCODE (10) count vectors indexed by card code. A deck is
[c1..c6, sig, land] (len 8) or [c1..c6, sig, land, draw] (len 9), summing to 98;
one implicit Sol Ring -> 99-card library.
"""
import numpy as np
from numba import int64, njit

LAND = 0
SIGNET = 7
SOLRING = 8
DRAW = 9        # "draw card": pay X mana, draw X cards; scores 0, resolves away
NCODE = 10      # hands/boards are length-NCODE count vectors (codes 0..9)


# --------------------------------------------------------------------------- #
# PRNG: seedable xorshift64 with splitmix64 seeding.                           #
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
# Library construction.                                                        #
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
    n_draw = deck_counts[8] if deck_counts.shape[0] > 8 else 0   # draw cards (opt.)
    for _ in range(n_draw):
        lib[idx] = DRAW
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


@njit(cache=True)
def deck9(deck_counts):
    """Convert a deck vector to a length-NCODE code-count vector (+ Sol Ring)."""
    d = np.zeros(NCODE, dtype=np.int64)
    d[LAND] = deck_counts[7]
    for k in range(1, 7):
        d[k] = deck_counts[k - 1]
    d[SIGNET] = deck_counts[6]
    d[SOLRING] = 1
    d[DRAW] = deck_counts[8] if deck_counts.shape[0] > 8 else 0
    return d


@njit(cache=True)
def build_library_codes(counts9):
    n = 0
    for i in range(NCODE):
        n += counts9[i]
    lib = np.empty(n, dtype=np.int8)
    idx = 0
    for code in range(NCODE):
        for _ in range(counts9[code]):
            lib[idx] = code
            idx += 1
    return lib
