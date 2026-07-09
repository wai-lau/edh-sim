# EDH Mana-Curve Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate Karsten's Commander mana-curve experiment — a Monte Carlo
simulator of expected compounded mana over 7 turns plus a local-search optimizer
reproducing his 5-row optimal-deck table (commander MV 2–6).

**Architecture:** A Numba `@njit` hot path (`sim_core.py`) does all per-game work
on fixed-size int arrays (build library, seeded shuffle, mulligan, bottoming,
7-turn play, scoring). A Python orchestration layer (`optimizer.py`) runs local
search with common-random-numbers variance reduction. A CLI (`main.py`) produces
the table and validates golden anchors.

**Tech Stack:** Python 3.11+, Numba, NumPy, pytest, uv.

## Global Constraints

- Deck = int array length 8 `[c1,c2,c3,c4,c5,c6, signets, lands]`, `sum == 98`;
  one Sol Ring is implicit → 99-card library.
- Card codes (int8): `0=land, 1..6=k-drop, 7=signet, 8=Sol Ring`.
- Hand/board = int array length 9 indexed by card code.
- Criterion worth: k-drop = k (k∈1..5), six-drop = **6.2**, commander = raw MV,
  land/signet/Sol Ring = 0. Summed at each turn end, turns 1..7.
- Always on the draw: draw one card every turn including turn 1.
- Free first mulligan; London bottoming; mulligan floor = to four.
- Mulligan keep-test "lands" = actual lands only; Sol Ring counts as a land only
  during bottoming.
- Commander: free MV-N spell, castable any turn, cast once, never recast.
- Six-drop worth 6.2; commander (even MV 6) worth raw MV.
- Goldfish: everything untaps every turn; no opponent interaction.
- All Monte Carlo uses a seedable xorshift64 PRNG; per-game seeds derived from a
  deck's base seed so decks can share seeds (CRN).

---

### Task 1: Scaffold + PRNG

**Files:**
- Create: `pyproject.toml`, `sim_core.py`, `tests/test_prng.py`

**Interfaces:**
- Produces: `LAND, SIGNET, SOLRING` code constants; `new_rng(seed)->uint64[1]`;
  `next_u64(rng)->uint64` (mutates `rng[0]`); `rand_below(rng, n)->int64`.

- [ ] **Step 1: `pyproject.toml`**

```toml
[project]
name = "edh-sim"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numba>=0.59", "numpy>=1.26"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
markers = ["slow: Monte Carlo / optimizer tests (deselect with -m 'not slow')"]
```

- [ ] **Step 2: Write failing test** `tests/test_prng.py`

```python
import numpy as np
from sim_core import new_rng, next_u64, rand_below

def test_prng_deterministic():
    a = new_rng(42); b = new_rng(42)
    assert next_u64(a) == next_u64(b)

def test_prng_differs_by_seed():
    assert next_u64(new_rng(1)) != next_u64(new_rng(2))

def test_rand_below_in_range():
    rng = new_rng(7)
    for _ in range(1000):
        v = rand_below(rng, 10)
        assert 0 <= v < 10
```

- [ ] **Step 3: Run test, verify fail** — `uv run pytest tests/test_prng.py -q` → ImportError.

- [ ] **Step 4: Implement** in `sim_core.py`

```python
import numpy as np
from numba import njit, int8, int64

LAND = 0
SIGNET = 7
SOLRING = 8

@njit(cache=True)
def new_rng(seed):
    s = np.empty(1, dtype=np.uint64)
    # splitmix64 to avoid zero state / decorrelate seeds
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
```

- [ ] **Step 5: Run test, verify pass** — `uv run pytest tests/test_prng.py -q` → 3 passed.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: seedable xorshift64 PRNG core"`

---

### Task 2: build_library + shuffle

**Files:** Modify `sim_core.py`; Create `tests/test_library.py`

**Interfaces:**
- Consumes: PRNG from Task 1.
- Produces: `build_library(deck_counts:int64[8]) -> int8[99]`;
  `shuffle(lib, rng)` (in-place Fisher-Yates).

- [ ] **Step 1: Failing test** `tests/test_library.py`

```python
import numpy as np
from sim_core import build_library, shuffle, new_rng

DECK = np.array([9,0,20,14,9,4,0,42], dtype=np.int64)  # 2-mv row, sums 98

def test_library_composition():
    lib = build_library(DECK)
    assert lib.shape[0] == 99
    counts = np.bincount(lib, minlength=9)
    assert counts[0] == 42          # lands
    assert list(counts[1:7]) == [9,0,20,14,9,4]
    assert counts[7] == 0           # signets
    assert counts[8] == 1           # exactly one Sol Ring

def test_shuffle_is_permutation():
    lib = build_library(DECK); ref = lib.copy()
    shuffle(lib, new_rng(123))
    assert sorted(lib.tolist()) == sorted(ref.tolist())

def test_shuffle_deterministic():
    a = build_library(DECK); b = build_library(DECK)
    shuffle(a, new_rng(5)); shuffle(b, new_rng(5))
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
@njit(cache=True)
def build_library(deck_counts):
    lib = np.empty(99, dtype=np.int8)
    idx = 0
    for k in range(1, 7):          # k-drops -> codes 1..6
        for _ in range(deck_counts[k-1]):
            lib[idx] = k; idx += 1
    for _ in range(deck_counts[6]):  # signets
        lib[idx] = SIGNET; idx += 1
    for _ in range(deck_counts[7]):  # lands
        lib[idx] = LAND; idx += 1
    lib[idx] = SOLRING; idx += 1     # the one Sol Ring
    return lib

@njit(cache=True)
def shuffle(lib, rng):
    n = lib.shape[0]
    for i in range(n - 1, 0, -1):
        j = rand_below(rng, i + 1)
        tmp = lib[i]; lib[i] = lib[j]; lib[j] = tmp
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: build_library + seeded Fisher-Yates shuffle"`

---

### Task 3: score_board (criterion) — golden §5.1

**Files:** Modify `sim_core.py`; Create `tests/test_scoring.py`

**Interfaces:**
- Produces: `score_board(board:int64[9], commander_on:int64, commander_mv:int64) -> float64`.

- [ ] **Step 1: Failing test** `tests/test_scoring.py`

```python
import numpy as np
from sim_core import score_board

def board(**kw):
    b = np.zeros(9, dtype=np.int64)
    for code, n in kw.items():
        b[int(code)] = n
    return b

def test_worked_example_sums_to_23():
    # T1,T2 empty; T3,T4 one-drop; T5-T7 one-drop + MV6 commander
    cmv = 6
    per_turn = [
        score_board(board(**{1:0}), 0, cmv),          # T1: 0
        score_board(board(**{1:0}), 0, cmv),          # T2: 0
        score_board(board(**{1:1}), 0, cmv),          # T3: 1
        score_board(board(**{1:1}), 0, cmv),          # T4: 1
        score_board(board(**{1:1}), 1, cmv),          # T5: 1+6
        score_board(board(**{1:1}), 1, cmv),          # T6: 7
        score_board(board(**{1:1}), 1, cmv),          # T7: 7
    ]
    assert sum(per_turn) == 23

def test_six_drop_worth_6point2():
    assert score_board(board(**{6:1}), 0, 4) == 6.2

def test_rocks_and_lands_excluded():
    b = board(**{0:5, 7:3, 8:1})  # lands, signets, sol ring only
    assert score_board(b, 0, 4) == 0.0

def test_commander_raw_mv_not_6point2():
    # MV6 commander scores 6.0, not 6.2
    assert score_board(board(), 1, 6) == 6.0
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
@njit(cache=True)
def score_board(board, commander_on, commander_mv):
    total = 0.0
    for k in range(1, 6):          # 1..5 drops worth k
        total += board[k] * k
    total += board[6] * 6.2        # six-drops worth 6.2
    if commander_on:
        total += commander_mv      # commander worth raw MV
    return total
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: score_board criterion (golden example -> 23)"`

---

### Task 4: mulligan_keep — §3.2

**Files:** Modify `sim_core.py`; Create `tests/test_mulligan.py`

**Interfaces:**
- Produces: `mulligan_keep(hand:int64[9], attempt:int64) -> bool`. `attempt` is
  1-based hand number (1..5). Uses `L=hand[LAND]`, `S=hand[SIGNET]`,
  `SR=hand[SOLRING]>0`.

- [ ] **Step 1: Failing test** `tests/test_mulligan.py`

```python
import numpy as np
from sim_core import mulligan_keep

def hand(land=0, sig=0, sr=0, **drops):
    h = np.zeros(9, dtype=np.int64)
    h[0]=land; h[7]=sig; h[8]=sr
    for code,n in drops.items(): h[int(code)]=n
    return h

def test_hand1_keeps_3_5_lands():
    assert mulligan_keep(hand(land=4, **{2:3}), 1)
    assert not mulligan_keep(hand(land=2, **{2:5}), 1)  # 2 lands, no SR
    assert not mulligan_keep(hand(land=6, **{2:1}), 1)  # 6 lands

def test_hand1_land_signet_cap():
    # 3 lands + 3 signets = 6 combined > 5 -> mull (unless SR branch)
    assert not mulligan_keep(hand(land=3, sig=3, **{2:1}), 1)

def test_hand1_solring_branch():
    assert mulligan_keep(hand(land=1, sr=1, **{2:5}), 1)   # SR + 1 land
    assert not mulligan_keep(hand(land=0, sr=1, **{2:6}), 1)  # SR + 0 land

def test_hand2_keeps_two_land():
    assert not mulligan_keep(hand(land=2, **{2:5}), 1)
    assert mulligan_keep(hand(land=2, **{2:5}), 2)         # 2-land ok on hand 2

def test_hand3_and_4_thresholds():
    for a in (3,4):
        assert mulligan_keep(hand(land=2, **{2:5}), a)
        assert mulligan_keep(hand(land=4, **{2:3}), a)
        assert not mulligan_keep(hand(land=5, **{2:2}), a)  # 5 lands -> mull
        assert not mulligan_keep(hand(land=1, **{2:6}), a)  # 1 land no SR
        assert mulligan_keep(hand(land=1, sr=1, **{2:5}), a)  # 1 land + SR

def test_hand5_always_keep():
    assert mulligan_keep(hand(land=0, **{6:7}), 5)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
@njit(cache=True)
def mulligan_keep(hand, attempt):
    L = hand[LAND]; S = hand[SIGNET]; SR = hand[SOLRING] > 0
    if attempt >= 5:
        return True                              # to four: always keep
    if attempt <= 2:
        base = (3 <= L <= 5) and (L + S <= 5)
        base = base or (SR and (1 <= L <= 5))
        if attempt == 2:
            base = base or (L == 2)
        return base
    # attempt 3 or 4 (-> six / -> five)
    return (2 <= L <= 4) or (L == 1 and SR)
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: mulligan keep policy"`

---

### Task 5: bottom_cards — §3.3

**Files:** Modify `sim_core.py`; add tests to `tests/test_mulligan.py`

**Interfaces:**
- Produces: `bottom_cards(hand:int64[9], n_bottom:int64)` — mutates `hand`,
  removing `n_bottom` cards: superfluous signets (keep ≤1) → lands down toward 3
  (Sol Ring counts as a land, never bottomed) → spells most-expensive-first.

- [ ] **Step 1: Failing tests** (append to `tests/test_mulligan.py`)

```python
from sim_core import bottom_cards

def total(h): return int(h.sum())

def test_bottom_superfluous_signets_first():
    h = hand(land=3, sig=3, **{2:1})  # 8 cards; bottom 2
    bottom_cards(h, 2)
    assert h[7] == 1                  # kept exactly one signet
    assert h[0] == 3 and h[2] == 1    # lands/spells untouched

def test_bottom_lands_toward_three():
    h = hand(land=6, **{2:1})         # bottom 3 -> want 3 lands left
    bottom_cards(h, 3)
    assert h[0] == 3

def test_solring_counts_as_land_never_bottomed():
    h = hand(land=3, sr=1, **{2:2})   # effective 4 lands; bottom 1
    bottom_cards(h, 1)
    assert h[8] == 1                  # sol ring stays
    assert h[0] == 2                  # one real land bottomed (3+SR ->3 effective)

def test_bottom_spells_most_expensive_first():
    h = hand(land=3, **{2:1, 5:1, 6:1})  # 3 lands already; bottom 2 spells
    bottom_cards(h, 2)
    assert h[6] == 0 and h[5] == 0 and h[2] == 1  # 6 then 5 bottomed
    assert h[0] == 3
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
@njit(cache=True)
def bottom_cards(hand, n_bottom):
    left = n_bottom
    # 1. superfluous signets (keep at most one)
    extra_sig = hand[SIGNET] - 1
    if extra_sig > 0:
        take = extra_sig if extra_sig < left else left
        hand[SIGNET] -= take; left -= take
    # 2. lands toward 3 (Sol Ring counts as a land, never bottomed)
    if left > 0:
        eff_lands = hand[LAND] + hand[SOLRING]
        removable = eff_lands - 3
        if removable > 0:
            take = removable if removable < left else left
            take = take if take < hand[LAND] else hand[LAND]
            hand[LAND] -= take; left -= take
    # 3. spells, most expensive first (6 -> 1)
    k = 6
    while left > 0 and k >= 1:
        if hand[k] > 0:
            take = hand[k] if hand[k] < left else left
            hand[k] -= take; left -= take
        else:
            k -= 1
    # 4. last resort: the kept signet
    if left > 0 and hand[SIGNET] > 0:
        take = hand[SIGNET] if hand[SIGNET] < left else left
        hand[SIGNET] -= take; left -= take
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: London bottoming policy"`

---

### Task 6: draw_opening_hand (mulligan loop)

**Files:** Modify `sim_core.py`; Create `tests/test_opening.py`

**Interfaces:**
- Produces: `draw_opening_hand(lib:int8[99], rng) -> (hand:int64[9], draw_ptr:int64)`.
  Reshuffles `lib` each mulligan attempt; on keep, applies `bottom_cards`; the
  draw pile is `lib[7:]` from that final shuffle; `draw_ptr` starts at 7.

- [ ] **Step 1: Failing test** `tests/test_opening.py`

```python
import numpy as np
from sim_core import build_library, draw_opening_hand, new_rng

DECK = np.array([6,12,13,0,13,8,7,39], dtype=np.int64)  # 4-mv row, sums 98

def test_opening_hand_size_and_ptr():
    lib = build_library(DECK)
    hand, ptr = draw_opening_hand(lib, new_rng(99))
    assert 4 <= int(hand.sum()) <= 7    # 7 minus 0..3 bottomed
    assert ptr == 7

def test_opening_deterministic():
    h1,_ = draw_opening_hand(build_library(DECK), new_rng(3))
    h2,_ = draw_opening_hand(build_library(DECK), new_rng(3))
    assert np.array_equal(h1, h2)

def test_kept_hand_passes_policy_or_is_floor():
    # over many seeds, every kept hand has >=? just assert never crashes & size valid
    for s in range(200):
        h,_ = draw_opening_hand(build_library(DECK), new_rng(s))
        assert int(h.sum()) >= 4
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
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
```

Note: `attempt` bottom count — attempts 1,2 → 0; 3 → 1; 4 → 2; 5 → 3 (and
attempt 5 always keeps via `mulligan_keep`). Bottomed cards leave `hand` but
remain physically in `lib[0:7]`; the draw pile is `lib[7:]`, so bottomed cards
are never redrawn within 7 turns. Correct for the criterion.

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: mulligan loop / opening hand"`

---

### Task 7: play_turn — gameplay logic §3.4

**Files:** Modify `sim_core.py`; Create `tests/test_gameplay.py`

**Interfaces:**
- Produces: helper `cast_value(hand, board, cstate)` and
  `play_turn(hand, board, cstate, turn, lib, draw_ptr, rng) -> draw_ptr`.
  `cstate = [commander_mv, commander_cast_flag]` (int64[2]). `play_turn` first
  draws `lib[draw_ptr]` into `hand` (turn's draw), then executes steps 1–8,
  mutating `hand`/`board`/`cstate`. Returns the advanced `draw_ptr`.
- Mana each turn: `board[LAND] + 2*board[SOLRING] + board[SIGNET]`.

- [ ] **Step 1: Failing test** `tests/test_gameplay.py`

```python
import numpy as np
from sim_core import play_turn, mana_of

def blank(): return np.zeros(9, dtype=np.int64)
def cst(mv, cast=0): return np.array([mv, cast], dtype=np.int64)
# a library long enough; draws that don't matter set to LAND(0)
def lib_of(*cards, pad=20):
    a = list(cards) + [0]*pad
    return np.array(a, dtype=np.int8)

def test_turn1_solring_stops_turn():
    hand = blank(); hand[0]=1; hand[8]=1; hand[2]=1   # land, sol ring, 2-drop
    board = blank(); cs = cst(4)
    from sim_core import new_rng
    # draw a spare land so drawing doesn't add relevant cards
    ptr = play_turn(hand, board, cs, 1, lib_of(0), 0, new_rng(1))
    assert board[0]==1 and board[8]==1   # land + sol ring down
    assert hand[2]==1                    # 2-drop NOT cast (turn-1 stop)

def test_turn3_rock_plus_nminus1():
    # 3 lands in play (N=3 after land drop from hand); hand has signet + 2-drop
    hand = blank(); hand[0]=1; hand[7]=1; hand[2]=1
    board = blank(); board[0]=2                        # 2 lands already
    cs = cst(5)
    from sim_core import new_rng
    play_turn(hand, board, cs, 3, lib_of(0), 0, new_rng(1))
    assert board[7]==1 and board[2]==1                 # signet + 2-drop (N-1)

def test_gapfill_two_plus_nminus2():
    # turn 5, N=5, no 5-drop, hand has 2-drop + 3-drop -> cast both
    hand = blank(); hand[2]=1; hand[3]=1
    board = blank(); board[0]=5
    cs = cst(4, cast=1)                                 # commander already cast
    from sim_core import new_rng
    play_turn(hand, board, cs, 5, lib_of(0), 0, new_rng(1))
    assert board[2]==1 and board[3]==1

def test_greedy_highest_first():
    hand = blank(); hand[6]=1; hand[2]=1
    board = blank(); board[0]=6
    cs = cst(3, cast=1)
    from sim_core import new_rng
    play_turn(hand, board, cs, 6, lib_of(0), 0, new_rng(1))
    assert board[6]==1                                 # six-drop cast

def test_commander_cast_on_curve():
    # turn 4, N=4, commander MV4 uncast, no 4-drop in hand -> cast commander
    hand = blank()
    board = blank(); board[0]=4
    cs = cst(4)
    from sim_core import new_rng
    play_turn(hand, board, cs, 4, lib_of(0), 0, new_rng(1))
    assert cs[1]==1                                    # commander cast
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
@njit(cache=True)
def mana_of(board):
    return board[LAND] + 2*board[SOLRING] + board[SIGNET]

@njit(cache=True)
def _has_drop(hand, cstate, mv):
    # True if a castable spell of value mv is available (real card or commander)
    if 1 <= mv <= 6 and hand[mv] > 0:
        return True
    if mv == cstate[0] and cstate[1] == 0:
        return True
    return False

@njit(cache=True)
def _cast_mv(hand, board, cstate, mv):
    # cast one spell of value mv, prefer a real card, else commander. no-op if none.
    if 1 <= mv <= 6 and hand[mv] > 0:
        hand[mv] -= 1; board[mv] += 1; return True
    if mv == cstate[0] and cstate[1] == 0:
        cstate[1] = 1; return True
    return False

@njit(cache=True)
def play_turn(hand, board, cstate, turn, lib, draw_ptr, rng):
    # draw for the turn (always on the draw, every turn incl. T1)
    hand[lib[draw_ptr]] += 1
    draw_ptr += 1

    # 1. play a land
    if hand[LAND] > 0:
        hand[LAND] -= 1; board[LAND] += 1
    # 2. Sol Ring
    played_solring = False
    if hand[SOLRING] > 0:
        hand[SOLRING] -= 1; board[SOLRING] += 1; played_solring = True
    # 3. turn-1 Sol Ring stop
    if turn == 1 and played_solring:
        return draw_ptr
    # 4. T1/T2 signet
    if turn <= 2 and hand[SIGNET] > 0 and mana_available(board) >= 2:
        hand[SIGNET] -= 1; board[SIGNET] += 1

    # remaining mana this turn = untapped sources minus what a signet/solring cast
    # cost. We recompute spendable mana directly:
    spent = _turn_spent(board)  # mana already committed casting rocks this turn
    N = mana_of(board) - spent

    # 5. turns 3 & 4: rock + (N-1)-drop
    if turn == 3 or turn == 4:
        if hand[SIGNET] > 0 and N >= 2:
            avail_after = N - 2 + 1              # signet pays 2, taps for 1
            if _has_drop(hand, cstate, avail_after) and avail_after >= 1:
                hand[SIGNET] -= 1; board[SIGNET] += 1
                _cast_mv(hand, board, cstate, avail_after)
                N = mana_of(board) - _turn_spent(board)

    # 6. gap-fill: no N-drop but 2-drop + distinct (N-2)-drop
    while N >= 3:
        if not _has_drop(hand, cstate, N):
            n2 = N - 2
            ok = False
            if n2 == 2:
                if hand[2] >= 2: ok = True
            else:
                if hand[2] >= 1 and _has_drop(hand, cstate, n2): ok = True
            if ok:
                _cast_mv(hand, board, cstate, 2)
                _cast_mv(hand, board, cstate, n2)
                N = mana_of(board) - _turn_spent(board)
                continue
        break

    # 7. greedy: highest-MV castable, down from 6
    progress = True
    while progress and N >= 1:
        progress = False
        mv = 6
        while mv >= 1:
            if mv <= N and _has_drop(hand, cstate, mv):
                _cast_mv(hand, board, cstate, mv)
                N = mana_of(board) - _turn_spent(board)
                progress = True
                break
            mv -= 1

    # 8. retroactive rock if mana left
    if N >= 2 and hand[SIGNET] > 0:
        hand[SIGNET] -= 1; board[SIGNET] += 1
    return draw_ptr
```

Helper for mana bookkeeping — casting a rock this turn consumes mana we must not
double-spend. Model spent-this-turn via a per-turn scratch. Simplest correct
approach: track `spent` locally instead of a board scan.

Replace the `spent`/`_turn_spent` scheme with an explicit local counter passed
through the steps. Final implementation uses a local `spent` int initialized to
account for the T1/T2 signet and Sol Ring casts, incremented on each rock cast:

```python
# canonical version — track `spent` locally, no _turn_spent helper
@njit(cache=True)
def mana_available(board):        # total untapped sources
    return board[LAND] + 2*board[SOLRING] + board[SIGNET]

@njit(cache=True)
def play_turn(hand, board, cstate, turn, lib, draw_ptr, rng):
    hand[lib[draw_ptr]] += 1; draw_ptr += 1
    spent = 0
    if hand[LAND] > 0:
        hand[LAND] -= 1; board[LAND] += 1
    played_solring = False
    if hand[SOLRING] > 0:
        hand[SOLRING] -= 1; board[SOLRING] += 1; spent += 1; played_solring = True
    if turn == 1 and played_solring:
        return draw_ptr
    if turn <= 2 and hand[SIGNET] > 0 and (mana_available(board) - spent) >= 2:
        hand[SIGNET] -= 1; board[SIGNET] += 1; spent += 2
    N = mana_available(board) - spent
    if turn == 3 or turn == 4:
        if hand[SIGNET] > 0 and N >= 2:
            avail_after = (N - 2) + 1
            if avail_after >= 1 and _has_drop(hand, cstate, avail_after):
                hand[SIGNET] -= 1; board[SIGNET] += 1; spent += 2  # +1 from tap:
                spent -= 1                                          # net signet cost 1 this turn
                N = mana_available(board) - spent
                _cast_mv(hand, board, cstate, avail_after); spent += avail_after
                N = mana_available(board) - spent
    while N >= 3 and not _has_drop(hand, cstate, N):
        n2 = N - 2; ok = False
        if n2 == 2:
            ok = hand[2] >= 2
        else:
            ok = hand[2] >= 1 and _has_drop(hand, cstate, n2)
        if not ok: break
        _cast_mv(hand, board, cstate, 2); spent += 2
        _cast_mv(hand, board, cstate, n2); spent += n2
        N = mana_available(board) - spent
    progress = True
    while progress and N >= 1:
        progress = False
        mv = 6
        while mv >= 1:
            if mv <= N and _has_drop(hand, cstate, mv):
                _cast_mv(hand, board, cstate, mv); spent += mv
                N = mana_available(board) - spent; progress = True; break
            mv -= 1
    if N >= 2 and hand[SIGNET] > 0:
        hand[SIGNET] -= 1; board[SIGNET] += 1
    return draw_ptr
```

Use only the canonical version above; delete the first draft. `_has_drop` and
`_cast_mv` are as defined earlier. `mana_available` replaces `mana_of` (tests
importing `mana_of` should import `mana_available`; update the test accordingly).

- [ ] **Step 4: Run, verify pass** — fix any scenario mismatches against §3.4.
- [ ] **Step 5: Commit** — `git commit -am "feat: per-turn gameplay logic"`

---

### Task 8: simulate_game (full game)

**Files:** Modify `sim_core.py`; Create `tests/test_game.py`

**Interfaces:**
- Produces: `simulate_game(deck_counts:int64[8], commander_mv:int64, seed) -> float64`.
  Builds library, draws opening hand, plays turns 1..7, sums `score_board` each
  turn end. One shared `rng` seeded from `seed`.

- [ ] **Step 1: Failing test** `tests/test_game.py`

```python
import numpy as np
from sim_core import simulate_game

DECK4 = np.array([6,12,13,0,13,8,7,39], dtype=np.int64)

def test_game_runs_and_is_positive():
    v = simulate_game(DECK4, 4, 12345)
    assert v > 0

def test_game_deterministic():
    assert simulate_game(DECK4, 4, 7) == simulate_game(DECK4, 4, 7)

def test_seed_changes_outcome():
    vals = {simulate_game(DECK4, 4, s) for s in range(50)}
    assert len(vals) > 1
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
@njit(cache=True)
def simulate_game(deck_counts, commander_mv, seed):
    rng = new_rng(seed)
    lib = build_library(deck_counts)
    hand, ptr = draw_opening_hand(lib, rng)
    board = np.zeros(9, dtype=np.int64)
    cstate = np.array([commander_mv, 0], dtype=np.int64)
    total = 0.0
    for turn in range(1, 8):
        ptr = play_turn(hand, board, cstate, turn, lib, ptr, rng)
        total += score_board(board, cstate[1], cstate[0])
    return total
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: full 7-turn game simulation"`

---

### Task 9: simulate_deck (Monte Carlo + CRN seeds)

**Files:** Modify `sim_core.py`; Create `tests/test_montecarlo.py`

**Interfaces:**
- Produces: `simulate_deck(deck_counts, commander_mv, n_games, base_seed) -> (mean, var)`.
  Per-game seed = `splitmix` of `(base_seed, i)` so two decks called with the
  same `base_seed`/`n_games` use identical game seeds (CRN). Welford variance.

- [ ] **Step 1: Failing test** `tests/test_montecarlo.py`

```python
import numpy as np
from sim_core import simulate_deck

DECK4 = np.array([6,12,13,0,13,8,7,39], dtype=np.int64)

def test_mean_deterministic():
    m1,_ = simulate_deck(DECK4, 4, 5000, 1)
    m2,_ = simulate_deck(DECK4, 4, 5000, 1)
    assert m1 == m2

def test_crn_same_seed_same_games():
    # a deck vs itself with same base_seed => identical means (obviously),
    # and variance >= 0
    m,v = simulate_deck(DECK4, 4, 2000, 9)
    assert v >= 0 and m > 0

def test_reasonable_range():
    m,_ = simulate_deck(DECK4, 4, 20000, 2)
    assert 60 < m < 90     # ballpark around his ~72.5
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
@njit(cache=True)
def _game_seed(base_seed, i):
    z = np.uint64(base_seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(i) + np.uint64(1)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(31))
    return z

@njit(cache=True)
def simulate_deck(deck_counts, commander_mv, n_games, base_seed):
    mean = 0.0; m2 = 0.0
    for i in range(n_games):
        s = _game_seed(base_seed, i)
        x = simulate_game(deck_counts, commander_mv, s)
        d = x - mean
        mean += d / (i + 1)
        m2 += d * (x - mean)
    var = m2 / (n_games - 1) if n_games > 1 else 0.0
    return mean, var
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: Monte Carlo simulate_deck with CRN seeds"`

---

### Task 10: Regression checkpoints — §5.4 (slow)

**Files:** Create `tests/test_regression.py`

**Interfaces:** Consumes `simulate_deck`. Uses high K so SE resolves the 0.031
checkpoint gap. These are the hard anchors from the article.

- [ ] **Step 1: Write test** `tests/test_regression.py`

```python
import numpy as np, math, pytest
from sim_core import simulate_deck

OPT4 = np.array([6,12,13,0,13,8,7,39], dtype=np.int64)     # 72.465
PERT = np.array([5,11,12,0,14,9,8,39], dtype=np.int64)     # -1/-1/-1 +5+6+sig -> 72.434
#   ^ from OPT4: c1 6->5, c2 12->11, c3 13->12, c5 13->14, c6 8->9, signet 7->8, land 39

@pytest.mark.slow
def test_checkpoint_absolute():
    K = 2_000_000
    m, v = simulate_deck(OPT4, 4, K, 20240828)
    se = math.sqrt(v / K)
    assert abs(m - 72.465) < 5*se + 0.05

@pytest.mark.slow
def test_checkpoint_ordering():
    K = 4_000_000
    m_opt,_ = simulate_deck(OPT4, 4, K, 111)
    m_pert,_ = simulate_deck(PERT, 4, K, 111)   # CRN: same base seed
    assert m_opt > m_pert                        # 72.465 > 72.434
    assert abs((m_opt - m_pert) - (72.465-72.434)) < 0.02
```

Note: verify `PERT` sums to 98 (`5+11+12+0+14+9=51, +8=59, +39=98`). ✓

- [ ] **Step 2: Run** — `uv run pytest tests/test_regression.py -m slow -q`. Expect
  PASS. If the absolute value is off, first re-check the gameplay logic against
  §3.4 (this test is the whole point — do not loosen tolerance to force a pass;
  fix the model). Document the measured mean + SE in the commit message.

- [ ] **Step 3: Commit** — `git commit -am "test: Karsten criterion checkpoints (72.465 / 72.434)"`

---

### Task 11: Neighborhoods (cross + star)

**Files:** Create `optimizer.py`, `tests/test_neighbors.py`

**Interfaces:**
- Produces: `neighbors_cross(deck)->list[np.ndarray]`, `neighbors_star(deck)->list[np.ndarray]`.
  Each neighbor is a length-8 int64 array, `sum==98`, all `>=0`. Pure Python
  (runs rarely; hot path stays in `sim_core`).

- [ ] **Step 1: Failing test** `tests/test_neighbors.py`

```python
import numpy as np
from optimizer import neighbors_cross, neighbors_star

DECK = np.array([6,12,13,0,13,8,7,39], dtype=np.int64)

def _valid(ds):
    for d in ds:
        assert d.sum() == 98 and (d >= 0).all() and d.shape[0] == 8

def test_cross_are_single_swaps():
    ns = neighbors_cross(DECK); _valid(ns)
    for d in ns:
        diff = d - DECK
        assert np.abs(diff).sum() == 2 and diff.max() == 1 and diff.min() == -1

def test_star_within_one_each_axis():
    ns = neighbors_star(DECK); _valid(ns)
    for d in ns:
        assert np.abs(d - DECK).max() <= 1
    assert len(ns) > 100   # ~1107 balanced +-1 vectors (minus negatives)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** `optimizer.py`

```python
import itertools
import numpy as np

def neighbors_cross(deck):
    out = []
    for a in range(8):
        if deck[a] == 0:
            continue
        for b in range(8):
            if a == b:
                continue
            d = deck.copy(); d[a] -= 1; d[b] += 1
            out.append(d)
    return out

def neighbors_star(deck):
    out = []
    for delta in itertools.product((-1, 0, 1), repeat=8):
        dv = np.array(delta, dtype=np.int64)
        if dv.sum() != 0:
            continue
        if np.abs(dv).sum() == 0:
            continue
        cand = deck + dv
        if (cand >= 0).all():
            out.append(cand)
    return out
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: cross + star neighborhoods"`

---

### Task 12: local_search (one restart)

**Files:** Modify `optimizer.py`; Create `tests/test_search.py`

**Interfaces:**
- Produces: `local_search(commander_mv, start_deck, base_seed, max_sims=200_000,
  sim_step=1000, sim_start=10_000, switch_star=150_000) -> (best_deck, best_mean)`.
  Escalating sim budget; per-deck sim cache pools prior sims; CRN within each
  iteration (all neighbors + current best share that iteration's base seed);
  fresh seed per iteration; cross neighborhood until best has ≥`switch_star`
  accumulated sims, then star; stop when best has >`max_sims` sims and is a
  local optimum.

- [ ] **Step 1: Failing test** `tests/test_search.py`

```python
import numpy as np, pytest
from optimizer import local_search

@pytest.mark.slow
def test_search_improves_and_stops():
    start = np.array([10,10,10,10,10,10,0,38], dtype=np.int64)  # sums 98
    best, mean = local_search(4, start, base_seed=1,
                              max_sims=40_000, sim_start=4000, sim_step=1000,
                              switch_star=20_000)
    assert best.sum() == 98 and (best >= 0).all()
    from sim_core import simulate_deck
    m_start,_ = simulate_deck(start, 4, 60_000, 999)
    m_best,_  = simulate_deck(best, 4, 60_000, 999)
    assert m_best >= m_start - 0.1     # never ends worse than start
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** (append to `optimizer.py`)

```python
from sim_core import simulate_deck

def _key(deck):
    return tuple(int(x) for x in deck)

def local_search(commander_mv, start_deck, base_seed,
                 max_sims=200_000, sim_step=1000, sim_start=10_000,
                 switch_star=150_000):
    # cache: key -> (n_games, mean)  (pooled across iterations)
    cache = {}

    def evaluate(deck, n_games, seed):
        k = _key(deck)
        m, _ = simulate_deck(np.asarray(deck, dtype=np.int64),
                             commander_mv, n_games, seed)
        prev = cache.get(k)
        if prev is None:
            cache[k] = (n_games, m)
        else:
            pn, pm = prev
            tot = pn + n_games
            cache[k] = (tot, (pm * pn + m * n_games) / tot)
        return cache[k][1]

    best = np.asarray(start_deck, dtype=np.int64).copy()
    it = 0
    sims = sim_start
    seed = base_seed
    best_mean = evaluate(best, sims, seed)

    while True:
        it += 1
        seed = base_seed + it * 7919          # fresh seed batch each iteration
        best_sims = cache[_key(best)][0]
        if best_sims >= switch_star:
            hood = neighbors_star(best)
        else:
            hood = neighbors_cross(best)
        # CRN: every candidate + current best evaluated on the same seed this iter
        cur = evaluate(best, sims, seed)
        best_cand = best; best_cand_mean = cur
        for cand in hood:
            m = evaluate(cand, sims, seed)
            if m > best_cand_mean:
                best_cand_mean = m; best_cand = cand
        moved = not np.array_equal(best_cand, best)
        if moved:
            best = best_cand.copy(); best_mean = best_cand_mean
        sims += sim_step
        if not moved and cache[_key(best)][0] > max_sims:
            break
        if sims > max_sims * 3:               # safety cap
            break
    return best, best_mean
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: local search with CRN + escalating sims"`

---

### Task 13: optimize_commander (multi-restart) + integration §5.5 (slow)

**Files:** Modify `optimizer.py`; Create `tests/test_integration.py`

**Interfaces:**
- Produces: `optimize_commander(commander_mv, restarts=3, master_seed=0, **kw) ->
  (best_deck, best_mean)`. Runs `local_search` from several start decks / seeds,
  returns the highest-criterion result. `START_DECKS[mv]` provides seed decks.

- [ ] **Step 1: Implement start decks + wrapper** (append to `optimizer.py`)

```python
# Reasonable start decks per commander MV (sum 98). Scaled from his 60-card optima.
START_DECKS = {
    2: np.array([8, 6, 10, 8, 6, 3, 0, 57-16], dtype=np.int64),  # placeholder-safe
    3: np.array([6, 12, 8, 10, 6, 3, 4, 49], dtype=np.int64),
    4: np.array([6, 12, 10, 8, 8, 5, 7, 42], dtype=np.int64),
    5: np.array([6, 12, 10, 10, 6, 6, 8, 40], dtype=np.int64),
    6: np.array([6, 12, 10, 10, 8, 4, 9, 39], dtype=np.int64),
}

def _fix_sum(deck):
    deck = deck.copy()
    while deck.sum() > 98:
        deck[np.argmax(deck)] -= 1
    while deck.sum() < 98:
        deck[7] += 1
    return deck

def optimize_commander(commander_mv, restarts=3, master_seed=0, **kw):
    base = _fix_sum(START_DECKS[commander_mv])
    results = []
    for r in range(restarts):
        # jitter the start a little per restart for diversity
        start = base.copy()
        if r > 0:
            start[(r * 3) % 8] += 1
            start = _fix_sum(start)
        best, mean = local_search(commander_mv, start,
                                  base_seed=master_seed + r * 104729, **kw)
        results.append((mean, best))
    results.sort(key=lambda t: t[0], reverse=True)
    return results[0][1], results[0][0]
```

Note: verify each `START_DECKS[mv]` sums to 98 at import; `_fix_sum` guarantees
it regardless. (Task-time check: adjust the row 2 entry so it sums to 98 without
relying on the fixer — replace the arithmetic placeholder with explicit ints.)

- [ ] **Step 2: Integration test** `tests/test_integration.py`

```python
import numpy as np, pytest
from optimizer import optimize_commander

EXPECTED_LANDS = {2:42, 3:42, 4:39, 5:39, 6:38}

@pytest.mark.slow
@pytest.mark.parametrize("mv", [4])   # start with the well-anchored 4-mv row
def test_reproduce_land_count(mv):
    best, mean = optimize_commander(mv, restarts=3, master_seed=1,
                                    max_sims=200_000)
    lands = int(best[7])
    assert abs(lands - EXPECTED_LANDS[mv]) <= 2   # same or adjacent
    assert best[mv-1] <= 3     # few/zero drops at the commander's own MV
```

- [ ] **Step 3: Run** — `uv run pytest tests/test_integration.py -m slow -q`
  (minutes). If land count is far off, re-examine the gameplay logic and mulligan
  policy before touching tolerances.

- [ ] **Step 4: Commit** — `git commit -am "feat: multi-restart optimizer + integration test"`

---

### Task 14: CLI (`main.py`)

**Files:** Create `main.py`, `tests/test_cli.py`, `README.md`

**Interfaces:**
- Produces: `python main.py run [--mv N] [--restarts R] [--max-sims M]` prints the
  table; `python main.py validate` runs the checkpoint anchors and reports
  measured value ± SE vs target. Uses `multiprocessing.Pool` across MVs for `run`.

- [ ] **Step 1: Failing smoke test** `tests/test_cli.py`

```python
import subprocess, sys

def test_validate_runs():
    r = subprocess.run([sys.executable, "main.py", "validate", "--quick"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0
    assert "72." in r.stdout        # prints a criterion near his checkpoint
```

- [ ] **Step 2: Implement** `main.py`

```python
import argparse, math
from multiprocessing import Pool
import numpy as np
from sim_core import simulate_deck
from optimizer import optimize_commander

MVS = [2, 3, 4, 5, 6]
LABELS = ["1", "2", "3", "4", "5", "6", "Sig", "Land"]

def _run_one(args):
    mv, kw = args
    best, mean = optimize_commander(mv, **kw)
    return mv, best, mean

def cmd_run(a):
    kw = dict(restarts=a.restarts, max_sims=a.max_sims, master_seed=a.seed)
    with Pool(len(MVS)) as p:
        rows = p.map(_run_one, [(mv, kw) for mv in MVS])
    rows.sort()
    print(f"{'MV':>3} | " + " ".join(f"{l:>4}" for l in LABELS) + " | SolRing | crit")
    for mv, best, mean in rows:
        cells = " ".join(f"{int(x):>4}" for x in best)
        print(f"{mv:>3} | {cells} |    1    | {mean:.3f}")

def cmd_validate(a):
    K = 200_000 if a.quick else 3_000_000
    opt4 = np.array([6,12,13,0,13,8,7,39], dtype=np.int64)
    m, v = simulate_deck(opt4, 4, K, 20240828)
    se = math.sqrt(v / K)
    print(f"4-mv optimal deck: {m:.3f} +/- {se:.3f}  (Karsten: 72.465)")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--restarts", type=int, default=3)
    r.add_argument("--max-sims", type=int, default=200_000)
    r.add_argument("--seed", type=int, default=0); r.set_defaults(fn=cmd_run)
    v = sub.add_parser("validate"); v.add_argument("--quick", action="store_true")
    v.set_defaults(fn=cmd_validate)
    a = ap.parse_args(); a.fn(a)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `README.md`** — usage, the model summary, his real-deck
  adjustment guidance (cut a land per 2–3 rocks; Cultivate = rock + 3-drop;
  fewer 1-drops with many tapped lands), and the honest caveat that exact
  bit-matching of his table is not expected.

- [ ] **Step 4: Run** — `uv run pytest tests/test_cli.py -q` then
  `uv run python main.py validate --quick`.

- [ ] **Step 5: Commit** — `git commit -am "feat: CLI run/validate + README"`

---

## Self-Review

**Spec coverage:** §2 criterion → T3; §3.2 mulligan → T4; §3.3 bottoming → T5;
§3.4 gameplay → T7; §4.1 sim_core → T1-3,6-9; §4.2 optimizer/CRN → T11-13; §4.3
CLI → T14; §5 anchors → T3 (23), T4/5/7 (crafted), T10 (72.465/72.434), T13
(land counts). All covered.

**Placeholder scan:** Task 7 contains a deliberate two-draft note (delete the
first draft, keep the canonical `play_turn`); Task 13 `START_DECKS[2]` has an
arithmetic placeholder to replace with explicit ints summing to 98. Both flagged
in-task. No other placeholders.

**Type consistency:** hand/board/deck are int64 arrays; `mana_available` (not
`mana_of`) is the final name — Task 8/tests import `mana_available`. `cstate` is
`[mv, cast]` throughout. `simulate_deck` returns `(mean, var)` everywhere.

**Known risks to watch during execution:**
1. The `spent` bookkeeping in `play_turn` step 5 (signet net cost = 1 this turn)
   is the subtlest arithmetic — the crafted `test_turn3_rock_plus_nminus1` guards
   it. If T10's absolute value drifts, suspect this first.
2. Bottomed-cards-stay-in-`lib[0:7]` relies on the draw pile being `lib[7:]`;
   holds because ≤7 draws happen. Do not "compact" the library.
3. T10 absolute anchor may need K≈2-4M to sit within tolerance; that's seconds in
   Numba after warmup. Never loosen tolerance to force a pass — fix the model.
