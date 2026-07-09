# EDH Mana-Curve Simulator — Design Spec

**Date:** 2026-07-09
**Goal:** Replicate Frank Karsten's Commander mana-curve experiment ("What's an
Optimal Mana Curve and Land/Ramp Count for Commander?", ChannelFireball, updated
2025-08-28): a Monte Carlo simulator of the *expected compounded mana spent over
the first seven turns* plus a local-search optimizer that reproduces his
five-row optimal-deck table (Commander mana value 2–6).

Stack: **Python + Numba** (njit hot path). Faithful to his model; the Numba core
removes his multi-day runtime and lets us run enough games to sharpen his flagged
Sol Ring variance. Full replication scope (engine + optimizer + table).

---

## 1. The model

### 1.1 Deck

A 99-card deck is 8 integer category counts plus exactly one Sol Ring:

```
[c1, c2, c3, c4, c5, c6, signets, lands]   with sum == 98
+ 1 Sol Ring                                (always exactly 1)
```

- `c1..c6` = one-drops … six-drops (blank permanents of that mana value).
- `signets` = Arcane-Signet-equivalent mana rocks (Signet/Talisman/Fellwar/
  Three Visits/Nature's Lore all modeled identically: a 2-mana rock).
- Sol Ring is its own always-present rock (costs 1, taps for 2).
- Cards are abstracted: a three-drop is "a 3-mana permanent with a blank text
  box". No colors, no tapped lands, no card draw, no mana dorks, no mana sinks.

### 1.2 Commander

A free spell of mana value **N** (N ∈ {2,3,4,5,6}) sits in the command zone from
turn 0, castable any turn once mana allows. Cast **once**; never recast (goldfish
opponent, so no removal to dodge). Modeled as an always-in-hand card of mana
value N that the gameplay logic may cast.

### 1.3 Mana available

Each turn, untapped mana is:

```
N_mana = lands_in_play + 2*(SolRing_in_play) + 1*(signets_in_play)
```

All sources untap every turn (goldfish; nothing taps for abilities).

---

## 2. Criterion — expected compounded mana over 7 turns

At the **end of each turn t = 1..7**, sum the mana worth of every on-board
permanent that is **not a mana rock and not a land**:

| Permanent | Worth |
|---|---|
| k-drop, k ∈ 1..5 | k |
| six-drop | **6.2** (power scales super-linearly past 5) |
| commander (MV N) | **N** (raw MV) |
| land | 0 |
| Sol Ring / Signet | 0 |

The game's score = Σ over t=1..7 of that turn-end board sum. The deck's criterion
= **expectation of that score over random games**, estimated by Monte Carlo.

**Golden worked example (from the article).** Mulligan into a bad hand; first
play is a one-drop on T3, a Signet on T4, the MV-6 commander on T5, nothing else:

```
T1,T2: 0     (nothing on board)
T3: 1        (one-drop)
T4: 1        (one-drop; signet excluded)
T5: 1+6 = 7  (one-drop + commander)
T6: 7
T7: 7
score = 0+0+1+1+7+7+7 = 23
```

Note the commander is scored at raw MV 6 here (1+6 = 7), **not** 6.2. Six-drops
(the deck category) are the only thing worth 6.2. This example is the primary
unit test for the scorer.

**Modeling notes carried from the article (do not "fix"):**
- Mana rocks contribute 0 to the criterion — their value is enabling faster
  N-drops, which *do* count.
- Opponents don't interact (goldfish). A curve-out that would force interaction
  still counts as good.
- Relevant game length = 7 turns (matches his Game Knights sample; the
  early-to-mid game is where curving out matters).

---

## 3. Game rules encoded (exact per article)

### 3.1 Start of game

- Multiplayer Commander: **free first mulligan** (CR 103.4c — first mulligan
  doesn't count toward cards bottomed) and **always on the draw** (CR 800.7 —
  starting player does not skip turn-1 draw). So a card is drawn every turn
  including turn 1.
- London mulligan: each mulligan draws a fresh 7; on keep after M *counted*
  mulligans, put M cards on the bottom.

### 3.2 Mulligan policy (his hand-crafted heuristic — encode faithfully)

Let L = lands in the 7-card hand (Sol Ring counts as a land for keep tests),
S = signets, SR = Sol Ring present.

| Hand attempt | Counted mulls so far | Bottom on keep | Keep condition |
|---|---|---|---|
| 1st | 0 | 0 | (3 ≤ L ≤ 5 ∧ L+S ≤ 5) ∨ (SR ∧ 1 ≤ L ≤ 5) |
| 2nd | 0 (free mull) | 0 | same as 1st **∨ L == 2** |
| 3rd (→ six) | 1 | 1 | (2 ≤ L ≤ 4) ∨ (L == 1 ∧ SR) |
| 4th (→ five) | 2 | 2 | (2 ≤ L ≤ 4) ∨ (L == 1 ∧ SR) |
| 5th (→ four) | 3 | 3 | **always keep** (floor) |

Keep tests for the → six / → five hands are evaluated conceptually "after
bottoming"; in practice the keep decision uses the drawn 7 and the same L/SR
thresholds, then bottoming (3.3) is applied.

### 3.3 Bottoming (need to bottom B cards)

In order, stop once B reached:

1. **Superfluous signets first:** bottom up to `max(0, signets_in_hand - 1)`
   signets (keep at most one signet).
2. **Lands toward 3:** bottom lands until land count (with Sol Ring counted as a
   land, and Sol Ring never bottomed) is ≤ 3.
3. **Spells, most expensive first:** if still bottoming, bottom the highest-MV
   spells (6-drop → … → 1-drop). The single kept signet is not treated as a
   bottomable spell unless nothing else remains.

### 3.4 Gameplay logic per turn (his hand-crafted heuristic — encode faithfully)

At the start of each turn: untap all, draw one card. Then, in order:

1. **Play a land** if one is in hand.
2. **Cast Sol Ring** if in hand (costs 1).
3. **Turn-1 Sol Ring stop:** if it's turn 1 and Sol Ring was just cast, end the
   turn (cast nothing else — this preempts the signet step below). Interpretation
   of "after a turn one Sol Ring, we're done for the turn."
4. **On turns 1 and 2 only:** cast an **Arcane Signet** if possible (costs 2).
   (On turn 1 this is reached only when no Sol Ring was cast, per step 3.)
5. Let N = mana still available. **On turns 3 and 4:** if possible, cast a **mana
   rock and an (N−1)-drop** (deploy a signet for ramp while still curving — the
   signet pays 2 and taps for 1, net leaving enough for an (N−1)-drop).
6. **Gap-fill:** on any turn, if we do **not** hold an N-drop but do hold a
   two-drop **and** a distinct (N−2)-drop, cast both (prefer 2 + (N−2) over an
   N-1-drop that wastes a mana). Commander counts as "holding an N-drop" of its
   own MV.
7. **Greedy fill:** cast the highest-MV castable spell, from six-drops down,
   repeat until nothing is castable. Commander is a castable spell of MV N here.
8. **Retroactive rock:** if mana is left over at end of turn and a signet is in
   hand that could have been cast, cast it now.

These policies are explicitly heuristics (Karsten's words: "need not be
optimal"). We reproduce them, we do not optimize them.

---

## 4. Architecture

### 4.1 `sim_core.py` — Numba njit hot path

Card encoding as `int8`: `0 = land, 1..6 = k-drops, 7 = signet, 8 = Sol Ring`.
Commander tracked separately (MV + a "cast?" flag).

- `simulate_game(deck_counts: int8[8], commander_mv: int, seed: uint64,
  injected_library: optional int8[]) -> float`
  - Build the 99-card library array from counts, Fisher-Yates shuffle with a
    seeded PRNG (numba-friendly, e.g. xorshift/PCG on `uint64` state), or use
    `injected_library` verbatim (for deterministic tests).
  - Run mulligan loop (§3.2) + bottoming (§3.3) → opening hand.
  - Play 7 turns (§3.4), accumulate criterion (§2).
  - Return the game score.
- `simulate_deck(deck_counts, commander_mv, n_games, base_seed) -> (mean, var)`
  - Loop `n_games`, Welford's online mean+variance. Seeds derived from
    `base_seed` so a deck+base_seed pair is reproducible and supports CRN
    (§4.2). Returns mean and variance (variance → standard error for stopping).

All of mulligan, bottoming, turn play, scoring live inside njit functions
operating on fixed-size int arrays — no Python objects on the hot path.

### 4.2 `optimizer.py` — local search (Python orchestration)

Faithful to Karsten's local search, plus CRN variance reduction (approved
option B).

- **Start deck:** a reasonable seed deck per commander MV (scaled from his 40/60
  card optima), sum == 98 + Sol Ring.
- **Neighborhoods** (99-card balanced moves — every cut matched by an add):
  - *Cross* (used while best-so-far sim count < 150k): cut ≤1 card and add ≤1
    card total (single swap between two categories).
  - *Star* (used once best-so-far ≥ 150k sims): every deck whose category counts
    each differ by ≤1 from the current best.
  - Always also evaluate the best deck seen in all prior iterations.
- **Escalating simulation budget:** iteration 1 uses 10k games/deck; +1k each
  iteration. A per-deck sim cache accumulates games across iterations (revisited
  decks combine prior + new sims via pooled mean/variance).
- **Common random numbers (CRN):** when ranking the decks of one neighborhood,
  evaluate them all against the **same batch of game seeds** for that iteration
  → the Sol Ring's presence is shared across decks and cancels in the pairwise
  comparison, sharply lowering the variance of the ranking. CRN is applied
  **per-comparison only**; the seed batch is **refreshed each iteration** (fresh
  games as the budget grows) to avoid overfitting one seed set.
- **Move rule:** move to the neighbor with the best CRN-estimated criterion.
- **Stopping:** stop when the best deck has > 200k accumulated sims and is a
  local optimum (no neighbor better).
- **Multi-restart:** run the whole search ≥3 times from different start decks /
  master seeds; keep the deck with the highest criterion. Guards local maxima
  (search topology) — orthogonal to CRN.

### 4.3 `main.py` — CLI

- `run` — optimize all five commander MVs (2–6), print the results table (curve
  + signets + lands). Runs the five MVs in parallel via `multiprocessing`.
- `validate` — evaluate the golden anchors (§5) and report pass/fail with the
  measured value, standard error, and tolerance.
- Config: games-per-deck budget, restart count, master seed, MV subset — CLI
  flags with sane defaults.

### 4.4 Project layout

```
edh-sim/
  sim_core.py
  optimizer.py
  main.py
  pyproject.toml          # numba, pytest, numpy; uv-managed
  tests/
    test_scoring.py       # §5.1 worked example -> 23
    test_mulligan.py      # §5.2 keep/mull + bottoming on crafted hands
    test_gameplay.py      # §5.3 crafted turn scenarios
    test_regression.py    # §5.4 criterion checkpoints (slow, marked)
    test_optimizer_integration.py  # §5.5 reproduce a row (slow, marked)
  docs/superpowers/specs/2026-07-09-edh-mana-curve-sim-design.md
```

---

## 5. Validation — golden anchors from the article

Built test-first. Anchors 1–3 are deterministic/cheap; 4–5 are Monte Carlo and
marked slow.

1. **Scorer:** the §2 worked-example trace scores **exactly 23** (inject the
   library / force the draw sequence so no shuffle is involved).
2. **Mulligan + bottoming:** crafted 7-card hands hit the expected keep/mull
   decisions (§3.2) and bottom the expected cards (§3.3).
3. **Gameplay:** crafted small scenarios exercise each numbered step of §3.4
   (turn-1 Sol Ring stop; T3/T4 rock + (N−1)-drop; gap-fill 2 + (N−2);
   retroactive rock).
4. **Criterion checkpoints (high K, tight SE):** the 4-mana-commander optimal
   deck `{6,12,13,0,13,8, SolRing+7 signet, 39 lands}` scores **≈ 72.465**; the
   perturbation (−1 one-drop, −1 two-drop, −1 three-drop, +1 five-drop, +1
   six-drop, +1 signet) scores **≈ 72.434**. Run enough games that the standard
   error resolves the 0.031 gap (Numba makes millions of games cheap).
5. **Optimizer integration (slow):** the optimizer reproduces his table land
   counts (2mv→42, 3mv→42, 4mv→39, 5mv→39, 6mv→38) and curve shape.

**Honest caveat.** We will not bit-match his exact table. His own runs varied
slightly ("very similar yet slightly different outcomes in different runs"), so
success = landing on the **same or an adjacent deck** for each MV and reproducing
the **criterion checkpoints within standard error**. The checkpoint numbers
(72.465 / 72.434) are the hard regression anchors; the table is the softer
integration target.

---

## 6. Testing & workflow

- **TDD:** write each test in §5 before the code it exercises; scorer and
  mulligan/gameplay first (deterministic), then the Monte Carlo layer, then the
  optimizer.
- **Determinism:** seeded PRNG throughout; `simulate_game` accepts an injected
  library so tests pin exact card sequences.
- **`uv run pytest`**; slow Monte Carlo / optimizer tests marked and excluded
  from the default fast run.

## 7. Reference code

Karsten links his own Python. We **reimplement from the prose** (that is the
replication). *After* our engine passes anchors 1–4, optionally locate his
published code and diff *behavior* (not source) on a few decks as an extra
cross-check. We do not copy his code.

## 8. Out of scope (his stated model limits — keep them)

Colors / mana fixing / tapped lands; card draw & cantrips; mana dorks; mana
sinks; sweepers; opponent interaction; cEDH turn-3 combo; recasting the
commander; optimizing the mulligan or gameplay policies. All abstracted away, as
in the article. Real-deck adjustments (e.g. "cut a land per 2–3 rocks", "Cultivate
= rock + 3-drop") are documented as guidance in the README, not modeled.
