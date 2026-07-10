# Methodology

How this project estimates optimal Commander mana curves — the model, the
objective, the game rules it encodes, the Monte Carlo estimator, the optimizer,
and the horizon/bracket extension layered on top of Karsten's original design.

It is a faithful reimplementation of Frank Karsten's *"What's an Optimal Mana
Curve and Land/Ramp Count for Commander?"* (ChannelFireball, 2025), built from
the article's prose, with two additions of our own: **common-random-numbers**
variance reduction in the optimizer, and a **game-length / bracket** layer that
sweeps the turn horizon and folds the results into per-bracket curves.

---

## 1. The question

Given a commander of mana value **N**, what distribution of a 99-card deck across
one-drops … six-drops, mana rocks, and lands lets you spend your mana most
effectively in the early-to-mid game? "Most effectively" is made precise by an
objective (§3). Colors, specific cards, synergies, and interaction are
abstracted away; every card is a blank permanent of a given mana value.

---

## 2. Deck model

A deck is eight integer counts plus one implicit Sol Ring:

```
[ c1, c2, c3, c4, c5, c6, signets, lands ]   with sum == 98
+ 1 Sol Ring                                  → 99-card library
```

- `c1…c6` — one- to six-drops (blank permanents of that mana value).
- `signets` — two-mana rocks that tap for one (Arcane Signet and its many
  equivalents: Talisman, Fellwar Stone, Nature's Lore, Three Visits, …).
- `Sol Ring` — always exactly one; costs one, taps for two.
- The **commander** is a free spell of mana value N in the command zone,
  castable any turn once mana allows, cast once, never recast (goldfish, so no
  removal to dodge).

Everything is encoded as small integers on fixed-size arrays so the whole
per-game engine compiles under Numba (`@njit`). Card codes:
`0=land, 1..6=k-drop, 7=signet, 8=Sol Ring`. Hands and battlefields are
length-9 count vectors indexed by that code.

### Abstractions carried from the article (kept, not "fixed")

No colors, no tapped lands, no card draw, no mana dorks, no mana sinks, no
sweepers, no opponent interaction, no cEDH combo. Real-deck translations
(a `Cultivate` ≈ rock + three-drop, a `Llanowar Elves` ≈ rock + one-drop, an
MDFC land ≈ half-land) are guidance for the reader, not part of the model.

---

## 3. Objective — expected compounded mana over the horizon

At the **end of each turn t = 1 … H** (H = horizon, default 7), sum the mana
worth of every on-board permanent that is **not a mana rock and not a land**:

| Permanent | Worth |
|---|---|
| k-drop, k ∈ 1..5 | k |
| six-drop | **6.2** (power scales super-linearly past five) |
| commander (MV N) | **N** (raw mana value, even for N = 6) |
| land, Sol Ring, signet | 0 |

A game's score is the sum of those turn-end board values over turns 1…H. The
deck's objective value is the **expectation of that score over random games**.

The intuition (Karsten's): a permanent generates advantage every turn it sits on
the battlefield, roughly proportional to its mana value, and that advantage
compounds. Mana rocks score zero directly — their value is *enabling* the
scoring permanents sooner. Games are usually won by whoever compounds the most
mana, so a deck that curves out while the table stumbles tends to win.

**Worked check.** A bad game — one-drop on T3, signet T4, MV-6 commander T5,
nothing else — scores `0+0+1+1+7+7+7 = 23` over seven turns (the commander is
scored at its raw MV 6, so T5 = 1 + 6 = 7). This is the primary unit test of the
scorer.

---

## 4. Game rules encoded

Multiplayer Commander specifics, per the comprehensive rules:

- **Free first mulligan** (CR 103.4c) and **always on the draw** (CR 800.7 — the
  starting player does not skip the turn-1 draw; a card is drawn every turn
  including turn one).
- **London mulligan:** each mulligan draws a fresh seven; on keep after M
  *counted* mulligans, put M cards on the bottom. The free first mulligan does
  not count, so the bottoming floor is: keep hand 1 or 2 → bottom 0; hand 3 → 1;
  hand 4 → 2; hand 5 (mulligan to four) → 3, and hand 5 is always kept.

### Mulligan keep policy (Karsten's hand-crafted heuristic)

With L = actual lands (Sol Ring is *not* counted as a land here — only in
bottoming), S = signets, SR = Sol Ring present:

| Hand | Keep if |
|---|---|
| 1st | (3 ≤ L ≤ 5 ∧ L+S ≤ 5) ∨ (SR ∧ 1 ≤ L ≤ 5) |
| 2nd | same, **or** L = 2 |
| 3rd / 4th | (2 ≤ L ≤ 4) ∨ (L = 1 ∧ SR) |
| 5th | always keep |

### Bottoming (need to bottom B cards)

In order until B reached: superfluous signets (keep ≤ 1) → lands until land count
(with Sol Ring counted as a land, never bottomed) ≤ 3 → spells most-expensive
first.

### Per-turn gameplay (Karsten's hand-crafted heuristic)

Untap, draw one, then:

1. Play a land if held.
2. Cast Sol Ring if held.
3. **Turn-1 Sol Ring stop** — if it is turn 1 and Sol Ring was just cast, end the
   turn (preempts the signet step).
4. On turns 1–2, cast a signet if affordable.
5. On turns 3–4, if possible cast **a rock and an (N−1)-drop** (deploy ramp while
   still curving; the signet pays 2 and taps for 1, netting an (N−1)-drop).
6. **Gap-fill** — if no N-drop is held but a two-drop and a distinct (N−2)-drop
   are, cast both (prefer 2 + (N−2) over an N-1-drop that wastes a mana). The
   commander counts as an N-drop of its own MV.
7. **Greedy** — cast the highest-MV castable spell, from six down, repeat.
8. **Retroactive rock** — if mana is left and a signet is held, deploy it.

Both policies are heuristics ("need not be optimal", in Karsten's words). We
reproduce them verbatim; we do **not** optimize them.

---

## 5. Monte Carlo estimation

A deck's objective is estimated by simulating many games and averaging.

- **PRNG.** A seedable xorshift64 generator, seeded through splitmix64 (both
  Numba-friendly). Each game gets a deterministic per-game seed derived from a
  deck's *base seed* and the game index, so a `(deck, base_seed, n_games)` triple
  is fully reproducible — and two decks sharing a base seed run on *identical*
  game shuffles (the basis for CRN, §6).
- **Shuffling** is Fisher-Yates over the 99-card int array. Bottomed cards remain
  physically at the front of the shuffled library; the draw pile is the tail, so
  within a ≤12-turn game they are never redrawn.
- **Accumulation** uses Welford's online algorithm for a numerically stable mean
  and variance in one pass. The variance gives a standard error
  (`SE = √(var/n)`) used to judge when two decks are distinguishable.

Throughput is ~1 M games/sec per core after Numba warm-up; a two-million-game
evaluation takes ~2 s, which is what makes the optimizer (below) tractable — the
original ran for days.

---

## 6. Optimizer — local search with common random numbers

The search follows Karsten's local search, plus a variance-reduction fix.

**Neighborhoods** (all moves keep the deck at 99 cards; Sol Ring is fixed and not
a category):

- *Cross* — cut one card from category A, add one to B (a single swap). ~56
  neighbors. Used while the incumbent has few accumulated sims.
- *Star* — every deck whose category counts each differ by ≤1 from the incumbent
  (a balanced ±1 move across many axes). ~1,100 neighbors. Used for final
  polishing once the incumbent is well-sampled.

**Escalating budget.** Early iterations evaluate cheaply (few thousand games per
deck) to explore; the per-deck budget grows each iteration. A per-deck cache
pools games across iterations, so a revisited deck accumulates precision rather
than restarting.

**Common random numbers (CRN).** Within a single iteration, every candidate in
the neighborhood is scored on the **same batch of game seeds**. Because the Sol
Ring (and every other card) then appears in the *same* games for every candidate,
its luck cancels in the pairwise comparison — the variance of the *difference*
between two decks collapses even though each deck's absolute estimate is still
noisy. This directly addresses the "Sol Ring variance" Karsten flagged as the
weakest part of his study, where a deck that happened to draw Sol Ring more often
than average could win a comparison on luck. The seed batch is **refreshed every
iteration** (fresh games as the budget grows) so the search never overfits one
seed set.

**Move rule.** Move to the neighbor with the best CRN-estimated objective.

**Stopping.** Stop when the incumbent has surpassed the max-sims budget and no
neighbor beats it (a local optimum).

**Multi-restart.** The whole search is run several times from different start
decks / master seeds; the best result is kept. This guards against local maxima
— a property of the *search topology*, orthogonal to CRN, which only sharpens
*ranking*.

### Why CRN is not a local-maximum trap

CRN fixes seeds only *within a comparison*, to make ranking fair; it does not
freeze the search. Local maxima come from the neighborhood structure and are
handled by restarts + the star neighborhood. If anything, CRN *reduces* one trap:
without it, evaluation noise can create spurious "lucky bumps" that halt the
search early; with it, the search follows the true objective gradient more
faithfully.

---

## 7. Horizon extension — game length as a bracket proxy

Karsten fixed the horizon at seven turns (roughly when a Commander game is
decided). We parameterize it (`n_turns`) and sweep it, because **game length is
the single biggest lever on the optimal curve** and it varies systematically with
deck power:

- Weaker / more casual tables → longer games → the horizon should be later.
- Stronger / more optimized tables → faster games → earlier horizon.
- cEDH → games can end turns 1–3 (rarely turn 1–2, but real).

Sweeping T = 1 … 12 and re-optimizing at each horizon shows the trend cleanly:
as the horizon lengthens, cheap one-drops fall away (a wasted slot when there is
time to deploy bigger threats), while six-drops and mana rocks climb (they
compound over more turns and there is time to ramp into them). Land counts stay
remarkably flat (~40) across horizons — land is the horizon-*insensitive* slot.

### Bracket weighting

Real games in a given power bracket do not all end on the same turn; they cluster
around a characteristic length with spread. We model that with a **normal
distribution over the horizon**: a bracket with center μ turns and spread σ
weights each per-horizon optimal deck by

```
w(T) = exp( −(T − μ)² / (2σ²) ),   normalized over the swept T range,
```

and the bracket's curve is the weighted average of the per-horizon optima,
`Σ_T w(T) · deck*(T)`, rounded to integers (lands absorb the rounding residual so
the deck sums to 98 + Sol Ring).

- **σ** is the standard deviation, *in turns*, of how long that bracket's games
  last — e.g. σ = 2 puts ~68% of the weight within μ ± 2 turns. Small σ → the
  bracket curve approaches the single-horizon optimum at μ; large σ → brackets
  blend toward a common average. It is a modeling knob (`--sigma`); set it from
  real game-length data if available.
- **Centers** map the five official Commander brackets to characteristic game
  lengths. The default keeps centers interior to the sweep range so the normal's
  tails are captured rather than truncated:

  | Center μ | Brackets |
  |---|---|
  | 5 turns | 4 Optimized, 5 cEDH |
  | 7 turns | 3 Upgraded |
  | 9 turns | 1 Exhibition, 2 Core |

  (Brackets sharing a center produce the same weighted curve by construction.)

A caveat worth stating: the weighted-average deck is a **blend of per-horizon
optima**, not the deck re-optimized against the normal-mixture objective directly
(the optimum of a mixture ≠ the mixture of optima). The two are close on the flat
objective ridge, but a direct-mixture optimization is the more principled version
if the distinction matters for a use case.

---

## 8. Differences from Karsten's original work

Everything in §2–§4 (deck model, objective, mulligan/gameplay heuristics) is a
faithful copy of his design. The differences are in the **engine, the optimizer,
and two additions on top** — plus the resulting numeric divergences.

### 8a. What we changed or added (methodology)

| | Karsten's original | This project |
|---|---|---|
| **Engine** | Plain Python; runs reported in *days* | Numba `@njit` hot path, ~1 M games/s; a 2 M-game eval in ~2 s |
| **Sample sizes** | Practically limited (10k → ~200k per deck) | Millions per deck are cheap; checkpoints resolve to a few thousandths |
| **Sol Ring variance** | Flagged as the study's weakest part; independent RNG per deck lets a lucky Sol Ring draw win a comparison | **Common random numbers** — every candidate in a neighborhood scored on the *same* game seeds, cancelling that luck in the comparison |
| **Restarts** | Re-ran manually ≥3× and picked the best | Systematic multi-restart with fresh master seeds |
| **Horizon** | Fixed at **7 turns** | **Parameterized and swept (T = 1…12)**; game length treated as the primary lever |
| **Power levels** | One table per commander MV, single horizon | **Bracket curves** — per-horizon optima folded via a normal weighting over game length, mapped to the five official Commander brackets |
| **Source** | His own Python | Reimplemented from the article's prose (then cross-checked against his numbers) |

The **CRN** and **bracket/horizon** layers are the substantive original
contributions; the rest is a faster, better-sampled re-execution of his method.

### 8b. What stayed identical (faithful)

Deck model and card abstractions; the compounded-mana objective including the
six-drop = 6.2 and commander = raw-MV scoring; the free first mulligan and
always-on-the-draw rules; the London bottoming; the exact mulligan keep policy
and the eight-step per-turn gameplay heuristic; the goldfish assumption; the
local-search shape (cross → star neighborhoods, escalating per-deck budget,
stop at a well-sampled local optimum).

### 8c. Resulting numeric differences

- **Absolute objective ~0.6% low.** His 4-MV optimal deck scores ~71.98 here vs
  his 72.465. His heuristics have unspecified degrees of freedom; small
  interpretation differences move the absolute number. The model-independent
  anchor — the *ordering* of his named perturbation (72.465 > 72.434) —
  reproduces with the same sign.
- **Insight #2 reproduced exactly** — the optimizer independently drives the
  commander's own mana value to ~zero drops in every case.
- **Land counts reproduced** (~38–42, matching or slightly above his 38–42).
- **Ramp undervalued at 7 turns.** For expensive commanders our optimizer keeps
  fewer signets than his (e.g. 0/0/0/4/4 vs his 0/0/7/8/9 across MV 2–6) and a
  denser low curve. This is exactly the parameter he flagged as noisiest, and it
  is **horizon-driven**: extending to 8–12 turns raises optimal ramp to meet or
  exceed his counts. The divergence is a statement about *assumed game length*,
  not a contradiction — and it is precisely what motivated the horizon/bracket
  layer (§7).
- Everything sits on the **near-flat optimum ridge** he describes: decks within a
  fraction of a percent of optimal are, for practical purposes, equivalent.

None of this is gospel. It is an idealized curve model with a single-minded
objective and a stack of deliberate abstractions, meant as a starting template.
Aggro wants a lower curve, control a higher one, and real synergy beats raw
curve every time.

---

## 9. Reproducing

```bash
uv sync
uv run pytest                 # 42 tests (37 fast, 5 slow anchors)

uv run python main.py validate            # objective vs his 72.465 checkpoint
uv run python main.py run                 # optimize the five commander MVs (7-turn)

# horizon / bracket sweep with live progress + ETA:
uv run python sweep.py --turns-min 1 --turns-max 12 --restarts 3 \
                       --max-sims 300000 --sigma 2 --out full.json
```

Determinism is total: seeded PRNG throughout, per-game seeds derived from a base
seed. Same script + same seeds → identical results. `simulate_game` accepts an
injected library so tests can pin exact card sequences.
