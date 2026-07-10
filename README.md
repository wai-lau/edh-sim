# edh-sim — optimal Commander mana curves

Starts as a from-scratch reimplementation of Frank Karsten's *"What's an Optimal
Mana Curve and Land/Ramp Count for Commander?"* (ChannelFireball, updated
2025-08-28) — a Monte Carlo simulator of expected **compounded board mana** plus a
local-search optimizer over the 99-card list — then **extends past his goldfish**
with interaction (board wipes), diminishing returns (a per-turn score cap), card
draw, a swept game-length horizon, and per-bracket curves.

Python + Numba (`@njit`-compiled engine). Throughput ~1M games/s for the bare
Karsten model; **~0.2–0.5M/s** for the full model (wipes + cap + draw).

---

## Results — mana curves by Commander bracket

Beyond Karsten's goldfish replication, this model adds **interaction and realism**
— board wipes, diminishing returns on over-development, and card draw — then
sweeps the game-length horizon (T = 2–15) and folds the per-horizon optima into
per-bracket curves via a normal weighting centered on each bracket's
characteristic game length (r/EDH midpoints: B4≈7, B3≈9, B2≈11 turns; **σ = 1.5**):

![Bracket turn-weighting](docs/bracket_weights.svg)

Format `[1d 2d 3d 4d 5d 6d | Draw | Signets | Lands]`, **+ 1 Sol Ring** each.
"Draw" = a card-draw spell (pay X, draw X); "Signets" = any 2-mana rock.

### Bracket 4 — Optimized (fast, ~7-turn games)
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Draw | Sig | Land |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:----:|:---:|:----:|
| 2 | 13 | 0 | 19 | 12 | 7 | 3 | 1 | 0 | 43 |
| 3 | 13 | 16 | 0 | 15 | 7 | 4 | 0 | 0 | 43 |
| 4 | 13 | 15 | 13 | 0 | 8 | 5 | 1 | 2 | 41 |
| 5 | 12 | 15 | 13 | 8 | 0 | 5 | 2 | 2 | 41 |
| 6 | 11 | 16 | 14 | 10 | 4 | 1 | 1 | 2 | 39 |

### Bracket 3 — Upgraded (mid, ~9-turn games)
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Draw | Sig | Land |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:----:|:---:|:----:|
| 2 | 5 | 0 | 14 | 12 | 10 | 9 | 5 | 0 | 43 |
| 3 | 5 | 10 | 0 | 13 | 10 | 10 | 4 | 3 | 43 |
| 4 | 5 | 7 | 9 | 0 | 11 | 12 | 8 | 6 | 40 |
| 5 | 5 | 7 | 10 | 10 | 0 | 12 | 8 | 6 | 40 |
| 6 | 5 | 7 | 10 | 11 | 8 | 4 | 8 | 6 | 39 |

### Bracket 2 — Core (slow, ~11-turn games)
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Draw | Sig | Land |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:----:|:---:|:----:|
| 2 | 2 | 0 | 10 | 10 | 9 | 13 | 14 | 1 | 39 |
| 3 | 3 | 4 | 0 | 9 | 10 | 14 | 14 | 5 | 39 |
| 4 | 1 | 3 | 6 | 0 | 10 | 16 | 17 | 7 | 38 |
| 5 | 2 | 2 | 6 | 10 | 0 | 16 | 17 | 8 | 37 |
| 6 | 2 | 2 | 6 | 10 | 9 | 8 | 17 | 8 | 36 |

(Brackets 1 Exhibition and 5 cEDH — the extremes the model fits worst — are
omitted. Numbers from the interim quick pass; the deep run tightens the ±1s.)

**Draw cards are cheap cantrips.** Across MVs the draw spells are cast at an
average **X ≈ 1.9–2.2** (≈65% at X = 1) — a 1-mana leftover-mana filter to refill
the hand, not a big draw-7. Slightly higher X for cheaper commanders.

**Read across the brackets:** fast → cheap curve, ~0 ramp/draw. Slow → 1-drops
vanish; six-drops + ~6–8 signets + **14–17 card-draw** dominate, i.e.
**wipe-resilient draw-go control** that rebuilds to the development cap after each
board wipe. **Draw switches on with game length** — 0 through turn ~7, then 7 at
turn 9, up to ~20 by turn 12+. Lands drift 36–44 as draw/ramp substitute. The `0`
on the diagonal is Karsten's Insight #2 (no drops at the commander's own MV); it
breaks at MV 6 because **6 is the model's ceiling** (no 7+ drops) and after each
wipe you need a *stock* of top-end to rebuild — so you run 6-drops even though the
MV-6 commander is one (both worth 6.2).

### The full model (beyond Karsten's goldfish)

- **Board wipes** (interaction): each turn ≥5, chance `0.10 × 1.2^(wipe-free turns)`;
  a wipe kills creatures + sends the commander to the command zone, but **rocks,
  lands, and your hand survive**. This is what makes ramp and draw earn their keep.
- **Development cap** (diminishing returns): each turn contributes `min(board
  value, 12)` to the score — over-committing past ~12 mana of board is wasted, so
  you hold cards (which then survive wipes).
- **Card draw:** a pay-X-draw-X spell, played last, only when hand < 7 (or stuck).
  Mostly fires as a **1-mana leftover cantrip** (avg X ≈ 2).
- **Optimizer:** explore-cheap → select-precise (many cheap restarts, then a
  high-sim showdown of the finalists) — avoids the max-of-noisy-estimates bias.

**Chosen constants (magic numbers, not derived).** These are hand-picked and
tunable, not fit to data: wipe **base 10% / ×1.2 escalation / start turn 5**;
score **cap = 12**; weighting **σ = 1.5**. Six-drop / MV-6-commander = **6.2** is
Karsten's experience-based super-linear premium (also a fudge, but justified).
Bracket centers (7/9/11) come from r/EDH game-length data; hand-limit 7 is a rule.

### The mulligan (Karsten's open problem, solved)

We derived the optimal mulligan by value-function DP — the piece Karsten left as
future work — and distilled it into a fast rule that beats his heuristic at every
horizon:

> **Keep 3–4 lands. Mull the fifth (flood). 2 lands only with a mana rock, 1 only
> behind a Sol Ring.** Fast decks also want ≥1 non-rock play.

Switching from his mulligan to this one **shifted the optimal decks toward more
lands and fewer rocks** — a smarter mulligan handles flood, so the deck no longer
needs rocks as land-consistency insurance. (That DP was run on the *pre-wipe*
model; the full model uses the same distilled rule but hasn't re-derived it.)
Caveats and full derivation in [`docs/methodology.md`](docs/methodology.md).

---

## Criterion & model

- Deck = `[1,2,3,4,5,6-drop, Signet, Land, Draw]` (draw slot optional) summing to
  98, plus one Sol Ring → 99 cards. Commander = a free MV-N spell in the command
  zone (cast once, recastable after a wipe).
- **Criterion:** at each turn end, sum the mana worth of on-board non-rock,
  non-land permanents, **capped per turn** (`min(·, cap)`). k-drop = k,
  six-drop = **6.2**, commander scored the same (raw MV for 1–5, **6.2 at MV6**).
  Rocks, lands, draw spells score 0. Averaged over random games; the horizon T is
  a parameter (Karsten's base = 7; we sweep **2–15**).
- Multiplayer rules: **free first mulligan**, **always on the draw**, London
  bottoming.
- **Base = Karsten (goldfish, no cap, no draw).** The full model adds wipes + cap
  + draw (above); the faithful base is recovered with `wipes=False`, no cap, no
  draw slot.

Full detail: the spec under `docs/superpowers/specs/` and
[`docs/methodology.md`](docs/methodology.md).

## Usage

```bash
uv sync
make test                           # 59 fast tests (uv run pytest)
make lint                           # ruff + 500-line file cap
uv run pytest -m slow               # 5 Monte Carlo / optimizer anchors (~3.5 min)

# FULL model (wipes + cap + draw), explore-cheap optimizer -> per-bracket curves:
uv run python run_final.py --seed-json <seed.json> --restarts 16 --final-sims 1200000

# FAITHFUL Karsten base (7-turn goldfish, no wipes/cap/draw):
uv run python main.py run           # optimize all 5 commander MVs -> table
uv run python main.py validate      # criterion vs his 72.465 checkpoint

# analysis: horizon sweep, self-pacing overnight runner, value-function mulligan DP:
uv run python sweep.py --turns-min 2 --turns-max 15
uv run python -c "import mulligan"
```

## Faithful replication (Karsten's 7-turn goldfish base)

Before the extensions, the bare model reproduces Karsten's published table
(`[1d 2d 3d 4d 5d 6d | Sig | Land]`, all decks + 1 Sol Ring):

| MV | ours | Karsten |
|----|------|---------|
| 2 | `9 0 21 13 9 4 \| 0 \| 42` | `9 0 20 14 9 4 \| 0 \| 42` |
| 3 | `8 18 0 16 10 4 \| 0 \| 42` | `8 19 0 16 10 3 \| 0 \| 42` |
| 4 | `7 18 16 0 10 5 \| 0 \| 42` | `6 12 13 0 13 8 \| 7 \| 39` |
| 5 | `7 15 13 11 0 8 \| 4 \| 40` | `6 12 10 13 0 10 \| 8 \| 39` |
| 6 | `6 16 13 12 7 0 \| 4 \| 40` | `6 12 10 14 9 0 \| 9 \| 38` |

### What reproduced

- **Insight #2 — zero N-drops at the commander's own MV:** exactly, for all five
  commanders (the `0` on the diagonal). The optimizer discovers this on its own.
- **Cheap commanders (MV 2, 3):** near-exact whole-deck match — same 42 lands,
  same 0 signets, curve within ±1 card per slot.
- **Insight #4 — high land counts:** ours 42/42/42/40/40 vs his 42/42/39/39/38.
  We land, if anything, *higher*.
- **Insight #1 — bulk on cheap spells:** the mass sits on 2s/3s/4s.
- **Insight #3 — more ramp for pricier commanders (directional):** our signet
  counts 0/0/0/4/4 rise with MV, as his 0/0/7/8/9 do.

### Where it diverges (honest)

- **Absolute criterion ~0.6% low.** His 4-MV optimal deck scores 72.465 in his
  model, ~71.98 in ours. His mulligan/gameplay heuristics have unspecified
  degrees of freedom; small interpretation differences shift the absolute number.
  The *ordering* of his named perturbation (72.465 > 72.434) reproduces with the
  same sign — the model-independent anchor.
- **Ramp undervalued for expensive commanders.** For MV 4–6 our optimizer keeps
  fewer signets (0/4/4 vs his 7/8/9) and a denser low curve instead of his
  top-heavy ramp curve. Both decks sit on the near-flat optimum ridge Karsten
  describes in Insight #5 (our MV-4 optimum and his differ by ~0.5 compounded
  mana, ~0.7%). Notably, **ramp/Sol-Ring value is exactly the part Karsten
  flagged as his noisiest, weakest result** ("one of the weaker parts of this
  study"), so a divergence here is expected. Our engine converts ramp to
  compounded mana slightly less generously than his did.

## Design choices worth noting

- **Common Random Numbers (CRN):** within each local-search iteration every
  candidate deck is scored on the *same* batch of game seeds, so Sol Ring's luck
  cancels in the comparison — a variance-reduction fix for the noise Karsten
  flagged. Seeds refresh each iteration (no seed overfit); multiple restarts
  guard local maxima.
- **Numba engine:** removes his multi-day runtime — a 2M-game evaluation is ~2 s
  for the bare model, ~5–10 s for the full model (wipes + cap + draw).

## Applying this to real decks

As in the article, this is an idealized curve model. For real decks: cut a land
per 2–3 mana rocks (or per 3–4 cantrips / mana dorks) but don't drop below ~37;
treat `Cultivate` as rock + 3-drop, `Llanowar Elves` as rock + 1-drop, an MDFC
land as half-land; run slightly fewer 1-drops if you have many tapped lands. And
don't chase the table exactly — aggro wants a lower curve, control a higher one,
and synergy beats raw curve.
