# edh-sim — replication of Karsten's Commander mana-curve experiment

A from-scratch reimplementation of Frank Karsten's *"What's an Optimal Mana Curve
and Land/Ramp Count for Commander?"* (ChannelFireball, updated 2025-08-28): a
Monte Carlo simulator of the **expected compounded mana spent over the first
seven turns** plus a local-search optimizer that searches for the best 99-card
mana curve for a given commander mana value (MV).

Reimplemented from the article's prose (not his code). Python + Numba: the whole
per-game engine is `@njit`-compiled, ~1M games/sec on one core.

---

## Results — mana curves by Commander bracket

Beyond Karsten's goldfish replication, this model adds **interaction and realism**
— board wipes, diminishing returns on over-development, and card draw — then
sweeps the game-length horizon (T = 2–15) and folds the per-horizon optima into
per-bracket curves via a normal weighting centered on each bracket's
characteristic game length (from r/EDH data; σ = 2 turns):

![Bracket turn-weighting](docs/bracket_weights.svg)

Format `[1d 2d 3d 4d 5d 6d | Signets | Lands | Draw]`, **+ 1 Sol Ring** each.
"Signets" = any 2-mana rock; "Draw" = a card-draw spell (pay X, draw X).

### Bracket 4 — Optimized (fast, ~7-turn games)
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Sig | Land | Draw |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:----:|:----:|
| 2 | 14 | 0 | 19 | 11 | 7 | 4 | 0 | 42 | 1 |
| 3 | 14 | 16 | 0 | 14 | 7 | 4 | 1 | 41 | 1 |
| 4 | 14 | 14 | 12 | 0 | 7 | 5 | 3 | 40 | 3 |
| 5 | 13 | 14 | 12 | 7 | 0 | 6 | 3 | 40 | 3 |
| 6 | 12 | 15 | 13 | 9 | 4 | 1 | 3 | 38 | 3 |

### Bracket 3 — Upgraded (mid, ~9-turn games)
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Sig | Land | Draw |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:----:|:----:|
| 2 | 6 | 0 | 14 | 11 | 9 | 8 | 0 | 44 | 6 |
| 3 | 6 | 10 | 0 | 13 | 9 | 9 | 3 | 43 | 5 |
| 4 | 6 | 8 | 9 | 0 | 10 | 11 | 5 | 40 | 9 |
| 5 | 6 | 7 | 10 | 10 | 0 | 12 | 6 | 38 | 9 |
| 6 | 5 | 8 | 10 | 10 | 7 | 4 | 6 | 39 | 9 |

### Bracket 2 — Core (slow, ~11-turn games)
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Sig | Land | Draw |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:----:|:----:|
| 2 | 3 | 0 | 10 | 10 | 9 | 12 | 1 | 40 | 13 |
| 3 | 3 | 6 | 0 | 10 | 9 | 13 | 4 | 40 | 13 |
| 4 | 2 | 4 | 7 | 0 | 10 | 15 | 6 | 38 | 16 |
| 5 | 2 | 3 | 7 | 10 | 0 | 15 | 7 | 38 | 16 |
| 6 | 2 | 3 | 7 | 10 | 9 | 7 | 8 | 36 | 16 |

(Brackets 1 Exhibition and 5 cEDH — the extremes the model fits worst — are
omitted. Numbers from the interim converge run; the deep run tightens the ±1s.)

**Read across the brackets:** fast → cheap curve, ~0 ramp/draw. Slow → 1-drops
vanish; six-drops + ~6–8 signets + **13–16 card-draw** dominate, i.e.
**wipe-resilient draw-go control** that rebuilds to the development cap after each
board wipe. **Draw switches on with game length** — 0 through turn ~7, then 7 at
turn 9, up to ~20 by turn 12+. Lands drift 36–44 as draw/ramp substitute. The `0`
on the diagonal is Karsten's Insight #2 (no drops at the commander's own MV),
which holds except at MV 6 (six-drops are the 6.2-premium ceiling — nothing higher
to reach).

### The full model (beyond Karsten's goldfish)

- **Board wipes** (interaction): each turn ≥5, chance 0.10 × 1.2^(wipe-free turns);
  a wipe kills creatures + sends the commander to the command zone, but **rocks,
  lands, and your hand survive**. This is what makes ramp and draw earn their keep.
- **Development cap** (diminishing returns): each turn contributes `min(board
  value, 12)` to the score — over-committing past ~12 mana of board is wasted, so
  you hold cards (which then survive wipes).
- **Card draw:** a pay-X-draw-X spell, played last, only when hand < 7 (or stuck).
- **Optimizer:** explore-cheap → select-precise (many cheap restarts, then a
  high-sim showdown of the finalists) — avoids the max-of-noisy-estimates bias.

### The mulligan (Karsten's open problem, solved)

We derived the optimal mulligan by value-function DP — the piece Karsten left as
future work — and distilled it into a fast rule that beats his heuristic at every
horizon:

> **Keep 3–4 lands. Mull the fifth (flood). 2 lands only with a mana rock, 1 only
> behind a Sol Ring.** Fast decks also want ≥1 non-rock play.

Switching from his mulligan to this one **shifts every optimal deck toward more
lands and fewer rocks** (mid/slow brackets: **+2–3 lands, −1–2 signets**) — a
smarter mulligan handles flood, so the deck no longer needs rocks as
land-consistency insurance. Caveats and full derivation in
[`docs/methodology.md`](docs/methodology.md).

---

## Model (his, faithfully)

- Deck = counts of `[1,2,3,4,5,6-drop, Signet, Land]` summing to 98, plus one
  Sol Ring → 99 cards. Commander = a free MV-N spell in the command zone, cast
  once.
- **Criterion:** at each turn end (turns 1–7) sum the mana worth of on-board
  non-rock, non-land permanents. k-drop = k, six-drop = **6.2**, commander = raw
  MV. Rocks and lands score 0. Averaged over random games.
- Multiplayer rules: **free first mulligan**, **always on the draw** (draw every
  turn incl. turn 1). London bottoming. His hand-crafted mulligan + gameplay
  heuristics, encoded verbatim (see the spec).
- Goldfish opponent; everything untaps every turn.

Full detail: `docs/superpowers/specs/2026-07-09-edh-mana-curve-sim-design.md`.

## Usage

```bash
uv sync
uv run pytest                       # 49 fast tests
uv run pytest -m slow               # 5 Monte Carlo / optimizer anchors (~3.5 min)

uv run python sweep.py --turns-min 2 --turns-max 12   # horizon sweep + brackets
uv run python overnight.py --hours 7                  # heavy, self-pacing, crash-safe
uv run python -c "import mulligan"                    # value-function mulligan DP

uv run python main.py validate      # criterion vs his 72.465 checkpoint
uv run python main.py run           # optimize all 5 commander MVs -> table
uv run python main.py run --mv 4    # one MV, verbose search trace
uv run python main.py run --star    # enable star-neighborhood polishing (slow)
```

## Results

Optimizer output vs Karsten's published table (`[1d 2d 3d 4d 5d 6d | Sig | Land]`,
all decks + 1 Sol Ring):

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
- **Numba engine:** removes his multi-day runtime; a 2M-game evaluation is ~2 s.

## Applying this to real decks

As in the article, this is an idealized curve model. For real decks: cut a land
per 2–3 mana rocks (or per 3–4 cantrips / mana dorks) but don't drop below ~37;
treat `Cultivate` as rock + 3-drop, `Llanowar Elves` as rock + 1-drop, an MDFC
land as half-land; run slightly fewer 1-drops if you have many tapped lands. And
don't chase the table exactly — aggro wants a lower curve, control a higher one,
and synergy beats raw curve.
