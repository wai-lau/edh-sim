# edh-sim — optimal Commander mana curves

A Monte Carlo simulator + local-search optimizer for **Magic: the Gathering
Commander** mana curves. It starts as a faithful reimplementation of Frank
Karsten's [*"What's an Optimal Mana Curve and Land/Ramp Count for Commander?"*
(ChannelFireball, 2025)](https://www.tcgplayer.com/content/article/What-s-an-Optimal-Mana-Curve-and-Land-Ramp-Count-for-Commander) — reproducing his published table from the article's prose
— then **extends past his goldfish** with the things that actually shape a
Commander deck: **interaction (board wipes), diminishing returns (a per-turn
development cap), card draw, a swept game-length horizon, and per-bracket curves.**

The whole per-game engine is `@njit`-compiled (Numba). Throughput ~1M games/s for
the bare Karsten model, ~0.2–0.5M/s for the full model (wipes + cap + draw).

---

## Results — mana curves by Commander bracket

Game length is the biggest lever on the curve, and it tracks deck power. We
optimize the deck at each horizon **T = 2–15**, then fold the per-horizon optima
into **per-bracket curves** via a normal weighting centered on each bracket's
characteristic game length (r/EDH midpoints: B4 ≈ 7, B3 ≈ 9, B2 ≈ 11 turns; **σ =
1.5**):

<img src="docs/bracket_weights.svg" alt="Bracket turn-weighting" width="460">

Format `[1d 2d 3d 4d 5d 6d | Draw | Signets | Lands]`, **+ 1 Sol Ring** in every
deck (99 cards). "Draw" = a card-draw spell (pay X, draw X); "Signets" = any
2-mana rock. Deep converged run: 16 restarts, 1.2M-game final showdown, **cap = 15**.

### Bracket 4 — Optimized (fast, ~7-turn games)
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Draw | Sig | Land |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:----:|:---:|:----:|
| 2 | 16 | 0 | 22 | 13 | 5 | 0 | 0 | 0 | 42 |
| 3 | 16 | 20 | 0 | 15 | 6 | 0 | 0 | 0 | 41 |
| 4 | 18 | 19 | 15 | 0 | 5 | 0 | 1 | 0 | 40 |
| 5 | 16 | 19 | 14 | 6 | 0 | 1 | 1 | 0 | 41 |
| 6 | 13 | 19 | 15 | 9 | 1 | 0 | 0 | 1 | 40 |

### Bracket 3 — Upgraded (mid, ~9-turn games)
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Draw | Sig | Land |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:----:|:---:|:----:|
| 2 | 11 | 0 | 20 | 14 | 8 | 2 | 3 | 0 | 40 |
| 3 | 11 | 17 | 0 | 15 | 8 | 3 | 4 | 0 | 40 |
| 4 | 12 | 16 | 13 | 0 | 9 | 3 | 5 | 1 | 39 |
| 5 | 12 | 15 | 12 | 9 | 0 | 3 | 4 | 2 | 41 |
| 6 | 10 | 13 | 13 | 11 | 4 | 0 | 2 | 4 | 41 |

### Bracket 2 — Core (slow, ~11-turn games)
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Draw | Sig | Land |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:----:|:---:|:----:|
| 2 | 7 | 0 | 17 | 13 | 8 | 5 | 10 | 0 | 38 |
| 3 | 7 | 12 | 0 | 13 | 8 | 6 | 10 | 1 | 41 |
| 4 | 7 | 12 | 11 | 0 | 9 | 6 | 11 | 2 | 40 |
| 5 | 8 | 12 | 10 | 10 | 0 | 6 | 10 | 4 | 38 |
| 6 | 7 | 8 | 11 | 11 | 7 | 0 | 8 | 7 | 39 |

*(Brackets 1 Exhibition and 5 cEDH — the extremes the model fits worst — are
omitted.)*

**Read across the brackets:**
- **Fast (B4):** a **cheap creature curve** (16–18 one-drops), ~0 ramp/draw.
- **Slow (B2):** 1-drops thin out and a **~8–11 card-draw** engine + a few signets
  appears — a resilient, low-ish deck that rebuilds to the cap from hand after
  each board wipe. Draw grows with game length (0 short → ~4 mid → ~10 long).
- Lands hold **~38–42** throughout.
- The `0` on the diagonal is **Karsten's Insight #2** (no drops at the commander's
  own mana value) — the free commander already fills that slot. It breaks only at
  MV 6: 6 is the model's ceiling (no 7+ drops), and you keep a *stock* of top-end
  to rebuild after wipes.
- **Draw cards are cheap cantrips:** cast at an average **X ≈ 2.3** (~55–59% at
  X = 1) — a 1-mana leftover-mana filter, with a thin tail of big post-wipe digs.

**Cap sensitivity — the biggest knob.** The per-turn development cap governs how
much draw and ramp you want: **lower cap → more draw + ramp** (harsh diminishing
returns make efficient post-wipe rebuild matter, so a draw/ramp engine pays off);
**higher cap → back toward a straight creature curve.** Slow-bracket draw runs
~14–17 at cap 12 but ~8–11 at cap 15.

---

## The model

### Criterion & deck
- **Deck** = `[1,2,3,4,5,6-drop, Signet, Land, Draw]` (the Draw slot is optional)
  summing to 98, plus one implicit **Sol Ring** → 99 cards. The **commander** is a
  free MV-N spell in the command zone (cast once; recastable after a wipe).
- **Criterion** = expected *compounded board mana*. At each turn end, sum the mana
  worth of on-board non-rock, non-land permanents, then **cap it** per turn
  (`min(·, cap)`). A k-drop is worth k; a **six-drop is 6.2** (Karsten's
  super-linear premium); the commander is scored the same (raw MV for 1–5, **6.2
  at MV 6**). Rocks, lands, and draw spells score 0. The horizon T (turns summed)
  is a parameter — Karsten's base is 7; we sweep **2–15**.
- Multiplayer rules: **free first mulligan**, **always on the draw**, London
  bottoming.

### The full model (beyond Karsten's goldfish)
- **Board wipes** (interaction): each turn ≥ 5, wipe chance `0.10 × 1.2^(wipe-free
  turns)`. A wipe kills creatures and sends the commander to the command zone, but
  **rocks, lands, and your hand survive**. This is what makes ramp and draw earn
  their keep — you rebuild from the hand that survived.
- **Development cap** (diminishing returns): the per-turn `min(board value, 15)`
  above. Over-committing past ~15 mana of board is wasted, so you **hold cards**
  (which then survive the next wipe).
- **Card draw:** a pay-X-draw-X spell, played *last*, only when your hand is below
  7 cards (or you're stuck with nothing else castable — a rare dig).
- **Optimizer:** *explore-cheap → select-precise* — many cheap local-search
  restarts explore, then the unique finalists get one high-sim showdown on a
  shared seed. This avoids the max-of-noisy-estimates bias of naive best-of-N.

### Chosen constants (magic numbers — not fit to data)
These are hand-picked and tunable, not derived:

| Constant | Value | Basis |
|---|---|---|
| score cap | **15** | chosen (tried 10 / 12 / 15) |
| wipe chance | **10% base, ×1.2/turn, from turn 5** | chosen |
| weighting σ | **1.5** | chosen (pointier = fewer tail games) |
| six-drop / MV-6 cmdr | **6.2** | Karsten's experience-based super-linear premium |
| bracket centers | **7 / 9 / 11** | r/EDH game-length midpoints |
| hand size | **7** | MTG rule |

---

## The mulligan (Karsten's open problem, solved)

Karsten flagged the optimal mulligan as future work. We derived it by
**value-function dynamic programming** (keep hand *h* iff its simulated value ≥ the
value of mulliganing again; solved by backward induction), then distilled it into
a fast rule:

> **Keep 3–4 lands. Mull the fifth (flood).** 2 lands only with a mana rock, 1
> only behind a Sol Ring. Fast decks also want ≥ 1 non-rock play.

This *tightened* the classic "keep 3–5 lands" (5-land hands flood), and on the
pre-wipe model it beat his heuristic at every horizon, shifting decks toward more
lands / fewer rocks (a smarter mulligan handles flood, so rocks aren't needed as
land-consistency insurance). The full model reuses the same distilled rule.
Full derivation: [`docs/methodology.md`](docs/methodology.md).

---

## Faithful replication (validation vs Karsten)

With the extensions off (goldfish, 7 turns, no cap/draw), the bare model
reproduces Karsten's published table (`[1d 2d 3d 4d 5d 6d | Sig | Land]` + Sol Ring):

| MV | ours | Karsten |
|----|------|---------|
| 2 | `9 0 21 13 9 4 \| 0 \| 42` | `9 0 20 14 9 4 \| 0 \| 42` |
| 3 | `8 18 0 16 10 4 \| 0 \| 42` | `8 19 0 16 10 3 \| 0 \| 42` |
| 4 | `7 18 16 0 10 5 \| 0 \| 42` | `6 12 13 0 13 8 \| 7 \| 39` |
| 5 | `7 15 13 11 0 8 \| 4 \| 40` | `6 12 10 13 0 10 \| 8 \| 39` |
| 6 | `6 16 13 12 7 0 \| 4 \| 40` | `6 12 10 14 9 0 \| 9 \| 38` |

**Reproduced:** Insight #2 (zero N-drops at the commander's MV — exactly, all
five); near-exact whole-deck match for cheap commanders (MV 2–3); high land counts
(Insight #4); bulk on 2s/3s/4s (#1); more ramp for pricier commanders, directionally
(#3). **Diverges:** absolute criterion ~0.6% low (his 4-MV deck 72.465 → ~71.98
here — his heuristics have unspecified slack; the *ordering* of his named
perturbation reproduces with the same sign), and ramp is undervalued for expensive
commanders at a 7-turn horizon — which is exactly the "noisiest, weakest" part he
flagged, and which the horizon/wipe extensions later address.

---

## Methods

- **Common random numbers (CRN):** within each optimizer iteration, every candidate
  deck is scored on the *same* batch of game seeds, so luck (esp. Sol Ring)
  cancels in the comparison — a variance-reduction fix for the noise Karsten
  flagged. Seeds refresh each iteration (no seed overfit).
- **Cumulative sim:** one game to the max horizon records the criterion at every
  turn, so a single pass yields *all* horizons — a horizon-T game is the
  horizon-(T+1) game truncated.
- **Numba engine:** removes his multi-day runtime — a 2M-game evaluation is ~2 s
  for the bare model, ~5–10 s for the full model.

---

## Usage

```bash
uv sync
make test                           # 59 fast tests
make lint                           # ruff + 500-line file cap
uv run pytest -m slow               # Monte Carlo / optimizer anchors (~3.5 min)

# FULL model (wipes + cap + draw), explore-cheap optimizer -> per-bracket curves:
uv run python run_final.py --seed-json <seed.json> --restarts 16 --final-sims 1200000

# FAITHFUL Karsten base (7-turn goldfish, no wipes/cap/draw):
uv run python main.py run           # optimize all 5 commander MVs -> table
uv run python main.py validate      # criterion vs his 72.465 checkpoint

# analysis: horizon sweep, self-pacing overnight runner, value-function mulligan DP:
uv run python sweep.py --turns-min 2 --turns-max 15
uv run python -c "import mulligan"
```

## Project layout

| File | Role |
|---|---|
| `cards.py` | card codes, seedable PRNG, library construction |
| `sim_core.py` | `@njit` engine: mulligan, gameplay, wipes, scoring, cumulative sim, draw-stats |
| `optimizer.py` | local search + CRN, explore-cheap→select-precise, joint sweep, neighborhoods |
| `run_final.py` | full-model per-bracket runner (per-cell crash-safe save) |
| `main.py` / `sweep.py` / `overnight.py` / `mulligan.py` | faithful CLI / horizon sweep / self-pacing runner / mulligan DP |
| `docs/` | methodology, design spec, bracket-weighting graph |

---

## Applying this to real decks

An idealized curve model, not gospel. In practice: cut a land per 2–3 mana rocks
(or per 3–4 cantrips / mana dorks) but don't go below ~37; treat `Cultivate` as
rock + 3-drop, `Llanowar Elves` as rock + 1-drop, an MDFC land as half-land; run
fewer 1-drops if you have many tapped lands. And don't chase the tables exactly —
aggro wants a lower curve, control a higher one, and real synergy beats raw curve.

## Caveats

The magic numbers above (cap, wipe rate, σ) are chosen, not fit to real game data.
Draw is modeled as a one-shot cantrip, not an ongoing engine (Rhystic Study), so
its *selection/consistency* value is under-counted. No colors, tapped lands, mana
dorks, tutors, or combos. cEDH (turn-3 wins) is out of scope. The full-model optima
sit on a flat-ish ridge — deeper search can shift ±1s and occasionally jump basins.
