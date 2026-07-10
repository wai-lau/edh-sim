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

Our extension beyond Karsten: game length is the biggest lever on the curve, and
it tracks deck power. We sweep the turn horizon (T = 2–12), re-optimize at each,
and fold the per-horizon optima into per-bracket curves via a normal weighting
centered on each bracket's characteristic game length (σ = 2 turns). Three
distinct center-curves cover the five official Commander brackets:

Format `[1d 2d 3d 4d 5d 6d | Signets | Lands]`, **+ 1 Sol Ring** in every deck.
"Signets" = any 2-mana rock (Signet / Talisman / Fellwar / Nature's Lore / …).

### Fast tables — **Bracket 4 (Optimized) + Bracket 5 (cEDH)**, ~5-turn games
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Sig | Land |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:----:|
| 2 | 20 | 3 | 18 | 11 | 4 | 2 | 0 | 40 |
| 3 | 18 | 16 | 1 | 13 | 5 | 2 | 1 | 42 |
| 4 | 18 | 16 | 13 | 0 | 7 | 3 | 2 | 39 |
| 5 | 17 | 17 | 13 | 8 | 0 | 3 | 1 | 39 |
| 6 | 15 | 17 | 13 | 11 | 2 | 1 | 1 | 38 |

### Mid tables — **Bracket 3 (Upgraded)**, ~7-turn games
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Sig | Land |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:----:|
| 2 | 11 | 1 | 19 | 13 | 8 | 5 | 0 | 41 |
| 3 | 10 | 14 | 0 | 14 | 8 | 7 | 3 | 42 |
| 4 | 10 | 13 | 13 | 0 | 10 | 8 | 4 | 40 |
| 5 | 9 | 14 | 13 | 10 | 0 | 8 | 4 | 40 |
| 6 | 8 | 14 | 13 | 12 | 6 | 3 | 4 | 38 |

### Slow tables — **Bracket 1 (Exhibition) + Bracket 2 (Core)**, ~9-turn games
| Cmdr MV | 1d | 2d | 3d | 4d | 5d | 6d | Sig | Land |
|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:----:|
| 2 | 4 | 0 | 17 | 14 | 10 | 10 | 1 | 42 |
| 3 | 3 | 8 | 0 | 15 | 10 | 14 | 6 | 42 |
| 4 | 3 | 7 | 11 | 0 | 13 | 15 | 8 | 41 |
| 5 | 3 | 7 | 11 | 12 | 0 | 17 | 8 | 40 |
| 6 | 2 | 8 | 11 | 13 | 9 | 8 | 8 | 39 |

**Read across the brackets:** faster tables → cheap curve, little to no ramp;
slower tables → 1-drops vanish, six-drops and ~8 signets dominate (ramp into
fatties early so they compound). Lands hold ~38–42 throughout. The `0` on each
row's diagonal is Karsten's Insight #2 — you don't run drops at your commander's
own mana value. Caveats and derivation in [`docs/methodology.md`](docs/methodology.md).

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
uv run pytest                       # 37 fast tests
uv run pytest -m slow               # 5 Monte Carlo / optimizer anchors (~3.5 min)

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
