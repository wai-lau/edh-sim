#!/usr/bin/env python
"""Generate docs/bracket_weights.svg: the normal turn-weighting for the three
brackets (one chart, three bell curves) used to fold per-horizon optima into
per-bracket curves."""
import math
import pathlib

TURNS = list(range(2, 16))
SIGMA = 2.0
BRACKETS = [                       # (center, label, colour)
    (7, "B4 Optimized (fast)", "#2563eb"),
    (9, "B3 Upgraded (mid)", "#d97706"),
    (11, "B2 Core (slow)", "#dc2626"),
]

W, H = 680, 430
L, R, T, B = 62, 210, 46, 52       # margins (R leaves room for the legend)
PW, PH = W - L - R, H - T - B


def weights(mu):
    w = {t: math.exp(-((t - mu) ** 2) / (2 * SIGMA ** 2)) for t in TURNS}
    z = sum(w.values())
    return {t: w[t] / z for t in TURNS}


ymax = max(max(weights(mu).values()) for mu, _, _ in BRACKETS)
ymax = math.ceil(ymax * 20) / 20   # round up to a 0.05 gridline


def px(t):
    return L + (t - TURNS[0]) / (TURNS[-1] - TURNS[0]) * PW


def py(v):
    return T + PH - (v / ymax) * PH


s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
     f'font-family="system-ui,sans-serif" font-size="13">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
s.append(f'<text x="{L}" y="26" font-size="16" font-weight="600">'
         f'Turn weighting by bracket (normal, σ={SIGMA:g})</text>')

# y gridlines + labels
yt = 0.0
while yt <= ymax + 1e-9:
    y = py(yt)
    s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+PW}" y2="{y:.1f}" '
             f'stroke="#e5e7eb"/>')
    s.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" fill="#6b7280">'
             f'{yt*100:.0f}%</text>')
    yt += 0.05
# x ticks + labels
for t in TURNS:
    if t % 2 == 0:
        s.append(f'<text x="{px(t):.1f}" y="{T+PH+20}" text-anchor="middle" '
                 f'fill="#6b7280">{t}</text>')
s.append(f'<text x="{L+PW/2:.0f}" y="{H-10}" text-anchor="middle" '
         f'fill="#374151">game length (turn)</text>')

# curves + legend
for i, (mu, label, col) in enumerate(BRACKETS):
    w = weights(mu)
    pts = " ".join(f"{px(t):.1f},{py(w[t]):.1f}" for t in TURNS)
    s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
             f'stroke-width="2.5"/>')
    ly = T + 6 + i * 22
    s.append(f'<line x1="{L+PW+18}" y1="{ly}" x2="{L+PW+40}" y2="{ly}" '
             f'stroke="{col}" stroke-width="2.5"/>')
    s.append(f'<text x="{L+PW+46}" y="{ly+4}" fill="#374151">'
             f'{label} μ={mu}</text>')
s.append('</svg>')

out = pathlib.Path("docs/bracket_weights.svg")
out.write_text("\n".join(s), encoding="utf-8")
print(f"wrote {out}")
