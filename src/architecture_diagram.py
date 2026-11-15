"""Architecture figure for the VQ-VAE action-masking agent, rendered twice from ONE scene description.

  plots/architecture/architecture.png / .pdf   (matplotlib: 300 dpi raster + vector, for the paper)
  plots/architecture/architecture.drawio       (diagrams.net file with the same elements, for hand editing)

Layout (two stacked halves + a loss band):
  I   Discretisation into events (encoder only, no decoder):
        observation -> CNN -> z_e tensor -> codebook 1 (nearest embedding per cell) -> code-id grid M
        -> embed & pool -> g_t -> codebook 2 (nearest of N prototype grids) -> event e_t
  II  Event-transition head and agent (the loop runs clockwise):
        e_t -> one-layer perceptron W (N x N, a weight vector per event) -> softmax over the next events -> the score at
        the observed e_{t+1} is the penalty -> shaped reward -> DQN agent -> a_t -> environment -> x_{t+1} back to I
        (the action never enters the head: the Q-network credits the shaped reward to the action it took)
  L   Total loss = task term (replaces reconstruction) + codebook-1 terms + codebook-2 terms

Every drawn element is a primitive in SCENE (rect, circle, line, arrow, text, image, cuboid, tag); both renderers
walk the same list, so the two outputs cannot drift apart.  Coordinates are layout units with y UP; draw.io gets
PX pixels per unit with y flipped.

Run from the repository root:  python src/architecture_diagram.py
"""

import base64
import os
import time
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import numpy as np

OUT_DIR = Path("plots/architecture")
FRAME_PNG = OUT_DIR / "frame_rgb.png"
W, H, PX = 176, 118, 8                        # layout units (y up); draw.io pixels per unit
FIG_WIDTH_IN = 7.4

# ── palette ───────────────────────────────────────────────────────────────────
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"
ARROW, GRAD = "#4f86d8", "#e34948"                                       # flow arrows; gradient arrows (as in the paper)
P1_BG, P1_EDGE = "#f3f7fd", "#b9cff0"                                    # half I background / outline (cool)
P2_BG, P2_EDGE = "#fdf7f2", "#f0c7ad"                                    # half II background / outline (warm)
BAND_BG, BAND_EDGE = "#ffffff", "#c9c8c3"
GREEN_FILL, GREEN_SIDE, GREEN_LINE, GREEN_DARK = "#dcefd9", "#c9e5c6", "#8cc98a", "#5aa457"    # z_e tensor
VIOLET_FILL, VIOLET_LINE, VIOLET_DARK = "#e4def6", "#a99bdf", "#6250d6"                      # codebooks, events
BLUE_FILL, BLUE_LINE, BLUE_DARK = "#dbe9f8", "#9ec5f4", "#2a78d6"                            # code-id grid
ORANGE_FILL, ORANGE_LINE, ORANGE_DARK = "#fdeee7", "#f5b89a", "#eb6834"                      # head, penalty, loss
AQUA_FILL, AQUA_DARK = "#e3f6ee", "#1baf7a"                                                  # agent, environment, actions
WIRE, WIRE_ACTIVE_E, WIRE_ACTIVE_A = "#d9d9d6", VIOLET_DARK, AQUA_DARK
TAG_COLOUR = {1: ORANGE_DARK, 2: BLUE_DARK, 3: VIOLET_DARK}
CODE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e8b931", "#e34948", "#9ec5f4", "#f5b89a", "#8fdcbd", "#b9b0ea"]
MATRIX = [[1, 8, 2], [4, 9, 10], [21, 40, 0]]                            # from the whiteboard
SOFTMAX = [0.04, 0.10, 0.62, 0.07, 0.17]                                 # illustrative p(e_{t+1} | e_t, a_t)

# ══════════════════════════════════════════════════════════════════════════════
# scene primitives (shared by both renderers)
# ══════════════════════════════════════════════════════════════════════════════
SCENE = []


def rect(x, y, w, h, fill, stroke, lw=0.8, r=0.0, dashed=False, z=2):
    SCENE.append(dict(kind="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, lw=lw, r=r, dashed=dashed, z=z))


def circle(cx, cy, r, fill, stroke="none", lw=0.6, z=3):
    SCENE.append(dict(kind="circle", cx=cx, cy=cy, r=r, fill=fill, stroke=stroke, lw=lw, z=z))


def line(pts, stroke, lw=0.6, dashed=False, z=2.5, alpha=1.0):
    SCENE.append(dict(kind="line", pts=pts, stroke=stroke, lw=lw, dashed=dashed, z=z, alpha=alpha))


def arrow(pts, stroke=ARROW, lw=0.9, dashed=False, curve=None, z=2.8):
    """pts: polyline end-to-end.  curve=(a, ctrl, b) draws one quadratic Bezier instead."""
    SCENE.append(dict(kind="arrow", pts=pts, stroke=stroke, lw=lw, dashed=dashed, curve=curve, z=z))


def text(x, y, mpl, html=None, size=6.5, colour=INK, ha="center", va="center", bold=False, z=4, w=None):
    """mpl: matplotlib string (mathtext allowed).  html: draw.io label; defaults to mpl with $ removed."""
    SCENE.append(dict(kind="text", x=x, y=y, mpl=mpl, html=html if html is not None else mpl.replace("$", ""),
                      size=size, colour=colour, ha=ha, va=va, bold=bold, z=z, w=w))


def image(x, y, w, h):
    SCENE.append(dict(kind="image", x=x, y=y, w=w, h=h, z=3))


def cuboid(x, y, w, h, d, grid=5):
    SCENE.append(dict(kind="cuboid", x=x, y=y, w=w, h=h, d=d, grid=grid, z=3))


def tag(x, y, n, r=1.25):
    """Numbered circle (1, 2, 3) marking the three loss terms."""
    circle(x, y, r, TAG_COLOUR[n], z=4.5)
    text(x, y, str(n), size=4.6, colour="white", bold=True, z=4.6)


def brace(x0, x1, y, d=1.5):
    SCENE.append(dict(kind="brace", x0=x0, x1=x1, y=y, d=d, z=3))


def mini_grid(x, y, cell, colours, stroke="white"):
    """3 x 3 grid of coloured cells with bottom-left corner (x, y)."""
    for r in range(3):
        for c in range(3):
            rect(x + c * cell, y + (2 - r) * cell, cell, cell, colours[(r * 3 + c) % len(colours)], stroke, lw=0.35, z=3.6)


# ══════════════════════════════════════════════════════════════════════════════
# the figure
# ══════════════════════════════════════════════════════════════════════════════
def build_scene():
    rng = np.random.default_rng(7)

    # ── half I: discretisation ────────────────────────────────────────────────
    rect(3, 64, 170, 52, P1_BG, P1_EDGE, lw=0.9, r=2.5, z=1)
    text(5.5, 114.7, r"I   Discretisation into events   (encoder only, no decoder: the reconstruction term is replaced by the task term of $\mathcal{L}$)",
         "I   Discretisation into events   (encoder only, no decoder: the reconstruction term is replaced by the task term of 𝓛)",
         size=7.4, ha="left", va="top", bold=True)

    # observation
    image(6, 80, 11, 14.44)
    text(11.5, 95.6, r"observation $x_t$", "observation <i>x</i><sub>t</sub>", size=6.2, va="bottom")
    arrow([(17.3, 87.2), (24.7, 87.2)])
    text(21, 88.6, r"CNN $E_\phi$", "CNN <i>E</i><sub>φ</sub>", size=5.4, colour=INK_SOFT, va="bottom")

    # z_e tensor
    cuboid(25, 81, 12, 12, 3)
    text(31.5, 79.4, r"$z_e(x_t)\in\mathbb{R}^{D\times G\times G}$", "<i>z</i><sub>e</sub>(<i>x</i><sub>t</sub>) ∈ ℝ<sup>D×G×G</sup>", size=5.8, va="top")
    text(24.3, 83, r"$G$", "<i>G</i>", size=5.2, colour=INK_SOFT, ha="right")
    text(38.5, 95.7, r"$D$", "<i>D</i>", size=5.2, colour=INK_SOFT, va="bottom")
    arrow([(40.3, 87.5), (43.4, 87.1)])

    # codebook 1 and the first lookup
    rect(28, 98, 36, 12, "white", VIOLET_LINE, lw=0.9, r=1.6, z=3)
    text(46, 108.6, r"Codebook 1 (cell level):  $e_1\ldots e_K\in\mathbb{R}^{D}$",
         "Codebook 1 (cell level):  <i>e</i><sub>1</sub> … <i>e</i><sub>K</sub> ∈ ℝ<sup>D</sup>", size=5.8, va="top")
    col_w, gap, col_h, col_y = 1.6, 0.5, 6.3, 99.6
    for i in range(6):
        rect(31 + i * (col_w + gap), col_y, col_w, col_h, VIOLET_FILL, VIOLET_LINE, lw=0.5, z=3.5)
    for i in range(3):
        rect(61 - (3 - i) * (col_w + gap) + gap, col_y, col_w, col_h, VIOLET_FILL, VIOLET_LINE, lw=0.5, z=3.5)
    for k in (-1, 0, 1):
        circle(49.3 + k * 1.6, col_y + col_h / 2, 0.45, VIOLET_LINE, z=3.5)
    circle(45, 87, 1.5, ARROW, z=4)
    arrow(None, curve=((46, 98), (49.5, 94), (45.7, 88.6)))
    text(47, 84.6, r"$\arg\min_k\|z_e-e_k\|$", "argmin<sub>k</sub> ‖<i>z</i><sub>e</sub> − <i>e</i><sub>k</sub>‖", size=4.8, colour=INK_SOFT, va="top")
    arrow(None, stroke=GRAD, lw=0.8, curve=((44.2, 89.1), (42.4, 91.2), (40.6, 89.8)))
    text(42.4, 91.4, r"$\nabla_z\mathcal{L}$", "∇<sub>z</sub>𝓛", size=4.8, colour=GRAD, va="bottom")
    arrow([(46.6, 87), (52.7, 87)])

    # code-id grid M
    cell = 4.2
    x0, y0 = 53.7, 80.9
    for r_, row in enumerate(MATRIX):
        for c_, v in enumerate(row):
            rect(x0 + c_ * cell, y0 + (2 - r_) * cell, cell, cell, BLUE_FILL, BLUE_LINE, lw=0.6, z=3)
            text(x0 + (c_ + 0.5) * cell, y0 + (2.5 - r_) * cell, str(v), size=6.4, z=4)
    text(60, 79.4, r"code-id grid $M\in\{1..K\}^{G\times G}$", "code-id grid <i>M</i> ∈ {1..K}<sup>G×G</sup>", size=5.6, va="top")
    arrow([(66.6, 87), (75.6, 87)])
    text(71.1, 88.5, r"pool($z_q$)", "pool(<i>z</i><sub>q</sub>)", size=5.0, colour=INK_SOFT, va="bottom")

    # g_t vector and codebook 2
    shades = ["#cfc6f0", "#b9b0ea", "#a99bdf", "#8f80d8", "#7a69d4", "#6250d6"]
    for i in range(6):
        rect(76, 81 + i * 2.0, 3.2, 2.0, shades[i], "white", lw=0.35, z=3.5)
    rect(76, 81, 3.2, 12, "none", VIOLET_DARK, lw=0.6, z=3.6)
    text(77.6, 79.4, r"$g_t\in\mathbb{R}^{D'}$", "<i>g</i><sub>t</sub> ∈ ℝ<sup>D′</sup>", size=5.6, va="top")
    arrow([(79.5, 87.2), (86.4, 87.1)])
    circle(88, 87, 1.5, ARROW, z=4)
    text(88.5, 84.6, r"$\arg\min_n\|g_t-c_n\|$", "argmin<sub>n</sub> ‖<i>g</i><sub>t</sub> − <i>c</i><sub>n</sub>‖", size=5.0, colour=INK_SOFT, va="top")
    arrow(None, stroke=GRAD, lw=0.8, curve=((87.2, 89.1), (83.6, 91.3), (80.1, 89.8)))

    rect(72, 97, 50, 14, "white", VIOLET_LINE, lw=0.9, r=1.6, z=3)
    text(97, 109.6, r"Codebook 2 (grid level):  $N$ prototype grids $c_1\ldots c_N\in\mathbb{R}^{D'}$",
         "Codebook 2 (grid level):  <i>N</i> prototype grids <i>c</i><sub>1</sub> … <i>c</i><sub>N</sub> ∈ ℝ<sup>D′</sup>", size=5.5, va="top")
    proto_x = [75.5 + i * 5.4 for i in range(6)] + [112.5]
    for i, gx in enumerate(proto_x):
        mini_grid(gx, 101.4, 1.1, [CODE_COLOURS[j] for j in rng.permutation(len(CODE_COLOURS))])
    for k in (-1, 0, 1):
        circle(108.7 + k * 1.3, 103.05, 0.4, VIOLET_LINE, z=3.5)
    text(77.15, 100.6, r"$c_1$", "<i>c</i><sub>1</sub>", size=4.6, colour=INK_SOFT, va="top")
    text(114.15, 100.6, r"$c_N$", "<i>c</i><sub>N</sub>", size=4.6, colour=INK_SOFT, va="top")
    text(124.5, 104, "$N$ = size of codebook 2\n= bound on the number of\ndistinct discrete states (events)",
         "<i>N</i> = size of codebook 2<br>= bound on the number of<br>distinct discrete states (events)", size=5.2, colour=INK_SOFT, ha="left")
    arrow(None, curve=((92, 97), (94.5, 92), (88.7, 88.6)))
    arrow([(89.6, 87), (108.2, 87)])
    text(98.9, 88.5, "nearest prototype", size=5.0, colour=INK_SOFT, va="bottom")

    # event token
    rect(108.5, 81, 22, 12, VIOLET_FILL, VIOLET_DARK, lw=0.9, r=1.4, z=3)
    text(119.5, 90.3, r"event $e_t\in\{1,\ldots,N\}$", "event <i>e</i><sub>t</sub> ∈ {1, …, N}", size=6.2, z=4)
    mini_grid(117.85, 83.7, 1.1, [CODE_COLOURS[j] for j in rng.permutation(len(CODE_COLOURS))])
    text(119.5, 82.3, "the discrete state", size=4.8, colour=INK_SOFT, z=4)

    # braces under half I (labels double as the loss tags)
    brace(20, 41, 75.3)
    text(30.5, 72.7, r"Encoder $E_\phi$", "Encoder <i>E</i><sub>φ</sub>", size=5.8, va="top")
    brace(42.5, 70, 75.3)
    tag(45.3, 71.6, 2)
    text(47.3, 71.6, "Codebook 1 loss (cell level)", size=5.6, ha="left")
    brace(71.5, 112, 75.3)
    tag(79.6, 71.6, 3)
    text(81.6, 71.6, "Codebook 2 loss (grid level)", size=5.6, ha="left")

    # ── half II: head and agent ───────────────────────────────────────────────
    rect(3, 16, 170, 46, P2_BG, P2_EDGE, lw=0.9, r=2.5, z=1)
    text(5.5, 59.6, "II   Event-transition head and agent", size=7.4, ha="left", va="top", bold=True)
    text(52, 59.3, r"loop (clockwise):  $x_t\to e_t\to\rho_t\to\tilde{r}_t\to a_t\to x_{t+1}$",
         "loop (clockwise):  <i>x</i><sub>t</sub> → <i>e</i><sub>t</sub> → ρ<sub>t</sub> → <i>r̃</i><sub>t</sub> → <i>a</i><sub>t</sub> → <i>x</i><sub>t+1</sub>",
         size=5.4, colour=INK_SOFT, ha="left", va="top")

    def box(x, y, w, h, fill, edge, label, label_html, sub, sub_html):
        rect(x, y, w, h, fill, edge, lw=0.9, r=1.4, z=3)
        if sub:
            text(x + w / 2, y + h * 0.63, label, label_html, size=6.4, z=4)
            text(x + w / 2, y + h * 0.26, sub, sub_html, size=5.2, colour=INK_SOFT, z=4)
        else:
            text(x + w / 2, y + h / 2, label, label_html, size=6.4, z=4)

    box(6, 28, 16, 12, AQUA_FILL, AQUA_DARK, "Demon Attack", "Demon Attack", "ALE, 18 actions", "ALE, 18 actions")
    box(27, 28, 22, 12, AQUA_FILL, AQUA_DARK, r"DQN agent $Q_\psi(x_t,\cdot)$", "DQN agent <i>Q</i><sub>ψ</sub>(<i>x</i><sub>t</sub>, ·)",
        "ε-greedy, replay, target net", "ε-greedy, replay, target net")
    box(54, 28, 16, 12, ORANGE_FILL, ORANGE_DARK, r"$\tilde{r}_t = r_t-\lambda\,\rho_t$", "<i>r̃</i><sub>t</sub> = <i>r</i><sub>t</sub> − λ ρ<sub>t</sub>",
        "shaped reward", "shaped reward")
    box(72.5, 28, 18.5, 12, ORANGE_FILL, ORANGE_DARK, r"penalty $\rho_t$", "penalty ρ<sub>t</sub>",
        r"$= p(e_{t+1}\mid e_t)$, the score", "= p(<i>e</i><sub>t+1</sub> | <i>e</i><sub>t</sub>), the score")
    arrow([(72.2, 34), (70.3, 34)])
    arrow([(53.7, 34), (49.3, 34)])
    arrow([(26.7, 34), (22.3, 34)])
    text(24.5, 35.2, r"$a_t$", "<i>a</i><sub>t</sub>", size=5.4, colour=INK_SOFT, va="bottom")
    arrow([(11, 40.3), (11, 79.7)])
    text(12.3, 50, r"$x_{t+1}$", "<i>x</i><sub>t+1</sub>", size=5.4, colour=INK_SOFT, ha="left")
    tag(14.8, 23.3, 1)
    text(17.2, 23.3, r"task loss, term 1 of $\mathcal{L}$:  maximise the number of kills," + "\n" + r"then minimise the index $i_{\mathrm{last}}$ of the last kill",
         "task loss, term 1 of 𝓛: maximise the number of kills,<br>then minimise the index <i>i</i><sub>last</sub> of the last kill", size=5.3, colour=INK_SOFT, ha="left")

    # the MLP: one-hot event in, softmax over the N next events out; the action never enters the head
    rect(95, 18, 57, 41, "white", ORANGE_LINE, lw=0.9, r=1.6, z=2.2)
    e_x = o_x = [112, 115.75, 119.5, 123.25, 130.75]
    y_in, y_out, y_base = 50, 37, 27.5
    active_e = 119.5
    for ex in e_x:                                      # wires first (below the nodes)
        for ox in o_x:
            on = ex == active_e
            line([(ex, y_in - 1.2), (ox, y_out + 1.2)], WIRE_ACTIVE_E if on else WIRE, lw=0.7 if on else 0.35, z=2.4)
    for ex in e_x:
        circle(ex, y_in, 1.2, VIOLET_DARK if ex == active_e else "white", VIOLET_DARK, lw=0.7, z=3.5)
    for k in (-1, 0, 1):
        circle(127 + k * 0.9, y_in, 0.28, VIOLET_DARK, z=3.5)
    for ox in o_x:
        circle(ox, y_out, 1.2, "white", ORANGE_DARK, lw=0.7, z=3.5)
    for k in (-1, 0, 1):
        circle(127 + k * 0.9, y_out, 0.28, ORANGE_DARK, z=3.5)
    text(127.5, 53.3, r"current event, one-hot over the $N$ events", "current event, one-hot over the <i>N</i> events", size=5.2, colour=INK_SOFT)
    text(110.3, y_in, r"$e_1$", "<i>e</i><sub>1</sub>", size=4.8, colour=INK_SOFT, ha="right")
    text(132.6, y_in, r"$e_N$", "<i>e</i><sub>N</sub>", size=4.8, colour=INK_SOFT, ha="left")
    text(110.3, y_out, r"next event $e_{t+1}$", "next event <i>e</i><sub>t+1</sub>", size=4.8, colour=INK_SOFT, ha="right")
    line([(109.5, y_base), (133.5, y_base)], INK_MUTED, lw=0.5, z=3)
    best = int(np.argmax(SOFTMAX))
    for i, (ox, p) in enumerate(zip(o_x, SOFTMAX)):
        rect(ox - 1.2, y_base, 2.4, 7.0 * p, ORANGE_DARK if i == best else ORANGE_FILL, ORANGE_DARK, lw=0.5, z=3.4)
    text(121.5, 25.2, r"softmax:  $p(e_{t+1}\mid e_t)$ over the $N$ events;  the bar at the observed $e_{t+1}$ is the penalty",
         "softmax:  p(<i>e</i><sub>t+1</sub> | <i>e</i><sub>t</sub>) over the <i>N</i> events;  the bar at the observed <i>e</i><sub>t+1</sub> is the penalty", size=5.0, colour=INK_SOFT)
    text(123.5, 21.3, r"one-layer perceptron $W\in\mathbb{R}^{N\times N}$:  one weight vector per event",
         "one-layer perceptron <i>W</i> ∈ ℝ<sup>N×N</sup>:  one weight vector per event", size=5.4)
    text(136.5, 44.5, r"filled node =" + "\n" + r"current $e_t$", "filled node =<br>current <i>e</i><sub>t</sub>", size=4.6, colour=INK_SOFT, ha="left")
    arrow([(119.5, 80.7), (119.5, 51.5)])                    # the event enters the head
    text(121, 66, r"$e_t$", "<i>e</i><sub>t</sub>", size=5.4, colour=INK_SOFT, ha="left")
    arrow([(118.2, 30), (91.3, 33.6)])                       # the score leaves as the penalty

    # ── loss band ────────────────────────────────────────────────────────────
    rect(3, 1, 170, 13, BAND_BG, BAND_EDGE, lw=0.8, r=2, z=1)
    text(5.5, 11.3, r"Total loss  $\mathcal{L}\;=$", "Total loss  𝓛 =", size=6.4, ha="left", bold=True)
    tag(24.6, 11.3, 1)
    text(26.9, 11.3, r"$\min_\pi\; i_{\mathrm{last}}\left[\max_\pi\,\sum_t \mathbb{1}(r_t\!=\!10)\right]$",
         "min<sub>π</sub> <i>i</i><sub>last</sub> [ max<sub>π</sub> Σ<sub>t</sub> 𝟙(<i>r</i><sub>t</sub> = 10) ]", size=6.2, ha="left")
    text(62.5, 11.3, "+", size=6.4)
    tag(66.2, 11.3, 2)
    text(68.5, 11.3, r"$\|\mathrm{sg}[z_e]-e\|^2+\beta\,\|z_e-\mathrm{sg}[e]\|^2$",
         "‖sg[<i>z</i><sub>e</sub>] − <i>e</i>‖² + β ‖<i>z</i><sub>e</sub> − sg[<i>e</i>]‖²", size=6.2, ha="left")
    text(104.5, 11.3, "+", size=6.4)
    tag(108.2, 11.3, 3)
    text(110.5, 11.3, r"$\|\mathrm{sg}[g_t]-c\|^2+\beta\,\|g_t-\mathrm{sg}[c]\|^2+\gamma\,\mathcal{L}_{\mathrm{use}}$",
         "‖sg[<i>g</i><sub>t</sub>] − <i>c</i>‖² + β ‖<i>g</i><sub>t</sub> − sg[<i>c</i>]‖² + γ 𝓛<sub>use</sub>", size=6.2, ha="left")
    text(26.9, 7.3, r"task (replaces reconstruction): max kills, then min $i_{\mathrm{last}}$",
         "task (replaces reconstruction): max kills, then min <i>i</i><sub>last</sub>", size=5.1, colour=INK_SOFT, ha="left")
    text(68.5, 7.3, "codebook 1 (cell level): unchanged", size=5.1, colour=INK_SOFT, ha="left")
    text(110.5, 7.3, r"codebook 2 (grid level): + usage of the $N$ prototypes",
         "codebook 2 (grid level): + usage of the <i>N</i> prototypes", size=5.1, colour=INK_SOFT, ha="left")
    text(5.5, 3.5, r"sg = stop-gradient;  $e$, $c$ = the selected entries of codebooks 1 and 2;  $\beta$ = commitment weight;  "
                   r"$\gamma\,\mathcal{L}_{\mathrm{use}}$ keeps all $N$ prototypes in use;  $i_{\mathrm{last}}$ = time index of the last kill (+10);  "
                   r"gradients reach $E_\phi$ through the straight-through estimator ($\nabla_z\mathcal{L}$).",
         "sg = stop-gradient;  <i>e</i>, <i>c</i> = the selected entries of codebooks 1 and 2;  β = commitment weight;  γ 𝓛<sub>use</sub> keeps all <i>N</i> prototypes in use;  "
         "<i>i</i><sub>last</sub> = time index of the last kill (+10);  gradients reach <i>E</i><sub>φ</sub> through the straight-through estimator (∇<sub>z</sub>𝓛).",
         size=4.8, colour=INK_MUTED, ha="left")


# ══════════════════════════════════════════════════════════════════════════════
# matplotlib renderer
# ══════════════════════════════════════════════════════════════════════════════
def mpl_render(frame):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * H / W))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    dash = (0, (3, 2))
    for p in sorted(SCENE, key=lambda p: p["z"]):
        k = p["kind"]
        if k == "rect":
            if p["r"]:
                ax.add_patch(mpatches.FancyBboxPatch((p["x"], p["y"]), p["w"], p["h"], boxstyle=f"round,pad=0,rounding_size={p['r']}",
                                                     fc=p["fill"], ec=p["stroke"], lw=p["lw"], ls=dash if p["dashed"] else "-", zorder=p["z"]))
            else:
                ax.add_patch(mpatches.Rectangle((p["x"], p["y"]), p["w"], p["h"], fc=p["fill"], ec=p["stroke"], lw=p["lw"],
                                                ls=dash if p["dashed"] else "-", zorder=p["z"]))
        elif k == "circle":
            ax.add_patch(mpatches.Circle((p["cx"], p["cy"]), p["r"], fc=p["fill"], ec=p["stroke"], lw=p["lw"], zorder=p["z"]))
        elif k == "line":
            xs, ys = zip(*p["pts"])
            ax.add_line(Line2D(xs, ys, color=p["stroke"], lw=p["lw"], ls=dash if p["dashed"] else "-", zorder=p["z"], alpha=p["alpha"], solid_capstyle="round"))
        elif k == "arrow":
            style = dict(color=p["stroke"], lw=p["lw"], ls=dash if p["dashed"] else "-", zorder=p["z"])
            if p["curve"]:
                a, c, b = p["curve"]
                path = MplPath([a, c, b], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
                ax.add_patch(mpatches.FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=7, fc=p["stroke"], **style))
            else:
                pts = p["pts"]
                for q0, q1 in zip(pts[:-2], pts[1:-1]):
                    ax.add_line(Line2D([q0[0], q1[0]], [q0[1], q1[1]], solid_capstyle="round", **style))
                ax.add_patch(mpatches.FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, shrinkA=0, shrinkB=0, fc=p["stroke"], **style))
        elif k == "text":
            ax.text(p["x"], p["y"], p["mpl"], fontsize=p["size"], color=p["colour"], ha=p["ha"], va=p["va"],
                    fontweight="bold" if p["bold"] else "normal", zorder=p["z"], linespacing=1.15)
        elif k == "image":
            ax.imshow(frame, extent=[p["x"], p["x"] + p["w"], p["y"], p["y"] + p["h"]], zorder=p["z"], interpolation="nearest")
            ax.add_patch(mpatches.Rectangle((p["x"], p["y"]), p["w"], p["h"], fc="none", ec=INK_SOFT, lw=0.6, zorder=p["z"] + 0.1))
        elif k == "cuboid":
            x, y, w, h, d, g = p["x"], p["y"], p["w"], p["h"], p["d"], p["grid"]
            ax.add_patch(mpatches.Polygon([(x, y + h), (x + d, y + h + d * 0.8), (x + w + d, y + h + d * 0.8), (x + w, y + h)], closed=True,
                                          fc=GREEN_FILL, ec=GREEN_LINE, lw=0.6, zorder=p["z"]))
            ax.add_patch(mpatches.Polygon([(x + w, y), (x + w + d, y + d * 0.8), (x + w + d, y + h + d * 0.8), (x + w, y + h)], closed=True,
                                          fc=GREEN_SIDE, ec=GREEN_LINE, lw=0.6, zorder=p["z"]))
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fc=GREEN_FILL, ec=GREEN_LINE, lw=0.7, zorder=p["z"] + 0.5))
            for i in range(1, g):
                ax.add_line(Line2D([x + i * w / g] * 2, [y, y + h], color=GREEN_LINE, lw=0.45, zorder=p["z"] + 0.6))
                ax.add_line(Line2D([x, x + w], [y + i * h / g] * 2, color=GREEN_LINE, lw=0.45, zorder=p["z"] + 0.6))
                ax.add_line(Line2D([x + i * w / g, x + i * w / g + d], [y + h, y + h + d * 0.8], color=GREEN_LINE, lw=0.35, zorder=p["z"] + 0.6))
        elif k == "brace":
            x0, x1, y, d = p["x0"], p["x1"], p["y"], p["d"]
            xm = (x0 + x1) / 2
            verts = [(x0, y + d), (x0, y), (x0 + d, y), (xm - d, y), (xm, y), (xm, y - d), (xm, y), (xm + d, y), (x1 - d, y), (x1, y), (x1, y + d)]
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
                     MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
            ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), fc="none", ec=ARROW, lw=0.8, zorder=p["z"]))
    fig.subplots_adjust(0, 0, 1, 1)
    for ext in ("png", "pdf"):
        save_with_retry(fig, OUT_DIR / f"architecture.{ext}", dpi=300)
    plt.close(fig)


def save_with_retry(fig, path, attempts=6, **kwargs):
    """OneDrive and viewers briefly lock files: write to a temporary name, then swap it in, retrying."""
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    for attempt in range(attempts):
        try:
            fig.savefig(tmp, bbox_inches="tight", pad_inches=0.04, facecolor="white", **kwargs)
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(1.5)


# ══════════════════════════════════════════════════════════════════════════════
# draw.io renderer
# ══════════════════════════════════════════════════════════════════════════════
def pt(x, y):
    return round(x * PX, 1), round((H - y) * PX, 1)


def geom(x, y, w, h):
    return round(x * PX, 1), round((H - y - h) * PX, 1), round(w * PX, 1), round(h * PX, 1)


def vtx(cid, value, style, x, y, w, h):
    return (f'<mxCell id="{cid}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edg(eid, value, style, a, b, points=None):
    inner = f'<mxPoint x="{a[0]}" y="{a[1]}" as="sourcePoint"/><mxPoint x="{b[0]}" y="{b[1]}" as="targetPoint"/>'
    if points:
        inner += '<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points) + "</Array>"
    return (f'<mxCell id="{eid}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">{inner}</mxGeometry></mxCell>')


def drawio_render():
    font = "fontFamily=Helvetica;"
    cells = []
    for i, p in enumerate(sorted(SCENE, key=lambda p: p["z"])):
        k, cid = p["kind"], f"c{i}"
        if k == "rect":
            style = (f"rounded={1 if p['r'] else 0};whiteSpace=wrap;html=1;fillColor={p['fill']};strokeColor={p['stroke']};"
                     f"strokeWidth={max(0.5, p['lw'])};{'dashed=1;' if p['dashed'] else ''}")
            if p["r"]:
                style += f"arcSize={min(50, round(100 * p['r'] / max(1e-6, min(p['w'], p['h']))))};"
            cells.append(vtx(cid, "", style, *geom(p["x"], p["y"], p["w"], p["h"])))
        elif k == "circle":
            cells.append(vtx(cid, "", f"ellipse;aspect=fixed;fillColor={p['fill']};strokeColor={p['stroke']};strokeWidth={p['lw']};",
                             *geom(p["cx"] - p["r"], p["cy"] - p["r"], 2 * p["r"], 2 * p["r"])))
        elif k == "line":
            pts = [pt(*q) for q in p["pts"]]
            cells.append(edg(cid, "", f"endArrow=none;html=1;strokeColor={p['stroke']};strokeWidth={p['lw'] * 1.6:.2f};edgeStyle=none;"
                                      f"{'dashed=1;' if p['dashed'] else ''}opacity={round(100 * p['alpha'])};", pts[0], pts[-1], pts[1:-1]))
        elif k == "arrow":
            base = f"endArrow=blockThin;endFill=1;html=1;strokeColor={p['stroke']};strokeWidth={p['lw'] * 1.6:.2f};{'dashed=1;' if p['dashed'] else ''}"
            if p["curve"]:
                a, c, b = p["curve"]
                cells.append(edg(cid, "", base + "edgeStyle=none;curved=1;", pt(*a), pt(*b), [pt(*c)]))
            else:
                pts = [pt(*q) for q in p["pts"]]
                cells.append(edg(cid, "", base + ("edgeStyle=orthogonalEdgeStyle;rounded=1;" if len(pts) > 2 else "edgeStyle=none;"), pts[0], pts[-1], pts[1:-1]))
        elif k == "text":
            plain = p["mpl"].replace("$", "").replace("\\", "")
            lines = plain.split("\n")
            w = p["w"] or (max(len(s) for s in lines) * 0.62 * p["size"] / 6.5 + 1.5)
            h = len(lines) * 2.6 * p["size"] / 6.5 + 0.4
            x = {"left": p["x"], "center": p["x"] - w / 2, "right": p["x"] - w}[p["ha"]]
            y = {"bottom": p["y"], "center": p["y"] - h / 2, "top": p["y"] - h}[p["va"]]
            # no whiteSpace=wrap: labels stay on one line (explicit <br> still breaks) and may overflow their cell
            style = (f"text;html=1;overflow=visible;align={p['ha']};verticalAlign={'middle' if p['va'] == 'center' else p['va']};"
                     f"fontSize={round(p['size'] * 2.7)};fontColor={p['colour']};{font}{'fontStyle=1;' if p['bold'] else ''}spacing=0;")
            cells.append(vtx(cid, p["html"], style, *geom(x, y, w, h)))
        elif k == "image":
            b64 = base64.b64encode(FRAME_PNG.read_bytes()).decode()
            cells.append(vtx(cid, "", f"shape=image;imageAspect=0;strokeColor={INK_SOFT};imageBorder={INK_SOFT};image=data:image/png,{b64};",
                             *geom(p["x"], p["y"], p["w"], p["h"])))
        elif k == "cuboid":
            cells.append(vtx(cid, "", f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;"
                                      f"size={round(p['d'] * PX)};fillColor={GREEN_FILL};strokeColor={GREEN_DARK};",
                             *geom(p["x"], p["y"], p["w"] + p["d"], p["h"] + p["d"] * 0.8)))
        elif k == "brace":
            x0, x1, y = p["x0"], p["x1"], p["y"]
            span, thick = x1 - x0, 3.5
            cells.append(vtx(cid, "", f"shape=curlyBracket;rounded=1;rotation=90;html=1;strokeColor={ARROW};size=0.5;",
                             *geom((x0 + x1) / 2 - thick / 2, y - p["d"] - span / 2 + thick / 2 + 0.2, thick, span)))
    xml = ('<mxfile host="app.diagrams.net" type="device"><diagram id="vqvae-masking" name="Architecture">'
           f'<mxGraphModel dx="1400" dy="900" grid="1" gridSize="8" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
           f'pageScale="1" pageWidth="{W * PX + 40}" pageHeight="{H * PX + 40}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
           + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
    (OUT_DIR / "architecture.drawio").write_text(xml, encoding="utf-8")
    return xml


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_scene()
    mpl_render(plt.imread(FRAME_PNG))
    xml = drawio_render()
    print("wrote", OUT_DIR / "architecture.png", OUT_DIR / "architecture.pdf", OUT_DIR / "architecture.drawio",
          f"({len(SCENE)} primitives, {len(xml) // 1024} KB)")
