"""
plot_learned_penalty.py - What did the penalty head learn, and did it help?
===========================================================================
Reads the run directories written by ddqn_baseline.py, ddqn_reward_shaping.py and ddqn_learned_penalty.py
for one environment and draws:

Training curves (mean over seeds, shaded band = min..max across seeds)
  learned_penalty_probability.png   p(penalise) of the head on "stuck" transitions (observation unchanged)
                                    versus "moving" transitions, over training.  If the learned rule rediscovers
                                    the hand-written one, the stuck curve rises towards 1 and the moving curve
                                    falls towards 0.
  stuck_rate_by_agent.png           fraction of steps on which the observation did not change, for all agents.
  episode_length_by_agent.png       episode length over training for all agents (the "minimise steps" objective).

The learned penalty table itself (one figure per learned-penalty run, read from penalty_net.pt)
  <env>_penalty_table_seed<k>.png   heat maps on the grid of the penalty probability the head assigns to
                                    (row 1) staying in the same state, s -> s: a wall bump, a pickup of nothing;
                                    (row 2) moving forward one cell, s -> s + direction;
                                    (row 3) turning on the spot, s -> (x, y, direction +- 1).
                                    One column per facing direction (East, South, West, North); walls are dark,
                                    S / G / K / D mark start, goal, key and door.  With --penalty-state position
                                    the table has no direction, so rows 1 and 2 collapse to one panel each.
  <env>_penalty_table_summary.png   mean probability per transition type, one bar per seed: the head has learned
                                    the stuck rule when "stay" is high and "forward" / "turn" are low.
  <env>_penalty_transitions_seed<k>.png
                                    the learned penalties drawn as TRANSITIONS on one grid: a disc in every cell for
                                    s -> s (staying put) and one arrow per direction for the forward move into the
                                    neighbouring cell, each coloured by its penalty probability.
  <env>_penalty_histogram_seed<k>.png
                                    distribution of the penalty probabilities over all free cells, one histogram per
                                    transition type, with the mean of each type marked.

Usage:
  python src/plot_learned_penalty.py --env-id MiniGrid-DoorKey-6x6-v0 --plots-dir plots/learned_penalty
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dqn_common import make_env                                   # noqa: E402
from learned_penalty import DiscreteStateSpace, TransitionPenaltyHead   # noqa: E402

AGENTS = [  # exp_name, label, colour (same colours as plot_comparison.py)
    ("ddqn_baseline", "Baseline DDQN", "#1f77b4"),
    ("ddqn_reward_shaping", "RS-DDQN (manual penalty)", "#ff7f0e"),
    ("ddqn_learned_penalty", "LP-DDQN (learned penalty)", "#2ca02c"),
]
DIR_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
DIR_OFFSETS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
TRANSITION_COLOURS = {"stay": "#d62728", "forward": "#2ca02c", "turn": "#1f77b4", "neighbour": "#2ca02c"}


# ────────────────────────────────────────────────────────────────────────────
# run discovery and CSV helpers
# ────────────────────────────────────────────────────────────────────────────
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_runs(results_dir, env_id, exp_name):
    """{seed: (run_dir, config, episode rows, metric rows)} for every run of one agent on one environment."""
    runs = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}__*")):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("exp_name") != exp_name:
            continue
        episodes = read_csv(run_dir / "episodes.csv") if (run_dir / "episodes.csv").exists() else []
        metrics = read_csv(run_dir / "metrics.csv") if (run_dir / "metrics.csv").exists() else []
        runs[int(config.get("seed", len(runs) + 1))] = (run_dir, config, episodes, metrics)
    return runs


def series(rows, x_key, y_key):
    """(x, y) arrays from CSV rows, skipping blank or non-numeric values."""
    xs, ys = [], []
    for row in rows:
        try:
            x, y = float(row[x_key]), float(row[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def binned(x, y, edges):
    """One seed's series resampled onto the common step grid: bin means where a bin has data, linear
    interpolation between the logged points elsewhere, so sparse logs still give a continuous curve."""
    centres = (edges[:-1] + edges[1:]) / 2
    out = np.full(len(centres), np.nan)
    for i in range(len(centres)):
        inside = (x >= edges[i]) & (x < edges[i + 1])
        if inside.any():
            out[i] = y[inside].mean()
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    if len(x_sorted) >= 2:
        covered = (centres >= x_sorted[0]) & (centres <= x_sorted[-1])
        fill = np.isnan(out) & covered
        out[fill] = np.interp(centres[fill], x_sorted, y_sorted)
    return out


def plot_band(ax, centres, per_seed, colour, label):
    """Mean line with a min..max band over seeds; per_seed is a list of equally binned arrays."""
    stack = np.vstack(per_seed)
    mean = np.nanmean(stack, axis=0)
    lo, hi = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
    ax.plot(centres, mean, color=colour, lw=1.8, label=f"{label} (n={len(per_seed)})")
    if len(per_seed) > 1:
        ax.fill_between(centres, lo, hi, color=colour, alpha=0.15, lw=0)


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ────────────────────────────────────────────────────────────────────────────
# training curves
# ────────────────────────────────────────────────────────────────────────────
def plot_training_curves(env_id, runs_by_agent, plots_dir, bins):
    total = 0
    for runs in runs_by_agent.values():
        for _, _, episodes, metrics in runs.values():
            for rows in (episodes, metrics):
                if rows:
                    total = max(total, max(float(r["global_step"]) for r in rows if r.get("global_step")))
    if total == 0:
        print(f"[plot_learned_penalty] no episode or metric logs found for {env_id}")
        return
    edges = np.linspace(0, total + 1, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    learned = runs_by_agent["ddqn_learned_penalty"]
    if learned:
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, colour, label in (("penalty_prob_stuck", "#d62728", "stuck transitions (observation unchanged)"),
                                   ("penalty_prob_move", "#2ca02c", "moving transitions")):
            per_seed = []
            for _, _, _, metrics in learned.values():
                x, y = series(metrics, "global_step", key)
                if len(x):
                    per_seed.append(binned(x, y, edges))
            if per_seed:
                plot_band(ax, centres, per_seed, colour, label)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=9)
        style(ax, "Training step", "penalty score (penalty = scale × score)", f"Learned penalty head: what it penalises ({env_id})")
        fig.tight_layout()
        fig.savefig(plots_dir / "learned_penalty_probability.png", dpi=150)
        plt.close(fig)

    for key, source, ylabel, title, stem in (
            ("stuck_rate", "metrics", "Stuck rate (fraction of steps)", "Stuck rate over training", "stuck_rate_by_agent"),
            ("episodic_length", "episodes", "Episode length (steps)", "Episode length over training", "episode_length_by_agent")):
        fig, ax = plt.subplots(figsize=(7, 4))
        drew = False
        for exp, label, colour in AGENTS:
            per_seed = []
            for _, _, episodes, metrics in runs_by_agent[exp].values():
                x, y = series(metrics if source == "metrics" else episodes, "global_step", key)
                if len(x):
                    per_seed.append(binned(x, y, edges))
            if per_seed:
                plot_band(ax, centres, per_seed, colour, label)
                drew = True
        if drew:
            ax.legend(loc="best", fontsize=9)
            style(ax, "Training step", ylabel, f"{title} ({env_id})")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{stem}.png", dpi=150)
        plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# the learned table on the grid
# ────────────────────────────────────────────────────────────────────────────
def grid_layout(env):
    """Walls and annotated cells of the environment's current layout."""
    u = env.unwrapped
    walls = np.zeros((u.width, u.height), dtype=bool)
    marks = {tuple(u.agent_pos): "S"}
    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "wall":
                walls[x, y] = True
            elif cell.type == "goal":
                marks[(x, y)] = "G"
            elif cell.type == "key":
                marks[(x, y)] = "K"
            elif cell.type == "door":
                marks[(x, y)] = "D"
    return walls, marks


def load_head(run_dir, config, space):
    """The trained head of one run, or None when its size does not match the state space."""
    path = run_dir / "penalty_net.pt"
    if not path.exists():
        return None
    head = TransitionPenaltyHead(space.size, int(config.get("penalty_rank", 0)))
    try:
        head.load_state_dict(torch.load(path, map_location="cpu"))
    except RuntimeError as error:
        print(f"[plot_learned_penalty] {run_dir.name}: head does not match the state space ({error}); skipped")
        return None
    head.eval()
    return head


@torch.no_grad()
def penalty_maps(head, space, walls):
    """Probability of penalising each transition type from every state, as (W, H) maps per direction.

    Returns {"stay": [map per direction], "forward": [...], "turn": [...]} for position_dir / full tables, or
    {"stay": [map], "neighbour": [map]} for position-only tables.  Wall cells and moves into walls are NaN.
    """
    W, H = space.width, space.height
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
    free = ~walls

    def prob(s_from, s_to, valid):
        p = np.full((W, H), np.nan)
        if valid.any():
            logits = head.logits(torch.as_tensor(s_from[valid]), torch.as_tensor(s_to[valid]))
            p[valid] = torch.sigmoid(logits).numpy()
        return p

    maps = {}
    if space.with_direction:
        maps = {"stay": [], "forward": [], "turn": []}
        for d in range(4):
            s = np.vectorize(lambda x, y: space.index(x, y, d))(xs, ys)
            maps["stay"].append(prob(s, s, free))
            dx, dy = DIR_OFFSETS[d]
            nx, ny = xs + dx, ys + dy
            inside = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
            valid = free & inside
            valid[valid] &= ~walls[nx[valid], ny[valid]]
            s_fwd = np.zeros_like(s)
            s_fwd[valid] = np.vectorize(lambda x, y: space.index(x, y, d))(nx[valid], ny[valid])
            maps["forward"].append(prob(s, s_fwd, valid))
            turn = np.zeros((W, H))
            for d2 in ((d - 1) % 4, (d + 1) % 4):
                s_turn = np.vectorize(lambda x, y: space.index(x, y, d2))(xs, ys)
                turn += np.nan_to_num(prob(s, s_turn, free))
            turn[~free] = np.nan
            maps["turn"].append(turn / 2.0)
    else:
        s = np.vectorize(lambda x, y: space.index(x, y))(xs, ys)
        maps["stay"] = [prob(s, s, free)]
        maps["neighbour_by_direction"] = []
        total, count = np.zeros((W, H)), np.zeros((W, H))
        for d in range(4):
            dx, dy = DIR_OFFSETS[d]
            nx, ny = xs + dx, ys + dy
            inside = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
            valid = free & inside
            valid[valid] &= ~walls[nx[valid], ny[valid]]
            s_nb = np.zeros_like(s)
            s_nb[valid] = np.vectorize(lambda x, y: space.index(x, y))(nx[valid], ny[valid])
            p = prob(s, s_nb, valid)
            maps["neighbour_by_direction"].append(p)
            total += np.nan_to_num(p)
            count += valid
        neighbour = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        neighbour[~free] = np.nan
        maps["neighbour"] = [neighbour]
    return maps


def draw_layout(ax, walls, marks, W, H):
    """Walls, S/G/K/D marks and the minor grid, without any data."""
    for x in range(W):
        for y in range(H):
            if walls[x, y]:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#333333"))
    for (x, y), label in marks.items():
        ax.text(x + 0.33, y - 0.33, label, color="white", fontsize=9, fontweight="bold", ha="center", va="center",
                path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#888888", linestyle="-", linewidth=0.6)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def plot_transition_map(env_id, seed, space, maps, walls, marks, plots_dir, subtitle):
    """One grid: a disc per cell for s -> s and one arrow per direction for the forward move, coloured by penalty."""
    W, H = space.width, space.height
    cmap = plt.get_cmap("Reds")
    stay = np.nanmean(np.stack(maps["stay"]), axis=0)
    arrows = maps["forward"] if space.with_direction else maps["neighbour_by_direction"]
    fig, ax = plt.subplots(figsize=(0.95 * W + 2.4, 0.95 * H + 1.4))
    draw_layout(ax, walls, marks, W, H)
    for x in range(W):
        for y in range(H):
            if walls[x, y] or not np.isfinite(stay[x, y]):
                continue
            ax.add_patch(plt.Circle((x, y), 0.17, color=cmap(stay[x, y]), ec="#555555", lw=0.5, zorder=3))
            for d in range(4):
                p = arrows[d][x, y]
                if not np.isfinite(p):
                    continue
                dx, dy = DIR_OFFSETS[d]
                ax.annotate("", xy=(x + 0.47 * dx, y + 0.47 * dy), xytext=(x + 0.2 * dx, y + 0.2 * dy),
                            arrowprops=dict(arrowstyle="-|>", color=cmap(p), lw=1.8, mutation_scale=9, shrinkA=0, shrinkB=0), zorder=4)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("penalty score  (penalty = scale × score)", fontsize=9)
    ax.set_title(f"{env_id} — learned penalty per transition, seed {seed}\n"
                 f"disc = stay in place (mean over facing directions);  arrow = move one cell that way\n{subtitle}",
                 fontsize=9.5, fontweight="bold")
    fig.text(0.42, 0.01, "S = Start   G = Goal   K = Key   D = Door   dark = wall", ha="center", va="bottom", fontsize=8, color="#444444")
    fig.savefig(plots_dir / f"{env_id}_penalty_transitions_seed{seed}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_penalty_histogram(env_id, seed, maps, plots_dir):
    """Distribution of the penalty probabilities over free cells, one histogram per transition type."""
    kinds = [k for k in ("stay", "forward", "turn", "neighbour") if k in maps]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, 1, 26)
    for kind in kinds:
        values = np.concatenate([m[np.isfinite(m)].ravel() for m in maps[kind]])
        if not len(values):
            continue
        colour = TRANSITION_COLOURS.get(kind, "#7f7f7f")
        ax.hist(values, bins=bins, color=colour, alpha=0.45, label=f"{kind} (mean {values.mean():.2f}, n={len(values)})")
        ax.axvline(values.mean(), color=colour, lw=1.5, ls="--")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9, loc="best")
    style(ax, "penalty score  (penalty = scale × score)", "number of (cell, direction) entries", f"Learned penalty distribution by transition type ({env_id}, seed {seed})")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{env_id}_penalty_histogram_seed{seed}.png", dpi=150)
    plt.close(fig)


def draw_grid_panel(ax, values, walls, marks, title):
    """One (W, H) map on the grid: probability colour, dark walls, S/G/K/D marks, minor grid lines."""
    W, H = values.shape
    image = ax.imshow(values.T, origin="upper", cmap="Reds", vmin=0.0, vmax=1.0, aspect="equal")
    for x in range(W):
        for y in range(H):
            if walls[x, y]:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#333333"))
            elif np.isfinite(values[x, y]):
                ax.text(x, y, f"{values[x, y]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="black" if values[x, y] < 0.6 else "white")
    for (x, y), label in marks.items():
        ax.text(x + 0.33, y - 0.33, label, color="white", fontsize=9, fontweight="bold", ha="center", va="center",
                path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#888888", linestyle="-", linewidth=0.6)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    return image


def plot_penalty_tables(env_id, learned_runs, plots_dir):
    """One grid figure per learned-penalty run plus a per-seed summary of the three transition types."""
    summary = {}
    for seed, (run_dir, config, _, _) in sorted(learned_runs.items()):
        env = make_env(env_id, seed, config.get("action_set", "task"))
        env.reset(seed=seed)
        space = DiscreteStateSpace(env, env_id, config.get("penalty_state", "position_dir"))
        head = load_head(run_dir, config, space)
        if head is None:
            env.close()
            continue
        walls, marks = grid_layout(env)
        env.close()
        maps = penalty_maps(head, space, walls)
        rank = int(config.get("penalty_rank", 0))
        subtitle = f"state = {space.mode}, {space.size} states, {'full table' if rank == 0 else f'rank {rank}'}"
        plot_transition_map(env_id, seed, space, maps, walls, marks, plots_dir, subtitle)
        plot_penalty_histogram(env_id, seed, maps, plots_dir)
        rows = [k for k in ("stay", "forward", "turn", "neighbour") if k in maps]
        cols = 4 if space.with_direction else 1
        fig, axes = plt.subplots(len(rows), cols, figsize=(3.4 * cols + 1.2, 3.3 * len(rows)), squeeze=False)
        titles = {"stay": "stay in place  s → s", "forward": "move forward one cell", "turn": "turn on the spot",
                  "neighbour": "move to a neighbouring cell"}
        image = None
        for r, kind in enumerate(rows):
            for c in range(cols):
                head_title = f"{titles[kind]}\nfacing {DIR_NAMES[c]}" if space.with_direction else titles[kind]
                image = draw_grid_panel(axes[r, c], maps[kind][c], walls, marks, head_title)
        fig.subplots_adjust(right=0.9, wspace=0.08, hspace=0.3)
        cbar = fig.colorbar(image, cax=fig.add_axes([0.92, 0.2, 0.015, 0.6]))
        cbar.set_label("penalty score  (penalty = scale × score)", fontsize=9)
        fig.suptitle(f"{env_id} — learned penalty table, seed {seed}  ({subtitle})", fontsize=11, fontweight="bold", y=0.995)
        fig.text(0.44, 0.005, "S = Start   G = Goal   K = Key   D = Door   dark = wall   NaN = move into a wall (not drawn)",
                 ha="center", va="bottom", fontsize=8, color="#444444")
        fig.savefig(plots_dir / f"{env_id}_penalty_table_seed{seed}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        summary[seed] = {kind: float(np.nanmean(np.stack(maps[kind]))) for kind in rows}
        print(f"[plot_learned_penalty] seed {seed}: " + ", ".join(f"{k} {v:.3f}" for k, v in summary[seed].items()))

    if summary:
        kinds = [k for k in ("stay", "forward", "turn", "neighbour") if any(k in v for v in summary.values())]
        seeds = sorted(summary)
        fig, ax = plt.subplots(figsize=(1.4 * len(seeds) + 3.5, 3.6))
        width = 0.8 / len(kinds)
        for k, kind in enumerate(kinds):   # a seed trained with another --penalty-state simply lacks some bars
            ax.bar(np.arange(len(seeds)) + (k - (len(kinds) - 1) / 2) * width, [summary[s].get(kind, np.nan) for s in seeds],
                   width=width, color=TRANSITION_COLOURS.get(kind, "#7f7f7f"), label=kind)
        ax.set_xticks(np.arange(len(seeds)))
        ax.set_xticklabels([f"seed {s}" for s in seeds])
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="#888888", lw=0.8, ls="--")
        ax.legend(fontsize=9, loc="best")
        style(ax, "", "mean penalty score over free cells", f"Learned penalty by transition type ({env_id})")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{env_id}_penalty_table_summary.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots/learned_penalty")
    parser.add_argument("--bins", type=int, default=40, help="number of training-progress bins for the curves")
    args = parser.parse_args()
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs_by_agent = {exp: load_runs(args.results_dir, args.env_id, exp) for exp, _, _ in AGENTS}
    if not any(runs_by_agent.values()):
        print(f"[plot_learned_penalty] no runs found for {args.env_id} in {args.results_dir}")
        return
    plot_training_curves(args.env_id, runs_by_agent, plots_dir, args.bins)
    if runs_by_agent["ddqn_learned_penalty"]:
        plot_penalty_tables(args.env_id, runs_by_agent["ddqn_learned_penalty"], plots_dir)
    print(f"[plot_learned_penalty] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
