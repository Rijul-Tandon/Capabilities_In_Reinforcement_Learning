"""
plot_learned_penalty.py - What did the penalty MLP learn, and did it help?
==========================================================================
Reads the run directories written by ddqn_baseline.py, ddqn_reward_shaping.py and ddqn_learned_penalty.py
for one environment and draws three figures (mean over seeds, shaded band = min..max across seeds):

  learned_penalty_probability.png   p(penalise) of the MLP on "stuck" transitions (observation unchanged)
                                    versus "moving" transitions, over training.  If the learned rule rediscovers
                                    the hand-written one, the stuck curve rises towards 1 and the moving curve
                                    falls towards 0.
  stuck_rate_by_agent.png           fraction of steps on which the observation did not change, for all agents.
  episode_length_by_agent.png       episode length over training for all agents (the "minimise steps" objective).

Usage:
  python src/plot_learned_penalty.py --env-id MiniGrid-DoorKey-6x6-v0 --plots-dir plots/learned_penalty
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AGENTS = [  # exp_name, label, colour (same colours as plot_comparison.py)
    ("ddqn_baseline", "Baseline DDQN", "#1f77b4"),
    ("ddqn_reward_shaping", "RS-DDQN (manual penalty)", "#ff7f0e"),
    ("ddqn_learned_penalty", "LP-DDQN (learned penalty)", "#2ca02c"),
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_runs(results_dir, env_id, exp_name):
    """{seed: (episode rows, metric rows)} for every run of one agent on one environment."""
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
        runs[int(config.get("seed", len(runs) + 1))] = (episodes, metrics)
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
    total = 0
    for runs in runs_by_agent.values():
        for episodes, metrics in runs.values():
            for rows in (episodes, metrics):
                if rows:
                    total = max(total, max(float(r["global_step"]) for r in rows if r.get("global_step")))
    if total == 0:
        print(f"[plot_learned_penalty] no runs found for {args.env_id} in {args.results_dir}")
        return
    edges = np.linspace(0, total + 1, args.bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    # 1. what the MLP learned: p(penalise) on stuck vs moving transitions
    learned = runs_by_agent["ddqn_learned_penalty"]
    if learned:
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, colour, label in (("penalty_prob_stuck", "#d62728", "stuck transitions (observation unchanged)"),
                                   ("penalty_prob_move", "#2ca02c", "moving transitions")):
            per_seed = []
            for _, metrics in learned.values():
                x, y = series(metrics, "global_step", key)
                if len(x):
                    per_seed.append(binned(x, y, edges))
            if per_seed:
                plot_band(ax, centres, per_seed, colour, label)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=9)
        style(ax, "Training step", "P(penalise transition)", f"Learned penalty head: what it penalises ({args.env_id})")
        fig.tight_layout()
        fig.savefig(plots_dir / "learned_penalty_probability.png", dpi=150)
        plt.close(fig)

    # 2. stuck rate per agent, 3. episode length per agent
    for key, source, ylabel, title, stem in (
            ("stuck_rate", "metrics", "Stuck rate (fraction of steps)", "Stuck rate over training", "stuck_rate_by_agent"),
            ("episodic_length", "episodes", "Episode length (steps)", "Episode length over training", "episode_length_by_agent")):
        fig, ax = plt.subplots(figsize=(7, 4))
        drew = False
        for exp, label, colour in AGENTS:
            per_seed = []
            for episodes, metrics in runs_by_agent[exp].values():
                x, y = series(metrics if source == "metrics" else episodes, "global_step", key)
                if len(x):
                    per_seed.append(binned(x, y, edges))
            if per_seed:
                plot_band(ax, centres, per_seed, colour, label)
                drew = True
        if drew:
            ax.legend(loc="best", fontsize=9)
            style(ax, "Training step", ylabel, f"{title} ({args.env_id})")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{stem}.png", dpi=150)
        plt.close(fig)
    print(f"[plot_learned_penalty] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
