import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rolling(values, window):
    if len(values) == 0:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def load_runs(results_dir, env_id):
    runs = []
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__*")):
        config_path = run_dir / "config.json"
        episode_path = run_dir / "episodes.csv"
        if not config_path.exists() or not episode_path.exists():
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        rows = []
        with open(episode_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "global_step": int(row["global_step"]),
                        "episodic_return": float(row["episodic_return"]),
                        "goal_reached": float(row["goal_reached"]),
                    }
                )
        if rows:
            runs.append((run_dir.name, config, rows))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--rolling-window", type=int, default=20)
    args = parser.parse_args()

    runs = load_runs(args.results_dir, args.env_id)
    if not runs:
        raise SystemExit(f"No runs found for {args.env_id} in {args.results_dir}")

    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for _, config, rows in runs:
        label = f"{config['exp_name']} seed={config['seed']}"
        steps = [r["global_step"] for r in rows]
        returns = rolling([r["episodic_return"] for r in rows], args.rolling_window)
        goals = rolling([r["goal_reached"] for r in rows], args.rolling_window)
        axes[0].plot(steps, returns, label=label)
        axes[1].plot(steps, goals, label=label)

    axes[0].set_title(f"{args.env_id} episodic return")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.3)
    axes[1].set_title(f"{args.env_id} goal reached rate")
    axes[1].set_ylabel("Goal rate")
    axes[1].set_xlabel("Global step")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()

    out = Path(args.plots_dir) / f"{args.env_id}_comparison.png"
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
