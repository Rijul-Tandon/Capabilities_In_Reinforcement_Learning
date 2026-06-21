import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AGENT_COLORS = {
    "random_mujoco_agent": "dimgray",
    "ppo_mujoco_baseline": "seagreen",
    "ppo_mujoco_reward_shaping": "darkorange",
}


def rolling(values, window):
    if not values:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def load_runs(results_dir, env_id):
    runs_by_exp = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__*")):
        config_path = run_dir / "config.json"
        episodes_path = run_dir / "episodes.csv"
        metrics_path = run_dir / "metrics.csv"
        if not config_path.exists() or not episodes_path.exists():
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        episode_rows = []
        with open(episodes_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                entry = {
                    "global_step": int(row["global_step"]),
                    "episodic_return": float(row["episodic_return"]),
                    "goal_reached": float(row["goal_reached"]),
                }
                if "epsilon" in row and row["epsilon"] != "":
                    entry["epsilon"] = float(row["epsilon"])
                episode_rows.append(entry)

        metric_rows = []
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    parsed = {"global_step": int(row["global_step"])}
                    for key, value in row.items():
                        if key == "global_step" or value in (None, ""):
                            continue
                        try:
                            parsed[key] = float(value)
                        except ValueError:
                            continue
                    metric_rows.append(parsed)

        if episode_rows:
            exp_name = config.get("exp_name", run_dir.name.split("__")[1])
            seed = int(config.get("seed", 1))
            if exp_name not in runs_by_exp:
                runs_by_exp[exp_name] = {}
            runs_by_exp[exp_name][seed] = (run_dir.name, config, episode_rows, metric_rows)

    return {exp_name: list(seed_map.values()) for exp_name, seed_map in runs_by_exp.items()}


def pick_metric(runs_by_exp, candidates):
    for _, run_list in runs_by_exp.items():
        for _, _, _, metric_rows in run_list:
            if not metric_rows:
                continue
            keys = set().union(*(row.keys() for row in metric_rows))
            for candidate in candidates:
                if candidate in keys:
                    return candidate
    return None


def plot_curves(runs_by_exp, env_id, output_path, rolling_window):
    has_epsilon = any(
        run_list and any("epsilon" in row for _, _, rows, _ in run_list for row in rows)
        for run_list in runs_by_exp.values()
    )
    secondary_metric = pick_metric(
        runs_by_exp,
        ["td_loss", "value_loss", "policy_loss", "approx_kl", "mean_penalty"],
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=not has_epsilon)

    for exp_name, run_list in runs_by_exp.items():
        color = AGENT_COLORS.get(exp_name, None)
        for i, (_, config, rows, metric_rows) in enumerate(run_list):
            seed = int(config.get("seed", 1))
            label = f"{exp_name} s={seed}"
            linestyle = ["-", "--", ":", "-."][i % 4]

            x_values = [row["epsilon"] for row in rows] if has_epsilon and all("epsilon" in row for row in rows) else [row["global_step"] for row in rows]
            returns = rolling([row["episodic_return"] for row in rows], rolling_window)
            success = rolling([row["goal_reached"] for row in rows], rolling_window)

            axes[0].plot(x_values, returns, label=label, color=color, linestyle=linestyle, linewidth=1.6)
            axes[1].plot(x_values, success, label=label, color=color, linestyle=linestyle, linewidth=1.6)

            if secondary_metric and metric_rows:
                mx = [row["global_step"] for row in metric_rows if secondary_metric in row]
                my = [row[secondary_metric] for row in metric_rows if secondary_metric in row]
                if mx and my:
                    axes[2].plot(mx, rolling(my, rolling_window), label=label, color=color, linestyle=linestyle, linewidth=1.6)

    axes[0].set_title(f"{env_id} - Episodic Return")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"{env_id} - Success Rate")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    metric_title = secondary_metric if secondary_metric else "No metric available"
    axes[2].set_title(f"{env_id} - {metric_title}")
    axes[2].set_ylabel(metric_title)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Epsilon" if has_epsilon else "Training Steps")

    if has_epsilon:
        for ax in axes[:2]:
            ax.invert_xaxis()
            ax.set_xlabel("Epsilon")
        axes[2].set_xlabel("Training Steps")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--rolling-window", type=int, default=20)
    args = parser.parse_args()

    runs_by_exp = load_runs(args.results_dir, args.env_id)
    if not runs_by_exp:
        raise SystemExit(f"No runs found for {args.env_id} in {args.results_dir}")

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / f"{args.env_id}_benchmark_curves.png"
    plot_curves(runs_by_exp, args.env_id, output_path, args.rolling_window)
