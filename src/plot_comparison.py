"""
plot_comparison.py - Training Curve & Decay Comparison Plotter
==============================================================
Generates multi-panel comparison plots showing how different agents and exploration
decay strategies perform on MiniGrid environments.

Supported modes:
  1. Standard Agent Comparison (Random vs Baseline DDQN vs Reward-Shaped DDQN)
  2. Decay Strategy Comparison (Comparing Epsilon Decay schedules for Baseline and Reward Shaped DDQN)

Usage:
  python plot_comparison.py --env-id MiniGrid-Empty-6x6-v0
  python plot_comparison.py --env-id MiniGrid-DoorKey-6x6-v0 --decay-comparison
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

import argparse
import csv
import json
from pathlib import Path

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rolling(values, window):
    """
    Computes a rolling (moving) average over a list of values.
    """
    if len(values) == 0:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def load_runs(results_dir, env_id):
    """
    Scans the results directory for all experiment runs matching a specific environment,
    and returns the LATEST run per (experiment type, seed).
    """
    runs_by_exp = {}

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
                        "epsilon": float(row.get("epsilon", 0.0)),
                    }
                )

        metric_path = run_dir / "metrics.csv"
        metric_rows = []
        if metric_path.exists():
            with open(metric_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    loss_val = row.get("td_loss", "nan")
                    try:
                        loss_val = float(loss_val)
                    except ValueError:
                        loss_val = float("nan")
                    metric_rows.append(
                        {
                            "global_step": int(row["global_step"]),
                            "td_loss": loss_val,
                        }
                    )

        if rows:
            exp_name = config.get("exp_name", "unknown")
            seed = int(config.get("seed", 1))
            
            if exp_name not in runs_by_exp:
                runs_by_exp[exp_name] = {}
            
            runs_by_exp[exp_name][seed] = (run_dir.name, config, rows, metric_rows)

    for exp_name in runs_by_exp:
        runs_by_exp[exp_name] = list(runs_by_exp[exp_name].values())

    return runs_by_exp


# ============================================================================
# FIGURE HELPER & COLOR PALETTE
# ============================================================================

AGENT_COLORS = {
    "random_agent":          "dimgray",
    "ddqn_baseline":         "#1f77b4",  # Blue
    "ddqn_reward_shaping":   "#ff7f0e",  # Orange
    "dqn_baseline":          "#1f77b4",
    "dqn_reward_shaping":    "#ff7f0e",
    "dqn_vanilla":           "#2ca02c",  # Green
}

DISTINCT_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-Green
    "#17becf",  # Cyan
]

LINE_STYLES = ["-", "--", ":", "-."]


def format_exp_label(exp_name):
    """
    Formats raw experiment directory names into clean legend labels.
    e.g., 'ddqn_baseline_decay_linear' -> 'Baseline (linear)'
    """
    if "_decay_" in exp_name:
        parts = exp_name.split("_decay_")
        agent_clean = parts[0].replace("ddqn_", "").replace("_", " ").title()
        return f"{agent_clean} ({parts[1]})"
    return exp_name.replace("ddqn_", "").replace("_", " ").title()


def get_best_exp(exp_dict):
    """
    Given a dict of {exp_name: run_list}, computes the average performance
    in the FIRST 40% of training steps for each experiment, and returns the exp_name
    with the highest performance.
    """
    best_exp = None
    best_score = -float("inf")
    
    for exp_name, run_list in exp_dict.items():
        scores = []
        for _, _, rows, _ in run_list:
            if not rows:
                continue
            num_rows = len(rows)
            first_portion = rows[:max(1, int(num_rows * 0.4))]
            if not first_portion:
                first_portion = rows
            avg_goal = np.mean([r["goal_reached"] for r in first_portion])
            avg_ret = np.mean([r["episodic_return"] for r in first_portion])
            score = avg_goal + 0.1 * avg_ret
            scores.append(score)
            
        if scores:
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_exp = exp_name
                
    return best_exp


def plot_figure(runs_by_exp, env_id, output_path, rolling_window, title_suffix=""):
    """
    Creates and saves a 3-panel comparison figure from a runs_by_exp dictionary.
    Assigns distinct colors to each experiment line so they are clearly distinguishable.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    any_dqn_config = None

    # Pre-assign distinct colors to each experiment name
    exp_colors = {}
    for idx, exp_name in enumerate(sorted(runs_by_exp.keys())):
        if exp_name in AGENT_COLORS and len(runs_by_exp) <= 3:
            exp_colors[exp_name] = AGENT_COLORS[exp_name]
        else:
            exp_colors[exp_name] = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]

    for exp_name, run_list in runs_by_exp.items():
        base_color = exp_colors[exp_name]
        clean_label = format_exp_label(exp_name)

        for i, (_, config, rows, metric_rows) in enumerate(run_list):
            seed = config.get("seed", 1)
            ls = LINE_STYLES[i % len(LINE_STYLES)]

            if len(run_list) > 1:
                label = f"{clean_label} (s={seed})"
            else:
                label = clean_label

            steps = [r["global_step"] for r in rows]
            returns = rolling([r["episodic_return"] for r in rows], rolling_window)
            goals = rolling([r["goal_reached"] for r in rows], rolling_window)

            axes[0].plot(steps, returns, label=label, color=base_color, linestyle=ls, linewidth=1.8)
            axes[1].plot(steps, goals, label=label, color=base_color, linestyle=ls, linewidth=1.8)

            if metric_rows:
                ls_steps = [r["global_step"] for r in metric_rows]
                lv = rolling([r["td_loss"] for r in metric_rows], rolling_window)
                valid = [(s, l) for s, l in zip(ls_steps, lv) if l == l]
                if valid:
                    vs, vl = zip(*valid)
                    axes[2].plot(vs, vl, label=label, color=base_color, linestyle=ls, linewidth=1.8)

            if exp_name != "random_agent":
                any_dqn_config = config

    max_x = any_dqn_config.get("total_timesteps", 200000) if any_dqn_config else 200000
    axes[0].set_xlim(0, max_x)

    axes[0].set_title(f"{env_id} — Episodic Return{title_suffix}")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")

    axes[1].set_title(f"{env_id} — Goal Reached Rate{title_suffix}")
    axes[1].set_ylabel("Goal Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper left")

    axes[2].set_title(f"{env_id} — TD Loss{title_suffix}")
    axes[2].set_ylabel("TD Loss")
    axes[2].set_xlabel("Training Steps")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8, loc="upper left")

    # Secondary X-axis: Epsilon
    ax_eps = axes[2].twiny()
    ax_eps.set_xlim(axes[2].get_xlim())
    ax_eps.xaxis.set_ticks_position("bottom")
    ax_eps.xaxis.set_label_position("bottom")
    ax_eps.spines["bottom"].set_position(("outward", 40))
    ax_eps.set_xlabel("Exploration Rate (Epsilon)")

    ticks = axes[2].get_xticks()
    ax_eps.set_xticks(ticks)

    if any_dqn_config:
        start_e = any_dqn_config.get("start_e", 1.0)
        end_e = any_dqn_config.get("end_e", 0.1)
        frac = any_dqn_config.get("exploration_fraction", 0.5)
        total = any_dqn_config.get("total_timesteps", 200000)
        slope = (end_e - start_e) / (frac * total)
        eps_labels = []
        for t in ticks:
            if t < 0 or t > total:
                eps_labels.append("")
            else:
                eps_labels.append(f"{max(slope * t + start_e, end_e):.2f}")
        ax_eps.set_xticklabels(eps_labels)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def main():
    """
    Entry point: parses arguments, loads all runs, and generates comparison plots.
    Supports standard agent comparison as well as decay strategy comparison mode.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--decay-comparison", action="store_true",
                        help="Generate decay comparison plots for Baseline, Reward Shaped, and Best of Both.")
    args = parser.parse_args()

    runs_by_exp = load_runs(args.results_dir, args.env_id)
    if not runs_by_exp:
        raise SystemExit(f"No runs found for {args.env_id} in {args.results_dir}")

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect if decay runs are present
    is_decay_run = args.decay_comparison or any("_decay_" in exp for exp in runs_by_exp.keys())

    if is_decay_run:
        print(f"=== Generating Per-Seed Epsilon Decay Comparisons for {args.env_id} ===")

        # Collect all seeds present across decay runs
        decay_seeds = set()
        for run_list in runs_by_exp.values():
            for _, config, _, _ in run_list:
                decay_seeds.add(int(config.get("seed", 1)))
        decay_seeds = sorted(decay_seeds)

        def filter_by_seed(exp_dict, target_seed):
            res = {}
            for exp, r_list in exp_dict.items():
                s_runs = [r for r in r_list if int(r[1].get("seed", 1)) == target_seed]
                if s_runs:
                    res[exp] = s_runs
            return res

        baseline_runs = {exp: runs for exp, runs in runs_by_exp.items() if "ddqn_baseline" in exp or "dqn_baseline" in exp}
        shaped_runs = {exp: runs for exp, runs in runs_by_exp.items() if "ddqn_reward_shaping" in exp or "dqn_reward_shaping" in exp}

        for s in decay_seeds:
            seed_b_runs = filter_by_seed(baseline_runs, s)
            seed_s_runs = filter_by_seed(shaped_runs, s)

            # 1. Compare all decay strategies for Baseline DDQN on seed `s`
            if seed_b_runs:
                out_b_s = plots_dir / f"{args.env_id}_decay_comparison_baseline_seed{s}.png"
                plot_figure(seed_b_runs, args.env_id, out_b_s, args.rolling_window,
                            title_suffix=f" | Baseline DDQN (Seed {s})")

            # 2. Compare all decay strategies for Reward Shaped DDQN on seed `s`
            if seed_s_runs:
                out_s_s = plots_dir / f"{args.env_id}_decay_comparison_reward_shaping_seed{s}.png"
                plot_figure(seed_s_runs, args.env_id, out_s_s, args.rolling_window,
                            title_suffix=f" | Reward Shaped DDQN (Seed {s})")

            # 3. Compare Winner Baseline Decay vs Winner Reward Shaped Decay on seed `s`
            winner_b_s = get_best_exp(seed_b_runs) if seed_b_runs else None
            winner_s_s = get_best_exp(seed_s_runs) if seed_s_runs else None
            
            best_s_dict = {}
            if winner_b_s and winner_b_s in seed_b_runs:
                best_s_dict[winner_b_s] = seed_b_runs[winner_b_s]
            if winner_s_s and winner_s_s in seed_s_runs:
                best_s_dict[winner_s_s] = seed_s_runs[winner_s_s]

            if best_s_dict:
                out_best_s = plots_dir / f"{args.env_id}_decay_comparison_best_of_both_seed{s}.png"
                plot_figure(best_s_dict, args.env_id, out_best_s, args.rolling_window,
                            title_suffix=f" | Best Baseline vs Best Shaped (Seed {s})")

    else:
        # Standard seed-by-seed agent comparison plots
        all_seeds = set()
        for exp_name, run_list in runs_by_exp.items():
            if exp_name == "random_agent":
                continue
            for _, config, _, _ in run_list:
                all_seeds.add(int(config.get("seed", 1)))
        all_seeds = sorted(all_seeds)

        print(f"Found DQN seeds: {all_seeds} for {args.env_id}")

        first_seed = all_seeds[0] if all_seeds else None
        for seed in all_seeds:
            filtered = {}
            for exp_name, run_list in runs_by_exp.items():
                if exp_name == "random_agent":
                    if seed == first_seed:
                        filtered[exp_name] = run_list[:1]
                else:
                    seed_runs = [(d, c, r, m) for d, c, r, m in run_list if int(c.get("seed", 1)) == seed]
                    if seed_runs:
                        filtered[exp_name] = seed_runs

            if not filtered:
                continue

            out = plots_dir / f"{args.env_id}_comparison_seed{seed}.png"
            plot_figure(filtered, args.env_id, out, args.rolling_window,
                        title_suffix=f" | Seed {seed}")


if __name__ == "__main__":
    main()
