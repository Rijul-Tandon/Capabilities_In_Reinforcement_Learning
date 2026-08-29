"""
plot_comparison.py - Training Curve & Decay Comparison Plotter
==============================================================
Generates multi-panel comparison plots showing how different agents and exploration
decay strategies perform on MiniGrid environments.

Supported modes:
  1. Standard Agent Comparison (Random vs Baseline DDQN vs Reward-Shaped DDQN)
  2. Decay Strategy Comparison (Comparing Epsilon Decay schedules for Baseline and Reward Shaped DDQN)

Usage:
  python plot_comparison.py --env-id MiniGrid-Empty-Random-6x6-v0
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
                if not row or "global_step" not in row or row["global_step"] is None or row["global_step"] == "":
                    continue
                try:
                    rows.append(
                        {
                            "global_step": int(float(row["global_step"])),
                            "episodic_return": float(row.get("episodic_return", 0.0)),
                            "goal_reached": float(row.get("goal_reached", 0.0)),
                            "epsilon": float(row.get("epsilon", 0.0)),
                        }
                    )
                except (ValueError, KeyError) as e:
                    continue

        metric_path = run_dir / "metrics.csv"
        metric_rows = []
        if metric_path.exists():
            with open(metric_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not row or "global_step" not in row or row["global_step"] is None or row["global_step"] == "":
                        continue
                    try:
                        loss_val = float(row.get("td_loss", "nan"))
                    except (ValueError, TypeError):
                        loss_val = float("nan")
                    try:
                        metric_rows.append(
                            {
                                "global_step": int(float(row["global_step"])),
                                "td_loss": loss_val,
                            }
                        )
                    except (ValueError, KeyError):
                        continue

        if rows:
            exp_name = config.get("exp_name", "unknown")
            if exp_name == "random_agent":
                continue
            seed_val = config.get("seed", 1)
            if isinstance(seed_val, list):
                seed = int(seed_val[0])
            else:
                try:
                    seed = int(seed_val)
                except (ValueError, TypeError):
                    seed = 1
            
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
    e.g., 'ddqn_baseline_decay_linear' -> 'Baseline DDQN (linear)'
          'ddqn_reward_shaping_decay_linear' -> 'RS-DDQN (linear)'
    """
    label = exp_name
    if "reward_shaping" in label:
        label = label.replace("ddqn_reward_shaping", "RS-DDQN").replace("dqn_reward_shaping", "RS-DDQN")
    elif "baseline" in label:
        label = label.replace("ddqn_baseline", "Baseline DDQN").replace("dqn_baseline", "Baseline DDQN")
    
    if "_decay_" in exp_name:
        parts = exp_name.split("_decay_")
        decay_part = parts[1]
        if "RS-DDQN" in label:
            return f"RS-DDQN ({decay_part})"
        elif "Baseline DDQN" in label:
            return f"Baseline DDQN ({decay_part})"
        else:
            return f"{parts[0].replace('_', ' ').title()} ({decay_part})"
    return label.replace("_", " ").title() if label == exp_name else label


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


def plot_single_metric(runs_by_exp, env_id, output_path, rolling_window, metric_key, metric_title, y_label, y_limits=None, title_suffix=""):
    """
    Creates and saves a publication-grade standalone plot for a single metric.
    Includes rolling standard deviation / variance shading.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    any_dqn_config = None

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

            if metric_key in ["episodic_return", "goal_reached"]:
                steps = [r["global_step"] for r in rows]
                vals = [r[metric_key] for r in rows]
            elif metric_key == "td_loss" and metric_rows:
                steps = [r["global_step"] for r in metric_rows]
                vals = [r["td_loss"] for r in metric_rows]
            else:
                continue

            if not vals:
                continue

            smooth_vals = rolling(vals, rolling_window)
            
            # Compute rolling std for shaded confidence band
            std_vals = []
            for idx_v in range(len(vals)):
                st = max(0, idx_v - rolling_window + 1)
                std_vals.append(float(np.std(vals[st : idx_v + 1])))

            smooth_vals = np.array(smooth_vals)
            std_vals = np.array(std_vals)

            ax.plot(steps, smooth_vals, label=label, color=base_color, linestyle=ls, linewidth=2.0)
            ax.fill_between(steps, smooth_vals - std_vals * 0.5, smooth_vals + std_vals * 0.5,
                            color=base_color, alpha=0.12)

            if exp_name != "random_agent":
                any_dqn_config = config

    max_x = any_dqn_config.get("total_timesteps", 200000) if any_dqn_config else 200000
    ax.set_xlim(0, max_x)
    if y_limits:
        ax.set_ylim(y_limits)

    ax.set_title(f"{env_id} — {metric_title}{title_suffix}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Training Steps", fontsize=9.5, fontweight="medium")
    ax.set_ylabel(y_label, fontsize=9.5, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.legend(fontsize=8.5, loc="best", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved publication plot → {output_path}")


def plot_figure(runs_by_exp, env_id, output_path, rolling_window, title_suffix=""):
    """
    Creates and saves individual publication-grade plots for Return, Goal Reached Rate, and TD Loss.
    """
    out_dir = Path(output_path).parent
    out_name = Path(output_path).stem
    plot_single_metric(runs_by_exp, env_id, out_dir / f"{out_name}_episodic_return.png",
                       rolling_window, "episodic_return", "Episodic Return", "Episodic Return", title_suffix=title_suffix)
    plot_single_metric(runs_by_exp, env_id, out_dir / f"{out_name}_goal_reach.png",
                       rolling_window, "goal_reached", "Goal Reach Rate", "Goal Success Rate", y_limits=(-0.05, 1.05), title_suffix=title_suffix)
    plot_single_metric(runs_by_exp, env_id, out_dir / f"{out_name}_td_loss.png",
                       rolling_window, "td_loss", "TD Loss", "TD Loss", title_suffix=title_suffix)


def plot_single_mean_metric(runs_by_exp, env_id, output_path, rolling_window, metric_key, metric_title, y_label, y_limits=None, title_suffix=""):
    """
    Creates and saves a standalone plot for the mean of a single metric across all seeds.
    Includes shaded error bands (±1 std) showing seed variability.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    any_dqn_config = None

    exp_colors = {}
    for idx, exp_name in enumerate(sorted(runs_by_exp.keys())):
        if exp_name in AGENT_COLORS and len(runs_by_exp) <= 3:
            exp_colors[exp_name] = AGENT_COLORS[exp_name]
        else:
            exp_colors[exp_name] = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]

    for exp_name, run_list in runs_by_exp.items():
        if not run_list:
            continue
        base_color = exp_colors[exp_name]
        clean_label = format_exp_label(exp_name)
        num_seeds = len(run_list)
        label = f"{clean_label} (Mean across {num_seeds} seeds)" if num_seeds > 1 else clean_label

        all_seed_vals = []
        max_timestep = 0

        for _, config, rows, metric_rows in run_list:
            if exp_name != "random_agent":
                any_dqn_config = config

            if metric_key in ["episodic_return", "goal_reached"]:
                if rows:
                    steps = np.array([r["global_step"] for r in rows])
                    vals = np.array(rolling([r[metric_key] for r in rows], rolling_window))
                    all_seed_vals.append((steps, vals))
                    if steps[-1] > max_timestep:
                        max_timestep = steps[-1]
            elif metric_key == "td_loss" and metric_rows:
                m_steps = np.array([r["global_step"] for r in metric_rows])
                l_vals = np.array(rolling([r["td_loss"] for r in metric_rows], rolling_window))
                valid = [(s, l) for s, l in zip(m_steps, l_vals) if not np.isnan(l)]
                if valid:
                    vs, vl = zip(*valid)
                    all_seed_vals.append((np.array(vs), np.array(vl)))
                    if vs[-1] > max_timestep:
                        max_timestep = vs[-1]

        if not all_seed_vals:
            continue

        if max_timestep == 0:
            max_timestep = any_dqn_config.get("total_timesteps", 200000) if any_dqn_config else 200000

        grid_steps = np.linspace(0, max_timestep, num=500)
        interp_vals = []
        for s_arr, v_arr in all_seed_vals:
            interp_vals.append(np.interp(grid_steps, s_arr, v_arr))

        mean_v = np.mean(interp_vals, axis=0)
        std_v = np.std(interp_vals, axis=0) if num_seeds > 1 else np.zeros_like(mean_v)

        ax.plot(grid_steps, mean_v, label=label, color=base_color, linewidth=2.0)
        if num_seeds > 1:
            if metric_key == "goal_reached":
                lower = np.clip(mean_v - std_v, 0, 1)
                upper = np.clip(mean_v + std_v, 0, 1)
            elif metric_key == "td_loss":
                lower = np.maximum(0, mean_v - std_v)
                upper = mean_v + std_v
            else:
                lower = mean_v - std_v
                upper = mean_v + std_v
            ax.fill_between(grid_steps, lower, upper, color=base_color, alpha=0.15)

    max_x = any_dqn_config.get("total_timesteps", 200000) if any_dqn_config else 200000
    ax.set_xlim(0, max_x)
    if y_limits:
        ax.set_ylim(y_limits)

    ax.set_title(f"{env_id} — {metric_title}{title_suffix}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Training Steps", fontsize=9.5, fontweight="medium")
    ax.set_ylabel(y_label, fontsize=9.5, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.legend(fontsize=8.5, loc="best", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mean publication plot → {output_path}")


def plot_mean_figure(runs_by_exp, env_id, output_path, rolling_window, title_suffix=""):
    """
    Creates and saves individual aggregated publication plots showing the MEAN performance across ALL seeds.
    """
    out_dir = Path(output_path).parent
    out_name = Path(output_path).stem
    plot_single_mean_metric(runs_by_exp, env_id, out_dir / f"{out_name}_episodic_return.png",
                            rolling_window, "episodic_return", "Mean Episodic Return", "Episodic Return", title_suffix=title_suffix)
    plot_single_mean_metric(runs_by_exp, env_id, out_dir / f"{out_name}_goal_reach.png",
                            rolling_window, "goal_reached", "Mean Goal Reach Rate", "Goal Success Rate", y_limits=(-0.05, 1.05), title_suffix=title_suffix)
    plot_single_mean_metric(runs_by_exp, env_id, out_dir / f"{out_name}_td_loss.png",
                            rolling_window, "td_loss", "Mean TD Loss", "TD Loss", title_suffix=title_suffix)


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
    parser.add_argument("--plots-dir", type=str, default="plots/reward_comparison")
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

        # 1. Per-seed comparison plots (kept as requested)
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

        # 2. Mean comparison plot across ALL executed seeds
        if all_seeds:
            num_seeds = len(all_seeds)
            print(f"Generating mean aggregated comparison plots across all {num_seeds} seeds for {args.env_id} ...")
            out_mean = plots_dir / f"{args.env_id}_mean.png"
            plot_mean_figure(runs_by_exp, args.env_id, out_mean, args.rolling_window,
                             title_suffix=f" | Mean Across {num_seeds} Seeds")


if __name__ == "__main__":
    main()
