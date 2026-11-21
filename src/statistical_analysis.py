"""Is RS-DDQN better than the Baseline DDQN across seeds?  For each environment: final episodic return and goal rate
of every matched seed and the paired Wilcoxon signed-rank test with effect size r = |Z| / sqrt(N), written as
Markdown, CSV and Excel."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon


def discover_runs(results_dir, env_id, exp_name):
    """{seed: path of the episode log} for the runs of one agent on one environment (the latest run of each seed)."""
    episode_logs = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}*")):
        config_path = run_dir / "config.json"
        episode_path = run_dir / "episodes.csv"
        if not episode_path.exists():
            episode_path = run_dir / "episode_log.csv"      # the file name written by older runs
        if not config_path.exists() or not episode_path.exists():
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                seed = int(json.load(config_file)["seed"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        episode_logs[seed] = episode_path
    return episode_logs


def final_performance(episode_path, last_pct, last_n_episodes):
    """Mean episodic return and goal rate of one run over the end of its training.

    The end of training is the last `last_pct` fraction of the training steps, or the last `last_n_episodes`
    episodes when that number is given.
    """
    episodes = []
    with open(episode_path, "r", encoding="utf-8") as episode_file:
        for row in csv.DictReader(episode_file):
            episodes.append({
                "global_step": float(row.get("global_step", len(episodes))),
                "episodic_return": float(row["episodic_return"]),
                "goal_reached": float(row["goal_reached"]),
            })
    if not episodes:
        return {"mean_reward": np.nan, "mean_goal_rate": np.nan, "eval_episodes": 0}

    if last_n_episodes is not None and last_n_episodes > 0:
        final_episodes = episodes[-last_n_episodes:]
    else:
        last_step = max(episode["global_step"] for episode in episodes)
        cutoff_step = last_step * (1.0 - last_pct)
        final_episodes = [episode for episode in episodes if episode["global_step"] >= cutoff_step]
        if not final_episodes:
            final_episodes = episodes[-max(1, int(len(episodes) * last_pct)):]

    return {
        "mean_reward": float(np.mean([episode["episodic_return"] for episode in final_episodes])),
        "mean_goal_rate": float(np.mean([episode["goal_reached"] for episode in final_episodes])),
        "eval_episodes": len(final_episodes),
    }


def compare_metric(baseline_values, shaped_values):
    """Baseline vs shaped agent on one metric, given one value per matched seed (same seed order in both lists).

    Returns the mean and SD of each agent, the relative improvement of the shaped agent, and the two-sided
    Wilcoxon signed-rank test on the paired values: statistic W, p-value, Z and effect size r = |Z| / sqrt(N).
    """
    baseline = np.array(baseline_values)
    shaped = np.array(shaped_values)

    baseline_mean = np.mean(baseline_values)
    shaped_mean = np.mean(shaped_values)
    if baseline_mean != 0:
        pct_improvement = (shaped_mean - baseline_mean) / abs(baseline_mean) * 100
    else:
        pct_improvement = 0.0

    if np.all(shaped - baseline == 0):
        # both agents have the same value on every seed: the test is undefined
        statistic, p_value, z_score, effect_size = np.nan, 1.0, 0.0, 0.0
    else:
        try:
            test = wilcoxon(baseline, shaped, alternative="two-sided", method="exact")
        except ValueError:
            test = wilcoxon(baseline, shaped, alternative="two-sided")
        statistic = test.statistic
        p_value = test.pvalue
        z_score = norm.ppf(1 - p_value / 2)         # the normal quantile that has this two-sided p-value
        effect_size = round(abs(z_score) / np.sqrt(len(baseline)), 4)
        z_score = round(z_score, 4)

    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"

    if abs(effect_size) >= 0.5:
        effect_label = "Large"
    elif abs(effect_size) >= 0.3:
        effect_label = "Medium"
    elif abs(effect_size) >= 0.1:
        effect_label = "Small"
    else:
        effect_label = "Negligible"

    return {
        "baseline_values": baseline_values,
        "shaped_values": shaped_values,
        "baseline_mean": round(baseline_mean, 4),
        "baseline_std": round(np.std(baseline_values, ddof=1), 4),
        "shaped_mean": round(shaped_mean, 4),
        "shaped_std": round(np.std(shaped_values, ddof=1), 4),
        "pct_improvement": round(pct_improvement, 2),
        "W": statistic,
        "Z": z_score,
        "p_value": p_value,
        "effect_size": effect_size,
        "significance": significance,
        "effect_label": effect_label,
    }


def write_markdown_report(comparisons, output_path, evaluation_window):
    """The comparison tables of every environment as one Markdown file."""
    lines = [
        "# Statistical Analysis: Baseline DDQN vs RS-DDQN",
        "",
        "> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)",
        "> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant",
        "> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)",
        f"> **Metrics**: Averaged over the last {evaluation_window} of training steps for each seed's training run",
        "",
    ]

    for comparison in comparisons:
        seeds = comparison["seeds"]
        reward = comparison["reward"]
        goal = comparison["goal"]

        lines.append(f"## {comparison['env_id'].replace('MiniGrid-', '')}")
        lines.append(f"**Seeds**: {seeds}   |   **N** = {len(seeds)}")
        lines.append("")

        lines.append("### Summary Comparison")
        lines.append("")
        lines.append("| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |")
        lines.append("|--------|:------------:|:------:|:-------------:|")
        lines.append(f"| Mean Reward (±SD) | {reward['baseline_mean']:.2f} ± {reward['baseline_std']:.2f} "
                     f"| {reward['shaped_mean']:.2f} ± {reward['shaped_std']:.2f} | {reward['pct_improvement']:+.1f}% |")
        lines.append(f"| Goal Rate (±SD) | {goal['baseline_mean']:.3f} ± {goal['baseline_std']:.3f} "
                     f"| {goal['shaped_mean']:.3f} ± {goal['shaped_std']:.3f} | {goal['pct_improvement']:+.1f}% |")
        lines.append("")

        lines.append("### Wilcoxon Signed-Rank Test Results")
        lines.append("")
        lines.append("| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |")
        lines.append("|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|")
        for metric_name, statistics in (("Episodic Reward", reward), ("Goal Success Rate", goal)):
            w_text = "—" if np.isnan(statistics["W"]) else f"{statistics['W']:.1f}"
            lines.append(f"| {metric_name} | {w_text} | {statistics['Z']:.3f} | {statistics['p_value']:.4f} | {statistics['significance']} "
                         f"| {statistics['effect_size']:.3f} | {statistics['effect_label']} |")
        lines.append("")

        lines.append("### Per-Seed Raw Data")
        lines.append("")
        lines.append("| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |")
        lines.append("|:----:|:---------------:|:--------------:|:-------------:|:------------:|")
        for position, seed in enumerate(seeds):
            lines.append(f"| {seed} | {reward['baseline_values'][position]:.2f} | {reward['shaped_values'][position]:.2f} "
                         f"| {goal['baseline_values'][position]:.3f} | {goal['shaped_values'][position]:.3f} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
    print(f"  Markdown report saved → {output_path}")


def write_csv_and_excel_reports(comparisons, stats_dir):
    """One summary row per environment and one row per (environment, seed), each as CSV and as Excel."""
    summary_rows = []
    per_seed_rows = []
    for comparison in comparisons:
        seeds = comparison["seeds"]
        reward = comparison["reward"]
        goal = comparison["goal"]

        # columns: baseline_reward_mean, ..., reward_effect_label, then the same eleven columns for the goal rate
        summary_row = {"env_id": comparison["env_id"], "N": len(seeds)}
        for metric, statistics in (("reward", reward), ("goal", goal)):
            summary_row[f"baseline_{metric}_mean"] = statistics["baseline_mean"]
            summary_row[f"baseline_{metric}_std"] = statistics["baseline_std"]
            summary_row[f"shaped_{metric}_mean"] = statistics["shaped_mean"]
            summary_row[f"shaped_{metric}_std"] = statistics["shaped_std"]
            summary_row[f"{metric}_pct_improvement"] = statistics["pct_improvement"]
            summary_row[f"{metric}_W"] = statistics["W"]
            summary_row[f"{metric}_Z"] = statistics["Z"]
            summary_row[f"{metric}_p"] = statistics["p_value"]
            summary_row[f"{metric}_r"] = statistics["effect_size"]
            summary_row[f"{metric}_sig"] = statistics["significance"]
            summary_row[f"{metric}_effect_label"] = statistics["effect_label"]
        summary_rows.append(summary_row)

        for position, seed in enumerate(seeds):
            per_seed_rows.append({
                "env_id": comparison["env_id"],
                "seed": seed,
                "baseline_mean_reward": reward["baseline_values"][position],
                "rs_ddqn_mean_reward": reward["shaped_values"][position],
                "reward_diff": reward["shaped_values"][position] - reward["baseline_values"][position],
                "baseline_goal_rate": goal["baseline_values"][position],
                "rs_ddqn_goal_rate": goal["shaped_values"][position],
                "goal_rate_diff": goal["shaped_values"][position] - goal["baseline_values"][position],
                "baseline_eval_episodes": comparison["baseline_eval_episodes"][position],
                "rs_ddqn_eval_episodes": comparison["shaped_eval_episodes"][position],
            })

    summary_table = pd.DataFrame(summary_rows)
    per_seed_table = pd.DataFrame(per_seed_rows)

    summary_table.to_csv(stats_dir / "statistical_summary.csv", index=False)
    per_seed_table.to_csv(stats_dir / "per_seed_metrics.csv", index=False)
    print(f"  CSV summary saved    → {stats_dir / 'statistical_summary.csv'}")
    print(f"  CSV per-seed saved   → {stats_dir / 'per_seed_metrics.csv'}")

    # The CSV files above are already written, so a missing Excel engine only costs the two workbooks.
    try:
        with pd.ExcelWriter(stats_dir / "statistical_analysis.xlsx", engine="openpyxl") as workbook:
            summary_table.to_excel(workbook, sheet_name="Summary", index=False)
            per_seed_table.to_excel(workbook, sheet_name="Per-Seed Metrics", index=False)
        per_seed_table.to_excel(stats_dir / "per_seed_metrics.xlsx", engine="openpyxl", index=False)
        print(f"  Excel workbook saved → {stats_dir / 'statistical_analysis.xlsx'}")
        print(f"  Excel per-seed saved → {stats_dir / 'per_seed_metrics.xlsx'}")
    except Exception as error:
        print(f"  ⚠ Could not write Excel workbook: {error}")


def main():
    parser = argparse.ArgumentParser(description="Statistical comparison of Baseline DDQN vs RS-DDQN across seeds.")
    parser.add_argument("--env-id", type=str, default=None,
                        help="Specific environment to analyse (default: auto-discover all)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="plots")
    parser.add_argument("--last-pct", type=float, default=0.10,
                        help="Fraction of final training steps to evaluate over (default: 0.10 for last 10%%)")
    parser.add_argument("--last-n-episodes", type=int, default=None,
                        help="Optional override to average over fixed N final episodes instead of step percentage")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory '{results_dir}' does not exist. Exiting.")
        sys.exit(1)

    if args.env_id:
        env_ids = [args.env_id]
    else:
        # run folders are named <env_id>__<exp_name>__<seed>__<timestamp>
        env_ids = set()
        for run_dir in results_dir.iterdir():
            if run_dir.is_dir() and "__" in run_dir.name:
                env_ids.add(run_dir.name.split("__")[0])
        env_ids = sorted(env_ids)
    if not env_ids:
        print("No experiment results found. Exiting.")
        sys.exit(1)

    if args.last_n_episodes:
        evaluation_window = f"fixed last {args.last_n_episodes} episodes"
    else:
        evaluation_window = f"last {args.last_pct * 100:g}% of training steps"
    print("=" * 70)
    print("STATISTICAL ANALYSIS: Baseline DDQN vs RS-DDQN")
    print("=" * 70)
    print(f"Results directory : {results_dir}")
    print(f"Evaluation window : {evaluation_window}")
    print(f"Environments      : {env_ids}")
    print()

    comparisons = []
    for env_id in env_ids:
        print(f"--- {env_id} ---")

        # each agent is looked up under its current experiment name first, then under older names
        baseline_logs = discover_runs(results_dir, env_id, "ddqn_baseline")
        if not baseline_logs:
            baseline_logs = discover_runs(results_dir, env_id, "dqn_baseline")
        shaped_logs = discover_runs(results_dir, env_id, "ddqn_reward_shaping")
        if not shaped_logs:
            shaped_logs = discover_runs(results_dir, env_id, "dqn_reward_shaping")
        if not shaped_logs:
            shaped_logs = discover_runs(results_dir, env_id, "dqn_vqvae_masked")

        common_seeds = sorted(set(baseline_logs) & set(shaped_logs))
        if len(common_seeds) < 2:
            print(f"  ⚠ Need at least 2 matched seeds, found {len(common_seeds)}. Skipping.\n")
            continue
        print(f"  Matched seeds: {common_seeds} (N={len(common_seeds)})")

        baseline_runs = [final_performance(baseline_logs[seed], args.last_pct, args.last_n_episodes) for seed in common_seeds]
        shaped_runs = [final_performance(shaped_logs[seed], args.last_pct, args.last_n_episodes) for seed in common_seeds]
        reward = compare_metric([run["mean_reward"] for run in baseline_runs], [run["mean_reward"] for run in shaped_runs])
        goal = compare_metric([run["mean_goal_rate"] for run in baseline_runs], [run["mean_goal_rate"] for run in shaped_runs])
        comparisons.append({
            "env_id": env_id,
            "seeds": common_seeds,
            "reward": reward,
            "goal": goal,
            "baseline_eval_episodes": [run["eval_episodes"] for run in baseline_runs],
            "shaped_eval_episodes": [run["eval_episodes"] for run in shaped_runs],
        })

        print(f"  Reward : Baseline {reward['baseline_mean']:.2f} vs RS-DDQN {reward['shaped_mean']:.2f} "
              f"({reward['pct_improvement']:+.1f}%) p={reward['p_value']:.4f} {reward['significance']}")
        print(f"  Goal   : Baseline {goal['baseline_mean']:.3f} vs RS-DDQN {goal['shaped_mean']:.3f} "
              f"({goal['pct_improvement']:+.1f}%) p={goal['p_value']:.4f} {goal['significance']}")
        print()

    if not comparisons:
        print("No valid comparisons could be made. Exiting.")
        sys.exit(1)

    stats_dir = Path(args.output_dir) / "statistical_analysis"
    stats_dir.mkdir(parents=True, exist_ok=True)
    write_markdown_report(comparisons, stats_dir / "statistical_report.md", f"{args.last_pct * 100:g}%")
    write_csv_and_excel_reports(comparisons, stats_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
