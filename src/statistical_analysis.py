"""
statistical_analysis.py — Statistical Comparison
==================================================================
Compares Baseline DDQN vs RS-DDQN across multiple seeds for each environment.

For every environment, this script:
  1. Loads all per-seed episode logs (episode_log.csv)
  2. Computes per-seed summary metrics (mean reward, mean goal rate)
  3. Runs the Wilcoxon signed-rank test (paired, non-parametric)
  4. Computes effect size r = |Z| / sqrt(N)
  5. Outputs a summary table as both CSV and Markdown

Usage:
  python src/statistical_analysis.py --results-dir results --output-dir plots
  python src/statistical_analysis.py --env-id MiniGrid-DoorKey-6x6-v0

The Wilcoxon signed-rank test is chosen because:
  - It is non-parametric (no normality assumption on reward distributions)
  - It is paired (each seed is a matched pair: baseline vs shaped)
  - It is appropriate for small sample sizes (N = 3..10 seeds)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test (scipy)
# ---------------------------------------------------------------------------
try:
    from scipy.stats import wilcoxon
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy")
    sys.exit(1)


# ============================================================================
# DATA LOADING
# ============================================================================

def discover_runs(results_dir, env_id, exp_name):
    """
    Finds all run directories for a given (env_id, exp_name) pair.

    Returns
    -------
    dict[int, Path]
        Mapping from seed number → run directory path.
    """
    runs = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}__*")):
        config_path = run_dir / "config.json"
        episode_path = run_dir / "episode_log.csv"

        if not config_path.exists() or not episode_path.exists():
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            seed = int(config["seed"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

        runs[seed] = run_dir
    return runs


def load_episode_metrics(run_dir, last_n_episodes=100):
    """
    Loads the episode log and returns summary metrics for the LAST N episodes.

    Parameters
    ----------
    run_dir : Path
        Path to the run directory containing episode_log.csv.
    last_n_episodes : int
        Number of final episodes to average over (default: 100).
        This captures the agent's converged performance, not early exploration.

    Returns
    -------
    dict with keys: mean_reward, mean_goal_rate, total_episodes
    """
    episode_path = run_dir / "episode_log.csv"
    rows = []
    with open(episode_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "episodic_return": float(row["episodic_return"]),
                "goal_reached":   float(row["goal_reached"]),
            })

    if not rows:
        return {"mean_reward": np.nan, "mean_goal_rate": np.nan, "total_episodes": 0}

    # Take the last N episodes for converged performance
    tail = rows[-last_n_episodes:]

    return {
        "mean_reward":    np.mean([r["episodic_return"] for r in tail]),
        "mean_goal_rate": np.mean([r["goal_reached"]   for r in tail]),
        "total_episodes": len(rows),
    }


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def run_wilcoxon_test(baseline_values, shaped_values):
    """
    Runs a two-sided Wilcoxon signed-rank test on paired observations.

    Parameters
    ----------
    baseline_values : array-like
        Per-seed metric values for the Baseline DDQN agent.
    shaped_values : array-like
        Per-seed metric values for the RS-DDQN agent (same seed order).

    Returns
    -------
    dict with keys: W, p_value, Z, r_effect_size, N, significant_005, significant_001
    """
    baseline = np.array(baseline_values)
    shaped   = np.array(shaped_values)
    N = len(baseline)

    differences = shaped - baseline

    # If all differences are zero, the test is undefined
    if np.all(differences == 0):
        return {
            "W": np.nan, "p_value": 1.0, "Z": 0.0,
            "r_effect_size": 0.0, "N": N,
            "significant_005": False, "significant_001": False,
        }

    # Wilcoxon signed-rank test (two-sided)
    # method="exact" for small N, falls back to "approx" if N > 25
    try:
        result = wilcoxon(baseline, shaped, alternative="two-sided", method="exact")
    except ValueError:
        # If exact method fails (e.g., ties), fall back to approximate
        result = wilcoxon(baseline, shaped, alternative="two-sided")

    W = result.statistic
    p = result.pvalue

    # Effect size: r = |Z| / sqrt(N)
    # For small samples, approximate Z from the p-value using the normal inverse
    from scipy.stats import norm
    Z = norm.ppf(1 - p / 2)  # two-tailed Z approximation
    r = abs(Z) / np.sqrt(N)

    return {
        "W": W,
        "p_value": p,
        "Z": round(Z, 4),
        "r_effect_size": round(r, 4),
        "N": N,
        "significant_005": p < 0.05,
        "significant_001": p < 0.01,
    }


def significance_label(p_value):
    """Returns a human-readable significance label."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def effect_size_label(r):
    """Cohen's benchmarks for effect size r."""
    r = abs(r)
    if r >= 0.5:
        return "Large"
    elif r >= 0.3:
        return "Medium"
    elif r >= 0.1:
        return "Small"
    else:
        return "Negligible"


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_comparison_table(env_id, baseline_metrics, shaped_metrics, common_seeds):
    """
    Builds a complete comparison summary for one environment.

    Returns a dict containing all the statistics needed for the table.
    """
    b_rewards = [baseline_metrics[s]["mean_reward"]    for s in common_seeds]
    s_rewards = [shaped_metrics[s]["mean_reward"]      for s in common_seeds]
    b_goals   = [baseline_metrics[s]["mean_goal_rate"] for s in common_seeds]
    s_goals   = [shaped_metrics[s]["mean_goal_rate"]   for s in common_seeds]

    # Descriptive statistics
    b_reward_mean, b_reward_std = np.mean(b_rewards), np.std(b_rewards, ddof=1)
    s_reward_mean, s_reward_std = np.mean(s_rewards), np.std(s_rewards, ddof=1)
    b_goal_mean,   b_goal_std  = np.mean(b_goals),   np.std(b_goals, ddof=1)
    s_goal_mean,   s_goal_std  = np.mean(s_goals),   np.std(s_goals, ddof=1)

    # Percentage improvement
    reward_pct = ((s_reward_mean - b_reward_mean) / abs(b_reward_mean) * 100
                  if b_reward_mean != 0 else 0.0)
    goal_pct   = ((s_goal_mean - b_goal_mean) / abs(b_goal_mean) * 100
                  if b_goal_mean != 0 else 0.0)

    # Wilcoxon tests
    reward_test = run_wilcoxon_test(b_rewards, s_rewards)
    goal_test   = run_wilcoxon_test(b_goals, s_goals)

    return {
        "env_id": env_id,
        "N": len(common_seeds),
        "seeds": common_seeds,

        # Reward statistics
        "baseline_reward_mean": round(b_reward_mean, 4),
        "baseline_reward_std":  round(b_reward_std, 4),
        "shaped_reward_mean":   round(s_reward_mean, 4),
        "shaped_reward_std":    round(s_reward_std, 4),
        "reward_pct_improvement": round(reward_pct, 2),
        "reward_W":             reward_test["W"],
        "reward_Z":             reward_test["Z"],
        "reward_p":             reward_test["p_value"],
        "reward_r":             reward_test["r_effect_size"],
        "reward_sig":           significance_label(reward_test["p_value"]),
        "reward_effect_label":  effect_size_label(reward_test["r_effect_size"]),

        # Goal rate statistics
        "baseline_goal_mean":   round(b_goal_mean, 4),
        "baseline_goal_std":    round(b_goal_std, 4),
        "shaped_goal_mean":     round(s_goal_mean, 4),
        "shaped_goal_std":      round(s_goal_std, 4),
        "goal_pct_improvement": round(goal_pct, 2),
        "goal_W":               goal_test["W"],
        "goal_Z":               goal_test["Z"],
        "goal_p":               goal_test["p_value"],
        "goal_r":               goal_test["r_effect_size"],
        "goal_sig":             significance_label(goal_test["p_value"]),
        "goal_effect_label":    effect_size_label(goal_test["r_effect_size"]),

        # Raw per-seed data for transparency
        "per_seed_baseline_rewards": b_rewards,
        "per_seed_shaped_rewards":   s_rewards,
        "per_seed_baseline_goals":   b_goals,
        "per_seed_shaped_goals":     s_goals,
    }


def write_markdown_report(results, output_path):
    """
    Writes a publication-ready Markdown report with all statistical tables.
    """
    lines = [
        "# Statistical Analysis: Baseline DDQN vs RS-DDQN",
        "",
        "> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)",
        "> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant",
        "> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)",
        "> **Metrics**: Averaged over the last 100 episodes of each seed's training run",
        "",
    ]

    for res in results:
        env_short = res["env_id"].replace("MiniGrid-", "")
        lines.append(f"## {env_short}")
        lines.append(f"**Seeds**: {res['seeds']}   |   **N** = {res['N']}")
        lines.append("")

        # Summary comparison table
        lines.append("### Summary Comparison")
        lines.append("")
        lines.append("| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |")
        lines.append("|--------|:------------:|:------:|:-------------:|")
        lines.append(
            f"| Mean Reward (±SD) | {res['baseline_reward_mean']:.2f} ± {res['baseline_reward_std']:.2f} "
            f"| {res['shaped_reward_mean']:.2f} ± {res['shaped_reward_std']:.2f} "
            f"| {res['reward_pct_improvement']:+.1f}% |"
        )
        lines.append(
            f"| Goal Rate (±SD) | {res['baseline_goal_mean']:.3f} ± {res['baseline_goal_std']:.3f} "
            f"| {res['shaped_goal_mean']:.3f} ± {res['shaped_goal_std']:.3f} "
            f"| {res['goal_pct_improvement']:+.1f}% |"
        )
        lines.append("")

        # Statistical test results
        lines.append("### Wilcoxon Signed-Rank Test Results")
        lines.append("")
        lines.append("| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |")
        lines.append("|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|")

        reward_W = f"{res['reward_W']:.1f}" if not np.isnan(res['reward_W']) else "—"
        goal_W   = f"{res['goal_W']:.1f}"   if not np.isnan(res['goal_W'])   else "—"

        lines.append(
            f"| Episodic Reward | {reward_W} | {res['reward_Z']:.3f} "
            f"| {res['reward_p']:.4f} | {res['reward_sig']} "
            f"| {res['reward_r']:.3f} | {res['reward_effect_label']} |"
        )
        lines.append(
            f"| Goal Success Rate | {goal_W} | {res['goal_Z']:.3f} "
            f"| {res['goal_p']:.4f} | {res['goal_sig']} "
            f"| {res['goal_r']:.3f} | {res['goal_effect_label']} |"
        )
        lines.append("")

        # Per-seed raw data
        lines.append("### Per-Seed Raw Data")
        lines.append("")
        lines.append("| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |")
        lines.append("|:----:|:---------------:|:--------------:|:-------------:|:------------:|")
        for i, seed in enumerate(res["seeds"]):
            lines.append(
                f"| {seed} "
                f"| {res['per_seed_baseline_rewards'][i]:.2f} "
                f"| {res['per_seed_shaped_rewards'][i]:.2f} "
                f"| {res['per_seed_baseline_goals'][i]:.3f} "
                f"| {res['per_seed_shaped_goals'][i]:.3f} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Markdown report saved → {output_path}")


def write_csv_summary(results, output_path):
    """
    Writes a flat CSV file with one row per environment for easy import into
    Excel, LaTeX, or statistical software.
    """
    fieldnames = [
        "env_id", "N",
        "baseline_reward_mean", "baseline_reward_std",
        "shaped_reward_mean", "shaped_reward_std",
        "reward_pct_improvement", "reward_W", "reward_Z",
        "reward_p", "reward_r", "reward_sig", "reward_effect_label",
        "baseline_goal_mean", "baseline_goal_std",
        "shaped_goal_mean", "shaped_goal_std",
        "goal_pct_improvement", "goal_W", "goal_Z",
        "goal_p", "goal_r", "goal_sig", "goal_effect_label",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"  CSV summary saved    → {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of Baseline DDQN vs RS-DDQN across seeds."
    )
    parser.add_argument("--env-id", type=str, default=None,
                        help="Specific environment to analyse (default: auto-discover all)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="plots")
    parser.add_argument("--last-n-episodes", type=int, default=100,
                        help="Number of final episodes to average for converged performance")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)

    # Auto-discover environments if not specified
    if args.env_id:
        env_ids = [args.env_id]
    else:
        env_ids = set()
        for run_dir in results_dir.iterdir():
            if run_dir.is_dir() and "__" in run_dir.name:
                env_id = run_dir.name.split("__")[0]
                env_ids.add(env_id)
        env_ids = sorted(env_ids)

    if not env_ids:
        print("No experiment results found. Exiting.")
        sys.exit(1)

    print("=" * 70)
    print("STATISTICAL ANALYSIS: Baseline DDQN vs RS-DDQN")
    print("=" * 70)
    print(f"Results directory : {results_dir}")
    print(f"Last N episodes   : {args.last_n_episodes}")
    print(f"Environments      : {env_ids}")
    print()

    all_results = []

    for env_id in env_ids:
        print(f"--- {env_id} ---")

        # Discover runs for both agents
        baseline_runs = discover_runs(results_dir, env_id, "ddqn_baseline")
        if not baseline_runs:
            baseline_runs = discover_runs(results_dir, env_id, "dqn_baseline")

        shaped_runs = discover_runs(results_dir, env_id, "ddqn_reward_shaping")
        if not shaped_runs:
            shaped_runs = discover_runs(results_dir, env_id, "dqn_reward_shaping")

        common_seeds = sorted(set(baseline_runs.keys()) & set(shaped_runs.keys()))

        if len(common_seeds) < 2:
            print(f"  ⚠ Need at least 2 matched seeds, found {len(common_seeds)}. Skipping.\n")
            continue

        print(f"  Matched seeds: {common_seeds} (N={len(common_seeds)})")

        # Load metrics
        baseline_metrics = {}
        shaped_metrics   = {}
        for seed in common_seeds:
            baseline_metrics[seed] = load_episode_metrics(
                baseline_runs[seed], args.last_n_episodes
            )
            shaped_metrics[seed] = load_episode_metrics(
                shaped_runs[seed], args.last_n_episodes
            )

        # Generate comparison
        result = generate_comparison_table(env_id, baseline_metrics, shaped_metrics, common_seeds)
        all_results.append(result)

        # Print quick summary to console
        print(f"  Reward : Baseline {result['baseline_reward_mean']:.2f} vs "
              f"RS-DDQN {result['shaped_reward_mean']:.2f} "
              f"({result['reward_pct_improvement']:+.1f}%) "
              f"p={result['reward_p']:.4f} {result['reward_sig']}")
        print(f"  Goal   : Baseline {result['baseline_goal_mean']:.3f} vs "
              f"RS-DDQN {result['shaped_goal_mean']:.3f} "
              f"({result['goal_pct_improvement']:+.1f}%) "
              f"p={result['goal_p']:.4f} {result['goal_sig']}")
        print()

    if not all_results:
        print("No valid comparisons could be made. Exiting.")
        sys.exit(1)

    # Write outputs
    stats_dir = output_dir / "statistical_analysis"
    write_markdown_report(all_results, stats_dir / "statistical_report.md")
    write_csv_summary(all_results, stats_dir / "statistical_summary.csv")

    print("\nAll done.")


if __name__ == "__main__":
    main()
