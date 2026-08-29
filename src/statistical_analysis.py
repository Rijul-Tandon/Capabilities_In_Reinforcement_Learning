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
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}*")):
        config_path = run_dir / "config.json"
        episode_path = run_dir / "episodes.csv"
        if not episode_path.exists():
            episode_path = run_dir / "episode_log.csv"

        if not config_path.exists() or not episode_path.exists():
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            seed_val = config.get("seed", 1)
            if isinstance(seed_val, list):
                if not seed_val: continue
                seed = int(seed_val[0])
            else:
                seed = int(seed_val)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue

        runs[seed] = run_dir
    return runs


import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)

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
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}*")):
        config_path = run_dir / "config.json"
        episode_path = run_dir / "episodes.csv"
        if not episode_path.exists():
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


def load_episode_metrics(run_dir, last_pct=0.10, last_n_episodes=None):
    """
    Loads the episode log and returns summary metrics for the LAST 10% of training steps
    (or last N episodes if last_n_episodes is explicitly set).

    Parameters
    ----------
    run_dir : Path
        Path to the run directory containing episodes.csv or episode_log.csv.
    last_pct : float
        Fraction of final steps to evaluate over (default: 0.10 for last 10%).
    last_n_episodes : int or None
        Optional fixed count of final episodes to average.

    Returns
    -------
    dict with keys: mean_reward, mean_goal_rate, total_episodes, eval_episodes, max_global_step
    """
    episode_path = run_dir / "episodes.csv"
    if not episode_path.exists():
        episode_path = run_dir / "episode_log.csv"

    rows = []
    with open(episode_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "global_step": float(row.get("global_step", len(rows))),
                "episodic_return": float(row["episodic_return"]),
                "goal_reached":   float(row["goal_reached"]),
            })

    if not rows:
        return {
            "mean_reward": np.nan,
            "mean_goal_rate": np.nan,
            "total_episodes": 0,
            "eval_episodes": 0,
            "max_global_step": 0,
        }

    if last_n_episodes is not None and last_n_episodes > 0:
        tail = rows[-last_n_episodes:]
    else:
        max_step = max(r["global_step"] for r in rows)
        cutoff_step = max_step * (1.0 - last_pct)
        tail = [r for r in rows if r["global_step"] >= cutoff_step]
        if not tail:
            num_tail = max(1, int(len(rows) * last_pct))
            tail = rows[-num_tail:]

    return {
        "mean_reward":    float(np.mean([r["episodic_return"] for r in tail])),
        "mean_goal_rate": float(np.mean([r["goal_reached"]   for r in tail])),
        "total_episodes": len(rows),
        "eval_episodes":  len(tail),
        "max_global_step": max(r["global_step"] for r in rows) if rows else 0,
    }


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def run_wilcoxon_test(baseline_values, shaped_values):
    """
    Runs a two-sided Wilcoxon signed-rank test on paired observations.
    """
    baseline = np.array(baseline_values)
    shaped   = np.array(shaped_values)
    N = len(baseline)

    differences = shaped - baseline

    if np.all(differences == 0):
        return {
            "W": np.nan, "p_value": 1.0, "Z": 0.0,
            "r_effect_size": 0.0, "N": N,
            "significant_005": False, "significant_001": False,
        }

    try:
        result = wilcoxon(baseline, shaped, alternative="two-sided", method="exact")
    except ValueError:
        result = wilcoxon(baseline, shaped, alternative="two-sided")

    W = result.statistic
    p = result.pvalue

    from scipy.stats import norm
    Z = norm.ppf(1 - p / 2)
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
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def effect_size_label(r):
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
    """
    b_rewards = [baseline_metrics[s]["mean_reward"]    for s in common_seeds]
    s_rewards = [shaped_metrics[s]["mean_reward"]      for s in common_seeds]
    b_goals   = [baseline_metrics[s]["mean_goal_rate"] for s in common_seeds]
    s_goals   = [shaped_metrics[s]["mean_goal_rate"]   for s in common_seeds]

    b_reward_mean, b_reward_std = np.mean(b_rewards), np.std(b_rewards, ddof=1)
    s_reward_mean, s_reward_std = np.mean(s_rewards), np.std(s_rewards, ddof=1)
    b_goal_mean,   b_goal_std  = np.mean(b_goals),   np.std(b_goals, ddof=1)
    s_goal_mean,   s_goal_std  = np.mean(s_goals),   np.std(s_goals, ddof=1)

    reward_pct = ((s_reward_mean - b_reward_mean) / abs(b_reward_mean) * 100
                  if b_reward_mean != 0 else 0.0)
    goal_pct   = ((s_goal_mean - b_goal_mean) / abs(b_goal_mean) * 100
                  if b_goal_mean != 0 else 0.0)

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

        # Raw per-seed data
        "per_seed_baseline_rewards": b_rewards,
        "per_seed_shaped_rewards":   s_rewards,
        "per_seed_baseline_goals":   b_goals,
        "per_seed_shaped_goals":     s_goals,
        "per_seed_baseline_eval_episodes": [baseline_metrics[s]["eval_episodes"] for s in common_seeds],
        "per_seed_shaped_eval_episodes":   [shaped_metrics[s]["eval_episodes"] for s in common_seeds],
    }


def write_markdown_report(results, output_path, last_pct_str="10%"):
    """
    Writes a publication-ready Markdown report with all statistical tables.
    """
    lines = [
        "# Statistical Analysis: Baseline DDQN vs RS-DDQN",
        "",
        "> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)",
        "> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant",
        "> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)",
        f"> **Metrics**: Averaged over the last {last_pct_str} of training steps for each seed's training run",
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


def write_excel_and_csv_reports(results, stats_dir):
    """
    Writes flat CSV summary files and Excel workbooks (.xlsx) containing both
    the aggregate statistical summary and full per-seed metrics.
    """
    stats_dir.mkdir(parents=True, exist_ok=True)

    summary_fieldnames = [
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
    summary_rows = [{k: res[k] for k in summary_fieldnames if k in res} for res in results]
    df_summary = pd.DataFrame(summary_rows)

    per_seed_rows = []
    for res in results:
        env_id = res["env_id"]
        for i, seed in enumerate(res["seeds"]):
            per_seed_rows.append({
                "env_id": env_id,
                "seed": seed,
                "baseline_mean_reward": res["per_seed_baseline_rewards"][i],
                "rs_ddqn_mean_reward": res["per_seed_shaped_rewards"][i],
                "reward_diff": res["per_seed_shaped_rewards"][i] - res["per_seed_baseline_rewards"][i],
                "baseline_goal_rate": res["per_seed_baseline_goals"][i],
                "rs_ddqn_goal_rate": res["per_seed_shaped_goals"][i],
                "goal_rate_diff": res["per_seed_shaped_goals"][i] - res["per_seed_baseline_goals"][i],
                "baseline_eval_episodes": res["per_seed_baseline_eval_episodes"][i],
                "rs_ddqn_eval_episodes": res["per_seed_shaped_eval_episodes"][i],
            })
    df_per_seed = pd.DataFrame(per_seed_rows)

    csv_summary_path = stats_dir / "statistical_summary.csv"
    csv_per_seed_path = stats_dir / "per_seed_metrics.csv"
    df_summary.to_csv(csv_summary_path, index=False)
    df_per_seed.to_csv(csv_per_seed_path, index=False)
    print(f"  CSV summary saved    → {csv_summary_path}")
    print(f"  CSV per-seed saved   → {csv_per_seed_path}")

    excel_path = stats_dir / "statistical_analysis.xlsx"
    excel_per_seed_path = stats_dir / "per_seed_metrics.xlsx"

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            df_per_seed.to_excel(writer, sheet_name="Per-Seed Metrics", index=False)
        df_per_seed.to_excel(excel_per_seed_path, engine="openpyxl", index=False)
        print(f"  Excel workbook saved → {excel_path}")
        print(f"  Excel per-seed saved → {excel_per_seed_path}")
    except Exception as e:
        print(f"  ⚠ Could not write Excel workbook: {e}")


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
    parser.add_argument("--last-pct", type=float, default=0.10,
                        help="Fraction of final training steps to evaluate over (default: 0.10 for last 10%%)")
    parser.add_argument("--last-n-episodes", type=int, default=None,
                        help="Optional override to average over fixed N final episodes instead of step percentage")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Results directory '{results_dir}' does not exist. Exiting.")
        sys.exit(1)

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

    eval_label = f"fixed last {args.last_n_episodes} episodes" if args.last_n_episodes else f"last {args.last_pct*100:g}% of training steps"

    print("=" * 70)
    print("STATISTICAL ANALYSIS: Baseline DDQN vs RS-DDQN")
    print("=" * 70)
    print(f"Results directory : {results_dir}")
    print(f"Evaluation window : {eval_label}")
    print(f"Environments      : {env_ids}")
    print()

    all_results = []

    for env_id in env_ids:
        print(f"--- {env_id} ---")

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

        baseline_metrics = {}
        shaped_metrics   = {}
        for seed in common_seeds:
            baseline_metrics[seed] = load_episode_metrics(
                baseline_runs[seed], last_pct=args.last_pct, last_n_episodes=args.last_n_episodes
            )
            shaped_metrics[seed] = load_episode_metrics(
                shaped_runs[seed], last_pct=args.last_pct, last_n_episodes=args.last_n_episodes
            )

        result = generate_comparison_table(env_id, baseline_metrics, shaped_metrics, common_seeds)
        all_results.append(result)

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

    stats_dir = output_dir / "statistical_analysis"
    write_markdown_report(all_results, stats_dir / "statistical_report.md", last_pct_str=f"{args.last_pct*100:g}%")
    write_excel_and_csv_reports(all_results, stats_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
