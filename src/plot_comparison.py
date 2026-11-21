"""Training curves of all agents on one environment: episodic return, goal-reach rate and TD loss over the training
steps, per seed and as the mean over seeds.  The other modes (--decay-comparison, --combine-all-runs,
--mean-reward-only, --per-seed-mean-reward-only) are described in the argument help."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

# Colour of each agent, used when a figure shows at most three experiments
AGENT_COLORS = {
    "ddqn_baseline":         "#1f77b4",  # blue
    "ddqn_vqvae_masked":     "#ff7f0e",  # orange
    "ddqn_reward_shaping":   "#ff7f0e",  # orange
    "ddqn_learned_penalty":  "#2ca02c",  # green
    "dqn_baseline":          "#1f77b4",  # blue
    "dqn_reward_shaping":    "#ff7f0e",  # orange
    "dqn_vqvae_masked":      "#ff7f0e",  # orange
    "dqn_vanilla":           "#2ca02c",  # green
}
# Otherwise every experiment gets its own colour, in alphabetical order of the experiment names
DISTINCT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
# Runs of the same experiment in one figure differ by line style
LINE_STYLES = ["-", "--", ":", "-."]

# metric -> (plot title, y-axis label, y-axis limits, ending of the output file name)
METRICS = {
    "episodic_return": ("Episodic Return", "Episodic Return", None, "episodic_return"),
    "goal_reached":    ("Goal Reach Rate", "Goal Success Rate", (-0.05, 1.05), "goal_reach"),
    "td_loss":         ("TD Loss", "TD Loss", None, "td_loss"),
}


def load_runs(results_dir, env_id, exp_name_contains):
    """{experiment name: [run, ...]} for one environment, with the latest run of every (experiment, seed).

    A run is a dict of the run's seed and total_timesteps and the logged columns as lists:
    episode_steps, episodic_return, goal_reached (episodes.csv) and metric_steps, td_loss (metrics.csv).
    """
    runs_by_exp_and_seed = {}
    # An Atari id such as ALE/DemonAttack-v5 is stored with "_" in place of "/" in the run-folder name.
    for run_dir in sorted(Path(results_dir).glob(f"{env_id.replace('/', '_')}__*")):
        if not (run_dir / "config.json").exists() or not (run_dir / "episodes.csv").exists():
            continue
        with open(run_dir / "config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        exp_name = config.get("exp_name", "unknown")
        if exp_name == "random_agent":
            continue
        if exp_name_contains and not any(text in exp_name for text in exp_name_contains):
            continue

        seed = config.get("seed", 1)
        if isinstance(seed, list):      # runs that trained on several seeds are filed under the first one
            seed = seed[0] if seed else 1
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            seed = 1

        run = {"seed": seed, "total_timesteps": config.get("total_timesteps", 200000),
               "episode_steps": [], "episodic_return": [], "goal_reached": [], "metric_steps": [], "td_loss": []}

        with open(run_dir / "episodes.csv", "r", encoding="utf-8") as episode_file:
            for row in csv.DictReader(episode_file):
                if not row.get("global_step"):
                    continue
                try:
                    step = int(float(row["global_step"]))
                    episodic_return = float(row.get("episodic_return", 0.0))
                    goal_reached = float(row.get("goal_reached", 0.0))      # logs without this column count as "not reached"
                except ValueError:
                    continue
                run["episode_steps"].append(step)
                run["episodic_return"].append(episodic_return)
                run["goal_reached"].append(goal_reached)
        if not run["episode_steps"]:
            continue

        if (run_dir / "metrics.csv").exists():
            with open(run_dir / "metrics.csv", "r", encoding="utf-8") as metric_file:
                for row in csv.DictReader(metric_file):
                    if not row.get("global_step"):
                        continue
                    try:
                        td_loss = float(row.get("td_loss", "nan"))
                    except (ValueError, TypeError):
                        td_loss = float("nan")
                    try:
                        step = int(float(row["global_step"]))
                    except ValueError:
                        continue
                    run["metric_steps"].append(step)
                    run["td_loss"].append(td_loss)

        # run folders are read in sorted order, so a later run of the same experiment and seed replaces the earlier one
        runs_by_exp_and_seed.setdefault(exp_name, {})[seed] = run

    runs_by_exp = {}
    for exp_name, runs_by_seed in runs_by_exp_and_seed.items():
        runs_by_exp[exp_name] = list(runs_by_seed.values())
    return runs_by_exp


def select_seed(runs_by_exp, seed):
    """The same experiments restricted to the runs of one seed; experiments without that seed are left out."""
    selected = {}
    for exp_name, runs in runs_by_exp.items():
        seed_runs = [run for run in runs if run["seed"] == seed]
        if seed_runs:
            selected[exp_name] = seed_runs
    return selected


def legend_label(exp_name):
    """Readable legend label of an experiment name, e.g.
        ddqn_baseline_decay_linear                  -> Baseline DDQN (linear)
        ddqn_reward_shaping                         -> RS-DDQN
        ddqn_baseline_transfer_doorkey_to_empty     -> Baseline DDQN | Transfer Doorkey → Empty
    """
    if "_transfer_" in exp_name:
        agent = "RS-DDQN" if "reward_shaping" in exp_name else "Baseline DDQN"
        direction = exp_name.split("_transfer_", 1)[1].replace("_to_", " → ")
        return f"{agent} | Transfer {direction.title()}"
    if "_scratch_" in exp_name:
        agent = "RS-DDQN" if "reward_shaping" in exp_name else "Baseline DDQN"
        context = exp_name.split("_scratch_", 1)[1].replace("_for_", " (for ").replace("_transfer", ")")
        return f"{agent} | Fresh {context.replace('_', ' ').title()}"

    label = exp_name
    if "ddqn_vqvae_masked" in label:
        label = label.replace("ddqn_vqvae_masked", "RS-DDQN (VQ-VAE Masked)")
    elif "dqn_vqvae_masked" in label:
        label = label.replace("dqn_vqvae_masked", "RS-DQN (VQ-VAE Masked)")
    if "learned_penalty" in label:
        label = label.replace("ddqn_learned_penalty", "LP-DDQN (learned penalty)")
    if "reward_shaping" in label:
        label = label.replace("ddqn_reward_shaping", "RS-DDQN").replace("dqn_reward_shaping", "RS-DDQN")
    elif "baseline" in label:
        label = label.replace("ddqn_baseline", "Baseline DDQN").replace("dqn_baseline", "Baseline DQN")

    if "_decay_" in exp_name:
        name_before_decay, decay_schedule = exp_name.split("_decay_")[:2]
        if "RS-DDQN" in label:
            return f"RS-DDQN ({decay_schedule})"
        if "Baseline DDQN" in label:
            return f"Baseline DDQN ({decay_schedule})"
        if "Baseline DQN" in label:
            return f"Baseline DQN ({decay_schedule})"
        return f"{name_before_decay.replace('_', ' ').title()} ({decay_schedule})"
    if label == exp_name:       # no known agent name in it
        return label.replace("_", " ").title()
    return label


def rolling(values, window, statistic):
    """`statistic` (np.mean or np.std) of the last `window` values at every position; fewer values at the start."""
    return np.array([float(statistic(values[max(0, end - window):end])) for end in range(1, len(values) + 1)])


def plot_metric(runs_by_exp, env_id, output_path, rolling_window, metric, title_suffix, mean_over_seeds):
    """One PNG with the curves of one metric over the training steps.

    mean_over_seeds = False:  one curve per run, the rolling mean, with a band of +- half the rolling standard deviation
    mean_over_seeds = True:   one curve per experiment, the mean of its runs' rolling means, with a band of +- 1
                              standard deviation across the runs
    """
    title, y_label, y_limits, _ = METRICS[metric]
    if mean_over_seeds:
        title = f"Mean {title}"
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    total_timesteps = 200000        # right end of the x-axis; replaced by the value of the runs that are drawn

    for exp_name, runs in runs_by_exp.items():
        if exp_name in AGENT_COLORS and len(runs_by_exp) <= 3:
            color = AGENT_COLORS[exp_name]
        else:
            color = DISTINCT_COLORS[sorted(runs_by_exp).index(exp_name) % len(DISTINCT_COLORS)]
        label = legend_label(exp_name)
        num_runs = len(runs)

        if not mean_over_seeds:
            for run_number, run in enumerate(runs):
                if metric == "td_loss":
                    steps, values = run["metric_steps"], run["td_loss"]
                else:
                    steps, values = run["episode_steps"], run[metric]
                if not values:
                    continue
                rolling_mean = rolling(values, rolling_window, np.mean)
                rolling_std = rolling(values, rolling_window, np.std)
                ax.plot(steps, rolling_mean, label=f"{label} (s={run['seed']})" if num_runs > 1 else label,
                        color=color, linestyle=LINE_STYLES[run_number % len(LINE_STYLES)], linewidth=2.0)
                ax.fill_between(steps, rolling_mean - rolling_std * 0.5, rolling_mean + rolling_std * 0.5, color=color, alpha=0.12)
                total_timesteps = run["total_timesteps"]

        else:
            run_curves = []         # (steps, rolling mean) of every run that has data
            last_step = 0
            for run in runs:
                total_timesteps = run["total_timesteps"]
                if metric == "td_loss":
                    steps = np.array(run["metric_steps"])
                    values = rolling(run["td_loss"], rolling_window, np.mean)
                    has_loss = ~np.isnan(values)        # the loss is NaN until the first gradient step
                    steps, values = steps[has_loss], values[has_loss]
                else:
                    steps = np.array(run["episode_steps"])
                    values = rolling(run[metric], rolling_window, np.mean)
                if len(steps) > 0:
                    run_curves.append((steps, values))
                    last_step = max(last_step, steps[-1])
            if not run_curves:
                continue
            if last_step == 0:
                last_step = total_timesteps

            # the runs log at different steps, so each curve is interpolated onto the same 500 steps before averaging
            common_steps = np.linspace(0, last_step, num=500)
            curves = [np.interp(common_steps, steps, values) for steps, values in run_curves]
            mean_curve = np.mean(curves, axis=0)
            ax.plot(common_steps, mean_curve, label=f"{label} (Mean across {num_runs} seeds)" if num_runs > 1 else label,
                    color=color, linewidth=2.0)
            if num_runs > 1:
                std_curve = np.std(curves, axis=0)
                lower, upper = mean_curve - std_curve, mean_curve + std_curve
                if metric == "goal_reached":        # a rate stays within [0, 1]
                    lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
                elif metric == "td_loss":           # a squared error is never negative
                    lower = np.maximum(0, lower)
                ax.fill_between(common_steps, lower, upper, color=color, alpha=0.15)

    ax.set_xlim(0, total_timesteps)
    if y_limits:
        ax.set_ylim(y_limits)
    ax.set_title(f"{env_id} — {title}{title_suffix}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Training Steps", fontsize=9.5, fontweight="medium")
    ax.set_ylabel(y_label, fontsize=9.5, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    ax.legend(fontsize=8.5, loc="best", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {'mean ' if mean_over_seeds else ''}publication plot → {output_path}")


def plot_all_metrics(runs_by_exp, env_id, output_prefix, rolling_window, title_suffix, mean_over_seeds):
    """The figures of the three metrics: <output_prefix>_episodic_return.png, _goal_reach.png and _td_loss.png."""
    for metric in METRICS:
        file_ending = METRICS[metric][3]
        plot_metric(runs_by_exp, env_id, Path(f"{output_prefix}_{file_ending}.png"), rolling_window, metric, title_suffix, mean_over_seeds)


def best_experiment(runs_by_exp):
    """The experiment with the best early performance: goal rate + 0.1 * episodic return over the first 40% of the
    episodes of each run, averaged over its runs."""
    best_name = None
    best_score = -float("inf")
    for exp_name, runs in runs_by_exp.items():
        run_scores = []
        for run in runs:
            num_early_episodes = max(1, int(len(run["goal_reached"]) * 0.4))
            mean_goal = np.mean(run["goal_reached"][:num_early_episodes])
            mean_return = np.mean(run["episodic_return"][:num_early_episodes])
            run_scores.append(mean_goal + 0.1 * mean_return)
        score = float(np.mean(run_scores))
        if score > best_score:
            best_name, best_score = exp_name, score
    return best_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots/reward_comparison")
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--exp-name-contains", action="append", default=[],
                        help="Only plot runs whose experiment name contains this value. May be repeated.")
    parser.add_argument("--combine-all-runs", action="store_true",
                        help="Draw all matching runs in one comparison figure, even when their seeds differ.")
    parser.add_argument("--mean-reward-only", action="store_true",
                        help="Generate only the mean episodic-reward comparison plot.")
    parser.add_argument("--per-seed-mean-reward-only", action="store_true",
                        help="Generate one mean episodic-reward comparison plot for each seed.")
    parser.add_argument("--decay-comparison", action="store_true",
                        help="Generate decay comparison plots for Baseline, Reward Shaped, and Best of Both.")
    args = parser.parse_args()

    runs_by_exp = load_runs(args.results_dir, args.env_id, args.exp_name_contains)
    if not runs_by_exp:
        raise SystemExit(f"No runs found for {args.env_id} in {args.results_dir}")

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Every output file starts with the environment id; titles keep the "/" of an Atari id, file names cannot.
    file_prefix = plots_dir / args.env_id.replace("/", "_")

    seeds = set()
    for runs in runs_by_exp.values():
        for run in runs:
            seeds.add(run["seed"])
    seeds = sorted(seeds)

    if args.mean_reward_only:
        plot_metric(runs_by_exp, args.env_id, Path(f"{file_prefix}_mean_reward_comparison.png"), args.rolling_window,
                    "episodic_return", " | Transfer Performance", mean_over_seeds=True)
        return

    if args.per_seed_mean_reward_only:
        for seed in seeds:
            plot_metric(select_seed(runs_by_exp, seed), args.env_id, Path(f"{file_prefix}_seed{seed}_mean_reward_comparison.png"),
                        args.rolling_window, "episodic_return", f" | Seed {seed} Transfer Performance", mean_over_seeds=True)
        return

    if args.combine_all_runs:
        plot_all_metrics(runs_by_exp, args.env_id, f"{file_prefix}_combined_comparison", args.rolling_window,
                         " | Combined Transfer Comparison", mean_over_seeds=False)
        return

    if args.decay_comparison or any("_decay_" in exp_name for exp_name in runs_by_exp):
        print(f"=== Generating Per-Seed Epsilon Decay Comparisons for {args.env_id} ===")
        # "dqn_baseline" also matches "ddqn_baseline", and likewise for reward shaping
        baseline_runs = {exp_name: runs for exp_name, runs in runs_by_exp.items() if "dqn_baseline" in exp_name}
        shaped_runs = {exp_name: runs for exp_name, runs in runs_by_exp.items() if "dqn_reward_shaping" in exp_name}

        for seed in seeds:
            seed_baseline_runs = select_seed(baseline_runs, seed)
            seed_shaped_runs = select_seed(shaped_runs, seed)
            best_of_both = {}
            if seed_baseline_runs:
                plot_all_metrics(seed_baseline_runs, args.env_id, f"{file_prefix}_decay_comparison_baseline_seed{seed}",
                                 args.rolling_window, f" | Baseline DDQN (Seed {seed})", mean_over_seeds=False)
                best_baseline = best_experiment(seed_baseline_runs)
                if best_baseline is not None:
                    best_of_both[best_baseline] = seed_baseline_runs[best_baseline]
            if seed_shaped_runs:
                plot_all_metrics(seed_shaped_runs, args.env_id, f"{file_prefix}_decay_comparison_reward_shaping_seed{seed}",
                                 args.rolling_window, f" | Reward Shaped DDQN (Seed {seed})", mean_over_seeds=False)
                best_shaped = best_experiment(seed_shaped_runs)
                if best_shaped is not None:
                    best_of_both[best_shaped] = seed_shaped_runs[best_shaped]
            if best_of_both:
                plot_all_metrics(best_of_both, args.env_id, f"{file_prefix}_decay_comparison_best_of_both_seed{seed}",
                                 args.rolling_window, f" | Best Baseline vs Best Shaped (Seed {seed})", mean_over_seeds=False)
        return

    print(f"Found DQN seeds: {seeds} for {args.env_id}")
    for seed in seeds:
        plot_all_metrics(select_seed(runs_by_exp, seed), args.env_id, f"{file_prefix}_comparison_seed{seed}",
                         args.rolling_window, f" | Seed {seed}", mean_over_seeds=False)
    print(f"Generating mean aggregated comparison plots across all {len(seeds)} seeds for {args.env_id} ...")
    plot_all_metrics(runs_by_exp, args.env_id, f"{file_prefix}_mean", args.rolling_window,
                     f" | Mean Across {len(seeds)} Seeds", mean_over_seeds=True)


if __name__ == "__main__":
    main()
