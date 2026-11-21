"""JSON report of the irrelevant actions taken during training, from the saved state-action counts: "forward" into a
wall, "pickup" away from the key, "toggle" away from the door and every "drop", per seed and averaged over seeds."""

import argparse
import json
from pathlib import Path

import numpy as np

from dqn_common import action_names, make_env

# (key in the report, label)
AGENTS = [("baseline", "Baseline DDQN"), ("rs_ddqn", "RS-DDQN"), ("lp_ddqn", "LP-DDQN")]
# (name in the report, suffix of the counts file)
TIME_WINDOWS = [("full_training", ""), ("last_50_pct", "_last_half"), ("last_25_pct", "_last_quarter")]
# (name in the report, index on the stage axis of the counts; None = all stages summed)
DOORKEY_STAGES = [("all_stages", None), ("stage_0_initial", 0), ("stage_1_key_picked", 1), ("stage_2_door_opened", 2)]
PERCENTAGE_NAMES = ["pct_irrelevant", "pct_wall_bumps", "pct_suboptimal_pickups", "pct_suboptimal_toggles", "pct_suboptimal_drops"]
DIRECTION_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
DIRECTION_OFFSETS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}      # (dx, dy) of one step forward


def find_run_dirs(results_dir, env_id, exp_name):
    """{seed: run folder} of one experiment.  When a seed was run more than once, the newest run is used."""
    run_dirs = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}__*")):
        if not (run_dir / "state_action_counts.npy").exists() or not (run_dir / "config.json").exists():
            continue
        with open(run_dir / "config.json", "r", encoding="utf-8") as config_file:
            seed = int(json.load(config_file)["seed"])
        run_dirs[seed] = run_dir
    return run_dirs


def load_counts(run_dir, file_suffix, stage_index):
    """Action counts of one run, shape (width, height, 4 directions, num_actions).

    file_suffix selects the time window; a run without that file falls back to the counts of the whole training.
    stage_index selects one task stage of the saved (width, height, 4, num_actions, 3 stages) array; None sums the stages.
    """
    counts_path = run_dir / f"state_action_counts{file_suffix}.npy"
    if not counts_path.exists():
        counts_path = run_dir / "state_action_counts.npy"
    counts = np.load(counts_path)
    if counts.ndim == 5:
        if stage_index is not None and stage_index < counts.shape[4]:
            return counts[:, :, :, :, stage_index]
        return counts.sum(axis=-1)
    return counts


def percentage(part, total):
    return round(part / total * 100, 2) if total > 0 else 0.0


def irrelevant_action_percentages(counts, wall_mask, key_positions, door_positions, names):
    """Percentages of irrelevant actions in one counts array (width, height, 4 directions, num_actions).

    Returns the percentages (keys: PERCENTAGE_NAMES) and the number of wall bumps for each facing direction.
    """
    width, height = wall_mask.shape
    forward_index = names.index("forward") if "forward" in names else -1
    pickup_index = names.index("pickup") if "pickup" in names else -1
    toggle_index = names.index("toggle") if "toggle" in names else -1
    drop_index = names.index("drop") if "drop" in names else -1

    wall_bumps = {0: 0, 1: 0, 2: 0, 3: 0}      # facing direction -> count
    suboptimal_pickups = 0
    suboptimal_toggles = 0
    suboptimal_drops = 0
    for x in range(width):
        for y in range(height):
            if wall_mask[x, y]:
                continue
            for direction in range(4):
                step_x, step_y = DIRECTION_OFFSETS[direction]
                front_x, front_y = x + step_x, y + step_y
                faces_wall = front_x < 0 or front_x >= width or front_y < 0 or front_y >= height or wall_mask[front_x, front_y]
                if forward_index != -1 and faces_wall:
                    wall_bumps[direction] += int(counts[x, y, direction, forward_index])
                if pickup_index != -1 and (front_x, front_y) not in key_positions:
                    suboptimal_pickups += int(counts[x, y, direction, pickup_index])
                if toggle_index != -1 and (front_x, front_y) not in door_positions:
                    suboptimal_toggles += int(counts[x, y, direction, toggle_index])
                if drop_index != -1:
                    suboptimal_drops += int(counts[x, y, direction, drop_index])

    total_actions = int(counts.sum())
    total_wall_bumps = sum(wall_bumps.values())
    percentages = {
        "pct_irrelevant": percentage(total_wall_bumps + suboptimal_pickups + suboptimal_toggles + suboptimal_drops, total_actions),
        "pct_wall_bumps": percentage(total_wall_bumps, total_actions),
        "pct_suboptimal_pickups": percentage(suboptimal_pickups, total_actions),
        "pct_suboptimal_toggles": percentage(suboptimal_toggles, total_actions),
        "pct_suboptimal_drops": percentage(suboptimal_drops, total_actions),
    }
    wall_bumps_by_direction = {}
    for direction in range(4):
        if wall_bumps[direction] > 0:
            wall_bumps_by_direction[DIRECTION_NAMES[direction]] = wall_bumps[direction]
    return percentages, wall_bumps_by_direction


def build_analysis_report(env_id, results_dir, action_set):
    """The whole report as a dictionary, or None when no agent has a run for this environment."""
    baseline_runs = find_run_dirs(results_dir, env_id, "ddqn_baseline") or find_run_dirs(results_dir, env_id, "dqn_baseline")
    shaped_runs = find_run_dirs(results_dir, env_id, "ddqn_reward_shaping") or find_run_dirs(results_dir, env_id, "dqn_reward_shaping")
    learned_runs = find_run_dirs(results_dir, env_id, "ddqn_learned_penalty")
    run_dirs = {"baseline": baseline_runs, "rs_ddqn": shaped_runs, "lp_ddqn": learned_runs}
    seeds = sorted(set(baseline_runs) | set(shaped_runs) | set(learned_runs))
    if not seeds:
        return None

    env = make_env(env_id, seeds[0], action_set, max_steps=10)
    width, height = env.unwrapped.width, env.unwrapped.height
    names = action_names(env_id, action_set, env.action_space.n)
    stages = DOORKEY_STAGES if "DoorKey" in env_id else [("all_stages", None)]

    # ---- per-seed details ----
    all_percentages = {}        # (seed, window name, stage name, agent key) -> percentages of irrelevant actions
    per_seed_details = {}
    for seed in seeds:
        # Map layout of THIS seed: walls, key and door.  A run trained with --fixed-layout sees the map of its own
        # seed in every episode, so its actions must be judged against that map.
        env.reset(seed=seed)
        wall_mask = np.zeros((width, height), dtype=bool)
        key_positions = set()
        door_positions = set()
        for x in range(width):
            for y in range(height):
                cell = env.unwrapped.grid.get(x, y)
                if cell is None:
                    continue
                if cell.type == "wall":
                    wall_mask[x, y] = True
                elif cell.type == "key":
                    key_positions.add((x, y))
                elif cell.type == "door":
                    door_positions.add((x, y))

        seed_details = {}
        for window_name, file_suffix in TIME_WINDOWS:
            seed_details[window_name] = {}
            for stage_name, stage_index in stages:
                stage_details = {}
                for agent_key, _ in AGENTS:
                    if seed not in run_dirs[agent_key]:
                        continue
                    counts = load_counts(run_dirs[agent_key][seed], file_suffix, stage_index)
                    percentages, wall_bumps_by_direction = irrelevant_action_percentages(
                        counts, wall_mask, key_positions, door_positions, names)
                    all_percentages[(seed, window_name, stage_name, agent_key)] = percentages
                    stage_details[agent_key] = {
                        "pct_irrelevant": percentages["pct_irrelevant"],
                        "pct_wall_bumps": percentages["pct_wall_bumps"],
                        "pct_suboptimal_pickups": percentages["pct_suboptimal_pickups"],
                        "pct_suboptimal_drops": percentages["pct_suboptimal_drops"],
                    }
                    if wall_bumps_by_direction:
                        stage_details[agent_key]["wall_bumps_by_direction"] = wall_bumps_by_direction
                seed_details[window_name][stage_name] = stage_details
        per_seed_details[f"seed_{seed}"] = seed_details
    env.close()

    # ---- aggregated summary: means over the seeds, and the improvement over the baseline ----
    aggregated_summary = {}
    for window_name, _ in TIME_WINDOWS:
        aggregated_summary[window_name] = {}
        for stage_name, _ in stages:
            stage_summary = {}
            mean_pct_irrelevant = {}        # agent key -> mean over seeds, not rounded
            for agent_key, agent_label in AGENTS:
                percentages_of_seeds = []
                for seed in seeds:
                    if (seed, window_name, stage_name, agent_key) in all_percentages:
                        percentages_of_seeds.append(all_percentages[(seed, window_name, stage_name, agent_key)])
                if not percentages_of_seeds:
                    continue
                agent_summary = {}
                for name in PERCENTAGE_NAMES:
                    values = [percentages[name] for percentages in percentages_of_seeds]
                    agent_summary[f"mean_{name}"] = round(float(np.mean(values)), 2)
                agent_summary["num_seeds"] = len(percentages_of_seeds)
                stage_summary[agent_label] = agent_summary
                mean_pct_irrelevant[agent_key] = float(np.mean([percentages["pct_irrelevant"] for percentages in percentages_of_seeds]))

            for agent_key in ("rs_ddqn", "lp_ddqn"):
                if "baseline" in mean_pct_irrelevant and agent_key in mean_pct_irrelevant:
                    reduction = mean_pct_irrelevant["baseline"] - mean_pct_irrelevant[agent_key]
                    stage_summary[f"{agent_key}_improvement"] = {
                        "absolute_reduction_pct_points": round(reduction, 2),
                        "relative_improvement_pct": percentage(reduction, mean_pct_irrelevant["baseline"]),
                    }
            aggregated_summary[window_name][stage_name] = stage_summary

    return {
        "environment": env_id,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "grid_size": f"{width}x{height}",
        "aggregated_summary": aggregated_summary,
        "per_seed_details": per_seed_details,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON analysis of the irrelevant actions taken during training.")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    parser.add_argument("--plots-dir", type=str, default="plots/action_freq")
    args = parser.parse_args()

    print(f"=== Qualitative Action Analysis for {args.env_id} ===")
    report = build_analysis_report(args.env_id, args.results_dir, args.action_set)
    if report is None:
        print(f"No trained models found for {args.env_id} in {args.results_dir}")
        raise SystemExit(1)

    output_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.env_id}_action_analysis.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(f"\nReport saved to {report_path}")

    for window_name, window_summary in report["aggregated_summary"].items():
        for stage_name, stage_summary in window_summary.items():
            print(f"\n  [{window_name} | {stage_name}]")
            for _, agent_label in AGENTS:
                if agent_label in stage_summary:
                    agent_summary = stage_summary[agent_label]
                    print(f"    {agent_label}: {agent_summary['mean_pct_irrelevant']:.1f}% irrelevant "
                          f"(wall: {agent_summary['mean_pct_wall_bumps']:.1f}%, "
                          f"pickup: {agent_summary['mean_pct_suboptimal_pickups']:.1f}%, "
                          f"drop: {agent_summary['mean_pct_suboptimal_drops']:.1f}%)")
            for improvement_key, agent_label in (("rs_ddqn_improvement", "RS-DDQN"), ("lp_ddqn_improvement", "LP-DDQN")):
                if improvement_key in stage_summary:
                    improvement = stage_summary[improvement_key]
                    print(f"    → {agent_label} Improvement: {improvement['absolute_reduction_pct_points']:.1f} pp "
                          f"({improvement['relative_improvement_pct']:.1f}% relative)")

    print(f"\nDone. Analysis saved to {report_path}")
