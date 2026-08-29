"""
plot_training_action_freq.py - Qualitative Action Analysis (JSON Report)
=========================================================================
Generates a structured JSON report quantifying irrelevant/ineffective actions
taken during training, including wall bumps, suboptimal pickups/drops/toggles.

Report structure:
  1. AGGREGATED SUMMARY — Mean metrics across all seeds, RS-DDQN improvement %
  2. PER-STAGE BREAKDOWN — Initial, Key Picked, Door Opened (DoorKey envs)
  3. TIME-WINDOW ANALYSIS — Full training, Last 50%, Last 25%
  4. PER-SEED DETAILS — Compact per-seed stats with direction-wise wall bumps

No heatmap plots are generated; only the qualitative JSON analysis.

Usage:
  python src/plot_training_action_freq.py --env-id MiniGrid-DoorKey-8x8-v0
"""

import argparse
import json
from pathlib import Path
import numpy as np

from dqn_common import action_names, make_env


# ============================================================================
# HELPERS
# ============================================================================

def get_dirs_by_seed(results_dir, env_id, exp_name):
    """Finds the run directories for a given experiment, grouped by seed."""
    dirs = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}__*")):
        counts_path = run_dir / "state_action_counts.npy"
        config_path = run_dir / "config.json"

        if not counts_path.exists() or not config_path.exists():
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            seed = int(config["seed"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

        dirs[seed] = run_dir
    return dirs


def extract_layout(env):
    """Extracts walls and annotations from the environment."""
    width = env.unwrapped.width
    height = env.unwrapped.height
    grid = env.unwrapped.grid

    wall_mask = np.zeros((width, height), dtype=bool)
    annotations = {}

    start_pos = tuple(env.unwrapped.agent_pos)
    annotations[start_pos] = "S"

    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is not None:
                if cell.type == "wall":
                    wall_mask[x, y] = True
                elif cell.type == "goal":
                    annotations[(x, y)] = "G"
                elif cell.type == "key":
                    annotations[(x, y)] = "K"
                elif cell.type == "door":
                    annotations[(x, y)] = "D"

    return wall_mask, annotations, width, height


DIR_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
DIR_OFFSETS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _pct(num, denom):
    """Return rounded percentage."""
    return round(num / denom * 100, 2) if denom > 0 else 0.0


def compute_actions_for_counts(counts_array, wall_mask, annotations, names, width, height):
    """
    Compute irrelevant action stats from a state_action_counts array.
    counts_array shape: (W, H, 4_dirs, N_actions)
    Returns a compact dict of metrics.
    """
    fwd_idx = names.index("forward") if "forward" in names else -1
    pickup_idx = names.index("pickup") if "pickup" in names else -1
    toggle_idx = names.index("toggle") if "toggle" in names else -1
    drop_idx = names.index("drop") if "drop" in names else -1

    key_pos = {pos for pos, label in annotations.items() if label == "K"}
    door_pos = {pos for pos, label in annotations.items() if label == "D"}

    total_actions = int(counts_array.sum())
    wall_bumps = 0
    wall_bumps_by_dir = {0: 0, 1: 0, 2: 0, 3: 0}
    suboptimal_pickups = 0
    suboptimal_toggles = 0
    suboptimal_drops = 0

    for x in range(width):
        for y in range(height):
            if wall_mask[x, y]:
                continue
            for d_idx in range(4):
                dx, dy = DIR_OFFSETS[d_idx]
                fx, fy = x + dx, y + dy
                is_front_wall = (
                    fx < 0 or fx >= width or fy < 0 or fy >= height or wall_mask[fx, fy]
                )
                is_front_key = (fx, fy) in key_pos
                is_front_door = (fx, fy) in door_pos

                if fwd_idx != -1 and is_front_wall:
                    bump_count = int(counts_array[x, y, d_idx, fwd_idx])
                    wall_bumps += bump_count
                    wall_bumps_by_dir[d_idx] += bump_count
                if pickup_idx != -1 and not is_front_key:
                    suboptimal_pickups += int(counts_array[x, y, d_idx, pickup_idx])
                if toggle_idx != -1 and not is_front_door:
                    suboptimal_toggles += int(counts_array[x, y, d_idx, toggle_idx])
                if drop_idx != -1:
                    suboptimal_drops += int(counts_array[x, y, d_idx, drop_idx])

    total_irrelevant = wall_bumps + suboptimal_pickups + suboptimal_toggles + suboptimal_drops

    result = {
        "total_actions": total_actions,
        "wall_bumps": wall_bumps,
        "pct_wall_bumps": _pct(wall_bumps, total_actions),
        "suboptimal_pickups": suboptimal_pickups,
        "pct_suboptimal_pickups": _pct(suboptimal_pickups, total_actions),
        "suboptimal_toggles": suboptimal_toggles,
        "pct_suboptimal_toggles": _pct(suboptimal_toggles, total_actions),
        "suboptimal_drops": suboptimal_drops,
        "pct_suboptimal_drops": _pct(suboptimal_drops, total_actions),
        "total_irrelevant": total_irrelevant,
        "pct_irrelevant": _pct(total_irrelevant, total_actions),
    }
    return result, wall_bumps_by_dir


def load_counts(run_dir, suffix="", stage_idx=None):
    """Load state_action_counts array, optionally slice by stage."""
    fname = f"state_action_counts{suffix}.npy"
    path = run_dir / fname
    fallback = run_dir / "state_action_counts.npy"

    if path.exists():
        raw = np.load(path)
    elif fallback.exists():
        raw = np.load(fallback)
    else:
        return None

    # raw shape: (W, H, 4_dirs, N_actions, N_stages) for 5D
    if raw.ndim == 5:
        if stage_idx is not None and stage_idx < raw.shape[4]:
            return raw[:, :, :, :, stage_idx]
        else:
            return raw.sum(axis=-1)
    return raw


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def build_analysis_report(env_id, results_dir, action_set="task"):
    """Build the complete structured JSON analysis report."""
    # Find experiment directories
    baseline_dirs = get_dirs_by_seed(results_dir, env_id, "ddqn_baseline")
    if not baseline_dirs:
        baseline_dirs = get_dirs_by_seed(results_dir, env_id, "dqn_baseline")
    shaped_dirs = get_dirs_by_seed(results_dir, env_id, "ddqn_reward_shaping")
    if not shaped_dirs:
        shaped_dirs = get_dirs_by_seed(results_dir, env_id, "dqn_reward_shaping")

    seeds = sorted(set(baseline_dirs.keys()) | set(shaped_dirs.keys()))
    if not seeds:
        return None

    # Create env for layout analysis
    env = make_env(env_id, 1, action_set, capture_video=False, run_name="dummy_analysis", max_steps=10)
    env.reset(seed=1)
    wall_mask, annotations, width, height = extract_layout(env)
    num_actions = env.action_space.n
    names = action_names(env_id, action_set, num_actions)
    env.close()

    is_doorkey = "DoorKey" in env_id

    # Define time windows and stages
    time_windows = [
        ("full_training", ""),
        ("last_50_pct", "_last_half"),
        ("last_25_pct", "_last_quarter"),
    ]

    stages = []
    if is_doorkey:
        stages = [
            ("all_stages", None),
            ("stage_0_initial", 0),
            ("stage_1_key_picked", 1),
            ("stage_2_door_opened", 2),
        ]
    else:
        stages = [("all_stages", None)]

    # ---- Collect per-seed data ----
    per_seed_data = {}
    for seed in seeds:
        b_dir = baseline_dirs.get(seed)
        s_dir = shaped_dirs.get(seed)
        seed_data = {}

        for tw_name, tw_suffix in time_windows:
            tw_data = {}
            for stage_name, stage_idx in stages:
                for agent_key, agent_dir, agent_label in [
                    ("baseline", b_dir, "Baseline DDQN"),
                    ("rs_ddqn", s_dir, "RS-DDQN"),
                ]:
                    if agent_dir is None:
                        continue
                    counts = load_counts(agent_dir, suffix=tw_suffix, stage_idx=stage_idx)
                    if counts is None:
                        continue

                    stats, dir_bumps = compute_actions_for_counts(
                        counts, wall_mask, annotations, names, width, height
                    )
                    key = f"{agent_key}"
                    if stage_name not in tw_data:
                        tw_data[stage_name] = {}
                    tw_data[stage_name][key] = stats
                    # Store direction-wise bumps for detailed view
                    tw_data[stage_name][f"{key}_wall_bumps_by_direction"] = {
                        DIR_NAMES[d]: cnt for d, cnt in dir_bumps.items() if cnt > 0
                    }

            seed_data[tw_name] = tw_data
        per_seed_data[seed] = seed_data

    # ---- Build aggregated summary ----
    def _aggregate_metric(metric_name, seeds_data, tw_name, stage_name, agent_key):
        """Collect a specific metric across all seeds for averaging."""
        values = []
        for seed in seeds:
            try:
                val = seeds_data[seed][tw_name][stage_name][agent_key][metric_name]
                values.append(val)
            except KeyError:
                pass
        return values

    def _make_summary_block(tw_name, stage_name):
        """Create a compact summary for one (time_window, stage) combination."""
        block = {}
        for agent_key, label in [("baseline", "Baseline DDQN"), ("rs_ddqn", "RS-DDQN")]:
            pct_irr_vals = _aggregate_metric("pct_irrelevant", per_seed_data, tw_name, stage_name, agent_key)
            pct_wb_vals = _aggregate_metric("pct_wall_bumps", per_seed_data, tw_name, stage_name, agent_key)
            pct_sp_vals = _aggregate_metric("pct_suboptimal_pickups", per_seed_data, tw_name, stage_name, agent_key)
            pct_st_vals = _aggregate_metric("pct_suboptimal_toggles", per_seed_data, tw_name, stage_name, agent_key)
            pct_sd_vals = _aggregate_metric("pct_suboptimal_drops", per_seed_data, tw_name, stage_name, agent_key)

            if pct_irr_vals:
                block[label] = {
                    "mean_pct_irrelevant": round(float(np.mean(pct_irr_vals)), 2),
                    "mean_pct_wall_bumps": round(float(np.mean(pct_wb_vals)), 2),
                    "mean_pct_suboptimal_pickups": round(float(np.mean(pct_sp_vals)), 2) if pct_sp_vals else 0.0,
                    "mean_pct_suboptimal_toggles": round(float(np.mean(pct_st_vals)), 2) if pct_st_vals else 0.0,
                    "mean_pct_suboptimal_drops": round(float(np.mean(pct_sd_vals)), 2) if pct_sd_vals else 0.0,
                    "num_seeds": len(pct_irr_vals),
                }

        # Compute RS-DDQN improvement
        b_vals = _aggregate_metric("pct_irrelevant", per_seed_data, tw_name, stage_name, "baseline")
        s_vals = _aggregate_metric("pct_irrelevant", per_seed_data, tw_name, stage_name, "rs_ddqn")
        if b_vals and s_vals:
            b_mean = float(np.mean(b_vals))
            s_mean = float(np.mean(s_vals))
            reduction = b_mean - s_mean
            pct_improvement = _pct(reduction, b_mean) if b_mean > 0 else 0.0
            block["rs_ddqn_improvement"] = {
                "absolute_reduction_pct_points": round(reduction, 2),
                "relative_improvement_pct": round(pct_improvement, 2),
            }

        return block

    # ---- Build report structure ----
    report = {
        "environment": env_id,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "grid_size": f"{width}x{height}",
    }

    # Section 1: Aggregated summary
    aggregated = {}
    for tw_name, _ in time_windows:
        tw_summary = {}
        for stage_name, _ in stages:
            block = _make_summary_block(tw_name, stage_name)
            if block:
                tw_summary[stage_name] = block
        if tw_summary:
            aggregated[tw_name] = tw_summary
    report["aggregated_summary"] = aggregated

    # Section 2: Per-seed compact details (only full_training, all_stages + direction bumps)
    per_seed_section = {}
    for seed in seeds:
        seed_entry = {}
        for tw_name, _ in time_windows:
            tw_entry = {}
            for stage_name, _ in stages:
                stage_entry = {}
                for agent_key in ["baseline", "rs_ddqn"]:
                    try:
                        stats = per_seed_data[seed][tw_name][stage_name][agent_key]
                        dir_bumps = per_seed_data[seed][tw_name][stage_name].get(
                            f"{agent_key}_wall_bumps_by_direction", {}
                        )
                        # Compact per-seed entry: just key percentages + direction bumps
                        stage_entry[agent_key] = {
                            "pct_irrelevant": stats["pct_irrelevant"],
                            "pct_wall_bumps": stats["pct_wall_bumps"],
                            "pct_suboptimal_pickups": stats["pct_suboptimal_pickups"],
                            "pct_suboptimal_drops": stats["pct_suboptimal_drops"],
                        }
                        if dir_bumps:
                            stage_entry[agent_key]["wall_bumps_by_direction"] = dir_bumps
                    except KeyError:
                        pass
                if stage_entry:
                    tw_entry[stage_name] = stage_entry
            if tw_entry:
                seed_entry[tw_name] = tw_entry
        per_seed_section[f"seed_{seed}"] = seed_entry
    report["per_seed_details"] = per_seed_section

    return report


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate qualitative JSON analysis of irrelevant/suboptimal actions during training."
    )
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

    # Save report
    output_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.env_id}_action_analysis.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print compact summary to console
    print(f"\nReport saved to {report_path}")
    agg = report.get("aggregated_summary", {})
    for tw_name, tw_data in agg.items():
        for stage_name, stage_data in tw_data.items():
            print(f"\n  [{tw_name} | {stage_name}]")
            for agent_label in ["Baseline DDQN", "RS-DDQN"]:
                if agent_label in stage_data:
                    d = stage_data[agent_label]
                    print(f"    {agent_label}: {d['mean_pct_irrelevant']:.1f}% irrelevant "
                          f"(wall: {d['mean_pct_wall_bumps']:.1f}%, "
                          f"pickup: {d['mean_pct_suboptimal_pickups']:.1f}%, "
                          f"drop: {d['mean_pct_suboptimal_drops']:.1f}%)")
            if "rs_ddqn_improvement" in stage_data:
                imp = stage_data["rs_ddqn_improvement"]
                print(f"    → RS-DDQN Improvement: {imp['absolute_reduction_pct_points']:.1f} pp "
                      f"({imp['relative_improvement_pct']:.1f}% relative)")

    print(f"\nDone. Analysis saved to {report_path}")
