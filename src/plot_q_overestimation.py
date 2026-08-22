"""
plot_q_overestimation.py - Q-Value Overestimation Comparison
=============================================================
Visualises how much standard (vanilla) DQN overestimates Q-values compared
to Double DQN (DDQN) across multiple MiniGrid environments.

For every reachable (non-wall) cell in the grid, and for all 4 agent
directions (North, East, South, West), the script:
  1. Constructs the wrapped observation by setting agent_pos / agent_dir
     on the unwrapped environment and running it through the wrapper pipeline.
  2. Passes the observation through the trained Q-network and records Q(s, a).
  3. Displays a 4×3 grid of heatmaps for each facing direction (Rows = North, East,
     South, West; Columns = Agent A Max-Q, Agent B Max-Q, Difference A − B).

Usage:
  python plot_q_overestimation.py
  python plot_q_overestimation.py --results-dir results --action-set task
  python plot_q_overestimation.py --hidden-size 256

Output:
  plots/<env_id>_q_overestimation_seed<seed>.png   (one per env × seed)
  plots/q_overestimation_comparison.png             (summary bar chart)
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

import argparse
import json
import warnings
from pathlib import Path

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
import numpy as np
import torch

# ============================================================================
# LOCAL IMPORTS
# ============================================================================

from dqn_common import QNetwork, make_env, action_names


# ============================================================================
# CONSTANTS & HELPERS
# ============================================================================

ENV_IDS = [
    "MiniGrid-Empty-6x6-v0",
    "MiniGrid-DoorKey-6x6-v0",
]

NUM_DIRECTIONS = 4


def get_models_by_seed(results_dir, env_id, exp_name):
    """Finds all trained model files for a given experiment, grouped by seed."""
    models = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}__*")):
        model_path = run_dir / "q_net.pt"
        if not model_path.exists():
            continue
        parts = run_dir.name.split("__")
        try:
            seed = int(parts[2])
        except (IndexError, ValueError):
            continue
        models[seed] = model_path
    return models


def get_decay_str(model_path):
    """Reads config.json from model parent directory to get epsilon_schedule."""
    if not model_path:
        return ""
    cfg_path = model_path.parent / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                c = json.load(f)
            sched = c.get("epsilon_schedule", "")
            if sched:
                return f" ({sched})"
        except Exception:
            pass
    return ""


def get_wrapped_obs(env):
    """Generates raw observation and pushes it through observation wrappers."""
    obs = env.unwrapped.gen_obs()
    wrapper = env
    obs_wrappers = []
    while hasattr(wrapper, 'env'):
        if hasattr(wrapper, 'observation'):
            obs_wrappers.append(wrapper)
        wrapper = wrapper.env
        
    for w in reversed(obs_wrappers):
        obs = w.observation(obs)
        
    return obs


def compute_q_values_grid_for_stage(env, q_net, seed, device, stage=1):
    """
    Computes Q(s, a) for reachable cells across all 4 facing directions for a specific environment stage.
    Stage 1: Initial (Key on ground, Door locked/closed)
    Stage 2: Key Carrying (Key picked up, Door locked/closed)
    Stage 3: Door Opened (Door open, Key used/dropped)
    """
    env.reset(seed=seed)
    width = env.unwrapped.width
    height = env.unwrapped.height
    num_actions = env.action_space.n

    # Locate key and door positions
    key_pos = None
    door_pos = None
    key_cell = None
    door_cell = None

    for x in range(width):
        for y in range(height):
            c = env.unwrapped.grid.get(x, y)
            if c is not None:
                if c.type == "key":
                    key_pos = (x, y)
                    key_cell = c
                elif c.type == "door":
                    door_pos = (x, y)
                    door_cell = c

    # Modify map state according to requested stage
    if stage == 2 and key_pos is not None:
        # Key picked up (remove key from ground)
        env.unwrapped.grid.set(key_pos[0], key_pos[1], None)
        env.unwrapped.carrying = key_cell
    elif stage == 3:
        if key_pos is not None:
            env.unwrapped.grid.set(key_pos[0], key_pos[1], None)
            env.unwrapped.carrying = None
        if door_cell is not None:
            door_cell.is_open = True
            door_cell.is_locked = False

    q_grid = np.full((width, height, 4, num_actions), np.nan, dtype=np.float32)
    wall_mask = np.zeros((width, height), dtype=bool)
    annotations = {}

    start_pos = tuple(env.unwrapped.agent_pos)
    annotations[start_pos] = "S"

    for x in range(width):
        for y in range(height):
            cell = env.unwrapped.grid.get(x, y)
            if cell is not None:
                if cell.type == "wall":
                    wall_mask[x, y] = True
                elif cell.type == "goal":
                    annotations[(x, y)] = "G"
                elif cell.type == "key" and stage == 1:
                    annotations[(x, y)] = "K"
                elif cell.type == "door":
                    annotations[(x, y)] = "D(open)" if getattr(cell, 'is_open', False) else "D"

    with torch.no_grad():
        for x in range(width):
            for y in range(height):
                if wall_mask[x, y]:
                    continue

                for d in range(NUM_DIRECTIONS):
                    env.unwrapped.agent_pos = np.array([x, y])
                    env.unwrapped.agent_dir = d

                    obs = get_wrapped_obs(env)
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_net(obs_t).squeeze(0)
                    q_grid[x, y, d, :] = q_values.cpu().numpy()

    return q_grid, wall_mask, annotations


def compute_q_values_grid(env, q_net, seed, device):
    """Wrapper to compute stage 1 Q-values for backwards compatibility."""
    return compute_q_values_grid_for_stage(env, q_net, seed, device, stage=1)


# ============================================================================
# PER-ENVIRONMENT HEATMAP PLOT
# ============================================================================

def plot_heatmap_for_env(env_id, grid_a, grid_b, wall_mask, annotations, seed,
                        label_a="Agent A", label_b="Agent B", comparison_tag="", stage_name=""):
    width, height, num_dirs, num_actions = grid_a.shape
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grid_a_max = np.nanmax(grid_a, axis=3) # shape (W, H, 4)
        grid_b_max = np.nanmax(grid_b, axis=3) # shape (W, H, 4)
    
    diff_grid_max = grid_a_max - grid_b_max # shape (W, H, 4)
    diff_grid = grid_a - grid_b             # shape (W, H, 4, num_actions)

    dir_info = [
        (3, "North"),
        (0, "East"),
        (1, "South"),
        (2, "West")
    ]

    fig, axes = plt.subplots(4, 3, figsize=(24, 20))
    title_suffix = f" [{stage_name}]" if stage_name else ""
    fig.suptitle(
        f"{env_id}{title_suffix}  —  {label_a} vs {label_b}  (seed={seed})",
        fontsize=18, fontweight="bold", y=0.98
    )

    vmin_q = np.nanmin([grid_a_max, grid_b_max])
    vmax_q = np.nanmax([grid_a_max, grid_b_max])

    col_titles = [
        f"{label_a}  Max Q",
        f"{label_b}  Max Q",
        f"Difference ({label_a} − {label_b})"
    ]

    names = action_names(env_id, "task", num_actions)
    abbr_map = {
        "left": "L", "right": "R", "forward": "F",
        "pickup": "P", "drop": "Dp", "toggle": "T", "done": "Dn"
    }
    action_labels = [abbr_map.get(n, n[:1].upper()) for n in names]

    for row, (d_idx, d_name) in enumerate(dir_info):
        panels = [
            (grid_a_max[:, :, d_idx], grid_a[:, :, d_idx, :], "viridis", None),
            (grid_b_max[:, :, d_idx], grid_b[:, :, d_idx, :], "viridis", None),
            (diff_grid_max[:, :, d_idx], diff_grid[:, :, d_idx, :], "RdBu_r", "diverging"),
        ]

        for col, (grid_max_d, grid_full_d, cmap, style) in enumerate(panels):
            ax = axes[row, col]
            display = grid_max_d.T.copy()

            if style == "diverging":
                abs_max = np.nanmax(np.abs(diff_grid[:, :, d_idx, :]))
                if abs_max == 0:
                    abs_max = 1.0
                norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
                im = ax.imshow(display, origin="upper", cmap=cmap, norm=norm)
            else:
                im = ax.imshow(display, origin="upper", cmap=cmap, vmin=vmin_q, vmax=vmax_q)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=14, fontweight="bold")

            if col == 0:
                ax.set_ylabel(f"Facing {d_name}", fontsize=14, fontweight="bold")

            for x in range(width):
                for y in range(height):
                    if wall_mask[x, y]:
                        ax.add_patch(plt.Rectangle(
                            (x - 0.5, y - 0.5), 1, 1,
                            facecolor="dimgray", edgecolor="none",
                        ))

                    if wall_mask[x, y] or annotations.get((x, y)) == "G":
                        continue

                    else:
                        if (x, y) in annotations:
                            ax.text(x + 0.45, y - 0.45, annotations[(x, y)],
                                    ha='right', va='top', color='white', 
                                    fontsize=10, fontweight='bold',
                                    path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
                        
                        q_vals = grid_full_d[x, y, :]
                        best_a = int(np.argmax(q_vals)) if not np.isnan(q_vals).all() else -1
                        text_lines = []
                        for a in range(len(q_vals)):
                            if a < len(action_labels):
                                star = "*" if (a == best_a and style != "diverging") else ""
                                if style == "diverging":
                                    val_str = f"{q_vals[a]:+.2f}"
                                    text_lines.append(f"{action_labels[a]}: {val_str}")
                                else:
                                    val_str = f"{q_vals[a]:.2f}"
                                    text_lines.append(f"{action_labels[a]}{star}: {val_str}")
                        
                        text_str = "\n".join(text_lines)
                        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none")
                        ax.text(x, y, text_str, ha='center', va='center', 
                                color='black', fontsize=6, bbox=bbox_props)

            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_xticks(np.arange(0, width, 1))
            ax.set_yticks(np.arange(0, height, 1))

    legend_parts = [f"{abbr_map.get(n, n[:1].upper())} = {n}" for n in names]
    legend_text = "   |   ".join(legend_parts)
    fig.text(
        0.5, 0.015,
        f"Action Key:  {legend_text}    ---    Layout Annotations: S = Start, G = Goal, K = Key, D = Door",
        ha="center", va="bottom",
        fontsize=11, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#aaaaaa", alpha=0.8),
    )

    # Format clean folder names (e.g. DoorKey, Empty)
    env_clean = "DoorKey" if "DoorKey" in env_id else ("Empty" if "Empty" in env_id else env_id)
    output_dir = Path("plots") / "overestimation" / env_clean / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_file_suffix = f"_{stage_name.lower().replace(' ', '_')}" if stage_name else ""
    output_path = output_dir / f"q_overestimation{stage_file_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {output_path}")


# ============================================================================
# SUMMARY BAR CHART
# ============================================================================

def plot_bar_chart(summary, label_a="Agent A", label_b="Agent B", comparison_tag=""):
    if not summary:
        print("No data for summary bar chart — skipping.")
        return

    env_labels = [s["env_id"].replace("MiniGrid-", "") for s in summary]
    a_means  = [s["a_mean"]  for s in summary]
    b_means  = [s["b_mean"]  for s in summary]
    a_stds   = [s.get("a_std", 0)  for s in summary]
    b_stds   = [s.get("b_std", 0)  for s in summary]

    x = np.arange(len(env_labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_a = ax.bar(
        x - bar_width / 2, a_means, bar_width,
        yerr=a_stds, capsize=4,
        label=label_a, color="#E74C3C", edgecolor="black", linewidth=0.6,
    )
    bars_b = ax.bar(
        x + bar_width / 2, b_means, bar_width,
        yerr=b_stds, capsize=4,
        label=label_b, color="#3498DB", edgecolor="black", linewidth=0.6,
    )

    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel("Average Max Q-Value", fontsize=12)
    ax.set_title(
        f"Q-Value Comparison: {label_a} vs {label_b}",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars_a, bars_b]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    output_dir = Path("plots") / comparison_tag if comparison_tag else Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "q_overestimation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nBar chart saved → {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare Q-value overestimation between two trained agents."
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"),
                        default=["dqn_vanilla", "ddqn_baseline"])
    parser.add_argument("--label-a", type=str, default=None)
    parser.add_argument("--label-b", type=str, default=None)

    args = parser.parse_args()

    exp_a, exp_b = args.compare
    label_a = args.label_a or exp_a
    label_b = args.label_b or exp_b
    comparison_tag = f"{exp_a}_vs_{exp_b}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Results directory: {args.results_dir}")
    print(f"Comparing: {label_a}  vs  {label_b}")
    print()

    bar_chart_summary = []

    for env_id in ENV_IDS:
        print(f"{'=' * 60}")
        print(f"Environment: {env_id}")
        print(f"{'=' * 60}")

        models_a = get_models_by_seed(args.results_dir, env_id, exp_a)
        models_b = get_models_by_seed(args.results_dir, env_id, exp_b)

        common_seeds = sorted(set(models_a.keys()) & set(models_b.keys()))
        if not common_seeds:
            print(f"  ⚠  No matching seed pair found for {env_id} — skipping.\n")
            continue

        print(f"  Common seeds: {common_seeds}")

        env = make_env(env_id, common_seeds[0], args.action_set)
        obs_dim     = int(np.prod(env.observation_space.shape))
        num_actions = env.action_space.n

        seed_a_avgs = []
        seed_b_avgs = []

        for seed in common_seeds:
            print(f"\n  Seed {seed}:")

            q_net_a = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
            q_net_a.load_state_dict(
                torch.load(models_a[seed], map_location=device, weights_only=True)
            )
            q_net_a.eval()

            q_net_b = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
            q_net_b.load_state_dict(
                torch.load(models_b[seed], map_location=device, weights_only=True)
            )
            q_net_b.eval()

            env_seed = make_env(env_id, seed, args.action_set)

            stages_to_run = [
                (1, "initial"),
                (2, "key_picked"),
                (3, "door_opened"),
            ] if "DoorKey" in env_id else [(1, "")]

            for stage_num, stage_name in stages_to_run:
                print(f"    Computing {label_a} Q-values for {stage_name or 'Stage 1'} …")
                grid_a, wall_mask, annotations = compute_q_values_grid_for_stage(
                    env_seed, q_net_a, seed, device, stage=stage_num
                )

                print(f"    Computing {label_b} Q-values for {stage_name or 'Stage 1'} …")
                grid_b, _, _ = compute_q_values_grid_for_stage(
                    env_seed, q_net_b, seed, device, stage=stage_num
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    grid_a_max = np.nanmax(grid_a, axis=3)
                    grid_b_max = np.nanmax(grid_b, axis=3)

                seed_a_avgs.append(np.nanmean(grid_a_max))
                seed_b_avgs.append(np.nanmean(grid_b_max))

                seed_label_a = f"{label_a}{get_decay_str(models_a[seed])}"
                seed_label_b = f"{label_b}{get_decay_str(models_b[seed])}"

                print(f"    Generating 4×3 directional heatmap plot ({stage_name}) …")
                plot_heatmap_for_env(
                    env_id, grid_a, grid_b, wall_mask, annotations, seed,
                    label_a=seed_label_a, label_b=seed_label_b, comparison_tag=comparison_tag,
                    stage_name=stage_name,
                )
                
                diff_avg = np.nanmean(grid_a_max) - np.nanmean(grid_b_max)
                print(f"    {seed_label_a} avg max Q : {np.nanmean(grid_a_max):.4f}")
                print(f"    {seed_label_b} avg max Q : {np.nanmean(grid_b_max):.4f}")
                print(f"    Difference                : {diff_avg:+.4f}")

        env.close()

        bar_chart_summary.append({
            "env_id":  env_id,
            "a_mean":  float(np.mean(seed_a_avgs)),
            "b_mean":  float(np.mean(seed_b_avgs)),
            "a_std":   float(np.std(seed_a_avgs))  if len(seed_a_avgs) > 1 else 0.0,
            "b_std":   float(np.std(seed_b_avgs))   if len(seed_b_avgs) > 1 else 0.0,
        })

    plot_bar_chart(bar_chart_summary, label_a=label_a, label_b=label_b,
                   comparison_tag=comparison_tag)

    print("\nAll done.")


if __name__ == "__main__":
    main()
