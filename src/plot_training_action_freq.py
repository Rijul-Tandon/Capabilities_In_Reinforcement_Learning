"""
plot_training_action_freq.py - Cumulative Action Frequency During Training
==========================================================================
Visualizes the total number of times each action was taken in every grid cell
(broken down by the direction the agent was facing) OVER THE ENTIRE TRAINING RUN.

This creates a grid of heatmaps:
- Rows: Baseline DDQN and Reward-Shaped DDQN, plus Random Agent on the first seed only
- Columns: Facing North (3), Facing East (0), Facing South (1), Facing West (2)

Cells are colored based on how often the active actions were taken from each
state/direction pair.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# We only need action_names to label the text inside the cells.
from dqn_common import action_names, make_env

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
    """Extracts walls and annotations from the environment for overlaying on the plot."""
    width = env.unwrapped.width
    height = env.unwrapped.height
    grid = env.unwrapped.grid
    
    wall_mask = np.zeros((width, height), dtype=bool)
    annotations = {}
    
    # Start position
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

def get_decay_str(run_dir):
    if not run_dir:
        return ""
    cfg_path = run_dir / "config.json"
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

def plot_3x4_frequencies(env_id, results_dir, seed, action_set, suffix="", title_suffix="(All Steps)", plots_dir="plots/action_freq_plots", stage_idx=None, stage_name=""):
    # Find directories
    baseline_dirs = get_dirs_by_seed(results_dir, env_id, "ddqn_baseline")
    if not baseline_dirs:
        baseline_dirs = get_dirs_by_seed(results_dir, env_id, "dqn_baseline")
    shaped_dirs = get_dirs_by_seed(results_dir, env_id, "ddqn_reward_shaping")
    if not shaped_dirs:
        shaped_dirs = get_dirs_by_seed(results_dir, env_id, "dqn_reward_shaping")

    b_dir = baseline_dirs.get(seed)
    s_dir = shaped_dirs.get(seed)

    agents = [
        (f"Baseline DDQN{get_decay_str(b_dir)}", b_dir),
        (f"RS-DDQN{get_decay_str(s_dir)}", s_dir),
    ]

    # Directions in MiniGrid: 0=East, 1=South, 2=West, 3=North
    # We map them to columns in a logical order
    dir_info = [
        (3, "North"),
        (0, "East"),
        (1, "South"),
        (2, "West")
    ]

    # Dummy env to get layout and action names
    env = make_env(env_id, seed, action_set, capture_video=False, run_name="dummy", max_steps=10)
    env.reset(seed=seed)
    wall_mask, annotations, width, height = extract_layout(env)
    
    num_actions = env.action_space.n
    names = action_names(env_id, action_set, num_actions)
    env.close()

    # Create one row per plotted agent and four columns for facing direction.
    fig, axes = plt.subplots(len(agents), 4, figsize=(20, 5 * len(agents)))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Cumulative Action Frequencies During Training {title_suffix} ({env_id} | Seed {seed})", fontsize=20, y=0.95)

    abbr_map = {
        "left": "L", "right": "R", "forward": "F",
        "pickup": "P", "drop": "D", "toggle": "T", "done": "X"
    }

    for row, (agent_name, run_dir) in enumerate(agents):
        # Load the tracking array
        counts_path = run_dir / f"state_action_counts{suffix}.npy" if run_dir else None
        fallback_path = run_dir / "state_action_counts.npy" if run_dir else None
        if counts_path and counts_path.exists():
            raw_counts = np.load(counts_path)
        elif fallback_path and fallback_path.exists():
            raw_counts = np.load(fallback_path)
        else:
            raw_counts = np.zeros((width, height, 4, num_actions), dtype=np.int64)

        if raw_counts.ndim == 5:
            if stage_idx is not None and stage_idx < raw_counts.shape[4]:
                counts = raw_counts[:, :, :, :, stage_idx]
            else:
                counts = raw_counts.sum(axis=-1)
        else:
            counts = raw_counts

        for col, (d_idx, d_name) in enumerate(dir_info):
            ax = axes[row, col]
            
            # Row labels on the far left
            if col == 0:
                ax.set_ylabel(agent_name, fontsize=16, fontweight="bold")
            
            # Column labels on the top
            if row == 0:
                ax.set_title(f"Facing {d_name}", fontsize=16)

            # Color each cell by total active-action frequency for this facing direction.
            action_total = counts[:, :, d_idx, :].sum(axis=-1)
            
            # Create a heatmap background (transpose because imshow expects (y, x))
            ax.imshow(np.log1p(action_total).T, cmap="Blues", origin="upper", aspect="equal")
            
            # Overlay walls in dark gray
            wall_layer = np.full((height, width, 4), [0.0, 0.0, 0.0, 0.0])
            wall_layer[wall_mask.T] = [0.2, 0.2, 0.2, 1.0] # Dark gray, opaque
            ax.imshow(wall_layer, origin="upper", aspect="equal")

            # Write text inside the cells
            for x in range(width):
                for y in range(height):
                    if wall_mask[x, y]:
                        continue
                        
                    cell_counts = counts[x, y, d_idx, :]
                    
                    lines = []
                    for act_idx, act_name in enumerate(names):
                        c = cell_counts[act_idx]
                        if c <= 0:
                            continue
                        abbr = abbr_map.get(act_name, act_name[:1].upper())
                        c_str = f"{c/1000:.1f}k" if c >= 1000 else str(c)
                        lines.append(f"{abbr}: {c_str}")
                            
                    if lines:
                        text = "\n".join(lines)
                        max_total = action_total.max()
                        text_color = "white" if max_total > 0 and np.log1p(action_total[x, y]) > np.log1p(max_total) * 0.55 else "black"
                        ax.text(x, y, text, ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")
                        
                    # Add annotation (S, G, K, D) if present in top-right of cell
                    if (x, y) in annotations:
                        label = annotations[(x, y)]
                        ax.text(x + 0.32, y - 0.32, label, color='white', fontsize=10, 
                                fontweight='bold', ha='center', va='center',
                                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

            # Grid lines
            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            ax.grid(which="minor", color="#888888", linestyle="-", linewidth=0.8)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # Action Legend
    legend_text = "  |  ".join(f"{abbr_map.get(name, name[:1].upper())} = {name}" for name in names)
    fig.text(0.5, 0.02, f"Action key:  {legend_text}    ---    Red labels (S=Start, G=Goal, K=Key, D=Door)", ha="center", va="bottom", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.95))

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    # Save inside the output folder requested
    output_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_file_suffix = f"_{stage_name.lower().replace(' ', '_')}" if stage_name else ""
    output_path = output_dir / f"{env_id}_training_action_freq{suffix}_seed{seed}{stage_file_suffix}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    parser.add_argument("--plots-dir", type=str, default="plots/action_freq_plots")
    args = parser.parse_args()

    # Find seeds available in baseline models
    baseline_dirs = get_dirs_by_seed(args.results_dir, args.env_id, "ddqn_baseline")
    if not baseline_dirs:
        baseline_dirs = get_dirs_by_seed(args.results_dir, args.env_id, "dqn_baseline")
        
    seeds = sorted(list(baseline_dirs.keys()))
    if not seeds:
        print(f"No trained baseline models found for {args.env_id} in {args.results_dir}. Cannot determine seeds.")
        raise SystemExit(1)
        
    stages_to_run = [
        (0, "initial"),
        (1, "key_picked"),
        (2, "door_opened"),
    ] if "DoorKey" in args.env_id else [(None, "")]

    env_clean = "DoorKey" if "DoorKey" in args.env_id else ("Empty" if "Empty" in args.env_id else args.env_id)

    for seed in seeds:
        dir_50 = Path(args.plots_dir) / "last_50_percent" / env_clean / f"seed_{seed}"
        dir_25 = Path(args.plots_dir) / "last_25_percent" / env_clean / f"seed_{seed}"

        for stage_idx, stage_name in stages_to_run:
            s_title = f" [{stage_name}]" if stage_name else ""
            print(f"Generating training action freq plot for {args.env_id} seed={seed} (Last 50%{s_title}) ...")
            plot_3x4_frequencies(
                args.env_id, args.results_dir, seed, args.action_set, 
                suffix="_last_half", title_suffix=f"(Last 50%{s_title})", 
                plots_dir=dir_50,
                stage_idx=stage_idx, stage_name=stage_name
            )
            print(f"Generating training action freq plot for {args.env_id} seed={seed} (Last 25%{s_title}) ...")
            plot_3x4_frequencies(
                args.env_id, args.results_dir, seed, args.action_set, 
                suffix="_last_quarter", title_suffix=f"(Last 25%{s_title})", 
                plots_dir=dir_25,
                stage_idx=stage_idx, stage_name=stage_name
            )
