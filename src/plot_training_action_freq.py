"""
plot_training_action_freq.py - Cumulative Action Frequency During Training
==========================================================================
Visualizes the total number of times each action was taken in every grid cell
(broken down by the direction the agent was facing) OVER THE ENTIRE TRAINING RUN.

This creates a 3x4 grid of heatmaps:
- Rows: Random Agent, Baseline DDQN, Reward-Shaped DDQN
- Columns: Facing North (3), Facing East (0), Facing South (1), Facing West (2)

Cells are colored based on the "Futile Action Ratio" (how often the agent took
an action that didn't change its state, such as walking into a wall).
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

def plot_3x4_frequencies(env_id, results_dir, seed, action_set, suffix="", title_suffix="(All Steps)"):
    # Find directories
    random_dirs = get_dirs_by_seed(results_dir, env_id, "random_agent")
    baseline_dirs = get_dirs_by_seed(results_dir, env_id, "ddqn_baseline")
    if not baseline_dirs:
        baseline_dirs = get_dirs_by_seed(results_dir, env_id, "dqn_baseline")
    shaped_dirs = get_dirs_by_seed(results_dir, env_id, "ddqn_reward_shaping")
    if not shaped_dirs:
        shaped_dirs = get_dirs_by_seed(results_dir, env_id, "dqn_reward_shaping")

    agents = [
        ("Random Agent", random_dirs.get(seed)),
        ("Baseline DDQN", baseline_dirs.get(seed)),
        ("Shaped DDQN", shaped_dirs.get(seed)),
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

    # Find the indices for pickup and toggle
    try:
        pickup_idx = names.index("pickup")
    except ValueError:
        pickup_idx = -1
    try:
        toggle_idx = names.index("toggle")
    except ValueError:
        toggle_idx = -1

    # Create 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"Cumulative Action Frequencies During Training {title_suffix} ({env_id} | Seed {seed})", fontsize=20, y=0.95)

    abbr_map = {
        "left": "L", "right": "R", "forward": "F",
        "pickup": "P", "drop": "D", "toggle": "T", "done": "X"
    }

    for row, (agent_name, run_dir) in enumerate(agents):
        # Load the tracking array
        counts_path = run_dir / f"state_action_counts{suffix}.npy" if run_dir else None
        if counts_path and counts_path.exists():
            counts = np.load(counts_path)
            # shape = (width, height, 4, num_actions)
        else:
            counts = np.zeros((width, height, 4, num_actions), dtype=np.int64)

        for col, (d_idx, d_name) in enumerate(dir_info):
            ax = axes[row, col]
            
            # Row labels on the far left
            if col == 0:
                ax.set_ylabel(agent_name, fontsize=16, fontweight="bold")
            
            # Column labels on the top
            if row == 0:
                ax.set_title(f"Facing {d_name}", fontsize=16)

            # We want to color the cell based on the total number of Pickup and Toggle actions
            # since these are the "futile" actions that get masked by reward shaping.
            if pickup_idx != -1 and toggle_idx != -1:
                pt_total = counts[:, :, d_idx, pickup_idx] + counts[:, :, d_idx, toggle_idx]
            elif pickup_idx != -1:
                pt_total = counts[:, :, d_idx, pickup_idx]
            elif toggle_idx != -1:
                pt_total = counts[:, :, d_idx, toggle_idx]
            else:
                pt_total = np.zeros((width, height))
            
            # Create a heatmap background (transpose because imshow expects (y, x))
            heatmap = ax.imshow(np.log1p(pt_total).T, cmap="Blues", origin="upper", aspect="equal")
            
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
                    # Only show Pickup and Toggle
                    if pickup_idx != -1:
                        c = cell_counts[pickup_idx]
                        c_str = f"{c/1000:.1f}k" if c >= 1000 else str(c)
                        lines.append(f"P: {c_str}")
                    if toggle_idx != -1:
                        c = cell_counts[toggle_idx]
                        c_str = f"{c/1000:.1f}k" if c >= 1000 else str(c)
                        lines.append(f"T: {c_str}")
                            
                    if lines:
                        text = "\n".join(lines)
                        # If the cell is very dark (high count), use white text
                        text_color = "white" if np.log1p(pt_total[x, y]) > np.log1p(pt_total.max()) * 0.5 else "black"
                        ax.text(x, y, text, ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")
                        
                    # Add annotation (S, G, K, D) if present
                    if (x, y) in annotations:
                        label = annotations[(x, y)]
                        # Draw label in bottom-right corner of the cell
                        ax.text(x + 0.35, y + 0.35, label, color='red', fontsize=10, 
                                fontweight='bold', ha='center', va='center')

            # Grid lines
            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # Action Legend
    legend_text = "P = pickup  |  T = toggle"
    fig.text(0.5, 0.02, f"Action key:  {legend_text}    ---    Red labels (S=Start, G=Goal, K=Key, D=Door)", ha="center", va="bottom", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#aaaaaa", alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    # Save inside the sub-folder requested by the user
    output_dir = Path("plots/action_freq_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{env_id}_training_action_freq{suffix}_seed{seed}.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    args = parser.parse_args()

    # Find seeds available in baseline models
    baseline_dirs = get_dirs_by_seed(args.results_dir, args.env_id, "ddqn_baseline")
    if not baseline_dirs:
        baseline_dirs = get_dirs_by_seed(args.results_dir, args.env_id, "dqn_baseline")
        
    seeds = sorted(list(baseline_dirs.keys()))
    if not seeds:
        print(f"No trained baseline models found for {args.env_id} in {args.results_dir}. Cannot determine seeds.")
        raise SystemExit(1)
        
    for seed in seeds:
        print(f"Generating training action freq plot for {args.env_id} seed={seed} (All Steps) ...")
        plot_3x4_frequencies(args.env_id, args.results_dir, seed, args.action_set, suffix="", title_suffix="(All Steps)")
        print(f"Generating training action freq plot for {args.env_id} seed={seed} (Second Half) ...")
        plot_3x4_frequencies(args.env_id, args.results_dir, seed, args.action_set, suffix="_second_half", title_suffix="(Second Half)")
