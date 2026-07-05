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
    """Extracts walls from the environment for overlaying on the plot."""
    width = env.unwrapped.width
    height = env.unwrapped.height
    grid = env.unwrapped.grid
    
    wall_mask = np.zeros((width, height), dtype=bool)
    
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "wall":
                wall_mask[x, y] = True
                
    return wall_mask, width, height

def plot_3x4_frequencies(env_id, results_dir, seed, action_set):
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
    env.reset()
    wall_mask, width, height = extract_layout(env)
    
    num_actions = env.action_space.n
    names = action_names(env_id, action_set, num_actions)
    env.close()

    # Create 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"Cumulative Action Frequencies During Training ({env_id} | Seed {seed})", fontsize=20, y=0.95)

    abbr_map = {
        "left": "L", "right": "R", "forward": "F",
        "pickup": "P", "drop": "D", "toggle": "T", "done": "X"
    }

    for row, (agent_name, run_dir) in enumerate(agents):
        # Load the tracking array
        if run_dir and (run_dir / "state_action_counts.npy").exists():
            counts = np.load(run_dir / "state_action_counts.npy")
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

            # We want to color the cell based on the total number of actions taken
            # in that direction. We'll use a log scale so heavily visited areas don't wash out everything.
            dir_total = counts[:, :, d_idx, :].sum(axis=-1)
            
            # Create a heatmap background (transpose because imshow expects (y, x))
            # We use log1p to compress the dynamic range of training counts (which can hit 100k+)
            heatmap = ax.imshow(np.log1p(dir_total).T, cmap="Blues", origin="upper", aspect="equal")
            
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
                    total_cell = cell_counts.sum()
                    if total_cell == 0:
                        continue
                        
                    lines = []
                    # Only show actions that were taken at least once
                    for a_idx, a_name in enumerate(names):
                        c = cell_counts[a_idx]
                        if c > 0:
                            abbr = abbr_map.get(a_name, a_name)
                            # Convert count to K if it's large
                            c_str = f"{c/1000:.1f}k" if c >= 1000 else str(c)
                            lines.append(f"{abbr}:{c_str}")
                            
                    if lines:
                        # Split into columns if there are many actions to fit in the box
                        if len(lines) > 3:
                            half = len(lines) // 2 + len(lines) % 2
                            col1 = "\n".join(lines[:half])
                            col2 = "\n".join(lines[half:])
                            text = f"{col1}\n--\n{col2}"
                        else:
                            text = "\n".join(lines)
                            
                        # If the cell is very dark (high count), use white text
                        text_color = "white" if np.log1p(total_cell) > np.log1p(dir_total.max()) * 0.5 else "black"
                        
                        ax.text(x, y, text, ha="center", va="center", fontsize=7, color=text_color, fontweight="bold")

            # Grid lines
            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # Action Legend
    legend_parts = [f"{abbr_map.get(n, n)}={n}" for n in names]
    legend_text = "  |  ".join(legend_parts)
    fig.text(0.5, 0.02, f"Action key:  {legend_text}", ha="center", va="bottom", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#aaaaaa", alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    # Save inside the sub-folder requested by the user
    output_dir = Path("plots/action_freq_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{env_id}_training_action_freq_seed{seed}.png"
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
        print(f"Generating training action freq plot for {args.env_id} seed={seed} ...")
        plot_3x4_frequencies(args.env_id, args.results_dir, seed, args.action_set)
