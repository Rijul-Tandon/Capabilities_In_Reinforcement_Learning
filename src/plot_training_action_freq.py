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

def plot_3x4_frequencies(env_id, results_dir, seed, action_set, suffix="", title_suffix="(All Steps)", include_random=True):
    # Find directories
    random_dirs = get_dirs_by_seed(results_dir, env_id, "random_agent")
    baseline_dirs = get_dirs_by_seed(results_dir, env_id, "ddqn_baseline")
    if not baseline_dirs:
        baseline_dirs = get_dirs_by_seed(results_dir, env_id, "dqn_baseline")
    shaped_dirs = get_dirs_by_seed(results_dir, env_id, "ddqn_reward_shaping")
    if not shaped_dirs:
        shaped_dirs = get_dirs_by_seed(results_dir, env_id, "dqn_reward_shaping")

    random_seed = min(random_dirs.keys()) if random_dirs else None
    agents = []
    if include_random and random_seed is not None:
        agents.append((f"Random Agent (seed {random_seed})", random_dirs[random_seed]))
    agents.extend([
        ("Baseline DDQN", baseline_dirs.get(seed)),
        ("Shaped DDQN", shaped_dirs.get(seed)),
    ])

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

            # Color each cell by total active-action frequency for this facing direction.
            # Empty has only navigation actions, so plotting only pickup/toggle would be blank.
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
                        # If the cell is very dark (high count), use white text
                        max_total = action_total.max()
                        text_color = "white" if max_total > 0 and np.log1p(action_total[x, y]) > np.log1p(max_total) * 0.5 else "black"
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
    legend_text = "  |  ".join(f"{abbr_map.get(name, name[:1].upper())} = {name}" for name in names)
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
        
    first_seed = seeds[0]
    for seed in seeds:
        include_random = seed == first_seed
        print(f"Generating training action freq plot for {args.env_id} seed={seed} (All Steps) ...")
        plot_3x4_frequencies(args.env_id, args.results_dir, seed, args.action_set, suffix="", title_suffix="(All Steps)", include_random=include_random)
        print(f"Generating training action freq plot for {args.env_id} seed={seed} (Second Half) ...")
        plot_3x4_frequencies(args.env_id, args.results_dir, seed, args.action_set, suffix="_second_half", title_suffix="(Second Half)", include_random=include_random)
