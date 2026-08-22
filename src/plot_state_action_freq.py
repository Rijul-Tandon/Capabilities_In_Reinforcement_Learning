"""
plot_state_action_freq.py - State Visit & Action Frequency Heatmaps
====================================================================
Generates a 2x3 grid of heatmaps comparing how three different agents
(Random, Baseline DQN, Reward Shaped DQN) navigate a MiniGrid environment.

Top Row: State Visit Frequency Heatmaps
  - Shows which (x, y) tiles on the grid each agent visits most often.
  - Brighter/warmer colors = more visits. Useful for seeing if the agent
    has learned an efficient path or if it wanders aimlessly.

Bottom Row: Most Frequent Action Maps
  - Shows the most common action taken at each tile (e.g., "forward", "left").
  - Reveals whether the agent has developed a coherent navigation policy
    or is acting erratically.

The script automatically finds the latest trained models in the results
directory. No need to manually specify model paths.

Why only 10-20 episodes?
  This script does NOT look at the agent's behavior during training. Instead,
  it takes the fully trained, finished agent, drops it into a brand new
  environment, and evaluates it. Since the agent has already finished learning
  and its policy is locked in, running it for just 20 episodes is more than
  enough to see what paths it prefers to take and what its favorite actions are.
  If we ran it for thousands of episodes, the heatmap would just turn into a
  giant solid block of color and it would take a very long time to generate!

Usage:
  python plot_state_action_freq.py --env-id MiniGrid-Empty-8x8-v0
  python plot_state_action_freq.py --env-id MiniGrid-DoorKey-8x8-v0 --episodes 20

Output:
  plots/<env_id>_state_action_freq.png
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

# argparse: Parses command-line arguments (--env-id, --episodes, etc.)
import argparse

# Path: Object-oriented filesystem paths for finding model files and creating output dirs.
from pathlib import Path

# json: Available for reading config files if needed in future extensions.
import json

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

# gymnasium: The standard RL environment API. Not directly used here but available
#   for type consistency. The actual environment is created via dqn_common.make_env().
import gymnasium as gym

# matplotlib.pyplot (plt): Creates the heatmap visualizations.
#   plt.subplots() creates the 2x3 grid of subplot panels.
#   ax.imshow() renders 2D arrays as colored heatmaps.
#   ax.text() overlays text labels on the heatmap cells.
#   fig.colorbar() adds a color legend bar next to a heatmap.
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# numpy (np): Used for:
#   - np.zeros() to create the counting arrays for visits and actions.
#   - np.prod() to compute the total observation dimension from the shape tuple.
#   - np.argmax() to find the most frequently taken action at each cell.
#   - np.arange() to generate grid line positions.
import numpy as np

# torch: PyTorch deep learning framework. Used to:
#   - Load saved model weights from q_net.pt files using torch.load().
#   - Run forward passes through the Q-Network to get action selections.
#   - torch.no_grad() disables gradient tracking during evaluation (saves memory).
#   - torch.argmax() finds the action with the highest Q-value.
import torch

# ============================================================================
# LOCAL IMPORTS (from our own codebase)
# ============================================================================

# QNetwork: The neural network architecture (MLP) that maps observations to Q-values.
#   We create instances of this class and load trained weights into them.
# make_env: Creates the MiniGrid environment with all wrappers (FullyObs, FlatObs, etc.)
# action_names: Returns human-readable names for the actions (e.g., ["left", "right", "forward"])
from dqn_common import QNetwork, make_env, action_names


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_models_by_seed(results_dir, env_id, exp_name):
    """
    Finds all trained model files for a given experiment, grouped by seed.

    The directories are named like: {env_id}__{exp_name}__{seed}__{timestamp}
    If multiple runs exist for the same seed (re-runs), the newest timestamp wins.

    Parameters
    ----------
    results_dir : str
        Path to the parent directory containing all run folders.
    env_id : str
        The gymnasium environment ID (e.g., "MiniGrid-Empty-5x5-v0").
    exp_name : str
        The experiment name (e.g., "dqn_baseline" or "dqn_reward_shaping").

    Returns
    -------
    dict[int, Path]
        A dict mapping seed -> path to q_net.pt for that seed.
    """
    models = {}  # seed -> latest model path for that seed
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}__*")):
        model_path = run_dir / "q_net.pt"
        config_path = run_dir / "config.json"
        
        if not model_path.exists() or not config_path.exists():
            continue
            
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            seed = int(config["seed"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
            
        # Overwrite with newer run (sorted order ensures newest is last)
        models[seed] = model_path
    return models  # {seed: Path}


def get_agent_data(env, q_net, episodes, seed, num_actions, device, target_stage=None):
    """
    Runs an agent through the environment for multiple episodes and records
    which tiles it visits and which actions it takes, optionally filtered to a specific task stage:
      target_stage=1: Key on ground (initial state)
      target_stage=2: Key picked up (carrying key, door closed)
      target_stage=3: Door opened
    """
    width = env.unwrapped.width
    height = env.unwrapped.height

    state_action_counts = np.zeros((width, height, num_actions, 4), dtype=int)
    visit_counts = np.zeros((width, height), dtype=int)
    layout = {"start": None, "goal": None, "walls": [], "doors": [], "keys": []}

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        
        if ep == 0:
            layout["start"] = env.unwrapped.agent_pos
            for x in range(width):
                for y in range(height):
                    cell = env.unwrapped.grid.get(x, y)
                    if cell is not None:
                        if cell.type == 'wall':
                            layout["walls"].append((x, y))
                        elif cell.type == 'goal':
                            layout["goal"] = (x, y)
                        elif cell.type == 'door':
                            layout["doors"].append((x, y))
                        elif cell.type == 'key':
                            layout["keys"].append((x, y))

        done = False
        has_picked_key = False
        has_opened_door = False

        while not done:
            agent_pos = env.unwrapped.agent_pos
            carrying = env.unwrapped.carrying
            
            if carrying is not None and getattr(carrying, 'type', None) == "key":
                has_picked_key = True

            if not has_opened_door:
                for dx, dy in layout["doors"]:
                    cell = env.unwrapped.grid.get(dx, dy)
                    if cell is not None and getattr(cell, "is_open", False):
                        has_opened_door = True
                        break

            if has_opened_door:
                current_stage = 3
            elif has_picked_key:
                current_stage = 2
            else:
                current_stage = 1

            # Record metrics if no filter or matching requested target stage
            if target_stage is None or current_stage == target_stage:
                if agent_pos is not None:
                    x, y = agent_pos
                    visit_counts[x, y] += 1

            if q_net is None:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_net(obs_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())

            if (target_stage is None or current_stage == target_stage) and agent_pos is not None:
                agent_dir = env.unwrapped.agent_dir
                state_action_counts[agent_pos[0], agent_pos[1], action, agent_dir] += 1

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return visit_counts, state_action_counts, layout


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def get_decay_str(model_path):
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

def plot_all_frequencies(env_id, results_dir, episodes=5, seed=1, hidden_size=256, action_set="task", include_random=True, target_stage=None, stage_name="", plots_dir="plots/reward_comparison"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(env_id, seed, action_set)
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    num_actions = env.action_space.n
    names = action_names(env_id, action_set, num_actions)
    width = env.unwrapped.width
    height = env.unwrapped.height

    baseline_models = get_models_by_seed(results_dir, env_id, "ddqn_baseline")
    if not baseline_models:
        baseline_models = get_models_by_seed(results_dir, env_id, "dqn_baseline")
    shaped_models = get_models_by_seed(results_dir, env_id, "ddqn_reward_shaping")
    if not shaped_models:
        shaped_models = get_models_by_seed(results_dir, env_id, "dqn_reward_shaping")

    baseline_model_path = baseline_models.get(seed)
    shaped_model_path   = shaped_models.get(seed)

    agents = [("Random Agent", None)] if include_random else []

    if baseline_model_path:
        q_net_base = QNetwork(obs_dim, num_actions, hidden_size).to(device)
        q_net_base.load_state_dict(torch.load(baseline_model_path, map_location=device))
        q_net_base.eval()
        agents.append((f"Baseline DQN{get_decay_str(baseline_model_path)}", q_net_base))
    else:
        agents.append(("Baseline DQN (Not Found)", None))

    if shaped_model_path:
        q_net_shape = QNetwork(obs_dim, num_actions, hidden_size).to(device)
        q_net_shape.load_state_dict(torch.load(shaped_model_path, map_location=device))
        q_net_shape.eval()
        agents.append((f"Reward Shaped DQN{get_decay_str(shaped_model_path)}", q_net_shape))
    else:
        agents.append(("Reward Shaped (Not Found)", None))

    # Set publication styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    fig, axes = plt.subplots(2, len(agents), figsize=(6.2 * len(agents), 11.0), squeeze=False)
    title_suffix = f" [{stage_name}]" if stage_name else ""
    fig.suptitle(
        f"{env_id}{title_suffix} — Evaluation Test (Seed {seed})",
        fontsize=17, fontweight="bold", y=0.975
    )

    legend_handles = {}

    for col, (title, q_net) in enumerate(agents):
        if "Not Found" in title:
            axes[0, col].set_title(title, fontsize=12, fontweight="bold")
            axes[1, col].set_title(title, fontsize=12, fontweight="bold")
            continue

        visit_counts, state_action_counts, layout = get_agent_data(
            env, q_net, episodes, seed, num_actions, device, target_stage=target_stage
        )

        im = axes[0, col].imshow(visit_counts.T, origin="upper", cmap="YlOrRd", aspect="equal")
        axes[0, col].set_title(f"{title}\nVisit Frequencies", fontsize=12, fontweight="bold", pad=8)
        cbar = fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        if visit_counts.sum() == 0:
            axes[0, col].text(width / 2 - 0.5, height / 2 - 0.5, "Stage Not Reached",
                              ha="center", va="center", color="red", fontsize=14, fontweight="bold")
            axes[1, col].text(width / 2 - 0.5, height / 2 - 0.5, "Stage Not Reached",
                              ha="center", va="center", color="red", fontsize=14, fontweight="bold")

        if layout["start"] and "Start" not in legend_handles:
            h, = axes[0, col].plot(layout["start"][0], layout["start"][1], 'bo', markersize=9, markeredgecolor='white', label="Start")
            legend_handles["Start"] = h
        elif layout["start"]:
            axes[0, col].plot(layout["start"][0], layout["start"][1], 'bo', markersize=9, markeredgecolor='white')

        if layout["goal"] and "Goal" not in legend_handles:
            h, = axes[0, col].plot(layout["goal"][0], layout["goal"][1], 'g*', markersize=14, markeredgecolor='white', label="Goal")
            legend_handles["Goal"] = h
        elif layout["goal"]:
            axes[0, col].plot(layout["goal"][0], layout["goal"][1], 'g*', markersize=14, markeredgecolor='white')

        for i, (wx, wy) in enumerate(layout["walls"]):
            if "Wall" not in legend_handles:
                h, = axes[0, col].plot(wx, wy, 's', color='#333333', markersize=10, label="Wall")
                legend_handles["Wall"] = h
            else:
                axes[0, col].plot(wx, wy, 's', color='#333333', markersize=10)

        for i, (dx, dy) in enumerate(layout["doors"]):
            if "Door" not in legend_handles:
                h, = axes[0, col].plot(dx, dy, 's', color='saddlebrown', markersize=10, markeredgecolor='white', label="Door")
                legend_handles["Door"] = h
            else:
                axes[0, col].plot(dx, dy, 's', color='saddlebrown', markersize=10, markeredgecolor='white')

        for i, (kx, ky) in enumerate(layout["keys"]):
            if "Key" not in legend_handles:
                h, = axes[0, col].plot(kx, ky, 'yD', markersize=8, markeredgecolor='black', label="Key")
                legend_handles["Key"] = h
            else:
                axes[0, col].plot(kx, ky, 'yD', markersize=8, markeredgecolor='black')

        axes[1, col].imshow(visit_counts.T, origin="upper", cmap="Blues", alpha=0.35, aspect="equal")
        axes[1, col].set_title(f"{title}\nAction Counts", fontsize=12, fontweight="bold", pad=8)

        abbr_map = {"left": "L", "right": "R", "forward": "F", "pickup": "P", "drop": "Dp", "toggle": "T", "done": "Dn"}
        dir_map = {0: "E", 1: "S", 2: "W", 3: "N"}

        for x in range(width):
            for y in range(height):
                f_parts = []
                other_parts = []
                for act_idx in range(num_actions):
                    act_name = names[act_idx]
                    abbr = abbr_map.get(act_name, act_name[:1].upper())
                    
                    if act_name == "forward":
                        for d_idx in range(4):
                            count = state_action_counts[x, y, act_idx, d_idx]
                            if count > 0:
                                f_parts.append(f"{dir_map[d_idx]}:{count}")
                    else:
                        count = np.sum(state_action_counts[x, y, act_idx, :])
                        if count > 0:
                            other_parts.append(f"{abbr}:{count}")
                
                lines = []
                if other_parts:
                    if len(other_parts) <= 2:
                        lines.append("  ".join(other_parts))
                    else:
                        lines.append("  ".join(other_parts[:2]))
                        lines.append("  ".join(other_parts[2:]))
                if f_parts:
                    lines.append("F: " + " ".join(f_parts))

                cell_text = "\n".join(lines)
                alpha_val = 1.0 if visit_counts[x, y] > 0 else 0.2
                
                if cell_text:
                    axes[1, col].text(
                        x, y, cell_text,
                        ha="center", va="center", fontsize=6.5, fontweight="bold", color="black", alpha=alpha_val
                    )

        for ax in [axes[0, col], axes[1, col]]:
            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            ax.grid(which="minor", color="#888888", linestyle="-", linewidth=0.8)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

            # Add bold white text markers with black stroke for S, G, D, K
            annotations = {}
            if layout["start"]:
                annotations[layout["start"]] = "S"
            if layout["goal"]:
                annotations[layout["goal"]] = "G"
            for dx, dy in layout["doors"]:
                annotations[(dx, dy)] = "D"
            for kx, ky in layout["keys"]:
                annotations[(kx, ky)] = "K"

            for (ax_x, ax_y), label in annotations.items():
                ax.text(
                    ax_x + 0.32, ax_y - 0.32, label,
                    color='white', fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    path_effects=[path_effects.withStroke(linewidth=2, foreground='black')]
                )

    # Top layout legend
    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc="upper center", bbox_to_anchor=(0.5, 0.935),
            ncol=len(legend_handles), fontsize=10, frameon=True,
            facecolor="white", edgecolor="#cccccc", framealpha=0.95
        )

    legend_parts = []
    for act_name in names:
        abbr = abbr_map.get(act_name, act_name[:1].upper())
        legend_parts.append(f"{abbr} = {act_name}")
    legend_text = "   |   ".join(legend_parts)

    fig.text(
        0.5, 0.02,
        f"Action key:  {legend_text}",
        ha="center", va="bottom",
        fontsize=10, fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.95),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    suffix = f"_{stage_name.lower().replace(' ', '_')}" if stage_name else ""
    output_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{env_id}_test_seed{seed}{suffix}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Test plot saved to {output_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    parser.add_argument("--plots-dir", type=str, default="plots/reward_comparison")

    args = parser.parse_args()

    baseline_models = get_models_by_seed(args.results_dir, args.env_id, "ddqn_baseline")
    if not baseline_models:
        baseline_models = get_models_by_seed(args.results_dir, args.env_id, "dqn_baseline")
    shaped_models = get_models_by_seed(args.results_dir, args.env_id, "ddqn_reward_shaping")
    if not shaped_models:
        shaped_models = get_models_by_seed(args.results_dir, args.env_id, "dqn_reward_shaping")

    all_seeds = sorted(set(baseline_models.keys()) | set(shaped_models.keys()))

    if not all_seeds:
        print(f"No trained models found for {args.env_id} in {args.results_dir}")
        raise SystemExit(1)

    print(f"Found seeds: {all_seeds} for {args.env_id}")

    first_seed = all_seeds[0]
    stages_to_run = [(None, "")]

    for seed in all_seeds:
        for stage_num, stage_name in stages_to_run:
            print(f"Generating combined test plot for seed={seed} ...")
            plot_all_frequencies(
                env_id=args.env_id,
                results_dir=args.results_dir,
                episodes=args.episodes,
                seed=seed,
                action_set=args.action_set,
                include_random=seed == first_seed,
                target_stage=stage_num,
                stage_name=stage_name,
                plots_dir=args.plots_dir,
            )

    print(f"Done. Performance test plots saved to {args.plots_dir}")
