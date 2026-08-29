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

# random: Python's random module for uniform sampling.
import random

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

DIR_ARROWS = {0: "→", 1: "↓", 2: "←", 3: "↑"}

def get_action_arrow_or_symbol(act_name, facing_dir):
    """
    Given an action name and agent's facing direction (0=E, 1=S, 2=W, 3=N),
    returns the resulting movement direction arrow along with the action abbreviation.
    """
    abbr_map = {
        "left": "L", "right": "R", "forward": "F",
        "pickup": "P", "drop": "Dp", "toggle": "T", "done": "Dn"
    }
    abbr = abbr_map.get(act_name.lower(), act_name[:1].upper())

    if act_name.lower() == "forward":
        return f"{DIR_ARROWS[facing_dir]} {abbr}"
    elif act_name.lower() == "left":
        return f"{DIR_ARROWS[(facing_dir - 1) % 4]} {abbr}"
    elif act_name.lower() == "right":
        return f"{DIR_ARROWS[(facing_dir + 1) % 4]} {abbr}"
    return abbr


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
            seed_val = config.get("seed", 1)
            if isinstance(seed_val, list):
                if not seed_val: continue
                seed = int(seed_val[0])
            else:
                seed = int(seed_val)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue
            
        # Overwrite with newer run (sorted order ensures newest is last)
        models[seed] = model_path
    return models  # {seed: Path}


def get_agent_data(env, q_net, episodes, seed, num_actions, device, target_stage=None, epsilon=0.0):
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
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        q_values = q_net(obs_tensor).squeeze(0)
                        max_q = torch.max(q_values)
                        max_indices = (q_values == max_q).nonzero(as_tuple=True)[0].tolist()
                        action = int(random.choice(max_indices))

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

def plot_all_frequencies(env_id, results_dir, episodes=5, seed=1, hidden_size=256, action_set="task", include_random=True, target_stage=None, stage_name="", plots_dir="plots/reward_comparison", epsilon=0.0):
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
        agents.append((f"Baseline DDQN{get_decay_str(baseline_model_path)}", q_net_base))
    else:
        agents.append(("Baseline DDQN (Not Found)", None))

    if shaped_model_path:
        q_net_shape = QNetwork(obs_dim, num_actions, hidden_size).to(device)
        q_net_shape.load_state_dict(torch.load(shaped_model_path, map_location=device))
        q_net_shape.eval()
        agents.append((f"RS-DDQN{get_decay_str(shaped_model_path)}", q_net_shape))
    else:
        agents.append(("RS-DDQN (Not Found)", None))

    # Set publication styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    fig, axes = plt.subplots(1, len(agents), figsize=(8.5 * len(agents), 7.5), squeeze=False)
    title_suffix = f" [{stage_name}]" if stage_name else ""
    eps_label = "Greedy (Q-values)" if epsilon == 0.0 else f"5% Exploration (ε={epsilon:g})"
    fig.suptitle(
        f"{env_id}{title_suffix} — Evaluation Test [{eps_label}] (Seed {seed})",
        fontsize=17, fontweight="bold", y=1.05
    )

    legend_handles = {}

    for col, (title, q_net) in enumerate(agents):
        if "Not Found" in title:
            axes[0, col].set_title(title, fontsize=12, fontweight="bold")
            continue

        visit_counts, state_action_counts, layout = get_agent_data(
            env, q_net, episodes, seed, num_actions, device, target_stage=target_stage, epsilon=epsilon
        )

        if visit_counts.sum() == 0:
            axes[0, col].text(width / 2 - 0.5, height / 2 - 0.5, "Stage Not Reached",
                              ha="center", va="center", color="red", fontsize=14, fontweight="bold")

        for ax_row in [0]:
            if layout["start"] and "Start" not in legend_handles:
                h, = axes[ax_row, col].plot(layout["start"][0] - 0.35, layout["start"][1] - 0.35, 'bo', markersize=9, markeredgecolor='white', label="Start")
                legend_handles["Start"] = h
            elif layout["start"]:
                axes[ax_row, col].plot(layout["start"][0] - 0.35, layout["start"][1] - 0.35, 'bo', markersize=9, markeredgecolor='white')

            if layout["goal"] and "Goal" not in legend_handles:
                h, = axes[ax_row, col].plot(layout["goal"][0], layout["goal"][1], 'g*', markersize=14, markeredgecolor='white', label="Goal")
                legend_handles["Goal"] = h
            elif layout["goal"]:
                axes[ax_row, col].plot(layout["goal"][0], layout["goal"][1], 'g*', markersize=14, markeredgecolor='white')

            for i, (wx, wy) in enumerate(layout["walls"]):
                if "Wall" not in legend_handles:
                    h, = axes[ax_row, col].plot(wx, wy, 's', color='#333333', markersize=10, label="Wall")
                    legend_handles["Wall"] = h
                else:
                    axes[ax_row, col].plot(wx, wy, 's', color='#333333', markersize=10)

            for i, (dx, dy) in enumerate(layout["doors"]):
                if "Door" not in legend_handles:
                    h, = axes[ax_row, col].plot(dx, dy, 's', color='saddlebrown', markersize=10, markeredgecolor='white', label="Door")
                    legend_handles["Door"] = h
                else:
                    axes[ax_row, col].plot(dx, dy, 's', color='saddlebrown', markersize=10, markeredgecolor='white')

            for i, (kx, ky) in enumerate(layout["keys"]):
                if "Key" not in legend_handles:
                    h, = axes[ax_row, col].plot(kx, ky, 'yD', markersize=8, markeredgecolor='black', label="Key")
                    legend_handles["Key"] = h
                else:
                    axes[ax_row, col].plot(kx, ky, 'yD', markersize=8, markeredgecolor='black')

        im1 = axes[0, col].imshow(np.log1p(visit_counts.T), origin="upper", cmap="Blues", alpha=0.50, aspect="equal")
        axes[0, col].set_title(f"{title}\nAction Counts", fontsize=12, fontweight="bold", pad=8)
        cbar1 = fig.colorbar(im1, ax=axes[0, col], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)

        abbr_map = {"left": "L", "right": "R", "forward": "F", "pickup": "P", "drop": "Dp", "toggle": "T", "done": "Dn"}
        dir_map = {0: "E", 1: "S", 2: "W", 3: "N"}

        for x in range(width):
            for y in range(height):
                symbol_counts = {}
                for act_idx in range(num_actions):
                    act_name = names[act_idx]
                    for d_idx in range(4):
                        cnt = state_action_counts[x, y, act_idx, d_idx]
                        if cnt > 0:
                            sym = get_action_arrow_or_symbol(act_name, d_idx)
                            symbol_counts[sym] = symbol_counts.get(sym, 0) + cnt
                
                if symbol_counts:
                    max_cnt = max(symbol_counts.values())
                    max_symbols = [s for s, c in symbol_counts.items() if c == max_cnt]
                    arrow_str = "\n".join(max_symbols)
                    cell_text = f"{arrow_str}\n{visit_counts[x, y]}"
                    alpha_val = 1.0 if visit_counts[x, y] > 0 else 0.2
                    axes[0, col].text(
                        x, y, cell_text,
                        ha="center", va="center", fontsize=8.0, fontweight="bold", color="black", alpha=alpha_val
                    )

        for ax in [axes[0, col]]:
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
                offset_x, offset_y = (-0.35, -0.35) if label == "S" else (0.32, -0.32)
                ax.text(
                    ax_x + offset_x, ax_y + offset_y, label,
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

    if epsilon == 0.0:
        eps_suffix = "_q_values"
    else:
        eps_suffix = f"_eps_{epsilon:g}".replace(".", "_")

    output_path = output_dir / f"{env_id}_test_seed{seed}{eps_suffix}{suffix}.png"
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
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    parser.add_argument("--plots-dir", type=str, default="plots/reward_comparison")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.0, 0.05], help="Epsilon exploration values for evaluation (default: 0.0 0.05)")

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
        import sys
        sys.exit(0)

    print(f"Found seeds: {all_seeds} for {args.env_id}")

    stages_to_run = [(None, "")]

    for seed in all_seeds:
        for stage_num, stage_name in stages_to_run:
            for eps in args.epsilons:
                mode_desc = "Greedy Q-values" if eps == 0.0 else f"{eps*100:g}% random exploration"
                print(f"Generating combined test plot for seed={seed} ({mode_desc}) ...")
                plot_all_frequencies(
                    env_id=args.env_id,
                    results_dir=args.results_dir,
                    episodes=args.episodes,
                    seed=seed,
                    action_set=args.action_set,
                    include_random=False,
                    target_stage=stage_num,
                    stage_name=stage_name,
                    plots_dir=args.plots_dir,
                    epsilon=eps,
                )

    print(f"Done. Performance test plots saved to {args.plots_dir}")

