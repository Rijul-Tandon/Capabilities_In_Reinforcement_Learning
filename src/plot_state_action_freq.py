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
        if not model_path.exists():
            continue
        # Directory name format: env__exp__seed__timestamp
        parts = run_dir.name.split("__")
        try:
            seed = int(parts[2])
        except (IndexError, ValueError):
            continue
        # Overwrite with newer run (sorted order ensures newest is last)
        models[seed] = model_path
    return models  # {seed: Path}


def get_agent_data(env, q_net, episodes, seed, num_actions, device):
    """
    Runs an agent (either trained or random) through the environment for multiple
    episodes and records which tiles it visits and which actions it takes.

    The function handles two cases:
      - q_net is None: Simulates a random agent (env.action_space.sample())
      - q_net is a QNetwork: Uses the trained model to select greedy actions

    Parameters
    ----------
    env : gym.Env
        The wrapped MiniGrid environment. Must have .unwrapped.agent_pos
        and .unwrapped.width/.height accessible for position tracking.
    q_net : QNetwork or None
        The trained neural network for action selection, or None for random actions.
    episodes : int
        Number of episodes to run for data collection.
    seed : int
        Base seed for environment resets. Each episode uses seed + episode_number
        to ensure different starting configurations.
    num_actions : int
        Total number of possible actions (needed for the counting array dimensions).
    device : torch.device
        The device (CPU/GPU) to run the neural network on.

    Returns
    -------
    tuple of np.ndarray
        visit_counts: shape (width, height) - how many times each tile was visited.
        state_action_counts: shape (width, height, num_actions) - how many times
            each action was taken at each tile.
    """
    width = env.unwrapped.width
    height = env.unwrapped.height

    # Initialize counting arrays with zeros
    state_action_counts = np.zeros((width, height, num_actions), dtype=int)
    visit_counts = np.zeros((width, height), dtype=int)

    for ep in range(episodes):
        # Reset with a different seed each episode to get varied layouts
        # (important for randomized environments like DoorKey and FourRooms)
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            # Access the raw MiniGrid environment (bypassing all wrappers)
            # to read the agent's actual (x, y) position on the grid.
            # The wrappers modify observations but don't change the underlying position.
            agent_pos = env.unwrapped.agent_pos

            if agent_pos is not None:
                x, y = agent_pos
                visit_counts[x, y] += 1

                if q_net is None:
                    # Random agent: uniformly sample from the action space
                    action = env.action_space.sample()
                else:
                    # Trained agent: use the neural network to select the best action
                    with torch.no_grad():  # Disable gradients (evaluation mode, not training)
                        # Convert numpy observation to PyTorch tensor
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        # unsqueeze(0) adds batch dimension: (obs_dim,) -> (1, obs_dim)
                        # Forward pass returns Q-values for all actions: shape (1, num_actions)
                        q_values = q_net(obs_tensor)
                        # argmax returns the action index with the highest Q-value
                        action = int(torch.argmax(q_values, dim=1).item())

                # Record which action was taken at this position
                state_action_counts[x, y, action] += 1

            # Step the environment with the chosen action
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    return visit_counts, state_action_counts


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def plot_all_frequencies(env_id, results_dir, episodes=20, seed=1, hidden_size=256, action_set="task"):
    """
    Generates the 2x3 comparison plot for all three agents FOR A SPECIFIC SEED.
    Saves the output as: plots/{env_id}_state_action_freq_seed{seed}.png

    Layout:
      Column 0: Random Agent  | Column 1: Baseline DQN    | Column 2: Reward Shaped DQN
      Row 0: Visit Frequencies  (heatmaps showing where each agent goes)
      Row 1: Action Counts      (text labels: count for every action at every cell)

    Parameters
    ----------
    env_id : str
        The gymnasium environment ID.
    results_dir : str
        Path to the directory containing run folders with trained models.
    episodes : int
        Number of episodes to evaluate each agent for.
    seed : int
        Which trained seed to load for the DQN agents. Also used as the
        base seed for environment resets during evaluation.
    hidden_size : int
        Number of neurons in the Q-Network hidden layers.
    action_set : str
        "task" or "full" — must match the value used during training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the environment to get its structural properties
    env = make_env(env_id, seed, action_set)
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))   # Flattened observation size (e.g., 192)
    num_actions = env.action_space.n     # Number of available actions
    names = action_names(env_id, action_set, num_actions)  # Human-readable action names
    width = env.unwrapped.width          # Grid width (e.g., 8)
    height = env.unwrapped.height        # Grid height (e.g., 8)

    # --- Find Trained Models for This Seed ---
    baseline_models = get_models_by_seed(results_dir, env_id, "dqn_baseline")
    shaped_models   = get_models_by_seed(results_dir, env_id, "dqn_reward_shaping")

    baseline_model_path = baseline_models.get(seed)
    shaped_model_path   = shaped_models.get(seed)

    # Build the list of agents to plot. Each entry is (display_name, q_net_or_None).
    # The Random Agent has no neural network, so q_net=None.
    # Random agent is always evaluated with seed 1 regardless of the DQN seed.
    agents = [("Random Agent", None)]

    # --- Load Baseline DQN Model ---
    if baseline_model_path:
        # Create a fresh Q-Network with the same architecture as during training
        q_net_base = QNetwork(obs_dim, num_actions, hidden_size).to(device)
        # Load the saved weights from disk into the network
        # map_location=device ensures weights are loaded onto the correct device (CPU/GPU)
        q_net_base.load_state_dict(torch.load(baseline_model_path, map_location=device))
        # Set to evaluation mode: disables dropout and batch normalization layers
        # (our current network doesn't have these, but it's good practice)
        q_net_base.eval()
        agents.append(("Baseline DQN", q_net_base))
    else:
        agents.append(("Baseline DQN (Not Found)", None))

    # --- Load Reward Shaped DQN Model ---
    if shaped_model_path:
        q_net_shape = QNetwork(obs_dim, num_actions, hidden_size).to(device)
        q_net_shape.load_state_dict(torch.load(shaped_model_path, map_location=device))
        q_net_shape.eval()
        agents.append(("Reward Shaped DQN", q_net_shape))
    else:
        agents.append(("Reward Shaped (Not Found)", None))

    # --- Create the Figure ---
    # 2 rows x 3 columns = 6 subplots
    # figsize=(18, 10) gives enough space for all heatmaps and text labels
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # suptitle adds a title above all subplots
    fig.suptitle(
        f"{env_id} - State/Action Frequencies ({episodes} episodes, DQN seed={seed})",
        fontsize=14
    )

    # --- Generate Plots for Each Agent ---
    for col, (title, q_net) in enumerate(agents):
        # Skip agents whose model files weren't found
        if "Not Found" in title:
            axes[0, col].set_title(title)
            axes[1, col].set_title(title)
            continue

        # Run the agent through the environment and collect visit/action statistics
        visit_counts, state_action_counts = get_agent_data(
            env, q_net, episodes, seed, num_actions, device
        )

        # --- Top Row: Visit Frequency Heatmap ---
        # imshow() displays a 2D numpy array as a colored image.
        # .T transposes the array so that x=columns and y=rows (matching the grid layout).
        # origin="upper" puts (0,0) at the top-left corner (standard for grids).
        # cmap="YlOrRd" uses a Yellow-Orange-Red color map (brighter = more visits).
        im = axes[0, col].imshow(visit_counts.T, origin="upper", cmap="YlOrRd")
        axes[0, col].set_title(f"{title}\nVisit Frequencies")
        # colorbar adds a legend showing what colors map to what visit counts
        # fraction and pad control the colorbar's size and spacing
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        # --- Bottom Row: All Action Frequencies ---
        # Light blue background heatmap (just for visual context, not the main data)
        axes[1, col].imshow(visit_counts.T, origin="upper", cmap="Blues", alpha=0.3)
        axes[1, col].set_title(f"{title}\nAction Counts")

        # Map full action names to 1-letter abbreviations to save space
        abbr_map = {"left": "L", "right": "R", "forward": "F", "pickup": "P", "drop": "D", "toggle": "T", "done": "DN"}

        # Overlay text on each cell showing the count for EVERY action
        for x in range(width):
            for y in range(height):
                lines = []
                for act_idx in range(num_actions):
                    act_name = names[act_idx]
                    abbr = abbr_map.get(act_name, act_name[:1].upper())
                    count = state_action_counts[x, y, act_idx]
                    lines.append(f"{abbr}: {count}")
                
                cell_text = "\n".join(lines)
                
                # Make text slightly faded for unvisited cells so it's less visually overwhelming
                alpha_val = 1.0 if visit_counts[x, y] > 0 else 0.2
                
                axes[1, col].text(
                    x, y, cell_text,
                    ha="center", va="center", fontsize=7, fontweight="bold", alpha=alpha_val
                )

        # --- Draw Grid Lines ---
        # These lines separate the individual tiles of the MiniGrid environment,
        # making it easy to see which cell each heatmap value belongs to.
        for ax in [axes[0, col], axes[1, col]]:
            # Minor ticks at half-integer positions create grid lines between cells
            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
            # Hide the minor tick marks themselves (we only want the grid lines)
            ax.tick_params(which="minor", bottom=False, left=False)
            # Major ticks at integer positions label the cell coordinates (0, 1, 2, ...)
            ax.set_xticks(np.arange(0, width, 1))
            ax.set_yticks(np.arange(0, height, 1))

    # --- Action Abbreviation Legend ---
    # Build a legend string showing only the abbreviations that are actually used
    # in this environment's action set so the legend stays minimal and relevant.
    legend_parts = []
    for act_name in names:
        abbr = abbr_map.get(act_name, act_name[:1].upper())
        legend_parts.append(f"{abbr} = {act_name}")
    legend_text = "   |   ".join(legend_parts)

    # Place the legend as a centered text below all the subplots.
    # transform=fig.transFigure means x/y are fractions of the full figure (0.0 to 1.0).
    fig.text(
        0.5, 0.01,
        f"Action key:  {legend_text}",
        ha="center", va="bottom",
        fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#aaaaaa", alpha=0.8),
    )

    # --- Save the Plot ---
    # rect=[left, bottom, right, top] — bottom=0.05 reserves space for the legend text
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path = Path("plots") / f"{env_id}_state_action_freq_seed{seed}.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)  # free memory before the next seed's figure
    print(f"Plot saved to {output_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --env-id: Which environment's models and results to visualize.
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")

    # --results-dir: Where the trained model files (q_net.pt) are stored.
    parser.add_argument("--results-dir", type=str, default="results")

    # --episodes: How many episodes to evaluate each agent for.
    #   More episodes = more accurate heatmaps but longer runtime.
    parser.add_argument("--episodes", type=int, default=10)

    # --action-set: Must match the action set used during training.
    #   "task" = environment-specific subset, "full" = all 7 MiniGrid actions.
    parser.add_argument("--action-set", choices=["task", "full"], default="task")

    args = parser.parse_args()

    # Discover which seeds are available for this environment
    baseline_models = get_models_by_seed(args.results_dir, args.env_id, "dqn_baseline")
    shaped_models   = get_models_by_seed(args.results_dir, args.env_id, "dqn_reward_shaping")

    # Union of all seeds found across both agents
    all_seeds = sorted(set(baseline_models.keys()) | set(shaped_models.keys()))

    if not all_seeds:
        print(f"No trained models found for {args.env_id} in {args.results_dir}")
        raise SystemExit(1)

    print(f"Found seeds: {all_seeds} for {args.env_id}")

    for seed in all_seeds:
        print(f"Generating heatmap for seed={seed} ...")
        plot_all_frequencies(
            env_id=args.env_id,
            results_dir=args.results_dir,
            episodes=args.episodes,
            seed=seed,
            action_set=args.action_set,
        )

    print(f"Done. {len(all_seeds)} heatmap(s) saved to plots/")
