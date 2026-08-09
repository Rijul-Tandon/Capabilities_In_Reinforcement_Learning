"""
plot_q_overestimation.py - Q-Value Overestimation Comparison
=============================================================
Visualises how much standard (vanilla) DQN overestimates Q-values compared
to Double DQN (DDQN) across multiple MiniGrid environments.

For every reachable (non-wall) cell in the grid, and for all 4 agent
directions (E=0, S=1, W=2, N=3), the script:
  1. Constructs the wrapped observation by setting agent_pos / agent_dir
     on the unwrapped environment and running it through the wrapper pipeline.
  2. Passes the observation through the trained Q-network and records
     max_a Q(s, a).
  3. Averages the max-Q across the 4 directions to get a single value per cell.

Two types of plots are produced:

  A) Per-environment heatmap grid (1×3):
       DQN Max Q  |  DDQN Max Q  |  Difference (DQN − DDQN)
     Wall cells are shown in dark gray. A diverging colour map highlights
     cells where DQN overestimates most.

  B) A grouped bar chart across all environments:
       Average max-Q for DQN vs DDQN, with one group per environment.

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

# argparse: Parses command-line arguments (--results-dir, --action-set, etc.)
import argparse

# warnings: Used to suppress expected runtime warnings (like all-NaN slices)
import warnings

# Path: Object-oriented filesystem paths for locating model files and
#   creating output directories without manual string concatenation.
from pathlib import Path

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

# matplotlib.pyplot (plt): The plotting library used for heatmaps and bar
#   charts.  plt.subplots() creates grids of axes, ax.imshow() renders 2-D
#   arrays as colour-mapped images, and ax.bar() draws grouped bar charts.
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# matplotlib.colors: Provides Normalize and TwoSlopeNorm for mapping data
#   values to the [0, 1] colour-map range.  TwoSlopeNorm centres a diverging
#   colour map at zero so that positive and negative differences are visually
#   symmetric even when the data range is asymmetric.
import matplotlib.colors as mcolors

# numpy (np): Numerical computing — used for grid arrays, masking wall cells,
#   computing means, and constructing bar-chart positions.
import numpy as np

# torch: PyTorch deep-learning framework.  Used to load saved Q-network
#   weights (torch.load), run inference (forward pass), and disable gradient
#   tracking during evaluation (torch.no_grad).
import torch

# ============================================================================
# LOCAL IMPORTS (from our own codebase)
# ============================================================================

# QNetwork: The MLP that maps a flattened observation to Q-values for each
#   action.  Constructor signature: QNetwork(obs_dim, num_actions, hidden_size).
# make_env: Creates a MiniGrid environment wrapped with FullyObsWrapper →
#   ImgObsWrapper → FlatImageAndDirectionWrapper → RecordEpisodeStatistics.
# action_names: Returns human-readable names for the active action subset.
from dqn_common import QNetwork, make_env, action_names


# ============================================================================
# CONSTANTS
# ============================================================================

# The four MiniGrid environments we analyse.  This list mirrors the set of
# environments used throughout the project's benchmarks.
ENV_IDS = [
    "MiniGrid-Empty-6x6-v0",
    "MiniGrid-DoorKey-6x6-v0",
    # "MiniGrid-FourRooms-v0",
    # "MiniGrid-MultiRoom-N2-S4-v0",
]

# MiniGrid direction codes → human-readable labels (used in comments only).
#   0 = East (right), 1 = South (down), 2 = West (left), 3 = North (up)
NUM_DIRECTIONS = 4


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_models_by_seed(results_dir, env_id, exp_name):
    """
    Finds all trained model files for a given experiment, grouped by seed.

    The run directories follow the naming convention:
        {env_id}__{exp_name}__{seed}__{timestamp}
    If multiple runs exist for the same seed (re-runs), the newest timestamp
    wins because sorted() processes them in chronological order and the last
    one overwrites earlier entries.

    Parameters
    ----------
    results_dir : str
        Path to the parent directory containing all run folders.
    env_id : str
        The gymnasium environment ID (e.g., "MiniGrid-Empty-6x6-v0").
    exp_name : str
        The experiment name (e.g., "dqn_vanilla" or "ddqn_baseline").

    Returns
    -------
    dict[int, Path]
        A dict mapping seed → path to q_net.pt for that seed.
    """
    models = {}  # seed → latest model path for that seed
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


def get_wrapped_obs(env):
    """
    Generates the raw observation and manually pushes it through the pipeline
    of ObservationWrappers. This avoids AttributeError when the top-level
    wrapper (like RecordEpisodeStatistics) does not expose an observation() method.
    """
    obs = env.unwrapped.gen_obs()
    
    # Collect all observation wrappers in the stack (outermost to innermost)
    wrapper = env
    obs_wrappers = []
    while hasattr(wrapper, 'env'):
        if hasattr(wrapper, 'observation'):
            obs_wrappers.append(wrapper)
        wrapper = wrapper.env
        
    # Apply them from innermost to outermost
    for w in reversed(obs_wrappers):
        obs = w.observation(obs)
        
    return obs


def compute_q_values_grid(env, q_net, seed, device):
    """
    Computes max_a Q(s, a) for every reachable (non-wall) cell and all 4
    agent directions, using a trained Q-network.

    Approach:
      1. Reset the environment with the given seed so the procedural generator
         produces a deterministic layout (walls, doors, keys, goal).
      2. Walk every (x, y) cell in the grid; skip walls.
      3. For each reachable cell and each direction d ∈ {0,1,2,3}:
         - Directly set env.unwrapped.agent_pos = (x, y)
         - Directly set env.unwrapped.agent_dir = d
         - Obtain the wrapped observation via the wrapper pipeline:
               obs = env.observation(env.unwrapped.gen_obs())
         - Forward-pass through q_net to get Q-values for all actions.
         - Record max_a Q(s, a).
      4. Average the 4 directional max-Q values for each cell.

    Parameters
    ----------
    env : gym.Env
        A fully-wrapped MiniGrid environment (from make_env).
    q_net : torch.nn.Module
        A trained QNetwork instance (already in eval mode).
    seed : int
        The seed used to reset the environment (determines the layout).
    device : torch.device
        CPU or CUDA device for tensor operations.

    Returns
    -------
    q_grid : np.ndarray, shape (width, height, num_actions)
        Average Q-value for each action at each cell. Wall cells contain np.nan.
    wall_mask : np.ndarray, shape (width, height), dtype bool
        True where a cell is a wall (used for gray overlay in heatmaps).
    annotations : dict[tuple(int, int), str]
        Mapping of (x, y) coordinates to character labels (e.g., 'S' for start,
        'G' for goal, 'K' for key, 'D' for door).
    """
    # Reset env to establish the deterministic grid layout for this seed.
    env.reset(seed=seed)

    width = env.unwrapped.width
    height = env.unwrapped.height
    num_actions = env.action_space.n

    # Pre-allocate: NaN for walls, will be filled for reachable cells.
    q_grid = np.full((width, height, num_actions), np.nan, dtype=np.float32)
    wall_mask = np.zeros((width, height), dtype=bool)
    annotations = {}

    # Identify the start position
    start_pos = tuple(env.unwrapped.agent_pos)
    annotations[start_pos] = "S"

    # Identify wall cells, goals, keys, and doors from the grid object.
    for x in range(width):
        for y in range(height):
            cell = env.unwrapped.grid.get(x, y)
            if cell is not None:
                if cell.type == "wall":
                    wall_mask[x, y] = True
                elif cell.type == "goal":
                    annotations[(x, y)] = "G"
                elif cell.type == "key":
                    annotations[(x, y)] = "K"
                elif cell.type == "door":
                    annotations[(x, y)] = "D"

    # Compute Q-values for every reachable cell, averaged over 4 directions.
    with torch.no_grad():
        for x in range(width):
            for y in range(height):
                if wall_mask[x, y]:
                    continue  # skip wall cells

                dir_qvals = []
                for d in range(NUM_DIRECTIONS):
                    # Directly set the agent's position and direction on the
                    # unwrapped (raw MiniGrid) environment.
                    env.unwrapped.agent_pos = np.array([x, y])
                    env.unwrapped.agent_dir = d

                    # Generate the raw MiniGrid observation dict, then push it
                    # through the full wrapper pipeline (FullyObs → ImgObs →
                    # FlatImageAndDirection) to get the flat float vector that
                    # the Q-network expects.
                    obs = get_wrapped_obs(env)

                    # Convert to tensor: (obs_dim,) → (1, obs_dim) with batch dim.
                    obs_t = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                    q_values = q_net(obs_t).squeeze(0) # shape (num_actions,)
                    dir_qvals.append(q_values.cpu().numpy())

                q_grid[x, y, :] = np.mean(dir_qvals, axis=0)

    return q_grid, wall_mask, annotations


# ============================================================================
# PER-ENVIRONMENT HEATMAP PLOT
# ============================================================================

def plot_heatmap_for_env(env_id, grid_a, grid_b, wall_mask, annotations, seed,
                        label_a="Agent A", label_b="Agent B", comparison_tag=""):
    """
    Creates a 1×3 heatmap figure for a single environment and seed:
        Panel 0: Agent A Max Q
        Panel 1: Agent B Max Q
        Panel 2: Difference (A − B)

    Wall cells are overlaid in dark gray.  The difference panel uses a
    diverging colour map centred at zero so that overestimation (positive
    values) and underestimation (negative values) are visually symmetric.
    The specific state-action Q-values are printed inside each cell.

    Parameters
    ----------
    env_id : str
        Environment name (used in title and filename).
    grid_a : np.ndarray, shape (W, H, num_actions)
        Average Q-values for all actions from agent A.  Walls are np.nan.
    grid_b : np.ndarray, shape (W, H, num_actions)
        Average Q-values for all actions from agent B.  Walls are np.nan.
    wall_mask : np.ndarray, shape (W, H), dtype bool
        True for wall cells.
    annotations : dict[tuple, str]
        Mapping of (x,y) to layout labels (S, G, K, D).
    seed : int
        Training seed (shown in title and filename).
    label_a : str
        Human-readable label for agent A (used in panel titles).
    label_b : str
        Human-readable label for agent B (used in panel titles).
    comparison_tag : str
        Short tag appended to the output filename to distinguish comparisons.
    """
    width, height, num_actions = grid_a.shape
    
    # The background color of the cell will be based on the maximum Q-value
    # Use warnings.catch_warnings to ignore the expected all-NaN slice warnings for wall cells
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grid_a_max = np.nanmax(grid_a, axis=2)
        grid_b_max = np.nanmax(grid_b, axis=2)
    
    diff_grid_max = grid_a_max - grid_b_max  # positive ⇒ A overestimates vs B
    
    # For the text inside the cell, we want the specific Q-values or difference
    diff_grid = grid_a - grid_b

    # Make the figure larger so the text fits comfortably
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(
        f"{env_id}  —  {label_a} vs {label_b}  (seed={seed})",
        fontsize=16, fontweight="bold",
    )

    # Shared min/max across both agent panels for comparable colour scales.
    vmin_q = np.nanmin([grid_a_max, grid_b_max])
    vmax_q = np.nanmax([grid_a_max, grid_b_max])

    panels = [
        (f"{label_a}  Max Q", grid_a_max, grid_a, "viridis", None),
        (f"{label_b}  Max Q", grid_b_max, grid_b, "viridis", None),
        (f"Difference ({label_a} − {label_b})", diff_grid_max, diff_grid, "RdBu_r", "diverging"),
    ]

    # Action labels for the text overlay
    action_labels = ["L", "R", "F", "P", "D", "T", "Dn"]

    for col, (title, grid_max, grid_full, cmap, style) in enumerate(panels):
        ax = axes[col]

        # Build a display array.  Walls get a sentinel value so we can paint
        # them gray after the main imshow call.
        display = grid_max.T.copy()  # transpose so x→columns, y→rows

        if style == "diverging":
            # Centre the diverging colour map at zero.
            abs_max = np.nanmax(np.abs(diff_grid))
            if abs_max == 0:
                abs_max = 1.0  # avoid degenerate norm
            norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            im = ax.imshow(display, origin="upper", cmap=cmap, norm=norm)
        else:
            im = ax.imshow(
                display, origin="upper", cmap=cmap, vmin=vmin_q, vmax=vmax_q
            )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Overlay dark gray squares on wall cells and add annotations/text to free cells
        for x in range(width):
            for y in range(height):
                if wall_mask[x, y]:
                    ax.add_patch(plt.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1,
                        facecolor="dimgray", edgecolor="none",
                    ))
                else:
                    # Draw static layout annotations (S, G, K, D)
                    if (x, y) in annotations:
                        # Place symbol in the top-right corner of the cell
                        ax.text(x + 0.45, y - 0.45, annotations[(x, y)],
                                ha='right', va='top', color='white', 
                                fontsize=10, fontweight='bold',
                                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
                    
                    # Draw action Q-values as text inside the cell
                    q_vals = grid_full[x, y, :]
                    text_lines = []
                    for a in range(len(q_vals)):
                        if a < len(action_labels):
                            # Use +/- sign explicitly for difference plot
                            val_str = f"{q_vals[a]:+.2f}" if style == "diverging" else f"{q_vals[a]:.2f}"
                            text_lines.append(f"{action_labels[a]}: {val_str}")
                    
                    text_str = "\n".join(text_lines)
                    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none")
                    ax.text(x, y, text_str, ha='center', va='center', 
                            color='black', fontsize=6, bbox=bbox_props)

        # Draw grid lines between cells.
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks(np.arange(0, width, 1))
        ax.set_yticks(np.arange(0, height, 1))
        ax.set_title(title, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_dir = Path("plots") / comparison_tag if comparison_tag else Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{env_id}_q_overestimation_seed{seed}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {output_path}")


# ============================================================================
# SUMMARY BAR CHART
# ============================================================================

def plot_bar_chart(summary, label_a="Agent A", label_b="Agent B", comparison_tag=""):
    """
    Creates a grouped bar chart comparing average max-Q for two agents
    across all environments.

    Parameters
    ----------
    summary : list[dict]
        Each entry has keys: "env_id", "a_mean", "b_mean",
        and optionally "a_std", "b_std" for error bars when multiple
        seeds are available.
    label_a : str
        Human-readable label for agent A.
    label_b : str
        Human-readable label for agent B.
    comparison_tag : str
        Short tag appended to the output filename to distinguish comparisons.
    """
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

    # Annotate each bar with its numeric value.
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
    """
    Entry point: parses arguments, iterates over environments & seeds,
    computes Q-value grids for both agents, and produces all plots.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Compare Q-value overestimation between two trained agents."
    )
    # --results-dir: Parent directory where run folders live.
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing training run folders.")
    # --action-set: Must match the action set used during training.
    parser.add_argument("--action-set", choices=["task", "full"], default="task",
                        help="Action subset used during training ('task' or 'full').")
    # --hidden-size: Must match the hidden layer width used during training.
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden layer size of the Q-Network (must match training).")
    # --compare: Two experiment names to compare (replaces hardcoded dqn_vanilla/ddqn_baseline).
    parser.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"),
                        default=["dqn_vanilla", "ddqn_baseline"],
                        help="Experiment names of the two agents to compare.")
    # --label-a / --label-b: Human-readable labels for plot titles and legend.
    parser.add_argument("--label-a", type=str, default=None,
                        help="Display label for agent A (defaults to exp name).")
    parser.add_argument("--label-b", type=str, default=None,
                        help="Display label for agent B (defaults to exp name).")

    args = parser.parse_args()

    exp_a, exp_b = args.compare
    label_a = args.label_a or exp_a
    label_b = args.label_b or exp_b
    # Create a filesystem-safe tag from the two experiment names
    comparison_tag = f"{exp_a}_vs_{exp_b}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Results directory: {args.results_dir}")
    print(f"Action set: {args.action_set}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Comparing: {label_a}  vs  {label_b}")
    print(f"  (exp names: {exp_a}  vs  {exp_b})")
    print()

    # Accumulate per-environment summary statistics for the bar chart.
    bar_chart_summary = []

    # --- Iterate Over Environments ---
    for env_id in ENV_IDS:
        print(f"{'=' * 60}")
        print(f"Environment: {env_id}")
        print(f"{'=' * 60}")

        # Discover trained models for both agents.
        models_a = get_models_by_seed(args.results_dir, env_id, exp_a)
        models_b = get_models_by_seed(args.results_dir, env_id, exp_b)

        # We need both agents for at least one common seed.
        common_seeds = sorted(set(models_a.keys()) & set(models_b.keys()))
        if not common_seeds:
            print(f"  ⚠  No matching seed pair found for {env_id} — skipping.\n")
            continue

        print(f"  Common seeds: {common_seeds}")

        # Create the environment once to get observation/action dimensions.
        env = make_env(env_id, common_seeds[0], args.action_set)
        obs_dim     = int(np.prod(env.observation_space.shape))
        num_actions = env.action_space.n

        # Track per-seed averages for the bar chart error bars.
        seed_a_avgs = []
        seed_b_avgs = []

        for seed in common_seeds:
            print(f"\n  Seed {seed}:")

            # --- Load Agent A Model ---
            q_net_a = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
            q_net_a.load_state_dict(
                torch.load(models_a[seed], map_location=device, weights_only=True)
            )
            q_net_a.eval()
            print(f"    Loaded {label_a}: {models_a[seed]}")

            # --- Load Agent B Model ---
            q_net_b = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
            q_net_b.load_state_dict(
                torch.load(models_b[seed], map_location=device, weights_only=True)
            )
            q_net_b.eval()
            print(f"    Loaded {label_b}: {models_b[seed]}")

            # --- Compute Q-Value Grids ---
            env_seed = make_env(env_id, seed, args.action_set)

            print(f"    Computing {label_a} Q-values …")
            grid_a, wall_mask, annotations = compute_q_values_grid(
                env_seed, q_net_a, seed, device
            )

            print(f"    Computing {label_b} Q-values …")
            grid_b, _, _ = compute_q_values_grid(
                env_seed, q_net_b, seed, device
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                grid_a_max = np.nanmax(grid_a, axis=2)
                grid_b_max = np.nanmax(grid_b, axis=2)

            seed_a_avgs.append(np.nanmean(grid_a_max))
            seed_b_avgs.append(np.nanmean(grid_b_max))

            # Generate the visualization heatmap.
            print("    Generating heatmap plot …")
            plot_heatmap_for_env(
                env_id, grid_a, grid_b, wall_mask, annotations, seed,
                label_a=label_a, label_b=label_b, comparison_tag=comparison_tag,
            )
            
            diff_avg = np.nanmean(grid_a_max) - np.nanmean(grid_b_max)
            print(f"    {label_a} avg max Q : {np.nanmean(grid_a_max):.4f}")
            print(f"    {label_b} avg max Q : {np.nanmean(grid_b_max):.4f}")
            print(f"    Difference          : {diff_avg:+.4f}")


        env.close()

        # --- Aggregate Across Seeds for the Bar Chart ---
        bar_chart_summary.append({
            "env_id":  env_id,
            "a_mean":  float(np.mean(seed_a_avgs)),
            "b_mean":  float(np.mean(seed_b_avgs)),
            "a_std":   float(np.std(seed_a_avgs))  if len(seed_a_avgs) > 1 else 0.0,
            "b_std":   float(np.std(seed_b_avgs))   if len(seed_b_avgs) > 1 else 0.0,
        })

    # --- Generate Summary Bar Chart ---
    plot_bar_chart(bar_chart_summary, label_a=label_a, label_b=label_b,
                   comparison_tag=comparison_tag)

    print("\nAll done.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
