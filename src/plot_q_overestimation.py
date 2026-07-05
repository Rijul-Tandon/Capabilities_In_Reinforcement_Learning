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
    "MiniGrid-FourRooms-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
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
    q_grid : np.ndarray, shape (width, height)
        Average max-Q value at each cell.  Wall cells contain np.nan.
    wall_mask : np.ndarray, shape (width, height), dtype bool
        True where a cell is a wall (used for gray overlay in heatmaps).
    """
    # Reset env to establish the deterministic grid layout for this seed.
    env.reset(seed=seed)

    width = env.unwrapped.width
    height = env.unwrapped.height

    # Pre-allocate: NaN for walls, will be filled for reachable cells.
    q_grid = np.full((width, height), np.nan, dtype=np.float32)
    wall_mask = np.zeros((width, height), dtype=bool)

    # Identify wall cells from the grid object.
    for x in range(width):
        for y in range(height):
            cell = env.unwrapped.grid.get(x, y)
            if cell is not None and cell.type == "wall":
                wall_mask[x, y] = True

    # Compute max-Q for every reachable cell, averaged over 4 directions.
    with torch.no_grad():
        for x in range(width):
            for y in range(height):
                if wall_mask[x, y]:
                    continue  # skip wall cells

                dir_maxq = []
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

                    q_values = q_net(obs_t)            # shape (1, num_actions)
                    max_q = q_values.max(dim=1).values.item()
                    dir_maxq.append(max_q)

                # Average across the 4 directions.
                q_grid[x, y] = np.mean(dir_maxq)

    return q_grid, wall_mask


# ============================================================================
# PER-ENVIRONMENT HEATMAP PLOT
# ============================================================================

def plot_heatmap_for_env(env_id, dqn_grid, ddqn_grid, wall_mask, seed):
    """
    Creates a 1×3 heatmap figure for a single environment and seed:
        Panel 0: DQN Max Q
        Panel 1: DDQN Max Q
        Panel 2: Difference (DQN − DDQN)

    Wall cells are overlaid in dark gray.  The difference panel uses a
    diverging colour map centred at zero so that overestimation (positive
    values) and underestimation (negative values) are visually symmetric.

    Parameters
    ----------
    env_id : str
        Environment name (used in title and filename).
    dqn_grid : np.ndarray, shape (W, H)
        Average max-Q from the vanilla DQN.  Walls are np.nan.
    ddqn_grid : np.ndarray, shape (W, H)
        Average max-Q from the Double DQN.  Walls are np.nan.
    wall_mask : np.ndarray, shape (W, H), dtype bool
        True for wall cells.
    seed : int
        Training seed (shown in title and filename).
    """
    width, height = dqn_grid.shape
    diff_grid = dqn_grid - ddqn_grid  # positive ⇒ DQN overestimates

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"{env_id}  —  Q-Value Overestimation  (seed={seed})",
        fontsize=14, fontweight="bold",
    )

    # Shared min/max across DQN and DDQN panels for comparable colour scales.
    vmin_q = np.nanmin([dqn_grid, ddqn_grid])
    vmax_q = np.nanmax([dqn_grid, ddqn_grid])

    panels = [
        ("DQN (Vanilla)  Max Q", dqn_grid, "viridis", None),
        ("DDQN  Max Q",          ddqn_grid, "viridis", None),
        ("Difference (DQN − DDQN)", diff_grid, "RdBu_r", "diverging"),
    ]

    for col, (title, grid, cmap, style) in enumerate(panels):
        ax = axes[col]

        # Build a display array.  Walls get a sentinel value so we can paint
        # them gray after the main imshow call.
        display = grid.T.copy()  # transpose so x→columns, y→rows

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

        # Overlay dark gray squares on wall cells.
        for x in range(width):
            for y in range(height):
                if wall_mask[x, y]:
                    ax.add_patch(plt.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1,
                        facecolor="dimgray", edgecolor="none",
                    ))

        # Draw grid lines between cells.
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks(np.arange(0, width, 1))
        ax.set_yticks(np.arange(0, height, 1))
        ax.set_title(title, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = Path("plots") / f"{env_id}_q_overestimation_seed{seed}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {output_path}")


# ============================================================================
# SUMMARY BAR CHART
# ============================================================================

def plot_bar_chart(summary):
    """
    Creates a grouped bar chart comparing average max-Q for DQN vs DDQN
    across all environments.

    Parameters
    ----------
    summary : list[dict]
        Each entry has keys: "env_id", "dqn_mean", "ddqn_mean",
        and optionally "dqn_std", "ddqn_std" for error bars when multiple
        seeds are available.
    """
    if not summary:
        print("No data for summary bar chart — skipping.")
        return

    env_labels = [s["env_id"].replace("MiniGrid-", "") for s in summary]
    dqn_means  = [s["dqn_mean"]  for s in summary]
    ddqn_means = [s["ddqn_mean"] for s in summary]
    dqn_stds   = [s.get("dqn_std", 0)  for s in summary]
    ddqn_stds  = [s.get("ddqn_std", 0) for s in summary]

    x = np.arange(len(env_labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_dqn = ax.bar(
        x - bar_width / 2, dqn_means, bar_width,
        yerr=dqn_stds, capsize=4,
        label="DQN (Vanilla)", color="#E74C3C", edgecolor="black", linewidth=0.6,
    )
    bars_ddqn = ax.bar(
        x + bar_width / 2, ddqn_means, bar_width,
        yerr=ddqn_stds, capsize=4,
        label="Double DQN", color="#3498DB", edgecolor="black", linewidth=0.6,
    )

    ax.set_xlabel("Environment", fontsize=12)
    ax.set_ylabel("Average Max Q-Value", fontsize=12)
    ax.set_title(
        "Q-Value Overestimation: DQN vs Double DQN",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotate each bar with its numeric value.
    for bars in [bars_dqn, bars_ddqn]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    output_path = Path("plots") / "q_overestimation_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nBar chart saved → {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Entry point: parses arguments, iterates over environments & seeds,
    computes Q-value grids for both DQN and DDQN, and produces all plots.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Compare Q-value overestimation between DQN and Double DQN."
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

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Results directory: {args.results_dir}")
    print(f"Action set: {args.action_set}")
    print(f"Hidden size: {args.hidden_size}")
    print()

    # Accumulate per-environment summary statistics for the bar chart.
    # Each entry: {"env_id", "dqn_mean", "ddqn_mean", "dqn_std", "ddqn_std"}
    bar_chart_summary = []

    # --- Iterate Over Environments ---
    for env_id in ENV_IDS:
        print(f"{'=' * 60}")
        print(f"Environment: {env_id}")
        print(f"{'=' * 60}")

        # Discover trained models for both DQN variants.
        dqn_models  = get_models_by_seed(args.results_dir, env_id, "dqn_vanilla")
        ddqn_models = get_models_by_seed(args.results_dir, env_id, "ddqn_baseline")

        # We need both a DQN and a DDQN model for at least one common seed.
        common_seeds = sorted(set(dqn_models.keys()) & set(ddqn_models.keys()))
        if not common_seeds:
            print(f"  ⚠  No matching seed pair found for {env_id} — skipping.\n")
            continue

        print(f"  Common seeds: {common_seeds}")

        # Create the environment once to get observation/action dimensions.
        # We use the first common seed; the layout will be re-generated per seed.
        env = make_env(env_id, common_seeds[0], args.action_set)
        obs_dim     = int(np.prod(env.observation_space.shape))
        num_actions = env.action_space.n

        # Track per-seed averages for the bar chart error bars.
        seed_dqn_avgs  = []
        seed_ddqn_avgs = []

        for seed in common_seeds:
            print(f"\n  Seed {seed}:")

            # --- Load DQN (Vanilla) Model ---
            q_net_dqn = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
            q_net_dqn.load_state_dict(
                torch.load(dqn_models[seed], map_location=device, weights_only=True)
            )
            q_net_dqn.eval()
            print(f"    Loaded DQN  : {dqn_models[seed]}")

            # --- Load Double DQN Model ---
            q_net_ddqn = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
            q_net_ddqn.load_state_dict(
                torch.load(ddqn_models[seed], map_location=device, weights_only=True)
            )
            q_net_ddqn.eval()
            print(f"    Loaded DDQN : {ddqn_models[seed]}")

            # --- Compute Q-Value Grids ---
            # Re-create env for this specific seed so the layout matches training.
            env_seed = make_env(env_id, seed, args.action_set)

            print("    Computing DQN  Q-values …")
            dqn_grid, wall_mask = compute_q_values_grid(
                env_seed, q_net_dqn, seed, device
            )

            print("    Computing DDQN Q-values …")
            ddqn_grid, _ = compute_q_values_grid(
                env_seed, q_net_ddqn, seed, device
            )

            env_seed.close()

            # --- Per-Seed Statistics ---
            # nanmean ignores wall cells (which are NaN).
            dqn_avg  = float(np.nanmean(dqn_grid))
            ddqn_avg = float(np.nanmean(ddqn_grid))
            diff_avg = dqn_avg - ddqn_avg

            print(f"    DQN  avg max Q : {dqn_avg:.4f}")
            print(f"    DDQN avg max Q : {ddqn_avg:.4f}")
            print(f"    Difference     : {diff_avg:+.4f}")

            seed_dqn_avgs.append(dqn_avg)
            seed_ddqn_avgs.append(ddqn_avg)

            # --- Generate Per-Environment Heatmap ---
            plot_heatmap_for_env(env_id, dqn_grid, ddqn_grid, wall_mask, seed)

        env.close()

        # --- Aggregate Across Seeds for the Bar Chart ---
        bar_chart_summary.append({
            "env_id":    env_id,
            "dqn_mean":  float(np.mean(seed_dqn_avgs)),
            "ddqn_mean": float(np.mean(seed_ddqn_avgs)),
            "dqn_std":   float(np.std(seed_dqn_avgs))  if len(seed_dqn_avgs) > 1 else 0.0,
            "ddqn_std":  float(np.std(seed_ddqn_avgs)) if len(seed_ddqn_avgs) > 1 else 0.0,
        })

    # --- Generate Summary Bar Chart ---
    plot_bar_chart(bar_chart_summary)

    print("\nAll done.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
