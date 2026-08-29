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
    "MiniGrid-Empty-Random-6x6-v0",
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
                    annotations[(x, y)] = "D"

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
# PUBLICATION STYLE SETUP
# ============================================================================

BLUES_CMAP = "Blues"          # sequential blue colormap for all heatmaps
HEATMAP_WALL_COLOR = "#f0f0f0"
HEATMAP_GRID_COLOR = "#b0b0b0"
HEATMAP_DPI = 300

DIR_INFO = [
    (3, "North"),
    (0, "East"),
    (1, "South"),
    (2, "West"),
]

DIR_ARROWS = {0: "→", 1: "↓", 2: "←", 3: "↑"}
DIR_NAME_TO_IDX = {"East": 0, "South": 1, "West": 2, "North": 3}

def get_action_arrow_or_symbol(act_name, facing_dir):
    """
    Given an action name and agent's facing direction (0=E, 1=S, 2=W, 3=N),
    returns the resulting movement direction arrow or interaction symbol.
    """
    if act_name == "forward":
        return DIR_ARROWS[facing_dir]
    elif act_name == "left":
        return DIR_ARROWS[(facing_dir - 1) % 4]
    elif act_name == "right":
        return DIR_ARROWS[(facing_dir + 1) % 4]
    elif act_name == "pickup":
        return "P"
    elif act_name == "drop":
        return "Dp"
    elif act_name == "toggle":
        return "T"
    elif act_name == "done":
        return "Dn"
    return act_name[:1].upper()


def _setup_pub_style():
    """Apply publication-grade matplotlib rcParams."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.6
    plt.rcParams['ytick.major.width'] = 0.6


def _style_heatmap_ax(ax, width, height):
    """Apply consistent grid, ticks, and spine styling to a heatmap axis."""
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color=HEATMAP_GRID_COLOR, linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _build_output_dir(plots_dir, env_id, seed):
    env_clean = "DoorKey" if "DoorKey" in env_id else ("Empty" if "Empty" in env_id else env_id)
    output_dir = Path(plots_dir) / "overestimation" / env_clean / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# PER-DIRECTION STANDALONE HEATMAP PLOT (for paper figures)
# ============================================================================

def _render_single_direction_plot(env_id, grid_a_d, grid_b_d,
                                  wall_mask, annotations, d_name,
                                  label_a, label_b, vmin_q, vmax_q,
                                  output_path, labels):
    """
    Renders a standalone 1×2 heatmap figure for a SINGLE facing direction.
    Designed for direct inclusion in a research paper.
    """
    _setup_pub_style()
    width, height = wall_mask.shape

    # Dynamic scaling based on grid dimensions (e.g. 6x6 vs 8x8 vs 10x10)
    fig_w = max(10.0, width * 1.6)
    fig_h = max(5.0, height * 0.8)
    if width >= 10:
        font_sz = 2.6
        star_sz = ""
        line_spacing = 0.82
        annot_sz = 6
    elif width >= 8:
        font_sz = 3.6
        star_sz = "*"
        line_spacing = 0.90
        annot_sz = 7
    else:
        font_sz = 5.0
        star_sz = "*"
        line_spacing = 1.0
        annot_sz = 9

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=HEATMAP_DPI)
    fig.patch.set_facecolor("white")

    cmap = plt.get_cmap(BLUES_CMAP).copy()
    cmap.set_bad(color="white")

    q_range = (vmax_q - vmin_q) if (vmax_q - vmin_q) != 0 else 1.0
    abbr_map = {"left": "L", "right": "R", "forward": "F", "pickup": "P", "drop": "Dp", "toggle": "T", "done": "Dn"}

    for col, (grid_d, agent_label) in enumerate([
        (grid_a_d, label_a),
        (grid_b_d, label_b),
    ]):
        ax = axes[col]
        ax.set_facecolor("white")
        display_max = np.nanmax(grid_d, axis=2)
        display = display_max.T.copy()
        im = ax.imshow(display, origin="upper", cmap=cmap, vmin=vmin_q, vmax=vmax_q)

        ax.set_title(f"{agent_label}", fontsize=10, fontweight="bold", pad=6)

        for x in range(width):
            for y in range(height):
                val_max = display_max[x, y]
                norm_v = (val_max - vmin_q) / q_range if not np.isnan(val_max) else 0
                if wall_mask[x, y]:
                    ax.add_patch(plt.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1,
                        facecolor=HEATMAP_WALL_COLOR, edgecolor="none",
                    ))
                    continue
                lbl = annotations.get((x, y))
                if lbl:
                    ax.text(x + 0.32, y - 0.32, lbl,
                            ha='center', va='center', color='white',
                            fontsize=annot_sz, fontweight='bold',
                            path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')])
                if not np.isnan(val_max):
                    txt_c = 'white' if norm_v > 0.6 else 'black'
                    cell_q_vals = grid_d[x, y]
                    if not np.isnan(cell_q_vals).all():
                        max_q = float(np.nanmax(cell_q_vals))
                        facing_idx = DIR_NAME_TO_IDX.get(d_name, 0)
                        max_act_indices = [i for i, q in enumerate(cell_q_vals) if np.isclose(q, max_q, atol=1e-5)]
                        arrows = []
                        for act_idx in max_act_indices:
                            if act_idx < len(labels):
                                sym = get_action_arrow_or_symbol(labels[act_idx], facing_idx)
                                if sym not in arrows:
                                    arrows.append(sym)
                        arrow_str = " ".join(arrows)
                        cell_text = f"{arrow_str}\n{max_q:.2f}"
                        ax.text(x, y, cell_text,
                                ha='center', va='center', color=txt_c,
                                fontsize=font_sz * 1.3, fontweight='bold', linespacing=1.1)

        _style_heatmap_ax(ax, width, height)

    # Shared colourbar
    fig.subplots_adjust(right=0.88, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Max Q(s, a)", fontsize=9, fontweight="medium")

    env_short = env_id.replace("MiniGrid-", "")
    fig.suptitle(f"{env_short} — Facing {d_name}",
                 fontsize=11, fontweight="bold", y=1.02)

    fig.savefig(output_path, dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Per-direction plot saved → {output_path}")


# ============================================================================
# PER-ENVIRONMENT HEATMAP PLOT (combined + per-direction)
# ============================================================================

def plot_heatmap_for_env(env_id, grid_a, grid_b, wall_mask, annotations, seed, labels,
                        label_a="Agent A", label_b="Agent B", comparison_tag="", stage_name="", plots_dir="plots"):
    """
    Generates:
      1. Combined 4×2 overview figure (all directions, both agents)
      2. Individual per-direction 1×2 figures for paper inclusion
    """
    _setup_pub_style()
    width, height, num_dirs, num_actions = grid_a.shape

    # Dynamic scaling based on grid dimensions (e.g. 6x6 vs 8x8 vs 10x10)
    fig_w = max(11.0, width * 1.8)
    fig_h = max(18.0, height * 2.5)
    if width >= 10:
        font_sz = 2.4
        star_sz = ""
        line_spacing = 0.80
        annot_sz = 6
    elif width >= 8:
        font_sz = 3.2
        star_sz = "*"
        line_spacing = 0.88
        annot_sz = 7
    else:
        font_sz = 4.5
        star_sz = "*"
        line_spacing = 1.0
        annot_sz = 9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grid_a_max = np.nanmax(grid_a, axis=3)  # (W, H, 4)
        grid_b_max = np.nanmax(grid_b, axis=3)  # (W, H, 4)

    vmin_q = float(np.nanmin([grid_a_max, grid_b_max]))
    vmax_q = float(np.nanmax([grid_a_max, grid_b_max]))
    q_range = (vmax_q - vmin_q) if (vmax_q - vmin_q) != 0 else 1.0

    output_dir = _build_output_dir(plots_dir, env_id, seed)
    
    # Extract grid size (e.g., 6x6, 8x8, 10x10) and env name
    grid_size_str = f"{width}x{height}"
    env_base = "doorkey" if "doorkey" in env_id.lower() else ("empty" if "empty" in env_id.lower() else env_id.lower().replace("minigrid-", "").replace("-v0", ""))
    env_tag = f"{env_base}_{grid_size_str}"
    stage_suffix = f"_{stage_name.lower().replace(' ', '_')}" if stage_name else ""

    # ------------------------------------------------------------------
    # 1) Combined 4×2 overview figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(fig_w, fig_h), dpi=HEATMAP_DPI)
    fig.patch.set_facecolor("white")

    title_suffix = f" [{stage_name}]" if stage_name else ""
    env_short = env_id.replace("MiniGrid-", "")
    fig.suptitle(
        f"{env_short}{title_suffix}  —  Max Q-Value Comparison  (seed {seed})",
        fontsize=13, fontweight="bold", y=0.995,
    )

    cmap = plt.get_cmap(BLUES_CMAP).copy()
    cmap.set_bad(color="white")

    col_titles = [
        f"{label_a}",
        f"{label_b}",
    ]

    abbr_map = {"left": "L", "right": "R", "forward": "F", "pickup": "P", "drop": "Dp", "toggle": "T", "done": "Dn"}
    for row, (d_idx, d_name) in enumerate(DIR_INFO):
        for col, grid_all in enumerate([grid_a, grid_b]):
            ax = axes[row, col]
            ax.set_facecolor("white")
            grid_d = grid_all[:, :, d_idx, :]
            display_max = np.nanmax(grid_d, axis=2)
            display = display_max.T.copy()

            im = ax.imshow(display, origin="upper", cmap=cmap, vmin=vmin_q, vmax=vmax_q)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(f"Facing {d_name}", fontsize=10.5, fontweight="bold")

            for x in range(width):
                for y in range(height):
                    val_max = display_max[x, y]
                    norm_v = (val_max - vmin_q) / q_range if not np.isnan(val_max) else 0
                    if wall_mask[x, y]:
                        ax.add_patch(plt.Rectangle(
                            (x - 0.5, y - 0.5), 1, 1,
                            facecolor=HEATMAP_WALL_COLOR, edgecolor="none",
                        ))
                        continue
                    lbl = annotations.get((x, y))
                    if lbl:
                        ax.text(x + 0.32, y - 0.32, lbl,
                                ha='center', va='center', color='white',
                                fontsize=annot_sz, fontweight='bold',
                                path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')])
                    if not np.isnan(val_max):
                        txt_c = 'white' if norm_v > 0.6 else 'black'
                        cell_q_vals = grid_d[x, y]
                        if not np.isnan(cell_q_vals).all():
                            max_q = float(np.nanmax(cell_q_vals))
                            max_act_indices = [i for i, q in enumerate(cell_q_vals) if np.isclose(q, max_q, atol=1e-5)]
                            arrows = []
                            for act_idx in max_act_indices:
                                if act_idx < len(labels):
                                    sym = get_action_arrow_or_symbol(labels[act_idx], d_idx)
                                    if sym not in arrows:
                                        arrows.append(sym)
                            arrow_str = " ".join(arrows)
                            cell_text = f"{arrow_str}\n{max_q:.2f}"
                            ax.text(x, y, cell_text,
                                    ha='center', va='center', color=txt_c,
                                    fontsize=font_sz * 1.3, fontweight='bold', linespacing=1.1)

            _style_heatmap_ax(ax, width, height)

    # Shared colourbar for combined figure
    fig.subplots_adjust(right=0.88, hspace=0.20, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.08, 0.02, 0.85])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Max Q(s, a)", fontsize=9.5, fontweight="medium")

    # Footer annotation key
    fig.text(
        0.44, 0.005,
        "S = Start   G = Goal   K = Key   D = Door",
        ha="center", va="bottom", fontsize=9, fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cccccc", alpha=0.95),
    )

    combined_path = output_dir / f"q_overestimation_{env_tag}_combined{stage_suffix}.png"
    fig.savefig(combined_path, dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined heatmap saved → {combined_path}")

    # ------------------------------------------------------------------
    # 2) Individual per-direction standalone figures
    # ------------------------------------------------------------------
    for d_idx, d_name in DIR_INFO:
        per_dir_path = output_dir / f"q_overestimation_{env_tag}_{d_name.lower()}{stage_suffix}.png"
        _render_single_direction_plot(
            env_id,
            grid_a[:, :, d_idx, :],
            grid_b[:, :, d_idx, :],
            wall_mask, annotations, d_name,
            label_a, label_b,
            vmin_q, vmax_q,
            per_dir_path,
            labels
        )


# ============================================================================
# CONSISTENT AGENT COLOURS (matches plot_comparison.py)
# ============================================================================

AGENT_BAR_COLORS = {
    "baseline":        "#1f77b4",  # Blue  (same as plot_comparison AGENT_COLORS)
    "reward_shaping":  "#ff7f0e",  # Orange
    "default_a":       "#1f77b4",
    "default_b":       "#ff7f0e",
}


# ============================================================================
# SUMMARY BAR CHART
# ============================================================================

def plot_bar_chart(summary, label_a="Agent A", label_b="Agent B", comparison_tag="", plots_dir="plots"):
    """Publication-grade bar chart with colours matching the training-curve pipeline."""
    if not summary:
        print("No data for summary bar chart — skipping.")
        return

    _setup_pub_style()

    env_labels = [s["env_id"].replace("MiniGrid-", "") for s in summary]
    a_means = [s["a_mean"] for s in summary]
    b_means = [s["b_mean"] for s in summary]
    a_stds  = [s.get("a_std", 0) for s in summary]
    b_stds  = [s.get("b_std", 0) for s in summary]

    # Determine colours from label hints
    color_a = AGENT_BAR_COLORS["default_a"]
    color_b = AGENT_BAR_COLORS["default_b"]
    if "reward" in label_a.lower() or "rs" in label_a.lower():
        color_a = AGENT_BAR_COLORS["reward_shaping"]
    if "reward" in label_b.lower() or "rs" in label_b.lower():
        color_b = AGENT_BAR_COLORS["reward_shaping"]
    if "baseline" in label_a.lower():
        color_a = AGENT_BAR_COLORS["baseline"]
    if "baseline" in label_b.lower():
        color_b = AGENT_BAR_COLORS["baseline"]

    x = np.arange(len(env_labels))
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    fig.patch.set_facecolor("white")

    bars_a = ax.bar(
        x - bar_width / 2, a_means, bar_width,
        yerr=a_stds, capsize=4, error_kw=dict(lw=1.0, capthick=1.0),
        label=label_a, color=color_a, edgecolor="white", linewidth=0.6,
    )
    bars_b = ax.bar(
        x + bar_width / 2, b_means, bar_width,
        yerr=b_stds, capsize=4, error_kw=dict(lw=1.0, capthick=1.0),
        label=label_b, color=color_b, edgecolor="white", linewidth=0.6,
    )

    ax.set_ylabel("Mean Maximum Q-Value", fontsize=10, fontweight="medium")
    ax.set_title("Q-Value Overestimation Comparison", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, fontsize=9.5, fontweight="medium")
    ax.tick_params(axis='y', labelsize=9)

    # Value annotations on bars
    for bars in [bars_a, bars_b]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="medium",
                )

    # Grid & spine cleanup
    ax.grid(True, axis='y', linestyle="--", alpha=0.35, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="best", frameon=True, facecolor="white",
              edgecolor="#e0e0e0", framealpha=0.95)

    fig.tight_layout()
    output_dir = Path(plots_dir) / "overestimation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "q_overestimation_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary bar chart saved → {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare Q-value overestimation between two trained agents."
    )
    parser.add_argument("--env-id", type=str, default=None, help="Specific environment ID to plot (default: run all in ENV_IDS)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots")
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
    target_envs = [args.env_id] if args.env_id else ENV_IDS

    for env_id in target_envs:
        print(f"{'=' * 60}")
        print(f"Environment: {env_id}")
        print(f"{'=' * 60}")

        models_a = get_models_by_seed(args.results_dir, env_id, exp_a)
        models_b = get_models_by_seed(args.results_dir, env_id, exp_b)

        common_seeds = sorted(set(models_a.keys()) & set(models_b.keys()))
        current_exp_a, current_exp_b = exp_a, exp_b
        current_label_a, current_label_b = label_a, label_b

        if not common_seeds:
            possible_pairs = [
                ("ddqn_baseline", "ddqn_reward_shaping"),
                ("dqn_baseline", "dqn_reward_shaping"),
                ("dqn_vanilla", "dqn_baseline"),
            ]
            for pa, pb in possible_pairs:
                ma = get_models_by_seed(args.results_dir, env_id, pa)
                mb = get_models_by_seed(args.results_dir, env_id, pb)
                cs = sorted(set(ma.keys()) & set(mb.keys()))
                if cs:
                    models_a, models_b = ma, mb
                    common_seeds = cs
                    current_exp_a, current_exp_b = pa, pb
                    current_label_a = pa.replace("ddqn_reward_shaping", "RS-DDQN").replace("dqn_reward_shaping", "RS-DDQN").replace("ddqn_baseline", "Baseline DDQN").replace("_", " ").title()
                    current_label_b = pb.replace("ddqn_reward_shaping", "RS-DDQN").replace("dqn_reward_shaping", "RS-DDQN").replace("ddqn_baseline", "Baseline DDQN").replace("_", " ").title()
                    print(f"  Auto-selected available experiment pair: {pa} vs {pb}")
                    break

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

                seed_label_a = f"{current_label_a}{get_decay_str(models_a[seed])}"
                seed_label_b = f"{current_label_b}{get_decay_str(models_b[seed])}"

                labels = action_names(env_id, args.action_set, num_actions)
                print(f"    Generating 4×3 directional heatmap plot ({stage_name}) …")
                plot_heatmap_for_env(
                    env_id, grid_a, grid_b, wall_mask, annotations, seed, labels,
                    label_a=seed_label_a, label_b=seed_label_b, comparison_tag=comparison_tag,
                    stage_name=stage_name, plots_dir=args.plots_dir,
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
                   comparison_tag=comparison_tag, plots_dir=args.plots_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
