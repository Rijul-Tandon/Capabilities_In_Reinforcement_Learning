"""Compares the Q-values of two trained agents: for every free cell and facing direction it draws max_a Q(s, a) of
both agents as heat maps (one figure per task stage) and a bar chart of the mean maximum Q-value."""

import argparse
import json
import warnings
from pathlib import Path

import gymnasium as gym
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn_common import QNetwork, action_names, make_env

DEFAULT_ENV_IDS = ["MiniGrid-Empty-Random-6x6-v0", "MiniGrid-DoorKey-6x6-v0"]      # used when --env-id is not given
# Tried in this order when the requested --compare pair has no seed with a trained model for both agents
FALLBACK_PAIRS = [("ddqn_baseline", "ddqn_reward_shaping"), ("dqn_baseline", "dqn_reward_shaping"), ("dqn_vanilla", "dqn_baseline")]

DIRECTION_ROWS = [(3, "North"), (0, "East"), (1, "South"), (2, "West")]             # (MiniGrid direction, name), in drawing order
DIRECTION_ARROWS = {0: "→", 1: "↓", 2: "←", 3: "↑"}
ACTION_ABBREVIATIONS = {"left": "L", "right": "R", "forward": "F", "pickup": "P", "drop": "Dp", "toggle": "T", "done": "Dn"}

FIGURE_DPI = 300
WALL_COLOR = "#f0f0f0"
GRID_COLOR = "#b0b0b0"
BLUE, ORANGE = "#1f77b4", "#ff7f0e"         # the agent colours of plot_comparison.py
HEATMAP_CMAP = plt.get_cmap("Blues").copy()
HEATMAP_CMAP.set_bad(color="white")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6


def find_models_by_seed(results_dir, env_id, exp_name):
    """{seed: path of q_net.pt} of one experiment; the seed is read from the run-folder name
    <env_id>__<exp_name>__<seed>__<timestamp>.  When a seed was run more than once, the newest run is used."""
    models = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}__*")):
        if not (run_dir / "q_net.pt").exists():
            continue
        try:
            seed = int(run_dir.name.split("__")[2])
        except (IndexError, ValueError):
            continue
        models[seed] = run_dir / "q_net.pt"
    return models


def epsilon_schedule_suffix(model_path):
    """' (<epsilon schedule>)' of the run that produced this model, for the panel titles; '' when it is not recorded."""
    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        return ""
    with open(config_path, "r", encoding="utf-8") as config_file:
        epsilon_schedule = json.load(config_file).get("epsilon_schedule", "")
    return f" ({epsilon_schedule})" if epsilon_schedule else ""


def wrapped_observation(env):
    """The observation of the environment's CURRENT state, passed through the observation wrappers.

    Only genuine ObservationWrapper instances are applied.  Gymnasium <= 0.29 forwards unknown attribute lookups
    from every wrapper to the wrapped env, so `hasattr(wrapper, 'observation')` is True for RecordEpisodeStatistics,
    TimeLimit, etc. as well, and the flattening wrapper would run more than once.
    """
    observation_wrappers = []       # outermost first
    wrapper = env
    while isinstance(wrapper, gym.Wrapper):
        if isinstance(wrapper, gym.ObservationWrapper):
            observation_wrappers.append(wrapper)
        wrapper = wrapper.env
    observation = env.unwrapped.gen_obs()
    for observation_wrapper in reversed(observation_wrappers):
        observation = observation_wrapper.observation(observation)
    return observation


def compute_q_values(env, q_net, seed, device, stage):
    """Q(s, a) for every free cell and facing direction, in one task stage of the layout of this seed.

    stage 1: initial (key on the ground, door closed);  2: the agent carries the key;  3: the door is open.

    Returns
      q_values       (width, height, 4 directions, num_actions), NaN on wall cells
      wall_mask      (width, height), True on wall cells
      annotations    {(x, y): "S" | "G" | "K" | "D"} for start, goal, key and door
    """
    env.reset(seed=seed)
    grid = env.unwrapped.grid
    width, height = env.unwrapped.width, env.unwrapped.height

    # ---- put the map into the requested stage ----
    key_position, key, door = None, None, None
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "key":
                key_position, key = (x, y), cell
            elif cell is not None and cell.type == "door":
                door = cell
    if stage >= 2 and key is not None:
        grid.set(key_position[0], key_position[1], None)
        env.unwrapped.carrying = key if stage == 2 else None        # stage 3: the key has been used
    if stage == 3 and door is not None:
        door.is_open = True
        door.is_locked = False

    # ---- place the agent on every free cell, facing every direction ----
    q_values = np.full((width, height, 4, env.action_space.n), np.nan, dtype=np.float32)
    wall_mask = np.zeros((width, height), dtype=bool)
    annotations = {tuple(env.unwrapped.agent_pos): "S"}
    with torch.no_grad():
        for x in range(width):
            for y in range(height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "wall":
                    wall_mask[x, y] = True
                    continue
                if cell is not None and cell.type == "goal":
                    annotations[(x, y)] = "G"
                elif cell is not None and cell.type == "key":
                    annotations[(x, y)] = "K"
                elif cell is not None and cell.type == "door":
                    annotations[(x, y)] = "D"

                for direction in range(4):
                    env.unwrapped.agent_pos = np.array([x, y])
                    env.unwrapped.agent_dir = direction
                    observation = torch.tensor(wrapped_observation(env), dtype=torch.float32, device=device).unsqueeze(0)
                    q_values[x, y, direction, :] = q_net(observation).squeeze(0).cpu().numpy()
    return q_values, wall_mask, annotations


def action_symbol(action_name, facing_direction):
    """Text drawn for an action: its abbreviation, preceded for forward / left / right by an arrow that points
    where the agent moves or faces afterwards."""
    abbreviation = ACTION_ABBREVIATIONS[action_name]
    if action_name == "forward":
        return f"{DIRECTION_ARROWS[facing_direction]} {abbreviation}"
    if action_name == "left":
        return f"{DIRECTION_ARROWS[(facing_direction - 1) % 4]} {abbreviation}"
    if action_name == "right":
        return f"{DIRECTION_ARROWS[(facing_direction + 1) % 4]} {abbreviation}"
    return abbreviation


def draw_max_q_panel(axis, q_values, wall_mask, annotations, facing_direction, action_labels, vmin_q, vmax_q, cell_font_size):
    """One heatmap of max_a Q(s, a) for one agent and one facing direction.  Returns the image, for the colour bar.

    q_values: (width, height, num_actions) for this facing direction.  Every free cell shows the action(s) with
    the highest Q-value and that value; walls are grey; S / G / K / D mark start, goal, key and door.
    """
    width, height = wall_mask.shape
    if width >= 10:
        annotation_font_size = 6
    elif width >= 8:
        annotation_font_size = 7
    else:
        annotation_font_size = 9
    q_range = (vmax_q - vmin_q) if (vmax_q - vmin_q) != 0 else 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)        # wall cells are all-NaN
        max_q_values = np.nanmax(q_values, axis=2)                      # (width, height)
    axis.set_facecolor("white")
    image = axis.imshow(max_q_values.T, origin="upper", cmap=HEATMAP_CMAP, vmin=vmin_q, vmax=vmax_q)

    for x in range(width):
        for y in range(height):
            if wall_mask[x, y]:
                axis.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=WALL_COLOR, edgecolor="none"))
                continue
            if (x, y) in annotations:
                axis.text(x + 0.32, y - 0.32, annotations[(x, y)], ha="center", va="center", color="white",
                          fontsize=annotation_font_size, fontweight="bold",
                          path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")])
            if np.isnan(max_q_values[x, y]):
                continue
            max_q = float(max_q_values[x, y])
            best_action_symbols = []        # every action whose Q-value ties with the maximum
            for action_index, q_value in enumerate(q_values[x, y]):
                if np.isclose(q_value, max_q, atol=1e-5):
                    symbol = action_symbol(action_labels[action_index], facing_direction)
                    if symbol not in best_action_symbols:
                        best_action_symbols.append(symbol)
            # white text on the dark cells of the upper part of the colour scale
            text_color = "white" if (max_q_values[x, y] - vmin_q) / q_range > 0.6 else "black"
            axis.text(x, y, " ".join(best_action_symbols) + f"\n{max_q:.2f}", ha="center", va="center", color=text_color,
                      fontsize=cell_font_size, fontweight="bold", linespacing=1.1)

    axis.set_xticks(np.arange(-0.5, width, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, height, 1), minor=True)
    axis.grid(which="minor", color=GRID_COLOR, linestyle="-", linewidth=0.5)
    axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def plot_stage_heatmaps(env_id, seed, stage_name, agents, wall_mask, annotations, action_labels, plots_dir):
    """Writes the combined 4 x 2 figure and the four per-direction figures of one seed and task stage.

    agents: [(label, q_values), (label, q_values)], q_values of shape (width, height, 4 directions, num_actions).
    """
    width, height = wall_mask.shape
    if width >= 10:
        combined_font_size, direction_font_size = 2.4 * 1.3, 2.6 * 1.3
    elif width >= 8:
        combined_font_size, direction_font_size = 3.2 * 1.3, 3.6 * 1.3
    else:
        combined_font_size, direction_font_size = 4.5 * 1.3, 5.0 * 1.3

    # one colour scale for both agents and all facing directions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)        # wall cells are all-NaN
        max_q_a = np.nanmax(agents[0][1], axis=3)                       # (width, height, 4)
        max_q_b = np.nanmax(agents[1][1], axis=3)
    vmin_q = float(np.nanmin([max_q_a, max_q_b]))
    vmax_q = float(np.nanmax([max_q_a, max_q_b]))

    if "DoorKey" in env_id:
        env_folder, env_tag = "DoorKey", "doorkey"
    elif "Empty" in env_id:
        env_folder, env_tag = "Empty", "empty"
    else:
        env_folder, env_tag = env_id, env_id.lower().replace("minigrid-", "").replace("-v0", "")
    output_dir = Path(plots_dir) / "overestimation" / env_folder / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"q_overestimation_{env_tag}_{width}x{height}"
    stage_suffix = f"_{stage_name}" if stage_name else ""
    env_short = env_id.replace("MiniGrid-", "")

    # ---- combined figure: one row per facing direction, one column per agent ----
    figure, axes = plt.subplots(4, 2, figsize=(max(11.0, width * 1.8), max(18.0, height * 2.5)), dpi=FIGURE_DPI)
    figure.patch.set_facecolor("white")
    stage_title = f" [{stage_name}]" if stage_name else ""
    figure.suptitle(f"{env_short}{stage_title}  —  Max Q-Value Comparison  (seed {seed})", fontsize=13, fontweight="bold", y=0.995)
    for row, (direction, direction_name) in enumerate(DIRECTION_ROWS):
        for column, (agent_label, q_values) in enumerate(agents):
            axis = axes[row, column]
            image = draw_max_q_panel(axis, q_values[:, :, direction, :], wall_mask, annotations, direction, action_labels,
                                     vmin_q, vmax_q, combined_font_size)
            if row == 0:
                axis.set_title(agent_label, fontsize=11, fontweight="bold", pad=6)
            if column == 0:
                axis.set_ylabel(f"Facing {direction_name}", fontsize=10.5, fontweight="bold")
    figure.subplots_adjust(right=0.88, hspace=0.20, wspace=0.08)
    color_bar = figure.colorbar(image, cax=figure.add_axes([0.90, 0.08, 0.02, 0.85]))
    color_bar.ax.tick_params(labelsize=8)
    color_bar.set_label("Max Q(s, a)", fontsize=9.5, fontweight="medium")
    figure.text(0.44, 0.005, "S = Start   G = Goal   K = Key   D = Door", ha="center", va="bottom", fontsize=9, fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.95))
    combined_path = output_dir / f"{file_prefix}_combined{stage_suffix}.png"
    figure.savefig(combined_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"  Combined heatmap saved → {combined_path}")

    # ---- one standalone figure per facing direction, for the paper ----
    for direction, direction_name in DIRECTION_ROWS:
        figure, axes = plt.subplots(1, 2, figsize=(max(10.0, width * 1.6), max(5.0, height * 0.8)), dpi=FIGURE_DPI)
        figure.patch.set_facecolor("white")
        for column, (agent_label, q_values) in enumerate(agents):
            image = draw_max_q_panel(axes[column], q_values[:, :, direction, :], wall_mask, annotations, direction, action_labels,
                                     vmin_q, vmax_q, direction_font_size)
            axes[column].set_title(agent_label, fontsize=10, fontweight="bold", pad=6)
        figure.subplots_adjust(right=0.88, wspace=0.08)
        color_bar = figure.colorbar(image, cax=figure.add_axes([0.90, 0.18, 0.02, 0.64]))
        color_bar.ax.tick_params(labelsize=8)
        color_bar.set_label("Max Q(s, a)", fontsize=9, fontweight="medium")
        figure.suptitle(f"{env_short} — Facing {direction_name}", fontsize=11, fontweight="bold", y=1.02)
        direction_path = output_dir / f"{file_prefix}_{direction_name.lower()}{stage_suffix}.png"
        figure.savefig(direction_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)
        print(f"    Per-direction plot saved → {direction_path}")


def plot_bar_chart(summary, label_a, label_b, plots_dir):
    """Mean max-Q of both agents for every environment.  summary: one dict per environment with env_id and the
    mean / standard deviation (over seeds and stages) of each agent."""
    if not summary:
        print("No data for summary bar chart — skipping.")
        return

    # baseline agents are blue and reward-shaped ones orange, as in the training curves
    bar_colors = []
    for label, default_color in ((label_a, BLUE), (label_b, ORANGE)):
        if "baseline" in label.lower():
            bar_colors.append(BLUE)
        elif "reward" in label.lower() or "rs" in label.lower():
            bar_colors.append(ORANGE)
        else:
            bar_colors.append(default_color)

    env_positions = np.arange(len(summary))
    bar_width = 0.32
    figure, axis = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    figure.patch.set_facecolor("white")
    bars_a = axis.bar(env_positions - bar_width / 2, [entry["a_mean"] for entry in summary], bar_width,
                      yerr=[entry["a_std"] for entry in summary], capsize=4, error_kw=dict(lw=1.0, capthick=1.0),
                      label=label_a, color=bar_colors[0], edgecolor="white", linewidth=0.6)
    bars_b = axis.bar(env_positions + bar_width / 2, [entry["b_mean"] for entry in summary], bar_width,
                      yerr=[entry["b_std"] for entry in summary], capsize=4, error_kw=dict(lw=1.0, capthick=1.0),
                      label=label_b, color=bar_colors[1], edgecolor="white", linewidth=0.6)

    axis.set_ylabel("Mean Maximum Q-Value", fontsize=10, fontweight="medium")
    axis.set_title("Q-Value Overestimation Comparison", fontsize=11, fontweight="bold", pad=10)
    axis.set_xticks(env_positions)
    axis.set_xticklabels([entry["env_id"].replace("MiniGrid-", "") for entry in summary], fontsize=9.5, fontweight="medium")
    axis.tick_params(axis="y", labelsize=9)

    for bar in list(bars_a) + list(bars_b):
        bar_height = bar.get_height()
        if bar_height > 0:
            axis.annotate(f"{bar_height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar_height), xytext=(0, 4),
                          textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="medium")

    axis.grid(True, axis="y", linestyle="--", alpha=0.35, color="#cccccc")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(fontsize=9, loc="best", frameon=True, facecolor="white", edgecolor="#e0e0e0", framealpha=0.95)

    figure.tight_layout()
    output_dir = Path(plots_dir) / "overestimation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "q_overestimation_comparison.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"  Summary bar chart saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Q-value overestimation between two trained agents.")
    parser.add_argument("--env-id", type=str, default=None, help="Specific environment ID to plot (default: run all in DEFAULT_ENV_IDS)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"), default=["dqn_vanilla", "ddqn_baseline"])
    parser.add_argument("--label-a", type=str, default=None)
    parser.add_argument("--label-b", type=str, default=None)
    parser.add_argument("--action-set", choices=["task", "full"], default="task")      # must match the trained agents
    parser.add_argument("--hidden-size", type=int, default=256)                         # must match the trained agents
    args = parser.parse_args()

    exp_a, exp_b = args.compare
    label_a = args.label_a or exp_a
    label_b = args.label_b or exp_b
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Results directory: {args.results_dir}")
    print(f"Comparing: {label_a}  vs  {label_b}")
    print()

    summary = []        # one entry per environment, for the bar chart
    for env_id in ([args.env_id] if args.env_id else DEFAULT_ENV_IDS):
        print(f"{'=' * 60}")
        print(f"Environment: {env_id}")
        print(f"{'=' * 60}")

        # ---- seeds for which both agents have a trained model ----
        models_a = find_models_by_seed(args.results_dir, env_id, exp_a)
        models_b = find_models_by_seed(args.results_dir, env_id, exp_b)
        common_seeds = sorted(set(models_a) & set(models_b))
        heatmap_label_a, heatmap_label_b = label_a, label_b
        if not common_seeds:
            for fallback_a, fallback_b in FALLBACK_PAIRS:
                fallback_models_a = find_models_by_seed(args.results_dir, env_id, fallback_a)
                fallback_models_b = find_models_by_seed(args.results_dir, env_id, fallback_b)
                fallback_seeds = sorted(set(fallback_models_a) & set(fallback_models_b))
                if fallback_seeds:
                    models_a, models_b, common_seeds = fallback_models_a, fallback_models_b, fallback_seeds
                    fallback_labels = []
                    for exp_name in (fallback_a, fallback_b):
                        fallback_labels.append(exp_name.replace("ddqn_reward_shaping", "RS-DDQN").replace("dqn_reward_shaping", "RS-DDQN")
                                               .replace("ddqn_baseline", "Baseline DDQN").replace("_", " ").title())
                    heatmap_label_a, heatmap_label_b = fallback_labels
                    print(f"  Auto-selected available experiment pair: {fallback_a} vs {fallback_b}")
                    break
        if not common_seeds:
            print(f"  ⚠  No matching seed pair found for {env_id} — skipping.\n")
            continue
        print(f"  Common seeds: {common_seeds}")

        if "DoorKey" in env_id:
            stages = [(1, "initial"), (2, "key_picked"), (3, "door_opened")]
        else:
            stages = [(1, "")]

        mean_max_q_a = []       # mean over cells and directions of max_a Q(s, a); one value per seed and stage
        mean_max_q_b = []
        for seed in common_seeds:
            print(f"\n  Seed {seed}:")
            env = make_env(env_id, seed, args.action_set)
            observation_dim = int(np.prod(env.observation_space.shape))
            num_actions = env.action_space.n
            action_labels = action_names(env_id, args.action_set, num_actions)

            q_net_a = QNetwork(observation_dim, num_actions, args.hidden_size).to(device)
            q_net_a.load_state_dict(torch.load(models_a[seed], map_location=device, weights_only=True))
            q_net_a.eval()
            q_net_b = QNetwork(observation_dim, num_actions, args.hidden_size).to(device)
            q_net_b.load_state_dict(torch.load(models_b[seed], map_location=device, weights_only=True))
            q_net_b.eval()
            seed_label_a = f"{heatmap_label_a}{epsilon_schedule_suffix(models_a[seed])}"
            seed_label_b = f"{heatmap_label_b}{epsilon_schedule_suffix(models_b[seed])}"

            for stage, stage_name in stages:
                print(f"    Computing {label_a} Q-values for {stage_name or 'Stage 1'} …")
                q_values_a, wall_mask, annotations = compute_q_values(env, q_net_a, seed, device, stage)
                print(f"    Computing {label_b} Q-values for {stage_name or 'Stage 1'} …")
                q_values_b, _, _ = compute_q_values(env, q_net_b, seed, device, stage)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)        # wall cells are all-NaN
                    stage_mean_a = np.nanmean(np.nanmax(q_values_a, axis=3))
                    stage_mean_b = np.nanmean(np.nanmax(q_values_b, axis=3))
                mean_max_q_a.append(stage_mean_a)
                mean_max_q_b.append(stage_mean_b)

                print(f"    Generating the directional heatmaps ({stage_name}) …")
                plot_stage_heatmaps(env_id, seed, stage_name, [(seed_label_a, q_values_a), (seed_label_b, q_values_b)],
                                    wall_mask, annotations, action_labels, args.plots_dir)
                print(f"    {seed_label_a} avg max Q : {stage_mean_a:.4f}")
                print(f"    {seed_label_b} avg max Q : {stage_mean_b:.4f}")
                print(f"    Difference                : {stage_mean_a - stage_mean_b:+.4f}")
            env.close()

        summary.append({
            "env_id": env_id,
            "a_mean": float(np.mean(mean_max_q_a)),
            "b_mean": float(np.mean(mean_max_q_b)),
            "a_std": float(np.std(mean_max_q_a)) if len(mean_max_q_a) > 1 else 0.0,
            "b_std": float(np.std(mean_max_q_b)) if len(mean_max_q_b) > 1 else 0.0,
        })

    plot_bar_chart(summary, label_a, label_b, args.plots_dir)
    print("\nAll done.")


if __name__ == "__main__":
    main()
