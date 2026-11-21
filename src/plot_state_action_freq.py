"""Runs every trained agent for a few evaluation episodes and draws, per agent, how often it stood on each cell and
which action it took there most often.  One figure per seed and evaluation epsilon."""

import argparse
import json
import random
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn_common import QNetwork, action_names, make_env

HIDDEN_SIZE = 256       # hidden units of the trained Q-networks (the --hidden-size default of the trainer)
DIRECTION_ARROWS = {0: "→", 1: "↓", 2: "←", 3: "↑"}      # MiniGrid directions: 0 east, 1 south, 2 west, 3 north
ACTION_ABBREVIATIONS = {"left": "L", "right": "R", "forward": "F", "pickup": "P", "drop": "Dp", "toggle": "T", "done": "Dn"}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def find_trained_models(results_dir, env_id, exp_name):
    """{seed: (path of q_net.pt, epsilon schedule of the run)} of one experiment.

    Run folders are named <env_id>__<exp_name>__<seed>__<timestamp>; when a seed was run more than once, the
    newest run is used.
    """
    models = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}__*")):
        if not (run_dir / "q_net.pt").exists() or not (run_dir / "config.json").exists():
            continue
        with open(run_dir / "config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        seed = config.get("seed", 1)
        if isinstance(seed, list):      # a multi-seed run lists all its seeds; it is filed under the first one
            seed = seed[0]
        models[int(seed)] = (run_dir / "q_net.pt", config.get("epsilon_schedule", ""))
    return models


def evaluate_agent(env, q_net, num_episodes, seed, epsilon, device):
    """Runs the agent for num_episodes episodes; episode i is reset with seed + i.

    Returns
      visit_counts           (width, height)                   number of steps the agent stood on each cell
      state_action_counts    (width, height, num_actions, 4)   actions taken on each cell, by facing direction
    """
    visit_counts = np.zeros((env.unwrapped.width, env.unwrapped.height), dtype=int)
    state_action_counts = np.zeros((env.unwrapped.width, env.unwrapped.height, env.action_space.n, 4), dtype=int)
    for episode in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            agent_x, agent_y = env.unwrapped.agent_pos
            agent_direction = env.unwrapped.agent_dir

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)   # (num_actions,)
                best_actions = (q_values == torch.max(q_values)).nonzero(as_tuple=True)[0].tolist()
                action = int(random.choice(best_actions))       # ties are broken at random

            visit_counts[agent_x, agent_y] += 1
            state_action_counts[agent_x, agent_y, action, agent_direction] += 1
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return visit_counts, state_action_counts


def action_symbol(action_name, facing_direction):
    """Text drawn for an action.  "forward" shows the direction of the move.  "left" and "right" are turns on the
    spot, so they get a rotation glyph: a loop of turns in one cell must not look like a blocked move."""
    abbreviation = ACTION_ABBREVIATIONS[action_name]
    if action_name == "forward":
        return f"{DIRECTION_ARROWS[facing_direction]} {abbreviation}"
    if action_name == "left":
        return f"↺ {abbreviation}"
    if action_name == "right":
        return f"↻ {abbreviation}"
    return abbreviation


def plot_evaluation(env_id, seed, epsilon, agents, num_episodes, action_set, plots_dir):
    """Evaluates the agents of one seed and writes one figure with a panel per agent.

    agents: list of (label, path of q_net.pt), with None as path when no trained model was found.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(env_id, seed, action_set)
    observation_dim = int(np.prod(env.observation_space.shape))
    num_actions = env.action_space.n
    names = action_names(env_id, action_set, num_actions)
    width, height = env.unwrapped.width, env.unwrapped.height

    # ---- map layout of the first evaluation episode ----
    env.reset(seed=seed)
    start_position = tuple(env.unwrapped.agent_pos)
    goal_position = None
    wall_positions, door_positions, key_positions = [], [], []
    for x in range(width):
        for y in range(height):
            cell = env.unwrapped.grid.get(x, y)
            if cell is None:
                continue
            if cell.type == "wall":
                wall_positions.append((x, y))
            elif cell.type == "goal":
                goal_position = (x, y)
            elif cell.type == "door":
                door_positions.append((x, y))
            elif cell.type == "key":
                key_positions.append((x, y))

    figure, axes = plt.subplots(1, len(agents), figsize=(8.5 * len(agents), 7.5), squeeze=False)
    evaluation_label = "Greedy (Q-values)" if epsilon == 0.0 else f"5% Exploration (ε={epsilon:g})"
    figure.suptitle(f"{env_id} — Evaluation Test [{evaluation_label}] (Seed {seed})", fontsize=17, fontweight="bold", y=1.05)

    legend_markers = {}         # legend label -> the first marker drawn for it
    for column, (label, model_path) in enumerate(agents):
        axis = axes[0, column]
        if model_path is None:
            axis.set_title(f"{label} (Not Found)", fontsize=12, fontweight="bold")
            continue

        q_net = QNetwork(observation_dim, num_actions, HIDDEN_SIZE).to(device)
        q_net.load_state_dict(torch.load(model_path, map_location=device))
        q_net.eval()
        visit_counts, state_action_counts = evaluate_agent(env, q_net, num_episodes, seed, epsilon, device)

        # ---- layout markers ----
        marker, = axis.plot(start_position[0] - 0.35, start_position[1] - 0.35, "bo", markersize=9, markeredgecolor="white")
        legend_markers.setdefault("Start", marker)
        if goal_position is not None:
            marker, = axis.plot(goal_position[0], goal_position[1], "g*", markersize=14, markeredgecolor="white")
            legend_markers.setdefault("Goal", marker)
        for x, y in wall_positions:
            marker, = axis.plot(x, y, "s", color="#333333", markersize=10)
            legend_markers.setdefault("Wall", marker)
        for x, y in door_positions:
            marker, = axis.plot(x, y, "s", color="saddlebrown", markersize=10, markeredgecolor="white")
            legend_markers.setdefault("Door", marker)
        for x, y in key_positions:
            marker, = axis.plot(x, y, "yD", markersize=8, markeredgecolor="black")
            legend_markers.setdefault("Key", marker)

        # ---- visit heatmap ----
        image = axis.imshow(np.log1p(visit_counts.T), origin="upper", cmap="Blues", alpha=0.50, aspect="equal")
        axis.set_title(f"{label}\nAction Counts", fontsize=12, fontweight="bold", pad=8)
        colour_bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colour_bar.ax.tick_params(labelsize=9)

        # ---- most frequent action(s) and number of visits of every visited cell ----
        for x in range(width):
            for y in range(height):
                symbol_counts = {}      # drawn symbol -> how often it was taken on this cell
                for action_index, action_name in enumerate(names):
                    for direction in range(4):
                        count = state_action_counts[x, y, action_index, direction]
                        if count > 0:
                            symbol = action_symbol(action_name, direction)
                            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + count
                if not symbol_counts:
                    continue
                highest_count = max(symbol_counts.values())
                most_frequent_symbols = [symbol for symbol, count in symbol_counts.items() if count == highest_count]
                axis.text(x, y, "\n".join(most_frequent_symbols) + f"\n{visit_counts[x, y]}",
                          ha="center", va="center", fontsize=8.0, fontweight="bold", color="black")

        axis.set_xticks(np.arange(-0.5, width, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, height, 1), minor=True)
        axis.grid(which="minor", color="#888888", linestyle="-", linewidth=0.8)
        axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        # ---- S / G / D / K letters next to the markers ----
        letters = {start_position: "S"}
        if goal_position is not None:
            letters[goal_position] = "G"
        for position in door_positions:
            letters[position] = "D"
        for position in key_positions:
            letters[position] = "K"
        for (x, y), letter in letters.items():
            offset_x, offset_y = (-0.35, -0.35) if letter == "S" else (0.32, -0.32)
            axis.text(x + offset_x, y + offset_y, letter, color="white", fontsize=10, fontweight="bold", ha="center", va="center",
                      path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])

    if legend_markers:
        figure.legend(handles=list(legend_markers.values()), labels=list(legend_markers.keys()),
                      loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=len(legend_markers), fontsize=10,
                      frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    action_key = "   |   ".join(f"{ACTION_ABBREVIATIONS[action_name]} = {action_name}" for action_name in names)
    figure.text(0.5, 0.02, f"Action key:  {action_key}", ha="center", va="bottom", fontsize=10, fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.95))
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])

    output_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if epsilon == 0.0:
        epsilon_suffix = "_q_values"
    else:
        epsilon_suffix = f"_eps_{epsilon:g}".replace(".", "_")
    output_path = output_dir / f"{env_id}_test_seed{seed}{epsilon_suffix}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Test plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots/reward_comparison")
    parser.add_argument("--episodes", type=int, default=1)      # evaluation episodes per agent
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.0, 0.05])      # one figure per value
    parser.add_argument("--action-set", choices=["task", "full"], default="task")      # must match the trained agents
    parser.add_argument("--model-spec", action="append", default=[], metavar="EXP_NAME|LABEL",
                        help="Evaluate a named experiment model with a custom label. May be repeated.")
    parser.add_argument("--seed", type=int, default=1)          # only used together with --model-spec
    args = parser.parse_args()

    # One column of the figure per entry: (label, {seed: (model path, epsilon schedule)})
    if args.model_spec:
        columns = []
        for spec in args.model_spec:
            if "|" not in spec:
                parser.error("--model-spec must use EXP_NAME|LABEL")
            exp_name, label = spec.split("|", 1)
            columns.append((label, find_trained_models(args.results_dir, args.env_id, exp_name)))
        seeds = [args.seed]
    else:
        baseline_models = (find_trained_models(args.results_dir, args.env_id, "ddqn_baseline")
                           or find_trained_models(args.results_dir, args.env_id, "dqn_baseline"))
        shaped_models = (find_trained_models(args.results_dir, args.env_id, "ddqn_reward_shaping")
                         or find_trained_models(args.results_dir, args.env_id, "dqn_reward_shaping"))
        learned_models = find_trained_models(args.results_dir, args.env_id, "ddqn_learned_penalty")
        columns = [("Baseline DDQN", baseline_models), ("RS-DDQN", shaped_models)]
        if learned_models:
            columns.append(("LP-DDQN", learned_models))
        seeds = sorted(set(baseline_models) | set(shaped_models) | set(learned_models))

    if not seeds:
        print(f"No trained models found for {args.env_id} in {args.results_dir}")
        raise SystemExit(0)
    print(f"Found seeds: {seeds} for {args.env_id}")

    for seed in seeds:
        agents = []         # (label, path of q_net.pt or None) for this seed
        for label, models in columns:
            if seed not in models:
                agents.append((label, None))
                continue
            model_path, epsilon_schedule = models[seed]
            if epsilon_schedule and not args.model_spec:
                agents.append((f"{label} ({epsilon_schedule})", model_path))      # default panels also name the run's epsilon schedule
            else:
                agents.append((label, model_path))

        for epsilon in args.epsilons:
            mode = "Greedy Q-values" if epsilon == 0.0 else f"{epsilon * 100:g}% random exploration"
            print(f"Generating combined test plot for seed={seed} ({mode}) ...")
            plot_evaluation(args.env_id, seed, epsilon, agents, args.episodes, args.action_set, args.plots_dir)

    print(f"Done. Performance test plots saved to {args.plots_dir}")
