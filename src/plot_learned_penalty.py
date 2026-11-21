"""What the learned penalty head learned: its scores on stuck and moving transitions over training, the stuck rate
and episode length of all agents, and the trained penalty table drawn on the grid (heat maps, transition arrows,
histograms and a per-seed summary)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn_common import make_env
from learned_penalty import DiscreteStateSpace, TransitionPenaltyHead

# experiment name, legend label, colour (the same colours as plot_comparison.py)
AGENTS = [
    ("ddqn_baseline", "Baseline DDQN", "#1f77b4"),
    ("ddqn_reward_shaping", "RS-DDQN (manual penalty)", "#ff7f0e"),
    ("ddqn_learned_penalty", "LP-DDQN (learned penalty)", "#2ca02c"),
]
DIRECTION_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
DIRECTION_OFFSETS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}      # (dx, dy) of one step forward; y grows downwards

# Transition types of the learned table.  Tables with a direction have stay / forward / turn; position-only tables
# (--penalty-state position) have stay / neighbour.
TRANSITION_TYPES = ["stay", "forward", "turn", "neighbour"]
TRANSITION_COLOURS = {"stay": "#d62728", "forward": "#2ca02c", "turn": "#1f77b4", "neighbour": "#2ca02c"}
TRANSITION_TITLES = {"stay": "stay in place  s → s", "forward": "move forward one cell", "turn": "turn on the spot",
                     "neighbour": "move to a neighbouring cell"}
SCORE_LABEL = "penalty score  (penalty = scale × score)"


# ============================================================================
# RUNS
# ============================================================================

def read_csv(path):
    """The rows of a CSV file as dicts; no rows if the file does not exist."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_runs(results_dir, env_id, exp_name):
    """{seed: run} for every run of one agent on one environment; a run is a dict of run_dir, config, episodes, metrics."""
    runs = {}
    for run_dir in sorted(Path(results_dir).glob(f"{env_id}__{exp_name}__*")):
        if not (run_dir / "config.json").exists():
            continue
        with open(run_dir / "config.json", encoding="utf-8") as config_file:
            config = json.load(config_file)
        if config.get("exp_name") != exp_name:
            continue
        seed = int(config.get("seed", len(runs) + 1))
        runs[seed] = {
            "run_dir": run_dir,
            "config": config,
            "episodes": read_csv(run_dir / "episodes.csv"),
            "metrics": read_csv(run_dir / "metrics.csv"),
        }
    return runs


# ============================================================================
# TRAINING CURVES
# ============================================================================

def style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_curve_over_seeds(ax, runs, log_name, column, bin_edges, colour, label):
    """Draws one logged column against the training step: the mean over seeds with a min..max band.

    log_name is "episodes" or "metrics".  Each seed's values are first resampled onto the common step bins.
    Returns False if no seed has data for this column.
    """
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    seed_curves = []
    for run in runs.values():
        # the logged (step, value) pairs, skipping blank or non-numeric entries
        steps, values = [], []
        for row in run[log_name]:
            try:
                step, value = float(row["global_step"]), float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(step) and np.isfinite(value):
                steps.append(step)
                values.append(value)
        if not steps:
            continue
        steps, values = np.asarray(steps), np.asarray(values)

        # one value per bin: the mean of the bin where it has data ...
        curve = np.full(len(bin_centres), np.nan)
        for bin_number in range(len(bin_centres)):
            in_bin = (steps >= bin_edges[bin_number]) & (steps < bin_edges[bin_number + 1])
            if in_bin.any():
                curve[bin_number] = values[in_bin].mean()
        # ... and linear interpolation between the logged points elsewhere, so sparse logs still give a continuous curve
        order = np.argsort(steps)
        sorted_steps, sorted_values = steps[order], values[order]
        if len(sorted_steps) >= 2:
            within_log = (bin_centres >= sorted_steps[0]) & (bin_centres <= sorted_steps[-1])
            empty_bins = np.isnan(curve) & within_log
            curve[empty_bins] = np.interp(bin_centres[empty_bins], sorted_steps, sorted_values)
        seed_curves.append(curve)

    if not seed_curves:
        return False
    seed_curves = np.vstack(seed_curves)
    ax.plot(bin_centres, np.nanmean(seed_curves, axis=0), color=colour, lw=1.8, label=f"{label} (n={len(seed_curves)})")
    if len(seed_curves) > 1:
        ax.fill_between(bin_centres, np.nanmin(seed_curves, axis=0), np.nanmax(seed_curves, axis=0), color=colour, alpha=0.15, lw=0)
    return True


def plot_training_curves(env_id, runs_by_agent, plots_dir, num_bins):
    last_step = 0
    for runs in runs_by_agent.values():
        for run in runs.values():
            for rows in (run["episodes"], run["metrics"]):
                if rows:
                    last_step = max(last_step, max(float(row["global_step"]) for row in rows if row.get("global_step")))
    if last_step == 0:
        print(f"[plot_learned_penalty] no episode or metric logs found for {env_id}")
        return
    bin_edges = np.linspace(0, last_step + 1, num_bins + 1)

    # What the head penalises: its score on stuck and on moving transitions
    learned_runs = runs_by_agent["ddqn_learned_penalty"]
    if learned_runs:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_curve_over_seeds(ax, learned_runs, "metrics", "penalty_prob_stuck", bin_edges, "#d62728", "stuck transitions (observation unchanged)")
        plot_curve_over_seeds(ax, learned_runs, "metrics", "penalty_prob_move", bin_edges, "#2ca02c", "moving transitions")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best", fontsize=9)
        style_axes(ax, "Training step", "penalty score (penalty = scale × score)", f"Learned penalty head: what it penalises ({env_id})")
        fig.tight_layout()
        fig.savefig(plots_dir / "learned_penalty_probability.png", dpi=150)
        plt.close(fig)

    # All agents side by side: stuck rate and episode length
    for log_name, column, ylabel, title, file_name in (
            ("metrics", "stuck_rate", "Stuck rate (fraction of steps)", "Stuck rate over training", "stuck_rate_by_agent.png"),
            ("episodes", "episodic_length", "Episode length (steps)", "Episode length over training", "episode_length_by_agent.png")):
        fig, ax = plt.subplots(figsize=(7, 4))
        any_curve_drawn = False
        for exp_name, label, colour in AGENTS:
            if plot_curve_over_seeds(ax, runs_by_agent[exp_name], log_name, column, bin_edges, colour, label):
                any_curve_drawn = True
        if any_curve_drawn:
            ax.legend(loc="best", fontsize=9)
            style_axes(ax, "Training step", ylabel, f"{title} ({env_id})")
            fig.tight_layout()
            fig.savefig(plots_dir / file_name, dpi=150)
        plt.close(fig)


# ============================================================================
# THE LEARNED TABLE ON THE GRID
# ============================================================================

@torch.no_grad()
def transition_scores(head, from_states, to_states, valid):
    """sigmoid(logit(s, s')) of one transition per cell: a (W, H) map with NaN where `valid` is False.

    from_states, to_states: (W, H) arrays of state indices; valid: (W, H) boolean array.
    """
    scores = np.full(valid.shape, np.nan)
    if valid.any():
        logits = head.logits(torch.as_tensor(from_states[valid]), torch.as_tensor(to_states[valid]))
        scores[valid] = torch.sigmoid(logits).numpy()
    return scores


def penalty_maps(head, space, walls):
    """The score of each transition type from every cell, as (W, H) maps; NaN for walls and for moves into a wall.

    Table with a direction:   {"stay": [...], "forward": [...], "turn": [...]}, one map per facing direction.
    Position-only table:      {"stay": [map], "neighbour": [map], "neighbour_by_direction": [one map per direction]}.
    """
    width, height = space.width, space.height
    free = ~walls
    stay_maps, move_maps, turn_maps = [], [], []
    move_score_sum = np.zeros((width, height))      # over the four directions, for the position-only table
    num_possible_moves = np.zeros((width, height))

    for direction in range(4):
        step_x, step_y = DIRECTION_OFFSETS[direction]
        states = np.zeros((width, height), dtype=np.int64)              # index of the state (x, y, direction)
        states_ahead = np.zeros((width, height), dtype=np.int64)        # index of the state one cell ahead
        can_move = np.zeros((width, height), dtype=bool)                # the cell ahead exists and is not a wall
        for x in range(width):
            for y in range(height):
                states[x, y] = space.index(x, y, direction)
                ahead_x, ahead_y = x + step_x, y + step_y
                if free[x, y] and 0 <= ahead_x < width and 0 <= ahead_y < height and not walls[ahead_x, ahead_y]:
                    can_move[x, y] = True
                    states_ahead[x, y] = space.index(ahead_x, ahead_y, direction)

        stay_maps.append(transition_scores(head, states, states, free))
        move_scores = transition_scores(head, states, states_ahead, can_move)
        move_maps.append(move_scores)
        move_score_sum += np.nan_to_num(move_scores)
        num_possible_moves += can_move

        if space.with_direction:
            # turning: the mean score of the turn to the left and the turn to the right
            turn_scores = np.zeros((width, height))
            for turned_direction in ((direction - 1) % 4, (direction + 1) % 4):
                states_turned = np.zeros((width, height), dtype=np.int64)
                for x in range(width):
                    for y in range(height):
                        states_turned[x, y] = space.index(x, y, turned_direction)
                turn_scores += np.nan_to_num(transition_scores(head, states, states_turned, free))
            turn_scores[~free] = np.nan
            turn_maps.append(turn_scores / 2.0)

    if space.with_direction:
        return {"stay": stay_maps, "forward": move_maps, "turn": turn_maps}

    # Position-only table: the state has no direction, so the four "stay" maps are identical, and "neighbour" is
    # the mean score of the moves to the free neighbouring cells.
    neighbour_scores = np.where(num_possible_moves > 0, move_score_sum / np.maximum(num_possible_moves, 1), np.nan)
    neighbour_scores[~free] = np.nan
    return {"stay": [stay_maps[0]], "neighbour": [neighbour_scores], "neighbour_by_direction": move_maps}


def draw_layout(ax, walls, marks):
    """Dark walls, the S / G / K / D marks and the cell grid lines."""
    width, height = walls.shape
    for x in range(width):
        for y in range(height):
            if walls[x, y]:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#333333"))
    for (x, y), mark in marks.items():
        ax.text(x + 0.33, y - 0.33, mark, color="white", fontsize=9, fontweight="bold", ha="center", va="center",
                path_effects=[path_effects.withStroke(linewidth=2, foreground="black")])
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#888888", linestyle="-", linewidth=0.6)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def plot_transition_map(env_id, seed, maps, walls, marks, plots_dir, subtitle):
    """One grid: a disc per cell for s -> s and one arrow per direction for the move to the next cell, coloured by score."""
    width, height = walls.shape
    colour_map = plt.get_cmap("Reds")
    stay_scores = np.nanmean(np.stack(maps["stay"]), axis=0)        # mean over the facing directions
    move_maps = maps["forward"] if "forward" in maps else maps["neighbour_by_direction"]

    fig, ax = plt.subplots(figsize=(0.95 * width + 2.4, 0.95 * height + 1.4))
    draw_layout(ax, walls, marks)
    for x in range(width):
        for y in range(height):
            if walls[x, y] or not np.isfinite(stay_scores[x, y]):
                continue
            ax.add_patch(plt.Circle((x, y), 0.17, color=colour_map(stay_scores[x, y]), ec="#555555", lw=0.5, zorder=3))
            for direction in range(4):
                move_score = move_maps[direction][x, y]
                if not np.isfinite(move_score):
                    continue
                step_x, step_y = DIRECTION_OFFSETS[direction]
                ax.annotate("", xy=(x + 0.47 * step_x, y + 0.47 * step_y), xytext=(x + 0.2 * step_x, y + 0.2 * step_y),
                            arrowprops=dict(arrowstyle="-|>", color=colour_map(move_score), lw=1.8, mutation_scale=9, shrinkA=0, shrinkB=0),
                            zorder=4)
    colour_bar = fig.colorbar(plt.cm.ScalarMappable(cmap=colour_map, norm=plt.Normalize(0, 1)), ax=ax, fraction=0.045, pad=0.03)
    colour_bar.set_label(SCORE_LABEL, fontsize=9)
    ax.set_title(f"{env_id} — learned penalty per transition, seed {seed}\n"
                 f"disc = stay in place (mean over facing directions);  arrow = move one cell that way\n{subtitle}",
                 fontsize=9.5, fontweight="bold")
    fig.text(0.42, 0.01, "S = Start   G = Goal   K = Key   D = Door   dark = wall", ha="center", va="bottom", fontsize=8, color="#444444")
    fig.savefig(plots_dir / f"{env_id}_penalty_transitions_seed{seed}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_penalty_histogram(env_id, seed, maps, plots_dir):
    """Distribution of the scores over the free cells, one histogram per transition type."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for transition_type in TRANSITION_TYPES:
        if transition_type not in maps:
            continue
        scores = np.concatenate([score_map[np.isfinite(score_map)].ravel() for score_map in maps[transition_type]])
        if not len(scores):
            continue
        colour = TRANSITION_COLOURS[transition_type]
        ax.hist(scores, bins=np.linspace(0, 1, 26), color=colour, alpha=0.45,
                label=f"{transition_type} (mean {scores.mean():.2f}, n={len(scores)})")
        ax.axvline(scores.mean(), color=colour, lw=1.5, ls="--")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9, loc="best")
    style_axes(ax, SCORE_LABEL, "number of (cell, direction) entries", f"Learned penalty distribution by transition type ({env_id}, seed {seed})")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{env_id}_penalty_histogram_seed{seed}.png", dpi=150)
    plt.close(fig)


def plot_penalty_tables(env_id, learned_runs, plots_dir):
    """The three figures of every learned-penalty run, then one summary figure over the seeds."""
    mean_scores_by_seed = {}
    for seed, run in sorted(learned_runs.items()):
        config = run["config"]
        penalty_rank = int(config.get("penalty_rank", 0))

        # the grid layout this run was trained on, and its state space
        env = make_env(env_id, seed, config.get("action_set", "task"))
        env.reset(seed=seed)
        space = DiscreteStateSpace(env, env_id, config.get("penalty_state", "position_dir"))
        grid = env.unwrapped.grid
        walls = np.zeros((space.width, space.height), dtype=bool)
        marks = {tuple(env.unwrapped.agent_pos): "S"}
        for x in range(space.width):
            for y in range(space.height):
                cell = grid.get(x, y)
                if cell is None:
                    continue
                if cell.type == "wall":
                    walls[x, y] = True
                elif cell.type == "goal":
                    marks[(x, y)] = "G"
                elif cell.type == "key":
                    marks[(x, y)] = "K"
                elif cell.type == "door":
                    marks[(x, y)] = "D"
        env.close()

        # the trained head
        head_path = run["run_dir"] / "penalty_net.pt"
        if not head_path.exists():
            continue
        head = TransitionPenaltyHead(space.size, penalty_rank)
        try:
            head.load_state_dict(torch.load(head_path, map_location="cpu"))
        except RuntimeError as error:
            print(f"[plot_learned_penalty] {run['run_dir'].name}: head does not match the state space ({error}); skipped")
            continue
        head.eval()

        maps = penalty_maps(head, space, walls)
        subtitle = f"state = {space.mode}, {space.size} states, {'full table' if penalty_rank == 0 else f'rank {penalty_rank}'}"
        plot_transition_map(env_id, seed, maps, walls, marks, plots_dir, subtitle)
        plot_penalty_histogram(env_id, seed, maps, plots_dir)

        # the table as heat maps: one row per transition type, one column per facing direction
        row_types = [transition_type for transition_type in TRANSITION_TYPES if transition_type in maps]
        num_columns = 4 if space.with_direction else 1
        fig, axes = plt.subplots(len(row_types), num_columns, figsize=(3.4 * num_columns + 1.2, 3.3 * len(row_types)), squeeze=False)
        for row, transition_type in enumerate(row_types):
            for column in range(num_columns):
                ax = axes[row, column]
                scores = maps[transition_type][column]
                image = ax.imshow(scores.T, origin="upper", cmap="Reds", vmin=0.0, vmax=1.0, aspect="equal")
                for x in range(space.width):
                    for y in range(space.height):
                        if not walls[x, y] and np.isfinite(scores[x, y]):
                            ax.text(x, y, f"{scores[x, y]:.2f}", ha="center", va="center", fontsize=6.5,
                                    color="black" if scores[x, y] < 0.6 else "white")
                draw_layout(ax, walls, marks)
                panel_title = TRANSITION_TITLES[transition_type]
                if space.with_direction:
                    panel_title += f"\nfacing {DIRECTION_NAMES[column]}"
                ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=6)
        fig.subplots_adjust(right=0.9, wspace=0.08, hspace=0.3)
        colour_bar = fig.colorbar(image, cax=fig.add_axes([0.92, 0.2, 0.015, 0.6]))
        colour_bar.set_label(SCORE_LABEL, fontsize=9)
        fig.suptitle(f"{env_id} — learned penalty table, seed {seed}  ({subtitle})", fontsize=11, fontweight="bold", y=0.995)
        fig.text(0.44, 0.005, "S = Start   G = Goal   K = Key   D = Door   dark = wall   NaN = move into a wall (not drawn)",
                 ha="center", va="bottom", fontsize=8, color="#444444")
        fig.savefig(plots_dir / f"{env_id}_penalty_table_seed{seed}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

        mean_scores_by_seed[seed] = {}
        for transition_type in row_types:
            mean_scores_by_seed[seed][transition_type] = float(np.nanmean(np.stack(maps[transition_type])))
        print(f"[plot_learned_penalty] seed {seed}: "
              + ", ".join(f"{transition_type} {mean_score:.3f}" for transition_type, mean_score in mean_scores_by_seed[seed].items()))

    # Summary: mean score per transition type, one group of bars per seed
    if mean_scores_by_seed:
        seeds = sorted(mean_scores_by_seed)
        bar_types = []
        for transition_type in TRANSITION_TYPES:
            if any(transition_type in mean_scores for mean_scores in mean_scores_by_seed.values()):
                bar_types.append(transition_type)
        bar_width = 0.8 / len(bar_types)
        fig, ax = plt.subplots(figsize=(1.4 * len(seeds) + 3.5, 3.6))
        for bar_number, transition_type in enumerate(bar_types):
            # a seed trained with another --penalty-state simply lacks some bars
            heights = [mean_scores_by_seed[seed].get(transition_type, np.nan) for seed in seeds]
            ax.bar(np.arange(len(seeds)) + (bar_number - (len(bar_types) - 1) / 2) * bar_width, heights,
                   width=bar_width, color=TRANSITION_COLOURS[transition_type], label=transition_type)
        ax.set_xticks(np.arange(len(seeds)))
        ax.set_xticklabels([f"seed {seed}" for seed in seeds])
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="#888888", lw=0.8, ls="--")
        ax.legend(fontsize=9, loc="best")
        style_axes(ax, "", "mean penalty score over free cells", f"Learned penalty by transition type ({env_id})")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{env_id}_penalty_table_summary.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plots-dir", type=str, default="plots/learned_penalty")
    parser.add_argument("--bins", type=int, default=40, help="number of training-progress bins for the curves")
    args = parser.parse_args()
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs_by_agent = {}
    for exp_name, _, _ in AGENTS:
        runs_by_agent[exp_name] = load_runs(args.results_dir, args.env_id, exp_name)
    if not any(runs_by_agent.values()):
        print(f"[plot_learned_penalty] no runs found for {args.env_id} in {args.results_dir}")
        return

    plot_training_curves(args.env_id, runs_by_agent, plots_dir, args.bins)
    if runs_by_agent["ddqn_learned_penalty"]:
        plot_penalty_tables(args.env_id, runs_by_agent["ddqn_learned_penalty"], plots_dir)
    print(f"[plot_learned_penalty] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
