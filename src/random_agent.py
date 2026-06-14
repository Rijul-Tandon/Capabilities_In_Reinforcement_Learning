"""
random_agent.py - Random Action Baseline Agent
================================================
This script serves as the absolute baseline for our experiments. The agent
has NO neural network and NO ability to learn. It takes completely random
actions at every single timestep.

Purpose:
  If a trained DQN agent cannot outperform the random agent, something is
  fundamentally wrong with the training setup (bad hyperparameters, broken
  observations, etc.). The random agent establishes the "floor" of performance.

Fake Epsilon:
  The random agent always acts randomly (epsilon = 1.0 effectively), but we
  compute a "fake" decaying epsilon value using the same schedule as the DQN
  agents. This fake epsilon is ONLY used for the X-axis in our comparison
  plots (which plot Return vs Epsilon and Goal Rate vs Epsilon). By matching
  the epsilon values, the random agent's flat performance line overlays
  perfectly on top of the DQN agents' improving curves, making it easy to
  visually compare them.

Usage:
  python random_agent.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 30000
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

# argparse: Parses command-line arguments from the terminal.
import argparse

# csv: Writes CSV files for episode-level metrics logging.
import csv

# json: Saves the experiment configuration as a JSON file for reproducibility.
import json

# time: Generates timestamps for unique run directory names.
import time

# Path: Object-oriented filesystem path handling (creating directories, joining paths).
from pathlib import Path

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

# gymnasium: The standard RL environment API (not directly used here but imported
#   for consistency; make_env from dqn_common handles the actual environment creation).
import gymnasium as gym

# minigrid: Importing this module registers all MiniGrid environments with gymnasium.
#   Without this import, gym.make("MiniGrid-...") would fail with "environment not found".
import minigrid

# numpy: Used for seeding the random number generator for reproducibility.
import numpy as np

# tqdm: Displays a progress bar in the terminal during the main loop.
from tqdm import tqdm

# ============================================================================
# LOCAL IMPORTS (from our own codebase)
# ============================================================================

# make_env: Creates the MiniGrid environment with all wrappers
#   (FullyObsWrapper, FlatObsWrapper, action subset, episode statistics).
# linear_schedule: Computes the decaying epsilon value for a given timestep.
#   We reuse it here to generate the "fake" epsilon for consistent plot X-axes.
from dqn_common import make_env, linear_schedule


def parse_args():
    """
    Parses command-line arguments for the random agent experiment.

    The arguments mirror those of the DQN agents so that:
      1. The run directory naming convention is consistent (env_id__exp_name__seed__timestamp)
      2. The episodes.csv format is identical (enabling shared plotting scripts)
      3. The fake epsilon schedule matches the DQN agents' real epsilon schedule

    Returns
    -------
    argparse.Namespace
        An object with attributes: exp_name, env_id, seed, total_timesteps,
        action_set, results_dir, start_e, end_e, exploration_fraction.
    """
    parser = argparse.ArgumentParser()

    # --exp-name: Experiment label used in directory names and plot legends.
    #   Default is "random_agent" to distinguish it from "dqn_baseline" and "dqn_reward_shaping".
    parser.add_argument("--exp-name", type=str, default="random_agent")

    # --env-id: Which MiniGrid environment to run in.
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")

    # --seed: Random seed for reproducibility.
    parser.add_argument("--seed", type=int, default=1)

    # --total-timesteps: How many environment steps to simulate.
    parser.add_argument("--total-timesteps", type=int, default=30000)

    # --action-set: "task" restricts actions to environment-relevant ones,
    #   "full" uses all 7 MiniGrid actions. Should match the DQN experiments.
    parser.add_argument("--action-set", choices=["task", "full"], default="task")

    # --results-dir: Parent directory where run folders are created.
    parser.add_argument("--results-dir", type=str, default="results")

    # --- Fake Epsilon Schedule Arguments ---
    # These control the fake epsilon computation for consistent plot X-axes.
    # They should match the defaults in dqn_common.parse_args() exactly.
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.3)
    parser.add_argument("--exploration-fraction", type=float, default=0.6)

    return parser.parse_args()


def main():
    """
    Main loop: runs the random agent through the environment for total_timesteps steps.

    The agent always calls env.action_space.sample() which returns a uniformly
    random action from the allowed action set. It never learns or improves.

    Outputs:
      results/<run_dir>/config.json   - Experiment configuration
      results/<run_dir>/episodes.csv  - Per-episode metrics (return, length, goal reached, epsilon)
    """
    args = parse_args()

    # Seed numpy's random number generator for reproducible random action sequences.
    np.random.seed(args.seed)

    # Create a uniquely named run directory using the same convention as DQN agents:
    # Format: {env_id}__{exp_name}__{seed}__{unix_timestamp}
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration as JSON so we can identify this run later.
    # vars(args) converts the Namespace object into a plain dictionary.
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Create the environment with the same wrappers as the DQN agents
    # (FullyObsWrapper, FlatObsWrapper, action subset) for a fair comparison.
    # capture_video=False: we don't need video recordings for the random agent.
    env = make_env(args.env_id, args.seed, args.action_set, False, run_name)

    # Reset the environment and get the initial observation.
    # The underscore (_) discards the 'info' dict returned by reset().
    obs, _ = env.reset(seed=args.seed)

    # --- CSV Logging Setup ---
    # The CSV format matches dqn_common's episodes.csv exactly so that
    # plot_comparison.py can read and plot all agents together.
    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(
        episode_file,
        fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "epsilon"],
    )
    episode_writer.writeheader()

    # Track cumulative reward and step count for the current episode
    episode_return = 0.0
    episode_length = 0

    # --- Main Loop ---
    pbar = tqdm(range(args.total_timesteps), desc=args.exp_name)
    for global_step in pbar:
        # Compute the "fake" epsilon value.
        # This agent ALWAYS acts randomly regardless of epsilon.
        # We only compute this so the X-axis in our comparison plots (Return vs Epsilon)
        # lines up correctly with the DQN agents' real epsilon values.
        fake_epsilon = linear_schedule(
            args.start_e,                                    # Starting epsilon (1.0)
            args.end_e,                                      # Ending epsilon (0.3)
            args.exploration_fraction * args.total_timesteps, # Duration of decay
            global_step,                                     # Current step
        )

        # Take a completely random action from the allowed action space.
        # env.action_space.sample() returns a uniformly random integer in [0, num_actions).
        action = env.action_space.sample()

        # Step the environment with the random action.
        # Returns: next_obs, reward, terminated (goal reached), truncated (time limit), info
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Accumulate the episode's total reward and step count
        episode_return += float(env_reward)
        episode_length += 1

        if done:
            # Episode finished. Log the results.
            # In MiniGrid, positive reward means the goal was reached.
            reached = episode_return > 0.0
            episode_writer.writerow(
                {
                    "global_step": global_step,
                    "episodic_return": episode_return,
                    "episodic_length": episode_length,
                    "goal_reached": int(reached),   # 1 if goal reached, 0 otherwise
                    "epsilon": fake_epsilon,          # Fake epsilon for plot alignment
                }
            )
            episode_file.flush()  # Force write to disk immediately

            # Reset the environment for the next episode
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
        else:
            # Episode continues: update the current observation
            obs = next_obs

    # --- Cleanup ---
    episode_file.close()
    env.close()
    print(f"Done. Results: {run_dir}")


if __name__ == "__main__":
    main()
