"""Random-action baseline on MiniGrid: the floor of performance.  It writes episodes.csv in the format of the DQN
agents; the epsilon column holds the value a DQN agent would have had at that step and is otherwise unused."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dqn_common import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="random_agent")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--fixed-layout", action="store_true")   # reset with the same seed every episode, so the map never changes
    parser.add_argument("--max-steps", type=int, default=-1)     # step limit of an episode; -1 keeps MiniGrid's own limit
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    # Only for the epsilon column of episodes.csv
    parser.add_argument("--epsilon-schedule", choices=["linear", "polynomial", "hardcoded", "cosine", "exponential", "cyclic"],
                        default="polynomial")
    parser.add_argument("--start-e", dest="start_epsilon", type=float, default=1.0)
    parser.add_argument("--end-e", dest="end_epsilon", type=float, default=0.1)
    parser.add_argument("--exploration-fraction", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2)

    env = make_env(args.env_id, args.seed, args.action_set, args.max_steps)
    env.reset(seed=args.seed)
    num_actions = env.action_space.n

    # how often each action was taken in each (x, y, direction)
    counts_shape = (env.unwrapped.width, env.unwrapped.height, 4, num_actions)
    state_action_counts = np.zeros(counts_shape, dtype=np.int64)
    state_action_counts_second_half = np.zeros(counts_shape, dtype=np.int64)

    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(episode_file, fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "epsilon"])
    episode_writer.writeheader()

    episode_return = 0.0
    episode_length = 0
    decay_steps = args.exploration_fraction * args.total_timesteps
    for global_step in tqdm(range(args.total_timesteps), desc=args.exp_name):
        # epsilon of a DQN agent at this step: linear if requested, otherwise the polynomial decay
        if args.epsilon_schedule == "linear":
            logged_epsilon = max((args.end_epsilon - args.start_epsilon) / decay_steps * global_step + args.start_epsilon, args.end_epsilon)
        else:
            progress = min(global_step / decay_steps, 1.0)
            logged_epsilon = args.end_epsilon + (args.start_epsilon - args.end_epsilon) * (1.0 - progress) ** 3.0

        action = env.action_space.sample()

        agent_x, agent_y = env.unwrapped.agent_pos
        agent_direction = env.unwrapped.agent_dir
        state_action_counts[agent_x, agent_y, agent_direction, action] += 1
        if global_step >= args.total_timesteps / 2:
            state_action_counts_second_half[agent_x, agent_y, agent_direction, action] += 1

        _, env_reward, terminated, truncated, _ = env.step(action)
        episode_return += float(env_reward)
        episode_length += 1

        if terminated or truncated:
            episode_writer.writerow({
                "global_step": global_step,
                "episodic_return": episode_return,
                "episodic_length": episode_length,
                "goal_reached": int(env_reward > 0),      # the reward is positive only on the step that reaches the goal
                "epsilon": logged_epsilon,
            })
            episode_file.flush()
            if args.fixed_layout:
                env.reset(seed=args.seed)
            else:
                env.reset()
            episode_return = 0.0
            episode_length = 0

    episode_file.close()
    np.save(run_dir / "state_action_counts.npy", state_action_counts)
    np.save(run_dir / "state_action_counts_second_half.npy", state_action_counts_second_half)
    env.close()
    print(f"Done. Results: {run_dir}")
