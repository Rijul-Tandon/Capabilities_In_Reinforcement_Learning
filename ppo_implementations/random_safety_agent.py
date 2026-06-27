import argparse
import csv
import json
import time
from pathlib import Path

import safety_gymnasium
import gymnasium as gym
import numpy as np
from tqdm import tqdm


def episode_success(episode_return, reward_threshold):
    if reward_threshold is None:
        return float(episode_return) > 0.0
    return float(episode_return) >= reward_threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="random_safety_agent")
    parser.add_argument("--env-id", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    env = gym.make(args.env_id)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    reward_threshold = getattr(getattr(env.unwrapped, "spec", None), "reward_threshold", None)

    obs, _ = env.reset(seed=args.seed)
    episode_cost = 0.0
    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(
        episode_file,
        fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "episodic_cost"],
    )
    episode_writer.writeheader()

    pbar = tqdm(range(args.total_timesteps), desc=args.exp_name)
    for global_step in pbar:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_cost += float(info.get("cost", 0.0))
        if terminated or truncated:
            episode_return = float(np.asarray(info["episode"]["r"]).reshape(-1)[0])
            episode_length = int(np.asarray(info["episode"]["l"]).reshape(-1)[0])
            success = episode_success(episode_return, reward_threshold)
            episode_writer.writerow(
                {
                    "global_step": global_step + 1,
                    "episodic_return": episode_return,
                    "episodic_length": episode_length,
                    "goal_reached": int(success),
                    "episodic_cost": episode_cost,
                }
            )
            episode_file.flush()
            episode_cost = 0.0
            obs, _ = env.reset()

    episode_file.close()
    env.close()
    print(f"Done. Results: {run_dir}")
