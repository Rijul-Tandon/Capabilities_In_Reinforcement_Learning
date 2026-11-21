"""Standard Atari Double DQN baseline: no VQ-VAE, no penalty, and the game's minimal action space.
Network, replay buffer and update rule are the same as in dqn_atari_vqvae_masked.py."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

gym.register_envs(ale_py)


# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Standard Atari Double DQN baseline (minimal action space)")

    # Experiment
    parser.add_argument("--env-id", default="ALE/DemonAttack-v5")
    parser.add_argument("--agent", default="dqn_baseline_standard")     # label of the run in folder names and plots
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--results-dir", default="results")

    # Double DQN
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=100_000)                # replay buffer capacity, in transitions
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-starts", type=int, default=10_000)             # steps collected before the first update
    parser.add_argument("--train-frequency", type=int, default=4)                  # one gradient step every this many env steps
    parser.add_argument("--target-network-frequency", type=int, default=1000)      # copy q_net -> target_net every this many env steps

    return parser.parse_args()


MAX_GRADIENT_NORM = 10.0       # the DQN gradient is clipped to this norm
LOG_INTERVAL = 1000            # steps between two rows of metrics.csv


# ============================================================================
# Q-NETWORK, REPLAY BUFFER, ENVIRONMENT
# ============================================================================

class QNetwork(nn.Module):
    """Nature-DQN network: (batch, 4, 84, 84) uint8 frames -> (batch, num_actions) Q-values."""

    def __init__(self, num_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),                # (4, 84, 84)  -> (32, 20, 20)
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),               # (32, 20, 20) -> (64, 9, 9)
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),  # (64, 9, 9)  -> (64, 7, 7) -> (3136,)
        )
        self.head = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions))

    def forward(self, observations):
        return self.head(self.features(observations.float() / 255.0))


class ReplayBuffer:
    """Circular buffer of transitions (s, a, r, s', terminated), sampled uniformly at random."""

    def __init__(self, capacity, observation_shape, device):
        self.capacity = capacity
        self.device = device
        self.position = 0       # where the next transition is written
        self.size = 0           # number of transitions stored so far (at most capacity)
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.terminated = np.zeros(capacity, dtype=np.float32)

    def add(self, observation, action, reward, next_observation, terminated):
        self.observations[self.position] = np.asarray(observation)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = np.asarray(next_observation)
        self.terminated[self.position] = terminated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        buffer_indices = np.random.randint(self.size, size=batch_size)
        return (
            torch.as_tensor(self.observations[buffer_indices], device=self.device),
            torch.as_tensor(self.actions[buffer_indices], device=self.device),
            torch.as_tensor(self.rewards[buffer_indices], device=self.device),
            torch.as_tensor(self.next_observations[buffer_indices], device=self.device),
            torch.as_tensor(self.terminated[buffer_indices], device=self.device),
        )


def make_env(env_id, seed):
    """Standard Atari preprocessing: frame skip 4, grayscale 84x84, 4 stacked frames; the game's minimal action set."""
    env = gym.make(env_id, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.action_space.seed(seed)     # makes the random actions repeatable for a given seed
    env.reset(seed=seed)
    return env


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.env_id, args.seed)
    num_actions = env.action_space.n
    print(f"{args.env_id}: actions={num_actions}  meanings={env.unwrapped.get_action_meanings()}")

    # ---- run folder and loggers ----
    run_dir = Path(args.results_dir) / f"{args.env_id.replace('/', '_')}__{args.agent}__{args.seed}__{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["exp_name"] = args.agent     # plot_comparison.py groups runs by this field
    with open(run_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)

    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metric_file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(episode_file, fieldnames=["global_step", "episodic_return", "episodic_length", "epsilon"])
    metric_writer = csv.DictWriter(metric_file, fieldnames=["global_step", "epsilon", "td_loss"])
    episode_writer.writeheader()
    metric_writer.writeheader()

    # ---- networks ----
    q_net = QNetwork(num_actions).to(device)
    target_net = QNetwork(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space.shape, device)

    observation, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    last_td_loss = float("nan")

    for global_step in tqdm(range(args.total_timesteps), desc=f"{args.env_id} [{args.agent}]"):

        # ---- choose an action: epsilon falls linearly from 1.0 to 0.01 over the first 10% of training ----
        epsilon = max(0.01, 1.0 - 0.99 * global_step / max(1, args.total_timesteps * 0.1))
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.as_tensor(np.asarray(observation), device=device).unsqueeze(0))     # (1, num_actions)
                action = q_values.argmax(1).item()

        next_observation, reward, terminated, truncated, _ = env.step(action)

        # `terminated` is stored, not `terminated or truncated`: a time-limit cut still bootstraps from s'.
        replay_buffer.add(observation, action, float(reward), next_observation, float(terminated))
        observation = next_observation
        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            episode_writer.writerow({
                "global_step": global_step,
                "episodic_return": episode_return,
                "episodic_length": episode_length,
                "epsilon": epsilon,
            })
            episode_file.flush()
            observation, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        # ---- one Double DQN gradient step ----
        if global_step >= args.learning_starts and global_step % args.train_frequency == 0:
            batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_terminated = replay_buffer.sample(args.batch_size)
            with torch.no_grad():
                # the online network selects a*, the target network evaluates it
                best_next_actions = q_net(batch_next_observations).argmax(1, keepdim=True)                     # (batch, 1)
                next_q_values = target_net(batch_next_observations).gather(1, best_next_actions).squeeze(1)    # (batch,)
                td_target = batch_rewards + args.gamma * next_q_values * (1.0 - batch_terminated)
            # Q(s, a) of the actions that were taken: (batch, num_actions) -> (batch,)
            q_values_taken = q_net(batch_observations).gather(1, batch_actions.long().unsqueeze(1)).squeeze(1)
            td_loss = F.mse_loss(q_values_taken, td_target)
            optimizer.zero_grad()
            td_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()
            last_td_loss = td_loss.item()

        if global_step % args.target_network_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        if global_step % LOG_INTERVAL == 0:
            metric_writer.writerow({"global_step": global_step, "epsilon": epsilon, "td_loss": last_td_loss})
            metric_file.flush()

    episode_file.close()
    metric_file.close()
    env.close()
    print(f"Done. Results saved to {run_dir}")


if __name__ == "__main__":
    train(parse_args())
