"""Standard Atari DQN baseline (Double-DQN).

This is a clean implementation with NO VQ-VAE components and uses the
environment's *minimal* action space (the standard Atari benchmark setting,
typically 4–6 actions depending on the game).  It intentionally mirrors the
architecture in ``dqn_atari_vqvae_masked.py`` so that performance differences
can be attributed solely to the action-masking mechanism.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ─── network ────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """Nature-DQN convolutional architecture."""
    def __init__(self, num_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(states.float() / 255.0))


# ─── replay buffer ──────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple, device: torch.device):
        self.capacity, self.device, self.pos, self.size = capacity, device, 0, 0
        self.obs = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, next_obs, action, reward, terminated):
        self.obs[self.pos] = np.asarray(obs)
        self.next_obs[self.pos] = np.asarray(next_obs)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminated[self.pos] = terminated
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(self.size, size=batch_size)
        return tuple(
            torch.as_tensor(arr[indices], device=self.device)
            for arr in (self.obs, self.next_obs, self.actions, self.rewards, self.terminated)
        )


# ─── environment ────────────────────────────────────────────────────────────
def make_env(env_id: str, seed: int) -> gym.Env:
    """Standard Atari preprocessing with *minimal* action space."""
    env = gym.make(env_id, frameskip=1)           # minimal action set by default
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.reset(seed=seed)
    return env


# ─── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard Atari DQN baseline (minimal action space)")
    parser.add_argument("--env-id", default="ALE/DemonAttack-v5")
    parser.add_argument("--agent", default="dqn_baseline_standard",
                        help="Label for this run (used in config.json / plots).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-network-frequency", type=int, default=1000)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


# ─── schedule ───────────────────────────────────────────────────────────────
def epsilon(step: int, total: int) -> float:
    """Linear ε decay: 1.0 → 0.01 over the first 10 % of training."""
    return max(0.01, 1.0 - 0.99 * step / max(1, total * 0.1))


# ─── training loop ──────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env_id, args.seed)
    num_actions = env.action_space.n
    print(f"{args.env_id}: actions={num_actions}  meanings={env.unwrapped.get_action_meanings()}")

    run_dir = (
        Path(args.results_dir)
        / f"{args.env_id.replace('/', '_')}__{args.agent}__{args.seed}__{int(time.time())}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["exp_name"] = args.agent
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    q_net = QNetwork(num_actions).to(device)
    target_net = QNetwork(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.buffer_size, env.observation_space.shape, device)

    episodes_fh = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metrics_fh = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(
        episodes_fh, fieldnames=["global_step", "episodic_return", "episodic_length", "epsilon"],
    )
    metric_writer = csv.DictWriter(
        metrics_fh, fieldnames=["global_step", "epsilon", "td_loss"],
    )
    episode_writer.writeheader()
    metric_writer.writeheader()

    obs = env.reset(seed=args.seed)[0]
    episode_return, episode_length, last_loss = 0.0, 0, float("nan")

    for step in tqdm(range(args.total_timesteps), desc=f"{args.env_id} [{args.agent}]"):
        eps = epsilon(step, args.total_timesteps)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q_net(
                    torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
                ).argmax(1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, next_obs, action, float(reward), float(terminated))
        obs = next_obs
        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            episode_writer.writerow({
                "global_step": step,
                "episodic_return": episode_return,
                "episodic_length": episode_length,
                "epsilon": eps,
            })
            episodes_fh.flush()
            obs, _ = env.reset()
            episode_return, episode_length = 0.0, 0

        # ── DDQN update ──
        if step >= args.learning_starts and step % args.train_frequency == 0:
            b_obs, b_next, b_actions, b_rewards, b_term = replay.sample(args.batch_size)
            with torch.no_grad():
                next_actions = q_net(b_next).argmax(1, keepdim=True)
                target_q = (
                    b_rewards
                    + args.gamma
                    * target_net(b_next).gather(1, next_actions).squeeze(1)
                    * (1.0 - b_term)
                )
            predicted_q = q_net(b_obs).gather(1, b_actions.long().unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(predicted_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()
            last_loss = loss.item()

        if step % args.target_network_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % 1000 == 0:
            metric_writer.writerow({
                "global_step": step,
                "epsilon": eps,
                "td_loss": last_loss,
            })
            metrics_fh.flush()

    episodes_fh.close()
    metrics_fh.close()
    env.close()
    print(f"Done. Results saved to {run_dir}")


if __name__ == "__main__":
    train(parse_args())
