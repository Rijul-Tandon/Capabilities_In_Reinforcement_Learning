"""DQN for Atari with VQ-VAE action-alias masking.

The masker penalizes an action when it either produces a VQ-VAE self-loop or
duplicates a previously observed discrete transition (source code -> target
code) that has already been produced by a *different* action.
"""

import argparse
import csv
import json
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        distances = (inputs.square().sum(1, keepdim=True)
                     + self.embedding.weight.square().sum(1)
                     - 2 * inputs @ self.embedding.weight.T)
        indices = distances.argmin(1)
        quantized = self.embedding(indices)
        loss = F.mse_loss(quantized, inputs.detach()) + self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
        return inputs + (quantized - inputs).detach(), loss, indices


class AtariVQVAE(nn.Module):
    """A compact convolutional VQ-VAE for four stacked 84x84 frames."""
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, embedding_dim),
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64 * 7 * 7), nn.ReLU(), nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, 3, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 8, 4), nn.Sigmoid(),
        )

    def codes(self, states):
        _, _, indices = self.vq(self.encoder(states))
        return indices

    def forward(self, states):
        z = self.encoder(states)
        zq, vq_loss, indices = self.vq(z)
        return self.decoder(zq), vq_loss, indices


class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions))

    def forward(self, states):
        return self.head(self.features(states.float() / 255.0))


class ReplayBuffer:
    def __init__(self, capacity, observation_shape, device):
        self.capacity, self.device, self.pos, self.size = capacity, device, 0, 0
        self.obs = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, next_obs, action, reward, terminated):
        self.obs[self.pos], self.next_obs[self.pos] = np.asarray(obs), np.asarray(next_obs)
        self.actions[self.pos], self.rewards[self.pos], self.terminated[self.pos] = action, reward, terminated
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return tuple(torch.as_tensor(values[indices], device=self.device) for values in
                     (self.obs, self.next_obs, self.actions, self.rewards, self.terminated))


def make_env(env_id, seed):
    # full_action_space=True exposes all 18 Atari actions, including LEFTFIRE and UPLEFT.
    env = gym.make(env_id, frameskip=1, full_action_space=True)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)
    env.reset(seed=seed)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Atari DQN with VQ-VAE action-alias masking")
    parser.add_argument("--env-id", default="ALE/DemonAttack-v5")
    parser.add_argument("--agent", choices=["dqn_baseline", "dqn_vqvae_masked"], default="dqn_vqvae_masked")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-network-frequency", type=int, default=1000)
    parser.add_argument("--mask-penalty", type=float, default=1.0)
    parser.add_argument("--vqvae-epochs", type=int, default=10)
    parser.add_argument("--dataset-size", type=int, default=5000)
    parser.add_argument("--num-embeddings", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def epsilon(step, total):
    return max(0.01, 1.0 - 0.99 * step / max(1, total * 0.1))


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env_id, args.seed)
    if env.action_space.n != 18:
        raise RuntimeError(f"Expected full Atari action space (18 actions), got {env.action_space.n}.")
    print(f"{args.env_id}: actions={env.unwrapped.get_action_meanings()}")
    run_dir = Path(args.results_dir) / f"{args.env_id.replace('/', '_')}__{args.agent}__{args.seed}__{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    vqvae = None
    if args.agent == "dqn_vqvae_masked":
        vqvae = AtariVQVAE(args.embedding_dim, args.num_embeddings).to(device)
        states, state = [], env.reset(seed=args.seed)[0]
        for _ in range(args.dataset_size):
            states.append(np.asarray(state))
            state, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                state, _ = env.reset()
        loader = torch.utils.data.DataLoader(torch.as_tensor(np.asarray(states)), batch_size=32, shuffle=True)
        optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)
        for epoch in range(args.vqvae_epochs):
            losses = []
            for batch in loader:
                batch = batch.to(device).float() / 255.0
                reconstruction, vq_loss, _ = vqvae(batch)
                loss = F.mse_loss(reconstruction, batch) + vq_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
            print(f"VQ-VAE epoch {epoch + 1}/{args.vqvae_epochs}: loss={np.mean(losses):.5f}")
        vqvae.eval()

    q_net, target_net = QNetwork(env.action_space.n).to(device), QNetwork(env.action_space.n).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.buffer_size, env.observation_space.shape, device)
    transition_actions = defaultdict(set)
    recent_masks, recent_alias_masks = deque(maxlen=1000), deque(maxlen=1000)
    episodes = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metrics = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(episodes, fieldnames=["global_step", "episodic_return", "episodic_length", "epsilon"])
    metric_writer = csv.DictWriter(metrics, fieldnames=["global_step", "epsilon", "td_loss", "mask_penalty_frequency", "transition_alias_penalty_frequency"])
    episode_writer.writeheader(); metric_writer.writeheader()
    obs, episode_return, episode_length, last_loss = env.reset(seed=args.seed)[0], 0.0, 0, np.nan

    for step in tqdm(range(args.total_timesteps), desc=f"{args.env_id} [{args.agent}]"):
        eps = epsilon(step, args.total_timesteps)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q_net(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)).argmax(1).item()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        mask_applied = alias_mask_applied = 0.0
        shaped_reward = float(reward)
        if vqvae is not None:
            with torch.no_grad():
                pair = tuple(vqvae.codes(torch.as_tensor(np.asarray(frame), device=device).unsqueeze(0).float() / 255.0).item()
                             for frame in (obs, next_obs))
            prior_actions = transition_actions[pair]
            self_loop, action_alias = pair[0] == pair[1], any(previous != action for previous in prior_actions)
            if self_loop or action_alias:
                shaped_reward -= args.mask_penalty
                mask_applied, alias_mask_applied = 1.0, float(action_alias)
            prior_actions.add(action)
        recent_masks.append(mask_applied); recent_alias_masks.append(alias_mask_applied)
        replay.add(obs, next_obs, action, shaped_reward, float(terminated))
        obs, episode_return, episode_length = next_obs, episode_return + float(reward), episode_length + 1
        if terminated or truncated:
            episode_writer.writerow({"global_step": step, "episodic_return": episode_return, "episodic_length": episode_length, "epsilon": eps})
            episodes.flush(); obs, _ = env.reset(); episode_return, episode_length = 0.0, 0
        if step >= args.learning_starts and step % args.train_frequency == 0:
            b_obs, b_next, b_actions, b_rewards, b_terminated = replay.sample(args.batch_size)
            with torch.no_grad():
                next_actions = q_net(b_next).argmax(1, keepdim=True)
                target = b_rewards + args.gamma * target_net(b_next).gather(1, next_actions).squeeze(1) * (1 - b_terminated)
            predicted = q_net(b_obs).gather(1, b_actions.long().unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(predicted, target); optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0); optimizer.step(); last_loss = loss.item()
        if step % args.target_network_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())
        if step % 1000 == 0:
            metric_writer.writerow({"global_step": step, "epsilon": eps, "td_loss": last_loss,
                                    "mask_penalty_frequency": np.mean(recent_masks) if recent_masks else 0.0,
                                    "transition_alias_penalty_frequency": np.mean(recent_alias_masks) if recent_alias_masks else 0.0})
            metrics.flush()
    episodes.close(); metrics.close(); env.close()


if __name__ == "__main__":
    train(parse_args())
