"""Discrete-action PPO baseline and VQ-VAE reward-shaped PPO for Atari."""

import argparse
import csv
import json
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from dqn_atari_vqvae_masked import AtariVQVAE, make_env


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class AtariPPO(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.features = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, 4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(), nn.Flatten(),
        )
        self.actor = nn.Sequential(layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(), layer_init(nn.Linear(512, actions), 0.01))
        self.critic = nn.Sequential(layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(), layer_init(nn.Linear(512, 1), 1.0))

    def _features(self, obs):
        return self.features(obs.float() / 255.0)

    def value(self, obs):
        return self.critic(self._features(obs)).squeeze(-1)

    def action_and_value(self, obs, action=None):
        features = self._features(obs)
        distribution = Categorical(logits=self.actor(features))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(features).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Atari PPO with optional VQ-VAE transition-alias reward shaping")
    parser.add_argument("--env-id", default="ALE/DemonAttack-v5")
    parser.add_argument("--agent", choices=["ppo_baseline", "ppo_vqvae_masked"], default="ppo_baseline")
    parser.add_argument("--full-action-space", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--mask-penalty", type=float, default=1.0)
    parser.add_argument("--vqvae-epochs", type=int, default=10)
    parser.add_argument("--dataset-size", type=int, default=5000)
    parser.add_argument("--num-embeddings", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def pretrain_vqvae(env, args, device):
    model = AtariVQVAE(args.embedding_dim, args.num_embeddings).to(device)
    state, _ = env.reset(seed=args.seed)
    states = []
    for _ in range(args.dataset_size):
        states.append(np.asarray(state))
        state, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            state, _ = env.reset()
    loader = torch.utils.data.DataLoader(torch.as_tensor(np.asarray(states)), batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(args.vqvae_epochs):
        epoch_loss = []
        for batch in loader:
            batch = batch.to(device).float() / 255.0
            reconstruction, vq_loss, _ = model(batch)
            loss = F.mse_loss(reconstruction, batch) + vq_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss.append(loss.item())
        print(f"VQ-VAE epoch {epoch + 1}/{args.vqvae_epochs}: loss={np.mean(epoch_loss):.5f}")
    model.eval()
    return model


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env_id, args.seed, args.full_action_space)
    if env.action_space.n != 18:
        raise RuntimeError(f"Expected 18 full Atari actions, got {env.action_space.n}.")
    run_dir = Path(args.results_dir) / f"{args.env_id.replace('/', '_')}__{args.agent}__{args.seed}__{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy(); config["exp_name"] = args.agent
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    vqvae = pretrain_vqvae(env, args, device) if args.agent == "ppo_vqvae_masked" else None
    agent = AtariPPO(env.action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    transition_actions = defaultdict(set)
    recent_masks, recent_aliases = deque(maxlen=1000), deque(maxlen=1000)
    episodes = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metrics = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(episodes, fieldnames=["global_step", "episodic_return", "episodic_length", "epsilon"])
    metric_writer = csv.DictWriter(metrics, fieldnames=["global_step", "policy_loss", "value_loss", "entropy", "mask_penalty_frequency", "transition_alias_penalty_frequency"])
    episode_writer.writeheader(); metric_writer.writeheader()

    obs, episode_return, episode_length, last_stats = env.reset(seed=args.seed)[0], 0.0, 0, {}
    updates = (args.total_timesteps + args.rollout_steps - 1) // args.rollout_steps
    for update in tqdm(range(updates), desc=f"{args.env_id} [{args.agent}]"):
        length = min(args.rollout_steps, args.total_timesteps - update * args.rollout_steps)
        observations = torch.zeros((length, *env.observation_space.shape), dtype=torch.uint8, device=device)
        actions = torch.zeros(length, dtype=torch.long, device=device)
        log_probs = torch.zeros(length, device=device); rewards = torch.zeros(length, device=device)
        dones = torch.zeros(length, device=device); values = torch.zeros(length, device=device)
        for step in range(length):
            observations[step] = torch.as_tensor(np.asarray(obs), device=device)
            with torch.no_grad():
                action, log_prob, _, value = agent.action_and_value(observations[step].unsqueeze(0))
            action = action.item(); next_obs, reward, terminated, truncated, _ = env.step(action)
            mask = alias_mask = 0.0
            if vqvae is not None:
                with torch.no_grad():
                    pair = tuple(vqvae.codes(torch.as_tensor(np.asarray(frame), device=device).unsqueeze(0).float() / 255.0).item() for frame in (obs, next_obs))
                prior = transition_actions[pair]
                alias = any(previous != action for previous in prior)
                if pair[0] == pair[1] or alias:
                    reward -= args.mask_penalty; mask, alias_mask = 1.0, float(alias)
                prior.add(action)
            # Reward clipping is the conventional Atari PPO baseline; returns below remain unmodified.
            rewards[step] = np.clip(reward, -1.0, 1.0)
            actions[step], log_probs[step], values[step] = action, log_prob, value
            dones[step] = float(terminated or truncated)
            recent_masks.append(mask); recent_aliases.append(alias_mask)
            episode_return += float(reward + (args.mask_penalty if mask else 0.0)); episode_length += 1
            obs = next_obs
            if terminated or truncated:
                global_step = update * args.rollout_steps + step
                episode_writer.writerow({"global_step": global_step, "episodic_return": episode_return, "episodic_length": episode_length, "epsilon": 0.0})
                episodes.flush(); obs, _ = env.reset(); episode_return, episode_length = 0.0, 0

        with torch.no_grad():
            next_value = agent.value(torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)).item()
        advantages = torch.zeros_like(rewards); last_advantage = 0.0
        for step in reversed(range(length)):
            next_nonterminal = 1.0 - dones[step]
            following_value = next_value if step == length - 1 else values[step + 1]
            delta = rewards[step] + args.gamma * following_value * next_nonterminal - values[step]
            last_advantage = delta + args.gamma * args.gae_lambda * next_nonterminal * last_advantage
            advantages[step] = last_advantage
        returns = advantages + values
        batch_indices = np.arange(length)
        for _ in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, length, args.minibatch_size):
                mb = batch_indices[start:start + args.minibatch_size]
                _, new_log_prob, entropy, new_value = agent.action_and_value(observations[mb], actions[mb])
                ratio = (new_log_prob - log_probs[mb]).exp()
                normalized_advantage = (advantages[mb] - advantages[mb].mean()) / (advantages[mb].std(unbiased=False) + 1e-8)
                policy_loss = torch.maximum(-normalized_advantage * ratio, -normalized_advantage * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                value_loss = F.mse_loss(new_value, returns[mb])
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy.mean()
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm); optimizer.step()
                last_stats = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.mean().item()}
        global_step = min((update + 1) * args.rollout_steps, args.total_timesteps)
        metric_writer.writerow({"global_step": global_step, **last_stats,
                                "mask_penalty_frequency": np.mean(recent_masks) if recent_masks else 0.0,
                                "transition_alias_penalty_frequency": np.mean(recent_aliases) if recent_aliases else 0.0})
        metrics.flush()
    episodes.close(); metrics.close(); env.close()


if __name__ == "__main__":
    train(parse_args())
