"""Standard Atari PPO baseline.

Clean PPO implementation with NO VQ-VAE components and using the environment's
*minimal* action space (typically 4–6 actions).  Mirror architecture of
``ppo_atari_vqvae_masked.py`` so that performance differences are attributable
solely to the action-masking mechanism.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm import tqdm

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)


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


# ─── network ────────────────────────────────────────────────────────────────
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class AtariPPO(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, 4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(), nn.Flatten(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, num_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs.float() / 255.0)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(obs)).squeeze(-1)

    def action_and_value(self, obs, action=None):
        feats = self._features(obs)
        dist = Categorical(logits=self.actor(feats))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(feats).squeeze(-1)


# ─── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard Atari PPO baseline (minimal action space)")
    parser.add_argument("--env-id", default="ALE/DemonAttack-v5")
    parser.add_argument("--agent", default="ppo_baseline_standard",
                        help="Label for this run (used in config.json / plots).")
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
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


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

    agent = AtariPPO(num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    episodes_fh = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metrics_fh = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(
        episodes_fh, fieldnames=["global_step", "episodic_return", "episodic_length", "epsilon"],
    )
    metric_writer = csv.DictWriter(
        metrics_fh, fieldnames=["global_step", "policy_loss", "value_loss", "entropy"],
    )
    episode_writer.writeheader()
    metric_writer.writeheader()

    obs = env.reset(seed=args.seed)[0]
    episode_return, episode_length = 0.0, 0
    last_stats: dict = {}
    updates = (args.total_timesteps + args.rollout_steps - 1) // args.rollout_steps

    for update in tqdm(range(updates), desc=f"{args.env_id} [{args.agent}]"):
        length = min(args.rollout_steps, args.total_timesteps - update * args.rollout_steps)
        observations = torch.zeros((length, *env.observation_space.shape), dtype=torch.uint8, device=device)
        actions = torch.zeros(length, dtype=torch.long, device=device)
        log_probs = torch.zeros(length, device=device)
        rewards = torch.zeros(length, device=device)
        dones = torch.zeros(length, device=device)
        values = torch.zeros(length, device=device)

        for step in range(length):
            observations[step] = torch.as_tensor(np.asarray(obs), device=device)
            with torch.no_grad():
                action, log_prob, _, value = agent.action_and_value(observations[step].unsqueeze(0))
            action = action.item()
            next_obs, reward, terminated, truncated, _ = env.step(action)

            rewards[step] = np.clip(reward, -1.0, 1.0)
            actions[step] = action
            log_probs[step] = log_prob
            values[step] = value
            dones[step] = float(terminated or truncated)

            episode_return += float(reward)
            episode_length += 1
            obs = next_obs

            if terminated or truncated:
                global_step = update * args.rollout_steps + step
                episode_writer.writerow({
                    "global_step": global_step,
                    "episodic_return": episode_return,
                    "episodic_length": episode_length,
                    "epsilon": 0.0,
                })
                episodes_fh.flush()
                obs, _ = env.reset()
                episode_return, episode_length = 0.0, 0

        # ── GAE ──
        with torch.no_grad():
            next_value = agent.value(
                torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
            ).item()

        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0
        for t in reversed(range(length)):
            nonterminal = 1.0 - dones[t]
            following_value = next_value if t == length - 1 else values[t + 1]
            delta = rewards[t] + args.gamma * following_value * nonterminal - values[t]
            last_advantage = delta + args.gamma * args.gae_lambda * nonterminal * last_advantage
            advantages[t] = last_advantage
        returns = advantages + values

        # ── PPO update ──
        batch_indices = np.arange(length)
        for _ in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, length, args.minibatch_size):
                mb = batch_indices[start : start + args.minibatch_size]
                _, new_log_prob, entropy, new_value = agent.action_and_value(
                    observations[mb], actions[mb],
                )
                ratio = (new_log_prob - log_probs[mb]).exp()
                norm_adv = (advantages[mb] - advantages[mb].mean()) / (advantages[mb].std(unbiased=False) + 1e-8)
                policy_loss = torch.maximum(
                    -norm_adv * ratio,
                    -norm_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                value_loss = F.mse_loss(new_value, returns[mb])
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                last_stats = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.mean().item(),
                }

        global_step = min((update + 1) * args.rollout_steps, args.total_timesteps)
        metric_writer.writerow({"global_step": global_step, **last_stats})
        metrics_fh.flush()

    episodes_fh.close()
    metrics_fh.close()
    env.close()
    print(f"Done. Results saved to {run_dir}")


if __name__ == "__main__":
    train(parse_args())
