"""Double DQN on Atari (full 18-action space) with a VQ-VAE penalty for redundant actions: a transition between
discrete codes costs --mask-penalty when the code does not change (self-loop) or when a different action has
already produced the same code transition (action alias).  --agent ddqn_baseline runs without VQ-VAE and penalty."""

import argparse
import csv
import json
import random
import time
from collections import defaultdict, deque
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
    parser = argparse.ArgumentParser(description="Atari Double DQN with VQ-VAE action-alias masking")

    # Experiment
    parser.add_argument("--env-id", default="ALE/DemonAttack-v5")
    parser.add_argument("--agent", choices=["ddqn_baseline", "ddqn_vqvae_masked"], default="ddqn_vqvae_masked")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--full-action-space", action=argparse.BooleanOptionalAction, default=True)   # all 18 Atari actions (required)

    # Double DQN
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=10000)                  # replay buffer capacity, in transitions
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-starts", type=int, default=10000)              # steps collected before the first update
    parser.add_argument("--train-frequency", type=int, default=4)                  # one gradient step every this many env steps
    parser.add_argument("--target-network-frequency", type=int, default=1000)      # copy q_net -> target_net every this many env steps

    # VQ-VAE and penalty (ddqn_vqvae_masked only)
    parser.add_argument("--mask-penalty", type=float, default=1.0)       # subtracted from the reward of a self-loop or action alias
    parser.add_argument("--dataset-size", type=int, default=5000)        # random-policy observations the VQ-VAE is trained on
    parser.add_argument("--vqvae-epochs", type=int, default=10)
    parser.add_argument("--num-embeddings", type=int, default=256)       # codebook size K = number of discrete states
    parser.add_argument("--embedding-dim", type=int, default=8)          # dimension of each codebook vector

    return parser.parse_args()


VQVAE_BATCH_SIZE = 32
VQVAE_LEARNING_RATE = 1e-3
COMMITMENT_COST = 0.25         # beta: weight of the commitment term of the VQ loss
MAX_GRADIENT_NORM = 10.0       # the DQN gradient is clipped to this norm
LOG_INTERVAL = 1000            # steps between two rows of metrics.csv


# ============================================================================
# VQ-VAE: observation -> one discrete code
# ============================================================================

class VectorQuantizer(nn.Module):
    """Replaces each encoder output by its nearest codebook vector."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, encoder_outputs):
        """z_e (batch, embedding_dim) -> quantised z_q (batch, embedding_dim), the VQ loss, the code index of each row."""
        codebook = self.embedding.weight                                    # (num_embeddings, embedding_dim)
        # ||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 z_e . e_k for every code k: (batch, num_embeddings)
        distances = encoder_outputs.square().sum(1, keepdim=True) + codebook.square().sum(1) - 2 * encoder_outputs @ codebook.T
        code_indices = distances.argmin(1)
        quantized = self.embedding(code_indices)

        codebook_loss = F.mse_loss(quantized, encoder_outputs.detach())        # pulls the chosen codes towards z_e
        commitment_loss = F.mse_loss(quantized.detach(), encoder_outputs)      # keeps z_e close to its code
        vq_loss = codebook_loss + COMMITMENT_COST * commitment_loss

        # Straight-through estimator: the forward value is z_q, the gradient passes to z_e unchanged.
        quantized = encoder_outputs + (quantized - encoder_outputs).detach()
        return quantized, vq_loss, code_indices


class AtariVQVAE(nn.Module):
    """Convolutional VQ-VAE for 4 stacked 84x84 frames scaled to [0, 1].  The whole observation gets ONE code."""

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),                # (4, 84, 84)  -> (32, 20, 20)
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),               # (32, 20, 20) -> (64, 9, 9)
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),               # (64, 9, 9)   -> (64, 7, 7)
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),            # average over the 7x7 map -> (64,)
            nn.Linear(64, embedding_dim),
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64 * 7 * 7), nn.ReLU(), nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, 3, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 8, 4), nn.Sigmoid(),
        )

    def forward(self, states):
        """(batch, 4, 84, 84) -> the reconstruction (same shape), the VQ loss, the code index of each state."""
        quantized, vq_loss, code_indices = self.vq(self.encoder(states))
        return self.decoder(quantized), vq_loss, code_indices

    def codes(self, states):
        """The discrete code of each state: (batch, 4, 84, 84) -> (batch,)."""
        _, _, code_indices = self.vq(self.encoder(states))
        return code_indices


# ============================================================================
# Q-NETWORK, REPLAY BUFFER, ENVIRONMENT
# ============================================================================

class QNetwork(nn.Module):
    """Nature-DQN network: (batch, 4, 84, 84) uint8 frames -> (batch, num_actions) Q-values."""

    def __init__(self, num_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
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


def make_env(env_id, seed, full_action_space=True):
    """Standard Atari preprocessing: frame skip 4, grayscale 84x84, 4 stacked frames; all 18 actions by default."""
    env = gym.make(env_id, frameskip=1, full_action_space=full_action_space)
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

    env = make_env(args.env_id, args.seed, args.full_action_space)
    num_actions = env.action_space.n
    if num_actions != 18:
        raise RuntimeError(f"Expected full Atari action space (18 actions), got {num_actions}.")
    print(f"{args.env_id}: actions={env.unwrapped.get_action_meanings()}")

    # ---- run folder and loggers ----
    run_dir = Path(args.results_dir) / f"{args.env_id.replace('/', '_')}__{args.agent}__{args.seed}__{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["exp_name"] = args.agent     # plot_comparison.py groups runs by this field
    with open(run_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)

    # ---- VQ-VAE: train once on random-policy observations, then freeze ----
    vqvae = None
    if args.agent == "ddqn_vqvae_masked":
        vqvae = AtariVQVAE(args.embedding_dim, args.num_embeddings).to(device)

        dataset_observations = []
        observation, _ = env.reset(seed=args.seed)
        for _ in range(args.dataset_size):
            dataset_observations.append(np.asarray(observation))
            observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                observation, _ = env.reset()
        data_loader = torch.utils.data.DataLoader(torch.as_tensor(np.asarray(dataset_observations)),
                                                  batch_size=VQVAE_BATCH_SIZE, shuffle=True)

        vqvae_optimizer = torch.optim.Adam(vqvae.parameters(), lr=VQVAE_LEARNING_RATE)
        for epoch in range(args.vqvae_epochs):
            epoch_losses = []
            for batch_states in data_loader:
                batch_states = batch_states.to(device).float() / 255.0      # (batch, 4, 84, 84) in [0, 1]
                reconstruction, vq_loss, _ = vqvae(batch_states)
                vqvae_loss = F.mse_loss(reconstruction, batch_states) + vq_loss
                vqvae_optimizer.zero_grad()
                vqvae_loss.backward()
                vqvae_optimizer.step()
                epoch_losses.append(vqvae_loss.item())
            print(f"VQ-VAE epoch {epoch + 1}/{args.vqvae_epochs}: loss={np.mean(epoch_losses):.5f}")
        vqvae.eval()

    # ---- networks ----
    q_net = QNetwork(num_actions).to(device)
    target_net = QNetwork(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space.shape, device)

    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metric_file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(episode_file, fieldnames=["global_step", "episodic_return", "episodic_length", "epsilon"])
    metric_writer = csv.DictWriter(metric_file, fieldnames=["global_step", "epsilon", "td_loss", "mask_penalty_frequency",
                                                           "transition_alias_penalty_frequency"])
    episode_writer.writeheader()
    metric_writer.writeheader()

    # actions_that_produced[(code, next_code)] = set of actions that have produced this code transition so far
    actions_that_produced = defaultdict(set)
    recent_penalised = deque(maxlen=1000)           # 1 / 0 per step: a penalty was applied
    recent_alias_penalised = deque(maxlen=1000)     # 1 / 0 per step: the penalty was (also) for an action alias
    observation, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    last_td_loss = np.nan

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

        # ---- penalty: was this action redundant in terms of discrete codes? ----
        shaped_reward = float(reward)
        penalised = 0.0
        alias_penalised = 0.0
        if vqvae is not None:
            with torch.no_grad():
                code = vqvae.codes(torch.as_tensor(np.asarray(observation), device=device).unsqueeze(0).float() / 255.0).item()
                next_code = vqvae.codes(torch.as_tensor(np.asarray(next_observation), device=device).unsqueeze(0).float() / 255.0).item()
            previous_actions = actions_that_produced[(code, next_code)]
            is_self_loop = code == next_code
            is_action_alias = any(previous_action != action for previous_action in previous_actions)
            if is_self_loop or is_action_alias:
                shaped_reward -= args.mask_penalty
                penalised = 1.0
                alias_penalised = float(is_action_alias)
            previous_actions.add(action)
        recent_penalised.append(penalised)
        recent_alias_penalised.append(alias_penalised)

        # `terminated` is stored, not `terminated or truncated`: a time-limit cut still bootstraps from s'.
        replay_buffer.add(observation, action, shaped_reward, next_observation, float(terminated))
        observation = next_observation
        episode_return += float(reward)     # the logs report the environment reward, never the shaped one
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
                td_target = batch_rewards + args.gamma * next_q_values * (1 - batch_terminated)
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
            metric_writer.writerow({
                "global_step": global_step,
                "epsilon": epsilon,
                "td_loss": last_td_loss,
                "mask_penalty_frequency": np.mean(recent_penalised) if recent_penalised else 0.0,
                "transition_alias_penalty_frequency": np.mean(recent_alias_penalised) if recent_alias_penalised else 0.0,
            })
            metric_file.flush()

    episode_file.close()
    metric_file.close()
    env.close()


if __name__ == "__main__":
    train(parse_args())
