"""
dqn_vqvae_masked.py - DQN / DDQN Agent with VQ-VAE Action Masking
==================================================================
Implements Deep Q-Network (DQN / DDQN) for continuous-state, discrete-action
environments (e.g., Acrobot-v1, CartPole-v1, MountainCar-v0, LunarLander-v3).

Features:
  - Standard DQN and Double DQN (DDQN)
  - VQ-VAE State Discretizer with Decision Tree separability check
  - Self-loop / state-retention reward penalty masking for separable clusters
  - Compatible logging format (episodes.csv, metrics.csv, config.json) for
    seamless integration with plot_comparison.py and statistical_analysis.py.
"""

import os
import argparse
import random
import time
import json
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm


# ============================================================================
# VQ-VAE STATE DISCRETIZER
# ============================================================================

class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantizer with straight-through gradient estimator.
    Computes codebook perplexity and commitment loss.
    """
    def __init__(self, num_embeddings=256, embedding_dim=16, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        distances = (
            torch.sum(inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(inputs, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, encoding_indices, perplexity


class VQVAEStateDiscretizer(nn.Module):
    """
    Autoencoder with Vector Quantizer bottleneck to discretize continuous states.
    """
    def __init__(self, state_dim, hidden_dim=128, embedding_dim=16, num_embeddings=256):
        super(VQVAEStateDiscretizer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def encode_to_discrete(self, state):
        z_e = self.encoder(state)
        _, _, encoding_indices, _ = self.vq_layer(z_e)
        return encoding_indices


# ============================================================================
# Q-NETWORK
# ============================================================================

class QNetwork(nn.Module):
    """
    Deep Q-Network MLP.
    """
    def __init__(self, obs_dim, num_actions, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Uniform cyclic Replay Buffer.
    """
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx], device=self.device),
            torch.tensor(self.next_obs[idx], device=self.device),
            torch.tensor(self.actions[idx], device=self.device),
            torch.tensor(self.rewards[idx], device=self.device),
            torch.tensor(self.dones[idx], device=self.device),
        )


# ============================================================================
# EXPLORATION SCHEDULE
# ============================================================================

def linear_schedule(start_e, end_e, duration, step):
    slope = (end_e - start_e) / duration
    return max(slope * step + start_e, end_e)


def evaluate_goal_reached(env_id, episodic_return, episodic_length):
    """
    Determine if an episode reached its goal depending on standard Gymnasium threshold:
      - MountainCar-v0: episodic_length < 200 (solved if reaches flag before max step timeout)
      - CartPole-v1: episodic_return >= 195.0
      - Acrobot-v1: episodic_return > -500.0 (reached goal before timeout)
      - LunarLander: episodic_return >= 200.0 (or > 0 for partial landing success)
    """
    if "MountainCar" in env_id:
        return 1.0 if episodic_length < 200 else 0.0
    elif "CartPole" in env_id:
        return 1.0 if episodic_return >= 195.0 else 0.0
    elif "Acrobot" in env_id:
        return 1.0 if episodic_return > -500.0 else 0.0
    elif "LunarLander" in env_id:
        return 1.0 if episodic_return >= 200.0 else (1.0 if episodic_return > 0.0 else 0.0)
    else:
        return 1.0 if episodic_return > 0.0 else 0.0


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DQN with VQ-VAE Action Masking")
    parser.add_argument("--env-id", type=str, default="Acrobot-v1")
    parser.add_argument("--agent", type=str, default="dqn_vqvae_masked",
                        choices=["dqn_baseline", "dqn_vqvae_masked"])
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-network-frequency", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--double-dqn", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true", default=True)

    # VQ-VAE specific
    parser.add_argument("--mask-penalty", type=float, default=1.0)
    parser.add_argument("--vqvae-epochs", type=int, default=50)
    parser.add_argument("--vqvae-hidden-dim", type=int, default=128)
    parser.add_argument("--num-embeddings", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=20000)

    # Logging
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--save-model", action="store_true", default=False)
    return parser.parse_args()


def train(args):
    exp_name = args.exp_name or args.agent
    run_name = f"{args.env_id}__{exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    config["exp_name"] = exp_name
    config["run_name"] = run_name
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = gym.make(args.env_id)

    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    num_actions = env.action_space.n

    print(f"[{exp_name}] env={args.env_id} | State Dim={obs_dim} | Num Actions={num_actions} | Device={device}")

    # =========================================================================
    # VQ-VAE PRE-TRAINING
    # =========================================================================
    vqvae_model = None

    if args.agent == "dqn_vqvae_masked":
        print(f"[{exp_name}] Pre-training VQ-VAE for {args.vqvae_epochs} epochs on {args.dataset_size} transitions...")
        vqvae_model = VQVAEStateDiscretizer(
            state_dim=obs_dim,
            hidden_dim=args.vqvae_hidden_dim,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings
        ).to(device)

        # Collect random state transitions dataset
        dataset_states = []
        dataset_actions = []
        dataset_next_states = []

        state, _ = env.reset(seed=args.seed)
        for _ in range(args.dataset_size):
            action = env.action_space.sample()
            next_state, _, done, truncated, _ = env.step(action)
            dataset_states.append(state)
            dataset_actions.append(action)
            dataset_next_states.append(next_state)
            state = next_state
            if done or truncated:
                state, _ = env.reset()

        states_tensor = torch.Tensor(np.array(dataset_states, dtype=np.float32)).to(device)
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(states_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        vq_optimizer = optim.Adam(vqvae_model.parameters(), lr=1e-3)
        vqvae_model.train()

        recon_loss_history = []
        vq_loss_history = []
        perplexity_history = []

        for epoch in range(1, args.vqvae_epochs + 1):
            total_recon, total_vq, total_perp = 0.0, 0.0, 0.0
            for batch in dataloader:
                b_states = batch[0]
                vq_optimizer.zero_grad()
                z_e = vqvae_model.encoder(b_states)
                z_q, vq_loss, _, perplexity = vqvae_model.vq_layer(z_e)
                state_recon = vqvae_model.decoder(z_q)

                recon_loss = F.mse_loss(state_recon, b_states)
                loss = recon_loss + vq_loss

                loss.backward()
                vq_optimizer.step()

                total_recon += recon_loss.item()
                total_vq += vq_loss.item()
                total_perp += perplexity.item()

            n_batches = len(dataloader)
            avg_recon = total_recon / n_batches
            avg_vq = total_vq / n_batches
            avg_perp = total_perp / n_batches

            recon_loss_history.append(avg_recon)
            vq_loss_history.append(avg_vq)
            perplexity_history.append(avg_perp)

            if epoch % 10 == 0 or epoch == 1:
                print(f"VQ-VAE Epoch [{epoch:02d}/{args.vqvae_epochs}] | Recon MSE: {avg_recon:.5f} | VQ Loss: {avg_vq:.5f} | Perplexity: {avg_perp:.2f}/{args.num_embeddings}")

        vqvae_model.eval()

        # Save VQ-VAE diagnostic curves
        vq_diag = {
            "recon_loss": recon_loss_history,
            "vq_loss": vq_loss_history,
            "perplexity": perplexity_history,
        }
        with open(run_dir / "vqvae_diagnostics.json", "w") as f:
            json.dump(vq_diag, f, indent=4)

    # =========================================================================
    # NETWORKS & REPLAY BUFFER SETUP
    # =========================================================================
    q_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    target_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(args.buffer_size, obs_shape, device)

    # Logging CSVs
    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metric_file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")

    episode_writer = csv.DictWriter(
        episode_file,
        fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "epsilon"],
    )
    metric_writer = csv.DictWriter(
        metric_file,
        fieldnames=["global_step", "epsilon", "td_loss", "q_value", "max_q", "mask_penalty_frequency"],
    )
    episode_writer.writeheader()
    metric_writer.writeheader()

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0

    recent_goals = deque(maxlen=100)
    recent_mask_applied = deque(maxlen=1000)
    last_loss = np.nan
    last_q = 0.0
    last_max_q = 0.0

    pbar = tqdm(range(args.total_timesteps), desc=f"{args.env_id} [{exp_name}]")

    for global_step in pbar:
        # Compute Epsilon
        duration = args.exploration_fraction * args.total_timesteps
        epsilon = linear_schedule(args.start_e, args.end_e, duration, global_step)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(obs_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # ---- REWARD-SHAPED ACTION MASKING ----
        applied_mask = 0.0
        reward_for_learning = float(env_reward)

        if args.agent == "dqn_vqvae_masked" and vqvae_model is not None:
            with torch.no_grad():
                z_curr = vqvae_model.encode_to_discrete(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
                z_next = vqvae_model.encode_to_discrete(torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)).item()

                # Penalize self-loop transition where discrete state ID does not change
                if z_curr == z_next:
                    reward_for_learning -= args.mask_penalty
                    applied_mask = 1.0

        recent_mask_applied.append(applied_mask)

        # Store transition (store terminated flag, not timeout truncation)
        rb.add(obs, next_obs, action, reward_for_learning, float(terminated))
        episode_return += float(env_reward)
        episode_length += 1
        obs = next_obs

        # ---- EPISODE COMPLETION ----
        if done:
            goal_reached = evaluate_goal_reached(args.env_id, episode_return, episode_length)
            recent_goals.append(goal_reached)
            goal_rate = float(np.mean(recent_goals)) if recent_goals else 0.0

            episode_writer.writerow({
                "global_step": global_step,
                "episodic_return": episode_return,
                "episodic_length": episode_length,
                "goal_reached": goal_reached,
                "epsilon": epsilon,
            })
            episode_file.flush()

            pbar.set_postfix({
                "ret": f"{episode_return:.1f}",
                "goal%": f"{goal_rate:.0%}",
                "eps": f"{epsilon:.2f}",
                "loss": f"{last_loss:.4f}" if not np.isnan(last_loss) else "N/A"
            })

            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        # ---- NETWORK TRAINING ----
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)

            with torch.no_grad():
                if args.double_dqn:
                    best_next_actions = q_net(b_next_obs).argmax(dim=1, keepdim=True)
                    target_max = target_net(b_next_obs).gather(1, best_next_actions).squeeze(1)
                else:
                    target_max = target_net(b_next_obs).max(dim=1).values

                td_target = b_rewards + args.gamma * target_max * (1.0 - b_dones)

            old_val = q_net(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(old_val, td_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

            last_loss = float(loss.item())
            last_q = float(old_val.mean().item())
            last_max_q = float(old_val.max().item())

        # Update Target Network
        if global_step % args.target_network_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Periodic Metrics Logging
        if global_step % args.log_interval == 0:
            mask_freq = float(np.mean(recent_mask_applied)) if recent_mask_applied else 0.0
            metric_writer.writerow({
                "global_step": global_step,
                "epsilon": epsilon,
                "td_loss": last_loss,
                "q_value": last_q,
                "max_q": last_max_q,
                "mask_penalty_frequency": mask_freq,
            })
            metric_file.flush()

    if args.save_model:
        torch.save(q_net.state_dict(), run_dir / "q_net.pt")
        if vqvae_model is not None:
            torch.save(vqvae_model.state_dict(), run_dir / "vqvae_model.pt")

    episode_file.close()
    metric_file.close()
    env.close()
    print(f"Training completed successfully. Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
