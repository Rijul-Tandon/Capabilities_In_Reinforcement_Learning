"""Double DQN for classic-control tasks (Acrobot, CartPole, MountainCar, LunarLander) with a VQ-VAE self-loop
penalty: a transition whose discrete code does not change costs --mask-penalty.  --agent dqn_baseline runs the
same Double DQN without VQ-VAE and penalty."""

import argparse
import csv
import json
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Double DQN with VQ-VAE self-loop penalty")

    # Experiment
    parser.add_argument("--env-id", type=str, default="Acrobot-v1")
    parser.add_argument("--agent", type=str, default="dqn_vqvae_masked", choices=["dqn_baseline", "dqn_vqvae_masked"])
    parser.add_argument("--exp-name", type=str, default=None)       # label of the run; defaults to the agent name
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--save-model", action="store_true", default=False)     # also write q_net.pt and vqvae_model.pt

    # Double DQN
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden-size", type=int, default=128)                    # units in each of the two hidden layers
    parser.add_argument("--buffer-size", type=int, default=50000)                  # replay buffer capacity, in transitions
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=5000)               # steps collected before the first update
    parser.add_argument("--train-frequency", type=int, default=4)                  # one gradient step every this many env steps
    parser.add_argument("--target-network-frequency", type=int, default=500)       # copy q_net -> target_net every this many env steps

    # Exploration: epsilon decays linearly from start to end over the first exploration_fraction of training
    parser.add_argument("--start-e", dest="start_epsilon", type=float, default=1.0)
    parser.add_argument("--end-e", dest="end_epsilon", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)

    # VQ-VAE and penalty (dqn_vqvae_masked only)
    parser.add_argument("--mask-penalty", type=float, default=1.0)       # subtracted from the reward of a self-loop transition
    parser.add_argument("--dataset-size", type=int, default=20000)       # random-policy states the VQ-VAE is trained on
    parser.add_argument("--vqvae-epochs", type=int, default=50)
    parser.add_argument("--vqvae-hidden-dim", type=int, default=128)
    parser.add_argument("--num-embeddings", type=int, default=256)       # codebook size K = number of discrete states
    parser.add_argument("--embedding-dim", type=int, default=8)          # dimension of each codebook vector

    return parser.parse_args()


VQVAE_BATCH_SIZE = 256
VQVAE_LEARNING_RATE = 1e-3
COMMITMENT_COST = 0.25         # beta: weight of the commitment term of the VQ loss
MAX_GRADIENT_NORM = 10.0       # the DQN gradient is clipped to this norm
LOG_INTERVAL = 1000            # steps between two rows of metrics.csv


# ============================================================================
# VQ-VAE: continuous state -> discrete code
# ============================================================================

class VectorQuantizer(nn.Module):
    """Replaces each encoder output by its nearest codebook vector."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, encoder_outputs):
        """z_e (batch, embedding_dim) -> quantised z_q, the VQ loss, the code index of each row, the codebook perplexity."""
        codebook = self.embedding.weight                                    # (num_embeddings, embedding_dim)
        # ||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 z_e . e_k for every code k: (batch, num_embeddings)
        distances = (
            torch.sum(encoder_outputs ** 2, dim=1, keepdim=True)
            + torch.sum(codebook ** 2, dim=1)
            - 2 * torch.matmul(encoder_outputs, codebook.t())
        )
        code_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(code_indices)

        commitment_loss = F.mse_loss(quantized.detach(), encoder_outputs)      # keeps z_e close to its code
        codebook_loss = F.mse_loss(quantized, encoder_outputs.detach())        # pulls the chosen codes towards z_e
        vq_loss = codebook_loss + COMMITMENT_COST * commitment_loss

        # Straight-through estimator: the forward value is z_q, the gradient passes to z_e unchanged.
        quantized = encoder_outputs + (quantized - encoder_outputs).detach()

        # Perplexity = exp(entropy of code usage in this batch): the effective number of codes in use.
        code_usage = F.one_hot(code_indices, self.num_embeddings).float().mean(dim=0)      # (num_embeddings,)
        perplexity = torch.exp(-torch.sum(code_usage * torch.log(code_usage + 1e-10)))

        return quantized, vq_loss, code_indices, perplexity


class VQVAEStateDiscretizer(nn.Module):
    """MLP autoencoder with a vector-quantiser bottleneck: state -> z_e -> nearest code z_q -> reconstructed state."""

    def __init__(self, state_dim, hidden_dim, embedding_dim, num_embeddings):
        super().__init__()
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

    def encode_to_discrete(self, states):
        """The discrete code of each state: (batch, state_dim) -> (batch,)."""
        _, _, code_indices, _ = self.vq_layer(self.encoder(states))
        return code_indices


# ============================================================================
# Q-NETWORK AND REPLAY BUFFER
# ============================================================================

class QNetwork(nn.Module):
    """MLP that maps a state to one Q-value per action: (batch, observation_dim) -> (batch, num_actions)."""

    def __init__(self, observation_dim, num_actions, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, observations):
        return self.network(observations)


class ReplayBuffer:
    """Circular buffer of transitions (s, a, r, s', terminated), sampled uniformly at random."""

    def __init__(self, capacity, observation_shape, device):
        self.capacity = capacity
        self.device = device
        self.position = 0       # where the next transition is written
        self.size = 0           # number of transitions stored so far (at most capacity)
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)

    def add(self, observation, action, reward, next_observation, terminated):
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.terminated[self.position] = terminated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        buffer_indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.observations[buffer_indices], device=self.device),
            torch.tensor(self.actions[buffer_indices], device=self.device),
            torch.tensor(self.rewards[buffer_indices], device=self.device),
            torch.tensor(self.next_observations[buffer_indices], device=self.device),
            torch.tensor(self.terminated[buffer_indices], device=self.device),
        )


# ============================================================================
# TRAINING
# ============================================================================

def episode_reached_goal(env_id, episode_return, episode_length):
    """1.0 if the episode counts as solved, else 0.0.  Each environment has its own criterion:

      MountainCar   the flag was reached before the 200-step time limit
      CartPole      return of at least 195
      Acrobot       the bar was reached before the 500-step time limit (return above -500)
      others        positive return (for LunarLander: landed, fully or partially)
    """
    if "MountainCar" in env_id:
        return 1.0 if episode_length < 200 else 0.0
    if "CartPole" in env_id:
        return 1.0 if episode_return >= 195.0 else 0.0
    if "Acrobot" in env_id:
        return 1.0 if episode_return > -500.0 else 0.0
    return 1.0 if episode_return > 0.0 else 0.0


def train(args):
    exp_name = args.exp_name or args.agent
    run_name = f"{args.env_id}__{exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), exp_name=exp_name, run_name=run_name)
    with open(run_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(args.env_id)
    env.action_space.seed(args.seed)    # makes the random actions repeatable for a given seed
    observation_shape = env.observation_space.shape
    observation_dim = int(np.prod(observation_shape))
    num_actions = env.action_space.n
    print(f"[{exp_name}] env={args.env_id} | State Dim={observation_dim} | Num Actions={num_actions} | Device={device}")

    # ---- VQ-VAE: train once on random-policy states, then freeze ----
    vqvae = None
    if args.agent == "dqn_vqvae_masked":
        print(f"[{exp_name}] Pre-training VQ-VAE for {args.vqvae_epochs} epochs on {args.dataset_size} transitions...")
        vqvae = VQVAEStateDiscretizer(observation_dim, args.vqvae_hidden_dim, args.embedding_dim, args.num_embeddings).to(device)

        dataset_states = []
        state, _ = env.reset(seed=args.seed)
        for _ in range(args.dataset_size):
            next_state, _, terminated, truncated, _ = env.step(env.action_space.sample())
            dataset_states.append(state)
            state = next_state
            if terminated or truncated:
                state, _ = env.reset()
        dataset_states = torch.Tensor(np.array(dataset_states, dtype=np.float32)).to(device)       # (dataset_size, state_dim)
        data_loader = DataLoader(TensorDataset(dataset_states), batch_size=VQVAE_BATCH_SIZE, shuffle=True)

        vqvae_optimizer = torch.optim.Adam(vqvae.parameters(), lr=VQVAE_LEARNING_RATE)
        vqvae.train()
        reconstruction_loss_history = []
        vq_loss_history = []
        perplexity_history = []
        for epoch in range(1, args.vqvae_epochs + 1):
            epoch_reconstruction_loss = 0.0
            epoch_vq_loss = 0.0
            epoch_perplexity = 0.0
            for (batch_states,) in data_loader:
                vqvae_optimizer.zero_grad()
                encoder_outputs = vqvae.encoder(batch_states)                                   # z_e
                quantized, vq_loss, _, perplexity = vqvae.vq_layer(encoder_outputs)             # z_q
                reconstructed_states = vqvae.decoder(quantized)
                reconstruction_loss = F.mse_loss(reconstructed_states, batch_states)
                vqvae_loss = reconstruction_loss + vq_loss
                vqvae_loss.backward()
                vqvae_optimizer.step()
                epoch_reconstruction_loss += reconstruction_loss.item()
                epoch_vq_loss += vq_loss.item()
                epoch_perplexity += perplexity.item()

            num_batches = len(data_loader)
            reconstruction_loss_history.append(epoch_reconstruction_loss / num_batches)
            vq_loss_history.append(epoch_vq_loss / num_batches)
            perplexity_history.append(epoch_perplexity / num_batches)
            if epoch % 10 == 0 or epoch == 1:
                print(f"VQ-VAE Epoch [{epoch:02d}/{args.vqvae_epochs}] | Recon MSE: {reconstruction_loss_history[-1]:.5f} | "
                      f"VQ Loss: {vq_loss_history[-1]:.5f} | Perplexity: {perplexity_history[-1]:.2f}/{args.num_embeddings}")
        vqvae.eval()

        with open(run_dir / "vqvae_diagnostics.json", "w") as diagnostics_file:
            json.dump({"recon_loss": reconstruction_loss_history, "vq_loss": vq_loss_history, "perplexity": perplexity_history},
                      diagnostics_file, indent=4)

    # ---- networks ----
    q_net = QNetwork(observation_dim, num_actions, args.hidden_size).to(device)
    target_net = QNetwork(observation_dim, num_actions, args.hidden_size).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size, observation_shape, device)

    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metric_file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(episode_file, fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "epsilon"])
    metric_writer = csv.DictWriter(metric_file, fieldnames=["global_step", "epsilon", "td_loss", "q_value", "max_q", "mask_penalty_frequency"])
    episode_writer.writeheader()
    metric_writer.writeheader()

    observation, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    recent_goals = deque(maxlen=100)            # 1 / 0 per episode: solved
    recent_penalised = deque(maxlen=1000)       # 1 / 0 per step: the self-loop penalty was applied
    last_td_loss = np.nan
    last_mean_q = 0.0
    last_max_q = 0.0
    decay_steps = args.exploration_fraction * args.total_timesteps

    progress_bar = tqdm(range(args.total_timesteps), desc=f"{args.env_id} [{exp_name}]")
    for global_step in progress_bar:

        # ---- choose an action (epsilon-greedy, linear decay) ----
        epsilon = max((args.end_epsilon - args.start_epsilon) / decay_steps * global_step + args.start_epsilon, args.end_epsilon)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))    # (1, num_actions)
                action = int(torch.argmax(q_values, dim=1).item())

        next_observation, env_reward, terminated, truncated, _ = env.step(action)

        # ---- penalty: the action left the agent in the same discrete state ----
        shaped_reward = float(env_reward)
        penalised = 0.0
        if vqvae is not None:
            with torch.no_grad():
                code = vqvae.encode_to_discrete(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)).item()
                next_code = vqvae.encode_to_discrete(torch.tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)).item()
            if code == next_code:
                shaped_reward -= args.mask_penalty
                penalised = 1.0
        recent_penalised.append(penalised)

        # `terminated` is stored, not `terminated or truncated`: a time-limit cut still bootstraps from s'.
        replay_buffer.add(observation, action, shaped_reward, next_observation, float(terminated))
        episode_return += float(env_reward)     # the logs report the environment reward, never the shaped one
        episode_length += 1
        observation = next_observation

        if terminated or truncated:
            goal_reached = episode_reached_goal(args.env_id, episode_return, episode_length)
            recent_goals.append(goal_reached)
            episode_writer.writerow({
                "global_step": global_step,
                "episodic_return": episode_return,
                "episodic_length": episode_length,
                "goal_reached": goal_reached,
                "epsilon": epsilon,
            })
            episode_file.flush()
            progress_bar.set_postfix({
                "ret": f"{episode_return:.1f}",
                "goal%": f"{float(np.mean(recent_goals)):.0%}",
                "eps": f"{epsilon:.2f}",
                "loss": f"{last_td_loss:.4f}" if not np.isnan(last_td_loss) else "N/A",
            })
            observation, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        # ---- one Double DQN gradient step ----
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_terminated = replay_buffer.sample(args.batch_size)
            with torch.no_grad():
                # the online network selects a*, the target network evaluates it
                best_next_actions = q_net(batch_next_observations).argmax(dim=1, keepdim=True)                  # (batch, 1)
                next_q_values = target_net(batch_next_observations).gather(1, best_next_actions).squeeze(1)     # (batch,)
                td_target = batch_rewards + args.gamma * next_q_values * (1.0 - batch_terminated)
            # Q(s, a) of the actions that were taken: (batch, num_actions) -> (batch,)
            q_values_taken = q_net(batch_observations).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            td_loss = F.mse_loss(q_values_taken, td_target)
            optimizer.zero_grad()
            td_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()
            last_td_loss = float(td_loss.item())
            last_mean_q = float(q_values_taken.mean().item())
            last_max_q = float(q_values_taken.max().item())

        if global_step % args.target_network_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        if global_step % LOG_INTERVAL == 0:
            metric_writer.writerow({
                "global_step": global_step,
                "epsilon": epsilon,
                "td_loss": last_td_loss,
                "q_value": last_mean_q,
                "max_q": last_max_q,
                "mask_penalty_frequency": float(np.mean(recent_penalised)) if recent_penalised else 0.0,
            })
            metric_file.flush()

    if args.save_model:
        torch.save(q_net.state_dict(), run_dir / "q_net.pt")
        if vqvae is not None:
            torch.save(vqvae.state_dict(), run_dir / "vqvae_model.pt")
    episode_file.close()
    metric_file.close()
    env.close()
    print(f"Training completed successfully. Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    train(parse_args())
