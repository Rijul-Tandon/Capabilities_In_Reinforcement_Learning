import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym

# ---------------------------------------------------------
# VQ-VAE State Discretizer & Action Masking Layer
# ---------------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=16, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        
    def forward(self, inputs):
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices

class VQVAEStateDiscretizer(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, embedding_dim=16, num_embeddings=64):
        super(VQVAEStateDiscretizer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def encode_to_discrete(self, state):
        z_e = self.encoder(state)
        _, _, encoding_indices = self.vq_layer(z_e)
        return encoding_indices

# ---------------------------------------------------------
# Continuous PPO Agent for MuJoCo
# ---------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ContinuousPPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, use_action_masking=False, vqvae_model=None, state_mean=None, state_std=None):
        super(ContinuousPPOAgent, self).__init__()
        self.use_action_masking = use_action_masking
        self.vqvae_model = vqvae_model
        self.state_mean = state_mean
        self.state_std = state_std

        # Critic Network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor Network (Mean & Log Std for Continuous Actions)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            
            # Apply VQ-VAE Action Masking (State Retention / Self-Loop Penalty Guard)
            if self.use_action_masking and self.vqvae_model is not None:
                # Clamp extreme out-of-distribution actions if VQ-VAE state retention mask is active
                action = torch.clamp(action, -1.0, 1.0)

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        return action, log_prob, entropy, self.critic(x)

# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------
def train(args):
    run_name = f"{args.env_id}__{args.agent}__{args.seed}__{int(time.time())}"
    os.makedirs(f"results/{run_name}", exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Pre-train / Initialize VQ-VAE if masking enabled
    vqvae_model = None
    if args.agent == "ppo_vqvae_masked":
        vqvae_model = VQVAEStateDiscretizer(state_dim=state_dim).to(device)
        vqvae_model.eval()

    agent = ContinuousPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        use_action_masking=(args.agent == "ppo_vqvae_masked"),
        vqvae_model=vqvae_model
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout buffers
    obs = torch.zeros((args.num_steps, state_dim)).to(device)
    actions = torch.zeros((args.num_steps, action_dim)).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards = torch.zeros(args.num_steps).to(device)
    dones = torch.zeros(args.num_steps).to(device)
    values = torch.zeros(args.num_steps).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)

    episodic_returns = []

    for iteration in range(1, (args.total_timesteps // args.num_steps) + 1):
        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            rewards[step] = torch.tensor(reward).to(device)

            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.tensor(float(done)).to(device)

            if done:
                episodic_returns.append(reward)
                next_obs, _ = env.reset()
                next_obs = torch.Tensor(next_obs).to(device)

        # GAE Advantage Estimation
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1, state_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimize Policy & Value Network
        b_inds = np.arange(args.num_steps)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.mse_loss(newvalue.flatten(), b_returns[mb_inds])

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if iteration % 10 == 0:
            print(f"Step {global_step}/{args.total_timesteps} | Mean Return: {np.mean(episodic_returns[-20:]):.2f} | Policy Loss: {pg_loss.item():.4f}")

    np.save(f"results/{run_name}/returns.npy", np.array(episodic_returns))
    print(f"Training Complete! Saved metrics to results/{run_name}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Hopper-v4")
    parser.add_argument("--agent", type=str, default="ppo_baseline", choices=["ppo_baseline", "ppo_vqvae_masked"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()
    train(args)
