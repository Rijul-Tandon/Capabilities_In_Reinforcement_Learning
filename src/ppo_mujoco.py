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

        # Compute Codebook Perplexity
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, encoding_indices, perplexity

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
        _, _, encoding_indices, _ = self.vq_layer(z_e)
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
        action_logstd = torch.clamp(action_logstd, -20.0, 2.0)
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
    recon_loss_history, vq_loss_history, perplexity_history = [], [], []
    if args.agent == "ppo_vqvae_masked":
        print(f"Pre-training VQ-VAE for {args.vqvae_epochs} epochs on {args.env_id} trajectory dataset...")
        vqvae_model = VQVAEStateDiscretizer(
            state_dim=state_dim,
            hidden_dim=args.vqvae_hidden_dim,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings
        ).to(device)

        # Collect random state transitions dataset
        dataset_states = []
        state, _ = env.reset(seed=args.seed)
        for _ in range(20000):
            action = env.action_space.sample()
            next_state, _, done, truncated, _ = env.step(action)
            dataset_states.append(state)
            state = next_state
            if done or truncated:
                state, _ = env.reset()

        states_tensor = torch.Tensor(np.array(dataset_states)).to(device)
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(states_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        vq_optimizer = optim.Adam(vqvae_model.parameters(), lr=1e-3)
        vqvae_model.train()

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
        print("VQ-VAE Pre-training Complete!")

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

            # Sanitize observation array against unexpected NaNs
            next_obs_np = np.nan_to_num(next_obs_np, nan=0.0, posinf=1.0, neginf=-1.0)

            # Apply VQ-VAE State Retention Action Mask Penalty (Penalize Self-Loop Transitions)
            if agent.use_action_masking and agent.vqvae_model is not None:
                with torch.no_grad():
                    z_curr = agent.vqvae_model.encode_to_discrete(next_obs.unsqueeze(0))
                    z_next = agent.vqvae_model.encode_to_discrete(torch.Tensor(next_obs_np).unsqueeze(0).to(device))
                    # Apply substantial negative penalty when action causes a self-loop (same discrete state transition)
                    if z_curr.item() == z_next.item():
                        reward -= args.mask_penalty

            rewards[step] = torch.tensor(reward).to(device)

            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.tensor(float(done)).to(device)

            if done:
                episodic_returns.append(reward)
                next_obs, _ = env.reset()
                next_obs_np = np.nan_to_num(next_obs, nan=0.0, posinf=1.0, neginf=-1.0)
                next_obs = torch.Tensor(next_obs_np).to(device)

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

    # Generate VQ-VAE & Action Masking plots if VQ-VAE agent
    if args.agent == "ppo_vqvae_masked" and vqvae_model is not None:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(recon_loss_history, label='Reconstruction MSE Loss', color='tab:blue')
            axes[0].plot(vq_loss_history, label='VQ Loss', color='tab:orange')
            axes[0].set_title(f'VQ-VAE Loss Convergence ({args.env_id})')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(perplexity_history, label='Active Codebook Utilization (Perplexity)', color='tab:green')
            axes[1].axhline(y=args.num_embeddings, color='r', linestyle='--', label=f'Max Codebook Capacity ({args.num_embeddings})')
            axes[1].set_title('Codebook Perplexity over Training')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Perplexity')
            axes[1].legend()
            axes[1].grid(True)
            plt.tight_layout()
            plt.savefig(f"results/{run_name}/vqvae_training_diagnostics.png", dpi=300)
            plt.close()
            print(f"Saved VQ-VAE training plot to results/{run_name}/vqvae_training_diagnostics.png")
        except Exception as e:
            print(f"Could not save VQ-VAE diagnostic plots: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Hopper-v4")
    parser.add_argument("--agent", type=str, default="ppo_baseline", choices=["ppo_baseline", "ppo_vqvae_masked"])
    parser.add_argument("--mask-penalty", type=float, default=5.0, help="Heavy negative reward penalty applied when an unmasked action causes an unwanted discrete state change")
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

    # VQ-VAE Discretizer Hyperparameters
    parser.add_argument("--vqvae-epochs", type=int, default=50, help="Number of pre-training epochs for VQ-VAE state discretizer")
    parser.add_argument("--num-embeddings", type=int, default=64, help="Number of discrete state clusters in codebook")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Latent embedding dimension of codebook vectors")
    parser.add_argument("--vqvae-hidden-dim", type=int, default=128, help="Hidden dimension of VQ-VAE encoder/decoder networks")

    args = parser.parse_args()
    train(args)
