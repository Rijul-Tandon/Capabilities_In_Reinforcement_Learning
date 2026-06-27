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
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


MUJOCO_ENV_IDS = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "HumanoidStandup-v4",
    "InvertedDoublePendulum-v4",
    "InvertedPendulum-v4",
    "Pusher-v4",
    "Reacher-v4",
    "Swimmer-v4",
    "Walker2d-v4",
]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(env_id, seed, idx, capture_video=False, run_name=""):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def get_reward_threshold(envs):
    base_env = envs.envs[0] if hasattr(envs, "envs") and envs.envs else None
    spec = getattr(base_env, "spec", None)
    if spec is not None and getattr(spec, "reward_threshold", None) is not None:
        return float(spec.reward_threshold)
    return None


def episode_success(episode_return, reward_threshold):
    if reward_threshold is None:
        return float(episode_return) > 0.0
    return float(episode_return) >= reward_threshold


def transition_penalty(obs, next_obs, threshold, penalty_value):
    obs_arr = np.asarray(obs, dtype=np.float32)
    next_obs_arr = np.asarray(next_obs, dtype=np.float32)
    delta = np.linalg.norm(next_obs_arr - obs_arr, axis=1)
    penalties = np.where(delta <= threshold, penalty_value, 0.0)
    return penalties.astype(np.float32)


class Agent(nn.Module):
    def __init__(self, envs, hidden_size):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32)
        action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def get_value(self, x):
        return self.critic(x).flatten()

    def get_action_and_value(self, x, latent_action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        if latent_action is None:
            latent_action = dist.rsample()
        squashed = torch.tanh(latent_action)
        action = squashed * self.action_scale + self.action_bias
        log_prob = dist.log_prob(latent_action)
        log_prob -= torch.log(self.action_scale * (1 - squashed.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        value = self.get_value(x)
        return action, log_prob, entropy, value, latent_action


def parse_args(default_exp_name, use_shaping):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=default_exp_name)
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--anneal-lr", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--norm-adv", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--save-model", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--cuda", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--capture-video", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--stuck-penalty", type=float, default=-0.05)
    parser.add_argument("--delta-threshold", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=1)
    args = parser.parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.use_shaping = use_shaping
    return args


def train(args, use_shaping):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "PPO MuJoCo expects continuous Box actions."
    assert isinstance(envs.single_observation_space, gym.spaces.Box), "PPO MuJoCo expects Box observations."

    reward_threshold = get_reward_threshold(envs)
    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    agent = Agent(envs, args.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, dtype=torch.float32, device=device)
    latent_actions = torch.zeros((args.num_steps, args.num_envs) + action_shape, dtype=torch.float32, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards_real = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    penalties = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    num_updates = args.total_timesteps // args.batch_size

    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metric_file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(
        episode_file,
        fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached"],
    )
    metric_writer = csv.DictWriter(
        metric_file,
        fieldnames=[
            "global_step",
            "learning_rate",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clipfrac",
            "mean_penalty",
            "success_rate",
        ],
    )
    episode_writer.writeheader()
    metric_writer.writeheader()

    recent_success = deque(maxlen=100)
    global_step = 0

    print(f"[{args.exp_name}] env={args.env_id}")
    print(f"[{args.exp_name}] obs_shape={obs_shape} action_shape={action_shape} device={device}")
    if reward_threshold is not None:
        print(f"[{args.exp_name}] reward_threshold={reward_threshold}")

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, latent_action = agent.get_action_and_value(next_obs)
            latent_actions[step] = latent_action
            logprobs[step] = logprob
            values[step] = value

            next_obs_np_raw = next_obs.detach().cpu().numpy()
            next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action.cpu().numpy())
            done_np = np.logical_or(terminated_np, truncated_np)
            penalty_np = transition_penalty(next_obs_np_raw, next_obs_np, args.delta_threshold, args.stuck_penalty)
            if not use_shaping:
                penalty_np.fill(0.0)

            rewards_real[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            penalties[step] = torch.tensor(penalty_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is None or "episode" not in info:
                        continue
                    episode_return = float(np.asarray(info["episode"]["r"]).reshape(-1)[0])
                    episode_length = int(np.asarray(info["episode"]["l"]).reshape(-1)[0])
                    success = episode_success(episode_return, reward_threshold)
                    recent_success.append(float(success))
                    episode_writer.writerow(
                        {
                            "global_step": global_step,
                            "episodic_return": episode_return,
                            "episodic_length": episode_length,
                            "goal_reached": int(success),
                        }
                    )
                    episode_file.flush()
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    writer.add_scalar(
                        "charts/success_rate",
                        float(np.mean(recent_success)) if recent_success else 0.0,
                        global_step,
                    )

        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages_value = torch.zeros_like(rewards_real)
            advantages_policy = torch.zeros_like(rewards_real)
            lastgaelam_value = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
            lastgaelam_policy = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
            shaped_rewards = rewards_real + penalties
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta_value = rewards_real[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                delta_policy = shaped_rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam_value = delta_value + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_value
                lastgaelam_policy = delta_policy + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_policy
                advantages_value[t] = lastgaelam_value
                advantages_policy[t] = lastgaelam_policy
            returns_value = advantages_value + values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_latent_actions = latent_actions.reshape((-1,) + action_shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages_policy = advantages_policy.reshape(-1)
        b_returns_value = returns_value.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_approx_kl = 0.0

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages_policy[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns_value[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns_value[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns_value[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                last_policy_loss = float(pg_loss.item())
                last_value_loss = float(v_loss.item())
                last_entropy = float(entropy_loss.item())
                last_approx_kl = float(approx_kl.item())

            if args.target_kl is not None and last_approx_kl > args.target_kl:
                break

        success_rate = float(np.mean(recent_success)) if recent_success else 0.0
        mean_penalty = float(penalties.mean().item())
        metric_writer.writerow(
            {
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "policy_loss": last_policy_loss,
                "value_loss": last_value_loss,
                "entropy": last_entropy,
                "approx_kl": last_approx_kl,
                "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "mean_penalty": mean_penalty,
                "success_rate": success_rate,
            }
        )
        metric_file.flush()
        writer.add_scalar("losses/policy_loss", last_policy_loss, global_step)
        writer.add_scalar("losses/value_loss", last_value_loss, global_step)
        writer.add_scalar("losses/entropy", last_entropy, global_step)
        writer.add_scalar("losses/approx_kl", last_approx_kl, global_step)
        writer.add_scalar("charts/mean_penalty", mean_penalty, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)

    if args.save_model:
        torch.save(agent.state_dict(), run_dir / "actor_critic.pt")

    episode_file.close()
    metric_file.close()
    writer.close()
    envs.close()
    print(f"Done. Results: {run_dir}")
    return float(np.mean(recent_success)) if recent_success else 0.0
