import argparse
import csv
import json
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from minigrid.wrappers import FlatObsWrapper
from tqdm import tqdm


MINIGRID_ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def minigrid_action_map(env_id, action_set):
    if action_set == "full":
        return None
    if action_set != "task":
        raise ValueError(f"Unknown action set {action_set!r}; use 'task' or 'full'.")
    if "Empty" in env_id or "FourRooms" in env_id:
        return [0, 1, 2]
    if "DoorKey" in env_id or "UnlockPickup" in env_id:
        return [0, 1, 2, 3, 5]
    return None


class MiniGridActionSubsetWrapper(gym.ActionWrapper):
    def __init__(self, env, actions):
        super().__init__(env)
        self.actions = list(actions)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[int(action)]


def make_env(env_id, seed, action_set):
    env = gym.make(env_id)
    action_map = minigrid_action_map(env_id, action_set)
    if action_map is not None:
        env = MiniGridActionSubsetWrapper(env, action_map)
    env = FlatObsWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def action_names(env_id, action_set, action_space_n):
    action_map = minigrid_action_map(env_id, action_set)
    if action_map is None:
        return MINIGRID_ACTION_NAMES[:action_space_n]
    return [MINIGRID_ACTION_NAMES[action] for action in action_map]


class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        return self.network(x.float() / 255.0)


class ReplayBuffer:
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


def linear_schedule(start_e, end_e, duration, step):
    slope = (end_e - start_e) / duration
    return max(slope * step + start_e, end_e)


def parse_args(default_exp_name, use_shaping):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=default_exp_name)
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-network-frequency", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.6)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    parser.add_argument("--cuda", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--save-model", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--stuck-penalty", type=float, default=-0.01)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
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

    env = make_env(args.env_id, args.seed, args.action_set)
    obs, _ = env.reset(seed=args.seed)
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    num_actions = env.action_space.n
    names = action_names(args.env_id, args.action_set, num_actions)

    print(f"[{args.exp_name}] env={args.env_id} obs_dim={obs_dim} actions={names} device={device}")

    q_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    target_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(args.buffer_size, obs_shape, device)

    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metric_file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(
        episode_file,
        fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "epsilon"],
    )
    metric_writer = csv.DictWriter(
        metric_file,
        fieldnames=["global_step", "epsilon", "td_loss", "q_value", "stuck_rate", "mean_penalty"],
    )
    episode_writer.writeheader()
    metric_writer.writeheader()

    recent_goals = deque(maxlen=100)
    recent_stuck = deque(maxlen=1000)
    recent_penalty = deque(maxlen=1000)
    episode_return = 0.0
    episode_length = 0
    last_loss = np.nan
    last_q = np.nan
    best_goal_rate = 0.0

    pbar = tqdm(range(args.total_timesteps), desc=args.exp_name)
    for global_step in pbar:
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(q_net(obs_tensor), dim=1).item())

        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        no_change = bool(np.array_equal(next_obs, obs))
        penalty = args.stuck_penalty if use_shaping and no_change else 0.0
        reward_for_learning = float(env_reward + penalty)
        recent_stuck.append(float(no_change))
        recent_penalty.append(float(penalty))

        rb.add(obs, next_obs, action, reward_for_learning, float(done))
        episode_return += float(env_reward)
        episode_length += 1

        obs = next_obs
        if done:
            reached = episode_return > 0.0
            recent_goals.append(float(reached))
            goal_rate = float(np.mean(recent_goals)) if recent_goals else 0.0
            best_goal_rate = max(best_goal_rate, goal_rate)
            episode_writer.writerow(
                {
                    "global_step": global_step,
                    "episodic_return": episode_return,
                    "episodic_length": episode_length,
                    "goal_reached": int(reached),
                    "epsilon": epsilon,
                }
            )
            episode_file.flush()
            pbar.set_postfix(
                {
                    "return": f"{episode_return:.2f}",
                    "goal%": f"{goal_rate:.0%}",
                    "best%": f"{best_goal_rate:.0%}",
                    "eps": f"{epsilon:.2f}",
                }
            )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)
            with torch.no_grad():
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

        if global_step % args.target_network_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        if global_step % args.log_interval == 0:
            metric_writer.writerow(
                {
                    "global_step": global_step,
                    "epsilon": epsilon,
                    "td_loss": last_loss,
                    "q_value": last_q,
                    "stuck_rate": float(np.mean(recent_stuck)) if recent_stuck else 0.0,
                    "mean_penalty": float(np.mean(recent_penalty)) if recent_penalty else 0.0,
                }
            )
            metric_file.flush()

    if args.save_model:
        torch.save(q_net.state_dict(), run_dir / "q_net.pt")

    episode_file.close()
    metric_file.close()
    env.close()
    print(f"Done. Results: {run_dir}")
