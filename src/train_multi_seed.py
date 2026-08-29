"""
train_multi_seed.py — Multi-Seed Parallel Training (Shared Replay Buffer)
==========================================================================
Trains a SINGLE agent across ALL seed instances of the same environment
simultaneously, using a shared replay buffer. This ensures the agent
learns a robust policy that generalises across different layout variations.

Architecture:
  - N environment instances, one per seed (all same env_id)
  - ONE Q-network and ONE target network (shared across all seeds)
  - ONE shared replay buffer receiving transitions from ALL seeds
  - Round-robin stepping: each global step cycles through seeds
  - Per-seed episode logging (episodes_seedX.csv) for evaluation
  - Per-seed state-action counts for heatmap evaluation

Usage:
  python src/train_multi_seed.py \
      --agent ddqn_baseline \
      --env-id MiniGrid-DoorKey-6x6-v0 \
      --num-seeds 3 \
      --total-timesteps 600000

  python src/train_multi_seed.py \
      --agent ddqn_reward_shaping \
      --env-id MiniGrid-Empty-Random-8x8-v0 \
      --num-seeds 10

The total-timesteps is the TOTAL budget across ALL seeds, so each seed
receives approximately total_timesteps / num_seeds steps.
"""

import argparse
import csv
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import shared infrastructure from dqn_common
from dqn_common import (
    QNetwork,
    ReplayBuffer,
    action_names,
    episode_success,
    linear_schedule,
    polynomial_schedule,
    hardcoded_schedule,
    cosine_schedule,
    exponential_schedule,
    cyclic_schedule,
    make_env,
    observation_unchanged,
)


# ============================================================================
# PER-SEED ENVIRONMENT STATE
# ============================================================================

class SeedEnvState:
    """Tracks the per-seed environment state during multi-seed training."""

    def __init__(self, seed, env, env_id):
        self.seed = seed
        self.env = env
        self.env_id = env_id

        # Current observation
        self.obs, _ = env.reset(seed=seed)

        # Episode tracking
        self.episode_return = 0.0
        self.episode_length = 0
        self.recent_goals = deque(maxlen=100)
        self.recent_stuck = deque(maxlen=1000)
        self.recent_penalty = deque(maxlen=1000)
        self.best_goal_rate = 0.0

        # DoorKey stage tracking (monotonic per episode)
        self.has_picked_key = False
        self.has_opened_door = False


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_multi_seed_args():
    parser = argparse.ArgumentParser(
        description="Train a single agent across multiple seeds with a shared replay buffer."
    )

    # --- Agent Selection ---
    parser.add_argument("--agent", type=str, required=True,
                        choices=["ddqn_baseline", "ddqn_reward_shaping"],
                        help="Which agent type to train")

    # --- Environment ---
    parser.add_argument("--env-id", type=str, required=True,
                        help="MiniGrid environment ID (e.g. MiniGrid-DoorKey-6x6-v0)")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Number of seed instances to train on simultaneously")
    parser.add_argument("--fixed-layout", action="store_true",
                        help="Use fixed layout for each seed (deterministic resets)")

    # --- Training ---
    parser.add_argument("--total-timesteps", type=int, default=600000,
                        help="Total training steps across ALL seeds combined")
    parser.add_argument("--learning-rate", type=float, default=0.00171)
    parser.add_argument("--buffer-size", type=int, default=50000,
                        help="Shared replay buffer capacity (should be larger for multi-seed)")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--train-frequency", type=int, default=10)
    parser.add_argument("--target-network-frequency", type=int, default=264)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=-1)

    # --- Exploration ---
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.01)
    parser.add_argument("--exploration-fraction", type=float, default=0.50)
    parser.add_argument("--epsilon-schedule", type=str, default="linear",
                        choices=["linear", "polynomial", "hardcoded", "cosine", "exponential", "cyclic"])

    # --- Reward Shaping ---
    parser.add_argument("--stuck-penalty", type=float, default=-1)
    parser.add_argument("--no-change-tolerance", type=float, default=0.0)

    # --- Action Space ---
    parser.add_argument("--action-set", choices=["task", "full"], default="task")

    # --- Hardware ---
    parser.add_argument("--cuda", type=lambda x: str(x).lower() == "true", default=True)

    # --- Output ---
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--save-model", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--log-interval", type=int, default=1000)

    args = parser.parse_args()

    # Derive reward shaping flag from agent type
    args.use_shaping = ("reward_shaping" in args.agent)
    args.double_dqn = True  # Always use Double DQN
    args.exp_name = args.agent

    return args


# ============================================================================
# MULTI-SEED TRAINING LOOP
# ============================================================================

def train_multi_seed(args):
    """
    Trains a SINGLE agent across multiple seed instances simultaneously.

    Architecture:
      - N environments (one per seed), stepped in round-robin
      - ONE shared Q-network, target network, and replay buffer
      - Transitions from ALL seeds are mixed in the buffer
      - Per-seed episode metrics logged to separate CSV files
    """
    num_seeds = args.num_seeds
    use_shaping = args.use_shaping
    seeds = list(range(1, num_seeds + 1))

    # --- Reproducibility ---
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- Run Directory ---
    run_name = f"{args.env_id}__{args.exp_name}__multiseed_{num_seeds}__{int(time.time())}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["seeds"] = seeds
    config["seed"] = seeds  # For compatibility with discover_runs
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # --- TensorBoard ---
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in config.items()])),
    )

    # --- Create ALL environments ---
    max_steps_override = args.max_steps if args.max_steps > 0 else None
    envs = []
    env_states = []
    for seed in seeds:
        env = make_env(args.env_id, seed, args.action_set, max_steps=max_steps_override)
        state = SeedEnvState(seed, env, args.env_id)
        envs.append(env)
        env_states.append(state)

    # Use env[0] for dimensions
    obs_shape = envs[0].observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    num_actions = envs[0].action_space.n
    names = action_names(args.env_id, args.action_set, num_actions)

    print(f"{'=' * 70}")
    print(f"MULTI-SEED TRAINING: {args.exp_name}")
    print(f"{'=' * 70}")
    print(f"Environment      : {args.env_id}")
    print(f"Seeds            : {seeds} (N={num_seeds})")
    print(f"Total timesteps  : {args.total_timesteps:,} (≈{args.total_timesteps//num_seeds:,} per seed)")
    print(f"State Space      : {obs_dim} features")
    print(f"Action Space     : {num_actions} actions {names}")
    print(f"Reward Shaping   : {use_shaping}")
    print(f"Shared Buffer    : {args.buffer_size:,} capacity")
    print(f"Device           : {device}")
    print()

    # --- Neural Networks (SINGLE shared network) ---
    q_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    target_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)

    # --- SHARED Replay Buffer ---
    rb = ReplayBuffer(args.buffer_size, obs_shape, device)

    # --- Per-Seed State-Action Counts ---
    num_stages = 3
    w = envs[0].unwrapped.width
    h = envs[0].unwrapped.height
    per_seed_state_action_counts = {}
    per_seed_state_action_counts_last_half = {}
    per_seed_state_action_counts_last_quarter = {}
    for seed in seeds:
        per_seed_state_action_counts[seed] = np.zeros(
            (w, h, 4, num_actions, num_stages), dtype=np.int64
        )
        per_seed_state_action_counts_last_half[seed] = np.zeros(
            (w, h, 4, num_actions, num_stages), dtype=np.int64
        )
        per_seed_state_action_counts_last_quarter[seed] = np.zeros(
            (w, h, 4, num_actions, num_stages), dtype=np.int64
        )

    # --- Per-Seed CSV Logging ---
    episode_files = {}
    episode_writers = {}
    for seed in seeds:
        ep_path = run_dir / f"episodes_seed{seed}.csv"
        fh = open(ep_path, "w", newline="", encoding="utf-8")
        ew = csv.DictWriter(
            fh,
            fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "epsilon"],
        )
        ew.writeheader()
        episode_files[seed] = fh
        episode_writers[seed] = ew

    # Combined episode log (all seeds merged) for compatibility with existing analysis
    combined_ep_fh = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    combined_ep_writer = csv.DictWriter(
        combined_ep_fh,
        fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "epsilon", "seed"],
    )
    combined_ep_writer.writeheader()

    # Metrics log (shared training metrics)
    metric_fh = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    metric_writer = csv.DictWriter(
        metric_fh,
        fieldnames=["global_step", "epsilon", "td_loss", "q_value", "max_q", "stuck_rate", "mean_penalty"],
    )
    metric_writer.writeheader()

    # --- Training State ---
    last_loss = float("nan")
    last_q = 0.0
    last_max_q = 0.0
    global_recent_goals = deque(maxlen=100 * num_seeds)

    # --- Main Training Loop (Round-Robin) ---
    pbar = tqdm(range(args.total_timesteps), desc=f"{args.exp_name} (multi-seed)")
    for global_step in pbar:
        # Round-robin: pick which seed environment to step
        env_idx = global_step % num_seeds
        es = env_states[env_idx]
        env = envs[env_idx]
        seed = seeds[env_idx]

        # ---- EPSILON SCHEDULE ----
        duration = args.exploration_fraction * args.total_timesteps
        if args.epsilon_schedule == "linear":
            epsilon = linear_schedule(args.start_e, args.end_e, duration, global_step)
        elif args.epsilon_schedule == "polynomial":
            epsilon = polynomial_schedule(args.start_e, args.end_e, duration, global_step, power=3.0)
        elif args.epsilon_schedule == "hardcoded":
            epsilon = hardcoded_schedule(args.total_timesteps, global_step)
        elif args.epsilon_schedule == "cosine":
            epsilon = cosine_schedule(args.start_e, args.end_e, duration, global_step)
        elif args.epsilon_schedule == "exponential":
            epsilon = exponential_schedule(args.start_e, args.end_e, duration, global_step)
        elif args.epsilon_schedule == "cyclic":
            epsilon = cyclic_schedule(args.start_e, args.end_e, args.total_timesteps, global_step)
        else:
            epsilon = linear_schedule(args.start_e, args.end_e, duration, global_step)

        # ---- ACTION SELECTION (epsilon-greedy) ----
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(es.obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(obs_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

        # ---- STAGE TRACKING (DoorKey) ----
        carrying = getattr(env.unwrapped, 'carrying', None)
        if carrying is not None and getattr(carrying, 'type', None) == "key":
            es.has_picked_key = True

        if not es.has_opened_door:
            grid = env.unwrapped.grid
            for wx in range(env.unwrapped.width):
                for wy in range(env.unwrapped.height):
                    c = grid.get(wx, wy)
                    if c is not None and getattr(c, 'type', None) == "door" and getattr(c, 'is_open', False):
                        es.has_opened_door = True
                        break
                if es.has_opened_door:
                    break

        if es.has_opened_door:
            stage_idx = 2
        elif es.has_picked_key:
            stage_idx = 1
        else:
            stage_idx = 0

        # Track state-action per seed
        ax, ay = env.unwrapped.agent_pos
        ad = env.unwrapped.agent_dir
        per_seed_state_action_counts[seed][ax, ay, ad, action, stage_idx] += 1
        if global_step >= args.total_timesteps * 0.50:
            per_seed_state_action_counts_last_half[seed][ax, ay, ad, action, stage_idx] += 1
        if global_step >= args.total_timesteps * 0.75:
            per_seed_state_action_counts_last_quarter[seed][ax, ay, ad, action, stage_idx] += 1

        # ---- STEP ENVIRONMENT ----
        next_obs, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ---- REWARD SHAPING ----
        no_change = observation_unchanged(es.obs, next_obs, args.no_change_tolerance)
        penalty = args.stuck_penalty if use_shaping and no_change else 0.0
        reward_for_learning = float(env_reward + penalty)

        es.recent_stuck.append(float(no_change))
        es.recent_penalty.append(float(penalty))

        # ---- STORE IN SHARED BUFFER ----
        rb.add(es.obs, next_obs, action, reward_for_learning, float(terminated))
        es.episode_return += float(env_reward)
        es.episode_length += 1
        es.obs = next_obs

        # ---- END OF EPISODE ----
        if done:
            reached = episode_success(args.env_id, es.episode_return)
            es.recent_goals.append(float(reached))
            global_recent_goals.append(float(reached))
            goal_rate = float(np.mean(es.recent_goals)) if es.recent_goals else 0.0
            es.best_goal_rate = max(es.best_goal_rate, goal_rate)

            row = {
                "global_step": global_step,
                "episodic_return": es.episode_return,
                "episodic_length": es.episode_length,
                "goal_reached": int(reached),
                "epsilon": epsilon,
            }
            episode_writers[seed].writerow(row)
            episode_files[seed].flush()

            combined_row = dict(row)
            combined_row["seed"] = seed
            combined_ep_writer.writerow(combined_row)
            combined_ep_fh.flush()

            writer.add_scalar(f"charts/episodic_return_seed{seed}", es.episode_return, global_step)
            writer.add_scalar(f"charts/goal_rate_seed{seed}", goal_rate, global_step)

            # Reset
            reset_kwargs = {"seed": seed} if args.fixed_layout else {}
            es.obs, _ = env.reset(**reset_kwargs)
            es.has_picked_key = False
            es.has_opened_door = False
            es.episode_return = 0.0
            es.episode_length = 0

        # ---- NEURAL NETWORK TRAINING ----
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

        # ---- TARGET NETWORK UPDATE ----
        if global_step % args.target_network_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ---- PERIODIC LOGGING ----
        if global_step % args.log_interval == 0:
            all_stuck = []
            all_penalty = []
            for s in env_states:
                all_stuck.extend(s.recent_stuck)
                all_penalty.extend(s.recent_penalty)

            metric_writer.writerow({
                "global_step": global_step,
                "epsilon": epsilon,
                "td_loss": last_loss,
                "q_value": last_q,
                "max_q": last_max_q,
                "stuck_rate": float(np.mean(all_stuck)) if all_stuck else 0.0,
                "mean_penalty": float(np.mean(all_penalty)) if all_penalty else 0.0,
            })
            metric_fh.flush()
            writer.add_scalar("losses/td_loss", last_loss, global_step)
            writer.add_scalar("losses/q_values", last_q, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

            global_goal_rate = float(np.mean(global_recent_goals)) if global_recent_goals else 0.0
            pbar.set_postfix({
                "seed": seed,
                "goal%": f"{global_goal_rate:.0%}",
                "eps": f"{epsilon:.2f}",
                "buf": rb.size,
            })

    # --- Save Model ---
    if args.save_model:
        model_path = run_dir / "q_net.pt"
        torch.save(q_net.state_dict(), model_path)
        print(f"Saved shared model → {model_path}")

    # --- Save Per-Seed State-Action Counts ---
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "state_action_counts.npy", per_seed_state_action_counts[seed])
        np.save(seed_dir / "state_action_counts_last_half.npy", per_seed_state_action_counts_last_half[seed])
        np.save(seed_dir / "state_action_counts_last_quarter.npy", per_seed_state_action_counts_last_quarter[seed])

        # Also copy q_net.pt and config.json into each seed dir for compatibility
        # with downstream plotting scripts (discover_runs expects per-seed dirs)
        torch.save(q_net.state_dict(), seed_dir / "q_net.pt")

        seed_config = dict(config)
        seed_config["seed"] = seed
        seed_config["training_mode"] = "multi_seed"
        seed_config["all_seeds"] = seeds
        with open(seed_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(seed_config, f, indent=2)

        # Copy the per-seed episode CSV into the seed dir as episodes.csv
        import shutil
        src_ep = run_dir / f"episodes_seed{seed}.csv"
        shutil.copy2(src_ep, seed_dir / "episodes.csv")

    # --- Cleanup ---
    for fh in episode_files.values():
        fh.close()
    combined_ep_fh.close()
    metric_fh.close()
    writer.close()
    for env in envs:
        env.close()

    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE: {args.exp_name}")
    print(f"{'=' * 70}")
    print(f"Results directory : {run_dir}")
    print(f"Seeds trained     : {seeds}")
    print(f"Total steps       : {args.total_timesteps:,}")
    for es in env_states:
        print(f"  Seed {es.seed}: best goal rate = {es.best_goal_rate:.1%}")
    print()

    return run_dir


if __name__ == "__main__":
    args = parse_multi_seed_args()
    train_multi_seed(args)
