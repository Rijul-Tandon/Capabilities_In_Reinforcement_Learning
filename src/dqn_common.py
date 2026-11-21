"""The DQN / Double DQN trainer shared by all MiniGrid agents: configuration, environment, Q-network, replay buffer,
exploration schedule and training loop.  The entry scripts (dqn_vanilla.py, ddqn_*.py) choose the TD target and
the penalty type and call train()."""

import argparse
import csv
import json
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  (importing it registers the MiniGrid environments with gymnasium)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from learned_penalty import DiscreteStateSpace, HindsightPenaltyTrainer, MetaGradientPenaltyTrainer, TransitionPenaltyHead


# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args(default_exp_name):
    """Every hyperparameter of a run.  Each one can be overridden from the command line, e.g. --gamma 0.99."""
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("--exp-name", type=str, default=default_exp_name)   # agent label in run-folder names and plot legends
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-Random-6x6-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--load-model", type=str, default="")   # path of a q_net.pt to start from (transfer); empty = train from scratch

    # Environment
    parser.add_argument("--fixed-layout", action="store_true")   # reset with the same seed every episode, so the map never changes
    parser.add_argument("--max-steps", type=int, default=-1)     # step limit of an episode; -1 keeps MiniGrid's own limit
    parser.add_argument("--action-set", choices=["task", "full"], default="task")   # task: only the actions the task needs; full: all 7

    # Q-learning
    parser.add_argument("--learning-rate", type=float, default=0.00171)
    parser.add_argument("--gamma", type=float, default=0.95)                    # discount factor
    parser.add_argument("--hidden-size", type=int, default=256)                 # units in each of the two hidden layers
    parser.add_argument("--buffer-size", type=int, default=30000)               # replay buffer capacity, in transitions
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=2000)            # steps collected before the first update
    parser.add_argument("--train-frequency", type=int, default=10)              # one gradient step every this many env steps
    parser.add_argument("--target-network-frequency", type=int, default=264)    # copy q_net -> target_net every this many env steps

    # Exploration: epsilon decays from start to end over the first exploration_fraction of training
    parser.add_argument("--exploration-strategy", choices=["epsilon_greedy", "softmax"], default="epsilon_greedy")
    parser.add_argument("--epsilon-schedule", choices=["linear", "polynomial", "hardcoded", "cosine", "exponential", "cyclic"],
                        default="polynomial")
    parser.add_argument("--start-e", dest="start_epsilon", type=float, default=1.0)
    parser.add_argument("--end-e", dest="end_epsilon", type=float, default=0.01)
    parser.add_argument("--exploration-fraction", type=float, default=0.75)

    # Fixed penalty (ddqn_reward_shaping.py): added to the reward when the observation does not change
    parser.add_argument("--stuck-penalty", type=float, default=-1.0)

    # Learned penalty (ddqn_learned_penalty.py); the method is explained in learned_penalty.py
    parser.add_argument("--penalty-learning", choices=["meta", "hindsight"], default="meta")   # how the head is trained
    parser.add_argument("--penalty-state", choices=["position_dir", "position", "full"], default="position_dir")   # state that indexes the head
    parser.add_argument("--penalty-rank", type=int, default=0)                     # 0: full N x N table; r > 0: low-rank table U V^T
    parser.add_argument("--penalty-scale", type=float, default=0.02)               # cap: penalty = scale * sigmoid(logit)
    parser.add_argument("--penalty-lr", type=float, default=5e-3)                  # Adam learning rate of the head
    parser.add_argument("--penalty-init-std", type=float, default=0.1)             # meta: the table starts at N(0, std)
    parser.add_argument("--penalty-meta-temperature", type=float, default=0.1)     # meta: softmax temperature of the policy in J
    parser.add_argument("--penalty-init-logit", type=float, default=-2.0)          # hindsight: initial bias (-2 -> score 0.12)

    return parser.parse_args()


LOG_INTERVAL = 1000            # steps between two rows of metrics.csv
MAX_GRADIENT_NORM = 10.0       # gradients are clipped to this norm before every optimizer step
MINIGRID_ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]   # MiniGrid action indices 0..6


# ============================================================================
# ENVIRONMENT
# ============================================================================

# Grid sizes that MiniGrid does not register itself
if "MiniGrid-Empty-Random-8x8-v0" not in gym.envs.registry.keys():
    gym.register(id="MiniGrid-Empty-Random-8x8-v0", entry_point="minigrid.envs.empty:EmptyEnv",
                 kwargs={"size": 8, "agent_start_pos": None})
if "MiniGrid-Empty-Random-10x10-v0" not in gym.envs.registry.keys():
    gym.register(id="MiniGrid-Empty-Random-10x10-v0", entry_point="minigrid.envs.empty:EmptyEnv",
                 kwargs={"size": 10, "agent_start_pos": None})
if "MiniGrid-DoorKey-10x10-v0" not in gym.envs.registry.keys():
    gym.register(id="MiniGrid-DoorKey-10x10-v0", entry_point="minigrid.envs.doorkey:DoorKeyEnv", kwargs={"size": 10})


def minigrid_action_map(env_id, action_set):
    """MiniGrid action indices the agent may use, or None for all 7.

    action_set "task" keeps only the actions the environment needs, so the agent does not spend exploration on
    useless ones (e.g. "drop" when there is nothing to drop); "full" keeps all 7.
    """
    if action_set == "full":
        return None
    if "Empty" in env_id or "FourRooms" in env_id:
        return [0, 1, 2]                # left, right, forward
    if "MultiRoom" in env_id:
        return [0, 1, 2, 5]             # + toggle (doors)
    if "DoorKey" in env_id:
        return [0, 1, 2, 3, 5]          # + pickup (key) and toggle (door)
    if "UnlockPickup" in env_id:
        return [0, 1, 2, 3, 4, 5]       # + drop (the key must be dropped before the box can be picked up)
    return None


def action_names(env_id, action_set, num_actions):
    """Names of the agent's actions, in the order of the Q-network outputs."""
    action_map = minigrid_action_map(env_id, action_set)
    if action_map is None:
        return MINIGRID_ACTION_NAMES[:num_actions]
    return [MINIGRID_ACTION_NAMES[minigrid_action] for minigrid_action in action_map]


class MiniGridActionSubsetWrapper(gym.ActionWrapper):
    """Restricts the action space: agent action i is executed as MiniGrid action allowed_actions[i]."""

    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = list(allowed_actions)
        self.action_space = gym.spaces.Discrete(len(self.allowed_actions))

    def action(self, action):
        return self.allowed_actions[int(action)]


class MarkovianStepPenaltyWrapper(gym.RewardWrapper):
    """Reward +1 for reaching the goal and -0.01 for every other step.

    MiniGrid's own goal reward shrinks with the number of steps taken; this one depends only on (s, a, s').
    """

    def reward(self, reward):
        if reward > 0:
            return 1.0
        return -0.01


class FlatImageAndDirectionWrapper(gym.ObservationWrapper):
    """Turns the (width, height, 3) grid image into one flat vector in [0, 1] and appends the agent's direction.

    The three channels of a cell are (object type, colour, state); each is divided by its largest value.
    The direction (0..3) is appended as a one-hot vector of length 4.
    """

    def __init__(self, env):
        super().__init__(env)
        flat_image_size = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(flat_image_size + 4,), dtype=np.float32)

    def observation(self, observation):
        image = observation.astype(np.float32)
        image[:, :, 0] /= 10.0      # object type
        image[:, :, 1] /= 5.0       # colour
        image[:, :, 2] /= 3.0       # state (door open / closed / locked)
        direction_one_hot = np.zeros(4, dtype=np.float32)
        direction_one_hot[self.env.unwrapped.agent_dir] = 1.0
        return np.concatenate([image.flatten(), direction_one_hot])


def make_env(env_id, seed, action_set, max_steps=-1):
    """The MiniGrid environment with all wrappers.  max_steps <= 0 keeps MiniGrid's own episode step limit."""
    if max_steps is not None and max_steps > 0:
        env = gym.make(env_id, max_steps=max_steps)
    else:
        env = gym.make(env_id)
    env = MarkovianStepPenaltyWrapper(env)
    action_map = minigrid_action_map(env_id, action_set)
    if action_map is not None:
        env = MiniGridActionSubsetWrapper(env, action_map)
    env = FullyObsWrapper(env)      # the whole grid instead of the agent's 7x7 view, so the problem is an MDP
    env = ImgObsWrapper(env)        # keep only the grid image of the observation dictionary
    env = FlatImageAndDirectionWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def update_key_door_flags(env, has_picked_key, has_opened_door):
    """(has_picked_key, has_opened_door) for the environment's current state.

    Both flags are sticky within an episode: once the key has been carried or the door has been open, they stay True.
    """
    carried_object = env.unwrapped.carrying
    if carried_object is not None and carried_object.type == "key":
        has_picked_key = True
    if not has_opened_door:
        for x in range(env.unwrapped.width):
            for y in range(env.unwrapped.height):
                cell = env.unwrapped.grid.get(x, y)
                if cell is not None and cell.type == "door" and cell.is_open:
                    has_opened_door = True
    return has_picked_key, has_opened_door


# ============================================================================
# Q-NETWORK
# ============================================================================

class QNetwork(nn.Module):
    """MLP that maps an observation to one Q-value per action: (batch, observation_dim) -> (batch, num_actions)."""

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
        return self.network(observations.float())


# ============================================================================
# REPLAY BUFFER
# ============================================================================

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
        self.position = (self.position + 1) % self.capacity      # wrap around: the oldest transition is overwritten
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """A random batch as tensors, each with batch_size as first dimension, plus the sampled buffer positions."""
        buffer_indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.observations[buffer_indices], device=self.device),
            torch.tensor(self.actions[buffer_indices], device=self.device),
            torch.tensor(self.rewards[buffer_indices], device=self.device),
            torch.tensor(self.next_observations[buffer_indices], device=self.device),
            torch.tensor(self.terminated[buffer_indices], device=self.device),
            buffer_indices,
        )


# ============================================================================
# EXPLORATION SCHEDULE
# ============================================================================

def epsilon_at_step(args, step):
    """Probability of a random action at this step, for the chosen --epsilon-schedule."""
    start, end = args.start_epsilon, args.end_epsilon
    decay_steps = args.exploration_fraction * args.total_timesteps
    progress = min(step / decay_steps, 1.0)          # 0 at the start, 1 once the decay is over

    if args.epsilon_schedule == "linear":
        return max((end - start) / decay_steps * step + start, end)
    if args.epsilon_schedule == "polynomial":        # slow at first, then faster
        return end + (start - end) * (1.0 - progress) ** 3.0
    if args.epsilon_schedule == "cosine":
        return end + (start - end) * (0.5 * (1.0 + np.cos(np.pi * progress)))
    if args.epsilon_schedule == "exponential":
        return start * ((end / start) ** progress)
    if args.epsilon_schedule == "cyclic":            # three cosine decays in a row; each restart boosts exploration again
        cycle_length = args.total_timesteps / 3
        cycle_progress = (step % cycle_length) / cycle_length
        return end + (start - end) * (0.5 * (1.0 + np.cos(np.pi * cycle_progress)))
    # "hardcoded": three fixed levels, independent of start, end and exploration_fraction
    training_progress = step / args.total_timesteps
    if training_progress < 0.10:
        return 1.0
    if training_progress < 0.50:
        return 0.7
    return 0.1


# ============================================================================
# TRAINING
# ============================================================================

def train(args, double_dqn, penalty_type):
    """
    Trains one agent and writes config.json, episodes.csv, metrics.csv, the state-action counts and q_net.pt to
    <results_dir>/<env_id>__<exp_name>__<seed>__<timestamp>/.

    double_dqn      True:   y = r + gamma * Q_target(s', argmax_a Q_online(s', a))     (Double DQN)
                    False:  y = r + gamma * max_a Q_target(s', a)                       (DQN)
    penalty_type    "none":     the agent learns from the environment reward r
                    "stuck":    r + args.stuck_penalty whenever the observation does not change
                    "learned":  r - penalty(s, s'), where the penalty comes from a learned head (learned_penalty.py)
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- run folder and loggers ----
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), double_dqn=double_dqn, penalty_type=penalty_type)
    with open(run_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)

    tensorboard = SummaryWriter(f"runs/{run_name}")
    tensorboard.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{key}|{value}|" for key, value in config.items()))

    episode_file = open(run_dir / "episodes.csv", "w", newline="", encoding="utf-8")
    metric_file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(episode_file, fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "epsilon"])
    metric_writer = csv.DictWriter(metric_file, fieldnames=[
        "global_step", "epsilon", "td_loss", "q_value", "max_q", "stuck_rate", "mean_penalty",
        "penalty_prob_stuck", "penalty_prob_move", "head_loss", "head_wasted_fraction", "true_q_loss",   # learned penalty only
    ])
    episode_writer.writeheader()
    metric_writer.writeheader()

    # ---- environment ----
    env = make_env(args.env_id, args.seed, args.action_set, args.max_steps)
    observation, _ = env.reset(seed=args.seed)
    observation_shape = env.observation_space.shape
    observation_dim = int(np.prod(observation_shape))
    num_actions = env.action_space.n
    print(f"[{args.exp_name}] env={args.env_id}")
    print(f"[{args.exp_name}] State Space Size: {observation_dim} features")
    print(f"[{args.exp_name}] Action Space Size: {num_actions} actions {action_names(args.env_id, args.action_set, num_actions)}")
    print(f"[{args.exp_name}] device={device}")

    # ---- networks ----
    # q_net is trained and chooses the actions; target_net is a periodically refreshed copy that provides the TD targets.
    q_net = QNetwork(observation_dim, num_actions, args.hidden_size).to(device)
    target_net = QNetwork(observation_dim, num_actions, args.hidden_size).to(device)
    if args.load_model:
        # Continue from a trained network: no random warm-up, and exploration restarts at 0.3 instead of 1.0
        # so that the agent explores the new task without discarding what it already knows.
        print(f"[{args.exp_name}] Loading pre-trained model from {args.load_model}")
        q_net.load_state_dict(torch.load(args.load_model, map_location=device))
        args.learning_starts = 0
        args.start_epsilon = 0.3
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size, observation_shape, device)

    # ---- learned penalty: a head over the enumerated states, and the rule that trains it ----
    if penalty_type == "learned":
        state_space = DiscreteStateSpace(env, args.env_id, args.penalty_state)
        penalty_head = TransitionPenaltyHead(state_space.size, args.penalty_rank).to(device)
        if args.penalty_learning == "meta":
            penalty_trainer = MetaGradientPenaltyTrainer(penalty_head, observation_dim, num_actions, device, args)
        else:
            penalty_trainer = HindsightPenaltyTrainer(penalty_head, device, args)
        num_head_parameters = sum(parameter.numel() for parameter in penalty_head.parameters())
        print(f"[{args.exp_name}] learned penalty head: state = {args.penalty_state}, {state_space.size} discrete states, "
              f"{num_head_parameters:,} parameters, learning = {args.penalty_learning}")
        recent_score_stuck = deque(maxlen=1000)     # head scores on transitions where the observation did not change
        recent_score_moved = deque(maxlen=1000)     # head scores on all other transitions

    # ---- counters ----
    # How often each action was taken in each (x, y, direction), split by task stage:
    # 0 = start, 1 = key picked up, 2 = door opened.  Environments without a key stay in stage 0.
    counts_shape = (env.unwrapped.width, env.unwrapped.height, 4, num_actions, 3)
    state_action_counts = np.zeros(counts_shape, dtype=np.int64)
    state_action_counts_last_half = np.zeros(counts_shape, dtype=np.int64)
    state_action_counts_last_quarter = np.zeros(counts_shape, dtype=np.int64)

    recent_goals = deque(maxlen=100)        # 1 / 0 per episode: goal reached
    recent_stuck = deque(maxlen=1000)       # 1 / 0 per step: observation unchanged
    recent_penalty = deque(maxlen=1000)     # penalty of each step
    best_goal_rate = 0.0
    episode_return = 0.0
    episode_length = 0
    has_picked_key = False
    has_opened_door = False
    last_td_loss = np.nan
    last_mean_q = 0.0
    last_max_q = 0.0

    progress_bar = tqdm(range(args.total_timesteps), desc=args.exp_name)
    for global_step in progress_bar:

        # ---- choose an action ----
        if args.exploration_strategy == "softmax":
            # Sample a ~ softmax(Q(s, .) / temperature); the temperature falls linearly from 1.0 to 0.1 over 75% of training.
            temperature = max(1.0 * (1.0 - global_step / (args.total_timesteps * 0.75)), 0.1)
            epsilon = temperature       # written to the epsilon column of the logs
            with torch.no_grad():
                q_values = q_net(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))    # (1, num_actions)
                action_probabilities = torch.softmax(q_values / temperature, dim=1)
                action = int(torch.multinomial(action_probabilities, num_samples=1).item())
        else:
            epsilon = epsilon_at_step(args, global_step)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))    # (1, num_actions)
                    action = int(torch.argmax(q_values, dim=1).item())

        # ---- count the (state, action) pair ----
        has_picked_key, has_opened_door = update_key_door_flags(env, has_picked_key, has_opened_door)
        if has_opened_door:
            stage = 2
        elif has_picked_key:
            stage = 1
        else:
            stage = 0
        agent_x, agent_y = env.unwrapped.agent_pos
        agent_direction = env.unwrapped.agent_dir
        state_action_counts[agent_x, agent_y, agent_direction, action, stage] += 1
        if global_step >= args.total_timesteps * 0.50:
            state_action_counts_last_half[agent_x, agent_y, agent_direction, action, stage] += 1
        if global_step >= args.total_timesteps * 0.75:
            state_action_counts_last_quarter[agent_x, agent_y, agent_direction, action, stage] += 1
        if penalty_type == "learned":
            state_index = state_space.index(agent_x, agent_y, agent_direction, has_picked_key, has_opened_door)
            full_state = (int(agent_x), int(agent_y), int(agent_direction), bool(has_picked_key), bool(has_opened_door))

        # ---- act ----
        next_observation, env_reward, terminated, truncated, _ = env.step(action)
        observation_unchanged = bool(np.array_equal(observation, next_observation))     # e.g. the agent walked into a wall

        # ---- penalty of this transition (zero or negative) ----
        if penalty_type == "learned":
            # The head sees only the transition s -> s', never the action.
            next_has_key, next_door_open = update_key_door_flags(env, has_picked_key, has_opened_door)
            next_x, next_y = env.unwrapped.agent_pos
            next_direction = env.unwrapped.agent_dir
            next_state_index = state_space.index(next_x, next_y, next_direction, next_has_key, next_door_open)
            next_full_state = (int(next_x), int(next_y), int(next_direction), bool(next_has_key), bool(next_door_open))
            with torch.no_grad():
                penalty_score = float(torch.sigmoid(penalty_head.logits(state_index, next_state_index)).item())   # in (0, 1)
            penalty = -(args.penalty_scale * penalty_score)
            if observation_unchanged:
                recent_score_stuck.append(penalty_score)
            else:
                recent_score_moved.append(penalty_score)
            if args.penalty_learning == "hindsight":
                penalty_trainer.remember_transition(state_index, next_state_index, full_state, next_full_state)
        elif penalty_type == "stuck" and observation_unchanged:
            penalty = args.stuck_penalty
        else:
            penalty = 0.0
        recent_stuck.append(float(observation_unchanged))
        recent_penalty.append(float(penalty))

        # ---- store the transition ----
        # `terminated` is stored, not `terminated or truncated`: when an episode is cut by the step limit, s' is an
        # ordinary state, so the TD target must still bootstrap from it.
        if penalty_type == "learned" and args.penalty_learning == "meta":
            # The buffer keeps the TRUE reward; the head's current penalty is subtracted when the batch is sampled.
            penalty_trainer.record_transition(replay_buffer.position, state_index, next_state_index)
            replay_buffer.add(observation, action, float(env_reward), next_observation, float(terminated))
        else:
            replay_buffer.add(observation, action, float(env_reward + penalty), next_observation, float(terminated))
        episode_return += float(env_reward)     # the logs always report the environment reward, never the shaped one
        episode_length += 1
        observation = next_observation

        # ---- end of episode ----
        if terminated or truncated:
            # The reward is positive only on the step that reaches the goal.  (The episode return cannot be used:
            # with -0.01 per step it is negative for a goal reached after more than 100 steps.)
            reached_goal = env_reward > 0
            if penalty_type == "learned" and args.penalty_learning == "hindsight":
                penalty_trainer.end_episode(reached_goal)
            recent_goals.append(float(reached_goal))
            goal_rate = float(np.mean(recent_goals))
            best_goal_rate = max(best_goal_rate, goal_rate)

            episode_writer.writerow({
                "global_step": global_step,
                "episodic_return": episode_return,
                "episodic_length": episode_length,
                "goal_reached": int(reached_goal),
                "epsilon": epsilon,
            })
            episode_file.flush()
            tensorboard.add_scalar("charts/episodic_return", episode_return, global_step)
            tensorboard.add_scalar("charts/episodic_length", episode_length, global_step)
            tensorboard.add_scalar("charts/goal_rate", goal_rate, global_step)
            progress_bar.set_postfix({
                "return": f"{episode_return:.2f}",
                "goal%": f"{goal_rate:.0%}",
                "best%": f"{best_goal_rate:.0%}",
                "eps": f"{epsilon:.2f}",
            })

            if args.fixed_layout:
                observation, _ = env.reset(seed=args.seed)
            else:
                observation, _ = env.reset()
            has_picked_key = False
            has_opened_door = False
            episode_return = 0.0
            episode_length = 0

        # ---- one gradient step on a replay batch ----
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            (batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_terminated,
             buffer_indices) = replay_buffer.sample(args.batch_size)

            if penalty_type == "learned" and args.penalty_learning == "meta":
                # First one meta-gradient step of the head on the true rewards of this batch,
                # then the batch is shaped with the head as it is now.
                penalty_trainer.update(q_net, target_net, batch_observations, batch_actions, batch_rewards,
                                       batch_next_observations, batch_terminated, buffer_indices, double_dqn)
                with torch.no_grad():
                    batch_rewards = batch_rewards - penalty_trainer.penalties(buffer_indices)

            # TD target y = r + gamma * Q(s', a*) * (1 - terminated); no gradient flows through the target
            with torch.no_grad():
                if double_dqn:
                    # the online network selects a*, the target network evaluates it
                    best_next_actions = q_net(batch_next_observations).argmax(dim=1, keepdim=True)                  # (batch, 1)
                    next_q_values = target_net(batch_next_observations).gather(1, best_next_actions).squeeze(1)     # (batch,)
                else:
                    # the target network both selects and evaluates a*, which overestimates Q
                    next_q_values = target_net(batch_next_observations).max(dim=1).values                          # (batch,)
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

        # ---- refresh the target network ----
        if global_step % args.target_network_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ---- periodic metrics ----
        if global_step % LOG_INTERVAL == 0:
            metric_row = {
                "global_step": global_step,
                "epsilon": epsilon,
                "td_loss": last_td_loss,
                "q_value": last_mean_q,
                "max_q": last_max_q,
                "stuck_rate": float(np.mean(recent_stuck)) if recent_stuck else 0.0,
                "mean_penalty": float(np.mean(recent_penalty)) if recent_penalty else 0.0,
            }
            if penalty_type == "learned":
                metric_row["penalty_prob_stuck"] = float(np.mean(recent_score_stuck)) if recent_score_stuck else ""
                metric_row["penalty_prob_move"] = float(np.mean(recent_score_moved)) if recent_score_moved else ""
                metric_row["head_loss"] = penalty_trainer.last_head_loss
                if args.penalty_learning == "meta":
                    metric_row["true_q_loss"] = penalty_trainer.last_true_q_loss
                else:
                    metric_row["head_wasted_fraction"] = penalty_trainer.last_wasted_fraction
            metric_writer.writerow(metric_row)
            metric_file.flush()
            tensorboard.add_scalar("losses/td_loss", last_td_loss, global_step)
            tensorboard.add_scalar("losses/q_values", last_mean_q, global_step)
            tensorboard.add_scalar("losses/max_q", last_max_q, global_step)
            tensorboard.add_scalar("charts/epsilon", epsilon, global_step)

    # ---- save ----
    torch.save(q_net.state_dict(), run_dir / "q_net.pt")
    if penalty_type == "learned":
        torch.save(penalty_head.state_dict(), run_dir / "penalty_net.pt")
    np.save(run_dir / "state_action_counts.npy", state_action_counts)
    np.save(run_dir / "state_action_counts_last_half.npy", state_action_counts_last_half)
    np.save(run_dir / "state_action_counts_last_quarter.npy", state_action_counts_last_quarter)
    episode_file.close()
    metric_file.close()
    tensorboard.close()
    env.close()
    print(f"Done. Results: {run_dir}")
