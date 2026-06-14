"""
dqn_common.py - Shared DQN Infrastructure
==========================================
This is the core module for all our DQN experiments. It contains everything needed
to train a Deep Q-Network agent on MiniGrid environments:

  - Environment setup with observation/action wrappers
  - The Q-Network (neural network) architecture
  - The Replay Buffer for experience storage
  - The epsilon-greedy exploration schedule
  - The main training loop
  - Argument parsing for hyperparameter configuration

Both dqn_baseline.py and dqn_reward_shaping.py import from this file.
The only difference between them is whether reward shaping (stuck penalty) is enabled.
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

# argparse: Parses command-line arguments (e.g., --env-id, --total-timesteps).
#   Lets us configure experiments from the terminal without editing code.
import argparse

# csv: Reads and writes CSV (Comma Separated Values) files.
#   We use it to log episode-level and step-level metrics to disk.
import csv

# json: Reads and writes JSON files.
#   We save the full experiment configuration (all hyperparameters) as config.json.
import json

# random: Python's built-in random number generator.
#   Used for epsilon-greedy action selection (random.random() < epsilon).
import random

# time: Provides time-related functions.
#   We use time.time() to generate unique timestamps for run directory names.
import time

# deque (double-ended queue): A list-like container from the collections module
#   with fast appends and pops from both ends. When maxlen is set, it automatically
#   discards the oldest items when new ones are added. We use it to track the
#   most recent 100 goal-reach outcomes for computing a rolling success rate.
from collections import deque

# Path: Object-oriented filesystem paths from the pathlib module.
#   Makes it easy to create directories, join paths, and check if files exist
#   without string concatenation (e.g., Path("results") / "run_name" / "config.json").
from pathlib import Path


# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

# gymnasium (gym): The standard API for Reinforcement Learning environments.
#   Formerly known as OpenAI Gym. It defines the interface that all RL environments
#   follow: env.reset() returns an observation, env.step(action) returns
#   (next_obs, reward, terminated, truncated, info). It also provides "wrappers"
#   which are decorators that modify environment behavior without changing the
#   underlying environment code.
import gymnasium as gym

# minigrid: A collection of lightweight grid-world environments for RL research.
#   Simply importing this module registers all MiniGrid environments with gymnasium,
#   so that gym.make("MiniGrid-Empty-8x8-v0") works. Without this import,
#   gymnasium would not know these environments exist.
#   noqa: F401 tells the linter to ignore the "imported but unused" warning.
import minigrid  # noqa: F401 - registers MiniGrid envs

# numpy (np): The fundamental package for numerical computing in Python.
#   We use it for array operations (observations are numpy arrays), random sampling
#   in the replay buffer, and computing statistics like mean stuck rate.
import numpy as np

# torch: PyTorch, the deep learning framework.
#   We use it to build, train, and run the neural network (Q-Network).
import torch

# torch.nn: Contains neural network building blocks (layers, activation functions).
#   nn.Module is the base class for all neural networks in PyTorch.
#   nn.Linear is a fully-connected (dense) layer: output = input @ weights + bias.
#   nn.ReLU is the Rectified Linear Unit activation function: max(0, x).
#   nn.Sequential chains layers together so data flows through them in order.
import torch.nn as nn

# torch.nn.functional (F): Provides functions (as opposed to nn's module-based approach)
#   for operations like loss computation. We use F.mse_loss() to compute
#   Mean Squared Error between predicted Q-values and target Q-values.
import torch.nn.functional as F

# torch.optim: Contains optimization algorithms that update neural network weights.
#   optim.Adam is an adaptive learning rate optimizer that works well in practice.
#   It adjusts the learning rate for each weight individually based on past gradients.
import torch.optim as optim

# FlatObsWrapper (from minigrid.wrappers):
#   By default, MiniGrid returns observations as a Python dictionary containing:
#     - "image": a 3D numpy array of shape (width, height, 3) where each cell has
#       3 values: [object_type, color, state]. For example, a wall tile might be [2, 5, 0].
#     - "direction": an integer (0-3) for the agent's facing direction.
#     - "mission": a text string like "reach the goal".
#   Neural networks cannot process dictionaries. FlatObsWrapper extracts the "image"
#   array and flattens it into a 1D vector. For an 8x8 grid, this produces a vector
#   of 8 * 8 * 3 = 192 numbers that can be fed directly into an MLP.
#
# FullyObsWrapper (from minigrid.wrappers):
#   By default, MiniGrid gives the agent a PARTIAL observation: a 7x7 grid of tiles
#   directly in front of the agent (like a flashlight cone). The agent cannot see
#   what is behind it or far away. This makes it a POMDP (Partially Observable
#   Markov Decision Process), which is much harder to solve.
#   FullyObsWrapper overrides this and gives the agent the ENTIRE map as its
#   observation. This converts the problem into a standard MDP, making it much
#   easier for a memoryless MLP to solve.
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper

# tqdm: Displays a progress bar in the terminal during long loops.
#   Wrapping range() with tqdm() shows elapsed time, iterations per second,
#   and a visual progress indicator. We also use pbar.set_postfix() to display
#   live training statistics (return, goal rate, epsilon) next to the progress bar.
from tqdm import tqdm

# SummaryWriter (from torch.utils.tensorboard):
#   TensorBoard is a visualization tool originally built for TensorFlow but also
#   works with PyTorch. SummaryWriter logs scalar values (loss, reward, epsilon)
#   to disk. You can then run `tensorboard --logdir runs/` in your terminal to
#   view interactive, real-time training charts in your browser.
from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# CONSTANTS
# ============================================================================

# The 7 standard actions available in any MiniGrid environment, indexed 0-6.
# Not all environments need all actions. For example, "Empty" only needs
# left (0), right (1), and forward (2). "DoorKey" also needs pickup (3)
# and toggle (5) to interact with keys and doors.
MINIGRID_ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def minigrid_action_map(env_id, action_set):
    """
    Determines which subset of actions are useful for a given MiniGrid environment.
    Removing irrelevant actions speeds up learning because the agent doesn't waste
    time exploring useless actions like "drop" when there is nothing to drop.

    Parameters
    ----------
    env_id : str
        The gymnasium environment ID (e.g., "MiniGrid-Empty-8x8-v0").
        We check if the name contains "Empty", "DoorKey", etc. to decide which actions to keep.
    action_set : str
        Either "task" (use only task-relevant actions) or "full" (use all 7 actions).

    Returns
    -------
    list[int] or None
        A list of MiniGrid action indices to keep, or None if all actions should be used.
    """
    if action_set == "full":
        # Use all 7 actions, no filtering needed
        return None
    if action_set != "task":
        raise ValueError(f"Unknown action set {action_set!r}; use 'task' or 'full'.")

    # Navigation-only environments: the agent just needs to move around
    if "Empty" in env_id or "FourRooms" in env_id:
        return [0, 1, 2]  # left, right, forward

    # Interaction environments: the agent also needs to pick up keys and toggle doors
    if "DoorKey" in env_id or "UnlockPickup" in env_id:
        return [0, 1, 2, 3, 5]  # left, right, forward, pickup, toggle

    # Unknown environment: use all actions as a safe default
    return None


class MiniGridActionSubsetWrapper(gym.ActionWrapper):
    """
    A custom Gymnasium wrapper that restricts the environment's action space.

    Why this is needed:
      MiniGrid always exposes 7 actions (Discrete(7)), but most environments
      only need 3-5 of them. If we let the neural network output 7 Q-values,
      it wastes capacity learning about useless actions. This wrapper makes
      the neural network only output Q-values for the actions we actually need.

    How it works:
      The wrapper maintains a mapping list, e.g., [0, 1, 2] for Empty environments.
      When the neural network outputs action index 0, this wrapper translates it
      to MiniGrid action 0 (left). When it outputs action index 2, it translates
      to MiniGrid action 2 (forward). The neural network never sees the raw
      MiniGrid action indices; it only sees indices 0..len(actions)-1.

    Inherits from gym.ActionWrapper, which is a Gymnasium base class specifically
    designed for wrappers that only modify the action space (not the observations).

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    actions : list[int]
        The list of MiniGrid action indices to keep (e.g., [0, 1, 2] for left/right/forward).
    """
    def __init__(self, env, actions):
        super().__init__(env)
        self.actions = list(actions)
        # Override the action space so the neural network only has len(actions) outputs
        # Discrete(n) means the action space is {0, 1, 2, ..., n-1}
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        """
        Called automatically by Gymnasium when env.step(action) is called.
        Translates the wrapper's action index back to the real MiniGrid action index.

        Parameters
        ----------
        action : int
            The action index from the neural network (0 to len(self.actions)-1).

        Returns
        -------
        int
            The corresponding MiniGrid action index.
        """
        return self.actions[int(action)]


def make_env(env_id, seed, action_set, capture_video=False, run_name=""):
    """
    Creates a MiniGrid environment and wraps it with all necessary wrappers.

    The wrapper chain (order matters!):
      1. MiniGridActionSubsetWrapper - restricts actions to task-relevant ones
      2. FullyObsWrapper - gives the agent full map visibility (not just 7x7 cone)
      3. FlatObsWrapper - flattens the 3D grid into a 1D vector for the neural network
      4. RecordEpisodeStatistics - automatically tracks episode return and length

    Parameters
    ----------
    env_id : str
        The gymnasium environment ID (e.g., "MiniGrid-DoorKey-8x8-v0").
    seed : int
        Random seed for reproducibility of the action space sampling.
    action_set : str
        "task" to use only task-relevant actions, "full" to use all 7 actions.
    capture_video : bool, optional
        If True, records a video of the agent's behavior. Default is False.
    run_name : str, optional
        Name for the video recording directory. Only used when capture_video=True.

    Returns
    -------
    gym.Env
        The fully wrapped environment, ready for training.
    """
    if capture_video:
        # render_mode="rgb_array" makes the environment return pixel frames
        # that can be recorded as video, instead of rendering to a window
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)

    # Step 1: Restrict actions to only the ones needed for this environment
    action_map = minigrid_action_map(env_id, action_set)
    if action_map is not None:
        env = MiniGridActionSubsetWrapper(env, action_map)

    # Step 2: Give the agent full visibility of the entire map
    # Without this, the agent only sees a 7x7 cone in front of it (partial observability)
    env = FullyObsWrapper(env)

    # Step 3: Flatten the 3D grid observation (width x height x 3) into a 1D vector
    # This is required because our MLP neural network expects a flat input vector
    env = FlatObsWrapper(env)

    # Step 4: Automatically record episode statistics (total return, episode length)
    # These are stored in the 'info' dict returned by env.step() when an episode ends
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Seed the action space so that random action sampling is reproducible
    env.action_space.seed(seed)
    return env


def action_names(env_id, action_set, action_space_n):
    """
    Returns human-readable names for the actions being used.

    Parameters
    ----------
    env_id : str
        The gymnasium environment ID.
    action_set : str
        "task" or "full" action set.
    action_space_n : int
        The number of actions in the (possibly wrapped) action space.

    Returns
    -------
    list[str]
        Names like ["left", "right", "forward"] for display and logging.
    """
    action_map = minigrid_action_map(env_id, action_set)
    if action_map is None:
        return MINIGRID_ACTION_NAMES[:action_space_n]
    return [MINIGRID_ACTION_NAMES[action] for action in action_map]


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class QNetwork(nn.Module):
    """
    The Deep Q-Network (DQN) - a Multi-Layer Perceptron (MLP).

    Architecture:
      Input (obs_dim) -> Linear(hidden_size) -> ReLU -> Linear(hidden_size) -> ReLU -> Linear(num_actions)

    The network takes a flattened observation vector as input and outputs one
    Q-value for each possible action. The Q-value Q(s, a) represents the
    expected cumulative discounted reward the agent will receive if it takes
    action 'a' in state 's' and then follows the optimal policy afterwards.

    During action selection, the agent picks the action with the highest Q-value:
      action = argmax_a Q(s, a)

    Inherits from nn.Module, which is PyTorch's base class for all neural networks.
    It provides automatic gradient tracking, parameter management, and GPU support.

    Parameters
    ----------
    obs_dim : int
        The size of the flattened observation vector (e.g., 192 for an 8x8 fully observable grid).
    num_actions : int
        The number of possible actions (e.g., 3 for Empty, 5 for DoorKey).
    hidden_size : int
        The number of neurons in each hidden layer (default: 256).
    """
    def __init__(self, obs_dim, num_actions, hidden_size):
        super().__init__()
        # nn.Sequential chains layers together: data flows through them in order
        self.network = nn.Sequential(
            # First hidden layer: maps observation vector to hidden representation
            nn.Linear(obs_dim, hidden_size),
            # ReLU activation: max(0, x). Introduces non-linearity so the network
            # can learn complex patterns (without it, stacking linear layers is
            # mathematically equivalent to a single linear layer)
            nn.ReLU(),
            # Second hidden layer: further processes the hidden representation
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # Output layer: produces one Q-value per action (no activation function
            # because Q-values can be any real number, positive or negative)
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        """
        Forward pass: computes Q-values for all actions given an observation.

        NOTE: We cast to float but do NOT divide by 255. MiniGrid observations
        are categorical integers (object_type, color, state) ranging from 0 to ~10,
        NOT RGB pixel values (0-255). Dividing by 255 would squash these values
        close to zero, making it nearly impossible for the network to distinguish
        between different objects.

        Parameters
        ----------
        x : torch.Tensor
            A batch of observations, shape (batch_size, obs_dim).

        Returns
        -------
        torch.Tensor
            Q-values for each action, shape (batch_size, num_actions).
        """
        return self.network(x.float())


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Experience Replay Buffer - stores past (s, a, r, s', done) transitions.

    Why we need this:
      In supervised learning, we assume training samples are independent and
      identically distributed (i.i.d.). But in RL, consecutive transitions are
      highly correlated (step 100 is very similar to step 101). Training a neural
      network on correlated data causes instability and poor convergence.

      The replay buffer solves this by storing thousands of past transitions and
      sampling random mini-batches for training. This breaks the temporal correlation
      and makes the training data approximately i.i.d.

    Implementation:
      Uses pre-allocated numpy arrays (not Python lists) for memory efficiency.
      Operates as a circular buffer: when full, new transitions overwrite the oldest ones.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store (e.g., 100,000).
    obs_shape : tuple
        Shape of a single observation (e.g., (192,) for a flattened 8x8 grid).
    device : torch.device
        The device (CPU or GPU) to move sampled tensors to.
    """
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.pos = 0     # Current write position in the circular buffer
        self.size = 0    # Number of transitions currently stored (up to capacity)

        # Pre-allocate numpy arrays for each component of a transition
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)       # Current observation s
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)  # Next observation s'
        self.actions = np.zeros(capacity, dtype=np.int64)                    # Action taken a
        self.rewards = np.zeros(capacity, dtype=np.float32)                  # Reward received r
        self.dones = np.zeros(capacity, dtype=np.float32)                    # Episode done flag (0 or 1)

    def add(self, obs, next_obs, action, reward, done):
        """
        Store a single transition (s, s', a, r, done) in the buffer.

        Parameters
        ----------
        obs : np.ndarray
            The current observation (state) before taking the action.
        next_obs : np.ndarray
            The observation (state) after taking the action.
        action : int
            The action that was taken.
        reward : float
            The reward received (may include shaping penalties).
        done : float
            1.0 if the episode ended after this step, 0.0 otherwise.
        """
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        # Circular buffer: wrap around to the beginning when we reach capacity
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Randomly sample a mini-batch of transitions for training.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample (e.g., 128).

        Returns
        -------
        tuple of torch.Tensors
            (observations, next_observations, actions, rewards, dones)
            Each tensor has batch_size as its first dimension.
        """
        # Uniformly sample random indices from the filled portion of the buffer
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
    """
    Computes the current epsilon value for epsilon-greedy exploration.

    Epsilon controls the exploration-exploitation tradeoff:
      - epsilon = 1.0: 100% random actions (pure exploration)
      - epsilon = 0.0: 100% greedy actions (pure exploitation)

    This function linearly decays epsilon from start_e to end_e over 'duration'
    steps, then holds it at end_e for the rest of training. This ensures the
    agent explores broadly at the start and gradually shifts to exploiting
    what it has learned, while always maintaining some minimum exploration.

    Parameters
    ----------
    start_e : float
        Starting epsilon value (typically 1.0 for full exploration).
    end_e : float
        Minimum epsilon value (e.g., 0.3 to always keep 30% random actions).
    duration : float
        Number of steps over which to decay from start_e to end_e.
    step : int
        The current training step.

    Returns
    -------
    float
        The current epsilon value, clamped to be at least end_e.
    """
    slope = (end_e - start_e) / duration
    return max(slope * step + start_e, end_e)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args(default_exp_name, use_shaping):
    """
    Parses command-line arguments for configuring a DQN training run.

    All hyperparameters can be overridden from the terminal. For example:
      python dqn_baseline.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 50000

    Parameters
    ----------
    default_exp_name : str
        Name for this experiment type (e.g., "dqn_baseline" or "dqn_reward_shaping").
        Used in directory names and plot labels to distinguish between runs.
    use_shaping : bool
        Whether this experiment uses reward shaping. Stored in args for reference.

    Returns
    -------
    argparse.Namespace
        An object containing all hyperparameters as attributes (e.g., args.env_id).
    """
    parser = argparse.ArgumentParser()

    # --- Experiment Identity ---
    # --exp-name: A label for this experiment, used in directory names and plot legends
    parser.add_argument("--exp-name", type=str, default=default_exp_name)
    # --env-id: Which MiniGrid environment to train on
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    # --seed: Random seed for reproducibility (same seed = same results)
    parser.add_argument("--seed", type=int, default=1)

    # --- Training Duration ---
    # --total-timesteps: How many environment steps to train for
    parser.add_argument("--total-timesteps", type=int, default=100000)

    # --- DQN Hyperparameters ---
    # --learning-rate: How fast the neural network adjusts its weights.
    #   Too high = unstable training. Too low = slow learning. 2.5e-4 is a good default.
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    # --buffer-size: Maximum number of transitions stored in the replay buffer
    parser.add_argument("--buffer-size", type=int, default=100000)
    # --gamma: Discount factor for future rewards (0 = greedy, 1 = far-sighted).
    #   0.99 means the agent values a reward 100 steps away at 0.99^100 ≈ 0.37 of its face value.
    parser.add_argument("--gamma", type=float, default=0.99)
    # --target-network-frequency: How often (in steps) to copy q_net weights to target_net.
    #   The target network provides stable Q-value targets during training.
    parser.add_argument("--target-network-frequency", type=int, default=1000)
    # --batch-size: Number of transitions sampled from the replay buffer per training step
    parser.add_argument("--batch-size", type=int, default=128)
    # --learning-starts: Number of random steps before training begins.
    #   This seeds the replay buffer with diverse experiences before the network starts learning.
    parser.add_argument("--learning-starts", type=int, default=5000)
    # --train-frequency: Train the network every N environment steps (not every single step)
    parser.add_argument("--train-frequency", type=int, default=4)

    # --- Exploration Schedule ---
    # --start-e: Initial epsilon (exploration rate). 1.0 = 100% random actions at the start.
    parser.add_argument("--start-e", type=float, default=1.0)
    # --end-e: Minimum epsilon. 0.3 = always keep at least 30% random actions.
    #   This prevents the agent from getting stuck in local optima, especially in
    #   environments with randomized layouts (DoorKey, FourRooms) where the agent
    #   needs to keep exploring to handle new configurations.
    parser.add_argument("--end-e", type=float, default=0.3)
    # --exploration-fraction: Fraction of total timesteps over which epsilon decays.
    #   0.6 means epsilon reaches end_e at 60% of training, then stays flat.
    parser.add_argument("--exploration-fraction", type=float, default=0.6)

    # --- Network Architecture ---
    # --hidden-size: Number of neurons in each hidden layer of the Q-Network
    parser.add_argument("--hidden-size", type=int, default=256)

    # --- Action Space ---
    # --action-set: "task" uses only environment-relevant actions, "full" uses all 7
    parser.add_argument("--action-set", choices=["task", "full"], default="task")

    # --- Hardware ---
    # --cuda: Whether to use GPU acceleration (if available)
    parser.add_argument("--cuda", type=lambda x: str(x).lower() == "true", default=True)

    # --- Model Saving ---
    # --save-model: Whether to save the trained Q-Network weights to disk as q_net.pt
    parser.add_argument("--save-model", type=lambda x: str(x).lower() == "true", default=True)

    # --- Reward Shaping ---
    # --stuck-penalty: Negative reward applied when the agent's observation doesn't change
    #   (e.g., bumping into a wall or doing nothing). Only used when use_shaping=True.
    parser.add_argument("--stuck-penalty", type=float, default=-0.01)

    # --- Logging ---
    # --log-interval: How often (in steps) to write training metrics to CSV and TensorBoard
    parser.add_argument("--log-interval", type=int, default=1000)
    # --results-dir: Parent directory where run folders are created
    parser.add_argument("--results-dir", type=str, default="results")

    # --- Optional: Weights & Biases Integration ---
    # --track: Enable Weights & Biases (wandb) logging for cloud-based experiment tracking
    parser.add_argument("--track", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL")
    parser.add_argument("--wandb-entity", type=str, default=None)

    # --- Optional: Video Recording ---
    parser.add_argument("--capture-video", type=lambda x: str(x).lower() == "true", default=False)

    args = parser.parse_args()
    # Store the shaping flag so it can be accessed alongside other args
    args.use_shaping = use_shaping
    return args


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(args, use_shaping):
    """
    The main DQN training loop. This function:
      1. Creates the environment and neural networks
      2. Collects experience using epsilon-greedy exploration
      3. Stores transitions in a replay buffer
      4. Periodically samples mini-batches and trains the Q-Network
      5. Logs metrics to CSV files and TensorBoard
      6. Saves the trained model to disk

    The DQN Algorithm (Mnih et al., 2015):
      For each step:
        - With probability epsilon, take a random action (explore)
        - Otherwise, take action = argmax_a Q(s, a) (exploit)
        - Store transition (s, a, r, s', done) in replay buffer
        - Sample a random mini-batch from the buffer
        - Compute target: y = r + gamma * max_a' Q_target(s', a') * (1 - done)
        - Update Q-Network to minimize: loss = (Q(s, a) - y)^2
        - Periodically copy Q-Network weights to Target Network

    Parameters
    ----------
    args : argparse.Namespace
        All hyperparameters and configuration (from parse_args).
    use_shaping : bool
        If True, applies a stuck penalty when the agent's observation doesn't change.
        This discourages the agent from wasting steps bumping into walls.
    """
    # --- Reproducibility ---
    # Setting seeds for all random number generators ensures that running the same
    # experiment with the same seed produces identical results.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Makes CUDA operations deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    # Use GPU if available and requested, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- Run Directory Setup ---
    # Create a unique directory name using: environment__experiment__seed__timestamp
    # Example: "MiniGrid-Empty-8x8-v0__dqn_baseline__1__1718300000"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save the full configuration as JSON for later reference and reproducibility
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # --- Optional: Weights & Biases ---
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # --- TensorBoard Writer ---
    # Creates a TensorBoard log directory under runs/. View with: tensorboard --logdir runs/
    writer = SummaryWriter(f"runs/{run_name}")
    # Log all hyperparameters as a markdown table in TensorBoard
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # --- Environment Setup ---
    env = make_env(args.env_id, args.seed, args.action_set, args.capture_video, run_name)
    obs, _ = env.reset(seed=args.seed)

    # Get observation and action space dimensions from the wrapped environment
    obs_shape = env.observation_space.shape   # e.g., (192,) for 8x8 fully observable
    obs_dim = int(np.prod(obs_shape))         # Total number of input features
    num_actions = env.action_space.n          # Number of available actions
    names = action_names(args.env_id, args.action_set, num_actions)

    print(f"[{args.exp_name}] env={args.env_id} obs_dim={obs_dim} actions={names} device={device}")

    # --- Neural Networks ---
    # DQN uses TWO copies of the same network:
    #   q_net (online network): The network being actively trained. Used for action selection.
    #   target_net (target network): A frozen copy used to compute stable Q-value targets.
    #     Without this, training is unstable because the target Q-values shift every update.
    #     The target network is periodically updated by copying weights from q_net.
    q_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    target_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    target_net.load_state_dict(q_net.state_dict())  # Initialize target_net with same weights

    # Adam optimizer: adjusts q_net's weights to minimize the TD loss
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)

    # Create the replay buffer to store past experiences
    rb = ReplayBuffer(args.buffer_size, obs_shape, device)

    # --- CSV Logging Setup ---
    # episodes.csv: One row per completed episode (return, length, goal reached, epsilon)
    # metrics.csv: One row per log_interval steps (loss, Q-values, stuck rate)
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

    # --- Rolling Statistics ---
    # deque with maxlen automatically discards old values, giving us a sliding window
    recent_goals = deque(maxlen=100)    # Track last 100 episodes' goal success (0 or 1)
    recent_stuck = deque(maxlen=1000)   # Track last 1000 steps' stuck status (0 or 1)
    recent_penalty = deque(maxlen=1000) # Track last 1000 steps' penalty values
    episode_return = 0.0   # Cumulative reward for the current episode
    episode_length = 0     # Number of steps in the current episode
    last_loss = np.nan     # Most recent TD loss value (for logging)
    last_q = np.nan        # Most recent mean Q-value (for logging)
    best_goal_rate = 0.0   # Highest goal rate achieved so far

    # --- Main Loop ---
    pbar = tqdm(range(args.total_timesteps), desc=args.exp_name)
    for global_step in pbar:
        # ---- EPSILON-GREEDY ACTION SELECTION ----
        # Calculate current exploration rate (decays linearly over training)
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        if random.random() < epsilon:
            # EXPLORE: Take a completely random action
            action = env.action_space.sample()
        else:
            # EXPLOIT: Use the neural network to pick the best action
            with torch.no_grad():  # Disable gradient computation (not training here)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                # unsqueeze(0) adds a batch dimension: shape (obs_dim,) -> (1, obs_dim)
                # argmax returns the index of the highest Q-value = the best action
                action = int(torch.argmax(q_net(obs_tensor), dim=1).item())

        # ---- ENVIRONMENT STEP ----
        # Take the action in the environment and observe the result
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        # terminated: True if the agent reached the goal (natural end)
        # truncated: True if the episode hit the maximum step limit (forced end)
        done = terminated or truncated

        # ---- REWARD SHAPING (optional) ----
        # Compare the current and next observations to detect if the agent is "stuck"
        # (e.g., walking into a wall, where the observation doesn't change)
        no_change = bool(np.array_equal(next_obs, obs))
        # Apply a small negative penalty if shaping is enabled and the agent didn't move
        penalty = args.stuck_penalty if use_shaping and no_change else 0.0
        # The reward used for training includes the penalty; the original env_reward is
        # used for tracking performance metrics (so plots show true environment reward)
        reward_for_learning = float(env_reward + penalty)

        recent_stuck.append(float(no_change))
        recent_penalty.append(float(penalty))

        # ---- STORE TRANSITION ----
        # Save the experience (s, s', a, r, done) to the replay buffer for later training
        rb.add(obs, next_obs, action, reward_for_learning, float(done))
        episode_return += float(env_reward)  # Track ORIGINAL reward (not shaped) for metrics
        episode_length += 1

        obs = next_obs

        # ---- END OF EPISODE ----
        if done:
            # In MiniGrid, reaching the goal gives a positive reward (> 0)
            # Not reaching the goal gives 0 reward (the episode times out)
            reached = episode_return > 0.0
            recent_goals.append(float(reached))
            goal_rate = float(np.mean(recent_goals)) if recent_goals else 0.0
            best_goal_rate = max(best_goal_rate, goal_rate)

            # Log episode metrics to CSV
            episode_writer.writerow(
                {
                    "global_step": global_step,
                    "episodic_return": episode_return,
                    "episodic_length": episode_length,
                    "goal_reached": int(reached),
                    "epsilon": epsilon,
                }
            )
            # Log to TensorBoard
            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)
            writer.add_scalar("charts/goal_rate", goal_rate, global_step)
            episode_file.flush()  # Force write to disk so data is available even if crashed

            # Update the progress bar with live statistics
            pbar.set_postfix(
                {
                    "return": f"{episode_return:.2f}",
                    "goal%": f"{goal_rate:.0%}",
                    "best%": f"{best_goal_rate:.0%}",
                    "eps": f"{epsilon:.2f}",
                }
            )
            # Reset for the next episode
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        # ---- NEURAL NETWORK TRAINING ----
        # Only start training after the buffer has enough random experiences,
        # and only train every train_frequency steps (not every single step)
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            # Sample a random mini-batch of transitions from the replay buffer
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)

            with torch.no_grad():
                # --- Compute TD Target ---
                # The Temporal Difference (TD) target is what we want Q(s, a) to be:
                #   y = r + gamma * max_a' Q_target(s', a') * (1 - done)
                # If the episode ended (done=1), the target is just r (no future rewards).
                # We use the TARGET network (not q_net) to compute max_a' Q(s', a')
                # because it provides more stable targets.
                target_max = target_net(b_next_obs).max(dim=1).values
                td_target = b_rewards + args.gamma * target_max * (1.0 - b_dones)

            # --- Compute Current Q-Values ---
            # q_net(b_obs) returns Q-values for ALL actions, shape (batch_size, num_actions)
            # .gather(1, ...) selects only the Q-value for the action that was actually taken
            # This gives us Q(s, a) for the specific (state, action) pairs in our batch
            old_val = q_net(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)

            # --- Compute Loss ---
            # MSE Loss: mean of (Q(s,a) - y)^2 across the batch
            # This measures how far our Q-value predictions are from the TD targets
            loss = F.mse_loss(old_val, td_target)

            # --- Backpropagation and Weight Update ---
            optimizer.zero_grad()   # Clear old gradients from the previous step
            loss.backward()         # Compute gradients of the loss w.r.t. network weights
            # Clip gradients to prevent "exploding gradients" (very large weight updates
            # that destabilize training). Max gradient norm of 10.0.
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()        # Update the network weights using the Adam optimizer

            last_loss = float(loss.item())
            last_q = float(old_val.mean().item())

        # ---- TARGET NETWORK UPDATE ----
        # Periodically copy q_net's weights to target_net (hard update).
        # This keeps the target network's Q-value estimates slowly tracking the
        # improving q_net, while remaining stable enough to provide good targets.
        if global_step % args.target_network_frequency == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ---- PERIODIC LOGGING ----
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
            writer.add_scalar("losses/td_loss", last_loss, global_step)
            writer.add_scalar("losses/q_values", last_q, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

    # --- Cleanup ---
    # Save the trained Q-Network weights so we can load them later for evaluation
    if args.save_model:
        torch.save(q_net.state_dict(), run_dir / "q_net.pt")

    episode_file.close()
    metric_file.close()
    writer.close()
    env.close()
    print(f"Done. Results: {run_dir}")
