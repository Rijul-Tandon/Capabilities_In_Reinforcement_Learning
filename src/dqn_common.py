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

# ImgObsWrapper (from minigrid.wrappers):
#   By default, MiniGrid returns observations as a Python dictionary containing:
#     - "image": a 3D numpy array of shape (width, height, 3) where each cell has
#       3 values: [object_type, color, state]. For example, a wall tile might be [2, 5, 0].
#     - "direction": an integer (0-3) for the agent's facing direction.
#     - "mission": a text string like "reach the goal".
#   Neural networks cannot process dictionaries or text easily. ImgObsWrapper extracts
#   ONLY the "image" array and drops the text description completely.
#
# FullyObsWrapper (from minigrid.wrappers):
#   By default, MiniGrid gives the agent a PARTIAL observation: a 7x7 grid of tiles
#   directly in front of the agent (like a flashlight cone). The agent cannot see
#   what is behind it or far away. This makes it a POMDP (Partially Observable
#   Markov Decision Process), which is much harder to solve.
#   FullyObsWrapper overrides this and gives the agent the ENTIRE map as its
#   observation. This converts the problem into a standard MDP, making it much
#   easier for a memoryless MLP to solve.
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper

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


def episode_success(env_id, episode_return):
    """Maps an episode return to a simple success flag."""
    return float(episode_return) > 0.0


def observation_unchanged(obs, next_obs, tolerance):
    """
    Detects whether a transition produced a meaningful observation change.
    For categorical MiniGrid observations we require exact equality.
    """
    obs_arr = np.asarray(obs, dtype=np.float32)
    next_obs_arr = np.asarray(next_obs, dtype=np.float32)
    if tolerance <= 0.0:
        return bool(np.array_equal(obs_arr, next_obs_arr))
    return bool(np.linalg.norm(next_obs_arr - obs_arr) <= tolerance)


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

    # Navigation-only environments: the agent just needs to move around.
    # FourRooms has open doorways (no lockable doors), just navigation.
    if "Empty" in env_id or "FourRooms" in env_id:
        return [0, 1, 2]  # left, right, forward

    # MultiRoom: rooms are separated by DOORS that must be toggled open.
    # No keys or objects to pick up — toggle is the only interaction needed.
    if "MultiRoom" in env_id:
        return [0, 1, 2, 5]  # left, right, forward, toggle

    # DoorKey: The agent picks up the key, toggles the door, and walks to the goal (can hold key).
    if "DoorKey" in env_id:
        return [0, 1, 2, 3, 5]  # left, right, forward, pickup, toggle

    # UnlockPickup: The agent MUST drop the key after unlocking the door to pick up the box.
    if "UnlockPickup" in env_id:
        return [0, 1, 2, 3, 4, 5]  # left, right, forward, pickup, drop, toggle

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


class MarkovianStepPenaltyWrapper(gym.RewardWrapper):
    """
    Fixed step penalty per action + fixed goal reward.
    Strictly Markovian: reward depends ONLY on (s, a, s').
    """
    def __init__(self, env, step_penalty=0.01, goal_reward=1.0):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

    def reward(self, reward):
        if reward > 0:  # Goal reached
            return self.goal_reward
        return -self.step_penalty  # Step penalty


class FlatImageAndDirectionWrapper(gym.ObservationWrapper):
    """
    A custom wrapper that takes the 3D image output from ImgObsWrapper,
    applies component-wise min-max normalization ([0, 1]) to each channel
    (object_type / 10.0, color / 5.0, state / 3.0), flattens it into a 1D array,
    and appends the normalized agent direction (direction / 3.0).
    """
    def __init__(self, env):
        super().__init__(env)
        image_shape = env.observation_space.shape
        flat_size = int(np.prod(image_shape))
        
        # Observation space bounds are now normalized between 0.0 and 1.0
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(flat_size + 1,),
            dtype=np.float32
        )

    def observation(self, obs):
        # 'obs' is the 3D image array from ImgObsWrapper of shape (width, height, 3)
        img = obs.astype(np.float32)
        
        # Component-wise min-max scaling to [0, 1]
        img[:, :, 0] /= 10.0  # Object type (max 10)
        img[:, :, 1] /= 5.0   # Color (max 5)
        img[:, :, 2] /= 3.0   # State (max 3)
        
        flat_image = img.flatten()
        
        # Normalize agent direction (0..3) to [0, 1]
        direction = np.array([self.env.unwrapped.agent_dir / 3.0], dtype=np.float32)
        
        return np.concatenate([flat_image, direction])


def make_env(env_id, seed, action_set, capture_video=False, run_name="", max_steps=None):
    """
    Creates a MiniGrid environment and applies the wrappers needed by our DQN pipeline.

    Parameters
    ----------
    max_steps : int or None
        Override MiniGrid's built-in episode step limit. MiniGrid's default is
        4 * width * height (e.g. 100 for 5x5, 256 for 8x8). Pass a larger value
        (e.g. 500) to give the agent more time per episode in harder environments.
        If None, the environment's own default is used.
    """
    make_kwargs = {}
    if max_steps is not None and max_steps > 0:
        make_kwargs["max_steps"] = max_steps

    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array", **make_kwargs)
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id, **make_kwargs)

    env = MarkovianStepPenaltyWrapper(env)

    action_map = minigrid_action_map(env_id, action_set)
    if action_map is not None:
        env = MiniGridActionSubsetWrapper(env, action_map)

    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = FlatImageAndDirectionWrapper(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def action_names(env_id, action_set, action_space_n):
    """Returns human-readable action names for logging and debug prints."""
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
    Computes the current epsilon value for epsilon-greedy exploration using a linear decay.
    """
    slope = (end_e - start_e) / duration
    return max(slope * step + start_e, end_e)

def polynomial_schedule(start_e, end_e, duration, step, power=4.0):
    """
    Computes the current epsilon value for epsilon-greedy exploration using a polynomial decay.
    
    Decays epsilon slowly at first, then increasingly fast as t approaches `duration`.
    power=1.0 reduces to your existing linear_schedule.
    power=2-4 gives a pronounced slow-start/fast-finish curve.
    """
    frac = min(step / duration, 1.0)
    decay = (1.0 - frac) ** power
    return end_e + (start_e - end_e) * decay


def hardcoded_schedule(total_timesteps, step):
    """
    Hardcoded piecewise decay:
      - 1.0 for 0-10% of total steps
      - 0.7 for 10%-50% of total steps
      - 0.1 for 50%+ steps
    """
    frac = step / total_timesteps
    if frac < 0.10:
        return 1.0
    elif frac < 0.50:
        return 0.7
    else:
        return 0.1


def cosine_schedule(start_e, end_e, duration, step):
    """
    Cosine annealing decay from start_e to end_e over `duration` steps.
    """
    frac = min(step / duration, 1.0)
    decay = 0.5 * (1.0 + np.cos(np.pi * frac))
    return end_e + (start_e - end_e) * decay


def exponential_schedule(start_e, end_e, duration, step):
    """
    Exponential decay from start_e to end_e over `duration` steps.
    """
    frac = min(step / duration, 1.0)
    return start_e * ((end_e / start_e) ** frac)


def cyclic_schedule(start_e, end_e, total_timesteps, step, num_cycles=3):
    """
    Cyclic / Warm-restart decay:
    Repeatedly decays epsilon using cosine annealing over `num_cycles` periods,
    periodically boosting exploration back up to escape local minima.
    """
    cycle_length = total_timesteps / num_cycles
    cycle_step = step % cycle_length
    frac = cycle_step / cycle_length
    decay = 0.5 * (1.0 + np.cos(np.pi * frac))
    return end_e + (start_e - end_e) * decay


def softmax_tau_schedule(total_timesteps, step):
    """
    Computes the current temperature (tau) for softmax exploration using a simple linear decay.
    tau = max(1.0 * (1 - step / (total_timesteps * 0.75)), 0.1)
    At the start tau=1.0 (high randomness); decays linearly to a minimum of 0.1.
    """
    return max(1.0 * (1.0 - step / (total_timesteps * 0.75)), 0.1)

def softmax_action(q_values_tensor, tau):
    """
    Selects an action by sampling from a softmax distribution over Q-values.
    Lower tau -> distribution closer to greedy; higher tau -> more uniform (more exploration).
    """
    probs = torch.softmax(q_values_tensor / tau, dim=1)
    return int(torch.multinomial(probs, num_samples=1).item())


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
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-6x6-v0")
    # --seed: Random seed for reproducibility (same seed = same results)
    parser.add_argument("--seed", type=int, default=1)

    # --- Training Duration ---
    # --total-timesteps: How many environment steps to train for
    parser.add_argument("--total-timesteps", type=int, default=200000)

    # --- DQN Hyperparameters ---
    # --learning-rate: How fast the neural network adjusts its weights.
    #   Too high = unstable training. Too low = slow learning. 2.5e-4 is a good default.
    parser.add_argument("--learning-rate", type=float, default=0.00171)
    # --buffer-size: Maximum number of transitions to store in the replay buffer.
    parser.add_argument("--buffer-size", type=int, default=30000)
    # --gamma: Discount factor for future rewards (0 = greedy, 1 = far-sighted).
    parser.add_argument("--gamma", type=float, default=0.95)

    # --fixed-layout: If set, the environment uses the same seed on every single reset.
    #   This forces the procedural generation to create the exact same map layout every episode,
    #   allowing the agent to overfit to a single map (great for debugging if learning works at all).
    parser.add_argument("--fixed-layout", action="store_true")

    # --double-dqn: If set, uses Double DQN for the TD target computation.
    #   Standard DQN: target = r + γ * max_a Q_target(s', a)  (target net selects AND evaluates)
    #   Double DQN:   target = r + γ * Q_target(s', argmax_a Q_online(s', a))  (online selects, target evaluates)
    #   Double DQN reduces the overestimation bias inherent in standard DQN.
    parser.add_argument("--double-dqn", action="store_true")

    # --- Target Network Parameters ---quency: How often (in steps) to copy q_net weights to target_net.
    #   The target network provides stable Q-value targets during training.
    parser.add_argument("--target-network-frequency", type=int, default=264)
    # --batch-size: Number of transitions sampled from the replay buffer per training step
    parser.add_argument("--batch-size", type=int, default=128)
    # --learning-starts: Number of random steps before training begins.
    #   This seeds the replay buffer with diverse experiences before the network starts learning.
    parser.add_argument("--learning-starts", type=int, default=2000)
    # --train-frequency: Train the network every N environment steps (not every single step).
    #   CleanRL uses 10 here. Training every single step (=1) causes correlated updates
    #   because consecutive steps come from very similar states, destabilising learning.
    parser.add_argument("--train-frequency", type=int, default=10)
    # --max-steps: Maximum steps per episode. Overrides MiniGrid's built-in limit.
    #   MiniGrid defaults: 4*width*height (100 for 5x5, 256 for 8x8, 400 for FourRooms).
    #   Increasing this gives the agent more time to find the goal in harder environments
    #   (e.g. DoorKey, FourRooms) where the default limit may truncate too aggressively.
    #   Set to -1 to use each environment's own built-in default.
    parser.add_argument("--max-steps", type=int, default=-1)

    # --- Exploration Schedule ---
    # --start-e: Initial epsilon (exploration rate). 1.0 = 100% random actions at the start.
    parser.add_argument("--start-e", type=float, default=1.0)
    # --end-e: Minimum epsilon. 0.0 = completely greedy at the end.
    #   This prevents the agent from getting stuck in local optima, especially in
    #   environments with randomized layouts (DoorKey, FourRooms) where the agent
    #   needs to keep exploring to handle new configurations.
    parser.add_argument("--end-e", type=float, default=0.1)
    # --exploration-fraction: What fraction of training to decay epsilon over
    parser.add_argument("--exploration-fraction", type=float, default=0.50)
    
    # --epsilon-schedule: Which decay schedule to use for epsilon
    parser.add_argument("--epsilon-schedule", choices=["linear", "polynomial", "hardcoded", "cosine", "exponential", "cyclic"], default="polynomial")

    # --exploration-strategy: Whether to use epsilon-greedy or softmax exploration.
    #   epsilon_greedy: standard random action with probability epsilon.
    #   softmax: samples action from softmax(Q/tau) where tau decays linearly.
    parser.add_argument("--exploration-strategy", choices=["epsilon_greedy", "softmax"], default="epsilon_greedy")

    # --- Hyperparameters ---
    # --hidden-size: Number of neurons in each hidden layer of the Q-Network
    parser.add_argument("--hidden-size", type=int, default=256)

    # --- Action Space ---
    # --action-set: "task" uses only environment-relevant actions, "full" uses all 7
    parser.add_argument("--action-set", choices=["task", "full"], default="task")

    # --- Hardware ---
    # --cuda: Whether to use GPU acceleration (if available)
    parser.add_argument("--cuda", type=lambda x: str(x).lower() == "true", default=True)

    # --- Model Saving & Loading ---
    # --save-model: Whether to save the trained Q-Network weights to disk as q_net.pt
    parser.add_argument("--save-model", type=lambda x: str(x).lower() == "true", default=True)
    # --load-model: Path to a pre-trained q_net.pt to load before training
    parser.add_argument("--load-model", type=str, default="")
    # --- Reward Shaping ---
    # --stuck-penalty: Negative reward applied when the agent's observation doesn't change
    parser.add_argument("--stuck-penalty", type=float, default=-1)

    # --- Logging ---
    # --log-interval: How often (in steps) to write training metrics to CSV and TensorBoard
    parser.add_argument("--log-interval", type=int, default=1000)
    # --results-dir: Parent directory where run folders are created
    parser.add_argument("--results-dir", type=str, default="results")
    # --run-dir: Explicit directory to save/append to
    parser.add_argument("--run-dir", type=str, default="")
    # --global-step-offset: Starting step number
    parser.add_argument("--global-step-offset", type=int, default=0)

    # --- Optional: Weights & Biases Integration ---
    # --track: Enable Weights & Biases (wandb) logging for cloud-based experiment tracking
    parser.add_argument("--track", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL")
    parser.add_argument("--wandb-entity", type=str, default=None)

    # --- Optional: Video Recording ---
    parser.add_argument("--capture-video", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--no-change-tolerance", type=float, default=None)

    args = parser.parse_args()
    if args.no_change_tolerance is None:
        args.no_change_tolerance = 0.0
    
        
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
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_name = run_dir.name
    else:
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
    # Pass max_steps only if the user explicitly set it (i.e. not the sentinel -1).
    max_steps_override = args.max_steps if args.max_steps > 0 else None
    env = make_env(
        args.env_id, 
        args.seed, 
        args.action_set, 
        args.capture_video, 
        run_name, 
        max_steps_override,
    )
    obs, _ = env.reset(seed=args.seed)

    # Get observation and action space dimensions from the wrapped environment
    obs_shape = env.observation_space.shape   # e.g., (192,) for 8x8 fully observable
    obs_dim = int(np.prod(obs_shape))         # Total number of input features
    num_actions = env.action_space.n          # Number of available actions
    names = action_names(args.env_id, args.action_set, num_actions)

    print(f"[{args.exp_name}] env={args.env_id}")
    print(f"[{args.exp_name}] State Space Size: {obs_dim} features")
    print(f"[{args.exp_name}] Action Space Size: {num_actions} actions {names}")
    print(f"[{args.exp_name}] device={device}")

    # --- Neural Networks ---
    # DQN uses TWO copies of the same network:
    #   q_net (online network): The network being actively trained. Used for action selection.
    #   target_net (target network): A frozen copy used to compute stable Q-value targets.
    #     Without this, training is unstable because the target Q-values shift every update.
    #     The target network is periodically updated by copying weights from q_net.
    q_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    target_net = QNetwork(obs_dim, num_actions, args.hidden_size).to(device)
    
    # If loading a pre-trained model (e.g. for curriculum learning across stages)
    if args.load_model:
        print(f"[{args.exp_name}] Loading pre-trained model from {args.load_model}")
        q_net.load_state_dict(torch.load(args.load_model, map_location=device))
        
        # When continuing from a previous stage, we shouldn't spend thousands of steps
        # doing purely random actions, because the network is already trained.
        args.learning_starts = 0
        
        # We also shouldn't start epsilon at 1.0. A small bump (e.g., 0.3) helps it 
        # explore the new difficulty without forgetting its old knowledge.
        args.start_e = 0.3

    target_net.load_state_dict(q_net.state_dict())  # Initialize target_net with same weights

    # Adam optimizer: adjusts q_net's weights to minimize the TD loss
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)

    # Create the replay buffer to store past experience
    rb = ReplayBuffer(args.buffer_size, obs_shape, device)

    # --- State-Action Frequency Tracking ---
    # We track how many times each action was taken in each (x, y, dir) state per stage.
    num_stages = 3
    counts_file = run_dir / "state_action_counts.npy"
    if args.run_dir and counts_file.exists():
        state_action_counts = np.load(counts_file)
        state_action_counts_last_quarter = np.load(run_dir / "state_action_counts_last_quarter.npy")
        state_action_counts_last_half = np.load(run_dir / "state_action_counts_last_half.npy")
        if state_action_counts.ndim == 4:
            state_action_counts = np.expand_dims(state_action_counts, axis=-1)
            state_action_counts_last_quarter = np.expand_dims(state_action_counts_last_quarter, axis=-1)
            state_action_counts_last_half = np.expand_dims(state_action_counts_last_half, axis=-1)
    else:
        state_action_counts = np.zeros(
            (env.unwrapped.width, env.unwrapped.height, 4, num_actions, num_stages), 
            dtype=np.int64
        )
        state_action_counts_last_quarter = np.zeros(
            (env.unwrapped.width, env.unwrapped.height, 4, num_actions, num_stages), 
            dtype=np.int64
        )
        state_action_counts_last_half = np.zeros(
            (env.unwrapped.width, env.unwrapped.height, 4, num_actions, num_stages), 
            dtype=np.int64
        )

    # --- CSV Logging Setup ---
    # episodes.csv: One row per completed episode (return, length, goal reached, epsilon)
    # metrics.csv: One row per log_interval steps (loss, Q-values, stuck rate)
    file_mode = "a" if args.run_dir and (run_dir / "episodes.csv").exists() else "w"
    episode_file = open(run_dir / "episodes.csv", file_mode, newline="", encoding="utf-8")
    metric_file = open(run_dir / "metrics.csv", file_mode, newline="", encoding="utf-8")
    episode_writer = csv.DictWriter(
        episode_file,
        fieldnames=["global_step", "episodic_return", "episodic_length", "goal_reached", "epsilon"],
    )
    metric_writer = csv.DictWriter(
        metric_file,
        fieldnames=["global_step", "epsilon", "td_loss", "q_value", "max_q", "stuck_rate", "mean_penalty"],
    )
    if file_mode == "w":
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
    last_q = 0.0           # Most recent mean Q-value (for logging)
    last_max_q = 0.0       # Most recent max Q-value in batch (for overestimation tracking)
    best_goal_rate = 0.0   # Highest goal rate achieved so far

    has_picked_key = False
    has_opened_door = False

    # --- Main Loop ---
    pbar = tqdm(range(args.total_timesteps), desc=args.exp_name)
    for local_step in pbar:
        global_step = local_step + args.global_step_offset
        
        # ---- ACTION SELECTION ----
        # Either epsilon-greedy or softmax, depending on --exploration-strategy.
        was_random = False
        if args.exploration_strategy == "softmax":
            # Softmax exploration: sample from softmax(Q/tau)
            tau = softmax_tau_schedule(args.total_timesteps, local_step)
            epsilon = tau  # log tau as "epsilon" for CSV/TensorBoard continuity
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(obs_tensor)
                action = softmax_action(q_values, tau)
        else:
            # Epsilon-greedy: calculate current exploration rate
            duration = args.exploration_fraction * args.total_timesteps
            if args.epsilon_schedule == "linear":
                epsilon = linear_schedule(args.start_e, args.end_e, duration, local_step)
            elif args.epsilon_schedule == "polynomial":
                epsilon = polynomial_schedule(args.start_e, args.end_e, duration, local_step, power=3.0)
            elif args.epsilon_schedule == "hardcoded":
                epsilon = hardcoded_schedule(args.total_timesteps, local_step)
            elif args.epsilon_schedule == "cosine":
                epsilon = cosine_schedule(args.start_e, args.end_e, duration, local_step)
            elif args.epsilon_schedule == "exponential":
                epsilon = exponential_schedule(args.start_e, args.end_e, duration, local_step)
            elif args.epsilon_schedule == "cyclic":
                epsilon = cyclic_schedule(args.start_e, args.end_e, args.total_timesteps, local_step)
            else:
                epsilon = linear_schedule(args.start_e, args.end_e, duration, local_step)

            if random.random() < epsilon:
                # Explore: pick a random action
                action = env.action_space.sample()
                was_random = True
            else:
                # Exploit: pure greedy action choice (argmax Q-value)
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = q_net(obs_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())

        # Determine current stage monotonically for the episode
        carrying = getattr(env.unwrapped, 'carrying', None)
        if carrying is not None and getattr(carrying, 'type', None) == "key":
            has_picked_key = True
        
        if not has_opened_door:
            grid = env.unwrapped.grid
            for wx in range(env.unwrapped.width):
                for wy in range(env.unwrapped.height):
                    c = grid.get(wx, wy)
                    if c is not None and getattr(c, 'type', None) == "door" and getattr(c, 'is_open', False):
                        has_opened_door = True
                        break
                if has_opened_door:
                    break

        if has_opened_door:
            stage_idx = 2
        elif has_picked_key:
            stage_idx = 1
        else:
            stage_idx = 0

        # Track the action taken at the current state per stage
        ax, ay = env.unwrapped.agent_pos
        ad = env.unwrapped.agent_dir
        state_action_counts[ax, ay, ad, action, stage_idx] += 1
        if local_step >= args.total_timesteps * 0.50:
            state_action_counts_last_half[ax, ay, ad, action, stage_idx] += 1
        if local_step >= args.total_timesteps * 0.75:
            state_action_counts_last_quarter[ax, ay, ad, action, stage_idx] += 1

        # Execute the chosen action in the environment
        next_obs, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ---- REWARD SHAPING ----
        # Compare the current and next observations to detect if the agent is "stuck"
        # (e.g., walking into a wall, where the observation doesn't change)
        no_change = observation_unchanged(obs, next_obs, args.no_change_tolerance)
        # Apply a negative penalty if shaping is enabled and the agent did not move.
        # Penalizing all no-change transitions, including exploratory ones, lets the
        # replay buffer teach the Q-network that the action is bad in this state.
        penalty = args.stuck_penalty if use_shaping and no_change else 0.0
        # The reward used for training includes the penalty; the original env_reward is
        # used for tracking performance metrics (so plots show true environment reward)
        reward_for_learning = float(env_reward + penalty)

        recent_stuck.append(float(no_change))
        recent_penalty.append(float(penalty))

        # ---- STORE TRANSITION ----
        # Save the experience (s, s', a, r, s'_terminal) to the replay buffer.
        #
        # IMPORTANT: we store `terminated`, NOT `done` (= terminated OR truncated).
        #
        # Why this matters (truncation vs. termination):
        #   - terminated=True  → the agent reached a genuine end-state (e.g. goal tile).
        #                        s' has no future value → TD target = r  (no bootstrap).
        #   - truncated=True   → the episode hit the step-limit; s' is a normal state.
        #                        The agent *could* keep going → TD target = r + γ·max Q(s').
        #
        # The TD-target formula is:  r + γ · max Q(s') · (1 - done_flag)
        # If we store `done` (terminated OR truncated) the (1-done_flag) term kills the
        # bootstrap on every timeout, making productive states look valueless.
        # Storing only `terminated` keeps the bootstrap alive for timeout transitions.
        rb.add(obs, next_obs, action, reward_for_learning, float(terminated))
        episode_return += float(env_reward)  # Track ORIGINAL reward (not shaped) for metrics
        episode_length += 1

        obs = next_obs

        # ---- END OF EPISODE ----
        if done:
            # In MiniGrid, reaching the goal gives a positive reward (> 0)
            # Not reaching the goal gives 0 reward (the episode times out)
            reached = episode_success(args.env_id, episode_return)
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
            # If fixed-layout is enabled, we pass the same seed again so the layout doesn't change.
            reset_kwargs = {"seed": args.seed} if args.fixed_layout else {}
            obs, _ = env.reset(**reset_kwargs)
            has_picked_key = False
            has_opened_door = False
            episode_return = 0.0
            episode_length = 0

        # ---- NEURAL NETWORK TRAINING ----
        # Only start training after the buffer has enough random experiences,
        # and only train every train_frequency steps (not every single step)
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            # Sample a random mini-batch of transitions from the replay buffer
            b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)

            with torch.no_grad():
                if args.double_dqn:
                    # --- Double DQN: Decoupled Selection & Evaluation ---
                    # The ONLINE network selects the best action for the next state,
                    # but the TARGET network evaluates its value.
                    # This prevents the overestimation bias present in standard DQN.
                    best_next_actions = q_net(b_next_obs).argmax(dim=1, keepdim=True)
                    target_max = target_net(b_next_obs).gather(1, best_next_actions).squeeze(1)
                else:
                    # --- Standard DQN: Single-Network Max ---
                    # The TARGET network both selects AND evaluates the best action.
                    # This is prone to overestimation because max over noisy estimates
                    # is a biased estimator of the true max value.
                    target_max = target_net(b_next_obs).max(dim=1).values

                # Compute the TD target: r + γ * Q(s', a*) * (1 - done)
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
            last_max_q = float(old_val.max().item())

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
                    "max_q": last_max_q,
                    "stuck_rate": float(np.mean(recent_stuck)) if recent_stuck else 0.0,
                    "mean_penalty": float(np.mean(recent_penalty)) if recent_penalty else 0.0,
                }
            )
            metric_file.flush()
            writer.add_scalar("losses/td_loss", last_loss, global_step)
            writer.add_scalar("losses/q_values", last_q, global_step)
            writer.add_scalar("losses/max_q", last_max_q, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

    # --- Cleanup ---
    # Save the trained Q-Network weights so we can load them later for evaluation
    if args.save_model:
        torch.save(q_net.state_dict(), run_dir / "q_net.pt")

    episode_file.close()
    metric_file.close()
    # Save the cumulative state-action counts
    np.save(run_dir / "state_action_counts.npy", state_action_counts)
    np.save(run_dir / "state_action_counts_last_half.npy", state_action_counts_last_half)
    np.save(run_dir / "state_action_counts_last_quarter.npy", state_action_counts_last_quarter)

    writer.close()
    env.close()
    print(f"Done. Results: {run_dir}")

    # --- Return final performance ---
    # Read the episode log back and compute mean goal_reached over the last 20% of training.
    # This is the objective Optuna will maximize.
    import csv as _csv
    episode_rows = []
    with open(run_dir / "episodes.csv", newline="") as ef:
        for row in _csv.DictReader(ef):
            episode_rows.append(float(row["goal_reached"]))
    if len(episode_rows) == 0:
        return 0.0
    cutoff = min(len(episode_rows) - 1, int(len(episode_rows) * 0.80))
    return float(np.mean(episode_rows[cutoff:]))
