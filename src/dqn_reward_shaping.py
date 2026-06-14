"""
dqn_reward_shaping.py - DQN Agent WITH Reward Shaping
======================================================
This script trains a Deep Q-Network (DQN) agent on a MiniGrid environment
WITH reward shaping enabled. In addition to the environment's sparse reward,
the agent receives a small negative penalty (default: -0.01) whenever its
observation does not change between steps.

When does the observation stay the same?
  - Walking into a wall (the agent stays in the same position)
  - Using "pickup" when there is nothing to pick up
  - Using "toggle" when there is no door to open

This penalty discourages the agent from wasting steps on futile actions
and encourages it to explore more effectively by trying different actions.

By comparing this agent's performance against dqn_baseline.py (no shaping),
we can evaluate whether this simple reward shaping technique improves
learning speed, goal success rate, or both.

Usage:
  python dqn_reward_shaping.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 50000

All command-line arguments are defined in dqn_common.parse_args().
"""

# Import the shared training infrastructure from dqn_common.py:
#   parse_args: Parses command-line arguments into a configuration namespace
#   train: The main DQN training loop (environment interaction, replay buffer, backprop)
from dqn_common import parse_args, train


if __name__ == "__main__":
    # parse_args() configures the experiment:
    #   default_exp_name="dqn_reward_shaping" -> used in run directory names and plot labels
    #   use_shaping=True -> tells the training loop to apply stuck penalties
    args = parse_args(default_exp_name="dqn_reward_shaping", use_shaping=True)

    # train() runs the full DQN training loop:
    #   args: all hyperparameters (env_id, timesteps, learning_rate, epsilon schedule, etc.)
    #   use_shaping=True: the agent receives a penalty when its observation doesn't change
    #     The penalty value is controlled by --stuck-penalty (default: -0.01)
    train(args, use_shaping=True)
