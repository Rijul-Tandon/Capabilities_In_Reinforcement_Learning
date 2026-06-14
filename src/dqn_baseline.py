"""
dqn_baseline.py - Baseline DQN Agent (No Reward Shaping)
=========================================================
This script trains a standard Deep Q-Network (DQN) agent on a MiniGrid environment
WITHOUT any reward shaping. The agent only receives the environment's original
sparse reward signal:
  - Positive reward (> 0) when it reaches the goal.
  - Zero reward for every other step.

This serves as the control experiment. By comparing its performance against
dqn_reward_shaping.py (which adds a stuck penalty), we can measure whether
reward shaping actually helps the agent learn faster or more reliably.

Usage:
  python dqn_baseline.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 30000

All command-line arguments are defined in dqn_common.parse_args().
"""

# Import the shared training infrastructure from dqn_common.py:
#   parse_args: Parses command-line arguments into a configuration namespace
#   train: The main DQN training loop (environment interaction, replay buffer, backprop)
from dqn_common import parse_args, train


if __name__ == "__main__":
    # parse_args() configures the experiment:
    #   default_exp_name="dqn_baseline" -> used in run directory names and plot labels
    #   use_shaping=False -> tells the training loop NOT to apply stuck penalties
    args = parse_args(default_exp_name="dqn_baseline", use_shaping=False)

    # train() runs the full DQN training loop:
    #   args: all hyperparameters (env_id, timesteps, learning_rate, epsilon schedule, etc.)
    #   use_shaping=False: the agent uses raw environment rewards only
    train(args, use_shaping=False)
