"""
dqn_vanilla.py - Standard DQN Agent (No Double DQN, No Reward Shaping)
=======================================================================
This script trains a standard (vanilla) Deep Q-Network agent on a MiniGrid
environment. It uses the original DQN algorithm where the TARGET network
both selects AND evaluates the best next action:

    target = r + γ * max_a Q_target(s', a)

This is known to suffer from Q-value overestimation because the max operator
applied to noisy Q-value estimates produces a positively biased estimate of
the true max value.

By comparing this agent's Q-values against ddqn_baseline.py (Double DQN),
we can directly measure the overestimation bias and demonstrate why Double
DQN was introduced as an improvement.

Usage:
  python dqn_vanilla.py --env-id MiniGrid-Empty-Random-6x6-v0 --total-timesteps 100000

All command-line arguments are defined in dqn_common.parse_args().
"""

# Import the shared training infrastructure from dqn_common.py:
#   parse_args: Parses command-line arguments into a configuration namespace
#   train: The main DQN training loop (environment interaction, replay buffer, backprop)
from dqn_common import parse_args, train


if __name__ == "__main__":
    # parse_args() configures the experiment:
    #   default_exp_name="dqn_vanilla" -> used in run directory names and plot labels
    #   use_shaping=False -> tells the training loop NOT to apply stuck penalties
    args = parse_args(default_exp_name="dqn_vanilla", use_shaping=False)

    # Standard DQN: do NOT set double_dqn flag.
    # args.double_dqn is already False by default (store_true).

    # train() runs the full DQN training loop:
    #   args: all hyperparameters (env_id, timesteps, learning_rate, epsilon schedule, etc.)
    #   use_shaping=False: the agent uses raw environment rewards only
    train(args, use_shaping=False)
