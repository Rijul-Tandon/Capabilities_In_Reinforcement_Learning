"""
dqn_baseline.py - Double DQN Baseline Agent (No Reward Shaping)
================================================================
This script trains a Double DQN (DDQN) agent on a MiniGrid environment
WITHOUT any reward shaping. The agent only receives the environment's original
sparse reward signal:
  - Positive reward (> 0) when it reaches the goal.
  - Zero reward for every other step.

Double DQN decouples action selection from action evaluation in the TD target:
    a* = argmax_a Q_online(s', a)       (online network SELECTS)
    target = r + γ * Q_target(s', a*)   (target network EVALUATES)

This reduces the overestimation bias present in standard DQN. By comparing
this agent against dqn_vanilla.py (standard DQN), we can measure the
overestimation. By comparing against dqn_reward_shaping.py (DDQN with stuck
penalty), we can measure whether reward shaping helps.

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
    #   default_exp_name="ddqn_baseline" -> used in run directory names and plot labels
    #   use_shaping=False -> tells the training loop NOT to apply stuck penalties
    args = parse_args(default_exp_name="ddqn_baseline", use_shaping=False)

    # Enable Double DQN for this agent.
    # This uses the online network to select the best next action and the
    # target network to evaluate it, reducing overestimation bias.
    args.double_dqn = True

    # train() runs the full DQN training loop:
    #   args: all hyperparameters (env_id, timesteps, learning_rate, epsilon schedule, etc.)
    #   use_shaping=False: the agent uses raw environment rewards only
    train(args, use_shaping=False)
