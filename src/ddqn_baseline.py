"""
ddqn_baseline.py - Double DQN Baseline Agent (No Reward Shaping)
================================================================
This script trains a Double DQN (DDQN) agent on a MiniGrid environment.
It uses the Double DQN algorithm where the ONLINE network selects the best
next action, and the TARGET network evaluates its Q-value:

    target = r + γ * Q_target(s', argmax_a Q_online(s', a))

This decouples action selection from action evaluation, mitigating the Q-value
overestimation bias inherent in standard DQN.

Usage:
  python ddqn_baseline.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 30000

All command-line arguments are defined in dqn_common.parse_args().
"""

from dqn_common import parse_args, train

if __name__ == "__main__":
    args = parse_args(default_exp_name="ddqn_baseline", use_shaping=False)
    args.double_dqn = True
    train(args, use_shaping=False)
