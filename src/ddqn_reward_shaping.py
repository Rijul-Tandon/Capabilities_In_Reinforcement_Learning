"""
ddqn_reward_shaping.py - Double DQN Agent WITH Reward Shaping
=============================================================
This script trains a Double DQN (DDQN) agent on a MiniGrid environment
WITH reward shaping enabled. In addition to the environment's sparse reward,
the agent receives a negative penalty (default: -0.10) whenever its
observation does not change between steps.

Usage:
  python ddqn_reward_shaping.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 50000

All command-line arguments are defined in dqn_common.parse_args().
"""

from dqn_common import parse_args, train

if __name__ == "__main__":
    args = parse_args(default_exp_name="ddqn_reward_shaping", use_shaping=True)
    args.double_dqn = True
    train(args, use_shaping=True)
