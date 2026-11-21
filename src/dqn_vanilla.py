"""Standard DQN on MiniGrid (no Double DQN, no reward shaping): target = r + gamma * max_a Q_target(s', a).
Its Q-values are compared with ddqn_baseline.py to measure overestimation.  All arguments: dqn_common.parse_args."""

from dqn_common import parse_args, train

if __name__ == "__main__":
    args = parse_args(default_exp_name="dqn_vanilla")
    train(args, double_dqn=False, penalty_type="none")
