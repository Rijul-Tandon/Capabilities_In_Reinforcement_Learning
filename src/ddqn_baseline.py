"""Double DQN on MiniGrid without reward shaping: target = r + gamma * Q_target(s', argmax_a Q_online(s', a)).
Run: python src/ddqn_baseline.py --env-id MiniGrid-DoorKey-6x6-v0   (all arguments: dqn_common.parse_args)"""

from dqn_common import parse_args, train

if __name__ == "__main__":
    args = parse_args(default_exp_name="ddqn_baseline")
    train(args, double_dqn=True, penalty_type="none")
