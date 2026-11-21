"""Double DQN on MiniGrid with a learned penalty: a one-layer head scores every transition s -> s', and
--penalty-scale times that score is subtracted from the reward.  The head is explained in learned_penalty.py."""

from dqn_common import parse_args, train

if __name__ == "__main__":
    args = parse_args(default_exp_name="ddqn_learned_penalty")
    train(args, double_dqn=True, penalty_type="learned")
