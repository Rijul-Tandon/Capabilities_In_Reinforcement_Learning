"""Double DQN on MiniGrid with the fixed stuck penalty: --stuck-penalty is added to the reward whenever the
observation does not change (e.g. the agent walked into a wall).  All arguments: dqn_common.parse_args."""

from dqn_common import parse_args, train

if __name__ == "__main__":
    args = parse_args(default_exp_name="ddqn_reward_shaping")
    train(args, double_dqn=True, penalty_type="stuck")
