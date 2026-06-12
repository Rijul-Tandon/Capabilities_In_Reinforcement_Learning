from dqn_common import parse_args, train


if __name__ == "__main__":
    args = parse_args(default_exp_name="dqn_reward_shaping", use_shaping=True)
    train(args, use_shaping=True)
