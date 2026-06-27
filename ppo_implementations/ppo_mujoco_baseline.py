from ppo_mujoco_common import parse_args, train


if __name__ == "__main__":
    args = parse_args(default_exp_name="ppo_mujoco_baseline", use_shaping=False)
    train(args, use_shaping=False)
