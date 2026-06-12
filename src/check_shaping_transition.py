import argparse

import numpy as np

from dqn_common import make_env


def find_key(env):
    unwrapped = env.unwrapped
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is not None and cell.type == "key":
                return x, y
    raise RuntimeError("No key found in this environment.")


def place_agent_facing_key(env, key_pos):
    unwrapped = env.unwrapped
    key_x, key_y = key_pos
    candidates = [
        (key_x - 1, key_y, 0),  # west of key, facing east
        (key_x + 1, key_y, 2),  # east of key, facing west
        (key_x, key_y - 1, 1),  # north of key, facing south
        (key_x, key_y + 1, 3),  # south of key, facing north
    ]
    for x, y, direction in candidates:
        if unwrapped.grid.get(x, y) is None:
            unwrapped.agent_pos = (x, y)
            unwrapped.agent_dir = direction
            unwrapped.carrying = None
            return
    raise RuntimeError("Could not place the agent next to the key.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--action-set", choices=["task", "full"], default="task")
    args = parser.parse_args()

    env = make_env(args.env_id, args.seed, args.action_set)

    obs, _ = env.reset(seed=args.seed)
    pickup_action = 3 if args.action_set == "task" else 3
    next_obs, _, _, _, _ = env.step(pickup_action)
    no_key_no_change = bool(np.array_equal(obs, next_obs))

    obs, _ = env.reset(seed=args.seed + 1)
    place_agent_facing_key(env, find_key(env))
    obs = env.observation(env.unwrapped.gen_obs())
    next_obs, _, _, _, _ = env.step(pickup_action)
    key_present_no_change = bool(np.array_equal(obs, next_obs))
    carrying = getattr(env.unwrapped.carrying, "type", None)

    env.close()

    print(f"pickup with no key -> no_change={no_key_no_change}")
    print(f"pickup with key    -> no_change={key_present_no_change}, carrying={carrying}")
    print()
    if no_key_no_change and not key_present_no_change and carrying == "key":
        print("OK: pickup is penalized only in the no-key state.")
    else:
        raise SystemExit("FAILED: pickup transition check did not match the expected behavior.")


if __name__ == "__main__":
    main()
