# MiniGrid DQN Reward Shaping

Clean, small DQN experiments for MiniGrid.

The goal is to compare:

- `dqn_baseline.py`: standard DQN, no reward shaping.
- `dqn_reward_shaping.py`: same DQN, same action space, with a small penalty only when a sampled action causes no observable state change.

For example, `pickup` with no key in front receives the stuck penalty, but `pickup` with a key in front changes the state and receives no penalty.

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Smoke Test

```powershell
python src\dqn_reward_shaping.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 2000 --save-model False
```

## Basic Comparison

Start with `MiniGrid-Empty-8x8-v0` because it should learn before DoorKey:

```powershell
python src\dqn_baseline.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000 --seed 1
python src\dqn_reward_shaping.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000 --seed 1
python src\plot_comparison.py --env-id MiniGrid-Empty-8x8-v0
```

For DoorKey, use more timesteps:

```powershell
python src\dqn_baseline.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 300000 --seed 1
python src\dqn_reward_shaping.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 300000 --seed 1
python src\plot_comparison.py --env-id MiniGrid-DoorKey-8x8-v0
```

Results are written to `results/`; plots are written to `plots/`.

For a more detailed command checklist, see `RUN_STEPS.md`.
