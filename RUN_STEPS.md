# Run Steps

Detailed command checklist for running experiments.

## 1. Create and Activate the Virtual Environment

```powershell
python -m venv venv_capabilities
.\venv_capabilities\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv_capabilities\Scripts\Activate.ps1
```

## 2. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Run a Small Smoke Test

```powershell
python src\dqn_baseline.py --env-id MiniGrid-Empty-5x5-v0 --total-timesteps 5000 --seed 1 --save-model False
python src\dqn_reward_shaping.py --env-id MiniGrid-Empty-5x5-v0 --total-timesteps 5000 --seed 1 --save-model False
python src\plot_comparison.py --env-id MiniGrid-Empty-5x5-v0
```

## 4. Run Full Experiments (Automated)

The `run_comparison.ps1` script runs all three agents (random, baseline, reward shaped) and generates the comparison plot automatically:

```powershell
.\run_comparison.ps1 -EnvId "MiniGrid-Empty-5x5-v0" -Timesteps 30000
.\run_comparison.ps1 -EnvId "MiniGrid-DoorKey-5x5-v0" -Timesteps 50000
.\run_comparison.ps1 -EnvId "MiniGrid-UnlockPickup-v0" -Timesteps 100000
```

**Tip:** Open 3 separate terminals (activate the venv in each) and run one environment per terminal to train them in parallel.

## 5. Run Full Experiments (Manual)

### Empty (easiest — fixed layout, navigation only)

```powershell
python src\random_agent.py --env-id MiniGrid-Empty-5x5-v0 --total-timesteps 30000 --seed 1
python src\dqn_baseline.py --env-id MiniGrid-Empty-5x5-v0 --total-timesteps 30000 --seed 1
python src\dqn_reward_shaping.py --env-id MiniGrid-Empty-5x5-v0 --total-timesteps 30000 --seed 1
python src\plot_comparison.py --env-id MiniGrid-Empty-5x5-v0
python src\plot_state_action_freq.py --env-id MiniGrid-Empty-5x5-v0
```

### DoorKey (hardest — randomized layout, requires key + door interaction)

```powershell
python src\dqn_baseline.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 150000 --seed 1
python src\dqn_reward_shaping.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 150000 --seed 1
python src\plot_comparison.py --env-id MiniGrid-DoorKey-8x8-v0
python src\plot_state_action_freq.py --env-id MiniGrid-DoorKey-8x8-v0
```

### FourRooms (medium — large fixed map, randomized start and goal)

```powershell
python src\random_agent.py --env-id MiniGrid-FourRooms-v0 --total-timesteps 100000 --seed 1
python src\dqn_baseline.py --env-id MiniGrid-FourRooms-v0 --total-timesteps 100000 --seed 1
python src\dqn_reward_shaping.py --env-id MiniGrid-FourRooms-v0 --total-timesteps 100000 --seed 1
python src\plot_comparison.py --env-id MiniGrid-FourRooms-v0
python src\plot_state_action_freq.py --env-id MiniGrid-FourRooms-v0
```

## Notes

- All three agents use the same environment wrappers (FullyObsWrapper + FlatObsWrapper + FlatObsWithDirectionWrapper) and the same action subsets for fair comparison.
- Default `--action-set task` uses:
  - Empty / FourRooms: `left`, `right`, `forward` (3 actions)
  - DoorKey / UnlockPickup: `left`, `right`, `forward`, `pickup`, `toggle` (5 actions)
- Use `--action-set full` if you want all seven MiniGrid actions.
- The shaped agent uses `--stuck-penalty -0.01` by default.
- Minimum epsilon is `0.3` for all agents (never drops below 30% exploration).
- Environment return is logged separately from shaped learning reward, so comparison plots use the original MiniGrid reward.
- The plotting scripts automatically find the **latest** run for each experiment type, so re-running training and then plotting will always show the newest results.
