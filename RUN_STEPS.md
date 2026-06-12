# Run Steps

Use these commands from:

```powershell
cd C:\Users\letsc\Desktop\cleanrl\cleanrl\minigrid_dqn_shaping
```

## 1. Create and Activate the Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Verify the State-Specific Shaping Logic

```powershell
python src\check_shaping_transition.py --env-id MiniGrid-DoorKey-8x8-v0
```

Expected output:

```text
pickup with no key -> no_change=True
pickup with key    -> no_change=False, carrying=key
OK: pickup is penalized only in the no-key state.
```

## 4. Run a Small Smoke Test

```powershell
python src\dqn_baseline.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 5000 --seed 1 --save-model False
python src\dqn_reward_shaping.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 5000 --seed 1 --save-model False
python src\plot_comparison.py --env-id MiniGrid-Empty-8x8-v0
```

## 5. Run the First Real Comparison

Start with Empty because the baseline should learn there before DoorKey:

```powershell
python src\dqn_baseline.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000 --seed 1
python src\dqn_reward_shaping.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000 --seed 1
python src\plot_comparison.py --env-id MiniGrid-Empty-8x8-v0
```

Then DoorKey:

```powershell
python src\dqn_baseline.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 300000 --seed 1
python src\dqn_reward_shaping.py --env-id MiniGrid-DoorKey-8x8-v0 --total-timesteps 300000 --seed 1
python src\plot_comparison.py --env-id MiniGrid-DoorKey-8x8-v0
```

## Notes

- Both agents use the same DQN code and the same action set.
- Default `--action-set task` uses:
  - Empty/FourRooms: `left`, `right`, `forward`
  - DoorKey/UnlockPickup: `left`, `right`, `forward`, `pickup`, `toggle`
- Use `--action-set full` if you want all seven MiniGrid actions.
- The shaped agent uses `--stuck-penalty -0.01` by default.
- Environment return is logged separately from shaped learning reward, so the comparison plot uses the original MiniGrid reward.
