param(
    [string]$EnvId = "MiniGrid-Empty-5x5-v0",
    [int]$Timesteps = 50000,
    [int]$Seed = 1,
    [int]$MaxSteps = -1
)

python "$PSScriptRoot\src\random_agent.py" --env-id $EnvId --total-timesteps $Timesteps --seed $Seed --max-steps $MaxSteps
python "$PSScriptRoot\src\dqn_baseline.py" --env-id $EnvId --total-timesteps $Timesteps --seed $Seed --max-steps $MaxSteps
python "$PSScriptRoot\src\dqn_reward_shaping.py" --env-id $EnvId --total-timesteps $Timesteps --seed $Seed --max-steps $MaxSteps
python "$PSScriptRoot\src\plot_comparison.py" --env-id $EnvId
python "$PSScriptRoot\src\plot_state_action_freq.py" --env-id $EnvId
