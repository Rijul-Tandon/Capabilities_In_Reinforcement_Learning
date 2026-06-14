param(
    [string]$EnvId = "MiniGrid-Empty-8x8-v0",
    [int]$Timesteps = 100000,
    [int]$Seed = 1
)

python "$PSScriptRoot\src\random_agent.py" --env-id $EnvId --total-timesteps $Timesteps --seed $Seed
python "$PSScriptRoot\src\dqn_baseline.py" --env-id $EnvId --total-timesteps $Timesteps --seed $Seed
python "$PSScriptRoot\src\dqn_reward_shaping.py" --env-id $EnvId --total-timesteps $Timesteps --seed $Seed
python "$PSScriptRoot\src\plot_comparison.py" --env-id $EnvId
