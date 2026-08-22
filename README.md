# Capabilities in Reinforcement Learning — MiniGrid DQN Experiments

## Overview

This project investigates how reinforcement learning agents can learn to avoid
wasteful actions by understanding their own **capabilities** in a given state.
We train Deep Q-Network (DQN) agents on MiniGrid grid-world environments and
compare four agent types:

| Agent | Description |
|---|---|
| **Random Agent** | Takes uniformly random actions. Establishes the performance floor. |
| **Vanilla DQN** | Standard DQN using the original TD target (`max_a Q_target`). Prone to Q-value overestimation. |
| **DDQN Baseline** | Double DQN — decouples action selection from evaluation to reduce overestimation. No reward shaping. |
| **DDQN Reward-Shaped** | Same Double DQN with a penalty (`-0.10`) when an action causes no observable state change (e.g., walking into a wall). |

The reward shaping approach is a lightweight, environment-agnostic proxy for
capability-aware learning: if an action has no effect, the agent should learn
to avoid it in that state.

---

## Environments

We run experiments on three [MiniGrid](https://minigrid.farama.org/) environments
of increasing difficulty:

### MiniGrid-Empty-8x8-v0
| Property | Value |
|---|---|
| Grid Size | 8 × 8 |
| Layout | Fixed — empty room with walls on the border |
| Goal | Fixed position (bottom-right corner) |
| Randomization | None. The map is identical every episode. |
| Actions Used | `left`, `right`, `forward` (3 actions) |
| Difficulty | Easy — tests basic navigation |

### MiniGrid-DoorKey-8x8-v0
| Property | Value |
|---|---|
| Grid Size | 8 × 8 |
| Layout | **Randomized** — a wall divides the room, with a locked door |
| Goal | **Randomized** position (always behind the locked door) |
| Randomization | Key position, door position, wall position, goal position all change every episode |
| Actions Used | `left`, `right`, `forward`, `pickup`, `toggle` (5 actions) |
| Difficulty | Hard — the agent must find a key, pick it up, navigate to the door, open it, then reach the goal |

### MiniGrid-FourRooms-v0
| Property | Value |
|---|---|
| Grid Size | 19 × 19 |
| Layout | Fixed — four rooms connected by narrow doorways |
| Goal | **Randomized** position |
| Randomization | Agent start position and goal position change every episode |
| Actions Used | `left`, `right`, `forward` (3 actions) |
| Difficulty | Medium — larger map requiring efficient long-distance navigation |

---

## State Space (Observation) & Normalization

MiniGrid observations go through a pipeline of wrappers before reaching the neural network:

```
Raw MiniGrid Output          FullyObsWrapper              ImgObsWrapper          FlatImageAndDirectionWrapper
─────────────────────   →   ──────────────────   →   ─────────────────────   →   ────────────────────────────
Dictionary:                 Dictionary:              3D numpy array:             1D numpy array:
  "image": 7×7×3 array       "image": 6×6×3 array     (6×6×3 categorical)         [img / max_vals, dir / 3.0]
  "direction": int            "direction": int                                     (109 normalized elements)
  "mission": string           "mission": string                                    ↑ component-wise scaled to [0, 1]
```

**Key details:**
- **FullyObsWrapper** replaces the agent's limited 7×7 forward-facing view with the **entire map**. This converts the POMDP into a standard MDP.
- **Component-wise Min-Max Scaling:** Input channels are normalized to $[0.0, 1.0]$:
  - **Object Type Channel (`[:, :, 0]`):** Divided by `10.0` (Max object ID = 10)
  - **Color Channel (`[:, :, 1]`):** Divided by `5.0` (Max color ID = 5)
  - **State Channel (`[:, :, 2]`):** Divided by `3.0` (Max state ID = 3)
  - **Agent Direction (`agent_dir`):** Divided by `3.0` (Max direction ID = 3)
- **Multi-Stage Heatmaps (DoorKey):** For `DoorKey` environments, Q-value overestimation and state-action frequency heatmaps are generated across **all 3 distinct task stages**:
  1. `initial`: Key on ground, Door locked/closed.
  2. `key_picked`: Key in inventory, Door locked/closed.
  3. `door_opened`: Door unlocked and opened.

---

## Action Space

Actions are restricted per environment using `MiniGridActionSubsetWrapper` to prevent the agent from wasting exploration on irrelevant actions:

| Action | Name      | What it does                                                     |
| ------ | --------- | ---------------------------------------------------------------- |
| 0      | `left`    | Rotate the agent **90° left** (does **not** move)                |
| 1      | `right`   | Rotate the agent **90° right** (does **not** move)               |
| 2      | `forward` | Move one cell **in the direction the agent is currently facing** |
| 3      | `pickup`  | Pick up an object in front of the agent                          |
| 4      | `drop`    | Drop the carried object                                          |
| 5      | `toggle`  | Open/close door or activate object in front                      |
| 6      | `done`    | Unused                                                           |

- **Empty / FourRooms:** Only actions 0, 1, 2 (navigation only)
- **DoorKey:** Actions 0, 1, 2, 3, 5 (navigation + interaction)

---

## Markovian Reward Structure

We enforce a strictly Markovian reward structure across all environments:

| Event | Reward | Description |
|---|---|---|
| Reaching the Goal | `+1.0` | Fixed goal reward |
| Every Other Step | `-1.0` (or `-0.01` baseline) | Fixed step penalty dependent ONLY on transition $(s, a, s')$ |

This replaces step-count-dependent rewards, ensuring that the reward depends strictly on the current transition $(s, a, s')$.

---

## DQN Architecture

```
Input (193) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(num_actions)
```

| Hyperparameter | Default Value |
|---|---|
| Hidden layers | 2 × 256 neurons |
| Learning rate | 1.71e-3 (Adam optimizer) |
| Replay buffer size | 10% of total timesteps (CleanRL default) |
| Batch size | 128 |
| Discount factor (γ) | 0.915 |
| Target network update | Every 788 steps |
| Learning starts | After 2,000 random steps |
| Train frequency | Every 10 environment steps |
| Start epsilon | 1.0 (100% random) |
| End epsilon | 0.1 (10% random — never drops below) |
| Exploration fraction | 0.6 (epsilon decays over 60% of training, up to 80% on hard maps) |

---

## Project Structure

```
Capabilities_In_Reinforcement_Learning/
├── src/
│   ├── dqn_common.py             # Shared DQN/DDQN infrastructure (network, buffer, training loop, wrappers)
│   ├── dqn_vanilla.py            # Entry point: trains standard DQN (no double, no shaping)
│   ├── dqn_baseline.py           # Entry point: trains Double DQN baseline (no shaping)
│   ├── dqn_reward_shaping.py     # Entry point: trains Double DQN with stuck penalty
│   ├── random_agent.py           # Entry point: runs a random action baseline
│   ├── plot_comparison.py        # Generates Return & Goal Rate vs Epsilon comparison plots
│   ├── plot_state_action_freq.py # Generates state-visit and action-frequency heatmaps
│   └── plot_q_overestimation.py  # Generates DQN vs DDQN Q-value overestimation heatmaps & bar charts
├── .github/workflows/
│   ├── run_experiments.yml       # Main workflow: trains all agents on all envs (3 seeds)
│   └── run_overestimation.yml    # Overestimation workflow: DQN vs DDQN comparison (1 seed)
├── run_comparison.ps1            # PowerShell script to run all agents + plot for one environment
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── results/                      # Training output (CSV logs, model weights, configs)
├── plots/                        # Generated comparison plots and heatmaps
│   ├── overestimation/           # Q-value overestimation heatmaps
│   │   ├── DoorKey/
│   │   │   ├── seed_1/
│   │   │   └── seed_2/
│   │   └── Empty/
│   │       ├── seed_1/
│   │       └── seed_2/
│   ├── reward_comparison/        # Return, Goal Rate, and TD Loss curves
│   └── action_freq/              # Cumulative action frequency heatmaps during training
│       ├── last_50_percent/      # Evaluated over last 50% steps
│       └── last_25_percent/      # Evaluated over last 25% steps
```

---

## Setup

```powershell
python -m venv venv_capabilities
.\venv_capabilities\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv_capabilities\Scripts\Activate.ps1
```

---

## Running Experiments

### Quick: Use the automated script

The `run_comparison.ps1` script runs all three agents (random, baseline, shaped) and generates the comparison plot automatically:

```powershell
.\run_comparison.ps1 -EnvId "MiniGrid-Empty-8x8-v0" -Timesteps 100000
.\run_comparison.ps1 -EnvId "MiniGrid-DoorKey-8x8-v0" -Timesteps 150000
.\run_comparison.ps1 -EnvId "MiniGrid-FourRooms-v0" -Timesteps 100000
```

### Parallel execution (recommended)

Open 3 separate terminals (activate the venv in each) and run one environment per terminal for faster results.

### Manual: Run agents individually

```powershell
# Train all three agents
python src/random_agent.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000
python src/dqn_baseline.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000
python src/dqn_reward_shaping.py --env-id MiniGrid-Empty-8x8-v0 --total-timesteps 100000

# Generate comparison plot (Return & Goal Rate vs Epsilon)
python src/plot_comparison.py --env-id MiniGrid-Empty-8x8-v0

# Generate state-visit and action-frequency heatmaps
python src/plot_state_action_freq.py --env-id MiniGrid-Empty-8x8-v0
```

---

## Outputs

### Results Directory
Each training run creates a timestamped folder under `results/`:
```
results/MiniGrid-Empty-8x8-v0__dqn_baseline__1__1718300000/
├── config.json    # All hyperparameters used
├── episodes.csv   # Per-episode: return, length, goal reached, epsilon
├── metrics.csv    # Per-interval: TD loss, Q-values, stuck rate
└── q_net.pt       # Saved PyTorch model weights
```

### Plots Directory
- `plots/<env_id>_comparison.png` — Return and Goal Rate vs Epsilon for all agents
- `plots/<env_id>_state_action_freq.png` — 2×3 heatmap grid comparing visit patterns and preferred actions

---

## Key Design Decisions

1. **Minimum epsilon = 0.1:** We let exploration decay to 10%. On hard environments (DoorKey, FourRooms, MultiRoom) we decay slowly over 80% of training.

2. **FullyObsWrapper:** MiniGrid's default 7×7 partial view makes it a POMDP. Since our DQN has no memory (no LSTM/attention), it cannot solve POMDPs. Full observability converts it to a standard MDP.

3. **Direction appended to observation:** The standard `FlatObsWrapper` discards the agent's facing direction. We re-append it because knowing which way you face is crucial for deciding whether to turn or move forward.

4. **Environment reward for logging, shaped reward for learning:** The stuck penalty is only added to the reward used in the replay buffer. All logged metrics and plots use the original environment reward, ensuring fair comparison.

5. **Task-specific action subsets:** Removing irrelevant actions (like `drop` in Empty) shrinks the action space and accelerates learning.

6. **DQN vs DDQN toggle:** The `--double-dqn` flag controls whether the TD target uses standard DQN (`max_a Q_target`) or Double DQN (online selects, target evaluates). This allows direct comparison of Q-value overestimation.

7. **Dynamic replay buffer:** Buffer size defaults to 10% of total timesteps (CleanRL standard), scaling automatically with training duration.

---

## DQN vs DDQN Overestimation Experiment

The `run_overestimation.yml` workflow trains both **Standard DQN** (`dqn_vanilla.py`) and **Double DQN** (`dqn_baseline.py`) on all 4 environments with a fixed layout (seed=1). After training, `plot_q_overestimation.py` generates:

1. **Q-Value Heatmaps**: For each environment, a 3-panel grid showing `DQN Max Q | DDQN Max Q | Difference (DQN - DDQN)`. Red cells in the difference panel indicate where standard DQN overestimates relative to DDQN.

2. **Bar Chart**: A grouped bar chart comparing average max Q-values across all environments, clearly showing the overestimation gap.

### Why Standard DQN Overestimates

Standard DQN uses `max_a Q_target(s', a)` as the TD target. The `max` operator applied to noisy Q-value estimates is a **positively biased estimator** — it systematically picks the noisiest (highest) estimate, inflating Q-values over training.

Double DQN fixes this by decoupling selection from evaluation:
- The **online network** selects the best action: `a* = argmax_a Q_online(s', a)`
- The **target network** evaluates it: `Q_target(s', a*)`

Since these are different networks, the noise doesn't compound, reducing overestimation.
