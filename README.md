# Capabilities in Reinforcement Learning — MiniGrid DQN Experiments

## Overview

This project investigates how reinforcement learning agents can learn to avoid
wasteful actions by understanding their own **capabilities** in a given state.
We train Deep Q-Network (DQN) agents on MiniGrid grid-world environments and
compare three agent types:

| Agent | Description |
|---|---|
| **Random Agent** | Takes uniformly random actions. Establishes the performance floor. |
| **Baseline DQN** | Standard DQN using only the environment's sparse reward signal. |
| **Reward-Shaped DQN** | Same DQN with a small penalty (`-0.02`) when an action causes no observable state change (e.g., walking into a wall). |

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

## State Space (Observation)

MiniGrid observations go through a pipeline of wrappers before reaching the neural network:

```
Raw MiniGrid Output          FullyObsWrapper              FlatObsWrapper         FlatObsWithDirectionWrapper
─────────────────────   →   ──────────────────   →   ──────────────────────   →   ──────────────────────────
Dictionary:                 Dictionary:              1D numpy array:             1D numpy array:
  "image": 7×7×3 array       "image": 8×8×3 array     [2, 5, 0, 1, 0, ...]      [2, 5, 0, 1, 0, ..., 1]
  "direction": int            "direction": int          (192 elements)             (193 elements)
  "mission": string           "mission": string                                    ↑ direction appended
```

**Key details:**
- The raw "image" is **not** pixel data. Each grid cell is encoded as 3 categorical integers: `[object_type, color, state]`. Values range from 0 to ~10.
- **FullyObsWrapper** replaces the agent's limited 7×7 forward-facing view with the **entire map**. This converts the problem from a POMDP (Partially Observable MDP) to a standard MDP.
- **FlatObsWrapper** discards the "mission" text and "direction" integer, then flattens the 3D grid into a 1D vector.
- **FlatObsWithDirectionWrapper** (custom) re-appends the direction integer to the end of the flattened vector, giving the agent 193 input features instead of 192.

**Why we do NOT divide by 255:** MiniGrid outputs categorical integers (0–10), not RGB pixels (0–255). Dividing by 255 would squash all values near zero, making objects indistinguishable.

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

## Reward Structure

| Event | Reward |
|---|---|
| Reaching the goal | `1 - 0.9 * (step_count / max_steps)` — a positive value that decreases the longer the agent takes |
| Any other step | `0.0` (extremely sparse!) |
| Stuck penalty (shaped agent only) | `-0.02` when `observation == next_observation` |

The shaped agent's penalty is only used for training the neural network. The
plots always show the **original environment reward** so comparisons are fair.

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
| End epsilon | 0.01 (1% random — never drops below) |
| Exploration fraction | 0.6 (epsilon decays over 60% of training, up to 80% on hard maps) |

---

## Project Structure

```
Capabilities_In_Reinforcement_Learning/
├── src/
│   ├── dqn_common.py             # Shared DQN infrastructure (network, buffer, training loop, wrappers)
│   ├── dqn_baseline.py           # Entry point: trains baseline DQN (no shaping)
│   ├── dqn_reward_shaping.py     # Entry point: trains DQN with stuck penalty
│   ├── random_agent.py           # Entry point: runs a random action baseline
│   ├── plot_comparison.py        # Generates Return & Goal Rate vs Epsilon comparison plots
│   └── plot_state_action_freq.py # Generates state-visit and action-frequency heatmaps
├── run_comparison.ps1            # PowerShell script to run all 3 agents + plot for one environment
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── RUN_STEPS.md                  # Detailed command checklist
├── presentation.tex              # LaTeX Beamer presentation
├── results/                      # Training output (CSV logs, model weights, configs)
└── plots/                        # Generated comparison plots and heatmaps
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

1. **Minimum epsilon = 0.3:** We never drop exploration below 30%. In randomized environments (DoorKey, FourRooms), the agent needs sustained exploration to handle varied layouts.

2. **FullyObsWrapper:** MiniGrid's default 7×7 partial view makes it a POMDP. Since our DQN has no memory (no LSTM/attention), it cannot solve POMDPs. Full observability converts it to a standard MDP.

3. **Direction appended to observation:** The standard `FlatObsWrapper` discards the agent's facing direction. We re-append it because knowing which way you face is crucial for deciding whether to turn or move forward.

4. **Environment reward for logging, shaped reward for learning:** The stuck penalty is only added to the reward used in the replay buffer. All logged metrics and plots use the original environment reward, ensuring fair comparison.

5. **Task-specific action subsets:** Removing irrelevant actions (like `drop` in Empty) shrinks the action space and accelerates learning.
