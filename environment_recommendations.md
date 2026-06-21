# Environment Recommendations for Capability-Aware RL

## Your Approach — What We Need to Showcase

Based on the [presentation.tex](file:///c:/Users/RijulTandon/OneDrive%20-%20ProcDNA%20Analytics%20Pvt.%20Ltd/Desktop/RL/Capabilities_In_Reinforcement_Learning/presentation.tex) and [README.md](file:///c:/Users/RijulTandon/OneDrive%20-%20ProcDNA%20Analytics%20Pvt.%20Ltd/Desktop/RL/Capabilities_In_Reinforcement_Learning/README.md), your core idea is:

> Learn a capability function $\phi_\theta(s)$ that predicts which actions are **infeasible** at a given state, using environment-agnostic signals (S1: identity transitions, S2: neighbor consensus, upper-bound masking), and integrate this with the RL policy via post-posed masking or reward shaping.

The ideal environments to demonstrate this should have:

1. **Many actions that are frequently infeasible** — large action spaces where only a subset is valid at any given state
2. **Physical constraints / walls / boundaries** — so actions genuinely produce $\|s' - s\| = 0$ or near-zero transitions
3. **State-dependent feasibility** — the *same* action works in some states but not others (this is where S2 shines)
4. **Sparse rewards** — so the exploration savings from capability-aware masking translate into faster convergence
5. **Safety-critical scenarios** — where avoiding infeasible/dangerous actions has real value beyond just efficiency

---

## Tier 1 — High Impact, Strong Fit (Recommended Starting Points)

### 1. Safety Gymnasium (Gymnasium-based Safe RL)
| Property | Details |
|---|---|
| **Package** | `pip install safety-gymnasium` |
| **Env IDs** | `SafetyPointGoal1-v0`, `SafetyCarGoal1-v0`, `SafetyAntGoal1-v0`, `SafetyPointButton1-v0` |
| **Action Space** | Continuous (Box) |
| **Why it fits** | Environments have **hazard zones** (cost regions) and **physical boundaries**. Actions that push the agent into walls, hazard zones, or out-of-bounds produce near-zero or penalized transitions. Your capability function can learn to mask actions leading to hazard/boundary collisions *without* explicit constraint specifications — demonstrating your "agnostic, unconstrained safety" claim (Slide 20). |
| **Key showcase** | Compare your approach vs. Constrained Policy Optimization (CPO), TRPO-Lagrangian. Show that $\phi_\theta$ achieves comparable safety *without* requiring predefined cost functions. This directly addresses the future work on your Slide 21. |
| **Effort** | Medium — uses Gymnasium API, PPO-compatible, similar to your MuJoCo setup. |

### 2. Gymnasium-Robotics: Fetch & Shadow Hand
| Property | Details |
|---|---|
| **Package** | `pip install gymnasium-robotics` |
| **Env IDs** | `FetchReach-v3`, `FetchPush-v3`, `FetchPickAndPlace-v3`, `FetchSlide-v3`, `HandReach-v2`, `HandManipulateBlock-v2` |
| **Action Space** | Continuous (Box), 4D for Fetch, 20D+ for Shadow Hand |
| **Why it fits** | **Extremely sparse rewards** (binary: goal reached or not). Fetch tasks have **joint limits**, **table boundaries**, and **object interaction constraints** — the gripper can't close on nothing, can't push through the table, etc. Shadow Hand has a 20+ dimensional action space where most joint configurations are physically infeasible at any given moment. |
| **Key showcase** | Shadow Hand is the killer demo — with 20+ actions, the combinatorial explosion of infeasible joint states makes vanilla exploration nearly hopeless. Your capability function can dramatically shrink the effective action space. Compare sample efficiency: baseline HER vs. HER + $\phi_\theta$ masking. |
| **Effort** | Medium-High — requires MuJoCo, may need HER (Hindsight Experience Replay). |

### 3. PettingZoo / Overcooked-AI (Multi-Agent Coordination)
| Property | Details |
|---|---|
| **Package** | `pip install pettingzoo` or `pip install overcooked-ai` |
| **Env IDs** | PettingZoo: `pistonball_v6`, `cooperative_pong_v5`; Overcooked: various layouts |
| **Action Space** | Discrete (typically 4-6 actions) |
| **Why it fits** | In multi-agent environments, **other agents create dynamic obstacles**. An action that is feasible when Agent B is in position X becomes infeasible when Agent B moves to position Y. This is exactly the scenario where S2 (neighbor consensus) excels — the action works in some states but is blocked in others due to the dynamic environment. |
| **Key showcase** | Demonstrate that $\phi_\theta$ can learn *dynamic* capability constraints imposed by other agents — not just static walls. This extends your work beyond static environments and shows generalization. |
| **Effort** | Medium — discrete action spaces are easier, but multi-agent training adds complexity. |

---

## Tier 2 — Strong Fit, Moderate Effort

### 4. Gymnasium Classic Control: Acrobot, MountainCar
| Property | Details |
|---|---|
| **Package** | Already in `gymnasium` |
| **Env IDs** | `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0` |
| **Action Space** | Discrete (Acrobot: 3, MountainCar: 3) or Continuous |
| **Why it fits** | **MountainCar** is a textbook case: at the bottom of the valley, applying force in either direction barely moves the car — $\|s' - s\| \approx 0$ for low-energy actions. Your S1 signal detects this perfectly. The agent must learn that "push right" is infeasible at the valley bottom (not enough momentum) but feasible on the slopes. |
| **Key showcase** | Simple, well-understood, and easy to visualize. Perfect for a clean ablation study showing S1 vs S2 vs baseline on a problem everyone in the RL community understands. |
| **Effort** | Low — trivial to set up, already Gymnasium compatible. |

### 5. Minigrid + BabyAI Extensions
| Property | Details |
|---|---|
| **Package** | `pip install minigrid` (already installed), `pip install babyai` |
| **Env IDs** | `BabyAI-GoToRedBall-v0`, `BabyAI-PutNextLocal-v0`, `BabyAI-UnlockPickup-v0`, `BabyAI-SynthSeq-v0` |
| **Action Space** | Discrete (7 actions) |
| **Why it fits** | BabyAI extends MiniGrid with **language-conditioned tasks** and **longer action sequences**. The environments have all the wall-collision infeasibility of MiniGrid but with more complex tasks where `pickup`, `drop`, `toggle` are context-dependent (can't pick up if nothing there, can't toggle if no door). Your S2 signal perfectly distinguishes "pickup does nothing because there's nothing to pick up" vs "pickup does nothing because it's a wait action." |
| **Key showcase** | Natural extension of your existing MiniGrid work. Shows scaling within the same framework to harder tasks. |
| **Effort** | Very Low — almost identical to your current codebase. |

### 6. Crafter (Procedural Open-World Survival)
| Property | Details |
|---|---|
| **Package** | `pip install crafter` |
| **Env IDs** | `Crafter-v1` |
| **Action Space** | Discrete (17 actions!) |
| **Why it fits** | **17 discrete actions** in a procedurally generated survival game. Actions include: move (4 dirs), do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, eat_cow, eat_plant, drink, attack_skeleton. **Most actions are infeasible most of the time** — you can't craft without materials, can't eat without food nearby, can't place without inventory. |
| **Key showcase** | **This is your killer environment.** With 17 actions and only 2-4 being feasible at any moment, a standard agent wastes ~75-80% of its exploration on infeasible actions. Your capability function can learn this feasibility structure and dramatically accelerate learning. The improvement should be massive and visually striking. |
| **Effort** | Medium — pixel observations (64×64), may need CNN or use their provided feature observations. |

> [!IMPORTANT]
> **Crafter is likely the single best environment for your thesis.** The ratio of infeasible-to-feasible actions at any state is extremely high (often 13 out of 17 actions do nothing), making the case for capability-aware learning undeniable.

---

## Tier 3 — Aspirational / High Effort, Maximum Impact

### 7. NetHack Learning Environment (NLE)
| Property | Details |
|---|---|
| **Package** | `pip install nle` |
| **Env IDs** | `NetHackScore-v0`, `NetHackChallenge-v0` |
| **Action Space** | Discrete (93 actions!) |
| **Why it fits** | 93 discrete actions in a procedurally generated dungeon. At any state, only a tiny fraction are valid (can't eat without food, can't cast spells without mana, can't open doors without being adjacent to one). The massive action space makes exploration extremely inefficient without masking. |
| **Key showcase** | If your approach works here, it's a landmark result. Show that $\phi_\theta$ discovers action feasibility in a 93-action space purely from transitions. |
| **Effort** | High — complex environment, large observation space, requires significant compute. Linux-only. |

### 8. MiniWorld (3D Navigation with Physics)
| Property | Details |
|---|---|
| **Package** | `pip install miniworld` |
| **Env IDs** | `MiniWorld-Maze-v0`, `MiniWorld-FourRooms-v0`, `MiniWorld-CollectHealth-v0` |
| **Action Space** | Discrete or Continuous |
| **Why it fits** | 3D version of MiniGrid with actual physics. Wall collisions, door interactions, object manipulation — all produce identity transitions when infeasible. Moving from 2D (MiniGrid) to 3D (MiniWorld) with the same approach validates generalization across observation modalities. |
| **Key showcase** | Direct comparison: same task (FourRooms) in 2D vs 3D. Shows your signals are truly environment-agnostic. |
| **Effort** | Medium — visual observations require CNN, but Gymnasium-compatible. |

---

## Summary Matrix

| # | Environment | Action Space | Infeasibility Density | Reward Sparsity | Setup Effort | Impact |
|---|---|---|---|---|---|---|
| 1 | Safety Gymnasium | Continuous | ★★★☆ | ★★☆☆ | Medium | ★★★★★ |
| 2 | Fetch / Shadow Hand | Continuous | ★★★★ | ★★★★★ | Medium-High | ★★★★★ |
| 3 | PettingZoo / Overcooked | Discrete | ★★★☆ | ★★★☆ | Medium | ★★★★☆ |
| 4 | MountainCar / Acrobot | Discrete | ★★☆☆ | ★★★★ | **Very Low** | ★★★☆☆ |
| 5 | BabyAI | Discrete | ★★★☆ | ★★★★ | **Very Low** | ★★★☆☆ |
| 6 | **Crafter** | **Discrete (17)** | **★★★★★** | **★★★★★** | Medium | **★★★★★** |
| 7 | NetHack (NLE) | Discrete (93) | ★★★★★ | ★★★★★ | High | ★★★★★ |
| 8 | MiniWorld | Mixed | ★★★☆ | ★★★☆ | Medium | ★★★★☆ |

## Recommended Strategy

> [!TIP]
> **Phased approach for your PhD work:**
> 
> **Phase 1 (Quick wins):** MountainCar + BabyAI extensions — minimal code changes, clean ablation results, extends existing work naturally.
>
> **Phase 2 (Main contribution):** **Crafter** — this is where your approach will shine the most. 17 actions with extreme context-dependent feasibility makes the strongest case for capability learning.
>
> **Phase 3 (Continuous control):** Safety Gymnasium — directly demonstrates the Safe RL connection (your Slide 21 future work) and positions the thesis within a hot research area.
>
> **Phase 4 (Stretch goal):** Fetch/Shadow Hand or NetHack — if time permits, these would be landmark results.

## Open Questions

1. **Do you want to prioritize discrete or continuous environments?** Your current codebase supports both (DQN for discrete, PPO for continuous), but the capability function architecture differs.
2. **Are you planning to implement the full $\phi_\theta$ neural network masking (Slides 3-5), or continue with the reward-shaping proxy?** This affects which environments would show the biggest delta.
3. **Compute budget?** Crafter and NetHack require significantly more training steps than MiniGrid.
4. **Do you want me to implement a prototype for any of these environments?**
