# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 100 episodes of each seed's training run

## DoorKey-10x10-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -10.00 ± 0.00 | -3.00 ± 6.06 | +70.0% |
| Goal Rate (±SD) | 0.000 ± 0.000 | 0.637 ± 0.552 | +0.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |
| Goal Success Rate | 0.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -10.00 | 0.35 | 0.000 | 0.920 |
| 2 | -10.00 | -10.00 | 0.000 | 0.000 |
| 3 | -10.00 | 0.64 | 0.000 | 0.990 |

---

## Empty-Random-10x10-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -1.73 ± 2.32 | 0.91 ± 0.04 | +152.6% |
| Goal Rate (±SD) | 0.380 ± 0.537 | 1.000 ± 0.000 | +163.2% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |
| Goal Success Rate | 0.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -3.28 | 0.92 | 0.050 | 1.000 |
| 2 | 0.94 | 0.94 | 1.000 | 1.000 |
| 3 | -2.84 | 0.87 | 0.090 | 1.000 |

---
