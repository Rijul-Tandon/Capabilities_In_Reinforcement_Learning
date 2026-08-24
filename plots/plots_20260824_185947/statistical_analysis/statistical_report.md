# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 100 episodes of each seed's training run

## DoorKey-10x10-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -10.00 ± 0.00 | -0.04 ± 2.06 | +99.7% |
| Goal Rate (±SD) | 0.000 ± 0.000 | 0.914 ± 0.205 | +0.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 3.097 | 0.0020 | ** | 0.979 | Large |
| Goal Success Rate | 0.0 | 3.097 | 0.0020 | ** | 0.979 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -10.00 | 0.76 | 0.000 | 1.000 |
| 2 | -10.00 | 0.81 | 0.000 | 1.000 |
| 3 | -10.00 | 0.79 | 0.000 | 0.990 |
| 4 | -10.00 | -5.80 | 0.000 | 0.340 |
| 5 | -10.00 | 0.66 | 0.000 | 0.980 |
| 6 | -10.00 | 0.86 | 0.000 | 1.000 |
| 7 | -10.00 | 0.55 | 0.000 | 0.970 |
| 8 | -10.00 | 0.70 | 0.000 | 0.990 |
| 9 | -10.00 | 0.78 | 0.000 | 0.990 |
| 10 | -10.00 | -0.46 | 0.000 | 0.880 |

---

## Empty-Random-10x10-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.13 ± 1.66 | 0.51 ± 1.37 | +285.4% |
| Goal Rate (±SD) | 0.807 ± 0.397 | 0.911 ± 0.281 | +12.9% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 11.0 | 0.402 | 0.6875 | ns | 0.127 | Small |
| Goal Success Rate | 0.0 | 1.150 | 0.2500 | ns | 0.364 | Medium |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -2.99 | 0.92 | 0.060 | 1.000 |
| 2 | 0.94 | 0.94 | 1.000 | 1.000 |
| 3 | -3.05 | -3.39 | 0.050 | 0.110 |
| 4 | 0.96 | 0.96 | 1.000 | 1.000 |
| 5 | 0.99 | 0.99 | 1.000 | 1.000 |
| 6 | 0.92 | 0.92 | 1.000 | 1.000 |
| 7 | 0.97 | 0.97 | 1.000 | 1.000 |
| 8 | 0.76 | 0.94 | 0.960 | 1.000 |
| 9 | 0.96 | 0.96 | 1.000 | 1.000 |
| 10 | 0.87 | 0.87 | 1.000 | 1.000 |

---
