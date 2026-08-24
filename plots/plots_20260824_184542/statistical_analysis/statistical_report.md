# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 100 episodes of each seed's training run

## DoorKey-8x8-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -2.78 ± 3.82 | 0.58 ± 0.74 | +121.0% |
| Goal Rate (±SD) | 0.498 ± 0.525 | 0.960 ± 0.113 | +92.8% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 10.0 | 1.728 | 0.0840 | ns | 0.546 | Large |
| Goal Success Rate | 2.0 | 1.987 | 0.0469 | * | 0.628 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.82 | 0.78 | 1.000 | 1.000 |
| 2 | -6.40 | 0.79 | 0.000 | 1.000 |
| 3 | -6.40 | 0.67 | 0.000 | 0.970 |
| 4 | -6.40 | 0.89 | 0.000 | 1.000 |
| 5 | 0.86 | 0.86 | 1.000 | 1.000 |
| 6 | 0.90 | 0.90 | 1.000 | 1.000 |
| 7 | 0.80 | 0.81 | 0.980 | 0.990 |
| 8 | -6.40 | 0.76 | 0.000 | 1.000 |
| 9 | 0.87 | -1.52 | 1.000 | 0.640 |
| 10 | -6.40 | 0.90 | 0.000 | 1.000 |

---

## Empty-Random-8x8-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.70 ± 0.77 | 0.71 ± 0.71 | +0.8% |
| Goal Rate (±SD) | 0.921 ± 0.246 | 0.920 ± 0.236 | -0.1% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 6.0 | 0.776 | 0.4375 | ns | 0.245 | Small |
| Goal Success Rate | 4.0 | 0.157 | 0.8750 | ns | 0.050 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.90 | 0.90 | 0.990 | 0.980 |
| 2 | -1.48 | -1.31 | 0.220 | 0.250 |
| 3 | 0.90 | 0.89 | 1.000 | 1.000 |
| 4 | 0.95 | 0.95 | 1.000 | 1.000 |
| 5 | 1.00 | 1.00 | 1.000 | 1.000 |
| 6 | 0.94 | 0.94 | 1.000 | 1.000 |
| 7 | 0.98 | 0.98 | 1.000 | 1.000 |
| 8 | 0.95 | 0.95 | 1.000 | 1.000 |
| 9 | 0.97 | 0.93 | 1.000 | 0.990 |
| 10 | 0.90 | 0.83 | 1.000 | 0.980 |

---
