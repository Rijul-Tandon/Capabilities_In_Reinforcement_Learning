# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 100 episodes of each seed's training run

## DoorKey-6x6-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.80 ± 0.20 | 0.87 ± 0.05 | +8.8% |
| Goal Rate (±SD) | 0.974 ± 0.055 | 0.993 ± 0.013 | +1.9% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 27.5 | 0.000 | 1.0000 | ns | 0.000 | Negligible |
| Goal Success Rate | 6.0 | 0.237 | 0.8125 | ns | 0.075 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.88 | 0.88 | 1.000 | 1.000 |
| 2 | 0.37 | 0.86 | 0.840 | 1.000 |
| 3 | 0.87 | 0.82 | 1.000 | 0.990 |
| 4 | 0.89 | 0.89 | 1.000 | 1.000 |
| 5 | 0.90 | 0.90 | 1.000 | 1.000 |
| 6 | 0.85 | 0.78 | 0.990 | 0.960 |
| 7 | 0.49 | 0.90 | 0.910 | 1.000 |
| 8 | 0.92 | 0.92 | 1.000 | 1.000 |
| 9 | 0.90 | 0.83 | 1.000 | 0.980 |
| 10 | 0.91 | 0.91 | 1.000 | 1.000 |

---

## Empty-Random-6x6-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.96 ± 0.02 | 0.96 ± 0.03 | -0.4% |
| Goal Rate (±SD) | 1.000 ± 0.000 | 0.997 ± 0.009 | -0.3% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 2.0 | 0.887 | 0.3750 | ns | 0.281 | Small |
| Goal Success Rate | 0.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.96 | 0.96 | 1.000 | 1.000 |
| 2 | 0.98 | 0.98 | 1.000 | 1.000 |
| 3 | 0.94 | 0.94 | 1.000 | 1.000 |
| 4 | 0.97 | 0.97 | 1.000 | 1.000 |
| 5 | 0.95 | 0.91 | 1.000 | 0.970 |
| 6 | 0.96 | 0.96 | 1.000 | 1.000 |
| 7 | 0.99 | 0.99 | 1.000 | 1.000 |
| 8 | 0.97 | 0.97 | 1.000 | 1.000 |
| 9 | 0.99 | 0.99 | 1.000 | 1.000 |
| 10 | 0.93 | 0.93 | 1.000 | 1.000 |

---
