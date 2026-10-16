# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## DoorKey-6x6-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.88 ± 0.02 | 0.87 ± 0.04 | -0.9% |
| Goal Rate (±SD) | 0.997 ± 0.003 | 0.995 ± 0.009 | -0.2% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 25.0 | 0.195 | 0.8457 | ns | 0.061 | Negligible |
| Goal Success Rate | 24.0 | 0.293 | 0.7695 | ns | 0.093 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.88 | 0.87 | 1.000 | 0.996 |
| 2 | 0.85 | 0.84 | 0.998 | 0.996 |
| 3 | 0.86 | 0.86 | 0.997 | 0.999 |
| 4 | 0.89 | 0.89 | 0.999 | 1.000 |
| 5 | 0.88 | 0.77 | 0.994 | 0.972 |
| 6 | 0.84 | 0.84 | 0.993 | 0.987 |
| 7 | 0.89 | 0.90 | 0.999 | 1.000 |
| 8 | 0.92 | 0.92 | 0.999 | 1.000 |
| 9 | 0.89 | 0.90 | 0.999 | 1.000 |
| 10 | 0.88 | 0.91 | 0.991 | 0.999 |

---
