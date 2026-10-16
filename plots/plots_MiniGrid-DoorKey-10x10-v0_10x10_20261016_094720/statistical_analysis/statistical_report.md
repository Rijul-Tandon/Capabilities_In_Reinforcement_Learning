# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## DoorKey-10x10-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -7.75 ± 4.50 | 0.74 ± 0.07 | +109.6% |
| Goal Rate (±SD) | 0.197 ± 0.416 | 0.989 ± 0.009 | +401.9% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 1.0 | 2.886 | 0.0039 | ** | 0.912 | Large |
| Goal Success Rate | 1.0 | 2.886 | 0.0039 | ** | 0.912 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -9.35 | 0.72 | 0.000 | 0.997 |
| 2 | -10.00 | 0.76 | 0.000 | 0.994 |
| 3 | -9.98 | 0.74 | 0.000 | 0.985 |
| 4 | -10.00 | 0.64 | 0.000 | 0.974 |
| 5 | 0.81 | 0.80 | 0.999 | 0.996 |
| 6 | -9.78 | 0.83 | 0.000 | 0.996 |
| 7 | -10.00 | 0.78 | 0.000 | 0.993 |
| 8 | -10.00 | 0.62 | 0.000 | 0.973 |
| 9 | -9.91 | 0.73 | 0.000 | 0.988 |
| 10 | 0.74 | 0.82 | 0.972 | 0.996 |

---
