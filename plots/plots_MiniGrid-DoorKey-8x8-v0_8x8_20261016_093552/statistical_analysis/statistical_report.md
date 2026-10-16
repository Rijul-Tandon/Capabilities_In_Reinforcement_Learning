# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## DoorKey-8x8-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -4.23 ± 3.48 | 0.81 ± 0.08 | +119.2% |
| Goal Rate (±SD) | 0.299 ± 0.481 | 0.991 ± 0.011 | +231.8% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 3.0 | 2.584 | 0.0098 | ** | 0.817 | Large |
| Goal Success Rate | 3.0 | 2.584 | 0.0098 | ** | 0.817 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.80 | 0.81 | 0.993 | 0.996 |
| 2 | 0.80 | 0.70 | 0.996 | 0.976 |
| 3 | -6.40 | 0.76 | 0.000 | 0.976 |
| 4 | -6.34 | 0.88 | 0.000 | 0.997 |
| 5 | 0.85 | 0.85 | 0.997 | 0.998 |
| 6 | -6.40 | 0.89 | 0.000 | 0.998 |
| 7 | -6.40 | 0.83 | 0.000 | 0.996 |
| 8 | -6.40 | 0.65 | 0.000 | 0.973 |
| 9 | -6.40 | 0.86 | 0.000 | 0.997 |
| 10 | -6.40 | 0.90 | 0.000 | 1.000 |

---
