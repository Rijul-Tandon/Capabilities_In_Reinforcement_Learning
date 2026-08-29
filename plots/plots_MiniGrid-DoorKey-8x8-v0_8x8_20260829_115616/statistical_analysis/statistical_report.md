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
| Mean Reward (±SD) | -2.79 ± 3.80 | 0.80 ± 0.07 | +128.6% |
| Goal Rate (±SD) | 0.494 ± 0.521 | 0.990 ± 0.008 | +100.2% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 5.0 | 2.335 | 0.0195 | * | 0.739 | Large |
| Goal Success Rate | 4.0 | 2.466 | 0.0137 | * | 0.780 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.78 | 0.78 | 0.988 | 0.989 |
| 2 | -6.40 | 0.77 | 0.000 | 0.992 |
| 3 | -6.40 | 0.82 | 0.000 | 0.995 |
| 4 | -6.40 | 0.76 | 0.000 | 0.978 |
| 5 | 0.79 | 0.84 | 0.985 | 0.995 |
| 6 | 0.87 | 0.89 | 0.993 | 0.997 |
| 7 | 0.78 | 0.83 | 0.985 | 0.996 |
| 8 | -6.40 | 0.67 | 0.000 | 0.976 |
| 9 | 0.83 | 0.75 | 0.992 | 0.982 |
| 10 | -6.40 | 0.89 | 0.000 | 0.998 |

---
