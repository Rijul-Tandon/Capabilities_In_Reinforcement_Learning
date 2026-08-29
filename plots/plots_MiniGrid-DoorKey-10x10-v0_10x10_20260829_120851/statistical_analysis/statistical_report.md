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
| Mean Reward (±SD) | -9.17 ± 2.45 | -0.34 ± 3.39 | +96.2% |
| Goal Rate (±SD) | 0.016 ± 0.050 | 0.886 ± 0.312 | +5544.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 1.0 | 2.886 | 0.0039 | ** | 0.912 | Large |
| Goal Success Rate | 0.0 | 2.886 | 0.0039 | ** | 0.912 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -9.58 | 0.70 | 0.000 | 0.993 |
| 2 | -10.00 | 0.75 | 0.000 | 0.992 |
| 3 | -10.00 | 0.72 | 0.000 | 0.982 |
| 4 | -2.22 | 0.66 | 0.157 | 0.976 |
| 5 | -10.00 | 0.78 | 0.000 | 0.990 |
| 6 | -10.00 | 0.81 | 0.000 | 0.991 |
| 7 | -10.00 | 0.78 | 0.000 | 0.991 |
| 8 | -9.98 | -10.00 | 0.000 | 0.000 |
| 9 | -9.95 | 0.75 | 0.000 | 0.993 |
| 10 | -10.00 | 0.61 | 0.000 | 0.956 |

---
