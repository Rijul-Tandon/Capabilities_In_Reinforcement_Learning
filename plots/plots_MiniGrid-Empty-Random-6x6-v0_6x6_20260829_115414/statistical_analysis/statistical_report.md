# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## Empty-Random-6x6-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.96 ± 0.02 | 0.96 ± 0.02 | +0.0% |
| Goal Rate (±SD) | 0.999 ± 0.001 | 1.000 ± 0.001 | +0.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 16.0 | 0.681 | 0.4961 | ns | 0.215 | Small |
| Goal Success Rate | 14.0 | 0.467 | 0.6406 | ns | 0.148 | Small |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.96 | 0.96 | 0.999 | 1.000 |
| 2 | 0.98 | 0.98 | 1.000 | 1.000 |
| 3 | 0.94 | 0.94 | 1.000 | 1.000 |
| 4 | 0.96 | 0.97 | 0.998 | 1.000 |
| 5 | 0.96 | 0.96 | 1.000 | 0.999 |
| 6 | 0.96 | 0.96 | 1.000 | 1.000 |
| 7 | 0.99 | 0.99 | 1.000 | 1.000 |
| 8 | 0.97 | 0.96 | 1.000 | 0.998 |
| 9 | 0.99 | 0.99 | 1.000 | 1.000 |
| 10 | 0.92 | 0.93 | 0.998 | 0.999 |

---
