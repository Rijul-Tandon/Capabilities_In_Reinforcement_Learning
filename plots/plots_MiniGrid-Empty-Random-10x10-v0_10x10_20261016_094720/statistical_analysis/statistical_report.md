# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## Empty-Random-10x10-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.92 ± 0.04 | 0.48 ± 1.38 | -47.5% |
| Goal Rate (±SD) | 1.000 ± 0.000 | 0.903 ± 0.306 | -9.7% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 10.0 | 1.391 | 0.1641 | ns | 0.440 | Medium |
| Goal Success Rate | 0.0 | 1.863 | 0.0625 | ns | 0.589 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.91 | 0.91 | 1.000 | 1.000 |
| 2 | 0.93 | 0.93 | 1.000 | 1.000 |
| 3 | 0.85 | 0.84 | 1.000 | 0.998 |
| 4 | 0.95 | -3.43 | 0.999 | 0.033 |
| 5 | 0.99 | 0.99 | 1.000 | 1.000 |
| 6 | 0.91 | 0.91 | 1.000 | 1.000 |
| 7 | 0.96 | 0.96 | 1.000 | 1.000 |
| 8 | 0.93 | 0.93 | 1.000 | 1.000 |
| 9 | 0.95 | 0.95 | 1.000 | 1.000 |
| 10 | 0.85 | 0.85 | 1.000 | 1.000 |

---
