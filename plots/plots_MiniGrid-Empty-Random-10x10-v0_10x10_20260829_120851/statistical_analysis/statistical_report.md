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
| Mean Reward (±SD) | 0.56 ± 0.61 | 0.92 ± 0.05 | +63.2% |
| Goal Rate (±SD) | 0.863 ± 0.225 | 1.000 ± 0.001 | +15.8% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 14.0 | 1.301 | 0.1934 | ns | 0.411 | Medium |
| Goal Success Rate | 13.0 | 0.602 | 0.5469 | ns | 0.191 | Small |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -0.61 | 0.91 | 0.424 | 1.000 |
| 2 | -0.17 | 0.93 | 0.602 | 1.000 |
| 3 | -0.11 | 0.85 | 0.608 | 1.000 |
| 4 | 0.94 | 0.94 | 0.998 | 0.998 |
| 5 | 0.99 | 0.99 | 1.000 | 1.000 |
| 6 | 0.91 | 0.91 | 1.000 | 1.000 |
| 7 | 0.96 | 0.96 | 1.000 | 1.000 |
| 8 | 0.93 | 0.93 | 1.000 | 0.999 |
| 9 | 0.95 | 0.95 | 1.000 | 1.000 |
| 10 | 0.85 | 0.85 | 1.000 | 0.999 |

---
