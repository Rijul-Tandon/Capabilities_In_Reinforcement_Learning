# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## Empty-Random-8x8-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.95 ± 0.03 | 0.95 ± 0.03 | +0.1% |
| Goal Rate (±SD) | 0.999 ± 0.002 | 1.000 ± 0.001 | +0.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 27.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |
| Goal Success Rate | 10.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.94 | 0.94 | 1.000 | 1.000 |
| 2 | 0.96 | 0.96 | 0.999 | 1.000 |
| 3 | 0.90 | 0.90 | 1.000 | 0.999 |
| 4 | 0.95 | 0.95 | 1.000 | 1.000 |
| 5 | 1.00 | 1.00 | 1.000 | 1.000 |
| 6 | 0.92 | 0.94 | 0.995 | 1.000 |
| 7 | 0.98 | 0.97 | 1.000 | 0.997 |
| 8 | 0.95 | 0.95 | 1.000 | 1.000 |
| 9 | 0.97 | 0.97 | 1.000 | 1.000 |
| 10 | 0.90 | 0.90 | 1.000 | 1.000 |

---
