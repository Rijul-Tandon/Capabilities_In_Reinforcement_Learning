# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## Empty-Random-8x8-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.92 ± 0.03 | 0.03 ± 1.51 | -96.9% |
| Goal Rate (±SD) | 0.997 ± 0.002 | 0.717 ± 0.478 | -28.1% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 1.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |
| Goal Success Rate | 1.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.92 | 0.94 | 0.995 | 1.000 |
| 2 | 0.95 | -1.71 | 0.998 | 0.165 |
| 3 | 0.90 | 0.86 | 0.999 | 0.987 |

---
