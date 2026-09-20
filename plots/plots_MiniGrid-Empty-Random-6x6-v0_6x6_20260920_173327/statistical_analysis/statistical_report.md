# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## Empty-Random-6x6-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.96 ± 0.02 | 0.96 ± 0.02 | +0.0% |
| Goal Rate (±SD) | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |
| Goal Success Rate | — | 0.000 | 1.0000 | ns | 0.000 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.96 | 0.96 | 1.000 | 1.000 |
| 2 | 0.98 | 0.98 | 1.000 | 1.000 |
| 3 | 0.94 | 0.94 | 1.000 | 1.000 |

---
