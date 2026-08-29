# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## Empty-Random-10x10-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -0.30 ± 0.27 | 0.90 ± 0.04 | +402.6% |
| Goal Rate (±SD) | 0.544 ± 0.104 | 1.000 ± 0.000 | +83.6% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 1.150 | 0.2500 | ns | 0.664 | Large |
| Goal Success Rate | 0.0 | 1.150 | 0.2500 | ns | 0.664 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -0.61 | 0.91 | 0.424 | 1.000 |
| 2 | -0.17 | 0.93 | 0.602 | 1.000 |
| 3 | -0.11 | 0.85 | 0.608 | 1.000 |

---
