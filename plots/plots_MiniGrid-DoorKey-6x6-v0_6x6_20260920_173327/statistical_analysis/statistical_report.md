# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## DoorKey-6x6-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.86 ± 0.01 | 0.86 ± 0.01 | -0.9% |
| Goal Rate (±SD) | 0.999 ± 0.002 | 0.997 ± 0.002 | -0.2% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 1.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |
| Goal Success Rate | 1.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.88 | 0.87 | 1.000 | 0.996 |
| 2 | 0.86 | 0.84 | 0.999 | 0.996 |
| 3 | 0.86 | 0.86 | 0.997 | 0.999 |

---
