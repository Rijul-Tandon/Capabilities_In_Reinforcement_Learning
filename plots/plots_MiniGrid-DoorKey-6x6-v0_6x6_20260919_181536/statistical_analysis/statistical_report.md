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
| Mean Reward (±SD) | 0.85 ± 0.03 | 0.84 ± 0.05 | -0.6% |
| Goal Rate (±SD) | 0.993 ± 0.008 | 0.993 ± 0.012 | +0.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 3.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |
| Goal Success Rate | 3.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.87 | 0.88 | 0.998 | 1.000 |
| 2 | 0.81 | 0.85 | 0.984 | 0.999 |
| 3 | 0.86 | 0.79 | 0.996 | 0.979 |

---
