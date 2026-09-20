# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## DoorKey-8x8-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -1.60 ± 4.16 | 0.76 ± 0.05 | +147.2% |
| Goal Rate (±SD) | 0.663 ± 0.574 | 0.983 ± 0.012 | +48.2% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 2.0 | 0.319 | 0.7500 | ns | 0.184 | Small |
| Goal Success Rate | 2.0 | 0.319 | 0.7500 | ns | 0.184 | Small |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.80 | 0.81 | 0.993 | 0.996 |
| 2 | 0.80 | 0.70 | 0.996 | 0.976 |
| 3 | -6.40 | 0.76 | 0.000 | 0.976 |

---
