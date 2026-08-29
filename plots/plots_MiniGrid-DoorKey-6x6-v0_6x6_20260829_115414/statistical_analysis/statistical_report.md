# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## DoorKey-6x6-v0
**Seeds**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   |   **N** = 10

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.86 ± 0.04 | 0.88 ± 0.03 | +1.8% |
| Goal Rate (±SD) | 0.993 ± 0.007 | 0.998 ± 0.003 | +0.4% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 3.097 | 0.0020 | ** | 0.979 | Large |
| Goal Success Rate | 0.0 | 2.886 | 0.0039 | ** | 0.912 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.87 | 0.88 | 0.998 | 1.000 |
| 2 | 0.81 | 0.84 | 0.984 | 0.992 |
| 3 | 0.86 | 0.86 | 0.996 | 0.998 |
| 4 | 0.88 | 0.89 | 0.999 | 0.999 |
| 5 | 0.90 | 0.90 | 1.000 | 1.000 |
| 6 | 0.80 | 0.84 | 0.981 | 0.993 |
| 7 | 0.87 | 0.88 | 0.994 | 0.994 |
| 8 | 0.90 | 0.92 | 0.996 | 1.000 |
| 9 | 0.89 | 0.89 | 0.998 | 0.999 |
| 10 | 0.87 | 0.90 | 0.986 | 0.999 |

---
