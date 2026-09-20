# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 10% of training steps for each seed's training run

## DoorKey-10x10-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -9.78 ± 0.37 | 0.74 ± 0.02 | +107.6% |
| Goal Rate (±SD) | 0.000 ± 0.000 | 0.992 ± 0.006 | +0.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 1.150 | 0.2500 | ns | 0.664 | Large |
| Goal Success Rate | 0.0 | 1.150 | 0.2500 | ns | 0.664 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -9.35 | 0.72 | 0.000 | 0.997 |
| 2 | -10.00 | 0.76 | 0.000 | 0.994 |
| 3 | -9.98 | 0.74 | 0.000 | 0.985 |

---
