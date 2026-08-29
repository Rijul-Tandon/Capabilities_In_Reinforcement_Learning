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
| Mean Reward (±SD) | -9.83 ± 0.22 | 0.72 ± 0.03 | +107.4% |
| Goal Rate (±SD) | 0.000 ± 0.000 | 0.987 ± 0.005 | +0.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 1.150 | 0.2500 | ns | 0.664 | Large |
| Goal Success Rate | 0.0 | 1.150 | 0.2500 | ns | 0.664 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -9.58 | 0.69 | 0.000 | 0.988 |
| 2 | -10.00 | 0.75 | 0.000 | 0.992 |
| 3 | -9.92 | 0.72 | 0.000 | 0.982 |

---
