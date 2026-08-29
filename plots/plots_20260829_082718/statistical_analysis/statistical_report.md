# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 100 episodes of each seed's training run

## DoorKey-8x8-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -4.06 ± 4.05 | 0.80 ± 0.03 | +119.8% |
| Goal Rate (±SD) | 0.307 ± 0.531 | 1.000 ± 0.000 | +226.1% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 1.150 | 0.2500 | ns | 0.664 | Large |
| Goal Success Rate | 0.0 | 1.150 | 0.2500 | ns | 0.664 | Large |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.62 | 0.78 | 0.920 | 1.000 |
| 2 | -6.40 | 0.79 | 0.000 | 1.000 |
| 3 | -6.40 | 0.83 | 0.000 | 1.000 |

---

## Empty-Random-8x8-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.11 ± 1.38 | 0.16 ± 1.28 | +51.3% |
| Goal Rate (±SD) | 0.737 ± 0.448 | 0.743 ± 0.427 | +0.9% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 3.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |
| Goal Success Rate | 1.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.90 | 0.90 | 0.990 | 0.980 |
| 2 | -1.48 | -1.31 | 0.220 | 0.250 |
| 3 | 0.90 | 0.89 | 1.000 | 1.000 |

---
