# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 100 episodes of each seed's training run

## DoorKey-10x10-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -10.00 ± 0.00 | -3.58 ± 5.57 | +64.2% |
| Goal Rate (±SD) | 0.000 ± 0.000 | 0.570 ± 0.494 | +0.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 0.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |
| Goal Success Rate | 0.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -10.00 | -0.70 | 0.000 | 0.830 |
| 2 | -10.00 | -10.00 | 0.000 | 0.000 |
| 3 | -10.00 | -0.04 | 0.000 | 0.880 |

---

## Empty-Random-10x10-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -0.85 ± 2.06 | -0.74 ± 2.83 | +12.9% |
| Goal Rate (±SD) | 0.587 ± 0.501 | 0.667 ± 0.577 | +13.6% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 3.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |
| Goal Success Rate | 1.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | -3.13 | 0.92 | 0.030 | 1.000 |
| 2 | -0.28 | -4.00 | 0.730 | 0.000 |
| 3 | 0.87 | 0.87 | 1.000 | 1.000 |

---
