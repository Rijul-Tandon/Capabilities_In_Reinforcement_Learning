# Statistical Analysis: Baseline DDQN vs RS-DDQN

> **Test**: Wilcoxon signed-rank test (two-sided, paired, non-parametric)
> **Significance**: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
> **Effect size**: r = |Z| / √N (Small ≥0.1, Medium ≥0.3, Large ≥0.5)
> **Metrics**: Averaged over the last 100 episodes of each seed's training run

## DoorKey-6x6-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | 0.71 ± 0.29 | 0.85 ± 0.03 | +20.9% |
| Goal Rate (±SD) | 0.947 ± 0.092 | 0.997 ± 0.006 | +5.3% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 2.0 | 0.319 | 0.7500 | ns | 0.184 | Small |
| Goal Success Rate | 1.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.88 | 0.88 | 1.000 | 1.000 |
| 2 | 0.37 | 0.86 | 0.840 | 1.000 |
| 3 | 0.87 | 0.82 | 1.000 | 0.990 |

---

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
| Episodic Reward | 0.0 | 0.000 | 1.0000 | ns | 0.000 | Negligible |
| Goal Success Rate | — | 0.000 | 1.0000 | ns | 0.000 | Negligible |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.96 | 0.96 | 1.000 | 1.000 |
| 2 | 0.98 | 0.98 | 1.000 | 1.000 |
| 3 | 0.94 | 0.94 | 1.000 | 1.000 |

---
