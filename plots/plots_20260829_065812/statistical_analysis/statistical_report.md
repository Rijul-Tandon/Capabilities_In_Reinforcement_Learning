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
| Mean Reward (±SD) | -1.68 ± 4.09 | 0.82 ± 0.02 | +148.9% |
| Goal Rate (±SD) | 0.647 ± 0.561 | 1.000 ± 0.000 | +54.6% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 1.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |
| Goal Success Rate | 0.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.82 | 0.82 | 1.000 | 1.000 |
| 2 | -6.40 | 0.81 | 0.000 | 1.000 |
| 3 | 0.53 | 0.85 | 0.940 | 1.000 |

---

## Empty-Random-8x8-v0
**Seeds**: [1, 2, 3]   |   **N** = 3

### Summary Comparison

| Metric | Baseline DDQN | RS-DDQN | Δ Improvement |
|--------|:------------:|:------:|:-------------:|
| Mean Reward (±SD) | -0.92 ± 1.64 | 0.93 ± 0.03 | +201.5% |
| Goal Rate (±SD) | 0.437 ± 0.497 | 1.000 ± 0.000 | +129.0% |

### Wilcoxon Signed-Rank Test Results

| Metric | W | Z | p-value | Sig. | Effect Size (r) | Interpretation |
|--------|:-:|:-:|:-------:|:----:|:---------------:|:--------------:|
| Episodic Reward | 1.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |
| Goal Success Rate | 0.0 | 0.674 | 0.5000 | ns | 0.389 | Medium |

### Per-Seed Raw Data

| Seed | Baseline Reward | RS-DDQN Reward | Baseline Goal | RS-DDQN Goal |
|:----:|:---------------:|:--------------:|:-------------:|:------------:|
| 1 | 0.94 | 0.94 | 1.000 | 1.000 |
| 2 | -1.53 | 0.96 | 0.250 | 1.000 |
| 3 | -2.16 | 0.90 | 0.060 | 1.000 |

---
