**Matched-configuration audit (query 1):** PASS - 5 seeds, 6 runs each, one dataset/split/budget per seed.

### Test results per variant (mean ± sd across seeds; from query 4)

| Variant | Seeds | Test accuracy | Macro-F1 | Test loss | Params |
|---|---|---|---|---|---|
| baseline | 5 | 76.80% ± 8.93 | 0.7655 ± 0.0882 | 0.5686 ± 0.1883 | 4388 |
| full_improved | 5 | 95.05% ± 0.67 | 0.9503 ± 0.0070 | 0.1456 ± 0.0080 | 4436 |
| no_augmentation | 5 | 93.25% ± 1.09 | 0.9323 ± 0.0109 | 0.1737 ± 0.0195 | 4436 |
| no_batchnorm | 5 | 67.45% ± 22.93 | 0.6614 ± 0.2413 | 0.7725 ± 0.4379 | 4388 |
| no_dropout | 5 | 94.70% ± 1.34 | 0.9464 ± 0.0139 | 0.1420 ± 0.0291 | 4436 |
| no_scheduler | 5 | 95.05% ± 0.96 | 0.9496 ± 0.0103 | 0.1284 ± 0.0056 | 4436 |

### Test accuracy by seed

| Variant | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| baseline | 72.00% | 68.25% | 90.75% | 72.75% | 80.25% |
| full_improved | 94.00% | 94.75% | 95.50% | 95.50% | 95.50% |
| no_augmentation | 93.00% | 92.50% | 94.25% | 92.00% | 94.50% |
| no_batchnorm | 58.50% | 79.00% | 93.75% | 72.75% | 33.25% |
| no_dropout | 96.50% | 94.00% | 93.75% | 93.50% | 95.75% |
| no_scheduler | 93.75% | 95.50% | 95.25% | 96.25% | 94.50% |

### Difference from full_improved, same seed (from query 6)

Negative accuracy delta = scored lower than the full model. Accuracy deltas in percentage points (pp).

| Variant | Technique removed | Δ accuracy per seed (pp) | Mean Δ acc (pp) | 95% CI of mean Δ (pp, paired t) | Mean Δ macro-F1 | Mean Δ test loss | Seeds lower than full |
|---|---|---|---|---|---|---|---|
| baseline | architecture + all four | -22.00, -26.50, -4.75, -22.75, -15.25 | -18.25 | [-28.89, -7.61] | -0.1848 | +0.4229 | 5 of 5 |
| no_augmentation | augmentation | -1.00, -2.25, -1.25, -3.50, -1.00 | -1.80 | [-3.14, -0.46] | -0.0180 | +0.0280 | 5 of 5 |
| no_batchnorm | batch norm | -35.50, -15.75, -1.75, -22.75, -62.25 | -27.60 | [-56.01, +0.81] includes 0 | -0.2889 | +0.6269 | 5 of 5 |
| no_dropout | dropout | +2.50, -0.75, -1.75, -2.00, +0.25 | -0.35 | [-2.62, +1.92] includes 0 | -0.0039 | -0.0037 | 3 of 5 |
| no_scheduler | LR scheduler | -0.25, +0.75, -0.25, +0.75, -1.00 | -0.00 | [-0.93, +0.93] includes 0 | -0.0007 | -0.0172 | 3 of 5 |

### Validation / training evidence (from query 2)

| Variant | Best val loss | Val acc at best | Best epoch (0-based) per seed | Final train loss |
|---|---|---|---|---|
| baseline | 0.5841 | 75.70% | 25, 28, 7, 8, 12 | 0.3437 |
| full_improved | 0.1298 | 96.00% | 26, 29, 23, 20, 20 | 0.1625 |
| no_augmentation | 0.1525 | 94.50% | 25, 22, 28, 27, 22 | 0.0581 |
| no_batchnorm | 0.7869 | 64.60% | 21, 26, 21, 29, 3 | 0.8764 |
| no_dropout | 0.1216 | 96.60% | 26, 24, 25, 23, 22 | 0.1114 |
| no_scheduler | 0.1150 | 96.40% | 25, 27, 29, 28, 28 | 0.1567 |

### Most common confusions (summed over seeds, from query 7)

| Variant | Errors | Top confusions (true → predicted: count) |
|---|---|---|
| baseline | 464 | block → horizontal: 93; horizontal → block: 92; vertical → block: 91 |
| full_improved | 99 | block → horizontal: 33; block → vertical: 19; horizontal → block: 13 |
| no_augmentation | 135 | block → horizontal: 44; horizontal → block: 30; block → vertical: 25 |
| no_batchnorm | 651 | block → horizontal: 139; vertical → horizontal: 81; block → vertical: 75 |
| no_dropout | 106 | block → horizontal: 33; block → vertical: 24; vertical → block: 15 |
| no_scheduler | 99 | block → vertical: 30; block → horizontal: 28; vertical → block: 13 |
