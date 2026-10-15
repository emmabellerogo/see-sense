**Matched-configuration audit (query 1):** PASS - 5 seeds, 6 runs each, one dataset/split/budget per seed.

### Test results per variant (mean ± sd across seeds; from query 4)

| Variant | Seeds | Test accuracy | Macro-F1 | Test loss | Params |
|---|---|---|---|---|---|
| baseline | 5 | 73.00% ± 11.64 | 0.7217 ± 0.1285 | 0.6189 ± 0.2127 | 4388 |
| full_improved | 5 | 95.20% ± 0.69 | 0.9513 ± 0.0071 | 0.1407 ± 0.0097 | 4436 |
| no_augmentation | 5 | 94.65% ± 0.68 | 0.9465 ± 0.0065 | 0.1610 ± 0.0186 | 4436 |
| no_batchnorm | 5 | 44.00% ± 31.80 | 0.3583 ± 0.3860 | 1.0528 ± 0.5332 | 4388 |
| no_dropout | 5 | 94.55% ± 1.20 | 0.9448 ± 0.0123 | 0.1466 ± 0.0255 | 4436 |
| no_scheduler | 5 | 94.60% ± 1.40 | 0.9460 ± 0.0138 | 0.1314 ± 0.0205 | 4436 |

### Test accuracy by seed

| Variant | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| baseline | 71.75% | 71.75% | 90.25% | 73.75% | 57.50% |
| full_improved | 94.75% | 94.25% | 96.00% | 95.50% | 95.50% |
| no_augmentation | 94.25% | 94.50% | 94.75% | 94.00% | 95.75% |
| no_batchnorm | 57.75% | 22.75% | 94.00% | 22.75% | 22.75% |
| no_dropout | 96.25% | 93.75% | 93.25% | 94.25% | 95.25% |
| no_scheduler | 94.75% | 92.25% | 96.00% | 95.00% | 95.00% |

### Difference from full_improved, same seed (from query 6)

Negative accuracy delta = scored lower than the full model. Accuracy deltas in percentage points (pp).

| Variant | Technique removed | Δ accuracy per seed (pp) | Mean Δ acc (pp) | 95% CI of mean Δ (pp, paired t) | Mean Δ macro-F1 | Mean Δ test loss | Seeds lower than full |
|---|---|---|---|---|---|---|---|
| baseline | architecture + all four | -23.00, -22.50, -5.75, -21.75, -38.00 | -22.20 | [-36.37, -8.03] | -0.2296 | +0.4782 | 5 of 5 |
| no_augmentation | augmentation | -0.50, +0.25, -1.25, -1.50, +0.25 | -0.55 | [-1.57, +0.47] includes 0 | -0.0048 | +0.0203 | 3 of 5 |
| no_batchnorm | batch norm | -37.00, -71.50, -2.00, -72.75, -72.75 | -51.20 | [-90.28, -12.12] | -0.5931 | +0.9121 | 5 of 5 |
| no_dropout | dropout | +1.50, -0.50, -2.75, -1.25, -0.25 | -0.65 | [-2.57, +1.27] includes 0 | -0.0066 | +0.0059 | 4 of 5 |
| no_scheduler | LR scheduler | +0.00, -2.00, +0.00, -0.50, -0.50 | -0.60 | [-1.62, +0.42] includes 0 | -0.0053 | -0.0093 | 3 of 5 |

### Validation / training evidence (from query 2)

| Variant | Best val loss | Val acc at best | Best epoch (0-based) per seed | Final train loss |
|---|---|---|---|---|
| baseline | 0.6390 | 71.60% | 25, 28, 7, 8, 4 | 0.4372 |
| full_improved | 0.1343 | 96.50% | 26, 29, 20, 23, 20 | 0.1737 |
| no_augmentation | 0.1384 | 95.00% | 23, 17, 20, 24, 25 | 0.0522 |
| no_batchnorm | 1.0515 | 43.20% | 21, 0, 25, 0, 10 | 1.0941 |
| no_dropout | 0.1220 | 96.60% | 26, 24, 25, 23, 20 | 0.1104 |
| no_scheduler | 0.1201 | 95.80% | 26, 29, 29, 26, 25 | 0.1632 |

### Most common confusions (summed over seeds, from query 7)

| Variant | Errors | Top confusions (true → predicted: count) |
|---|---|---|
| baseline | 540 | horizontal → block: 125; vertical → block: 111; block → horizontal: 104 |
| full_improved | 96 | block → horizontal: 30; block → vertical: 24; vertical → block: 11 |
| no_augmentation | 107 | block → horizontal: 40; horizontal → block: 23; block → vertical: 18 |
| no_batchnorm | 1120 | horizontal → vertical: 375; block → vertical: 318; diagonal → vertical: 299 |
| no_dropout | 109 | block → horizontal: 32; block → vertical: 27; vertical → block: 16 |
| no_scheduler | 108 | block → horizontal: 31; horizontal → block: 19; vertical → block: 18 |
