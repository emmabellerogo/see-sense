# SQL provenance

- source: **supabase-sql-editor**
- export file: `fashion-official-1-supabase-export.csv` sha256 `474ee3db5f915e89cef75560f7e231b39ecd800a6f20347908831f9271d42c20`
- executed_at (reported by the database): 2026-10-17T22:49:57.518155+00:00
- server_version: PostgreSQL 17.11 on aarch64-unknown-linux-gnu, compiled by gcc (GCC) 15.2.0, 64-bit
- database: postgres
- experiment_group: fashion-official-1
- queries file: docs/comparison_queries.sql sha256 `5d31a4fe21db31b8c9f4e5d8731cb7e369532b9b026307caf71661b4c4741385`

| query | rows |
|---|---|
| q1 | 3 |
| q10 | 180 |
| q2 | 18 |
| q3 | 360 |
| q4 | 18 |
| q5 | 6 |
| q6 | 15 |
| q7 | 1800 |
| q8 | 5 |
| q9 | 18 |

**Matched-configuration audit (query 1):** PASS - 3 seeds, 6 runs each, one dataset/split/budget per seed.

### Test results per variant (mean ± sd across seeds; from query 4)

| Variant | Seeds | Test accuracy | Macro-F1 | Test loss | Params |
|---|---|---|---|---|---|
| baseline | 3 | 89.73% ± 0.47 | 0.8973 ± 0.0047 | 0.2884 ± 0.0093 | 9098 |
| full_improved | 3 | 88.10% ± 0.29 | 0.8801 ± 0.0031 | 0.3327 ± 0.0057 | 9146 |
| no_augmentation | 3 | 90.30% ± 0.13 | 0.9024 ± 0.0015 | 0.2720 ± 0.0043 | 9146 |
| no_batchnorm | 3 | 87.12% ± 0.30 | 0.8701 ± 0.0031 | 0.3591 ± 0.0072 | 9098 |
| no_dropout | 3 | 88.94% ± 0.07 | 0.8888 ± 0.0006 | 0.3093 ± 0.0029 | 9146 |
| no_scheduler | 3 | 88.23% ± 0.40 | 0.8810 ± 0.0039 | 0.3274 ± 0.0094 | 9146 |

### Test accuracy by seed

| Variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| baseline | 90.01% | 89.19% | 89.98% |
| full_improved | 87.82% | 88.10% | 88.39% |
| no_augmentation | 90.29% | 90.44% | 90.18% |
| no_batchnorm | 86.89% | 87.02% | 87.46% |
| no_dropout | 88.97% | 88.86% | 88.98% |
| no_scheduler | 87.77% | 88.50% | 88.41% |

### Difference from full_improved, same seed (from query 6)

Negative accuracy delta = scored lower than the full model. Accuracy deltas in percentage points (pp).

| Variant | Technique removed | Δ accuracy per seed (pp) | Mean Δ acc (pp) | 95% CI of mean Δ (pp, paired t) | Mean Δ macro-F1 | Mean Δ test loss | Seeds lower than full |
|---|---|---|---|---|---|---|---|
| baseline | architecture + all four | +2.19, +1.09, +1.59 | +1.62 | [+0.26, +2.99] | +0.0172 | -0.0443 | 0 of 3 |
| no_augmentation | augmentation | +2.47, +2.34, +1.79 | +2.20 | [+1.30, +3.10] | +0.0223 | -0.0607 | 0 of 3 |
| no_batchnorm | batch norm | -0.93, -1.08, -0.93 | -0.98 | [-1.20, -0.76] | -0.0101 | +0.0264 | 3 of 3 |
| no_dropout | dropout | +1.15, +0.76, +0.59 | +0.83 | [+0.12, +1.55] | +0.0087 | -0.0234 | 0 of 3 |
| no_scheduler | LR scheduler | -0.05, +0.40, +0.02 | +0.12 | [-0.48, +0.72] includes 0 | +0.0009 | -0.0054 | 1 of 3 |

### Validation / training evidence (from query 2)

| Variant | Best val loss | Val acc at best | Best epoch (0-based) per seed | Final train loss |
|---|---|---|---|---|
| baseline | 0.2571 | 90.75% | 18, 15, 18 | 0.2173 |
| full_improved | 0.3072 | 88.98% | 17, 19, 19 | 0.3951 |
| no_augmentation | 0.2472 | 91.25% | 19, 19, 19 | 0.2602 |
| no_batchnorm | 0.3340 | 88.04% | 18, 19, 18 | 0.4207 |
| no_dropout | 0.2810 | 89.84% | 19, 18, 19 | 0.3266 |
| no_scheduler | 0.3013 | 89.09% | 13, 18, 19 | 0.3920 |

### Fit at the last epoch: training vs validation (from query 3, mean over seeds)

Training accuracy is the whole un-augmented training split scored in eval mode, so it is comparable to validation accuracy: a small gap with modest accuracy suggests under-fitting, a large gap over-fitting. Training LOSS is different: it is the mean mini-batch loss in training mode on augmented data with dropout, so it sits above the validation loss for the regularized variants and is not comparable to it.

| Variant | Train acc | Val acc | Gap (pp) | Train loss | Val loss |
|---|---|---|---|---|---|
| baseline | 92.39% | 90.69% | +1.70 | 0.2173 | 0.2641 |
| full_improved | 88.95% | 88.97% | -0.02 | 0.3951 | 0.3074 |
| no_augmentation | 92.23% | 91.25% | +0.98 | 0.2602 | 0.2472 |
| no_batchnorm | 88.03% | 88.01% | +0.02 | 0.4207 | 0.3341 |
| no_dropout | 90.06% | 89.85% | +0.21 | 0.3266 | 0.2810 |
| no_scheduler | 89.12% | 88.99% | +0.13 | 0.3920 | 0.3041 |

### Most common confusions (summed over seeds, from query 7)

| Variant | Errors | Top confusions (true → predicted: count) |
|---|---|---|
| baseline | 3082 | T-shirt/top → Shirt: 354; Shirt → T-shirt/top: 319; Shirt → Coat: 207; Shirt → Pullover: 202; Coat → Shirt: 202 |
| full_improved | 3569 | Shirt → T-shirt/top: 462; Shirt → Coat: 337; T-shirt/top → Shirt: 244; Pullover → Shirt: 240; Pullover → Coat: 220 |
| no_augmentation | 2909 | Shirt → T-shirt/top: 391; T-shirt/top → Shirt: 257; Shirt → Coat: 242; Shirt → Pullover: 204; Pullover → Coat: 184 |
| no_batchnorm | 3863 | Shirt → T-shirt/top: 514; Shirt → Coat: 359; Pullover → Coat: 261; Pullover → Shirt: 259; Shirt → Pullover: 222 |
| no_dropout | 3319 | Shirt → T-shirt/top: 419; Shirt → Coat: 297; T-shirt/top → Shirt: 257; Pullover → Shirt: 244; Shirt → Pullover: 190 |
| no_scheduler | 3532 | Shirt → T-shirt/top: 441; Shirt → Coat: 360; T-shirt/top → Shirt: 251; Pullover → Coat: 230; Shirt → Pullover: 220 |

### Per-class test precision and recall (mean ± sd over seeds; from query 7)

Computed from each run's saved confusion matrix. Precision = correct / predicted as that class; recall = correct / truly that class. A class never predicted by a run has precision scored as 0 (shown with *).

| Variant | Class | Precision | Recall |
|---|---|---|---|
| baseline | T-shirt/top | 85.2% ± 2.2 | 82.4% ± 2.7 |
| baseline | Trouser | 98.1% ± 0.4 | 98.4% ± 0.0 |
| baseline | Pullover | 83.8% ± 5.2 | 84.2% ± 3.3 |
| baseline | Dress | 90.0% ± 0.9 | 89.7% ± 1.6 |
| baseline | Coat | 83.3% ± 1.0 | 84.0% ± 0.2 |
| baseline | Sandal | 97.6% ± 0.6 | 97.6% ± 1.0 |
| baseline | Shirt | 70.8% ± 2.3 | 71.2% ± 3.5 |
| baseline | Sneaker | 93.9% ± 1.5 | 97.0% ± 0.3 |
| baseline | Bag | 97.5% ± 0.9 | 97.9% ± 0.7 |
| baseline | Ankle boot | 97.8% ± 0.6 | 94.8% ± 0.8 |
| full_improved | T-shirt/top | 82.2% ± 0.9 | 83.7% ± 0.6 |
| full_improved | Trouser | 98.6% ± 0.3 | 97.5% ± 0.3 |
| full_improved | Pullover | 84.1% ± 1.4 | 82.5% ± 0.9 |
| full_improved | Dress | 84.8% ± 1.0 | 89.8% ± 0.5 |
| full_improved | Coat | 78.6% ± 0.2 | 83.8% ± 0.5 |
| full_improved | Sandal | 97.8% ± 0.4 | 94.1% ± 1.0 |
| full_improved | Shirt | 69.4% ± 1.1 | 61.5% ± 1.6 |
| full_improved | Sneaker | 91.4% ± 0.7 | 96.1% ± 0.3 |
| full_improved | Bag | 98.0% ± 0.4 | 96.9% ± 0.7 |
| full_improved | Ankle boot | 95.3% ± 0.4 | 95.2% ± 0.3 |
| no_augmentation | T-shirt/top | 84.2% ± 1.0 | 86.2% ± 0.6 |
| no_augmentation | Trouser | 99.1% ± 0.2 | 97.5% ± 0.2 |
| no_augmentation | Pullover | 84.9% ± 0.4 | 85.4% ± 0.4 |
| no_augmentation | Dress | 88.9% ± 0.1 | 91.4% ± 0.2 |
| no_augmentation | Coat | 83.3% ± 1.2 | 86.0% ± 0.8 |
| no_augmentation | Sandal | 98.0% ± 0.2 | 97.6% ± 0.2 |
| no_augmentation | Shirt | 74.0% ± 0.5 | 67.8% ± 1.4 |
| no_augmentation | Sneaker | 94.8% ± 0.3 | 96.9% ± 0.1 |
| no_augmentation | Bag | 97.7% ± 0.4 | 97.9% ± 0.3 |
| no_augmentation | Ankle boot | 97.4% ± 0.1 | 96.2% ± 0.2 |
| no_batchnorm | T-shirt/top | 80.4% ± 1.4 | 83.4% ± 1.6 |
| no_batchnorm | Trouser | 98.6% ± 0.2 | 97.7% ± 0.3 |
| no_batchnorm | Pullover | 82.6% ± 0.7 | 80.4% ± 1.0 |
| no_batchnorm | Dress | 84.0% ± 0.8 | 88.4% ± 0.7 |
| no_batchnorm | Coat | 76.6% ± 0.6 | 83.0% ± 1.0 |
| no_batchnorm | Sandal | 97.6% ± 0.4 | 93.4% ± 0.5 |
| no_batchnorm | Shirt | 67.1% ± 1.1 | 58.1% ± 1.7 |
| no_batchnorm | Sneaker | 90.9% ± 0.2 | 95.7% ± 0.6 |
| no_batchnorm | Bag | 97.6% ± 0.2 | 96.3% ± 0.4 |
| no_batchnorm | Ankle boot | 94.9% ± 0.8 | 95.0% ± 0.6 |
| no_dropout | T-shirt/top | 83.3% ± 1.5 | 84.2% ± 0.5 |
| no_dropout | Trouser | 98.3% ± 0.4 | 97.7% ± 0.3 |
| no_dropout | Pullover | 84.3% ± 0.4 | 83.8% ± 1.3 |
| no_dropout | Dress | 86.9% ± 1.4 | 90.3% ± 0.3 |
| no_dropout | Coat | 80.9% ± 0.2 | 83.5% ± 0.5 |
| no_dropout | Sandal | 97.6% ± 0.5 | 95.6% ± 0.3 |
| no_dropout | Shirt | 70.6% ± 0.8 | 65.5% ± 1.2 |
| no_dropout | Sneaker | 92.8% ± 0.4 | 96.4% ± 0.5 |
| no_dropout | Bag | 98.3% ± 0.2 | 97.1% ± 0.5 |
| no_dropout | Ankle boot | 95.9% ± 0.4 | 95.3% ± 0.2 |
| no_scheduler | T-shirt/top | 82.8% ± 0.3 | 83.0% ± 1.6 |
| no_scheduler | Trouser | 98.8% ± 0.2 | 97.6% ± 0.1 |
| no_scheduler | Pullover | 83.4% ± 2.8 | 82.8% ± 2.5 |
| no_scheduler | Dress | 83.9% ± 2.4 | 90.7% ± 0.2 |
| no_scheduler | Coat | 78.0% ± 2.0 | 84.6% ± 3.3 |
| no_scheduler | Sandal | 97.3% ± 0.9 | 95.5% ± 0.8 |
| no_scheduler | Shirt | 70.8% ± 1.4 | 60.2% ± 1.1 |
| no_scheduler | Sneaker | 93.3% ± 0.6 | 95.2% ± 1.4 |
| no_scheduler | Bag | 98.1% ± 0.4 | 96.9% ± 0.8 |
| no_scheduler | Ankle boot | 94.9% ± 1.8 | 95.7% ± 0.9 |

**Cross-check:** stored per-class precision/recall (query 10, 180 rows) vs recomputed from confusion matrices (query 7): max absolute difference 4.44e-16.

### Calibration: temperature scaling fitted on validation (query 9)

ECE = expected calibration error (15 bins; lower is better); NLL = negative log-likelihood. 'Before' is plain softmax (T = 1); 'after' divides the logits by the temperature fitted on validation data. Test columns are held-out measurements, never used to fit T. Mean over seeds.

| Variant | Mean T | Test ECE before | Test ECE after | Test NLL before | Test NLL after |
|---|---|---|---|---|---|
| baseline | 1.085 | 0.0134 | 0.0071 | 0.2884 | 0.2856 |
| full_improved | 0.884 | 0.0173 | 0.0069 | 0.3327 | 0.3325 |
| no_augmentation | 0.935 | 0.0087 | 0.0068 | 0.2720 | 0.2726 |
| no_batchnorm | 0.884 | 0.0190 | 0.0079 | 0.3591 | 0.3586 |
| no_dropout | 0.953 | 0.0073 | 0.0082 | 0.3093 | 0.3100 |
| no_scheduler | 0.917 | 0.0108 | 0.0071 | 0.3274 | 0.3277 |
