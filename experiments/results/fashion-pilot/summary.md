# Fashion-MNIST runtime pilot — summary

**Scope:** training and validation only. The official test files were never opened (`fashion.load(include_test=False)`), nothing was written to
Supabase, and none of these numbers is a result. Date: 2026-10-17. Machine: Apple M2, 8 cores, 16 GB, CPU only, other applications running (load average about 9).
Settings: Adam, lr 0.001, batch 128, 54,000 training / 6,000 validation images, `split_seed 0`, `train_seed 0` unless noted.

## What was measured
- Seconds per epoch: **15.4 s** mean over the six variants run sequentially (13.4 s `no_dropout` … 17.3 s `no_batchnorm`); 10–12 s for a single process on a quieter machine.
- Parallelism: three processes with 2 threads each ran at 22 s/epoch (baseline) and 31 s/epoch (full_improved) *each*, about 1.6–1.9× the single-process throughput.
- 20 epochs of one run ≈ 5 min busy; six variants × 3 seeds ≈ 1.5 h sequential, ≈ 50–60 min with three processes. See `docs/FASHION_PROTOCOL.md`.

## What the curves showed (validation-based; used only to size the budget, not to tune)
- 20-epoch baseline: validation loss 0.413 → 0.266 (best epoch 19 of 20, 0.257), validation accuracy 0.908, training accuracy 0.924: still improving slowly, mild train/validation gap, no overfitting collapse.
- 20-epoch full_improved: validation loss 0.459 → 0.306, validation accuracy 0.888, training accuracy 0.889: it **under-fits relative to the baseline** at this budget (dropout 0.3 + augmentation + a small network). Dropout, shift size, learning rate and epochs were **not** changed in response.
- Same seed, same settings, different thread counts (2 vs 4): identical epoch losses for baseline and full_improved on this machine.


## A. all six variants, 3 epochs (machine under other load)

File: `pilot_all_variants_3ep.jsonl`; 6 variant-runs; epochs 3, batch 128, lr 0.001, 54000 train / 6000 validation images, torch threads 4.

| Variant | seed | s/epoch (mean) | s/epoch (min–max) | final train loss | final val loss | final train acc | final val acc | best val-loss epoch |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 11.0 | 9.9–13.1 | 0.3378 | 0.3179 | 0.8922 | 0.8872 | 3 |
| full_improved | 0 | 14.3 | 12.6–16.1 | 0.5123 | 0.3891 | 0.8601 | 0.8592 | 3 |
| no_augmentation | 0 | 14.5 | 12.4–16.9 | 0.3556 | 0.3123 | 0.8930 | 0.8893 | 3 |
| no_batchnorm | 0 | 19.0 | 11.6–33.9 | 0.5811 | 0.4556 | 0.8355 | 0.8303 | 3 |
| no_dropout | 0 | 12.2 | 11.4–13.5 | 0.4440 | 0.3631 | 0.8691 | 0.8683 | 3 |
| no_scheduler | 0 | 13.9 | 11.5–18.2 | 0.5106 | 0.3732 | 0.8649 | 0.8658 | 3 |

Mean over all epochs in this file: **14.2 s/epoch** (n = 18).

## B. convergence check, baseline + full_improved, 20 epochs

File: `pilot_convergence_20ep.jsonl`; 2 variant-runs; epochs 20, batch 128, lr 0.001, 54000 train / 6000 validation images, torch threads 4.

| Variant | seed | s/epoch (mean) | s/epoch (min–max) | final train loss | final val loss | final train acc | final val acc | best val-loss epoch |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 11.1 | 9.6–16.5 | 0.2157 | 0.2660 | 0.9239 | 0.9078 | 19 |
| full_improved | 0 | 19.0 | 12.1–31.0 | 0.3971 | 0.3055 | 0.8890 | 0.8878 | 18 |

Mean over all epochs in this file: **15.1 s/epoch** (n = 40).

## C. timing, all six variants, 2 epochs, sequential

File: `pilot_timing_sequential_2ep.jsonl`; 6 variant-runs; epochs 2, batch 128, lr 0.001, 54000 train / 6000 validation images, torch threads 4.

| Variant | seed | s/epoch (mean) | s/epoch (min–max) | final train loss | final val loss | final train acc | final val acc | best val-loss epoch |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 14.5 | 14.3–14.7 | 0.3828 | 0.3714 | 0.8682 | 0.8667 | 2 |
| full_improved | 0 | 17.0 | 16.4–17.6 | 0.5589 | 0.4166 | 0.8469 | 0.8483 | 2 |
| no_augmentation | 0 | 16.5 | 15.2–17.8 | 0.3863 | 0.3336 | 0.8853 | 0.8805 | 2 |
| no_batchnorm | 0 | 17.3 | 15.8–18.9 | 0.6350 | 0.4927 | 0.8149 | 0.8125 | 2 |
| no_dropout | 0 | 13.4 | 12.3–14.4 | 0.4864 | 0.3908 | 0.8578 | 0.8562 | 2 |
| no_scheduler | 0 | 13.7 | 12.8–14.6 | 0.5567 | 0.4042 | 0.8529 | 0.8517 | 2 |

Mean over all epochs in this file: **15.4 s/epoch** (n = 12).

## D. timing, 3 processes in parallel (2 threads each), baseline + full_improved, 2 epochs

File: `pilot_timing_parallel3_2ep.jsonl`; 6 variant-runs; epochs 2, batch 128, lr 0.001, 54000 train / 6000 validation images, torch threads 2.

| Variant | seed | s/epoch (mean) | s/epoch (min–max) | final train loss | final val loss | final train acc | final val acc | best val-loss epoch |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 22.4 | 20.8–24.1 | 0.4075 | 0.3776 | 0.8682 | 0.8610 | 2 |
| full_improved | 1 | 31.3 | 28.4–34.2 | 0.5458 | 0.4101 | 0.8480 | 0.8485 | 2 |
| baseline | 2 | 22.3 | 20.7–23.9 | 0.3804 | 0.3426 | 0.8801 | 0.8777 | 2 |
| full_improved | 2 | 31.5 | 27.9–35.2 | 0.5687 | 0.4297 | 0.8479 | 0.8475 | 2 |
| baseline | 0 | 22.3 | 20.9–23.8 | 0.3828 | 0.3714 | 0.8682 | 0.8667 | 2 |
| full_improved | 0 | 31.9 | 29.1–34.7 | 0.5589 | 0.4166 | 0.8469 | 0.8483 | 2 |

Mean over all epochs in this file: **27.0 s/epoch** (n = 12).

