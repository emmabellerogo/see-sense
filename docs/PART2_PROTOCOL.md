# Part 2: frozen experimental protocol

Frozen on **2026-10-15**, before any official run and before any test metric
was looked at. Every choice below was made from training and validation
evidence only. The exploration script (`experiments/explore.py`) deletes test
metrics before printing, and its raw output is in `experiments/results/explore_*.jsonl`.

## Frozen protocol

| Item | Value |
|---|---|
| Dataset ID / rows / noise / data seed | **new dataset** `shapes-2000-noise1.0`: 2000 rows, noise 1.0, seed 42 (ID 2 in the local run; register it in Supabase with `POST /datasets`, see below) |
| `split_seed`, test fraction, validation fraction | 0, 0.2 (400 images), 0.1 (200 images); train = 1400 |
| Training seeds | 0, 1, 2, 3, 4 (5 seeds → 30 runs) |
| Epochs, batch size, initial learning rate (Adam) | 30, 32, 0.01 |
| Dropout, augmentation shift, scheduler | 0.3, 1 px (+ flips), cosine |
| Model selection | each run keeps the weights of its lowest-validation-loss epoch; test scored once afterwards (unchanged from Part 1) |
| Experiment group | `official-1` |
| Protocol fingerprint (dataset ID 2) | `cf05f9546cbbb0d4eb1cadfe0f7854bee883ffd9899ba5b1388aab3d6d9a578d` |
| Frozen on (date) | 2026-10-15 |

## Evidence used (validation and training only)

Exploration used training seeds 100 and 101, never the official seeds, with
`split_seed` 0.

**1. Dataset difficulty** (`explore_difficulty*.jsonl`; lr 0.01, batch 32, 15 epochs)

| Rows | Noise | Baseline val acc | Full-improved val acc |
|---|---|---|---|
| 800 | 0.1 (Part 1 default) | 100%, 100% | 100%, 100% |
| 2000 | 0.5 | 100%, 100% | 100%, 100% |
| 2000 | 0.9 | 93%, 90% | 98.5%, 97% |
| 2000 | **1.0** | **85.5%, 91.5%** | **95%, 93.5%** |
| 1200 | 1.0 | 81.7%, 76.7% | 90.8%, 87.5% |

The default dataset puts every model at 100% on validation, so no comparison
is possible (handoff §2). Noise 1.0 is the highest the API allows and takes
both models clearly off the ceiling. 2000 rows were chosen over 800 or 1200
because they give a 200-image validation set (instead of 80) for model
selection and a 400-image test set, so one test image is 0.25 pp.

**2. Training budget** (`explore_budget.jsonl`; 2000 rows / noise 1.0; baseline, full_improved, no_scheduler)

- **lr 0.03 is unfair to the baseline:** the baseline never learns at that
  rate (val acc 23–33%, chance level) at either batch size.
- **lr 0.01 with batch 64:** one baseline seed collapsed (val acc 32.5%), so it is unstable.
- **lr 0.003:** stable but slow; the improved models are under-trained at 15 epochs.
- **lr 0.01, batch 32** trains every model stably. At 30 epochs the improved
  models reach their lowest validation loss (≈0.146 vs ≈0.21 at 15 epochs), and
  their best epoch is late (25–29), so the extra epochs are used. The baseline
  peaks earlier (best epoch 13–17) and then overfits; best-validation-epoch
  selection protects it, and its best validation loss at 30 epochs (0.292) is no
  worse than at 15 (0.322). Its best setting anywhere in the grid (0.266 at
  lr 0.003 / batch 64 / 30 epochs) is close, so the shared budget does not
  meaningfully disadvantage it.

**3. Technique strengths** (`explore_techniques.jsonl`; full_improved, lr 0.01, batch 32, 30 epochs)

| Dropout \ shift | 1 px | 2 px | 3 px |
|---|---|---|---|
| 0.2 | 0.125 | 0.132 | 0.163 |
| 0.3 | 0.126 | 0.146 | 0.171 |
| 0.5 | 0.144 | 0.180 | 0.184 |

(mean best validation loss over 2 seeds; lower is better)

A smaller shift was better at every dropout level, so the **shift is 1 px**.
Dropout 0.2 and 0.3 are within noise of each other, and 0.5 is worse. **0.3 is
kept** (the Part 1 default) instead of over-tuning on a 200-image validation set.
Cosine is kept as the scheduler: in the budget grid, cosine and no-scheduler
were close on validation, which is exactly what the `no_scheduler` ablation will
measure.

**4. Seeds:** 5 instead of 3. The validation differences between variants are
small (around 0.01–0.02 in loss), and runs take about 8 s on CPU, so 30 runs is
cheap and gives a better view of seed-to-seed noise.

## Commands

Look first (no training):
```bash
python -m experiments.run_experiments --dry-run \
  --dataset-id 2 --group official-1 --seeds 0,1,2,3,4 \
  --epochs 30 --batch-size 32 --lr 0.01 --split-seed 0 \
  --dropout 0.3 --max-shift 1 --scheduler cosine
```

Official run:
```bash
python -m experiments.run_experiments --api-url http://127.0.0.1:8000 \
  --dataset-id 2 --group official-1 --seeds 0,1,2,3,4 \
  --epochs 30 --batch-size 32 --lr 0.01 --split-seed 0 \
  --dropout 0.3 --max-shift 1 --scheduler cosine \
  --expect-fingerprint cf05f9546cbbb0d4eb1cadfe0f7854bee883ffd9899ba5b1388aab3d6d9a578d \
  --out official-1-runs.jsonl --show-test
```

### For Emma: running the same protocol on Supabase

1. Register the dataset once:
   `curl -X POST <API>/datasets -H "Content-Type: application/json" -d '{"name":"shapes-2000-noise1.0","n_rows":2000,"noise":1.0,"seed":42}'`
   and note the returned `id`.
2. If that `id` is not 2, run the dry run with `--dataset-id <id>`. The
   fingerprint then differs **only** because the dataset ID is part of the
   hashed plan; every other value must stay exactly as above. Record the new
   fingerprint here, then run with `--expect-fingerprint <new hash>`.
3. Pull results with `docs/comparison_queries.sql` (group `official-1`), then run
   `python -m experiments.analyze <folder of q1..q7 csv> <out folder>`.
