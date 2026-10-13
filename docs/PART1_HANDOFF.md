# Part 1 handoff: model and experiment foundation

For Komal, who runs the official experiments in Part 2. Written by Emma's Part 1
workstream; everything below is in the working tree and **not yet committed**.

## 1. Where things stand

- **Built:** a baseline CNN and an improved CNN, reproducible training with
  separate seeds, a validation split, label-safe augmentation, an LR scheduler,
  and full per-run records (curves, test metrics, confusion matrix, parameter
  count, duration) saved through the API to Supabase.
- **Not established:** that the improved model is better. Nothing has been
  compared officially. See section 2.
- **Your job in Part 2:** pick the experimental protocol from training and
  validation evidence, freeze it, run the six matched configurations over
  several seeds, and report the results honestly.

## 2. Important: earlier 100% scores are not results

During development, one preliminary 800-row, 6-epoch, single-seed check gave
**100% test accuracy for the baseline, the improved model and the augmented
variant** alike, with identical confusion matrices. Those were plumbing checks to
see that training, saving and reading back worked. They show only that the
default task is easy enough for every model to ceiling. They do **not** show that
any technique helps, and no improvement has been established. A separate
200-row, 3-epoch plumbing run (section 6) even had the full improved model
scoring *below* the baseline for the first seed. That is equally meaningless at that
size, and a reminder that the improved model is not guaranteed to win.

Dataset difficulty was deliberately **not** changed in Part 1. If the task stays
this easy, a comparison cannot separate the variants; deciding whether and how to
make it harder (more noise, fewer rows, fewer epochs) is part of choosing the
protocol.

## 3. What exists

| Piece | Where |
|---|---|
| Baseline `SmallCNN`, `ImprovedCNN` (batch norm and dropout switchable independently), seeds, splits, augmentation, scheduler, records | [api/training.py](../api/training.py) (`run_experiment`) |
| `/train` accepts variant, switches, seeds, group and run name; `/runs` returns saved metadata | [api/main.py](../api/main.py), [shared/schemas.py](../shared/schemas.py) |
| Saving and the column mapping | [api/db.py](../api/db.py), [db/migrations/002_experiment_tracking.sql](../db/migrations/002_experiment_tracking.sql) and its [guide](../db/migrations/002_experiment_tracking.md) |
| Result validation and "same split" check | [api/experiments.py](../api/experiments.py) |
| The six matched configurations | [experiments/configs.py](../experiments/configs.py) |
| Runner that posts them to the API | [experiments/run_experiments.py](../experiments/run_experiments.py) |
| SQL for pulling results | [docs/comparison_queries.sql](comparison_queries.sql) |
| Tests | `tests/` |

**Two kinds of data:** the trained weights are in `run_artifacts.model_b64` (API
only). The numbers *about* a run (settings, seeds, curves, test metrics) are
columns on `runs`. Raw images and Grad-CAM PNGs are never stored.

## 4. The six configurations

All six use the same dataset, `split_seed`, `train_seed`, epochs, batch size,
Adam optimizer and initial learning rate. Only the declared technique changes.
`check_matched` enforces this and fails if an ablation differs in anything else.

| Variant | Model | Batch norm | Dropout | Augmentation | LR scheduler |
|---|---|---|---|---|---|
| `baseline` | SmallCNN | off | 0 | off | none |
| `full_improved` | ImprovedCNN | on | 0.3 | on (shift 2 px + flips) | cosine |
| `no_augmentation` | ImprovedCNN | on | 0.3 | **off** | cosine |
| `no_batchnorm` | ImprovedCNN | **off** | 0.3 | on | cosine |
| `no_dropout` | ImprovedCNN | on | **0** | on | cosine |
| `no_scheduler` | ImprovedCNN | on | 0.3 | on | **none** |

Run names are `<variant>-seed<N>`, e.g. `no_dropout-seed1`, so they are unique
within an experiment group. The dropout strength (0.3), shift size (2) and
scheduler (cosine) are **untuned defaults**; they are protocol choices you can
change before freezing (`--dropout`, `--max-shift`, `--scheduler`).

Things to remember when interpreting results:
- Baseline vs `full_improved` changes the architecture *and* all four techniques
  at once. The ablations are what isolate each technique.
- Flipped diagonals run in a direction the data generator never draws, so
  augmentation adds training examples beyond the generator's original diagonals.
  Shifts and flips were checked to keep each of the four classes recognizable;
  there are no 90° rotations.
- Leave-one-out measures each technique's contribution *given the others are on*.
  It does not measure its effect on its own.

## 5. Setup and commands

Use Python 3.11 (the macOS default `python` may be 2.7). From the repo root:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                      # expect all pass, 1 skipped (live Supabase)
```

**Credentials.** The API needs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in a
local `.env` (copy `.env.example`). Emma owns the Supabase project and applied
migrations 001 and 002 and seeded dataset ID 1 (800 rows, noise 0.1, seed 42).
You need either the **API URL** of a running instance Emma hosts, or the keys
through a private channel. Never commit `.env`, and never paste the service key
into chat or a doc. Part 1 does not know whether the API is deployed anywhere.

```bash
uvicorn api.main:app --port 8000     # terminal 1 (if you run the API yourself)
```

**Look before you run (no network, nothing trained).** The runner has no
defaults for protocol values on purpose:

```bash
python -m experiments.run_experiments --dry-run \
  --dataset-id 1 --group explore-1 --seeds 0 \
  --epochs 15 --batch-size 32 --lr 0.01 --split-seed 0
```

It prints the six request bodies and a **protocol fingerprint**.

**Exploration** (choosing the protocol): use a group name like `explore-*`. The
console shows only validation-side evidence (best validation loss, final training
loss, duration); test metrics are hidden unless `--show-test` is given.

**Official run** (after freezing, section 9):

```bash
python -m experiments.run_experiments \
  --api-url http://127.0.0.1:8000 \
  --dataset-id 1 --group official-1 --seeds 0,1,2 \
  --epochs <E> --batch-size <B> --lr <LR> --split-seed <S> \
  --expect-fingerprint <hash from the dry run> \
  --out official-1-runs.jsonl --show-test
```

What the runner does for each run: `POST /train`, then `GET /runs/{id}` to
confirm the saved row matches the request (variant, switches, seeds, budget,
curves of the right length, confusion matrix, split hash). Afterwards it checks
that all runs of one seed shared the same dataset and split. If a run name
already exists in the group it is **skipped, not retrained**, so a rerun cannot
create duplicates. It stops at the first failure unless `--keep-going`. Exit code
0 means every run was saved or already existed and no problem was found. Training
is synchronous and a run can take a while on CPU; `--timeout` (seconds per
request, default 900) matters on a slow or free-tier host.

## 6. Verification results

| Check | Result | By |
|---|---|---|
| Full test suite | **79 passed, 1 skipped** (the live-Supabase round-trip: no credentials) | run during Part 1 |
| Runner against mocked APIs | 20 tests: matched configs, unique names, all six variants x 2 seeds persisted through the real API code on a fake database, skip on rerun, failure handling, mismatch detection, dry run, fingerprint, health check | run during Part 1 |
| Migrations 001 + 002 on a real Postgres engine (PGlite, local) | 002 applies twice cleanly; an old row stays readable with NULLs; duplicate name, bad variant, short history, bad dropout and non-object config are rejected; RLS still on for all four tables with only the `runs` anon-read policy | run during Part 1 |
| `docs/comparison_queries.sql` | all 8 queries execute on that engine against 12 generated rows and return the expected shapes | run during Part 1 |
| Live Supabase: 3-epoch baseline through `/train` | returned 200; experiment metadata and matching checkpoint verified in Supabase | **reported by Emma** (done by her; not re-run here) |

The local engine is PGlite (Postgres compiled to WebAssembly), not Supabase
itself. Nothing in Part 1 used cloud credentials or ran official experiments.

## 7. Querying results

[docs/comparison_queries.sql](comparison_queries.sql) is read-only; paste a query
into the Supabase SQL editor and replace `REPLACE-WITH-GROUP`.

| # | Purpose | Uses test columns? |
|---|---|---|
| 1 | Matched-configuration audit per seed (all counts should be 1) | no |
| 2 | Settings, best validation loss/accuracy, final training loss | no |
| 3 | Per-epoch loss and LR curves for plotting | no |
| 4 | Per-run test accuracy, macro-F1, test loss | **yes** |
| 5 | Mean and standard deviation across seeds per variant | **yes** |
| 6 | Each variant's test difference from `full_improved` (same seed) | **yes** |
| 7 | Confusion-matrix cells, with class names | **yes** |
| 8 | List of experiment groups | no |

Test metrics are stored in the database the moment a run finishes, including
during exploration. Queries 1 to 3 are the safe ones before freezing; keep to them.

## 8. Known limitations

- **Task difficulty:** unchanged and likely too easy (section 2).
- **Test data discipline is by convention.** The database cannot hide test columns
  from you. Choosing settings by looking at test numbers would invalidate the
  results.
- **Selection rule:** each run keeps the weights from its lowest-validation-loss
  epoch; the test set is scored once afterwards. The validation set is small
  (10% of 800 = 80 images), so that choice is noisy.
- **Few seeds:** a standard deviation from 2 or 3 seeds is a rough guide only.
- **Untuned techniques:** dropout 0.3, shift 2, cosine over the full epoch count.
- **Reproducibility:** identical seeds on the same machine and library versions
  gave identical results in tests; bitwise equality across machines or torch
  versions is not guaranteed. `train_seconds` depends on the machine.
- **API:** no authentication or rate limiting; `/train` blocks until done. Two
  simultaneous requests with the same new run name could both pass the early
  check; the database's unique index would then reject the second with a 500 rather
  than a clean 409.
- **`/runs`** returns up to 50 runs including their histories.
- **Dashboard:** not updated; it does not show the new fields. The old anon query
  still works.
- **Existing runs:** the run from Emma's live verification is in the database
  under some group; use fresh group names and exclude it (query 8 lists groups).
- **Repository state:** nothing is committed. There is also an untracked
  `ai-usage/` folder that Part 1 code did not create or modify.

## 9. Part 2 tasks for Komal

1. **Get access:** an API URL or credentials from Emma (section 5). Run the test
   suite, then a `--dry-run`.
2. **Explore on validation evidence only**, in an `explore-*` group: decide the
   epoch budget, batch size, initial learning rate, dropout strength, shift size,
   scheduler, number of seeds, and whether the dataset needs to be harder (a new
   dataset is registered with `POST /datasets`). Use queries 1 to 3 and the
   runner's default console output.
3. **Freeze the protocol** before looking at any official test result. Fill in the
   table below, record the dry-run fingerprint, and commit it.
4. **Run the official experiment** (at least 3 seeds), passing `--expect-fingerprint`
   so any later change to the protocol is refused.
5. **Analyze:** training/validation loss curves (query 3), test accuracy, macro-F1
   and test loss (queries 4 and 5), confusion matrices (query 7), each
   technique's effect (query 6). Report spread across seeds and say plainly when
   a difference is within noise or when a technique did not help.
6. **Write up and present:** update `MODEL_CARD.md` and the dashboard for Part 2,
   including the limitations above.

### Frozen protocol (fill in before the official run)

| Item | Value |
|---|---|
| Dataset ID / rows / noise / data seed | |
| `split_seed`, test fraction, validation fraction | |
| Training seeds | |
| Epochs, batch size, initial learning rate (Adam) | |
| Dropout, augmentation shift, scheduler | |
| Evidence used to choose these (validation / training only) | |
| Experiment group | |
| Protocol fingerprint | |
| Frozen on (date) | |
