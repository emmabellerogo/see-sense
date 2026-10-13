# 002 — Experiment tracking

`002_experiment_tracking.sql` adds columns to the existing `runs` table so each
training run can carry its full experiment record. It only **adds** things: no
data is deleted, rewritten or backfilled, and `001_init.sql` is unchanged.

## Two different things

| | What it is | Where it lives | Who can read it |
|---|---|---|---|
| **Model checkpoint** | The trained network's weights (base64 pickle). Needed to predict and draw Grad-CAM. | `run_artifacts.model_b64` (unchanged) | API only (service key) |
| **Experiment metadata** | Numbers *about* the run: settings, seeds, losses, test results. Needed to compare runs. | New columns on `runs` | Dashboard (anon key, read-only) |

Never stored: raw images, Grad-CAM PNGs, or weights outside `run_artifacts`.

## New fields on `runs`

Existing columns that keep their meaning: `dataset_id`, `lr` (initial rate),
`batch_size`, `epochs` (the budget), `accuracy` and `macro_f1` (final **test**
results).

| Group | Columns |
|---|---|
| Identity | `experiment_group`, `run_name` (unique together when both set) |
| Model & techniques actually used | `model_variant` (`baseline`/`improved`), `use_batchnorm`, `dropout`, `augment`, `augment_max_shift`, `scheduler` (`none`/`cosine`/`step`) |
| Seeds | `data_seed` (images), `split_seed` (train/val/test), `train_seed` (weights, batch order, augmentation, dropout) |
| Split identity | `split_sha256`, `n_train`, `n_val`, `n_test` — same hash means identical examples |
| Configuration | `optimizer`, `config` (full JSON) |
| Test results | `test_loss`, `confusion_matrix` (rows = true, columns = predicted) |
| Histories (one value per epoch) | `train_loss_history`, `val_loss_history`, `val_accuracy_history`, `lr_history` |
| Summary | `best_epoch` (0-based), `param_count`, `train_seconds`, `artifact_version` |

A baseline run should be stored with `use_batchnorm = false` and `dropout = 0`.

`artifact_version` describes the format of the checkpoint in `run_artifacts`:
NULL or 1 is the original format, 2 adds `arch`/`use_batchnorm`/`dropout`.

**Old rows:** every new column is nullable, so rows from before the migration
still read fine and simply show NULL for the new fields.

**Security:** RLS is unchanged. Anonymous users can still only `SELECT` from
`runs` (so the dashboard can chart the histories and confusion matrix); all
writes still need the service-role key. `run_artifacts` is not touched.

**Checks the database enforces:** valid variant/scheduler names, dropout in
[0, 1), non-negative counts, JSON shape, and each history having exactly
`epochs` entries.

## Applying it later

1. Open the Supabase dashboard → **SQL Editor** → **New query**.
2. Confirm `001_init.sql` was already applied (the `runs` table exists).
3. Paste the contents of `002_experiment_tracking.sql` and click **Run**.
4. Verify: **Table Editor → runs** shows the new columns, and existing rows are
   still there. Running the file a second time is harmless.

The API and tests do not need this migration yet; the API code that writes the
new columns will be added in a later step, and until then the old inserts keep
working because every new column is optional.
