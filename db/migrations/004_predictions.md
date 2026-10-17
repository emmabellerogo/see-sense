# 004: predictions log, run join view, training-accuracy column (NOT YET APPLIED)

`004_predictions.sql` is additive and safe to run twice. It adds:

| Object | Purpose | Anon access |
|---|---|---|
| table `predictions` (`image_sha256`, `predicted_label`, `predicted_class`, `probabilities` jsonb, `confidence`, `served_by_run_id` → `runs.id`, `created_at`) | The assignment's prediction log, including the full probability vector | none (RLS on, no policy; it holds image hashes) |
| view `recent_predictions` (newest 200 predictions joined to `runs`; no hash) | Recent Classifications tab and drill-down to the producing run | `SELECT` only |
| column `runs.train_accuracy_history` (jsonb, nullable, length = `epochs`) | Training-accuracy curve for runs trained after this change | inherits `runs` policy |

`image_metadata` and `recent_classifications` (001, 003) are untouched. The API now writes `image_metadata`
and `predictions` together through the `log_prediction` function from **migration 005**, so apply 005 too.

## Order of operations (for Emma)

1. Supabase → SQL Editor, run as `postgres`. Paste and run `004_predictions.sql`.
2. Run `004_verify.sql` block by block; compare with the expectations in its comments.
3. Also apply `005_fashion_mnist_and_atomic_logging.sql` (see its `.md`) **before** deploying the API from the
   `correction/fashion-mnist` branch. If the API is deployed first, `/predict` answers HTTP 503 ("prediction logging is
   unavailable") rather than returning a result it could not log.
4. Make one prediction in Try It, press Refresh in Recent Classifications, and check the
   row appears with a run name.

## What it does not do

- It does not back-fill old `image_metadata` rows into `predictions` (their probability vectors were never stored).
- It does not fill `train_accuracy_history` for the 30 existing `official-supabase-1` runs. Those
  per-epoch training accuracies were never recorded; only a new training run can produce them.

## Roll back

See the commented block at the end of `004_predictions.sql`. It deletes any logged predictions.
