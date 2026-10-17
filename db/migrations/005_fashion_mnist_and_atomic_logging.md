# 005: Fashion-MNIST provenance and atomic prediction logging (NOT YET APPLIED)

Additive and safe to run twice. Requires 001–004 to be applied first.

| Object | Purpose |
|---|---|
| `datasets.source` (default `'synthetic_shapes'`), `class_names`, `identity` | Label every dataset by origin. All existing rows become `synthetic_shapes`, which is what they are; nothing is relabelled as Fashion-MNIST. |
| `runs.calibration`, `runs.test_per_class` | Measured temperature-scaling results and per-class test precision/recall for new runs. NULL on old runs. |
| function `log_prediction(...)` | Writes `image_metadata` **and** `predictions` in one transaction. Executable by `service_role` only. |

## Why the function

Two separate inserts can leave a prediction in `image_metadata` but not in `predictions` (or the
reverse) if the second one fails. The API now calls this function, and answers HTTP 503 (with no
Grad-CAM result) if logging is unavailable. It never reports `logged: true` unless both rows exist.

## Order of operations (for Emma, when ready; nothing was applied by Claude)

1. Supabase → SQL Editor, as `postgres`: run `004_predictions.sql`, then `004_verify.sql`.
2. Run `005_fashion_mnist_and_atomic_logging.sql`, then `005_verify.sql` block by block.
3. **Apply both before deploying this branch's API.** The new `/predict` needs `log_prediction`; without it, it answers 503 rather than pretending to log.
4. Register the dataset from the training machine: start the API locally against Supabase and `POST /datasets` (see README).

## Roll back

See the commented block at the end of the `.sql`. It does not delete prediction rows.
