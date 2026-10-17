# Model Card — See-Sense

## Summary
See-Sense classifies small synthetic images into four shapes and explains each
prediction with a Grad-CAM heatmap. It is a teaching and evaluation demo and is
not a general photo classifier. The numbers below come from the official experiment
group `official-supabase-1` (30 runs: 6 model variants × 5 training seeds),
stored in Supabase and exported with the SQL in `docs/comparison_queries.sql`.

## Model details
Two architectures share the same layout: two blocks of
`Conv2d(3×3, padding 1) → [BatchNorm] → ReLU → MaxPool(2)` with 8 then 16
channels, followed by a linear layer on the 16×7×7 feature maps.

| Variant | What it is | Parameters |
|---|---|---|
| `baseline` | `SmallCNN`: no batch norm, no dropout, no augmentation, constant learning rate | 4,388 |
| `full_improved` | `ImprovedCNN` with batch norm, dropout 0.3, augmentation and a cosine learning-rate schedule | 4,436 |
| `no_augmentation`, `no_batchnorm`, `no_dropout`, `no_scheduler` | `full_improved` with exactly one technique removed | 4,436 (4,388 without batch norm) |

- **Augmentation:** random shifts of up to 2 pixels (zero padded) and random
  horizontal/vertical flips. Flips keep the label, but a flipped diagonal runs the
  other way, which the data generator never draws.
- **Training:** Adam, 30 epochs, batch size 32, initial learning rate 0.01,
  cross-entropy loss, five training seeds (0–4). Each run keeps the weights of its
  lowest-validation-loss epoch, and the test set is scored once afterwards.
- **Explainability:** Grad-CAM on the output of the last pooling block (16 maps of
  7×7), for the predicted class: gradients are averaged into one weight per map,
  the weighted maps are summed, passed through ReLU, scaled to [0, 1], upsampled to
  28×28 and overlaid in red. Each 7×7 cell has a 10×10-pixel receptive field.
- **Served model:** the Try It tab uses `full_improved-seed0` (run ID 4), chosen in
  advance from the saved experiment records, not by test score.

## Intended use
- **Primary:** showing how a small CNN, regularization techniques and Grad-CAM
  behave, with every number traceable to a stored run.
- **Out of scope:** real photos, medical imaging, surveillance, content moderation or
  any decision about people. A photo of a face or a cat would still be assigned one
  of four shape labels, and that label would mean nothing.

## Data
- Synthetic 28×28 grayscale images in four classes: `horizontal` bar, `vertical`
  bar, `diagonal` line and filled `block`.
- Official dataset `shapes-2000-noise1.0`: 2,000 images, Gaussian pixel noise with
  standard deviation 1.0 (values clipped to [0, 1]), data seed 42. This is the
  highest noise level the API allows; it was chosen before any test result was
  viewed because the default easy dataset put every model at 100% validation accuracy.
- Split (split seed 0): 1,400 train, 200 validation, 400 test. All variants and
  seeds use the same split, verified by a split hash.
- Only the generation parameters are stored in Supabase; pixels are regenerated
  from the seed. The Try It "generated sample" option uses much lower noise (0.1)
  than the evaluation data, so its predictions are easier than the reported scores.
- Uploaded images are converted to 28×28 grayscale before classification.

## Evaluation protocol
- Protocol frozen before the official runs (`docs/PART2_PROTOCOL.md`); the matched-
  configuration audit (SQL query 1) passed: within each seed all six runs share the
  dataset, split, budget, optimizer and learning rate.
- Metrics: test accuracy and macro-F1 on the 400-image test set (one image is
  0.25 percentage points), plus test loss, training curves and confusion matrices.
- Ablations remove one technique from `full_improved` while the others stay on, and
  compare the two on the same seed. They measure a technique's contribution
  *given the others*, not its effect on its own.

## Results (mean ± sd over 5 seeds)
| Variant | Test accuracy | Macro-F1 | Test loss |
|---|---|---|---|
| full_improved | 95.05% ± 0.67 | 0.9503 ± 0.0070 | 0.1456 |
| no_scheduler | 95.05% ± 0.96 | 0.9496 ± 0.0103 | 0.1284 |
| no_dropout | 94.70% ± 1.34 | 0.9464 ± 0.0139 | 0.1420 |
| no_augmentation | 93.25% ± 1.09 | 0.9323 ± 0.0109 | 0.1737 |
| baseline | 76.80% ± 8.93 | 0.7655 ± 0.0882 | 0.5686 |
| no_batchnorm | 67.45% ± 22.93 | 0.6614 ± 0.2413 | 0.7725 |

Paired difference from `full_improved` (same seed, 95% interval, 5 seeds):

| Removed | Mean Δ accuracy | 95% interval |
|---|---|---|
| everything (baseline) | −18.25 pp | [−28.89, −7.61] |
| augmentation | −1.80 pp | [−3.14, −0.46] |
| batch norm | −27.60 pp | [−56.01, +0.81], includes 0 |
| dropout | −0.35 pp | [−2.62, +1.92], includes 0 |
| LR scheduler | −0.00 pp | [−0.93, +0.93], includes 0 |

What the evidence supports:
- The improved model beat the baseline on every seed, and was far more stable
  (baseline accuracy ranged from 68% to 91% across seeds).
- Removing augmentation lowered accuracy on all five seeds. Removing batch norm
  lowered it on all five, but by very different amounts, so its interval is wide and
  includes 0.
- **Dropout and the learning-rate scheduler show no clear accuracy benefit** in
  these data: their intervals include 0 and the seeds disagree in sign.
- The most common errors are `block` confused with the bar classes (for example
  block → horizontal, 33 of the 99 `full_improved` errors).

## Grad-CAM evidence
`docs/SUPABASE_GRADCAM_FAILURE_ANALYSIS.md` and the Compare Models tab show 5 correct
and 8 incorrect test examples from `full_improved-seed0` (run ID 4; 376 of 400 correct,
matching the stored 94.00%). One correct example is flagged uncertain (39% confidence).
Two failures have written interpretations: example 924 (horizontal → diagonal, 95%
confidence) fits heavy noise obscuring the shape, and example 1783 (horizontal → block,
59%) fits a reliance on background patterns. These are interpretations, not proof.

## Privacy and data handling
- For every served prediction the API stores the image's **SHA-256 hash, width, height,
  predicted class, label index and confidence** in `image_metadata`. **Pixels are never
  stored**, and uploaded files are processed in memory.
- The public Recent Classifications tab reads the view `recent_classifications` with
  the **anon key**. The view exposes only `id`, `run_id`, `class_name`, `confidence`
  and `created_at` for the latest 200 rows. Image hashes are not exposed.
- `run_artifacts` (model checkpoints) and `image_metadata` are not readable by the anon
  key. All writes use the service-role key, which stays on the API server and is never
  in the UI. Run metrics are readable with the anon key.

## Limitations
- **Synthetic, narrow data:** four clean geometric shapes. Performance says nothing
  about natural images or other kinds of noise.
- **One dataset, one split, five seeds.** Intervals from five seeds are rough; the
  400-image test set limits precision; conclusions about dropout and scheduling
  cannot be separated from zero.
- **Training instability:** the baseline and the no-batch-norm variants can fail on
  some seeds (as low as 33% accuracy), so their averages hide a wide spread.
- **Grad-CAM is coarse** (7×7 cells) and class-discriminative only; it shows what is
  associated with a score and can look plausible on a wrong prediction. A confident
  prediction can still be wrong.
- Hyperparameters were tuned only on validation data from separate exploration seeds,
  but they were tuned: the same budget may favour some variants.
- Results are saved experiment records. The app does not retrain, and the public UI
  has no training controls.

## Ethical considerations
Tell non-technical users plainly that this is a demonstration on synthetic data, not a
validated vision system. Grad-CAM can build trust by showing evidence, but a plausible
heatmap is not proof of correctness. Report the spread across seeds, not only the best
run, and do not describe the dropout or scheduler effects as improvements.
