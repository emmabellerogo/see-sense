# Model Card — See-Sense (Fashion-MNIST)

> **Status:** Official Fashion-MNIST results (group `fashion-official-1`, 18 runs, ids 53-70) are reported below.
> They are real saved run records, and the tables were produced by running the project's SQL queries in the **Supabase SQL editor**
> (export ingested 2026-10-17; see *Results*). The earlier four-class *synthetic-shape* experiment
> is documented separately under **Historical** at the end; it is not Fashion-MNIST and must not be read as Fashion-MNIST performance.

## Summary
See-Sense classifies 28×28 grayscale pictures of clothing into the ten Fashion-MNIST categories and explains
each prediction with a Grad-CAM heatmap and its five most probable classes. It is a teaching and evaluation
demo and is not a general photo classifier.

## Model details
Two architectures share the same layout: two blocks of
`Conv2d(3×3, padding 1) → [BatchNorm] → ReLU → MaxPool(2)` with 8 then 16
channels, followed by a linear layer from the 16×7×7 feature maps to **10 class scores**.

| Variant | What it is | Parameters |
|---|---|---|
| `baseline` | `SmallCNN`: no batch norm, no dropout, no augmentation, constant learning rate | 9,098 |
| `full_improved` | `ImprovedCNN` with batch norm, dropout 0.3, augmentation and a cosine learning-rate schedule | 9,146 |
| `no_augmentation`, `no_batchnorm`, `no_dropout`, `no_scheduler` | `full_improved` with exactly one technique removed | 9,146 (9,098 without batch norm) |

- **Augmentation (training batches only):** random shifts of up to 2 pixels (zero = black background, no wrap-around)
  and a random **horizontal** flip. There is **no vertical flip**: an upside-down dress or boot is not a realistic
  garment, so flipping vertically would teach the model something false. No rotations or crops are used.
- **Training:** Adam, cross-entropy loss. Epochs, batch size, learning rate and seeds are frozen in
  `config/fashion_protocol.json` before the official runs (see *Evaluation protocol*). Each run keeps the
  weights of its lowest-**validation**-loss epoch; the official test set is scored once afterwards.
- **Per-epoch records:** training loss and accuracy, validation loss and accuracy, and learning rate, for every epoch of
  every run (training accuracy is the un-augmented training split scored in eval mode).
- **Calibration:** softmax probabilities are not automatically calibrated. Each checkpoint stores a single
  **temperature** fitted by *temperature scaling* on the validation logits only (`softmax(logits / T)`); the
  Top-5 probabilities and the confidence in the app are the temperature-scaled ones. Calibration is measured with ECE
  (15 equal-width bins), negative log-likelihood and the Brier score, before and after, on validation and on the held-out
  test set (`runs.calibration`). **A fitted temperature is not evidence that the model is well calibrated;**
  the app and this card claim calibration only to the extent the stored test measures show it. *Result: see Results below (ECE fell for five of six variants; NLL barely moved).*
- **Explainability:** Grad-CAM on the output of the last pooling block (16 maps of 7×7) for the predicted class,
  computed from the **raw class logit** (so the temperature never changes the heatmap): gradients are averaged into one
  weight per map, the weighted maps are summed, passed through ReLU, scaled to [0, 1], upsampled to 28×28 and overlaid
  in red. Each 7×7 cell has a 10×10-pixel receptive field.
- **Checkpoint contents (format 3):** weights, ordered class names, preprocessing (grayscale, 28×28, ÷255, mean 0.2860,
  std 0.3530), architecture options, dataset identity (source, file SHA-256 hashes, split seed and index hashes),
  temperature, selected epoch and artifact version. Earlier four-class checkpoints (format 1–2) still load and are
  labelled `synthetic_shapes`.
- **Served model:** `full_improved-seed0` = **run 54**, chosen in advance in `config/served_model.json` (fixed on 2026-10-17
  before any official run), not picked by test score. Verified: group `fashion-official-1`, dataset 24, ten ordered class
  names, mean 0.286 / std 0.353, regenerated test accuracy 87.82% = the stored value. It is **not the best model in the
  comparison** (see *Results*); it is the pre-registered one.

## Intended use
- **Primary:** showing how a small CNN, regularization techniques, calibration and Grad-CAM behave on a standard
  benchmark, with every number traceable to a stored run.
- **Out of scope:** real photos of clothing from shops or phones, fashion recommendation, any decision about
  people, medical or safety use. Uploaded pictures are converted to 28×28 grayscale but are **not** inverted or
  background-removed: Fashion-MNIST items are light on a dark background, and a dark garment on a white
  background is far outside the training distribution. A picture of a cat or a face would still be given one of the
  ten clothing labels, and that label would mean nothing.

## Data
- **Fashion-MNIST** (Zalando Research; MIT licence): 70,000 grayscale 28×28 images, ten classes, official partitions of
  60,000 training and 10,000 test images (1,000 per class in test). Details: [docs/DATASET_CARD.md](docs/DATASET_CARD.md).
- **Validation** images are carved out of the *official training* images (10%, 6,000, class-stratified, split seed in the
  protocol). The official test images are never used to train, to choose an epoch, to fit the temperature, or to choose
  the served model.
- Only dataset identity and file hashes are stored in Supabase; pixels are not.

## Evaluation protocol
- Frozen in `config/fashion_protocol.json` and `docs/FASHION_PROTOCOL.md` before the official runs. Within each seed the six
  variants share the dataset, validation split, seed, epoch budget, batch size, optimizer and initial learning rate; only the
  declared technique differs (checked by `experiments/configs.py` and SQL query 1).
- Metrics on the 10,000 official test images: accuracy, macro-F1, loss, the 10×10 confusion matrix and per-class precision and
  recall, plus calibration (ECE / NLL / Brier).
- Ablations remove one technique from `full_improved` while the others stay on and compare the two on the same seed. They
  measure a technique's contribution *given the others*, not its effect on its own.
- Nothing was tuned after looking at test results; the budget came from a validation-only pilot
  (`experiments/results/fashion-pilot/`).

## Results
Protocol frozen before any run: 3 training seeds (0, 1, 2) × 6 variants = 18 runs, 20 epochs, batch 128, Adam, learning rate 0.001,
54,000 train / 6,000 validation / 10,000 official test images, matched splits (checked: one split hash, identical settings except the
declared technique). Source of the numbers: SQL queries 1-7, 9 and 10 of `docs/comparison_queries.sql` over the 18 saved run records.
**Provenance:** the CSVs in `experiments/results/fashion-official-1-sql/` come from the Supabase SQL-editor export of
`docs/fashion_official_1_export.sql` (PostgreSQL 17.11, executed 2026-10-17T22:49:57Z, query-file hash recorded in `PROVENANCE.md`). They matched an earlier
local-engine dry run of the same queries cell for cell.

| Variant | Test accuracy (mean ± sd, 3 seeds) | Macro-F1 | Test loss |
|---|---|---|---|
| no_augmentation | 90.30% ± 0.13 | 0.9024 | 0.2720 |
| baseline | 89.73% ± 0.47 | 0.8973 | 0.2884 |
| no_dropout | 88.94% ± 0.07 | 0.8888 | 0.3093 |
| no_scheduler | 88.23% ± 0.40 | 0.8810 | 0.3274 |
| **full_improved** | **88.10% ± 0.29** | 0.8801 | 0.3327 |
| no_batchnorm | 87.12% ± 0.30 | 0.8701 | 0.3591 |

**The improved configuration did not beat the baseline.** Paired by seed against `full_improved` (95% interval of the mean difference,
3 seeds, so wide):

| Removed from full_improved | Mean Δ accuracy | 95% interval | Reading |
|---|---|---|---|
| everything (the baseline) | +1.62 pp | [+0.26, +2.99] | baseline was higher on all 3 seeds |
| augmentation | +2.20 pp | [+1.30, +3.10] | removing it **helped** on all 3 seeds |
| dropout | +0.83 pp | [+0.12, +1.55] | removing it helped on all 3 seeds |
| batch norm | −0.98 pp | [−1.20, −0.76] | removing it hurt on all 3 seeds: **the only technique that helped** |
| LR scheduler | +0.12 pp | [−0.48, +0.72] | no clear difference |

- **Why (evidence, not proof):** at 20 epochs the regularized variants are not over-fitting: for `full_improved` training accuracy
  (88.95%) equals validation accuracy (88.97%), while the baseline fits more (92.39% vs 90.69%). With about 9,000 parameters the model
  appears capacity-limited, so augmentation and dropout cost more than they save. This is consistent with the validation-only pilot, and
  the protocol was deliberately not changed afterwards. More epochs, a larger model or milder regularization might change the ranking;
  that was not tested.
- **Served model (run 54):** test accuracy 87.82%, macro-F1 0.8771, test loss 0.3393. It is the lowest of the three `full_improved` seeds
  (88.10% mean). Per-class F1: Shirt 0.64 (weakest), T-shirt/top 0.82, Coat 0.81, Pullover 0.83, Dress 0.87, Sneaker 0.94, Ankle boot 0.95,
  Sandal 0.96, Bag 0.98, Trouser 0.98. Per-class tables for every variant: `fashion-official-1-analysis/per_class_metrics.csv`.
- **Most common errors (all variants):** Shirt ↔ T-shirt/top, Shirt → Coat / Pullover, Pullover ↔ Coat, and footwear swaps (Ankle boot →
  Sneaker, Sandal → Sneaker).
- **Calibration (measured, held-out test set, mean over 3 seeds):** the temperature fitted on validation data lowered test ECE for five of
  six variants (for `full_improved` 0.0173 → 0.0069); for `no_dropout` it did not (0.0073 → 0.0082). Test NLL barely changed (0.3327 → 0.3325
  for `full_improved`). For run 54: T = 0.871 (below 1, i.e. the raw softmax was slightly *under*-confident), ECE 0.0168 → 0.0091, NLL
  0.3393 → 0.3400. All variants were already within about 2 percentage points ECE before scaling, so the gain is small. Individual
  confident mistakes still occur (next section).

## Grad-CAM evidence
Generated from the saved run-54 checkpoint (`experiments/results/fashion-official-1-gradcam/`): regenerated test accuracy 87.82%, identical
to the stored value. Fixed selection rule: one correct example per class (10) and the ten most confident errors (one per distinct true →
predicted pair first), so the error set is **not a typical sample of errors**. Two interpretations (drafted by an AI assistant; **not yet confirmed by a person**) are in
`docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md`: footwear confusions at over 99% confidence look like ambiguous 28×28 images rather than the model
looking at the wrong place; and **Grad-CAM is empty for 97.8% of Dress and 87.6% of Coat predictions** (and for at most 1.2% of any other
class), so for those classes the heat map cannot explain the decision. This is a property of the method and this checkpoint, found by
looking at the results; the method was not changed.

**Baseline (run 53) and side-by-side.** The same method was applied to the saved baseline checkpoint
(`experiments/results/fashion-official-1-gradcam-baseline/`): same group, dataset, split, labels and preprocessing (`identity_check.json`),
regenerated test accuracy 90.01%, identical to the stored value; 10 correct and 10 incorrect examples by the same rule, and the improved
model's 20 gallery images shown for both models. The baseline's predicted-class map is blank for **100% of Dress, Coat, Sneaker and
Ankle boot predictions and 72.9% of Pullover**, so only some of its examples can be read as heat maps (5 of its 10 correct and 5 of its 10
incorrect). On the improved model's ten confident errors the baseline predicted the same wrong class on nine and was right on one (#9922).
A blank map means this method found no positive evidence; it does not mean no region supports the class. One seed per model; saliency
shows where a score is sensitive, not why the model decides, and these figures do not explain why the baseline scored higher. The
written comparison (`docs/FASHION_GRADCAM_BASELINE_COMPARISON.md`) is an AI draft that has **not** been confirmed by a person.

## Privacy and data handling
- For every served prediction the API stores, in one transaction, a row in `image_metadata` (SHA-256 hash, width, height, label,
  class, confidence) and a row in `predictions` (SHA-256 hash, predicted label and class, the **full probability vector**,
  `served_by_run_id`, timestamp). **Pixels are never stored**, and uploaded files are processed in memory. If the
  database cannot log the prediction, the API returns an error instead of a result.
- The public Recent Classifications tab reads the view `recent_predictions` with the **anon key**: class, confidence,
  probabilities and the producing run, for the latest 200 rows. **Image hashes are not exposed**, and neither are model
  checkpoints. The older `recent_classifications` view remains as a fallback.
- `run_artifacts` (model checkpoints), `image_metadata` and `predictions` are not readable by the anon key. All writes use the
  service-role key, which stays on the API server and is never in the UI. Run metrics are readable with the anon key.
- The public `GET /predictions` endpoint likewise omits hashes.

## Limitations
- **Small, standard benchmark:** Fashion-MNIST images are centred, low-resolution and uniform. Accuracy here says little about real clothing photos.
- **Light-background inputs are pushed towards Bag (distribution shift).** The model was trained only on light items on dark backgrounds and the
  app does not invert, crop or reject uploads. In a supplementary robustness check run **after** the official evaluation (not an official benchmark
  result, not part of the 18-run comparison): the 10 bundled held-out test images, served model run 54, direct inference with nothing logged. With
  the background turned white and the item pixels unchanged, 8 of the 9 non-bag images were predicted Bag (confidence 0.63-1.00; only the pullover
  survived). With a dark item on white, 4 of the 9 became Bag at 0.97-1.00 (sneaker, ankle boot, sandal, shirt), 3 became other wrong classes (dress
  → trouser, T-shirt → pullover, coat → shirt) and 2 stayed correct (trouser, pullover). A wrong answer can carry very high confidence, and
  the temperature scaling does not fix that. Ordinary photos usually have light backgrounds, so expect this. These are small probes (10 images,
  one model, one construction of the background), not a measured failure rate; the probe script is not part of the saved evidence. The Try It tab
  warns before upload and recommends the bundled examples.
- **Confusable classes:** Shirt is the weakest class (F1 0.64 for the served model) and is mostly confused with T-shirt/top, Coat and
  Pullover; these four upper-body classes account for most errors in every variant.
- **Small model:** about 9,000 parameters, chosen for transparency and cheap training, not for the best possible accuracy.
- **Three seeds:** limited by this machine's compute (see `docs/FASHION_PROTOCOL.md`); the paired 95% intervals use t = 4.30 and are rough. All runs came from one machine.
- **Grad-CAM is blank for most Dress and Coat predictions** of the served model (see above), so a missing overlay in the app is expected for those classes.
- **The served model is not the most accurate configuration** (the no-augmentation and baseline variants scored higher); it was fixed in advance, as the protocol requires.
- **Grad-CAM is coarse** (7×7 cells) and class-discriminative only; it shows what is associated with a score and can look
  plausible on a wrong prediction. A confident prediction can still be wrong.
- **Calibration** is fitted on one validation split and measured on one test set; it may not hold for uploads from outside the benchmark distribution.
- Results are saved experiment records. The app does not retrain, and the public UI has no training controls.

## Ethical considerations
Tell non-technical users plainly that this is a demonstration on a benchmark dataset, not a validated vision system. Grad-CAM
can build trust by showing evidence, but a plausible heatmap is not proof of correctness. Report the spread across seeds, not
only the best run. See [docs/ETHICS_AND_DISCLOSURE.md](docs/ETHICS_AND_DISCLOSURE.md).

---

# Historical: the earlier four-class synthetic-shape experiment (NOT Fashion-MNIST)

> Everything below describes the **first version of the project**, built on the synthetic four-shape dataset that the
> starter shipped with (horizontal bar, vertical bar, diagonal line, filled block; 2,000 generated images, noise 1.0, group
> `official-supabase-1`, 30 runs, 4,388 / 4,436 parameters, trained 30 epochs at batch 32, lr 0.01, with 1-pixel shifts and
> horizontal+vertical flips). The assignment asks for CIFAR-10 or Fashion-MNIST, so this is **not** the final evidence. It is kept
> for transparency. Those runs have no training-accuracy history, no calibration and no per-class table. The deployed app at
> the time served `full_improved-seed0` (run ID 4) from this experiment.

## Historical results (synthetic shapes; mean ± sd over 5 seeds)
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

## Historical Grad-CAM evidence (synthetic shapes)
`docs/SUPABASE_GRADCAM_FAILURE_ANALYSIS.md` and the Compare Models tab (historical section) show 5 correct and 8 incorrect
test examples from `full_improved-seed0` of the synthetic experiment (run ID 4; 376 of 400 correct, matching the stored 94.00%).
Two failures have written interpretations: example 924 (horizontal → diagonal, 95% confidence) fits heavy noise obscuring the
shape, and example 1783 (horizontal → block, 59%) fits a reliance on background patterns. These are interpretations, not proof,
and say nothing about clothing.

## Historical limitations (synthetic shapes)
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

