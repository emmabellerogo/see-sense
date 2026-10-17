# Dataset Card — HISTORICAL synthetic shapes (NOT the final dataset)

> **Historical document.** This describes the four-class synthetic shapes the project started with
> (the starter's dataset). It is NOT Fashion-MNIST and not the final dataset. The final dataset is documented in
> [DATASET_CARD.md](DATASET_CARD.md). The generator now lives in `shared/synthetic_data.py` and is kept only so old
> results and checkpoints stay reproducible and loadable.


Sources: generator `shared/synthetic_data.py` (moved from `shared/data.py`), splitting and augmentation in `api/training.py`,
frozen settings in `docs/PART2_PROTOCOL.md` (frozen 2026-10-15, before any official run).
Class counts below were obtained by regenerating the official dataset locally with these
functions; they are not stored anywhere in the repository.

## Summary
All images are synthetic. No real photographs, people or third-party data are used. The
data exist so that a small CNN and Grad-CAM can be evaluated on a task whose correct
answer is known exactly.

## Classes
Four classes, 28×28 grayscale, label index in this order (`CLASS_NAMES`):

| Index | Class | Clean shape (before noise) |
|---|---|---|
| 0 | `horizontal` | Full-width bar. Centre row drawn uniformly from 3–24; half-thickness `t` drawn from {1, 2}, so the bar is 2 or 4 pixels tall. |
| 1 | `vertical` | Full-height bar, same rule on columns. |
| 2 | `diagonal` | Line along the main diagonal only (top-left to bottom-right), width 3 or 5 pixels (`t` from {1, 2}, offsets −t…+t). The other diagonal direction is never generated. |
| 3 | `block` | Filled square, side drawn from 6–11, top-left corner placed uniformly so it fits in the image. |

Shape pixels are 1.0 and background 0.0. Only the block and the bars vary in position;
the diagonal varies only in thickness.

## Generation
- `generate_images(n_rows, img_size=28, noise, seed)` draws one random generator from
  `seed`. For each image it draws the class uniformly at random from the four classes,
  then the shape parameters, then the noise, in that order. The same arguments always
  give the same dataset.
- Classes are sampled independently, so they are only approximately balanced.
  Official dataset counts (2,000 images): horizontal 515, vertical 494, diagonal 479,
  block 512.
- Pixels are never stored. Supabase `datasets` holds only the generation parameters
  (`n_rows`, `img_size`, `noise`, `seed`); the pixels are regenerated from them at
  training and evaluation time.
- `generate_one` (single image, used by the Try It "generated sample" option) is a
  separate code path with its own seed per image and does not reproduce the
  `generate_images` sequence.

## Noise
- Additive Gaussian noise with standard deviation `noise` is added to every pixel of the
  [0, 1] image, the result is clipped to [0, 1], then converted to 8-bit
  (`uint8`, 0–255). Inputs to the model are divided by 255.
- The API accepts `noise` from 0.0 to 1.0.
- **Official dataset `shapes-2000-noise1.0`:** 2,000 rows, noise 1.0 (the maximum),
  data seed 42. At this level the noise is as large as the signal, so the shape is
  heavily obscured (see the Grad-CAM gallery). Noise 1.0 was chosen from training and
  validation evidence before any test result was seen: easier settings put both models
  at or near 100% validation accuracy (`docs/PART2_PROTOCOL.md`, section 1).
- Clipping makes the noise non-Gaussian and non-symmetric at the extremes; background
  pixels cannot go below 0 and shape pixels cannot go above 1.
- The Try It "generated sample" uses noise 0.1, so its predictions are easier than the
  reported test scores.

## Splits
- `make_splits(n, test_size=0.2, val_size=0.1, split_seed=0)`: one random permutation of
  the indices from `split_seed`, cut into test, validation and train. It is **not
  stratified**.
- Official sizes: **1,400 train, 200 validation, 400 test.** One fixed split for all
  30 runs. A split hash and matched-configuration audit (SQL query 1) checked that
  within each training seed all six variants shared dataset, split, budget, optimizer
  and learning rate.
- Class counts per split (regenerated locally; chance differences from the lack of
  stratification are visible):

| Class | Train | Validation | Test |
|---|---|---|---|
| horizontal | 356 | 46 | 113 |
| vertical | 363 | 40 | 91 |
| diagonal | 331 | 49 | 99 |
| block | 350 | 65 | 97 |

- The validation split selects each run's best epoch (lowest validation loss). The
  test split is scored once afterwards. Hyperparameter exploration used training seeds
  100 and 101 and validation metrics only; the exploration script strips test metrics.
- The test set is small: one image is 0.25 percentage points of accuracy.

## Augmentation (training batches only)
Applied only to the training split, and only in variants with augmentation on
(`full_improved`, `no_scheduler`, `no_dropout`, `no_batchnorm`):
- Random shift of up to **1 pixel** in each direction, zero padded, no wrap-around
  (official `--max-shift 1`; 1 px beat 2 and 3 px on validation loss in exploration).
  The code default of 2 was not used for the official runs.
- Random horizontal flip and random vertical flip, each with probability 0.5.
- Caveat: a flipped diagonal runs bottom-left to top-right, a direction the generator
  never draws. The label is kept, so augmentation teaches the model that both
  directions are `diagonal`, which is outside the test distribution.
- Validation and test images are never augmented.

## Limitations
- Synthetic, four simple shapes with a known rule; says nothing about natural images or
  other noise types.
- One dataset, one seed (42), one split. Results across five training seeds do not
  capture variation from regenerating the data or re-splitting.
- Classes are not exactly balanced and the split is not stratified.
- Diagonal images vary only in thickness, so diagonal is the least varied class, while
  blocks vary in size and position.
- Upload handling is lossy: a user image is converted to grayscale and resized to
  28×28 without cropping, so it is far from this distribution. Predictions on real
  photos are meaningless.
- At noise 1.0 some images are close to unrecognisable; some errors are expected and
  not necessarily model faults (see `docs/SUPABASE_GRADCAM_FAILURE_ANALYSIS.md`).

## Data handling
- No personal data is collected for training: the dataset is generated from a seed.
- Stored in Supabase: dataset parameters; run settings, metrics, loss histories,
  confusion matrices; model checkpoints (`run_artifacts`, not readable with the anon key).
- Per served prediction the API stores SHA-256 hash, width, height, predicted class,
  label index and confidence in `image_metadata` (not readable with the anon key).
  Uploaded files are processed in memory and their pixels are not stored. Note that width
  and height are taken after resizing (always 28×28), so they carry no information
  about the original file; the SHA-256 is of the raw uploaded bytes.
- The public UI reads only `runs` and the `recent_classifications` view
  (`id, run_id, class_name, confidence, created_at`) with the anon key.
- See `MODEL_CARD.md` (Privacy and data handling) and `docs/ETHICS_AND_DISCLOSURE.md`
  for the fuller discussion.
