# Dataset Card — Fashion-MNIST in See-Sense

> Describes the dataset and how this project uses it. Class counts and hashes below were measured from the
> downloaded official files on 2026-10-17. The 18 official runs (group `fashion-official-1`, Supabase dataset id 24) recorded this
> same dataset identity (file SHA-256 hashes match the files measured here). Model results are in [MODEL_CARD.md](../MODEL_CARD.md), not here.
> (The earlier synthetic-shape dataset is documented in [DATASET_CARD_HISTORICAL_SYNTHETIC.md](DATASET_CARD_HISTORICAL_SYNTHETIC.md);
> it is not the final dataset.)

## Why Fashion-MNIST
The assignment requires CIFAR-10 or Fashion-MNIST. The project's CNN takes 28×28 grayscale input, which Fashion-MNIST provides
natively (CIFAR-10 would need 32×32 colour and a different network). The project originally used the synthetic shape generator
that came with the starter; it was replaced by Fashion-MNIST to meet the assignment, and the earlier experiment is kept only as
labelled historical evidence (see `docs/PROJECT_NOTES.md`).

## Source
- Fashion-MNIST, Zalando Research — https://github.com/zalandoresearch/fashion-mnist (MIT licence). Xiao, Rasul, Vollgraf, 2017.
- Downloaded **only** by `python tools/fetch_fashion_mnist.py` (local training machine), which verifies each file's published MD5.
  The API (Render) and the UI (Streamlit) never download the dataset; `shared/fashion.load` only reads local files and
  `tests/test_fashion_data.py` asserts it never touches the network.
- Files and their SHA-256 (recorded in every run's dataset identity):

| File | SHA-256 |
|---|---|
| `train-images-idx3-ubyte.gz` | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` |
| `train-labels-idx1-ubyte.gz` | `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` |
| `t10k-images-idx3-ubyte.gz` | `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` |
| `t10k-labels-idx1-ubyte.gz` | `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` |

## Content
28×28 grayscale pictures of single clothing items (one item per image, centred, light item on a dark background), `uint8` 0–255.
Ten classes, label index in this order (`shared.fashion.CLASS_NAMES`, stored with every checkpoint):

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| T-shirt/top | Trouser | Pullover | Dress | Coat | Sandal | Shirt | Sneaker | Bag | Ankle boot |

Official partitions (measured): **60,000 training images (6,000 per class)** and **10,000 test images (1,000 per class)**.
All ten classes are used; none are dropped or merged.

## Partitions used by this project
| Partition | Images | Source | Used for |
|---|---|---|---|
| Train | 54,000 | official training images minus validation | fitting weights |
| Validation | 6,000 (600 per class) | official training images, **class-stratified**, seeded | choosing the epoch (lowest validation loss), fitting the calibration temperature, pilot/protocol decisions |
| Test | 10,000 | the **official test partition**, unchanged | scored once per run; never used to train, select, calibrate or choose the served model |

- The validation split depends only on `(training labels, number of validation images, split_seed)` (`shared.fashion.split_train_val`)
  and is identical for every variant and training seed that shares a `split_seed`. The index sets are identified by SHA-256 hashes
  stored in each run's dataset identity (`train_index_sha256`, `val_index_sha256`) and by `runs.split_sha256`.
- **Subsets:** the plan uses the **full** dataset. A subset (`train_limit`) is supported only for pilots/sizing and, if ever used for an official
  run, must be justified in `docs/FASHION_PROTOCOL.md`; it is recorded in the run's config and verified by the experiment runner.
- **Official protocol:** split seed 0, validation fraction 0.1, full data (`train_limit` null); all 18 runs share split hash
  `786ed8107e43ef83d70d4905a8813c26927b352362675804427894557914b7b9` (checked from the saved records), so every variant and seed
  trained and was validated on the same images. Frozen in `config/fashion_protocol.json`.

## Preprocessing and normalization
1. Pixels are scaled to [0, 1] (÷255).
2. Standardized with the mean and standard deviation of the 60,000 official **training** pixels (labels and test data not involved):
   **mean 0.2860, std 0.3530** (measured 0.2860 / 0.3530; `tests/test_fashion_data.py` re-checks them when the files are present).
3. Augmentation (training batches only, applied in [0, 1] space before normalization): random shift up to ±2 px with black padding, and a
   random horizontal flip. **No vertical flips**, rotations or crops. Validation and test images are never augmented.
4. Serving (uploads): convert to grayscale, resize to 28×28 (PIL bicubic), ÷255, same mean/std. Uploads are **not** inverted,
   background-removed or otherwise adjusted, so photos far from the Fashion-MNIST look (dark item on white) are out of distribution.
   The preprocessing is stored in the checkpoint and the same function chain is used in training, evaluation and serving.

## Held-out examples in the app
`tools/export_samples.py` exports the first two official-**test** images of each class (in file order, 20 PNGs, `ui/assets/samples/`) with a manifest
of test indices and hashes. The rule does not look at any model output. The Try It tab sends them through the normal upload path.

## Known limitations of the data
- Low resolution, centred, no background or lighting variation, one item per image; the benchmark is well known to be easy relative to real photographs.
- Some categories are visually close (T-shirt/top, Shirt, Pullover, Coat) and a few labels are arguably ambiguous; this affects the ceiling of any model.
- Western-market product images; no people. The dataset contains no personal data.

## What is stored where
Supabase `datasets` holds only the dataset's identity (name, `source = fashion_mnist`, class names, file hashes); never pixels. Uploaded images are never stored.
