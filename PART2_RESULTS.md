# Part 2 results: baseline vs improved CNN on Fashion-MNIST

> **DRAFT.** Numbers are the official saved run records (group `fashion-official-1`, runs 53–70). The tables come from the project's SQL
> queries run in the Supabase SQL editor (export ingested 2026-10-17; `experiments/results/fashion-official-1-sql/PROVENANCE.md`).
> The earlier synthetic-shape experiment is kept separately in [docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md](docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md).

## Question and design
Does a small CNN improved with batch normalization, dropout, data augmentation and a cosine learning-rate schedule beat a plain CNN on
Fashion-MNIST, and which technique contributes what? The data are the full official set: 54,000 training, 6,000 validation (a seeded,
class-stratified 10% of the official training images) and the 10,000 official test images, ten classes. Six configurations were trained: the
baseline, the full improved model, and four variants each removing one technique. For every training seed (0, 1, 2) all six used the same
split, 20 epochs, batch size 128, Adam and learning rate 0.001. The protocol, seeds and a plan fingerprint were frozen before any
official run; the checkpoint of each run is the lowest-validation-loss epoch, and the test set was scored once per run. A matched-protocol
check on the saved records passed for all 18 runs.

## Results (test set, mean ± sd over 3 seeds)
| Variant | Accuracy | Macro-F1 |
|---|---|---|
| no_augmentation | 90.30% ± 0.13 | 0.902 |
| baseline | 89.73% ± 0.47 | 0.897 |
| no_dropout | 88.94% ± 0.07 | 0.889 |
| no_scheduler | 88.23% ± 0.40 | 0.881 |
| full_improved | 88.10% ± 0.29 | 0.880 |
| no_batchnorm | 87.12% ± 0.30 | 0.870 |

**The improved model did not beat the baseline.** Compared with `full_improved` on the same seed, the baseline scored 1.62 points higher
(95% interval +0.26 to +2.99), removing augmentation 2.20 higher (+1.30 to +3.10) and removing dropout 0.83 higher (+0.12 to +1.55), each
on all three seeds. Removing batch normalization lowered accuracy by 0.98 points (−1.20 to −0.76), so it was the only technique that
helped. Removing the scheduler made no clear difference (+0.12, interval includes 0). Three seeds give wide intervals, and ablations measure
a technique's contribution given the others.

A likely reason, consistent with the curves: `full_improved` ends with training accuracy 88.95% and validation accuracy 88.97%, so it is not
over-fitting, whereas the baseline reaches 92.39% training and 90.69% validation. A network of about 9,000 parameters appears
capacity-limited, so augmentation and dropout cost more than they save at 20 epochs. This was not tested with a larger model or longer
training, and the frozen protocol was not changed after the results.

## Errors, per-class results and calibration
Shirt is the hardest class in every variant (served model: precision 69%, recall 60%) and is confused mainly with T-shirt/top, Coat and
Pullover; Trouser, Bag, Sandal and Ankle boot are above 93%. Temperature scaling, fitted on validation logits only, lowered test expected
calibration error in five of six variants (`full_improved` 0.0173 → 0.0069) and left negative log-likelihood almost unchanged. Raw softmax was
already within 2 points of calibrated, so the gain is small, and confident mistakes still occur.

## Grad-CAM from the served model
The served model is `full_improved-seed0` (run 54), chosen before any result and therefore not the most accurate model (87.82%). Regenerating
its test predictions from the saved checkpoint reproduced the stored accuracy exactly. We examined 10 correct examples (one per class) and the
10 most confident errors, and drafted two cautious interpretations for review. Confident footwear swaps (sneaker ↔ ankle boot ↔ sandal) look
like ambiguous low-resolution images more than misplaced attention. The second is a limitation: Grad-CAM for the predicted class is empty for
97.8% of Dress and 87.6% of Coat predictions, so for those classes it cannot explain the decision. This is a property of the method and this
checkpoint, not evidence of reasoning, and the maps are coarse (7×7).

## System
A Streamlit interface (Concepts, Try It, Compare Models, Recent Classifications, Model Card) calls a FastAPI service that stores runs,
checkpoints and predictions in Supabase. Each prediction writes its image hash, label, probability vector and producing run in one
transaction, with no image pixels stored; if logging fails the API returns an error rather than a result. Public history reads a view
without hashes. Try It shows the top five probabilities and Grad-CAM, and loads the served checkpoint only after checking its group, dataset,
labels and preprocessing. Automated tests passed (mocks and a local Postgres engine); live Supabase checks passed on a small test run.

## Deviations and unresolved items
- SQL provenance: the tables come from the Supabase SQL-editor export; an earlier local-engine dry run of the same queries matched it cell for cell.
- Three seeds, not five, because of compute; intervals are rough. All runs came from one machine.
- The dataset was corrected from the starter's synthetic shapes to Fashion-MNIST late; the earlier results are preserved and labelled.
- No instructor approval exists for any of this; a question is drafted but unsent.
- Contribution statements, videos and the worldview reflection are the authors' own and are not written here.

*Contributions are recorded in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md); the Fashion-MNIST correction was an AI-assisted session on Emma's machine and is not attributed to Komal.*
