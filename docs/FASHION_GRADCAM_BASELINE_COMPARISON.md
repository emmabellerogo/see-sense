# Grad-CAM: baseline vs improved (DRAFT for human review)

**Status: drafted by Claude (an AI assistant) after viewing the generated figures. Neither Emma nor Komal has reviewed it; it is not their finding until they have read the figures and confirmed or rewritten it.** All numbers come from `experiments/results/fashion-official-1-gradcam-baseline/` and `experiments/results/fashion-official-1-gradcam/`.

**Models:** `baseline-seed0` (run 53, test accuracy 90.01%) and `full_improved-seed0` (run 54, 87.82%): same group, dataset 24, split, seed, class labels and preprocessing (`identity_check.json`). Both accuracies were regenerated from the stored checkpoints and equal the stored values. Nothing was retrained. One seed per model, so none of this says how the models differ in general.

## What the evidence is
- **Each model's own galleries:** 10 correct and 10 incorrect official test predictions, chosen by the same fixed rule (first correct example per class; most confident errors, one per distinct true→predicted pair first). Because the rule uses each model's own predictions, the two incorrect galleries are different sets of images, except that they overlap on some confident errors.
- **Shared images:** the 20 test images of the improved model's galleries, shown for both models with each model's own prediction. This is a fairness aid. It is **not** a random sample, and it was chosen from the improved model's errors, so it says nothing about how often the baseline errs.

## Blank maps: the largest difference, and a limit of the method
A map is blank when no cell has positive evidence for the predicted class, so the overlay shows nothing. Over all 10,000 test images:

| Predicted class | Baseline blank | Improved blank |
|---|---|---|
| Dress | 987 of 987 (100%) | 1,032 of 1,055 (97.8%) |
| Coat | 996 of 996 (100%) | 935 of 1,067 (87.6%) |
| Sneaker | 1,030 of 1,030 (100%) | 0 of 1,047 |
| Ankle boot | 958 of 958 (100%) | 0 of 1,001 |
| Pullover | 744 of 1,020 (72.9%) | 0 of 988 |
| every other class | at most 1.0% | at most 1.2% |

A blank map says only that this method, applied to this checkpoint, found no positive evidence. **It does not show that no region supports the class**, and it is not evidence that the model ignored the image: the class score may be driven by absent cues or by evidence that the positive-only map discards. For the baseline, five of its ten correct examples and five of its ten incorrect examples have blank maps, so only five of each can be read as heatmaps. For the improved model, five of the 20 shared images are blank.

## What each model highlights where both maps exist
Eight of the 20 shared images have a visible map for both models (T-shirt #19, Trouser #2, Sandal #8, Shirt #4, Bag #18, and the errors #3466, #4462, #9922).
- **T-shirt, Trouser, Shirt:** both models highlight parts of the garment. On the T-shirt both light the lower body, and the baseline also lights a patch at the upper right. On the Trouser both light the leg area. On the Shirt both light the sleeve and side areas.
- **Bag:** the baseline highlights a thin band along the bottom edge; the improved model highlights most of the bag body.
- **Sandal (#8, #9922):** both maps lie mostly on the dark background around the shoe, and the baseline's covers nearly the whole frame (garment share 0.023 and 0.038 against 0.100 and 0.067). Heat on background is a reminder that the map shows where the class score is sensitive, not where an object is.
- **Differences in where the heat falls are descriptions, not explanations.** With one seed each we cannot say they come from batch norm, dropout or augmentation, and a heatmap does not establish the model's reasoning.

## The shared errors: more often dataset-like than model-specific, cautiously
On the improved model's ten confident errors, the baseline predicted the **same wrong class** on nine (#3466, #4462, #2914, #1111, #3084, #4405, #787, #3188, #5077), and was **correct** on one (#9922, Sandal). So most of these confident errors are not peculiar to the improved model. Two readings fit and the data here cannot separate them: the images are genuinely ambiguous at 28×28 (a dataset limit), or both small networks of the same family share a weakness. This is consistent with, but does not prove, the dataset reading in `FASHION_GRADCAM_FAILURE_ANALYSIS.md` for #2914, whose baseline map is blank, and with the method limit for #787 and #3084, whose maps are blank for both models.

Three errors where only one model has a map are worth a person's look: #3188 (Shirt predicted Pullover; blank for the baseline, heat on the sleeves for the improved model), #4405 and #5077 (footwear swaps; blank for the baseline, heat above the shoe for the improved model).

## Limits
- One seed per model and 7×7 maps (each cell sees about 10×10 pixels).
- Shared images came from the improved model's galleries; the baseline's own galleries are in the baseline folder.
- Saliency maps are not proof of what a network used. We do not claim that either model's attention explains its accuracy, and the baseline scoring higher on the test set (90.01% vs 87.82% for these two runs; 89.73% vs 88.10% over three seeds) is not explained by these figures.
