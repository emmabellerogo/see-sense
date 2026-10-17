# Grad-CAM failure analysis: Fashion-MNIST (DRAFT for human review)

**Status: DRAFT written by Claude (an AI assistant) after viewing the generated figures, for Emma and Komal to read, check
against the figures themselves, and rewrite in their own words before submission.** It is not a finding until a person has
confirmed it. Every number comes from files in `experiments/results/fashion-official-1-gradcam/` and
`experiments/results/fashion-official-1-sql/`.

**Model:** `full_improved-seed0` (run 54, the model chosen in advance as the served model). Regenerated test accuracy 87.82%,
identical to the stored value. Figures: `gradcam_correct.png`, `gradcam_incorrect.png`; table: `examples.csv`.

**Selection rule, fixed before the figures were generated:** correct = first correctly classified test image of each class;
incorrect = wrong predictions in descending confidence, one per distinct (true → predicted) pair first. This means the incorrect
set is the model's **most confident mistakes**, not a random or typical sample of its errors. The most common error in the
data, Shirt confused with T-shirt/top, Coat or Pullover, appears only once in the ten (test #3188).

## Example 2914: Sneaker predicted as Ankle boot (99.9% confident)
**What the evidence shows.** The image is a tall sneaker whose upper rises towards the ankle, with a bulky sole. The Grad-CAM map for
the *predicted* class (Ankle boot) is spread over the shoe's upper outline and the area above it (`cam_mass_on_garment` 0.088, so
most of the highlighted mass lies on dark background pixels, not on the shoe); the map for the *true* class (Sneaker) is a small
blob in the empty upper-left. This is one of four of the ten most confident errors that are footwear-for-footwear swaps (also #4405
Sandal→Sneaker, #9922 Sandal→Ankle boot, #5077 Ankle boot→Sneaker). The same direction shows up in aggregate: summed over the three
`full_improved` seeds, 127 Ankle boots were called Sneakers, 84 Sneakers were called Ankle boots, and 133 Sandals were called Sneakers
(confusion matrices, query 7).

**Cautious reading.** This looks more like a **dataset problem** than a model problem: at 28×28 pixels a high-top sneaker and a low
ankle boot differ by a few rows of pixels, and a person looking at #2914 or at the closed-looking "sandal" #4405 could reasonably
hesitate. The model is, however, *confidently* wrong (more than 99%), so the confidence is not a signal that an item is ambiguous.
The aggregate calibration is good (test ECE 0.0168 → 0.0091 for this run after temperature scaling), but average calibration says
nothing about these individual cases.

**What would change this conclusion.** If a larger or better-regularized model, or a person asked to relabel these images blind,
resolved most of these cases, the problem would be the model's capacity or training, not the labels.

## Example 787: Trouser predicted as Dress (99.3% confident)
**What the evidence shows.** The image is one solid white vertical shape with no visible gap between the legs, which is the main cue that
separates a trouser from a dress. The Grad-CAM map for the *predicted* class (Dress) is **completely empty**
(`cam_mass_on_garment` 0.000); only the map for the true class (Trouser) shows any highlight (the legs and lower sides). The same
empty map appears for two other confident errors predicted as Dress (#1111 T-shirt/top→Dress, #3084 Coat→Dress) and for correct Dress
and Coat predictions (#13, #6). Checking this over all 10,000 test images (`blank_cam_by_class.csv`), the predicted-class map is empty
for **97.8% of Dress** (1,032 of 1,055) and **87.6% of Coat** (935 of 1,067) predictions, against at most 1.2% for every other class.

**Cautious reading.** Two separate things are visible. (1) For this image the input really is ambiguous (#4462, a Dress predicted as
Trouser, is the mirror case: two parallel vertical bars). (2) More important, **Grad-CAM cannot explain a Dress or Coat prediction of this
model**: the map keeps only locations that push the class logit up, and for these classes no location does. A plausible but untested
explanation is that the Dress/Coat logits are driven by what is *absent* (for example the lack of a gap or of sleeves), which a
positive-evidence map cannot show. I did not test this, and I did not change the Grad-CAM method after seeing results; the frozen
protocol uses the raw class logit. The practical consequence is that a blank overlay in the app is a property of the method and
model, not a failed upload.

**What would change this conclusion.** A different attribution method (for example one that also shows negative evidence), or the same
analysis on the other seeds' checkpoints. If other seeds show blank maps for different classes, this is an artifact of one training run.

## Interpretation limits
- The Grad-CAM resolution is 7×7 (each cell looks at a roughly 10×10-pixel patch of a 28×28 image), so it cannot localize fine cues.
- A heat map shows where the classification score was sensitive, not why the network decided; it is not proof of reasoning.
- Ten errors chosen by confidence are anecdotes. Neither example says how often this kind of mistake occurs; the confusion matrices
  and per-class tables do.
- Whether a particular image is mislabelled or ambiguous is my reading of 28×28 pictures; it needs a person's judgement.
- Only one checkpoint (seed 0, the lowest-scoring `full_improved` seed at 87.82% vs 88.10% and 88.39%) was examined.
- These interpretations apply to Fashion-MNIST only. The interpretations in `SUPABASE_GRADCAM_FAILURE_ANALYSIS.md` describe the earlier synthetic shapes and must not be reused.
