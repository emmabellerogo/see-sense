# Engineering report: Komal Khan

> **DRAFT written for Komal to review and rewrite.** It was drafted from the repository record (`ai-usage/Komal-Session1.md`, `Komal-Session2.md`, git history) and the team's verified results. Anything marked [Komal to confirm] is an inference, not something Komal has said. Her personal reflection is deliberately left blank.

## My focus: experimental method and evidence
My part of the project was making the comparison fair and the evidence checkable. Per the repository record, I ran the API locally with an offline database, fixed `.env` loading, and in Part 2 wrote the exploration, analysis, SQL-check and Grad-CAM evidence scripts. I also wrote the first protocol document. [Komal to confirm scope.] The final Fashion-MNIST correction was done afterwards, in an AI-assisted session on Emma's machine, and I do not claim it. What I contributed is the method it reused.

## Method decisions
**Explore on validation only.** `experiments/explore.py` calls the same runner as the API but deletes test metrics before printing and uses seeds 100 and 101, never the official ones. That prevents choosing a setup by looking at the test set. I used it to move the original task off a ceiling, where every model scored 100% and no comparison was possible.

**Freeze, then run.** The protocol was written before the official run, and the run was started with an expected fingerprint so a drifted configuration would fail rather than silently proceed. The Fashion-MNIST protocol followed the same rule: seeds 0–2, 20 epochs, identical split per seed.

**Controlled ablations.** Beyond the baseline and the full improved model, four variants each remove one technique: batch normalization, dropout, augmentation, and the cosine schedule. Because a seed uses the same split and budget for all six, differences can be paired by seed rather than compared as unrelated averages.

**Check numbers two ways.** The comparison queries run unchanged on a local Postgres engine, and I recomputed the summary in Python independently. On the earlier synthetic experiment this gave zero mismatches. The same discipline applied to the Fashion tables: the Supabase SQL-editor export matched the earlier local run cell for cell.

## What the evidence says
Averaged over three seeds, test accuracy was: no augmentation 90.30%, baseline 89.73%, no dropout 88.94%, no scheduler 88.23%, full improved 88.10%, no batch normalization 87.12%.

The headline is a negative result. Compared with the full improved model on the same seed, the baseline scored 1.62 points higher (interval +0.26 to +2.99) and the no-augmentation variant 2.20 higher (+1.30 to +3.10). Removing dropout gave +0.83 (+0.12 to +1.55). Removing batch normalization lowered accuracy by 0.98 (−1.20 to −0.76), so it alone helped. Removing the scheduler made no clear difference (+0.12, interval includes zero).

Two cautions apply. An ablation measures a technique's contribution given the others, so "dropout hurt" means "in this combination, at this budget." And with three seeds the intervals are wide; I would not rank variants that sit within one standard deviation of each other. A five-seed design was dropped for compute reasons.

On cause, the curves point to capacity, not over-fitting: the improved model ends with train and validation accuracy both near 89%, while the baseline gets 92.39% train and 90.69% validation. That is a hypothesis, not a result. A larger model or longer training would test it, and neither was run.

## Per-class errors and Grad-CAM interpretation
Shirt is the hardest class (precision 69%, recall 60% for the served model), confused mainly with T-shirt/top, Coat and Pullover. Those classes share silhouettes, so confusion there is plausible. Trouser, Bag, Sandal and Ankle boot exceed 93%.

For Grad-CAM, the pipeline selects genuinely correct and incorrect test predictions and checks that regenerated accuracy equals the stored value, so figures come from the stored model. The draft interpretations are in `docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md`. The first, that confident footwear swaps look like ambiguous images, is the one I find more convincing. The second is a limitation: the map is empty for most Dress and Coat predictions. I would not describe it as evidence the model "looks at the wrong thing", because the method simply gives no signal there. [Komal to confirm after reading the figures yourself; the interpretations were drafted for human review and are not yet approved.]

## What I would change
- Read the assignment before choosing a dataset. The synthetic shapes came from the starter, and the mismatch with the required Fashion-MNIST was caught late. Neither of us introduced it, but we both should have checked sooner.
- Run more seeds and a larger-capacity variant, to separate "techniques hurt" from "this model is too small."
- Compute provenance from Supabase from the start, not a local replica.
- Test beyond the benchmark earlier. Supplementary probes run after the official evaluation (10 bundled test images, run 54; not benchmark results) showed that a white background pushed most images to Bag with high confidence. The app now warns before upload; the model itself is unchanged.

## Ethics and my Christian worldview
[Komal: write your own reflection. Nothing about your beliefs, tradition or Scripture reading has been assumed here. You may connect it to reporting a null result honestly, to disclosing the dataset error, or to the privacy choices (hashes instead of pixels), but only if that reflects what you think.]

## AI use
My AI sessions are recorded in `ai-usage/Komal-Session1.md` and `Komal-Session2.md`. [Komal to add what you verified yourself and what you would not have done without the assistant.]
