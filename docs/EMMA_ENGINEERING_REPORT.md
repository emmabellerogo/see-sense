# Engineering report: Emma Rogoveanu

> **DRAFT.** Team numbers are the official saved records (group `fashion-official-1`, runs 53–70); the SQL tables come from the Supabase SQL-editor export.

## What I owned
I built the foundation and the serving side of See-Sense: the baseline and improved CNNs, reproducible training, the experiment runner, the Supabase schema and migrations, Grad-CAM loading from stored checkpoints, the five-tab dashboard, and the prediction log. When we discovered that the project used the starter's synthetic shapes instead of Fashion-MNIST, I carried out the correction in an AI-assisted session on my machine, and I record that here so it is not attributed to Komal.

## Design decisions
**Reproducibility first.** Each run records its seed, split hash, hyperparameters and a plan fingerprint, so a matched comparison can be audited from the database. The runner refuses to start if the fingerprint differs from the frozen one. The official Fashion-MNIST protocol was frozen before any official run: 54,000 training images, a seeded stratified 6,000-image validation set, the 10,000 official test images, 20 epochs, batch 128, Adam at 0.001, seeds 0–2. The checkpoint is the lowest-validation-loss epoch, and the test set is scored once per run.

**Atomic prediction logging.** Each prediction writes its image hash, label, probability vector and producing run in one database function. If logging fails, the API returns an error instead of a result, so the interface never shows a prediction that was not recorded. Raw pixels are never stored. Anonymous visitors read recent predictions through a view that omits hashes; the base tables deny anonymous access, and a live test confirms that.

**Honest serving.** The served model is `full_improved-seed0` (run 54), chosen before results were known. It is not the most accurate model (87.82%). Try It loads its checkpoint only after checking group, dataset, labels and preprocessing, so a mismatched checkpoint cannot be served silently.

## Results I rely on
Over three seeds the baseline reached 89.73% ± 0.47 test accuracy and the full improved model 88.10% ± 0.29. The improved model did not beat the baseline: paired by seed, the baseline was 1.62 points higher (95% interval +0.26 to +2.99). Removing augmentation gave the best variant (90.30%). Removing batch normalization cost 0.98 points, so batch normalization was the only technique that helped.

My reading is that the network, about 9,000 parameters, is capacity-limited. `full_improved` ends with training and validation accuracy both near 89%, so it is not over-fitting, while the baseline reaches 92.39% training and 90.69% validation. Augmentation and dropout therefore appear to cost more than they save in 20 epochs. I did not test a larger model or longer training.

Calibration was a smaller finding. Temperature scaling, fitted on validation logits only, reduced test expected calibration error in five of six variants (`full_improved` 0.0173 → 0.0069) and left log-likelihood almost unchanged. 

## Grad-CAM and its limits
From run 54 I generated 10 correct and 10 confident-error examples. Regenerating the test predictions from the checkpoint reproduced the stored accuracy exactly,. Confident sneaker, sandal and ankle-boot swaps look like ambiguous 28×28 images more than misplaced attention. The more important limit is that the map for the predicted class is empty for 97.8% of Dress and 87.6% of Coat predictions. For those classes Grad-CAM cannot explain the decision, and the dashboard says so. The maps are also only 7×7, so they show where, not why.

## Testing
Mocked and in-memory tests cover code paths, and a local Postgres engine covers migrations and the atomic logging function; neither proves anything about Supabase. A separate opt-in live test passed against Supabase on 2026-10-17 (run linkage, atomic logging, the hash-free view, denied anonymous table access).

## Limits and what I would do differently
- The SQL tables come from the Supabase SQL-editor export, which matched an earlier local dry run of the same queries cell for cell.
- Three seeds give wide intervals.
- The dataset error was caught late. The first deployed version used synthetic shapes and the deployed apps are not yet updated. Next time I would read the full assignment before building anything, not only the rubric and the starter.
- No instructor approval exists for the correction; a question is drafted and unsent.

## Worldview reflection
I am a Christian. A classifier pointed at people is an exercise of power over image-bearers: "So God created man in his own image" (Genesis 1:27, ESV). A label on a person is a claim about someone who carries that dignity, made without their consent and hard for them to contest.

**The use I would refuse to build:** retraining this pipeline to classify photos of people, such as inferring identity, emotion, health, or belonging to a group, or flagging individuals in a camera feed. Fashion-MNIST is a safe benchmark because its subjects are garments. The same code, with different labels, would be a surveillance tool, and our 87–90% accuracy and 7×7 Grad-CAM maps could not justify a judgment about a person, nor explain one.

**Why the refusal is engineering.** It is a requirement, not a mood, and it shows up in the system. The ten classes are garments only. The model card lists "any decision about people" as out of scope. Uploads are stored as hashes, not pixels, so the service cannot become an archive of faces, and public history omits hashes. The ethics document names surveillance as the foreseeable misuse and warns that a convincing heatmap could lend false credibility to a harmful decision. I also state the gap: no technical control stops someone retraining the code on people. Our refusal is a documented boundary, not a lock.

## AI use
I used Claude Code and Codex, recorded in `ai-usage/`. I ran and checked the tests, executed the official runs, applied migrations 004 and 005, and verified the results myself. The results above can be reproduced from the repository.
