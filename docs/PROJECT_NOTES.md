# Project working notes

Working notes, not a technical report. Technical documents (model card, dataset card, results) describe the final system only.

## 2026-10-17 — dataset requirement missed; correction started

**What happened.** The assignment requires the CNN to be trained on **CIFAR-10 or Fashion-MNIST**. The project was instead built, tested,
deployed and written up on the **synthetic four-shape dataset** that the starter already contained (horizontal bar, vertical bar, diagonal line, filled block).

**Why (accountability).** ChatGPT caused the planning error. It guided this project from the grading rubric and the starter README without
checking the full assignment instructions soon enough, explicitly advised keeping the synthetic shapes, and repeatedly understated how much
work remained. The synthetic shapes were already in the starter; Emma and Komal did not introduce that mismatch, and neither should be blamed
for following that guidance.

**What was preserved.** Nothing was deleted. `main` still holds the deployed synthetic version. The synthetic results (`experiments/results/official-supabase-1-*`),
their checkpoints (still loadable and labelled `synthetic_shapes`), the AI-usage logs and the contribution records are unchanged.
The old dataset card is kept as `docs/DATASET_CARD_HISTORICAL_SYNTHETIC.md`.

**Correction branch.** `correction/fashion-mnist` (local, uncommitted at the time of writing). It switches the project to Fashion-MNIST
(ten classes, official train/test partitions, validation carved from the training images), adds calibration, the `predictions` log and history views,
and moves the synthetic experiment to a clearly labelled historical section.

**Not done / not claimed.**
- Official Fashion-MNIST results now exist (group `fashion-official-1`, runs 53-70, 3 seeds): the improved model did not beat the baseline, and the reports say so. SQL provenance is the Supabase SQL-editor export (ingested 2026-10-17); an earlier local dry run matched it.
- Nothing was committed, pushed or deployed by the correction session. Migrations 004 and 005 were applied to Supabase by Emma and verified (live tests passed 2026-10-17); the deployed Render/Streamlit apps are still the earlier version.
- No instructor approval has been obtained for anything. A question is drafted in `docs/INSTRUCTOR_QUESTION_DRAFT.md` and has **not** been sent.

**Deviation to disclose in the final submission, if still unresolved then:** the first deployed version used synthetic shapes. The final evidence is
Fashion-MNIST only if the official runs, the migrations and the redeployment are completed before the deadline (2026-10-18 23:59 America/Phoenix).
