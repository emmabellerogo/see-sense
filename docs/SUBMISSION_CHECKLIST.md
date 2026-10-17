# Submission checklist (DRAFT — Fashion-MNIST correction, branch `correction/fashion-mnist`; updated after the official runs)

**Source of the categories:** the Product 3 rubric (`AIT-204-R-T3-Product3_SeeSense_VisionClassifiationService.docx`, 9 criteria) plus the
assignment's explicit requirements as stated by Emma: CIFAR-10 or Fashion-MNIST, controlled experiments, a `predictions` table with run linkage,
`GET /predictions`, calibrated probabilities, top-five probabilities, and ≥5 correct / ≥5 incorrect Grad-CAM examples with two evidence-based failure
interpretations. The rubric file itself does not name the dataset; the dataset requirement comes from the assignment text Emma supplied. **The full attached
assignment document was not available to the assistant that updated this list**, so each row below is checked against the rubric, Emma's
written requirements and the repository, not against the original assignment text: compare it with the real assignment before submitting.

Status: ✅ done and verified here · 🟡 code/docs ready, awaiting something below · ⏳ needs a person's action or the official run · ❌ missing.
"Evidence in repo" means the file exists; it does not mean a grader will accept it. Nothing was committed, pushed or deployed by the correction sessions. Emma applied migrations 004/005 herself and ran the 18 official runs; the assistant only read Supabase (plus one small, labelled live-test run, below).

| # | Requirement | Status | Evidence / what is missing |
|---|---|---|---|
| 0 | **Dataset is CIFAR-10 or Fashion-MNIST** | ✅ (local) / ⏳ deployed | Fashion-MNIST, 10 classes, official train/test, validation from train: [DATASET_CARD.md](DATASET_CARD.md), `shared/fashion.py`; 18 official runs on Supabase dataset 24. The **deployed** apps are still the synthetic version until this branch is deployed. |
| 1 | Baseline vs improved CNN, matched seeds and budgets | ✅ | Group `fashion-official-1`, runs 53-70, 3 seeds × 6 variants, matched-protocol check passed on all 18 records (`experiments/export_records.py`; SQL query 1). Result: **improved did not beat baseline** — [MODEL_CARD.md](../MODEL_CARD.md), [PART2_RESULTS.md](../PART2_RESULTS.md). |
| 2 | ≥3 techniques with controlled ablations | ✅ | Four leave-one-out variants with paired intervals ([summary.md](../experiments/results/fashion-official-1-analysis/summary.md)). Only batch norm helped; augmentation and dropout hurt at 20 epochs. Three seeds: wide intervals, disclosed. |
| 3 | Runs persisted; train **and** validation loss/accuracy every epoch; test accuracy, loss, macro-F1, confusion matrix, per-class precision/recall | ✅ | Verified on all 18 stored runs (20-epoch histories, 10×10 confusion, per-class arrays, calibration). Migrations 004/005 applied by Emma; columns confirmed. |
| 4 | Headline numbers sourced via SQL | ✅ | Tables come from `docs/comparison_queries.sql` (queries 1-7, 9, 10) run in the Supabase SQL editor (export ingested 2026-10-17, recorded in `fashion-official-1-sql/PROVENANCE.md`); the local-engine dry run matched cell for cell. |
| 5 | Grad-CAM: ≥5 correct, ≥5 genuinely incorrect, in the dashboard, + two evidence-based failure interpretations | 🟡 | 10 correct + 10 incorrect from run 54 (regenerated accuracy 87.82% = stored); shown in the Compare tab. Two interpretations are **drafted for human review** in [FASHION_GRADCAM_FAILURE_ANALYSIS.md](FASHION_GRADCAM_FAILURE_ANALYSIS.md). ⏳ Emma/Komal must read the figures and confirm or rewrite them. Note: Grad-CAM is blank for 97.8% of Dress and 87.6% of Coat predictions. |
| 6 | Calibrated probabilities | ✅ (measured) | Validation-fitted temperature; test ECE fell for 5 of 6 variants (full_improved 0.0173 → 0.0069), NLL ≈ unchanged; small gains, stated as such. |
| 7 | Tab 1 Concepts | ✅ | [ui/app.py](../ui/app.py), [ui/concepts.py](../ui/concepts.py); 10 classes. |
| 8 | Tab 2 Try It (upload + examples, top-5, Grad-CAM, cold start/errors, single POST) | ✅ locally / ⏳ deployed | `served_run_id` = 54, verified against group, dataset, labels, preprocessing. Real local check: Streamlit server + FastAPI serving run 54 (offline store, no cloud writes); the held-out example classified end to end. ⏳ Deploy. |
| 9 | Tab 3 Compare Models | ✅ locally | Final Fashion-MNIST tables, curves, confusion, per-class, calibration, Grad-CAM; provenance banner; historical synthetic evidence behind a labelled toggle. |
| 10 | Tab 4 Recent Classifications + drill-down | ✅ (live view read) | `recent_predictions` anon-readable without hashes (live test passed); drill-down tested. |
| 11 | Tab 5 Model Card | ✅ draft | [MODEL_CARD.md](../MODEL_CARD.md) with official results; draft until provenance is settled. |
| 12 | `predictions` table, `GET /predictions`, restricted anon history | ✅ | Live checks passed on Supabase (linkage, atomic function, hash-free anon view, anon denied on tables): `tests/test_live_supabase.py`. |
| 13 | Three-cloud deployment, URLs documented | ⏳ | URLs in [README.md](../README.md) still serve the synthetic version. Supabase URL placeholder. ⏳ Redeploy API and UI. |
| 14 | No raw images persisted | ✅ | README "Data" section; tests and the live row check assert no pixels/bytes in stored rows. |
| 15 | Dataset card, model card, architecture diagram | ✅ draft | [DATASET_CARD.md](DATASET_CARD.md), model card, README diagram. |
| 16 | Tests: multipart upload, Grad-CAM shape, regression, Supabase round trip, outage path | ✅ | `tests/` (mocks and in-memory DB) plus `tests/test_live_supabase.py` (live; 6 tests passed on 2026-10-17). Mocks and live runs are distinguished in the file headers. |
| 17 | CI green on every push | ⏳ | `.github/workflows/ci.yml` unchanged; confirm after pushing. |
| 18 | Ethics: two deployment contexts, safeguards, "not logged", Christian-worldview reflection | 🟡 | [ETHICS_AND_DISCLOSURE.md](ETHICS_AND_DISCLOSURE.md) updated. ⏳ §6 reflection is an AI-drafted scaffold; each author writes their own. **Komal's personal beliefs are left for her to write and review.** |
| 19 | Individual contribution statements and **individual videos** | ❌ | Placeholders only: [CONTRIBUTIONS.md](CONTRIBUTIONS.md). ⏳ Each person writes their own and records their own video. |
| 20 | AI-usage records | 🟡 | `ai-usage/` holds earlier sessions. ⏳ Add this correction/evidence session's record (export the transcript yourself). |
| 21 | Secrets not committed | 🟡 | `.gitignore` covers `.env`, `secrets.toml`; `data/`, `pilot-output/` ignored. The session read credentials only through the project's own code, never printed them. ⏳ Search git history for keys before pushing. |
| 22 | Changes committed and pushed | ❌ | Not done, by instruction. |
| 23 | Instructor question | ⏳ | Drafted, **not sent**: [INSTRUCTOR_QUESTION_DRAFT.md](INSTRUCTOR_QUESTION_DRAFT.md) (measured numbers filled in). No approval exists. |
| 24 | Two individual engineering reports, 750-1,000 words each, honest results | 🟡 draft | [EMMA_ENGINEERING_REPORT.md](EMMA_ENGINEERING_REPORT.md) (~980 words) and [KOMAL_ENGINEERING_REPORT.md](KOMAL_ENGINEERING_REPORT.md) (846 words), distinct, same verified team results. Emma's worldview reflection is in (Genesis 1:27, ESV; she should read it and make it her own). ⏳ Komal reviews/rewrites hers and confirms her contributions. Team results kept in [PART2_RESULTS.md](../PART2_RESULTS.md); historical: [PART2_RESULTS_HISTORICAL_SYNTHETIC.md](PART2_RESULTS_HISTORICAL_SYNTHETIC.md). **Not checked against the assignment text (not available).** |

## Remaining steps (manual)
1. Run `docs/fashion_official_1_export.sql` once in the Supabase SQL editor; download the result; `python -m experiments.ingest_sql_export <file> experiments/results/fashion-official-1-sql --source supabase-sql-editor`; `python -m experiments.analyze experiments/results/fashion-official-1-sql experiments/results/fashion-official-1-analysis`; check that `git diff` on the CSVs and `summary.md` shows no change in the numbers (if it does, update the model card and reports).
2. Read the Grad-CAM figures yourselves and confirm or rewrite the two interpretations.
3. Write your own contribution statements, worldview reflections and videos; add the AI-usage record for this session.
4. Decide whether to send the instructor question. Review, commit, push, confirm CI, redeploy.

## Tests (what each kind proves)
- **Mocked / in-memory (default `pytest`)**: code paths with a fake database and a tiny Fashion-shaped dataset. They prove nothing about Supabase.
- **Local Postgres engine (PGlite)**: migrations 001-005 apply and re-apply, `log_prediction` is atomic, anon reads only the hash-free view; the comparison queries and the packaged export run there. This is **not** Supabase.
- **Live Supabase (`SEE_SENSE_RUN_LIVE_TESTS=1`)**: `tests/test_live_supabase.py` — passed on 2026-10-17 (writes one small dataset row, one 1-epoch run in group `live-test`, two predictions + two image_metadata rows; nothing deleted).
- **Real-data tests**: `tests/test_fashion_data.py::TestRealFiles` run where the downloaded files exist; `tests/test_evidence_pipeline.py` checks the saved official records against the frozen protocol.
- **Real local UI check**: Streamlit server (health 200) + FastAPI serving run 54 from an offline copy; full app render and an end-to-end classification with no exception.
