# Contribution statements and individual videos

Each statement below is backed by the repository records named next to it (git history: Emma 25 commits, Komal 7; `ai-usage/` session records).
Where a statement could not be matched to those records, it is marked. Each person remains free to revise their own statement.

## Emma Rogoveanu

**Individual video:** [Emma's video walkthrough](https://www.loom.com/share/8a3f95c3eb824a01a19407af70d2c41c)
**Engineering report:** [EMMA_ENGINEERING_REPORT.md](EMMA_ENGINEERING_REPORT.md)

**Contribution statement** (assembled from Emma's report "What I owned" and the git/AI-session records; Emma to confirm or edit):
- Built the foundation: baseline and improved CNNs, reproducible training, the experiment runner, the Supabase schema and migrations 001–005, the FastAPI service, and the five-tab Streamlit dashboard.
- Applied migrations 004 and 005 to Supabase, ran the 18 official Fashion-MNIST runs (group `fashion-official-1`, run ids 53–70) and ingested the Supabase SQL-editor export.
- Carried out the Fashion-MNIST correction (dataset, protocol, calibration, atomic prediction logging, history views, Grad-CAM from stored checkpoints) in AI-assisted sessions on her machine; this work is **not** Komal's.
- Prepared the deployment configuration and the final documentation; wrote the ethics and worldview sections of her own report.
- Learned that the dataset requirement must be read from the full assignment before building, and that an honest null result ("improved" did not beat baseline) is still a result.

## Komal Khan

**Individual video:** [https://www.loom.com/share/d341818a31304cdb9add752bd470809a]
**Engineering report:** [KOMAL_ENGINEERING_REPORT.md](KOMAL_ENGINEERING_REPORT.md)

**Contribution statement** (Komal's wording, adjusted only where the repository records contradict it; see the reconciliation note below):
- Made the API runnable offline on a Windows laptop (`api/local_db.py`, `.env` loading fix, `docs/RUN_API_LOCALLY.md`, 3 tests) and verified the same dry-run fingerprint on both machines (`ai-usage/Komal-Session1.md`).
- Wrote the first experiment protocol (`docs/PART2_PROTOCOL.md`) and the exploring, analysis, SQL-checking and Grad-CAM-figure scripts (`experiments/explore.py`, `analyze.py`, `gradcam_evidence.py`), and ran the earlier **synthetic-shape** experiment offline (`ai-usage/Komal-Session2.md`, commit `2e5c231`).
- Reviewed the Fashion-MNIST results and Grad-CAM evidence and wrote her own engineering report, including the bias, honesty and humility reflections.
- Learned that a fair experiment matters more than a high score: the "improved" model did worse than the baseline, and that was still worth reporting.

> **Reconciliation note (from records, not from Komal).** Komal's earlier statement said she "ran all six configurations across three seeds (18 runs), saved them to Supabase and retrieved the results with SQL", that she "fixed setup before the official run (Fashion-MNIST, 20 epochs, batch size 128, learning rate 0.001, seeds 0–2)", and that she "explained two failures". The records show something different: the 18 official Fashion-MNIST runs were executed by Emma (Emma's session records and commits; Komal's sessions state that runs against the real Supabase project were out of scope, and her Part 2 runs were offline synthetic runs). The Fashion-MNIST protocol and setup are in Emma's commits (`5687587`), and the two Fashion-MNIST Grad-CAM failure interpretations were drafted by an AI assistant in Emma's session and are labelled drafts in `FASHION_GRADCAM_FAILURE_ANALYSIS.md`. The bullets above therefore keep only what the records support. Komal's report (which does not claim the official runs) is unchanged. **Komal should confirm or correct this before submission.**
>
> <details><summary>Komal's original wording (superseded pending her confirmation; preserved unchanged)</summary>
>
> I was responsible for the experiment and the evidence. I fixed setup before the official run (Fashion-MNIST, 20 epochs, batch size 128, learning rate 0.001, seeds 0–2), ran all six configurations across three seeds (18 runs), saved them to Supabase and retrieved the results with SQL. Also preparing the Grad-CAM evidence and explained two failures. The switch to Fashion-MNIST was done by Emma. I learned that fair experiment matters more than a high score, our "improved" model did worse than the baseline, and that was still worth reporting.
> </details>

## Task inventory (all tasks identified in the project)

Owner column is taken from git authorship and the session records; "not recorded" means the records do not say.

| # | Task | Owner (per records) | Evidence |
|---|---|---|---|
| 1 | Fork the template; choose Fashion-MNIST; document the choice in the README | Emma (first chose the starter's synthetic shapes, corrected 2026-10-17) | README, `docs/PROJECT_NOTES.md`, `DATASET_CARD.md` |
| 2 | SQL migrations for runs, predictions, image_metadata; apply to Supabase | Emma wrote and applied 001–005 | `db/migrations/`, live tests 2026-10-17 |
| 3 | Baseline CNN (≥2 conv blocks, no augmentation) and its persisted runs | Emma | `api/training.py`, runs 53, 59, 65 |
| 4 | Improved CNN (batch norm, dropout, augmentation, LR schedule), matched budget | Emma | same, runs 54, 60, 66 |
| 5 | Protocol, exploration, analysis, SQL-check and Grad-CAM scripts (first version) | Komal | `docs/PART2_PROTOCOL.md`, `experiments/` (commit `2e5c231`) |
| 6 | Fashion-MNIST frozen protocol and 18 official runs; six configurations, ablations | Emma | `docs/FASHION_PROTOCOL.md`, runs 53–70 |
| 7 | SQL comparison queries and Supabase SQL-editor export | Emma (queries committed by Emma; ran in the Supabase SQL editor) | `docs/comparison_queries.sql`, `fashion-official-1-sql/` |
| 8 | Calibration (temperature scaling fitted on validation) | Emma | `api/calibration.py` |
| 9 | FastAPI endpoints and OpenAPI tags | Emma | `api/main.py` |
| 10 | Offline local API mode and Windows verification | Komal | `docs/RUN_API_LOCALLY.md` |
| 11 | Five-tab Streamlit dashboard | Emma | `ui/` |
| 12 | Grad-CAM evidence for improved (run 54) and baseline (run 53), shared-image comparison, failure and comparison write-ups | Emma's session (AI-drafted interpretations; human confirmation pending) | `experiments/results/fashion-official-1-gradcam/`, `…-gradcam-baseline/`, `docs/FASHION_GRADCAM_*.md` |
| 13 | pytest suite and CI | Emma (authorship); Komal added 3 local-DB tests | `tests/`, `.github/workflows/ci.yml` |
| 14 | Deploy Supabase, Render and Streamlit | Configuration by Emma; who performed the deploy clicks is not recorded | `render.yaml`, live URLs |
| 15 | Ethics and disclosure document | Drafted in Emma's AI session; worldview reflections are in each person's own report | `ETHICS_AND_DISCLOSURE.md` |
| 16 | Dataset card, model card, architecture diagram | Emma (AI-assisted) | `DATASET_CARD.md`, `MODEL_CARD.md`, README |
| 17 | Individual engineering reports (750–1,000 words) | Each person, own report; Komal used an AI assistant to help draft and simplify | `docs/*_ENGINEERING_REPORT.md` |
| 18 | Individual videos (3–5 min) | Each person | links above |
| 19 | AI-usage records | Each person for their own sessions | `ai-usage/` |
| 20 | Question to instructor about the dataset correction | Drafted, **not sent**; no approval exists | `INSTRUCTOR_QUESTION_DRAFT.md` |

## Supporting records
- AI-usage records: [Emma Session 1](../ai-usage/Emma-Session1.md), [Emma Session 3](../ai-usage/Emma-Session3.md), [Emma Session 4](../ai-usage/Emma-Session4.md), [Komal Session 1](../ai-usage/Komal-Session1.md), [Komal Session 2](../ai-usage/Komal-Session2.md). There is no Emma Session 2 record; none was created.
- Git history: `git log --format='%an %s'`
