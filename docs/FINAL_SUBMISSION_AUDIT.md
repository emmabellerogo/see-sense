# Final submission audit — See-Sense

Audit date: 2026-10-18 (assignment due 2026-10-18 11:59 PM). **Follow-up the same day:** baseline Grad-CAM evidence was generated and added (see §4a); sections below are updated only where that follow-up changed a check. Authoritative sources: the assignment text (Halo "CLC – Product 3", read from the saved attachment `~/.codex/attachments/00f6abb4…/Pasted text.txt`) and the rubric (`AIT-204-R-T3-Product3_SeeSense_VisionClassifiationService.docx`, nine criteria totalling 100%).
This audit gives no grade and no instructor approval is implied (none exists for the dataset correction; the question in `INSTRUCTOR_QUESTION_DRAFT.md` was never sent).

Legend: **PASS** verified by something actually run or inspected; **FAIL** a defect remains; **NOT VERIFIED** could not be checked here. "Existing evidence" = produced before this audit and not re-run; "This audit" = run today.

## 1. Submission state

| Item | Finding |
|---|---|
| Git at start | Clean; local `main` = `origin/main` = `24a09d5`. |
| CI | All 17 recorded GitHub Actions runs succeeded (9 on `main`, 8 on the PR branch). "Green on every push" holds for recorded history. |
| Render | `/version` → `git_sha 24a09d5` (matches HEAD), `supabase_project_ref glcxpjaeobpppkizlcke`, torch 2.4.1+cpu; `/healthz` → ok, storage supabase. |
| Streamlit | URL reachable (HTTP 200 after Streamlit's cookie redirect, app is public). Its deployed *content/version* cannot be read without a browser: NOT VERIFIED. |
| Uncommitted work | **Created by this audit, not committed:** README, MODEL_CARD, PART2_RESULTS, `docs/` (reports, CONTRIBUTIONS, ETHICS, DATASET_CARD, FASHION_PROTOCOL, PROJECT_NOTES, this audit, checklist), `ui/app.py`, `ui/evidence.py`, one edited and one new test file, the new `experiments/gradcam_compare.py`, `docs/FASHION_GRADCAM_BASELINE_COMPARISON.md` and the new folder `experiments/results/fashion-official-1-gradcam-baseline/`. **Until these are committed and pushed, the graded `main` still lacks the required report sections.** |
| Discrepancies found | Stale README banner ("deployed apps are still synthetic", "004/005 NOT applied"); stale ETHICS T1 text and §6 placeholder; CONTRIBUTIONS said "Placeholders only" and Emma's statement was a TODO; Komal's statement over-attributed work (see §8); neither report had the required ethics section; no product pitch/dataset reasoning in README; Emma's video is 5:00.19. All fixed or flagged below. |
| Untracked required assets | None. `.env` and `ui/.streamlit/secrets.toml` exist locally and are git-ignored (not tracked, never printed). |

## 2. Requirement matrix

### 2a. Build steps, report, deliverables (assignment)

| # | Requirement | Evidence | Verification performed | Status | Remaining action |
|---|---|---|---|---|---|
| 1 | Choose CIFAR-10 or Fashion-MNIST; document choice and reasoning in README pitch | README "Product pitch and dataset choice" (added today); `DATASET_CARD.md` | Read; spot-checked: the train-images and t10k-images/labels hashes in the card equal those in the saved dataset-24 identity; the sample manifest repeats the dataset hashes | PASS (after push) | Commit/push README. |
| 2 | Migration for runs, predictions, image_metadata; no raw bytes; applied | `db/migrations/001–005`; no bytea/pixel columns | Grepped migrations; **This audit:** two production `/predict` calls logged successfully through the atomic function (proves 004/005 are applied); existing live test 2026-10-17 | PASS | none |
| 3 | Baseline CNN ≥2 conv blocks + FC head, no augmentation, stated epochs, persisted | Run 53/59/65: bn False, dropout 0, aug False, 20 epochs, 9,098 params | Recomputed from records; matches live `/runs` | PASS | none |
| 4 | Improved model ≥3 techniques, same budget, persisted | Runs 54/60/66: batch norm, dropout 0.3, augmentation, cosine schedule (4 techniques), 20 epochs, 9,146 params | Same | PASS | none |
| 5 | `POST /predict` multipart → label, probabilities, base64 Grad-CAM, logs to predictions + image_metadata | `api/main.py`; **This audit:** 2 production calls HTTP 200 | Response had label, 10 probs summing to 1, sorted top-5, `cam_png`, `prediction_id`, `logged: true` | PASS | none |
| 6 | `GET /runs`, `/predictions` (joined to runs), `/classes`, `/healthz`, `/version`, OpenAPI tags, `/docs` | **This audit:** all GET live | `/runs` 50 rows, no checkpoint field; `/predictions` joined run, no hash; `/classes` 10 labels; `/docs` 200; every operation has a tag (tag *descriptions* absent: cosmetic) | PASS | optional: add `openapi_tags` descriptions (needs redeploy). |
| 7 | Streamlit tabs: Concepts (LaTeX conv, RF formula, Grad-CAM derivation), Try It, Compare Models, Recent Classifications (sortable, histogram), Model Card | `ui/app.py`, `ui/concepts.py`; 323 tests incl. Streamlit AppTest renders | Source inspected: LaTeX for conv, output size, RF, Grad-CAM; sortable `st.dataframe`; `st.bar_chart` histogram; drill-down. **Deployed browser check not possible** | Code PASS / deployed **NOT VERIFIED** | Manual checklist §9. |
| 8 | Compare Models: baseline vs improved on gallery of ≥5 correct and ≥5 incorrect test images, sourced from runs | `fashion-official-1-gradcam/` (improved, run 54) and `fashion-official-1-gradcam-baseline/` (baseline, run 53): 10 correct + 10 incorrect each, plus 20 shared images and a blank-map table; Compare Models section "Baseline (baseline-seed0, run 53)…" | **Follow-up:** regenerated baseline accuracy 0.9001 = stored 0.9001 and improved 0.8782 = stored; identity check (group, dataset 24, split hash, seed, labels, preprocessing, local file hashes) passed; viewed all baseline and shared figures; 28 new tests incl. an AppTest render | PASS in repo (uncommitted); deployed tab **NOT VERIFIED** | Commit/push, then check the deployed tab. Caveat: only 5 of the baseline's 10 correct and 5 of its 10 incorrect examples have a non-blank map (§4a). |
| 9 | Video per member, 3–5 min, end-to-end demo, theory, individual contribution | README line 5 | Loom oEmbed: Emma **300.191 s (5:00.19)**; Komal **248.9 s (4:09)**; both share pages HTTP 200 anonymously | Emma **FAIL by 0.2 s** on a strict reading; Komal duration PASS; content of both **NOT VERIFIED** | Emma: trim or re-export under 5:00 (or accept risk). Both: watch to confirm demo, theory, contribution. |
| 10 | Report 750–1,000 words per person, in README or linked | Emma `docs/EMMA_ENGINEERING_REPORT.md`; Komal `docs/KOMAL_ENGINEERING_REPORT.md` | Re-counted after the follow-up (raw / without image lines / without markup): Emma 976 / 971 / 956; Komal 969 / 964 / 923 (inside 750–1,000 by every method) | PASS | Re-count if edited again. |
| 11 | Report: top-1 test accuracy per model; curves; confusion matrix; per-class precision/recall | Accuracy in text; loss curves, accuracy curves and confusion matrices **embedded** in both reports (relative paths checked); per-class table linked | Recomputed accuracy, confusion, P/R from records (§3); viewed the loss curves and confusion matrices | PASS | Per-class numbers beyond Shirt are in the linked CSV, not the report text. |
| 12 | Report: Grad-CAM qualitative, ≥2 failures argued dataset vs model | Emma: both models' overlays, blank-map comparison, Failure 1 (#2914, dataset-like), Failure 2 (#787, model/method); Komal: both models, shared errors, blank maps | Checked against the viewed figures, `blank_cam_by_class.csv` (both models) and `shared_examples.csv` | PASS (uncommitted) | Interpretations (`FASHION_GRADCAM_FAILURE_ANALYSIS.md`, `FASHION_GRADCAM_BASELINE_COMPARISON.md`) are AI drafts; **a person must confirm**. |
| 13 | Report: ethical reflection (harm contexts, two safeguards, what we did not log) | Added today to both reports | Safeguards labelled as not implemented where true | PASS (after push) | Push. |
| 14 | Report: worldview reflection (Genesis 1:27, one refused use, why engineering) | Emma: full; Komal: bias/honesty/humility with scripture | Read | Emma PASS; Komal PASS on content | Komal's refused-use is implicit (face recognition / medical imaging example), not stated as "a use I would refuse". Komal to confirm. |
| 15 | pytest: multipart smoke; Grad-CAM shape == input; fixed-reference regression; Supabase round-trip linking prediction to run | `tests/test_api_predict.py`, `test_reference_prediction.py`, `test_live_supabase.py` | **This audit:** `pytest` 323 passed, 6 skipped (the 6 live tests, opt-in); `pip check` clean | PASS (mocked) / round-trip **existing evidence** | Live test not re-run (it writes to production; not authorized). Production predictions 28/29 show the linkage by hand. |
| 16 | Deploy all three clouds; README links within first 15 lines | README lines 3–5 | Counted: Streamlit, Render, Supabase ref, both videos, reports all in lines 3–6 | PASS | none |
| 17 | Individual-contribution evidence: bullets + list of all tasks | `docs/CONTRIBUTIONS.md` | Rewritten today from git and session records; 20-row task inventory | PASS (after push) | Both statements need their owner's confirmation. |

### 2b. Rubric criteria

| Criterion (weight) | Evidence | Status | Biggest risk |
|---|---|---|---|
| Convolution, stride, padding, RF (12%) | Concepts tab: formulas, worked table (28→26/28/14), RF recursion + derivation, 3×3 stack 3→5→7, project CNN 10×10 RF, capacity paragraph added today | PASS (code); deployed render NOT VERIFIED | Capacity paragraph is new (needs push/redeploy). |
| Baseline vs improved CNN (12%) | Runs 53–70, matched seeds/budget, SQL-sourced, ablations | PASS | Improved did not beat baseline: honestly stated. |
| Grad-CAM (12%) | Derivation in Concepts; 10+10 overlays for each of run 53 and run 54, shared images, two failure analyses | PASS in repo (uncommitted) | Analyses AI-drafted, unconfirmed; many maps are blank (baseline: all Dress, Coat, Sneaker, Ankle boot), disclosed. |
| Augmentation/regularization/scheduling (12%) | Four leave-one-out variants, paired CIs, README + Compare tab | PASS | Three seeds: wide intervals, disclosed. |
| Three-cloud deployment & tabs (10%) | Render + Supabase verified live; Streamlit reachable | Render/Supabase PASS; tabs **NOT VERIFIED** | No browser available. |
| Code execution (10%) | 323 passed; CI green on 17/17 runs; outage and cold-start tests exist | PASS | Live tests skipped in CI. |
| Professional documentation (8%) | Pitch, URLs, model card, dataset card, persisted/not-persisted, diagram, justification, per-member statements | PASS (after push) | README edits uncommitted. |
| Participation & contribution (16%) | Git: Emma 25 commits, Komal 7; statements reconciled | At risk | Uneven contribution is visible in the records; Komal's original claim conflicted with them. |
| Ethics (8%) | Two contexts, 2 technical + 2 governance safeguards (status-labelled), "not logged" section, worldview in reports | PASS | Several safeguards are PROPOSED, not built, and say so. |

## 3. Experiment integrity and numbers (this audit unless noted)

- **Dataset/splits:** 18 runs (ids 53–70), group `fashion-official-1`, dataset 24, one split hash (`786ed810…`), n = 54,000/6,000/10,000, 20 epochs, batch 128, Adam 0.001, seeds 0–2 for every variant. Official file hashes match the dataset card.
- **Controlled ablations:** configuration flags differ from `full_improved` by exactly one technique per variant (checked on seed 0).
- **Checkpoint selection:** `best_epoch` equals the argmin of validation loss for all 18 runs; test set scored once (protocol).
- **Histories:** all 18 runs have 20-epoch train/val loss and accuracy arrays, none null. Their origin is the training loop per the protocol; I did not trace every code path, and a grep of the experiment code found no history back-filling.
- **Metrics:** confusion-matrix accuracy equals stored accuracy for all runs; stored per-class precision/recall equals values recomputed from the confusion matrices (tolerance 1e-6).
- **Means:** baseline 89.73 ± 0.47, full improved 88.10 ± 0.29 (recomputed); other variants match README, Model Card, PART2_RESULTS and both reports.
- **Live agreement:** public `/runs` returns the same accuracy, confusion, histories, calibration and per-class arrays as the saved records for all 18 runs.
- **Calibration:** temperature fitted on validation only (`fit_split: validation`, n = 6,000); test ECE fell in 5 of 6 variants (full improved 0.0173 → 0.0069; no_dropout rose 0.0073 → 0.0082); run 54 T = 0.8705, ECE 0.0168 → 0.0091. Docs match.
- **No claim that improved beat baseline:** searched README, Model Card, results, reports, UI: none.
- **SQL provenance:** `PROVENANCE.md` records source `supabase-sql-editor`, executed 2026-10-17T22:49:57Z, PostgreSQL 17.11, export sha256 and query-file sha256. I cannot independently prove from here that it was run in the Supabase editor (that is the team's attestation); the numbers in it agree with live Supabase.
- **Served model:** `config/served_model.json` names run 54 and says it was fixed before results; the date is self-attested (git history is the only external check). Live: 10 correct class names, temperature 0.8705, artifact preprocessing.
- **Historical synthetic evidence** is kept in separate folders/documents labelled historical; live `/runs` still lists those 30 old rows and 2 test groups, which the UI separates.

## 4. Grad-CAM and theory

- Viewed `gradcam_correct.png` (10) and `gradcam_incorrect.png` (10); all from run 54, official test indices. Correct: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. Incorrect: ten confident errors (selection rule documented; not a typical error sample).
- Blank Dress/Coat maps are visible in the figures and measured (97.8% / 87.6%); documents say this is a property of the method and checkpoint, not proof that no region supports the class.
- Failure analyses are in `FASHION_GRADCAM_FAILURE_ANALYSIS.md` and Emma's report; they match what I saw (#2914 map on background/upper; #787 predicted-class map empty). They remain AI-drafted.
- Arithmetic checked: 28/3/1/0→26; 28/3/1/1→28; 28/3/2/1→14; RF 3→5→7; project CNN RF 3→4→8→10 and jump 4; baseline parameter count 9,098.
- Baseline Grad-CAM was added in the follow-up (§4a).

### 4a. Baseline Grad-CAM follow-up (read-only, nothing retrained, no prediction endpoint called)

- Read run 53 (`baseline-seed0`) and run 54 records and checkpoints from Supabase with the existing private credentials (select-only; checkpoints held in memory). `experiments/gradcam_evidence.py` produced the baseline folder, `experiments/cam_blank_check.py` the blank-map table, and the new `experiments/gradcam_compare.py` the shared-image figures, `shared_examples.csv` and `identity_check.json`.
- **Accuracy:** baseline regenerated **0.9001** = stored 0.9001 (9,001 correct, 999 wrong); improved regenerated 0.8782 = stored 0.8782. The improved evidence folder is unchanged.
- **Identity checks (passed, enforced in code and tests):** same group `fashion-official-1`, dataset 24, split hash `786ed810…`, seed 0, ten class labels in the same order, identical preprocessing, local official file hashes equal to the stored dataset identity, same epochs/batch/lr/optimizer/split sizes.
- **Counts:** baseline 10 correct and 10 incorrect examples, genuinely so (predicted equals/differs from true; probability vectors agree). Shared set: the improved model's own 20 gallery images, with each model's prediction labelled separately (baseline right on 11 of 20, improved on 10 of 20; baseline right on #9922 where the improved model is wrong).
- **Blank maps (counted over 10,000 test images and seen in the figures):** baseline blank for 100% of Dress (987/987), Coat (996/996), Sneaker (1,030/1,030), Ankle boot (958/958) and 72.9% of Pullover (744/1,020); improved blank for 97.8% of Dress and 87.6% of Coat only. Only 5 of the baseline's 10 correct and 5 of its 10 incorrect examples show a heat map. Documents state that a blank map does not show that no region supports the class.
- **Findings recorded without over-claiming:** where both maps exist they light garment parts for T-shirt, Trouser and Shirt and mostly background for Sandal; the baseline predicted the same wrong class as the improved model on 9 of the improved model's 10 confident errors. No causal claim is made and nothing says the improved model won.
- `docs/FASHION_GRADCAM_BASELINE_COMPARISON.md` is an **AI draft; neither Emma nor Komal has reviewed it**, and the reports say so.
- **Tests added:** `tests/test_baseline_gradcam.py` (28 tests: loading, source identity, example counts, regenerated vs stored accuracy, shared-image identity, rejection of wrong run/dataset/split/group/accuracy/count, the identity guard, and an AppTest render of the new Compare Models section). Full suite after the follow-up: **323 passed, 6 skipped** (live tests); `pip check` clean. One earlier full run failed the new AppTest while the machine was under heavy load (load average about 11): the render exceeded the AppTest's 60 s limit. It passed alone and in a later full run (render about 27 s). The existing AppTest tests share this load sensitivity.

## 5. Deployed UI/API

- **Done (API):** all GET endpoints, OpenAPI, run-54 identity, label order. No browser tool was available, so **no deployed UI claim is made**.
- **Production writes (the two authorized):** multipart `POST /predict`, run 54, bundled official-test images, no retry:
  - **Prediction ID 28** — `8-bag-1.png` → Bag, 99.78%.
  - **Prediction ID 29** — `7-sneaker-1.png` → Sneaker, 99.75%.
  Both appear in `GET /predictions` linked to run 54, without hashes. Not deleted. (The app's bundled-sample path is also a multipart `/predict` call, so both used the same endpoint with different bundled files.) `/train` was not called and no datasets were created.

## 6. Stress tests

Existing mocked tests already cover, and all pass in the 323: missing/empty/oversize (413)/undecodable (422) uploads; unknown run (404); API timeout and cold start (`wake_api` retries GETs only, POST never retried); malformed/non-list responses; missing checkpoint (404/"isn't available"); Supabase outage (503, no leaks); failed atomic logging (503, no result); empty history; malformed/missing evidence files; probability-vector validity and top-5 ordering; Grad-CAM dimensions; fixed-reference regression. **This audit:** re-ran the whole suite (before and after edits), and ran the evidence/reference/Try-It tests from `/` with `-c` to confirm paths are working-directory independent (64 passed). Live-Supabase suite not re-run (writes to production); its last pass (2026-10-17, 6 tests) is existing evidence, including the anon-denial check. Anonymous denial was therefore **not re-verified today**.

## 7. Privacy and security

- Prediction↔run linkage and atomic logging: PASS (tests; live predictions 28/29).
- Public `/runs`, `/predictions`: no image hashes, no checkpoints. `/runs` exposes the dataset *split* hash (not an image hash).
- No raw pixels persisted: schema has no pixel columns; tests assert it; `/predict` returns the hash to the caller only.
- Secrets: no tracked `.env`/`secrets.toml`; pattern scan of the working tree and all history found no key-shaped strings (matches not printed); nothing rotated.
- Curated public benchmark samples (`ui/assets/samples`, official test images) are distinct from user uploads, which are processed in memory.
- Not implemented and documented as such: authentication, rate limits, restricted CORS (`*`), retention policy, pre-upload notice (a *warning* about out-of-distribution uploads exists), reporting contact. `POST /train` and `/datasets` are unauthenticated on the public API (the deployed service has no dataset files).
- Provider-side logs (Render, Streamlit, Supabase) were not inspected.

## 8. Individual deliverables

- Word counts and distinctness: see matrix row 10; texts differ (headings and content).
- **Emma's contribution statement** is now written from the records and her own report; it needs her confirmation.
- **Komal discrepancy (reconciled against records, not motives):** her statement said she ran the 18 official Supabase runs. Records show Emma ran them (Komal's sessions put Supabase out of scope; her runs were offline synthetic ones; the Fashion protocol and tooling commits are Emma's). `CONTRIBUTIONS.md` now keeps only record-supported items, keeps her original wording verbatim in a collapsed block, and asks her to confirm. Her report never claimed the runs; I fixed its truncated sentence and made the same attribution explicit.
- **Komal's worldview/AI-use:** present and in her own voice; I did not alter her beliefs. The ethics section I added to her report contains only project facts; she should read it, since it is now under her name.
- Ethics doc §6 placeholder text removed; it now points to each author's own reflection and invents nothing.
- AI transcripts preserved; `ai-usage/` unchanged. There is no Emma Session 2 record and none was made.
- Videos: both resolve anonymously; Emma 5:00.19 (over), Komal 4:09. Content, audio and transcripts **NOT VERIFIED**.

## 9. Manual checks left (cannot be done by this audit)

1. Watch both videos fully: end-to-end demo, theory, individual contribution; trim Emma's below 5:00.
2. In a fresh private browser window open the Streamlit URL: five tabs render; equations show; Try It: bundled sample classifies, top-5 and Grad-CAM show, upload warning appears; Compare Models: tables, curves, confusion, per-class, Grad-CAM gallery, historical toggle; Recent Classifications: table sorts, histogram, drill-down to run 54 (ids 28/29 should appear); Model Card renders.
3. Komal reads and confirms her statement, her report's new ethics section, and the reconciliation note; Emma confirms hers.
4. A person reads the Grad-CAM figures (improved, baseline and shared) and confirms or rewrites the failure interpretations and the baseline comparison text (both are AI drafts).
5. After pushing, check CI and that Streamlit redeployed (Concepts tab shows the capacity paragraph; Compare Models shows the baseline section, blank-map table and shared images).
