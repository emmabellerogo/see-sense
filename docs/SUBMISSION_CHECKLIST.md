# Submission checklist

Sources: the assignment text and the rubric `AIT-204-R-T3-Product3_SeeSense_VisionClassifiationService.docx`. Evidence and method for every row: [FINAL_SUBMISSION_AUDIT.md](FINAL_SUBMISSION_AUDIT.md).
Status: ✅ verified · ⚠️ partly met or at risk · ⏳ needs a person · ❌ not met · ❓ could not be verified.
Passing tests and existing files are not proof that a grader will accept them; no grade or instructor approval is implied.

| # | Item | Status | Note |
|---|---|---|---|
| 1 | Dataset choice (Fashion-MNIST) and reasoning in README | ✅ | README "Product pitch and dataset choice" (uncommitted until pushed) |
| 2 | Migrations 001–005 applied; no raw image columns | ✅ | Production predictions 28 and 29 logged through the atomic function |
| 3 | Baseline and improved CNN, matched seeds and budgets, persisted | ✅ | Runs 53–70; recomputed and equal to live Supabase |
| 4 | ≥3 techniques with controlled ablations | ✅ | Four leave-one-out variants; improved did **not** beat baseline (89.73% vs 88.10%) |
| 5 | Headline numbers sourced via SQL | ✅ | Supabase SQL-editor export, `fashion-official-1-sql/PROVENANCE.md` (team attestation; numbers agree with live data) |
| 6 | API endpoints, OpenAPI tags, `/docs` | ✅ | All GETs and two POSTs exercised on production |
| 7 | Five tabs in code (LaTeX theory, Try It, Compare, Recent with histogram and drill-down, Model Card) | ✅ | Source and AppTest only |
| 8 | Five tabs on the deployed Streamlit app | ❓ | No browser tool; manual list in the audit §9 |
| 9 | Compare Models gallery ≥5 correct and ≥5 incorrect, baseline and improved | ✅ in repo / ❓ deployed | 10 + 10 each for run 53 and run 54, plus 20 shared images; regenerated accuracies 0.9001 and 0.8782 equal stored; uncommitted. Many baseline maps are blank (disclosed) |
| 10 | Two failure analyses (dataset vs model) and baseline/improved comparison text | ⏳ | AI-drafted; neither author has reviewed them |
| 11 | Calibration fitted on validation only, measured | ✅ | ECE improved in 5 of 6 variants; stated as small |
| 12 | pytest: multipart, Grad-CAM shape, fixed reference, Supabase round trip | ✅ | 323 passed, 6 live tests skipped (one load-induced AppTest timeout in an earlier run, passed on re-run); round trip is existing evidence from 2026-10-17 |
| 13 | CI green | ✅ | 17 of 17 recorded runs succeeded; re-check after pushing |
| 14 | README links (Streamlit, Render, Supabase ref, videos) within first 15 lines | ✅ | Lines 3–5 |
| 15 | Emma video 3–5 minutes | ❌ | Loom reports 5:00.19 |
| 16 | Komal video 3–5 minutes | ✅ duration / ❓ content | 4:09 |
| 17 | Video content (demo, theory, contribution) | ❓ | Not viewable here; watch both |
| 18 | Two distinct reports, 750–1,000 words | ✅ | Emma 976 raw (956 body); Komal 969 (923) |
| 19 | Reports include ethics (contexts, safeguards, not-logged) and worldview | ✅ | Ethics sections added to both; Komal to read hers |
| 20 | Reports include curves, confusion matrices, per-class P/R | ✅ | Curves and confusion matrices embedded; per-class table linked |
| 21 | Contribution evidence: bullets + task inventory | ⏳ | Written from records; **both authors must confirm**; Komal's original claim conflicted with records |
| 22 | Ethics document honest about implemented vs proposed | ✅ | No auth, rate limit, retention policy or CORS restriction is claimed |
| 23 | No secrets in tracked files or history | ✅ | Pattern scan; matches not printed |
| 24 | Historical synthetic evidence separated | ✅ | Separate folders and documents, labelled |
| 25 | Changes committed, pushed, CI green, Streamlit redeployed | ❌ | **Not done by the audit, by instruction** |
| 26 | Instructor approval of the dataset correction | ❌ | None exists; question drafted, never sent |

## Before submitting
1. Review `git diff`, commit and push (batches in the audit summary); confirm CI and the Streamlit redeploy.
2. Trim Emma's video below 5:00 and update the link if it changes.
3. Watch both videos; run the browser checklist (audit §9).
4. Komal and Emma confirm their statements; a person confirms the two Grad-CAM interpretations.
5. Read the baseline and shared Grad-CAM figures and the comparison draft; confirm or rewrite.
