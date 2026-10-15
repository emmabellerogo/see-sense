# AI Usage Record — See-Sense

- **Student:** Komal Khan
- **Date:** 2026-10-15
- **Session title:** Part 2 experiment: protocol, 30 official runs, analysis, Grad-CAM
- **Project part / step:** Part 2 — steps 2–6 of `docs/PART1_HANDOFF.md` §9 (explore, freeze, official run, analyze, Grad-CAM)
- **AI tool:** Claude (agentic session in a cloud Linux workspace)
- **Purpose:** Complete the experiment tasks: choose a fair protocol from
  training/validation evidence only, run the six configurations over several
  seeds, retrieve results with SQL, interpret them honestly, and prepare
  Grad-CAM evidence.
- **Files changed (if any):**
  - `experiments/explore.py` — **new**, exploration that deletes test metrics before printing
  - `experiments/analyze.py` — **new**, tables and figures from SQL query output
  - `experiments/sql_check.mjs` — **new**, runs `docs/comparison_queries.sql` unchanged on local Postgres (PGlite)
  - `experiments/gradcam_evidence.py` — **new**, Grad-CAM gallery from real test predictions
  - `experiments/results/` — **new**, exploration logs, runner output, SQL CSVs, figures
  - `docs/PART2_PROTOCOL.md`, `docs/PART2_RESULTS.md` — **new**
  - `docs/PART1_HANDOFF.md` — filled in the frozen-protocol table
  - `ai-usage/Komal-Session2.md` — this file
- **Verification performed / results:**
  - Exploration printed and saved only training/validation numbers (test metrics deleted in code)
  - Protocol written to `docs/PART2_PROTOCOL.md` **before** the official run; run used `--expect-fingerprint`
  - Official run: 30 saved, all read-back checks passed, exit 0
  - SQL query 1 audit: PASS (5 seeds × 6 runs, one dataset/split/budget per seed)
  - Summary numbers recomputed independently in Python from the saved runs: **0 mismatches** with SQL
  - Every run: confusion matrix total 400 and diagonal/total = stored accuracy; 30-epoch curves; same split hash
  - Grad-CAM: regenerated test split reproduces the stored accuracy of `full_improved-seed0` exactly (94.75%)
  - `pytest -q`: 82 passed, 1 skipped
  - **Not done here:** the Supabase copy of these runs

## Transcript

Prompts, Claude's responses and the relevant tool and test results. Claude's
replies are condensed where they only narrated steps; commands and outputs are
quoted. No API keys, passwords or tokens were used or shown in this session.

---

**Komal:** _[screenshot of the Part 2 task list: 1. choose a fair experiment
setup; 2. run all six configurations; 3. repeat with matched training seeds;
4. save every run to Supabase and retrieve results with SQL; 5. interpret the
findings; 6. prepare the Grad-CAM evidence]_ i needed to complete these parts
which were not done

**Claude:** Planned the work in five stages: explore → freeze and run →
analyze → Grad-CAM → write up. Because Emma owns Supabase, it used the offline
database through the real API, and ran the project's SQL queries unchanged on
a local Postgres engine (PGlite) with migrations 001 + 002 applied.

**Step 1: exploration (validation only).** Wrote `experiments/explore.py`,
which calls the same `run_experiment` as the API, deletes test metrics, and uses
training seeds 100/101 (never the official ones).

```
Difficulty (lr 0.01, batch 32, 15 epochs), val accuracy per seed:
  800 rows, noise 0.1 (Part 1 default): baseline 1.0, 1.0 | full_improved 1.0, 1.0
  2000 rows, noise 0.9:                baseline 0.93, 0.90 | full_improved 0.985, 0.97
  2000 rows, noise 1.0:                baseline 0.855, 0.915 | full_improved 0.95, 0.935
→ chose 2000 rows / noise 1.0 (off the ceiling; 200-image validation set)

Budget (lr × batch × epochs; 72 runs):
  lr 0.03: baseline stuck at chance (val acc 0.23–0.33) → unfair to baseline
  lr 0.01, batch 64: one baseline seed collapsed (0.325)
  lr 0.01, batch 32, 30 epochs: stable for all; improved models' best val loss ≈0.146
→ chose lr 0.01, batch 32, 30 epochs

Technique strengths (full_improved, mean best val loss):
  shift 1 px best at every dropout level; dropout 0.2 ≈ 0.3 < 0.5
→ chose shift 1, kept dropout 0.3; 5 seeds (runs ≈8 s each)
```

**Step 2: freeze, then run.** Registered the dataset (`POST /datasets` → id 2),
dry run → fingerprint `cf05f954…`, wrote `docs/PART2_PROTOCOL.md`, then:
```
python -m experiments.run_experiments --api-url http://127.0.0.1:8000 --dataset-id 2 \
  --group official-1 --seeds 0,1,2,3,4 --epochs 30 --batch-size 32 --lr 0.01 --split-seed 0 \
  --dropout 0.3 --max-shift 1 --scheduler cosine --expect-fingerprint cf05f954... \
  --out experiments/results/official-1-runs.jsonl --show-test
→ done: 30 saved, 0 already existed, 0 failed, 0 not attempted   (exit 0)
```
_Fixes along the way: the SQL runner needed `OVERRIDING SYSTEM VALUE` for the
identity columns, and the query splitter first left part of each header line in
the SQL; both were fixed before the official data was loaded._

**Step 3: SQL and analysis.**
```
query 1 (audit): every count = 1 for all 5 seeds → PASS
query 5: full_improved 95.20% ± 0.69 | no_augmentation 94.65 | no_scheduler 94.60
         no_dropout 94.55 | baseline 73.00 ± 11.64 | no_batchnorm 44.00 ± 31.80
query 6 (vs full, same seed, mean Δ, 95% CI):
  augmentation −0.55 pp [−1.57, +0.47]   dropout −0.65 [−2.57, +1.27]
  scheduler −0.60 [−1.62, +0.42]          batch norm −51.20 [−90.28, −12.12]
```
**Claude:** Augmentation, dropout and the scheduler show no clear effect: all
intervals include zero, and signs differ across seeds. Batch norm: 3 of 5
no-batch-norm runs never learned. Their training loss stayed at 1.386 = ln(4)
and they predicted "vertical" for all 400 test images (22.75%), so the effect is
mostly training stability at lr 0.01. Noted as a limitation: exploration hadn't
tested `no_batchnorm`. The protocol was **not** changed after seeing results.

Independent check: recomputed means, sds and macro-F1 from the saved runs in
Python → 0 mismatches with SQL.

**Step 4: Grad-CAM.** `full_improved-seed0` (21 of 400 wrong). Rebuilt the
test split (accuracy reproduced exactly), and selected 5 correct and 8
incorrect predictions. The first figure was hard to read at noise 1.0, so it
was redrawn with the true shape (rebuilt by replaying the generator's random
stream) outlined in cyan. Two failures were explained:
- #292 horizontal → block (83%): the right region (on the bar), but only a
  dense fragment of it → read as a short thick patch = block.
- #135 block → horizontal (93%): the wrong region; the heatmap was on a chance
  horizontal streak of noise below the small block.

Fixed one claim in the write-up after checking it against the data (baseline
seed 4's training loss ended at 0.65, not ≈0.8). `pytest -q` → 82 passed, 1 skipped.
