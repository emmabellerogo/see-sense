# AI Usage Record — See-Sense

- **Student:** Emma Rogoveanu

The Supabase SQL export was imported and analysis completed. The per-class cross-check reports zero difference.

Prepare the correction branch for review and deployment:
1. Verify SQL provenance now says Supabase SQL Editor, remove the local-replica warning only where justified, and confirm all expected rows and figures exist.
2. Run final tests and dependency checks.
3. Check the local five-tab app uses Fashion-MNIST run 54, shows ten labels/top-five probabilities, and loads final evidence and prediction-to-run history.
4. Check both individual reports exist, are distinct, and remain 750–1,000 words. Leave Komal’s personal reflection pending if she hasn’t supplied it.
5. Check README’s first 15 lines contain all required links or clearly marked missing video links.
6. Inspect git status and give exact meaningful commit batches, excluding secrets, downloaded dataset files, caches, and temporary pilot files. Include required small sample assets and evidence.

Do not retrain, commit, push, deploy, or change cloud records. Report actual checks and remaining blockers.

The full assignment text is available in this local attachment; read it directly:
/Users/emmarogoveanu/.codex/attachments/00f6abb4-a071-4477-a9e8-18ddda411e90/Pasted text.txt
Show less
3:54 PM
I'll start by reading the assignment and checking the repo state.



Read Pasted text.txt
Bash
IN
git status --short | sed -n '55,400p'; echo ---; cat .gitignore; ls

OUT
?? docs/CONTRIBUTIONS.md
?? docs/DATASET_CARD.md
?? docs/DATASET_CARD_HISTORICAL_SYNTHETIC.md
?? docs/EMMA_ENGINEERING_REPORT.md
?? docs/ETHICS_AND_DISCLOSURE.md
?? docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md
?? docs/FASHION_PROTOCOL.md
?? docs/INSTRUCTOR_QUESTION_DRAFT.md
?? docs/KOMAL_ENGINEERING_REPORT.md
?? docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md
?? docs/PROJECT_NOTES.md
?? docs/SUBMISSION_CHECKLIST.md
?? docs/fashion_official_1_export.sql
?? experiments/cam_blank_check.py
?? experiments/export_records.py
?? experiments/ingest_sql_export.py
?? experiments/migration_check.mjs
?? experiments/package_sql.py
?? experiments/pilot.py
?? experiments/pilot_summary.py
?? experiments/results/fashion-official-1-analysis/
?? experiments/results/fashion-official-1-gradcam/
?? experiments/results/fashion-official-1-records/
?? experiments/results/fashion-official-1-runs.jsonl
?? experiments/results/fashion-official-1-sql/
?? experiments/results/fashion-official-1-supabase-export.csv
?? experiments/results/fashion-pilot/
?? experiments/results/official-supabase-1-analysis/accuracy_curves.png
?? experiments/results/official-supabase-1-analysis/per_class_metrics.csv
?? pytest.ini
?? shared/fashion.py
?? tests/fashion_fixture.py
?? tests/fixtures/
?? tests/test_api_predict.py
?? tests/test_calibration.py
?? tests/test_evidence_pipeline.py
?? tests/test_fashion_data.py
?? tests/test_live_supabase.py
?? tests/test_protocol_config.py
?? tests/test_reference_prediction.py
?? tests/test_ui_app_smoke.py
?? tools/
?? ui/assets/
---
# Secrets — never commit these
.env
.env.*
!.env.example
**/.streamlit/secrets.toml
!**/.streamlit/secrets.toml.example

# Python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
*.egg-info/

# OS
.DS_Store

# Local offline database (SEE_SENSE_LOCAL_DB)
.local_db.json
.local_db.json.tmp

# Local dataset downloads and training artifacts
/data/
/pilot-output/
MODEL_CARD.md
PART2_RESULTS.md
README.md
ai-usage
api
config
data
db
docs
experiments
pytest.ini
render.yaml
requirements-dev.txt
shared
tests
tools
ui
Bash
IN
head -30 README.md; echo ----; wc -w docs/EMMA_ENGINEERING_REPORT.md docs/KOMAL_ENGINEERING_REPORT.md; cmp docs/EMMA_ENGINEERING_REPORT.md docs/KOMAL_ENGINEERING_REPORT.md; echo ---; grep -rniE "local.replica|local replica|SQL Editor|provenance" --include=*.md --include=*.py --include=*.json --include=*.sql . -l 2>/dev/null | grep -v node_modules | head -30; ls experiments/results/ experiments/results/fashion-official-1-* tools ui/assets config tests/fixtures

OUT
# See-Sense — Fashion-MNIST CNN Classifier with Grad-CAM (three clouds)

**Streamlit UI ⇄ FastAPI model API ⇄ Supabase data.** A small PyTorch CNN classifies 28×28 grayscale pictures of clothing into the
ten **Fashion-MNIST** categories (T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot), shows its
five most probable classes, and returns a **Grad-CAM** heatmap showing where the network looked. It is a teaching and evaluation
project, not a general photo classifier.

> **Status (draft, branch `correction/fashion-mnist`).** The code on this branch is the Fashion-MNIST version and the official experiment has
> been run (group `fashion-official-1`, 18 runs, ids 53-70). The apps currently deployed at the URLs below are still the *earlier
> synthetic-shape version* (from `main`) until this branch is deployed. The results below are provisional: the tables were produced by the
> project's SQL queries on a local copy of the saved records, and the Supabase SQL-editor export has not yet been run and compared. See
> [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md) and [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md).

## Live deployment URLs

| Tier | Platform | URL |
|------|----------|-----|
| UI | Streamlit Community Cloud | https://see-sense-glpk38qzwsf9hhvnlkt798.streamlit.app/ |
| API | Render.com | https://see-sense-api-rhkb.onrender.com *(free tier sleeps; first call takes ~30–60 s; health check: `/healthz`)* |
| Data | Supabase | `https://<your-project-ref>.supabase.co` *(placeholder: the project URL is not documented in this repository; fill it in from the Supabase dashboard)* |

## The five tabs (`ui/app.py`)

1. **Concepts** — convolution, stride/padding/output-size arithmetic with a worked example, receptive-field growth (three stacked 3×3 layers: 3 → 5 → 7), and the Grad-CAM derivation and limits, using this project's 10-class network (9,098 parameters for the baseline).
2. **Try It** — classify an uploaded image or one of 20 bundled *held-out* clothing examples (the first two official-test images of each class). Shows the prediction, **top-five probabilities**, the input and Grad-CAM, and whether the prediction was logged. The model comes from [config/served_model.json](config/served_model.json) (fixed in advance, checked against the saved experiment records and verified against the API) — never a hard-coded run ID. Handles cold starts and errors; prediction requests are never retried automatically.
3. **Compare Models** — the final Fashion-MNIST comparison (baseline, full improved and four leave-one-technique-out variants: tables, training/validation curves, confusion matrices, per-class precision/recall, calibration, Grad-CAM gallery) **once its evidence files exist** — until then the tab says plainly that they are not available. The earlier synthetic-shape experiment is shown separately in a clearly labelled historical section.
4. **Recent Classifications** — reads the latest ≤200 rows of the hash-free `recent_predictions` view with the anon key, shows a class-distribution histogram, and lets you **drill down from a classification to the run that produced it**.
5. **Model Card** — [MODEL_CARD.md](MODEL_CARD.md).

Training is not exposed in the UI. It is available through the API (`POST /train`, run from a local machine that has the dataset) and the experiment tools in `experiments/`.
----
     976 docs/EMMA_ENGINEERING_REPORT.md
     846 docs/KOMAL_ENGINEERING_REPORT.md
    1822 total
docs/EMMA_ENGINEERING_REPORT.md docs/KOMAL_ENGINEERING_REPORT.md differ: char 23, line 1
---
(eval):1: no matches found: --include=*.md
experiments/results/fashion-official-1-runs.jsonl
experiments/results/fashion-official-1-supabase-export.csv

config:
fashion_protocol.json
served_model.json

experiments/results/:
explore_budget.jsonl
explore_difficulty.jsonl
explore_difficulty2.jsonl
explore_techniques.jsonl
fashion-official-1-analysis
fashion-official-1-gradcam
fashion-official-1-records
fashion-official-1-runs.jsonl
fashion-official-1-sql
fashion-official-1-supabase-export.csv
fashion-pilot
gradcam
official-1-analysis
official-1-console.txt
official-1-runs.jsonl
official-1-sql
official-supabase-1-analysis
official-supabase-1-gradcam
official-supabase-1-runs.jsonl
official-supabase-1-sql

experiments/results/fashion-official-1-analysis:
accuracy_curves.png
calibration.csv
confusion_matrices.png
loss_curves.png
per_class_metrics.csv
per_class_precision_recall.png
summary.md
test_metrics_by_seed.png

experiments/results/fashion-official-1-gradcam:
blank_cam_by_class.csv
examples.csv
gradcam_correct.png
gradcam_incorrect.png
source.json

experiments/results/fashion-official-1-records:
runs.json

experiments/results/fashion-official-1-sql:
PROVENANCE.md
q1.csv
q10.csv
q2.csv
q3.csv
q4.csv
q5.csv
q6.csv
q7.csv
q9.csv

tests/fixtures:
__init__.py
__pycache__
make_reference.py
reference_expected.json
reference_weights.json

tools:
export_samples.py
fetch_fashion_mnist.py

ui/assets:
samples
Bash
IN
ls .venv venv 2>/dev/null | head -3; which python python3; cat requirements*.txt | head -40; ls api/requirements* ui/requirements* 2>/dev/null

OUT
.venv:
bin
etc
/Library/Frameworks/Python.framework/Versions/2.7/bin/python
/opt/homebrew/bin/python3
# Local development + running the test suite from the repo root.
# Installs both tiers plus pytest. On the actual clouds, each tier installs only
# its own requirements (api/requirements.txt or ui/requirements.txt).
-r api/requirements.txt
-r ui/requirements.txt
pytest==8.3.3
httpx==0.27.2
PyYAML==6.0.2
# Grad-CAM evidence figures (experiments/gradcam_evidence.py)
matplotlib==3.11.2
api/requirements.txt
ui/requirements.txt
Bash
IN
cd /Users/emmarogoveanu/Documents/GitHub/see-sense/experiments/results && cat fashion-official-1-sql/PROVENANCE.md; echo ----; head -40 fashion-official-1-analysis/summary.md; echo; ls -la fashion-official-1-*.* ; head -c 600 fashion-official-1-supabase-export.csv; echo; wc -l fashion-official-1-supabase-export.csv fashion-official-1-runs.jsonl; ls fashion-official-1-sql ../../docs/fashion_official_1_export.sql; head -20 ../../docs/fashion_official_1_export.sql | cut -c1-200

OUT
# SQL provenance

- source: **LOCAL REPLICA, NOT Supabase SQL.** The 18 run rows (ids 53-70) were read from Supabase with a read-only
  PostgREST call (`experiments/export_records.py`), loaded into a local in-memory Postgres engine (PGlite) built from
  `db/migrations/001,002,004,005`, and `docs/comparison_queries.sql` was executed there unchanged (`experiments/sql_check.mjs`).
- The queries are the real ones and the data are the real records, but this is NOT an export from the Supabase SQL editor.
  It is provisional: run `docs/fashion_official_1_export.sql` once in Supabase and ingest it
  (`experiments/ingest_sql_export.py`) to produce the SQL-sourced set in `../fashion-official-1-sql/`.
----
# SQL provenance

- source: **LOCAL REPLICA, NOT Supabase SQL.** The 18 run rows (ids 53-70) were read from Supabase with a read-only
  PostgREST call (`experiments/export_records.py`), loaded into a local in-memory Postgres engine (PGlite) built from
  `db/migrations/001,002,004,005`, and `docs/comparison_queries.sql` was executed there unchanged (`experiments/sql_check.mjs`).
- The queries are the real ones and the data are the real records, but this is NOT an export from the Supabase SQL editor.
  It is provisional: run `docs/fashion_official_1_export.sql` once in Supabase and ingest it
  (`experiments/ingest_sql_export.py`) to produce the SQL-sourced set in `../fashion-official-1-sql/`.

**Matched-configuration audit (query 1):** PASS - 3 seeds, 6 runs each, one dataset/split/budget per seed.

### Test results per variant (mean ± sd across seeds; from query 4)

| Variant | Seeds | Test accuracy | Macro-F1 | Test loss | Params |
|---|---|---|---|---|---|
| baseline | 3 | 89.73% ± 0.47 | 0.8973 ± 0.0047 | 0.2884 ± 0.0093 | 9098 |
| full_improved | 3 | 88.10% ± 0.29 | 0.8801 ± 0.0031 | 0.3327 ± 0.0057 | 9146 |
| no_augmentation | 3 | 90.30% ± 0.13 | 0.9024 ± 0.0015 | 0.2720 ± 0.0043 | 9146 |
| no_batchnorm | 3 | 87.12% ± 0.30 | 0.8701 ± 0.0031 | 0.3591 ± 0.0072 | 9098 |
| no_dropout | 3 | 88.94% ± 0.07 | 0.8888 ± 0.0006 | 0.3093 ± 0.0029 | 9146 |
| no_scheduler | 3 | 88.23% ± 0.40 | 0.8810 ± 0.0039 | 0.3274 ± 0.0094 | 9146 |

### Test accuracy by seed

| Variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| baseline | 90.01% | 89.19% | 89.98% |
| full_improved | 87.82% | 88.10% | 88.39% |
| no_augmentation | 90.29% | 90.44% | 90.18% |
| no_batchnorm | 86.89% | 87.02% | 87.46% |
| no_dropout | 88.97% | 88.86% | 88.98% |
| no_scheduler | 87.77% | 88.50% | 88.41% |

### Difference from full_improved, same seed (from query 6)

Negative accuracy delta = scored lower than the full model. Accuracy deltas in percentage points (pp).

| Variant | Technique removed | Δ accuracy per seed (pp) | Mean Δ acc (pp) | 95% CI of mean Δ (pp, paired t) | Mean Δ macro-F1 | Mean Δ test loss | Seeds lower than full |
|---|---|---|---|---|---|---|---|
| baseline | architecture + all four | +2.19, +1.09, +1.59 | +1.62 | [+0.26, +2.99] | +0.0172 | -0.0443 | 0 of 3 |

-rw-r--r--  1 emmarogoveanu  staff    1884 Oct 17 15:18 fashion-official-1-runs.jsonl
-rw-r--r--  1 emmarogoveanu  staff  355369 Oct 17 15:51 fashion-official-1-supabase-export.csv
query_id,rows
q1,"[{""train_seed"":0,""n_runs"":6,""datasets"":1,""split_seeds"":1,""split_hashes"":1,""split_sizes"":1,""dataset_names"":1,""val_sizes"":1,""train_limits"":1,""preprocessings"":1,""epochs"":1,""batch_sizes"":1,""initial_lrs"":1,""optimizers"":1,""max_shifts"":1},{""train_seed"":1,""n_runs"":6,""datasets"":1,""split_seeds"":1,""split_hashes"":1,""split_sizes"":1,""dataset_names"":1,""val_sizes"":1,""train_limits"":1,""preprocessings"":1,""epochs"":1,""batch_sizes"":1,""initial_lrs"":1,""optimizers"":1,""max_shifts"":1},{""train_seed"":2,""n_runs"":6,""datasets"":1,""split_seeds
      11 fashion-official-1-supabase-export.csv
      18 fashion-official-1-runs.jsonl
      29 total
../../docs/fashion_official_1_export.sql

fashion-official-1-sql:
PROVENANCE.md
q1.csv
q10.csv
q2.csv
q3.csv
q4.csv
q5.csv
q6.csv
q7.csv
q9.csv
-- Packaged read-only export for experiment group 'fashion-official-1'.
-- Source: docs/comparison_queries.sql (sha256 5d31a4fe21db31b8c9f4e5d8731cb7e369532b9b026307caf71661b4c4741385); generated by experiments/package_sql.py.
-- Run ONCE in the Supabase SQL editor, then download the result (CSV or JSON).
select 'q1' as query_id, coalesce(json_agg(to_json(t)), '[]'::json) as rows from (
select train_seed,
       count(*)                                  as n_runs,
       count(distinct dataset_id)                as datasets,
       count(distinct split_seed)                as split_seeds,
       count(distinct split_sha256)              as split_hashes,
       count(distinct (n_train, n_val, n_test))  as split_sizes,
       count(distinct config ->> 'dataset')      as dataset_names,   -- fashion-mnist (NULL on historical synthetic runs)
       count(distinct config ->> 'val_size')     as val_sizes,
       count(distinct coalesce(config ->> 'train_limit', 'all')) as train_limits,
       count(distinct config -> 'preprocessing') as preprocessings,
       count(distinct epochs)                    as epochs,
       count(distinct batch_size)                as batch_sizes,
       count(distinct lr)                        as initial_lrs,
       count(distinct optimizer)                 as optimizers,
       count(distinct augment_max_shift)         as max_shifts
from runs
Bash
IN
ls experiments/*.py experiments/*.mjs; sed -n 1,60p experiments/ingest_sql_export.py; ls -la experiments/results/fashion-official-1-analysis experiments/results/fashion-official-1-sql

OUT
experiments/__init__.py
experiments/analyze.py
experiments/cam_blank_check.py
experiments/configs.py
experiments/explore.py
experiments/export_records.py
experiments/gradcam_evidence.py
experiments/ingest_sql_export.py
experiments/migration_check.mjs
experiments/package_sql.py
experiments/pilot.py
experiments/pilot_summary.py
experiments/run_experiments.py
experiments/sql_check.mjs
"""Turn the packaged SQL export (docs/fashion_official_1_export.sql, run once in Supabase) into q*.csv files.

    python -m experiments.ingest_sql_export <export.csv|export.json> <out_dir> [--source supabase-sql-editor]

Accepts the SQL editor's CSV download (columns query_id, rows) or "copy as JSON" / the JSON download.
Writes <out_dir>/q1.csv ... plus PROVENANCE.md (sha256 of the export, the executed_at / server_version / SQL
file hash reported by the database itself, row counts). Refuses to run without the ``meta`` row, because
without it nothing shows that the numbers came from a SQL execution.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

csv.field_size_limit(1 << 30)


def load(path: Path) -> dict[str, list[dict]]:
    raw = path.read_text()
    if raw.lstrip().startswith(("[", "{")):
        data = json.loads(raw)
        rows = data if isinstance(data, list) else data.get("rows", data)
    else:
        rows = list(csv.DictReader(raw.splitlines(keepends=True)))
    out = {}
    for r in rows:
        v = r["rows"]
        out[r["query_id"]] = json.loads(v) if isinstance(v, str) else v
    return out


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def main(export: str, out_dir: str, source: str = "unspecified") -> None:
    src = Path(export)
    res = load(src)
    if "meta" not in res:
        raise SystemExit("export has no 'meta' row: not produced by the packaged SQL (or truncated)")
    meta = res["meta"][0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for q, rows in sorted(res.items()):
        if q == "meta":
            continue
        cols = list(rows[0]) if rows else []
        with open(out / f"{q}.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(cols)
experiments/results/fashion-official-1-analysis:
total 1824
drwxr-xr-x  10 emmarogoveanu  staff     320 Oct 17 15:29 .
drwxr-xr-x@ 22 emmarogoveanu  staff     704 Oct 17 15:51 ..
-rw-r--r--   1 emmarogoveanu  staff  188574 Oct 17 15:53 accuracy_curves.png
-rw-r--r--   1 emmarogoveanu  staff    3059 Oct 17 15:53 calibration.csv
-rw-r--r--   1 emmarogoveanu  staff  372488 Oct 17 15:53 confusion_matrices.png
-rw-r--r--   1 emmarogoveanu  staff  176383 Oct 17 15:53 loss_curves.png
-rw-r--r--   1 emmarogoveanu  staff    3148 Oct 17 15:53 per_class_metrics.csv
-rw-r--r--   1 emmarogoveanu  staff  116309 Oct 17 15:53 per_class_precision_recall.png
-rw-r--r--   1 emmarogoveanu  staff    9962 Oct 17 15:53 summary.md
-rw-r--r--   1 emmarogoveanu  staff   47468 Oct 17 15:53 test_metrics_by_seed.png

experiments/results/fashion-official-1-sql:
total 312
drwxr-xr-x  12 emmarogoveanu  staff    384 Oct 17 15:26 .
drwxr-xr-x@ 22 emmarogoveanu  staff    704 Oct 17 15:51 ..
-rw-r--r--   1 emmarogoveanu  staff    700 Oct 17 15:25 PROVENANCE.md
-rw-r--r--   1 emmarogoveanu  staff    259 Oct 17 15:24 q1.csv
-rw-r--r--   1 emmarogoveanu  staff  14249 Oct 17 15:24 q10.csv
-rw-r--r--   1 emmarogoveanu  staff   2352 Oct 17 15:24 q2.csv
-rw-r--r--   1 emmarogoveanu  staff  41669 Oct 17 15:24 q3.csv
-rw-r--r--   1 emmarogoveanu  staff   1665 Oct 17 15:24 q4.csv
-rw-r--r--   1 emmarogoveanu  staff    456 Oct 17 15:24 q5.csv
-rw-r--r--   1 emmarogoveanu  staff    641 Oct 17 15:24 q6.csv
-rw-r--r--   1 emmarogoveanu  staff  69022 Oct 17 15:24 q7.csv
-rw-r--r--   1 emmarogoveanu  staff   3040 Oct 17 15:24 q9.csv
Bash
IN
cd /Users/emmarogoveanu/Documents/GitHub/see-sense && python3 -I - <<'E'
import csv,json,sys
csv.field_size_limit(1<<30)
rows=list(csv.DictReader(open('experiments/results/fashion-official-1-supabase-export.csv')))
for r in rows:
    v=json.loads(r['rows']); print(r['query_id'],len(v), (v[0] if r['query_id']=='meta' else ''))
E
find . -name PROVENANCE.md -not -path './.venv/*' -exec ls -la {} \; ; grep -rn "ingest_sql_export\|supabase-export" --include=*.md --include=*.py . 2>/dev/null | grep -v .venv | head -20

OUT
q1 3
q2 18
q3 360
q4 18
q5 6
q6 15
q7 1800
q8 5
q9 18
q10 180
meta 1 {'executed_at': '2026-10-17T22:49:57.518155+00:00', 'server_version': 'PostgreSQL 17.11 on aarch64-unknown-linux-gnu, compiled by gcc (GCC) 15.2.0, 64-bit', 'database': 'postgres', 'experiment_group': 'fashion-official-1', 'queries_file': 'docs/comparison_queries.sql', 'queries_file_sha256': '5d31a4fe21db31b8c9f4e5d8731cb7e369532b9b026307caf71661b4c4741385'}
-rw-r--r--  1 emmarogoveanu  staff  700 Oct 17 15:25 ./experiments/results/fashion-official-1-sql/PROVENANCE.md
(eval):8: no matches found: --include=*.md
The export looks genuine, but the committed fashion-official-1-sql/ still has the older local-replica files from 15:24. I'll ingest the new export into a scratch directory and diff it against what's there.
Bash
IN
S=/private/tmp/claude-501/-Users-emmarogoveanu-Documents-GitHub-see-sense/89f94cee-b0d6-48f7-817c-e0a7fcc377ca/scratchpad && source .venv/bin/activate && python -m experiments.ingest_sql_export experiments/results/fashion-official-1-supabase-export.csv $S/sqlnew --source supabase-sql-editor && cat $S/sqlnew/PROVENANCE.md && for q in 1 2 3 4 5 6 7 9 10; do cmp -s $S/sqlnew/q$q.csv experiments/results/fashion-official-1-sql/q$q.csv && echo q$q same || echo q$q DIFF; done; ls $S/sqlnew; sed -n 1,50p experiments/analyze.py | head -60; grep -n "argv\|argparse\|add_argument" experiments/analyze.py | head

OUT
wrote 10 csv files to /private/tmp/claude-501/-Users-emmarogoveanu-Documents-GitHub-see-sense/89f94cee-b0d6-48f7-817c-e0a7fcc377ca/scratchpad/sqlnew
# SQL provenance

- source: **supabase-sql-editor**
- export file: `fashion-official-1-supabase-export.csv` sha256 `474ee3db5f915e89cef75560f7e231b39ecd800a6f20347908831f9271d42c20`
- executed_at (reported by the database): 2026-10-17T22:49:57.518155+00:00
- server_version: PostgreSQL 17.11 on aarch64-unknown-linux-gnu, compiled by gcc (GCC) 15.2.0, 64-bit
- database: postgres
- experiment_group: fashion-official-1
- queries file: docs/comparison_queries.sql sha256 `5d31a4fe21db31b8c9f4e5d8731cb7e369532b9b026307caf71661b4c4741385`

| query | rows |
|---|---|
| q1 | 3 |
| q10 | 180 |
| q2 | 18 |
| q3 | 360 |
| q4 | 18 |
| q5 | 6 |
| q6 | 15 |
| q7 | 1800 |
| q8 | 5 |
| q9 | 18 |
q1 same
q2 DIFF
q3 DIFF
q4 DIFF
q5 DIFF
q6 DIFF
q7 same
q9 DIFF
q10 DIFF
PROVENANCE.md
q1.csv
q10.csv
q2.csv
q3.csv
q4.csv
q5.csv
q6.csv
q7.csv
q8.csv
q9.csv
"""Turn the SQL query results (docs/comparison_queries.sql) into tables and figures.

Input: a folder with q1.csv ... q7.csv, exactly as the queries return them
(from the Supabase SQL editor, or from experiments/sql_check.mjs locally).

    python -m experiments.analyze <sql_dir> <out_dir> [--classes fashion|legacy-synthetic]

Writes to <out_dir>:
    summary.md                   mean +/- sd per variant, per-technique deltas, audit
    loss_curves.png              train/val loss per variant (mean over seeds, band = min..max)
    accuracy_curves.png          validation accuracy per variant (+ training accuracy when q3 has it)
    per_class_metrics.csv        per-class test precision / recall per variant (from query 7)
    calibration.csv              per-run temperature and ECE / NLL before and after (query 9; optional)
    confusion_matrices.png       summed confusion matrix per variant
    test_metrics_by_seed.png     test accuracy per seed per variant
"""
from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Two-sided 95% t critical values by degrees of freedom (n_seeds - 1).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

VARIANTS = ["baseline", "full_improved", "no_augmentation", "no_batchnorm", "no_dropout", "no_scheduler"]
# Class order of the two datasets this tool can analyse. Fashion-MNIST is the project's dataset;
# the four synthetic shapes exist only to regenerate the historical official-supabase-1 outputs.
FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
LEGACY_SYNTHETIC_CLASSES = ["horizontal", "vertical", "diagonal", "block"]
CLASS_SETS = {"fashion": FASHION_CLASSES, "legacy-synthetic": LEGACY_SYNTHETIC_CLASSES}
TECHNIQUE = {
    "no_augmentation": "augmentation", "no_batchnorm": "batch norm",
    "no_dropout": "dropout", "no_scheduler": "LR scheduler",
}


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

388:    args = [a for a in sys.argv[1:] if not a.startswith("--classes")]
390:    if "--classes" in sys.argv:
391:        which = sys.argv[sys.argv.index("--classes") + 1]
Bash
IN
S=/private/tmp/claude-501/-Users-emmarogoveanu-Documents-GitHub-see-sense/89f94cee-b0d6-48f7-817c-e0a7fcc377ca/scratchpad && python3 -I - $S <<'E'
import csv,sys
S=sys.argv[1]
def num(x):
    try: return round(float(x),6)
    except: return x.strip()
for q in [1,2,3,4,5,6,7,9,10]:
    a=list(csv.reader(open(f'{S}/sqlnew/q{q}.csv'))); b=list(csv.reader(open(f'experiments/results/fashion-official-1-sql/q{q}.csv')))
    if a[0]!=b[0]: print(q,'header diff',a[0],b[0]); continue
    bad=0;ex=None
    for r1,r2 in zip(a,b):
        for x,y in zip(r1,r2):
            if num(x)!=num(y):
                bad+=1; ex=ex or (x[:50],y[:50])
    print(q,len(a),len(b),'cell diffs',bad,ex)
E
git diff --stat experiments/results/official-supabase-1-analysis/summary.md | tail -2

OUT
1 4 4 cell diffs 0 None
2 19 19 cell diffs 0 None
3 361 361 cell diffs 0 None
4 19 19 cell diffs 0 None
5 7 7 cell diffs 0 None
6 16 16 cell diffs 0 None
7 1801 1801 cell diffs 0 None
9 19 19 cell diffs 0 None
10 181 181 cell diffs 0 None
 .../official-supabase-1-analysis/summary.md        | 31 ++++++++++++++++++++++
 1 file changed, 31 insertions(+)
The Supabase export matches the local replica data (just a formatting difference), so I'm swapping in the Supabase-sourced CSVs locally and regenerating the analysis.
Bash
IN
S=/private/tmp/claude-501/-Users-emmarogoveanu-Documents-GitHub-see-sense/89f94cee-b0d6-48f7-817c-e0a7fcc377ca/scratchpad && source .venv/bin/activate && mkdir -p $S/oldsql && cp experiments/results/fashion-official-1-sql/* $S/oldsql/ && cp $S/sqlnew/* experiments/results/fashion-official-1-sql/ && python -m experiments.analyze experiments/results/fashion-official-1-sql experiments/results/fashion-official-1-analysis --classes fashion 2>&1 | tail -5; head -12 experiments/results/fashion-official-1-analysis/summary.md; grep -rniE "local replica|local copy|provisional|not yet been run|PGlite" --include=*.md --include=*.py --include=*.json . 2>/dev/null | grep -v "^./.venv" | cut -c1-220

OUT
| no_augmentation | 0.935 | 0.0087 | 0.0068 | 0.2720 | 0.2726 |
| no_batchnorm | 0.884 | 0.0190 | 0.0079 | 0.3591 | 0.3586 |
| no_dropout | 0.953 | 0.0073 | 0.0082 | 0.3093 | 0.3100 |
| no_scheduler | 0.917 | 0.0108 | 0.0071 | 0.3274 | 0.3277 |

# SQL provenance

- source: **supabase-sql-editor**
- export file: `fashion-official-1-supabase-export.csv` sha256 `474ee3db5f915e89cef75560f7e231b39ecd800a6f20347908831f9271d42c20`
- executed_at (reported by the database): 2026-10-17T22:49:57.518155+00:00
- server_version: PostgreSQL 17.11 on aarch64-unknown-linux-gnu, compiled by gcc (GCC) 15.2.0, 64-bit
- database: postgres
- experiment_group: fashion-official-1
- queries file: docs/comparison_queries.sql sha256 `5d31a4fe21db31b8c9f4e5d8731cb7e369532b9b026307caf71661b4c4741385`

| query | rows |
|---|---|
(eval):1: no matches found: --include=*.md
Bash
IN
grep -rniE "local replica|local copy|provisional|not yet been run|PGlite|replica" . --exclude-dir=.venv --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=__pycache__ -I 2>/dev/null | grep -v "\.csv:\|\.jsonl:\|\.json:" | cut -c1-260

OUT
README.md:10:> synthetic-shape version* (from `main`) until this branch is deployed. The results below are provisional: the tables were produced by the
README.md:11:> project's SQL queries on a local copy of the saved records, and the Supabase SQL-editor export has not yet been run and compared. See
MODEL_CARD.md:4:> They are real saved run records, but the tables were produced by running the project's SQL queries on a **local copy**
MODEL_CARD.md:85:has not been run yet. Treat the numbers as provisional until it has and has been compared.
PART2_RESULTS.md:4:> queries run on a local copy of those records; the Supabase SQL-editor export is still to be run and compared (see Deviations).
PART2_RESULTS.md:59:- The SQL provenance is a **local replica**, not a Supabase SQL export; this report is provisional until `docs/fashion_official_1_export.sql`
ui/app.py:655:                "were produced by the project's queries on a local copy, not by the Supabase SQL editor. They are "
ui/app.py:656:                "provisional until the packaged SQL export has been run once in Supabase and ingested."
experiments/migration_check.mjs:1:// Applies db/migrations/001-005 to a LOCAL in-memory Postgres (PGlite) and checks what the migrations promise:
experiments/migration_check.mjs:5:// Needs `npm install @electric-sql/pglite@0.2.17` in some scratch folder; copy this file there (node resolves
experiments/migration_check.mjs:8:// Last run: 2026-10-17, PGlite 0.2.17, all checks printed "ok"/expected results (see docs/SUBMISSION_CHECKLIST.md).
experiments/migration_check.mjs:9:import { PGlite } from '@electric-sql/pglite';
experiments/migration_check.mjs:12:const pg = new PGlite();
experiments/export_records.py:11:replica (experiments/sql_check.mjs) to check the queries before the real SQL export is run in Supabase.
experiments/sql_check.mjs:1:// Run docs/comparison_queries.sql verbatim on a local Postgres engine (PGlite).
experiments/sql_check.mjs:11:// Needs:  npm install @electric-sql/pglite@0.2.17  in some folder, then copy this file
experiments/sql_check.mjs:14:import { PGlite } from '@electric-sql/pglite';
experiments/sql_check.mjs:20:// SEE_SENSE_ROOT lets you run a copy of this file from the folder where PGlite is installed.
experiments/sql_check.mjs:24:const pg = new PGlite();
tests/test_evidence_pipeline.py:100:        for text, expect in [("- source: **supabase-sql-editor**", True), ("- source: **LOCAL REPLICA, NOT Supabase SQL.**", False)]:
docs/PROJECT_NOTES.md:24:- Official Fashion-MNIST results now exist (group `fashion-official-1`, runs 53-70, 3 seeds): the improved model did not beat the baseline, and the reports say so. SQL provenance is still a local replica until the Supabase export is ru
docs/PART1_HANDOFF.md:145:| Migrations 001 + 002 on a real Postgres engine (PGlite, local) | 002 applies twice cleanly; an old row stays readable with NULLs; duplicate name, bad variant, short history, bad dropout and non-object config are rejected; RLS still
docs/PART1_HANDOFF.md:149:The local engine is PGlite (Postgres compiled to WebAssembly), not Supabase
docs/FASHION_PROTOCOL.md:91:     and `python -m experiments.analyze experiments/results/fashion-official-1-sql experiments/results/fashion-official-1-analysis`. Until this is done, the CSVs in that folder are labelled in their `PROVENANCE.md` as a **local repl
docs/PART2_RESULTS.md:4:> queries run on a local copy of those records; the Supabase SQL-editor export is still to be run and compared (see Deviations).
docs/PART2_RESULTS.md:59:- The SQL provenance is a **local replica**, not a Supabase SQL export; this report is provisional until `docs/fashion_official_1_export.sql`
docs/KOMAL_ENGINEERING_REPORT.md:15:**Check numbers two ways.** The comparison queries run unchanged on a local Postgres engine, and I recomputed the summary in Python independently. On the earlier synthetic experiment this gave zero mismatches. The same disci
docs/KOMAL_ENGINEERING_REPORT.md:34:- Compute provenance from Supabase from the start, not a local replica.
docs/EMMA_ENGINEERING_REPORT.md:3:> **DRAFT.** Team numbers are the official saved records (group `fashion-official-1`, runs 53–70); the SQL tables came from a local replica until the Supabase export is run (see Limits).
docs/EMMA_ENGINEERING_REPORT.md:29:- The SQL provenance is a local replica. The Supabase export is prepared but not yet run, so these numbers are provisional until it matches.
docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md:17:> (PGlite) with migrations 001 + 002 applied (`experiments/sql_check.mjs`). They
docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md:204:# SQL on local Postgres (needs: npm install @electric-sql/pglite@0.2.17 in some folder,
docs/SUBMISSION_CHECKLIST.md:19:| 4 | Headline numbers sourced via SQL | 🟡 | Tables come from `docs/comparison_queries.sql` (queries 1-7, 9, 10) run on a **local Postgres replica** of the 18 records, labelled as such in `fashion-official-1-sql/PROVENANCE.md`.
docs/SUBMISSION_CHECKLIST.md:49:- **Local Postgres engine (PGlite)**: migrations 001-005 apply and re-apply, `log_prediction` is atomic, anon reads only the hash-free view; the comparison queries and the packaged export run there. This is **not** Supabase.
ai-usage/Komal-Session2.md:15:  - `experiments/sql_check.mjs` — **new**, runs `docs/comparison_queries.sql` unchanged on local Postgres (PGlite)
ai-usage/Komal-Session2.md:49:a local Postgres engine (PGlite) with migrations 001 + 002 applied.
ai-usage/Emma-Session1.md:3866:The SQL is syntactically valid, but I need to verify the queries actually return correct results. I'll check if a real Postgres engine like PGlite (Postgres compiled to WASM) is available locally for that.
ai-usage/Emma-Session1.md:3875:Bash Install PGlite in the scratch directory
ai-usage/Emma-Session1.md:3877:S=/private/tmp/claude-501/-Users-emmarogoveanu-Documents-GitHub-see-sense/7de1d29b-a753-4876-952b-bd5f326d9891/scratchpad; mkdir -p $S/pg && cd $S/pg && npm init -y >/dev/null 2>&1 && npm install --silent @electric-sql/pglite 2>&
ai-usage/Emma-Session1.md:3880:pglite
ai-usage/Emma-Session1.md:3883:PGlite installed. I'll generate realistic rows from real training runs, load them into a real Postgres engine after applying migrations 001 and 002, and execute the queries.
ai-usage/Emma-Session1.md:3909:import { PGlite } from '@electric-sql/pglite';
ai-usage/Emma-Session1.md:3913:const db=new PGlite();
ai-usage/Emma-Session1.md:4184:| Migrations 001 + 002 on a real Postgres engine (PGlite, local) | 002 applies twice cleanly; an old row stays readable with NULLs; duplicate name, bad variant, short history, bad dropout and non-object config are rejected; RLS s
ai-usage/Emma-Session1.md:4188:The local engine is PGlite (Postgres compiled to WebAssembly), not Supabase
ai-usage/Emma-Session1.md:4334:Real Postgres engine: I installed PGlite in the scratch folder, not the repo. It's Postgres compiled to WebAssembly. I applied migrations 001 and 002 on it, with 002 applied twice, and loaded 12 rows generated by real training ru
3:55 PM
Several documents still carry the "provisional / local replica" wording. Next I'll read each one and the UI banner logic.



Bash
IN
sed -n 640,665p ui/app.py; sed -n 85,115p tests/test_evidence_pipeline.py; sed -n 1,8p MODEL_CARD.md; sed -n 82,87p MODEL_CARD.md; sed -n 1,6p PART2_RESULTS.md; sed -n 55,62p PART2_RESULTS.md; diff -q PART2_RESULTS.md docs/PART2_RESULTS.md; sed -n 22,26p docs/PROJECT_NOTES.md; sed -n 89,93p docs/FASHION_PROTOCOL.md; sed -n 17,21p docs/SUBMISSION_CHECKLIST.md; sed -n 47,50p docs/SUBMISSION_CHECKLIST.md; grep -rn "cross-check\|crosscheck\|cross_check" --include=*.py --include=*.md . --exclude-dir=.venv --exclude-dir=ai-usage | head

OUT
    st.header("Compare Models")
    fm = evidence.FASHION
    st.subheader(fm.title)
    if fm.available():
        prov = evidence.load_provenance(fm.sql_dir)
        st.info(
            f"**Saved experiment results** — group `{fm.group}`. Tables are read from CSV files saved in "
            f"`experiments/results/{fm.group}-sql/`; figures were generated from them. This tab displays files "
            "saved in the repository; it does not re-query Supabase or retrain."
        )
        if prov["is_supabase_sql"]:
            st.caption(f"SQL provenance: **{prov['source']}** (see PROVENANCE.md next to the CSVs).")
        else:
            st.warning(
                f"**SQL provenance: {prov['source'].rstrip('.')}.** These numbers are the real saved run records, but the CSVs "
                "were produced by the project's queries on a local copy, not by the Supabase SQL editor. They are "
                "provisional until the packaged SQL export has been run once in Supabase and ingested."
            )
        _render_evidence_set(fm, historical=False)
    else:
        st.warning(
            "**Final Fashion-MNIST results are not available yet.** The official 10-class comparison has not "
            "been completed, so no numbers, curves or Grad-CAM examples are shown for it here. Nothing below "
            "this section describes Fashion-MNIST."
        )

        out = tmp_path / "o"
        ingest_sql_export.main(str(self._export(tmp_path, as_csv)), str(out), "supabase-sql-editor")
        assert list(csv.DictReader(open(out / "q4.csv"))) == [{"run_name": "baseline-seed0", "flag": "true", "x": "", "n": "1.5"}]
        prov = (out / "PROVENANCE.md").read_text()
        assert "supabase-sql-editor" in prov and "PostgreSQL 15" in prov

    def test_an_export_without_the_database_meta_row_is_refused(self, tmp_path):
        p = tmp_path / "e.json"
        p.write_text(json.dumps([{"query_id": "q4", "rows": []}]))
        with pytest.raises(SystemExit):
            ingest_sql_export.main(str(p), str(tmp_path / "o"))


class TestProvenanceLabel:
    def test_only_a_supabase_source_counts_as_supabase_sql(self, tmp_path):
        for text, expect in [("- source: **supabase-sql-editor**", True), ("- source: **LOCAL REPLICA, NOT Supabase SQL.**", False)]:
            (tmp_path / "PROVENANCE.md").write_text(text)
            assert evidence.load_provenance(tmp_path)["is_supabase_sql"] is expect
        assert evidence.load_provenance(tmp_path / "nope")["is_supabase_sql"] is False
# Model Card — See-Sense (Fashion-MNIST)

> **Status: DRAFT.** Official Fashion-MNIST results (group `fashion-official-1`, 18 runs, ids 53-70) are reported below.
> They are real saved run records, but the tables were produced by running the project's SQL queries on a **local copy**
> of those records, not yet by the Supabase SQL editor (see *Results*). The earlier four-class *synthetic-shape* experiment
> is documented separately under **Historical** at the end; it is not Fashion-MNIST and must not be read as Fashion-MNIST performance.

## Summary
declared technique). Source of the numbers: SQL queries 1-7, 9 and 10 of `docs/comparison_queries.sql` over the 18 saved run records.
**Provenance caveat:** the CSVs in `experiments/results/fashion-official-1-sql/` were produced by running those queries unchanged on a
local Postgres engine loaded with the 18 records read from Supabase; the Supabase SQL-editor export (`docs/fashion_official_1_export.sql`)
has not been run yet. Treat the numbers as provisional until it has and has been compared.

| Variant | Test accuracy (mean ± sd, 3 seeds) | Macro-F1 | Test loss |
# Part 2 results: baseline vs improved CNN on Fashion-MNIST

> **DRAFT.** Numbers are the official saved run records (group `fashion-official-1`, runs 53–70). The tables come from the project's SQL
> queries run on a local copy of those records; the Supabase SQL-editor export is still to be run and compared (see Deviations).
> The earlier synthetic-shape experiment is kept separately in [docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md](docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md).

without hashes. Try It shows the top five probabilities and Grad-CAM, and loads the served checkpoint only after checking its group, dataset,
labels and preprocessing. Automated tests passed (mocks and a local Postgres engine); live Supabase checks passed on a small test run.

## Deviations and unresolved items
- The SQL provenance is a **local replica**, not a Supabase SQL export; this report is provisional until `docs/fashion_official_1_export.sql`
  has been run once and its output matches.
- Three seeds, not five, because of compute; intervals are rough. All runs came from one machine.
- The dataset was corrected from the starter's synthetic shapes to Fashion-MNIST late; the earlier results are preserved and labelled.
Files PART2_RESULTS.md and docs/PART2_RESULTS.md differ

**Not done / not claimed.**
- Official Fashion-MNIST results now exist (group `fashion-official-1`, runs 53-70, 3 seeds): the improved model did not beat the baseline, and the reports say so. SQL provenance is still a local replica until the Supabase export is run.
- Nothing was committed, pushed, deployed or applied to Supabase by the correction session. Migrations 004 and 005 are prepared, not applied.
- No instructor approval has been obtained for anything. A question is drafted in `docs/INSTRUCTOR_QUESTION_DRAFT.md` and has **not** been sent.
   - **SQL export (once, in the Supabase SQL editor):** run `docs/fashion_official_1_export.sql` (generated by `python -m experiments.package_sql fashion-official-1 docs/fashion_official_1_export.sql`; read-only; one result row per query plus a `meta` row). Download the result as CSV (or copy as JSON), then
     `python -m experiments.ingest_sql_export <downloaded file> experiments/results/fashion-official-1-sql --source supabase-sql-editor`
     and `python -m experiments.analyze experiments/results/fashion-official-1-sql experiments/results/fashion-official-1-analysis`. Until this is done, the CSVs in that folder are labelled in their `PROVENANCE.md` as a **local replica** (same queries, same records read via the API), and every report says so.
   - Read-only verification of the saved records: `python -m experiments.export_records fashion-official-1 experiments/results/fashion-official-1-records`.
7. Grad-CAM evidence (done for run 54): `python -m experiments.gradcam_evidence --supabase full_improved-seed0 fashion-official-1 experiments/results/fashion-official-1-gradcam` and `python -m experiments.cam_blank_check --supabase full_improved-seed0 fashion-official-1 experiments/results/fashion-official-1-gradcam/blank_cam_by_class.csv`; the two failure interpretations are a **draft** in `docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md` for a person to review.
| 2 | ≥3 techniques with controlled ablations | ✅ | Four leave-one-out variants with paired intervals ([summary.md](../experiments/results/fashion-official-1-analysis/summary.md)). Only batch norm helped; augmentation and dropout hurt at 20 epochs. Three seeds: wide intervals, disclosed. |
| 3 | Runs persisted; train **and** validation loss/accuracy every epoch; test accuracy, loss, macro-F1, confusion matrix, per-class precision/recall | ✅ | Verified on all 18 stored runs (20-epoch histories, 10×10 confusion, per-class arrays, calibration). Migrations 004/005 applied by Emma; columns confirmed. |
| 4 | Headline numbers sourced via SQL | 🟡 | Tables come from `docs/comparison_queries.sql` (queries 1-7, 9, 10) run on a **local Postgres replica** of the 18 records, labelled as such in `fashion-official-1-sql/PROVENANCE.md`. ⏳ Run `docs/fashion_official_1_export.sql` once in Supabase, ingest it, re-run `analyze`, and compare (see Remaining steps). |
| 5 | Grad-CAM: ≥5 correct, ≥5 genuinely incorrect, in the dashboard, + two evidence-based failure interpretations | 🟡 | 10 correct + 10 incorrect from run 54 (regenerated accuracy 87.82% = stored); shown in the Compare tab. Two interpretations are **drafted for human review** in [FASHION_GRADCAM_FAILURE_ANALYSIS.md](FASHION_GRADCAM_FAILURE_ANALYSIS.md). ⏳ Emma/Komal must read the figures and confirm or rewrite them. Note: Grad-CAM is blank for 97.8% of Dress and 87.6% of Coat predictions. |
| 6 | Calibrated probabilities | ✅ (measured) | Validation-fitted temperature; test ECE fell for 5 of 6 variants (full_improved 0.0173 → 0.0069), NLL ≈ unchanged; small gains, stated as such. |
## Tests (what each kind proves)
- **Mocked / in-memory (default `pytest`)**: code paths with a fake database and a tiny Fashion-shaped dataset. They prove nothing about Supabase.
- **Local Postgres engine (PGlite)**: migrations 001-005 apply and re-apply, `log_prediction` is atomic, anon reads only the hash-free view; the comparison queries and the packaged export run there. This is **not** Supabase.
- **Live Supabase (`SEE_SENSE_RUN_LIVE_TESTS=1`)**: `tests/test_live_supabase.py` — passed on 2026-10-17 (writes one small dataset row, one 1-epoch run in group `live-test`, two predictions + two image_metadata rows; nothing deleted).
(eval):1: no matches found: --include=*.py
3:55 PM
I'll update the stale provenance wording in each place. First the UI banner is data-driven, so it will switch on its own now that PROVENANCE.md says supabase-sql-editor.



Bash
IN
python3 -I - <<'E'
import re,pathlib
def sub(p,old,new):
    f=pathlib.Path(p); t=f.read_text()
    assert old in t,(p,old[:40])
    f.write_text(t.replace(old,new,1))
sub('README.md',"""The results below are provisional: the tables were produced by the
> project's SQL queries on a local copy of the saved records, and the Supabase SQL-editor export has not yet been run and compared.""",
"""The result tables were produced by the
> project's SQL queries run in the Supabase SQL editor (export ingested 2026-10-17; see `experiments/results/fashion-official-1-sql/PROVENANCE.md`).""")
sub('MODEL_CARD.md',"""They are real saved run records, but the tables were produced by running the project's SQL queries on a **local copy**
> of those records, not yet by the Supabase SQL editor (see *Results*).""","""They are real saved run records, and the tables were produced by running the project's SQL queries in the **Supabase SQL editor**
> (export ingested 2026-10-17; see *Results*).""")
sub('MODEL_CARD.md',"""**Provenance caveat:** the CSVs in `experiments/results/fashion-official-1-sql/` were produced by running those queries unchanged on a
local Postgres engine loaded with the 18 records read from Supabase; the Supabase SQL-editor export (`docs/fashion_official_1_export.sql`)
has not been run yet. Treat the numbers as provisional until it has and has been compared.""","""**Provenance:** the CSVs in `experiments/results/fashion-official-1-sql/` come from the Supabase SQL-editor export of
`docs/fashion_official_1_export.sql` (PostgreSQL 17.11, executed 2026-10-17T22:49:57Z, query-file hash recorded in `PROVENANCE.md`). They matched an earlier
local-engine dry run of the same queries cell for cell.""")
for p in ['PART2_RESULTS.md','docs/PART2_RESULTS.md']:
    sub(p,"""queries run on a local copy of those records; the Supabase SQL-editor export is still to be run and compared (see Deviations).""","""queries run in the Supabase SQL editor (export ingested 2026-10-17; `experiments/results/fashion-official-1-sql/PROVENANCE.md`).""")
    sub(p,"""- The SQL provenance is a **local replica**, not a Supabase SQL export; this report is provisional until `docs/fashion_official_1_export.sql`
  has been run once and its output matches.""","""- SQL provenance: the tables come from the Supabase SQL-editor export; an earlier local-engine dry run of the same queries matched it cell for cell.""")
sub('docs/PROJECT_NOTES.md',"SQL provenance is still a local replica until the Supabase export is run.","SQL provenance is the Supabase SQL-editor export (ingested 2026-10-17); an earlier local dry run matched it.")
sub('docs/FASHION_PROTOCOL.md',"Until this is done, the CSVs in that folder are labelled in their `PROVENANCE.md` as a **local replica** (same queries, same records read via the API), and every report says so.","Done on 2026-10-17: `PROVENANCE.md` in that folder now records source `supabase-sql-editor`, the database's own timestamp/version and the export's sha256.")
sub('docs/SUBMISSION_CHECKLIST.md',"""| 🟡 | Tables come from `docs/comparison_queries.sql` (queries 1-7, 9, 10) run on a **local Postgres replica** of the 18 records, labelled as such in `fashion-official-1-sql/PROVENANCE.md`. ⏳ Run `docs/fashion_official_1_export.sql` once in Supabase, ingest it, re-run `analyze`, and compare (see Remaining steps). |""","""| ✅ | Tables come from `docs/comparison_queries.sql` (queries 1-7, 9, 10) run in the Supabase SQL editor (export ingested 2026-10-17, recorded in `fashion-official-1-sql/PROVENANCE.md`); the local-engine dry run matched cell for cell. |""")
sub('docs/EMMA_ENGINEERING_REPORT.md',"the SQL tables came from a local replica until the Supabase export is run (see Limits).","the SQL tables come from the Supabase SQL-editor export.")
sub('docs/EMMA_ENGINEERING_REPORT.md',"- The SQL provenance is a local replica. The Supabase export is prepared but not yet run, so these numbers are provisional until it matches.","- The SQL tables come from the Supabase SQL-editor export, which matched an earlier local dry run of the same queries cell for cell.")
E
sed -n 1,6p docs/EMMA_ENGINEERING_REPORT.md | cut -c1-250; sed -n 12,16p docs/KOMAL_ENGINEERING_REPORT.md | cut -c1-500; sed -n 30,36p docs/KOMAL_ENGINEERING_REPORT.md|cut -c1-250; sed -n 1,5p docs/KOMAL_ENGINEERING_REPORT.md|cut -c1-300

OUT
# Engineering report: Emma Rogoveanu

> **DRAFT.** Team numbers are the official saved records (group `fashion-official-1`, runs 53–70); the SQL tables come from the Supabase SQL-editor export.

## What I owned
I built the foundation and the serving side of See-Sense: the baseline and improved CNNs, reproducible training, the experiment runner, the Supabase schema and migrations, Grad-CAM loading from stored checkpoints, the five-tab dashboard, and the pred

**Controlled ablations.** Beyond the baseline and the full improved model, four variants each remove one technique: batch normalization, dropout, augmentation, and the cosine schedule. Because a seed uses the same split and budget for all six, differences can be paired by seed rather than compared as unrelated averages.

**Check numbers two ways.** The comparison queries run unchanged on a local Postgres engine, and I recomputed the summary in Python independently. On the earlier synthetic experiment this gave zero mismatches. The same discipline applies now: the Fashion tables are provisional until the Supabase export matches.


## What I would change
- Read the assignment before choosing a dataset. The synthetic shapes came from the starter, and the mismatch with the required Fashion-MNIST was caught late. Neither of us introduced it, but we both should have checked sooner.
- Run more seeds and a larger-capacity variant, to separate "techniques hurt" from "this model is too small."
- Compute provenance from Supabase from the start, not a local replica.

## Ethics and my Christian worldview
# Engineering report: Komal Khan

> **DRAFT written for Komal to review and rewrite.** It was drafted from the repository record (`ai-usage/Komal-Session1.md`, `Komal-Session2.md`, git history) and the team's verified results. Anything marked [Komal to confirm] is an inference, not something Komal has said. Her personal reflection i

## My focus: experimental method and evidence
Bash
IN
python3 -I - <<'E'
import pathlib
f=pathlib.Path('docs/KOMAL_ENGINEERING_REPORT.md'); t=f.read_text()
o="The same discipline applies now: the Fashion tables are provisional until the Supabase export matches."
assert o in t
f.write_text(t.replace(o,"The same discipline applied to the Fashion tables: the Supabase SQL-editor export matched the earlier local run cell for cell."))
E
source .venv/bin/activate
python3 -I - <<'E'
# independent check: recompute per-class precision/recall from the stored confusion matrices (records JSON) vs q7 (Supabase) and per_class_metrics.csv
import json,csv
d=json.load(open('experiments/results/fashion-official-1-records/runs.json'))
runs=d if isinstance(d,list) else d.get('runs',d)
print(type(d), len(runs), sorted(runs[0].keys())[:40])
E

OUT
<class 'dict'> 18 ['accuracy', 'artifact_version', 'augment', 'augment_max_shift', 'batch_size', 'best_epoch', 'calibration', 'config', 'confusion_matrix', 'created_at', 'data_seed', 'dataset_id', 'dropout', 'epochs', 'experiment_group', 'id', 'lr', 'lr_history', 'macro_f1', 'model_variant', 'n_test', 'n_train', 'n_val', 'optimizer', 'param_count', 'run_name', 'scheduler', 'split_seed', 'split_sha256', 'test_loss', 'test_per_class', 'train_accuracy_history', 'train_loss_history', 'train_seconds', 'train_seed', 'use_batchnorm', 'val_accuracy_history', 'val_loss_history']
Bash
IN
head -3 experiments/results/fashion-official-1-sql/q7.csv; head -3 experiments/results/fashion-official-1-analysis/per_class_metrics.csv; python3 -I - <<'E'
import json,csv
runs=json.load(open('experiments/results/fashion-official-1-records/runs.json'))
runs=list(runs.values()) if isinstance(runs,dict) else runs
print(list(json.load(open('experiments/results/fashion-official-1-records/runs.json')).keys())[:3])
r=runs[0]; print(r['id'], r['test_per_class'] if not isinstance(r['test_per_class'],dict) else list(r['test_per_class'])[:5], str(r['confusion_matrix'])[:80])
E

OUT
Exit code 1
run_name,train_seed,true_class,predicted_class,n
baseline-seed0,0,T-shirt/top,T-shirt/top,855
baseline-seed0,0,T-shirt/top,Trouser,1
variant,class,mean_precision,sd_precision,mean_recall,sd_recall,n_runs
baseline,T-shirt/top,0.8525,0.0224,0.8243,0.0267,3
baseline,Trouser,0.9811,0.0039,0.9840,0.0000,3
['datasets', 'provenance', 'runs']
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
TypeError: list indices must be integers or slices, not str
Bash
IN
python3 -I - <<'E'
import json,csv,statistics as st
from collections import defaultdict
runs=json.load(open('experiments/results/fashion-official-1-records/runs.json'))['runs']
C=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
print(len(runs), sorted(r['id'] for r in runs)[0], sorted(r['id'] for r in runs)[-1])
# recompute from stored confusion matrices
prec=defaultdict(list);rec=defaultdict(list)
for r in runs:
    cm=r['confusion_matrix']; v=r['model_variant'] if r['run_name'].split('-seed')[0]==r['model_variant'] else r['run_name'].split('-seed')[0]
    v=r['run_name'].split('-seed')[0]
    for i,c in enumerate(C):
        tp=cm[i][i]; col=sum(cm[j][i] for j in range(10)); row=sum(cm[i])
        prec[v,c].append(tp/col if col else 0); rec[v,c].append(tp/row if row else 0)
mx=0;n=0
for row in csv.DictReader(open('experiments/results/fashion-official-1-analysis/per_class_metrics.csv')):
    k=(row['variant'],row['class']); n+=1
    mx=max(mx,abs(st.mean(prec[k])-float(row['mean_precision'])),abs(st.mean(rec[k])-float(row['mean_recall'])))
print('rows',n,'max abs diff (rounded to 4dp csv):',mx)
# q7 vs stored confusion
q7=defaultdict(int)
for row in csv.DictReader(open('experiments/results/fashion-official-1-sql/q7.csv')): q7[row['run_name'],row['true_class'],row['predicted_class']]=int(row['n'])
bad=sum(1 for r in runs for i in range(10) for j in range(10) if r['confusion_matrix'][i][j]!=q7[r['run_name'],C[i],C[j]])
print('q7 vs stored confusion cell mismatches:',bad)
print('test sizes', {r['n_test'] for r in runs}, {r['n_train'] for r in runs},{r['n_val'] for r in runs}, {r['epochs'] for r in runs},{r['dataset_id'] for r in runs})
E

OUT
18 53 70
rows 60 max abs diff (rounded to 4dp csv): 4.99822336429423e-05
q7 vs stored confusion cell mismatches: 0
test sizes {10000} {54000} {6000} {20} {24}
3:55 PM
Provenance is fixed and the per-class cross-check holds. The 60 class rows differ only by 4-decimal rounding, and all 1,800 confusion cells match. Now the tests and dependency checks.



Bash
IN
source .venv/bin/activate && python -m pytest -q 2>&1 | tail -15; echo ---; pip check 2>&1 | tail -5; cat api/requirements.txt ui/requirements.txt; pip list --format=freeze 2>/dev/null | grep -iE "^(torch|fastapi|streamlit|numpy|pillow|supabase|httpx|uvicorn|pytest|matplotlib|pandas|python-multipart|requests)=="; cat render.yaml | head -30; ls .python-version runtime.txt 2>&1 | head

OUT
      # Updated
      @alt.theme.register('theme_name', enable=True)
      def custom_theme():
          return alt.theme.ThemeConfig(
              {'height': 400, 'width': 700}
          )

  See the updated User Guide for further details:
      https://altair-viz.github.io/user_guide/api.html#theme
      https://altair-viz.github.io/user_guide/customization.html#chart-themes
    with alt.themes.enable("none") if alt.themes.active == "default" else nullcontext():  # type: ignore[attr-defined,unused-ignore]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
293 passed, 6 skipped, 4 warnings in 100.92s (0:01:40)
---
No broken requirements found.
# API tier (deployed on Render.com)
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
python-multipart==0.0.9
# CPU-only torch keeps the Render build small and free-tier friendly. The extra
# index serves torch==2.4.1+cpu on Linux (without it pip installs the multi-GB CUDA
# build); macOS/Windows resolve the normal PyPI wheel. NumPy stays on 1.x for torch 2.4.
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.4.1
numpy==1.26.4
pillow==10.4.0
supabase==2.7.4
python-dotenv==1.0.1
# UI tier (deployed on Streamlit Community Cloud)
# No torch, no pillow, no model code here -- a thin client on purpose.
# Grad-CAM PNGs arrive already rendered from the API; the UI just base64-decodes.
streamlit==1.38.0
requests==2.32.3
pandas==2.2.2
supabase==2.7.4
# st.dataframe converts via pyarrow. Streamlit 1.38 only requires pyarrow>=7, so an
# unpinned install pulls the newest release, which needs NumPy>=2 and fails against
# the NumPy 1.26.4 that the API/torch stack is pinned to. 17.0.0 works with both.
pyarrow==17.0.0
fastapi==0.115.0
httpx==0.27.2
matplotlib==3.11.2
numpy==1.26.4
pandas==2.2.2
pillow==10.4.0
pytest==8.3.3
python-multipart==0.0.9
requests==2.32.3
streamlit==1.38.0
supabase==2.7.4
torch==2.4.1
uvicorn==0.30.6
# Render.com blueprint for the FastAPI model service (Cloud #2).
# In the Render dashboard: New -> Blueprint -> point at this repo.
# Set SUPABASE_URL and SUPABASE_SERVICE_KEY as environment variables (secrets)
# in the service settings after the first deploy. Never set SEE_SENSE_LOCAL_DB on
# Render: it switches the API to a throwaway local file and /healthz would report
# "offline" instead of "ok". The free plan sleeps when idle (cold start ~30-60 s);
# the UI's Try It tab warms the service up with /healthz before predicting.
services:
  - type: web
    name: see-sense-api
    runtime: python
    plan: free
    buildCommand: "pip install -r api/requirements.txt"
    startCommand: "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
    healthCheckPath: /healthz
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.9"
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_SERVICE_KEY
        sync: false
      - key: ALLOWED_ORIGINS
        value: "*"
ls: .python-version: No such file or directory
ls: runtime.txt: No such file or directory
Bash
IN
source .venv/bin/activate && python -m pytest -q -rs 2>&1 | grep SKIP; grep -hE "^(import|from) " ui/*.py | sort -u | grep -vE "^from (ui|shared|\.)" ; pip list --outdated 2>/dev/null | grep -iE "^(fastapi|streamlit|torch|pillow|supabase|requests|pandas|pyarrow|python-multipart|uvicorn) " ; pip-audit --version 2>&1 | head -1

OUT
SKIPPED [1] tests/test_live_supabase.py:60: Live tests write to the real Supabase project; set SEE_SENSE_RUN_LIVE_TESTS=1 (and SUPABASE_URL / SUPABASE_SERVICE_KEY) to run them.
SKIPPED [1] tests/test_live_supabase.py:68: Live tests write to the real Supabase project; set SEE_SENSE_RUN_LIVE_TESTS=1 (and SUPABASE_URL / SUPABASE_SERVICE_KEY) to run them.
SKIPPED [1] tests/test_live_supabase.py:88: Live tests write to the real Supabase project; set SEE_SENSE_RUN_LIVE_TESTS=1 (and SUPABASE_URL / SUPABASE_SERVICE_KEY) to run them.
SKIPPED [1] tests/test_live_supabase.py:108: Live tests write to the real Supabase project; set SEE_SENSE_RUN_LIVE_TESTS=1 (and SUPABASE_URL / SUPABASE_SERVICE_KEY) to run them.
SKIPPED [1] tests/test_live_supabase.py:121: Live tests write to the real Supabase project; set SEE_SENSE_RUN_LIVE_TESTS=1 (and SUPABASE_URL / SUPABASE_SERVICE_KEY) to run them.
SKIPPED [1] tests/test_live_supabase.py:140: Live tests write to the real Supabase project; set SEE_SENSE_RUN_LIVE_TESTS=1 (and SUPABASE_URL / SUPABASE_SERVICE_KEY) to run them.
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from supabase import create_client
from typing import NamedTuple, Optional
import base64
import concepts as concepts_math  # noqa: E402
import csv
import evidence  # noqa: E402  # sibling module: loaders for the saved experiment evidence
import json
import logging
import math
import pandas as pd
import re
import recent  # noqa: E402
import requests
import streamlit as st
import sys
import tryit  # noqa: E402
fastapi          0.115.0 0.143.3 wheel
pandas           2.2.2   3.0.6   wheel
pillow           10.4.0  13.0.0  wheel
pyarrow          17.0.0  26.0.0  wheel
python-multipart 0.0.9   0.0.32  wheel
requests         2.32.3  2.34.2  wheel
streamlit        1.38.0  1.66.0  wheel
supabase         2.7.4   2.33.0  wheel
torch            2.4.1   2.14.1  wheel
uvicorn          0.30.6  0.54.0  wheel
(eval):1: command not found: pip-audit
3:59 PM
Tests pass: 293 passed, 6 skipped (the live-Supabase tests, which write to the cloud and are off by default). Next is the local app. Let me see how it runs.



Bash
IN
cat config/served_model.json; sed -n 1,80p docs/RUN_API_LOCALLY.md; ls -la .local_db.json* ui/assets/samples | head -30; ls .env* ui/.streamlit 2>&1 | head

OUT
{
  "dataset": "fashion-mnist",
  "n_classes": 10,
  "experiment_group": "fashion-official-1",
  "run_name": "full_improved-seed0",
  "served_run_id": 54,
  "chosen_on": "2026-10-17",
  "chosen_before_official_results": true,
  "rationale": "The served model is the project's full improved configuration trained with the lowest training seed (0). It was fixed here on 2026-10-17, before any official Fashion-MNIST run or test result existed, so it is not selected by test score. Do not change run_name after results exist.",
  "how_to_fill_served_run_id": "After the official runs finish, copy the run_id of the 'full_improved-seed0' record from experiments/results/fashion-official-1-runs.jsonl into served_run_id. tests/test_ui_tryit.py and the UI verify it against those records and against GET /runs/{id}. Filled with 54 on 2026-10-17 after the 18 official runs (ids 53-70) were saved; verified against Supabase (group fashion-official-1, dataset 24, ten class names, mean 0.286 / std 0.353). If it is ever null Try It reports that no Fashion-MNIST model is configured instead of serving a different model."
}
# Running the See-Sense API locally (Part 2, step 1)

> **Note (2026-10-17, dataset correction):** this guide was written for the earlier synthetic-shape version and is kept as Komal wrote it.
> On branch `correction/fashion-mnist` some details differ: the offline store seeds **dataset id 1 = Fashion-MNIST** (ten classes; run
> `python tools/fetch_fashion_mnist.py` first), `POST /datasets` takes only a `name`, `POST /train` no longer takes `test_size`, `/predict_sample`
> is gone, and the `--dry-run` fingerprint, test counts and "4 classes" outputs below are historical. See the README and `docs/FASHION_PROTOCOL.md`.

Written by Komal for Part 2. It finishes step 1 of the
[Part 1 handoff](PART1_HANDOFF.md) section 9: "Get access ... Run the test
suite, then a `--dry-run`."

There are two ways to run the API on your own machine:

| Mode | Needs Supabase keys? | Where runs are saved | Use it for |
|---|---|---|---|
| **A. Supabase** (normal) | yes (Emma owns the project) | the shared Supabase project | exploration the team should see, and **all official runs** (run by Emma) |
| **B. Offline** (`SEE_SENSE_LOCAL_DB`) | no | a local JSON file (`.local_db.json`, git-ignored) | trying the API and runner before keys arrive, debugging, quick checks |

Offline mode is a development convenience only. The dashboard,
`docs/comparison_queries.sql` and the rest of the team cannot see what it saves,
so **official Part 2 results must go to Supabase (mode A).**

## 1. One-time setup

Use Python 3.11 (on macOS, plain `python` may be 2.7). From the repo root:

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt  # torch is large; this can take a few minutes
pytest -q                            # expect: 82 passed, 1 skipped
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # prompt should start with (.venv)
pip install -r requirements-dev.txt
pytest -q
```

Windows pitfalls hit during setup: `python3.11` and `source` do not exist in
PowerShell (use `py -3.11` and `.venv\Scripts\Activate.ps1`). If activation says
"running scripts is disabled", run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. Every new terminal
must be activated again. Run commands one per line; pasting two together (e.g.
`Activate.ps1python ...`) fails with "module '.venv' could not be loaded".
The zip unpacks to `see-sense-main-part2\see-sense-main`; `cd` into the inner
folder first.

The one skipped test is the live Supabase round-trip. It runs only when keys are
set.

## 2a. Start the API against Supabase (mode A)

```bash
cp .env.example .env
# edit .env: set SUPABASE_URL and SUPABASE_SERVICE_KEY (never commit .env)
uvicorn api.main:app --port 8000
```

The API now reads `.env` by itself (`load_dotenv()` in `api/db.py`). Before this
fix, only `db/seed.py` read it, so the API started with no credentials unless you
exported them in the shell.

Check that it is working in a second terminal:

```bash
curl http://127.0.0.1:8000/healthz   # {"status":"ok","model_loader":true,"supabase":true}
curl http://127.0.0.1:8000/version   # supabase_project_ref should be Emma's project
```

`"status":"degraded"` with `"supabase":false` means the keys are wrong or the
project can't be reached.

## 2b. Start the API offline (mode B)

(eval):1: no matches found: .local_db.json*
.env
.env.example

ui/.streamlit:
secrets.toml
secrets.toml.example
3:59 PM
Using the real Supabase keys would make /predict write a prediction row, which you've ruled out. I'll check whether run 54's weights are available for an offline local database.



Bash
IN
source .venv/bin/activate && python3 -I - <<'E'
import json
d=json.load(open('experiments/results/fashion-official-1-records/runs.json'))
r=[x for x in d['runs'] if x['id']==54][0]
print({k:(str(v)[:60]) for k,v in r.items() if k in('weights_uri','weights','artifact_version','run_name','experiment_group')}); print(d['provenance'] if 'provenance' in d else '')
E
grep -n "weights\|def \|class " api/local_db.py | head -50; sed -n 80,130p docs/RUN_API_LOCALLY.md; ls data 2>/dev/null; sed -n 1,60p tests/test_ui_app_smoke.py

OUT
{'artifact_version': '3', 'experiment_group': 'fashion-official-1', 'run_name': 'full_improved-seed0'}
{'group': 'fashion-official-1', 'kind': 'PostgREST read via service key (NOT a SQL export)', 'n_runs': 18, 'read_at_utc': '2026-10-17T22:24:10+00:00', 'run_ids': [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70], 'verification_problems': []}
43:def _now() -> str:
47:class LocalStore:
50:    def __init__(self, path: str | os.PathLike):
58:    def _read(self) -> dict:
61:    def _write(self, data: dict) -> None:
67:    def _next_id(rows: list) -> int:
71:    def ping(self) -> bool:
78:    def insert_dataset(self, name, n_rows, img_size, n_classes, noise=0.0, seed=0,
92:    def get_dataset(self, dataset_id: int) -> Optional[dict]:
95:    def insert_run(self, dataset_id, lr, batch_size, epochs, metrics, model_b64,
122:    def run_name_exists(self, experiment_group: str, run_name: str) -> bool:
128:    def get_run(self, run_id: int) -> Optional[dict]:
131:    def get_run_artifact(self, run_id: int) -> Optional[str]:
137:    def latest_runs(self, limit: int = 50) -> List[dict]:
140:    def log_prediction(self, run_id, sha256, width, height, label, class_name, probabilities,
164:    def _with_run(self, row: dict, runs: dict) -> dict:
170:    def list_predictions(self, limit: int = 50) -> List[dict]:
175:    def get_prediction(self, prediction_id: int) -> Optional[dict]:

```bash
SEE_SENSE_LOCAL_DB=.local_db.json uvicorn api.main:app --port 8000
# Windows PowerShell:  $env:SEE_SENSE_LOCAL_DB=".local_db.json"; uvicorn api.main:app --port 8000
```

Or put `SEE_SENSE_LOCAL_DB=.local_db.json` in `.env`. On first start the file is
created with **dataset id 1 = 800 rows, noise 0.1, seed 42**, the same settings
Emma seeded in Supabase, so every command in the handoff works unchanged. Delete
the file to start over. Remove the variable to go back to Supabase.

## 3. Look before you run (no training)

```bash
python -m experiments.run_experiments --dry-run \
  --dataset-id 1 --group explore-1 --seeds 0 \
  --epochs 15 --batch-size 32 --lr 0.01 --split-seed 0
```

This prints six request bodies and the protocol fingerprint. With exactly these
values the fingerprint was
`c82db51cf2b954a9654522e4cc988593fcd3bf9cd6af5ea813c91add20cb7ae0`. These are
example values, **not** the frozen protocol.

## 4. Smoke test through the running API

```bash
python -m experiments.run_experiments --api-url http://127.0.0.1:8000 \
  --dataset-id 1 --group explore-smoke --seeds 0 \
  --epochs 3 --batch-size 32 --lr 0.01 --split-seed 0
```

Expected output: `done: 6 saved, 0 already existed, 0 failed`. If you run it
again, all six are skipped (`6 already existed`), not retrained. Use an
`explore-*` group name for anything that is not the official run. Against
Supabase, these runs go into the shared database.

## 5. Verification done on 2026-10-14

### 5a. Development workspace (offline mode, Linux, Python 3.11, torch 2.4.1)

| Check | Result |
|---|---|
| `pytest -q` before any change | 79 passed, 1 skipped (matches the handoff) |
| `pytest -q` after adding offline mode + its 3 tests | 82 passed, 1 skipped |
| `GET /healthz`, `/version`, `/classes` | ok / torch 2.4.1 / 4 classes |
| `--dry-run` (values above) | 6 bodies, fingerprint `c82db51c…` |
| Runner, 6 configs × seed 0, 3 epochs, group `explore-smoke` | 6 saved, read-back checks passed, exit 0 |
| Same command again | 6 skipped, 0 retrained |
| `POST /predict` with an uploaded horizontal-bar PNG | `horizontal`, confidence 0.998, Grad-CAM returned |
| API reads `.env` (fake URL, real API code) | project ref read from `.env`; health `degraded` as expected with fake keys |
fashion-mnist
"""Render the whole Streamlit app (AppTest) with no network: every tab must draw without an exception.

The Supabase client is replaced by a fake. The committed configuration serves run 54 and the committed official
evidence (group fashion-official-1) is displayed, including its provenance label; a configuration with no served
run must still produce the honest "unavailable" state instead of a guess.
"""
from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).parent.parent / "ui" / "app.py"


class FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_):
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_):
        return self

    def eq(self, *_):
        return self

    def execute(self):
        return type("R", (), {"data": self.rows})()


class FakeSupabase:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return FakeTable(self.tables.get(name, []))


PROBS = [0.02, 0.01, 0.03, 0.6, 0.04, 0.05, 0.1, 0.05, 0.05, 0.05]
PRED_ROW = {"id": 1, "created_at": "2026-10-17T06:38:40+00:00", "predicted_class": "Dress", "confidence": 0.6,
            "probabilities": PROBS, "served_by_run_id": 7, "run_name": "full_improved-seed0",
            "experiment_group": "g", "model_variant": "improved", "run_test_accuracy": 0.9, "run_test_macro_f1": 0.89}
RUN_ROW = {"id": 7, "run_name": "full_improved-seed0", "experiment_group": "g", "model_variant": "improved",
           "accuracy": 0.9, "macro_f1": 0.89, "config": {"dataset": "fashion-mnist"}, "epochs": 20}


def _run(monkeypatch, tables):
    import streamlit as st
    import supabase

    st.cache_resource.clear()  # the app caches its anon client; start each test from a clean cache

    monkeypatch.setattr(supabase, "create_client", lambda *a, **k: FakeSupabase(tables))
Weights only live in Supabase, not in the stored records. I'll check run 54 read-only by loading its checkpoint from Supabase and running the API's prediction code in-process, without any logging or writes.
Bash
IN
grep -n "^def \|^async def \|^@app\|^class " api/main.py | head -60; grep -n "def " api/db.py | head -40; grep -n "SUPABASE\|secrets\|anon" ui/app.py | head -20; sed -n 60,140p tests/test_ui_app_smoke.py; ls ui/assets/samples | head -30

OUT
64:@app.exception_handler(httpx.TransportError)
65:async def _database_unreachable(_request, exc):
70:@app.exception_handler(APIError)
71:async def _database_error(_request, exc):
87:def _git_sha() -> str:
100:def _project_ref() -> str | None:
107:def _artifact_or_404(run_id: int) -> str:
117:def _to_gray_array(raw: bytes, img_size: int) -> np.ndarray:
130:@app.post("/datasets", response_model=Dataset, tags=["datasets"])
131:def create_dataset(req: DatasetCreate) -> Dataset:
157:@app.post("/train", response_model=TrainResponse, tags=["training"])
158:def train(req: TrainRequest) -> TrainResponse:
221:def _run_from_row(row: dict) -> Run:
226:@app.get("/runs/{run_id}", response_model=Run, tags=["training"])
227:def get_run(run_id: int) -> Run:
234:@app.get("/runs", response_model=list[Run], tags=["training"])
235:def list_runs() -> list[Run]:
242:def _predict_array(run_id: int, gray: np.ndarray, sha256: str, true_class: str | None):
273:@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
274:async def predict(run_id: int = Form(...), file: UploadFile = File(...)) -> PredictResponse:
293:def _prediction_record(row: dict) -> PredictionRecord:
301:@app.get("/predictions", response_model=list[PredictionRecord], tags=["prediction"])
302:def list_predictions(limit: int = Query(50, ge=1, le=200)) -> list[PredictionRecord]:
310:@app.get("/predictions/{prediction_id}", response_model=PredictionRecord, tags=["prediction"])
311:def get_prediction(prediction_id: int) -> PredictionRecord:
322:@app.get("/classes", response_model=ClassesResponse, tags=["meta"])
323:def classes() -> ClassesResponse:
331:@app.get("/healthz", response_model=Health, tags=["ops"])
332:def healthz() -> Health:
349:@app.get("/version", response_model=Version, tags=["ops"])
350:def version() -> Version:
50:def get_client() -> Client:
60:def ping() -> bool:
71:def insert_dataset(
86:def get_dataset(dataset_id: int) -> Optional[dict]:
101:def experiment_columns(
156:def insert_run(
192:def run_name_exists(experiment_group: str, run_name: str) -> bool:
205:def get_run(run_id: int) -> Optional[dict]:
212:def get_run_artifact(run_id: int) -> Optional[str]:
224:def latest_runs(limit: int = 50) -> List[dict]:
251:def log_prediction(
290:def _prediction_select(extra: str = "") -> str:
294:def list_predictions(limit: int = 50) -> List[dict]:
308:def get_prediction(prediction_id: int) -> Optional[dict]:
6:  * the ONLY database access is read-only anon-key queries (the `recent_predictions` and
12:Configuration comes from st.secrets (see .streamlit/secrets.toml.example):
14:    SUPABASE_URL         -> https://<ref>.supabase.co
15:    SUPABASE_ANON_KEY    -> the public anon key (safe to ship to the browser)
35:API_URL = st.secrets["API_URL"].rstrip("/")
43:def supabase_anon():
44:    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])
352:# Recent Classifications  (read-only anon-key queries; drill-down to the producing run)
357:        "Read directly from Supabase with the anon key through read-only views that contain no image "
364:                st.session_state["recent_pred_rows"] = recent.fetch_recent_predictions(supabase_anon)
370:                    st.session_state["recent_pred_rows"] = recent.fetch_recent(supabase_anon)
421:                            st.session_state["drill_run"] = recent.fetch_run(supabase_anon, row["Run ID"])
    monkeypatch.setattr(supabase, "create_client", lambda *a, **k: FakeSupabase(tables))
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    at.secrets["API_URL"] = "http://127.0.0.1:9"
    at.secrets["SUPABASE_URL"] = "https://example.invalid"
    at.secrets["SUPABASE_ANON_KEY"] = "anon"
    return at.run()


def test_app_renders_five_tabs_shows_the_final_evidence_and_labels_its_provenance(monkeypatch):
    at = _run(monkeypatch, {"recent_predictions": [PRED_ROW], "runs": [RUN_ROW]})
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Concepts", "Try It", "Compare Models", "Recent Classifications", "Model Card"]
    text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning) + " ".join(i.value for i in at.info)
    assert "Fashion-MNIST" in at.title[0].value
    assert "ten Fashion-MNIST clothing categories" in text
    assert "Final Fashion-MNIST results are not available yet" not in text
    assert "SQL provenance" in text + " ".join(c.value for c in at.caption)
    assert "Six-variant results" in " ".join(h.value for h in [*at.subheader, *at.header, *at.markdown])
    assert not any("not Fashion-MNIST" in m.value for m in at.warning)  # hidden until the toggle is switched on
    toggle = next(x for x in at.toggle if "historical" in x.label.lower() or "earlier experiment" in x.label.lower())
    assert "NOT Fashion-MNIST" in toggle.label and toggle.value is False
    at = toggle.set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("This is not Fashion-MNIST" in w.value for w in at.warning)  # the historical section is labelled
    assert not any("could not be read" in w.value for w in at.warning)      # every historical section rendered
    assert not any("Try It is unavailable" in w.value for w in at.warning)  # run 54 is configured and in the saved records


def test_without_a_served_run_id_try_it_is_unavailable_instead_of_guessing(monkeypatch, tmp_path):
    at, posts = _try_it(monkeypatch, tmp_path, RUN_OK, Resp(200, _predict_body()), served_run_id=None)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Try It is unavailable" in w.value for w in at.warning) and posts == []


def test_recent_tab_shows_the_prediction_and_drills_down_to_its_run(monkeypatch):
    at = _run(monkeypatch, {"recent_predictions": [PRED_ROW], "runs": [RUN_ROW]})
    assert not at.exception, [e.value for e in at.exception]
    tables = [d.value for d in at.dataframe]
    flat = " ".join(str(t) for t in tables)
    assert "Dress" in flat and "full_improved-seed0" in flat
    assert "sha256" not in flat.lower()
    assert any("Producing run 7" in m.value for m in at.markdown)


def test_recent_tab_falls_back_when_the_predictions_view_is_missing(monkeypatch):
    class Broken(FakeSupabase):
        def table(self, name):
            if name == "recent_predictions":
                raise RuntimeError("relation does not exist")
            return super().table(name)

    import streamlit as st
    import supabase

    st.cache_resource.clear()
    legacy = {"recent_classifications": [{"id": 1, "run_id": 4, "class_name": "block", "confidence": 0.9,
                                          "created_at": "2026-10-17T06:38:40+00:00"}]}
    monkeypatch.setattr(supabase, "create_client", lambda *a, **k: Broken(legacy))
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    at.secrets["API_URL"] = "http://127.0.0.1:9"
    at.secrets["SUPABASE_URL"] = "https://example.invalid"
    at.secrets["SUPABASE_ANON_KEY"] = "anon"
    at.run()
    assert not at.exception
    assert any("predictions view isn't available" in w.value for w in at.warning)


# ---- Try It flow with a mocked prediction service ---------------------------------------------------------
import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import requests  # noqa: E402


def _tiny_png_b64() -> str:
    from PIL import Image  # test-side only; the UI itself never imports PIL

    buf = io.BytesIO()
    Image.new("RGB", (28, 28), (200, 40, 40)).save(buf, format="PNG")
0-t-shirt-top-1.png
0-t-shirt-top-2.png
1-trouser-1.png
1-trouser-2.png
2-pullover-1.png
2-pullover-2.png
3-dress-1.png
3-dress-2.png
4-coat-1.png
4-coat-2.png
5-sandal-1.png
5-sandal-2.png
6-shirt-1.png
6-shirt-2.png
7-sneaker-1.png
7-sneaker-2.png
8-bag-1.png
8-bag-2.png
9-ankle-boot-1.png
9-ankle-boot-2.png
manifest.json
Bash
IN
sed -n 100,128p api/main.py; sed -n 242,292p api/main.py

OUT
def _project_ref() -> str | None:
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


def _artifact_or_404(run_id: int) -> str:
    model_b64 = db.get_run_artifact(run_id)
    if model_b64 is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return model_b64


MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # same cap the UI enforces before sending


def _to_gray_array(raw: bytes, img_size: int) -> np.ndarray:
    """Decode arbitrary image bytes to a grayscale [img_size, img_size] uint8 array.

    Same preprocessing the checkpoint declares: grayscale, resize to 28x28. Fashion-MNIST
    items are light on a dark background; photos are NOT inverted or otherwise adjusted.
    """
    img = Image.open(io.BytesIO(raw)).convert("L").resize((img_size, img_size))
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# datasets
def _predict_array(run_id: int, gray: np.ndarray, sha256: str, true_class: str | None):
    """Classify, then log. Logging is REQUIRED: if it fails the caller gets an error, not a result.

    ``db.log_prediction`` writes `image_metadata` and `predictions` in one transaction, so a
    failure leaves neither row. We never answer 200 with ``logged: true`` unless both exist.
    """
    model_b64 = _artifact_or_404(run_id)
    result = predict_with_cam(model_b64, gray)
    try:
        prediction_id = db.log_prediction(
            run_id=run_id,
            sha256=sha256,
            width=int(gray.shape[1]),
            height=int(gray.shape[0]),
            label=result["label"],
            class_name=result["class_name"],
            probabilities=result["probs"],
            confidence=result["confidence"],
        )
    except db.PredictionLogUnavailable as exc:
        _log.error("prediction logging unavailable (migrations 004/005 not applied?): %s", exc)
        raise HTTPException(
            status_code=503,
            detail="prediction logging is unavailable, so the image was not classified",
        )
    return PredictResponse(
        run_id=run_id, image_sha256=sha256, true_class=true_class,
        logged=True, prediction_id=prediction_id, **result,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(run_id: int = Form(...), file: UploadFile = File(...)) -> PredictResponse:
    """Classify an uploaded image and return top-5 probabilities and a Grad-CAM overlay.

    Only the image's sha256 hash, shape and prediction (including the probability
    vector, in `predictions`) are logged -- never pixels.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file larger than 5 MB")
    try:
        gray = _to_gray_array(raw, IMG_SIZE)
    except Exception:
        raise HTTPException(status_code=422, detail="could not decode image")
    sha256 = hashlib.sha256(raw).hexdigest()
    return _predict_array(run_id, gray, sha256, true_class=None)
Bash
IN
source .venv/bin/activate && S=/private/tmp/claude-501/-Users-emmarogoveanu-Documents-GitHub-see-sense/89f94cee-b0d6-48f7-817c-e0a7fcc377ca/scratchpad && cat > $S/check_live_readonly.py <<'E'
import base64, io, json, sys
sys.path.insert(0, "/Users/emmarogoveanu/Documents/GitHub/see-sense")
from fastapi.testclient import TestClient
from PIL import Image
from api import db, main
writes = []
db.log_prediction = lambda **k: writes.append(1) or -1   # STUB: no cloud write
c = TestClient(main.app)
r = c.get("/runs/54").json(); print("run54:", r["id"], r["run_name"], r["experiment_group"], "acc", r["accuracy"], r["config"].get("dataset"))
cl = c.get("/classes").json(); print("classes:", len(cl["classes"]) if isinstance(cl, dict) else len(cl), cl)
man = json.load(open("ui/assets/samples/manifest.json")); print("manifest keys", list(man)[:4] if isinstance(man, dict) else man[:1])
for name in ["3-dress-1.png", "7-sneaker-2.png"]:
    raw = open(f"ui/assets/samples/{name}", "rb").read()
    p = c.post("/predict", data={"run_id": 54}, files={"file": (name, raw, "image/png")}).json()
    probs = p["probs"]; top5 = sorted(zip(p["class_names"] if "class_names" in p else range(10), probs), key=lambda x: -x[1])[:5]
    cam = Image.open(io.BytesIO(base64.b64decode(p["gradcam_png_b64"] if "gradcam_png_b64" in p else p[[k for k in p if "cam" in k.lower()][0]])))
    print(name, "->", p["class_name"], round(p["confidence"], 3), "n_probs", len(probs), "sum", round(sum(probs), 4), "top5", [(a, round(b, 3)) for a, b in top5], "cam", cam.size, "keys", sorted(p))
print("stubbed log calls:", len(writes))
E
python $S/check_live_readonly.py 2>&1 | grep -v Warning | tail -20

OUT
You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
  warnings.warn(
run54: 54 full_improved-seed0 fashion-official-1 acc 0.8782 fashion-mnist
classes: 10 {'classes': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], 'n_classes': 10}
manifest keys ['dataset', 'partition', 'rule', 'class_names']
3-dress-1.png -> Dress 0.994 n_probs 10 sum 1.0 top5 [('Dress', 0.994), ('T-shirt/top', 0.005), ('Trouser', 0.001), ('Shirt', 0.0), ('Ankle boot', 0.0)] cam (28, 28) keys ['calibrated', 'cam_png', 'class_name', 'class_names', 'confidence', 'image_sha256', 'input_png', 'label', 'logged', 'prediction_id', 'probs', 'run_id', 'temperature', 'top5', 'true_class']
7-sneaker-2.png -> Sneaker 0.976 n_probs 10 sum 1.0 top5 [('Sneaker', 0.976), ('Ankle boot', 0.016), ('Sandal', 0.009), ('Bag', 0.0), ('Dress', 0.0)] cam (28, 28) keys ['calibrated', 'cam_png', 'class_name', 'class_names', 'confidence', 'image_sha256', 'input_png', 'label', 'logged', 'prediction_id', 'probs', 'run_id', 'temperature', 'top5', 'true_class']
stubbed log calls: 2
4:00 PM
Run 54 gives ten labels, ten probabilities summing to 1, a top-5 and a 28×28 Grad-CAM, with the log call stubbed. Next I'll render the full app against the real Supabase anon key, read-only.



Bash
IN
source .venv/bin/activate && S=/private/tmp/claude-501/-Users-emmarogoveanu-Documents-GitHub-see-sense/89f94cee-b0d6-48f7-817c-e0a7fcc377ca/scratchpad && cat > $S/check_ui.py <<'E'
import sys, tomllib
sys.path.insert(0, "/Users/emmarogoveanu/Documents/GitHub/see-sense")
from streamlit.testing.v1 import AppTest
sec = tomllib.load(open("/Users/emmarogoveanu/Documents/GitHub/see-sense/ui/.streamlit/secrets.toml", "rb"))
at = AppTest.from_file("/Users/emmarogoveanu/Documents/GitHub/see-sense/ui/app.py", default_timeout=90)
at.secrets["API_URL"] = "http://127.0.0.1:9"        # nothing listens: no API call can succeed or write
at.secrets["SUPABASE_URL"] = sec["SUPABASE_URL"]; at.secrets["SUPABASE_ANON_KEY"] = sec["SUPABASE_ANON_KEY"]
at.run()
print("exceptions:", [e.value for e in at.exception])
print("tabs:", [t.label for t in at.tabs])
print("title:", at.title[0].value)
print("warnings:", [w.value[:140] for w in at.warning])
print("captions with provenance:", [c.value for c in at.caption if "provenance" in c.value.lower()])
print("subheaders:", [s.value for s in at.subheader][:12])
for d in at.dataframe:
    df = d.value
    print("dataframe", df.shape, list(df.columns)[:10])
print("buttons:", [b.label for b in at.button])
E
python $S/check_ui.py 2>&1 | grep -v -i "warn" | tail -30

OUT
exceptions: []
tabs: ['Concepts', 'Try It', 'Compare Models', 'Recent Classifications', 'Model Card']
title: 🔍 See-Sense — Fashion-MNIST CNN Classifier with Grad-CAM
captions with provenance: ['SQL provenance: **supabase-sql-editor** (see PROVENANCE.md next to the CSVs).']
subheaders: ['1. Convolution', '2. Stride, padding and output size', '3. Receptive field', '4. Grad-CAM: where did the network look?', 'Final: Fashion-MNIST, 10 classes', 'Six-variant results', 'Matched-configuration audit', 'Paired ablation differences vs full_improved', 'Training and validation loss curves', 'Training and validation accuracy curves', 'Confusion matrices', 'Per-class precision and recall']
dataframe (3, 6) ['n', 'k', 'stride s', 'padding p', 'Calculation', 'Output']
dataframe (5, 4) ['Layer', 'Output size', 'Receptive field r', 'Jump j']
dataframe (4, 4) ['Layer', 'Output size', 'Receptive field r', 'Jump j']
dataframe (6, 7) ['Variant', 'Seeds', 'Test accuracy (mean ± sd)', 'Macro-F1 (mean ± sd)', 'Test loss (mean)', 'Params', 'Train s (mean)']
dataframe (3, 15) ['train_seed', 'n_runs', 'datasets', 'split_seeds', 'split_hashes', 'split_sizes', 'dataset_names', 'val_sizes', 'train_limits', 'preprocessings']
dataframe (5, 8) ['Variant', 'Technique removed', 'Δ accuracy per seed (pp)', 'Mean Δ acc (pp)', '95% CI (pp, paired t)', 'Mean Δ macro-F1', 'Mean Δ test loss', 'Seeds lower than full']
dataframe (60, 4) ['Variant', 'Class', 'Precision', 'Recall']
dataframe (6, 5) ['Variant', 'Runs', 'Mean T', 'Test ECE before → after', 'Test NLL before → after']
dataframe (10, 5) ['Test index', 'True', 'Predicted', 'Confidence', 'Note']
dataframe (10, 5) ['Test index', 'True', 'Predicted', 'Confidence', 'Note']
dataframe (2, 7) ['Time', 'Class', 'Confidence', 'Run ID', 'Run', 'Variant', 'Run test accuracy']
dataframe (5, 2) ['Class', 'Probability']
dataframe (8, 2) ['Field', 'Value']
buttons: ['Check again', 'Refresh']
Bash
IN
source .venv/bin/activate && S=/private/tmp/claude-501/-Users-emmarogoveanu-Documents-GitHub-see-sense/89f94cee-b0d6-48f7-817c-e0a7fcc377ca/scratchpad && cat >> $S/check_ui.py <<'E'
print(at.dataframe[-3].value.to_string()[:600])
print([m.value[:120] for m in at.markdown if "Producing run" in m.value or "ten" in m.value.lower()][:4])
E
python $S/check_ui.py 2>&1 | tail -6

OUT
dataframe (8, 2) ['Field', 'Value']
buttons: ['Check again', 'Refresh']
                      Time     Class  Confidence  Run ID         Run   Variant  Run test accuracy
0  2026-10-17 20:39:04 UTC  Pullover    0.366209      52  live-49130  improved             0.3335
1  2026-10-17 20:39:04 UTC   Sneaker    0.304618      52  live-49130  improved             0.3335
['The same weights are reused at every position, so the layer detects one pattern wherever it appears. A layer with severa', 'Temperature scaling divides the logits by a single number fitted on **validation** data. ECE (expected calibration error', '**What the evidence shows.** The image is a tall sneaker whose upper rises towards the ankle, with a bulky sole. The Gra', '- The Grad-CAM resolution is 7×7 (each cell looks at a roughly 10×10-pixel patch of a 28×28 image), so it cannot localiz']
4:01 PM
The history loads but holds only 2 live-test rows (run 52), so no run-54 prediction exists yet. That is expected until the branch is deployed. Now the reports and the README header.



Bash
IN
for f in docs/EMMA_ENGINEERING_REPORT.md docs/KOMAL_ENGINEERING_REPORT.md; do echo $f; wc -w < $f; python3 -I -c "
import re,sys
t=open('$f').read()
print('alnum words (all):',len(re.findall(r\"[A-Za-z0-9][A-Za-z0-9'’\-\.%,/]*\",t)))
body=re.sub(r'^> .*\$','',t,flags=re.M)
print('excluding blockquote lines:',len(re.findall(r\"[A-Za-z0-9][A-Za-z0-9'’\-\.%,/]*\",body)))
"; done; grep -n "^## " docs/EMMA_ENGINEERING_REPORT.md docs/KOMAL_ENGINEERING_REPORT.md; grep -n -i "pending\|\[Komal" docs/KOMAL_ENGINEERING_REPORT.md | cut -c1-200; echo; grep -n -i "video\|engineering report\|github.com\|supabase.co\|project ref\|ENGINEERING_REPORT" README.md | cut -c1-200; wc -l README.md

OUT
docs/EMMA_ENGINEERING_REPORT.md
     968
alnum words (all): 961
excluding blockquote lines: 939
docs/KOMAL_ENGINEERING_REPORT.md
     851
alnum words (all): 843
excluding blockquote lines: 799
docs/KOMAL_ENGINEERING_REPORT.md:5:## My focus: experimental method and evidence
docs/KOMAL_ENGINEERING_REPORT.md:8:## Method decisions
docs/KOMAL_ENGINEERING_REPORT.md:17:## What the evidence says
docs/KOMAL_ENGINEERING_REPORT.md:26:## Per-class errors and Grad-CAM interpretation
docs/KOMAL_ENGINEERING_REPORT.md:31:## What I would change
docs/KOMAL_ENGINEERING_REPORT.md:36:## Ethics and my Christian worldview
docs/KOMAL_ENGINEERING_REPORT.md:39:## AI use
docs/EMMA_ENGINEERING_REPORT.md:5:## What I owned
docs/EMMA_ENGINEERING_REPORT.md:8:## Design decisions
docs/EMMA_ENGINEERING_REPORT.md:15:## Results I rely on
docs/EMMA_ENGINEERING_REPORT.md:22:## Grad-CAM and its limits
docs/EMMA_ENGINEERING_REPORT.md:25:## Testing
docs/EMMA_ENGINEERING_REPORT.md:28:## Limits and what I would do differently
docs/EMMA_ENGINEERING_REPORT.md:34:## Worldview reflection
docs/EMMA_ENGINEERING_REPORT.md:41:## AI use
3:> **DRAFT written for Komal to review and rewrite.** It was drafted from the repository record (`ai-usage/Komal-Session1.md`, `Komal-Session2.md`, git history) and the team's verified results. Anyth
6:My part of the project was making the comparison fair and the evidence checkable. Per the repository record, I ran the API locally with an offline database, fixed `.env` loading, and in Part 2 wrote
29:For Grad-CAM, the pipeline selects genuinely correct and incorrect test predictions and checks that regenerated accuracy equals the stored value, so figures come from the stored model. The draft in
37:[Komal: write your own reflection. Nothing about your beliefs, tradition or Scripture reading has been assumed here. You may connect it to reporting a null result honestly, to disclosing the datase
40:My AI sessions are recorded in `ai-usage/Komal-Session1.md` and `Komal-Session2.md`. [Komal to add what you verified yourself and what you would not have done without the assistant.]

20:| Data | Supabase | `https://<your-project-ref>.supabase.co` *(placeholder: the project URL is not documented in this repository; fill it in from the Supabase dashboard)* |
37:Details, intervals, per-class results, calibration and Grad-CAM: [MODEL_CARD.md](MODEL_CARD.md), [PART2_RESULTS.md](PART2_RESULTS.md), individual engineering reports [Emma](docs/EMMA_ENGINEERING_RE
97:cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml   # UI: API_URL, SUPABASE_URL (…supabase.co), SUPABASE_ANON_KEY
     143 README.md
Both reports are distinct and within the 750–1,000 word range, but the README violates the first-15-lines rule since the URL table starts at line 17. I'll grab the Supabase project ref and GitHub URL to fix this.
Bash
IN
git remote -v | head -2; grep -E "^SUPABASE_URL" .env | sed -E 's#(https://)([a-z0-9]+)(\.supabase\.co.*)#\1\2\3#'; grep -rn "supabase_project_ref\|project_ref" docs/*.md README.md | head -3; sed -n 36,40p README.md | cut -c1-400; grep -n -i "youtube\|loom\|video" README.md docs/SUBMISSION_CHECKLIST.md | head

OUT
origin	https://github.com/emmabellerogo/see-sense.git (fetch)
origin	https://github.com/emmabellerogo/see-sense.git (push)
SUPABASE_URL=https://glcxpjaeobpppkizlcke.supabase.co
docs/RUN_API_LOCALLY.md:73:curl http://127.0.0.1:8000/version   # supabase_project_ref should be Emma's project
normalization helped; augmentation and dropout lowered accuracy at this 20-epoch budget. Served model (fixed in advance): `full_improved-seed0`, run 54, 87.82%.
Details, intervals, per-class results, calibration and Grad-CAM: [MODEL_CARD.md](MODEL_CARD.md), [PART2_RESULTS.md](PART2_RESULTS.md), individual engineering reports [Emma](docs/EMMA_ENGINEERING_REPORT.md) and [Komal](docs/KOMAL_ENGINEERING_REPORT.md),
[experiments/results/fashion-official-1-analysis/summary.md](experiments/results/fashion-official-1-analysis/summary.md). Protocol: [docs/FASHION_PROTOCOL.md](docs/FASHION_PROTOCOL.md) (frozen).

**Historical (not Fashion-MNIST):** the first version of the project was built on four generated shapes (group `official-supabase-1`, 2,000 images, 30 runs). That experiment is kept for transparency in [experiments/results/official-supabase-1-analysis/summary.md](experiments/results/official-supabase-1-analysis/summary.md), [docs/SUPABASE_GRADCAM_FAILURE_ANALYSIS.md](docs/SUPABASE_GRADCAM_FAILURE_
docs/SUBMISSION_CHECKLIST.md:34:| 19 | Individual contribution statements and **individual videos** | ❌ | Placeholders only: [CONTRIBUTIONS.md](CONTRIBUTIONS.md). ⏳ Each person writes their own and records their own video. |
docs/SUBMISSION_CHECKLIST.md:44:3. Write your own contribution statements, worldview reflections and videos; add the AI-usage record for this session.
Bash
IN
python3 -I - <<'E'
import pathlib
f=pathlib.Path('README.md'); t=f.read_text()
title="# See-Sense — Fashion-MNIST CNN Classifier with Grad-CAM (three clouds)\n"
assert t.startswith(title)
block=title+"""
- **GitHub:** https://github.com/emmabellerogo/see-sense · **Streamlit Cloud:** https://see-sense-glpk38qzwsf9hhvnlkt798.streamlit.app/
- **Render API:** https://see-sense-api-rhkb.onrender.com (`/healthz`, `/docs`) · **Supabase project ref:** `glcxpjaeobpppkizlcke`
- **Videos:** Emma — ❌ *MISSING: not yet recorded* · Komal — ❌ *MISSING: not yet recorded*
- **Engineering reports:** [Emma](docs/EMMA_ENGINEERING_REPORT.md) · [Komal](docs/KOMAL_ENGINEERING_REPORT.md) · Contributions: [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md)
"""
t=block+t[len(title):]
old="| Data | Supabase | `https://<your-project-ref>.supabase.co` *(placeholder: the project URL is not documented in this repository; fill it in from the Supabase dashboard)* |"
assert old in t
t=t.replace(old,"| Data | Supabase | `https://glcxpjaeobpppkizlcke.supabase.co` (project ref `glcxpjaeobpppkizlcke`) |")
f.write_text(t)
E
sed -n 1,9p README.md | cut -c1-160

OUT
# See-Sense — Fashion-MNIST CNN Classifier with Grad-CAM (three clouds)

- **GitHub:** https://github.com/emmabellerogo/see-sense · **Streamlit Cloud:** https://see-sense-glpk38qzwsf9hhvnlkt798.streamlit.app/
- **Render API:** https://see-sense-api-rhkb.onrender.com (`/healthz`, `/docs`) · **Supabase project ref:** `glcxpjaeobpppkizlcke`
- **Videos:** Emma — ❌ *MISSING: not yet recorded* · Komal — ❌ *MISSING: not yet recorded*
- **Engineering reports:** [Emma](docs/EMMA_ENGINEERING_REPORT.md) · [Komal](docs/KOMAL_ENGINEERING_REPORT.md) · Contributions: [docs/CONTRIBUTIONS.md](docs/CON

**Streamlit UI ⇄ FastAPI model API ⇄ Supabase data.** A small PyTorch CNN classifies 28×28 grayscale pictures of clothing into the
ten **Fashion-MNIST** categories (T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot), shows its
4:01 PM
Next: git inventory, sizes, a secrets scan, and the pilot and temp files.



Bash
IN
git status --short | sed -n 1,3p; git status --short -uall | grep -v "^ M\|^M " | awk '{print $2}' | grep -vE "^(docs|tests|db|config|tools|ui/assets)/" ; echo ---; du -sh experiments/results/* ui/assets tests/fixtures data 2>/dev/null | sort -h | tail -12; ls experiments/results/fashion-pilot | head; git check-ignore -v .env ui/.streamlit/secrets.toml data .local_db.json; git ls-files | grep -E "secrets.toml$|^\.env$"; echo scan; git ls-files -o --exclude-standard -z | xargs -0 grep -lIE "eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}|sb_secret_|service_role|sbp_[a-z0-9]{20,}" 2>/dev/null; git ls-files -m -z | xargs -0 grep -lIE "eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}|sb_secret_|sbp_[a-z0-9]{20,}" 2>/dev/null; grep -rl "$(grep ^SUPABASE_SERVICE_KEY .env | cut -d= -f2 | cut -c1-30)" . --exclude-dir=.venv --exclude-dir=.git -I 2>/dev/null

OUT
 M .gitignore
 M MODEL_CARD.md
 M PART2_RESULTS.md
shared/data.py
api/calibration.py
experiments/cam_blank_check.py
experiments/export_records.py
experiments/ingest_sql_export.py
experiments/migration_check.mjs
experiments/package_sql.py
experiments/pilot.py
experiments/pilot_summary.py
experiments/results/fashion-official-1-analysis/accuracy_curves.png
experiments/results/fashion-official-1-analysis/calibration.csv
experiments/results/fashion-official-1-analysis/confusion_matrices.png
experiments/results/fashion-official-1-analysis/loss_curves.png
experiments/results/fashion-official-1-analysis/per_class_metrics.csv
experiments/results/fashion-official-1-analysis/per_class_precision_recall.png
experiments/results/fashion-official-1-analysis/summary.md
experiments/results/fashion-official-1-analysis/test_metrics_by_seed.png
experiments/results/fashion-official-1-gradcam/blank_cam_by_class.csv
experiments/results/fashion-official-1-gradcam/examples.csv
experiments/results/fashion-official-1-gradcam/gradcam_correct.png
experiments/results/fashion-official-1-gradcam/gradcam_incorrect.png
experiments/results/fashion-official-1-gradcam/source.json
experiments/results/fashion-official-1-records/runs.json
experiments/results/fashion-official-1-runs.jsonl
experiments/results/fashion-official-1-sql/PROVENANCE.md
experiments/results/fashion-official-1-sql/q1.csv
experiments/results/fashion-official-1-sql/q10.csv
experiments/results/fashion-official-1-sql/q2.csv
experiments/results/fashion-official-1-sql/q3.csv
experiments/results/fashion-official-1-sql/q4.csv
experiments/results/fashion-official-1-sql/q5.csv
experiments/results/fashion-official-1-sql/q6.csv
experiments/results/fashion-official-1-sql/q7.csv
experiments/results/fashion-official-1-sql/q8.csv
experiments/results/fashion-official-1-sql/q9.csv
experiments/results/fashion-official-1-supabase-export.csv
experiments/results/fashion-pilot/pilot_all_variants_3ep.jsonl
experiments/results/fashion-pilot/pilot_all_variants_3ep.log
experiments/results/fashion-pilot/pilot_convergence_20ep.jsonl
experiments/results/fashion-pilot/pilot_convergence_20ep.log
experiments/results/fashion-pilot/pilot_timing_parallel3_2ep.jsonl
experiments/results/fashion-pilot/pilot_timing_parallel3_proc0.log
experiments/results/fashion-pilot/pilot_timing_parallel3_proc1.log
experiments/results/fashion-pilot/pilot_timing_parallel3_proc2.log
experiments/results/fashion-pilot/pilot_timing_sequential_2ep.jsonl
experiments/results/fashion-pilot/pilot_timing_sequential_2ep.log
experiments/results/fashion-pilot/summary.md
experiments/results/official-supabase-1-analysis/accuracy_curves.png
experiments/results/official-supabase-1-analysis/per_class_metrics.csv
pytest.ini
shared/fashion.py
---
116K	experiments/results/official-1-sql
156K	experiments/results/fashion-official-1-sql
168K	experiments/results/fashion-official-1-records
220K	tests/fixtures
280K	experiments/results/fashion-official-1-gradcam
280K	experiments/results/gradcam
316K	experiments/results/official-supabase-1-gradcam
348K	experiments/results/fashion-official-1-supabase-export.csv
372K	experiments/results/official-1-analysis
620K	experiments/results/official-supabase-1-analysis
912K	experiments/results/fashion-official-1-analysis
 29M	data
pilot_all_variants_3ep.jsonl
pilot_all_variants_3ep.log
pilot_convergence_20ep.jsonl
pilot_convergence_20ep.log
pilot_timing_parallel3_2ep.jsonl
pilot_timing_parallel3_proc0.log
pilot_timing_parallel3_proc1.log
pilot_timing_parallel3_proc2.log
pilot_timing_sequential_2ep.jsonl
pilot_timing_sequential_2ep.log
.gitignore:2:.env	.env
.gitignore:5:**/.streamlit/secrets.toml	ui/.streamlit/secrets.toml
.gitignore:24:/data/	data
.gitignore:20:.local_db.json	.local_db.json
scan
db/migrations/005_fashion_mnist_and_atomic_logging.md
db/migrations/005_fashion_mnist_and_atomic_logging.sql
db/migrations/005_verify.sql
experiments/migration_check.mjs
ai-usage/Emma-Session3.md
tests/test_ui_recent.py
Bash
IN
grep -noE ".{30}(eyJ[A-Za-z0-9_-]{20,}|sb_secret_|sbp_[a-z0-9]{20,}|service_role).{10}" ai-usage/Emma-Session3.md tests/test_ui_recent.py | cut -c1-140 | head -8; ANON=$(grep ^SUPABASE_ANON_KEY .env | cut -d= -f2 | cut -c1-25); grep -rl "$ANON" . --exclude-dir=.venv --exclude-dir=.git -I | head; git ls-files | grep -c "" ; git status --short | grep "^RM\|^R"; grep -rn "shared.data\|shared/data" --include=*.py --include=*.md --include=*.yaml . --exclude-dir=.venv --exclude-dir=ai-usage | head -3

OUT
ai-usage/Emma-Session3.md:3080:thenticated gets nothing, and service_role keeps its
ai-usage/Emma-Session3.md:3135:thenticated gets nothing, and service_role keeps its
ai-usage/Emma-Session3.md:5663:rIl -E 'eyJ[A-Za-z0-9_-]{20,}|sb_secret_|sb_publis
render.yaml
MODEL_CARD.md
pytest.ini
requirements-dev.txt
.gitignore
ui/requirements.txt
.env.example
README.md
ui/tryit.py
ui/evidence.py
105
RM shared/data.py -> shared/synthetic_data.py
(eval):1: no matches found: --include=*.py
Bash
IN
for k in SUPABASE_ANON_KEY SUPABASE_SERVICE_KEY; do V=$(grep ^$k .env | cut -d= -f2); echo $k len ${#V}; P=$(echo "$V" | cut -c45-80); grep -rlF "$P" . --exclude-dir=.venv --exclude-dir=.git -I | head; done; sed -n 5663p ai-usage/Emma-Session3.md | cut -c1-120

OUT
SUPABASE_ANON_KEY len 0
pytest.ini
.gitignore
render.yaml
ui/tryit.py
requirements-dev.txt
ui/concepts.py
ui/requirements.txt
MODEL_CARD.md
ui/evidence.py
.env.example
SUPABASE_SERVICE_KEY len 219
sed -n 255,290p api/db.py; sed -n 60,70p api/local_db.py; sed -n 30,135p tests/conftest.py; grep -n -E 'HTTPException|40
4:02 PM
No secrets in the tree: the service key's contents appear in no file, and .env, secrets.toml and data/ are all ignored. I'll re-run the full suite after the edits.



Bash
IN
source .venv/bin/activate && python -m pytest -q 2>&1 | tail -2; head -15 README.md | grep -c "http"; git status --short | grep -E "official-1-sql|\.local_db"; git diff --stat HEAD -- .gitignore | tail -1; git diff HEAD -- .gitignore | grep "^[+-]" | head; ls experiments/results/official-1-console.txt experiments/results/explore_*.jsonl; git ls-files experiments/results | head -5; git ls-files ui/assets tests/fixtures | head -3

OUT
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
293 passed, 6 skipped, 4 warnings in 101.36s (0:01:41)
2
?? experiments/results/fashion-official-1-sql/
 1 file changed, 4 insertions(+)
--- a/.gitignore
+++ b/.gitignore
+
+# Local dataset downloads and training artifacts
+/data/
+/pilot-output/
experiments/results/explore_budget.jsonl
experiments/results/explore_difficulty.jsonl
experiments/results/explore_difficulty2.jsonl
experiments/results/explore_techniques.jsonl
experiments/results/official-1-console.txt
experiments/results/explore_budget.jsonl
experiments/results/explore_difficulty.jsonl
experiments/results/explore_difficulty2.jsonl
experiments/results/explore_techniques.jsonl
experiments/results/gradcam/examples.csv
Bash
IN
grep -rln "fashion-pilot\|experiments.pilot\|pilot_summary\|experiments/pilot" . --exclude-dir=.venv --exclude-dir=.git --exclude-dir=ai-usage --exclude-dir=__pycache__ -I | head; git status --short | grep -E "^ M (db|api|shared|experiments)|^ D|^R" | head -30; ls db/migrations | head -20; git status --short | grep -E "^ M ai-usage|ai-usage"

OUT
MODEL_CARD.md
README.md
config/fashion_protocol.json
experiments/pilot.py
experiments/pilot_summary.py
docs/FASHION_PROTOCOL.md
 M api/db.py
 M api/experiments.py
 M api/local_db.py
 M api/main.py
 M api/training.py
 M db/seed.py
 M experiments/analyze.py
 M experiments/configs.py
 M experiments/gradcam_evidence.py
 M experiments/results/official-supabase-1-analysis/summary.md
 M experiments/run_experiments.py
 M experiments/sql_check.mjs
 M shared/schemas.py
RM shared/data.py -> shared/synthetic_data.py
 D tests/test_schema.py
 D tests/test_supabase_roundtrip.py
001_init.sql
002_experiment_tracking.md
002_experiment_tracking.sql
003_recent_classifications.md
003_recent_classifications_view.sql
003_verify.sql
004_predictions.md
004_predictions.sql
004_verify.sql
005_fashion_mnist_and_atomic_logging.md
005_fashion_mnist_and_atomic_logging.sql
005_verify.sql
 M ai-usage/Emma-Session3.md
Bash
IN
grep -n "pilot" README.md MODEL_CARD.md docs/FASHION_PROTOCOL.md config/fashion_protocol.json | cut -c1-230

OUT
README.md:79:experiments/           matched-run runner, pilot, analysis, Grad-CAM evidence, results/
README.md:117:python -m experiments.pilot --epochs 3 --batch-size 128 --lr 0.001 --out pilot.jsonl   # validation-only runtime pilot; never opens the test files
docs/FASHION_PROTOCOL.md:41:| Batch size | 128 | standard; measured cost in the pilot |
docs/FASHION_PROTOCOL.md:42:| Epochs | 20 | the 20-epoch validation pilot showed the baseline's validation loss flattening from about epoch 13; cosine decays over the same 20 epochs |
docs/FASHION_PROTOCOL.md:49:**No tuning toward a result.** The pilot looked only at training and validation numbers, and I deliberately did **not** change dropout, shift, learning rate or epochs after seeing them.
docs/FASHION_PROTOCOL.md:50:One pilot observation is recorded because it matters: over 20 epochs the full improved model ended with *lower* validation accuracy than the baseline (about 0.888 vs 0.908; training accuracy 0.889 vs 0.
docs/FASHION_PROTOCOL.md:52:## Measured cost on this Mac (validation-only pilot; no test data)
docs/FASHION_PROTOCOL.md:53:Apple M2, 8 cores, 16 GB, CPU only, 54,000 training images, batch 128. Raw data and per-epoch times: [`experiments/results/fashion-pilot/`](../experiments/results/fashion-pilot/summary.md).
docs/FASHION_PROTOCOL.md:54:The machine was shared with other applications during the pilot (load average about 9), so times are a realistic *busy* figure; single quiet runs were about 10–12 s/epoch.
docs/FASHION_PROTOCOL.md:59:| Spread of single epochs in the sequential pilots | about 10–19 s (one outlier 34 s) depending on variant and background load |
docs/FASHION_PROTOCOL.md:100:- Bitwise reproducibility across machines or thread counts is not guaranteed; in the pilot, 2-thread and 4-thread runs of the same seed gave identical epoch losses on this machine, but all official run
MODEL_CARD.md:76:- Nothing was tuned after looking at test results; the budget came from a validation-only pilot
MODEL_CARD.md:77:  (`experiments/results/fashion-pilot/`).
MODEL_CARD.md:109:  appears capacity-limited, so augmentation and dropout cost more than they save. This is consistent with the validation-only pilot, and
config/fashion_protocol.json:26:  "pilot": {
config/fashion_protocol.json:28:    "summary": "experiments/results/fashion-pilot/summary.md",
Bash
IN
python3 -I - <<'E'
import pathlib
f=pathlib.Path('docs/PROJECT_NOTES.md'); t=f.read_text()
o="Nothing was committed, pushed, deployed or applied to Supabase by the correction session. Migrations 004 and 005 are prepared, not applied."
assert o in t
f.write_text(t.replace(o,"Nothing was committed, pushed or deployed by the correction session. Migrations 004 and 005 were applied to Supabase by Emma and verified (live tests passed 2026-10-17); the deployed Render/Streamlit apps are still the earlier version."))
E
git diff --stat | tail -1

OUT
 46 files changed, 4935 insertions(+), 1868 deletions(-)
4:04 PM
I've done everything except the commit batches, and the README, provenance and report edits above are in the working tree. Nothing was retrained, committed, pushed, deployed or changed in the cloud.

1. Provenance and figures
Problem found: the export fashion-official-1-supabase-export.csv is genuine (meta row, PostgreSQL 17.11, executed 2026-10-17T22:49:57Z, all 10 queries). But fashion-official-1-sql/ and the analysis summary.md still carried the "LOCAL REPLICA" label, so the ingest had not been applied there.
Fix: I ingested it into a scratch directory first and compared. It matched the old files cell for cell (queries 1–7, 9, 10). I then copied it into fashion-official-1-sql/ and re-ran analyze. PROVENANCE.md now says supabase-sql-editor, and the UI shows that as its provenance caption.
Wording: I removed "provisional / local replica" from README, MODEL_CARD, both PART2_RESULTS copies, PROJECT_NOTES, FASHION_PROTOCOL, SUBMISSION_CHECKLIST, and both engineering reports. I also fixed PROJECT_NOTES, which wrongly said migrations 004/005 were "not applied".
Cross-check: I recomputed per-class precision and recall from the 18 stored confusion matrices. All 60 variant/class rows match per_class_metrics.csv to within 5e-5, which is rounding. All 1,800 confusion cells in q7 match the stored matrices. The records are 18 runs, ids 53–70, with 10,000 test, 54,000 train and 6,000 val images each, 20 epochs, dataset id 24.
2. Tests and dependencies
Tests: pytest gives 293 passed and 6 skipped, both before and after my edits. The skips are the live-Supabase tests, which write to the cloud and are off by default.
Dependencies: pip check finds no broken requirements. All pins are exact and the API and UI pins agree with each other.
Not run: I have no pip-audit, so I did not do a CVE scan.
3. Local app, run 54
Setup: I ran the real API code in-process against Supabase, reading only, with the log call stubbed so nothing was written.
API result: /runs/54 is full_improved-seed0, group fashion-official-1, accuracy 0.8782, dataset fashion-mnist. /classes returns 10 labels. Predicting a bundled dress and sneaker gave Dress at 0.994 and Sneaker at 0.976. Each returned 10 probabilities summing to 1, a top-5, and a 28×28 Grad-CAM.
UI: the full app rendered with the real anon key and no exceptions. It has five tabs and shows the final evidence, including the paired ablations, the curves and the per-class tables.
Recent Classifications: it loads the prediction-to-run history, but only 2 live-test rows exist (run 52). No run-54 prediction exists until the branch is deployed.
Not checked: I did not test Try It against a live API on a port.
4. Reports
Word counts: Emma's report has 968 words by wc (961 prose) and Komal's has 851 (843 prose). They are different files and both are inside 750–1,000.
Komal: her worldview reflection and AI-use section are still bracketed placeholders, as you asked.
5. README
Problem: the URL table started at line 17, so it failed the first-15-lines rule.
Fix: I added a link block at lines 3–6 with GitHub, Streamlit, Render, Supabase ref glcxpjaeobpppkizlcke (taken from your .env, which is a public identifier), and both reports. The two videos are marked ❌ MISSING: not yet recorded.
6. Commit batches
No secrets are in the tree. .env, secrets.toml and data/ (29 MB) are gitignored, and the service key appears in no file. Run these in order:


# 1 dataset correction, schema, API
git add -A -- .gitignore pytest.ini config tools shared api db
git commit -m "feat: switch to Fashion-MNIST, add predictions logging and migrations 004/005"

# 2 experiment protocol and tooling (no pilot files)
git add docs/comparison_queries.sql docs/fashion_official_1_export.sql docs/FASHION_PROTOCOL.md \
  experiments/*.mjs experiments/analyze.py experiments/cam_blank_check.py experiments/configs.py \
  experiments/export_records.py experiments/gradcam_evidence.py experiments/ingest_sql_export.py \
  experiments/package_sql.py experiments/run_experiments.py
git commit -m "feat: frozen Fashion-MNIST protocol, SQL export and ingest tooling"

# 3 UI
git add ui
git commit -m "feat: five-tab UI for Fashion-MNIST with top-5, evidence and run drill-down"

# 4 tests (deletions of test_schema.py and test_supabase_roundtrip.py included)
git add -A -- tests
git commit -m "test: Fashion-MNIST, calibration, evidence pipeline, API and UI smoke tests"

# 5 evidence
git add experiments/results/fashion-official-1-* experiments/results/official-supabase-1-analysis
git commit -m "docs: official Fashion-MNIST evidence (runs 53-70, Supabase SQL export)"

# 6 reports and docs
git add README.md MODEL_CARD.md PART2_RESULTS.md docs ai-usage/Emma-Session3.md
git commit -m "docs: README links, model card, results, engineering reports, ethics"
Each commit message needs the Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com> trailer. Batch 5's glob also matches the pilot-adjacent fashion-official-1-* paths only, so fashion-pilot/ stays out. docs/ already includes INSTRUCTOR_QUESTION_DRAFT.md, which is a draft that has not been sent.

The required small assets are included. ui/assets/samples holds 20 PNGs plus a manifest. tests/fixtures is about 220 KB, with __pycache__ ignored.

Remaining blockers
Videos: both are missing, so README and the checklist stay red until each person records one.
Komal's reflection: her worldview and AI-use sections are still placeholders. Her report has about 150 words of room.
Deployment: Render and Streamlit still serve the old synthetic version. Deploy, then do one real Try It prediction with run 54 and confirm the new row appears in Recent Classifications.
Pilot files: you asked to exclude them, but README, MODEL_CARD, FASHION_PROTOCOL and config/fashion_protocol.json link to experiments/results/fashion-pilot/summary.md. README also documents experiments/pilot.py. Leaving them out creates dangling links. My suggestion is to commit pilot.py, pilot_summary.py and fashion-pilot/summary.md, and exclude the .log and raw .jsonl files. Say if you'd rather drop the links instead.
Checklist items: SUBMISSION_CHECKLIST still has ⏳ items. The Grad-CAM failure interpretations in FASHION_GRADCAM_FAILURE_ANALYSIS.md are a draft that Emma or Komal must read and confirm. Grad-CAM is blank for 97.8% of Dress and 87.6% of Coat predictions.
Instructor question: INSTRUCTOR_QUESTION_DRAFT.md has not been sent.


Finalize the commit plan without committing yet.

- Include experiments/pilot.py, experiments/pilot_summary.py, and experiments/results/fashion-pilot/summary.md so documented evidence links resolve. Exclude temporary logs, downloaded data, and caches.
- Include any changed dependency files and CI configuration; they were missing from your listed batches.
- Keep related tests with their implementation where practical.
- Use exact file lists based on current git status rather than broad globs that could include unintended files.
- Do not treat the Claude co-author trailer as an assignment requirement. Follow repository instructions if they explicitly require it; preserve our AI-use records.
- Verify every intended changed file is assigned to a batch, and list intentionally excluded files.
- Keep missing videos, Komal’s placeholders, and unverified deployment checks marked incomplete.

Do not commit, push, deploy, or change cloud records. Give me the revised batches for GitHub Desktop.
Show more
Bash
IN
git status --short -uall | grep -v "__pycache__" ; echo ---; ls -a .github .github/workflows 2>&1 | head; git check-ignore -v tests/fixtures/__pycache__ ; git status --short -uall --ignored | grep '^!!' | grep -v __pycache__ | head; grep -rn "Co-Authored\|co-author" CLAUDE.md AGENTS.md .github 2>/dev/null | head -3

OUT
 M .gitignore
 M MODEL_CARD.md
 M PART2_RESULTS.md
 M README.md
 M ai-usage/Emma-Session3.md
 M api/db.py
 M api/experiments.py
 M api/local_db.py
 M api/main.py
 M api/training.py
 M db/seed.py
 M docs/PART1_HANDOFF.md
 M docs/PART2_PROTOCOL.md
 M docs/PART2_RESULTS.md
 M docs/RUN_API_LOCALLY.md
 M docs/comparison_queries.sql
 M experiments/analyze.py
 M experiments/configs.py
 M experiments/gradcam_evidence.py
 M experiments/results/official-supabase-1-analysis/summary.md
 M experiments/run_experiments.py
 M experiments/sql_check.mjs
 M shared/schemas.py
RM shared/data.py -> shared/synthetic_data.py
 M tests/conftest.py
 M tests/test_augmentation_safety.py
 M tests/test_deploy_readiness.py
 M tests/test_experiment_records.py
 M tests/test_experiment_runner.py
 M tests/test_gradcam_loading.py
 M tests/test_healthz.py
 M tests/test_local_db.py
 M tests/test_model_variants.py
 M tests/test_reproducibility.py
 D tests/test_schema.py
 D tests/test_supabase_roundtrip.py
 M tests/test_training.py
 M tests/test_ui_concepts_and_card.py
 M tests/test_ui_evidence.py
 M tests/test_ui_recent.py
 M tests/test_ui_tryit.py
 M ui/app.py
 M ui/concepts.py
 M ui/evidence.py
 M ui/recent.py
 M ui/tryit.py
?? api/calibration.py
?? config/fashion_protocol.json
?? config/served_model.json
?? db/migrations/004_predictions.md
?? db/migrations/004_predictions.sql
?? db/migrations/004_verify.sql
?? db/migrations/005_fashion_mnist_and_atomic_logging.md
?? db/migrations/005_fashion_mnist_and_atomic_logging.sql
?? db/migrations/005_verify.sql
?? docs/CONTRIBUTIONS.md
?? docs/DATASET_CARD.md
?? docs/DATASET_CARD_HISTORICAL_SYNTHETIC.md
?? docs/EMMA_ENGINEERING_REPORT.md
?? docs/ETHICS_AND_DISCLOSURE.md
?? docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md
?? docs/FASHION_PROTOCOL.md
?? docs/INSTRUCTOR_QUESTION_DRAFT.md
?? docs/KOMAL_ENGINEERING_REPORT.md
?? docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md
?? docs/PROJECT_NOTES.md
?? docs/SUBMISSION_CHECKLIST.md
?? docs/fashion_official_1_export.sql
?? experiments/cam_blank_check.py
?? experiments/export_records.py
?? experiments/ingest_sql_export.py
?? experiments/migration_check.mjs
?? experiments/package_sql.py
?? experiments/pilot.py
?? experiments/pilot_summary.py
?? experiments/results/fashion-official-1-analysis/accuracy_curves.png
?? experiments/results/fashion-official-1-analysis/calibration.csv
?? experiments/results/fashion-official-1-analysis/confusion_matrices.png
?? experiments/results/fashion-official-1-analysis/loss_curves.png
?? experiments/results/fashion-official-1-analysis/per_class_metrics.csv
?? experiments/results/fashion-official-1-analysis/per_class_precision_recall.png
?? experiments/results/fashion-official-1-analysis/summary.md
?? experiments/results/fashion-official-1-analysis/test_metrics_by_seed.png
?? experiments/results/fashion-official-1-gradcam/blank_cam_by_class.csv
?? experiments/results/fashion-official-1-gradcam/examples.csv
?? experiments/results/fashion-official-1-gradcam/gradcam_correct.png
?? experiments/results/fashion-official-1-gradcam/gradcam_incorrect.png
?? experiments/results/fashion-official-1-gradcam/source.json
?? experiments/results/fashion-official-1-records/runs.json
?? experiments/results/fashion-official-1-runs.jsonl
?? experiments/results/fashion-official-1-sql/PROVENANCE.md
?? experiments/results/fashion-official-1-sql/q1.csv
?? experiments/results/fashion-official-1-sql/q10.csv
?? experiments/results/fashion-official-1-sql/q2.csv
?? experiments/results/fashion-official-1-sql/q3.csv
?? experiments/results/fashion-official-1-sql/q4.csv
?? experiments/results/fashion-official-1-sql/q5.csv
?? experiments/results/fashion-official-1-sql/q6.csv
?? experiments/results/fashion-official-1-sql/q7.csv
?? experiments/results/fashion-official-1-sql/q8.csv
?? experiments/results/fashion-official-1-sql/q9.csv
?? experiments/results/fashion-official-1-supabase-export.csv
?? experiments/results/fashion-pilot/pilot_all_variants_3ep.jsonl
?? experiments/results/fashion-pilot/pilot_all_variants_3ep.log
?? experiments/results/fashion-pilot/pilot_convergence_20ep.jsonl
?? experiments/results/fashion-pilot/pilot_convergence_20ep.log
?? experiments/results/fashion-pilot/pilot_timing_parallel3_2ep.jsonl
?? experiments/results/fashion-pilot/pilot_timing_parallel3_proc0.log
?? experiments/results/fashion-pilot/pilot_timing_parallel3_proc1.log
?? experiments/results/fashion-pilot/pilot_timing_parallel3_proc2.log
?? experiments/results/fashion-pilot/pilot_timing_sequential_2ep.jsonl
?? experiments/results/fashion-pilot/pilot_timing_sequential_2ep.log
?? experiments/results/fashion-pilot/summary.md
?? experiments/results/official-supabase-1-analysis/accuracy_curves.png
?? experiments/results/official-supabase-1-analysis/per_class_metrics.csv
?? pytest.ini
?? shared/fashion.py
?? tests/fashion_fixture.py
?? tests/fixtures/__init__.py
?? tests/fixtures/make_reference.py
?? tests/fixtures/reference_expected.json
?? tests/fixtures/reference_weights.json
?? tests/test_api_predict.py
?? tests/test_calibration.py
?? tests/test_evidence_pipeline.py
?? tests/test_fashion_data.py
?? tests/test_live_supabase.py
?? tests/test_protocol_config.py
?? tests/test_reference_prediction.py
?? tests/test_ui_app_smoke.py
?? tools/export_samples.py
?? tools/fetch_fashion_mnist.py
?? ui/assets/samples/0-t-shirt-top-1.png
?? ui/assets/samples/0-t-shirt-top-2.png
?? ui/assets/samples/1-trouser-1.png
?? ui/assets/samples/1-trouser-2.png
?? ui/assets/samples/2-pullover-1.png
?? ui/assets/samples/2-pullover-2.png
?? ui/assets/samples/3-dress-1.png
?? ui/assets/samples/3-dress-2.png
?? ui/assets/samples/4-coat-1.png
?? ui/assets/samples/4-coat-2.png
?? ui/assets/samples/5-sandal-1.png
?? ui/assets/samples/5-sandal-2.png
?? ui/assets/samples/6-shirt-1.png
?? ui/assets/samples/6-shirt-2.png
?? ui/assets/samples/7-sneaker-1.png
?? ui/assets/samples/7-sneaker-2.png
?? ui/assets/samples/8-bag-1.png
?? ui/assets/samples/8-bag-2.png
?? ui/assets/samples/9-ankle-boot-1.png
?? ui/assets/samples/9-ankle-boot-2.png
?? ui/assets/samples/manifest.json
---
.github:
.
..
workflows

.github/workflows:
.
..
ci.yml
.gitignore:9:__pycache__/	tests/fixtures/__pycache__
!! .env
!! .pytest_cache/.gitignore
!! .pytest_cache/CACHEDIR.TAG
!! .pytest_cache/README.md
!! .pytest_cache/v/cache/lastfailed
!! .pytest_cache/v/cache/nodeids
!! .pytest_cache/v/cache/stepwise
!! .venv/bin/Activate.ps1
!! .venv/bin/activate
!! .venv/bin/activate.csh
Bash
IN
cat .github/workflows/ci.yml; git status --short -- requirements-dev.txt api/requirements.txt ui/requirements.txt render.yaml .github .env.example ui/.streamlit/secrets.toml.example; git ls-files ui/.streamlit .env.example; grep -rn "shared.data\|shared/data" . --include=*.py --include=*.yml --include=*.yaml --include=*.md --exclude-dir=.venv --exclude-dir=ai-usage -l | head

OUT
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: |
            api/requirements.txt
            ui/requirements.txt
            requirements-dev.txt
      - name: Install (CPU-only torch via api/requirements.txt)
        run: pip install -r requirements-dev.txt
      - name: Dependency consistency
        run: pip check
      - name: Tests (no Supabase credentials, so the live round-trip test skips)
        run: pytest -q
.env.example
ui/.streamlit/secrets.toml.example
(eval):1: no matches found: --include=*.py
4:13 PM
No dependency or CI files show as changed. One CI risk: data/ is gitignored, so I'll check whether the tests need it.



Bash
IN
grep -rn "data/\|fashion-mnist\"\|FASHION_DATA\|DATA_DIR\|skipif\|importorskip" tests/*.py shared/fashion.py pytest.ini | grep -iE "data_dir|data/|skip|environ" | head -15; ls data/fashion-mnist | head

OUT
tests/test_deploy_readiness.py:54:@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs a git checkout")
tests/test_fashion_data.py:101:@pytest.mark.skipif(not (REAL / "train-images-idx3-ubyte.gz").is_file(),
tests/test_live_supabase.py:30:    pytest.mark.skipif(
tests/test_live_supabase.py:35:    pytest.mark.skipif(
tests/test_ui_concepts_and_card.py:60:    torch = pytest.importorskip("torch")
tests/test_evidence_pipeline.py:20:@pytest.mark.skipif(not RECORDS.is_file(), reason="official records not exported")
tests/test_ui_app_smoke.py:13:streamlit_testing = pytest.importorskip("streamlit.testing.v1")
shared/fashion.py:54:    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
4:13 PM
Real-data tests skip when data/ is absent, so CI is safe. I'll generate the batch lists from current git status and verify full coverage by script.



Bash
IN
python3 -I - <<'E'
import subprocess,re
out=subprocess.run(["git","status","--short","-uall"],capture_output=True,text=True).stdout.splitlines()
files=[]
for l in out:
    p=l[3:]
    if " -> " in p: old,new=p.split(" -> "); files+= [old,new]
    else: files.append(p)
files=[f for f in files if "__pycache__" not in f]
excl=lambda f: f.startswith("experiments/results/fashion-pilot/") and f!="experiments/results/fashion-pilot/summary.md"
excluded=[f for f in files if excl(f)]
files=[f for f in files if not excl(f)]
B={i:[] for i in range(1,6)}
def b(f):
    if f.startswith("tests/"):
        n=f.split("/")[-1]
        if n.startswith("test_ui_"): return 3
        if n in("test_experiment_runner.py","test_experiment_records.py","test_evidence_pipeline.py"): return 2
        return 1
    if f.startswith("ui/"): return 3
    if f.startswith("experiments/results/"): return 2 if "fashion-pilot" in f else 4
    if f.startswith("experiments/"): return 2
    if f in("docs/comparison_queries.sql","docs/fashion_official_1_export.sql","docs/FASHION_PROTOCOL.md"): return 2
    if f.startswith(("docs/","ai-usage/")) or f in("README.md","MODEL_CARD.md","PART2_RESULTS.md"): return 5
    return 1
for f in files: B[b(f)].append(f)
names={1:"Fashion-MNIST data, schema, migrations and API (with tests)",2:"Frozen protocol, SQL export tooling and pilot (with tests)",3:"Streamlit UI and sample images (with tests)",4:"Official Fashion-MNIST evidence",5:"README, model card, results and reports"}
tot=0
for i in B:
    print(f"\n=== {i}: {names[i]} ({len(B[i])})"); tot+=len(B[i])
    print("\n".join(sorted(B[i])))
print("\nTOTAL",tot,"of",len(files),"excluded",len(excluded)); print("\n".join(excluded))
assert tot==len(files)==len(set(sum(B.values(),[])))
E

OUT
=== 1: Fashion-MNIST data, schema, migrations and API (with tests) (45)
.gitignore
api/calibration.py
api/db.py
api/experiments.py
api/local_db.py
api/main.py
api/training.py
config/fashion_protocol.json
config/served_model.json
db/migrations/004_predictions.md
db/migrations/004_predictions.sql
db/migrations/004_verify.sql
db/migrations/005_fashion_mnist_and_atomic_logging.md
db/migrations/005_fashion_mnist_and_atomic_logging.sql
db/migrations/005_verify.sql
db/seed.py
pytest.ini
shared/data.py
shared/fashion.py
shared/schemas.py
shared/synthetic_data.py
tests/conftest.py
tests/fashion_fixture.py
tests/fixtures/__init__.py
tests/fixtures/make_reference.py
tests/fixtures/reference_expected.json
tests/fixtures/reference_weights.json
tests/test_api_predict.py
tests/test_augmentation_safety.py
tests/test_calibration.py
tests/test_deploy_readiness.py
tests/test_fashion_data.py
tests/test_gradcam_loading.py
tests/test_healthz.py
tests/test_live_supabase.py
tests/test_local_db.py
tests/test_model_variants.py
tests/test_protocol_config.py
tests/test_reference_prediction.py
tests/test_reproducibility.py
tests/test_schema.py
tests/test_supabase_roundtrip.py
tests/test_training.py
tools/export_samples.py
tools/fetch_fashion_mnist.py

=== 2: Frozen protocol, SQL export tooling and pilot (with tests) (19)
docs/FASHION_PROTOCOL.md
docs/comparison_queries.sql
docs/fashion_official_1_export.sql
experiments/analyze.py
experiments/cam_blank_check.py
experiments/configs.py
experiments/export_records.py
experiments/gradcam_evidence.py
experiments/ingest_sql_export.py
experiments/migration_check.mjs
experiments/package_sql.py
experiments/pilot.py
experiments/pilot_summary.py
experiments/results/fashion-pilot/summary.md
experiments/run_experiments.py
experiments/sql_check.mjs
tests/test_evidence_pipeline.py
tests/test_experiment_records.py
tests/test_experiment_runner.py

=== 3: Streamlit UI and sample images (with tests) (31)
tests/test_ui_app_smoke.py
tests/test_ui_concepts_and_card.py
tests/test_ui_evidence.py
tests/test_ui_recent.py
tests/test_ui_tryit.py
ui/app.py
ui/assets/samples/0-t-shirt-top-1.png
ui/assets/samples/0-t-shirt-top-2.png
ui/assets/samples/1-trouser-1.png
ui/assets/samples/1-trouser-2.png
ui/assets/samples/2-pullover-1.png
ui/assets/samples/2-pullover-2.png
ui/assets/samples/3-dress-1.png
ui/assets/samples/3-dress-2.png
ui/assets/samples/4-coat-1.png
ui/assets/samples/4-coat-2.png
ui/assets/samples/5-sandal-1.png
ui/assets/samples/5-sandal-2.png
ui/assets/samples/6-shirt-1.png
ui/assets/samples/6-shirt-2.png
ui/assets/samples/7-sneaker-1.png
ui/assets/samples/7-sneaker-2.png
ui/assets/samples/8-bag-1.png
ui/assets/samples/8-bag-2.png
ui/assets/samples/9-ankle-boot-1.png
ui/assets/samples/9-ankle-boot-2.png
ui/assets/samples/manifest.json
ui/concepts.py
ui/evidence.py
ui/recent.py
ui/tryit.py

=== 4: Official Fashion-MNIST evidence (30)
experiments/results/fashion-official-1-analysis/accuracy_curves.png
experiments/results/fashion-official-1-analysis/calibration.csv
experiments/results/fashion-official-1-analysis/confusion_matrices.png
experiments/results/fashion-official-1-analysis/loss_curves.png
experiments/results/fashion-official-1-analysis/per_class_metrics.csv
experiments/results/fashion-official-1-analysis/per_class_precision_recall.png
experiments/results/fashion-official-1-analysis/summary.md
experiments/results/fashion-official-1-analysis/test_metrics_by_seed.png
experiments/results/fashion-official-1-gradcam/blank_cam_by_class.csv
experiments/results/fashion-official-1-gradcam/examples.csv
experiments/results/fashion-official-1-gradcam/gradcam_correct.png
experiments/results/fashion-official-1-gradcam/gradcam_incorrect.png
experiments/results/fashion-official-1-gradcam/source.json
experiments/results/fashion-official-1-records/runs.json
experiments/results/fashion-official-1-runs.jsonl
experiments/results/fashion-official-1-sql/PROVENANCE.md
experiments/results/fashion-official-1-sql/q1.csv
experiments/results/fashion-official-1-sql/q10.csv
experiments/results/fashion-official-1-sql/q2.csv
experiments/results/fashion-official-1-sql/q3.csv
experiments/results/fashion-official-1-sql/q4.csv
experiments/results/fashion-official-1-sql/q5.csv
experiments/results/fashion-official-1-sql/q6.csv
experiments/results/fashion-official-1-sql/q7.csv
experiments/results/fashion-official-1-sql/q8.csv
experiments/results/fashion-official-1-sql/q9.csv
experiments/results/fashion-official-1-supabase-export.csv
experiments/results/official-supabase-1-analysis/accuracy_curves.png
experiments/results/official-supabase-1-analysis/per_class_metrics.csv
experiments/results/official-supabase-1-analysis/summary.md

=== 5: README, model card, results and reports (19)
MODEL_CARD.md
PART2_RESULTS.md
README.md
ai-usage/Emma-Session3.md
docs/CONTRIBUTIONS.md
docs/DATASET_CARD.md
docs/DATASET_CARD_HISTORICAL_SYNTHETIC.md
docs/EMMA_ENGINEERING_REPORT.md
docs/ETHICS_AND_DISCLOSURE.md
docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md
docs/INSTRUCTOR_QUESTION_DRAFT.md
docs/KOMAL_ENGINEERING_REPORT.md
docs/PART1_HANDOFF.md
docs/PART2_PROTOCOL.md
docs/PART2_RESULTS.md
docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md
docs/PROJECT_NOTES.md
docs/RUN_API_LOCALLY.md
docs/SUBMISSION_CHECKLIST.md

TOTAL 144 of 144 excluded 10
experiments/results/fashion-pilot/pilot_all_variants_3ep.jsonl
experiments/results/fashion-pilot/pilot_all_variants_3ep.log
experiments/results/fashion-pilot/pilot_convergence_20ep.jsonl
experiments/results/fashion-pilot/pilot_convergence_20ep.log
experiments/results/fashion-pilot/pilot_timing_parallel3_2ep.jsonl
experiments/results/fashion-pilot/pilot_timing_parallel3_proc0.log
experiments/results/fashion-pilot/pilot_timing_parallel3_proc1.log
experiments/results/fashion-pilot/pilot_timing_parallel3_proc2.log
experiments/results/fashion-pilot/pilot_timing_sequential_2ep.jsonl
experiments/results/fashion-pilot/pilot_timing_sequential_2ep.log
Bash
IN
grep -n "\.log\|\.jsonl\|](" experiments/results/fashion-pilot/summary.md | head

OUT
20:File: `pilot_all_variants_3ep.jsonl`; 6 variant-runs; epochs 3, batch 128, lr 0.001, 54000 train / 6000 validation images, torch threads 4.
35:File: `pilot_convergence_20ep.jsonl`; 2 variant-runs; epochs 20, batch 128, lr 0.001, 54000 train / 6000 validation images, torch threads 4.
46:File: `pilot_timing_sequential_2ep.jsonl`; 6 variant-runs; epochs 2, batch 128, lr 0.001, 54000 train / 6000 validation images, torch threads 4.
61:File: `pilot_timing_parallel3_2ep.jsonl`; 6 variant-runs; epochs 2, batch 128, lr 0.001, 54000 train / 6000 validation images, torch threads 2.
4:13 PM
I've worked out the final six-batch plan. Nothing was committed, pushed, deployed or changed in the cloud.

Coverage check: a script compared the batch lists to git status -uall. All 144 changed or new files are in exactly one batch, with none duplicated or missing. Another 10 files are deliberately excluded, listed at the end.

Dependency and CI files: none changed. api/requirements.txt, ui/requirements.txt, requirements-dev.txt, render.yaml and .github/workflows/ci.yml are untouched, so no batch has anything to add for them. The CI workflow runs pip install -r requirements-dev.txt, pip check and pytest -q. Tests that need the real Fashion-MNIST download skip when data/ is absent, and the live-Supabase tests skip without credentials, so CI should not need either. The new pytest.ini is in batch 1.

Commit trailer: I found no CLAUDE.md, AGENTS.md or .github file that requires a co-author line, so none of the messages below carry one. Your AI-use records are untouched: ai-usage/Emma-Session3.md is in batch 5 and Komal's two session files are unchanged.

Pilot summary: fashion-pilot/summary.md mentions the excluded .jsonl files by name but does not link to them, so it still resolves in the repo. If you'd rather those raw .jsonl files be in the repo, they're the four pilot_*.jsonl files in the excluded list. The .log files are the ones I'd keep out.

Rename: in batch 1, check that shared/data.py (deleted) and shared/synthetic_data.py (added) are both ticked, so GitHub Desktop records it as a rename. Also tick the two deleted tests, tests/test_schema.py and tests/test_supabase_roundtrip.py.

Intermediate commits: I only ran the full test suite on the final tree, not at each intermediate commit. Batches 1 to 5 may not pass pytest in isolation, for example batch 3 depends on the evidence files from batch 4. If you want every commit green, squash them or push once after batch 5.

Batch 1: Fashion-MNIST data, schema, migrations and API (with tests), 45 files
Message: feat: switch to Fashion-MNIST with predictions logging and migrations 004/005


.gitignore
pytest.ini
api/calibration.py
api/db.py
api/experiments.py
api/local_db.py
api/main.py
api/training.py
config/fashion_protocol.json
config/served_model.json
db/migrations/004_predictions.md
db/migrations/004_predictions.sql
db/migrations/004_verify.sql
db/migrations/005_fashion_mnist_and_atomic_logging.md
db/migrations/005_fashion_mnist_and_atomic_logging.sql
db/migrations/005_verify.sql
db/seed.py
shared/data.py            (deleted/renamed)
shared/synthetic_data.py  (renamed)
shared/fashion.py
shared/schemas.py
tools/export_samples.py
tools/fetch_fashion_mnist.py
tests/conftest.py
tests/fashion_fixture.py
tests/fixtures/__init__.py
tests/fixtures/make_reference.py
tests/fixtures/reference_expected.json
tests/fixtures/reference_weights.json
tests/test_api_predict.py
tests/test_augmentation_safety.py
tests/test_calibration.py
tests/test_deploy_readiness.py
tests/test_fashion_data.py
tests/test_gradcam_loading.py
tests/test_healthz.py
tests/test_live_supabase.py
tests/test_local_db.py
tests/test_model_variants.py
tests/test_protocol_config.py
tests/test_reference_prediction.py
tests/test_reproducibility.py
tests/test_schema.py             (deleted)
tests/test_supabase_roundtrip.py (deleted)
tests/test_training.py
Batch 2: Frozen protocol, SQL export tooling and pilot (with tests), 19 files
Message: feat: frozen Fashion-MNIST protocol, SQL export/ingest tooling and validation pilot


docs/FASHION_PROTOCOL.md
docs/comparison_queries.sql
docs/fashion_official_1_export.sql
experiments/analyze.py
experiments/cam_blank_check.py
experiments/configs.py
experiments/export_records.py
experiments/gradcam_evidence.py
experiments/ingest_sql_export.py
experiments/migration_check.mjs
experiments/package_sql.py
experiments/pilot.py
experiments/pilot_summary.py
experiments/results/fashion-pilot/summary.md
experiments/run_experiments.py
experiments/sql_check.mjs
tests/test_evidence_pipeline.py
tests/test_experiment_records.py
tests/test_experiment_runner.py
Batch 3: Streamlit UI and sample images (with tests), 31 files
Message: feat: five-tab Fashion-MNIST UI with top-5, saved evidence and run drill-down


ui/app.py
ui/concepts.py
ui/evidence.py
ui/recent.py
ui/tryit.py
ui/assets/samples/ (all 20 PNGs and manifest.json)
tests/test_ui_app_smoke.py
tests/test_ui_concepts_and_card.py
tests/test_ui_evidence.py
tests/test_ui_recent.py
tests/test_ui_tryit.py
The ui/assets/samples/ folder holds 0-t-shirt-top-1.png through 9-ankle-boot-2.png (two per class) plus manifest.json. In GitHub Desktop, tick the whole folder, which is 21 files.

Batch 4: Official Fashion-MNIST evidence, 30 files
Message: docs: official Fashion-MNIST evidence (runs 53-70, Supabase SQL export)


experiments/results/fashion-official-1-analysis/ (8 files: 4 PNG, calibration.csv, per_class_metrics.csv, summary.md, per_class_precision_recall.png is one of the 4 PNGs)
experiments/results/fashion-official-1-gradcam/ (5 files)
experiments/results/fashion-official-1-records/runs.json
experiments/results/fashion-official-1-runs.jsonl
experiments/results/fashion-official-1-sql/ (PROVENANCE.md, q1 to q10)
experiments/results/fashion-official-1-supabase-export.csv
experiments/results/official-supabase-1-analysis/accuracy_curves.png
experiments/results/official-supabase-1-analysis/per_class_metrics.csv
experiments/results/official-supabase-1-analysis/summary.md
This folder list is exact: every file under those three fashion-official-1-* folders is intended, 8 + 5 + 10 files. The official-supabase-1-analysis entries are historical synthetic-run files that were modified.

Batch 5: README, model card, results and reports, 19 files
Message: docs: README links, model card, results, engineering reports and ethics


README.md
MODEL_CARD.md
PART2_RESULTS.md
ai-usage/Emma-Session3.md
docs/CONTRIBUTIONS.md
docs/DATASET_CARD.md
docs/DATASET_CARD_HISTORICAL_SYNTHETIC.md
docs/EMMA_ENGINEERING_REPORT.md
docs/ETHICS_AND_DISCLOSURE.md
docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md
docs/INSTRUCTOR_QUESTION_DRAFT.md
docs/KOMAL_ENGINEERING_REPORT.md
docs/PART1_HANDOFF.md
docs/PART2_PROTOCOL.md
docs/PART2_RESULTS.md
docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md
docs/PROJECT_NOTES.md
docs/RUN_API_LOCALLY.md
docs/SUBMISSION_CHECKLIST.md
This batch is six batches' worth of tracking, so there is one batch fewer than I sketched earlier (5 total, not 6). Two items in it are not yet complete:

docs/INSTRUCTOR_QUESTION_DRAFT.md is a draft that has not been sent.
docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md is a draft for human review.
Intentionally excluded (10 files)
All of these are under experiments/results/fashion-pilot/:

pilot_all_variants_3ep.jsonl and .log
pilot_convergence_20ep.jsonl and .log
pilot_timing_parallel3_2ep.jsonl
pilot_timing_parallel3_proc0.log, proc1.log and proc2.log
pilot_timing_sequential_2ep.jsonl and .log
Also not in any batch, because they are already gitignored: .env, ui/.streamlit/secrets.toml, data/ (the 29 MB dataset download), .local_db.json, .venv/, .pytest_cache/ and every __pycache__/. I found no secrets in the tree.

Still incomplete (committed as such, not hidden)
Videos: both are marked ❌ MISSING in README lines 3–6.
Komal's report: her worldview reflection and AI-use sections are still bracketed placeholders.
Deployment: Render and Streamlit still serve the old synthetic version. I did not check Try It against the deployed API, and no run-54 prediction exists in Recent Classifications yet. SUBMISSION_CHECKLIST.md still shows ⏳ items for this.