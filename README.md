# See-Sense — Fashion-MNIST CNN Classifier with Grad-CAM (three clouds)

- **GitHub:** https://github.com/emmabellerogo/see-sense · **Streamlit Cloud:** https://see-sense-glpk38qzwsf9hhvnlkt798.streamlit.app/
- **Render API:** https://see-sense-api-rhkb.onrender.com (`/healthz`, `/docs`) · **Supabase project ref:** `glcxpjaeobpppkizlcke`
- **Videos:** Emma — ❌ *MISSING: not yet recorded* · Komal — ❌ *MISSING: not yet recorded*
- **Engineering reports:** [Emma](docs/EMMA_ENGINEERING_REPORT.md) · [Komal](docs/KOMAL_ENGINEERING_REPORT.md) · Contributions: [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md)

**Streamlit UI ⇄ FastAPI model API ⇄ Supabase data.** A small PyTorch CNN classifies 28×28 grayscale pictures of clothing into the
ten **Fashion-MNIST** categories (T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot), shows its
five most probable classes, and returns a **Grad-CAM** heatmap showing where the network looked. It is a teaching and evaluation
project, not a general photo classifier.

> **Status (draft, branch `correction/fashion-mnist`).** The code on this branch is the Fashion-MNIST version and the official experiment has
> been run (group `fashion-official-1`, 18 runs, ids 53-70). The apps currently deployed at the URLs below are still the *earlier
> synthetic-shape version* (from `main`) until this branch is deployed. The result tables were produced by the
> project's SQL queries run in the Supabase SQL editor (export ingested 2026-10-17; see `experiments/results/fashion-official-1-sql/PROVENANCE.md`). See
> [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md) and [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md).

## Live deployment URLs

| Tier | Platform | URL |
|------|----------|-----|
| UI | Streamlit Community Cloud | https://see-sense-glpk38qzwsf9hhvnlkt798.streamlit.app/ |
| API | Render.com | https://see-sense-api-rhkb.onrender.com *(free tier sleeps; first call takes ~30–60 s; health check: `/healthz`)* |
| Data | Supabase | `https://glcxpjaeobpppkizlcke.supabase.co` (project ref `glcxpjaeobpppkizlcke`) |

## The five tabs (`ui/app.py`)

1. **Concepts** — convolution, stride/padding/output-size arithmetic with a worked example, receptive-field growth (three stacked 3×3 layers: 3 → 5 → 7), and the Grad-CAM derivation and limits, using this project's 10-class network (9,098 parameters for the baseline).
2. **Try It** — classify an uploaded image or one of 20 bundled *held-out* clothing examples (the first two official-test images of each class). Shows the prediction, **top-five probabilities**, the input and Grad-CAM, and whether the prediction was logged. The model comes from [config/served_model.json](config/served_model.json) (fixed in advance, checked against the saved experiment records and verified against the API) — never a hard-coded run ID. Handles cold starts and errors; prediction requests are never retried automatically.
3. **Compare Models** — the final Fashion-MNIST comparison (baseline, full improved and four leave-one-technique-out variants: tables, training/validation curves, confusion matrices, per-class precision/recall, calibration, Grad-CAM gallery) **once its evidence files exist** — until then the tab says plainly that they are not available. The earlier synthetic-shape experiment is shown separately in a clearly labelled historical section.
4. **Recent Classifications** — reads the latest ≤200 rows of the hash-free `recent_predictions` view with the anon key, shows a class-distribution histogram, and lets you **drill down from a classification to the run that produced it**.
5. **Model Card** — [MODEL_CARD.md](MODEL_CARD.md).

Training is not exposed in the UI. It is available through the API (`POST /train`, run from a local machine that has the dataset) and the experiment tools in `experiments/`.

## Results

**Fashion-MNIST (official, 3 seeds × 6 variants, test set, mean ± sd):** no_augmentation 90.30% ± 0.13, baseline 89.73% ± 0.47, no_dropout 88.94% ± 0.07,
no_scheduler 88.23% ± 0.40, **full_improved 88.10% ± 0.29**, no_batchnorm 87.12% ± 0.30. **The improved model did not beat the baseline**: only batch
normalization helped; augmentation and dropout lowered accuracy at this 20-epoch budget. Served model (fixed in advance): `full_improved-seed0`, run 54, 87.82%.
Details, intervals, per-class results, calibration and Grad-CAM: [MODEL_CARD.md](MODEL_CARD.md), [PART2_RESULTS.md](PART2_RESULTS.md), individual engineering reports [Emma](docs/EMMA_ENGINEERING_REPORT.md) and [Komal](docs/KOMAL_ENGINEERING_REPORT.md),
[experiments/results/fashion-official-1-analysis/summary.md](experiments/results/fashion-official-1-analysis/summary.md). Protocol: [docs/FASHION_PROTOCOL.md](docs/FASHION_PROTOCOL.md) (frozen).

**Historical (not Fashion-MNIST):** the first version of the project was built on four generated shapes (group `official-supabase-1`, 2,000 images, 30 runs). That experiment is kept for transparency in [experiments/results/official-supabase-1-analysis/summary.md](experiments/results/official-supabase-1-analysis/summary.md), [docs/SUPABASE_GRADCAM_FAILURE_ANALYSIS.md](docs/SUPABASE_GRADCAM_FAILURE_ANALYSIS.md) and [docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md](docs/PART2_RESULTS_HISTORICAL_SYNTHETIC.md). Its numbers say nothing about Fashion-MNIST.

**Known limitation (distribution shift).** The model expects Fashion-MNIST-style images: a centered, light clothing item on a dark background.
Ordinary photos, especially with light backgrounds, can receive incorrect high-confidence predictions; in our supplementary probes (10 bundled test
images, run 54, run after the official evaluation, not a benchmark result) a white background pushed most images to **Bag**. The app does not invert,
crop or reject uploads. Use the bundled held-out examples (Try It, Option B) to see expected behaviour. Details: [MODEL_CARD.md](MODEL_CARD.md) → Limitations.

## Data: persisted vs not persisted

| Persisted in Supabase | Never persisted |
|---|---|
| Dataset *identity* (name, source, class names, file hashes) | Image pixels (Fashion-MNIST files stay on the training machine) |
| Run metrics, settings, per-epoch training/validation loss and accuracy, confusion matrices, per-class test metrics, calibration measures | Uploaded files (processed in memory only) |
| Model checkpoints (`run_artifacts`, not readable with the anon key) | Grad-CAM images |
| Per prediction, written together in one transaction: SHA-256 hash, width, height, label, class, confidence (`image_metadata`) **and** hash, label, full probability vector, `served_by_run_id`, timestamp (`predictions`) — neither readable with the anon key | Service-role key in the UI |

**Why raw images are not persisted:** the hash lets us audit and de-duplicate predictions without keeping what people uploaded; uploads can contain faces, documents or anything else, and the project has no need to keep them.

The anon key can read only `runs` (policy in 001) and two views: `recent_classifications` (003: `id, run_id, class_name, confidence, created_at`) and `recent_predictions` (004: predictions joined to their run, **no image hash**). All writes go through the API with the service-role key. The API's public `GET /predictions` also omits hashes.

## Architecture

```
Streamlit Cloud (ui/app.py)  ──HTTPS──►  FastAPI on Render (api/main.py)  ──service-role──►  Supabase Postgres
 thin client, no torch                    CNN + Grad-CAM + temperature, only writer           datasets · runs · run_artifacts · image_metadata · predictions
        └──── anon key, read-only: runs, recent_predictions, recent_classifications ──────────────────────────►

Training machine (laptop):  tools/fetch_fashion_mnist.py → local Fashion-MNIST files → experiments/ + local API (POST /train) → Supabase
```

## Project structure

```
MODEL_CARD.md          model card (read by the UI)        render.yaml   Render blueprint
config/                served_model.json (model Try It serves), fashion_protocol.json (frozen experiment protocol)
shared/                fashion.py (dataset loader, split, identity), schemas.py, synthetic_data.py (historical)
api/                   FastAPI tier: main, training (CNN + Grad-CAM + checkpoints), calibration, db, experiments
ui/                    Streamlit tier: app.py + concepts/evidence/recent/tryit modules, assets/samples, requirements.txt
db/migrations/         001_init … 005 (+ *_verify.sql, *.md); 004 and 005 are prepared, NOT applied
experiments/           matched-run runner, pilot, analysis, Grad-CAM evidence, results/
tools/                 fetch_fashion_mnist.py (the only downloader), export_samples.py
docs/                  protocol, dataset card, ethics, checklist, notes, historical reports
tests/                 pytest suite (mocked database; live tests are opt-in)
```

## Setup (local)

Python **3.11** (Render uses 3.11.9; the pinned `pyarrow==17.0.0`/`pandas==2.2.2` UI stack supports 3.9–3.12). Model and UI dependencies are separate: `api/requirements.txt` (torch, no streamlit) and `ui/requirements.txt` (streamlit, no torch).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # API + UI + test tools
python tools/fetch_fashion_mnist.py     # training machine only: downloads + verifies the official files into ./data (git-ignored)
pytest -q                               # live Supabase tests are skipped unless SEE_SENSE_RUN_LIVE_TESTS=1
```

**Database:** in the Supabase SQL Editor, run in order `001_init.sql`, `002_experiment_tracking.sql`, `003_recent_classifications_view.sql` (+ `003_verify.sql`), then `004_predictions.sql` (+ `004_verify.sql`) and `005_fashion_mnist_and_atomic_logging.sql` (+ `005_verify.sql`). Migrations only add things and are safe to re-run. **Apply 004 and 005 before deploying this branch's API** (see `db/migrations/004_predictions.md`, `005_*.md`).

**Secrets (never commit; git-ignored at any depth):**

```bash
cp .env.example .env                                         # API: SUPABASE_URL + SUPABASE_SERVICE_KEY
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml   # UI: API_URL, SUPABASE_URL (…supabase.co), SUPABASE_ANON_KEY
```

**Run:**

```bash
uvicorn api.main:app --reload --port 8000     # terminal 1
cd ui && streamlit run app.py                 # terminal 2 (Streamlit reads .streamlit/ from the directory you start it in)
```

Offline API (no Supabase, development only): `SEE_SENSE_LOCAL_DB=.local_db.json uvicorn api.main:app`. `/healthz` then reports `"status": "offline"` and `"supabase": false`; never use it for official runs. See [docs/RUN_API_LOCALLY.md](docs/RUN_API_LOCALLY.md).

## Experiments (training machine)

```bash
python -m experiments.pilot --epochs 3 --batch-size 128 --lr 0.001 --out pilot.jsonl   # validation-only runtime pilot; never opens the test files
# official run: see docs/FASHION_PROTOCOL.md for the frozen command, e.g.
python -m experiments.run_experiments --dry-run ...                                      # prints the plan and its fingerprint
```

The experiment runner posts every run to a **local API process connected to Supabase**, reads each run back, verifies it was saved with the requested settings, and checks that all runs of one seed shared the same split. Test metrics stay hidden unless `--show-test` is given.

## Deployment

1. **Supabase:** apply migrations 001–005 (above).
2. **Render:** New → Blueprint → this repo (`render.yaml`; CPU-only torch via `api/requirements.txt`). Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in the service settings. Do not set `SEE_SENSE_LOCAL_DB`. Check `GET /healthz` returns `"status":"ok","storage":"supabase","supabase":true`. The service does not download the dataset and cannot train.
3. **Streamlit Community Cloud:** main file path `ui/app.py` (dependencies from `ui/requirements.txt`), Python 3.11 under Advanced settings, and paste `API_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` into the Secrets box. The repo must include `MODEL_CARD.md`, `config/`, `ui/assets/samples/`, `docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md` and the `experiments/results/*` evidence folders, which the UI reads at runtime.
4. After the official run, put its `run_id` in `config/served_model.json` (`served_run_id`) from the saved records — see that file.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/datasets` | Register the local Fashion-MNIST files (identity only; needs the files on that machine) |
| `POST` | `/train` | Train on Fashion-MNIST, persist run + checkpoint, return metrics (training machine only) |
| `GET`  | `/runs/{run_id}`, `/runs` | Fetch one run / list recent runs |
| `POST` | `/predict` | Multipart image upload → top-5 probabilities, Grad-CAM, and a logged prediction (503 if logging fails) |
| `GET`  | `/predictions`, `/predictions/{id}` | Recent logged predictions (newest first) with the producing run; no image hashes |
| `GET`  | `/classes` | The ten class labels, in probability-vector order |
| `GET`  | `/healthz`, `/version` | Liveness and storage mode (`ok` / `degraded` / `offline`); build SHA, torch version |

A database outage returns HTTP 503 `{"detail": "database unavailable"}`; if the prediction log (migrations 004/005) is unavailable, `/predict` returns 503 `prediction logging is unavailable…` and **no result** — it never reports a prediction as logged when it was not. The UI turns these into friendly messages.

## Tests and CI

`pytest -q` covers: Fashion-MNIST loading, validation split and identity; 10-class training and reproducibility; clothing-safe augmentation (no vertical flips); calibration; checkpoint contents and legacy-checkpoint loading; **multipart upload, Grad-CAM dimensions, a fixed-reference prediction regression**, prediction logging, history and drill-down; experiment matching and records; health modes and outages; and the UI modules.
Most tests use an **in-memory fake database and a tiny Fashion-shaped dataset** — they check the code, not Supabase. `tests/test_live_supabase.py` is the **live** check of prediction-to-database linkage, atomic logging and anon grants; it is skipped unless `SEE_SENSE_RUN_LIVE_TESTS=1` and needs applied migrations. GitHub Actions (`.github/workflows/ci.yml`) installs `requirements-dev.txt` on Python 3.11 and runs `pip check` and `pytest` on every push and pull request.
