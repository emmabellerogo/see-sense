# See-Sense — CNN Image Classifier with Grad-CAM (three clouds)

**Streamlit UI ⇄ FastAPI model API ⇄ Supabase data.** A small PyTorch CNN classifies
28×28 grayscale images into four **synthetic** shapes (horizontal bar, vertical bar,
diagonal line, filled block) and returns a **Grad-CAM** heatmap showing where the
network looked. It is a teaching and evaluation project, not a general photo classifier.

## Live deployment URLs

Placeholders until the deployment succeeds. Replace them with the real URLs afterwards.

| Tier | Platform | URL |
|------|----------|-----|
| UI | Streamlit Community Cloud | `https://<your-app>.streamlit.app` *(placeholder)* |
| API | Render.com | `https://<your-api>.onrender.com` *(placeholder; free tier sleeps, first call takes ~30–60 s)* |
| Data | Supabase | `https://<your-project-ref>.supabase.co` *(placeholder)* |

## The five tabs (`ui/app.py`)

1. **Concepts** — convolution, stride/padding/output-size arithmetic with a worked example, receptive-field growth (three stacked 3×3 layers: 3 → 5 → 7), and the Grad-CAM derivation and limits.
2. **Try It** — classify an uploaded image or a generated shape with the default model `full_improved-seed0` (run ID taken from the saved experiment records); shows prediction, confidence, input and Grad-CAM. Handles cold starts and errors; prediction requests are never retried automatically.
3. **Compare Models** — the official `official-supabase-1` results (5 seeds × 6 variants), matched-configuration audit, paired ablation differences, loss curves, confusion matrices, and a Grad-CAM gallery (5 correct and 8 incorrect examples with two written failure analyses). These are *saved* results read from `experiments/results/`.
4. **Recent Classifications** — reads the latest ≤200 rows of the `recent_classifications` view with the anon key and shows a class-distribution histogram.
5. **Model Card** — [MODEL_CARD.md](MODEL_CARD.md).

Training is not exposed in the UI. It remains available through the API (`POST /train`) and the experiment tools in `experiments/`.

## Official results (group `official-supabase-1`, stored in Supabase)

Test accuracy, mean ± sd over 5 training seeds, on one fixed 2,000-image dataset (noise 1.0; 1,400 train / 200 validation / 400 test):

| Variant | Test accuracy |
|---|---|
| full_improved | 95.05% ± 0.67 |
| no_scheduler | 95.05% ± 0.96 |
| no_dropout | 94.70% ± 1.34 |
| no_augmentation | 93.25% ± 1.09 |
| baseline | 76.80% ± 8.93 |
| no_batchnorm | 67.45% ± 22.93 |

Removing augmentation reliably lowered accuracy; the data do **not** show that dropout or the LR scheduler clearly improved it. Details: [experiments/results/official-supabase-1-analysis/summary.md](experiments/results/official-supabase-1-analysis/summary.md), [docs/SUPABASE_GRADCAM_FAILURE_ANALYSIS.md](docs/SUPABASE_GRADCAM_FAILURE_ANALYSIS.md), [docs/PART2_PROTOCOL.md](docs/PART2_PROTOCOL.md). (`experiments/results/official-1-*` are earlier offline-store runs and are not the official results.)

## Data: persisted vs not persisted

| Persisted in Supabase | Never persisted |
|---|---|
| Dataset *parameters* (rows, size, noise, seed) | Image pixels (datasets are regenerated from the seed) |
| Run metrics, settings, loss histories, confusion matrices | Uploaded files (processed in memory only) |
| Model checkpoints (`run_artifacts`, not readable with the anon key) | Grad-CAM images |
| Per prediction: SHA-256 hash, width, height, class, label, confidence (`image_metadata`, not readable with the anon key) | Service-role key in the UI |

The anon key can read only `runs` (policy in 001) and the `recent_classifications` view (003: `id, run_id, class_name, confidence, created_at`, no image hashes). All writes go through the API with the service-role key.

## Architecture

```
Streamlit Cloud (ui/app.py)  ──HTTPS──►  FastAPI on Render (api/main.py)  ──service-role──►  Supabase Postgres
 thin client, no torch                    CNN + Grad-CAM, only writer                         datasets · runs · run_artifacts · image_metadata
        └───────────────── anon key, read-only: runs, recent_classifications view ───────────────────►
```

## Project structure

```
MODEL_CARD.md          model card (read by the UI)        render.yaml   Render blueprint
shared/                Pydantic schemas, shape generator  .github/workflows/ci.yml   pytest on push/PR
api/                   FastAPI tier: main, training (CNN + Grad-CAM), db, experiments, requirements.txt
ui/                    Streamlit tier: app.py + concepts/evidence/recent/tryit modules, requirements.txt
db/migrations/         001_init, 002_experiment_tracking, 003_recent_classifications_view (+ 003_verify.sql)
experiments/           matched-run runner, analysis, Grad-CAM evidence, results/ (official-supabase-1-*)
docs/                  protocol, results, failure analysis, comparison_queries.sql
tests/                 pytest suite
```

## Setup (local)

Python **3.11** (Render uses 3.11.9; the pinned `pyarrow==17.0.0`/`pandas==2.2.2` UI stack supports 3.9–3.12).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # API + UI + test tools
pytest -q                               # live-Supabase test skips without credentials
```

**Database:** in the Supabase SQL Editor, run in order `db/migrations/001_init.sql`, `002_experiment_tracking.sql`, `003_recent_classifications_view.sql`, then `003_verify.sql` (see `003_recent_classifications.md`). Migrations only add things; they are safe to re-run.

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

To start the UI from the repo root instead, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and run `streamlit run ui/app.py`.

Offline API (no Supabase, development only): `SEE_SENSE_LOCAL_DB=.local_db.json uvicorn api.main:app`. `/healthz` then reports `"status": "offline"` and `"supabase": false`; never use it for official runs. See [docs/RUN_API_LOCALLY.md](docs/RUN_API_LOCALLY.md).

## Deployment

1. **Supabase:** apply the three migrations above.
2. **Render:** New → Blueprint → this repo (`render.yaml`; CPU-only torch via `api/requirements.txt`). Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in the service settings. Do not set `SEE_SENSE_LOCAL_DB`. Check `GET /healthz` returns `"status":"ok","storage":"supabase","supabase":true`.
3. **Streamlit Community Cloud:** main file path `ui/app.py` (dependencies from `ui/requirements.txt`), Python 3.11 under Advanced settings, and paste `API_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` into the Secrets box. The repo must include `MODEL_CARD.md`, `docs/SUPABASE_GRADCAM_FAILURE_ANALYSIS.md` and `experiments/results/official-supabase-1-*`, which the UI reads at runtime.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/datasets` | Register a synthetic dataset (parameters only) |
| `POST` | `/train` | Train, persist run + checkpoint, return metrics |
| `GET`  | `/runs/{run_id}`, `/runs` | Fetch one run / list recent runs |
| `POST` | `/predict` | Classify an uploaded image, return Grad-CAM, log the hash |
| `POST` | `/predict_sample` | Generate + classify a sample, return Grad-CAM |
| `GET`  | `/classes` | Class labels |
| `GET`  | `/healthz` | Liveness and storage mode (`ok` / `degraded` / `offline`) |
| `GET`  | `/version` | Build SHA, torch version |

A database outage returns HTTP 503 `{"detail": "database unavailable"}` from prediction/training routes; the UI turns it into a friendly message.

## Tests and CI

`pytest -q` covers data generation, training and reproducibility, experiment records, Grad-CAM evidence loading, upload and sample prediction through the API, health modes, database-outage handling, and the UI modules. GitHub Actions (`.github/workflows/ci.yml`) installs `requirements-dev.txt` on Python 3.11 and runs `pip check` and `pytest` on every push and pull request.
