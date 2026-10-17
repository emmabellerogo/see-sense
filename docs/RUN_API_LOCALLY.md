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

### 5b. Komal's laptop (offline mode, Windows, PowerShell, Python 3.11)

| Check | Result |
|---|---|
| `pytest -q` | **82 passed, 1 skipped**, 3 warnings (pydantic `model_` namespace; harmless) |
| `uvicorn` start | "Uvicorn running on http://127.0.0.1:8000" |
| `GET /healthz` in a browser | `{"status":"ok","model_loader":true,"supabase":true}` (in offline mode `supabase:true` means the local file works) |
| `--dry-run` (values in step 3) | fingerprint `c82db51c…`: **identical** to 5a, so the plan is the same on both machines |
| Runner, 6 configs × seed 0, 3 epochs, group `explore-smoke` | `done: 6 saved, 0 already existed, 0 failed, 0 not attempted` |
| Same command again | `0 saved, 6 already existed`: skipped, not retrained |

**Cross-machine comparison (smoke runs, seed 0, 3 epochs).** Final training
losses matched the Linux run exactly for all six variants. Best validation loss
matched exactly for the two models without batch norm, and differed for the
four with batch norm:

| Variant | Batch norm | best_val_loss Linux | best_val_loss Windows |
|---|---|---|---|
| baseline | off | 0.0367 | 0.0367 |
| full_improved | on | 0.0161 | 0.0140 |
| no_augmentation | on | 0.0192 | 0.0186 |
| no_batchnorm | off | 0.0827 | 0.0827 |
| no_dropout | on | 0.0188 | 0.0178 |
| no_scheduler | on | 0.0325 | 0.0128 |

This is a concrete case of the handoff's warning (§8) that bitwise equality
across machines is not guaranteed. It suggests evaluating the batch-norm models
is where small numerical differences appear. Practical rule: **all official
runs should come from one machine/host** (the Supabase-connected API Emma
runs), and these numbers are plumbing checks, not results.

### 5c. Supabase runs: owned by Emma

Runs against the real Supabase project (steps 2a–4 with real keys, exploration
and the official experiment) are done by Emma, who owns the project and its
keys. Komal's part ends at a verified local API.

The 3-epoch smoke numbers are plumbing checks only. Like the 100% scores in the
handoff, section 2, they are not results.

## What changed in the code

- `api/db.py`: calls `load_dotenv()`. At the bottom, it switches every function
  to `api/local_db.py` when `SEE_SENSE_LOCAL_DB` is set. Without that variable,
  behaviour is the same as before.
- `api/local_db.py` (new): a JSON-file store with the same functions as `db.py`.
  It applies the same duplicate (group, run_name) rule as migration 002's unique
  index.
- `tests/test_local_db.py` (new): seeding, the duplicate check, and a full
  train → read back → predict cycle through the API in offline mode.
- `.env.example`, `.gitignore`: document the variable and ignore the local file.
