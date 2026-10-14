# AI Usage Record — See-Sense

- **Student:** Komal Khan
- **Date:** 2026-10-14
- **Session title:** Run the API locally (Part 2 setup) on Windows and add AI-usage records
- **Project part / step:** Part 2 — Step 1 (get access, run the test suite, dry run), from `docs/PART1_HANDOFF.md` §9
- **AI tool:** Claude (agentic session in a cloud Linux workspace)
- **Purpose:** Set up and verify the FastAPI model service on a local machine
  following the Part 1 handoff, fix anything that stopped it from running locally,
  and record this AI session.
- **Files changed (if any):**
  - `api/db.py` — added `load_dotenv()`; added an opt-in offline switch (`SEE_SENSE_LOCAL_DB`)
  - `api/local_db.py` — **new**, JSON-file stand-in for Supabase (local development only)
  - `tests/test_local_db.py` — **new**, 3 tests for the offline mode
  - `docs/RUN_API_LOCALLY.md` — **new**, step-by-step guide plus verification table
  - `README.md` — test count corrected (6 → 82), link to the new guide
  - `.env.example`, `.gitignore` — documented and ignored the local DB file
  - `ai-usage/Komal-Session1.md` — this file
  - (later in the session) `docs/RUN_API_LOCALLY.md` — Windows steps, Windows results, cross-machine comparison
- **Verification performed / results:**
  - `pytest -q` on the untouched handoff code: **79 passed, 1 skipped** (matches handoff §6)
  - `pytest -q` after the changes: **82 passed, 1 skipped**
  - Real `uvicorn` server in offline mode: `/healthz` ok, `/version`, `/classes` correct
  - `--dry-run` printed 6 request bodies and fingerprint `c82db51c…` (example values, not frozen)
  - Runner against the live local server, 6 configs × seed 0 × 3 epochs: 6 saved, all read-back checks passed, exit 0; a rerun skipped all 6
  - `POST /predict` with an uploaded horizontal-bar PNG → `horizontal` (0.998) with Grad-CAM
  - `.env` loading confirmed with a fake URL (project ref read; health "degraded" as expected)
  - **On Komal's Windows laptop:** `pytest -q` 82 passed, 1 skipped; API started with
    uvicorn; `/healthz` ok in the browser; dry-run fingerprint `c82db51c…` **identical**
    to the Linux one; 6 smoke runs saved; rerun skipped all 6
  - Cross-machine check: training losses identical; best validation loss identical for
    the 2 models without batch norm, slightly different for the 4 with batch norm
  - **Out of scope (by team decision):** runs against the real Supabase project;

## Transcript

Prompts, Claude's responses and the relevant tool and test results. Claude's
replies are condensed where they only narrated steps; commands and outputs are
quoted. No API keys, passwords or tokens were used or shown in this session.

---

**Komal:** _[attached `see-sense-main.zip`]_ from file read part 1 handoff
complete the part to run api locally and also document ai transcripts

**Claude:** Unpacked the project and read `docs/PART1_HANDOFF.md`, `README.md`,
`.env.example`, `render.yaml`, the requirements files, `api/main.py`,
`api/db.py`, `tests/conftest.py`, the runner and `db/seed.py`. Tried to reach
Komal's computer through the desktop link; it was not connected, so all work was
done in the cloud workspace.

**Tool: environment setup**
```
python3.11 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
→ timed out: download.pytorch.org is not reachable from this workspace
pip install -r requirements-dev.txt        # plain PyPI
→ EXIT 0  (torch 2.4.1, fastapi 0.115.0)
```

**Claude:** Finding: the API never loads `.env`. `grep -rn dotenv` matched only
`db/seed.py`, while `api/db.py` reads `os.environ["SUPABASE_URL"]` directly. So
following handoff §5 (copy `.env.example` → `.env`, run `uvicorn`) would start
the API without credentials: `/healthz` would report degraded and `/train` would
fail. Fix: `load_dotenv()` at the top of `api/db.py`. It does not override
variables that are already set, so Render still behaves the same.

Also added an **opt-in offline mode**, because the handoff says Komal needs keys
or a hosted URL from Emma before she can run anything. With
`SEE_SENSE_LOCAL_DB=<file>` set, `api/db.py` swaps its functions for
`api/local_db.py`, a JSON-file store. It is seeded with dataset 1 (800 rows,
noise 0.1, seed 42, same as Emma's) and enforces the same unique
(group, run_name) rule as migration 002. Without the variable, behaviour is
unchanged. Clearly documented as not for official runs.

**Tool: tests**
```
pytest -q   (before changes)   → 79 passed, 1 skipped
pytest -q   (after changes)    → first try: 1 failure in my new test
             (used n_rows=80, below the schema minimum of 100, and the wrong
              field name gradcam_png_b64; the real field is cam_png) → fixed
pytest -q                      → 82 passed, 1 skipped
```

**Tool: live server (offline mode)**
```
SEE_SENSE_LOCAL_DB=.local_db.json uvicorn api.main:app --port 8000
GET /healthz  → {"status":"ok","model_loader":true,"supabase":true}
GET /version  → {"git_sha":"unknown","torch_version":"2.4.1+cu121","supabase_project_ref":null}
GET /classes  → ["horizontal","vertical","diagonal","block"]

python -m experiments.run_experiments --dry-run --dataset-id 1 --group explore-1 \
  --seeds 0 --epochs 15 --batch-size 32 --lr 0.01 --split-seed 0
→ plan: 6 runs, protocol fingerprint c82db51cf2b954a9654522e4cc988593fcd3bf9cd6af5ea813c91add20cb7ae0

python -m experiments.run_experiments --api-url http://127.0.0.1:8000 --dataset-id 1 \
  --group explore-smoke --seeds 0 --epochs 3 --batch-size 32 --lr 0.01 --split-seed 0
→ [1/6] baseline-seed0        saved run_id=1 params=4388 best_epoch=2 best_val_loss=0.0367
  [2/6] full_improved-seed0   saved run_id=2 params=4436 best_epoch=2 best_val_loss=0.0161
  [3/6] no_augmentation-seed0 saved run_id=3 params=4436 best_epoch=2 best_val_loss=0.0192
  [4/6] no_batchnorm-seed0    saved run_id=4 params=4388 best_epoch=2 best_val_loss=0.0827
  [5/6] no_dropout-seed0      saved run_id=5 params=4436 best_epoch=2 best_val_loss=0.0188
  [6/6] no_scheduler-seed0    saved run_id=6 params=4436 best_epoch=2 best_val_loss=0.0325
  done: 6 saved, 0 already existed, 0 failed, 0 not attempted   (exit 0)
(same command again) → done: 0 saved, 6 already existed

POST /predict (run_id=2, uploaded 28×28 PNG with a horizontal bar)
→ horizontal, confidence 0.998, Grad-CAM PNG returned
```
_Note: these 3-epoch, single-seed validation losses are plumbing checks only and
say nothing about which technique helps (handoff §2)._

**Tool: `.env` check**
```
copy of api/ + shared/ with a .env containing a FAKE url (https://fakeref123.supabase.co) and dummy key
GET /version → supabase_project_ref = fakeref123   (value came from .env)
GET /healthz → {"status":"degraded","model_loader":true,"supabase":false}   (expected: fake keys)
```

**Claude:** One slip during testing: the command used to stop the server
(`pkill -f "uvicorn api.main:app"`) also matched and ended its own shell, so
the `.env` check was rerun separately. It had no effect on the code.

Wrote `docs/RUN_API_LOCALLY.md`. It covers both modes (Supabase and offline),
setup, health checks, dry run, a smoke test and the verification table, and
lists the one remaining step: repeat the checks against the real Supabase
project once Emma shares keys or a hosted API URL. Updated the README quickstart.
Then wrote this record.

---

### Part B: running it on Komal's Windows laptop (same session, 14:49–15:10)

**Claude:** Gave an 8-step outline: unzip → venv and install → `pytest` → start
the API offline → `/healthz` → dry run and smoke test → get keys from Emma →
finish this record → commit. It gave Mac and Windows commands but used the
Mac form in places, which caused the next error.

**Komal:** _(pasted PowerShell errors)_
```
python3.11 : The term 'python3.11' is not recognized ...
source : The term 'source' is not recognized ...
PS C:\Users\...\Downloads\see-sense-main-part2>
```

**Claude:** Diagnosis: Windows PowerShell has no `python3.11` or `source`, and
the terminal was one folder too high (the zip contains `see-sense-main` inside
`see-sense-main-part2`). Gave the Windows forms: `cd see-sense-main`,
`py --list`, `py -3.11 -m venv .venv`, `.venv\Scripts\Activate.ps1`, the
execution-policy fix if needed, and the PowerShell way to set `SEE_SENSE_LOCAL_DB`.

**Komal:** do one step at a time

**Claude:** Switched to one step per message.

**Komal:** _(install log ending in `Installing collected packages: ... torch ... streamlit`)_ → **Claude:** wait for the prompt to come back.

**Komal:** `82 passed 1 skipped 3 warnings in 35.11s`

**Claude:** Matches the expected result; the warnings are harmless. Next, start the API:
`$env:SEE_SENSE_LOCAL_DB=".local_db.json"; uvicorn api.main:app --port 8000`

**Komal:**
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
→ **Claude:** open `http://127.0.0.1:8000/healthz` in a browser.

**Komal:** `{"status":"ok","model_loader":true,"supabase":true}`

**Claude:** Healthy. Noted that in offline mode `supabase:true` means the local
file works, not the real Supabase. Next, open a second PowerShell window (Windows
key → "PowerShell"), `cd` into the project and activate the venv.

**Komal:** _(pasted the activate and dry-run commands as one line)_
```
.venv\Scripts\Activate.ps1python -m experiments.run_experiments ...
.venv\Scripts\Activate.ps1python : The module '.venv' could not be loaded.
```
**Claude:** The two commands were joined; run them on separate lines.

**Komal:** _(after separating them)_
```
(.venv) PS ...> python -m experiments.run_experiments --dry-run --dataset-id 1 --group explore-1 --seeds 0 --epochs 15 --batch-size 32 --lr 0.01 --split-seed 0
plan: 6 runs, protocol fingerprint c82db51cf2b954a9654522e4cc988593fcd3bf9cd6af5ea813c91add20cb7ae0
(6 request bodies)
```
**Claude:** The fingerprint is identical to the Linux workspace, so the plan is the same on both machines.

**Komal:** _(smoke test, group `explore-smoke`, 3 epochs)_
```
[1/6] baseline-seed0        saved run_id=1 params=4388 best_epoch=2 best_val_loss=0.0367 final_train_loss=0.0769
[2/6] full_improved-seed0   saved run_id=2 params=4436 best_epoch=2 best_val_loss=0.0140 final_train_loss=0.0512
[3/6] no_augmentation-seed0 saved run_id=3 params=4436 best_epoch=2 best_val_loss=0.0186 final_train_loss=0.0280
[4/6] no_batchnorm-seed0    saved run_id=4 params=4388 best_epoch=2 best_val_loss=0.0827 final_train_loss=0.1497
[5/6] no_dropout-seed0      saved run_id=5 params=4436 best_epoch=2 best_val_loss=0.0178 final_train_loss=0.0310
[6/6] no_scheduler-seed0    saved run_id=6 params=4436 best_epoch=2 best_val_loss=0.0128 final_train_loss=0.0396
done: 6 saved, 0 already existed, 0 failed, 0 not attempted
```
**Claude:** The API runs locally. Training losses are identical to the Linux run.
Validation losses differ slightly for the 4 batch-norm models only, which is an
example of the handoff's warning that bitwise equality across machines is not
guaranteed. These are plumbing numbers, not results. Next, rerun the same command
to check skipping.

**Komal:** _(the rerun reported 6 already existed)_

**Claude:** Next, ask Emma privately for the Supabase URL and service key (or a
hosted API URL), and mention the `.env` fix.

**Komal:** from part 1 handoff file check all parts that have been completed

**Claude:** Status table. Done: tests, local API, dry run, smoke test and skip
check. Open: Supabase access and run, finishing this record, commit. Handoff §9
steps 2–6 (explore, freeze, official runs, analysis, write-up) not started;
check with the team who owns them.

**Komal:** update file RUN API LOCALLY

**Claude:** Updated `docs/RUN_API_LOCALLY.md` with Windows steps and
pitfalls, the Windows verification table, a Linux-vs-Windows comparison table,
and a note that Supabase runs are owned by Emma. Updated this record. Sent the
changed files and the commit commands to Komal.

**Remaining for Komal:**
1. Add your last name at the top and write "My review, decisions, and what I learned".
2. Copy the updated files into the project and commit (Emma does the Supabase runs).
