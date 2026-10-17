# Ethics and responsible disclosure — See-Sense

Status key used throughout:
- **IMPLEMENTED** — present in this repository today; evidence cited.
- **PROPOSED** — not implemented; a recommendation only. Nothing marked PROPOSED should be
  described as done.

> **DRAFT** — written for the Fashion-MNIST version on branch `correction/fashion-mnist`; it describes the code on that branch, not the earlier synthetic-shape deployment still online.

See-Sense is a teaching demo that classifies 28×28 grayscale clothing pictures (Fashion-MNIST, ten categories). The discussion
below asks what would go wrong if a system like it, or its prediction-logging design,
were reused in the real world.

## 1. Two realistic deployment contexts

### Context A — A classroom or public demo that accepts uploaded images (the intended deployment)
- **Who is affected:** anyone who uploads an image to the public Streamlit app.
- **Risks:** a user may upload a personal photo (a face, a child, a document). The model
  will still return one of ten clothing labels with a confidence, which can look
  authoritative but is meaningless. The API also stores a SHA-256 hash of the file; a
  hash is not reversible, but anyone who already holds the same file can check whether it
  was uploaded, so it is not anonymous in every sense. The public Recent Classifications
  tab shows class, confidence, the probability vector, time and the producing run for the latest 200 predictions, so other
  visitors can see that predictions are happening, but not images or hashes.
- **Where the code stands:** see safeguards T1–T3 and G1–G2 below. Known gaps are listed
  in section 4.

### Context B — The same pipeline adapted to a real vision task (for example, a camera-based screening or monitoring tool)
- **This is not what the project does.** It is the foreseeable misuse the model card
  already rules out of scope ("medical imaging, surveillance, content moderation or any
  decision about people").
- **Risks:** a small CNN that does well on a centred, low-resolution benchmark says nothing about real-world
  accuracy, and calibrated-looking percentages can be trusted too much. Grad-CAM heatmaps look convincing even on wrong
  predictions (the Compare Models gallery will show confident errors next to correct examples), so a heatmap
  could lend false credibility to a harmful decision. Logging per-prediction records
  with timestamps would, in a real deployment, become a record of people's behaviour.
- **Where the code stands:** the project only argues against this use (model card
  scope statement, UI wording). It has no technical control that would stop someone
  retraining the code on real data.

## 2. Technical safeguards

| ID | Safeguard | Status | Evidence |
|---|---|---|---|
| T1 | **Store a hash and numbers, never pixels — and log atomically.** Uploaded files are decoded in memory; `image_metadata` holds only SHA-256, width, height, label, class name, confidence and timestamp; `predictions` holds SHA-256, label, probability vector, serving run and timestamp. Both rows are written in one transaction, and `/predict` returns an error (not a result) if logging fails. Grad-CAM images are not stored. | IMPLEMENTED in code and migrations 004/005 (prepared, **not yet applied** to Supabase); local-engine and mocked tests pass, live check pending | `api/db.py` `log_prediction`; `db/migrations/004`–`005`; `api/main.py` `/predict`; `tests/test_api_predict.py`, `tests/test_live_supabase.py` (live, opt-in) |
| T2 | **Least-privilege database access.** The UI uses only the anon key, which can read `runs` and the restricted views `recent_predictions` and `recent_classifications` (no hashes). `image_metadata`, `predictions` and `run_artifacts` have row-level security with no anon policy. The public `GET /predictions` omits hashes too. Writes use the service-role key, which stays on the API server and is never in the UI. | IMPLEMENTED | `db/migrations/001`–`005` (`003_verify.sql`, `004_verify.sql`, `005_verify.sql` check grants); README "Data" section; `.gitignore` excludes secret files |
| T3 | **Errors do not leak internals.** Database failures return a generic 503 "database unavailable"; logs record the exception type or code only. Tests check that host names and keys do not appear in user-facing messages or logs. | IMPLEMENTED | `api/main.py` exception handlers; `tests/test_deploy_readiness.py`, `tests/test_ui_recent.py` |
| T4 | **Upload size limit, content-type check and rate limiting on `/predict`.** | PARTLY IMPLEMENTED | A 5 MB size limit (HTTP 413) and image decoding checks (HTTP 422) exist; there is no per-client rate limit. |
| T5 | **Restrict CORS to the real UI origin and add an API key or similar for `/train` and `/datasets`.** | PROPOSED | Currently `ALLOWED_ORIGINS="*"` in `render.yaml` and no authentication on any endpoint, including `POST /train`. |

## 3. Governance safeguards

| ID | Safeguard | Status | Evidence |
|---|---|---|---|
| G1 | **Honest scope statements.** The model card and UI say the model is a demonstration on a benchmark dataset, name out-of-scope uses, report spread across seeds rather than the best run, and state that a fitted temperature is not proof of calibration. Final claims about technique benefits will follow the measured results, whatever they are. | IMPLEMENTED | `MODEL_CARD.md` (Intended use, Limitations, Ethical considerations); README; Compare Models tab |
| G2 | **Frozen protocol and recorded AI use.** The experimental protocol was frozen before the official runs so results could not be chosen after seeing test scores; AI-assistant sessions are recorded. | IMPLEMENTED | `docs/PART2_PROTOCOL.md`; `ai-usage/` |
| G3 | **A short in-app privacy notice before upload** (what is stored, that a hash is kept, not to upload personal photos). | PROPOSED | Only a caption after prediction mentions the hash. There is no notice before upload. |
| G4 | **A retention and deletion policy** for `image_metadata`, and a named contact for reporting problems. | PROPOSED | No retention period is defined; rows are kept indefinitely. No contact or issue channel is documented. |

## 4. Known gaps (stated so they are not hidden)
- No authentication or rate limiting on any API endpoint; public `POST /train` can be
  called by anyone who knows the URL and consumes service resources and database rows.
- CORS is open (`*`).
- No per-client rate limit on uploads.
- No retention policy for stored prediction hashes.
- Platform-level logs (Render, Streamlit Community Cloud, Supabase) are controlled by
  those providers; this repository does not configure them and we have not checked what
  they retain. Do not read "we do not log X" below as covering provider infrastructure.

## 5. What we deliberately did not log or store
In the application code and database schema, the following are **not** collected:
- Image pixels, thumbnails, or Grad-CAM overlays.
- Original filenames, EXIF metadata, or the original image dimensions (width and height
  are recorded after resizing to 28×28, so they are always 28×28).
- User identifiers, accounts, session IDs, IP addresses, user-agent strings or any
  free-text input. The API code does not read client address or headers, and the table
  has no column for them.
- Error logs include only the exception type or database error code, not request bodies.

What **is** stored per prediction: SHA-256 of the raw uploaded bytes, 28×28 size, predicted
class, label index, confidence, the full probability vector (in `predictions`), timestamp, and the run ID used.
Uploads of the 20 bundled held-out examples are logged the same way (their hashes are those of the example PNG files).

## 6. Reflection: responsible disclosure from a Christian worldview

> **DRAFT — for Emma's personal review. This is a placeholder structure, not Emma's
> beliefs. The text below was drafted by an AI assistant from the project's facts and
> general themes; Emma should rewrite, replace or delete it in her own words before
> submission. No personal belief, church tradition or Scripture reading has been
> assumed or attributed to Emma or Komal.**

Prompts and project facts to reflect on (edit freely):

1. **Truthfulness about what the system is.** The project states plainly that it is a
   benchmark demo and will report whatever the final experiment shows, including weak or null
   results for individual techniques. It also discloses that an earlier version used the wrong
   (synthetic) dataset.
   *Reflection (Emma to write):* How does the commitment to truthful speech connect to
   reporting the seed spread and not only the best run? [your words here]
2. **Care for those who might be affected.** The hash-not-pixels design and the
   out-of-scope list exist because uploaded images might belong to real people.
   *Reflection (Emma to write):* What does it mean to love one's neighbour when you build
   something others will use without reading the fine print? [your words here]
3. **Humility about limits.** Grad-CAM can look plausible while wrong; known gaps in
   section 4 are published rather than hidden.
   *Reflection (Emma to write):* What place do confession of limits and accountability
   have in how you disclose a system's weaknesses? [your words here]
4. **Disclosure in practice.** Decide what responsible disclosure means for this project:
   who a user or reviewer should contact, and how quickly a reported problem would be
   fixed. This is currently undefined (G4). [your words here]

If you cite Scripture or a tradition, add the specific passage and how you read it
yourself; none has been chosen for you here.
