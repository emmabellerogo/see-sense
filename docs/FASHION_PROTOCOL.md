# Fashion-MNIST experimental protocol (FROZEN 2026-10-17; official results exist)

Machine-readable twin: [`config/fashion_protocol.json`](../config/fashion_protocol.json) (tests check that the two agree and that the plan is matched).
Written and **frozen 2026-10-17, before any official Fashion-MNIST run and before any official test image was scored.** The earlier synthetic-shape protocol is
[PART2_PROTOCOL.md](PART2_PROTOCOL.md) (historical).

## Frozen decisions
| Item | Value |
|---|---|
| Status | **frozen** (`config/fashion_protocol.json`) |
| Dataset | Supabase dataset id **24** (`fashion-mnist`) |
| Seeds | **0, 1, 2** (3 seeds × 6 variants = 18 runs); five seeds not used |
| Execution | **sequential**: one runner process, `--seeds 0,1,2`, one run after another (estimate about 1.5 h on a busy machine, see below) |
| Dry-run fingerprint | `287b1895e0de07d1f9578d9bbec81e04e3194dc14340fb744df8567f79317f5a` |
| Everything else | unchanged from the tables below (20 epochs, batch 128, Adam 0.001, split seed 0, val 10%, dropout 0.3, shift ≤2 px, cosine, full data) |

The runner is given `--expect-fingerprint` and aborts if the plan differs. `tests/test_protocol_config.py` recomputes the fingerprint from the JSON. After the first official run starts, no value here may change.

## What is compared
Six configurations, trained on the same data with the same seeds, epochs, batch size, optimizer and initial learning rate. Only the declared technique differs.

| Variant | Model | Batch norm | Dropout | Augmentation | LR schedule |
|---|---|---|---|---|---|
| `baseline` | SmallCNN | no | 0 | no | constant |
| `full_improved` | ImprovedCNN | yes | 0.3 | shift ≤2 px + horizontal flip | cosine |
| `no_augmentation` | as full | yes | 0.3 | **no** | cosine |
| `no_batchnorm` | as full | **no** | 0.3 | yes | cosine |
| `no_dropout` | as full | yes | **0** | yes | cosine |
| `no_scheduler` | as full | yes | 0.3 | yes | **constant** |

`experiments/configs.py::check_matched` fails the plan if any other field differs; SQL query 1 re-checks it from the saved rows; the runner re-checks that all runs of a seed shared the same split.
Ablations measure a technique's contribution *given the others*, not its stand-alone effect; `baseline` also removes the batch-norm/dropout architecture differences.

## Fixed values
| Item | Value | Why |
|---|---|---|
| Data | Fashion-MNIST, all 10 classes, full dataset: 54,000 train / 6,000 validation / 10,000 official test | assignment; no subset (`train_limit = null`) |
| Validation split | class-stratified 10% of the official training images, `split_seed = 0` | reproducible; identical for every run |
| Optimizer | Adam | fixed by the API |
| Initial learning rate | 0.001 | Adam's standard starting value; not tuned |
| Batch size | 128 | standard; measured cost in the pilot |
| Epochs | 20 | the 20-epoch validation pilot showed the baseline's validation loss flattening from about epoch 13; cosine decays over the same 20 epochs |
| Dropout / shift / schedule | 0.3 / ±2 px / cosine | the project's existing defaults, **not tuned** |
| Checkpoint | lowest **validation** loss epoch | validation only |
| Calibration | temperature scaling on validation logits | validation only |
| Test set | scored once per run after the checkpoint is chosen | the runner prints no test metric unless `--show-test` |
| Served model | `full_improved-seed0` | fixed in `config/served_model.json` before results; not chosen by test score |

**No tuning toward a result.** The pilot looked only at training and validation numbers, and I deliberately did **not** change dropout, shift, learning rate or epochs after seeing them.
One pilot observation is recorded because it matters: over 20 epochs the full improved model ended with *lower* validation accuracy than the baseline (about 0.888 vs 0.908; training accuracy 0.889 vs 0.924), i.e. the heavy regularization may under-fit a ~9,000-parameter network. That is a possible, legitimate result of the official comparison; the protocol will not be changed to avoid it, and the final report must say so if it persists.

## Measured cost on this Mac (validation-only pilot; no test data)
Apple M2, 8 cores, 16 GB, CPU only, 54,000 training images, batch 128. Raw data and per-epoch times: [`experiments/results/fashion-pilot/`](../experiments/results/fashion-pilot/summary.md).
The machine was shared with other applications during the pilot (load average about 9), so times are a realistic *busy* figure; single quiet runs were about 10–12 s/epoch.

| Measurement | Result |
|---|---|
| One epoch, all six variants, sequential (12 epochs) | **15.4 s mean** (13.4 no_dropout … 17.3 no_batchnorm) |
| Spread of single epochs in the sequential pilots | about 10–19 s (one outlier 34 s) depending on variant and background load |
| One 20-epoch run | about **5 min** busy / 3.5–4 min quiet |
| Three runner processes in parallel (2 threads each) | 22 s/epoch baseline, 31 s/epoch full_improved **per process** = about **1.6–1.9× the throughput** of one process |

Estimated wall-clock for the full comparison (6 variants × 20 epochs; plus ~2 s per run for test scoring, calibration and upload):

| Seeds | Runs | One process, busy machine | One process, quiet | Three processes in parallel |
|---|---|---|---|---|
| 3 (0,1,2) | 18 | **about 1.5 h** | about 1.1 h | about 50–60 min |
| 5 (0–4) | 30 | about 2.6 h | about 1.9 h | about 1.5 h |

## Recommended budget
**(Adopted) 3 training seeds × 6 variants = 18 runs, 20 epochs.** Five seeds are *not* required; they would tighten the paired intervals (t = 2.78 instead of 4.30 for the 95% interval) at roughly 1.7× the cost, and are feasible if the laptop stays awake. Decide **before** the first official run, never after seeing results; with 3 seeds, wide intervals that include 0 are expected for small effects and must be reported as such.

## Official procedure (proposed)
1. Migrations 004 and 005 applied (`db/migrations/*.md`). Fashion-MNIST downloaded (`python tools/fetch_fashion_mnist.py`).
2. Start the API **locally** against Supabase (training machine only; never the Render service):
   `OMP_NUM_THREADS=2 uvicorn api.main:app --port 8000` (credentials come from `.env`; `SEE_SENSE_LOCAL_DB` must be unset; `/healthz` must say `"storage":"supabase"`).
3. Dataset registered once (done): id **24**. Do not register another.
4. **Freeze (done):** the dry-run fingerprint and dataset id 24 are recorded in `config/fashion_protocol.json` (`frozen_fingerprint`, `frozen_dataset_id`, `status: "frozen"`).
5. Run **sequentially**, pinning the fingerprint (one process; the API must be running locally):
   ```bash
   python -m experiments.run_experiments --api-url http://127.0.0.1:8000 --dataset-id 24 \
     --group fashion-official-1 --seeds 0,1,2 --epochs 20 --batch-size 128 --lr 0.001 --split-seed 0 \
     --dropout 0.3 --max-shift 2 --scheduler cosine --val-size 0.1 --timeout 3600 \
     --expect-fingerprint 287b1895e0de07d1f9578d9bbec81e04e3194dc14340fb744df8567f79317f5a \
     --out experiments/results/fashion-official-1-runs.jsonl
   ```
   (Parallel per-seed processes were considered and are **not** used.)
6. Evidence (official runs done: group `fashion-official-1`, runs 53-70, all saved):
   - **SQL export (once, in the Supabase SQL editor):** run `docs/fashion_official_1_export.sql` (generated by `python -m experiments.package_sql fashion-official-1 docs/fashion_official_1_export.sql`; read-only; one result row per query plus a `meta` row). Download the result as CSV (or copy as JSON), then
     `python -m experiments.ingest_sql_export <downloaded file> experiments/results/fashion-official-1-sql --source supabase-sql-editor`
     and `python -m experiments.analyze experiments/results/fashion-official-1-sql experiments/results/fashion-official-1-analysis`. Done on 2026-10-17: `PROVENANCE.md` in that folder now records source `supabase-sql-editor`, the database's own timestamp/version and the export's sha256.
   - Read-only verification of the saved records: `python -m experiments.export_records fashion-official-1 experiments/results/fashion-official-1-records`.
7. Grad-CAM evidence (done for runs 54 and 53; the baseline folder `fashion-official-1-gradcam-baseline` was made with `python -m experiments.gradcam_evidence --supabase baseline-seed0 fashion-official-1 experiments/results/fashion-official-1-gradcam-baseline`, `python -m experiments.cam_blank_check --supabase baseline-seed0 ...` and `python -m experiments.gradcam_compare --supabase baseline-seed0 full_improved-seed0 fashion-official-1 experiments/results/fashion-official-1-gradcam experiments/results/fashion-official-1-gradcam-baseline`, all read-only, no retraining; improved, run 54): `python -m experiments.gradcam_evidence --supabase full_improved-seed0 fashion-official-1 experiments/results/fashion-official-1-gradcam` and `python -m experiments.cam_blank_check --supabase full_improved-seed0 fashion-official-1 experiments/results/fashion-official-1-gradcam/blank_cam_by_class.csv`; the two failure interpretations are a **draft** in `docs/FASHION_GRADCAM_FAILURE_ANALYSIS.md` for a person to review.
8. `served_run_id` = 54 is set in `config/served_model.json` (done); deploy is a manual step.

## Things that can go wrong (and what to do)
- A run fails: the runner stops at the first failure unless `--keep-going`; re-running the same command skips runs already saved (HTTP 409 → "exists"). Never edit settings between retries.
- The laptop sleeps or the API restarts mid-run: the run in progress is lost (nothing half-saved: the run row is removed if its checkpoint upload fails). Re-run the same command.
- Results disappoint: report them. Do not change the protocol after test results exist.
- Bitwise reproducibility across machines or thread counts is not guaranteed; in the pilot, 2-thread and 4-thread runs of the same seed gave identical epoch losses on this machine, but all official runs should still come from one machine.

## Known limitation to disclose
Few seeds (3 proposed) give wide paired intervals; five would be better but are not required. The number used is stated wherever results appear.
