-- 002_experiment_tracking.sql
-- Adds experiment metadata to `runs` so baseline-vs-improved comparisons can be
-- stored and queried. ADDITIVE ONLY: every new column is nullable, nothing is
-- dropped, renamed, rewritten or backfilled, and 001_init.sql is untouched.
-- Rows written before this migration stay valid; their new columns are NULL.
-- Safe to run more than once.
--
-- Apply in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- (See 002_experiment_tracking.md for what each column means.)
--
-- NOT stored anywhere: raw images, Grad-CAM PNGs, or model weights. The fitted
-- model checkpoint stays in `run_artifacts.model_b64`, exactly as before.

-- ---------------------------------------------------------------------------
-- Existing columns we KEEP USING (no change):
--   runs.dataset_id  -> which dataset row (datasets.n_rows / noise / seed)
--   runs.lr          -> initial learning rate
--   runs.batch_size  -> batch size
--   runs.epochs      -> epoch budget
--   runs.accuracy    -> final TEST accuracy
--   runs.macro_f1    -> final TEST macro-F1
-- ---------------------------------------------------------------------------

alter table runs
    -- Which experiment this run belongs to, and its name inside it.
    add column if not exists experiment_group text,
    add column if not exists run_name         text,

    -- Model variant and the techniques the model ACTUALLY uses. A baseline run
    -- records use_batchnorm = false, dropout = 0 (SmallCNN has neither).
    add column if not exists model_variant    text
        constraint runs_model_variant_chk check (model_variant in ('baseline', 'improved')),
    add column if not exists use_batchnorm    boolean,
    add column if not exists dropout          double precision
        constraint runs_dropout_chk check (dropout >= 0 and dropout < 1),
    add column if not exists augment          boolean,
    add column if not exists augment_max_shift integer
        constraint runs_augment_max_shift_chk check (augment_max_shift >= 0),
    add column if not exists scheduler        text
        constraint runs_scheduler_chk check (scheduler in ('none', 'cosine', 'step')),

    -- The three independent seeds: images, train/val/test split, training.
    add column if not exists data_seed        bigint,
    add column if not exists split_seed       bigint,
    add column if not exists train_seed       bigint,

    -- Split identity: hash of the exact train/val/test indices, and their sizes.
    -- Two runs with the same split_sha256 saw identical examples.
    add column if not exists split_sha256     text,
    add column if not exists n_train          integer,
    add column if not exists n_val            integer,
    add column if not exists n_test           integer,

    -- Full training configuration (optimizer, test/val fractions, noise, ...),
    -- exactly as run_experiment() returns it in result["config"].
    add column if not exists optimizer        text,
    add column if not exists config           jsonb,

    -- Final TEST results (accuracy / macro_f1 live in the existing columns).
    -- confusion_matrix: 4x4 list of lists, rows = true class, cols = predicted.
    add column if not exists test_loss        double precision,
    add column if not exists confusion_matrix jsonb,

    -- Per-epoch histories: JSON arrays with one number per epoch.
    add column if not exists train_loss_history   jsonb,
    add column if not exists val_loss_history     jsonb,
    add column if not exists val_accuracy_history jsonb,
    add column if not exists lr_history           jsonb,

    -- Summary of the run.
    add column if not exists best_epoch       integer      -- 0-based; chosen by lowest validation loss
        constraint runs_best_epoch_chk check (best_epoch >= 0),
    add column if not exists param_count      integer
        constraint runs_param_count_chk check (param_count > 0),
    add column if not exists train_seconds    double precision
        constraint runs_train_seconds_chk check (train_seconds >= 0),

    -- Format version of the checkpoint stored in run_artifacts.model_b64.
    -- NULL / 1 = original format (state_dict, img_size, n_classes).
    -- 2        = adds arch, use_batchnorm, dropout. (The API sets this later.)
    add column if not exists artifact_version integer
        constraint runs_artifact_version_chk check (artifact_version >= 1);

-- Shape checks on the JSON columns. NULL (old rows) always passes. Added in a
-- DO block because ADD CONSTRAINT has no IF NOT EXISTS.
do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'runs_confusion_matrix_chk') then
        alter table runs add constraint runs_confusion_matrix_chk
            check (confusion_matrix is null or jsonb_typeof(confusion_matrix) = 'array');
    end if;
    if not exists (select 1 from pg_constraint where conname = 'runs_config_chk') then
        alter table runs add constraint runs_config_chk
            check (config is null or jsonb_typeof(config) = 'object');
    end if;
    -- A history, when present, must have exactly one entry per epoch.
    if not exists (select 1 from pg_constraint where conname = 'runs_history_lengths_chk') then
        alter table runs add constraint runs_history_lengths_chk check (
            (train_loss_history   is null or (jsonb_typeof(train_loss_history)   = 'array' and jsonb_array_length(train_loss_history)   = epochs)) and
            (val_loss_history     is null or (jsonb_typeof(val_loss_history)     = 'array' and jsonb_array_length(val_loss_history)     = epochs)) and
            (val_accuracy_history is null or (jsonb_typeof(val_accuracy_history) = 'array' and jsonb_array_length(val_accuracy_history) = epochs)) and
            (lr_history           is null or (jsonb_typeof(lr_history)           = 'array' and jsonb_array_length(lr_history)           = epochs))
        );
    end if;
end
$$;

-- A run name is unique within its group (only enforced when both are set, so
-- old rows and ad-hoc runs are unaffected).
create unique index if not exists uq_runs_group_run_name
    on runs (experiment_group, run_name)
    where experiment_group is not null and run_name is not null;

create index if not exists idx_runs_experiment_group
    on runs (experiment_group)
    where experiment_group is not null;

-- ---------------------------------------------------------------------------
-- Row Level Security: unchanged. `runs` already has RLS enabled with a
-- SELECT-only policy for the anon key (001_init.sql), and new columns inherit
-- it, so the dashboard can read the metrics, histories and confusion matrix with
-- the anon key while all writes stay server-side (service-role key). The
-- re-assertion below changes nothing if RLS is already on. `run_artifacts`
-- (the model checkpoint) is NOT touched and stays unreadable by anon.
-- ---------------------------------------------------------------------------
alter table runs enable row level security;

-- Column documentation visible in the Supabase table editor.
comment on column runs.experiment_group      is 'Experiment this run belongs to (e.g. "baseline-vs-improved").';
comment on column runs.run_name              is 'Name of the run within its group; unique per group.';
comment on column runs.model_variant         is 'baseline (SmallCNN) or improved (ImprovedCNN).';
comment on column runs.use_batchnorm         is 'Whether the model really uses BatchNorm (false for baseline).';
comment on column runs.dropout               is 'Dropout probability the model really uses (0 for baseline).';
comment on column runs.data_seed             is 'Seed for generating the images.';
comment on column runs.split_seed            is 'Seed for the train/val/test partition.';
comment on column runs.train_seed            is 'Seed for weights init, batch order, augmentation, dropout.';
comment on column runs.split_sha256          is 'Hash of the exact train/val/test indices; equal hash = identical examples.';
comment on column runs.config                is 'Full training configuration as JSON (result["config"]).';
comment on column runs.accuracy              is 'Final TEST accuracy (best-validation weights).';
comment on column runs.macro_f1              is 'Final TEST macro-F1 (best-validation weights).';
comment on column runs.test_loss             is 'Final TEST cross-entropy loss.';
comment on column runs.confusion_matrix      is 'TEST confusion matrix; rows = true class, columns = predicted class.';
comment on column runs.best_epoch            is '0-based epoch with the lowest validation loss; its weights were kept.';
comment on column runs.artifact_version      is 'Format version of run_artifacts.model_b64 (NULL/1 = original, 2 = has arch/BN/dropout).';

-- ---------------------------------------------------------------------------
-- Rollback (manual, destructive -- deletes the experiment data, never run by
-- accident). Uncomment only if you really need to undo this migration:
--
-- drop index if exists uq_runs_group_run_name, idx_runs_experiment_group;
-- alter table runs
--     drop constraint if exists runs_confusion_matrix_chk,
--     drop constraint if exists runs_config_chk,
--     drop constraint if exists runs_history_lengths_chk,
--     drop column if exists experiment_group, drop column if exists run_name,
--     drop column if exists model_variant, drop column if exists use_batchnorm,
--     drop column if exists dropout, drop column if exists augment,
--     drop column if exists augment_max_shift, drop column if exists scheduler,
--     drop column if exists data_seed, drop column if exists split_seed,
--     drop column if exists train_seed, drop column if exists split_sha256,
--     drop column if exists n_train, drop column if exists n_val,
--     drop column if exists n_test, drop column if exists optimizer,
--     drop column if exists config, drop column if exists test_loss,
--     drop column if exists confusion_matrix, drop column if exists train_loss_history,
--     drop column if exists val_loss_history, drop column if exists val_accuracy_history,
--     drop column if exists lr_history, drop column if exists best_epoch,
--     drop column if exists param_count, drop column if exists train_seconds,
--     drop column if exists artifact_version;
-- ---------------------------------------------------------------------------
