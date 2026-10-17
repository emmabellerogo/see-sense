-- 005_fashion_mnist_and_atomic_logging.sql
-- (1) Records WHICH dataset a run came from, so Fashion-MNIST runs and the historical
--     synthetic-shape runs can never be confused: new nullable/defaulted columns on
--     `datasets` and `runs`.
-- (2) A Postgres function that writes `image_metadata` and `predictions` in ONE
--     transaction, so a prediction can never be half-logged.
-- ADDITIVE ONLY: no existing table, column, policy, view or row is dropped, rewritten or
-- re-labelled wrongly. Requires 001-004 first. Safe to run more than once.
--
-- NOT APPLIED. Apply in the Supabase dashboard: SQL Editor -> New query -> run as the
-- default `postgres` role -> paste -> Run. See 005_fashion_mnist_and_atomic_logging.md.

-- ---------------------------------------------------------------------------
-- datasets: provenance. Every EXISTING row gets source = 'synthetic_shapes' from the
-- column default, which is what those rows are (their pixels were generated from a seed).
-- New Fashion-MNIST rows are written with source = 'fashion_mnist' by the API.
-- ---------------------------------------------------------------------------
alter table public.datasets
    add column if not exists source      text  not null default 'synthetic_shapes',
    add column if not exists class_names jsonb,
    add column if not exists identity    jsonb;

do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'datasets_source_chk') then
        alter table public.datasets add constraint datasets_source_chk
            check (source in ('synthetic_shapes', 'fashion_mnist'));
    end if;
    if not exists (select 1 from pg_constraint where conname = 'datasets_class_names_chk') then
        alter table public.datasets add constraint datasets_class_names_chk
            check (class_names is null or (jsonb_typeof(class_names) = 'array'
                                           and jsonb_array_length(class_names) = n_classes));
    end if;
    if not exists (select 1 from pg_constraint where conname = 'datasets_identity_chk') then
        alter table public.datasets add constraint datasets_identity_chk
            check (identity is null or jsonb_typeof(identity) = 'object');
    end if;
end
$$;

comment on column public.datasets.source is
    'synthetic_shapes = historical generated 4-class data (pixels from a seed); fashion_mnist = official Fashion-MNIST files (identity holds their SHA-256).';

-- ---------------------------------------------------------------------------
-- runs: measured calibration and per-class test metrics (NULL on all earlier runs).
-- ---------------------------------------------------------------------------
alter table public.runs
    add column if not exists calibration    jsonb,
    add column if not exists test_per_class jsonb;

do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'runs_calibration_chk') then
        alter table public.runs add constraint runs_calibration_chk
            check (calibration is null or jsonb_typeof(calibration) = 'object');
    end if;
    if not exists (select 1 from pg_constraint where conname = 'runs_test_per_class_chk') then
        alter table public.runs add constraint runs_test_per_class_chk
            check (test_per_class is null or jsonb_typeof(test_per_class) = 'object');
    end if;
end
$$;

comment on column public.runs.calibration is
    'Temperature scaling fitted on VALIDATION logits, with ECE/NLL/Brier before and after on validation and test. A fitted temperature is not evidence of good calibration; read the measures.';
comment on column public.runs.test_per_class is
    'Per-class test precision / recall / F1 / support (lists in class order). NULL for runs trained before migration 005.';

-- ---------------------------------------------------------------------------
-- log_prediction: image_metadata row + predictions row in a single transaction.
-- A plpgsql function body is atomic: if either insert fails (unknown run, constraint),
-- both are rolled back and the API returns an error instead of reporting success.
-- SECURITY DEFINER with a pinned search_path; only service_role may execute it, so
-- anon/authenticated cannot write anything through it.
-- ---------------------------------------------------------------------------
create or replace function public.log_prediction(
    p_run_id        bigint,
    p_sha256        text,
    p_width         integer,
    p_height        integer,
    p_label         integer,
    p_class_name    text,
    p_probabilities jsonb,
    p_confidence    double precision
) returns bigint
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_id bigint;
begin
    insert into public.image_metadata (run_id, sha256, width, height, label, class_name, confidence)
    values (p_run_id, p_sha256, p_width, p_height, p_label, p_class_name, p_confidence);

    insert into public.predictions
        (image_sha256, predicted_label, predicted_class, probabilities, confidence, served_by_run_id)
    values
        (p_sha256, p_label, p_class_name, p_probabilities, p_confidence, p_run_id)
    returning id into v_id;

    return v_id;
end
$$;

revoke all on function public.log_prediction(bigint, text, integer, integer, integer, text, jsonb, double precision)
    from public, anon, authenticated;
grant execute on function public.log_prediction(bigint, text, integer, integer, integer, text, jsonb, double precision)
    to service_role;

-- ---------------------------------------------------------------------------
-- Rollback (manual; none of this deletes prediction rows):
--
-- drop function if exists public.log_prediction(bigint, text, integer, integer, integer, text, jsonb, double precision);
-- alter table public.runs     drop constraint if exists runs_calibration_chk, drop constraint if exists runs_test_per_class_chk;
-- alter table public.runs     drop column if exists calibration, drop column if exists test_per_class;
-- alter table public.datasets drop constraint if exists datasets_source_chk, drop constraint if exists datasets_class_names_chk, drop constraint if exists datasets_identity_chk;
-- alter table public.datasets drop column if exists source, drop column if exists class_names, drop column if exists identity;
-- ---------------------------------------------------------------------------
