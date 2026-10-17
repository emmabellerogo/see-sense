-- 004_predictions.sql
-- Adds the `predictions` log (image hash, predicted label, full probability vector,
-- serving run, timestamp), an anon-readable view that joins it to `runs`, and one
-- nullable column on `runs` for per-epoch TRAINING accuracy.
-- ADDITIVE ONLY: no existing table, column, policy, view or row is changed, dropped
-- or rewritten. 001-003 are untouched; `image_metadata` and `recent_classifications`
-- keep working exactly as before. Safe to run more than once.
--
-- NOT APPLIED. Apply in the Supabase dashboard: SQL Editor -> New query -> run as the
-- default `postgres` role -> paste -> Run. See 004_predictions.md for order and checks.
-- Apply BEFORE deploying API code that writes to `predictions`.

-- ---------------------------------------------------------------------------
-- predictions: one row per served prediction. Hash and numbers only, no pixels.
-- ---------------------------------------------------------------------------
create table if not exists public.predictions (
    id                bigint generated always as identity primary key,
    image_sha256      text             not null,   -- SHA-256 of the raw uploaded bytes
    predicted_label   integer          not null check (predicted_label >= 0),
    predicted_class   text             not null,
    probabilities     jsonb            not null,   -- softmax vector, one number per class
    confidence        double precision not null check (confidence >= 0 and confidence <= 1),
    served_by_run_id  bigint           not null references public.runs (id) on delete cascade,
    created_at        timestamptz      not null default now(),
    constraint predictions_probabilities_chk check (
        jsonb_typeof(probabilities) = 'array' and jsonb_array_length(probabilities) >= 2
    )
);

create index if not exists idx_predictions_run_id     on public.predictions (served_by_run_id);
create index if not exists idx_predictions_created_at on public.predictions (created_at desc, id desc);

-- Same stance as image_metadata: RLS on, NO anon policy. The table holds image
-- hashes, so anon reaches it only through the view below. The API writes with the
-- service-role key, which bypasses RLS.
alter table public.predictions enable row level security;
revoke all on public.predictions from anon, authenticated;

-- ---------------------------------------------------------------------------
-- recent_predictions: newest 200 predictions JOINED to the run that served them.
-- Exposes NO image hash. Owner rights (security_invoker = false) so anon never
-- needs a policy on `predictions`; security_barrier stops caller filters from
-- probing rows. Same pattern and rationale as 003.
-- ---------------------------------------------------------------------------
create or replace view public.recent_predictions
with (security_invoker = false, security_barrier = true) as
select p.id,
       p.created_at,
       p.predicted_class,
       p.predicted_label,
       p.confidence,
       p.probabilities,
       p.served_by_run_id,
       r.run_name,
       r.experiment_group,
       r.model_variant,
       r.accuracy  as run_test_accuracy,
       r.macro_f1  as run_test_macro_f1
from public.predictions p
join public.runs r on r.id = p.served_by_run_id
order by p.created_at desc, p.id desc
limit 200;

comment on view public.recent_predictions is
    'Newest 200 predictions joined to the serving run (no image hash). Read-only for anon; owner rights.';

revoke all on public.recent_predictions from public, anon, authenticated;
grant select on public.recent_predictions to anon;

-- ---------------------------------------------------------------------------
-- runs.train_accuracy_history: per-epoch accuracy on the TRAINING split (eval mode,
-- no augmentation), recorded by runs trained after this change. Existing runs stay
-- NULL: those histories were never recorded and are not back-filled or estimated.
-- ---------------------------------------------------------------------------
alter table public.runs
    add column if not exists train_accuracy_history jsonb;

do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'runs_train_accuracy_history_chk') then
        alter table public.runs add constraint runs_train_accuracy_history_chk check (
            train_accuracy_history is null or (
                jsonb_typeof(train_accuracy_history) = 'array'
                and jsonb_array_length(train_accuracy_history) = epochs
            )
        );
    end if;
end
$$;

comment on column public.runs.train_accuracy_history is
    'Per-epoch accuracy on the training split (eval mode, no augmentation). NULL for runs trained before migration 004.';

-- ---------------------------------------------------------------------------
-- Rollback (manual; deletes logged predictions -- never run by accident):
--
-- drop view  if exists public.recent_predictions;
-- drop table if exists public.predictions;
-- alter table public.runs drop constraint if exists runs_train_accuracy_history_chk;
-- alter table public.runs drop column if exists train_accuracy_history;
-- ---------------------------------------------------------------------------
