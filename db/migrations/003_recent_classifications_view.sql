-- 003_recent_classifications_view.sql
-- Read-only, anon-accessible window onto the most recent served predictions for
-- the "Recent Classifications" tab. ADDITIVE ONLY: one view and one index. No
-- table, column, policy, RLS setting or row is changed, dropped or rewritten.
-- Safe to run more than once.
--
-- Apply in the Supabase dashboard: SQL Editor -> New query -> paste -> Run
-- (as the default `postgres` role). See 003_recent_classifications.md.
--
-- WHY A VIEW AND NOT A POLICY ON image_metadata
--   RLS is row-level, not column-level. An anon SELECT policy on image_metadata
--   would expose EVERY column, including `sha256`. So image_metadata keeps RLS
--   enabled with NO anon policy (unchanged), and this view is the only door.
--
-- OWNER / SECURITY BEHAVIOUR (deliberate)
--   security_invoker = false: the view runs with its OWNER's rights (`postgres`,
--   the role that creates it in the SQL editor), which owns image_metadata and so
--   can read it despite RLS. anon never touches the base table; it can only run
--   this fixed query. security_invoker = true would instead apply anon's own
--   (policy-less) rights and return nothing -- and fixing that would need the
--   column-leaking policy above. (security_invoker needs PostgreSQL 15+, which
--   Supabase provides.)
--   security_barrier = true: filters a caller adds (e.g. PostgREST ?class_name=eq.x)
--   cannot run before the view's own query, so they cannot be used to probe
--   rows or functions through the owner's rights.
--
-- WHAT IS EXPOSED: id, run_id, class_name, confidence, created_at -- the newest
-- 200 rows only. NOT exposed: sha256, width, height, label, any model checkpoint
-- (run_artifacts is untouched), or any other table.

-- Supports "newest first" without scanning the whole table as it grows.
create index if not exists idx_image_metadata_created_at
    on image_metadata (created_at desc, id desc);

create or replace view public.recent_classifications
with (security_invoker = false, security_barrier = true) as
select id, run_id, class_name, confidence, created_at
from public.image_metadata
order by created_at desc, id desc
limit 200;

comment on view public.recent_classifications is
    'Newest 200 served predictions (no image hash, no pixels). Read-only for anon; runs with owner rights so image_metadata stays closed to anon.';

-- GRANTS. Supabase's default privileges give anon/authenticated broad rights on
-- new public objects, so reset first, then grant exactly one thing.
revoke all on public.recent_classifications from public, anon, authenticated;
grant select on public.recent_classifications to anon;
-- service_role keeps its default access. `authenticated` is intentionally NOT
-- granted: the app uses only the anon key.
