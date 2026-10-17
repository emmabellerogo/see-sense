# 003 — Recent Classifications view

`003_recent_classifications_view.sql` adds one read-only view,
`public.recent_classifications`, and one index. It changes no existing table,
column, policy, RLS setting or data, and it is safe to run twice.

## What anon can see

| Exposed (view columns) | Never exposed |
|---|---|
| `id`, `run_id`, `class_name`, `confidence`, `created_at` | `sha256`, `width`, `height`, `label`, pixels, model checkpoints (`run_artifacts`), any other table |

Only the newest 200 rows are returned.

## Design decisions

- **A view, not a policy.** RLS filters rows, not columns, so an anon policy on
  `image_metadata` would also expose `sha256`. The table stays closed (RLS on, no
  anon policy, unchanged); the view is the only door.
- **Owner rights (`security_invoker = false`).** The view runs as its owner
  (`postgres`, who owns `image_metadata` and is not subject to its RLS). anon can
  run only this fixed query. With `security_invoker = true` anon's own rights would
  apply and the view would return nothing. Requires PostgreSQL 15+ (Supabase).
- **`security_barrier = true`.** Filters added by the caller can't run ahead of the
  view's query, so they can't be used to probe the underlying rows.
- **Grants reset, then one grant.** Supabase gives anon/authenticated broad default
  rights on new public objects, so the script revokes everything and grants only
  `SELECT` to `anon`. `authenticated` gets nothing (the app uses the anon key
  only); `service_role` keeps its defaults.
- **Index** `idx_image_metadata_created_at` keeps "newest first" cheap as the log grows.

## How to apply (not yet applied)

1. Supabase dashboard → SQL Editor → New query. Confirm it runs as `postgres`.
2. Paste `003_recent_classifications_view.sql` → Run.
3. Paste `003_verify.sql` and run it block by block. Expect:
   - check 1: owner `postgres`, options `security_barrier=true, security_invoker=false`, the five columns;
   - check 2: one grant row, `anon | SELECT`;
   - checks 3–5: no `FAIL` error (they roll back, so nothing is written);
   - check 6: RLS on for all four tables; policies unchanged (only `anon can read runs`).
   Check 3 proves little if `image_metadata` is empty; make one prediction first.
4. Then the UI can query it with the anon key, e.g.
   `supabase.table("recent_classifications").select("*").execute()`.

## Notes for the UI step

- Filters/sorting from PostgREST apply **after** the 200-row cap, so a histogram
  from this view describes the latest 200 classifications, not all time. If an
  all-time distribution is wanted, add a separate aggregate view later.
- If the view returns an error mentioning permission, check the owner in
  verification check 1; an empty result with rows present means the owner can't
  read `image_metadata`.

## Roll back

`drop view if exists public.recent_classifications;` (the index can stay or be dropped
with `drop index if exists idx_image_metadata_created_at;`). No data is affected.
