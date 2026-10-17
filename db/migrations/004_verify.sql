-- 004_verify.sql: read-only checks after applying 004_predictions.sql. Run block by block.

-- 1. The view: owner postgres, security_barrier on, security_invoker off, no sha256 column.
select c.relname, pg_get_userbyid(c.relowner) as owner, c.reloptions
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public' and c.relname = 'recent_predictions';

select column_name from information_schema.columns
where table_schema = 'public' and table_name = 'recent_predictions' order by ordinal_position;
-- expect no image_sha256 / sha256 column.

-- 2. Grants: anon may SELECT the view and nothing on the table.
select grantee, table_name, privilege_type
from information_schema.role_table_grants
where table_schema = 'public' and table_name in ('predictions', 'recent_predictions')
  and grantee in ('anon', 'authenticated')
order by table_name, grantee;
-- expect exactly one row: anon | recent_predictions | SELECT

-- 3. RLS is on for predictions and there is no policy for it.
select relname, relrowsecurity from pg_class
where relname in ('predictions', 'runs', 'image_metadata', 'run_artifacts') and relkind = 'r';
select tablename, policyname from pg_policies where schemaname = 'public' order by 1, 2;
-- expect only: runs | anon can read runs

-- 4. New column and constraint on runs.
select column_name, data_type from information_schema.columns
where table_schema = 'public' and table_name = 'runs' and column_name = 'train_accuracy_history';
select conname from pg_constraint where conname = 'runs_train_accuracy_history_chk';

-- 5. After one prediction through the app: the join returns the serving run.
select id, predicted_class, confidence, served_by_run_id, run_name from public.recent_predictions limit 5;
