-- 005_verify.sql: checks after applying 005_fashion_mnist_and_atomic_logging.sql. Run block by block.

-- 1. New columns exist; every pre-existing dataset row is labelled synthetic_shapes.
select column_name, data_type, column_default from information_schema.columns
where table_schema = 'public' and table_name = 'datasets' and column_name in ('source', 'class_names', 'identity');
select source, count(*) from public.datasets group by source;
-- expect: only synthetic_shapes rows until the Fashion-MNIST dataset is registered.

select column_name, data_type from information_schema.columns
where table_schema = 'public' and table_name = 'runs' and column_name in ('calibration', 'test_per_class');

-- 2. The function exists and ONLY service_role may execute it.
select p.proname, p.prosecdef as security_definer, pg_get_userbyid(p.proowner) as owner
from pg_proc p join pg_namespace n on n.oid = p.pronamespace
where n.nspname = 'public' and p.proname = 'log_prediction';
select grantee, privilege_type from information_schema.routine_privileges
where routine_schema = 'public' and routine_name = 'log_prediction' order by grantee;
-- expect no row for anon, authenticated or PUBLIC.

-- 3. Atomicity. An unknown run id must fail AND leave no image_metadata row behind.
--    (Counts must be identical before and after; the failed insert may still consume an identity value.)
select (select count(*) from public.image_metadata) as image_metadata_rows,
       (select count(*) from public.predictions)    as prediction_rows;
-- Run this next statement ON ITS OWN; it is EXPECTED to error with a foreign-key violation:
-- select public.log_prediction(-1, 'verify-atomicity', 28, 28, 0, 'x', '[0.5,0.5]'::jsonb, 0.5);
select (select count(*) from public.image_metadata) as image_metadata_rows,
       (select count(*) from public.predictions)    as prediction_rows;

-- 4. Policies and anon grants are unchanged from 004's expectations.
select tablename, policyname from pg_policies where schemaname = 'public' order by 1, 2;
-- expect only: runs | anon can read runs
