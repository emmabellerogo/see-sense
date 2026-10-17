-- 003_verify.sql -- run AFTER 003_recent_classifications_view.sql, in the Supabase
-- SQL editor (postgres role). Each numbered block is independent; a failing check
-- raises an exception starting with FAIL. Nothing is committed: every block that
-- switches role or writes is wrapped in a transaction that is rolled back.
-- Needs at least one row in image_metadata for check 2 to prove anything.

-- 1. Definition: owner-rights, barrier on, exactly the five columns, no sha256.
select
    c.relname,
    pg_get_userbyid(c.relowner)  as owner,
    c.reloptions                 as options,          -- expect security_barrier=true, security_invoker=false
    (select string_agg(column_name, ', ' order by ordinal_position)
       from information_schema.columns
      where table_schema = 'public' and table_name = 'recent_classifications') as columns
from pg_class c
where c.oid = 'public.recent_classifications'::regclass;
-- expect columns = id, run_id, class_name, confidence, created_at

-- 2. Grants on the view: expect anon = SELECT only; no authenticated, no write privileges.
select grantee, privilege_type
from information_schema.role_table_grants
where table_schema = 'public' and table_name = 'recent_classifications'
  and grantee in ('anon', 'authenticated', 'PUBLIC')
order by grantee, privilege_type;
-- expect exactly one row: anon | SELECT

-- 3. Anonymous reads work, return only allowed columns, and respect the limit.
begin;
set local role anon;
do $$
declare n int; extra int;
begin
    select count(*) into n from public.recent_classifications;
    if n > 200 then raise exception 'FAIL: view returned % rows (> 200)', n; end if;
    if n = 0 then raise warning 'No rows in image_metadata yet; read path not proven.'; end if;
    -- hash and other columns must not exist on the view
    begin
        execute 'select sha256 from public.recent_classifications limit 1';
        raise exception 'FAIL: sha256 is selectable through the view';
    exception when undefined_column then null;
    end;
    begin
        execute 'select width from public.recent_classifications limit 1';
        raise exception 'FAIL: width is selectable through the view';
    exception when undefined_column then null;
    end;
end $$;
select id, run_id, class_name, confidence, created_at
from public.recent_classifications order by created_at desc limit 5;
rollback;

-- 4. Anonymous writes to the view are denied.
begin;
set local role anon;
do $$
begin
    begin
        insert into public.recent_classifications (run_id, class_name, confidence)
        values (1, 'block', 0.5);
        raise exception 'FAIL: anon INSERT on the view succeeded';
    exception when insufficient_privilege then null;
    end;
    begin
        update public.recent_classifications set class_name = 'x';
        raise exception 'FAIL: anon UPDATE on the view succeeded';
    exception when insufficient_privilege then null;
    end;
    begin
        delete from public.recent_classifications;
        raise exception 'FAIL: anon DELETE on the view succeeded';
    exception when insufficient_privilege then null;
    end;
end $$;
rollback;

-- 5. The base tables stay closed to anon (RLS or privilege denial both count).
begin;
set local role anon;
do $$
declare n int;
begin
    begin
        select count(*) into n from public.image_metadata;
        if n <> 0 then raise exception 'FAIL: anon can read % rows of image_metadata', n; end if;
    exception when insufficient_privilege then null;
    end;
    begin
        select count(*) into n from public.run_artifacts;
        if n <> 0 then raise exception 'FAIL: anon can read % rows of run_artifacts', n; end if;
    exception when insufficient_privilege then null;
    end;
    begin
        insert into public.image_metadata (run_id, sha256, width, height, label, class_name, confidence)
        values (1, 'x', 1, 1, 0, 'block', 0.5);
        raise exception 'FAIL: anon INSERT on image_metadata succeeded';
    exception when insufficient_privilege or check_violation then null;
    end;
end $$;
rollback;

-- 6. RLS is still enabled on every table, and the view added no policies.
select c.relname, c.relrowsecurity as rls_enabled
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public' and c.relname in ('datasets', 'runs', 'run_artifacts', 'image_metadata');
-- expect rls_enabled = true for all four

select tablename, policyname, roles, cmd
from pg_policies where schemaname = 'public' order by tablename, policyname;
-- expect the same policies as before 003 (only "anon can read runs" from 001);
-- none on image_metadata or run_artifacts

select 'verification finished: any FAIL above would have raised an error' as result;
