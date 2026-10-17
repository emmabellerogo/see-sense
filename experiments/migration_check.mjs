// Applies db/migrations/001-005 to a LOCAL in-memory Postgres (PGlite) and checks what the migrations promise:
// idempotent re-runs, historical datasets labelled synthetic_shapes, the atomic log_prediction function (a failed
// call leaves no orphan rows), and that anon can read only the hash-free view. This does NOT touch Supabase.
//
// Needs `npm install @electric-sql/pglite@0.2.17` in some scratch folder; copy this file there (node resolves
// the package relative to the script) and run:   SEE_SENSE_ROOT=<repo root> node migration_check.mjs
//
// Last run: 2026-10-17, PGlite 0.2.17, all checks printed "ok"/expected results (see docs/SUBMISSION_CHECKLIST.md).
import { PGlite } from '@electric-sql/pglite';
import fs from 'fs';
const root = process.env.SEE_SENSE_ROOT || process.argv[2];
const pg = new PGlite();
await pg.exec('create role anon; create role authenticated; create role service_role;');
const files = ['001_init.sql','002_experiment_tracking.sql','003_recent_classifications_view.sql','004_predictions.sql','005_fashion_mnist_and_atomic_logging.sql'];
for (const f of files) { await pg.exec(fs.readFileSync(`${root}/db/migrations/${f}`,'utf8')); console.log('applied', f); }
for (const f of files.slice(3)) { await pg.exec(fs.readFileSync(`${root}/db/migrations/${f}`,'utf8')); console.log('re-applied (idempotent)', f); }
// existing synthetic dataset row gets labelled synthetic_shapes
await pg.exec("insert into datasets (name,n_rows,img_size,n_classes,noise,seed) values ('old',800,28,4,0.1,42)");
console.log((await pg.query("select source, class_names from datasets")).rows);
await pg.exec("insert into datasets (name,n_rows,img_size,n_classes,noise,seed,source,class_names,identity) values ('fm',60000,28,10,0,0,'fashion_mnist','[\"a\",\"b\",\"c\",\"d\",\"e\",\"f\",\"g\",\"h\",\"i\",\"j\"]','{\"x\":1}')");
try { await pg.exec("insert into datasets (name,n_rows,img_size,n_classes,noise,seed,source,class_names) values ('bad',1,28,10,0,0,'fashion_mnist','[\"a\"]')"); console.log('BAD: class_names length not enforced'); } catch(e){ console.log('ok class_names check:', e.message.slice(0,60)); }
await pg.exec("insert into runs (dataset_id,lr,batch_size,epochs,accuracy,macro_f1) values (1,0.001,128,2,0.9,0.9)");
const before = async () => (await pg.query("select (select count(*) from image_metadata)::int a, (select count(*) from predictions)::int b")).rows[0];
console.log('counts before', await before());
const ok = await pg.query("select public.log_prediction(1,'abc',28,28,3,'Dress','[0.1,0.9]'::jsonb,0.9) as id");
console.log('logged id', ok.rows[0], await before());
try { await pg.query("select public.log_prediction(-1,'zzz',28,28,3,'Dress','[0.1,0.9]'::jsonb,0.9)"); console.log('BAD: no error'); } catch(e){ console.log('expected FK error:', e.message.slice(0,70)); }
console.log('counts after failed call (must equal previous)', await before());
try { await pg.query("select public.log_prediction(1,'zzz',28,28,3,'Dress','{\"a\":1}'::jsonb,0.9)"); console.log('BAD: bad probs accepted'); } catch(e){ console.log('expected probabilities check error:', e.message.slice(0,70)); }
console.log('counts after 2nd failed call', await before());
// grants
const g = await pg.query("select grantee, privilege_type from information_schema.routine_privileges where routine_name='log_prediction' order by 1");
console.log('routine privileges', g.rows);
const tg = await pg.query("select grantee, table_name, privilege_type from information_schema.role_table_grants where table_schema='public' and grantee in ('anon','authenticated') and table_name in ('predictions','recent_predictions','image_metadata','run_artifacts') order by 2,1");
console.log('anon/authenticated grants', tg.rows);
const v = await pg.query("select column_name from information_schema.columns where table_name='recent_predictions' order by ordinal_position");
console.log('view columns', v.rows.map(r=>r.column_name).join(','));
// anon can read the view but not the table
await pg.exec('set role anon');
console.log('anon view rows', (await pg.query('select id, predicted_class, run_name from recent_predictions')).rows);
for (const t of ['predictions','image_metadata','run_artifacts']) { try { await pg.query(`select * from ${t}`); console.log('BAD: anon read', t);} catch(e){ console.log('anon denied', t); } }
try { await pg.query("select public.log_prediction(1,'x',28,28,3,'Dress','[0.1,0.9]'::jsonb,0.9)"); console.log('BAD anon can call function'); } catch(e){ console.log('anon denied function'); }
await pg.exec('reset role');
