// Run docs/comparison_queries.sql verbatim on a local Postgres engine (PGlite).
//
// Applies db/migrations/001 + 002 (the same schema as Supabase), loads the runs
// of one experiment group from the offline API database (.local_db.json), then
// executes the requested queries from comparison_queries.sql unchanged except
// for the group name, writing each result to <outdir>/q<N>.csv.
//
//   node experiments/sql_check.mjs <local_db.json> <group> <outdir> 1,2,3[,4,5,6,7]
//
// Needs:  npm install @electric-sql/pglite@0.2.17  in some folder, then copy this file
//         there and run it with SEE_SENSE_ROOT=<repo root>.
// Only for checking queries locally. The source of truth for the team is Supabase.
import { PGlite } from '@electric-sql/pglite';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const [dbFile, group, outDir, which] = process.argv.slice(2);
// SEE_SENSE_ROOT lets you run a copy of this file from the folder where PGlite is installed.
const root = process.env.SEE_SENSE_ROOT || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const wanted = new Set(which.split(',').map(Number));

const pg = new PGlite();
await pg.exec('create role anon; create role authenticated; create role service_role;');
for (const m of ['001_init.sql', '002_experiment_tracking.sql']) {
  await pg.exec(fs.readFileSync(path.join(root, 'db', 'migrations', m), 'utf8'));
}

const local = JSON.parse(fs.readFileSync(dbFile, 'utf8'));
const runs = local.runs.filter(r => r.experiment_group === group);
if (!runs.length) throw new Error(`no runs in group ${group}`);
for (const d of local.datasets.filter(d => runs.some(r => r.dataset_id === d.id))) {
  await pg.query(
    'insert into datasets (id, name, n_rows, img_size, n_classes, noise, seed) overriding system value values ($1,$2,$3,$4,$5,$6,$7)',
    [d.id, d.name, d.n_rows, d.img_size, d.n_classes, d.noise, d.seed]);
}
const jsonCols = new Set(['config', 'confusion_matrix', 'train_loss_history', 'val_loss_history',
  'val_accuracy_history', 'lr_history']);
for (const r of runs) {
  const cols = Object.keys(r).filter(k => k !== 'created_at');
  const vals = cols.map(k => (jsonCols.has(k) && r[k] !== null ? JSON.stringify(r[k]) : r[k]));
  await pg.query(
    `insert into runs (${cols.join(',')}) overriding system value values (${cols.map((_, i) => '$' + (i + 1)).join(',')})`, vals);
}

// Split the SQL file into its numbered queries.
const sql = fs.readFileSync(path.join(root, 'docs', 'comparison_queries.sql'), 'utf8');
const blocks = sql.split(/^-- =+\n-- (\d+)\./m);
fs.mkdirSync(outDir, { recursive: true });
for (let i = 1; i < blocks.length; i += 2) {
  const n = Number(blocks[i]);
  if (!wanted.has(n)) continue;
  const body = blocks[i + 1].split('\n').slice(1).filter(l => !l.startsWith('--')).join('\n')
    .replaceAll('REPLACE-WITH-GROUP', group).trim();
  const res = await pg.query(body);
  const header = res.fields.map(f => f.name);
  const csv = [header.join(',')].concat(res.rows.map(row => header.map(h => {
    const v = row[h];
    return v === null ? '' : (typeof v === 'object' ? JSON.stringify(v) : String(v));
  }).join(','))).join('\n');
  fs.writeFileSync(path.join(outDir, `q${n}.csv`), csv + '\n');
  console.log(`query ${n}: ${res.rows.length} rows -> q${n}.csv`);
}
