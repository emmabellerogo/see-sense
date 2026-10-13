-- comparison_queries.sql
-- Read-only queries for the baseline-vs-improved experiment (Supabase SQL Editor).
-- In every query, replace 'REPLACE-WITH-GROUP' with your experiment_group.
-- Run names look like  <variant>-seed<N>  (e.g. no_dropout-seed1).
--
-- IMPORTANT: runs.accuracy, runs.macro_f1, runs.test_loss and runs.confusion_matrix
-- are TEST results. Use queries 1-3 (validation / training evidence only) while
-- choosing the protocol. Open queries 4-7 only after the protocol is frozen.

-- =========================================================================
-- 1. [validation only] Matched-configuration audit.
--    Every column should show 1 distinct value per training seed. Anything else
--    means that seed's runs are not matched (except for the declared technique).
-- =========================================================================
select train_seed,
       count(*)                                  as n_runs,
       count(distinct dataset_id)                as datasets,
       count(distinct data_seed)                 as data_seeds,
       count(distinct split_seed)                as split_seeds,
       count(distinct split_sha256)              as split_hashes,
       count(distinct (n_train, n_val, n_test))  as split_sizes,
       count(distinct config ->> 'n_rows')       as n_rows,
       count(distinct config ->> 'noise')        as noise,
       count(distinct epochs)                    as epochs,
       count(distinct batch_size)                as batch_sizes,
       count(distinct lr)                        as initial_lrs,
       count(distinct optimizer)                 as optimizers,
       count(distinct augment_max_shift)         as max_shifts
from runs
where experiment_group = 'REPLACE-WITH-GROUP'
group by train_seed
order by train_seed;

-- =========================================================================
-- 2. [validation only] Settings and training / validation evidence per run.
-- =========================================================================
select run_name,
       model_variant, use_batchnorm, dropout, augment, scheduler,
       train_seed, param_count, best_epoch,
       (val_loss_history   ->> best_epoch)::double precision   as best_val_loss,
       (val_accuracy_history ->> best_epoch)::double precision as val_accuracy_at_best,
       (train_loss_history ->> (epochs - 1))::double precision as final_train_loss,
       round(train_seconds::numeric, 1)                         as train_seconds
from runs
where experiment_group = 'REPLACE-WITH-GROUP'
order by train_seed, run_name;

-- =========================================================================
-- 3. [validation only] Loss and learning-rate curves, one row per run per epoch
--    (for plotting).
-- =========================================================================
select r.run_name, r.train_seed,
       t.ord - 1                  as epoch,
       t.v::double precision      as train_loss,
       v.v::double precision      as val_loss,
       a.v::double precision      as val_accuracy,
       l.v::double precision      as lr
from runs r
cross join lateral jsonb_array_elements_text(r.train_loss_history)   with ordinality as t(v, ord)
join  lateral jsonb_array_elements_text(r.val_loss_history)          with ordinality as v(v, ord) on v.ord = t.ord
join  lateral jsonb_array_elements_text(r.val_accuracy_history)      with ordinality as a(v, ord) on a.ord = t.ord
join  lateral jsonb_array_elements_text(r.lr_history)                with ordinality as l(v, ord) on l.ord = t.ord
where r.experiment_group = 'REPLACE-WITH-GROUP'
order by r.train_seed, r.run_name, epoch;

-- =========================================================================
-- 4. [TEST results - after protocol freeze] Per-run test metrics.
-- =========================================================================
select run_name, model_variant, train_seed,
       accuracy as test_accuracy, macro_f1 as test_macro_f1, test_loss,
       best_epoch, param_count, round(train_seconds::numeric, 1) as train_seconds
from runs
where experiment_group = 'REPLACE-WITH-GROUP'
order by train_seed, run_name;

-- =========================================================================
-- 5. [TEST results - after protocol freeze] Mean and spread across seeds per
--    variant. stddev is NULL with a single seed; do not read much into a spread
--    from two or three seeds.
-- =========================================================================
select split_part(run_name, '-seed', 1)       as variant,
       count(*)                                as n_seeds,
       round(avg(accuracy)::numeric, 4)        as mean_test_accuracy,
       round(stddev_samp(accuracy)::numeric, 4) as sd_test_accuracy,
       round(avg(macro_f1)::numeric, 4)        as mean_test_macro_f1,
       round(stddev_samp(macro_f1)::numeric, 4) as sd_test_macro_f1,
       round(avg(test_loss)::numeric, 4)       as mean_test_loss,
       round(avg(train_seconds)::numeric, 1)   as mean_train_seconds
from runs
where experiment_group = 'REPLACE-WITH-GROUP'
group by 1
order by mean_test_accuracy desc;

-- =========================================================================
-- 6. [TEST results - after protocol freeze] Ablation effect per seed: each
--    ablation (and the baseline) compared with full_improved of the SAME seed.
--    Negative delta = the run scored lower than full_improved.
-- =========================================================================
select a.train_seed,
       split_part(a.run_name, '-seed', 1)               as variant,
       round((a.accuracy - f.accuracy)::numeric, 4)     as delta_accuracy_vs_full,
       round((a.macro_f1 - f.macro_f1)::numeric, 4)     as delta_macro_f1_vs_full,
       round((a.test_loss - f.test_loss)::numeric, 4)   as delta_test_loss_vs_full
from runs a
join runs f
  on f.experiment_group = a.experiment_group
 and f.train_seed       = a.train_seed
 and f.run_name         = 'full_improved-seed' || a.train_seed
where a.experiment_group = 'REPLACE-WITH-GROUP'
  and a.run_name <> f.run_name
order by a.train_seed, variant;

-- =========================================================================
-- 7. [TEST results - after protocol freeze] Confusion matrices, one row per
--    (run, true class, predicted class) cell. Class order follows shared/data.py:
--    0 horizontal, 1 vertical, 2 diagonal, 3 block.
-- =========================================================================
select r.run_name, r.train_seed,
       (array['horizontal','vertical','diagonal','block'])[t.i]  as true_class,
       (array['horizontal','vertical','diagonal','block'])[p.j]  as predicted_class,
       (r.confusion_matrix -> (t.i - 1) ->> (p.j - 1))::integer  as n
from runs r
cross join generate_series(1, 4) as t(i)
cross join generate_series(1, 4) as p(j)
where r.experiment_group = 'REPLACE-WITH-GROUP'
order by r.train_seed, r.run_name, t.i, p.j;

-- =========================================================================
-- 8. Which runs are in which group (find the verification run and old groups).
-- =========================================================================
select experiment_group, count(*) as n_runs, min(created_at) as first_run, max(created_at) as last_run
from runs
group by experiment_group
order by last_run desc nulls last;
