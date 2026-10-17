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
       count(distinct split_seed)                as split_seeds,
       count(distinct split_sha256)              as split_hashes,
       count(distinct (n_train, n_val, n_test))  as split_sizes,
       count(distinct config ->> 'dataset')      as dataset_names,   -- fashion-mnist (NULL on historical synthetic runs)
       count(distinct config ->> 'val_size')     as val_sizes,
       count(distinct coalesce(config ->> 'train_limit', 'all')) as train_limits,
       count(distinct config -> 'preprocessing') as preprocessings,
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
-- 3. [validation only] Training AND validation loss and accuracy, plus learning rate,
--    one row per run per epoch (for plotting). train_accuracy is NULL for the historical
--    official-supabase-1 runs (never recorded) and present for runs trained after migration 004.
-- =========================================================================
select r.run_name, r.train_seed,
       t.ord - 1                  as epoch,
       t.v::double precision      as train_loss,
       v.v::double precision      as val_loss,
       ta.v::double precision     as train_accuracy,
       a.v::double precision      as val_accuracy,
       l.v::double precision      as lr
from runs r
cross join lateral jsonb_array_elements_text(r.train_loss_history)   with ordinality as t(v, ord)
join  lateral jsonb_array_elements_text(r.val_loss_history)          with ordinality as v(v, ord) on v.ord = t.ord
join  lateral jsonb_array_elements_text(r.val_accuracy_history)      with ordinality as a(v, ord) on a.ord = t.ord
join  lateral jsonb_array_elements_text(r.lr_history)                with ordinality as l(v, ord) on l.ord = t.ord
left join lateral jsonb_array_elements_text(r.train_accuracy_history) with ordinality as ta(v, ord) on ta.ord = t.ord
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
--    (run, true class, predicted class) cell. Class names come from each run's own
--    config (Fashion-MNIST order: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal,
--    Shirt, Sneaker, Bag, Ankle boot); historical synthetic runs fall back to their four shapes.
-- =========================================================================
select r.run_name, r.train_seed,
       coalesce(r.config -> 'class_names' ->> (t.i - 1),
                (array['horizontal','vertical','diagonal','block'])[t.i])  as true_class,
       coalesce(r.config -> 'class_names' ->> (p.j - 1),
                (array['horizontal','vertical','diagonal','block'])[p.j])  as predicted_class,
       (r.confusion_matrix -> (t.i - 1) ->> (p.j - 1))::integer           as n
from runs r
cross join lateral generate_series(1, jsonb_array_length(r.confusion_matrix)) as t(i)
cross join lateral generate_series(1, jsonb_array_length(r.confusion_matrix)) as p(j)
where r.experiment_group = 'REPLACE-WITH-GROUP'
order by r.train_seed, r.run_name, t.i, p.j;

-- =========================================================================
-- 8. Which runs are in which group (find the verification run and old groups).
-- =========================================================================
select experiment_group, count(*) as n_runs, min(created_at) as first_run, max(created_at) as last_run
from runs
group by experiment_group
order by last_run desc nulls last;

-- =========================================================================
-- 9. [TEST results - after protocol freeze] Calibration of the served probabilities:
--    temperature fitted on VALIDATION logits, and ECE / NLL before (T = 1) and after on the
--    held-out test set. Needs migration 005 and runs trained after it (NULL before).
--    Lower ECE/NLL after = better calibrated; this is a measurement, not an assumption.
-- =========================================================================
select run_name, train_seed,
       (calibration ->> 'temperature')::double precision                      as temperature,
       (calibration -> 'validation' -> 'before' ->> 'ece')::double precision  as val_ece_before,
       (calibration -> 'validation' -> 'after'  ->> 'ece')::double precision  as val_ece_after,
       (calibration -> 'test' -> 'before' ->> 'ece')::double precision        as test_ece_before,
       (calibration -> 'test' -> 'after'  ->> 'ece')::double precision        as test_ece_after,
       (calibration -> 'test' -> 'before' ->> 'nll')::double precision        as test_nll_before,
       (calibration -> 'test' -> 'after'  ->> 'nll')::double precision        as test_nll_after
from runs
where experiment_group = 'REPLACE-WITH-GROUP' and calibration is not null
order by train_seed, run_name;

-- =========================================================================
-- 10. [TEST results - after protocol freeze] Per-class test precision, recall, F1 and support,
--     one row per (run, class), exactly as stored with each run (runs.test_per_class).
--     Precision/recall of a class a run never predicts is stored as 0 (macro-F1 convention).
--     Cross-check: experiments.analyze recomputes the same numbers from query 7's confusion cells.
-- =========================================================================
select r.run_name, r.train_seed,
       t.i - 1                                                           as class_index,
       r.config -> 'class_names' ->> (t.i - 1)                           as class_name,
       (r.test_per_class -> 'precision' ->> (t.i - 1))::double precision as precision,
       (r.test_per_class -> 'recall'    ->> (t.i - 1))::double precision as recall,
       (r.test_per_class -> 'f1'        ->> (t.i - 1))::double precision as f1,
       (r.test_per_class -> 'support'   ->> (t.i - 1))::integer          as support
from runs r
cross join lateral generate_series(1, jsonb_array_length(r.test_per_class -> 'precision')) as t(i)
where r.experiment_group = 'REPLACE-WITH-GROUP' and r.test_per_class is not null
order by r.train_seed, r.run_name, t.i;
