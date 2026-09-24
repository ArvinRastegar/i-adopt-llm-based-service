-- Published results view: one row per rankable configuration of the three official
-- campaigns, with micro-averaged Precision, Recall and F1 in both scoring modes.
--
-- The campaign identifiers are pinned deliberately. The database also holds
-- short-lived development campaigns whose numbers are not comparable, so a general
-- view over every campaign would not reproduce the published tables. A future
-- official campaign needs a new forward migration rather than an edit here.
--
-- Metrics are micro-averaged exactly as the ranking computes them: the per-variable
-- confusion mass for the whole decomposition (component '__variable__') is summed
-- over the evaluation population, then the metric is computed once from those totals.
SET search_path TO iadopt_lab, public;

CREATE OR REPLACE VIEW configuration_results AS
WITH official AS (
  SELECT unnest(ARRAY[
    '5cdc9417-28c0-50b9-b24b-8048a6f58ffc',
    '844df00e-846c-50c2-9d0b-1a6deca6f3e7',
    '79b5ec5f-601b-54fd-9313-c0a3b99c3ed0'
  ]::uuid[]) AS campaign_id
),
rankable AS (
  SELECT rr.campaign_id, r.configuration_id
  FROM configuration_rank r
  JOIN ranking_run rr ON rr.id = r.ranking_id
  JOIN official o ON o.campaign_id = rr.campaign_id
  WHERE r.rank IS NOT NULL
),
mass AS (
  SELECT f.configuration_id, f.provider, f.model_id, f.prompt_variant,
         f.shot_count, f.temperature, f.reasoning_mode,
         sum(f.numerator / f.denominator) FILTER (WHERE f.mode = 'close' AND f.metric = 'tp') AS c_tp,
         sum(f.numerator / f.denominator) FILTER (WHERE f.mode = 'close' AND f.metric = 'fp') AS c_fp,
         sum(f.numerator / f.denominator) FILTER (WHERE f.mode = 'close' AND f.metric = 'fn') AS c_fn,
         sum(f.numerator / f.denominator) FILTER (WHERE f.mode = 'exact' AND f.metric = 'tp') AS e_tp,
         sum(f.numerator / f.denominator) FILTER (WHERE f.mode = 'exact' AND f.metric = 'fp') AS e_fp,
         sum(f.numerator / f.denominator) FILTER (WHERE f.mode = 'exact' AND f.metric = 'fn') AS e_fn
  FROM evaluation_facts f
  JOIN rankable k ON k.configuration_id = f.configuration_id
                 AND k.campaign_id     = f.campaign_id
  WHERE f.component = '__variable__'
  GROUP BY 1, 2, 3, 4, 5, 6, 7
)
SELECT model_id,
       provider,
       prompt_variant AS prompt,
       shot_count     AS shots,
       temperature,
       reasoning_mode AS reasoning,
       round(c_tp / nullif(c_tp + c_fp, 0), 4)              AS close_precision,
       round(c_tp / nullif(c_tp + c_fn, 0), 4)              AS close_recall,
       round(2 * c_tp / nullif(2 * c_tp + c_fp + c_fn, 0), 4) AS close_f1,
       round(e_tp / nullif(e_tp + e_fp, 0), 4)              AS exact_precision,
       round(e_tp / nullif(e_tp + e_fn, 0), 4)              AS exact_recall,
       round(2 * e_tp / nullif(2 * e_tp + e_fp + e_fn, 0), 4) AS exact_f1
FROM mass
ORDER BY close_f1 DESC;

-- Grant only to roles that exist: migrate() also runs against bare test databases.
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iadopt_lab_reader') THEN
    GRANT SELECT ON iadopt_lab.configuration_results TO iadopt_lab_reader;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'iadopt_lab_app') THEN
    GRANT SELECT ON iadopt_lab.configuration_results TO iadopt_lab_app;
  END IF;
END $$;
