WITH first_visit AS (
  SELECT subject_id, MIN(cohort_start_date) AS index_date
  FROM @cohort_database_schema.@cohort_table
  WHERE cohort_definition_id = @cohort_id
  GROUP BY subject_id
),
first_death AS (
  SELECT person_id, MIN(death_date) AS death_date
  FROM @cdm_database_schema.death
  GROUP BY person_id
)
SELECT
  f.subject_id,
  f.index_date,
  op.observation_period_end_date,
  p.year_of_birth,
  p.month_of_birth,
  p.day_of_birth,
  d.death_date
FROM first_visit f
LEFT JOIN @cdm_database_schema.observation_period op
  ON op.person_id = f.subject_id
  AND f.index_date >= op.observation_period_start_date
  AND f.index_date <= op.observation_period_end_date
LEFT JOIN @cdm_database_schema.person p
  ON p.person_id = f.subject_id
LEFT JOIN first_death d
  ON d.person_id = f.subject_id;
