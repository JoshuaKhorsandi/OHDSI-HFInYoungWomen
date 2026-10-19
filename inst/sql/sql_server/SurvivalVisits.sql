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
  v.visit_concept_id,
  MIN(CASE WHEN v.visit_start_date > f.index_date THEN v.visit_start_date END) AS first_visit_date,
  MAX(CASE WHEN d.death_date >= v.visit_start_date
            AND d.death_date <= COALESCE(v.visit_end_date, d.death_date) THEN 1 ELSE 0 END) AS died_during_visit
FROM first_visit f
JOIN @cdm_database_schema.visit_occurrence v
  ON v.person_id = f.subject_id
LEFT JOIN first_death d
  ON d.person_id = f.subject_id
WHERE v.visit_concept_id IN (@visit_concept_ids)
GROUP BY f.subject_id, v.visit_concept_id;
