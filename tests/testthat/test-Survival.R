test_that("event definitions pick the right dates and censor at observation end", {
  index <- as.Date("2020-01-01")
  cohort <- data.frame(
    subjectId = c(1, 2, 3, 4),
    indexDate = index,
    observationPeriodEndDate = index + c(1000, 1000, 1000, 1000),
    yearOfBirth = 1980, monthOfBirth = 6, dayOfBirth = 15,
    deathDate = index + c(300, 400, 500, NA)
  )
  visits <- data.frame(
    subjectId = c(1, 1, 3, 4),
    visitConceptId = c(9201, 9203, 9203, 9201),
    firstVisitDate = index + c(50, 20, 80, 200),
    diedDuringVisit = c(1, 0, 1, 0)
  )
  assignments <- data.frame(subjectId = 1:4, cluster = factor(c("A", "A", "B", "A")))

  inpatient <- buildSurvivalData(cohort, visits, assignments,
                                 list(id = "x", event = "visit", visit_concept_ids = list(9201)))
  either <- buildSurvivalData(cohort, visits, assignments,
                              list(id = "x", event = "visit", visit_concept_ids = list(9201, 9203)))
  deathInpatient <- buildSurvivalData(cohort, visits, assignments,
                                      list(id = "x", event = "death_during_visit", visit_concept_ids = list(9201)))
  mortality <- buildSurvivalData(cohort, visits, assignments,
                                 list(id = "x", event = "death", visit_concept_ids = list()))

  expect_equal(inpatient$duration, c(50, 1000, 1000, 200))
  expect_equal(inpatient$event, c(1, 0, 0, 1))
  expect_equal(either$duration, c(20, 1000, 80, 200))
  expect_equal(deathInpatient$event, c(1, 0, 0, 0))
  expect_equal(deathInpatient$duration, c(300, 1000, 1000, 1000))
  expect_equal(mortality$event, c(1, 1, 1, 0))
  expect_equal(round(mortality$age, 1), rep(39.5, 4))
  expect_equal(levels(mortality$cluster)[1], "A")
})

test_that("the Cox model uses the largest cluster as reference and recovers the hazard ratio", {
  set.seed(1)
  n <- 1500
  code <- sample(0:2, n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
  hazard <- c(1, 1, 2.5)[code + 1] / 900
  eventDay <- round(stats::rexp(n, hazard))
  censorDay <- round(stats::runif(n, 200, 3000))
  index <- as.Date("2019-01-01")

  cohort <- data.frame(
    subjectId = seq_len(n),
    indexDate = index,
    observationPeriodEndDate = index + censorDay,
    yearOfBirth = sample(1970:2000, n, replace = TRUE), monthOfBirth = NA, dayOfBirth = NA,
    deathDate = as.Date(NA)
  )
  observed <- eventDay < censorDay
  visits <- data.frame(subjectId = which(observed), visitConceptId = 9201,
                       firstVisitDate = index + eventDay[observed], diedDuringVisit = 0)
  assignments <- data.frame(subjectId = seq_len(n),
                            cluster = factor(code, levels = 0:2, labels = clusterLabels(0:2)))
  outcome <- list(id = "inpatient", group = "Rehospitalization", label = "Inpatient",
                  event = "visit", visit_concept_ids = list(9201))

  model <- fitSurvivalModels(buildSurvivalData(cohort, visits, assignments, outcome), outcome)

  expect_equal(model$cox$cluster[model$cox$reference], "Cluster 1")
  expect_equal(nrow(model$cox), 3)
  hr <- model$cox$hr[model$cox$cluster == "Cluster 3"]
  expect_gt(hr, 1.8)
  expect_lt(hr, 3.4)
  expect_lt(model$cox$p[model$cox$cluster == "Cluster 3"], 0.001)
  expect_equal(sum(model$counts$patients), n)
  expect_equal(sum(model$counts$events), sum(observed))
  expect_true("GLOBAL" %in% model$proportionalHazards$term)
})

test_that("cluster labels use reviewed names when given", {
  expect_equal(clusterLabels(0:1), c("Cluster 1", "Cluster 2"))
  expect_equal(clusterLabels(0:1, list(`2` = "Cardiorenal")), c("Cluster 1", "2 \u00b7 Cardiorenal"))
})

test_that("the survival report runs end to end on Eunomia", {
  skip_if_not_installed("Eunomia")
  skip_on_cran()
  databaseFile <- tryCatch(Eunomia::getDatabaseFile("GiBleed", overwrite = TRUE),
                           error = function(e) skip(paste("Eunomia data unavailable:", conditionMessage(e))))
  connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = "sqlite", server = databaseFile)
  connection <- DatabaseConnector::connect(connectionDetails)
  persons <- DatabaseConnector::querySql(
    connection,
    "SELECT op.person_id, op.observation_period_start_date
     FROM main.observation_period op
     WHERE op.person_id IN (SELECT DISTINCT person_id FROM main.visit_occurrence)
     ORDER BY op.person_id LIMIT 400;",
    snakeCaseToCamelCase = TRUE
  )
  cohort <- data.frame(
    cohort_definition_id = 1295,
    subject_id = persons$personId,
    cohort_start_date = asDate(persons$observationPeriodStartDate) + 30,
    cohort_end_date = asDate(persons$observationPeriodStartDate) + 60
  )
  DatabaseConnector::insertTable(connection, databaseSchema = "main", tableName = "hf_cohort",
                                 data = cohort, dropTableIfExists = TRUE, createTable = TRUE)
  DatabaseConnector::disconnect(connection)

  outputFolder <- file.path(tempdir(), "survival_test")
  assignmentsPath <- tempfile(fileext = ".parquet")
  nanoparquet::write_parquet(data.frame(rowId = as.numeric(persons$personId),
                                        cluster = rep(0:3, length.out = nrow(persons))),
                             assignmentsPath)

  results <- createSurvivalReport(connectionDetails, "main", "main", "hf_cohort", 1295,
                                  assignmentsPath, outputFolder)

  expect_setequal(unique(results$cox$outcome),
                  c("inpatient", "er", "inpatient_or_er", "death_inpatient", "death_er",
                    "death_inpatient_or_er", "mortality"))
  expect_equal(sum(results$counts$patients[results$counts$outcome == "inpatient"]), nrow(persons))
  expect_true(all(file.exists(file.path(outputFolder, c("cox_results.csv", "proportional_hazards.csv",
                                                        "outcome_counts.csv", "km_summary.csv",
                                                        "km_rehospitalization.png", "km_all_cause_mortality.png",
                                                        "forest_rehospitalization.png")))))
})

test_that("forest rows give one row per cluster and encounter, reference flagged", {
  cox <- data.frame(
    cluster = rep(c("Cluster 1", "Cluster 2"), times = 2),
    label = rep(c("Inpatient", "ER"), each = 2),
    reference = rep(c(TRUE, FALSE), times = 2),
    hr = c(1, 1.3, 1, 0.8), lower = c(1, 1.1, 1, 0.6), upper = c(1, 1.6, 1, 1.1), p = c(NA, 0.01, NA, 0.2)
  )

  rows <- forestRows(cox, c("Cluster 1", "Cluster 2", "Cluster 3"))

  expect_equal(nrow(rows), 4)
  expect_equal(rows$reference, c(TRUE, TRUE, FALSE, FALSE))
  expect_equal(levels(rows$label), c("Inpatient", "ER"))
  expect_true(all(diff(rows$y) < 0))
  expect_null(forestRows(transform(cox, hr = NA_real_), "Cluster 1"))
  expect_equal(formatP(c(NA, 0.0004, 0.0123)), c("NA", "<0.001", "0.012"))
})
