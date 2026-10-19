test_that("covariate windows map onto PheValuator's window arguments", {
  windows <- covariateWindows(matrix(c(0, 31, 181, 30, 180, 365), ncol = 2))

  expect_equal(windows, list(startDayWindow1 = 0, endDayWindow1 = 30, startDayWindow2 = 31,
                             endDayWindow2 = 180, startDayWindow3 = 181, endDayWindow3 = 365))
  expect_equal(covariateWindows(list(c(0, 9999))), list(startDayWindow1 = 0, endDayWindow1 = 9999))
  expect_error(covariateWindows(matrix(1:8, ncol = 2)), "at most three")
})

test_that("the default settings define every PheValuator cohort", {
  options <- jsonlite::read_json(defaultAnalysisSettings(), simplifyVector = TRUE)$phevaluator

  expect_setequal(names(options$cohort_ids), names(PHEVALUATOR_COHORT_SQL))
  for (file in PHEVALUATOR_COHORT_SQL) {
    expect_true(file.exists(system.file("sql/sql_server", file, package = "HFInYoungWomen")))
  }
})
