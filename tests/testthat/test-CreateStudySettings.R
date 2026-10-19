test_that("createStudySettings builds the settings list from arguments", {
  connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = "sqlite", server = ":memory:")

  settings <- createStudySettings(connectionDetails, "main", "scratch", "hf_cohort", cohortId = "42")

  expect_identical(settings$connectionDetails, connectionDetails)
  expect_equal(settings$cohortDatabaseSchema, "scratch")
  expect_identical(settings$cohortId, 42L)
  expect_equal(settings$outputFolder, "output")
  expect_null(settings$pythonEnv)
  expect_true(file.exists(settings$analysisSettingsPath))
})

test_that("createStudySettings rejects missing connection details and empty schemas", {
  connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = "sqlite", server = ":memory:")

  expect_error(createStudySettings(list(), "main", "scratch", "hf_cohort"), "connectionDetails")
  expect_error(createStudySettings(connectionDetails, "", "scratch", "hf_cohort"), "cdmDatabaseSchema")
})

test_that("execute rejects bad connection details before running any step", {
  expect_error(execute(connectionDetails = list(), cdmDatabaseSchema = "main",
                       cohortDatabaseSchema = "scratch", cohortTable = "hf_cohort"),
               "DatabaseConnector::createConnectionDetails")
})

test_that("the cohort step creates the cohort table and runs on Eunomia", {
  skip_if_not_installed("Eunomia")
  skip_on_cran()
  databaseFile <- tryCatch(Eunomia::getDatabaseFile("GiBleed", overwrite = TRUE),
                           error = function(e) skip(paste("Eunomia data unavailable:", conditionMessage(e))))
  connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = "sqlite", server = databaseFile)
  settings <- createStudySettings(connectionDetails, "main", "main", "hf_cohort",
                                  outputFolder = file.path(tempdir(), "hf_output"))
  count <- runCohortCreation(settings)

  connection <- DatabaseConnector::connect(settings$connectionDetails)
  on.exit(DatabaseConnector::disconnect(connection))
  rows <- DatabaseConnector::querySql(connection, "SELECT COUNT(*) AS n FROM main.hf_cohort;")
  expect_equal(count, rows[[1]])
})
