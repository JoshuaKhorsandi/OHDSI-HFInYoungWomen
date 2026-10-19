test_that("loadStudySettings reads connection details and schemas from the env file", {
  envPath <- writeEnvFile(c(
    "DB_DBMS=sqlite",
    "DB_SERVER=study.sqlite",
    "CDM_DATABASE_SCHEMA=main",
    "COHORT_DATABASE_SCHEMA=scratch",
    "COHORT_TABLE=hf_cohort",
    "COHORT_ID=42",
    "JAVA_PARAMETERS=-Da=1  -Db=2"
  ))

  settings <- loadStudySettings(envPath)

  expect_s3_class(settings$connectionDetails, "ConnectionDetails")
  expect_equal(settings$connectionDetails$dbms, "sqlite")
  expect_equal(settings$cdmDatabaseSchema, "main")
  expect_equal(settings$cohortDatabaseSchema, "scratch")
  expect_equal(settings$cohortTable, "hf_cohort")
  expect_identical(settings$cohortId, 42L)
  expect_equal(settings$outputFolder, "output")
  expect_null(settings$pythonEnv)
  expect_true(file.exists(settings$analysisSettingsPath))
  expect_equal(getOption("java.parameters"), c("-Da=1", "-Db=2"))
})

test_that("loadStudySettings names the missing key", {
  envPath <- writeEnvFile(c("DB_DBMS=sqlite", "DB_SERVER=study.sqlite", "CDM_DATABASE_SCHEMA=main"))

  expect_error(loadStudySettings(envPath), "COHORT_DATABASE_SCHEMA")
})

test_that("loadStudySettings points to the example when the env file is missing", {
  expect_error(loadStudySettings(tempfile(fileext = ".env")), "R.env.example")
})

test_that("the cohort step creates the cohort table and runs on Eunomia", {
  skip_if_not_installed("Eunomia")
  skip_on_cran()
  databaseFile <- tryCatch(Eunomia::getDatabaseFile("GiBleed", overwrite = TRUE),
                           error = function(e) skip(paste("Eunomia data unavailable:", conditionMessage(e))))
  envPath <- writeEnvFile(c(
    "DB_DBMS=sqlite",
    paste0("DB_SERVER=", databaseFile),
    "CDM_DATABASE_SCHEMA=main",
    "COHORT_DATABASE_SCHEMA=main",
    "COHORT_TABLE=hf_cohort",
    paste0("OUTPUT_FOLDER=", file.path(tempdir(), "hf_output"))
  ))

  settings <- loadStudySettings(envPath)
  count <- runCohortCreation(settings)

  connection <- DatabaseConnector::connect(settings$connectionDetails)
  on.exit(DatabaseConnector::disconnect(connection))
  rows <- DatabaseConnector::querySql(connection, "SELECT COUNT(*) AS n FROM main.hf_cohort;")
  expect_equal(count, rows[[1]])
})
