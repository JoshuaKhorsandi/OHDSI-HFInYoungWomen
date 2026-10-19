#' Load study settings from an env file
#'
#' Reads connection details, schemas, the cohort table and id, the output
#' folder and, optionally, a Python virtualenv from an env file. See
#' `R.env.example` for the keys. The result is what the [studySteps] take.
#'
#' @param envPath Path to the env file.
#' @param analysisSettingsPath JSON file with the VaDE hyperparameters, the
#'   Optuna options and the posthoc figure options.
#'
#' @returns A list with `connectionDetails`, `cdmDatabaseSchema`,
#'   `cohortDatabaseSchema`, `cohortTable`, `cohortId`, `outputFolder`,
#'   `pythonEnv` (the `PYTHON_VIRTUALENV` key, or `NULL` to let reticulate
#'   manage the Python environment with uv) and `analysisSettingsPath`.
#'
#' @export
loadStudySettings <- function(envPath = "R.env", analysisSettingsPath = defaultAnalysisSettings()) {
  connectionDetails <- getConnectionDetails(envPath)

  javaParameters <- Sys.getenv("JAVA_PARAMETERS")
  if (nzchar(javaParameters)) {
    options(java.parameters = strsplit(trimws(javaParameters), "\\s+")[[1]])
  }
  andromedaTempFolder <- Sys.getenv("ANDROMEDA_TEMP_FOLDER")
  if (nzchar(andromedaTempFolder)) {
    options(andromedaTempFolder = andromedaTempFolder)
  }

  list(
    connectionDetails = connectionDetails,
    cdmDatabaseSchema = requiredEnv("CDM_DATABASE_SCHEMA"),
    cohortDatabaseSchema = requiredEnv("COHORT_DATABASE_SCHEMA"),
    cohortTable = requiredEnv("COHORT_TABLE"),
    cohortId = as.integer(optionalEnv("COHORT_ID", "1295")),
    outputFolder = optionalEnv("OUTPUT_FOLDER", "output"),
    pythonEnv = optionalEnv("PYTHON_VIRTUALENV"),
    analysisSettingsPath = normalizePath(analysisSettingsPath, mustWork = TRUE)
  )
}

#' Create DatabaseConnector connection details from an env file
#'
#' @param envPath Path to the env file. `DB_DBMS` is required; `DB_SERVER`,
#'   `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_CONNECTION_STRING`,
#'   `DB_EXTRA_SETTINGS` and `DB_PATH_TO_DRIVER` are passed on when set.
#'
#' @returns A `ConnectionDetails` object.
#'
#' @export
getConnectionDetails <- function(envPath = "R.env") {
  if (!file.exists(envPath)) {
    stop("Env file '", envPath, "' not found. Copy R.env.example to R.env and fill it in.")
  }
  dotenv::load_dot_env(envPath)

  args <- list(
    dbms = requiredEnv("DB_DBMS"),
    server = optionalEnv("DB_SERVER"),
    port = optionalEnv("DB_PORT"),
    user = optionalEnv("DB_USER"),
    password = optionalEnv("DB_PASSWORD"),
    connectionString = optionalEnv("DB_CONNECTION_STRING"),
    extraSettings = optionalEnv("DB_EXTRA_SETTINGS"),
    pathToDriver = optionalEnv("DB_PATH_TO_DRIVER")
  )
  do.call(DatabaseConnector::createConnectionDetails, Filter(Negate(is.null), args))
}

requiredEnv <- function(name) {
  value <- Sys.getenv(name)
  if (!nzchar(value)) {
    stop("Required key '", name, "' is missing from the env file.")
  }
  value
}

optionalEnv <- function(name, default = NULL) {
  value <- Sys.getenv(name)
  if (nzchar(value)) value else default
}
