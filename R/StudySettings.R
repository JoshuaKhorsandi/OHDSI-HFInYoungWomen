#' Create study settings from connection details
#'
#' Collects what the [studySteps] need: connection details, schemas, the
#' cohort table and id, the output folder, the Python environment and the
#' analysis settings file. Its arguments are the same as [execute()]'s, so
#' `CodeToRun.R` can build the connection once and either run everything with
#' [execute()] or step through the study.
#'
#' @param connectionDetails Connection details from
#'   [DatabaseConnector::createConnectionDetails()].
#' @param cdmDatabaseSchema CDM and vocabulary schema.
#' @param cohortDatabaseSchema Writable schema for the cohort table and temp
#'   tables.
#' @param cohortTable Cohort table name, created if absent.
#' @param cohortId Cohort definition id of the heart failure cohort.
#' @param outputFolder Folder the steps write to.
#' @param pythonEnv `NULL` to let reticulate manage the Python environment
#'   with uv, or the name of, or path to, a virtualenv from
#'   [configurePython()].
#' @param analysisSettingsPath JSON file with the VaDE hyperparameters, the
#'   Optuna, survival, PheValuator and posthoc options.
#'
#' @returns A list with `connectionDetails`, `cdmDatabaseSchema`,
#'   `cohortDatabaseSchema`, `cohortTable`, `cohortId`, `outputFolder`,
#'   `pythonEnv` and `analysisSettingsPath`.
#'
#' @examples
#' \dontrun{
#' connectionDetails <- DatabaseConnector::createConnectionDetails(
#'   dbms = "postgresql",
#'   server = "localhost/ohdsi",
#'   user = "joe",
#'   password = Sys.getenv("DB_PASSWORD")
#' )
#' settings <- createStudySettings(
#'   connectionDetails = connectionDetails,
#'   cdmDatabaseSchema = "cdm",
#'   cohortDatabaseSchema = "scratch",
#'   cohortTable = "hf_young_women_cohort"
#' )
#' runCohortCreation(settings)
#' }
#'
#' @export
createStudySettings <- function(connectionDetails,
                                cdmDatabaseSchema,
                                cohortDatabaseSchema,
                                cohortTable,
                                cohortId = 1295,
                                outputFolder = "output",
                                pythonEnv = NULL,
                                analysisSettingsPath = defaultAnalysisSettings()) {
  if (!inherits(connectionDetails, "ConnectionDetails")) {
    stop("connectionDetails must come from DatabaseConnector::createConnectionDetails().")
  }
  for (name in c("cdmDatabaseSchema", "cohortDatabaseSchema", "cohortTable", "outputFolder")) {
    value <- get(name)
    if (!is.character(value) || length(value) != 1 || !nzchar(value)) {
      stop(name, " must be a single, non-empty string.")
    }
  }

  list(
    connectionDetails = connectionDetails,
    cdmDatabaseSchema = cdmDatabaseSchema,
    cohortDatabaseSchema = cohortDatabaseSchema,
    cohortTable = cohortTable,
    cohortId = as.integer(cohortId),
    outputFolder = outputFolder,
    pythonEnv = pythonEnv,
    analysisSettingsPath = normalizePath(analysisSettingsPath, mustWork = TRUE)
  )
}
