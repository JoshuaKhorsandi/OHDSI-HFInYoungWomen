#' Run the study end to end
#'
#' Runs each study step in order: [runCohortCreation()],
#' [runFeatureExtraction()], [runTable1()], [runMeasurementReport()],
#' optionally [runOptimization()], [runVadeTraining()],
#' [runPosthocAnalysis()] and [runSurvivalAnalysis()].
#' `extras/CodeToRun.R` shows how to set up the connection, and how to run the
#' same steps one at a time.
#'
#' @param connectionDetails Connection details from
#'   [DatabaseConnector::createConnectionDetails()].
#' @param cdmDatabaseSchema CDM and vocabulary schema.
#' @param cohortDatabaseSchema Writable schema for the cohort table and temp
#'   tables.
#' @param cohortTable Cohort table name, created if absent.
#' @param cohortId Cohort definition id of the heart failure cohort.
#' @param outputFolder Folder the steps write to.
#' @param steps Steps to run, any of `"cohort"`, `"features"`, `"table1"`,
#'   `"measurementReport"`, `"vade"`, `"posthoc"` and `"survival"`. Later
#'   steps read the files earlier steps wrote to the output folder, so a subset
#'   can be rerun.
#' @param optimize Re-optimize the VaDE hyperparameters with Optuna before
#'   training, instead of using those in `analysisSettingsPath`.
#' @param pythonEnv `NULL` to let reticulate manage the Python environment
#'   with uv, or the name of, or path to, a virtualenv from
#'   [configurePython()].
#' @param analysisSettingsPath JSON file with the VaDE hyperparameters, the
#'   Optuna, survival, PheValuator and posthoc options.
#'
#' @returns Invisibly, a list holding the settings and each step's result.
#'
#' @export
execute <- function(connectionDetails,
                    cdmDatabaseSchema,
                    cohortDatabaseSchema,
                    cohortTable,
                    cohortId = 1295,
                    outputFolder = "output",
                    steps = c("cohort", "features", "table1", "measurementReport", "vade", "posthoc",
                              "survival"),
                    optimize = FALSE,
                    pythonEnv = NULL,
                    analysisSettingsPath = defaultAnalysisSettings()) {
  steps <- match.arg(steps, several.ok = TRUE)
  settings <- createStudySettings(connectionDetails = connectionDetails,
                                  cdmDatabaseSchema = cdmDatabaseSchema,
                                  cohortDatabaseSchema = cohortDatabaseSchema,
                                  cohortTable = cohortTable,
                                  cohortId = cohortId,
                                  outputFolder = outputFolder,
                                  pythonEnv = pythonEnv,
                                  analysisSettingsPath = analysisSettingsPath)
  results <- list()

  if ("cohort" %in% steps) {
    results$cohort <- runCohortCreation(settings)
  }
  if ("features" %in% steps) {
    results$features <- runFeatureExtraction(settings)
  }
  if ("table1" %in% steps) {
    results$table1 <- runTable1(settings)
  }
  if ("measurementReport" %in% steps) {
    results$measurementReport <- runMeasurementReport(settings)
  }
  if (optimize) {
    results$optimization <- runOptimization(settings)
    settings$analysisSettingsPath <- results$optimization$settings_path
  }
  if ("vade" %in% steps) {
    results$vade <- runVadeTraining(settings)
  }
  if ("posthoc" %in% steps) {
    results$posthoc <- runPosthocAnalysis(settings)
  }
  if ("survival" %in% steps) {
    results$survival <- runSurvivalAnalysis(settings)
  }

  message("Done. Results are in ", normalizePath(settings$outputFolder))
  invisible(c(list(settings = settings), results))
}

#' Study steps
#'
#' The steps [execute()] runs, one function each, so they can be run and
#' inspected one at a time. Each takes the list returned by
#' [createStudySettings()] and writes to its own subfolder of `outputFolder`.
#' Later steps read what earlier ones wrote, so they must run in the order
#' listed here. [runOptimization()] is optional; to train with its result, set
#' `settings$analysisSettingsPath` to the `settings_path` it returns.
#' [runPheValuator()] is an optional validation of the cohort definition that
#' [execute()] never runs; call it after [runCohortCreation()]. It needs the
#' PheValuator package.
#'
#' @param settings List returned by [createStudySettings()].
#'
#' @returns
#' * `runCohortCreation()`: the number of patients in the cohort.
#' * `runFeatureExtraction()`: paths of the feature files written.
#' * `runTable1()`: the two Table 1 data frames.
#' * `runMeasurementReport()`: the missingness and fragmentation data frames.
#' * `runOptimization()`: see [optimizeVadeModel()].
#' * `runVadeTraining()`: see [trainVadeModel()].
#' * `runPosthocAnalysis()`: see [createPosthocFigures()].
#' * `runSurvivalAnalysis()`: see [createSurvivalReport()].
#' * `runPheValuator()`: see [createPheValuatorReport()].
#'
#' @name studySteps
NULL

#' @rdname studySteps
#' @export
runCohortCreation <- function(settings) {
  message("Creating cohort ", settings$cohortId, " in ",
          settings$cohortDatabaseSchema, ".", settings$cohortTable)
  createCohortTable(settings$connectionDetails, settings$cohortDatabaseSchema, settings$cohortTable)
  createCohort(connectionDetails = settings$connectionDetails,
               cdmDatabaseSchema = settings$cdmDatabaseSchema,
               cohortDatabaseSchema = settings$cohortDatabaseSchema,
               cohortTable = settings$cohortTable,
               cohortId = settings$cohortId)
  count <- cohortCount(settings)
  message("Cohort has ", count, " patients")
  invisible(count)
}

#' @rdname studySteps
#' @export
runFeatureExtraction <- function(settings) {
  message("Constructing features")
  files <- featureFiles(settings)
  constructFeatures(connectionDetails = settings$connectionDetails,
                    cdmDatabaseSchema = settings$cdmDatabaseSchema,
                    cohortDatabaseSchema = settings$cohortDatabaseSchema,
                    cohortTable = settings$cohortTable,
                    outputFolder = studyFolder(settings, "features"),
                    cohortId = settings$cohortId,
                    normalize = FALSE,
                    minFraction = 0.01,
                    name = files$name)
  invisible(files)
}

#' @rdname studySteps
#' @export
runTable1 <- function(settings) {
  message("Creating Table 1")
  invisible(createTable1Report(connectionDetails = settings$connectionDetails,
                               cdmDatabaseSchema = settings$cdmDatabaseSchema,
                               cohortDatabaseSchema = settings$cohortDatabaseSchema,
                               cohortTable = settings$cohortTable,
                               cohortId = settings$cohortId,
                               outputFolder = studyFolder(settings, "table1")))
}

#' @rdname studySteps
#' @export
runMeasurementReport <- function(settings) {
  message("Creating measurement report")
  outputFolder <- studyFolder(settings, "measurementReport")
  report <- measurementReport(featureFiles(settings)$measurements)
  readr::write_csv(report$missingness, file.path(outputFolder, "missingness.csv"))
  readr::write_csv(report$fragmentation, file.path(outputFolder, "fragmentation.csv"))
  invisible(report)
}

#' @rdname studySteps
#' @export
runOptimization <- function(settings) {
  message("Re-optimizing VaDE hyperparameters")
  files <- featureFiles(settings)
  optimizeVadeModel(featuresPath = files$features,
                    metadataPath = files$metadata,
                    outputFolder = studyFolder(settings, "optimization"),
                    analysisSettingsPath = settings$analysisSettingsPath,
                    pythonEnv = settings$pythonEnv)
}

#' @rdname studySteps
#' @export
runVadeTraining <- function(settings) {
  message("Training VaDE model and assigning clusters")
  files <- featureFiles(settings)
  trainVadeModel(featuresPath = files$features,
                 metadataPath = files$metadata,
                 outputFolder = studyFolder(settings, "vade"),
                 analysisSettingsPath = settings$analysisSettingsPath,
                 pythonEnv = settings$pythonEnv)
}

#' @rdname studySteps
#' @export
runPosthocAnalysis <- function(settings) {
  message("Creating posthoc figures")
  files <- featureFiles(settings)
  createPosthocFigures(featuresPath = files$features,
                       metadataPath = files$metadata,
                       assignmentsPath = file.path(studyFolder(settings, "vade"), "assignments.parquet"),
                       outputFolder = studyFolder(settings, "plots"),
                       analysisSettingsPath = settings$analysisSettingsPath,
                       pythonEnv = settings$pythonEnv)
}

#' @rdname studySteps
#' @export
runSurvivalAnalysis <- function(settings) {
  message("Fitting survival models by cluster")
  invisible(createSurvivalReport(connectionDetails = settings$connectionDetails,
                                 cdmDatabaseSchema = settings$cdmDatabaseSchema,
                                 cohortDatabaseSchema = settings$cohortDatabaseSchema,
                                 cohortTable = settings$cohortTable,
                                 cohortId = settings$cohortId,
                                 assignmentsPath = file.path(studyFolder(settings, "vade"), "assignments.parquet"),
                                 outputFolder = studyFolder(settings, "survival"),
                                 analysisSettingsPath = settings$analysisSettingsPath))
}

#' @rdname studySteps
#' @export
runPheValuator <- function(settings) {
  message("Validating cohort ", settings$cohortId, " with PheValuator")
  if (cohortCount(settings) == 0) {
    stop("Cohort ", settings$cohortId, " is empty or missing; run runCohortCreation() first.")
  }
  createPheValuatorReport(connectionDetails = settings$connectionDetails,
                          cdmDatabaseSchema = settings$cdmDatabaseSchema,
                          cohortDatabaseSchema = settings$cohortDatabaseSchema,
                          cohortTable = settings$cohortTable,
                          phenotypeCohortId = settings$cohortId,
                          outputFolder = studyFolder(settings, "phevaluator"),
                          analysisSettingsPath = settings$analysisSettingsPath)
}

studyFolder <- function(settings, name) {
  folder <- file.path(settings$outputFolder, name)
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
  folder
}

featureFiles <- function(settings, name = "HFCohort") {
  folder <- studyFolder(settings, "features")
  list(
    name = name,
    features = file.path(folder, paste0("features_", name, ".parquet")),
    metadata = file.path(folder, paste0("meta_", name, ".parquet")),
    measurements = file.path(folder, paste0("Measurement_", name))
  )
}

createCohortTable <- function(connectionDetails, cohortDatabaseSchema, cohortTable) {
  sqlPath <- system.file("sql/sql_server/CreateCohortTable.sql", package = "HFInYoungWomen")
  sql <- SqlRender::render(
    sql = readChar(sqlPath, file.info(sqlPath)$size),
    cohort_database_schema = cohortDatabaseSchema,
    cohort_table = cohortTable
  )
  sql <- SqlRender::translate(sql, targetDialect = connectionDetails$dbms,
                              tempEmulationSchema = cohortDatabaseSchema)
  connection <- DatabaseConnector::connect(connectionDetails)
  on.exit(DatabaseConnector::disconnect(connection))
  DatabaseConnector::executeSql(connection, sql, progressBar = FALSE, reportOverallTime = FALSE)
}

cohortCount <- function(settings) {
  sql <- SqlRender::render(
    "SELECT COUNT(DISTINCT subject_id) AS n FROM @cohort_database_schema.@cohort_table WHERE cohort_definition_id = @cohort_id;",
    cohort_database_schema = settings$cohortDatabaseSchema,
    cohort_table = settings$cohortTable,
    cohort_id = settings$cohortId
  )
  sql <- SqlRender::translate(sql, targetDialect = settings$connectionDetails$dbms)
  connection <- DatabaseConnector::connect(settings$connectionDetails)
  on.exit(DatabaseConnector::disconnect(connection))
  DatabaseConnector::querySql(connection, sql)[[1]]
}
