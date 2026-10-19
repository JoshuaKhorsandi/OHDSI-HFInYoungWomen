#' Validate the heart failure phenotype with PheValuator
#'
#' Builds the xSpec, xSens, prevalence and evaluation cohorts from the
#' package's original PheValuator SQL into `cohortTable`, then estimates the
#' sensitivity, specificity and predictive values of `phenotypeCohortId` with
#' PheValuator's probabilistic reference standard. The diagnostic model
#' excludes `phevaluator$excluded_concept_ids` (heart failure and NT-proBNP by
#' default) from its covariates so it cannot learn the phenotype's own codes.
#' Options are read from the `phevaluator` section of the analysis settings.
#'
#' Requires the PheValuator package, which is not installed with this one.
#' Not run by [execute()].
#'
#' @param connectionDetails connectionDetails object
#' @param cdmDatabaseSchema OMOP CDM schema
#' @param cohortDatabaseSchema Writable schema holding `cohortTable`, also
#'   used for PheValuator's work and temp tables.
#' @param cohortTable Cohort table that already holds `phenotypeCohortId`
#' @param phenotypeCohortId Id of the phenotype cohort to evaluate.
#' @param outputFolder Folder for PheValuator's working files and exports.
#' @param analysisSettingsPath JSON file with the `phevaluator` options.
#'
#' @returns Invisibly, the summary from
#'   `PheValuator::summarizePheValuatorAnalyses()`, also written to
#'   `phevaluator_summary.csv`.
#'
#' @export
createPheValuatorReport <- function(connectionDetails,
                                    cdmDatabaseSchema,
                                    cohortDatabaseSchema,
                                    cohortTable,
                                    phenotypeCohortId,
                                    outputFolder,
                                    analysisSettingsPath = defaultAnalysisSettings()) {
  if (!requireNamespace("PheValuator", quietly = TRUE)) {
    stop("The PheValuator step needs the PheValuator package: ",
         "remotes::install_github(\"OHDSI/PheValuator\")")
  }
  options <- jsonlite::read_json(analysisSettingsPath, simplifyVector = TRUE)$phevaluator
  cohortIds <- options$cohort_ids
  dir.create(outputFolder, recursive = TRUE, showWarnings = FALSE)

  for (cohort in names(PHEVALUATOR_COHORT_SQL)) {
    message("Creating PheValuator ", cohort, " cohort ", cohortIds[[cohort]])
    createCohort(connectionDetails = connectionDetails,
                 cdmDatabaseSchema = cdmDatabaseSchema,
                 cohortDatabaseSchema = cohortDatabaseSchema,
                 cohortTable = cohortTable,
                 cohortId = cohortIds[[cohort]],
                 sql_path = system.file("sql/sql_server", PHEVALUATOR_COHORT_SQL[[cohort]],
                                        package = "HFInYoungWomen"))
  }

  covariateSettings <- do.call(
    PheValuator::createDefaultCovariateSettings,
    c(list(excludedCovariateConceptIds = options$excluded_concept_ids, addDescendantsToExclude = TRUE),
      covariateWindows(options$covariate_windows))
  )
  analysis <- PheValuator::createPheValuatorAnalysis(
    analysisId = 1,
    description = options$phenotype,
    createEvaluationCohortArgs = PheValuator::createCreateEvaluationCohortArgs(
      xSpecCohortId = cohortIds$xspec,
      xSensCohortId = cohortIds$xsens,
      prevalenceCohortId = cohortIds$prevalence,
      covariateSettings = covariateSettings
    ),
    testPhenotypeAlgorithmArgs = PheValuator::createTestPhenotypeAlgorithmArgs(
      phenotypeCohortId = phenotypeCohortId
    )
  )

  referenceTable <- PheValuator::runPheValuatorAnalyses(
    phenotype = options$phenotype,
    connectionDetails = connectionDetails,
    cdmDatabaseSchema = cdmDatabaseSchema,
    cohortDatabaseSchema = cohortDatabaseSchema,
    cohortTable = cohortTable,
    workDatabaseSchema = cohortDatabaseSchema,
    tempEmulationSchema = cohortDatabaseSchema,
    outputFolder = normalizePath(outputFolder),
    pheValuatorAnalysisList = list(analysis)
  )

  summary <- PheValuator::summarizePheValuatorAnalyses(referenceTable, normalizePath(outputFolder))
  readr::write_csv(summary, file.path(outputFolder, "phevaluator_summary.csv"))
  invisible(summary)
}

covariateWindows <- function(windows) {
  windows <- if (is.matrix(windows)) windows else do.call(rbind, windows)
  if (nrow(windows) > 3) {
    stop("PheValuator takes at most three covariate windows; ", nrow(windows), " were given.")
  }
  args <- list()
  for (i in seq_len(nrow(windows))) {
    args[[paste0("startDayWindow", i)]] <- windows[i, 1]
    args[[paste0("endDayWindow", i)]] <- windows[i, 2]
  }
  args
}

PHEVALUATOR_COHORT_SQL <- c(
  xspec = "CreateXSpecCohort.sql",
  xsens = "CreateXSensCohort.sql",
  prevalence = "CreatePrevalenceCohort.sql",
  evaluation = "CreateEvaluationCohort.sql"
)
