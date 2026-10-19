#' Construct the desired features using a given cohort, as described by README
#'
#' @param connectionDetails connectionDetails object
#'
#' @param cdmDatabaseSchema cdmDatabaseSchema
#' @param cohortDatabaseSchema cohort database schema
#' @param cohortTable table with cohort id, subject id, start/end date
#' @param outputFolder folder to output features
#' @param cohortId cohort id in cohort table
#' @param normalize Normalize covs between 0 and 1?
#' @param minFraction Minimum fraction to keep
#' @param name Name of output .zip file, holds a duckdb object
#' @returns List of (covariates, metadata)
#'
#' @export
constructFeatures <- function(connectionDetails,
                              cdmDatabaseSchema,
                              cohortDatabaseSchema,
                              cohortTable,
                              outputFolder,
                              cohortId,
                              normalize = TRUE,
                              minFraction = 0.01,
                              name = "CovariateData.zip") {

  if (!file.exists(outputFolder)) {
    warning("Folder '", outputFolder, "' not found. Attempting to create")
    dir.create(outputFolder)
  }

  # Excludes insurance/practice related codes and heart failure itself, plus very common >80% in cohort codes
  excluded_path <- system.file('concept_sets/excluded_conceptIds.csv', package = 'HFInYoungWomen')
  excludedIds <- readr::read_csv(excluded_path) %>%
    pull(`Concept Id`)

  # selected features - validated for EHR prevalence
  conceptSet <- system.file('concept_sets/selected_features.csv', package = 'HFInYoungWomen')
  conceptIds <- readr::read_csv(conceptSet)

  conceptIds_include <- conceptIds %>%
    pull(concept_id)

  # Take measurements from -365 to 7 days rel. to index date
  covariateSettings_Measurements <- FeatureExtraction::createCovariateSettings(
    excludedCovariateConceptIds = excludedIds,
    addDescendantsToExclude = TRUE,
    includedCovariateConceptIds = conceptIds_include,
    addDescendantsToInclude = TRUE,
    longTermStartDays = -365,
    endDays = 7,
    useMeasurementValueLongTerm = TRUE, # numeric lab values
    #useMeasurementLongTerm = TRUE, # presence indicators
  )

  # All other values are from -730 to 0 end days or all time
  covariateSettings_ConditionsDrugsDems <- FeatureExtraction::createCovariateSettings(
    excludedCovariateConceptIds = excludedIds,
    addDescendantsToExclude = TRUE,
    includedCovariateConceptIds = conceptIds_include,
    addDescendantsToInclude = TRUE,
    useDemographicsAge = TRUE,
    useDemographicsRace = TRUE,
    useConditionEraAnyTimePrior = TRUE, # Condition Eras
    useDrugEraLongTerm = TRUE, # Drug Era
    useObservationValueAsConceptAnyTimePrior = TRUE, # Observation with outcome
    useObservationAnyTimePrior = TRUE, # presence Observation record
    useProcedureOccurrenceAnyTimePrior = TRUE,
    longTermStartDays = -730,
    endDays = 0
  )


  covariateData_ConditionsDrugsDems <- FeatureExtraction::getDbCovariateData(
    connectionDetails = connectionDetails,
    tempEmulationSchema = cohortDatabaseSchema,
    cohortDatabaseSchema = cohortDatabaseSchema,
    cohortTable = cohortTable,
    cohortIds = c(cohortId),
    cdmDatabaseSchema = cdmDatabaseSchema,
    rowIdField = "subject_id",
    covariateSettings = covariateSettings_ConditionsDrugsDems
  )

  covariateData_Measurements <- FeatureExtraction::getDbCovariateData(
    connectionDetails = connectionDetails,
    tempEmulationSchema = cohortDatabaseSchema,
    cohortDatabaseSchema = cohortDatabaseSchema,
    cohortTable = cohortTable,
    cohortIds = c(cohortId),
    cdmDatabaseSchema = cdmDatabaseSchema,
    rowIdField = "subject_id",
    covariateSettings = covariateSettings_Measurements
  )

  # Tidy  by optionally normalizing and removing redundancy
  tidyCovariatesConditionsDrugsDems <- FeatureExtraction::tidyCovariateData(
    covariateData = covariateData_ConditionsDrugsDems,
    minFraction = minFraction,
    normalize = normalize, # sets all values to between 0 and 1
    removeRedundancy = TRUE
  )

  tidyCovariatesMeasurements <- FeatureExtraction::tidyCovariateData(
    covariateData = covariateData_Measurements,
    minFraction = minFraction,
    normalize = normalize,
    removeRedundancy = TRUE
  )

  tidyCovariatesMeasurements_data <- tidyCovariatesMeasurements$covariates %>%
    collect()
  tidyCovariatesMeasurements_meta <- tidyCovariatesMeasurements$covariateRef %>%
    collect()

  tidyCovariatesConditionsDrugsDems_data <- tidyCovariatesConditionsDrugsDems$covariates %>%
    collect()
  tidyCovariatesConditionsDrugsDems_meta <- tidyCovariatesConditionsDrugsDems$covariateRef %>%
    collect()

  # Bind the measurements with DrugsDemsConditions
  covariates <- rbind(tidyCovariatesConditionsDrugsDems_data, tidyCovariatesMeasurements_data)
  metadata <- rbind(tidyCovariatesConditionsDrugsDems_meta, tidyCovariatesMeasurements_meta)

  # Save each CovariateData object as DuckDb at path
  FeatureExtraction::saveCovariateData(tidyCovariatesMeasurements, file.path(outputFolder, paste0("Measurement_",name)))
  FeatureExtraction::saveCovariateData(tidyCovariatesConditionsDrugsDems, file.path(outputFolder, paste0("CondDrugsDems_",name)))

  nanoparquet::write_parquet(covariates, file.path(outputFolder, paste0("features_", name, ".parquet")))
  nanoparquet::write_parquet(metadata, file.path(outputFolder, paste0("meta_", name, ".parquet")))

  list(covariates, metadata)
}
