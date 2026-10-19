#' Create Table 1 for the cohort
#'
#' Extracts aggregated covariates restricted to
#' `inst/concept_sets/selected_features.csv` and renders the Table 1 rows with
#' [FeatureExtraction::createTable1()]. Binary covariates are shown as N (%),
#' continuous ones as median \[Q1, Q3\].
#'
#' @param connectionDetails connectionDetails object
#' @param cdmDatabaseSchema OMOP CDM schema
#' @param cohortDatabaseSchema Cohort schema
#' @param cohortTable Cohort table
#' @param cohortId Cohort id in the cohort table
#' @param outputFolder Folder to write the two tab-separated tables to
#'
#' @returns A list of the conditions/drugs/demographics table and the
#'   measurements table.
#'
#' @export
createTable1Report <- function(connectionDetails,
                               cdmDatabaseSchema,
                               cohortDatabaseSchema,
                               cohortTable,
                               cohortId,
                               outputFolder) {
  specifications <- table1Specifications()
  conceptIds_include <- readr::read_csv(
    system.file("concept_sets/selected_features.csv", package = "HFInYoungWomen"),
    show_col_types = FALSE
  ) %>%
    pull("concept_id")

  covariateSettings_Measurements <- FeatureExtraction::createCovariateSettings(
    includedCovariateConceptIds = conceptIds_include,
    addDescendantsToInclude = TRUE,
    longTermStartDays = -365,
    endDays = 7,
    useMeasurementValueLongTerm = TRUE, # numeric lab values
    #useMeasurementLongTerm = TRUE, # presence indicators
  )

  # All other values are from -730 to 0 end days or all time
  covariateSettings_ConditionsDrugsDems <- FeatureExtraction::createCovariateSettings(
    includedCovariateConceptIds = conceptIds_include,
    addDescendantsToInclude = TRUE,
    useDemographicsAge = TRUE,
    useDemographicsRace = TRUE,
    useConditionEraAnyTimePrior = TRUE, # Condition Eras
    useDrugEraLongTerm = TRUE, # Drug Era
    useDrugEraOverlapping = TRUE,
    useDrugGroupEraLongTerm = TRUE,
    useDrugGroupEraOverlapping = TRUE,
    useObservationValueAsConceptAnyTimePrior = TRUE, # Observation with outcome
    useObservationAnyTimePrior = TRUE, # presence Observation record
    useProcedureOccurrenceAnyTimePrior = TRUE,
    longTermStartDays = -730,
    endDays = 0
  )

  tables <- list(
    conditionsDrugsProcedures = covariateSettings_ConditionsDrugsDems,
    measurements = covariateSettings_Measurements
  )
  tables <- lapply(tables, function(covariateSettings) {
    aggregatedTable1(connectionDetails, cdmDatabaseSchema, cohortDatabaseSchema, cohortTable,
                     cohortId, specifications, conceptIds_include, covariateSettings)
  })

  utils::write.table(tables$conditionsDrugsProcedures,
                     file = file.path(outputFolder, "conditions_drugs_proc_table.txt"), sep = "\t")
  utils::write.table(tables$measurements,
                     file = file.path(outputFolder, "measurements_table.txt"), sep = "\t")
  tables
}

table1Specifications <- function() {
  tibble::tribble(
    ~label,                             ~analysisId, ~covariateIds,

    # Demographics
    "Age, years",                                 2L, "1002",
    "Race",                                        4L, NA_character_,
    "Ethnicity",                                   5L, NA_character_,

    # Vitals
    "Systolic Blood Pressure (mmHg)",             706L, "3004249876706",
    "Diastolic Blood Pressure (mmHg)",            706L, "3012888876706",

    #  Diabetes, by type
    "Type 1 Diabetes Mellitus",                  201L, "201254201,4063042201,4099214201,318712201,43531008201",
    "Type 2 Diabetes Mellitus",                  201L, "43530685201,43531010201,443729201,43531578201,4063043201,201826201,4099651201",

    # Obesity
    "Obesity",                                   201L, "42872398201,433736201,4097996201,434005201,4100857201,45757112201",

    "GFR, mL/min/1.73m2",                        706L, "3049187870706",
    "Chronic Kidney Disease",                    201L, "45763854201,46271022201,44784621201,443611201,443601201,443612201,43531578201,443614201,443597201,45763855201,44782429201",

    "NT-proBNP, pg/mL",                          706L, "3029187845706",

    # Medications
    "ACE Inhibitor Use",                         402L, "1334456402,1335471402,1331235402,1340128402,1308216402,1373225402,1342001402,1363749402,1341927402",
    "Anticoagulant Use",                         402L, "43013024402,1301025402,1367571402,1310149402,40241331402",

    # Autoimmune disorders, by type
    "Systemic Lupus Erythematosus",              201L, "255891201,4066824201,4344158201,4149913201,4063581201,257628201,4055640201,4343923201",
    "Rheumatoid Arthritis",                      201L, "37160562201,4114439201,4117687201,256197201,4116441201,4116151201,4116440201,4035611201,4117686201,37207809201,4115161201,80809201,4083556201",
    "Sj\u00f6gren's Syndrome",                   201L, "4193620201,254443201",
    "Autoimmune Hemolytic Anemia",               201L, "4219853201,4160887201,441269201,36713763201",
    "Autoimmune Thyroiditis",                    201L, "4281109201",
    "Autoimmune Hepatitis",                      201L, "200762201",
    "Myasthenia Gravis",                         201L, "43531561201,43531560201",
    "Scleroderma",                               201L, "441928201",
    "Pernicious Anemia",                         201L, "432295201",

    #  Prior procedures
    "Prior CABG",                                501L, "2107222501,2107223501,2724740501,2724719501,2725227501,2107231501",
    "Prior PCI",                                 501L, "43527994501,43527998501,43527999501,43533247501,43528002501,43528000501"
  )
}

aggregatedTable1 <- function(connectionDetails,
                             cdmDatabaseSchema,
                             cohortDatabaseSchema,
                             cohortTable,
                             cohortId,
                             specifications,
                             conceptIds,
                             covariateSettings) {
  table1CovariateSettings <- FeatureExtraction::createTable1CovariateSettings(
    specifications = specifications,
    covariateSettings = covariateSettings,
    includedCovariateConceptIds = conceptIds,
    addDescendantsToInclude = TRUE
  )

  covariateData <- FeatureExtraction::getDbCovariateData(
    connectionDetails = connectionDetails,
    tempEmulationSchema = cohortDatabaseSchema,
    cohortDatabaseSchema = cohortDatabaseSchema,
    cohortTable = cohortTable,
    cohortIds = cohortId,
    cdmDatabaseSchema = cdmDatabaseSchema,
    rowIdField = "subject_id",
    covariateSettings = table1CovariateSettings,
    aggregated = TRUE
  )

  FeatureExtraction::createTable1(
    covariateData1 = covariateData,
    specifications = specifications,
    output = "one column",
    showCounts = TRUE,
    showPercent = TRUE,
    valueDigits = 1
  )
}
