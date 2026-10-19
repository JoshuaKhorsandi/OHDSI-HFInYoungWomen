library(HFInYoungWomen)

settings <- loadStudySettings("R.env")

cohortSize <- runCohortCreation(settings)

validatePhenotype <- FALSE
if (validatePhenotype) {
  phevaluator <- runPheValuator(settings)
}

featureFiles <- runFeatureExtraction(settings)

table1 <- runTable1(settings)
table1$conditionsDrugsProcedures
table1$measurements

measurements <- runMeasurementReport(settings)
measurements$fragmentation

reoptimize <- FALSE
if (reoptimize) {
  optimization <- runOptimization(settings)
  settings$analysisSettingsPath <- optimization$settings_path
}

vade <- runVadeTraining(settings)
vade$cluster_sizes
vade$metrics
vade$verification

posthoc <- runPosthocAnalysis(settings)
posthoc$auc
posthoc$figures

survival <- runSurvivalAnalysis(settings)
survival$cox
survival$proportionalHazards

assignments <- nanoparquet::read_parquet(file.path(settings$outputFolder, "vade", "assignments.parquet"))
table(assignments$cluster)
