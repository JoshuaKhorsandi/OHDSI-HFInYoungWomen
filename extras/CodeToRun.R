library(HFInYoungWomen)

dotenv::load_dot_env("R.env")

options(java.parameters = c("-Dno.netty.nativeTlsOpenSsl=false", "-Dno.netty.allocType=unpooled"))

connectionString <- Sys.getenv("CONNECTION_STRING") # or create directly

connectionDetails <- DatabaseConnector::createConnectionDetails(
  dbms = "spark",
  connectionString = connectionString,
  pathToDriver = Sys.getenv("PATH_TO_JDBC_DRIVER"),
  user = "token",
  password = Sys.getenv("DATABRICKS_TOKEN")
)

cdmDatabaseSchema <- Sys.getenv("CDM_SCHEMA")
cohortDatabaseSchema <- Sys.getenv("DATABASE_SCHEMA")
cohortTable <- "hf_young_women_cohort"
outputFolder <- "output"

runAll <- FALSE
if (runAll) {
  results <- execute(connectionDetails = connectionDetails,
                     cdmDatabaseSchema = cdmDatabaseSchema,
                     cohortDatabaseSchema = cohortDatabaseSchema,
                     cohortTable = cohortTable,
                     outputFolder = outputFolder)
}

settings <- createStudySettings(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = cdmDatabaseSchema,
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTable = cohortTable,
  outputFolder = outputFolder
)

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
