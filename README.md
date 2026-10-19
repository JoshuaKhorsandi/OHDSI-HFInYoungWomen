# Uncovering Distinct Phenotypes of Heart Failure in Women

- Analytics use case(s): **`Patient-Level Prediction`**
- Study type: **Clinical Application**
- Study lead: **Joshua Khorsandi**
- Study start date: **December, 2025**
- Study end date: **October, 2026**

OHDSI study package for unsupervised phenotyping of heart failure in women under 55. It builds the cohort on an OMOP CDM, extracts features, trains a Variational Deep Embedding (VaDE) model, assigns every cohort patient to a cluster and performs survival analysis.

| Step | Function | Output folder |
|--------------------------------------|-----------------|-----------------|
| Create the cohort table if needed and build the heart failure cohort | `runCohortCreation()` | – |
| Extract covariates with FeatureExtraction | `runFeatureExtraction()` | `features/` |
| Aggregated Table 1 | `runTable1()` | `table1/` |
| Measurement missingness and unit fragmentation | `runMeasurementReport()` | `measurementReport/` |
| *Optional:* re-optimize hyperparameters with Optuna | `runOptimization()` | `optimization/` |
| Train the VaDE (S3VDC schedule) and assign clusters | `runVadeTraining()` | `vade/` |
| XGBoost surrogate, SHAP, radar, ridgeline and ROC figures | `runPosthocAnalysis()` | `plots/` |
| Cox and Kaplan-Meier models of outcomes by cluster | `runSurvivalAnalysis()` | `survival/` |
| *Optional, not run by `execute()`:* validate the cohort with PheValuator | `runPheValuator()` | `phevaluator/` |

`execute()` runs them all in order. `extras/CodeToRun.R` runs them one at a time.

## Setup

``` r
remotes::install_local("HFInYoungWomen")
```

Python is managed for you. The package declares its requirements (`inst/python/requirements.txt`) with `reticulate::py_require()`, and the first time a Python step runs, reticulate uses [uv](https://docs.astral.sh/uv/) to download a matching Python (3.11 or 3.12) and build a cached environment. If uv isn't installed, reticulate downloads it.

Where uv can't be used, build a virtualenv with pip instead and pass `pythonEnv = "hfinyoungwomen"` to `execute()` or `createStudySettings()`:

``` r
HFInYoungWomen::configurePython("hfinyoungwomen")
```

Copy `R.env.example` to `R.env` and fill it in. `R.env` holds credentials and is git-ignored. `extras/CodeToRun.R` loads it with `dotenv::load_dot_env("R.env")`.

| Key | Required | Meaning |
|---------------------------------|------------------|----------------------|
| `DATABRICKS_SERVER_HOSTNAME` | yes | Databricks workspace hostname |
| `DATABRICKS_HTTP_PATH` | yes | HTTP path of the SQL warehouse or cluster |
| `DATABRICKS_TOKEN` | yes | Databricks personal access token |
| `PATH_TO_JDBC_DRIVER` | yes | folder holding the JDBC driver |
| `CDM_SCHEMA` | yes | CDM and vocabulary schema |
| `DATABASE_SCHEMA` | yes | writable schema for the cohort table and temp tables |

Any other key, such as `JAVA_HOME`, is exported as an environment variable.

## Running

Connection details are created as [`DatabaseConnector::createConnectionDetails()`](https://ohdsi.github.io/DatabaseConnector/reference/createConnectionDetails.html) objects. An example connection function is described in `extras/CodeToRun`. It assembles a JDBC URL from the `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH` and `DATABRICKS_TOKEN` environment variables, set in `.Renviron` or loaded with `dotenv::load_dot_env()`. Replace that block with the correct connection details, then pass it and the required tables and schemas (loaded from R.env) to `execute()`.

``` r
library(HFInYoungWomen)

dotenv::load_dot_env("R.env")

connectionString <- "<JDBC URL for your database, see extras/CodeToRun.R>"

connectionDetails <- DatabaseConnector::createConnectionDetails(
  dbms = "spark",
  connectionString = connectionString,
  pathToDriver = Sys.getenv("PATH_TO_JDBC_DRIVER"),
  user = "token",
  password = Sys.getenv("DATABRICKS_TOKEN")
)

cdmDatabaseSchema <- Sys.getenv("CDM_SCHEMA")
cohortDatabaseSchema <- Sys.getenv("DATABASE_SCHEMA") # must be writable
cohortTable <- "hf_young_women_cohort"
outputFolder <- "output"

execute(connectionDetails = connectionDetails,
        cdmDatabaseSchema = cdmDatabaseSchema,
        cohortDatabaseSchema = cohortDatabaseSchema,
        cohortTable = cohortTable,
        outputFolder = outputFolder)
```

To run the steps one at a time, build the same arguments into a list with `createStudySettings()` and pass it to each `run*()` function, as `CodeToRun.R` does. Pass `steps` to rerun part of the study, e.g. `execute(connectionDetails, ..., steps = c("vade", "posthoc"))`. Later steps read what earlier ones wrote to the output folder.

------------------------------------------------------------------------

### Hyperparameters

By default the VaDE is trained with parameters from the optimal trial of the original study. Those settings, along with the Optuna and posthoc figure options, are in `inst/settings/analysisSettings.json`.

To re-optimize on your own data, pass `optimize = TRUE`:

``` r
execute(connectionDetails = connectionDetails,
        cdmDatabaseSchema = cdmDatabaseSchema,
        cohortDatabaseSchema = cohortDatabaseSchema,
        cohortTable = cohortTable,
        outputFolder = outputFolder,
        optimize = TRUE)
```

This runs `optimization$n_trials` Optuna trials over the search space in `inst/python/hfvade/optimize.py`. Each trial minimises the held-out VaDE loss and maximises cluster quality (silhouette scaled by the share of clusters left active). The study is stored in `optimization/study.db`, so a rerun adds trials to it. One trial is then taken from the Pareto front by `optimization$selection`:

- `"knee"`, the default: the trial closest to the ideal point once both objectives are scaled across the front. On the original study this picks trial 25.
- `"quality"`: the best cluster quality.
- `"val_loss"`: the lowest held-out loss.

The chosen parameters are written to `optimization/optimizedSettings.json` and used for training and the posthoc figures. To reuse them later without re-optimizing, pass `analysisSettingsPath = "output/optimization/optimizedSettings.json"`.

### Survival models

`runSurvivalAnalysis()` follows each clustered patient from their first cohort entry to the first event or the end of the observation period containing that entry. For each outcome in `survival$outcomes` it fits a Cox model of cluster membership with `survival::coxph()`, stratified by age quartile, using the largest cluster as reference and Efron ties. It checks proportional hazards with `survival::cox.zph()` and draws Kaplan-Meier curves by cluster, with the numbers at risk written to `km_summary.csv`. The default outcomes are:

| Group | Outcomes | Event |
|-------------------|-----------------------------|------------------------|
| Rehospitalization | Inpatient (9201), ER (9203), either | first such visit after index |
| Death during encounter | Inpatient, ER, either | death during such a visit |
| All-cause mortality | All-cause | death |

Add or change outcomes in the settings file: `event` is `"visit"`, `"death_during_visit"` or `"death"`, and `visit_concept_ids` lists the visit concepts.

### PheValuator

`runPheValuator(settings)` estimates the sensitivity, specificity and predictive values of the heart failure cohort with [PheValuator](https://github.com/OHDSI/PheValuator). It isn't part of `execute()`, so run it on its own after `runCohortCreation()` and requires the PheValuator package:

``` r
remotes::install_github("OHDSI/PheValuator")
```

It builds the original xSpec, xSens, prevalence and evaluation cohorts (ids 1081 to 1084) into the study cohort table. Its diagnostic model leaves out the heart failure and NT-proBNP concepts, so it cannot learn the phenotype's own codes. The cohort ids, excluded concepts and covariate windows are in the `phevaluator` section of the settings file.

Once the clusters have been reviewed, set `posthoc.cluster_names` in a copy of the settings file to label the figures.

## Output

```         
output/
  features/            features_HFCohort.parquet, meta_HFCohort.parquet, Andromeda covariate data
  table1/              conditions_drugs_proc_table.txt, measurements_table.txt
  measurementReport/   missingness.csv, fragmentation.csv
  optimization/        study.db, sweep_results.csv, pareto_front.csv, optimizedSettings.json
  vade/                model.pt, assignments.parquet, metrics.json, verification.json,
                       cluster_enrichment.tsv, feature_eta2.tsv, training_curves.png
  plots/               SHAP, ROC, fidelity, radar, ridgeline and waterfall figures
  survival/            cox_results.csv, proportional_hazards.csv, outcome_counts.csv,
                       km_summary.csv, km_<group>.png, forest_<group>.png
  phevaluator/         PheValuator working files, exportFolder/, phevaluator_summary.csv
```

`features/` and `vade/assignments.parquet` are patient level and should not leave the site.

## All of Us

`AoU/` holds the notebooks for running the study on the All of Us Researcher Workbench. They run there rather than through `execute()`; see `AoU/README.md`.
