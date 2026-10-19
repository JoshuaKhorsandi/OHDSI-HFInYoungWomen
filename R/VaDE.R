#' Train the VaDE model and assign every patient to a cluster
#'
#' Preprocesses the feature matrix, trains a VaDE with the S3VDC schedule using
#' the hyperparameters in `analysisSettingsPath`, and predicts cluster
#' membership for every patient. Writes `model.pt`, `assignments.parquet`,
#' `metrics.json`, `training_history.csv`, `training_curves.png`,
#' `verification.json`, `feature_eta2.tsv` and `cluster_enrichment.tsv` to
#' `outputFolder`.
#'
#' The Python requirements in `inst/python/requirements.txt` are declared with
#' [reticulate::py_require()], so on first use reticulate builds and caches a
#' matching environment with uv, downloading uv itself if needed.
#'
#' @param featuresPath Parquet of (rowId, covariateId, covariateValue) written
#'   by [constructFeatures()].
#' @param metadataPath Matching covariate reference parquet.
#' @param outputFolder Folder to write results to.
#' @param analysisSettingsPath JSON file of hyperparameters.
#' @param pythonEnv `NULL` to let reticulate manage the environment with uv, or
#'   the name of, or path to, a virtualenv from [configurePython()].
#'
#' @returns A list with the evaluation metrics, cluster sizes, verification
#'   statistics and number of training steps run.
#'
#' @export
trainVadeModel <- function(featuresPath,
                           metadataPath,
                           outputFolder,
                           analysisSettingsPath = defaultAnalysisSettings(),
                           pythonEnv = NULL) {
  pipeline <- importPythonModule("hfvade.pipeline", pythonEnv)
  pipeline$run_vade(normalizePath(featuresPath, mustWork = TRUE),
                    normalizePath(metadataPath, mustWork = TRUE),
                    normalizePath(outputFolder, mustWork = FALSE),
                    normalizePath(analysisSettingsPath, mustWork = TRUE))
}

#' Re-optimize the VaDE hyperparameters with Optuna
#'
#' Runs a two-objective Optuna study, minimising the held-out VaDE loss and
#' maximising cluster quality (silhouette scaled by the share of clusters left
#' active), over the search space in `inst/python/hfvade/optimize.py`. Trials
#' are stored in `study.db` in `outputFolder`, so rerunning adds trials to the
#' same study. One trial is then taken from the Pareto front by the
#' `optimization$selection` rule in the settings: `"knee"` (closest to the
#' ideal point once both objectives are scaled to the front), `"quality"` or
#' `"val_loss"`. Its parameters are written to `optimizedSettings.json`.
#'
#' @inheritParams trainVadeModel
#'
#' @returns A list with `settings_path` (the optimized settings file to pass
#'   to [trainVadeModel()]), the selected trial and its scores, the number of
#'   finished trials and the trials on the Pareto front.
#'
#' @export
optimizeVadeModel <- function(featuresPath,
                              metadataPath,
                              outputFolder,
                              analysisSettingsPath = defaultAnalysisSettings(),
                              pythonEnv = NULL) {
  optimize <- importPythonModule("hfvade.optimize", pythonEnv)
  optimize$run_optimization(normalizePath(featuresPath, mustWork = TRUE),
                            normalizePath(metadataPath, mustWork = TRUE),
                            normalizePath(outputFolder, mustWork = FALSE),
                            normalizePath(analysisSettingsPath, mustWork = TRUE))
}

#' Draw posthoc explanation figures for the cluster assignments
#'
#' Trains an XGBoost surrogate to predict the VaDE cluster labels and writes
#' SHAP, ROC, fidelity, radar, ridgeline and per-cluster waterfall figures to
#' `outputFolder`, along with the surrogate's AUCs and classification report.
#'
#' @inheritParams trainVadeModel
#' @param assignmentsPath `assignments.parquet` written by [trainVadeModel()].
#'
#' @returns A list with the surrogate AUCs, the top features by mean absolute
#'   SHAP value and the names of the figures written.
#'
#' @export
createPosthocFigures <- function(featuresPath,
                                 metadataPath,
                                 assignmentsPath,
                                 outputFolder,
                                 analysisSettingsPath = defaultAnalysisSettings(),
                                 pythonEnv = NULL) {
  posthoc <- importPythonModule("hfvade.posthoc", pythonEnv)
  posthoc$run_posthoc(normalizePath(featuresPath, mustWork = TRUE),
                      normalizePath(metadataPath, mustWork = TRUE),
                      normalizePath(assignmentsPath, mustWork = TRUE),
                      normalizePath(outputFolder, mustWork = FALSE),
                      normalizePath(analysisSettingsPath, mustWork = TRUE))
}

#' Create a virtualenv for the VaDE model without uv
#'
#' Only needed where reticulate cannot use uv, for example on a machine
#' without access to uv's downloads. Builds a virtualenv with pip from the
#' packages pinned in `inst/python/requirements.txt`; set `PYTHON_VIRTUALENV`
#' in the env file to `envname` to use it.
#'
#' @param envname Name of, or path to, the virtualenv.
#' @param python Python interpreter (3.11 or 3.12) to build the environment
#'   from. `NULL` picks the newest one found outside the Microsoft Store, whose
#'   Python reticulate cannot load.
#'
#' @returns Invisibly, `envname`.
#'
#' @export
configurePython <- function(envname = "hfinyoungwomen", python = NULL) {
  if (is.null(python)) {
    python <- findStarterPython()
  }
  reticulate::virtualenv_create(envname = envname, python = python,
                                requirements = requirementsFile())
  invisible(envname)
}

defaultAnalysisSettings <- function() {
  system.file("settings", "analysisSettings.json", package = "HFInYoungWomen")
}

importPythonModule <- function(module, pythonEnv = NULL) {
  if (!reticulate::py_available()) {
    if (is.null(pythonEnv)) {
      reticulate::py_require(pythonRequirements(), python_version = PYTHON_VERSION)
    } else {
      reticulate::use_virtualenv(pythonEnv, required = TRUE)
    }
  }
  reticulate::import_from_path(module, path = system.file("python", package = "HFInYoungWomen"))
}

findStarterPython <- function() {
  starters <- reticulate::virtualenv_starter(PYTHON_VERSION, all = TRUE)
  starters <- starters[!grepl("WindowsApps", starters$path, fixed = TRUE), , drop = FALSE]
  if (nrow(starters) == 0) {
    stop("No Python 3.11 or 3.12 found outside the Microsoft Store. Install one with ",
         "reticulate::install_python(\"3.12\"), or pass its path as `python`.")
  }
  starters$path[order(starters$version, decreasing = TRUE)[1]]
}

pythonRequirements <- function() {
  lines <- trimws(readLines(requirementsFile()))
  lines[nzchar(lines)]
}

requirementsFile <- function() {
  system.file("python", "requirements.txt", package = "HFInYoungWomen")
}

PYTHON_VERSION <- ">=3.11,<3.13"
