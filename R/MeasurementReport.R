#' Analyze measurement missingness, prevalence, and unit fragmentation
#'
#' For each measurement concept/window that has both a "presence" analysis
#' (e.g. useMeasurementAnyTimePrior, analysis IDs 701-704) and a matching
#' "value" analysis (e.g. useMeasurementValueAnyTimePrior, analysis IDs
#' 705-708) turned on, this reports:
#'   - how many patients had the measurement ordered (presence = 1)
#'   - the prevalence of ordering it (as % of the full cohort)
#'   - how many of those patients actually got a recorded numeric value
#'   - how many were ordered but ended up with NO recorded value
#'     (i.e. the presence covariate fired but the value covariate didn't)
#'
#' It also flags likely "unit fragmentation": cases where what is
#' conceptually the same measurement (the same covariate name once any
#' unit annotation is stripped out) has been split across more than one
#' \code{conceptId}. This typically happens because the source vocabulary
#' maps the same lab test to different standard concepts depending on the
#' unit or method used at the source, which silently reduces the apparent
#' prevalence of each individual covariate. Detection here is name-based
#' (grouping covariates by their name with unit annotations stripped out),
#' since FeatureExtraction does not expose a documented internal flag for
#' this specific failure mode.
#'
#' @param covariateData        A FeatureExtraction \code{CovariateData} object
#'                             (non-aggregated / person-level), or a path
#'                             to one previously saved with
#'                             \code{saveCovariateData()}.
#' @param presenceAnalysisIds  Analysis IDs for presence/absence
#'                             (701:704 = AnyTimePrior/Long/Medium/Short).
#' @param valueAnalysisIds     Analysis IDs for the values
#'                             Default is the standard MeasurementValue
#'                             family (705:708).
#'
#' @return Invisibly returns a list with two data frames:
#'   \describe{
#'     \item{missingness}{One row per concept/window pair, sorted by how
#'       many patients were ordered-but-missing-a-value (worst first).}
#'     \item{fragmentation}{One row per covariate suspected of being part
#'       of a unit-fragmented measurement, grouped by normalized name.}
#'   }
#'   A summary of both is also printed to the console.
#'
#' @export
#' @import dplyr
#' @importFrom rlang .data
#' @importFrom FeatureExtraction loadCovariateData
measurementReport <- function(covariateData,
                                          presenceAnalysisIds = c(701, 702, 703, 704),
                                          valueAnalysisIds    = c(705, 706, 707, 708)) {

  # Allow either an already-loaded CovariateData object or a saved path
  if (is.character(covariateData)) {
    covariateData <- FeatureExtraction::loadCovariateData(covariateData)
  }
  if (!inherits(covariateData, "CovariateData")) {
    stop("covariateData must be a FeatureExtraction CovariateData object, ",
         "or a path to one saved with saveCovariateData().")
  }

  offset <- valueAnalysisIds[1] - presenceAnalysisIds[1]
  if (!all(valueAnalysisIds - presenceAnalysisIds == offset)) {
    stop("presenceAnalysisIds and valueAnalysisIds must be parallel vectors ",
         "(same length, constant offset), e.g. c(701,702,703,704) and c(705,706,707,708).")
  }

  # Small reference table - safe to pull fully into memory
  covariateRef <- covariateData$covariateRef %>% dplyr::collect()

  # Total cohort size, derived from the actual extraction rather than assumed
  totalPatients <- covariateData$covariates %>%
    dplyr::distinct(.data$rowId) %>%
    dplyr::tally() %>%
    dplyr::pull(.data$n)
  if (length(totalPatients) == 0 || is.na(totalPatients) || totalPatients == 0) {
    stop("Could not determine population size from covariateData$covariates.")
  }

  presenceRef <- covariateRef %>%
    dplyr::filter(.data$analysisId %in% presenceAnalysisIds) %>%
    dplyr::transmute(conceptId = .data$conceptId,
                     presenceCovariateId = .data$covariateId,
                     presenceAnalysisId = .data$analysisId,
                     measurementName = .data$covariateName)

  valueRef <- covariateRef %>%
    dplyr::filter(.data$analysisId %in% valueAnalysisIds) %>%
    dplyr::transmute(conceptId = .data$conceptId,
                     valueCovariateId = .data$covariateId,
                     valueAnalysisId = .data$analysisId)

  # Pair up presence & value covariates for the same concept + matching window
  pairs <- presenceRef %>%
    dplyr::mutate(valueAnalysisId = .data$presenceAnalysisId + offset) %>%
    dplyr::inner_join(valueRef, by = c("conceptId", "valueAnalysisId"))

  if (nrow(pairs) == 0) {
    warning("No matching presence/value measurement covariate pairs were found. ",
            "Check that both useMeasurement* and useMeasurementValue* settings ",
            "were enabled when the covariates were created.")
  }

  # --- Unit fragmentation detection ----------------------------------------
  # Strip parenthetical/bracketed unit annotations (e.g. "(mg/dL)", "[mmol/L]")
  # and whitespace, then flag any normalized name that maps to more than one
  # conceptId within the same analysis family.
  normalizeMeasurementName <- function(x) {
    x <- tolower(x)
    x <- gsub("\\s*\\([^)]*\\)", "", x)
    x <- gsub("\\s*\\[[^]]*\\]", "", x)
    trimws(x)
  }

  fragmentationCandidates <- covariateRef %>%
    dplyr::filter(.data$analysisId %in% c(presenceAnalysisIds, valueAnalysisIds)) %>%
    dplyr::mutate(
      baseName = normalizeMeasurementName(.data$covariateName),
      family = ifelse(.data$analysisId %in% presenceAnalysisIds, "presence", "value")
    ) %>%
    dplyr::group_by(.data$baseName, .data$family) %>%
    dplyr::filter(dplyr::n_distinct(.data$conceptId) > 1) %>%
    dplyr::ungroup()

  # --- Pull only the covariate rows we actually need from Andromeda -------
  # (the potentially large table), covering both the missingness pairs and
  # the fragmentation candidates, then collect just that filtered subset.
  relevantIds <- unique(c(pairs$presenceCovariateId, pairs$valueCovariateId,
                          fragmentationCandidates$covariateId))

  subset <- covariateData$covariates %>%
    dplyr::filter(.data$covariateId %in% relevantIds) %>%
    dplyr::select(.data$rowId, .data$covariateId) %>%
    dplyr::collect()

  rowIdsByCovariate <- split(subset$rowId, subset$covariateId)
  getRows <- function(id) {
    x <- rowIdsByCovariate[[as.character(id)]]
    if (is.null(x)) integer(0) else x
  }
  countPatients <- function(id) length(getRows(id))

  # --- Missingness summary --------------------------------------------------
  missingness <- pairs
  if (nrow(pairs) > 0) {
    missingness$nOrdered   <- sapply(pairs$presenceCovariateId, countPatients)
    missingness$nWithValue <- sapply(pairs$valueCovariateId, countPatients)
    missingness$nOrderedMissingValue <- mapply(
      function(p, v) length(setdiff(getRows(p), getRows(v))),
      pairs$presenceCovariateId, pairs$valueCovariateId
    )
    missingness <- missingness %>%
      dplyr::mutate(
        prevalence             = round(100 * .data$nOrdered / totalPatients, 1),
        pctMissingAmongOrdered = ifelse(.data$nOrdered > 0,
                                        round(100 * .data$nOrderedMissingValue / .data$nOrdered, 1),
                                        NA_real_)
      ) %>%
      dplyr::select(.data$conceptId, .data$measurementName,
                    .data$presenceAnalysisId, .data$valueAnalysisId,
                    .data$nOrdered, .data$prevalence, .data$nWithValue,
                    .data$nOrderedMissingValue, .data$pctMissingAmongOrdered) %>%
      dplyr::arrange(dplyr::desc(.data$nOrderedMissingValue))
  }

  # --- Fragmentation summary ------------------------------------------------
  fragmentation <- fragmentationCandidates
  if (nrow(fragmentationCandidates) > 0) {
    fragmentation$nPatients <- sapply(fragmentationCandidates$covariateId, countPatients)
    fragmentation <- fragmentation %>%
      dplyr::group_by(.data$baseName, .data$family) %>%
      dplyr::mutate(combinedPrevalence = round(100 * sum(.data$nPatients) / totalPatients, 1)) %>%
      dplyr::ungroup() %>%
      dplyr::mutate(individualPrevalence = round(100 * .data$nPatients / totalPatients, 1)) %>%
      dplyr::select(.data$baseName, .data$family, .data$conceptId, .data$covariateId,
                    .data$analysisId, .data$covariateName, .data$nPatients,
                    .data$individualPrevalence, .data$combinedPrevalence) %>%
      dplyr::arrange(dplyr::desc(.data$combinedPrevalence), .data$baseName)
  }

  # --- Console summary -------------------------------------------------------
  anyMissing <- nrow(missingness) > 0 && any(missingness$nOrderedMissingValue > 0, na.rm = TRUE)
  anyFragmented <- nrow(fragmentation) > 0

  cat(sprintf(
    "Measurement report: %d patients, %d measurement/window pairs checked\n",
    totalPatients, nrow(missingness)
  ))
  cat(sprintf("Any measurements ordered but missing a value? %s\n",
              ifelse(anyMissing, "YES", "No")))
  cat(sprintf("Any suspected unit fragmentation? %s\n\n",
              ifelse(anyFragmented, "YES", "No")))

  cat("--- Missingness (worst first) ---\n")
  if (nrow(missingness) > 0) {
    print(missingness, n = nrow(missingness))
  } else {
    cat("(no matching presence/value pairs found)\n")
  }

  cat("\n--- Suspected unit fragmentation ---\n")
  if (anyFragmented) {
    print(fragmentation, n = nrow(fragmentation))
  } else {
    cat("(none detected by name-based heuristic; this does not rule out ",
        "fragmentation under differently-worded covariate names)\n", sep = "")
  }

  invisible(list(missingness = missingness, fragmentation = fragmentation))
}
