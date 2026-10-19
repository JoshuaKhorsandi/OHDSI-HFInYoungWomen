#' Fit Cox and Kaplan-Meier models of outcomes by cluster
#'
#' For each outcome in the `survival` section of the analysis settings,
#' follows every clustered cohort patient from their first cohort entry to the
#' first event or the end of the observation period containing that entry.
#' It then fits a Cox model of cluster membership, stratified by age quantile,
#' with the largest cluster as reference and Efron ties. Proportional hazards
#' are tested with [survival::cox.zph()] and Kaplan-Meier curves are drawn per
#' cluster.
#'
#' An outcome's `event` is one of:
#' * `"visit"`: the first visit of a type in `visit_concept_ids` after index.
#' * `"death_during_visit"`: death during a visit of those types.
#' * `"death"`: death from any cause.
#'
#' Writes `cox_results.csv`, `proportional_hazards.csv`, `outcome_counts.csv`
#' and `km_summary.csv` (survival and numbers at risk each year) to
#' `outputFolder`. For each outcome `group` it also writes `km_<group>.png`,
#' with one Kaplan-Meier panel per outcome, and `forest_<group>.png`. Both
#' follow the cluster palette of the posthoc ridge and radar figures; the
#' forest plot's axis is set by `survival$forest_xlim` and `survival$forest_log`.
#'
#' @param connectionDetails connectionDetails object
#' @param cdmDatabaseSchema OMOP CDM schema
#' @param cohortDatabaseSchema Cohort schema
#' @param cohortTable Cohort table
#' @param cohortId Cohort id in the cohort table
#' @param assignmentsPath `assignments.parquet` written by [trainVadeModel()].
#' @param outputFolder Folder to write results to.
#' @param analysisSettingsPath JSON file with the `survival` outcomes and, for
#'   labels, `posthoc$cluster_names`.
#'
#' @returns Invisibly, a list of the `cox`, `proportionalHazards`, `counts` and
#'   `kmSummary` data frames, one block of rows per outcome.
#'
#' @importFrom survival cox.zph coxph strata Surv survfit
#' @export
createSurvivalReport <- function(connectionDetails,
                                 cdmDatabaseSchema,
                                 cohortDatabaseSchema,
                                 cohortTable,
                                 cohortId,
                                 assignmentsPath,
                                 outputFolder,
                                 analysisSettingsPath = defaultAnalysisSettings()) {
  analysisSettings <- jsonlite::read_json(analysisSettingsPath, simplifyVector = FALSE)
  survivalSettings <- analysisSettings$survival
  outcomes <- survivalSettings$outcomes
  dir.create(outputFolder, recursive = TRUE, showWarnings = FALSE)

  assignments <- readAssignments(assignmentsPath, analysisSettings$posthoc$cluster_names)
  colours <- stats::setNames(CLUSTER_COLOURS[seq_along(levels(assignments$cluster))],
                             levels(assignments$cluster))
  visitConceptIds <- unique(unlist(lapply(outcomes, function(outcome) outcome$visit_concept_ids)))
  extracted <- extractSurvivalData(connectionDetails, cdmDatabaseSchema, cohortDatabaseSchema,
                                   cohortTable, cohortId, visitConceptIds)

  models <- lapply(outcomes, function(outcome) {
    data <- buildSurvivalData(extracted$cohort, extracted$visits, assignments, outcome,
                              survivalSettings$age_strata)
    fitSurvivalModels(data, outcome)
  })

  results <- list(
    cox = do.call(rbind, lapply(models, `[[`, "cox")),
    proportionalHazards = do.call(rbind, lapply(models, `[[`, "proportionalHazards")),
    counts = do.call(rbind, lapply(models, `[[`, "counts")),
    kmSummary = do.call(rbind, lapply(models, kmSummary, times = seq(0, survivalSettings$max_years)))
  )
  readr::write_csv(results$cox, file.path(outputFolder, "cox_results.csv"))
  readr::write_csv(results$proportionalHazards, file.path(outputFolder, "proportional_hazards.csv"))
  readr::write_csv(results$counts, file.path(outputFolder, "outcome_counts.csv"))
  readr::write_csv(results$kmSummary, file.path(outputFolder, "km_summary.csv"))

  groups <- vapply(models, function(model) model$outcome$group, character(1))
  for (group in unique(groups)) {
    slug <- gsub("[^a-z0-9]+", "_", tolower(group))
    saveKaplanMeierPanels(models[groups == group], file.path(outputFolder, paste0("km_", slug, ".png")),
                          survivalSettings$max_years, colours)
    saveForestPlot(results$cox[results$cox$group == group, , drop = FALSE],
                   file.path(outputFolder, paste0("forest_", slug, ".png")), colours,
                   xlim = unlist(survivalSettings$forest_xlim), log = isTRUE(survivalSettings$forest_log))
  }

  invisible(results)
}

readAssignments <- function(assignmentsPath, clusterNames = NULL) {
  assignments <- nanoparquet::read_parquet(assignmentsPath)
  codes <- sort(unique(assignments$cluster))
  data.frame(
    subjectId = as.numeric(assignments$rowId),
    cluster = factor(assignments$cluster, levels = codes, labels = clusterLabels(codes, clusterNames))
  )
}

extractSurvivalData <- function(connectionDetails,
                                cdmDatabaseSchema,
                                cohortDatabaseSchema,
                                cohortTable,
                                cohortId,
                                visitConceptIds) {
  connection <- DatabaseConnector::connect(connectionDetails)
  on.exit(DatabaseConnector::disconnect(connection))

  cohort <- querySurvivalSql(connection, "SurvivalCohort.sql", cdmDatabaseSchema,
                             cohortDatabaseSchema, cohortTable, cohortId)
  visits <- if (length(visitConceptIds) > 0) {
    querySurvivalSql(connection, "SurvivalVisits.sql", cdmDatabaseSchema, cohortDatabaseSchema,
                     cohortTable, cohortId, visit_concept_ids = visitConceptIds)
  } else {
    data.frame(subjectId = numeric(), visitConceptId = numeric(),
               firstVisitDate = as.Date(character()), diedDuringVisit = numeric())
  }

  cohort[c("indexDate", "observationPeriodEndDate", "deathDate")] <-
    lapply(cohort[c("indexDate", "observationPeriodEndDate", "deathDate")], asDate)
  visits$firstVisitDate <- asDate(visits$firstVisitDate)
  list(cohort = cohort, visits = visits)
}

buildSurvivalData <- function(cohort, visits, assignments, outcome, ageStrata = 4) {
  data <- merge(cohort, assignments, by = "subjectId")
  data$eventDate <- outcomeEventDate(data, visits, outcome)
  endDate <- dplyr::coalesce(data$eventDate, data$observationPeriodEndDate)
  data$duration <- as.numeric(endDate - data$indexDate)
  data$event <- as.integer(!is.na(data$eventDate))

  birthDate <- as.Date(sprintf("%04d-%02d-%02d",
                               as.integer(data$yearOfBirth),
                               dplyr::coalesce(as.integer(data$monthOfBirth), 1L),
                               dplyr::coalesce(as.integer(data$dayOfBirth), 1L)),
                       optional = TRUE)
  data$age <- as.numeric(data$indexDate - birthDate) / 365.25

  data <- data[!is.na(data$duration) & data$duration >= 0 & !is.na(data$age), , drop = FALSE]
  data$ageStratum <- ageStrataOf(data$age, ageStrata)
  data$cluster <- droplevels(data$cluster)
  data$cluster <- stats::relevel(data$cluster, ref = names(which.max(table(data$cluster))))
  data
}

fitSurvivalModels <- function(data, outcome) {
  message("Survival: ", outcome$group, " - ", outcome$label, ", ",
          nrow(data), " patients, ", sum(data$event), " events")
  reference <- levels(data$cluster)[1]
  counts <- data %>%
    group_by(.data$cluster) %>%
    summarise(patients = n(), events = sum(.data$event), personYears = sum(.data$duration) / 365.25) %>%
    mutate(cluster = as.character(.data$cluster)) %>%
    as.data.frame()

  fit <- if (sum(data$event) == 0 || nlevels(data$cluster) < 2) NULL else tryCatch(
    coxph(Surv(duration, event) ~ cluster + strata(ageStratum), data = data, ties = "efron"),
    error = function(e) {
      warning("Cox model for '", outcome$id, "' failed: ", conditionMessage(e), call. = FALSE)
      NULL
    }
  )

  list(
    outcome = outcome,
    cox = withOutcome(coxTable(fit, reference, counts), outcome),
    proportionalHazards = withOutcome(proportionalHazardsTable(fit), outcome),
    counts = withOutcome(counts, outcome),
    km = survfit(Surv(duration / 365, event) ~ cluster, data = data, conf.type = "log-log")
  )
}

querySurvivalSql <- function(connection,
                             fileName,
                             cdmDatabaseSchema,
                             cohortDatabaseSchema,
                             cohortTable,
                             cohortId,
                             ...) {
  sqlPath <- system.file("sql/sql_server", fileName, package = "HFInYoungWomen")
  sql <- SqlRender::render(readChar(sqlPath, file.info(sqlPath)$size),
                           cdm_database_schema = cdmDatabaseSchema,
                           cohort_database_schema = cohortDatabaseSchema,
                           cohort_table = cohortTable,
                           cohort_id = cohortId,
                           ...)
  sql <- SqlRender::translate(sql, targetDialect = DatabaseConnector::dbms(connection),
                              tempEmulationSchema = cohortDatabaseSchema)
  DatabaseConnector::querySql(connection, sql, snakeCaseToCamelCase = TRUE)
}

outcomeEventDate <- function(data, visits, outcome) {
  matching <- visits[visits$visitConceptId %in% unlist(outcome$visit_concept_ids), , drop = FALSE]

  if (outcome$event == "visit") {
    matching <- matching[!is.na(matching$firstVisitDate), , drop = FALSE]
    if (nrow(matching) == 0) {
      return(as.Date(rep(NA, nrow(data))))
    }
    first <- matching %>%
      group_by(.data$subjectId) %>%
      summarise(eventDate = min(.data$firstVisitDate))
    return(first$eventDate[match(data$subjectId, first$subjectId)])
  }
  if (outcome$event == "death_during_visit") {
    dates <- data$deathDate
    dates[!data$subjectId %in% matching$subjectId[matching$diedDuringVisit == 1]] <- NA
    return(dates)
  }
  if (outcome$event == "death") {
    return(data$deathDate)
  }
  stop("Unknown survival event '", outcome$event, "' for outcome '", outcome$id,
       "'; use 'visit', 'death_during_visit' or 'death'.")
}

ageStrataOf <- function(age, strata) {
  breaks <- unique(stats::quantile(age, probs = seq(0, 1, length.out = strata + 1), na.rm = TRUE))
  if (length(breaks) < 2) {
    return(factor(rep("all", length(age))))
  }
  cut(age, breaks, include.lowest = TRUE)
}

coxTable <- function(fit, reference, counts) {
  if (is.null(fit)) {
    estimates <- data.frame(cluster = counts$cluster, hr = NA_real_, lower = NA_real_,
                            upper = NA_real_, p = NA_real_)
  } else {
    coefficients <- summary(fit)$coefficients
    intervals <- summary(fit)$conf.int
    estimates <- data.frame(
      cluster = c(reference, sub("^cluster", "", rownames(coefficients))),
      hr = c(1, coefficients[, "exp(coef)"]),
      lower = c(1, intervals[, "lower .95"]),
      upper = c(1, intervals[, "upper .95"]),
      p = c(NA, coefficients[, "Pr(>|z|)"])
    )
  }
  if (all(is.na(estimates$hr[estimates$cluster != reference]))) {
    estimates[c("hr", "lower", "upper", "p")] <- NA_real_
  }
  estimates$reference <- estimates$cluster == reference
  cbind(estimates, counts[match(estimates$cluster, counts$cluster), -1, drop = FALSE], row.names = NULL)
}

proportionalHazardsTable <- function(fit) {
  test <- if (is.null(fit)) NULL else tryCatch(cox.zph(fit, transform = "km"), error = function(e) NULL)
  if (is.null(test)) {
    return(data.frame(term = character(), chisq = numeric(), df = numeric(), p = numeric()))
  }
  data.frame(term = rownames(test$table), chisq = test$table[, "chisq"],
             df = test$table[, "df"], p = test$table[, "p"], row.names = NULL)
}

kmSummary <- function(model, times) {
  risk <- summary(model$km, times = times, extend = TRUE)
  strata <- names(kmStrata(model$km))
  cluster <- if (is.null(risk$strata)) strata else stripStrata(as.character(risk$strata))
  withOutcome(data.frame(cluster = cluster, years = risk$time, atRisk = risk$n.risk,
                         survival = risk$surv, lower = risk$lower, upper = risk$upper),
              model$outcome)
}

withOutcome <- function(table, outcome) {
  if (nrow(table) == 0) {
    return(cbind(data.frame(outcome = character(), group = character(), label = character()), table))
  }
  cbind(data.frame(outcome = outcome$id, group = outcome$group, label = outcome$label), table)
}

clusterLabels <- function(codes, clusterNames = NULL) {
  vapply(codes, function(code) {
    name <- clusterNames[[as.character(code + 1)]]
    if (is.null(name)) paste("Cluster", code + 1) else paste0(code + 1, " \u00b7 ", name)
  }, character(1))
}

kmStrata <- function(km) {
  if (is.null(km$strata)) {
    return(c(All = length(km$time)))
  }
  stats::setNames(as.integer(km$strata), stripStrata(names(km$strata)))
}

stripStrata <- function(strata) {
  sub("^cluster=", "", strata)
}

asDate <- function(x) {
  if (inherits(x, "Date")) {
    return(x)
  }
  if (inherits(x, "POSIXt")) {
    return(as.Date(x))
  }
  if (is.numeric(x)) {
    return(as.Date(x, origin = "1970-01-01"))
  }
  as.Date(x, optional = TRUE)
}

