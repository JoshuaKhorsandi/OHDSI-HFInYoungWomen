saveKaplanMeierPanels <- function(models, path, maxYears, colours) {
  panels <- vapply(models, function(model) model$outcome$label, character(1))
  curves <- do.call(rbind, lapply(models, kmPanelCurves))
  curves$panel <- factor(curves$panel, levels = panels)
  medians <- do.call(rbind, lapply(models, kmMedians, maxYears = maxYears))
  if (!is.null(medians)) {
    medians$panel <- factor(medians$panel, levels = panels)
  }
  colours <- colours[names(colours) %in% curves$cluster]

  plot <- ggplot2::ggplot(curves, ggplot2::aes(x = .data$time)) +
    ggplot2::geom_hline(yintercept = 0.5, colour = REF_GREY, linewidth = 0.35, linetype = "43") +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$lower, ymax = .data$upper, fill = .data$cluster),
                         alpha = CI_ALPHA, na.rm = TRUE) +
    ggplot2::geom_line(ggplot2::aes(y = .data$surv, colour = .data$cluster), linewidth = 0.75) +
    ggplot2::scale_colour_manual(values = colours, guide = "none") +
    ggplot2::scale_fill_manual(values = colours, breaks = names(colours),
                               labels = referenceLabels(names(colours), modelReference(models[[1]]), "(Ref)")) +
    ggplot2::guides(fill = ggplot2::guide_legend(nrow = if (length(models) == 1) 2 else 1,
                                                 override.aes = list(alpha = 1))) +
    ggplot2::facet_wrap(~panel, nrow = 1) +
    ggplot2::scale_x_continuous(breaks = seq(0, maxYears)) +
    ggplot2::scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
    ggplot2::coord_cartesian(xlim = c(0, maxYears), ylim = c(0, 1.02), expand = FALSE) +
    ggplot2::labs(x = "Years from heart failure index date", y = "Risk-free probability", fill = NULL) +
    themeCluster() +
    ggplot2::theme(legend.position = "top", panel.spacing = ggplot2::unit(1.2, "lines"))
  if (!is.null(medians)) {
    plot <- plot + ggplot2::geom_point(data = medians, ggplot2::aes(x = .data$median, y = 0.5, colour = .data$cluster),
                                       size = 1.9)
  }

  ggplot2::ggsave(path, plot, width = max(5.5, 1.2 + 3.3 * length(models)),
                  height = if (length(models) == 1) 4.4 else 3.8, dpi = 300, bg = BG)
}

saveForestPlot <- function(cox, path, colours, xlim = c(0, 4), log = FALSE, showHrText = TRUE) {
  rows <- forestRows(cox, names(colours))
  if (is.null(rows)) {
    return(invisible(NULL))
  }
  plot <- forestPlot(rows, colours, xlim, log, showHrText)
  height <- 1.3 + 0.27 * nrow(rows) + 0.25 * length(unique(rows$cluster))
  ggplot2::ggsave(path, plot, width = 9.5, height = height, dpi = 300, bg = BG)
}

themeCluster <- function(baseSize = 11) {
  ggplot2::theme_minimal(base_size = baseSize) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = BG, colour = NA),
      panel.background = ggplot2::element_rect(fill = BG, colour = NA),
      text = ggplot2::element_text(colour = INK),
      axis.text = ggplot2::element_text(colour = INK),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_line(colour = GRID, linewidth = 0.3),
      strip.text = ggplot2::element_text(colour = INK, face = "bold", hjust = 0),
      legend.key.size = ggplot2::unit(0.9, "lines")
    )
}

kmPanelCurves <- function(model) {
  km <- model$km
  sizes <- kmStrata(km)
  base <- data.frame(cluster = rep(names(sizes), sizes), time = km$time, surv = km$surv,
                     lower = km$lower, upper = km$upper)
  start <- data.frame(cluster = names(sizes), time = 0, surv = 1, lower = 1, upper = 1)
  curves <- rbind(start, base)
  curves <- do.call(rbind, lapply(split(curves, factor(curves$cluster, levels = names(sizes))), stairs))
  curves$panel <- model$outcome$label
  rownames(curves) <- NULL
  curves
}

kmMedians <- function(model, maxYears) {
  km <- model$km
  sizes <- kmStrata(km)
  cluster <- rep(names(sizes), sizes)
  reached <- km$surv <= 0.5 & km$time <= maxYears
  if (!any(reached)) {
    return(NULL)
  }
  first <- !duplicated(cluster[reached])
  data.frame(panel = model$outcome$label, cluster = cluster[reached][first], median = km$time[reached][first])
}

modelReference <- function(model) {
  model$cox$cluster[model$cox$reference & !is.na(model$cox$hr)][1]
}

forestRows <- function(cox, clusters) {
  if (all(is.na(cox$hr[!cox$reference]))) {
    return(NULL)
  }
  encounters <- unique(cox$label)
  blocks <- clusters[clusters %in% cox$cluster]
  rowGap <- 0.55
  blockGap <- 0.9

  rows <- list()
  y <- 0
  for (b in seq_along(blocks)) {
    ys <- y - (seq_along(encounters) - 1) * rowGap
    for (k in seq_along(encounters)) {
      estimate <- cox[cox$cluster == blocks[b] & cox$label == encounters[k], , drop = FALSE]
      rows[[length(rows) + 1]] <- data.frame(
        block = b, cluster = blocks[b], label = encounters[k], y = ys[k],
        top = ys[1] + blockGap / 2, bottom = ys[length(ys)] - blockGap / 2, middle = mean(ys),
        reference = nrow(estimate) == 1 && isTRUE(estimate$reference),
        hr = if (nrow(estimate) == 1) estimate$hr else NA_real_,
        lower = if (nrow(estimate) == 1) estimate$lower else NA_real_,
        upper = if (nrow(estimate) == 1) estimate$upper else NA_real_,
        p = if (nrow(estimate) == 1) estimate$p else NA_real_
      )
    }
    y <- ys[length(ys)] - blockGap
  }
  rows <- do.call(rbind, rows)
  rows$label <- factor(rows$label, levels = encounters)
  rows
}

forestPlot <- function(rows, colours, xlim, log, showHrText) {
  transform <- if (log) log10 else identity
  range <- transform(xlim)
  right <- if (showHrText) 1.42 else 0.05
  ticks <- if (log) c(0.25, 0.5, 1, 2, 4, 8) else pretty(xlim)
  ticks <- ticks[ticks >= xlim[1] & ticks <= xlim[2]]
  encounters <- levels(rows$label)
  encounterColours <- stats::setNames(rep_len(ENCOUNTER_COLOURS, length(encounters)), encounters)
  encounterShapes <- stats::setNames(rep_len(ENCOUNTER_SHAPES, length(encounters)), encounters)

  estimates <- rows[!rows$reference & !is.na(rows$hr), , drop = FALSE]
  estimates$x <- transform(pmin(pmax(estimates$hr, xlim[1]), xlim[2]))
  estimates$xmin <- transform(pmin(pmax(estimates$lower, xlim[1]), xlim[2]))
  estimates$xmax <- transform(pmin(pmax(estimates$upper, xlim[1]), xlim[2]))
  references <- rows[rows$reference, , drop = FALSE]
  blocks <- rows[!duplicated(rows$block), , drop = FALSE]
  bands <- blocks[blocks$block %% 2 == 0, , drop = FALSE]
  labels <- data.frame(
    y = blocks$middle,
    cluster = blocks$cluster,
    text = referenceLabels(blocks$cluster, unique(references$cluster), "(ref)")
  )
  rows$pText <- ifelse(rows$reference, "ref", formatP(rows$p))
  header <- max(rows$top) + 0.15
  bottom <- min(rows$bottom)
  center <- (FOREST_LEFT + 0.5) / (FOREST_LEFT + 1 + right)

  plot <- ggplot2::ggplot() +
    ggplot2::geom_rect(data = bands, ggplot2::aes(ymin = .data$bottom, ymax = .data$top),
                       xmin = axesX(-FOREST_LEFT, range), xmax = axesX(1 + right, range), fill = BAND) +
    ggplot2::annotate("segment", x = transform(ticks), xend = transform(ticks), y = bottom, yend = max(rows$top),
                      colour = GRID, linewidth = 0.3) +
    ggplot2::annotate("segment", x = transform(1), xend = transform(1), y = bottom, yend = max(rows$top),
                      colour = REF_GREY, linewidth = 0.35, linetype = "43") +
    ggplot2::geom_segment(data = estimates, ggplot2::aes(x = .data$xmin, xend = .data$xmax, y = .data$y,
                                                         yend = .data$y, colour = .data$label),
                          linewidth = 0.6, lineend = "butt") +
    ggplot2::geom_segment(data = estimates, ggplot2::aes(x = .data$xmin, xend = .data$xmin, y = .data$y - 0.12,
                                                         yend = .data$y + 0.12, colour = .data$label),
                          linewidth = 0.6, show.legend = FALSE) +
    ggplot2::geom_segment(data = estimates, ggplot2::aes(x = .data$xmax, xend = .data$xmax, y = .data$y - 0.12,
                                                         yend = .data$y + 0.12, colour = .data$label),
                          linewidth = 0.6, show.legend = FALSE) +
    ggplot2::geom_point(data = estimates, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$label,
                                                       shape = .data$label),
                        colour = "white", size = 2.6, stroke = 0.5) +
    ggplot2::geom_point(data = references, ggplot2::aes(x = transform(1), y = .data$y, shape = .data$label),
                        fill = "white", colour = REF_GREY, size = 2.4, stroke = 0.6, show.legend = FALSE) +
    ggplot2::geom_point(data = labels, ggplot2::aes(y = .data$y), x = axesX(0.03 - FOREST_LEFT, range), shape = 15,
                        size = 3.3, colour = colours[labels$cluster]) +
    ggplot2::geom_text(data = labels, ggplot2::aes(y = .data$y, label = .data$text), x = axesX(0.06 - FOREST_LEFT, range),
                       hjust = 0, size = 9.5 / ggplot2::.pt, colour = INK) +
    ggplot2::geom_text(data = rows, ggplot2::aes(y = .data$y, label = .data$pText), x = axesX(-0.04, range),
                       hjust = 1, size = 8 / ggplot2::.pt,
                       colour = ifelse(rows$reference, REF_GREY, INK)) +
    ggplot2::annotate("text", x = axesX(c(0.03 - FOREST_LEFT, -0.04), range), y = header, label = c("Cluster", "p-value"),
                      hjust = c(0, 1), vjust = 0, fontface = "bold", size = 9.5 / ggplot2::.pt, colour = INK) +
    ggplot2::scale_colour_manual(values = encounterColours, breaks = encounters, name = NULL) +
    ggplot2::scale_fill_manual(values = encounterColours, breaks = encounters, name = NULL) +
    ggplot2::scale_shape_manual(values = encounterShapes, breaks = encounters, name = NULL) +
    ggplot2::scale_x_continuous(breaks = transform(ticks), labels = formatC(ticks, format = "g")) +
    ggplot2::coord_cartesian(xlim = c(axesX(-FOREST_LEFT, range), axesX(1 + right, range)),
                             ylim = c(bottom, header + 0.45), expand = FALSE, clip = "off") +
    ggplot2::labs(x = "Hazard ratio (95% CI)", y = NULL) +
    themeCluster() +
    ggplot2::theme(panel.grid.major = ggplot2::element_blank(), axis.text.y = ggplot2::element_blank(),
                   axis.title.x = ggplot2::element_text(hjust = center),
                   legend.position = "top", legend.key.width = ggplot2::unit(2.2, "lines"))

  if (showHrText) {
    estimates$hrText <- sprintf("%.2f (%.2f\u2013%.2f)", estimates$hr, estimates$lower, estimates$upper)
    plot <- plot +
      ggplot2::geom_text(data = estimates, ggplot2::aes(y = .data$y, label = .data$hrText),
                         x = axesX(1.04, range), hjust = 0, size = 8 / ggplot2::.pt, colour = INK) +
      ggplot2::annotate("text", x = axesX(1.04, range), y = header, label = "HR (95% CI)", hjust = 0, vjust = 0,
                        fontface = "bold", size = 9.5 / ggplot2::.pt, colour = INK)
  }
  plot
}

stairs <- function(curve) {
  curve <- curve[order(curve$time), , drop = FALSE]
  index <- rep(seq_len(nrow(curve)), each = 2)
  steps <- curve[index[-length(index)], , drop = FALSE]
  steps$time <- curve$time[index[-1]]
  steps
}

referenceLabels <- function(clusters, reference, suffix) {
  ifelse(!is.na(clusters) & clusters %in% reference, paste(clusters, suffix), clusters)
}

axesX <- function(fraction, range) {
  range[1] + fraction * diff(range)
}

formatP <- function(p) {
  ifelse(is.na(p), "NA", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
}

CLUSTER_COLOURS <- c("#568F8B", "#CD7E59", "#D15252", "#DDB247",
                     "#6F7FB8", "#8E6AA8", "#7A9A4F", "#B06F8E")
BG <- "#FCFCFB"
INK <- "#1A1A1A"
REF_GREY <- "#8C8C8C"
GRID <- "#E6E6E6"
BAND <- "#F2F1EE"
CI_ALPHA <- 0.18
FOREST_LEFT <- 1.05
ENCOUNTER_COLOURS <- c("#0072B2", "#CC79A7", "#000000")
ENCOUNTER_SHAPES <- c(21, 22, 23)
