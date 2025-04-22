#' findConnections
#'
#' Identifies and quantifies functional connections between cells based on temporal proximity and spatial distance.
#'
#' @param object An object containing spike events (`@Spikes`) and image data (`@Image`).
#' @param img_df Optional data.frame with cell positions (columns: Cells, x, y). If NULL, computed from `object@Image`.
#' @param time_shift Maximum time difference to consider for functional connection (default = 10).
#' @param distance_max Maximum spatial distance allowed between connected cells (default = 0.1).
#' @param pal Color palette for plotting (default = viridis palette).
#' @param parameter_to_col A string specifying which connection feature to use for coloring (default = "dist").
#' @param cores Number of cores to use for parallel computation (default = 8).
#' @param Events_select Optional vector to manually select specific events.
#' @return Modified input object with a new slot `@Connections` containing:
#'   - Events: All detected spike events
#'   - Connections_unselected: Raw connection candidates
#'   - connections: Filtered connections
#'   - poly_connected: Connection edges with spatial coordinates for plotting
findConnections = function(object,
                           img_df = NULL,
                           time_shift = 10,
                           distance_max = 0.1, 
                           pal = viridis::viridis(50), 
                           parameter_to_col = c("dist", "Time")[1], 
                           cores = 8,
                           Events_select = NULL) {
  
  Spikes <- object@Spikes
  image <- object@Image
  
  message("Step 1/4: Create Events")
  Events <- do.call(rbind, pbmcapply::pbmclapply(1:length(Spikes), mc.cores = cores, function(i) {
    Time <- as.numeric(Spikes[[i]])
    if (length(Time) >= 1) {
      data.frame(Time = Time, Cells = paste0("Cell_", i))
    } else {
      data.frame(Time = NA, Cells = paste0("Cell_", i))
    }
  })) %>% na.omit() %>% dplyr::arrange(Time)
  
  if (!is.null(Events_select)) {
    Events <- Events[Events_select, ]
  }
  
  message(paste0("Number of Events extracted: ", nrow(Events)))
  message("Step 2/4: Quantify Connections and measure Distance")
  
  if (is.null(img_df)) {
    img_df <- image@.Data %>% 
      reshape2::melt() %>% 
      filter(value != 0) %>% 
      group_by(value) %>% 
      summarise(x = mean(Var1), y = mean(Var2)) %>% 
      rename(Cells = value)
  }
  
  Events <- Events %>% left_join(img_df, by = "Cells")
  
  Connections_unselected <- pbmcapply::pbmclapply(1:nrow(Events), mc.cores = cores, function(i) {
    collect <- Events[i, "Time"] + time_shift
    Events_i <- Events[i:max(which(Events$Time < collect)), ]
    pos_mat <- data.frame(p1x = Events_i[1, "x"], p1y = Events_i[1, "y"],
                          p2x = Events_i[, "x"], p2y = Events_i[, "y"])
    dist <- NeuroPhysiologyLab::getDistance(pos_mat$p1x, pos_mat$p1y, pos_mat$p2x, pos_mat$p2y)
    Events_i$distance <- dist
    Events_i <- Events_i[Events_i$distance < distance_max, ]
    if (nrow(Events_i) == 0 || length(unique(Events_i$Cells)) == 1) return(NA)
    return(Events_i)
  })
  
  message("Step 3/4: Optimize and Filter Connections")
  connections <- do.call(rbind, pbmcapply::pbmclapply(1:length(Connections_unselected), mc.cores = cores, function(i) {
    con <- Connections_unselected[[i]]
    if (is.data.frame(con) && length(unique(con$Cells)) > 1) {
      start <- con$Cells[1]
      lapply(2:nrow(con), function(j) {
        data.frame(From = start, To = con$Cells[j], col = "black",
                   dist = con$distance[j], Time = con$Time[j], Nr_of_con = i)
      }) %>% do.call(rbind, .)
    } else {
      data.frame(From = NA, To = NA, col = NA, dist = NA, Time = NA, Nr_of_con = NA)
    }
  })) %>% na.omit()
  
  message(paste0("Number of connections: ", nrow(connections)))
  
  message("Step 4/4: Create Plot File")
  cell_pos <- Events %>% distinct(Cells, x, y) %>% mutate(From = Cells, To = Cells, x1 = x, y1 = y, x2 = x, y2 = y)
  poly_connected <- connections %>% 
    left_join(cell_pos %>% select(From, x1, y1), by = "From") %>%
    left_join(cell_pos %>% select(To, x2, y2), by = "To")
  
  out <- list(Events, Connections_unselected, connections, poly_connected)
  names(out) <- c("Events", "Connections_unselected", "connections", "poly_connected")
  object@Connections <- out
  message("Finish Pipeline")
  return(object)
}

#' runSFT
#'
#' Computes scale-free topology diagnostics based on event frequencies.
#'
#' @param object Input object containing `@Connections$Events`.
#' @param breaks Number of histogram bins (default = 1000).
#' @return Modified object with `@Connections$SFT` slot containing degree distribution and R² fit.
runSFT <- function(object, breaks = 1000) {
  EV <- object@Connections$Events
  EV$Cells <- as.numeric(gsub("Cell_", "", EV$Cells))
  frq <- data.frame(cell = 1:max(EV$Cells), Freq = tabulate(EV$Cells))
  frq$connectivity <- frq$Freq / max(frq$Freq)
  a <- hist(frq$connectivity, breaks = breaks, plot = FALSE)
  plot.df <- data.frame(x = log10(a$counts), y = log10(a$mids)) %>% filter(is.finite(x))
  model <- lm(y ~ x, data = plot.df)
  object@Connections$SFT <- list(frequence = frq, histo = a, R2 = summary(model)$adj.r.squared)
  print(paste0("R2 is: ", summary(model)$adj.r.squared))
  return(object)
}

#' plotSFT
#'
#' Visualizes scale-free topology fit with optional linear regression.
#'
#' @param object Input object with `@Connections$SFT`.
#' @param filter Numeric threshold for log10(frequency) filter (default = -0.5).
#' @param export_R Logical flag to return R² instead of plot (default = FALSE).
#' @return ggplot object or R² value depending on `export_R`.
plotSFT <- function(object, filter = c(-0.5), export_R = FALSE) {
  plot.df <- data.frame(x = log10(object@Connections$SFT$histo$counts),
                        y = log10(object@Connections$SFT$histo$mids)) %>%
    filter(is.finite(x))
  if (!is.na(filter)) {
    plot.df <- plot.df %>% filter(y > filter)
  }
  model <- lm(x ~ y, data = plot.df)
  print(paste0("adj.r.squared: ", summary(model)$adj.r.squared))
  p <- ggplot2::ggplot(plot.df, ggplot2::aes(x, y)) +
    ggplot2::geom_point() +
    ggplot2::theme_classic() +
    ggplot2::geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +
    ggplot2::xlab("Log10(Connectivity)") +
    ggplot2::ylab("Log10(Frequency)")
  if (export_R) {
    return(summary(model)$adj.r.squared)
  } else {
    return(p)
  }
}
