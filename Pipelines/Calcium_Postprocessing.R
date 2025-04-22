# Calcium Imaging Full Analysis Pipeline (R + Python)

# Load Python environment and required R libraries
reticulate::use_condaenv("CaImg")
library(sf)
library(ggplot2)
library(jsonlite)
library(tidyverse)
library(reticulate)

# Source Python and R processing functions
reticulate::source_python("Calcium_Imaging.py")
source("Calcium_Postprocessing.R")

# Set working directories and load metadata
root <- "./your/path/to/data"
meta <- read.csv("./your/path/meta.csv")
samples <- paste0("MAX_", str_remove(meta$video_path, ".avi"))

# Loop through all samples
for (i in seq_along(samples)) {
  message(paste0("Run sample: ", samples[i]))
  
  # Load RDS structure and intensity data
  data <- readRDS(file.path(root, paste0(samples[i], ".RDS")))
  pathTraces <- file.path(root, paste0(samples[i], ".csv"))
  
  # -------------------- Python Processing --------------------
  reticulate::py_run_string("df = pd.read_csv(r.pathTraces)")
  reticulate::py_run_string("processed_df = ScaleTraces(df)")
  reticulate::py_run_string("processed_df = smooth_signal(processed_df, window_length=100, polyorder=4)")
  reticulate::py_run_string("processed_df = apply_als_baseline(processed_df, lam=1e7, p=0.05)")
  reticulate::py_run_string("peak_results = detect_peaks_and_features(processed_df, prominence=0.000004, distance=1)")
  reticulate::py_run_string("peak_df = calculate_peaks_for_all_cells(processed_df, prominence=0.000004, distance=1)")
  
  # -------------------- R Postprocessing --------------------
  Traces <- py$processed_df
  Spikes <- map(py$peak_results, ~.x$Peaks)
  peaks_df <- py$peak_df
  
  # Compute peak summary per cell
  nr_peak_cell <- map(Spikes, length) %>% unlist()
  
  get_mean <- function(data) mean(data, na.rm = TRUE)
  per_cell_df <- peaks_df %>%
    select(-c(1:3)) %>%
    group_by(Cell) %>%
    summarise_all(.funs = get_mean)
  
  per_cell_df <- left_join(
    data.frame(Cell = names(nr_peak_cell), nr_peaks = nr_peak_cell), 
    per_cell_df,
    by = "Cell"
  )
  
  # Create new NeuroPhysiology object
  object <- new(NeuroPhysiologyLab::.__C__NeuoPhyhysiology)
  object@Framerate <- 6
  object@Frames <- nrow(Traces)
  object@Spikes <- Spikes
  object@Traces <- Traces[, -1] %>% as.matrix()
  
  # Get coordinates for cell positions
  coords_all <- data$coords
  img_df <- coords_all %>%
    select(ID, X, Y) %>%
    group_by(ID) %>%
    summarise(x = mean(X), y = mean(Y), .groups = 'drop')
  
  # Detect pairwise cell connections
  object <- findConnections(
    object = object, 
    img_df = img_df, 
    time_shift = 10, 
    distance_max = 80
  )
  
  # Save result object if not already stored
  save_path <- file.path(root, paste0(samples[i], "NeuroPhysiology.RDS"))
  if (!file.exists(save_path)) {
    saveRDS(object, save_path)
  }
}
