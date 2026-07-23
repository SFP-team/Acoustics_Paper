library(tuneR)
library(seewave)
library(dplyr)
library(signal)

# Standard
extract_ae_parameters <- function(file_path, threshold_ratio = 0.05,
                                  low_cut = 2000, high_cut = 50000) {
  # Read audio
  wav <- readWave(file_path)
  signal <- wav@left
  fs <- wav@samp.rate
  dt <- 1 / fs
  time_us <- seq(0, length(signal) - 1) * dt * 1e6
  
  # Band-pass filtering (2–50 kHz)
  filtered <- seewave::fir(signal, f = fs, from = low_cut, to = high_cut, bandpass = TRUE, output = "samples")
  
  # Denoising
  threshold <- threshold_ratio * max(abs(filtered), na.rm = TRUE)
  cross_idx <- which(abs(filtered) >= threshold)
  
  if (length(cross_idx) < 2) {
    return(data.frame(
      file = file_path, counts = NA, energy = NA, duration_us = NA,
      peak_db = NA, rise_time_us = NA, psd_db = NA, peak_psd = NA, peak_loc = NA,
      env_entropy = NA, mod_index = NA, ENH = NA, ENL = NA, AVE = NA,
      inst_freq_mean = NA, inst_freq_sd = NA
    ))
  }
  
  # Signal of interest
  first <- min(cross_idx)
  last <- max(cross_idx)
  sig_segment <- filtered[first:last]
  # --- Time Domain Features ---
  counts <- length(cross_idx)
  energy <- sum(sig_segment^2) * dt
  peak_amp <- max(abs(sig_segment))
  peak_db <- 20 * log10(peak_amp)
  duration_us <- (last - first) * dt * 1e6
  peak_idx <- which.max(abs(sig_segment)) + first - 1
  rise_time_us <- (peak_idx - first) * dt * 1e6
  
  # PSD
  window <- seewave::hanning.w(length(sig_segment))
  windowed_signal <- sig_segment * window
  n <- length(windowed_signal)
  signal_fft <- fft(windowed_signal)
  psd <- Mod(signal_fft)^2 / n
  psd <- psd[1:(floor(n/2) + 1)]
  freq <- seq(0, fs/2, length.out = length(psd))
  log_psd <- log10(psd + 1e-12)
  psd_db <- sum(log_psd, na.rm = TRUE)
  # --- Frequency Domain Features ---
  peak_idx <- which.max(psd)
  peak_psd <- log_psd[peak_idx]
  peak_loc <- freq[peak_idx]
  
  # --- Hilbert Transform ---
  analytic <- seewave::hilbert(sig_segment, f = fs)
  envelope <- Mod(analytic)
  phase <- Arg(analytic)
  unwrap_phase <- function(p) {
    dp <- diff(p)
    dp_mod <- (dp + pi) %% (2 * pi) - pi
    dp_mod[dp_mod == -pi & dp > 0] <- pi
    p_unwrapped <- c(p[1], p[1] + cumsum(dp_mod))
    return(p_unwrapped)
  }
  unwrap_phase <- function(p) {
    dp <- diff(p)
    dp_mod <- (dp + pi) %% (2 * pi) - pi
    dp_mod[dp_mod == -pi & dp > 0] <- pi
    p_unwrapped <- c(p[1], p[1] + cumsum(dp_mod))
    return(p_unwrapped)
  }
  unwrapped <- unwrap_phase(phase)
  #unwrapped <- signal::unwrap(phase) 
  inst_freq <- c(0, diff(unwrapped)) * fs / (2 * pi)
  
  # --- Hilbert Domain Features ---
  AVE <- mean(envelope, na.rm = TRUE)
  mod_index <- sd(envelope, na.rm = TRUE) / AVE
  prob_env <- envelope / sum(envelope + 1e-12)
  env_entropy <- -sum(prob_env * log(prob_env + 1e-12), na.rm = TRUE)
  inst_freq_mean <- mean(inst_freq, na.rm = TRUE)
  inst_freq_sd <- sd(inst_freq, na.rm = TRUE)
  
  # --- Energy in frequency bands ---
  enh_idx <- which(freq >= 10000 & freq <= 15000)
  enl_idx <- which(freq >= 5000 & freq < 10000)
  ENH <- sum(psd[enh_idx], na.rm = TRUE)
  ENL <- sum(psd[enl_idx], na.rm = TRUE)
  
  # Return all
  data.frame(
    file = file_path,
    # --- Time Domain Features ---
    counts = counts,
    energy = energy,
    duration_us = duration_us,
    peak_db = peak_db,
    rise_time_us = rise_time_us,
    # --- Frequency Domain Features ---
    psd_db = psd_db,
    peak_psd = peak_psd,
    peak_loc = peak_loc,
    # --- Hilbert Domain Features ---
    env_entropy = env_entropy,
    mod_index = mod_index,
    ENH = ENH,
    ENL = ENL,
    AVE = AVE,
    inst_freq_mean = inst_freq_mean,
    inst_freq_sd = inst_freq_sd
  )
}

# Normalized - Works
extract_ae_parameters <- function(file_path, threshold_ratio = 0.05,
                                  low_cut = 2000, high_cut = 50000, eps = 1e-12) {
  # Read audio
  wav <- tuneR::readWave(file_path)
  x <- wav@left
  fs <- wav@samp.rate
  dt <- 1 / fs
  
  # Band-pass filtering (2–50 kHz)
  x_f <- seewave::fir(x, f = fs, from = low_cut, to = high_cut,
                      bandpass = TRUE, output = "samples")
  
  # -------- LEVEL NORMALIZATION (per-file) --------
  # 1) RMS normalize
  rms <- sqrt(mean(x_f^2, na.rm = TRUE))
  if (!is.finite(rms) || rms < eps) rms <- eps
  xn <- x_f / rms
  
  # 2) Robust re-scaling by 95th percentile of |xn| (limits influence of spikes)
  q95 <- stats::quantile(abs(xn), 0.95, na.rm = TRUE, names = FALSE)
  if (!is.finite(q95) || q95 < eps) q95 <- 1
  xn <- xn / q95
  
  # -------- Robust thresholding (relative) --------
  # Use MAD as a noise estimate; threshold is relative to normalized scale
  mad_x <- stats::mad(xn, constant = 1, na.rm = TRUE) # not multiplied by 1.4826
  thr <- max(threshold_ratio, 3 * mad_x)              # safety floor
  cross_idx <- which(abs(xn) >= thr)
  
  if (length(cross_idx) < 2) {
    return(data.frame(
      file = file_path, counts = NA, energy = NA, duration_us = NA,
      peak_db = NA, peak_db_rel = NA, rise_time_us = NA, psd_db = NA,
      peak_psd = NA, peak_loc = NA, env_entropy = NA, mod_index = NA,
      ENH = NA, ENL = NA, ENH_frac = NA, ENL_frac = NA, AVE = NA,
      inst_freq_mean = NA, inst_freq_sd = NA
    ))
  }
  
  first <- min(cross_idx); last <- max(cross_idx)
  seg <- xn[first:last]
  
  counts <- length(cross_idx)
  energy <- sum(seg^2) * dt                            # now scale-invariant
  peak_amp <- max(abs(seg))
  peak_db_rel <- 20 * log10((peak_amp + eps))          # dB relative to normalized RMS
  peak_db <- peak_db_rel                                # keep name for backward compat
  duration_us <- (last - first) * dt * 1e6
  peak_idx <- which.max(abs(seg)) + first - 1
  rise_time_us <- (peak_idx - first) * dt * 1e6
  
  # PSD (single FFT; you can switch to Welch if desired)
  w <- seewave::hanning.w(length(seg))
  xw <- seg * w
  n <- length(xw)
  X <- fft(xw)
  psd <- Mod(X)^2 / n
  psd <- psd[1:(floor(n/2) + 1)]
  freq <- seq(0, fs/2, length.out = length(psd))
  
  # Relative / log PSD
  psd_sum <- sum(psd, na.rm = TRUE) + eps
  psd_norm <- psd / psd_sum
  log_psd <- log10(psd + eps)
  psd_db <- sum(log_psd, na.rm = TRUE)                 # descriptive only
  pk <- which.max(psd_norm)
  peak_psd <- log_psd[pk]
  peak_loc <- freq[pk]
  
  # Hilbert domain on normalized segment
  analytic <- seewave::hilbert(seg, f = fs)
  env <- Mod(analytic)
  ph <- Arg(analytic)
  # unwrap
  dp <- diff(ph); dp_mod <- (dp + pi) %% (2*pi) - pi; dp_mod[dp_mod == -pi & dp > 0] <- pi
  ph_unw <- c(ph[1], ph[1] + cumsum(dp_mod))
  inst_freq <- c(0, diff(ph_unw)) * fs / (2*pi)
  
  AVE <- mean(env, na.rm = TRUE)
  mod_index <- stats::sd(env, na.rm = TRUE) / (AVE + eps)     # scale-invariant
  p_env <- env / (sum(env, na.rm = TRUE) + eps)               # prob. envelope
  env_entropy <- -sum(p_env * log(p_env + eps), na.rm = TRUE)
  inst_freq_mean <- mean(inst_freq, na.rm = TRUE)
  inst_freq_sd <- stats::sd(inst_freq, na.rm = TRUE)
  
  # Energy in bands (absolute and fraction of total)
  enh_idx <- which(freq >= 10000 & freq <= 15000)
  enl_idx <- which(freq >= 5000  & freq  < 10000)
  ENH <- sum(psd[enh_idx], na.rm = TRUE)
  ENL <- sum(psd[enl_idx], na.rm = TRUE)
  ENH_frac <- sum(psd_norm[enh_idx], na.rm = TRUE)
  ENL_frac <- sum(psd_norm[enl_idx], na.rm = TRUE)
  
  data.frame(
    file = file_path,
    counts = counts,
    energy = energy,
    duration_us = duration_us,
    peak_db = peak_db,                 # relative to normalized RMS
    peak_db_rel = peak_db_rel,         # explicit name if you prefer
    rise_time_us = rise_time_us,
    psd_db = psd_db,
    peak_psd = peak_psd,
    peak_loc = peak_loc,
    env_entropy = env_entropy,
    mod_index = mod_index,
    ENH = ENH,
    ENL = ENL,
    ENH_frac = ENH_frac,
    ENL_frac = ENL_frac,
    AVE = AVE,
    inst_freq_mean = inst_freq_mean,
    inst_freq_sd = inst_freq_sd
  )
}


# --- Function to Process All .wav Files in Directory ---
process_all_wav_files <- function(root_dir = ".") {
  wav_files <- list.files(path = root_dir, pattern = "\\.wav$", 
                          full.names = TRUE, recursive = TRUE)
  
  message("Processing ", length(wav_files), " .wav files...\n")
  
  results <- lapply(wav_files, extract_ae_parameters)
  do.call(rbind, results)
}

# --- Set working directory and run analysis ---
# Set path to the directory containing raw .wav audio files
path = "../data/raw_audio"
ae_results <- process_all_wav_files(path)

# --- Clean file names ---
paths <- ae_results$file
filenames <- basename(paths)
names_no_ext <- sub("\\.wav$", "", filenames)
ae_results$file <- names_no_ext

# --- View Results ---
head(ae_results)

library(openxlsx)
# Save parameters to the data directory
output_file <- "../data/ae_parameters.xlsx"
write.xlsx(ae_results, file = output_file, rowNames = FALSE)



