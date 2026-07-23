# ============================================================
# Acoustic classification (2 classes)
# ============================================================

library(readr)
library(openxlsx)
library(readxl)
library(dplyr)
library(tibble)
library(caret)
library(e1071)
library(randomForest)
library(xgboost)
library(reshape2)
library(data.table)
library(purrr)

set.seed(123)

# -----------------------------
# Paths & load data
# -----------------------------
path <- file.path("..", "data")
df_raw <- read.csv(file.path(path, "01_training_cv_data.csv"))

# -----------------------------
# Build target 
# -----------------------------
dat0 <- df_raw %>%
  mutate(
    r = case_when(
      SC %in% c(1, 2, 3) ~ 1,
      SC %in% c(4, 5)    ~ 2
    )
  ) %>%
  dplyr::select(Genotype, everything()) %>% 
  na.omit()

dat0$r <- factor(dat0$r, levels = c(1, 2), labels = c("class1", "class2"))

# -----------------------------
# Scale numeric predictors
# -----------------------------
num_cols  <- names(dat0)[sapply(dat0, is.numeric)]
num_cols  <- setdiff(num_cols, c("SC"))      # don't scale "SC"
pred_cols <- setdiff(names(dat0), c("Genotype","r","SC"))  # predictors used in model

ae_dat <- dat0
ae_dat[, intersect(pred_cols, num_cols)] <- scale(ae_dat[, intersect(pred_cols, num_cols)])

# Modeling formula
formula <- as.formula(paste("r ~", paste(pred_cols, collapse = " + ")))
optimal_features <- pred_cols

# -----------------------------
# CV control
# -----------------------------
train_ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final"
)

# -----------------------------
# Hyperparameter grids (as before)
# -----------------------------
grids <- list(
  rf = expand.grid(mtry = c(2:6)),
  svmRadial = expand.grid(
    sigma = c(0.01, 0.05, 0.1, 0.5),
    C     = c(0.01, 0.05, 0.1, 1, 5, 10)
  ),
  xgbTree = expand.grid(
    nrounds = c(50, 100),
    max_depth = c(2, 4, 6),
    eta = c(0.01, 0.1),
    gamma = 0,
    colsample_bytree = c(0.6, 0.8),
    min_child_weight = 1,
    subsample = 0.8
  )
)

# -----------------------------
# Train models
# -----------------------------
methods <- names(grids)
models  <- list()

for (m in methods) {
  cat("Training:", m, "\n")
  models[[m]] <- train(
    formula, data = ae_dat,
    method = m,
    trControl = train_ctrl,
    tuneGrid = grids[[m]],
    preProcess = c("center", "scale"),
    metric = "Accuracy"
  )
}

# -----------------------------
# Compare models
# -----------------------------
resamps <- resamples(models)

mean_metrics <- summary(resamps)$statistics
summary_table <- do.call(rbind, mean_metrics)
summary_df <- as.data.frame(summary_table)
summary_df$model <- sub("\\..*", "", rownames(summary_df))
n_models <- length(unique(summary_df$model))
metric_names <- rep(names(mean_metrics), each = n_models)
summary_df$metrics <- metric_names

summary_df1 <- dcast(as.data.table(summary_df[, c("model", "metrics", "Mean")]), model ~ metrics)
summary_df1 <- data.frame(summary_df1)
print(summary_df1)

# Save results
write.csv(summary_df1, file.path(path, "cv_metrics.csv"), row.names = FALSE)
