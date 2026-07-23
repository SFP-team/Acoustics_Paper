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
library(kernlab)  

set.seed(123)

# -----------------------------
# Paths & load data
# -----------------------------
path <- file.path("..", "data")

# Training data
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
# Predictors
# -----------------------------
num_cols  <- names(dat0)[sapply(dat0, is.numeric)]
num_cols  <- setdiff(num_cols, c("SC"))               
pred_cols <- setdiff(names(dat0), c("Genotype","r","SC"))

# Build a scaler on TRAINING predictors only
pp_train <- caret::preProcess(dat0[, pred_cols, drop = FALSE],
                              method = c("center", "scale"))

# Apply scaling to TRAINING data
ae_dat <- dat0
ae_dat[, pred_cols] <- predict(pp_train, dat0[, pred_cols, drop = FALSE])

# Modeling formula
formula <- as.formula(paste("r ~", paste(pred_cols, collapse = " + ")))

# ============================================================
# Train on ALL data
# ============================================================
train_ctrl_all <- trainControl(
  method = "none",
  classProbs = TRUE,
  savePredictions = "none"
)

# Tune grids
p <- length(pred_cols)
rf_grid  <- data.frame(mtry = max(1, floor(sqrt(p))))
sig      <- as.numeric(kernlab::sigest(as.matrix(ae_dat[, pred_cols, drop = FALSE])))[2]
svm_grid <- data.frame(sigma = sig, C = 1)
svml_grid <- data.frame(C = c(1))
xgb_grid <- data.frame(
  nrounds = 100,
  max_depth = 4,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

grids <- list(
  rf = rf_grid,
  svmRadial = svm_grid,
  svmLinear = svml_grid,
  xgbTree = xgb_grid
)

methods <- names(grids)
models  <- list()

set.seed(123)  
for (m in methods) {
  cat("Training (all data):", m, "\n")
  models[[m]] <- train(
    formula, data = ae_dat,
    method = m,
    trControl = train_ctrl_all,
    tuneGrid = grids[[m]],
    preProcess = NULL,        
    metric = "Accuracy"
  )
}

# ============================================================
# External VALIDATION
# ============================================================

# External Validation data
# Create r in df_summary
df_summary <- read_excel(file.path(path, "02_external_validation_data.xlsx"))
df_summary <- df_summary %>%
  mutate(
    r = case_when(
      SC %in% c(1, 2, 3) ~ "class1",
      SC %in% c(4, 5)    ~ "class2"
    )
  )
df_summary$r <- factor(df_summary$r, levels = c("class1","class2"))

# Ensure all predictor columns exist in df_summary
missing_pred <- setdiff(pred_cols, names(df_summary))
if (length(missing_pred)) {
  df_summary[missing_pred] <- NA_real_
}

df_valid <- df_summary[, c("Genotype", "r", pred_cols), drop = FALSE]

# ===== Scale df_valid using TRAINING scaler =====
df_valid_scaled <- df_valid
df_valid_scaled[, pred_cols] <- predict(pp_train, df_valid[, pred_cols, drop = FALSE])

# Predict validation set
pred_classes <- lapply(models, predict, newdata = df_valid_scaled, type = "raw")
pred_probs   <- lapply(models, predict, newdata = df_valid_scaled, type = "prob")

# Bind predictions into a single table
val_out <- df_valid %>%
  dplyr::select(Genotype, r)

for (m in methods) {
  val_out[[paste0(m, "_pred")]] <- pred_classes[[m]]
  if ("class2" %in% colnames(pred_probs[[m]])) {
    val_out[[paste0(m, "_prob_class2")]] <- pred_probs[[m]][,"class2"]
  } else {
    val_out[[paste0(m, "_prob_class2")]] <- pred_probs[[m]][,1]
  }
}

cm_rf <- confusionMatrix(as.factor(val_out$rf_pred), as.factor(val_out$r))
cat("RF Accuracy:", cm_rf$overall["Accuracy"], "\n")

cm_svm <- confusionMatrix(as.factor(val_out$svmRadial_pred), as.factor(val_out$r))
cat("SVM Accuracy:", cm_svm$overall["Accuracy"], "\n")

cm_xgb <- confusionMatrix(as.factor(val_out$xgbTree_pred), as.factor(val_out$r))
cat("XGB Accuracy:", cm_xgb$overall["Accuracy"], "\n")

# Save predictions
write.csv(val_out, file.path(path, "external_predictions.csv"), row.names = FALSE)

