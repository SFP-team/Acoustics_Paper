# Acoustic Classification of Blueberry Crunchiness

This repository contains the complete dataset and codebase for the manuscript on classifying blueberry crunchiness using acoustic emissions (AE) and machine learning. The project demonstrates the viability of acoustic signals—captured during tissue fracture—to accurately classify blueberries into crunchy and non-crunchy categories.

## Repository Structure

The repository is organized into two main directories: `data/` and `scripts/`.

### 1. `data/`
Contains the tabular datasets used for model training, cross-validation, and independent external validation.
- **`01_training_cv_data.csv`**: The primary dataset (3,210 samples across 109 genotypes) containing acoustic features and sensory crunchiness scores (SC). Used for internal model training and 10-fold cross-validation.
- **`02_external_validation_data.xlsx`**: An independent dataset comprising sensory-labeled samples from 11 distinct genotypes, used strictly for external validation of the models to assess their real-world generalizability.

### 2. `scripts/`
Contains all code required to replicate the feature extraction, model training, and performance evaluation described in the manuscript. The scripts are split between R (Classical Machine Learning) and Python (Deep Learning and specialized acoustic parameters).

#### Feature Extraction (Audio to Data)
- **`01a_feature_extraction_TDF_FDF_HDF.R`**: Processes raw `.wav` audio files. Applies a 2–50 kHz band-pass filter, root mean square scaling, and median absolute deviation thresholding. It extracts the Time Domain (TDF), Frequency Domain (FDF), and Hilbert Domain (HDF) features.
- **`01b_feature_extraction_MFCCs.ipynb`**: Extracts Mel-frequency cepstral coefficients (MFCCs) to capture the perceptual features of the acoustic emissions.
- **`01c_feature_extraction_MelSDF.ipynb`**: Computes Mel Spectrogram Density Features (SDF) to capture energy distributions across mel-frequency bands.
- **`01d_feature_extraction_Spectrogram.ipynb`**: Converts raw acoustic waveforms into two-dimensional spectrograms.
- **`01e_feature_extraction_Envelope.ipynb`**: Quantifies the envelope spread area of the acoustic signals, highlighting the sustained energy release typical of crunchy tissues.
- **`01f_feature_extraction_EventBurstSum.ipynb`**: Analyzes the cumulative energy of acoustic event bursts.
- **`01g_feature_extraction_LongestBurst.ipynb`**: Identifies and quantifies the longest continuous acoustic burst during the fracture event.
- **`01h_feature_extraction_utils.ipynb`**: A master utility notebook for batch processing and file-level feature aggregation.

#### Model Training & Validation (Data to Predictions)
- **`02a_model_training_ClassicalML.R`**: The core classical machine learning pipeline. It loads `01_training_cv_data.csv`, scales the features, and trains Random Forest (RF), Support Vector Machines (SVM), and XGBoost (XGB) using a 10-fold cross-validation repeated 3 times. It optimizes hyperparameters via grid search to match the parameters reported in the manuscript.
- **`02b_model_training_TabNet.ipynb`**: Implements the attentive interpretable tabular learning model (TabNet) in PyTorch. Includes hyperparameter tuning using `Optuna` and incorporates the specific learning rate scheduling and early stopping mechanisms detailed in the paper. It evaluates TabNet's performance, specifically demonstrating its tendency to favor the non-crunchy class compared to the tree-based ensemble models.
- **`03_external_validation_ClassicalML.R`**: Takes the models optimized in script `02a`, trains them on the entire training set, and predicts the crunchiness of the completely unseen external dataset (`02_external_validation_data.xlsx`). This script generates the final accuracy metrics highlighting RF's and XGB's superiority in generalization.

## Usage Instructions for Reviewers
1. **Prerequisites**: 
   - **R**: Requires `caret`, `randomForest`, `xgboost`, `e1071`, `tuneR`, `seewave`, and `dplyr`.
   - **Python**: Requires `pytorch_tabnet`, `optuna`, `librosa`, `scikit-learn`, and `pandas`.
2. **Execution Order**:
   - For feature extraction replication, start with `01a_feature_extraction_TDF_FDF_HDF.R` and the associated Python notebooks. *(Note: Requires raw `.wav` files).*
   - For modeling replication, execute `02a_model_training_ClassicalML.R` followed by `03_external_validation_ClassicalML.R` to reproduce the primary classical machine learning results from the paper.
   - Execute `02b_model_training_TabNet.ipynb` to replicate the deep learning comparative analysis.
