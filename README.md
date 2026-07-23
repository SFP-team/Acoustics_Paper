# Acoustic Classification of Blueberry Crunchiness

This repository contains the complete dataset and codebase for the manuscript on classifying blueberry crunchiness using acoustic emissions (AE) and machine learning. The project demonstrates the viability of acoustic signals—captured during tissue fracture—to accurately classify blueberries into crunchy and non-crunchy categories.

## Repository Structure

The repository is organized into two main directories: `data/` and `scripts/`.

### 1. `data/`
Contains the tabular datasets used for model training, cross-validation, and independent external validation.
- **`01_training_cv_data.csv`**: The dataset was collected from 3,210 samples across 109 genotypes.
- **`02_external_validation_data.xlsx`**: An independent dataset comprising sensory-labeled samples from 150 samples across 11 genotypes.
- **`Note`**: Audio files can be accessed through this link - https://tinyurl.com/37879pyb.

### 2. `scripts/`
Contains all code required to replicate the feature extraction, model training, and performance evaluation described in the manuscript. The scripts are split between R (Classical Machine Learning) and Python (Deep Learning and acoustic parameters).

#### Feature Extraction (Audio to Data)
- **`01a_feature_extraction_TDF_FDF_HDF.R`**: Processes raw .wav audio files and extracts the time domain, frequency domain, and Hilbert domain features.
- **`01b_feature_extraction_MFCCs.ipynb`**: Extracts Mel-frequency cepstral coefficients.
- **`01c_feature_extraction_MelSDF.ipynb`**: Captures Mel spectrogram density features.
- **`01d_feature_extraction_Spectrogram.ipynb`**: Converts raw acoustic waveforms into two-dimensional spectrograms.
- **`01e_feature_extraction_Envelope.ipynb`**: Extracts domain-derived features that quantify the envelope spread area of the acoustic signals, highlighting the sustained energy release typical of crunchy tissues.
- **`01f_feature_extraction_EventBurstSum.ipynb`**: Extracts domain-derived features that analyze the cumulative energy of acoustic event bursts.
- **`01g_feature_extraction_LongestBurst.ipynb`**: Extracts domain-derived features that identify and quantify the longest continuous acoustic burst during the fracture event.
- **`01h_feature_extraction_utils.ipynb`**: A master utility notebook for batch processing and file-level feature aggregation.

#### Model Training & Validation (Data to Predictions)
- **`02a_model_training_ClassicalML.R`**: Runs the machine learning pipeline. It loads, scales the features, and trains Random Forest (RF), Support Vector Machines (SVM), and XGBoost (XGB) using a 10-fold cross-validation.
- **`02b_model_training_TabNet.ipynb`**: It implements the TabNet learning model pipeline.
- **`03_external_validation_ClassicalML.R`**: Takes the models optimized in script `02a`, trains them on the entire training set, and predicts the crunchiness of the completely unseen external dataset (`02_external_validation_data.xlsx`).

## Usage Instructions for Reviewers
1. **Prerequisites**: 
   - **R**: Requires `caret`, `randomForest`, `xgboost`, `e1071`, `tuneR`, `seewave`, and `dplyr`.
   - **Python**: Requires `pytorch_tabnet`, `optuna`, `librosa`, `scikit-learn`, and `pandas`.
2. **Execution Order**:
   - For feature extraction replication, start with `01a_feature_extraction_TDF_FDF_HDF.R` and the associated Python notebooks. *(Note: Requires raw `.wav` files).*
   - For modeling replication, execute `02a_model_training_ClassicalML.R` followed by `03_external_validation_ClassicalML.R` to reproduce the primary classical machine learning results from the paper.
   - Execute `02b_model_training_TabNet.ipynb` to replicate the deep learning comparative analysis.
