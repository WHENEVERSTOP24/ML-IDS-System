# Machine Learning Intrusion Detection System (ML-IDS)

This repository contains a modular, production-ready Machine Learning Intrusion Detection System (ML-IDS) built in Python using the **NSL-KDD dataset**. 

It implements both **Binary Classification** (predicting whether network traffic is `Normal` or an `Anomaly`) and **Multiclass Classification** (predicting the specific class of intrusion: `Normal`, `DoS`, `Probe`, `R2L`, `U2R`).

---

## Project Directory Structure

```text
ml_ids/
├── data/                  # Raw and preprocessed dataset files
│   ├── KDDTrain+.txt      # Raw training data
│   ├── KDDTest+.txt       # Raw testing data
│   ├── train_processed.csv # Scaled and encoded training features
│   └── test_processed.csv  # Scaled and encoded testing features
├── models/                # Serialized model & pipeline files (.joblib)
│   ├── preprocessor.joblib # Fitted ColumnTransformer (scaling & encoding)
│   ├── binary_dt.joblib    # Trained Binary Decision Tree
│   ├── binary_rf.joblib    # Trained Binary Random Forest (Classifier)
│   ├── multiclass_dt.joblib # Trained Multiclass Decision Tree
│   ├── multiclass_rf.joblib # Trained Multiclass Random Forest (Classifier)
│   ├── bin_dt_cm.png       # Confusion matrix plot for Binary Decision Tree
│   ├── bin_rf_cm.png       # Confusion matrix plot for Binary Random Forest
│   ├── multi_dt_cm.png     # Confusion matrix for Multiclass Decision Tree
│   └── multi_rf_cm.png     # Confusion matrix for Multiclass Random Forest
├── notebooks/             # Jupyter Notebooks for EDA and demonstration
│   ├── 01_eda.ipynb       # Exploratory Data Analysis
│   └── 02_model_training.ipynb # Interactive training & evaluation demo
├── src/                   # Source code package
│   ├── __init__.py        # Package initializer
│   ├── config.py          # Column definitions, mappings, and local paths
│   ├── data_loader.py     # Downloads and parses raw files, drops difficulty column
│   ├── preprocessor.py    # Target mapping, scaling, encoding, and alignment
│   ├── trainer.py         # Model training script
│   ├── evaluator.py       # Metrics report and confusion matrix plotting
│   └── predict.py         # Inference pipeline for predicting new network records
├── requirements.txt       # Project dependencies
└── main.py                # Pipeline orchestrator CLI
```

---

## Installation & Setup

1. **Verify Python Installation**: Ensure you have Python 3.8+ installed.
2. **Install Dependencies**: Run the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Pipeline

You can run each step of the pipeline individually or execute the entire pipeline end-to-end using the command line orchestrator `main.py`.

### 1. Download the Dataset
Downloads `KDDTrain+.txt` and `KDDTest+.txt` from verified mirrors:
```bash
python main.py download
```

### 2. Preprocess features
Runs label mapping, scales numerical features using `StandardScaler`, encodes categorical variables (`protocol_type`, `service`, `flag`) using `OneHotEncoder`, and aligns column structure between train and test sets. It saves preprocessed CSVs and the pipeline object `models/preprocessor.joblib`:
```bash
python main.py preprocess
```

### 3. Train the Models
Fits Decision Tree and Random Forest classifiers on the processed training set and saves the serialized models into `models/`:
```bash
python main.py train
```

### 4. Evaluate the Models
Evaluates trained models on the test set, prints classification reports, and saves the confusion matrix plots in `models/`:
```bash
python main.py evaluate
```

### 5. Run the Entire Pipeline from Scratch
Executes all the above steps in sequence:
```bash
python main.py run-all
```

---

## Inference (Prediction on New Traffic Records)

To verify the inference pipeline or predict on custom records, use `src/predict.py`.

### Run Prediction on Mock Samples
Runs prediction on hardcoded Normal and DoS Neptune mock packets:
```bash
python -m src.predict
```

### Run Prediction on Custom CSV Files
To run prediction on a batch of unlabelled traffic records, pass a CSV containing the 41 feature columns (matching the feature columns of NSL-KDD):
```bash
python -m src.predict --file path/to/your/custom_traffic_records.csv
```

---

## Model Pipeline Details

- **Scaling**: Robust standardization is performed using `StandardScaler` on numerical columns (e.g. `duration`, `src_bytes`, `dst_bytes`, etc.).
- **One-Hot Encoding**: Handled using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to prevent mismatching feature vectors during inference when unseen categories appear.
- **Label Mappings**:
  - **Binary**: `normal` class -> `0` (Normal), any other attack name -> `1` (Anomaly).
  - **Multiclass**: Grouped into 5 classes:
    - `Normal` -> `0`
    - `DoS` (Denial of Service) -> `1`
    - `Probe` (Scanning/Probing) -> `2`
    - `R2L` (Remote to Local) -> `3`
    - `U2R` (User to Root) -> `4`
