# ML-IDS System
Machine Learning Intrusion Detection System (ML-IDS) Plan
This project implements a complete Machine Learning Intrusion Detection System (ML-IDS) using Python and the NSL-KDD dataset. The goal is to build a robust, reproducible, and modular pipeline that downloads the dataset, preprocesses it, trains classification models (for both binary and multi-class intrusion detection), evaluates their performance, and saves the trained models for inference.

We will set up the project under the local path: C:\Users\dell\.gemini\antigravity\scratch\ml_ids. We recommend setting this directory as the active workspace once created.

Directory Structure
text

ml_ids/
├── data/                  # Raw and processed dataset files
├── models/                # Saved trained model files (.joblib)
├── notebooks/             # Jupyter notebooks for EDA and experimentation
│   ├── 01_eda.ipynb       # Exploratory Data Analysis
│   └── 02_model_training.ipynb  # Interactive training & evaluation
├── src/                   # Python source code package
│   ├── __init__.py
│   ├── config.py          # Configuration parameters, paths, and feature names
│   ├── data_loader.py     # Script/module to download the NSL-KDD dataset
│   ├── preprocessor.py    # Pipeline for scaling, one-hot encoding, and label mapping
│   ├── trainer.py         # Model training script
│   ├── evaluator.py       # Metrics calculation & visualization
│   └── predict.py         # Inference script for classifying new network samples
├── requirements.txt       # Project dependencies
├── main.py                # Command-Line Interface to run the pipeline
└── README.md              # Project documentation and guide
Proposed Changes
Configuration
[NEW] 
config.py
Stores paths, column names, URL locations for dataset downloads, mapping definitions (individual attacks to DoS, Probe, R2L, U2R), scaling configurations, and hyperparameter dictionaries.

Data Loader
[NEW] 
data_loader.py
Downloads the raw NSL-KDD dataset from public repositories (specifically KDDTrain+.txt and KDDTest+.txt from standard mirrors like Defcom17 or WUSTL) and stores them in the data/ folder. It will use the requests library and show download progress.

Preprocessing
[NEW] 
preprocessor.py
Handles data transformations:

Columns assignment (43 features/attributes from NSL-KDD)
Binary labels mapping (normal vs attack)
Multi-class labels mapping (normal, dos, probe, r2l, u2r)
Dropping irrelevant columns (like the difficulty_level column)
Standardizing/Scaling numeric columns using StandardScaler
One-hot encoding of categorical variables (protocol_type, service, flag) with structural alignment between train and test sets
Saving preprocessing artifacts (scikit-learn pipeline/transformers) using joblib so they can be reused during inference.
Model Training
[NEW] 
trainer.py
Implements:

Training of multiple models (e.g., Random Forest, Decision Tree, and Logistic Regression / XGBoost equivalent)
Support for both Binary Classification (Intrusion vs. Normal) and Multi-Class Classification (5 classes)
Hyperparameter tuning configuration
Saving trained models to the models/ directory using joblib.
Model Evaluation
[NEW] 
evaluator.py
Generates classification reports, accuracy, precision, recall, F1-score, confusion matrix plots, and ROC/AUC scores for both training and test datasets. Saves confusion matrices and ROC charts as PNGs in the models/ or a separate reports/ directory.

Inference / Prediction
[NEW] 
predict.py
Provides a function/CLI interface to predict the class (binary/multi-class) of a single input record or a new batch of unlabelled traffic data, using saved preprocessor and model artifacts.

Notebooks
[NEW] 
01_eda.ipynb
Jupyter notebook demonstrating dataset loading, feature distributions, class imbalance, and correlation analysis.

[NEW] 
02_model_training.ipynb
Jupyter notebook showing step-by-step model training, comparison of multiple algorithms, and visualization of the evaluation results.

Orchestration & Dependencies
[NEW] 
requirements.txt
Specifies dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, requests, jupyter, ipykernel.

[NEW] 
main.py
Provides a CLI tool with commands:

download: Downloads dataset files.
preprocess: Runs preprocessor pipeline and saves transformers.
train: Trains and saves binary/multi-class models.
evaluate: Evaluates trained models on the test set and prints/saves reports.
run-all: Performs all steps in sequence.
[NEW] 
README.md
Detailed setup instructions, package requirements, usage commands, and explanation of model architecture.

Verification Plan
Automated Verification
We will run the complete pipeline using:

python main.py download to verify dataset downloading and integrity.
python main.py preprocess to verify correct data cleaning, mapping, and column alignment.
python main.py train to verify error-free model training and serialization.
python main.py evaluate to verify metric computation and visualization generation.
Create a test script to check the predict.py module on a mock input record.
Manual Verification
We will verify that files are generated in data/ and models/ folders.
We will review classification scores to ensure the models achieve standard performance metrics on NSL-KDD (typically >95% accuracy for binary, >75% for multi-class).
