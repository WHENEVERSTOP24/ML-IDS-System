# 🛡️ ML-IDS: Machine Learning Intrusion Detection System

## Overview

ML-IDS is a machine learning-based Intrusion Detection System (IDS) built using the NSL-KDD dataset to classify network traffic as either **Normal** or **Malicious**. The project explores multiple machine learning algorithms and compares their effectiveness in detecting cyber attacks.

This project was developed as a hands-on cybersecurity and machine learning portfolio project to understand the complete IDS development lifecycle, including data preprocessing, feature engineering, model training, and evaluation.

---

## Features

* Binary classification of network traffic:

  * Normal Traffic
  * Attack Traffic
* Data preprocessing and feature engineering
* One-hot encoding of categorical network features
* Class balancing using SMOTE
* Comparative analysis of multiple ML algorithms
* Model evaluation using Accuracy, Precision, Recall, and F1-Score
* Trained model serialization for future deployment

---

## Dataset

This project uses the **NSL-KDD** dataset.

The dataset is an improved version of the KDD Cup 1999 dataset and addresses issues such as redundant records.

Download the dataset from:

https://www.unb.ca/cic/datasets/nsl.html

Required files:

* `KDDTrain+.txt`
* `KDDTest+.txt`

Place these files inside the `data/` directory before running the project.

---

## Project Structure

ML-IDS/
├── data/
│   └── README.md
├── models/
│   └── xgboost_ids.pkl
├── notebooks/
│   └── ML_IDS.ipynb
├── reports/
├── src/
├── main.py
├── requirements.txt
└── README.md

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Jupyter Notebook
* Joblib

---

## Machine Learning Pipeline

1. Load NSL-KDD training and testing datasets.
2. Convert multiclass attack labels into binary labels.
3. Encode categorical features using one-hot encoding.
4. Handle class imbalance using SMOTE.
5. Train and evaluate multiple machine learning models.
6. Compare performance using standard evaluation metrics.
7. Save the best-performing model.

---

## Models Evaluated

| Model                 | Accuracy   |
| --------------------- | ---------- |
| Random Forest + SMOTE | 77.16%     |
| Random Forest         | 76.77%     |
| SVM + StandardScaler  | 78.69%     |
| XGBoost               | **79.35%** |

### Best Model: XGBoost

XGBoost achieved the highest overall performance:

* Accuracy: 79.35%
* Attack Precision: 97%
* Attack Recall: 66%
* Attack F1-Score: 78%

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/ML_IDS.ipynb
```

Run the notebook cells sequentially to reproduce the experiments.

---

## Future Improvements

* Multi-class attack classification
* Hyperparameter optimization
* Feature selection techniques
* Real-time packet capture integration
* Streamlit dashboard for interactive predictions
* Deployment as a web application

---

## Learning Outcomes

Through this project, I gained practical experience in:

* Cybersecurity dataset analysis
* Intrusion Detection Systems
* Machine Learning model development
* Feature engineering
* Model evaluation and comparison
* Debugging and optimizing ML pipelines
* End-to-end project development using Python

---

## Author

**Anubhav**

Cybersecurity Student | Ethical Hacking Enthusiast | Machine Learning Learner

Feel free to connect and provide feedback on this project.
