import os
import pandas as pd
import joblib
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/command-line plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.config import (
    MODELS_DIR, PROCESSED_TEST_PATH, BINARY_TARGET, MULTICLASS_TARGET
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target decoding labels for mapping integers back to string representation
MULTICLASS_LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
BINARY_LABELS = ['Normal', 'Anomaly']

def load_processed_test_data():
    """
    Loads preprocessed testing data.
    """
    if not os.path.exists(PROCESSED_TEST_PATH):
        raise FileNotFoundError(f"Processed test file not found at: {PROCESSED_TEST_PATH}. Please run preprocessing first.")
    
    logger.info(f"Loading processed test data from {PROCESSED_TEST_PATH}...")
    df = pd.read_csv(PROCESSED_TEST_PATH)
    
    # Separate features and target variables
    target_columns = [BINARY_TARGET, MULTICLASS_TARGET]
    feature_columns = [col for col in df.columns if col not in target_columns]
    
    X = df[feature_columns]
    y_binary = df[BINARY_TARGET]
    y_multiclass = df[MULTICLASS_TARGET]
    
    return X, y_binary, y_multiclass

def plot_and_save_confusion_matrix(cm, labels, title, filename):
    """
    Plots a confusion matrix using seaborn and saves it as a PNG file.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(MODELS_DIR, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")

def evaluate_models():
    """
    Loads test data and evaluates all binary and multiclass models.
    """
    X_test, y_test_bin, y_test_multi = load_processed_test_data()
    
    # Check if models exist
    required_models = ['binary_dt.joblib', 'binary_rf.joblib', 'multiclass_dt.joblib', 'multiclass_rf.joblib']
    for model_file in required_models:
        if not os.path.exists(os.path.join(MODELS_DIR, model_file)):
            raise FileNotFoundError(f"Model file not found: {model_file} in {MODELS_DIR}. Please run trainer first.")
            
    logger.info("--- EVALUATING BINARY MODELS ---")
    
    # Load binary models
    bin_dt = joblib.load(os.path.join(MODELS_DIR, 'binary_dt.joblib'))
    bin_rf = joblib.load(os.path.join(MODELS_DIR, 'binary_rf.joblib'))
    
    # Predict and evaluate Binary Decision Tree
    logger.info("Evaluating Binary Decision Tree...")
    y_pred_bin_dt = bin_dt.predict(X_test)
    acc_bin_dt = accuracy_score(y_test_bin, y_pred_bin_dt)
    f1_bin_dt = f1_score(y_test_bin, y_pred_bin_dt)
    print("\n=== Binary Decision Tree Classification Report ===")
    print(classification_report(y_test_bin, y_pred_bin_dt, target_names=BINARY_LABELS))
    
    # Predict and evaluate Binary Random Forest
    logger.info("Evaluating Binary Random Forest...")
    y_pred_bin_rf = bin_rf.predict(X_test)
    acc_bin_rf = accuracy_score(y_test_bin, y_pred_bin_rf)
    f1_bin_rf = f1_score(y_test_bin, y_pred_bin_rf)
    print("=== Binary Random Forest Classification Report ===")
    print(classification_report(y_test_bin, y_pred_bin_rf, target_names=BINARY_LABELS))
    
    # Generate binary confusion matrices
    cm_bin_dt = confusion_matrix(y_test_bin, y_pred_bin_dt)
    cm_bin_rf = confusion_matrix(y_test_bin, y_pred_bin_rf)
    plot_and_save_confusion_matrix(cm_bin_dt, BINARY_LABELS, "Binary Decision Tree Confusion Matrix", "bin_dt_cm.png")
    plot_and_save_confusion_matrix(cm_bin_rf, BINARY_LABELS, "Binary Random Forest Confusion Matrix", "bin_rf_cm.png")
    
    logger.info("--- EVALUATING MULTICLASS MODELS ---")
    
    # Load multiclass models
    multi_dt = joblib.load(os.path.join(MODELS_DIR, 'multiclass_dt.joblib'))
    multi_rf = joblib.load(os.path.join(MODELS_DIR, 'multiclass_rf.joblib'))
    
    # Predict and evaluate Multiclass Decision Tree
    logger.info("Evaluating Multiclass Decision Tree...")
    y_pred_multi_dt = multi_dt.predict(X_test)
    acc_multi_dt = accuracy_score(y_test_multi, y_pred_multi_dt)
    f1_multi_dt = f1_score(y_test_multi, y_pred_multi_dt, average='weighted')
    print("\n=== Multiclass Decision Tree Classification Report ===")
    print(classification_report(y_test_multi, y_pred_multi_dt, target_names=MULTICLASS_LABELS, zero_division=0))
    
    # Predict and evaluate Multiclass Random Forest
    logger.info("Evaluating Multiclass Random Forest...")
    y_pred_multi_rf = multi_rf.predict(X_test)
    acc_multi_rf = accuracy_score(y_test_multi, y_pred_multi_rf)
    f1_multi_rf = f1_score(y_test_multi, y_pred_multi_rf, average='weighted')
    print("=== Multiclass Random Forest Classification Report ===")
    print(classification_report(y_test_multi, y_pred_multi_rf, target_names=MULTICLASS_LABELS, zero_division=0))
    
    # Generate multiclass confusion matrices
    cm_multi_dt = confusion_matrix(y_test_multi, y_pred_multi_dt)
    cm_multi_rf = confusion_matrix(y_test_multi, y_pred_multi_rf)
    plot_and_save_confusion_matrix(cm_multi_dt, MULTICLASS_LABELS, "Multiclass Decision Tree Confusion Matrix", "multi_dt_cm.png")
    plot_and_save_confusion_matrix(cm_multi_rf, MULTICLASS_LABELS, "Multiclass Random Forest Confusion Matrix", "multi_rf_cm.png")
    
    # Summary Table
    print("\n" + "="*50)
    print("                 PERFORMANCE SUMMARY")
    print("="*50)
    summary_data = {
        'Model': ['Binary Decision Tree', 'Binary Random Forest', 'Multiclass Decision Tree', 'Multiclass Random Forest'],
        'Accuracy': [acc_bin_dt, acc_bin_rf, acc_multi_dt, acc_multi_rf],
        'F1-Score': [f1_bin_dt, f1_bin_rf, f1_multi_dt, f1_multi_rf]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_models()
