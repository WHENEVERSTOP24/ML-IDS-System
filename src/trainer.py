import os
import pandas as pd
import joblib
import logging
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.config import (
    MODELS_DIR, PROCESSED_TRAIN_PATH, BINARY_TARGET, MULTICLASS_TARGET
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data():
    """
    Loads preprocessed training data.
    """
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        raise FileNotFoundError(f"Processed training file not found at: {PROCESSED_TRAIN_PATH}. Please run preprocessing first.")
    
    logger.info(f"Loading processed training data from {PROCESSED_TRAIN_PATH}...")
    df = pd.read_csv(PROCESSED_TRAIN_PATH)
    
    # Separate features and target variables
    target_columns = [BINARY_TARGET, MULTICLASS_TARGET]
    feature_columns = [col for col in df.columns if col not in target_columns]
    
    X = df[feature_columns]
    y_binary = df[BINARY_TARGET]
    y_multiclass = df[MULTICLASS_TARGET]
    
    return X, y_binary, y_multiclass

def train_and_save_model(model, X_train, y_train, model_name, filename):
    """
    Utility function to train a model, track time, and save it.
    """
    logger.info(f"Starting training for {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    logger.info(f"Finished training {model_name} in {elapsed:.2f} seconds.")
    
    model_path = os.path.join(MODELS_DIR, filename)
    logger.info(f"Saving {model_name} to {model_path}...")
    joblib.dump(model, model_path)
    return model

def run_training_pipeline():
    """
    Loads processed data and trains all binary and multiclass models.
    """
    X_train, y_train_bin, y_train_multi = load_processed_data()
    
    logger.info("Initializing models...")
    
    # 1. Binary classification models
    binary_dt = DecisionTreeClassifier(random_state=42, max_depth=15)
    binary_rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15, n_jobs=-1)
    
    # 2. Multiclass classification models
    multiclass_dt = DecisionTreeClassifier(random_state=42, max_depth=15)
    multiclass_rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15, n_jobs=-1)
    
    # Train Binary Models
    logger.info("--- TRAINING BINARY MODELS ---")
    train_and_save_model(binary_dt, X_train, y_train_bin, "Binary Decision Tree", "binary_dt.joblib")
    train_and_save_model(binary_rf, X_train, y_train_bin, "Binary Random Forest", "binary_rf.joblib")
    
    # Train Multiclass Models
    logger.info("--- TRAINING MULTICLASS MODELS ---")
    train_and_save_model(multiclass_dt, X_train, y_train_multi, "Multiclass Decision Tree", "multiclass_dt.joblib")
    train_and_save_model(multiclass_rf, X_train, y_train_multi, "Multiclass Random Forest", "multiclass_rf.joblib")
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    run_training_pipeline()
