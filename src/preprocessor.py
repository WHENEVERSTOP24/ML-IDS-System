import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import (
    DATA_DIR, MODELS_DIR, CATEGORICAL_FEATURES, ATTACK_MAPPING,
    BINARY_TARGET, MULTICLASS_TARGET, PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH
)
from src.data_loader import get_datasets

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target encoding dictionaries
MULTICLASS_LABEL_MAP = {
    'normal': 0,
    'dos': 1,
    'probe': 2,
    'r2l': 3,
    'u2r': 4
}

def map_targets(df):
    """
    Creates binary and multiclass target columns from raw attack_type.
    """
    logger.info("Mapping raw attack labels to binary and multiclass targets...")
    
    # Strip any whitespace from attack names
    attack_series = df['attack_type'].astype(str).str.strip().str.lower()
    
    # 1. Binary target: 0 if normal, 1 if any attack
    df[BINARY_TARGET] = (attack_series != 'normal').astype(int)
    
    # 2. Multiclass target: map specific attack types to their category
    # Use 'dos' as default fallback for unknown attacks
    mapped_categories = attack_series.map(lambda x: ATTACK_MAPPING.get(x, 'dos'))
    df[MULTICLASS_TARGET] = mapped_categories.map(MULTICLASS_LABEL_MAP)
    
    return df

def fit_transform_pipeline(train_df, test_df):
    """
    Fits standard scaling and one-hot encoding on train features and transforms both datasets.
    """
    logger.info("Initializing and fitting preprocessing pipeline...")
    
    # Extract feature columns (all columns except the original attack_type and our mapped targets)
    target_columns = ['attack_type', BINARY_TARGET, MULTICLASS_TARGET]
    feature_columns = [col for col in train_df.columns if col not in target_columns]
    
    # Separate numeric and categorical features
    numeric_features = [col for col in feature_columns if col not in CATEGORICAL_FEATURES]
    
    logger.info(f"Numeric features count: {len(numeric_features)}")
    logger.info(f"Categorical features count: {len(CATEGORICAL_FEATURES)}")
    
    # Define Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    
    # Extract features and targets
    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    
    y_train_bin = train_df[BINARY_TARGET]
    y_train_multi = train_df[MULTICLASS_TARGET]
    
    y_test_bin = test_df[BINARY_TARGET]
    y_test_multi = test_df[MULTICLASS_TARGET]
    
    # Fit on train, transform both
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get output feature names
    feature_names = preprocessor.get_feature_names_out()
    
    # Reconstruct DataFrames
    X_train_processed = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    # Add target columns back to preprocessed DataFrames
    X_train_processed[BINARY_TARGET] = y_train_bin.values
    X_train_processed[MULTICLASS_TARGET] = y_train_multi.values
    
    X_test_processed[BINARY_TARGET] = y_test_bin.values
    X_test_processed[MULTICLASS_TARGET] = y_test_multi.values
    
    return X_train_processed, X_test_processed, preprocessor

def run_preprocessing_pipeline():
    """
    Orchestrates downloading, loading, preprocessing, and saving data & model artifacts.
    """
    train_df, test_df = get_datasets()
    
    # 1. Map target columns
    train_df = map_targets(train_df)
    test_df = map_targets(test_df)
    
    # 2. Preprocess features
    X_train_processed, X_test_processed, preprocessor = fit_transform_pipeline(train_df, test_df)
    
    # 3. Save preprocessor artifact for inference
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    logger.info(f"Saving preprocessor pipeline to {preprocessor_path}...")
    joblib.dump(preprocessor, preprocessor_path)
    
    # 4. Save processed datasets
    logger.info(f"Saving processed train data to {PROCESSED_TRAIN_PATH}...")
    X_train_processed.to_csv(PROCESSED_TRAIN_PATH, index=False)
    
    logger.info(f"Saving processed test data to {PROCESSED_TEST_PATH}...")
    X_test_processed.to_csv(PROCESSED_TEST_PATH, index=False)
    
    logger.info("Preprocessing completed successfully!")
    return X_train_processed, X_test_processed

if __name__ == "__main__":
    run_preprocessing_pipeline()
