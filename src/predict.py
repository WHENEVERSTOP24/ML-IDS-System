import os
import pandas as pd
import numpy as np
import joblib
import argparse
import logging
from src.config import MODELS_DIR, COLUMNS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output target lists
BINARY_LABELS = ['Normal', 'Anomaly']
MULTICLASS_LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# Feature columns used for training (first 41 columns)
FEATURE_COLUMNS = COLUMNS[:-2]

def load_inference_artifacts():
    """
    Loads preprocessor and trained models.
    """
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    binary_model_path = os.path.join(MODELS_DIR, 'binary_rf.joblib')
    multiclass_model_path = os.path.join(MODELS_DIR, 'multiclass_rf.joblib')
    
    if not all(os.path.exists(p) for p in [preprocessor_path, binary_model_path, multiclass_model_path]):
        raise FileNotFoundError("Missing preprocessing or model artifacts in models/ folder. Please run preprocess and train first.")
        
    preprocessor = joblib.load(preprocessor_path)
    binary_model = joblib.load(binary_model_path)
    multiclass_model = joblib.load(multiclass_model_path)
    
    return preprocessor, binary_model, multiclass_model

def predict(raw_data):
    """
    Predicts the classification for raw input data.
    Input raw_data can be a dictionary, list of dictionaries, or a Pandas DataFrame.
    """
    # Load artifacts
    preprocessor, binary_model, multiclass_model = load_inference_artifacts()
    
    # Convert input to DataFrame
    if isinstance(raw_data, dict):
        df = pd.DataFrame([raw_data])
    elif isinstance(raw_data, list):
        df = pd.DataFrame(raw_data)
    elif isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    else:
        raise ValueError("Invalid input format. Must be dict, list of dicts, or DataFrame.")
        
    # Ensure all required features are present
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing the following required features: {missing_cols}")
        
    # Order the columns exactly as they were during training
    X = df[FEATURE_COLUMNS]
    
    # Preprocess
    X_processed_arr = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed_arr, columns=feature_names)
    
    # Predict
    binary_preds = binary_model.predict(X_processed)
    multiclass_preds = multiclass_model.predict(X_processed)
    
    # Predict probabilities (if supported by models)
    try:
        binary_probs = binary_model.predict_proba(X_processed)[:, 1]
        multiclass_probs = np.max(multiclass_model.predict_proba(X_processed), axis=1)
    except Exception:
        binary_probs = [None] * len(df)
        multiclass_probs = [None] * len(df)
        
    # Format results
    results = []
    for i in range(len(df)):
        res = {
            'binary_prediction_id': int(binary_preds[i]),
            'binary_prediction': BINARY_LABELS[int(binary_preds[i])],
            'binary_confidence': float(binary_probs[i]) if binary_probs[i] is not None else None,
            'multiclass_prediction_id': int(multiclass_preds[i]),
            'multiclass_prediction': MULTICLASS_LABELS[int(multiclass_preds[i])],
            'multiclass_confidence': float(multiclass_probs[i]) if multiclass_probs[i] is not None else None
        }
        results.append(res)
        
    return results

def get_mock_sample():
    """
    Returns a mock network traffic sample for testing prediction (Normal and DoS features).
    """
    # Example of a normal traffic record (values typical of normal connections)
    normal_sample = {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
        'src_bytes': 215, 'dst_bytes': 4500, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0, 'root_shell': 0,
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
        'count': 1, 'srv_count': 1, 'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 1, 'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    }
    
    # Example of an anomaly traffic record (values typical of a DoS Neptune attack)
    anomaly_sample = {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'S0',
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0,
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
        'count': 229, 'srv_count': 10, 'serror_rate': 1.0, 'srv_serror_rate': 1.0, 'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0, 'same_srv_rate': 0.04, 'diff_srv_rate': 0.06, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 255, 'dst_host_srv_count': 10, 'dst_host_same_srv_rate': 0.04,
        'dst_host_diff_srv_rate': 0.06, 'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 1.0, 'dst_host_srv_serror_rate': 1.0,
        'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    }
    
    return [normal_sample, anomaly_sample]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict network traffic anomaly using ML-IDS.")
    parser.add_argument('--file', type=str, help="Path to CSV file containing features for prediction.")
    args = parser.parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
        else:
            logger.info(f"Loading data for prediction from {args.file}...")
            df = pd.read_csv(args.file)
            preds = predict(df)
            print("\n=== Predictions ===")
            for i, p in enumerate(preds):
                print(f"Row {i+1}: Binary -> {p['binary_prediction']} (conf: {p['binary_confidence']:.4f}), Multiclass -> {p['multiclass_prediction']} (conf: {p['multiclass_confidence']:.4f})")
    else:
        logger.info("No input file provided. Running prediction on mock samples...")
        samples = get_mock_sample()
        preds = predict(samples)
        
        print("\n" + "="*50)
        print("                 MOCK PREDICTIONS")
        print("="*50)
        print("Sample 1 (Expected Normal):")
        print(f"  Binary prediction:      {preds[0]['binary_prediction']} (confidence: {preds[0]['binary_confidence']:.4f})")
        print(f"  Multiclass prediction:  {preds[0]['multiclass_prediction']} (confidence: {preds[0]['multiclass_confidence']:.4f})")
        print("\nSample 2 (Expected Anomaly/DoS):")
        print(f"  Binary prediction:      {preds[1]['binary_prediction']} (confidence: {preds[1]['binary_confidence']:.4f})")
        print(f"  Multiclass prediction:  {preds[1]['multiclass_prediction']} (confidence: {preds[1]['multiclass_confidence']:.4f})")
        print("="*50 + "\n")
