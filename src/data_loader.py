import os
import requests
import pandas as pd
import numpy as np
import logging
from src.config import (
    TRAIN_URL, TEST_URL, RAW_TRAIN_PATH, RAW_TEST_PATH, COLUMNS
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, destination):
    """
    Downloads a file from a URL to a local destination if it doesn't already exist.
    """
    if os.path.exists(destination):
        logger.info(f"File already exists: {destination}. Skipping download.")
        return

    logger.info(f"Downloading from {url} to {destination}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download file from {url}. Error: {e}")
        raise

def download_dataset():
    """
    Downloads both train and test sets for the NSL-KDD dataset.
    """
    os.makedirs(os.path.dirname(RAW_TRAIN_PATH), exist_ok=True)
    download_file(TRAIN_URL, RAW_TRAIN_PATH)
    download_file(TEST_URL, RAW_TEST_PATH)

def load_raw_data(file_path):
    """
    Loads raw NSL-KDD file, applies column names, handles missing values,
    and removes the difficulty level column.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}. Please download it first.")

    logger.info(f"Loading raw dataset from {file_path}...")
    
    # Load dataset. NSL-KDD is comma-separated and has no header row
    df = pd.read_csv(file_path, header=None, names=COLUMNS)
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values in {file_path}. Filling missing values...")
        # Fill numeric with median and categorical with mode
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        logger.info("No missing values found.")

    # Remove the difficulty level column as requested
    if 'difficulty_level' in df.columns:
        logger.info("Removing unnecessary column 'difficulty_level'...")
        df = df.drop(columns=['difficulty_level'])

    return df

def get_datasets():
    """
    Orchestrates downloading and loading train and test datasets.
    """
    download_dataset()
    train_df = load_raw_data(RAW_TRAIN_PATH)
    test_df = load_raw_data(RAW_TEST_PATH)
    return train_df, test_df

if __name__ == "__main__":
    # Test data loader execution
    train_df, test_df = get_datasets()
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    print("Columns in Train dataset:", list(train_df.columns))
