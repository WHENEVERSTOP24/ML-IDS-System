import argparse
import sys
import logging

from src.data_loader import download_dataset
from src.preprocessor import run_preprocessing_pipeline
from src.trainer import run_training_pipeline
from src.evaluator import evaluate_models

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Machine Learning Intrusion Detection System (ML-IDS) Command Line Interface."
    )
    
    parser.add_argument(
        'command',
        choices=['download', 'preprocess', 'train', 'evaluate', 'run-all'],
        help="Pipeline command to execute: \n"
             "download: Fetches the raw NSL-KDD dataset\n"
             "preprocess: Cleans, scales, encodes categorical features, maps labels and saves processed data\n"
             "train: Fits Decision Trees and Random Forests on binary & multiclass targets\n"
             "evaluate: Reports test metric performance and confusion matrices\n"
             "run-all: Executes the entire pipeline from scratch"
    )
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    try:
        if args.command == 'download':
            logger.info("Executing task: Download NSL-KDD Dataset")
            download_dataset()
            
        elif args.command == 'preprocess':
            logger.info("Executing task: Preprocess Dataset")
            run_preprocessing_pipeline()
            
        elif args.command == 'train':
            logger.info("Executing task: Train Models")
            run_training_pipeline()
            
        elif args.command == 'evaluate':
            logger.info("Executing task: Evaluate Models")
            evaluate_models()
            
        elif args.command == 'run-all':
            logger.info("Executing Full Pipeline")
            logger.info("Step 1/4: Downloading dataset...")
            download_dataset()
            logger.info("Step 2/4: Preprocessing data...")
            run_preprocessing_pipeline()
            logger.info("Step 3/4: Training models...")
            run_training_pipeline()
            logger.info("Step 4/4: Evaluating models...")
            evaluate_models()
            logger.info("Full pipeline completed successfully!")
            
    except Exception as e:
        logger.error(f"Execution failed during '{args.command}' task. Error details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
