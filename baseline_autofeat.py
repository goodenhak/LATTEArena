import os
import sys
import warnings
import Evaluator
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from autofeat import AutoFeatClassifier

import os
import logging
import time
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='AutoFeat Feature Engineering Pipeline')

    # Logging related parameters
    parser.add_argument('--log_path', type=str, default='./log',
                       help='Path to log directory')
    parser.add_argument('--log_filename', type=str, default=None,
                       help='Log filename (default: {data_name}_AutoFeat.log)')

    # Main parameters
    parser.add_argument('--data_name', type=str, default='credit-g',
                       help='Dataset name')
    parser.add_argument('--seed', type=int, default=2,
                       help='Random seed')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size')

    # AutoFeat parameters
    parser.add_argument('--feateng_steps', type=int, default=2,
                       help='Feature engineering steps for AutoFeat')
    parser.add_argument('--max_gb', type=int, default=1,
                       help='Maximum memory usage in GB for AutoFeat')
    parser.add_argument('--n_jobs', type=int, default=10,
                       help='Number of parallel jobs for AutoFeat')

    return parser.parse_args()

def setup_logging(log_path, log_filename):
    """Setup logging configuration"""
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ensure log directory exists
    os.makedirs(log_path, exist_ok=True)

    log_file = os.path.join(log_path, log_filename)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    return logger

def main():
    # Parse command line arguments
    args = parse_arguments()
    task_type = 1

    # Set log filename (use dataset name if not specified)
    if args.log_filename is None:
        args.log_filename = f"{args.data_name}_AutoFeat_{args.seed}.log"

    # Setup logging
    logger = setup_logging(args.log_path, args.log_filename)
    logger.info("========== START ==========")
    logger.info(f"Arguments: {vars(args)}")

    # Start timing
    total_start_time = time.time()

    # Read and preprocess data
    csv_file_path = f'./tabular_data/{args.data_name}.csv'
    logger.info(f"Reading CSV file: '{csv_file_path}'...")

    # Read CSV file
    df = pd.read_csv(csv_file_path, encoding='utf-8', low_memory=False)
    logger.info(f"Original data shape: {df.shape}")

    # Separate features and target variable
    target_col = 'class'
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(target_col, axis=1)
    else:
        # If no 'class' column, assume last column is target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        target_col = df.columns[-1]

    logger.info(f"Target column: '{target_col}'")
    logger.info(f"Features shape: {X.shape}")

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"Numerical columns: {len(numerical_cols)}")
    logger.info(f"Categorical columns: {len(categorical_cols)}: {categorical_cols}")

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        logger.info("Handling missing values...")
        # Fill numerical columns with median
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())

        # Fill categorical columns with mode
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')

    # Process categorical variables - label encoding
    if categorical_cols:
        logger.info("Encoding categorical columns...")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode target variable if string type
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        logger.info("Encoding target variable...")
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))

    logger.info(f"Final preprocessed data shape: {X.shape}")
    logger.info(f"Missing values: {X.isnull().sum().sum()}")

    # Split data into train+validation and test sets by 4:1 ratio
    logger.info(f"Splitting data into train+val ({1-args.test_size}) and test ({args.test_size}) sets...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y  # Stratified sampling to maintain class ratio
    )

    # Further split train+validation set into train and validation sets
    val_ratio = args.val_size / (1 - args.test_size)  # Calculate validation set ratio of train+val
    logger.info(f"Splitting train+val set into train ({1-val_ratio}) and val ({val_ratio}) sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=args.seed,
        stratify=y_train_val  # Stratified sampling to maintain class ratio
    )

    logger.info("Data preprocessing and splitting completed!")

    # Merge features and labels into complete datasets
    train_val_data = X_train_val.copy()
    train_val_data[target_col] = y_train_val

    # Create training set DataFrame
    train_data = X_train.copy()
    train_data[target_col] = y_train

    # Create validation set DataFrame
    val_data = X_val.copy()
    val_data[target_col] = y_val

    # Create test set DataFrame
    test_data = X_test.copy()
    test_data[target_col] = y_test

    logger.info("Dataset merging completed!")

    # Use AutoFeat for feature engineering
    logger.info("Starting AutoFeat feature engineering...")

    # Separate training set features and labels for AutoFeat
    X_train_autofeat = train_val_data.drop(target_col, axis=1)
    y_train_autofeat = train_val_data[target_col]

    # Initialize AutoFeatClassifier
    afc = AutoFeatClassifier(
        verbose=1,
        feateng_steps=args.feateng_steps,  # Feature engineering steps
        max_gb=args.max_gb,  # Maximum memory usage (GB)
        n_jobs=args.n_jobs,  # Number of parallel jobs
    )

    # Fit AutoFeat on training data and generate new features
    logger.info("Fitting AutoFeat on training data...")
    X_train_new = afc.fit_transform(X_train_autofeat, y_train_autofeat)

    logger.info("AutoFeat feature engineering completed.")
    total_end_time = time.time()
    # Apply new features to training set
    logger.info("Transforming training data...")
    X_train_original = train_data.drop(target_col, axis=1)
    X_train_new = afc.transform(X_train_original)

    # Apply new features to validation set
    logger.info("Transforming validation data...")
    X_val_original = val_data.drop(target_col, axis=1)
    X_val_new = afc.transform(X_val_original)

    # Apply new features to test set
    logger.info("Transforming test data...")
    X_test_original = test_data.drop(target_col, axis=1)
    X_test_new = afc.transform(X_test_original)

    # Create complete datasets with new features
    logger.info("Creating new datasets with engineered features...")

    # Training set
    train_data_new = pd.DataFrame(X_train_new)
    train_data_new[target_col] = train_data[target_col].values

    # Validation set
    val_data_new = pd.DataFrame(X_val_new)
    val_data_new[target_col] = val_data[target_col].values

    # Test set
    test_data_new = pd.DataFrame(X_test_new)
    test_data_new[target_col] = test_data[target_col].values

    logger.info(f"Original feature count: {X_train.shape[1]}")
    logger.info(f"New feature count: {X_train_new.shape[1]}")

    # Evaluate model performance
    logger.info("Evaluating with AutoGluon...")
    predictor, val_acc = Evaluator.train_and_evaluate(train_data_new, val_data_new, target_col, task_type)
    test_acc = predictor.evaluate(test_data_new)['accuracy']
    logger.info(f"ag_acc = {test_acc}")

    logger.info("Evaluating with RandomForest...")
    predictor, val_acc = Evaluator.train_and_evaluate_rf(train_data_new, val_data_new, target_col, task_type)
    predictor, test_acc = Evaluator.train_and_evaluate_rf(train_data_new, test_data_new, target_col, task_type)
    
    logger.info(f"val_acc = {val_acc}")
    logger.info(f"test_acc = {test_acc}")
    
    # Save results
    output_dir = f"./tmp/{args.data_name}/"
    os.makedirs(output_dir, exist_ok=True)

    train_data_new.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_data_new.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_data_new.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Record time
    logger.info(f"Total time used = {total_end_time - total_start_time:.2f} seconds")
    logger.info("========== END ==========")

if __name__ == "__main__":
    main()