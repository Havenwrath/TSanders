#!/usr/bin/env python3
"""
Data Splitter - Prepares datasets for model evaluation.

Usage:
    python data_splitter.py <dataset_path> [--target <column_name>] [--test-size <float>] [--valid-size <float>] [--output-dir <path>] [--task <regression|classification>]

Examples:
    python data_splitter.py my_data.csv --target price
    python data_splitter.py my_data.csv --target label --task classification
    python data_splitter.py my_data.csv --target y --test-size 0.15 --valid-size 0.15
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def get_target_column(df):
    """Interactively ask user to select target column."""
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i + 1}. {col}")
    
    while True:
        choice = input("\nEnter the target column name or number: ").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(df.columns):
                return df.columns[idx]
            print(f"Invalid number. Please enter 1-{len(df.columns)}")
        elif choice in df.columns:
            return choice
        else:
            print(f"Column '{choice}' not found. Please try again.")


def detect_task_type(df, target_col):
    """Auto-detect if task is classification or regression."""
    unique_values = df[target_col].nunique()
    total_values = len(df[target_col])
    
    if df[target_col].dtype == 'object':
        return 'classification'
    elif unique_values <= 20 and unique_values / total_values < 0.05:
        return 'classification'
    else:
        return 'regression'


def split_and_save(df, target_col, test_size, valid_size, output_dir, task_type):
    """Split data and save to files."""
    train_size = 1.0 - test_size - valid_size
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    if task_type == 'classification':
        stratify = y
    else:
        stratify = None
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    
    valid_ratio = valid_size / (1.0 - test_size)
    
    if task_type == 'classification':
        stratify_temp = y_temp
    else:
        stratify_temp = None
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_ratio, random_state=42, stratify=stratify_temp
    )
    
    train_df = X_train.copy()
    train_df['target'] = y_train
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    valid_df = X_valid.copy()
    valid_df['target'] = y_valid
    
    if task_type == 'classification':
        suffix = '_classification'
    else:
        suffix = ''
    
    train_path = os.path.join(output_dir, f'train{suffix}.csv')
    test_path = os.path.join(output_dir, f'test{suffix}.csv')
    valid_path = os.path.join(output_dir, f'valid{suffix}.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    
    return train_path, test_path, valid_path, len(train_df), len(test_df), len(valid_df)


def main():
    parser = argparse.ArgumentParser(
        description='Split a dataset into train/test/valid files for model evaluation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_splitter.py my_data.csv --target price
  python data_splitter.py my_data.csv --target label --task classification
  python data_splitter.py my_data.csv --target y --test-size 0.15 --valid-size 0.15
        """
    )
    parser.add_argument('dataset', help='Path to the dataset file (CSV)')
    parser.add_argument('--target', '-t', help='Name of the target column')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion (default: 0.2)')
    parser.add_argument('--valid-size', type=float, default=0.2, help='Validation set proportion (default: 0.2)')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--task', choices=['regression', 'classification'], help='Task type (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: File '{args.dataset}' not found.")
        sys.exit(1)
    
    print(f"Loading dataset: {args.dataset}")
    
    if args.dataset.endswith('.csv'):
        df = pd.read_csv(args.dataset)
    elif args.dataset.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(args.dataset)
    elif args.dataset.endswith('.json'):
        df = pd.read_json(args.dataset)
    elif args.dataset.endswith('.parquet'):
        df = pd.read_parquet(args.dataset)
    else:
        df = pd.read_csv(args.dataset)
    
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    if args.target:
        if args.target not in df.columns:
            print(f"Error: Target column '{args.target}' not found in dataset.")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        target_col = args.target
    else:
        target_col = get_target_column(df)
    
    print(f"\nTarget column: {target_col}")
    
    if args.task:
        task_type = args.task
    else:
        task_type = detect_task_type(df, target_col)
        print(f"Auto-detected task type: {task_type}")
    
    if args.test_size + args.valid_size >= 1.0:
        print("Error: test-size + valid-size must be less than 1.0")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nSplitting data (train: {1 - args.test_size - args.valid_size:.0%}, valid: {args.valid_size:.0%}, test: {args.test_size:.0%})...")
    
    train_path, test_path, valid_path, train_len, test_len, valid_len = split_and_save(
        df, target_col, args.test_size, args.valid_size, args.output_dir, task_type
    )
    
    print(f"\nFiles created:")
    print(f"  Train: {train_path} ({train_len} samples)")
    print(f"  Valid: {valid_path} ({valid_len} samples)")
    print(f"  Test:  {test_path} ({test_len} samples)")
    
    print(f"\nReady for {task_type} model evaluation!")
    if task_type == 'classification':
        print("Run classifiers with: python Models/<Classifier>_Eval.py")
    else:
        print("Run regressors with: python Models/<Regressor>_Eval.py")


if __name__ == "__main__":
    main()
