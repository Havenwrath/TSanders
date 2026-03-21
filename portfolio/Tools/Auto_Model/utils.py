# Shared utilities for the Auto_Model project

def load_csv_data(filepath):
    """Load CSV data and return features and target."""
    import pandas as pd
    df = pd.read_csv(filepath)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def save_model(model, filepath):
    """Save model to file."""
    import joblib
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load model from file."""
    import joblib
    return joblib.load(filepath)