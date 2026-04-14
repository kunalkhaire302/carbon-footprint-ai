# ===========================
# IMPORT REQUIRED LIBRARIES
# ===========================
import os                  # For file/directory operations
import json                # For saving results in JSON format
import pandas as pd        # Data handling (DataFrames)
import numpy as np         # Numerical computations

# Scikit-learn modules for ML pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ML models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Evaluation metrics
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import joblib              # For saving trained models


# ===========================
# LOAD DATA FUNCTION
# ===========================
def load_data(filepath='data/carbon_data.csv'):
    """
    Load dataset from CSV file.

    Args:
        filepath (str): Path to dataset file

    Returns:
        DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run dataset.py first.")
    
    return pd.read_csv(filepath)


# ===========================
# PREPROCESSING PIPELINE
# ===========================
def build_preprocessor(numeric_features, categorical_features):
    """
    Create preprocessing pipeline for numerical and categorical data.

    Steps:
    - Numerical:
        1. Fill missing values using median
        2. Scale features (standardization)
    - Categorical:
        1. Fill missing values using most frequent value
        2. Convert categories to numeric using One-Hot Encoding

    Args:
        numeric_features (list): List of numeric column names
        categorical_features (list): List of categorical column names

    Returns:
        ColumnTransformer: Combined preprocessing pipeline
    """

    # Pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())                    # Normalize data
    ])

    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing
        ('encoder', OneHotEncoder(handle_unknown='ignore'))    # Convert to numeric
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# ===========================
# MODEL EVALUATION FUNCTION
# ===========================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model using standard regression metrics.

    Metrics:
    - R² Score  : How well model explains variance (higher is better)
    - MAE       : Average absolute error
    - RMSE      : Penalizes large errors (root of MSE)

    Args:
        model: Trained ML pipeline
        X_test: Test features
        y_test: Actual target values

    Returns:
        dict: Dictionary containing evaluation metrics
    """

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return {'r2': r2, 'mae': mae, 'rmse': rmse}


# ===========================
# MAIN TRAINING FUNCTION
# ===========================
def train_models():
    """
    Main workflow:
    1. Load dataset
    2. Split features and target
    3. Identify column types
    4. Train multiple models
    5. Evaluate performance
    6. Select best model
    7. Save model + results
    """

    # Load dataset
    df = load_data()
    
    # Separate target variable (output)
    y = df['total_footprint_tco2e']
    
    # Feature variables (input)
    X = df.drop(columns=['total_footprint_tco2e'])
    
    # Identify column types automatically
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),                     # Simple baseline model
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),  # Ensemble model
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)  # Boosting model
    }

    # Store results
    results = {}

    # Track best model
    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None

    print("Training and evaluating models...")

    # Loop through each model
    for name, model in models.items():

        # Combine preprocessing + model into one pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test data
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        # Perform cross-validation (more reliable evaluation)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        
        # Store results
        results[name] = metrics

        # Print results
        print(f"\n[{name}]")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"MAE:      {metrics['mae']:.4f} tCO2e/yr")
        print(f"RMSE:     {metrics['rmse']:.4f} tCO2e/yr")
        print(f"CV R²:    {metrics['cv_r2_mean']:.4f}")

        # Update best model if current is better
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest model selected: {best_model_name} with R² = {best_r2:.4f}")

    # ===========================
    # SAVE RESULTS
    # ===========================

    # Save metrics to JSON file
    with open('metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Create directory if not exists
    os.makedirs('model', exist_ok=True)

    # Save model and preprocessor separately
    joblib.dump(best_pipeline.named_steps['model'], 'model/carbon_model.pkl')
    joblib.dump(best_pipeline.named_steps['preprocessor'], 'model/preprocessor.pkl')
    
    print("Files saved to 'model/carbon_model.pkl', 'model/preprocessor.pkl', and 'metrics.json'.")


# ===========================
# ENTRY POINT
# ===========================
if __name__ == '__main__':
    # Run full training pipeline
    train_models()