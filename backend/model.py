import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_data(filepath='data/carbon_data.csv'):
    """Loads dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run dataset.py first.")
    return pd.read_csv(filepath)

def build_preprocessor(numeric_features, categorical_features):
    """Builds a scikit-learn preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns R2, MAE, RMSE."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return {'r2': r2, 'mae': mae, 'rmse': rmse}

def train_models():
    """Main function to train and compare models, then save the best one."""
    df = load_data()
    
    # Target variable
    y = df['total_footprint_tco2e']
    X = df.drop(columns=['total_footprint_tco2e'])
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None

    print("Training and evaluating models...")
    for name, model in models.items():
        # Create pipeline for each model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        # Cross validation score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        
        results[name] = metrics
        print(f"\\n[{name}]")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"MAE:      {metrics['mae']:.4f} tCO2e/yr")
        print(f"RMSE:     {metrics['rmse']:.4f} tCO2e/yr")
        print(f"CV R²:    {metrics['cv_r2_mean']:.4f}")

        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model_name = name
            best_pipeline = pipeline

    print(f"\\nBest model selected: {best_model_name} with R² = {best_r2:.4f}")

    # Save metrics to json
    with open('metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Save best model to model/ directory (Note: we just save the whole pipeline to ensure preprocessing is contained)
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_pipeline.named_steps['model'], 'model/carbon_model.pkl')
    joblib.dump(best_pipeline.named_steps['preprocessor'], 'model/preprocessor.pkl')
    
    # We optionally can just save the whole pipeline together, but the prompt asked for model/carbon_model.pkl and model/preprocessor.pkl separately.
    print("Files saved to 'model/carbon_model.pkl', 'model/preprocessor.pkl', and 'metrics.json'.")

if __name__ == '__main__':
    train_models()
