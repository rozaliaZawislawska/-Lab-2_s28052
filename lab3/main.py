import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

URL = 'https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv'
ARTEFACT_DIR = 'report_artifacts'
TARGET_VARIABLE = 'score'
LOG_FILE = 'analysis_log.txt'
RANDOM_STATE = 42
MODELS_TO_TRAIN = {
    'Linear_Regression': LinearRegression(),
    'Random_Forest': RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100),
    'Gradient_Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=100)
}

# Setup logging
def setup_logger():
    
    logger = logging.getLogger('Lab3 Logger')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(os.path.join(ARTEFACT_DIR, LOG_FILE), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    globals()['logger'] = logger
    
    logger.info("Logger setup complete. All output directed to console and log file.")
    sns.set_style("whitegrid")

def load_data():
    logger.info("Starting data loading process...")
    try:
        data = pd.read_csv(URL, index_col=0) 
        logger.info(f"Data loaded successfully. Initial shape: {data.shape}")
        
        if 'unnamed: 0' in data.columns.str.lower():
             data = data.loc[:, ~data.columns.str.contains('^unnamed')]
             logger.info("Removed an 'Unnamed' index column.")
        
        return data
    except Exception as e:
        logger.error(f"Error loading data from URL: {e}")
        return None

def handle_missing_values(data):
    
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if not missing_data.empty:
        logger.info(f'There are {missing_data.sum()} missing values in the dataset across {len(missing_data)} columns.')

    else:
        logger.info("No missing values (NaN) detected. Data is clean.")
        
    return data


def perform_eda_and_visualizations(data):    
    logger.info("\n--- 3. Statistical Analysis ---")
    logger.info("Descriptive Statistics for Numerical Features:")
    logger.info(data.describe().T.to_string())

    numerical_cols = data.select_dtypes(include=np.number).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    # --- Visualization 1: Target Distribution (Histogram) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data[TARGET_VARIABLE], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of Target Variable: {TARGET_VARIABLE}', fontsize=14)
    plt.xlabel('Test Score (score)')
    plt.savefig(os.path.join(ARTEFACT_DIR, '01_target_distribution.png'))
    plt.close()
    logger.info(f"Saved: Distribution plot for {TARGET_VARIABLE}")

    # --- Visualization 2: Correlation Matrix (Heatmap) ---
    correlation_matrix = data[numerical_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features', fontsize=14)
    plt.savefig(os.path.join(ARTEFACT_DIR, '02_correlation_matrix.png'))
    plt.close()
    logger.info("Saved: Correlation Matrix Heatmap")
    
    # --- Visualization 3: Target vs. Categorical Feature (Box Plot) ---
    if 'gender' in data.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='gender', y=TARGET_VARIABLE, data=data)
        plt.title(f'{TARGET_VARIABLE} vs. Gender', fontsize=14)
        plt.savefig(os.path.join(ARTEFACT_DIR, '03_score_by_gender_boxplot.png'))
        plt.close()
        logger.info("Saved: Box Plot (score vs. gender)")
    
    return data

def feature_engineering_and_preparation(data):
    logger.info("\n--- Stage 2: Feature Engineering and Data Preparation ---")
    
    # 1. Identify Feature Types
    X = data.drop(columns=[TARGET_VARIABLE])
    y = data[TARGET_VARIABLE]
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Numerical features identified: {numerical_features}")
    logger.info(f"Categorical features identified: {categorical_features}")

    # 2. Define Preprocessing Pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    logger.info("Numerical features will be standardized (StandardScaler).")
    
    # One-Hot Encoding for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    logger.info("Categorical features will be encoded (OneHotEncoder).")
    
    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    logger.info("Preprocessing pipeline (ColumnTransformer) defined.")

    # 3. Split Data (Training and Testing Sets)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info(f"Data split completed (80% Train, 20% Test). Random State: {RANDOM_STATE}")
    logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_and_select_best_model(X_train, X_test, y_train, y_test, preprocessor):
    logger.info("\n--- Stage 3: Model Training and Selection ---")
    
    results = {}
    best_r2 = -float('inf')
    best_model_name = None
    
    for name, model in MODELS_TO_TRAIN.items():
        logger.info(f"Training model: {name}...")
        
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        full_pipeline.fit(X_train, y_train)
        logger.info(f"Model {name} training complete.")
        
        y_pred = full_pipeline.predict(X_test)
        
        # --- OBLICZANIE METRYK ---
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred) # <-- NOWA LINIA
        mse = mean_squared_error(y_test, y_pred)   # <-- NOWA LINIA
        # ------------------------
        
        results[name] = {'r2': r2, 'mae': mae, 'mse': mse, 'pipeline': full_pipeline} # <-- ZMIENIONA LINIA
        
        logger.info(f"Model {name} R-squared on test set: {r2:.4f}")
        logger.info(f"Model {name} MAE on test set: {mae:.4f}") # <-- NOWA LINIA
        logger.info(f"Model {name} MSE on test set: {mse:.4f}") # <-- NOWA LINIA
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
    
    logger.info(f"\nBest model selected: {best_model_name} (R^2: {best_r2:.4f})")
    
    return best_model_name, results

if __name__ == "__main__":
    os.makedirs(ARTEFACT_DIR, exist_ok=True)
    
    setup_logger()
    
    logger.info("--- Starting Lab3: Predictive Model Analysis (Stage 1: EDA) ---")
    
    df = load_data()
    
    if df is not None:
        df_cleaned = handle_missing_values(df.copy())
        df_processed = perform_eda_and_visualizations(df_cleaned)
        logger.info("Stage 1 (EDA) completed.")
        
        # --- Stage 2: Feature Engineering and Data Preparation ---
        X_train, X_test, y_train, y_test, preprocessor = feature_engineering_and_preparation(df_processed)
        logger.info("Stage 2 (Feature Engineering) completed.")
        
        # --- Stage 3: Model Training and Selection ---
        best_model_name, training_results = train_and_select_best_model(
            X_train, X_test, y_train, y_test, preprocessor
        )
        logger.info("Stage 3 (Model Training and Selection) completed.")

        # --- Stage 4 (Evaluation and Reporting) would follow here ---
        # (This will be implemented in the next step, using 'training_results')
        
        logger.info("\nScript execution halted after Stage 3. Ready for Evaluation (Stage 4).")
        