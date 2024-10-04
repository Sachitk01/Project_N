import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import logging
import os

MODEL_PATH = "D:/Project_N/models/cutoff_prediction_model_v2.json"

def train_cutoff_prediction_model(cutoff_data):
    logging.info("Training cutoff prediction model...")
    try:
        # Combine data from 2022, 2023, and new data if available
        df_r1_2023 = cutoff_data['2023_R1']
        df_r2_2023 = cutoff_data['2023_R2']
        df_r1_2022 = cutoff_data['2022_R1']
        df_r2_2022 = cutoff_data['2022_R2']
        
        if '2024_R2' in cutoff_data:
            df_r2_2024 = cutoff_data['2024_R2']
            # Add new data to the training data (if available)
            X_train = pd.concat([
                df_r1_2023.drop(columns=['Score', 'College Name', 'College Code']),
                df_r1_2022.drop(columns=['Score', 'College Name', 'College Code']),
                df_r2_2024.drop(columns=['Score', 'College Name', 'College Code'])
            ])
            y_train = pd.concat([
                df_r2_2023['Score'] - df_r1_2023['Score'],
                df_r2_2022['Score'] - df_r1_2022['Score'],
                df_r2_2024['Score']
            ])
        else:
            # Use only historical data if no new data is available
            X_train = pd.concat([
                df_r1_2023.drop(columns=['Score', 'College Name', 'College Code']),
                df_r1_2022.drop(columns=['Score', 'College Name', 'College Code'])
            ])
            y_train = pd.concat([
                df_r2_2023['Score'] - df_r1_2023['Score'],
                df_r2_2022['Score'] - df_r1_2022['Score']
            ])

        # Hyperparameter tuning
        model = xgb.XGBRegressor()
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Save the model
        best_model.save_model(MODEL_PATH)
        logging.info("Cutoff prediction model training completed. Best parameters: %s", grid_search.best_params_)
    except Exception as e:
        logging.error("Error in training cutoff prediction model: %s", str(e))
