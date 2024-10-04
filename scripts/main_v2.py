import sys
import os

# Add the scripts folder to the system path to help Python find the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import logging
import concurrent.futures
import pandas as pd
import xgboost as xgb
from admission_probability_v2 import calculate_admission_probability_round_2
from data_loader_v2 import load_data
from model_training_v2 import train_cutoff_prediction_model, predict_round_2_cutoff
from logging_setup_v2 import setup_logging

MODEL_PATH = "D:/Project_N/models/cutoff_prediction_model_v2.json"

def get_student_data():
    return {
        "name": input("Enter your name: "),
        "gender": input("Enter your gender (M/F/O): ").upper(),
        "NEET Score": int(input("Enter your NEET Score: ")),
        "AI Rank": int(input("Enter your AI Rank: ")),
        "State Merit Rank": int(input("Enter your State Merit Rank: ")),
        "category": input("Enter your category (e.g., OCG, SCG, STG): ").upper(),
        "minority_status": input("Do you belong to PWD, CAP, EWS, MIN, or NONE? ").upper(),
        "priority_list": input("Enter your priority list of colleges (comma-separated, or leave blank if not available): ").split(",")
    }

def save_probabilities(output_file, admission_probabilities):
    with pd.ExcelWriter(output_file, mode="a", engine="openpyxl") as writer:
        pd.DataFrame(admission_probabilities).to_excel(writer, index=False)

def main():
    setup_logging()
    logging.info("Starting NEET UG Admission Tool")

    # Load data, including new data if available
    seat_matrix_2024, allotment_2024_r1, cutoff_2024_r1, college_list, cutoff_data_2023_2022, new_data = load_data()
    logging.info("Data loading completed.")

    # Check if new data is available for retraining the model
    retrain_model = bool(new_data)  # Retrain if there is new data

    # Train or load the model based on new data availability
    if not os.path.exists(MODEL_PATH) or retrain_model:
        logging.info("Training or retraining the model with new data...")
        train_cutoff_prediction_model(cutoff_data_2023_2022)
    else:
        logging.info("Loading the pre-trained model...")

    # Load the model
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    # Predict 2024 Round 2 cutoffs
    logging.info("Predicting 2024 Round 2 cutoffs...")
    predicted_round_2_cutoffs = predict_round_2_cutoff(model, cutoff_2024_r1)
    logging.info("Prediction of Round 2 cutoffs completed.")

    # Save predicted cutoffs immediately
    predicted_round_2_cutoffs.to_excel("D:/Project_N/output/predicted_round_2_cutoffs_2024.xlsx", index=False)
    logging.info("Predicted Round 2 cutoffs saved.")

    # Use new seat matrix for Round 2 if available for calculating admission probabilities
    seat_matrix_to_use = new_data.get('seat_matrix_2024_r2', seat_matrix_2024)

    # Calculate admission probabilities for each student
    while True:
        student_data = get_student_data()
        admission_probabilities = calculate_admission_probability_round_2(
            student_data, predicted_round_2_cutoffs, seat_matrix_to_use, college_list
        )

        # Save the output
        output_file = f"D:/Project_N/output/{student_data['name']}_admission_probabilities.xlsx"
        save_probabilities(output_file, admission_probabilities)

        if input("Do you want to enter data for another student? (yes/no): ").lower() != "yes":
            break

    logging.info("Program execution completed.")

if __name__ == "__main__":
    main()
