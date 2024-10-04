import logging
import concurrent.futures
import pandas as pd
from admission_probability_v1 import calculate_admission_probability_round_2
from data_loader_v1 import load_data
from model_training_v1 import train_cutoff_prediction_model, predict_round_2_cutoff, train_seat_vacancy_model
from logging_setup_v1 import setup_logging

def main():
    setup_logging()
    logging.info("Starting NEET UG Admission Tool")

    # Load data
    seat_matrix_2024, allotment_2024_r1, cutoff_2024_r1, college_list, cutoff_data_2023_2022 = load_data()
    logging.info("Data loading completed.")

    logging.info("Seat Matrix Columns: %s", seat_matrix_2024.columns.tolist())
    logging.info("Allotment 2024 R1 Columns: %s", allotment_2024_r1.columns.tolist())
    logging.info("Cutoff 2024 R1 Columns: %s", cutoff_2024_r1.columns.tolist())
    logging.info("College List Columns: %s", college_list.columns.tolist())

    # Training models in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_cutoff_model = executor.submit(train_cutoff_prediction_model, cutoff_data_2023_2022)
        future_seat_vacancy_model = executor.submit(train_seat_vacancy_model, seat_matrix_2024, allotment_2024_r1)
    
    # Wait for both models to complete
    future_cutoff_model.result()
    future_seat_vacancy_model.result()
    logging.info("Model training completed.")

    # Predict 2024 Round 2 cutoffs
    logging.info("Predicting 2024 Round 2 cutoffs...")
    predicted_round_2_cutoffs = predict_round_2_cutoff(cutoff_data_2023_2022, cutoff_2024_r1)
    logging.info("Prediction of Round 2 cutoffs completed.")

    # Save predicted cutoffs immediately
    predicted_round_2_cutoffs.to_excel("D:/Project_N/output/predicted_round_2_cutoffs_2024.xlsx", index=False)
    logging.info("Predicted Round 2 cutoffs saved.")

    while True:
        # Gather user input
        student_data = {
            "name": input("Enter your name: "),
            "gender": input("Enter your gender (M/F/O): ").upper(),
            "NEET Score": int(input("Enter your NEET Score: ")),
            "AI Rank": int(input("Enter your AI Rank: ")),
            "State Merit Rank": int(input("Enter your State Merit Rank: ")),
            "category": input("Enter your category (e.g., OCG, SCG, STG): ").upper(),
            "minority_status": input("Do you belong to PWD, CAP, EWS, MIN, or NONE? ").upper(),
            "priority_list": input("Enter your priority list of colleges (comma-separated, or leave blank if not available): ").split(",")
        }

        # Calculate admission probabilities based on predicted Round 2 cutoffs
        admission_probabilities = calculate_admission_probability_round_2(student_data, predicted_round_2_cutoffs, seat_matrix_2024, college_list)

        # Save output
        output_file = f"D:/Project_N/output/{student_data['name']}_admission_probabilities.xlsx"
        with pd.ExcelWriter(output_file, mode="a", engine="openpyxl") as writer:
            pd.DataFrame(admission_probabilities).to_excel(writer, sheet_name=student_data["name"], index=False)
        logging.info(f"Admission probabilities saved for {student_data['name']} in {output_file}.")

        if input("Do you want to enter data for another student? (yes/no): ").lower() != "yes":
            break

    logging.info("Program execution completed.")

if __name__ == "__main__":
    main()
