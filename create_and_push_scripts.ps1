# Set up base directory and GitHub repo URL
$baseDir = "D:\Project_N"
$scriptsDir = "$baseDir\scripts"
$repoUrl = "https://github.com/Sachitk01/Project_N.git"

# Ensure the scripts directory exists
if (-Not (Test-Path $scriptsDir)) {
    Write-Output "Scripts directory does not exist. Please create the directory structure first."
    exit
}

# Create the Python scripts with relevant code
Write-Output "Creating Python scripts in $scriptsDir..."

# main_v1.py
$mainV1 = @'
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
'@
Set-Content -Path "$scriptsDir\main_v1.py" -Value $mainV1

# Repeat for other scripts

# logging_setup_v1.py
$loggingSetupV1 = @'
import logging
import os

def setup_logging():
    log_dir = "D:/Project_N/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "admission_tool.log")

    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Logging initialized. Logs will be saved to admission_tool.log")
'@
Set-Content -Path "$scriptsDir\logging_setup_v1.py" -Value $loggingSetupV1

# data_loader_v1.py
$dataLoaderV1 = @'
import pandas as pd
import os

data_cache = {}

def load_data():
    global data_cache
    if data_cache:
        return (data_cache["seat_matrix_2024"], data_cache["allotment_2024_r1"], 
                data_cache["cutoff_2024_r1"], data_cache["college_list"], 
                data_cache["cutoff_data_2023_2022"])
    
    base_dir = "D:/Project_N/data"

    seat_matrix_2024 = pd.read_excel(os.path.join(base_dir, "2024_Seat_matrix.xlsx"))
    allotment_2024_r1 = pd.read_excel(os.path.join(base_dir, "2024-R1_seat_allotment_data.xlsx"))
    cutoff_2024_r1 = pd.read_excel(os.path.join(base_dir, "2024-R1_cutoff_data.xlsx"))
    college_list = pd.read_excel(os.path.join(base_dir, "list_colleges.xlsx"))
    cutoff_data_2023_2022 = pd.read_excel(os.path.join(base_dir, "2023-2022_cutoff_data.xlsx"), sheet_name=None)

    data_cache = {
        "seat_matrix_2024": seat_matrix_2024,
        "allotment_2024_r1": allotment_2024_r1,
        "cutoff_2024_r1": cutoff_2024_r1,
        "college_list": college_list,
        "cutoff_data_2023_2022": cutoff_data_2023_2022
    }
    
    return seat_matrix_2024, allotment_2024_r1, cutoff_2024_r1, college_list, cutoff_data_2023_2022
'@
Set-Content -Path "$scriptsDir\data_loader_v1.py" -Value $dataLoaderV1

# model_training_v1.py
$modelTrainingV1 = @'
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import logging

def train_cutoff_prediction_model(cutoff_data):
    logging.info("Training cutoff prediction model...")
    try:
        # Combine data from 2022 and 2023
        df_r1_2023 = cutoff_data["2023_R1"]
        df_r2_2023 = cutoff_data["2023_R2"]
        df_r1_2022 = cutoff_data["2022_R1"]
        df_r2_2022 = cutoff_data["2022_R2"]

        # Prepare training data
        X_train = pd.concat([
            df_r1_2023.drop(columns=["Score", "College Name", "College Code"]),
            df_r1_2022.drop(columns=["Score", "College Name", "College Code"])
        ])
        y_train = pd.concat([
            df_r2_2023["Score"] - df_r1_2023["Score"],
            df_r2_2022["Score"] - df_r1_2022["Score"]
        ])

        # Hyperparameter tuning
        model = xgb.XGBRegressor()
        param_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [100, 200]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Save the model
        best_model.save_model("D:/Project_N/models/cutoff_prediction_model.json")
        logging.info("Cutoff prediction model training completed. Best parameters: %s", grid_search.best_params_)
    except Exception as e:
        logging.error("Error in training cutoff prediction model: %s", str(e))
'@
Set-Content -Path "$scriptsDir\model_training_v1.py" -Value $modelTrainingV1

# admission_probability_v1.py
$admissionProbabilityV1 = @'
import pandas as pd
import logging

def calculate_admission_probability_round_2(student_data, predicted_cutoff_data, seat_matrix, college_list):
    """
    Calculates admission probability based on predicted Round 2 cutoffs.
    """
    probabilities = []
    student_category = student_data['category']
    student_score = student_data['NEET Score']

    # Filter eligible colleges based on minority status
    eligible_colleges = college_list.copy()
    if student_data['minority_status'] == 'MIN':
        eligible_colleges = eligible_colleges[eligible_colleges['Minority_Status'] == 'Minority']
    else:
        eligible_colleges = eligible_colleges[eligible_colleges['Minority_Status'] != 'Minority']

    # Filter based on priority list if provided
    if student_data['priority_list']:
        eligible_colleges = eligible_colleges[eligible_colleges['College Name'].isin(student_data['priority_list'])]

    # Merge predicted cutoffs with eligible colleges
    merged_data = pd.merge(eligible_colleges, predicted_cutoff_data, on='College Code', how='left')

    for _, row in merged_data.iterrows():
        cutoff_score = row.get('Predicted_Score', None)
        if pd.notnull(cutoff_score):
            probability = min(1.0, student_score / cutoff_score) * 100
            probabilities.append({
                'College Name': row['College Name'],
                'College Code': row['College Code'],
                'District': row['District'],
                'Year of Establishment': row['Year of Establishment'],
                'Predicted Cutoff Score (R2)': cutoff_score,
                'Admission Probability': f"{probability:.2f}%"
            })
        else:
            logging.warning(f"No predicted cutoff data for {row['College Name']} in category {student_category}.")

    return probabilities
'@
Set-Content -Path "$scriptsDir\admission_probability_v1.py" -Value $admissionProbabilityV1

Write-Output "All scripts created successfully!"

# Navigate to Project N directory and push to GitHub
Set-Location -Path $baseDir

# Initialize git, add remote, and push
git init
git remote add origin $repoUrl
git add .
git commit -m "Initial commit with all scripts"
git branch -M main
git push -u origin main

Write-Output "Pushed to GitHub repository successfully!"
