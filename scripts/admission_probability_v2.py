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
            # Calculate probability based on the cutoff score
            if student_category in row:
                category_cutoff = row[student_category]
                probability = min(1.0, student_score / category_cutoff) * 100
            else:
                probability = min(1.0, student_score / cutoff_score) * 100

            # Append probability information
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
