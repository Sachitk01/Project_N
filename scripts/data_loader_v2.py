import pandas as pd
import os

data_cache = {}

def load_data():
    global data_cache
    if data_cache:
        return (data_cache['seat_matrix_2024'], data_cache['allotment_2024_r1'],
                data_cache['cutoff_2024_r1'], data_cache['college_list'],
                data_cache['cutoff_data_2023_2022'], data_cache['new_data'])

    base_dir = "D:/Project_N/data"

    # Load known data files
    seat_matrix_2024 = pd.read_excel(os.path.join(base_dir, '2024_Seat_matrix.xlsx'))
    allotment_2024_r1 = pd.read_excel(os.path.join(base_dir, '2024-R1_seat_allotment_data.xlsx'))
    cutoff_2024_r1 = pd.read_excel(os.path.join(base_dir, '2024-R1_cutoff_data.xlsx'))
    college_list = pd.read_excel(os.path.join(base_dir, 'list_colleges.xlsx'))
    cutoff_data_2023_2022 = pd.read_excel(os.path.join(base_dir, '2023-2022_cutoff_data.xlsx'), sheet_name=None)

    # Check for new data files
    new_data = {}
    if os.path.exists(os.path.join(base_dir, '2024-R2_seat_matrix.xlsx')):
        seat_matrix_2024_r2 = pd.read_excel(os.path.join(base_dir, '2024-R2_seat_matrix.xlsx'))
        if verify_data_quality(seat_matrix_2024_r2, ['College Code', 'Available Seats']):
            new_data['seat_matrix_2024_r2'] = seat_matrix_2024_r2

    if os.path.exists(os.path.join(base_dir, '2024-R2_cutoff_data.xlsx')):
        cutoff_2024_r2 = pd.read_excel(os.path.join(base_dir, '2024-R2_cutoff_data.xlsx'))
        if verify_data_quality(cutoff_2024_r2, ['College Code', 'Category', 'Cutoff Score']):
            new_data['cutoff_2024_r2'] = cutoff_2024_r2

    # Store the data in the cache for reuse
    data_cache = {
        'seat_matrix_2024': seat_matrix_2024,
        'allotment_2024_r1': allotment_2024_r1,
        'cutoff_2024_r1': cutoff_2024_r1,
        'college_list': college_list,
        'cutoff_data_2023_2022': cutoff_data_2023_2022,
        'new_data': new_data  # Add any new data that passed the quality check
    }

    return (seat_matrix_2024, allotment_2024_r1, cutoff_2024_r1, college_list, cutoff_data_2023_2022, new_data)

def verify_data_quality(dataframe, required_columns):
    """Verify if the new data is complete and useful for model training."""
    if all(col in dataframe.columns for col in required_columns):
        if dataframe[required_columns].isnull().sum().sum() == 0:  # Check for missing values
            return True
    return False
