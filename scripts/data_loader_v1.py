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
