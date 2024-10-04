import logging
import os

def setup_logging():
    log_dir = "D:/Project_N/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "admission_tool.log")

    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Logging initialized. Logs will be saved to admission_tool.log")
