import os
import sys
import logging
from datetime import datetime

# Define the logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Create a logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Generate a log file with the current timestamp as the filename
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filepath = os.path.join(log_dir, f"running_logs_{current_time}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),  # Save logs to the file
        logging.StreamHandler(sys.stdout)  # Print logs to the console
    ]
)

# Create a logger instance
logger = logging.getLogger("cnnClassifierLogger")