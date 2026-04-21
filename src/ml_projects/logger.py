import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler (log file এ save)
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(logging.Formatter(
    "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
))

# Console handler (terminal-এ show)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(levelname)s - %(message)s"
))

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)