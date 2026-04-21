import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Custom exception and logging imports
from src.ml_projects.exception import CustomException
from src.ml_projects.logger import logging
from src.ml_projects.utils.main_utils import read_sql_data
from src.ml_projects.config.configuration import ConfigurationManager, DataIngestionConfig
from src.ml_projects.constants import DB_NAME, TABLE_NAME

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the Data Ingestion component with configuration.
        """
        self.ingestion_config = config

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # 1. Attempt to read from MySQL
            logging.info("Attempting to read data from MySQL database")
            df = read_sql_data(table_name=TABLE_NAME)
            
            # 2. Fallback to CSV if SQL fails or is empty
            if df is None or df.empty:
                logging.warning("SQL data unavailable or empty, falling back to local CSV")
                if not os.path.exists(self.ingestion_config.source_data_path):
                    raise FileNotFoundError(
                        f"Source data file not found at {self.ingestion_config.source_data_path}. "
                        f"Please ensure the MySQL table exists or check the data directory."
                    )
                df = pd.read_csv(self.ingestion_config.source_data_path)

            logging.info(f"Dataset read successfully with shape: {df.shape}")

            # 3. Standardize Column Names (Replacing spaces with underscores)
            # This ensures compatibility with downstream models and SQL schemas.
            # This automatically handles 'Journey Date' -> 'Journey_Date', 'Station Name' -> 'Station_Name', etc.
            df.columns = [col.replace(' ', '_') for col in df.columns]

            logging.info(f"Standardized dataframe columns: {df.columns.tolist()}")

            # 4. Create Artifact Directories
            os.makedirs(os.path.dirname(self.ingestion_config.training_file_path), exist_ok=True)

            # 5. Save Raw Data to Feature Store
            df.to_csv(self.ingestion_config.feature_store_file_path, index=False, header=True)

            # 6. Train-Test Split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 7. Save Split Datasets
            train_set.to_csv(self.ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Data Ingestion process completed successfully")

            return (
                self.ingestion_config.training_file_path,
                self.ingestion_config.testing_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_ingestion = DataIngestion(config=config_manager.get_data_ingestion_config())
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.error(f"Error in data ingestion main: {e}")