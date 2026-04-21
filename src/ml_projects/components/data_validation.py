import os
import sys
import pandas as pd
from pandas import DataFrame

from src.ml_projects.components.data_drift import DataDriftDetector
from src.ml_projects.exception import CustomException
from src.ml_projects.logger import logging
from src.ml_projects.utils.main_utils import read_yaml_file, write_yaml_file
from src.ml_projects.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.ml_projects.config.configuration import DataValidationConfig
from src.ml_projects.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        """
        :param config: configuration for data validation
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            self._drift_detector = DataDriftDetector(config=self.data_validation_config)
        except Exception as e:
            raise CustomException(e, sys)

    @property
    def standardized_schema_columns(self) -> list:
        """Returns schema columns with spaces replaced by underscores."""
        schema_columns = self._schema_config.get("columns", {})
        return [col.replace(' ', '_') for col in schema_columns.keys()]

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            # Check if all columns defined in schema exist in the dataframe
            # This is more robust than checking exact length
            status = all(col in dataframe.columns for col in self.standardized_schema_columns)
            logging.info(f"Required columns presence validation status: [{status}]")
            return status
        except Exception as e:
            raise CustomException(e, sys)

    def get_missing_columns(self, df: DataFrame) -> list:
        """
        Checks if all columns defined in schema.yaml exist in the dataframe.
        Returns a list of missing columns.
        """
        try:
            dataframe_columns = df.columns
            missing_columns = [col for col in self.standardized_schema_columns if col not in dataframe_columns]
            if len(missing_columns) > 0:
                logging.error(f"Missing columns detected: {missing_columns}")

            return missing_columns
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Detects if the distribution of data has changed using the DataDriftDetector component.
        """
        try:
            return self._drift_detector.detect_dataset_drift(reference_df, current_df)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_messages = []
            logging.info("Starting data validation for Railway Ticket Demand Prediction")
            
            if not self._schema_config:
                error_messages.append("Schema configuration file is missing or empty")
            else:
                # Reading data using paths from ingestion artifact
                train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
                test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

                for df_type, df in [("training", train_df), ("testing", test_df)]:
                    # 1. Validate Column Count
                    if not self.validate_number_of_columns(dataframe=df):
                        error_messages.append(f"Column count mismatch in {df_type} dataframe")
                    
                    # 2. Validate Column Existence
                    missing_cols = self.get_missing_columns(df=df)
                    if missing_cols:
                        error_messages.append(f"Specific columns missing in {df_type} dataframe: {missing_cols}")

                # 3. Detect Drift if basic schema is valid
                if not error_messages:
                    drift_status = self.detect_dataset_drift(train_df, test_df)
                    if drift_status:
                        error_messages.append("Drift detected in the dataset")
            
            validation_status = len(error_messages) == 0
            validation_msg = ". ".join(error_messages) if error_messages else "Validation Successful"

            if not validation_status:
                logging.error(f"Validation Error: {validation_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact created: {data_validation_artifact}")
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
        