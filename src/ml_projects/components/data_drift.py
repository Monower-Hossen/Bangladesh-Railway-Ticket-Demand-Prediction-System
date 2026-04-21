import os
import sys
import pandas as pd
import numpy as np
from pandas import DataFrame
import json

from src.ml_projects.exception import CustomException
from src.ml_projects.logger import logging
from src.ml_projects.utils.main_utils import write_yaml_file, read_yaml_file
from src.ml_projects.config.configuration import ConfigurationManager, DataValidationConfig
from src.ml_projects.constants import SCHEMA_FILE_PATH

class DataDriftDetector:
    def __init__(self, config: DataValidationConfig):
        try:
            self.data_validation_config = config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Detects if the distribution of data has changed between reference and current datasets.
        Saves the report to the configured path.
        """
        try:
            try:
                # Lazy loading heavy monitoring dependencies
                from evidently.report import Report
                from evidently import ColumnMapping
                from evidently.metric_preset import DataDriftPreset
            except ImportError:
                logging.warning("Evidently library is not installed. Skipping data drift detection. "
                                "To enable this feature, run: pip install evidently")
                return False

            logging.info("Starting Data Drift detection using Evidently")
            
            # Standardize column names from schema to match ingested data (underscores instead of spaces)
            target_col = self._schema_config.get("target_column", "").replace(' ', '_')
            num_cols = [c.replace(' ', '_') for c in self._schema_config.get("numerical_columns", [])]
            cat_cols = [c.replace(' ', '_') for c in self._schema_config.get("categorical_columns", [])]

            def get_clean_df(df: DataFrame) -> DataFrame:
                """
                Force conversion to vanilla Python/Numpy types to bypass Pydantic 
                validation errors in Evidently 0.4.x caused by Pandas 2.0 ExtensionArrays.
                """
                # Standardize column selection based on schema and available data
                # Filter out high-cardinality columns (e.g., Ticket_No, Passenger_Name) 
                # which cause Evidently's drift tests to hang. 
                # We exempt the target_col from this check to ensure we monitor target drift.
                potential_cols = [c for c in (num_cols + cat_cols) if c and c in df.columns]
                relevant_cols = [c for c in potential_cols if df[c].nunique() < 100]
                
                if target_col and target_col in df.columns:
                    relevant_cols.append(target_col)

                skipped_cols = list(set(potential_cols) - set(relevant_cols))
                if skipped_cols:
                    logging.info(f"Skipping high-cardinality columns for drift detection: {skipped_cols}")
                
                # Create a fresh DataFrame with explicitly enforced base types
                # This prevents Pandas 2.0+ from using specialized ExtensionArrays that break Pydantic validation.
                cleaned_df = pd.DataFrame(index=df.index)
                for col in relevant_cols:
                    if col in num_cols or col == target_col:
                        cleaned_df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                    else:
                        # Standardize categorical nulls to "None" string to prevent Evidently mapping errors
                        cleaned_df[col] = df[col].fillna("None").astype(str).replace(['nan', 'NaN', 'null'], 'None')
                        
                        # Ensure it stays as object type for Evidently
                        cleaned_df[col] = cleaned_df[col].astype(object)
                
                return cleaned_df.reset_index(drop=True)

            ref_df = get_clean_df(reference_df)
            curr_df = get_clean_df(current_df)

            # Filter out columns that are entirely null (empty) in the reference dataset.
            # Evidently raises ValueError if a column defined in ColumnMapping has no data.
            # We also ensure the column exists in the current_df to avoid mapping errors.
            valid_num_cols = [c for c in num_cols if c in ref_df.columns and c in curr_df.columns 
                             and not ref_df[c].isnull().all()]
            valid_cat_cols = [c for c in cat_cols if c in ref_df.columns and c in curr_df.columns 
                             and not (ref_df[c] == "None").all()]

            # Explicitly define column mapping using the schema
            column_mapping = ColumnMapping()
            column_mapping.target = target_col if target_col in ref_df.columns else None
            column_mapping.numerical_features = valid_num_cols
            column_mapping.categorical_features = valid_cat_cols

            # Initialize the report with DataDriftPreset
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)
            
            report_dict = drift_report.as_dict()

            # Save the drift report to artifacts
            # We use JSON for saving the report because PyYAML cannot handle complex Evidently objects
            report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report_dict, f, indent=4)
            logging.info(f"Drift report saved at: {report_path}")

            # Extract drift status from the dictionary structure
            # Iterating through metrics ensures we find the drift result even if the order changes
            drift_status = False
            for metric in report_dict.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    drift_status = metric.get("result", {}).get("dataset_drift", False)
                    break

            logging.info(f"Dataset drift detected: {drift_status}")
            return drift_status

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Initialize configuration
        config_manager = ConfigurationManager()
        val_config = config_manager.get_data_validation_config()
        ingestion_config = config_manager.get_data_ingestion_config()
        
        # Load existing data artifacts to test drift detection
        if os.path.exists(ingestion_config.training_file_path) and os.path.exists(ingestion_config.testing_file_path):
            df_ref = pd.read_csv(ingestion_config.training_file_path)
            df_curr = pd.read_csv(ingestion_config.testing_file_path)
            
            detector = DataDriftDetector(config=val_config)
            detector.detect_dataset_drift(df_ref, df_curr)
        else:
            logging.warning("Ingested data not found. Please run data ingestion or 'main.py' first.")
    except Exception as e:
        raise CustomException(e, sys)
