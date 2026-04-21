import os
from pathlib import Path
from src.ml_projects.logger import logging
from src.ml_projects.constants import *
from src.ml_projects.utils.main_utils import read_yaml_file
from src.ml_projects.config.config import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            ModelTrainerConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = self._safe_read_yaml(config_filepath, "Config")
        self.params = self._safe_read_yaml(params_filepath, "Params")
        self.schema = self._safe_read_yaml(schema_filepath, "Schema")

        artifacts_root = self.config.get('artifacts_root', ARTIFACT_DIR)
        os.makedirs(artifacts_root, exist_ok=True)

    def _safe_read_yaml(self, path, name) -> dict:
        if os.path.exists(path):
            content = read_yaml_file(path)
            return content if content else {}
        logging.warning(f"{name} file not found at {path}. Using empty defaults.")
        return {}

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.get('data_ingestion', {})
        artifacts_root = self.config.get('artifacts_root', ARTIFACT_DIR)
        root_dir = os.path.join(artifacts_root, DATA_INGESTION_DIR_NAME)
        os.makedirs(root_dir, exist_ok=True)
        return DataIngestionConfig(
            root_dir=Path(root_dir),
            feature_store_file_path=config.get('raw_data_path', os.path.join(root_dir, FILE_NAME)),
            training_file_path=config.get('ingested_train_dir', os.path.join(root_dir, TRAIN_FILE_NAME)),
            testing_file_path=config.get('ingested_test_dir', os.path.join(root_dir, TEST_FILE_NAME)),
            source_data_path=config.get('source_data_path', os.path.join("notebook", "data", FILE_NAME))
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.get('data_validation', {})
        artifacts_root = self.config.get('artifacts_root', ARTIFACT_DIR)
        root_dir = os.path.join(artifacts_root, DATA_VALIDATION_DIR_NAME)
        os.makedirs(root_dir, exist_ok=True)
        return DataValidationConfig(
            root_dir=Path(root_dir),
            drift_report_file_path=Path(config.get('drift_report', os.path.join(root_dir, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)))
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.get('data_transformation', {})
        artifacts_root = self.config.get('artifacts_root', ARTIFACT_DIR)
        root_dir = os.path.join(artifacts_root, DATA_TRANSFORMATION_DIR_NAME)
        os.makedirs(root_dir, exist_ok=True)
        
        transformed_obj_dir = config.get('transformed_obj_dir', root_dir)
        return DataTransformationConfig(
            root_dir=Path(root_dir),
            preprocessor_obj_file_path=Path(os.path.join(transformed_obj_dir, PREPROCESSING_OBJECT_FILE_NAME)),
            transformed_train_file_path=Path(os.path.join(root_dir, "train.npy")),
            transformed_test_file_path=Path(os.path.join(root_dir, "test.npy"))
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.get('model_trainer', {})
        artifacts_root = self.config.get('artifacts_root', ARTIFACT_DIR)
        root_dir = os.path.join(artifacts_root, MODEL_TRAINER_DIR_NAME)
        os.makedirs(root_dir, exist_ok=True)
        return ModelTrainerConfig(
            root_dir=Path(root_dir),
            trained_model_file_path=Path(config.get('model_path', os.path.join(root_dir, MODEL_FILE_NAME))),
            expected_accuracy=MODEL_TRAINER_EXPECTED_SCORE
        )