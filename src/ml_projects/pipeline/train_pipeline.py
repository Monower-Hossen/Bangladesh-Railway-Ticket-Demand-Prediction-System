import sys
from src.ml_projects.logger import logging
from src.ml_projects.exception import CustomException

# Components
from src.ml_projects.utils.main_utils import get_regression_metrics
from src.ml_projects.components.data_ingestion import DataIngestion
from src.ml_projects.components.data_validation import DataValidation
from src.ml_projects.components.data_transformation import DataTransformation
from src.ml_projects.components.model_trainer import ModelTrainer

# Config Entities
from src.ml_projects.config.config import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

# Artifact Entities
from src.ml_projects.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact
)

class TrainPipeline:
    def __init__(self):
        # Initializing configurations for each stage
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Step 1: Starting Data Ingestion")
            data_ingestion = DataIngestion(config=self.data_ingestion_config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=train_data_path,
                test_file_path=test_data_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Step 2: Starting Data Validation")
            data_validation = DataValidation(
                config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Step 3: Starting Data Transformation")
            data_transformation = DataTransformation(
                config=self.data_transformation_config
            )
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=data_ingestion_artifact.trained_file_path,
                test_path=data_ingestion_artifact.test_file_path
            )
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=preprocessor_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact, train_arr, test_arr
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, train_arr, test_arr) -> ModelTrainerArtifact:
        try:
            logging.info("Step 4: Starting Model Trainer")
            model_trainer = ModelTrainer(
                config=self.model_trainer_config
            )
            r2, mae, rmse = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            metric_artifact = RegressionMetricArtifact(
                r2_score=r2, 
                mae=mae,
                rmse=rmse
            )
            
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Starting the Training Pipeline")

            # 1. Ingestion
            data_ingestion_artifact = self.start_data_ingestion()

            # 2. Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            if not data_validation_artifact.validation_status:
                raise Exception(f"Data Validation Failed: {data_validation_artifact.message}")

            # 3. Transformation
            data_transformation_artifact, train_arr, test_arr = self.start_data_transformation(data_ingestion_artifact)

            # 4. Model Training
            model_trainer_artifact = self.start_model_trainer(train_arr, test_arr)

            logging.info(f"Pipeline Run Successful. R2 Score: {model_trainer_artifact.metric_artifact.r2_score}")
            print(f"Success! Model Score: {model_trainer_artifact.metric_artifact.r2_score}")
            
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()