import sys
from src.ml_projects.config.configuration import ConfigurationManager
from src.ml_projects.components.data_ingestion import DataIngestion
from src.ml_projects.components.data_validation import DataValidation
from src.ml_projects.components.data_transformation import DataTransformation
from src.ml_projects.components.model_trainer import ModelTrainer
from src.ml_projects.entity.artifact_entity import DataIngestionArtifact
from src.ml_projects.exception import CustomException
from src.ml_projects.logger import logging

def main():
    try:
        config_manager = ConfigurationManager()
        
        # 1. Data Ingestion
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train path: {train_path}, Test path: {test_path}")

        # 2. Data Validation & Drift Detection
        # Constructing the artifact needed for validation
        ingestion_artifact = DataIngestionArtifact(
            trained_file_path=train_path,
            test_file_path=test_path
        )
        
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(
            config=data_validation_config, 
            data_ingestion_artifact=ingestion_artifact
        )
        validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data Validation completed. Status: {validation_artifact.validation_status}")

        if validation_artifact.validation_status:
            # 3. Data Transformation
            data_transformation_config = config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=train_path, test_path=test_path
            )
            logging.info("Data Transformation completed")

            # 4. Model Training
            model_trainer_config = config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            r2, mae, rmse = model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
            logging.info(f"Model Training completed. Metrics - R2: {r2}, MAE: {mae}, RMSE: {rmse}")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()