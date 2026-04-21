import os
from datetime import date

# Database Connection Constants (MySQL)
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "admin123"
DB_NAME = "bangladesh_railway_tickets"
TABLE_NAME = "bangladesh_railway_tickets"

# General Project Constants
PIPELINE_NAME: str = "ml_projects"
ARTIFACT_DIR: str = "artifacts"

# File Names
MODEL_FILE_NAME = "model.pkl"
# Note: Spelled to match your pipeline's likely usage
PREPROCESSING_OBJECT_FILE_NAME = "preprocessor.pkl" 

FILE_NAME: str = "bangladesh_railway_tickets.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Target & Schema
TARGET_COLUMN = "Seats_Booked" 
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
CONFIG_FILE_PATH = os.path.join("config", "config.yaml")
PARAMS_FILE_PATH = "params.yaml"

# AWS Configuration (For S3 Model Archiving)
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

# Data Ingestion Constants

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data Validation Constants

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_STATUS_FILE_NAME: str = "status.txt"


# Data Transformation Constants

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


# Model Trainer Constants

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6  # Minimum R2 Score
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")


# Model Evaluation & Deployment Constants
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "bangladesh_railway_tickets-artifacts"
MODEL_PUSHER_S3_KEY = "model-registry"

# App Configuration
APP_HOST = "0.0.0.0"
APP_PORT = 8080