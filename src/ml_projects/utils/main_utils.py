import os
import sys
import numpy as np
import pandas as pd
import dill
import yaml
from sqlalchemy import create_engine
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.ml_projects.exception import CustomException
from src.ml_projects.logger import logging
from src.ml_projects.constants import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

# Database Credentials - fetching from environment or using defaults
host = os.getenv("host", DB_HOST)
user = os.getenv("user", DB_USER)
password = os.getenv("password", DB_PASSWORD)
db = os.getenv('db', DB_NAME)

def read_yaml_file(file_path: str) -> dict:
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as yaml_file:
                return yaml.safe_load(yaml_file)
        logging.warning(f"YAML file not found at {file_path}. Returning empty dictionary.")
        return {}
    except Exception as e:
        raise CustomException(e, sys)

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)

def read_sql_data(table_name: str):
    """Reads railway data from a MySQL database."""
    logging.info("Reading SQL database started")
    try:
        if not table_name:
            raise ValueError("Table name must be provided for SQL read operation.")
            
        # Construct the connection string for SQLAlchemy
        connection_string = f"mysql+pymysql://{user}:{password}@{host}/{db}"
        engine = create_engine(connection_string)
        
        logging.info(f"Connection Established via SQLAlchemy: {host}/{db}")
        df = pd.read_sql_query(f'SELECT * FROM {table_name}', engine)
        logging.info(f"Data successfully read from SQL. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.warning(f"Could not read from SQL database: {e}")
        return None

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Performs hyperparameter tuning and returns a report containing R2 scores and fitted models.
    """
    try:
        report = {}
        for model_name, model in models.items():
            para = param.get(model_name, {})

            logging.info(f"Hyperparameter tuning started for: {model_name}")
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1, refit=True)
            gs.fit(X_train, y_train)

            # gs.best_estimator_ is already fitted on the full X_train/y_train
            y_test_pred = gs.best_estimator_.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            # Return both score and the fitted best estimator
            report[model_name] = (test_model_score, gs.best_estimator_)
            logging.info(f"Model: {model_name} | Best R2 Score: {test_model_score}")

        if not report:
            raise Exception("No models were successfully evaluated.")
            
        return report

    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    """Saves a python object (model/preprocessor) as a dill/pickle file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Loads a python object from a file."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def get_regression_metrics(y_true, y_pred):
    """Helper to get all regression metrics at once."""
    try:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return r2, mae, rmse
    except Exception as e:
        raise CustomException(e, sys)