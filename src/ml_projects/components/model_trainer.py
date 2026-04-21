import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.ml_projects.exception import CustomException
from src.ml_projects.logger import logging
from src.ml_projects.utils.main_utils import save_object, evaluate_models, get_regression_metrics
from src.ml_projects.config.configuration import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.model_trainer_config = config

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
            }

            # Every model in the 'models' dict MUST have a key here to avoid KeyErrors
            params = {
                "Random Forest": {"n_estimators": [32, 64, 128]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [32, 64, 128]
                },
                "Linear Regression": {}, # Empty dict if no tuning is needed
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [32, 64, 128]
                }
            }

            # Returns a dict of model_name: r2_score
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Get the best model based on R2 score (the first element of the tuple value)
            best_model_name = max(model_report, key=lambda k: model_report[k][0])
            best_model_score, best_model = max(model_report.values(), key=lambda x: x[0])

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise CustomException(f"No best model found with R2 score >= {self.model_trainer_config.expected_accuracy}")
            
            logging.info(f"Best found model: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2, mae, rmse = get_regression_metrics(y_test, predicted)
            
            return r2, mae, rmse

        except Exception as e:
            raise CustomException(e, sys)