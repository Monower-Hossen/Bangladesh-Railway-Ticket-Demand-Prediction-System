import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml_projects.exception import CustomException
from src.ml_projects.logger import logging
from src.ml_projects.utils.main_utils import save_object, read_yaml_file
from src.ml_projects.constants import SCHEMA_FILE_PATH
from src.ml_projects.config.configuration import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.data_transformation_config = config
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

    def get_data_transformer_object(self, num_cols, cat_cols):
        """
        Creates the preprocessing object for numerical and categorical columns.
        """
        try:
            # Numerical Pipeline: Handle missing values + Scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline: Handle missing values + OneHot Encoding + Scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipeline", cat_pipeline, cat_cols)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_column_name = self._schema_config.get("target_column", "").replace(' ', '_')
            num_cols = [c.replace(' ', '_') for c in self._schema_config.get("numerical_columns", [])]
            potential_cat_cols = [c.replace(' ', '_') for c in self._schema_config.get("categorical_columns", [])]

            # Filter out high-cardinality categorical columns (e.g., IDs, Names) 
            # to prevent OneHotEncoder from causing MemoryErrors.
            final_cat_cols = [col for col in potential_cat_cols if train_df[col].nunique() < 100]
            
            dropped_cols = list(set(potential_cat_cols) - set(final_cat_cols))
            if dropped_cols:
                logging.info(f"Dropped high-cardinality categorical columns from transformation: {dropped_cols}")

            actual_num_cols = [c for c in num_cols if c in train_df.columns]
            input_cols = actual_num_cols + final_cat_cols

            # Ensure all required columns exist in both dataframes to avoid KeyError
            for df_name, df in [("train", train_df), ("test", test_df)]:
                missing = [col for col in input_cols if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns in {df_name} data for transformation: {missing}")

            preprocessing_obj = self.get_data_transformer_object(num_cols=actual_num_cols, cat_cols=final_cat_cols)

            # 1. Extract Target and Input Features using explicit column lists
            target_feature_train_df = train_df[[target_column_name]]
            target_feature_test_df = test_df[[target_column_name]]

            input_feature_train_df = train_df[input_cols]
            input_feature_test_df = test_df[input_cols]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # 3. Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 4. Concatenate features with target
            input_feature_train_arr = np.array(input_feature_train_arr)
            input_feature_test_arr = np.array(input_feature_test_arr)
            
            # Target is already 2D due to double brackets [[target_column_name]]
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # 5. Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # 6. Save transformed data
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)