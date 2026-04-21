import sys
import pandas as pd
import os
from src.ml_projects.exception import CustomException
from src.ml_projects.logger import logging
from src.ml_projects.utils.main_utils import load_object, read_yaml_file
from src.ml_projects.config.configuration import ModelTrainerConfig, DataTransformationConfig
from src.ml_projects.constants import SCHEMA_FILE_PATH

class PredictPipeline:
    def __init__(self):
        try:
            self.model_config = ModelTrainerConfig()
            self.transformation_config = DataTransformationConfig()
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            
            # Pre-load artifacts to avoid redundant I/O and handle potential file missing errors early
            model_path = self.model_config.trained_model_file_path
            preprocessor_path = self.transformation_config.preprocessor_obj_file_path
            
            if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Artifacts missing. Model: {model_path}, Preprocessor: {preprocessor_path}")
                
            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, features):
        try:
            # Map schema names to match dataframe underscores
            num_cols = [c.replace(' ', '_') for c in self._schema_config.get("numerical_columns", [])]
            cat_cols = [c.replace(' ', '_') for c in self._schema_config.get("categorical_columns", [])]
            
            # Scikit-learn transformers are sensitive to column order.
            # We select the exact features in the order defined by our transformation pipeline logic.
            expected_input_columns = num_cols + cat_cols
            
            # Check for missing features before calling the transformer to avoid cryptic 500 errors
            missing_cols = [col for col in expected_input_columns if col not in features.columns]
            if missing_cols:
                raise ValueError(f"Input features missing required columns defined in schema: {missing_cols}")

            # Transform raw data and make prediction
            data_scaled = self.preprocessor.transform(features[expected_input_columns])
            preds = self.model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 Train_Name: str,
                 From: str,
                 To: str,
                 Coach: str,
                 Fare: float,
                 Journey_Date: str,
                 Departure_Time: str,
                 Issue_Date: str,
                 Issue_Time: str,
                 Group_Size: int,
                 Search_Volume: int,
                 Is_Holiday: int,
                 Payment_Method: str,
                 Ticket_No: str = "0",
                 Passenger_Name: str = "Unknown"):
        
        # Mapping inputs to class attributes
        self.Train_Name = Train_Name
        self.From = From
        self.To = To
        self.Coach = Coach
        self.Fare = Fare
        self.Journey_Date = Journey_Date
        self.Departure_Time = Departure_Time
        self.Issue_Date = Issue_Date
        self.Issue_Time = Issue_Time
        self.Group_Size = Group_Size
        self.Payment_Method = Payment_Method
        self.Search_Volume = Search_Volume
        self.Is_Holiday = Is_Holiday
        self.Ticket_No = Ticket_No
        self.Passenger_Name = Passenger_Name

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Train_Name": [self.Train_Name],
                "From": [self.From],
                "To": [self.To],
                "Coach": [self.Coach],
                "Fare": [float(self.Fare)], # Ensure numeric type for calculations
                "Journey_Date": [self.Journey_Date],
                "Departure_Time": [self.Departure_Time],
                "Issue_Date": [self.Issue_Date],
                "Issue_Time": [self.Issue_Time],
                "Group_Size": [self.Group_Size],
                "Payment_Method": [self.Payment_Method],
                "Search_Volume": [self.Search_Volume],
                "Is_Holiday": [self.Is_Holiday],
                "Ticket_No": [self.Ticket_No],
                "Passenger_Name": [self.Passenger_Name]
            }
            
            # Create DataFrame with standardized column names (underscores instead of spaces)
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)