import os
import sys 
from dataclasses import dataclass 

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklear.preprocessing import OneHotEncoder,StandardScaler 

from src.exception import CustomException 
from src.logger import logging 

@dataclass
class DataTransformationConfig: 
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_onject(sekf):
        try:
            numerical_columns = [
                "rank", 
                "rank_change",
                "revenue",
                "profit",
                "num_employees"
            ]
            categorical_columns = [
                "sector",
                "city",
                "state",
                "newcomer",
            ]

            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median"))
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent"))
                    ("one_hot_encoder",OneHotEncoder())
                    ("scaler",StandardScaler())
                ]
            )
            
            logging.info("Standard scaling of numerical columns is now complete")
            logging.info("Encoding of categorical columns is now complete")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns)
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)