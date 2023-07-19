import os 
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor 

from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object,evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts",'model.pkl')

class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data.")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandoomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "k-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=false),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_model(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                            models=models)

            ## to get best model score from dict 
            best_model_score = max(sorted(model_report.values()))

            ## To get the best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best mode found.")
            logging.info("Best model has been found in both training and testing datasets")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
