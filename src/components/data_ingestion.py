import os 
import sys 
from src.exception import CustomException
from src.logger import logging 
import pandas as pd 

from sklearn.model_selection import train_test_split 
from dataclasses import dataclass 

@dataclass 
class DataIngestionConfig: 
    train_data_path: str=os.path.join('artifact', "train.csv")
    test_data_path: str=os.path.join('artifact', "test.csv")
    raw_data_path: str=os.path.join('artifact', "data.csv")

