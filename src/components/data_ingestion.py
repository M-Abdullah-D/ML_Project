# This file will contain all the codes to import data from different sources like csv, excel, database, etc. and return the data in a pandas dataframe.
# it may also contain codes to split the data into train and test sets.

import os
import sys
from src.exception import custom_exception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass as dc

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def import_data(self, data_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(data_path)
            return data
        except Exception as e:
            logging.error(f"Error in importing data from {data_path}")
            raise custom_exception.DataIngestionError(f"Error in importing data from {data_path}")

    def split_data(self, data: pd.DataFrame, target: str, test_size: float) -> pd.DataFrame:
        try:
            X = data.drop(target, axis=1)
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data")
            raise custom_exception.DataIngestionError(f"Error in splitting data")

    def run(self):
        try:
            data = self.import_data(self.config.raw_data_path)
            X_train, X_test, y_train, y_test = self.split_data(data, target="target", test_size=0.2)
            X_train.to_csv(self.config.train_data_path, index=False)
            X_test.to_csv(self.config.test_data_path, index=False)
            logging.info("Data Ingestion completed successfully")
        except Exception as e:
            logging.error(f"Error in Data Ingestion")
            raise custom_exception.DataIngestionError(f"Error in Data Ingestion")