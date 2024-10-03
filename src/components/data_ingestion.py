# This file will contain all the codes to import data from different sources like csv, excel, database, etc. and return the data in a pandas dataframe.
# it may also contain codes to split the data into train and test sets.

import os
import sys
from src.exception import custom_exception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass as dataclass
import time



# Custom exception class for more specific error handling
class DataIngestionError(Exception):
    pass

# Configuration class for managing paths
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    source_data_path: str = os.path.join("source", "data.csv")
    test_size: float = 0.2

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def import_data(self, data_path: str) -> pd.DataFrame:
        """Reads CSV data from a given file path."""
        logging.info(f"Importing data from {data_path}")
        start_time = time.time()
        try:
            data = pd.read_csv(data_path)
            logging.info(f"Successfully loaded data from {data_path} with shape {data.shape} in {time.time() - start_time:.2f} seconds.")
            return data
        except Exception as e:
            raise custom_exception(e, sys)

    def split_data(self, data: pd.DataFrame, test_size: float) -> tuple:
        """Splits data into training and testing sets."""
        logging.info("Splitting data into train and test sets")
        try:
            
            train_data, test_data = train_test_split(data, test_size=self.config.test_size, random_state=42)
            logging.info("Data splitting successful")
            return train_data, test_data
        except Exception as e:
            raise custom_exception(e, sys)

    def run(self):
        """Main function to run the data ingestion process."""
        try:
            # Step 1: Load the raw data
            data = self.import_data(self.config.source_data_path)
            
            # Step 2: Split the data into training and testing sets
            train_data, test_data = self.split_data(data, test_size=self.config.test_size)
            
            # Step 3: Save the training and testing data
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            data.to_csv(self.config.raw_data_path, index=False, header=True)
            logging.info(f"Saving training data to {self.config.train_data_path}")
            train_data.to_csv(self.config.train_data_path, index=False)
            logging.info(f"Saving testing data to {self.config.test_data_path}")
            test_data.to_csv(self.config.test_data_path, index=False)
            return self.config.train_data_path, self.config.test_data_path
            logging.info("Data ingestion completed successfully")
        except Exception as e:
            raise custom_exception(e, sys)


if __name__ == "__main__":
    # Define configuration paths
    config = DataIngestionConfig()
    
    # Create an instance of DataIngestion and run the process
    data_ingestion = DataIngestion(config)
    train_data,test_data=data_ingestion.run()

    from src.components.data_transformation import DataTransformationConfig, DataTransformation
    data_transformation = DataTransformation(DataTransformationConfig())
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data)

    from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
    model_trainer = ModelTrainer(ModelTrainerConfig())
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
