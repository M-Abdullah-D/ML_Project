import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object
from src.components.data_ingestion import DataIngestionConfig, DataIngestion

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self, config: DataTransformationConfig,target:str):
        self.config = config
        self.target=target

    def get_data_transformer_object(self,data:pd.DataFrame):
        try:
            num_features = data.select_dtypes(exclude="object").columns.drop(self.target)
            cat_features = data.select_dtypes(include="object").columns
            
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler(with_mean=False))
            ])
            logging.info("Numarical columns scaling completed")
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('onehot', OneHotEncoder()),
                ('std_scaler', StandardScaler(with_mean=False))
            ])
            logging.info("Categorical columns encoding and scaling completed")

            preprocessor = ColumnTransformer([
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ])
            return preprocessor
        except Exception as e:
            raise custom_exception(e, sys)
        
    def initiate_data_transformation(self,train_data, test_data):
        try:
            train_df=pd.read_csv(train_data)
            test_df=pd.read_csv(test_data)
            logging.info("Reading train and test data completed")

            logging.info("Obtaining preprocessor object")
            preprocessor = self.get_data_transformer_object(train_df)
            X_train = train_df.drop(self.target, axis=1)
            y_train = train_df[self.target]
            X_test = test_df.drop(self.target, axis=1)
            y_test = test_df[self.target]
            logging.info("Applying preprocessor object on train and test data")
            X_train_arr=preprocessor.fit_transform(X_train)
            X_test_arr=preprocessor.transform(X_test)

            train_arr=np.c_[X_train_arr,np.array(y_train)]
            test_arr=np.c_[X_test_arr,np.array(y_test)]

            logging.info("Saved preprocessing object")

            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)

            return (train_arr, test_arr,self.config.preprocessor_obj_file_path)
        except Exception as e:
            raise custom_exception(e, sys)