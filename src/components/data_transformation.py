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
    data = pd.read_csv(DataIngestionConfig.raw_data_path)
    target = input("Please provide the target variable (column name) for your data: ")
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self):
        target=DataTransformation.target
        data=DataTransformation.data
        try:
            num_features = data.select_dtypes(exclude="object").columns.drop(target)
            cat_features = data.select_dtypes(include="object").columns
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler(with_mean=False))
            ])
            logging.info("Numarical columns scaling completed")
            cat_pipeline = Pipeline(steps=[
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
            preprocessing_obj = self.get_data_transformer_object()
            target=DataTransformation.target
            data=DataTransformation.data
            num_features = data.select_dtypes(exclude="object").columns.drop(target)
            target_feature = target
            X_train = train_df.drop(target_feature, axis=1)
            y_train = train_df[target_feature]
            X_test = test_df.drop(target_feature, axis=1)
            y_test = test_df[target_feature]
            logging.info("Applying preprocessor object on train and test data")
            X_train_arr=preprocessing_obj.fit_transform(X_train)
            X_test_arr=preprocessing_obj.transform(X_test)

            train_arr=np.c_[X_train_arr,np.array(y_train)]
            test_arr=np.c_[X_test_arr,np.array(y_test)]

            logging.info("Saved preprocessing object")

            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return (train_arr, test_arr,self.config.preprocessor_obj_file_path)
        except Exception as e:
            raise custom_exception(e, sys)