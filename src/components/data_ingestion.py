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