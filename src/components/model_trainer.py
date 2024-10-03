import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path:str=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spilt training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
            }
            model_report: dict = evaluate_model(Xt=X_train, yt=y_train,Xs=X_test,ys=y_test, models=models)
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise custom_exception("No best model found", sys)
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")
            
            save_object(
                file_path=self.config.trained_model_path,
                obj=best_model
            )
            prediction = best_model.predict(X_test)
            r2 = r2_score(y_test, prediction)
            return "R2 =",r2
        except Exception as e:
            raise custom_exception(e,sys)
        
    