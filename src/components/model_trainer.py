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
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
            }
            params={
                "Decision Tree": {'criterion': ['squared_error', 'poisson']},
                "Random Forest": {'n_estimators': [16, 32, 64, 128]},
                "Gradient Boosting": {'learning_rate': [0.1, 0.05], 'n_estimators': [32, 64, 128]},
                "XGBoost": {'learning_rate': [0.1, 0.05], 'n_estimators': [32, 64, 128]},
                "CatBoost": {'depth': [6, 8], 'learning_rate': [0.05, 0.1], 'iterations': [50, 100, 200]},
                "AdaBoost": {'learning_rate': [0.05, 0.1], 'n_estimators': [32, 64, 128]}
            }
            model_report: dict = evaluate_model(Xt=X_train, yt=y_train,Xs=X_test,ys=y_test, models=models,params=params)
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)

            if best_model_score < 0.6:
                raise custom_exception("No suitable model found", sys)
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")
            
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            
            save_object(
                file_path=self.config.trained_model_path,
                obj=best_model
            )
            prediction = best_model.predict(X_test)
            r2 = r2_score(y_test, prediction)
            logging.info(f"Model Performance (R2 Score): {r2}")
            return "R2 =",r2
        except Exception as e:
            raise custom_exception(e,sys)
        
    