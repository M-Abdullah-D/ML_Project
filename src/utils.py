# search about this file
import os
import sys
import dill

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import custom_exception
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise custom_exception(e, sys)
    

def evaluate_model(Xt, yt, Xs, ys, models, params):
    try:
        model_report = {}
        for model_name, model in models.items():
            logging.info(f"Evaluating {model_name}")
            param_grid = params.get(model_name, {})
            
            # Use GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
            gs.fit(Xt, yt)
            
            best_model = gs.best_estimator_
            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            
            # Make predictions on the test set
            y_pred = best_model.predict(Xs)
            
            # Calculate R2 score
            test_model_score = r2_score(ys, y_pred)
            model_report[model_name] = test_model_score
            
            logging.info(f"{model_name} R2 Score: {test_model_score}")
        
        return model_report
    except Exception as e:
        raise custom_exception(e, sys)