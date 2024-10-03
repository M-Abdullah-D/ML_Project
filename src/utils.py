# search about this file
import os
import sys
import dill

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from src.exception import custom_exception

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise custom_exception(e, sys)
    

def evaluate_model(Xt, yt, Xs, ys, models):
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(Xt, yt)
            y_pred = model.predict(Xs)
            model_report[model_name] = r2_score(ys, y_pred)
        return model_report
    except Exception as e:
        raise custom_exception(e, sys)