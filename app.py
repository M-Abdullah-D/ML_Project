import pickle
from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from src.pipline.predict_pipline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

## Route to the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("race_ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=int(request.form.get("reading_score")),
                writing_score=int(request.form.get("writing_score"))
            )
            pred_df=data.get_data_dataframe()
            print(pred_df)

            pipeline = PredictPipeline()
            results=pipeline.predict(pred_df)
            return render_template('home.html',results=results[0])
        except Exception as e:
            # Handle any errors that occur during the prediction process
            return render_template('home.html', results="Error in prediction process.")
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)