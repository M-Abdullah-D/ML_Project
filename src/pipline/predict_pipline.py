import sys
import pandas as pd
from src.exception import custom_exception
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        """
        Predicts the output based on the input features.
        Loads the preprocessor and model objects, scales the data, and performs prediction.
        """
        try:
            # Paths to the saved model and preprocessor
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            
            # Load model and preprocessor
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            
            # Preprocess input features
            data_scaled = preprocessor.transform(features)
            
            # Perform prediction
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise custom_exception(e, sys)
class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, 
                 test_preparation_course, reading_score, writing_score):
        # Initialize input data attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_dataframe(self):
        """
        Converts the input data into a pandas DataFrame for prediction.
        """
        try:
            data = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise custom_exception(e,sys)