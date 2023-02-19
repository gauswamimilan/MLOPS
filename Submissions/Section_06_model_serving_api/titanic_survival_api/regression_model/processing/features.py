import re
import numpy as np


class ExtractLetterTransformer():
    # Extract fist letter of variable

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y, **fit_params):
        return self
    
    def extract_letters_only(self, input_string):
        if input_string is np.nan:
            return input_string
        return re.findall(r'[a-zA-Z]+', input_string)[0]

    def transform(self, df):
        df[self.column_name] = df[self.column_name].apply(self.extract_letters_only)
        return df
