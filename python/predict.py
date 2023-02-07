import os
import pickle
import pandas as pd
from cleantext import clean

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer

class Predict:
    def __init__(self, model_path = '../model/model.pkl'):
        self.model = pickle.load(open(model_path, 'rb'))
        
    def clean_data(self, name, title):
        data = name + ' ' + title
        data = data.replace('[^A-Za-z0-9 ]+', ' ')
        data = clean(data, clean_all = False, 
                           extra_spaces = True, 
                           stemming = False,
                           stopwords = True, 
                           lowercase = True, 
                           numbers = True, 
                           punct = True
                    )
        data = ' '.join([lemmatizer.lemmatize(word) for word in data.split()])
        return data
    
    def predict(self, inp):
        return category_dict[self.model.predict([inp])[0]]