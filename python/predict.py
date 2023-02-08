import os
import pickle
import pandas as pd
from cleantext import clean

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer

class Predict:
    def __init__(self, cat_map_path = None, model_path = None):
        self.cat_map_path = cat_map_path if cat_map_path else 'cat_mapping_en.pkl'
        self.model_path = model_path if model_path else 'model.pkl'
        self.data_path = data_path if data_path else '../data/'

        self.category_dict = pickle.load(open(cat_map_path, 'rb'))
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
        return self.category_dict[self.model.predict([inp])[0]]