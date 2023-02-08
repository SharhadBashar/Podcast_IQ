import os
import pickle
import pandas as pd

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from data import Data

class Train:
    def __init__(self, clean_filename = None, model_filename = None, data_path = None):

        self.clean_filename = clean_filename if clean_filename else 'podcasts_en_cleaned.csv'
        self.model_filename = model_filename if model_filename else 'model.pkl'
        self.data_path = data_path if data_path else '../data/'

        if not os.path.isdir('model'):
            os.makedirs('model')

        self.train(clean_filename = self.clean_filename, model_filename = self.model_filename)
        
    def get_data(self, clean_filename):
        clean_data = input('Has data been cleaned? [y/n]:')
        if (clean_data.lower() == 'n'):
            print('Data cleaning started')
            Data()
        print('Starting Training')
        df = shuffle(pd.read_csv(os.path.join(self.data_path, clean_filename)).dropna())
        X = df['name_title']
        y = df['target']
        return X, y
    
    def train(self, clean_filename , model_filename):
        X, y = self.get_data(clean_filename = clean_filename)
        clf = Pipeline([
             ('vect', CountVectorizer(stop_words = 'english')),
             ('tfidf', TfidfTransformer()),
             ('clf', LogisticRegression(C = 100.0, random_state = 1, solver = 'lbfgs', multi_class = 'ovr'))
        ])
        model = clf.fit(X, y)
        pickle.dump(model, open(os.path.join(self.data_path, model_filename), 'wb'))
        print('Trained Model saved at {}'.format(model_filename))
