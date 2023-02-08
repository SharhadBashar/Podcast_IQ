import os
import pickle
import pandas as pd

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class Train:
    def __init__(self, clean_filename = None, model_filename = None, data_path = None, model_path = None):

        self.clean_filename = clean_filename if clean_filename else 'podcasts_en_cleaned.csv'
        self.model_filename = model_filename if model_filename else 'model.pkl'
        self.data_path = data_path if data_path else '../data/'
        self.model_path = model_path if model_path else '../model/'

        if not os.path.isdir('model'):
            os.makedirs('model')

        self.train(clean_filename = self.clean_filename, model_filename = self.model_filename)
        
    def get_data(self, clean_filename):
        clean_data = input('Has data been cleaned? [y/n]:')
        if (clean_data.lower() == 'n'):
            print('Data cleaning started')
            from data import Data
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
             ('clf', RandomForestClassifier()
        )])
        model = clf.fit(X, y)
        pickle.dump(model, open(os.path.join(self.model_path, model_filename), 'wb'))
        print('Trained Model saved at {}'.format(os.path.join(self.model_path, model_filename)))
