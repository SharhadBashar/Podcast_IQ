import os
import pickle
import pandas as pd

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from data import Data

class Train:
    def __init__(self, 
                 in_filename,
                 clean_filename = 'model_train.csv',
                 model_filename = 'model/model.pkl',
                 delete_csv = False,
                 delete_excel = False
                 ):
        if not os.path.isdir('model'):
            os.makedirs('model')

        self.train(in_filename, clean_filename = 'model_train.csv', model_filename = model_filename)
        
        delete_clean_file = input('Do you want to delete \'model_train.csv\'? If so enter Y: ')
        if (delete_clean_file.lower() == 'y'):
            os.remove('model_train.csv')
            print('\'model_train.csv\' deleted')
        
    def get_data(self, in_filename, clean_filename = 'model_train.csv'):
        Data(in_filename, clean_filename = 'model_train.csv', train = Train)
        df = shuffle(pd.read_csv(clean_filename).dropna())
        X = df['name']
        y = df['class']
        return X, y
    
    def train(self, in_filename, clean_filename = 'model_train.csv', model_filename = 'model/model.pkl'):
        X, y = self.get_data(in_filename = in_filename, clean_filename = 'model_train.csv',)
        movie_clf = Pipeline([
             ('vect', CountVectorizer(stop_words = 'english')),
             ('tfidf', TfidfTransformer()),
             ('clf', LogisticRegression(C = 100.0, random_state = 1, solver = 'lbfgs', multi_class = 'ovr'))
        ])
        model = movie_clf.fit(X, y)
        pickle.dump(model, open(model_filename, 'wb'))
        print('Model saved at {}'.format(model_filename))
