import os
import pickle
import pandas as pd

from data import Data

class Predict:
    def __init__(self, 
                 in_filename,
                 clean_filename = 'model_predict.csv',
                 out_filename = 'predictions.csv',
                 model_filename = 'model/model.pkl'):
        self.predict(in_filename,
                     clean_filename = 'model_predict.csv',
                     out_filename = 'predictions.csv',
                     model_filename = 'model/model.pkl')
        os.remove('model_predict.csv')
        
    def get_data(self, 
                 in_filename,
                 clean_filename = 'model_predict.csv'):
        Data(in_filename, clean_filename, train = False)
        df = pd.read_csv(clean_filename).dropna()
        original, X = df['title'], df['name']
        return original, X
    
    def predict(self, 
                in_filename,
                clean_filename = 'model_predict.csv',
                out_filename = 'predictions.csv',
                model_filename = 'model/model.pkl'):
        original, X = self.get_data(in_filename, clean_filename)
        model = pickle.load(open(model_filename, 'rb'))
        y_predict = model.predict(X)

        df = pd.DataFrame({'title': original, 'class': y_predict})
        df['class'] = df['class'].map({0: 'Entertainment', 1: 'News', 2: 'Sports'})
        df.to_csv(out_filename, index = False)
        print('Predictions saved at {}'.format(out_filename))
