import os
import time
import pickle
import pandas as pd
from cleantext import clean

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

class Data:
    def __init__(self, in_filename = None, 
        en_filename = None, 
        clean_filename = None,
        map_filename = None, 
        data_path = None,
        train = True, 
        to_drop = None):

        self.in_filename = in_filename if in_filename else 'Podcasts.csv'
        self.en_filename = en_filename if en_filename else 'Podcasts_en.csv'
        self.clean_filename = clean_filename if clean_filename else 'podcasts_en_cleaned.csv'
        self.map_filename = map_filename if map_filename else 'cat_mapping_en.pkl'
        self.data_path = data_path if data_path else '../data/'
        self.to_drop = to_drop if to_drop else ['Unnamed: 0', 'ContentUrl', 'Country', 'Language']

        self.get_en(os.path.join(self.data_path, self.in_filename), 
            save_file = True, 
            save_path = os.path.join(self.data_path, self.en_filename))
        start = time.time()
        df = self.read_csv(os.path.join(self.data_path, self.en_filename), to_drop = to_drop)
        df = drop_cat(df)
        df = augment_cols(df, map_filename = self.map_filename)
        df = clean_data(df)
        self.save_df(df, os.path.join(data_path, self.clean_filename))
        print('Finished data cleaning in,' time.time() - start)

    def read_csv(self, filename, to_drop = []):
        df = pd.read_csv(filename)
        df = df.drop(to_drop, axis = 1)
        return df

    def get_en(self, filename, save_file = False, save_path = None):
        df = pd.DataFrame(filename)
        df[df['Language'] == 'en']
        if save_file:
            df.to_csv(save_path)

    def drop_cat(self, df, n = 10):
        cat = df['stylename'].value_counts()
        cat_to_drop = list(cat[cat < n].index)
        cat_to_drop.append('Miscellaneous')
        df = df[~df['stylename'].isin(cat_to_drop)]
        return df

    def augment_cols(df, map_filename):
        df['name_title'] = df['podcastname'].astype(str) + ' ' + df['Title'].astype(str)
        df['target'], map = pd.factorize(df['stylename'])
        map = dict(zip(range(len(map)), map))
        with open(os.path.join(self.data_path, map_filename), 'wb') as file:
            pickle.dump(map, file, protocol = pickle.HIGHEST_PROTOCOL)
        return df 

    def clean_data(df, translate = False):
        lemmatizer = WordNetLemmatizer()
        stop = stopwords.words('english')
        # df['name_title'] = df['name_title'].apply(lambda x: re.sub(date, ' ', x))
        # if translate: 
        #     df['name'] = df['name'].apply(lambda x: translator.translate(x, dest = 'en'))
        df['name_title'] = df['name_title'].str.replace('[^A-Za-z0-9 ]+', ' ')
        df['name_title'] = df['name_title'].apply(lambda x: clean(x, clean_all = False, 
                                                                extra_spaces = True,                                                   
                                                                stemming = False,
                                                                stopwords = True,
                                                                lowercase = True,
                                                                numbers = True,
                                                                punct = True))

        df['name'] = df['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df['name_title'] = df['name_title'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
        df = df.dropna()
        return df

    def save_df(self, df, filename):
        df.to_csv(filename, index = None)
        print('Data has been cleaned and saved at {}'.format(filename))
