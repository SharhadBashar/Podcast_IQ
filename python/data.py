import os
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
        data_path = None,
        train = True, 
        to_drop = ['Unnamed: 0', 'ContentUrl', 'Country', 'Language']):

        in_filename = in_filename if in_filename else 'Podcasts.csv'
        en_filename = en_filename if en_filename else 'Podcasts_en.csv'
        clean_filename = clean_filename if clean_filename else 'podcasts_en_cleaned.csv'
        data_path = data_path if data_path else '../data/'

        self.get_en(os.path.join(data_path, in_filename), 
            save_file = True, 
            save_path = os.path.join(data_path, en_filename))

        df = read_csv(os.path.join(data_path, en_filename), 
              to_drop = to_drop)

        headers = ['name', 'class'] if train else ['name']
        if (in_filename.split('.')[-1] == 'xlsx'):
            df = pd.DataFrame(pd.read_excel(in_filename, names = headers))
        else:
            df = pd.read_csv(in_filename, names = headers)
        df = self.clean_data(df, translate = False, stem = False, lemm = True, train = train)
        self.save_df(df, filename = clean_filename)

    def read_csv(self, filename, to_drop = []):
        df = pd.read_csv(filename)
        df = df.drop(to_drop, axis = 1)
        return df

    def get_en(self, filename, save_file = False, save_path = ''):
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

    def augment_cols(df):
        df['name_title'] = df['podcastname'].astype(str) + ' ' + df['Title'].astype(str)
        df['target'] = pd.factorize(df['stylename'])[0]
        return df 

    def clean_data(self, df, translate = False, stem = False, lemm = True, train = True):
        #translator = Translator()
        stop = stopwords.words('english')
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        df['title'] = df['name']
        df['name'] = df['name'].str.replace('[^A-Za-z0-9 ]+', ' ')
        if translate: 
            df['name'] = df['name'].apply(lambda x: translator.translate(x, dest = 'en'))
        df['name'] = df['name'].apply(lambda x: clean(x))
        df['name'] = df['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        if stem:
            df['name'] = df['name'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
        if lemm:
            df['name'] = df['name'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

        if train:
            df['class'] = df['class'].map({'Entertainment': 0, 'News': 1, 'Sports': 2})
        df = df.dropna()
        return df

    def save_df(self, df, filename):
        df.to_csv(filename, index = None)
        print('Data has been cleaned and saved at {}'.format(filename))
