import pandas as pd
# from googletrans import Translator
from cleantext import clean

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

class Data:
    def __init__(self, 
                 in_filename, 
                 clean_filename,
                 train = True, 
                 translate = False, 
                 stem = False, 
                 lemm = True):
        headers = ['name', 'class'] if train else ['name']
        if (in_filename.split('.')[-1] == 'xlsx'):
            df = pd.DataFrame(pd.read_excel(in_filename, names = headers))
        else:
            df = pd.read_csv(in_filename, names = headers)
        df = self.clean_data(df, translate = False, stem = False, lemm = True, train = train)
        self.save_df(df, filename = clean_filename)

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
