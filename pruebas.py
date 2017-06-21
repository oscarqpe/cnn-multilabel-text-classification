import csv
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words
from sklearn.metrics import pairwise_distances

class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer(r'some_regular_expression')
        self.stemmer = LancasterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(token) 
                for token in self.tok.tokenize(doc)]

stop_words = get_stop_words('en')
vectorizer = TfidfVectorizer(stop_words=stop_words, 
                             use_idf=True, 
                             smooth_idf=True)

svd_model = TruncatedSVD(n_components=500, 
                         algorithm='randomized',
                         n_iter=10, random_state=42)

svd_transformer = Pipeline([('tfidf', vectorizer), 
                            ('svd', svd_model)])
texts = []
with open('data/ag_news/train.csv', 'r') as f:
	reader = csv.reader(f)
	texts = list(reader)
	texts = np.array(texts)
texts_train = list(texts[0:120000, 2])

svd_matrix = svd_transformer.fit_transform(texts_train)

#query = texts_train[1]
#query_vector = svd_transformer.transform([texts_train[0], texts_train[1]])

print(query_vector)
print(np.shape(query_vector))
