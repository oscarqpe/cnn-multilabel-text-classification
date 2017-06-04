from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
class LemmaTokenizer(object):
	def __init__(self):
		self.stemmer = PorterStemmer()
	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in word_tokenize(doc)]