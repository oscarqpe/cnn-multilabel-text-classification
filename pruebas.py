'''
import tensorflow as tf
import numpy as np
config_tf = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
bpmll_out_module = tf.load_op_library('custom/bp_mll.so')
with tf.Session(config=config_tf) as sess:
	input = tf.random_normal([86016], mean=0.0, stddev=0.9)
	labels = tf.constant(np.random.randint(2, size=86016))
	input = tf.cast(input, tf.float32)
	labels = tf.cast(labels, tf.float32)
	input = tf.reshape(input, [128, 672])
	labels = tf.reshape(labels, [128, 672])
	print (bpmll_out_module.bp_mll(input, labels).eval())
	'''
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
vect = TfidfVectorizer(tokenizer=LemmaTokenizer())
corpus = ["Hello World father mother book vocabulary tokenizer", "stemming stemmer lemmatization tokenizer"]
vect.fit(corpus)
print(vect.vocabulary_)

x = vect.transform(corpus).toarray()
print(x)