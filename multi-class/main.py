import xml.etree.ElementTree as et
import numpy as np
import tensorflow as tf
import time
import sys
import pickle
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import FeatureUnion

import config
import utils

import class_Dataset as ds
import cnn as ml
from stop_words import get_stop_words
env = sys.argv[1]
mlp = ml.Cnn()
# Construct model
pred = mlp.network(mlp.x, mlp.weights, mlp.biases, mlp.dropout)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=mlp.y))
#cross_entropy_cnn = -1 * mlp.y * tf.nn.log_softmax(pred)  #method (2)
#cost = tf.reduce_sum(cross_entropy_cnn)
optimizer = tf.train.AdamOptimizer(learning_rate=mlp.learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=mlp.learning_rate, momentum=0.9).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate=mlp.learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(mlp.y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
path = ""
if env == "local":
	path = "/home/oscarqpe/Documentos/maestria/tesis/cnn-multilabel-text-classification/data/reuters/"
elif env == "server":
	path = "/home/citeclabs/oscarqpe/cnn-multilabel-text-classification/data/reuters/"
data = None
data = ds.Dataset(path, config.batch_size)
data.all_data()
#collator = icu.Collator.createInstance(icu.Locale('UTF-8'))
'''
stop_words = get_stop_words('en')

vectorizer = CountVectorizer(min_df=1, stop_words = stop_words)
vectorizer2 = CountVectorizer(min_df=1, stop_words = stop_words)
vectorizer3 = CountVectorizer(min_df=1, stop_words = stop_words)
vectorizer4 = CountVectorizer(min_df=1, stop_words = stop_words)
vectorizer5 = CountVectorizer(min_df=1, stop_words = stop_words)
vectorizer6 = CountVectorizer(min_df=1, stop_words = stop_words)

print("Start")

vectorizer.fit_transform(list(data.data[0:20000,2])).toarray()
vectorizer2.fit_transform(list(data.data[20000:40000,2])).toarray()
vectorizer3.fit_transform(list(data.data[40000:60000,2])).toarray()
vectorizer4.fit_transform(list(data.data[60000:80000,2])).toarray()
vectorizer5.fit_transform(list(data.data[80000:100000,2])).toarray()
vectorizer6.fit_transform(list(data.data[100000:120000,2])).toarray()

vectorizer.vocabulary_.update(vectorizer2.vocabulary_)
vectorizer.vocabulary_.update(vectorizer3.vocabulary_)
vectorizer.vocabulary_.update(vectorizer4.vocabulary_)
vectorizer.vocabulary_.update(vectorizer5.vocabulary_)
vectorizer.vocabulary_.update(vectorizer6.vocabulary_)

values = []
for key in vectorizer.vocabulary_:
	values.append(key)
print("Values: ", len(values))
print(values[0])
vectorizer = None
vectorizer2 = None
vectorizer3 = None
vectorizer4 = None
vectorizer5 = None
vectorizer6 = None
vocvoc = dict((values[i], i) for i in range(0, len(values)))
'''
#realvec = CountVectorizer(min_df=1, stop_words = stop_words, vocabulary = vocvoc)
#pickle.dump(realvec, open("vectors/vectorizer-ag-news.pickle", "wb"))
#realvec = pickle.load(open("vectors/vectorizer-ag-news.pickle", "rb"))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
config_tf = tf.ConfigProto(device_count = {'GPU': 0})
config.training_iters = 720000
with tf.Session(config=config_tf) as sess:
	sess.run(init)
	t = time.asctime()
	print (t)
	print("TRAINING")
	step = 1
	# Keep training until reach max iterations
	#data = ds.Dataset("/home/rcoronado/project/cnn-multilabel-text-classification/data/reuters/", config.batch_size)
	epoch = 1
	model_saving = 0
	print("Epoch: " + str(epoch))
	saver.restore(sess, "models/model_cnn_1.ckpt")
	#data.total_characters()
	
	while step * config.batch_size < config.training_iters:
		data.next_batch()
		#data.generate_batch() # bag of words
		data.generate_batch_hot()

		#batch_x = realvec.transform(data.texts_train).toarray() # bag of words
		batch_x = np.array(data.texts_train) # one hot
		#batch_x = batch_x.reshape(config.batch_size, config.dictionary_size) # bag of words
		batch_x = batch_x.reshape(config.batch_size, config.vocabulary_size * config.max_characters) # one hot
		
		batch_y = np.array(data.labels_train)
		batch_y = batch_y.reshape(config.batch_size, config.label_size)
		#print(len(batch_x), len(batch_x[0]))
		#print("Y shape: ", batch_y.shape)
		sess.run(optimizer, feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: mlp.dropout})

		if step % 1 == 0:
			#print "Get Accuracy: "
			loss, acc = sess.run([cost, accuracy], feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1.})
			ou = sess.run(pred, feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1})
			ou = tf.nn.softmax(ou)
			ou = np.array(ou.eval())
			
			for i in range(0, len(ou)):
				print("[", end=" ")
				for j in range(0, len(ou[i])):
					print("{:.6f}".format(ou[i][j]), end = ", ")
				print("]")
			print ("Iter " + str(step * config.batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
		if data.end == data.total_texts:
			epoch += 1
			print("Epoch: " + str(epoch))
			data.shuffler()
		if step % 2000 == 0:
			print("Save weights!!!")
			save_path = saver.save(sess, "models/model_cnn_" + str(model_saving) + ".ckpt")
			model_saving += 1
		step += 1
		