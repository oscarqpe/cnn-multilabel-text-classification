import numpy as np
import tensorflow as tf
import time
import pickle
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import config
import utils
#utils.read_labels("rcv")
import class_DatasetAgN as ds
import mlp as ml
import lemma_tokenizer as lt
from stop_words import get_stop_words

env = sys.argv[1]

mlp = ml.Mlp()
# Construct model
pred = mlp.network(mlp.x, mlp.weights, mlp.biases, mlp.dropout)

# Define loss and optimizer
cost = -tf.reduce_sum(((mlp.y * tf.log(pred + 1e-9)) + ((1-mlp.y) * tf.log(1 - pred + 1e-9)) )  , name='entropy' ) + 0.01 * (tf.nn.l2_loss(mlp.weights['wd1']) + tf.nn.l2_loss(mlp.weights['out']))
optimizer = tf.train.AdamOptimizer(learning_rate=mlp.learning_rate).minimize(cost)

#cost = tf.reduce_mean(bpmll_out_module.bp_mll(pred, mlp.y))# + 0.01 * (tf.nn.l2_loss(mlp.weights['wd1']) + tf.nn.l2_loss(mlp.weights['out']))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=mlp.y))
optimizer = tf.train.AdamOptimizer(learning_rate=mlp.learning_rate).minimize(cost)

#collator = icu.Collator.createInstance(icu.Locale('UTF-8'))
stop_words = get_stop_words('en')

#vectorizer = CountVectorizer(min_df=1, stop_words = stop_words) #bag of words
#vectorizer = TfidfVectorizer(min_df=1, stop_words = stop_words) #tfidf
#vectorizer = CountVectorizer(min_df=1, stop_words = stop_words, tokenizer = lt.LemmaTokenizer()) #bag of words stemm
#vectorizer = TfidfVectorizer(min_df=1, stop_words = stop_words, tokenizer = lt.LemmaTokenizer()) #tfidf stemm

data = None

print("Start")

path = ""
if env == "local":
	path = "/home/oscarqpe/Documentos/maestria/tesis/cnn-multilabel-text-classification/data/reuters/"
elif env == "server":
	path = "/home/citeclabs/oscarqpe/cnn-multilabel-text-classification/data/reuters/"
data = ds.Dataset(path, config.batch_size)

#data.read_rcv_vectors() # bibtext, rcv
data.all_data() # agnews
#vectorizer.fit(list(data.texts[0:120000,2]))
#print("Vocabulary: ", len(vectorizer.vocabulary_))
#data.read_rcv_vectors_test(0)
'''
data.read_text(0, 14408)#14408

print("Data Read: ", len(data.texts))
vectorizer.fit(data.texts)
print(len(vectorizer.vocabulary_))
'''
#pickle.dump(vectorizer, open("data/ag_news/vectorizer/vectorizer_tfidf_stemm.pickle", "wb"))

#vectorizer = pickle.load(open("data/bibtex/over200/vectorizer/vectorizer_bow.pickle", "rb"))
vectorizer = pickle.load(open("data/ag_news/vectorizer/vectorizer_bow.pickle", "rb"))
#vectorizer = pickle.load(open("data/rcv1-2/vectorizer/vectorizer_bow.pickle", "rb"))
#print(vectorizer.vocabulary_)

t = time.asctime()
print (t)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
config_tf = tf.ConfigProto(device_count = {'GPU': 0})

with tf.Session(config=config_tf) as sess:
	sess.run(init)
	t = time.asctime()
	print (t)
	model_saving = 2
	saver.restore(sess, "mlp_weights_agnews/model2_bow.ckpt")
	train = True
	if train == True:
		step = 1
		epoch = 1
		print("TRAINING")
		print("Epoch: " + str(epoch))
		config.training_iters = 128#640000 # 5000 * 128
		while step * config.batch_size <= config.training_iters:
			data.next_batch()
			#data.generate_batch() # vectors
			data.generate_batch() # entire text
			#print data.texts_train.shape
			#print config.batch_size
			batch_x = vectorizer.transform(data.texts_train).toarray() # vectors
			#batch_x = data.texts_train
			batch_x = batch_x.reshape(config.batch_size, config.dictionary_size)
			#print("X shape: ", batch_x.shape)
			
			#batch_y = data.labels_train # bibtex, rcv1
			batch_y = np.array(data.labels_train) # agnews
			batch_y = batch_y.reshape(config.batch_size, config.label_size)
			#print(len(batch_x), len(batch_x[0]))
			#print("Y shape: ", batch_y.shape)
			
			sess.run(optimizer, feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: mlp.dropout})

			if step % 1 == 0:
				#print Get Accuracy: "
				loss = sess.run([cost], feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1.})
				#print loss
				ou = sess.run(pred, feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1})
				#data.ids[data.start:data.end]
				[hammin_loss, one_error, coverage, ranking_loss, average_precision, subset_accuracy, accuracy, precision, recall, f_beta] = utils.get_accuracy_test(ou, batch_y)
				#print(data.ids[data.start:data.end])
				print ("Iter " + str(step * config.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss[0]))
				print ("hammin_loss: ", "{:.6f}".format(hammin_loss))
				print ("subset_accuracy: ", "{:.6f}".format(subset_accuracy))
				print ("accuracy: ", "{:.6f}".format(accuracy))
				print ("precision: ", "{:.6f}".format(precision))
				print ("recall: ", "{:.6f}".format(recall))
				print ("f_beta: ", "{:.6f}".format(f_beta))

			if data.end == data.total_texts:
				epoch += 1
				print("Epoch: " + str(epoch))
				data.shuffler()
			if step % 5000 == 0:
				save_path = saver.save(sess, "mlp_weights/model" + str(model_saving) + "_bow.ckpt")
				model_saving += 1
			step += 1
		
		data = None
		data = ds.Dataset(path, config.batch_size)
		#data.read_labels_test(0) # bibtext, RCV
		data.all_data_test() # agnews
		print("TESTING")
		step = 0
		total_test = data.total_texts
		print (total_test)
		hammin_loss_sum = 0
		subset_accuracy_sum = 0
		accuracy_sum = 0
		precision_sum = 0
		recall_sum = 0
		f_beta_sum = 0

		while step * config.batch_size < total_test:
			data.next_batch()
			#data.read_data()
			data.generate_batch()
			#data.generate_batch_test_text()
			#print data.texts_train.shape
			#print config.batch_size
			batch_x = vectorizer.transform(data.texts_train).toarray()
			#batch_x = data.texts_train
			batch_x = batch_x.reshape(config.batch_size, config.dictionary_size)
			#print("X shape: ", batch_x.shape)
			
			#batch_y = data.labels_train # bibtex, rcv1
			batch_y = np.array(data.labels_train) # agnews
			batch_y = batch_y.reshape(config.batch_size, config.label_size)

			ou = sess.run(pred, feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1})
			#print(ou)
			[hammin_loss, one_error, coverage, ranking_loss, average_precision, subset_accuracy, accuracy, precision, recall, f_beta] = utils.get_accuracy_test(ou, batch_y)
			loss = sess.run([cost], feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1.})
			#print loss
			hammin_loss_sum += hammin_loss
			subset_accuracy_sum += subset_accuracy
			accuracy_sum += accuracy
			precision_sum += precision
			recall_sum += recall
			f_beta_sum += f_beta
			#print(acc)
			print ("Iter " + str(step * config.batch_size) + ", Minibatch Loss= " + str(loss[0]))
			
			print ("hammin_loss: ", "{:.6f}".format(hammin_loss))
			print ("subset_accuracy: ", "{:.6f}".format(subset_accuracy))
			print ("accuracy: ", "{:.6f}".format(accuracy))
			print ("precision: ", "{:.6f}".format(precision))
			print ("recall: ", "{:.6f}".format(recall))
			print ("f_beta: ", "{:.6f}".format(f_beta))
			
			step += 1
		print ("PROMEDIO:")
		print ("hammin_loss_sum: ", hammin_loss_sum / step)
		print ("subset_accuracy_sum: ", subset_accuracy_sum / step)
		print ("accuracy_sum: ", accuracy_sum / step)
		print ("precision_sum: ", precision_sum / step)
		print ("recall_sum: ", recall_sum / step)
		print ("f_beta_sum: ", f_beta_sum / step)
		