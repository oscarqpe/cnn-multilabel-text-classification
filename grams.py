import xml.etree.ElementTree as et
import numpy as np
import tensorflow as tf
import time
import sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
import config
import utils
config.max_characters = 1932
config.vocabulary_size = 4
utils.read_labels("rcv")
import class_DatasetRcv as ds
import cnn as cn
if len(sys.argv) > 1:
    env = sys.argv[1]
else:
    env = "local"

# print(labels)
print("Total labels: ", len(config.labels))
print (config.vocabulary_size)

path = ""
if env == "local":
    path = "data/reuters/"
elif env == "server":
    path = "data/reuters/"

cnn = cn.Cnn()
# Construct model
pred = cnn.network(cnn.x, cnn.weights, cnn.biases, cnn.dropout)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=cnn.y))
#cost = tf.reduce_mean(bpmll_out_module.bp_mll(pred, cnn.y))
cost = -tf.reduce_sum(((cnn.y * tf.log(pred + 1e-9)) + ((1-cnn.y) * tf.log(1 - pred + 1e-9))) , name='xentropy') + 0.01 * (tf.nn.l2_loss(cnn.weights['wd1']) + tf.nn.l2_loss(cnn.weights['out']))
#cost = -tf.reduce_sum(((mlp.y * tf.log(pred + 1e-9)) + ((1-mlp.y) * tf.log(1 - pred + 1e-9)) )  , name='entropy' ) + 0.01 * (tf.nn.l2_loss(mlp.weights['wd1']) + tf.nn.l2_loss(mlp.weights['out']))
optimizer = tf.train.AdamOptimizer(learning_rate=cnn.learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(cnn.y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#accuracy = get_accuracy(logits=pred, labels=y)
data = ds.Dataset(path, config.batch_size)
data.read_labels() # bibtex, RCV
#data.all_data() # AgNews
data.read_text(0, 23149)
stop_words = get_stop_words('en')

vectorizer1 = TfidfVectorizer(min_df=1, stop_words = stop_words, max_features=1932, ngram_range=(1, 1)) #tfidf
vectorizer2 = TfidfVectorizer(min_df=1, stop_words = stop_words, max_features=1932, ngram_range=(2, 2)) #tfidf
vectorizer3 = TfidfVectorizer(min_df=1, stop_words = stop_words, max_features=1932, ngram_range=(3, 3)) #tfidf
vectorizer4 = TfidfVectorizer(min_df=1, stop_words = stop_words, max_features=1932, ngram_range=(4, 4)) #tfidf

vectorizer1.fit(list(data.texts))
vectorizer2.fit(list(data.texts))
vectorizer3.fit(list(data.texts))
vectorizer4.fit(list(data.texts))

pickle.dump(vectorizer1, open("data/rcv1-2/vectorizer/vectorizer_1gram.pickle", "wb"))
pickle.dump(vectorizer2, open("data/rcv1-2/vectorizer/vectorizer_2gram.pickle", "wb"))
pickle.dump(vectorizer3, open("data/rcv1-2/vectorizer/vectorizer_3gram.pickle", "wb"))
pickle.dump(vectorizer4, open("data/rcv1-2/vectorizer/vectorizer_4gram.pickle", "wb"))

print(vectorizer1.vocabulary_)
print("Vocabulary: ", len(vectorizer1.vocabulary_))

#vectorizer1 = pickle.load(open("data/rcv1-2/vectorizer/vectorizer_1gram.pickle", "rb"))
#vectorizer2 = pickle.load(open("data/rcv1-2/vectorizer/vectorizer_2gram.pickle", "rb"))
#vectorizer3 = pickle.load(open("data/rcv1-2/vectorizer/vectorizer_3gram.pickle", "rb"))
#vectorizer4 = pickle.load(open("data/rcv1-2/vectorizer/vectorizer_4gram.pickle", "rb"))
#data.next_batch()
#data.generate_batch_text()

'''
for z in range(128):
	for i in range(4):
		for j in range(1932):
			print(vector1[z, i, j], end=", ")
		print("XD")
'''
#print(vector)
#utils.ngrams_picture(vector)
#utils.ngrams_test(data.texts)
#data.distribution_characters()
#data.distribution_num_labels()
#data.read_text(0, 14408)
#data.distribution_train_labels()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

c = tf.ConfigProto(device_count = {'GPU': 0})
#c = tf.ConfigProto()
#c.gpu_options.visible_device_list = "0,1"
#with tf.Session(config=c) as sess:
with tf.Session() as sess:
	sess.run(init)
	#print(sess.run("{:.5f}".format(cnn.weights['wc1'])))
	print("TRAINING")
	step = 1
	# Keep training until reach max iterations
	epoch = 1
	model_saving = 1
	print("Epoch: " + str(epoch))
	#saver.restore(sess, "cnn_weights/model_cnn_grams_0_word.ckpt")
	#data.shuffler()
	plot_x = []
	plot_y = []
	config.training_iters = 128#640000 # 5000 * 128
	data.shuffler()
	t = time.asctime()
	print (t)
	train = False
	if train == True:
		while step * config.batch_size <= config.training_iters:
			data.next_batch()
			data.generate_batch_text() # vector
			#data.generate_batch_hot() # bibtex
			#print data.texts_train.shape
			#print config.batch_size
			#batch_x = np.array(data.texts_train)
			vector1 = vectorizer1.transform(data.texts_train).toarray()
			vector2 = vectorizer2.transform(data.texts_train).toarray()
			vector3 = vectorizer3.transform(data.texts_train).toarray()
			vector4 = vectorizer4.transform(data.texts_train).toarray()
			vector1 = np.column_stack((vector1, vector2))
			vector1 = np.column_stack((vector1, vector3))
			vector1 = np.column_stack((vector1, vector4))
			batch_x = vector1
			#b = batch_x.reshape([-1, 70, 70, 1])
			#print("b shape: ", b.shape)
			#print("X shape: ", batch_x.shape)
			batch_y = np.array(data.labels_train)
			batch_y = batch_y.reshape(config.batch_size, config.label_size)
			#print(len(batch_x), len(batch_x[0]))
			#print("Y shape: ", batch_y.shape)
			#ou = sess.run(pred, feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: 1})
			#print(ou.shape)
			sess.run(optimizer, feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: cnn.dropout})

			if step % 1 == 0:
				#print "Get Accuracy: "
				loss = sess.run([cost], feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: 1.})
				#print loss
				ou = sess.run(pred, feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: 1})
				#print ou.shape
				#print batch_y.shape
				[hammin_loss, one_error, coverage, ranking_loss, average_precision, subset_accuracy, accuracy, precision, recall, f_beta] = utils.get_accuracy_test(ou, batch_y)
				#print(acc)
				plot_x.append(step * config.batch_size)
				plot_y.append(loss)
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
				#save_path = saver.save(sess, "cnn_weights_agnews/model_cnn_" + str(model_saving) + ".ckpt")
				save_path = saver.save(sess, "cnn_weights/model_cnn_grams_" + str(model_saving) + "_word.ckpt")
				model_saving += 1
			step += 1
		print(plot_x)
		print(plot_y)
		t = time.asctime()
		print (t)
		print ("TESTING")
		data = None
		data = ds.Dataset(path, config.batch_size)
		data.read_labels_test(0) # bibtext, RCV
		#data.all_data_test() # AgNEWS
		step = 1
		total_test = data.total_texts
		print (total_test)
		hammin_loss_sum = 0
		subset_accuracy_sum = 0
		accuracy_sum = 0
		precision_sum = 0
		recall_sum = 0
		f_beta_sum = 0
		data.shuffler()
		t = time.asctime()
		print (t)
		while step * config.batch_size <= total_test:
			data.next_batch()
			#data.read_data()
			data.generate_batch_test_text()
			#print data.texts_train.shape
			#print config.batch_size
			#batch_x = np.array(data.texts_train)
			#batch_x = batch_x.reshape(config.batch_size, 70 * 70)
			vector1 = vectorizer1.transform(data.texts_train).toarray()
			vector2 = vectorizer2.transform(data.texts_train).toarray()
			vector3 = vectorizer3.transform(data.texts_train).toarray()
			vector4 = vectorizer4.transform(data.texts_train).toarray()
			vector1 = np.column_stack((vector1, vector2))
			vector1 = np.column_stack((vector1, vector3))
			vector1 = np.column_stack((vector1, vector4))
			batch_x = vector1
			#print batch_x.shape
			batch_y = np.array(data.labels_train)
			batch_y = batch_y.reshape(config.batch_size, config.label_size)

			ou = sess.run(pred, feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: 1})
			#print(ou)
			[hammin_loss, one_error, coverage, ranking_loss, average_precision, subset_accuracy, accuracy, precision, recall, f_beta] = utils.get_accuracy_test(ou, batch_y)
			loss = sess.run([cost], feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: 1.})
			#print loss
			hammin_loss_sum += hammin_loss
			subset_accuracy_sum += subset_accuracy
			accuracy_sum += accuracy
			precision_sum += precision
			recall_sum += recall
			f_beta_sum += f_beta
			#print(acc)
			print ("Iter " + str(step) + ", Minibatch Loss= " + str(loss[0]))
			print ("hammin_loss: ", "{:.6f}".format(hammin_loss))
			print ("subset_accuracy: ", "{:.6f}".format(subset_accuracy))
			print ("accuracy: ", "{:.6f}".format(accuracy))
			print ("precision: ", "{:.6f}".format(precision))
			print ("recall: ", "{:.6f}".format(recall))
			print ("f_beta: ", "{:.6f}".format(f_beta))
			step += 1
		step -= 1
		print ("PROMEDIO:")
		print ("hammin_loss_sum: ", hammin_loss_sum / step)
		print ("subset_accuracy_sum: ", subset_accuracy_sum / step)
		print ("accuracy_sum: ", accuracy_sum / step)
		print ("precision_sum: ", precision_sum / step)
		print ("recall_sum: ", recall_sum / step)
		print ("f_beta_sum: ", f_beta_sum / step)
		t = time.asctime()
		print (t)
