import numpy as np
import tensorflow as tf
import time
import pickle
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import config
import utils
utils.read_labels("bibtex")
import class_DatasetRcv as ds

import encoder as au
import mlpau as ml
from stop_words import get_stop_words
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
env = sys.argv[1]

bpmll_out_module = tf.load_op_library('custom/bp_mll.so')
bpmll_grad_out_module = tf.load_op_library('custom/bp_mll_grad.so')

@ops.RegisterGradient("BpMll")
def _bp_mll_grad(op, grad):
	return bpmll_grad_out_module.bp_mll_grad(grad=grad, logits=op.inputs[0], labels=op.inputs[1])

mlp = ml.Mlp()
# Construct model
pred = mlp.network(mlp.x, mlp.weights, mlp.biases, mlp.dropout)

# Define loss and optimizer
cost = -tf.reduce_sum( (  (mlp.y * tf.log(pred + 1e-9)) + ((1-mlp.y) * tf.log(1 - pred + 1e-9)) )  , name='xentropy' ) + 0.01 * (tf.nn.l2_loss(mlp.weights['wd1']) + tf.nn.l2_loss(mlp.weights['out']))
#cost = tf.reduce_mean(bpmll_out_module.bp_mll(pred, mlp.y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=mlp.y))
optimizer = tf.train.AdamOptimizer(learning_rate=mlp.learning_rate).minimize(cost)


encoder = au.Encoder()
# Construct model
encoder_op = encoder.encoder(encoder.x)
decoder_op = encoder.decoder(encoder_op)
# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = encoder.x

# Define loss and optimizer, minimize the squared error
au_cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
au_optimizer = tf.train.RMSPropOptimizer(learning_rate=encoder.learning_rate).minimize(au_cost)

#collator = icu.Collator.createInstance(icu.Locale('UTF-8'))
#stop_words = get_stop_words('en')
#vectorizer = CountVectorizer(min_df=1, stop_words = stop_words)
#vectorizer = TfidfVectorizer(min_df=1, stop_words = stop_words)
#analyze = vectorizer.build_analyzer()

data = None

config.training_iters = 640001
print("Start")

path = ""
if env == "local":
	path = "/home/oscarqpe/Documentos/maestria/tesis/cnn-multilabel-text-classification/data/reuters/"
elif env == "server":
	path = "/home/citeclabs/oscarqpe/cnn-multilabel-text-classification/data/reuters/"
data = ds.Dataset(path, config.batch_size)

#data.read_rcv_vectors_test(0)
#vectorizer.fit_transform(data.texts_train).toarray()
#pickle.dump(vectorizer, open("mlp_weights/vectorizer.pickle", "wb"))
#vectorizer = pickle.load(open("mlp_weights/vectorizer.pickle", "rb"))
#print(vectorizer.vocabulary_)
#print ("Dic Size: ", len(vectorizer.vocabulary_))
t = time.asctime()
print (t)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
config_tf = tf.ConfigProto(device_count = {'GPU': 0})
#config=config_tf
with tf.Session(config=config_tf) as sess:
	sess.run(init)
	t = time.asctime()
	print (t)
	model_saving = 5
	saver.restore(sess, "ae_weights/model4.ckpt")
	auto_encoder = True
	train = False
	data.read_rcv_vectors()
	print("Data Read: ", len(data.ids))
	
	if auto_encoder == True:
		# Training cycle
		training_epochs = 4000
		step = 1
		epoch = 1
		print("TRAINING AUTO ENCODER")
		print("Epoch: " + str(epoch))
		while step * config.batch_size <= training_epochs * config.batch_size:
			data.next_batch()
			data.generate_batch()
			batch_x = data.texts_train
			batch_x = batch_x.reshape(config.batch_size, config.dictionary_size)

			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([au_optimizer, au_cost], feed_dict={encoder.x: batch_x})
			# Display logs per epoch step
			if step % 20 == 0:
				print("Iter: ", str(step * config.batch_size), "cost=", "{:.9f}".format(c))
			if data.end == data.total_texts:
				epoch += 1
				print("Epoch: " + str(epoch))
				data.shuffler()
			if step % 2000 == 0:
				print("Saving weights!!!")
				save_path = saver.save(sess, "ae_weights/model" + str(model_saving) + ".ckpt")
				model_saving += 1
			step += 1
		print("Optimization Finished!")
	if train == True:
		step = 1
		epoch = 1
		training_iters = 2000
		print("TRAINING MLP")
		print("Epoch: " + str(epoch))
		while step * config.batch_size <= training_iters * config.batch_size:
			data.next_batch()
			data.generate_batch()
			#print data.texts_train.shape
			#print config.batch_size
			#batch_x = vectorizer.transform(data.texts_train).toarray()
			batch_x = data.texts_train
			batch_x = batch_x.reshape(config.batch_size, config.dictionary_size)
			'''
			pred = sess.run(y_pred, feed_dict={encoder.x: batch_x})
			print(batch_x)
			for i in range(len(batch_x[0])):
				print(batch_x[0][i], end = ", ")
			print("->")
			for i in range(len(pred[0])):
				print(pred[0][i], end = ", ")
			print("->")
			'''
			batch_x = sess.run(encoder_op, feed_dict={encoder.x: batch_x})
			batch_x = batch_x.reshape(config.batch_size, 2048)
			#print("X shape: ", batch_x.shape)
			
			batch_y = data.labels_train
			
			batch_y = batch_y.reshape(config.batch_size, config.label_size)
			#print(len(batch_x), len(batch_x[0]))
			#print("Y shape: ", batch_y.shape)
			
			sess.run(optimizer, feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: mlp.dropout})

			if step % 10 == 0:
				#print "Get Accuracy: "
				loss = sess.run([cost], feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1.})
				#print loss
				ou = sess.run(pred, feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1})
				#print(ou[0])
				#print(batch_y[0])
				#print ou.shape
				#print batch_y.shape
				acc = utils.get_accuracy(data.ids[data.start:data.end], ou, batch_y)
				#print(data.ids[data.start:data.end])
				print ("Iter " + str(step * config.batch_size) + ", Minibatch Loss= " + \
					"{:.6f}".format(loss[0]) + ", Training Accuracy= " + \
					str(acc) + "/" + str(config.batch_size) + " correctos, " + "{:.5f}".format((acc * 100 / config.batch_size)) + " %")
			if data.end == data.total_texts:
				epoch += 1
				print("Epoch: " + str(epoch))
				data.shuffler()
			if step % 2000 == 0:
				print("Saving weights!!!")
				save_path = saver.save(sess, "ae_weights/model" + str(model_saving) + ".ckpt")
				model_saving += 1
			step += 1
		data = None
		data = ds.Dataset(path, config.batch_size)
		data.read_rcv_vectors_test(0)
		print("TESTING MLP")
		step = 0
		total_test = data.total_test
		print (total_test)
		true_positive = 0
		hammin_loss_sum = 0
		one_error_sum = 0
		coverage_sum = 0
		ranking_loss_sum = 0
		average_precision_sum = 0
		subset_accuracy_sum = 0
		accuracy_sum = 0
		precision_sum = 0
		recall_sum = 0
		while step * config.batch_size < total_test:
			data.next_test()
			#data.read_data()
			data.generate_batch_test()
			#print data.texts_train.shape
			#print config.batch_size
			batch_x = data.texts_train
			batch_x = batch_x.reshape(config.batch_size, config.dictionary_size)
			#print("X shape: ", batch_x.shape)
			
			batch_y = data.labels_train
			
			batch_y = batch_y.reshape(config.batch_size, config.label_size)

			ou = sess.run(pred, feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1})
			#print(ou)
			[acc, hammin_loss, one_error, coverage, ranking_loss, average_precision, subset_accuracy, accuracy, precision, recall, f_beta] = utils.get_accuracy_test(ou, batch_y)
			loss = sess.run([cost], feed_dict={mlp.x: batch_x, mlp.y: batch_y, mlp.keep_prob: 1.})
			#print loss
			true_positive += acc
			hammin_loss_sum += hammin_loss
			one_error_sum += one_error
			coverage_sum += coverage
			ranking_loss_sum += ranking_loss
			average_precision_sum += average_precision
			subset_accuracy_sum += subset_accuracy
			accuracy_sum += accuracy
			precision_sum += precision
			recall_sum += recall
			#print(acc)
			print ("Iter " + str(step * config.batch_size) + ", Minibatch Loss= " + \
				str(loss[0]) + ", Training Accuracy= " + \
				str(acc) + "/" + str(config.batch_size) + " correctos, " + str((acc * 100 / config.batch_size)) + " %")
			step += 1
		print ("Total Accuracy: " + str (true_positive) + "/" + str(total_test) + ", " + str((true_positive * 100 / total_test)) + " %")
		print ("Hammin Loss: ", hammin_loss_sum / total_test)
		print ("One Error: ", one_error_sum / total_test)
		print ("Coverage: ", coverage_sum / total_test)
		print ("Ranking Loss: ", ranking_loss_sum / total_test)
		print ("Average Precision: ", average_precision_sum / total_test)
		print ("Subset Accuracy: ", subset_accuracy_sum / total_test)
		print ("Accuracy: ", accuracy_sum / total_test)
		print ("Precision: ", precision_sum / total_test)
		print ("Recall: ", recall_sum / total_test)

