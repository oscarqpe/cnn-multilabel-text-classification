import numpy as np
import tensorflow as tf
import config

class Cnn:
	def __init__ (self) :
		# Parameters
		self.learning_rate = 0.003
		#self.global_step = tf.Variable(0, trainable=False)
		#self.starter_learning_rate = 0.001
		#self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 10000, 0.0096, staircase=True)
		self.training_iters = 100
		self.batch_size = 128
		self.display_step = 10
		# Network Parameters
		self.n_input = config.vocabulary_size * config.max_characters # vocabulary * text character size (img shape: l * 924)
		self.n_classes = config.label_size # reuters total classes
		self.dropout = 0.5 # Dropout, probability to keep units
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])
		self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
		# Store layers weight & bias
		self.weights = {
			# vocabulary_size x 7 conv, 1 input, 128 outputs
			'wc1': tf.Variable(tf.random_normal([7, config.vocabulary_size, 128], mean=0.0, stddev=0.02)),
			# 5x5 conv, 32 inputs, 64 outputs
			'wc2': tf.Variable(tf.random_normal([7, 128, 128], mean=0.0, stddev=0.02)),
			# fully connected, 7*7*64 inputs, 1024 outputs
			'wd1': tf.Variable(tf.random_normal([110 * 128, 1024], mean=0.0, stddev=0.02)),
			'wd2': tf.Variable(tf.random_normal([1024, 1024], mean=0.0, stddev=0.02)),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.random_normal([1024, config.label_size], mean=0.0, stddev=0.02))
		}
		self.biases = {
			'bc1': tf.Variable(tf.random_normal([128], mean=0.0, stddev=0.02)),
			'bc2': tf.Variable(tf.random_normal([128], mean=0.0, stddev=0.02)),
			'bd1': tf.Variable(tf.random_normal([1024], mean=0.0, stddev=0.02)),
			'bd2': tf.Variable(tf.random_normal([1024], mean=0.0, stddev=0.02)),
			'out': tf.Variable(tf.random_normal([config.label_size], mean=0.0, stddev=0.02))
		}
	# Create some wrappers for simplicity
	def convolution_1d(self, x, filters, bias, strides=1):
		# Conv1D wrapper, with bias and relu activation
		# x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.conv1d(x, filters, strides, padding='VALID')
		x = tf.nn.bias_add(x, bias)
		return tf.nn.relu(x)

	def max_pool_1d(self, x, lenght, output, k=2):
		# MaxPool2D wrapper
		x = tf.reshape(x, shape=[-1, 1, lenght, output])

		x = tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1, 1, k, 1], padding='VALID')
		x = tf.reshape(x, shape=[-1, int(lenght / k), output])
		return x

	def network(self, x, weights, biases, dropout):
		#print(x)
		#da = tf.reshape(x, [-1])
		#print(da)
		input_data = tf.reshape(x, shape=[config.batch_size, config.max_characters, config.vocabulary_size])
		#print input_data
		#input_data = tf.Variable(da.astype(np.float32))
		
		conv1 = self.convolution_1d(input_data, weights['wc1'], biases['bc1'], strides=1)
		#print conv1
		conv1 = self.max_pool_1d(conv1, config.max_characters - 7 + 1, 128, 3)
		#print conv1

		conv2 = self.convolution_1d(conv1, weights['wc2'], biases['bc2'], strides=1)
		#print conv2
		conv2 = self.max_pool_1d(conv2, 336 - 7 + 1, 128, 3)
		#print conv2

		# Fully connected layer
		# Reshape conv2 output to fit fully connected layer input
		#print(weights['wd1'])
		fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
		#print fc1
		fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
		fc1 = tf.nn.softmax(fc1)
		#print(fc1)
		fc1 = tf.nn.dropout(fc1, dropout)
		#print(fc1)
		fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
		#print fc2
		fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
		fc2 = tf.nn.softmax(fc2)
		#print(fc2)
		fc2 = tf.nn.dropout(fc2, dropout)
		# Output, class prediction
		
		out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
		#print out
		out = tf.nn.softmax(out)
		return out