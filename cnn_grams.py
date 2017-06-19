import numpy as np
import tensorflow as tf
import config

class Cnn:
	def __init__ (self) :
		# Parameters
		self.learning_rate = 0.001
		#self.global_step = tf.Variable(0, trainable=False)
		#self.starter_learning_rate = 0.001
		#self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 10000, 0.0096, staircase=True)
		# Network Parameters
		self.n_input = 70 * 70
		self.n_classes = config.label_size # reuters total classes
		self.dropout = 0.75 # Dropout, probability to keep units
		self.output_conv = 32
		self.hidden_size = 1024
		self.gaussian = 0.05
		self.gaussian_h = 0.05
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])
		self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
		# Store layers weight & bias
		self.weights = {
			# vocabulary_size x 7 conv, 1 input, 256 outputs
			'wc1': tf.Variable(tf.random_normal([7, 7, 1, 32], mean=0.0, stddev=self.gaussian), name="wc1"),
			# 5x5 conv, 32 inputs, 64 outputs
			'wc2': tf.Variable(tf.random_normal([7, 7, 32, 32], mean=0.0, stddev=self.gaussian), name="wc2"),
			# fully connected, 7*7*64 inputs, self.output_conv outputs
			'wd1': tf.Variable(tf.random_normal([18 * 18 * 32, self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="wd1"),
			#'wd2': tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="wd2"),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.random_normal([self.hidden_size, config.label_size], mean=0.0, stddev=self.gaussian_h), name="out")
		}
		self.biases = {
			'bc1': tf.Variable(tf.random_normal([32], mean=0.0, stddev=self.gaussian), name="bc1"),
			'bc2': tf.Variable(tf.random_normal([32], mean=0.0, stddev=self.gaussian), name="bc2"),
			'bd1': tf.Variable(tf.random_normal([self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="bd1"),
			#'bd2': tf.Variable(tf.random_normal([self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="bd2"),
			'out': tf.Variable(tf.random_normal([config.label_size], mean=0.0, stddev=self.gaussian_h), name="bout")
		}
	def conv2d(self, x, W, b, strides=1):
		# Conv2D wrapper, with bias and relu activation
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return tf.nn.relu(x)


	def maxpool2d(self, x, k=2):
		# MaxPool2D wrapper
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

	def network(self, x, weights, biases, dropout):
		# Reshape input picture
		x = tf.reshape(x, shape=[-1, 70, 70, 1])
		print(np.shape(x))
		# Convolution Layer
		conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
		print(np.shape(conv1))
		# Max Pooling (down-sampling)
		conv1 = self.maxpool2d(conv1, k=2)
		print(np.shape(conv1))
		# Convolution Layer
		conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
		print(np.shape(conv2))
		# Max Pooling (down-sampling)
		conv2 = self.maxpool2d(conv2, k=2)
		print(np.shape(conv2))
		# Fully connected layer
		# Reshape conv2 output to fit fully connected layer input
		fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
		print(np.shape(fc1))
		fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
		print(np.shape(fc1))
		fc1 = tf.nn.relu(fc1)
		# Apply Dropout
		fc1 = tf.nn.dropout(fc1, dropout)

		# Output, class prediction
		out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
		out = tf.nn.sigmoid(out)
		print(np.shape(out))
		return out