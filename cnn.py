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
		self.n_input = config.vocabulary_size * config.max_characters # vocabulary * text character size (img shape: l * 924)
		self.n_classes = config.label_size # reuters total classes
		self.dropout = 0.5 # Dropout, probability to keep units
		self.output_conv = 256
		self.hidden_size = 2048
		self.gaussian = 0.05
		self.gaussian_h = 0.02
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])
		self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
		# Store layers weight & bias
		self.weights = {
			# vocabulary_size x 7 conv, 1 input, 256 outputs
			'wc1': tf.Variable(tf.random_normal([7, config.vocabulary_size, self.output_conv], mean=0.0, stddev=self.gaussian), name="wc1"),
			# 5x5 conv, 32 inputs, 64 outputs
			'wc2': tf.Variable(tf.random_normal([7, self.output_conv, self.output_conv], mean=0.0, stddev=self.gaussian), name="wc2"),
			'wc3': tf.Variable(tf.random_normal([3, self.output_conv, self.output_conv], mean=0.0, stddev=self.gaussian), name="wc3"),
			'wc4': tf.Variable(tf.random_normal([3, self.output_conv, self.output_conv], mean=0.0, stddev=self.gaussian), name="wc4"),
			'wc5': tf.Variable(tf.random_normal([3, self.output_conv, self.output_conv], mean=0.0, stddev=self.gaussian), name="wc5"),
			'wc6': tf.Variable(tf.random_normal([3, self.output_conv, self.output_conv], mean=0.0, stddev=self.gaussian), name="wc6"),
			# fully connected, 7*7*64 inputs, self.output_conv outputs
			'wd1': tf.Variable(tf.random_normal([34 * self.output_conv, self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="wd1"),
			#'wd2': tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="wd2"),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.random_normal([self.hidden_size, config.label_size], mean=0.0, stddev=self.gaussian_h), name="out")
		}
		self.biases = {
			'bc1': tf.Variable(tf.random_normal([self.output_conv], mean=0.0, stddev=self.gaussian), name="bc1"),
			'bc2': tf.Variable(tf.random_normal([self.output_conv], mean=0.0, stddev=self.gaussian), name="bc2"),
			'bc3': tf.Variable(tf.random_normal([self.output_conv], mean=0.0, stddev=self.gaussian), name="bc3"),
			'bc4': tf.Variable(tf.random_normal([self.output_conv], mean=0.0, stddev=self.gaussian), name="bc4"),
			'bc5': tf.Variable(tf.random_normal([self.output_conv], mean=0.0, stddev=self.gaussian), name="bc5"),
			'bc6': tf.Variable(tf.random_normal([self.output_conv], mean=0.0, stddev=self.gaussian), name="bc6"),
			'bd1': tf.Variable(tf.random_normal([self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="bd1"),
			#'bd2': tf.Variable(tf.random_normal([self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="bd2"),
			'out': tf.Variable(tf.random_normal([config.label_size], mean=0.0, stddev=self.gaussian_h), name="bout")
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
		print(input_data.shape)
		#input_data = tf.Variable(da.astype(np.float32))
		
		conv1 = self.convolution_1d(input_data, weights['wc1'], biases['bc1'], strides=1)
		print(conv1.shape)
		conv1 = self.max_pool_1d(conv1, config.max_characters - 7 + 1, self.output_conv, 3)
		print(conv1.shape)

		conv2 = self.convolution_1d(conv1, weights['wc2'], biases['bc2'], strides=1)
		print(conv2.shape)
		conv2 = self.max_pool_1d(conv2, 336 - 7 + 1, self.output_conv, 3)
		print(conv2.shape)

		conv3 = self.convolution_1d(conv2, weights['wc3'], biases['bc3'], strides=1)
		print(conv3.shape)

		conv4 = self.convolution_1d(conv3, weights['wc4'], biases['bc4'], strides=1)
		print(conv4.shape)
		conv5 = self.convolution_1d(conv4, weights['wc5'], biases['bc5'], strides=1)
		print(conv5.shape)
		conv6 = self.convolution_1d(conv5, weights['wc6'], biases['bc6'], strides=1)
		print(conv6.shape)
		pool6 = self.max_pool_1d(conv6, 102, self.output_conv, 3)
		print(pool6.shape)
		# Fully connected layer
		# Reshape conv2 output to fit fully connected layer input
		#print(weights['wd1'])
		fc1 = tf.reshape(pool6, [-1, weights['wd1'].get_shape().as_list()[0]])
		print(fc1.shape)
		fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
		fc1 = tf.nn.relu(fc1)
		print(fc1.shape)
		fc1 = tf.nn.dropout(fc1, dropout)
		'''
		fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
		#print fc2
		fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
		fc2 = tf.nn.relu(fc2)
		#print(fc2)
		fc2 = tf.nn.dropout(fc2, dropout)
		print(fc2.shape)
		# Output, class prediction
		'''
		out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
		print(out.shape)
		out = tf.nn.sigmoid(out)
		return out