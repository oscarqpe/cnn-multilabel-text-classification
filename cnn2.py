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
		self.output_conv = 16
		self.hidden_size = 2048
		self.gaussian = 0.05
		self.gaussian_h = 0.02
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])
		self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
		# Store layers weight & bias
		self.weights = {
			# vocabulary_size x 7 conv, 1 input, 128 outputs
			'wc1': tf.Variable(tf.random_normal([7, config.vocabulary_size, self.output_conv], mean=0.0, stddev=self.gaussian), name="wc1"),
			# 5x5 conv, 32 inputs, 64 outputs
			'wc2': tf.Variable(tf.random_normal([7, 1, self.output_conv], mean=0.0, stddev=self.gaussian), name="wc2"),
			'wc3': tf.Variable(tf.random_normal([7, 1, self.output_conv], mean=0.0, stddev=self.gaussian), name="wc3"),
			# fully connected, 7*7*64 inputs, 1024 outputs
			'wd1': tf.Variable(tf.random_normal([336 * 3 * self.output_conv, self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="wd3"),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.random_normal([self.hidden_size, config.label_size], mean=0.0, stddev=self.gaussian_h), name="out")
		}
		self.biases = {
			'bc1': tf.Variable(tf.random_normal([self.output_conv], mean=0.0, stddev=self.gaussian), name="bc1"),
			'bc2': tf.Variable(tf.random_normal([self.output_conv], mean=0.0, stddev=self.gaussian), name="bc2"),
			'bc3': tf.Variable(tf.random_normal([self.output_conv], mean=0.0, stddev=self.gaussian), name="bc3"),
			'bd1': tf.Variable(tf.random_normal([self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="bd1"),
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
		input_data = tf.reshape(x, shape=[-1, config.max_characters, config.vocabulary_size])
		print(np.shape(input_data))
		#input_data = tf.Variable(da.astype(np.float32))
		pool = []
		conv1 = self.convolution_1d(input_data, weights['wc1'], biases['bc1'], strides=1)
		print(np.shape(conv1))
		conv1 = self.max_pool_1d(conv1, config.max_characters - 7 + 1, self.output_conv, 3)
		print(np.shape(conv1))
		pool.append(conv1)
		print(np.shape(pool))
		conv2 = self.convolution_1d(input_data, weights['wc2'], biases['bc2'], strides=1)
		print(np.shape(conv2))
		conv2 = self.max_pool_1d(conv2, config.max_characters - 7 + 1, self.output_conv, 3)
		print(np.shape(conv2))
		pool.append(conv2)
		print(np.shape(pool))
		conv3 = self.convolution_1d(input_data, weights['wc3'], biases['bc3'], strides=1)
		print(np.shape(conv3))
		conv3 = self.max_pool_1d(conv3, config.max_characters - 7 + 1, self.output_conv, 3)
		print(np.shape(conv3))
		pool.append(conv3)
		print(np.shape(pool))

		h_pool = tf.concat(pool, 2)
		print("FINAL POOL")
		print(np.shape(h_pool))
		h_pool_flat = tf.reshape(h_pool, [-1, 336 * 3 * self.output_conv])
		print(np.shape(h_pool_flat))

		# Fully connected layer
		# Reshape conv2 output to fit fully connected layer input
		#print(weights['wd1'])
		fc1 = tf.reshape(pool, [-1, weights['wd1'].get_shape().as_list()[0]])
		print(np.shape(fc1))
		fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
		fc1 = tf.nn.relu(fc1)
		print(np.shape(fc1))
		fc1 = tf.nn.dropout(fc1, dropout)

		out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
		#print out
		out = tf.nn.sigmoid(out)
		print(np.shape(out))
		return out