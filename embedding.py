import numpy as np
import tensorflow as tf
import config

class Embedding:
	def __init__ (self) :
		# Parameters
		self.learning_rate = 0.0005
		#self.global_step = tf.Variable(0, trainable=False)
		#self.starter_learning_rate = 0.001
		#self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, 10000, 0.0096, staircase=True)
		# Network Parameters
		self.n_input = config.to_embedding # to_embedding
		self.embedding = 256
		self.num_filters = 256
		self.n_classes = config.label_size # reuters total classes
		self.dropout = 0.5 # Dropout, probability to keep units
		self.hidden_size = 256
		self.gaussian = 0.05
		self.gaussian_h = 0.05
		# tf Graph input
		self.x = tf.placeholder(tf.int32, [None, self.n_input])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])
		self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
		self.filters = [3,4,5]
		# Store layers weight & bias
		self.weights = {
			# embedding
			'emb': tf.Variable(tf.random_uniform([self.n_input, self.embedding], -1.0, 1.0), name="emb"),
			# vocabulary_size x 7 conv, 1 input, 256 outputs
			'wc1': tf.Variable(tf.random_normal([self.filters[0], self.embedding, 1, self.num_filters], mean=0.0, stddev=self.gaussian), name="wc1"),
			# 5x5 conv, 32 inputs, 64 outputs
			'wc2': tf.Variable(tf.random_normal([self.filters[1], self.embedding, 1, self.num_filters], mean=0.0, stddev=self.gaussian), name="wc2"),
			'wc3': tf.Variable(tf.random_normal([self.filters[2], self.embedding, 1, self.num_filters], mean=0.0, stddev=self.gaussian), name="wc3"),
			# fully connected, 7*7*64 inputs, self.output_conv outputs
			'wd1': tf.Variable(tf.random_normal([self.num_filters * 3, self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="wd1"),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.random_normal([self.hidden_size, config.label_size], mean=0.0, stddev=self.gaussian_h), name="out")
		}
		self.biases = {
			'bc1': tf.Variable(tf.random_normal([self.num_filters], mean=0.0, stddev=self.gaussian), name="bc1"),
			'bc2': tf.Variable(tf.random_normal([self.num_filters], mean=0.0, stddev=self.gaussian), name="bc2"),
			'bc3': tf.Variable(tf.random_normal([self.num_filters], mean=0.0, stddev=self.gaussian), name="bc3"),
			
			'bd1': tf.Variable(tf.random_normal([self.hidden_size], mean=0.0, stddev=self.gaussian_h), name="bd1"),
			
			'out': tf.Variable(tf.random_normal([config.label_size], mean=0.0, stddev=self.gaussian_h), name="bout")
		}

	def network(self, x, weights, biases, dropout):
		
		embedded_chars = tf.nn.embedding_lookup(weights['emb'], x)
		embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
		pool = []
		conv1 = tf.nn.conv2d(embedded_chars_expanded, weights['wc1'], strides=[1, 1, 1, 1], padding="VALID", name="conv1")
		pool1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']), name="relu")
		pool1 = tf.nn.max_pool(pool1, ksize=[1, self.n_input - self.filters[0] + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool1")
		pool.append(pool1)
		conv2 = tf.nn.conv2d(embedded_chars_expanded, weights['wc2'], strides=[1, 1, 1, 1], padding="VALID", name="conv1")
		pool2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']), name="relu")
		pool2 = tf.nn.max_pool(pool2, ksize=[1, self.n_input - self.filters[1] + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool2")
		pool.append(pool2)
		conv3 = tf.nn.conv2d(embedded_chars_expanded, weights['wc3'], strides=[1, 1, 1, 1], padding="VALID", name="conv1")
		pool3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']), name="relu")
		pool3 = tf.nn.max_pool(pool3, ksize=[1, self.n_input - self.filters[2] + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool3")
		pool.append(pool3)
		num_filters_total = self.num_filters * 3
		print("filters total:", num_filters_total)
		h_pool = tf.concat(pool, 3)
		print("FINAL POOL")
		print(np.shape(h_pool))
		h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
		print(np.shape(h_pool_flat))

		fc1 = tf.add(tf.matmul(h_pool_flat, weights['wd1']), biases['bd1'])
		fc1 = tf.nn.relu(fc1)
		#print(fc1)
		fc1 = tf.nn.dropout(fc1, dropout)

		out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
		#print out
		out = tf.nn.sigmoid(out)
		return out