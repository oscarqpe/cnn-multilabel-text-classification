import numpy as np
import tensorflow as tf
import config

class Mlp:
	def __init__ (self) :
		# Parameters
		self.learning_rate = 0.001
		#self.global_step = tf.Variable(0, trainable=False)
		#self.starter_learning_rate = 0.001
		# Network Parameters
		self.n_input = config.dictionary_size # 
		self.n_classes = config.label_size # reuters total classes
		self.dropout = 0.5 # Dropout, probability to keep units
		self.hidden_size = 2048
		self.gaussian = 0.02
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])
		self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
		# Store layers weight & bias
		self.weights = {
			# fully connected, 7*7*64 inputs, 1024 outputs
			'wd1': tf.Variable(tf.random_normal([config.dictionary_size , self.hidden_size], mean=0.0, stddev=self.gaussian), name = "wd1"),
			#'wd2': tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size], mean=0.0, stddev=self.gaussian), name = "wd2"),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.random_normal([self.hidden_size, config.label_size], mean=0.0, stddev=self.gaussian), name = "out")
		}
		self.biases = {
			'bd1': tf.Variable(tf.random_normal([self.hidden_size], mean=0.0, stddev=self.gaussian), name = "bd1"),
			'bd2': tf.Variable(tf.random_normal([self.hidden_size], mean=0.0, stddev=self.gaussian), name = "bd2"),
			'out': tf.Variable(tf.random_normal([config.label_size], mean=0.0, stddev=self.gaussian), name = "bout")
		}

	def network(self, x, weights, biases, dropout):
		# Fully connected layer
		# Reshape conv2 output to fit fully connected layer input
		#print(weights['wd1'])
		fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
		print(fc1)
		fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
		fc1 = tf.nn.relu(fc1)
		#print(fc1)
		fc1 = tf.nn.dropout(fc1, dropout)
		#print(fc1)
		#fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
		#print fc2
		#fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
		#fc2 = tf.nn.relu(fc2)
		#print(fc2)
		#fc2 = tf.nn.dropout(fc2, dropout)
		# Output, class prediction
		
		out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
		#print out
		out = tf.nn.sigmoid(out)
		return out
