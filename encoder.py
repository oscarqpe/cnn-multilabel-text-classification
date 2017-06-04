import numpy as np
import tensorflow as tf
import config

class Encoder:
	def __init__ (self) :
		# Parameters
		self.learning_rate = 0.01
		#self.global_step = tf.Variable(0, trainable=False)
		#self.starter_learning_rate = 0.001
		# Network Parameters
		self.n_input = config.dictionary_size # 
		self.n_hidden_1 = 2048 # 1st layer num features
		self.gaussian = 0.02
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		# Store layers weight & bias
		self.weights = {
			'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1], mean=0.0, stddev=self.gaussian), name="encoder_h1"),
			'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input], mean=0.0, stddev=self.gaussian), name="decoder_h1"),
		}
		self.biases = {
			'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1], mean=0.0, stddev=self.gaussian), name="encoder_b1"),
			'decoder_b1': tf.Variable(tf.random_normal([self.n_input], mean=0.0, stddev=self.gaussian), name="decoder_b1"),
		}

	# Building the encoder
	def encoder(self, x):
		# Encoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
		# Decoder Hidden layer with sigmoid activation #2
		return layer_1


	# Building the decoder
	def decoder(self, x):
		# Encoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
		# Decoder Hidden layer with sigmoid activation #2
		return layer_1
