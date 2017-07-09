import xml.etree.ElementTree as et
import numpy as np
import os
import config
import random
import csv
import utils
import pandas as pd

class Dataset:
	def __init__(self, path_data = "", batch = 25):
		#assert os.path.exists(path_data), 'No existe el archivo con los datos de entrada ' + path_data
		self.path_data = path_data
		self.names = []
		self.names_test = []
		self.texts_train = []
		self.labels_train = []
		self.batch = batch
		self.texts = []
		#self.total_texts = 12288#12337
		self.total_texts = 0#119936#7270 # first 3
		self.start = 0
		self.end = 0
		self.start_test = 0
		self.end_test = 0
		'''
		for i in range(0, 120000):
			self.names.append("text_" + str(i) + ".xml")
		for i in range(0, self.total_test):
			self.names_test.append("text_" + str(i) + ".xml")
		'''
	def next_batch(self):
		if self.end == 0:
			self.start = 0
			self.end = self.batch
		elif self.end + self.batch >= self.total_texts:
			self.start = self.end
			self.end = self.total_texts
		else:
			self.start = self.end
			self.end = self.end + self.batch

	def prev_batch(self):
		if self.start == 0:
			self.start = 0
			self.end = self.end
		elif self.start - self.batch <= 0:
			self.start = 0
			self.end = self.batch
		else:
			self.end = self.start
			self.start = self.start - self.batch
	def generate_embedding (self):
		start = self.start
		end = self.end
		self.texts_train = []
		self.labels_train = []
		#print(self.texts[start:end, 2])
		#self.texts_train = list(self.texts[start:end, 2])
		titles = list(self.texts[start:end, 1])
		texts = list(self.texts[start:end, 2])
		for i in range(0, len(texts)):
			text = titles[i] + " " + texts[i]
			self.texts_train.append(text)
		labels = list(self.texts[start: end, 0])
		for i in range(0, len(labels)):
			if labels[i] == '1':
				self.labels_train.append([1,0,0,0])
			if labels[i] == '2':
				self.labels_train.append([0,1,0,0])
			if labels[i] == '3':
				self.labels_train.append([0,0,1,0])
			if labels[i] == '4':
				self.labels_train.append([0,0,0,1])
	def generate_batch(self):
		start = self.start
		end = self.end
		self.texts_train = []
		self.labels_train = []
		data_split = self.ids[start:end]
		#print("Batch: ", len(data_split))
		for i in range(0, len(data_split)):
			index = int(data_split[i])
			self.texts_train.append(self.texts[index])
			labels = self.labels[index]
			#for i in range(0, len(labels)):
			#print("label: ", labels)
			temp = np.zeros(config.label_size)
			temp[int(labels) - 1] = 1
			self.labels_train.append(temp)
			
	def generate_batch_hot (self):
		start = self.start
		end = self.end
		self.texts_train = []
		self.labels_train = []
		#print(self.data[start:end, 2])
		titles = list(self.texts[start:end, 1])
		texts = list(self.texts[start:end, 2])
		for i in range(0, len(texts)):
			text = titles[i] + texts[i]
			text = text.replace(" ", "")
			matrix = utils.one_hot_encoder(text)
			self.texts_train.append(matrix)
		labels = list(self.texts[start: end, 0])
		for i in range(0, len(labels)):
			temp = np.zeros(config.label_size)
			temp[int(labels[i]) - 1] = 1
			self.labels_train.append(temp)

	def generate_batch_test(self):
		start = self.start_test
		end = self.end_test
		self.texts_train = np.array([])
		self.labels_train = np.array([])
		for i in range(start, end):
			self.read_data(self.names_test[i], 2)

	def next_test(self):
		if self.end_test == 0:
			self.start_test = 0
			self.end_test = self.batch
		elif self.end_test + self.batch >= self.total_test:
			self.start_test = self.end_test
			self.end_test = self.total_test
		else:
			self.start_test = self.end_test
			self.end_test = self.end_test + self.batch
	def total_characters(self):
		aux = 0
		for i in range(0, len(self.texts)):
			if len(self.texts[i][2]) > 500:
				aux += 1
				print(aux)
	def all_data(self, data):
		if data == 0:
			self.total_texts = 119936#120000
			df = pd.read_csv('data/ag_news/train.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:120000, 1])
			texts = list(df.ix[0:120000, 2])
			self.texts = []
			self.labels = list(df.ix[0:120000, 0])
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 1:
			self.total_texts = 560000
			df = pd.read_csv('data/dbpedia/train.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:560000, 1])
			texts = list(df.ix[0:560000, 2])
			self.texts = []
			self.labels = list(df.ix[0:560000, 0])
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 2:
			self.total_texts = 649984#650000
			df = pd.read_csv('data/yelp/train.csv', header=None)
			print(np.shape(df))
			texts = list(df.ix[0:650000, 1])
			self.texts = []
			self.labels = list(df.ix[0:650000, 0])
			for i in range(0, len(texts)):
				self.texts.append(texts[i])
			self.ids = np.arange(len(self.texts))
		elif data == 3:
			self.total_texts = 1399936#1400000
			df = pd.read_csv('data/yahoo/train.csv', header=None)
			print(np.shape(df))
			question = list(df.ix[0:1400000, 1])
			content = list(df.ix[0:1400000, 2])
			answer = list(df.ix[0:1400000, 3])
			self.texts = []
			self.labels = list(df.ix[0:1400000, 0])
			for i in range(0, len(question)):
				text = str(question[i]) + " " + str(content[i]) + " " + str(answer[i])
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 4:
			self.total_texts = 449920#450000
			df = pd.read_csv('data/sogou/train.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:450000, 1])
			texts = list(df.ix[0:450000, 2])
			self.texts = []
			self.labels = list(df.ix[0:450000, 0])
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 5:
			self.total_texts = 2999936#3000000
			df = pd.read_csv('data/amazon/train.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:3000000, 1])
			texts = list(df.ix[0:3000000, 2])
			self.texts = []
			self.labels = list(df.ix[0:3000000, 0])
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
	def all_data_vectorizer(self, data):
		if data == 0:
			with open('data/ag_news/train.csv', 'r') as f:
				reader = csv.reader(f)
				self.texts = list(reader)
				self.texts = np.array(self.texts)
			titles = list(self.texts[0:120000, 1])
			texts = list(self.texts[0:120000, 2])
			self.texts = []
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 1:
			df = pd.read_csv('data/dbpedia/train.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:560000, 1])
			texts = list(df.ix[0:560000, 2])
			self.texts = []
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 2:
			df = pd.read_csv('data/yelp/train.csv', header=None)
			print(np.shape(df))
			texts = list(df.ix[0:650000, 1])
			self.texts = []
			for i in range(0, len(texts)):
				self.texts.append(texts[i])
			self.ids = np.arange(len(self.texts))
		elif data == 3:
			df = pd.read_csv('data/yahoo/train.csv', header=None)
			print(np.shape(df))
			question = list(df.ix[0:1400000, 1])
			content = list(df.ix[0:1400000, 2])
			answer = list(df.ix[0:1400000, 3])
			self.texts = []
			for i in range(0, len(question)):
				text = question[i] + " " + content[i] + " " + answer[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 4:
			df = pd.read_csv('data/sogou/train.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:450000, 1])
			texts = list(df.ix[0:450000, 2])
			self.texts = []
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 5:
			df = pd.read_csv('data/amazon/train.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:3000000, 1])
			texts = list(df.ix[0:3000000, 2])
			self.texts = []
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
	def all_data_test(self, data):
		if data == 0:
			self.total_texts = 7552
			df = pd.read_csv('data/ag_news/test.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:7600, 1])
			texts = list(df.ix[0:7600, 2])
			self.texts = []
			self.labels = list(df.ix[0:7600, 0])
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 1:
			self.total_texts = 69888#70000
			df = pd.read_csv('data/dbpedia/test.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:70000, 1])
			texts = list(df.ix[0:70000, 2])
			self.texts = []
			self.labels = list(df.ix[0:70000, 0])
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 2:
			self.total_texts = 49920#50000
			df = pd.read_csv('data/yelp/test.csv', header=None)
			print(np.shape(df))
			texts = list(df.ix[0:50000, 1])
			self.texts = []
			self.labels = list(df.ix[0:50000, 0])
			for i in range(0, len(texts)):
				self.texts.append(texts[i])
			self.ids = np.arange(len(self.texts))
		elif data == 3:
			self.total_texts = 59904#60000
			df = pd.read_csv('data/yahoo/test.csv', header=None)
			print(np.shape(df))
			question = list(df.ix[0:60000, 1])
			content = list(df.ix[0:60000, 2])
			answer = list(df.ix[0:60000, 3])
			self.texts = []
			self.labels = list(df.ix[0:60000, 0])
			for i in range(0, len(question)):
				text = str(question[i]) + " " + str(content[i]) + " " + str(answer[i])
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 4:
			self.total_texts = 59904#60000
			df = pd.read_csv('data/sogou/test.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:60000, 1])
			texts = list(df.ix[0:60000, 2])
			self.texts = []
			self.labels = list(df.ix[0:60000, 0])
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
		elif data == 5:
			self.total_texts = 649984#650000
			df = pd.read_csv('data/amazon/test.csv', header=None)
			print(np.shape(df))
			titles = list(df.ix[0:650000, 1])
			texts = list(df.ix[0:650000, 2])
			self.texts = []
			self.labels = list(df.ix[0:650000, 0])
			for i in range(0, len(texts)):
				text = titles[i] + " " + texts[i]
				self.texts.append(text)
			self.ids = np.arange(len(self.texts))
	def distribution_train_labels(self):
		distribution = np.zeros((config.label_size,), dtype=np.int)
		distribution_l = np.zeros((20,), dtype=np.int)
		i = 0
		for text in self.texts:
			l = label[0].split(" ")
		for i in range(len(distribution)):
			print(distribution[i], end = ", ")
	def shuffler(self):
		print ("shuffling texts")
		np.random.shuffle(self.ids)
		self.end = 0
		self.start = 0
