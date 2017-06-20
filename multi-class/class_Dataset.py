import xml.etree.ElementTree as et
import numpy as np
import os
import config
import random
import csv
import utils

class Dataset:
	def __init__(self, path_data = "", batch = 25):
		#assert os.path.exists(path_data), 'No existe el archivo con los datos de entrada ' + path_data
		self.path_data = path_data
		self.names = []
		self.names_test = []
		self.texts_train = []
		self.labels_train = []
		self.batch = batch
		self.data = []
		#self.total_texts = 12288#12337
		self.total_texts = 119936#7270 # first 3
		self.total_test = 4864#4891
		self.start = 0
		self.end = 0
		self.start_test = 0
		self.end_test = 0
		for i in range(0, 120000):
			self.names.append("text_" + str(i) + ".xml")
		for i in range(0, self.total_test):
			self.names_test.append("text_" + str(i) + ".xml")

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

	def generate_batch(self):
		start = self.start
		end = self.end
		self.texts_train = [np.array([])]
		self.labels_train = []
		#print(self.data[start:end, 2])
		self.texts_train = list(self.data[start:end, 2])
		labels = list(self.data[start: end, 0])
		for i in range(0, len(labels)):
			if labels[i] == '1':
				self.labels_train.append([1,0,0,0])
			if labels[i] == '2':
				self.labels_train.append([0,1,0,0])
			if labels[i] == '3':
				self.labels_train.append([0,0,1,0])
			if labels[i] == '4':
				self.labels_train.append([0,0,0,1])
	def generate_batch_hot (self):
		start = self.start
		end = self.end
		self.texts_train = []
		self.labels_train = []
		#print(self.data[start:end, 2])
		titles = list(self.data[start:end, 1])
		texts = list(self.data[start:end, 2])
		for i in range(0, len(texts)):
			text = titles[i] + texts[i]
			text = text.replace(" ", "")
			matrix = utils.one_hot_encoder(text)
			self.texts_train.append(matrix)
		labels = list(self.data[start: end, 0])
		for i in range(0, len(labels)):
			if labels[i] == '1':
				self.labels_train.append([1,0,0,0])
			if labels[i] == '2':
				self.labels_train.append([0,1,0,0])
			if labels[i] == '3':
				self.labels_train.append([0,0,1,0])
			if labels[i] == '4':
				self.labels_train.append([0,0,0,1])

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
		for i in range(0, len(self.data)):
			if len(self.data[i][2]) > 500:
				aux += 1
				print(aux)
	def all_data(self):
		with open('../data/ag_news/train.csv', 'r') as f:
			reader = csv.reader(f)
			self.data = list(reader)
			self.data = np.array(self.data)
	def all_data_test(self):
		self.total_texts = 7552
		with open('../data/ag_news/test.csv', 'r') as f:
			reader = csv.reader(f)
			self.data = list(reader)
			self.data = np.array(self.data)
	def shuffler(self):
		print ("shuffling data")
		np.random.shuffle(self.data)
		self.end = 0
		self.start = 0
