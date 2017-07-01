import xml.etree.ElementTree as et
import numpy as np
import os
import utils
import config
import random
import csv
import pandas as pd

class Dataset:
	def __init__(self, path_data = "", batch = 128):
		self.batch = batch
		self.total_texts = 0
		self.total_test = 0
		self.start = 0
		self.end = 0
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
		self.texts_train = np.array([])
		self.labels_train = np.array([])
		data_split = self.ids[start:end]
		#print("Batch: ", len(data_split))
		for i in range(0, len(data_split)):
			index = int(data_split[i])
			text = self.texts[index]
			labels = self.labels[index][0]
			split_labels = labels.split(" ")
			labels_temp = np.zeros(config.label_size)
			for j in range(1, len(split_labels)):
				try:
					label_index = utils.find_label_index(split_labels[j])
					labels_temp[label_index] = 1.0
				except ValueError:
					print("Not have label: ", split_labels[j])
			self.labels_train = np.append(self.labels_train, labels_temp)
			self.texts_train = np.append(self.texts_train, text)
	def read_ids_labels(self):
		with open('data/rcv1-2/train0/ids_index_test0.txt', 'r') as f:
			reader = csv.reader(f)
			self.ids = np.array(list(reader))
		self.data = np.array([])
		with open('data/rcv1-2/train0/labels_test0.txt', 'r') as f:
			reader = csv.reader(f)
			self.labels = list(reader)
			self.labels = np.array(self.labels)
	def load_data_train(self):
		with open('data/rcv1-2/train1014/train.csv', 'r') as f:
			self.texts = f.readlines()
		self.ids = np.arange(len(self.texts))
		self.total_texts = int(len(self.texts) / self.batch) * self.batch
		with open('data/rcv1-2/train1014/labels_train.csv', 'r') as f:
			reader = csv.reader(f)
			self.labels = np.array(list(reader))

	def load_data_test(self):
		with open('data/rcv1-2/train1014/test.csv', 'r') as f:
			self.texts = f.readlines()
		self.ids = np.arange(len(self.texts))
		self.total_texts = int(len(self.texts) / self.batch) * self.batch
		with open('data/rcv1-2/train1014/labels_test.csv', 'r') as f:
			reader = csv.reader(f)
			self.labels = np.array(list(reader))
	def distribution_num_labels(self):
		distribution = np.zeros((20,), dtype=np.int)
		print(distribution)
		for label in self.labels:
			l = label[0].split(" ")
			distribution[len(l) - 1] += 1
		print(distribution)

	def distribution_train_labels(self):
		distribution = np.zeros((103,), dtype=np.int)
		i = 0
		for label in self.labels:
			l = label[0].split(" ")
			for j in range(1, len(l)):
				try:
					label_index = utils.find_label_index(l[j])
					distribution[label_index] += 1
				except ValueError:
					print("Not have label: ", l[j])
		for i in range(len(distribution)):
			print(distribution[i], end = ", ")
	def distribution_characters(self):
		i = 0
		j = 0
		#distribution = np.zeros((40000,), dtype=np.int)
		media = 0
		self.texts_temp = []
		self.labels_temp = []
		for text in self.texts:
			text = text.replace("\n"," ")
			#distribution[len(list(text)) - 1] += 1
			media += len(text)
			if len(text) <= 1014:
				print(i, self.ids[j], self.labels[j], len(text))
				self.texts_temp.append(text)
				self.labels_temp.append(self.labels[i][0])
				i += 1
			j += 1
		print(media / len(self.texts))
		#self.save_text()
	def save_text(self):
		target = open("data/rcv1-2/train1014/train1.csv", 'w')
		target2 = open("data/rcv1-2/train1014/labels_train1.csv", 'w')
		#target.truncate()
		for i in range(len(self.texts_temp)):
			line = self.texts_temp[i]
			target.write(line)
			target.write("\n")
			line = self.labels_temp[i]
			target2.write(line)
			target2.write("\n")
		target.close()
		target2.close()
		
	def distribution_words (self):
		i = 0
		j = 0
		#distribution = np.zeros((40000,), dtype=np.int)
		media = 0
		for text in self.texts:
			text = text.split(" ")
			#distribution[len(list(text)) - 1] += 1
			media += len(text)
			if len(text) < 1014:
				print(i, self.ids[j], len(text))
				i += 1
			j += 1
		print(media / len(self.texts))
	def shuffler(self):
		print ("shuffling data")
		np.random.shuffle(self.ids)
		self.end = 0
		self.start = 0