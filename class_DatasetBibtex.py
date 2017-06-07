import xml.etree.ElementTree as et
import numpy as np
import os
import utils
import config
import random
import csv

class Dataset:
	def __init__(self, path_data = "", batch = 25):
		#assert os.path.exists(path_data), 'No existe el archivo con los datos de entrada ' + path_data
		self.path_data = path_data
		self.names = []
		self.names_test = []
		self.texts_train = []
		self.labels_train = []
		self.batch = batch
		self.total_texts = 14336# 14408
		# test 0
		self.total_test = 14336#
		self.start = 0
		self.end = 0
		self.start_test = 0
		self.end_test = 0
		self.vocabulary_size = 47236

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
	def generate_batch_hot(self):
		start = self.start
		end = self.end
		self.texts_train = []
		self.labels_train = []
		data_split = self.ids[start:end]
		for i in range(0, len(data_split)):
			ids_index = data_split[i][0].split(" ")
			id = int(ids_index[0])
			index = int(ids_index[1])
			labels = self.labels[index][0]
			split_labels = labels.split(" ")
			labels_temp = np.zeros(config.label_size)
			for j in range(1, len(split_labels)):
				try:
					label_index = utils.find_label_index(split_labels[j])
					labels_temp[label_index] = 1.0
				except ValueError:
					print("Not have label: ", split_labels[j])
			self.labels_train.append(labels_temp)
			text_name = str(id) + "text.txt"
			temp_text = ""
			with open('data/bibtex/over200/train/' + text_name, 'r') as f:
				temp_text = f.read()
			temp_text = temp_text + temp_text.replace(" ", "")
			temp_text = temp_text + temp_text.replace(" ", "").replace("\t","")
			matrix = utils.one_hot_encoder(temp_text)
			self.texts_train.append(matrix)
	def generate_batch(self):
		start = self.start
		end = self.end
		self.texts_train = np.array([])
		self.labels_train = np.array([])
		data_split = self.ids[start:end]
		#print("Batch: ", len(data_split))
		for i in range(0, len(data_split)):
			ids_index = data_split[i][0].split(" ")
			id = int(ids_index[0])
			index = int(ids_index[1])
			lines = ""
			with open('data/rcv1-2/train-vectors/'+str(index) +'_vector.txt', 'r') as f:
				lines = f.readlines()
			text = lines[0]
			labels = self.labels[index][0]
			split_text = text.split(" ")
			split_labels = labels.split(" ")
			vector = np.zeros(self.vocabulary_size)
			labels_temp = np.zeros(config.label_size)
			for j in range(1, len(split_text)):
				#print(split_text[j])
				valores = split_text[j].split(":")
				#print(id, ": ", valores[0])
				vector[int(valores[0]) - 1] = float(valores[1])
			for j in range(1, len(split_labels)):
				try:
					label_index = utils.find_label_index(split_labels[j])
					labels_temp[label_index] = 1.0
				except ValueError:
					print("Not have label: ", split_labels[j])
			self.labels_train = np.append(self.labels_train, labels_temp)
			self.texts_train = np.append(self.texts_train, vector)

	def generate_batch_test(self):
		start = self.start_test
		end = self.end_test
		self.texts_train = np.array([])
		self.labels_train = np.array([])
		data_split = self.ids[start:end]
		#print("Batch: ", len(data_split))
		for i in range(0, len(data_split)):
			ids_index = data_split[i][0].split(" ")
			id = int(ids_index[0])
			index = int(ids_index[1])
			lines = ""
			with open('data/rcv1-2/test-vectors0-0/'+str(index) +'_vector.txt', 'r') as f:
				lines = f.readlines()
			text = lines[0]
			labels = self.labels[index][0]
			split_text = text.split(" ")
			split_labels = labels.split(" ")
			vector = np.zeros(self.vocabulary_size)
			labels_temp = np.zeros(config.label_size)
			for j in range(1, len(split_text)):
				#print(split_text[j])
				valores = split_text[j].split(":")
				#print(id, ": ", valores[0])
				vector[int(valores[0]) - 1] = float(valores[1])
			for j in range(1, len(split_labels)):
				try:
					label_index = utils.find_label_index(split_labels[j])
					labels_temp[label_index] = 1.0
				except ValueError:
					print("Not have label: ", split_labels[j])
			self.labels_train = np.append(self.labels_train, labels_temp)
			self.texts_train = np.append(self.texts_train, vector)
	def generate_batch_text(self):
		start = self.start
		end = self.end
		self.texts_train = np.array([])
		self.labels_train = np.array([])
		data_split = self.ids[start:end]
		for i in range(0, len(data_split)):
			ids_index = data_split[i][0].split(" ")
			id = int(ids_index[0])
			index = int(ids_index[1])
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
			text_name = str(id) + "text.txt"
			temp_text = ""
			with open('data/bibtex/over200/train/' + text_name, 'r') as f:
				temp_text = f.read()
			self.texts_train = np.append(self.texts_train, temp_text)
	def generate_batch_test_text(self):
		start = self.start_test
		end = self.end_test
		self.texts_train = np.array([])
		self.labels_train = np.array([])
		data_split = self.ids[start:end]
		for i in range(0, len(data_split)):
			ids_index = data_split[i][0].split(" ")
			id = int(ids_index[0])
			index = int(ids_index[1])
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
			text_name = str(id) + "newsML.xml"
			reuters = et.parse("data/rcv1-2/test-text0-0/" + text_name, et.XMLParser(encoding='ISO-8859-1')).getroot()
			temp_text = ""
			for text in reuters.findall("title"):
				#print(text.text)
				temp_text = temp_text + text.text#.replace(" ", "")
			for text in reuters.findall("text"):
				for p in text.findall("p"):
					temp_text = temp_text + p.text#.replace(" ", "").replace("\t","")
			#print("ID TExt: ", id)
			#print(temp_text)
			self.texts_train = np.append(self.texts_train, temp_text)
	def generate_batch_stemm(self):
		start = self.start
		end = self.end
		self.texts_train = np.array([])
		self.labels_train = np.array([])
		data_split = self.ids[start:end]
		for i in range(0, len(data_split)):
			ids_index = data_split[i][0].split(" ")
			id = int(ids_index[0])
			index = int(ids_index[1])
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
			text_name = str(id) + "token.txt"
			with open("data/rcv1-2/train-tokens/" + text_name, 'r') as f:
				temp_text = f.read()
				self.texts_train = np.append(self.texts_train, temp_text)
	def generate_batch_stemm_test(self):
		start = self.start_test
		end = self.end_test
		self.texts_train = np.array([])
		self.labels_train = np.array([])
		data_split = self.ids[start:end]
		for i in range(0, len(data_split)):
			ids_index = data_split[i][0].split(" ")
			id = int(ids_index[0])
			index = int(ids_index[1])
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
			text_name = str(id) + "token.txt"
			with open("data/rcv1-2/test-tokens0-0/" + text_name, 'r') as f:
				temp_text = f.read()
				self.texts_train = np.append(self.texts_train, temp_text)
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
	def all_data(self, type):
		for i in range(0, 7270):
			#print(self.names[i])
			self.read_data(self.names[i], type)
	def read_rcv_vectors(self):
		with open('data/bibtex/over200/ids_index_train.txt', 'r') as f:
			reader = csv.reader(f)
			self.ids = np.array(list(reader))
		'''
		target = open("data/rcv1-2/ids_index_train.txt", 'w')
		target.truncate()
		for i in range(0, len(self.ids)):
			target.write(self.ids[i][0] + " " + str(i))
			target.write("\n")
		target.close()
		'''
		
		self.data = np.array([])
		with open('data/bibtex/over200/labels_train.txt', 'r') as f:
			reader = csv.reader(f)
			self.labels = list(reader)
			self.labels = np.array(self.labels)
	
	def read_labels(self):
		with open('data/bibtex/over200/ids_index_train.txt', 'r') as f:
			reader = csv.reader(f)
			self.ids = np.array(list(reader))
		with open('data/bibtex/over200/labels_train.txt', 'r') as f:
			reader = csv.reader(f)
			self.labels = list(reader)
			self.labels = np.array(self.labels)

	def read_rcv_vectors_test(self, test):
		with open('data/bibtex/over200/ids_index_test0_' + str(test) + '.txt', 'r') as f:
			reader = csv.reader(f)
			self.ids = np.array(list(reader))
		'''
		target = open("data/bibtex/over200/ids_index_test0_" + str(test) + ".txt", 'w')
		target.truncate()
		for i in range(0, len(self.ids)):
			target.write(self.ids[i][0] + " " + str(i))
			target.write("\n")
		target.close()
		'''
		with open('data/bibtex/over200/labels_test0_' + str(test) + '.txt', 'r') as f:
			reader = csv.reader(f)
			self.labels = np.array(list(reader))

	def read_text(self, init, end):
		with open('data/bibtex/over200/ids_index_train.txt', 'r') as f:
			reader = csv.reader(f)
			self.ids = np.array(list(reader))
		self.texts = []
		for i in range(init, end):
			ids_index = self.ids[i][0].split(" ")
			id = int(ids_index[0])
			text_name = str(id) + "text.txt"
			temp_text = ""
			with open('data/bibtex/over200/train/' + text_name, 'r') as f:
				temp_text = f.read()
			self.texts.append(temp_text)
	def read_text_stemming(self, init, end):
		self.temp_stemming = ""
		with open('data/rcv1-2/lyrl2004_tokens_train.dat', 'r') as f:
			self.temp_stemming = f.read()
		self.temp_stemming = self.temp_stemming.split(".I")
		self.temp_stemming.remove(self.temp_stemming[0])

		self.texts = []
		for i in range(init, end):
			temp_text = self.temp_stemming[i]
			temp_text = temp_text.split(".W")
			self.texts.append(temp_text[1])
	def shuffler(self):
		print ("shuffling data")
		np.random.shuffle(self.ids)
		self.end = 0
		self.start = 0
