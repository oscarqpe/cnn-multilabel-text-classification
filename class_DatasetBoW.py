import xml.etree.ElementTree as et
import numpy as np
import os
import utils
import config
import random

class Dataset:
	def __init__(self, path_data = "", batch = 25):
		#assert os.path.exists(path_data), 'No existe el archivo con los datos de entrada ' + path_data
		self.path_data = path_data
		self.names = []
		self.names_test = []
		self.texts_train = []
		self.labels_train = []
		self.batch = batch
		self.total_texts = 12288#12337
		#self.total_texts = 7168#7270 # first 3
		self.total_test = 4864#4891
		self.start = 0
		self.end = 0
		self.start_test = 0
		self.end_test = 0
		self.first0 = 0
		self.first1 = 0
		self.first2 = 0
		self.first3 = 0
		self.first4 = 0
		self.first5 = 0
		self.first6 = 0
		self.first7 = 0
		self.first8 = 0
		self.first9 = 0
		self.label_examples = np.zeros(config.label_size)
		for i in range(0, 12337):
			self.names.append("text_" + str(i) + ".xml")
		for i in range(0, self.total_test):
			self.names_test.append("text_" + str(i) + ".xml")
	def read_data(self, name, type = 1):
		#print "extract: " + self.path_data + name
		ruta = ""
		if type == 1:
			ruta = self.path_data + "train/" + name
		elif type == 2:
			ruta = self.path_data + "test/" + name
		elif type == 3:
			ruta = self.path_data + "first3/" + name
		reuters = et.parse(ruta, et.XMLParser(encoding='ISO-8859-1')).getroot()
		extract_labels = False
		#print reuters
		#for reuters in xml.findall('REUTERS'):
		#	print reuters
		matrix = []
		for text in reuters.findall("TEXT"):
			body = utils.extract_body(text)
			if body != "" and body != None:
				extract_labels = True
			#if extract_labels == True:
				labels_temp = np.zeros(config.label_size)
				all_labels = 0
				for a_topic in reuters.findall("TOPICS"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							self.label_examples[label_index] += 1
							all_labels += 1
						except ValueError:
							extract_labels = True
				for a_topic in reuters.findall("PLACES"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							self.label_examples[label_index] += 1
							all_labels += 1
						except ValueError:
							extract_labels = True
				for a_topic in reuters.findall("PEOPLE"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							self.label_examples[label_index] += 1
							all_labels += 1
						except ValueError:
							extract_labels = True
				for a_topic in reuters.findall("ORGS"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							self.label_examples[label_index] += 1
							all_labels += 1
						except ValueError:
							extract_labels = True
				for a_topic in reuters.findall("EXCHANGES"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							self.label_examples[label_index] += 1
							all_labels += 1
						except ValueError:
							extract_labels = True
				if all_labels != 0:
					#print("READ...")
					self.labels_train = np.append(self.labels_train, labels_temp)
					self.texts_train = np.append(self.texts_train, utils.stop_characters(body.text))
					extract_labels = False
				else:
					extract_labels = False

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

	def generate_batch(self, type):
		start = self.start
		end = self.end
		self.texts_train = np.array([])
		self.labels_train = np.array([])
		for i in range(start, end):
			self.read_data(self.names[i], type)

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
	def all_data(self, type):
		for i in range(0, 7270):
			#print(self.names[i])
			self.read_data(self.names[i], type)

	def shuffler(self):
		print ("shuffling data")
		random.shuffle(self.names)
		self.end = 0
		self.start = 0
