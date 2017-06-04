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
				temp_text = body.text.replace(" ", "")
				body = list(temp_text)
			#if extract_labels == True:
				labels_temp = np.zeros(config.label_size)
				all_labels = 0
				for a_topic in reuters.findall("TOPICS"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							all_labels += 1
						except ValueError:
							extract_labels = True
				for a_topic in reuters.findall("PLACES"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							all_labels += 1
						except ValueError:
							extract_labels = True
				for a_topic in reuters.findall("PEOPLE"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							all_labels += 1
						except ValueError:
							extract_labels = True
				for a_topic in reuters.findall("ORGS"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							all_labels += 1
						except ValueError:
							extract_labels = True
				for a_topic in reuters.findall("EXCHANGES"):
					for a_d in a_topic.findall("D"):
						try:
							label_index = utils.find_label_index(a_d.text)
							labels_temp[label_index] = 1.0
							all_labels += 1
						except ValueError:
							extract_labels = True
				if all_labels != 0:
					self.labels_train = np.append(self.labels_train, labels_temp)
					matrix = utils.one_hot_encoder(body)
					#print matrix[1]
					self.texts_train = np.append(self.texts_train, matrix)
					#self.texts_train.append(matrix)
					#print len(self.texts_train)
					extract_labels = False
				else:
					extract_labels = False
		#self.texts_train = np.concatenate([block for block in self.texts_train], 0)
		#print self.texts_train.shape
		#np.reshape(self.texts_train, (self.batch, config.vocabulary_size * config.max_characters))

	def split_data(self, count_train, count_test):
		print (self.path_data)
		xml = et.parse(self.path_data, et.XMLParser(encoding='ISO-8859-1')).getroot()
		extract_labels = False
		ruta_train = "/home/oscarqpe/proyectos/qt/vigilancia-tecnologica/clasificador-multietiqueta/python/data/reuters/first9"
		ruta_test = "/home/oscarqpe/proyectos/qt/vigilancia-tecnologica/clasificador-multietiqueta/python/data/reuters/test"
		count_train = count_train
		count_test = count_test
		for reuters in xml.findall('REUTERS'):
			matrix = []

			for text in reuters.findall("TEXT"):
				body = utils.extract_body(text)
				if body != "" and body != None:
					extract_labels = True
					body = list(body.text)
				#if extract_labels == True:
					labels_temp = np.zeros(config.label_size)
					all_labels = 0
					for a_topic in reuters.findall("TOPICS"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
							except ValueError:
								extract_labels = True
					for a_topic in reuters.findall("PLACES"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
							except ValueError:
								extract_labels = True
					for a_topic in reuters.findall("PEOPLE"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
							except ValueError:
								extract_labels = True
					for a_topic in reuters.findall("ORGS"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
							except ValueError:
								extract_labels = True
					for a_topic in reuters.findall("EXCHANGES"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
							except ValueError:
								extract_labels = True
					
					if all_labels != 0:
						if reuters.get("LEWISSPLIT") == "TRAIN":
							if all_labels > 1 and all_labels <= 9:
								target = open(ruta_train + "/text_" + str(self.first9) + ".xml", 'w')
								count_train += 1
								target.truncate()
								content = et.tostring(reuters, encoding='ISO-8859-1', method='xml')
								target.write(content)
								target.write("\n")
								target.close()

								self.labels_train.append(labels_temp)
								matrix = utils.one_hot_encoder(body)
								self.texts_train.append(matrix)
								
						'''
						if reuters.get("LEWISSPLIT") == "TEST":
							target = open(ruta_test + "/text_" + str(count_test) + ".xml", 'w')
							count_test += 1
							target.truncate()
							content = et.tostring(reuters, encoding='ISO-8859-1', method='xml')
							target.write(content)
							target.write("\n")
							#print "And finally, we close it."
							target.close()

							config.labels_test.append(labels_temp)
							matrix = utils.one_hot_encoder(body)
							config.texts_test.append(matrix)
						'''
						extract_labels = False
					else:
						extract_labels = False
					if all_labels != 0:
						if reuters.get("LEWISSPLIT") == "TRAIN":
							if all_labels > 1 and all_labels <= 3:
								self.first3 += 1
							if all_labels > 1 and all_labels <= 4:
								self.first4 += 1
							if all_labels > 1 and all_labels <= 5:
								self.first5 += 1
							if all_labels > 1 and all_labels <= 6:
								self.first6 += 1
							if all_labels > 1 and all_labels <= 7:
								self.first7 += 1
							if all_labels > 1 and all_labels <= 8:
								self.first8 += 1
							if all_labels > 1 and all_labels <= 9:
								self.first9 += 1
	def extract_label_vector (self, ruta):
		print (self.path_data)
		extract_labels = False
		real_labels = []
		for i in range(0, 128):#7168):
			reuters = et.parse(ruta + "/text_" + str(i) + ".xml", et.XMLParser(encoding='ISO-8859-1')).getroot()
			for text in reuters.findall("TEXT"):
				body = utils.extract_body(text)
				if body != "" and body != None:
					extract_labels = True
					body = list(body.text)
				#if extract_labels == True:
					labels_temp = np.zeros(config.label_size)
					all_labels = 0
					for a_topic in reuters.findall("TOPICS"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
								try:
									index = real_labels.index(config.labels[label_index])
								except:
									print("Labels first3: ", len(real_labels))
									real_labels.append(config.labels[label_index])
									extract_labels = True
							except ValueError:
								extract_labels = True
					for a_topic in reuters.findall("PLACES"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
								try:
									index = real_labels.index(config.labels[label_index])
								except:
									print("Labels first3: ", len(real_labels))
									real_labels.append(config.labels[label_index])
									extract_labels = True
							except ValueError:
								extract_labels = True
					for a_topic in reuters.findall("PEOPLE"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
								try:
									index = real_labels.index(config.labels[label_index])
								except:
									print("Labels first3: ", len(real_labels))
									real_labels.append(config.labels[label_index])
									extract_labels = True
							except ValueError:
								extract_labels = True
					for a_topic in reuters.findall("ORGS"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
								try:
									index = real_labels.index(config.labels[label_index])
								except:
									print("Labels first3: ", len(real_labels))
									real_labels.append(config.labels[label_index])
									extract_labels = True
							except ValueError:
								extract_labels = True
					for a_topic in reuters.findall("EXCHANGES"):
						for a_d in a_topic.findall("D"):
							try:
								label_index = utils.find_label_index(a_d.text)
								labels_temp[label_index] = 1.0
								all_labels += 1
								try:
									index = real_labels.index(config.labels[label_index])
								except:
									print("Labels first3: ", len(real_labels))
									real_labels.append(config.labels[label_index])
									extract_labels = True
							except ValueError:
								extract_labels = True
		print("Labels first3->: ", len(real_labels))
		print(real_labels)
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

	def shuffler(self):
		print ("shuffling data")
		random.shuffle(self.names)
		self.end = 0
		self.start = 0
