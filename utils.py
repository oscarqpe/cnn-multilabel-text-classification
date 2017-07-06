import numpy as np
import config
import re
from PIL import Image
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def one_hot_encoder (text):
	matrix_one_hot = np.array([np.zeros(config.vocabulary_size, dtype=np.float64)])
	if len(text) > config.max_characters:
		for count_character, character in enumerate(text):
			if count_character == config.max_characters:
				break
			try:
				index_character = config.vocabulary.index(character.lower())
				array_encoded = np.zeros(config.vocabulary_size)
				array_encoded[index_character] = 1.0
				matrix_one_hot = np.append(matrix_one_hot, [array_encoded], axis = 0)
			except ValueError:
				matrix_one_hot = np.append(matrix_one_hot, [np.zeros(config.vocabulary_size)], axis = 0)
	else:
		limit_character = 0
		for count_character, character in enumerate(text):
			limit_character = count_character
			try:
				index_character = config.vocabulary.index(character.lower())
				array_encoded = np.zeros(config.vocabulary_size)
				array_encoded[index_character] = 1.0
				matrix_one_hot = np.append(matrix_one_hot, [array_encoded], axis = 0)
			except ValueError:
				matrix_one_hot = np.append(matrix_one_hot, [np.zeros(config.vocabulary_size)], axis = 0)
		for count_character in range(limit_character + 1, config.max_characters):
			matrix_one_hot = np.append(matrix_one_hot, [np.zeros(config.vocabulary_size)], axis = 0)
	matrix_one_hot = np.delete(matrix_one_hot, 0, 0)
	return matrix_one_hot#.transpose()

def extract_body ( text ):
	if text.get("TYPE") is "BRIEF":
		return None
	for body in text.findall("BODY"):
		return body
	return ""

def extract_label ( reuters, topic ):
	for a_topic in reuters.findall(topic):
		for a_d in a_topic.findall("D"):
			try:
				label_index = find_label_index(a_d.text)
				return label_index
			except ValueError:
				return -1
def find_label_index ( label ):
	return config.labels.index(label)

def read_labels (type):
	if type == 1:
		with open("data/reuters/labels.txt", "r") as ins:
			for line in ins:
				config.labels.append(line.strip("\n"))
				config.label_size = len(config.labels)
	elif type == 3:
		with open("data/reuters/labels-first3.txt", "r") as ins:
			for line in ins:
				config.labels.append(line.strip("\n"))
				config.label_size = len(config.labels)
	elif type == 128:
		with open("data/reuters/labels-first3-128.txt", "r") as ins:
			for line in ins:
				config.labels.append(line.strip("\n"))
				config.label_size = len(config.labels)
	elif type == "rcv":
		with open("data/rcv1-2/rcv1.topics.txt", "r") as ins:
			for line in ins:
				config.labels.append(line.strip("\n"))
				config.label_size = len(config.labels)
	elif type == "bibtex":
		with open("data/bibtex/over200/topics.txt", "r") as ins:
			for line in ins:
				config.labels.append(line.strip("\n"))
				config.label_size = len(config.labels)

def get_path_file(i):
	print ("Extrayendo textos archivo " + str(i))
	e = ""
	if i < 10:
		return 'data/reuters/reut2-00' + str(i) + '.xml'
	else:
		return 'data/reuters/reut2-0' + str(i) + '.xml'

def multilabel(logits, labels):
	out = tf.nn.softmax(logits)
	out = -tf.reduce_sum(labels * tf.log(out))
	return out
def ngrams (text):
	array = np.zeros([config.vocabulary_size_grams,config.vocabulary_size_grams])
	temp_text = text.replace("\t"," ").lower()
	missing = ""
	for l in range(0,len(temp_text) - 1):
		try:
			index_character = config.vocabulary_grams.index(temp_text[l].lower())
			index = config.vocabulary_grams.index(temp_text[l + 1].lower())
			array[index_character, index] += 1
		except ValueError:
			missing += temp_text[l]
	total_characters = len(temp_text) - 1
	#print(total_characters)
	array = array * (1 / total_characters)
	return array

def ngrams_test (texts):
	for i in range(len(texts)):
		array = np.zeros([config.vocabulary_size_grams,config.vocabulary_size_grams])
		temp_text = texts[i].replace("\t"," ").lower()
		missing = ""
		print(temp_text)
		for l in range(0,len(temp_text) - 1):
			try:
				index_character = config.vocabulary_grams.index(temp_text[l].lower())
				index = config.vocabulary_grams.index(temp_text[l + 1].lower())
				array[index_character, index] += 1
			except ValueError:
				missing += temp_text[l]
		print("missing:", missing)
		print(array)
		total_characters = len(temp_text) - 1
		print(total_characters)
		array = array * (1 / total_characters)
		print(array)
		plt.imshow(array, cmap='gray')
		plt.show()
def ngrams_picture (array):
	array = array.reshape([4, 1932])
	for i in range(4):
		for j in range(1932):
			print(array[i, j], end=", ")
		print("XD")
	print(array.shape)
	plt.imshow(array)
	plt.show()
def get_accuracy (index, logits, labels):
	#print (logits.shape)
	#print (labels.shape)
	#print(logits[0])
	#print(labels[0])
	count = 0
	for j in range(0, config.batch_size):
		total_etiquetas = np.sum(labels[j])
		#print("total etiquetas:", total_etiquetas)
		'''print("[")
		for k in range(0, len(logits[j])):
			print("{:.7f}".format(logits[j][k]), end =", ")
		print("]")
		'''
		logits_ = np.copy(logits[j]).reshape(1, config.label_size)
		#print logits_.shape
		labels_ = np.copy(labels[j]).reshape(1, config.label_size)
		total = total_etiquetas.astype(int)
		max_x = np.empty(total, dtype=int)
		max_y = np.empty(total, dtype=int)
		#print("labels[i]:", total)
		#print(logits_)
		for i in range(0, total):
			indice = np.argmax(logits_, 1)
			#print(total,i,indice[0])
			max_x[i] = indice[0]
			logits_[0][indice[0]] = 0
			indice = np.argmax(labels_, 1)
			#print(total,i,indice[0])
			max_y[i] = indice[0]
			labels_[0][indice[0]] = 0
		#print(index[j][0], j, "(X, Y): ", max_x, max_y)
		c = np.in1d(max_x,max_y)
		cc = np.where(c == True)[0]
		if len(cc) != 0:
			count += 1
	return count

def get_accuracy_test (logits, labels):
	umbral = 0.50

	hammin_loss = 0
	one_error = 0
	coverage = 0
	ranking_loss = 0
	average_precision = 0
	subset_accuracy = 0
	accuracy = 0
	precision = 0
	recall = 0
	f_beta = 0

	hammin_loss_sum = 0
	one_error_sum = 0
	coverage_sum = 0
	ranking_loss_sum = 0
	average_precision_sum = 0
	subset_accuracy_sum = 0
	accuracy_sum = 0
	precision_sum = 0
	recall_sum = 0
	f_beta_sum = 0

	for j in range(0, config.batch_size):
		total_etiquetas = np.sum(labels[j])
		'''
		print("[")
		for k in range(0, len(logits[j])):
			print("{:.7f}".format(logits[j][k]), end =", ")
		print("]")
		'''
		#print("total etiquetas:", total_etiquetas)
		logits_ = np.array(logits[j]).reshape(1, config.label_size)
		logits_2 = np.array(logits[j]).reshape(1, config.label_size)
		#print logits_.shape
		labels_ = np.copy(labels[j]).reshape(1, config.label_size)
		total = total_etiquetas.astype(int)
		max_x = []#np.empty(total, dtype=int)
		max_x_val = []
		max_y = np.empty(total, dtype=int)
		max_y_val = np.empty(total, dtype=np.float64)

		#print("labels[i]:", total)
		#print(logits_)
		ranking_all_y = rankdata(logits_[0])
		for i in range(0, total):
			indice = np.argmax(labels_, 1)
			max_y[i] = indice[0]
			max_y_val[i] = labels_[0][indice[0]]
			labels_[0][indice[0]] = 0
			#while np.argmax(logits_, 1) > umbral:
			indice = np.argmax(logits_, 1)
			max_x.append(indice[0])
			max_x_val.append(logits_[0][indice[0]])
			logits_[0][indice[0]] = 0
		
		max_h = []#np.empty(total, dtype=int)
		max_h_val = []
		while logits_2[0][np.argmax(logits_2, 1)[0]] > umbral:
			#print(logits_2[0][np.argmax(logits_2, 1)[0]], " > ", umbral)
			indice = np.argmax(logits_2, 1)
			max_h.append(indice[0])
			max_h_val.append(logits_2[0][indice[0]])
			logits_2[0][indice[0]] = 0
		if len(max_h) == 0:
			logits_2 = np.array(logits[j]).reshape(1, config.label_size)
			while logits_2[0][np.argmax(logits_2, 1)[0]] > 0.35:
				#print(logits_2[0][np.argmax(logits_2, 1)[0]], " > ", umbral)
				indice = np.argmax(logits_2, 1)
				max_h.append(indice[0])
				max_h_val.append(logits_2[0][indice[0]])
				logits_2[0][indice[0]] = 0
		if len(max_h) == 0:
			logits_2 = np.array(logits[j]).reshape(1, config.label_size)
			while logits_2[0][np.argmax(logits_2, 1)[0]] > 0.25:
				#print(logits_2[0][np.argmax(logits_2, 1)[0]], " > ", umbral)
				indice = np.argmax(logits_2, 1)
				max_h.append(indice[0])
				max_h_val.append(logits_2[0][indice[0]])
				logits_2[0][indice[0]] = 0
		if len(max_h) == 0:
			logits_2 = np.array(logits[j]).reshape(1, config.label_size)
			while logits_2[0][np.argmax(logits_2, 1)[0]] > 0.125:
				#print(logits_2[0][np.argmax(logits_2, 1)[0]], " > ", umbral)
				indice = np.argmax(logits_2, 1)
				max_h.append(indice[0])
				max_h_val.append(logits_2[0][indice[0]])
				logits_2[0][indice[0]] = 0
		
		if len(max_h) == 0:
			max_h = max_x
		max_h = max_x ## for multi class
		max_x = np.array(max_x)
		ranking_y_predicted = rankdata(max_x_val)
		#print("(X, Y): ", max_h, max_y, max_x, max_x_val)
		### HAMMING LOSS ###
		hammin_loss_sum += len(np.setdiff1d(max_x, max_y)) + len(np.setdiff1d(max_y, max_x))
		### ONE ERROR ###
		if np.argmin(ranking_all_y) in max_y:
			one_error_sum += 1
		### COVERAGE
		coverage_sum += np.max(ranking_y_predicted) - 1
		### RANKING LOSS
		temp_sum = 0
		for i in range(len(max_x)):
			for j in range(len(ranking_all_y)):
				if max_x[i] != j:
					if ranking_all_y[max_x[i]] > ranking_all_y[j]:
						temp_sum += 1
		ranking_loss_sum += (temp_sum) / (len(max_x) * (len(ranking_all_y) - len(max_x)))
		### AVERAGE PRECISION
		average_precision_sum = 0
		### SUBSET ACCURACY ###
		#print (max_h, max_x, max_y)
		if len (max_h) == len(max_y):
			if len(np.setdiff1d(max_y, max_h)) == 0 and len(np.setdiff1d(max_h, max_y)) == 0:
				subset_accuracy_sum += 1

		accuracy_sum += len(np.intersect1d(max_h, max_y)) / len(np.union1d(max_h, max_y))
		precision_sum += len(np.intersect1d(max_h, max_y)) / len(max_h)
		recall_sum += len(np.intersect1d(max_h, max_y)) / len(max_y)
		f_beta_sum += (2 * len(np.intersect1d(max_h, max_y))) / (len(max_h) + len(max_y))
	#print(accuracy_sum, end=", ")
	hammin_loss = hammin_loss_sum / config.batch_size
	one_error = one_error_sum / config.batch_size
	coverage = coverage_sum / config.batch_size
	ranking_loss = ranking_loss_sum / config.batch_size
	average_precision = average_precision_sum / config.batch_size
	subset_accuracy = subset_accuracy_sum / config.batch_size
	accuracy = accuracy_sum / config.batch_size
	precision = precision_sum / config.batch_size
	recall = recall_sum / config.batch_size
	f_beta = f_beta_sum / config.batch_size
	return [hammin_loss, one_error, coverage, ranking_loss, average_precision, subset_accuracy, accuracy, precision, recall, f_beta]

def stop_characters (text):
	chars = '0123456789'
	for char in chars:
		text = text.replace(char, '')
	return text
def draw_one_hot(array):
	array = array.transpose()
	data = np.zeros((69, 1014, 3), dtype=np.uint8)
	for i in range(0, len(array)):
		for j in range(0, len(array[i])):
			if array[i][j] == 1:
				data[i, j] = [255, 255, 255]
	img = Image.fromarray(data, 'RGB')
	img.show()