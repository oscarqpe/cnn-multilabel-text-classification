import numpy as np
import config
import re
from PIL import Image

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

def get_accuracy (logits, labels):
	#print logits.shape
	#print labels.shape
	count = 0
	for j in range(0, config.batch_size):
		total_etiquetas = np.sum(labels[j])
		#print("total etiquetas:", total_etiquetas)
		logits_ = np.copy(logits[j]).reshape(1, config.label_size)
		'''print("[")
		for k in range(0, len(logits[j])):
			print("{:.7f}".format(logits[j][k]), end =", ")
		print("]")
		'''
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
		#print(j, "(X, Y): ", max_x, max_y)
		c = np.in1d(max_x,max_y)
		cc = np.where(c == True)[0]
		if len(cc) != 0:
			count += 1
	return count

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