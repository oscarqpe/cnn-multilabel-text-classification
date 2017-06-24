global labels
labels = []
global label_size
# AG NEWS
label_size = 4
global max_characters
max_characters = 1014#1932
global batch_size
batch_size = 128
global to_embedding
global dictionary_size
# RCV1
#dictionary_size = 47236 # default
#dictionary_size = 85913 # bow
#dictionary_size = 85913 # tfidf
#dictionary_size = 47115 # bow_stemm
#dictionary_size = 47115 # bow_stemm_tfidf
to_embedding = 8502
# BIBTEX
#dictionary_size = 55443 # bow
#dictionary_size = 55443 # tfidf
#dictionary_size = 55443 # bow_stemm
#dictionary_size = 55443 # bow_stemm_tfidf

# AGNEWS
dictionary_size = 60619 # bow
#dictionary_size = 60619 # tfidf
#dictionary_size = 73964 # bow_stemm
#dictionary_size = 73964 # bow_stemm_tfidf
#dictionary_size = 1014 # lsi
global vocabulary  

vocabulary = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
	't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ',', ';', '.', 
	'!', '?', ':', '\'', '"', '/', '\\', '|', '_', '@', '#', '$', '%', '^', '&', '*', '~', '`', '+', '-', 
	'=', '<', '>', '(', ')', '[', ']', '{', '}']
vocabulary = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
	't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ',', ';', '.', 
	'!', '?', ':', '\'', '"', '/', '\\', '|', '_', '@', '#', '$', '%', '^', '&', '*', '~', '`', '+', '-', 
	'=', '<', '>', '(', ')', '[', ']', '{', '}']
global vocabulary_size 

vocabulary_size = len(vocabulary)

global vocabulary_grams

vocabulary_grams = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
	't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ',', ';', '.', 
	'!', '?', ':', '\'', '"', '/', '\\', '|', '_', '@', '#', '$', '%', '^', '&', '*', '~', '`', '+', '-', 
	'=', '<', '>', '(', ')', '[', ']', '{', '}', ' ']

global vocabulary_size_grams

vocabulary_size_grams = len(vocabulary_grams)

global texts_test
texts_test = []
global labels_test
labels_test = []