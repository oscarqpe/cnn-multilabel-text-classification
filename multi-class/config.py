global labels
labels = []
global label_size
label_size = 4
global max_characters
max_characters = 1014
global output_conv_layer_size
output_conv_layer_size = 256
global training_iters
training_iters = 20000
global batch_size
batch_size = 128
global display_step
display_step = 10
global dictionary_size
dictionary_size = 60619
global vocabulary  

vocabulary = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
	't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ',', ';', '.', 
	'!', '?', ':', '\'', '"', '/', '\\', '|', '_', '@', '#', '$', '%', '^', '&', '*', '~', '`', '+', '-', 
	'=', '<', '>', '(', ')', '[', ']', '{', '}']

global vocabulary_size 

vocabulary_size = len(vocabulary)

global texts_test
texts_test = []
global labels_test
labels_test = []