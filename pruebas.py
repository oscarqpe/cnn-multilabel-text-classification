import xml.etree.ElementTree as et
import numpy as np
import tensorflow as tf
import time
import sys
import config
import utils
utils.read_labels("rcv")
import class_DatasetRcv as ds
import cnn as cn
path =""
data = ds.Dataset(path, config.batch_size)
#data.read_labels() # bibtex, RCV
data.read_text(0, 199450)
print("Total text: ", len(data.texts))
data.write_cvs()
#utils.ngrams(data.texts)
#data.distribution_characters()
#data.distribution_num_labels()
#data.read_text(0, 14408)
#data.distribution_train_labels()
