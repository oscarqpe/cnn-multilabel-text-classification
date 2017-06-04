import xml.etree.ElementTree as et
import numpy as np
import tensorflow as tf
import time
import csv
import matplotlib.pyplot as plt
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import icu
import MySQLdb

collator = icu.Collator.createInstance(icu.Locale('UTF-8'))

stop_words = get_stop_words('en')
# print(labels)
vectorizer = CountVectorizer(min_df=1, stop_words = stop_words) #bag of words
text = ""
with open('data/bibtex/tags.txt', 'r') as f:
	text = f.readlines()
plt.figure(figsize=(400, 200))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
x = vectorizer.fit_transform(text).toarray()
print(len(vectorizer.vocabulary_))

v = vectorizer.vocabulary_
print (len(v))
size_dic = len(v)
keys = np.fromiter(iter(v.keys()), dtype=np.dtype('<U25'))
print(len(keys))
keys = sorted(keys, key=collator.getSortKey)
keys = np.array(keys)
vals = np.fromiter(iter(v.values()), dtype=np.dtype(int))
vals = np.sort(vals)
real_keys = []
for val in range(0,len(vals)):
    for key, value in v.items():
        if value == val:
            real_keys.append(key)
arr1inds = np.argsort(x[0])
sorted_arr1 = x[0][arr1inds[::-1]]
print(len(sorted_arr1))
sorted_arr2 = vals[arr1inds[::-1]]
real_keys = np.array(real_keys)
real_keys = real_keys[arr1inds[::-1]]

#print(real_keys)
#print(sorted_arr1)
max_labels = []
total_text = 0
for i in range(size_dic):
	if sorted_arr1[i] > 50:
		print(i, real_keys[i], sorted_arr1[i])
		total_text += sorted_arr1[i]
		max_labels.append(real_keys[i])
print (total_text)
# > 200 -> 56  -> 30070
# > 100 -> 154 -> 43294
# > 50  -> 375 -> 58826
'''
plt.subplot(321)
plt.title("Bag of Words", fontsize='small')
#plt.scatter(np.arange(size_dic), sorted_arr1, marker='o')
plt.scatter(np.arange(size_dic), sorted_arr1, marker='o', cmap=plt.get_cmap('Spectral'))
#labels = labels[arr1inds]
plt.xticks(range(len(real_keys)), real_keys, fontsize=14, rotation='vertical')
plt.xticks(np.arange(min(np.arange(size_dic)), max(np.arange(size_dic))+1, 1))
plt.tick_params(axis='x', which='major', pad=15)
plt.subplots_adjust(bottom=-2, right=2)
plt.ylim(ymin=-1)
plt.xlim(xmin=-1, xmax=100)
plt.show()
'''

texts = []
labels = []
ids = []

db = MySQLdb.connect(host="localhost", user="root", passwd="sistemas", db="bibsonomy")
cur = db.cursor()

# Use all the SQL you like
query = "select \
	b.content_id, \
	b.title, \
	b.bibtexAbstract, \
	( \
		select GROUP_CONCAT(t.tag) \
		from tas t \
		where t.content_id = b.content_id \
		and t.content_type = 2 \
	) as tags_name, \
	( \
		select count(t.tag) \
		from tas t \
		where t.content_id = b.content_id \
		and t.content_type = 2 \
	) as tags \
from bibtex b where b.bibtexAbstract <> '' \
order by tags desc"

cur.execute(query)
labels_separated = []
# print all the first cell of all the rows
for row in cur.fetchall():
	real_tag = ""
	#print(row[0], row[1], row[2], row[3], row[4])
	tags = str(row[3]).split(",")
	for i in range(len(tags)):
		try:
			label_index = max_labels.index(tags[i])
			real_tag += str(max_labels[label_index]) + " "
			try:
				index = labels_separated.index(max_labels[label_index])
			except ValueError:
				labels_separated.append(max_labels[label_index])
		except ValueError:
			real_tag = real_tag
	if real_tag != "":
		texts.append(str(row[1]) + " " + str(row[2]))
		ids.append(str(row[0]))
		labels.append(str(row[0]) + " " + real_tag)
db.close()



for i in range(len(texts)):
	print(i, ids[i], labels[i])

print(labels_separated)
print(len(labels_separated))

target = open("data/bibtex/over50/ids_index_train.txt", 'w')
target.truncate()
for i in range(0, len(ids)):
	target.write(ids[i] + " " + str(i))
	target.write("\n")
target.close()

target = open("data/bibtex/over50/labels_train.txt", 'w')
target.truncate()
for i in range(0, len(labels)):
	target.write(labels[i])
	target.write("\n")
target.close()

for i in range(len(texts)):
	target = open("data/bibtex/over50/train/" + str(ids[i]) + "text.txt", 'w')
	target.truncate()
	line = texts[i]
	target.write(line)
	target.close()

print("total text: ", len(texts))
print("total ids: ", len(ids))
print("total labels: ", len(labels))