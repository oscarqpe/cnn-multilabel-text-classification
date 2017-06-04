import xml.etree.ElementTree as et
import numpy as np
import tensorflow as tf
import time
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import icu
collator = icu.Collator.createInstance(icu.Locale('UTF-8'))
def extract_body ( text ):
    if text.get("TYPE") is "BRIEF":
        return None
    for body in text.findall("BODY"):
        return body
    return ""

reuters = et.parse("data/lerolero.xml", et.XMLParser(encoding='utf-8')).getroot()

plt.figure(figsize=(400, 200))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

vectorizer = CountVectorizer(min_df=1)

matrix = []
for text in reuters.findall("TEXT"):
    body = extract_body(text)
    if body != "" and body != None:
        #body = list(body.text)
        matrix.append(body.text)
        vectorizer.fit(matrix)
        x = vectorizer.transform(matrix).toarray()
        #print(x[0])
        #print (y)
        v = vectorizer.vocabulary_
        print(len(v))
        keys = np.fromiter(iter(v.keys()), dtype=np.dtype('<U25'))
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
        sorted_arr2 = vals[arr1inds[::-1]]
        real_keys = np.array(real_keys)
        real_keys = real_keys[arr1inds[::-1]]
        plt.subplot(321)
        plt.title("Bag of Words", fontsize='small')
        #plt.scatter(np.arange(370), sorted_arr1, marker='o')
        plt.scatter(np.arange(370), sorted_arr1, marker='o', cmap=plt.get_cmap('Spectral'))
        #labels = labels[arr1inds]
        plt.xticks(range(len(real_keys)), real_keys, fontsize=14, rotation='vertical')
        plt.xticks(np.arange(min(np.arange(370)), max(np.arange(370))+1, 1))
        plt.tick_params(axis='x', which='major', pad=15)
        plt.subplots_adjust(bottom=-0.5, right=4)
        plt.ylim(ymin=-1)
        plt.xlim(xmin=-1, xmax=100)
plt.show()