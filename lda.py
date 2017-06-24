import xml.etree.ElementTree as et
import numpy as np
import tensorflow as tf
import time
import sys
import pickle
import config
import utils
#utils.read_labels("rcv")
import class_DatasetAgN as ds
import mlp as cn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import LatentDirichletAllocation

env = sys.argv[1]
config.dictionary_size = 1014
config.vocabulary_size = 1
# print(labels)
print("Total labels: ", len(config.labels))
print (config.vocabulary_size)

path = ""
if env == "local":
    path = "data/reuters/"
elif env == "server":
    path = "data/reuters/"

cnn = cn.Mlp()
# Construct model
pred = cnn.network(cnn.x, cnn.weights, cnn.biases, cnn.dropout)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=cnn.y))
#cost = tf.reduce_mean(bpmll_out_module.bp_mll(pred, cnn.y))
cost = -tf.reduce_sum(((cnn.y * tf.log(pred + 1e-9)) + ((1-cnn.y) * tf.log(1 - pred + 1e-9)))  , name='xentropy' ) + 0.01 * (tf.nn.l2_loss(cnn.weights['wd1']) + tf.nn.l2_loss(cnn.weights['out']))
optimizer = tf.train.AdamOptimizer(learning_rate=cnn.learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(cnn.y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#accuracy = get_accuracy(logits=pred, labels=y)
data = ds.Dataset(path, config.batch_size)
#data.read_labels() # bibtex, RCV
#data.all_data() # AgNews
data.all_data_vectorizer() # AgNews
#data.read_text(0, 199328)
stop_words = get_stop_words('en')
vectorizer = TfidfVectorizer(stop_words=stop_words, 
                             use_idf=True, 
                             smooth_idf=True)
svd_model = TruncatedSVD(n_components=1014, 
                         algorithm='randomized',
                         n_iter=10, random_state=42)

svd_transformer = Pipeline([('tfidf', vectorizer), 
                            ('svd', svd_model)])
#texts_train = list(data.texts[0:120000, 2])
#print(np.shape(data.texts))
#print(data.texts[0])
#svd_transformer.fit_transform(data.texts)
lda = LatentDirichletAllocation(n_topics=50, max_iter=10, 
                                learning_method='online',                 
                                learning_offset=50., random_state=42)
tf = vectorizer.fit_transform(data.texts)
lda.fit(tf)
print(vectorizer.get_feature_names())
#lda.print_topics(20)
#pickle.dump(svd_transformer, open("data/agnews/vectorizer/vectorizer_lsi_2grams.pickle", "wb"))
#svd_transformer = pickle.load(open("data/rcv1-2/vectorizer/vectorizer_lsi2.pickle", "rb"))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

config_tf = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
with tf.Session(config=config_tf) as sess:
    sess.run(init)
    #print(sess.run("{:.5f}".format(cnn.weights['wc1'])))
    t = time.asctime()
    print (t)
    print("TRAINING")
    step = 1
    # Keep training until reach max iterations
    epoch = 1
    model_saving = 10
    print("Epoch: " + str(epoch))
    #saver.restore(sess, "mlp_weights/model_lsi_5.ckpt")
    #data.shuffler()
    plot_x = []
    plot_y = []
    config.training_iters = 2560000 # 5000 * 128
    data.shuffler()
    print("TOTAL Training: ", data.total_texts)
    train = False
    if train == True:
        while step * config.batch_size <= config.training_iters:
            data.next_batch()
            data.generate_batch_text()
            #print data.texts_train.shape
            #print config.batch_size
            #batch_x = np.array(data.texts_train)
            batch_x = svd_transformer.transform(data.texts_train) 
            batch_x = batch_x.reshape(config.batch_size, config.dictionary_size)
            #batch_x = batch_x.reshape(config.batch_size, config.vocabulary_size * config.max_characters)
            #print("X shape: ", batch_x.shape)
            batch_y = np.array(data.labels_train)
            batch_y = batch_y.reshape(config.batch_size, config.label_size)
            #print(len(batch_x), len(batch_x[0]))
            #print("Y shape: ", batch_y.shape)
            sess.run(optimizer, feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: cnn.dropout})

            if step % 20 == 0:
                #print "Get Accuracy: "
                loss = sess.run([cost], feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: 1.})
                #print loss
                ou = sess.run(pred, feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: 1})
                #print ou.shape
                #print batch_y.shape
                [hammin_loss, one_error, coverage, ranking_loss, average_precision, subset_accuracy, accuracy, precision, recall, f_beta] = utils.get_accuracy_test(ou, batch_y)
                #print(acc)
                plot_x.append(step * config.batch_size)
                plot_y.append(loss)
                print ("Iter " + str(step * config.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss[0]))
                print ("hammin_loss: ", "{:.6f}".format(hammin_loss))
                print ("subset_accuracy: ", "{:.6f}".format(subset_accuracy))
                print ("accuracy: ", "{:.6f}".format(accuracy))
                print ("precision: ", "{:.6f}".format(precision))
                print ("recall: ", "{:.6f}".format(recall))
                print ("f_beta: ", "{:.6f}".format(f_beta))
            if data.end == data.total_texts:
                epoch += 1
                print("Epoch: " + str(epoch))
                data.shuffler()
            if step % 5000 == 0:
                save_path = saver.save(sess, "mlp_weights/model_lsi2_" + str(model_saving) + ".ckpt")
                model_saving += 1
            step += 1
        print(plot_x)
        print(plot_y)
        print ("TESTING")
        data = None
        data = ds.Dataset(path, config.batch_size)
        data.read_labels_test(0) # bibtext, RCV
        #data.all_data_test() # AgNEWS
        step = 1
        total_test = data.total_texts
        print (total_test)
        hammin_loss_sum = 0
        subset_accuracy_sum = 0
        accuracy_sum = 0
        precision_sum = 0
        recall_sum = 0
        f_beta_sum = 0
        while step * config.batch_size <= total_test:
            data.next_batch()
            #data.read_data()
            data.generate_batch_test_text()
            #print data.texts_train.shape
            #print config.batch_size
            #batch_x = np.array(data.texts_train)
            batch_x = svd_transformer.transform(data.texts_train) 
            batch_x = batch_x.reshape(config.batch_size, config.dictionary_size)
            #batch_x = batch_x.reshape(config.batch_size, config.vocabulary_size * config.max_characters)
            #print batch_x.shape
            batch_y = np.array(data.labels_train)
            batch_y = batch_y.reshape(config.batch_size, config.label_size)

            ou = sess.run(pred, feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: 1})
            #print(ou)
            [hammin_loss, one_error, coverage, ranking_loss, average_precision, subset_accuracy, accuracy, precision, recall, f_beta] = utils.get_accuracy_test(ou, batch_y)
            loss = sess.run([cost], feed_dict={cnn.x: batch_x, cnn.y: batch_y, cnn.keep_prob: 1.})
            #print loss
            hammin_loss_sum += hammin_loss
            subset_accuracy_sum += subset_accuracy
            accuracy_sum += accuracy
            precision_sum += precision
            recall_sum += recall
            f_beta_sum += f_beta
            #print(acc)
            print ("Iter " + str(step * config.batch_size) + ", Minibatch Loss= " + str(loss[0]))
            print ("hammin_loss: ", "{:.6f}".format(hammin_loss))
            print ("subset_accuracy: ", "{:.6f}".format(subset_accuracy))
            print ("accuracy: ", "{:.6f}".format(accuracy))
            print ("precision: ", "{:.6f}".format(precision))
            print ("recall: ", "{:.6f}".format(recall))
            print ("f_beta: ", "{:.6f}".format(f_beta))
            step += 1
        step -= 1
        print ("PROMEDIO:")
        print ("hammin_loss_sum: ", hammin_loss_sum / step)
        print ("subset_accuracy_sum: ", subset_accuracy_sum / step)
        print ("accuracy_sum: ", accuracy_sum / step)
        print ("precision_sum: ", precision_sum / step)
        print ("recall_sum: ", recall_sum / step)
        print ("f_beta_sum: ", f_beta_sum / step)
        
    t = time.asctime()
    print (t)