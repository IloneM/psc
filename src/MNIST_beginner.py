import numpy as np
from extractfeatures import FeaturesExtractor as fe
import tensorflow as tf
import feeder

#featureext = ef.lrft.melspectrogram
#import data base
#ex = ef.Examples(workingpath, featureext, nblabels, batchsize=n)
print("entering feeder")
exfeeder = feeder.Feeder(fe.workingpath, batchsize=fe.batchsize)
print("exiting feeder")
n = exfeeder.features.shape[1] #size of the vectors (a modifier)
nblabels = exfeeder.labels.shape[1]

##train_examples = np.random.rand(10000,n)
##train_labels = np.random.rand(10000,nblabels)
##
##control_examples = np.random.rand(1000,n)
##control_labels = np.random.rand(1000,nblabels)
##
##control_examples2 = np.random.rand(1000,10,n)
##control_labels2 = np.random.rand(1000,1,nblabels)

##data_size = np.shape(control_labels)[0]
data_size = exfeeder.nbtests

##print(np.shape(train_examples))

#define batching
next_batch = exfeeder
#train_examples = np.zeros((n, *ex.featuresshape))
#train_labels = np.zeros((n, nblabels))

##### batch_xs = np.zeros((n, *ex.featuresshape))
##### batch_ys = np.zeros((n, nblabels))
##### 
##### control_examples = np.zeros((ex.nbtests, *ex.featuresshape))
##### control_labels = np.zeros((ex.nbtests, nblabels))

##def next_batch(features, labels, batch_size):
##  data_size = np.shape(features)[0]
##  feature_size = np.shape(features)[1]
##  label_size = np.shape(labels)[1]
##  batch_features = np.zeros((batch_size, feature_size))
##  batch_labels = np.zeros((batch_size, label_size))
##  for i in range(batch_size):
##    x = np.floor(np.random.random(1)[0] * data_size) #pour moi ce n'est pas size qu'il faut mettre ici.
##    #la variable aléatoire doit parcourir l'ensemble de la base de données donc à la place de size il faudrait mettre
##    #le nombre d'exemples
##    batch_features[i] = features[x]
##    batch_labels[i] = labels[x]
##    return batch_features, batch_labels

#import tensorfl

#emplacements variables
x = tf.placeholder(tf.float32, [None, n])

W = tf.Variable(tf.zeros([n, nblabels]))
b = tf.Variable(tf.zeros([nblabels]))

control_predictions = tf.Variable(tf.zeros([n]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, nblabels])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(exfeeder.nbexamples // exfeeder.batchsize):
#batch_xs, batch_ys = next_batch(train_examples, train_labels)
    print("batch %d/%d" % (i+1, exfeeder.nbexamples // exfeeder.batchsize))

    batch_xs, batch_ys = next_batch()
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
batch_xs, batch_ys = next_batch(batchsize = exfeeder.nbexamples % exfeeder.batchsize)
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###vote = tf.argmax(tf.reduce_sum(y, 1),1)

exfeeder.switchmode()
control_examples, control_labels = next_batch(batchsize = exfeeder.nbtests)

print("pourcentage de vecteurs de controle labellés correctement")
print(sess.run(accuracy, feed_dict={x: control_examples, y_: control_labels}))

#avec le systeme de vote
#il faudrait avoir la base stockée comme des images et non plus des vecteurs



###print("pourcentage d'extraits musicaux de controle labellés correctement")
###
###for i in range(data_size):
###    control_predictions[i] = sess.run(vote,feed_dict={x: control_examples2[i], y: control_labels2[i]})
###print(tf.reduce_mean(tf.cast(tf.equal(control_predictions, tf.argmax(control_labels2, 1)))))
