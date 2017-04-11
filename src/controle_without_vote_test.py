import numpy as np
from extractfeatures import FeaturesExtractor as fe
import tensorflow as tf
import feeder
import training_en_reg_lin
import numpy as np
import numpy as np
#from extractfeatures import FeaturesExtractor as fe
from extractfeatures import ExtractMonoAudioFiles as emaf
import tensorflow as tf
import feedermysql as feeder
from matplotlib import pyplot as plt
import numpy as np
from tkinter import *

exfeeder = feeder.AudioFeederContext(emaf.inpath, opts={'batchsize': emaf.batchsize})

n = exfeeder.nbfeatures

nblabels = exfeeder.nblabels

data_size = exfeeder.nbtests

next_batch = exfeeder

#définition des placeholders

sess = tf.Session()

#features
x = tf.placeholder(tf.float32, [None, n])

#paramètres (poids)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

W = weight_variable([n, nblabels])
b = bias_variable([nblabels])

#approximation de y (vecteur stochastique de proba)
y = tf.nn.softmax(tf.matmul(x, W) + b)

#le label des features x
y_ = tf.placeholder(tf.float32, [None, nblabels])

#fonction de cout (permet de

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.pow(y-y_,2)))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tours = 100

entropy_graph = np.zeros(tours * exfeeder.nbsamples // emaf.batchsize)
accuracy_graph = np.zeros(tours * exfeeder.nbsamples // emaf.batchsize)
time = np.zeros(tours * exfeeder.nbsamples // emaf.batchsize)

sess.run(init)



for i in range(tours * exfeeder.nbsamples // emaf.batchsize):
    print("batch %d/%d" % (i + 1, tours * exfeeder.nbsamples // emaf.batchsize))
    time[i] = i
    # on effectue une étape de l'entrainement (c'est a dire un gradient descent sur tout le batch)
    batch_xs, batch_ys = next_batch()
    #print(sess.run(y, feed_dict={x: batch_xs, y_: batch_ys}))
    ent1, accu1 = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}), sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    print("entropy1  "+ str(ent1))
    print("accuracy1  " + str(accu1))
    accuracy_graph[i] = accu1
    print (accuracy_graph)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    ent2, accu2 = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}), sess.run(accuracy,feed_dict={x: batch_xs, y_: batch_ys})
    print("entropy2  " + str(ent2))
    print("accuracy2  "+str(accu2))
    entropy_graph[i] = ent2

    #for gv in grads_and_vars:
     #   print(str(sess.run(gv[0])) + " - " + gv[1].name, feed_dict={x: batch_xs, y_: batch_ys})

plt.figure(1)
plt.plot(time, entropy_graph)
plt.figure(2)
plt.plot(time, accuracy_graph)
plt.show()

training_en_reg_lin.exfeeder.switchmode()
x_control_base,y_control_base = training_en_reg_lin.exfeeder.getbatch(batchsize=training_en_reg_lin.exfeeder.nbtests)

print(sess.run(accuracy, feed_dict={x: x_control_base, y_: y_control_base}))

