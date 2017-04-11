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

exfeeder = feeder.AudioFeeder(emaf.inpath, opts={'batchsize': emaf.batchsize})

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

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tours = 1

entropy_graph = np.zeros(1000)
accuracy_graph = np.zeros(1000)
time = np.zeros(1000)

sess.run(init)


#tours * exfeeder.nbsamples // emaf.batchsize
for i in range(1000):
    print("batch %d/%d" % (i + 1, tours * exfeeder.nbsamples // emaf.batchsize))
    time[i] = i
    # on effectue une étape de l'entrainement (c'est a dire un gradient descent sur tout le batch)
    batch_xs, batch_ys = next_batch()
    #print(sess.run(y, feed_dict={x: batch_xs, y_: batch_ys}))
    ent1, accu1 = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
    print("entropy1  "+ str(ent1))
    print("accuracy1  " + str(accu1))
    accuracy_graph[i] = accu1
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    ent2, accu2 = sess.run([cross_entropy, accuracy],feed_dict={x: batch_xs, y_: batch_ys})
    print("entropy2  " + str(ent2))
    print("accuracy2  "+str(accu2))
    entropy_graph[i] = ent2



#training_en_reg_lin.exfeeder.switchmode()
exfeeder.switchmode()
x_control_base,y_control_base = exfeeder.getbatch(batchsize=exfeeder.nbtests)

print(sess.run(accuracy, feed_dict={x: x_control_base, y_: y_control_base}))

plt.figure(1)
plt.plot(time, entropy_graph)
plt.figure(2)
plt.plot(time, accuracy_graph)
plt.show()