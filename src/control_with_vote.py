import numpy as np
from extractfeatures import FeaturesExtractor as fe
import tensorflow as tf
import feeder
#import training_en_reg_lin
import numpy as np
import numpy as np
#from extractfeatures import FeaturesExtractor as fe
from extractfeatures import ExtractMonoAudioFiles as emaf
import tensorflow as tf
import feedermysql as feeder
from matplotlib import pyplot as plt
import numpy as np
from tkinter import *

exfeeder = feeder.AudioFeederFullContext(opts={'batchsize': emaf.batchsize})

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

vote = tf.argmax(tf.reduce_sum(y, 0), 0)

tours =1

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
    ent1, accu1 = sess.run([cross_entropy,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
    print("entropy1  "+ str(ent1))
    print("accuracy1  " + str(accu1))
    accuracy_graph[i] = accu1
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    ent2, accu2 = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
    print("entropy2  " + str(ent2))
    print("accuracy2  "+str(accu2))
    entropy_graph[i] = ent2

    #for gv in grads_and_vars:
     #   print(str(sess.run(gv[0])) + " - " + gv[1].name, feed_dict={x: batch_xs, y_: batch_ys})

plt.figure(1)
plt.plot(time, entropy_graph)
plt.figure(2)
plt.plot(time, accuracy_graph)


print("pourcentage d'extraits musicaux de controle labellés correctement")
# dans cette partie on va mettre en place le système de vote. Il faudrait alors avoir#c'est une liste dont les éléments sont un couple (spectrogram, label) correspondant à un extrait musical entier
# la base de donnée sous une forme pratique
# c'est une liste dont les éléments sont un couple (spectrogram, label) correspondant à un extrait musical entier

#af = feeder.AudioFeeder(emaf.inpath, opts={'contextmode': True})
af = exfeeder
#af.opts['contextmode'] = True
af.switchmode()

samples = af.getbatch()

s = 0.
total = 0.
for e in samples:
    current_x = e[0]
    current_label = np.argmax(e[1][0])
    #print("current x :")
    #print(current_x)

    if (sess.run(vote, feed_dict={x: current_x, y_: e[1]}) == current_label):
        s = s + 1.
    else:
        print("sample number " + str(total)+" unsuccessful")
        print("current label :")
        print(current_label)
        print("vote")
        print(sess.run(vote, feed_dict={x: current_x, y_: e[1]}))
    total += 1.

print("pourcentage d'enculé ")
print(s / total)

plt.show()
