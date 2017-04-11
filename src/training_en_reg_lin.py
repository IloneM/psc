import numpy as np
#from extractfeatures import FeaturesExtractor as fe
from extractfeatures import ExtractMonoAudioFiles as emaf
import tensorflow as tf
import feedermysql as feeder

#featureext = ef.lrft.melspectrogram
#import data base
#ex = ef.Examples(workingpath, featureext, nblabels, batchsize=n)
#print("entering feeder")
exfeeder = feeder.AudioFeederContext(emaf.inpath, opts={'batchsize': emaf.batchsize})
#exfeeder = feeder.AudioFeeder(emaf.inpath, opts={'batchsize': emaf.batchsize})
#print("exiting feeder")
n = exfeeder.nbfeatures #size of the vectors (a modifier)
nblabels = exfeeder.nblabels

data_size = exfeeder.nbtests

next_batch = exfeeder

#définition des placeholders

#features
x = tf.placeholder(tf.float32, [None, n])

#paramètres (poids)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

W = weight_variable([n, nblabels])
b = bias_variable([nblabels])



#approximation de y (vecteur stochastique de proba)
y = tf.nn.softmax(tf.matmul(x, W) + b) #modèle

#le label des features x
y_ = tf.placeholder(tf.float32, [None, nblabels])

#fonction de cout (permet de
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



init = tf.global_variables_initializer()

sess = tf.Session()


def getWb(tours):
    sess.run(init)

    for i in range(tours * exfeeder.nbsamples // emaf.batchsize):
        #il n'y pas de raison pour que le nombre d'itérations soit ca (Raph)
        #avec tours = 1 on passe en moyenne une fois par exemple

        print("batch %d/%d" % (i+1, tours * exfeeder.nbsamples // emaf.batchsize))

        #on effectue une étape de l'entrainement (c'est a dire un gradient descent sur tout le batch)
        batch_xs, batch_ys = next_batch()
        print(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))

    #enfin on récupère les paramètres qui ont été optimisés pour répondre au mieux à la régression linéaire
    #ces paramètres sont récupérés dans la classe controle
    return W,b,n,nblabels
