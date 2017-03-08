import numpy as np
#from extractfeatures import FeaturesExtractor as fe
from extractfeatures import ExtractMonoAudioFiles as emaf
import tensorflow as tf
import feeder

#featureext = ef.lrft.melspectrogram
#import data base
#ex = ef.Examples(workingpath, featureext, nblabels, batchsize=n)
print("entering feeder")
exfeeder = feeder.AudioFeeder(emaf.inpath, opts={'batchsize': emaf.batchsize})
print("exiting feeder")
n = exfeeder.features.shape[1] #size of the vectors (a modifier)
nblabels = exfeeder.labels.shape[1]


data_size = exfeeder.nbtests


next_batch = exfeeder

#définition des placeholders

#features
x = tf.placeholder(tf.float32, [None, n])

#paramètres (poids)
W = tf.Variable(tf.zeros([n, nblabels]))
b = tf.Variable(tf.zeros([nblabels]))

#approximation de y (vecteur stochastique de proba)
y = tf.nn.softmax(tf.matmul(x, W) + b)

#le label des features x
y_ = tf.placeholder(tf.float32, [None, nblabels])

#fonction de cout (permet de
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



init = tf.initialize_all_variables()

sess = tf.Session()


def getWb(tours):
    sess.run(init)

    for i in range(tours* exfeeder.nbsamples // emaf.batchsize):
        #il n'y pas de raison pour que le nombre d'itérations soit ca (Raph)
        #avec tours = 1 on passe en moyenne une fois par exemple

        print("batch %d/%d" % (i+1, exfeeder.nbsamples // emaf.batchsize))

        #on effectue une étape de l'entrainement (c'est a dire un gradient descent sur tout le batch)
        batch_xs, batch_ys = next_batch()
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #enfin on récupère les paramètres qui ont été optimisés pour répondre au mieux à la régression linéaire
    #ces paramètres sont récupérés dans la classe controle
    return W,b,n,nblabels
