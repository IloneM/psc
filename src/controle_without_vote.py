import numpy as np
from extractfeatures import FeaturesExtractor as fe
import tensorflow as tf
import feeder
import training_en_reg_lin
import numpy as np

#W, b, n, nblabels = training_en_reg_lin.getWb(1)
W, b, n, nblabels = training_en_reg_lin.getWb(1)

#print(W)
#print(b)

x = tf.placeholder(tf.float32, [None, n])

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, nblabels])

#à ce jour y est un vecteur de probabilité par rapport à chacune des notes

#on compare les argmax des vecteurs y avec les vecteurs y_ (argmax = indice de la plus grande valeur)
#on obtient un vecteur de booléens
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#ce vecteur de booléens est transformé via "cast" en vecteur de 0 et de 1. Il suffit ensuite d'effectuer la moyenne
#pour obtenir le pourcentage de réussite sur la simulation
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# get x_control_base (ici il nous faut la base de donnée sous la forme d'une suite de vecteurs (donc une matrice)
# get y_control_base (ceci est aussi une suite de vecteurs sous la forme donc une matrice)
training_en_reg_lin.exfeeder.switchmode()
x_control_base,y_control_base = training_en_reg_lin.exfeeder.getbatch(batchsize=training_en_reg_lin.exfeeder.nbtests)

print(sess.run(accuracy, feed_dict={x: x_control_base, y_: y_control_base}))

