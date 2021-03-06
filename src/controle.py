import numpy as np
from extractfeatures import FeaturesExtractor as fe
from extractfeatures import ExtractMonoAudioFiles as emaf
import tensorflow as tf
import feeder
import training_en_reg_lin
import numpy as np

W, b, n, nblabels = training_en_reg_lin.getWb(1)
print(W)
print(b)

x = tf.placeholder(tf.float32, [None, n])

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, nblabels])

vote = tf.argmax(tf.reduce_sum(y, 1), 0)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)



print("pourcentage d'extraits musicaux de controle labellés correctement")
# dans cette partie on va mettre en place le système de vote. Il faudrait alors avoir#c'est une liste dont les éléments sont un couple (spectrogram, label) correspondant à un extrait musical entier
# la base de donnée sous une forme pratique
# c'est une liste dont les éléments sont un couple (spectrogram, label) correspondant à un extrait musical entier

#af = feeder.AudioFeeder(emaf.inpath, opts={'contextmode': True})
af = training_en_reg_lin.exfeeder
#af.opts['contextmode'] = True
af.switchmode()

samples = af.getbatch()

s = 0
total = 0
for e in samples:
    # get current_x
    # get current_label
    current_x = e[0]
    current_label = np.argmax(e[1][0])

    if (sess.run(vote, feed_dict={x: current_x}) == current_label):
        s = s + 1
    total += 1

print(s / total)

