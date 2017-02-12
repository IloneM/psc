import numpy as np

n = 100 #size of the vectors (a modifier)



#import data base
train_examples = np.random.rand(10000,n)
train_labels = np.random.rand(10000,88)

control_examples = np.random.rand(1000,n)
control_labels = np.random.rand(1000,88)

control_examples2 = np.random.rand(1000,10,n)
control_labels2 = np.random.rand(1000,1,88)

data_size = np.shape(control_labels)[0]

print(np.shape(train_examples))

#define batching
def next_batch(features, labels, batch_size):
  data_size = np.shape(features)[0]
  feature_size = np.shape(features)[1]
  label_size = np.shape(labels)[1]
  batch_features = np.zeros((batch_size, feature_size))
  batch_labels = np.zeros((batch_size, label_size))
  for i in range(batch_size):
    x = np.floor(np.random.random(1)[0] * data_size) #pour moi ce n'est pas size qu'il faut mettre ici.
    #la variable aléatoire doit parcourir l'ensemble de la base de données donc à la place de size il faudrait mettre
    #le nombre d'exemples
    batch_features[i] = features[x]
    batch_labels[i] = labels[x]
    return batch_features, batch_labels



#import tensorflow
import tensorflow as tf



#emplacements variables
x = tf.placeholder(tf.float32, [None, n])

W = tf.Variable(tf.zeros([n, 88]))
b = tf.Variable(tf.zeros([88]))

control_predictions = tf.Variable(tf.zeros([n]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 88])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = next_batch(train_examples, train_labels, 100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

vote = tf.argmax(tf.reduce_sum(y, 1),1)

print("pourcenatge de vecteurs de controle labellés correctement")
print(sess.run(accuracy, feed_dict={x: control_examples, y_: control_labels}))

#avec le systeme de vote
#il faudrait avoir la base stockée comme des images et non plus des vecteurs



print("pourcenatge d'extraits musicaux de controle labellés correctement")

for i in range(data_size):
    control_predictions[i] = sess.run(vote,feed_dict={x: control_examples2[i], y: control_labels2[i]})
print(tf.reduce_mean(tf.cast(tf.equal(control_predictions, tf.argmax(control_labels2, 1)))))
