
#import data base


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#import numpy
import numpy as np

#define batching
def next_batch(features, labels, size):
    batch_features, batch_labels = np.zeros(size)
    for i in range(size):
        x = np.floor(np.random * size)
        batch_features[i] = features[x]
        batch_labels[i] = labels[x]
    return batch_features, batch_labels

#import tensorflow
import tensorflow as tf

#emplacements variables
x = tf.placeholder(tf.float32, [None, ?])

W = tf.Variable(tf.zeros([?, 88]))
b = tf.Variable(tf.zeros([88]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 88])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = next_batch(features, labels, 100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))