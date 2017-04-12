import tensorflow as tf
import numpy as np
x_ = [[0,1,2,3,4,5,6,7,8,9] for i in range(10)]
x_ = np.asarray(x_)
print(x_)
x = tf.placeholder(tf.float32,[10,10])
v = tf.reduce_sum(x, 0)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
res = sess.run(v, feed_dict= {x: x_})
print(res)