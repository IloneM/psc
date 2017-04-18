import feeder
import tensorflow as tf
sess=tf.Session()

deepbatchsize = 64

exfeeder = feeder.Feeder(feeder.emaf.outpath, opts={'batchsize': deepbatchsize})

nb_labels = exfeeder.nblabels  #a retrouver
drop = 1.0
plein = 200
nb_features = exfeeder.nbfeatures # a retrouver
n = deepbatchsize #a definir


x=tf.placeholder(tf.float32, [None, nb_features,n])
y_ = tf.placeholder(tf.float32, shape=[None, nb_labels])


#first layer
i1=4
j1=2
shape_W_conv1= [j1,i1,1,32]
W_conv1 = tf.truncated_normal(shape_W_conv1, stddev=0.1)
b_conv1 = tf.fill([32],0.1)
x_image = tf.reshape(x, [-1,nb_features,n,1])
conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1= tf.nn.max_pool(h_conv1, ksize=[1, i1, j1, 1],strides=[1, 1, 4, 1], padding='SAME')

#second layer
i2=4
j2=2
shape_W_conv2= [j2,i2,1,64]
W_conv2 = tf.truncated_normal(shape_W_conv1, stddev=0.1)
b_conv2 = tf.fill([64],0.1)
conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(conv1 + b_conv1)
h_pool2= tf.nn.max_pool(h_conv2, ksize=[1, i2, j2, 1],strides=[1, 1, 8, 1], padding='SAME')


#densely connected layer
W_fc1 = tf.truncated_normal([2*nb_features*64, 1024],stddev=0.1)
b_fc1 = tf.fill([1024],0.1)

h_pool2_flat = tf.reshape(h_pool2, [-1, 2*nb_features * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = tf.truncated_normal([1024, nb_labels],stddev=0.1)
b_fc2 = tf.fill([nb_labels],0.1)

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())


for i in range(plein):
    #batch = getbatch
    batch = exfeeder.getdeepbatch()
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: drop})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#batch = get control db
batch = exfeeder.getfulltests()
print("test accuracy %g" % accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: drop}))