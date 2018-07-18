import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_model

# Initialize session
sess = tf.InteractiveSession()
mnist_model_path = "./mnist_model/mnist_base"

# Download data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Initialize model inputs and load model variables
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
y_conv, keep_prob = mnist_model.base_model(x)

# Define loss and optimization variables
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Load pretrained model if it exits, and train new one otherwise
if(tf.gfile.Exists(mnist_model_path + ".index")):
    print("pretrained model found: restoring")
    saver = tf.train.Saver([var for var in tf.global_variables()])
    saver.restore(sess, mnist_model_path)
else:
    print("pretrained model not found: training")
    sess.run(tf.global_variables_initializer())

    # Train model
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    saver = tf.train.Saver([var for var in tf.global_variables()])
    if not os.path.exists("./mnist_model/"):
        os.makedirs("./mnist_model/")
    saver.save(sess, mnist_model_path)

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
