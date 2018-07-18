''' Generate white box adversarial examples using the iterative fgsm method. 
The adversarial image generation code here is written by me based on the paper:
https://arxiv.org/pdf/1611.01236.pdf
'''
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import mnist_model

import matplotlib.pyplot as plt

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

# Load pretrained model
print("restoring pretrained model")
saver = tf.train.Saver([var for var in tf.global_variables()])
saver.restore(sess, mnist_model_path)


# -- Largely Original Code Below 
index_twos = np.where(np.argmax(mnist.test.labels, 1) == 2)[0]
selected_twos = index_twos[np.random.randint(0, len(index_twos), 10)]
sixes = tf.one_hot(6*np.ones(10), depth=10) # TODO: Switch over to tf 

cross_entropy_adv = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=sixes, logits=y_conv))
grad = tf.gradients(cross_entropy_adv, x)

e = 0.14 # Cap on adversarial perturbation magnitude
learning_rate = 0.04 # Learning rate for adversarial image generation
fgsm_images = mnist.test.images[selected_twos].copy()

# Iterative fgsm
for i in range(300):
    feed_dict = {x: fgsm_images,
                    y_: mnist.test.labels[selected_twos], keep_prob: 1.0}
    fgsm_grads = sess.run(grad, feed_dict=feed_dict)
    
    signed_grads = np.sign(fgsm_grads[0]) # Zero indexed as the input is shape (1, 10000, 784)

    fgsm_images -= learning_rate*signed_grads
    fgsm_images = np.clip(
        fgsm_images, mnist.test.images[selected_twos]-e, mnist.test.images[selected_twos]+e)
    if(i % 100 == 0):
        print('iteration {} loss: {}'.format(
            i, cross_entropy_adv.eval(feed_dict=feed_dict)))

perturbations = mnist.test.images[selected_twos] - fgsm_images
predictions = sess.run(y_conv, feed_dict={x: mnist.test.images[selected_twos], y_: mnist.test.labels[selected_twos], keep_prob:1.0})
adv_predictions = sess.run(y_conv, feed_dict={x: fgsm_images, y_: mnist.test.labels[selected_twos], keep_prob:1.0})

# Plotting & saving images  
plt.figure(figsize=(10, 30))
for i in range(10):
    ax = plt.subplot(10, 3, (i*3)+1)
    plt.title('Pred: {}'.format(np.argmax(predictions[i])))
    plt.imshow(mnist.test.images[selected_twos][i].reshape((28, 28)))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = plt.subplot(10, 3, (i*3)+2)
    plt.imshow(perturbations[i].reshape((28, 28)))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = plt.subplot(10, 3, (i*3)+3)
    plt.title('Pred: {}'.format(np.argmax(adv_predictions[i])))  
    plt.imshow(fgsm_images[i].reshape((28, 28)))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

if not os.path.exists("./results/"):
    os.makedirs("./results/")
plt.savefig("./results/iterative_fgsm.png")
