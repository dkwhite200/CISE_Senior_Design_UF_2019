# Regression On IAPS Images

import numpy as np
import tensorflow as tf
from scipy import misc

n_inputs = 768*768 # Image Size
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X,W) + b
        if activation=="relu":
            return tf.nn.relu(z)
        else:
            return z

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

from os import walk

def pull_image_fnames(happypath, sadpath):
    image_fnames = []
    image_labels = []

    for (dirpath, dirnames, filenames) in walk(happypath):
        image_fnames.extend(filenames)
        break

    image_labels = [1 for i in range(0, len(image_fnames))]

    for (dirpath, dirnames, filenames) in walk(sadpath):
        image_fnames.extend(filenames)
        break
    length = len(image_fnames) - len(image_labels)
    image_labels = image_labels + [0 for i in range(0,length)]

    return image_fnames, image_labels

def parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [768, 768])
  return image_resized, label

def parse_tensor_data(X_data, y_data):
	tensor_data = tf.data.Dataset.from_tensor_slices((X_data, y_data))
	tensor_data = tensor_data.map(parse_function)
	tensor_data = tensor_data.shuffle(buffer_size=2000)
	tensor_data = tensor_data.batch(32)
	return tensor_data

img_fnames, img_labels = pull_image_fnames("../pics/Happy/", "../pics/Sad/")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(img_fnames, img_labels, test_size=.8)
train_data = parse_tensor_data(X_train, y_train)
test_data = parse_tensor_data(X_test, y_test)
train_iterator = train_data.make_one_shot_iterator()
next_train = train_iterator.get_next()

#Training Phase
with tf.Session() as sess:
	init.run()
	for _ in range(400):
		sess.run(next_train)
		acc_train = accuracy.eval(train_data)
		acc_test = accuracy.eval(test_data)
		print(epoch, "Train accuracy:", acc_train, "Test accuracy", acc_test)
