# Regression On IAPS Images

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy import misc
from os import walk

# Returns a list of all image file names from the happy and sad image directories
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

# Returns the image from the file name after resizing
def parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [768, 768])
    return image_resized

# Returns a tf Dataset with 2 lists (images_decoded, labels) in shuffled batches of 32 pairs
def parse_tensor_data(X_data, y_data):
    tensor_labels = tf.data.Dataset.from_tensor_slices(y_data)
    tensor_labels = tensor_labels.batch(50)
    tensor_data = tf.data.Dataset.from_tensor_slices(X_data)
    tensor_data = tensor_data.map(parse_function)
    tesnor_data = tensor_data.batch(50)
    return tensor_data, tensor_labels

img_fnames, img_labels = pull_image_fnames("../pics/Happy/", "../pics/Sad/")

# Split lists of file names and labels between a training and testing set
X_train, X_test, y_train, y_test = train_test_split(img_fnames, img_labels, test_size=.8)
train_data, train_labels = parse_tensor_data(X_train, y_train)
test_data, test_labels = parse_tensor_data(X_test, y_test)

# Create Iterators for the datasets
train_dataI = train_data.make_one_shot_iterator()
train_labelsI = train_labels.make_one_shot_iterator()
test_dataI = test_data.make_one_shot_iterator()
test_labelsI = test_labels.make_one_shot_iterator()

n_inputs = 768*768 # Image Size
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 2 # happy or sad

<<<<<<< HEAD
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
=======
#creates placeholders that will have data assigned at a later date
#X saves the information created by passing in images
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
#y saves the value of whether or not the image is happy
y = tf.placeholder(tf.int64, shape=(None), name="y")
>>>>>>> 5f8d79082622d7003ce7925e5f512cc8ac9e02cd

#constructs the neural network
def neuron_layer(X, n_neurons, name, activation=None):
    #gives a name to the layer being created using name that is passed in
    with tf.name_scope(name):
        #takes in size of the image as the number of inputs
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X,W) + b
        if activation=="relu":
            return tf.nn.relu(z)
        else:
            return z

#creates three neuron layers and connects the input and outputs
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")

#uses cross entropy to define a cost parse_function
#Cross entropy penalizes models that estimate a low probabilityy for the target class
# logits = output of the network before softmax activation function
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

#gradient descent optimizer - tweaks parameters to minimize cost function
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#gets the accuracy by checking if the highest logit is equal to the target class
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    #casts correct to floats (1.0 if correct 0.0 if incorrect) and takes the average of all of the values to get overall accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#creates the initial node to initialize all variables
init = tf.global_variables_initializer()

<<<<<<< HEAD
=======
#method used to generate file names from a directory
from os import walk

def pull_image_fnames(happypath, sadpath):
    image_fnames = []
    image_labels = []

    #adds all of the names of files in happypath to the array of files
    for (dirpath, dirnames, filenames) in walk(happypath):
        image_fnames.extend(filenames)
        break

    #adds a 1 for every image added from happypath to image_labels
    image_labels = [1 for i in range(0, len(image_fnames))]

    #adds all of the names of files in happypath to the array of files
    for (dirpath, dirnames, filenames) in walk(sadpath):
        image_fnames.extend(filenames)
        break

    #adds a 0 for every image added from sadpath to image_labels
    length = len(image_fnames) - len(image_labels)
    image_labels = image_labels + [0 for i in range(0,length)]

    return image_fnames, image_labels

#decodes an image and resizes it to 768x768
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

with tf.Session() as sess:
    init.run()
    for epoch in range(400):
        train_data_handle = sess.run(train_dataI.string_handle())
        train_label_handle = sess.run(train_labelsI.string_handle())
        sess.run(training_op, feed_dict={X: train_data_handle, y: train_label_handle})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
    print(epoch, "Train Accuracy: ", acc_train)
