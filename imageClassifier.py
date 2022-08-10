import tensorflow as tf

# Read image data using numpy
import numpy as np

from tensorflow import keras

# Printing stuff
import matplotlib.pyplot as plt


# Load a pre-defined dataset (70k of 28x28)
fashion_mnist = keras.datasets.fashion_mnist

# Pull out data from dataset
# Returns 60k training images in first set (train_images, train_labels)
# Returns 10k testing images in second set
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Show data in terminal
print(train_labels[0])
print(train_images[0])

# show image, grayscaled, 0 to 255 means 2^6 (8 bit image black and white)
plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

# Define neural net structure
model = keras.Sequential([

    # create input layer
    # input is a 28x28 image ("Flattens" the 28x28 into a single 784x1 input layer)
    # we flatten it to simplify the structure of the neural net, since we want 
    # one entire column for the data
    keras.layers.Flatten(input_shape=(28, 28)),

    # create output layer
    # output is 0-10 (depending on clothing). return maximum
    # 10 nodes corresponding to label (label is # for each clothing (ie. 1 = trouser))
    keras.layers.Dense(units=10, activation=tf.nn.softmax)

])