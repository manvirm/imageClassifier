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

# show image, grayscaled
plt.imshow(train_images[0], cmap='gray', vmin=0, vmax=255)
plt.show()

# Define neural net structure
model = keras.Sequential([
    # input layer (compatible with dataset)
    # one flat layer of our data
    keras.layers.Flatten(input_shape=(28, 28))


])