from model import AlexNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Load sample images
china = load_sample_image("china.jpg")/255

images = np.array([china])
images = tf.image.resize(images, [227,227])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, 11, strides=4, activation="relu", input_shape=[227,227,3]),
    tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(256, 5, activation="relu", padding="SAME"),
    tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(384, 3, activation="relu", padding="SAME"),
    tf.keras.layers.Conv2D(384, 3, activation="relu", padding="SAME"),
    tf.keras.layers.Conv2D(256, 3, activation="relu", padding="SAME"),
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dense(1000, activation="softmax"),
])

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

#print("Summary:", model.summary())
#print("Outputs:", outputs.shape)
