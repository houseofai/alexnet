import tensorflow as tf
import numpy as np

class AlexNet(tf.keras.Model):
    def __init__(self):
        super(AlexNet, self).__init__()

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
        bias = tf.keras.initializers.Ones()
        bias0 = tf.keras.initializers.Zeros()

        # Input is 227 and not 224 as stated on the paper.
        # See issue: https://stackoverflow.com/questions/36733636/the-number-of-neurons-in-alexnet
        self.conv1 = tf.keras.layers.Conv2D(96, 11, strides=4, activation="relu", input_shape=[227,227,3], kernel_initializer=initializer, bias_initializer=bias0)
        # Output: 227 - 11 / 4 + 1 = 55
        # Maxpool: 55 / 2 = 27.5 = ~27

        self.conv2 = tf.keras.layers.Conv2D(256, 5, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        # Output: 27
        # Maxpool: 27 / 2 = 13.5 = ~13

        self.conv3 = tf.keras.layers.Conv2D(384, 3, activation="relu", kernel_initializer=initializer, bias_initializer=bias0, padding="SAME")
        # Output: 13

        self.conv4 = tf.keras.layers.Conv2D(384, 3, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        # Output: 13

        self.conv5 = tf.keras.layers.Conv2D(256, 3, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        # Output: 13

        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))
        # Output: 13 / 2 = 6.5 = ~6

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation="relu", kernel_initializer=initializer, bias_initializer=bias)
        self.fc2 = tf.keras.layers.Dense(4096, activation="relu", kernel_initializer=initializer, bias_initializer=bias)
        self.fc3 = tf.keras.layers.Dense(1000, activation="softmax", kernel_initializer=initializer, bias_initializer=bias0)

    def call(self, input):

        x = self.conv1(input)
        x = tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=0.001, beta=0.75)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=0.001, beta=0.75)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool(x)

        ## Fully Connected Layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
