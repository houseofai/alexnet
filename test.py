import numpy as np
import tensorflow as tf
import logging
import datetime
from data import data_augmentation as da
import resource

if False:
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    print("###LIMITS", low, high)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


ds_train = da.prepare_trainset("imagenette/full-size-v2", "train", 128, True, 2)
ds_train = ds_train.cache()

initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
bias = tf.keras.initializers.Ones()
bias0 = tf.keras.initializers.Zeros()


class LRN(tf.keras.layers.Layer):
    def __init__(self):
        super(LRN, self).__init__()

    def call(self, inputs):
        print(inputs.shape)
        return tf.nn.local_response_normalization(inputs, depth_radius=5, bias=2, alpha=0.001, beta=0.75)


model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(96, 11, strides=4, activation="relu", input_shape=[227, 227, 3],
                               kernel_initializer=initializer, bias_initializer=bias0),
        LRN(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        tf.keras.layers.Conv2D(256, 5, activation="relu", kernel_initializer=initializer,
                               bias_initializer=bias, padding="SAME"),
        LRN(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        tf.keras.layers.Conv2D(384, 3, activation="relu", kernel_initializer=initializer,
                               bias_initializer=bias0, padding="SAME"),
        tf.keras.layers.Conv2D(384, 3, activation="relu", kernel_initializer=initializer,
                               bias_initializer=bias, padding="SAME"),
        tf.keras.layers.Conv2D(256, 3, activation="relu", kernel_initializer=initializer,
                               bias_initializer=bias, padding="SAME"),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Fully Connected Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation="relu", kernel_initializer=initializer, bias_initializer=bias),
        tf.keras.layers.Dense(4096, activation="relu", kernel_initializer=initializer, bias_initializer=bias),
        tf.keras.layers.Dense(1000, activation="softmax", kernel_initializer=initializer,
                              bias_initializer=bias0),
    ]
)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=500, batch_size=128)
