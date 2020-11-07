import tensorflow as tf


class AlexNet(object):
    def __init__(self, optimizer, loss):
        super(AlexNet, self).__init__()

        self.optimizer = optimizer
        self.loss = loss

        self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
        self.bias = tf.keras.initializers.Ones()
        self.bias0 = tf.keras.initializers.Zeros()

        # Init model
        self.model_cnn12_gpu_1 = self.__get_cnn12()
        self.model_cnn12_gpu_2 = self.__get_cnn12()

        self.model_cnn345_gpu_1 = self.__get_cnn345()
        self.model_cnn345_gpu_2 = self.__get_cnn345()

        self.model_fc12_gpu_1 = self.__get_fc12()
        self.model_fc12_gpu_2 = self.__get_fc12()

        self.model_fc3 = self.__get_fc3()

    def __call__(self, input, y=None, training=False):
        prediction = None
        loss_val = None
        with tf.GradientTape() as tape:
            x1 = self.model_cnn12_gpu_1(input, training=training)
            x2 = self.model_cnn12_gpu_2(input, training=training)

            x = tf.keras.layers.concatenate([x1, x2])

            x1 = self.model_cnn345_gpu_1(x, training=training)
            x2 = self.model_cnn345_gpu_2(x, training=training)

            x = tf.keras.layers.concatenate([x1, x2])

            x1 = self.model_fc12_gpu_1(x, training=training)
            x2 = self.model_fc12_gpu_2(x, training=training)

            x = tf.keras.layers.concatenate([x1, x2])

            prediction = self.model_fc3(x, training=training)

            loss_val = self.loss(y, prediction)

        if training:
            all_vars = self.model_cnn12_gpu_1.trainable_variables \
                       + self.model_cnn12_gpu_2.trainable_variables \
                       + self.model_cnn345_gpu_1.trainable_variables \
                       + self.model_cnn345_gpu_2.trainable_variables \
                       + self.model_fc12_gpu_1.trainable_variables \
                       + self.model_fc12_gpu_2.trainable_variables \
                       + self.model_fc3.trainable_variables

            grads = tape.gradient(loss_val, all_vars)
            self.optimizer.apply_gradients(zip(grads, all_vars))

        return prediction, loss_val

    def __get_cnn12(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(48, 11, strides=4, activation="relu", input_shape=[227, 227, 3],
                                       kernel_initializer=self.initializer, bias_initializer=self.bias0),
                LRN(depth_radius=5, bias=2, alpha=0.001, beta=0.75),
                tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                tf.keras.layers.Conv2D(128, 5, activation="relu", kernel_initializer=self.initializer,
                                       bias_initializer=self.bias, padding="SAME"),
                LRN(depth_radius=5, bias=2, alpha=0.001, beta=0.75),
                tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            ]
        )

    def __get_cnn345(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(192, 3, activation="relu", kernel_initializer=self.initializer,
                                       bias_initializer=self.bias, padding="SAME"),
                tf.keras.layers.Conv2D(192, 3, activation="relu", kernel_initializer=self.initializer,
                                       bias_initializer=self.bias, padding="SAME"),
                tf.keras.layers.Conv2D(128, 3, activation="relu", kernel_initializer=self.initializer,
                                       bias_initializer=self.bias, padding="SAME"),
                tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            ]
        )

    def __get_fc12(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2048, activation="relu", kernel_initializer=self.initializer,
                                      bias_initializer=self.bias),
                tf.keras.layers.Dropout(.5),
                tf.keras.layers.Dense(2048, activation="relu", kernel_initializer=self.initializer,
                                      bias_initializer=self.bias),
                tf.keras.layers.Dropout(.5)
            ]
        )

    def __get_fc3(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1000, activation="softmax")
            ]
        )


class LRN(tf.keras.layers.Layer):
    def __init__(self, depth_radius=5, bias=2, alpha=0.001, beta=0.75):
        super(LRN, self).__init__()
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def __call__(self, inputs):
        return tf.nn.local_response_normalization(inputs, depth_radius=self.depth_radius, bias=self.bias,
                                                  alpha=self.alpha, beta=self.beta)
