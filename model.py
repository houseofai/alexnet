import tensorflow as tf

class AlexNet(tf.keras.Model):
    def __init__(self):
        super(AlexNet, self).__init__()


        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
        bias = tf.keras.initializers.Ones()
        bias0 = tf.keras.initializers.Zeros()
        # Input is 227 and not 224 as stated on the paper.
        # See issue: https://stackoverflow.com/questions/36733636/the-number-of-neurons-in-alexnet
        self.conv1_1 = tf.keras.layers.Conv2D(48, 11, strides=4, activation="relu", input_shape=[227,227,3], kernel_initializer=initializer, bias_initializer=bias0)
        self.conv1_2 = tf.keras.layers.Conv2D(48, 11, strides=4, activation="relu", input_shape=[227,227,3], kernel_initializer=initializer, bias_initializer=bias0)
        # Output: 227 - 11 / 4 + 1 = 55
        # Maxpool: 55 / 2 = 27.5 = ~27

        self.conv2_1 = tf.keras.layers.Conv2D(128, 5, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        self.conv2_2 = tf.keras.layers.Conv2D(128, 5, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        # Output: 27
        # Maxpool: 27 / 2 = 13.5 = ~13

        self.conv3_1 = tf.keras.layers.Conv2D(192, 3, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        self.conv3_2 = tf.keras.layers.Conv2D(192, 3, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        # Output: 13

        self.conv4_1 = tf.keras.layers.Conv2D(192, 3, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        self.conv4_2 = tf.keras.layers.Conv2D(192, 3, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        # Output: 13

        self.conv5_1 = tf.keras.layers.Conv2D(128, 3, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        self.conv5_2 = tf.keras.layers.Conv2D(128, 3, activation="relu", kernel_initializer=initializer, bias_initializer=bias, padding="SAME")
        # Output: 13

        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))
        # Output: 13 / 2 = 6.5 = ~6

        self.flatten = tf.keras.layers.Flatten()

        # Input: 6 * 6 * 128 * 2 = 9216
        self.fc1_1 = tf.keras.layers.Dense(2048, activation="relu", kernel_initializer=initializer, bias_initializer=bias)
        self.fc1_2 = tf.keras.layers.Dense(2048, activation="relu", kernel_initializer=initializer, bias_initializer=bias)

        self.fc2_1 = tf.keras.layers.Dense(2048, activation="relu", kernel_initializer=initializer, bias_initializer=bias)
        self.fc2_2 = tf.keras.layers.Dense(2048, activation="relu", kernel_initializer=initializer, bias_initializer=bias)

        self.fc3 = tf.keras.layers.Dense(1000, activation="softmax")

    def call(self, input):
        # GPU1
        x1 = self.conv1_1(input)
        x1 = tf.nn.local_response_normalization(x1, depth_radius=5, bias=2, alpha=0.001, beta=0.75)
        x1 = self.max_pool(x1)
        x1 = self.conv2_1(x1)
        x1 = tf.nn.local_response_normalization(x1, depth_radius=5, bias=2, alpha=0.001, beta=0.75)
        x1 = self.max_pool(x1)

        # GPU2
        x2 = self.conv1_2(input)
        x2 = tf.nn.local_response_normalization(x2, depth_radius=5, bias=2, alpha=0.001, beta=0.75)
        x2 = self.max_pool(x2)
        x2 = self.conv2_2(x2)
        x2 = tf.nn.local_response_normalization(x2, depth_radius=5, bias=2, alpha=0.001, beta=0.75)
        x2 = self.max_pool(x2)

        x = tf.keras.layers.concatenate([x1,x2])

        x1 = self.conv3_1(x)
        x2 = self.conv3_2(x)

        # GPU1
        x1 = self.conv4_1(x1)
        x1 = self.conv5_1(x1)
        x1 = self.max_pool(x1)
        # GPU2
        x2 = self.conv4_2(x2)
        x2 = self.conv5_2(x2)
        x2 = self.max_pool(x2)

        ## Fully Connected Layers
        x = tf.keras.layers.concatenate([x1,x2])
        x = self.flatten(x)

        # GPU1
        x1 = self.fc1_1(x)
        x1 = self.fc2_1(x1)
        # GPU2
        x2 = self.fc1_2(x)
        x2 = self.fc2_2(x2)

        x = tf.keras.layers.concatenate([x1,x2])

        x = self.fc3(x)

        return x
