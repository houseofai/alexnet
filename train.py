import logging
import tensorflow as tf
from model import AlexNet

"""
Initiate the model, the metrics and prepare the context for training
"""


class Train:
    """
    Class to prepare the context to train the model
    """
    logging.basicConfig(level=logging.INFO)

    def __init__(self, config):
        """
        Initialize the parameters for the training
        :param config: Configuration object to initialize the parameters
        """
        self.log = logging.getLogger(__name__)

        #self.model_path = "{}/{}".format(config.model.dir, config.model.name)

        self.optimizer = tf.keras.optimizers.SGD(lr=config.optimizer.learning_rate,
                                                 momentum=config.optimizer.momentum)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.__init_model()

        self.__init_metrics()

    def __init_model(self):
        """
        Initialize the model
        """
        self.log.info("--- Model ---")

        self.log.info("* Building AlexNet model...")
        self.model = AlexNet()

        self.log.info("* New model built")

    def __init_metrics(self):
        """
        Initialize the metrics
        """
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    def __init_gpus(self):
        """
        Initialize and check the amount of GPUs
        """
        self.log.info("--- GPU ---")
        self.gpus = tf.config.experimental.list_physical_devices('GPU')
        if self.gpus is None:
            self.log.info("* No GPU found")
        elif len(self.gpus) < 2:
            self.log.info("* Found {} GPUs. Need at least 2 GPUs".format(len(self.gpus)))
        else:
            self.log.info("* Found {} GPUs".format(len(self.gpus)))
            for gpu in self.gpus:
                print("* Name:", gpu.name, "  Type:", gpu.device_type)

    def train(self, x, y):
        """
        Train and test the model
        :param x: The feature(s)
        :param y: The label(s)
        """
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            t_loss = self.loss(y, predictions)

        grads = tape.gradient(t_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss.update_state(t_loss)
        self.train_accuracy.update_state(y, predictions)

    def test(self, x, y):
        """
        Test the model
        :param x: The feature(s)
        :param y: The label(s)
        :return: The prediction and the loss
        """
        predictions = self.model(x)
        t_loss = self.loss(y, predictions)

        self.test_loss.update_state(t_loss)
        self.test_accuracy.update_state(y, predictions)

        return predictions, t_loss

    def predict(self, x):
        """
        Predict
        :param x: The feature(s)
        :return: The prediction
        """
        predictions = self.model(x)
        return predictions
