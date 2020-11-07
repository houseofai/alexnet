import logging
import tensorflow as tf
from model import AlexNet
import numpy as np


class Train:
    logging.basicConfig(level=logging.INFO)

    def __init__(self, config):
        self.log = logging.getLogger(__name__)

        self.model_path = "{}/{}".format(config.model.dir, config.model.name)

        self.optimizer = tf.keras.optimizers.SGD(lr=config.optimizer.learning_rate,
                                                 momentum=config.optimizer.momentum)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.__init_model()

        self.__init_metrics()

    def __init_model(self):
        self.log.info("--- Model ---")

        self.log.info("* Building AlexNet model...")
        self.model = AlexNet()

        self.log.info("* New model built")

    def __init_metrics(self):
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    def __init_gpus(self):
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
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            t_loss = self.loss(y, predictions)

        grads = tape.gradient(t_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss.update_state(t_loss)
        self.train_accuracy.update_state(y, predictions)

    def test(self, x, y):
        predictions = self.model(x, y)
        t_loss = self.loss(y, predictions)

        self.test_loss.update_state(t_loss)
        self.test_accuracy.update_state(y, predictions)

    def predict(self, x):
        predictions = self.model(x)

        self.log.info("Outputs Shape:", predictions.shape)
        self.log.info("Outputs Max:", max(predictions[0]))
        self.log.info("Outputs Index:", np.argmax(predictions[0]))
        return predictions

    def save(self):
        self.log.info("Saving model to {}".format(self.model_path))
        # self.model.save(self.model_path)
