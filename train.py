import logging
import tensorflow as tf
from model import AlexNet
import numpy as np


class Train:
    logging.basicConfig(level=logging.INFO)

    def __init__(self, config):
        self.log = logging.getLogger(__name__)
        self.log.info("--- Distributed training ---")
        self.strategy = tf.distribute.MirroredStrategy()
        nb_gpu = self.strategy.num_replicas_in_sync
        self.log.info("* Found {} GPU".format(nb_gpu))
        self.global_batch_size = config.training.batch_size * self.strategy.num_replicas_in_sync
        self.model_path = "{}/{}".format(config.model.dir, config.model.name)

        with self.strategy.scope():
            self.optimizer = tf.keras.optimizers.SGD(lr=config.optimizer.learning_rate,
                                                     momentum=config.optimizer.momentum)
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                      reduction=tf.keras.losses.Reduction.NONE)
            self.__init_model()
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
            self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
            self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        # Train loss is processed over the scope
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

    def __init_model(self):
        self.log.info("--- Model ---")

        with self.strategy.scope():
            self.log.info("* Building Lenet model...")
            self.model = AlexNet()
            self.model.compile(optimizer=self.optimizer, loss=self.loss)
            self.model.build((None, 227, 227, 3))

        self.log.info("* New model built")
        self.log.info("* Summary:")
        self.log.info("{}".format(self.model.summary()))

    def __init_metrics(self):
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    def distribute_dataset(self, ds):
        return self.strategy.experimental_distribute_dataset(ds)

    @tf.function
    def distributed_train(self, x, y):
        per_replica_losses = self.strategy.run(self.train, args=(x, y,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test(self, x, y):
        self.strategy.run(self.test, args=(x, y,))

    def train(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss(y, predictions)
            replicas_losses = tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)

        grads = tape.gradient(replicas_losses, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_accuracy.update_state(y, predictions)

        return replicas_losses

    def test(self, x, y):
        predictions = self.model(x)
        loss_value = self.loss(y, predictions)

        self.test_loss.update_state(loss_value)
        self.test_accuracy.update_state(y, predictions)

    def predict(self, x):
        outputs = self.model.predict(x)

        self.log.info("Outputs Shape:", outputs.shape)
        self.log.info("Outputs Max:", max(outputs[0]))
        self.log.info("Outputs Index:", np.argmax(outputs[0]))
        return outputs

    def save(self):
        self.log.info("Saving model to {}".format(self.model_path))
        self.model.save(self.model_path)
