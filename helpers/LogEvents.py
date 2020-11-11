import tensorflow as tf
import datetime
import logging
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class LogEvents:
    """
    Class to log events for Tensorboard
    """

    def __init__(self, log_dir):
        """
        Initialize the folders and writers to write the events
        :param log_dir: The log dir
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = '{}/{}/'.format(log_dir, current_time)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def log(self, step, train_loss, train_accuracy, test_loss, test_accuracy):
        """
        Log an event
        :param step: the current step
        :param train_loss: The current training loss
        :param train_accuracy: The current training accuracy
        :param test_loss: The current testing loss
        :param test_accuracy: The current testing accuracy
        """
        log.info(
            "[epoch:{}] train loss: {:.4} - train accuracy: {:.4} - test loss: {:.4} - test accuracy: {:.4}".format(
                step, train_loss, train_accuracy, test_loss, test_accuracy))

        self.log_train(step, train_loss, train_accuracy)
        self.log_test(step, test_loss, test_accuracy)

    def log_train(self, step, loss, accuracy):
        self.__log(step, 'train_loss', loss)
        self.__log(step, 'train_accuracy', accuracy)

    def log_test(self, step, loss, accuracy):
        self.__log(step, 'test_loss', loss)
        self.__log(step, 'test_accuracy', accuracy)

    def __log(self, step, name, value):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def reset(self, train_loss, train_accuracy, test_loss, test_accuracy):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
