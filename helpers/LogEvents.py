import tensorflow as tf
import datetime
import logging
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class LogEvents():
    def __init__(self, log_dir):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = '{}/{}/train'.format(log_dir, current_time)
        self.test_log_dir = '{}/{}/test'.format(log_dir, current_time)
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir, name="trainOdy")
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir, name="testOdy")


    def log(self, epoch, train_loss, train_accuracy, test_loss, test_accuracy):
        log.info("[epoch:{}] train loss: {:.4} - train accuracy: {:.4} - test loss: {:.4} - test accuracy: {:.4}".format(epoch, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()))

        self.log_train(epoch, train_loss, train_accuracy)
        self.log_test(epoch, test_loss, test_accuracy)

    def log_train(self, epoch, train_loss, train_accuracy):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        train_loss.reset_states()
        train_accuracy.reset_states()

    def log_test(self, epoch, test_loss, test_accuracy):

        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        test_loss.reset_states()
        test_accuracy.reset_states()
