import tensorflow as tf
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class ManageCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        self.checkpoint_manager = checkpoint_manager
        self.start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.checkpoint_manager.save()

        log.info("Epoch {} Execution time: {}".format(epoch, time.perf_counter() - self.start_time))
