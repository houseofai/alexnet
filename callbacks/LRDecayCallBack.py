
import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class LearningRateDecay(tf.keras.callbacks.Callback):

    def __init__(self, patience=0):
        super(LearningRateDecay, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        self.time_decay = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # LR Decay
                self.time_decay+=1
                # Max time we can decay: 3
                if self.time_decay > 3:
                    log.info("Too many decay!")
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                else:
                    prev_lr = self.model.optimizer.lr
                    self.model.optimizer.lr=prev_lr/10
                    log.info("* Decreasing Learning Rate (x{}): {} (old {})"
                        .format(self.time_decay, self.model.optimizer.lr, prev_lr))
