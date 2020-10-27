from model import AlexNet
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
from sklearn.datasets import load_sample_image

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


epochs = 90
checkpoints_directory = './checkpoints'
checkpoint_name = 'alexnet'


def transform(image, label):
    image = tf.image.resize(image, [227,227])
    return image, label

def img_shape(ds):
    image, label = next(iter(ds))
    return image.shape

def show_key_weights(model):
    layer = model.layers[0]
    weights = layer.get_weights()
    log.info("* Key Weights - First Layer: {}".format(weights[0][0][0][0][0]))

    layer = model.layers[8]
    weights = layer.get_weights()
    log.info("* Key Weights - Last Layer: {}".format(weights[0][0][0]))

log.info("--- Dataset ---")
log.info("* Loading ImageNet dataset")
ds = tfds.load('imagenet_a', split='test', as_supervised=True, batch_size=128)

log.info("* Transforming the dataset")
ds = ds.map(transform)




log.info("--- Model ---")

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9)
loss = tf.keras.losses.MeanSquaredError()

# Load model
log.info("* Building Alexnet model...")
model = AlexNet()
model.compile(optimizer=optimizer, loss=loss)
model.build(img_shape(ds))
log.info("* New model built")

log.info("* Summary")
log.info("{}".format(model.summary()))

log.info("Key weights BEFORE loading checkpoint")
show_key_weights(model)

# Checkpoint manager
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)#, iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, directory=checkpoints_directory, checkpoint_name=checkpoint_name, max_to_keep=3)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    log.info("Restored from {}".format(manager.latest_checkpoint))
else:
    log.info("Initializing from scratch.")

log.info("Key weights AFTER loading checkpoint")
show_key_weights(model)

class ManageCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        self.checkpoint_manager = checkpoint_manager

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.checkpoint_manager.save()

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

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

callbacks = [ManageCheckpoints(manager), LearningRateDecay(), tensorboard_callback]

# Train
log.info("Start training")
log.info("* epochs: {}".format(epochs))
model.fit(ds, epochs=epochs, callbacks=callbacks)
















#
