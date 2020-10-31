# Internal libs
from callbacks import LRDecayCallBack as lrcb
from callbacks import ManageCheckpointsCallBack as mcc
from model import AlexNet
import data_augmentation as da
# 3rd party sys libs
from pathlib import Path
import logging
import sys
import yaml
from munch import munchify
import argparse
# 3rd party frameworks
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tf.executing_eagerly()

parser = argparse.ArgumentParser()
parser.add_argument("--conf", default="orignal", help="'orignal' or 'test' config file")
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

log.info("--- Configuration file ---")
config_file = "original"
if args.conf.lower() == "test":
    config_file = "test"

log.info("* Loading configuration file '{}'".format(config_file))
config = munchify(yaml.safe_load(open("config/{}.yml".format(config_file))))
log.info("** Loaded")

log.info("--- Distributed training ---")
strategy = tf.distribute.MirroredStrategy()
nb_gpu = strategy.num_replicas_in_sync
log.info("* Found {} GPU".format(nb_gpu))

log.info("--- Dataset ---")
BATCH_SIZE = config.training.batch_size * strategy.num_replicas_in_sync
ds = da.processing(BATCH_SIZE, config.data.crop_amount)

log.info("--- Model ---")
optimizer = tf.keras.optimizers.SGD(learning_rate=config.optimizer.learning_rate, momentum=config.optimizer.momentum)
loss = tf.keras.losses.MeanSquaredError()

# Load model
with strategy.scope():
    log.info("* Building Alexnet model...")
    model = AlexNet()
    model.compile(optimizer=optimizer, loss=loss)
    model.build((None, 227, 227, 3))
    log.info("* New model built")

log.info("* Summary:")
log.info("{}".format(model.summary()))

log.info("--- Checkpoint ---")
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)#, iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, directory=config.checkpoint.dir, checkpoint_name=config.checkpoint.name, max_to_keep=config.checkpoint.max_to_keep)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    log.info("* Restored from {}".format(manager.latest_checkpoint))
    last_epoch = manager.latest_checkpoint[-1]
else:
    log.info("* Initializing from scratch.")

# Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.tensorboard.dir)
callbacks = [mcc.ManageCheckpoints(manager), lrcb.LearningRateDecay(patience=3), tensorboard_callback]

# Train
log.info("Start training")
log.info("* epochs: {}".format(config.training.epochs))
model.fit(ds, epochs=config.training.epochs, batch_size=BATCH_SIZE, callbacks=callbacks)






#
