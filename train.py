# Internal libs
from helpers import LogEvents as lev
from helpers import TimeManager as tm
from helpers import CheckpointManager as cm
from config import ConfigManager as cfg
import objects as obj
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

def compute_loss(train_obj, y, predictions):
  per_example_loss = train_obj.loss(y, predictions)
  return tf.nn.compute_average_loss(per_example_loss, global_batch_size=train_obj.global_batch_size)


def train_step(train_obj, x, y):

    with tf.GradientTape() as tape:
        predictions = train_obj.model(x, training=True)
        loss_value = compute_loss(train_obj, y, predictions)

    grads = tape.gradient(loss_value, train_obj.model.trainable_variables)
    train_obj.optimizer.apply_gradients(zip(grads, train_obj.model.trainable_variables))
    return loss_value, predictions

@tf.function
def distributed_train_step(train_obj, x,y):
    per_replica_losses, per_replica_predictions = train_obj.strategy.run(train_step, args=(train_obj, x,y,))

    return train_obj.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None), per_replica_predictions

def get_model(config):
    log.info("--- Model ---")
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.optimizer.learning_rate, momentum=config.optimizer.momentum)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    log.info("* Building Alexnet model...")
    model = AlexNet()
    model.compile(optimizer=optimizer, loss=loss)
    model.build((None, 227, 227, 3))
    log.info("* New model built")
    log.info("* Summary:")
    log.info("{}".format(model.summary()))
    return model, loss, optimizer

def main(args):

    cfgManager = cfg.ConfigManager(args.conf)
    config = cfgManager.get_conf()

    log.info("--- Distributed training ---")
    strategy = tf.distribute.MirroredStrategy()
    nb_gpu = strategy.num_replicas_in_sync
    log.info("* Found {} GPU".format(nb_gpu))

    log.info("--- Dataset ---")
    global_batch_size = config.training.batch_size * strategy.num_replicas_in_sync
    ds_train, ds_test = da.processing(config.data.dataset, global_batch_size, config.data.crop_amount)
    ds_train = strategy.experimental_distribute_dataset(ds_train)


    # Load model
    with strategy.scope():
        model, loss, optimizer = get_model(config)

        # Helpers
        ckpt_manager = cm.CheckpointManager(ds_train, model, optimizer, config)
        logevents = lev.LogEvents(log_dir=config.tensorboard.dir)
        timemanager = tm.TimeManager(config.training.epochs)

    # Tracking metrics
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    train_obj = obj.TrainObj(strategy, model, loss, optimizer, global_batch_size, train_loss, train_accuracy)

    # Train
    log.info("Start training")
    log.info("* epochs: {}".format(config.training.epochs))
    log.info("* Processing the images. Might take a while depending on the CPU")

    last_epoch = int(ckpt_manager.get_last_epoch())
    log.info("Training from epochs {} to {}".format(last_epoch, config.training.epochs))
    for epoch in range(last_epoch, config.training.epochs):
        log.info("Start of epoch {}".format(epoch))

        for step, (x_batch_train, y_batch_train) in enumerate(ds_train):
            loss_values, predictions = distributed_train_step(train_obj, x_batch_train, y_batch_train)

            with strategy.scope():
                train_loss(loss_values)
                train_accuracy(y_batch_train, predictions)

        for step, (x_test, y_test) in enumerate(ds_test):
            predictions = train_obj.model(x_test)
            loss_value = train_obj.loss(y_test, predictions)

            test_loss(loss_value)
            test_accuracy(y_test, predictions)

        # Save checkpoint
        ckpt_manager.save(train_loss.result())
        # Save tensorboard event log
        logevents.log(epoch, train_loss, train_accuracy, test_loss, test_accuracy)
        # Display timing info
        timemanager.display(epoch)

    model_path = "{}/{}".format(config.model.dir, config.model.name)
    log.info("Saving model to {}".format(model_path))
    model.save(model_path)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="orignal", help="'orignal' or 'test' config file")
    args = parser.parse_args()

    main(args)
