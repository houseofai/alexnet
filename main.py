# Internal libs
from helpers import LogEvents as lev
from helpers import TimeManager as tm
from helpers import CheckpointManager as cm
from config import ConfigManager as cfg
import data_augmentation as da
import train as tr
# 3rd party sys libs
import logging
import argparse
# 3rd party frameworks
import tensorflow as tf


def predict(config, image_path):
    training = tr.Train(config)
    img = tf.keras.utils.get_file(origin=image_path)
    cm.CheckpointManager(img, training.model, training.optimizer, config)
    training.predict(img)


def train(config):
    training = tr.Train(config)

    log.info("--- Dataset ---")
    ds_train, ds_test = da.processing(config.data.dataset, training.global_batch_size)
    ds_train = training.distribute_dataset(ds_train)
    ds_test = training.distribute_dataset(ds_test)

    with training.strategy.scope():
        # Helpers
        ckpt_manager = cm.CheckpointManager(ds_train, training.model, training.optimizer, config)
        logevents = lev.LogEvents(log_dir=config.tensorboard.dir)
        timemanager = tm.TimeManager(config.training.epochs)

    # Train
    log.info("Start training")
    log.info("* epochs: {}".format(config.training.epochs))
    log.info("* Processing the images. Might take a while depending on the CPU")

    last_epoch = int(ckpt_manager.get_last_epoch())
    log.info("Training from epochs {} to {}".format(last_epoch, config.training.epochs))
    for epoch in range(last_epoch, config.training.epochs):
        log.info("Start of epoch {}".format(epoch))

        total_loss = 0.0
        num_batches = 0

        for x_batch_train, y_batch_train in ds_train:
            total_loss += training.distributed_train(x_batch_train, y_batch_train)
            num_batches += 1

        training.train_loss.update_state(total_loss / num_batches)

        for step, (x_test, y_test) in enumerate(ds_test):
            training.distributed_test(x_test, y_test)

        with training.strategy.scope():
            # Save checkpoint
            ckpt_manager.save(training.train_loss.result())
            # Save tensorboard event log
            logevents.log(epoch, training.train_loss, training.train_accuracy, training.test_loss,
                          training.test_accuracy)
            # Display timing info
            timemanager.display(epoch)

    training.save()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="orignal", help="'orignal' or 'test' config file")
    parser.add_argument("--mode", default="train", help="train or predict")
    parser.add_argument("--image", default="", help="path to an image (test mode)")
    args = parser.parse_args()

    config_manager = cfg.ConfigManager(args.conf)
    config = config_manager.get_conf()

    if args.mode == "predict":
        if args.image is not None:
            predict(config, args.image)
        else:
            ValueError("'image' argument needs to be specified")
    elif args.mode == "train":
        # Default mode
        train(config)
