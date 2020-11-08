# Standard library imports
import logging
import argparse
import datetime
# Related third party imports
import numpy as np
import tensorflow as tf
# Local application/library specific imports
from helpers import LogEvents as lev
from helpers import TimeManager as tm
from helpers import CheckpointManager as cm
from helpers import LearningRateManager as lrm
from config import ConfigManager as cfg
import data_augmentation as da
import train as tr


def debug(dir):
    mytime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tf.debugging.experimental.enable_dump_debug_info(dir + "/" + mytime, tensor_debug_mode="FULL_HEALTH",
                                                     circular_buffer_size=-1)


def predict(config, image_path):
    training = tr.Train(config)
    img = tf.keras.utils.get_file(origin=image_path)
    # TODO Transform
    cm.CheckpointManager(training.model, training.optimizer, config)
    predictions = training.predict(img)

    log.info("Outputs Shape:", predictions.shape)
    log.info("Outputs Max:", max(predictions[0]))
    log.info("Outputs Index:", np.argmax(predictions[0]))


def train(config):
    training = tr.Train(config)

    if config.tensorboard.debug:
        log.info("--- Debug activated ---")
        debug(config.tensorboard.dir)

    log.info("--- Dataset ---")
    ds_train, ds_test = da.processing(config.data.dataset, config.training.batch_size, config.data.crop_amount)
    ds_size = tf.data.experimental.cardinality(ds_train).numpy()

    # Helpers
    lrd = lrm.LearningRateDecay(config.training.patience)
    ckpt_manager = cm.CheckpointManager(training.model, training.optimizer, config)
    logevents = lev.LogEvents(log_dir=config.tensorboard.dir)
    timemanager = tm.TimeManager(config.training.epochs, ds_size)

    # Train
    log.info("Start training")
    log.info("* epochs: {}".format(config.training.epochs))
    log.info("* Processing the images. Might take a while depending on the CPU")

    last_epoch = int(ckpt_manager.get_last_epoch())
    log.info("Training from epochs {} to {}".format(last_epoch, config.training.epochs))
    for epoch in range(last_epoch, config.training.epochs):
        log.info("Start of epoch {}".format(epoch))

        for step, (x_batch, y_batch) in enumerate(ds_train):
            training.train(x_batch, y_batch)

            # Display timing info
            timemanager.display_batch(step, training.train_loss.result())

        for x_test, y_test in ds_test:
            training.test(x_test, y_test)

        # Save checkpoint
        ckpt_manager.save(training.train_loss.result())
        # Save tensorboard event log
        logevents.log(epoch, training.train_loss, training.train_accuracy, training.test_loss,
                      training.test_accuracy)
        # Display timing info
        timemanager.display(epoch)

        if lrd.early_stop(training.train_loss.result()):
            log.info("Early Stop !")
            break

    training.save()


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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
