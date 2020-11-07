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
    # TODO Transform
    cm.CheckpointManager(img, training.model, training.optimizer, config)
    training.predict(img)


def train(config):
    training = tr.Train(config)

    log.info("--- Dataset ---")
    ds_train, ds_test = da.processing(config.data.dataset, config.training.batch_size, config.data.crop_amount)

    # Helpers
    #ckpt_manager = cm.CheckpointManager(ds_train, training.model, training.optimizer, config)
    logevents = lev.LogEvents(log_dir=config.tensorboard.dir)
    timemanager = tm.TimeManager(config.training.epochs)

    # Train
    log.info("Start training")
    log.info("* epochs: {}".format(config.training.epochs))
    log.info("* Processing the images. Might take a while depending on the CPU")

    last_epoch = 0 #int(ckpt_manager.get_last_epoch())
    log.info("Training from epochs {} to {}".format(last_epoch, config.training.epochs))
    for epoch in range(last_epoch, config.training.epochs):
        log.info("Start of epoch {}".format(epoch))

        for x_batch_train, y_batch_train in ds_train:
            training.train(x_batch_train, y_batch_train)

        for x_test, y_test in ds_test:
            training.test(x_test, y_test)

        # Save checkpoint
        #ckpt_manager.save(training.train_loss.result())
        # Save tensorboard event log
        logevents.log(epoch, training.train_loss, training.train_accuracy, training.test_loss,
                      training.test_accuracy)
        # Display timing info
        timemanager.display(epoch)

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
